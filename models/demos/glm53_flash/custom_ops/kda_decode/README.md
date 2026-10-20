# `kda_decode` — fused KDA single-token decode block (model-local `ttnn.generic_op`)

One program for everything between the packed projection matmuls and `o_proj` of a GLM-5.3-Flash KDA layer
(design 8.1): causal conv (kernel 4) + SiLU on the device-owned tile history, per-head L2 norms, the per-key-channel
forget gate, beta and output gates, the gated delta-rule step on the recurrent state and the gated RMSNorm, packed
as the bf16 TILE input of `o_proj`. Kernels: `kernels/{reader,writer,compute}_kda_decode.cpp`; launcher
`op.py::KdaDecode`; torch reference `KdaDecode.golden`; test `tests/custom_ops/test_kda_decode.py`. Used by
`tt/kda.py` (`GLM53_KDA_FUSED=1`, the default): the block is 5 ops (qkv matmul, two gate matmuls, this program, o_proj).

## Contract (all tensors interleaved DRAM/L1 on the same device; K = V = 128, conv kernel 4)

| tensor | shape | dtype / layout | role |
|---|---|---|---|
| `qkv` | `[1, 1, B, 3*H*K]` | bf16 TILE | raw projections `[q \| k \| v]`, head-major inside a branch |
| `gates` | `[1, 1, B, H*K + H*V + Hb]` | fp32 TILE | `[f pre-activation \| og pre-activation \| beta logits]`, `Hb = ceil(H/32)*32` |
| `conv_hist` | `[3, B, 3*H*K]` | bf16 TILE | tap-major conv history (tap 0 oldest, tap 2 the previous token); **updated in place** (tap j+1 -> j, raw -> tap 2) |
| `recurrent` | `[B, H, K, V]` | fp32 TILE | the recurrent state; **updated in place** |
| `taps` | `[1, 1, 4, 3*H*K]` | bf16 TILE | conv taps, row j = tap j (row 3 multiplies the current token); bf16 like the checkpoint weight |
| `chan_consts` | `[1, 1, 2, H*K]` | fp32 TILE | row 0 = `dt_bias`, row 1 = `exp(A_log)` per key channel |
| `o_norm_w` | `[1, 1, 1, V]` | fp32 TILE | gated RMSNorm weight |
| `selectors` | `[34*32, 32]` | bf16 TILE | `op.kda_selector_tiles()`: tile r < 32 = E_r^T (a 1 at (r, 0)), tile 32 all ones, tile 33 identity |
| `out` | `[1, 1, B, H*V]` | bf16 TILE | **pre-allocated by the caller**; normed and gated, the `o_proj` input |

Math per user b, head h (q, k L2-normalised, q scaled by K^-0.5): `S = diag(exp g) S; u = beta (v - k S);
S += k^T u; o = q S; out = rmsnorm(o) * w * sigmoid(og)` with `g = lb * sigmoid(exp(A_log) (f + dt_bias))` per key
channel, `beta = sigmoid(b)`. Padding rows of `qkv` / `gates` must be finite (zero when produced by ttnn matmuls on
zero-padded inputs); rows of the history beyond `B` are never read for live users.

Constructor: `KdaDecode(num_heads=H, head_dim=128, gate_lower_bound=-5.0, l2_eps=1e-6, norm_eps=1e-5,
math_fidelity=HiFi4)`. Call: `op(qkv, gates, conv_hist, recurrent, taps, chan_consts, o_norm_w, selectors, out) -> out`.
Buffer addresses are baked into the runtime args: under trace every io tensor must be persistent or allocated
inside the trace (the model allocates `qkv` / `gates` / `out` inside its captured decode step). The same program
runs on every device of a mesh (SPMD `ProgramDescriptor`); heads are the device's local heads.

The layer keeps the history in this layout only on the block path (`TtKDA.conv_tiled`); `prefill_forward` converts
to / from ttKDA's row-major `[B, 3, W]` at its boundary and `user_conv_state` / `write_user_conv_state` /
`conv_history_torch` / `upload_conv_history` serve the per-user prefill hand-offs and the tests.

## Program

* Work items `(head h, user tile row tb, user part)`; `KdaDecode.placement` puts the `parts` cores of a
  (head, row) slot in one column of the grid (the largest `parts` with `grid.x * (grid.y // parts) >= H * Bt`; 5 for
  16 heads, 1 for 64 heads on the 12x10 grid, then one item per core round-robin). The live users of a tile row are
  split evenly over the parts (`up = ceil(n_row / parts)`, users `[part*up, min((part+1)*up, n_row))`).
* Reader: the part-0 core of a slot reads the slot's inputs once -- the 3 x (3 history + raw) tile blocks of the head,
  the taps, the gate blocks, the channel constants, ~172 KB -- and multicasts them into the same CB addresses of
  the other parts (ready / valid semaphores), then shifts the history in place from L1. Every part reads its
  users' selector tile and state (the first user's with the item reads, overlapping the multicast wait).
* Compute, per item on the whole 32-row block: conv + SiLU per branch (row broadcasts of the tap rows), L2 norms
  (row sums by a matmul with the all-ones tile), `decay = exp(lb sigmoid(a (f + dt)))`, the beta column
  (`gb . E_c^T`), `wg = sigmoid(og) * w`, then `kd = k decay`, `qd = q decay`, `qk = rowsum(q k)`, `kcol = k^T`,
  `decT = decay^T`. SFPU work runs only over the live rows when the tile row has <= 16 users (faces 0 / 1 with
  `ceil(rows / 2)` iterations per face, >= 2).
* Per user r (no row lifting -- a whole-tile op's row r is user r's result): `dcol = decT . E_r^T`,
  `vread = kd . S`, `oq = qd . S`, `u = beta_r e_r (col bcast) (v - vread)`, `S' = diag(decay_r) S + kcol (x) u`
  (column broadcast of the decay column, then the rank-1 matmul accumulated onto it in DST: one pack per state
  tile, straight to the writer), `o_r = oq_r + (q_r . k_r) u_r` accumulated at row r of the part's output block via
  the projector matmuls `diag(e_r) oq + (q.k)_r diag(e_r) u`.
* Writer: each user's state (in place); a part's output block goes to the slot's part-0 core (gather CB +
  semaphore), whose compute sums the parts, applies the gated RMSNorm and packs the head's bf16 output tiles of
  the tile row; the part-0 writer stores them.
* Compile-time args -- compute: `Kt, Vt, Bt, B, parts, lb, l2_eps, norm_eps, K^-0.5, 1/V, it_half` (floats as
  uint32 bits); reader: `Kt, Vt, Bt, B, H, Wq_t, Wg_t, HK_t, HV_t, parts, ONES_PAGE` + `TensorAccessorArgs` of qkv,
  conv_hist, taps, gates, chan_consts, o_norm_w, selectors, recurrent; writer: `Kt, Vt, Bt, B, H, parts, HV_t` +
  accessors of out, recurrent. Runtime args per core -- reader: `item0, n_items, stride, <8 buffer addresses>,
  part-0 NoC x/y, last-part NoC x/y`; writer: `item0, n_items, stride, out, recurrent addresses, part-0 NoC x/y`;
  compute: `item0, n_items, stride`. Semaphores 0..3 (ready, valid, the constant one, gather).
* CBs (indices 0..31, per core): see `KdaDecode.CB` and the compute kernel header; ~1 MB of L1 per core.

## Measured (chip 3, real layer-4 weights, traced KDA layer `tests/test_kda_perf.py`, device time per step)

| heads | users | `kda_decode` us | block ops | block device us | KDA layer ms/step |
|---|---|---|---|---|---|
| 16 (TP4 shard) | 4 | 70.7 (was 145) | 5 (was 6) | 199 | 0.248 (was 0.343) |
| 16 (TP4 shard) | 32 | 257 (was 288) | 5 | 385 | 0.438 |

With 16 heads the four matmuls are 128 us (qkv 77.5 us = 325 GB/s of bfp8 weights, o_proj 27 us = 310 GB/s, the
two gate matmuls 23 us); the tilize is gone (TILE output). At B=32 the program moves the fp32 state twice (64 MB on
a 16-head shard): 257 us is 250 GB/s, so its floor on this chip is ~200 us (the matmuls' 325 GB/s). Op-level
correctness vs the torch golden: output / state PCC >= 0.999995 (B = 1/4/32/40, H = 16/64, multi-step, trace replay).
