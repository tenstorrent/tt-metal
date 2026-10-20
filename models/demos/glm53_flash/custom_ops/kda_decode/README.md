# `kda_decode` — fused KDA single-token decode block (model-local `ttnn.generic_op`)

One program for everything between the packed projection matmuls and `o_proj` of a GLM-5.3-Flash KDA layer
(design 8.1): causal conv (kernel 4) + SiLU on the device-owned history, per-head L2 norms, the per-key-channel
forget gate, beta and output gates, the gated delta-rule step on the recurrent state and the gated RMSNorm.
Kernels: `kernels/{reader,writer,compute}_kda_decode.cpp`; launcher `op.py::KdaDecode`; torch reference
`KdaDecode.golden`; test `tests/custom_ops/test_kda_decode.py`. Used by `tt/kda.py` (`GLM53_KDA_FUSED=1`).

## Contract (all tensors interleaved DRAM/L1 on the same device; K = V = 128, conv kernel 4)

| tensor | shape | dtype / layout | role |
|---|---|---|---|
| `qkv` | `[1, 1, B, 3*H*K]` | bf16 TILE | raw projections `[q \| k \| v]`, head-major inside a branch |
| `gates` | `[1, 1, B, H*K + H*V + Hb]` | fp32 TILE | `[f pre-activation \| og pre-activation \| beta logits]`, `Hb = ceil(H/32)*32` |
| `conv_state` | `[B, 3, 3*H*K]` | bf16 ROW_MAJOR | the three previous raw rows (oldest first); **updated in place** (shift + append) |
| `recurrent` | `[B, H, K, V]` | fp32 TILE | the recurrent state; **updated in place** |
| `taps` | `[1, 1, 4, 3*H*K]` | fp32 TILE | conv taps, row j = tap j (row 3 multiplies the current token) |
| `chan_consts` | `[1, 1, 2, H*K]` | fp32 TILE | row 0 = `dt_bias`, row 1 = `exp(A_log)` per key channel |
| `o_norm_w` | `[1, 1, 1, V]` | fp32 TILE | gated RMSNorm weight |
| `selectors` | `[33*32, 32]` | bf16 TILE | `op.selector_tiles()`: tile r < 32 has a 1 at (0, r), tile 32 is all ones |
| `out` | `[1, 1, B, H*V]` | bf16 ROW_MAJOR | **pre-allocated by the caller**; one row per user, normed and gated (tilize it for `o_proj`) |

Math per user b, head h (q, k L2-normalised, q scaled by K^-0.5): `S = diag(exp g) S; u = beta (v - k S);
S += k^T u; o = q S; out = rmsnorm(o) * w * sigmoid(og)` with `g = lb * sigmoid(exp(A_log) (f + dt_bias))` per key
channel, `beta = sigmoid(b)`. Padding rows of `qkv` / `gates` must be finite (zero when produced by ttnn matmuls on
zero-padded inputs).

Constructor: `KdaDecode(num_heads=H, head_dim=128, gate_lower_bound=-5.0, l2_eps=1e-6, norm_eps=1e-5)`.
Call: `op(qkv, gates, conv_state, recurrent, taps, chan_consts, o_norm_w, selectors, out) -> out`.

## Program

* Work items `(head h, user tile row tb, user part)`: `parts = clamp(cores // (H * Bt), 1, 32)`, item `i` ->
  `h = i // (Bt*parts)`, `tb = (i % (Bt*parts)) // parts`, `part = i % parts`; the live users of a tile row are split
  evenly over the parts (`up = ceil(n_row / parts)`, users `[part*up, min((part+1)*up, n_row))`); items are dealt
  round-robin over `min(items, cores)` cores (core c: items c, c + NC, ...).
* Per item the core computes the conv + SiLU, the L2 norms and the gates on the whole 32-row block, then walks its
  users: row r is lifted into tile row 0 by the selector matmul `E_r X`, the recurrence runs on the user's state
  (read once from `recurrent`, written once in place), and the output row is accumulated with `E_r^T o_r`; the block
  gets the gated RMSNorm and is untilized into the row-major output (the writer writes only this part's rows).
* Compile-time args — compute: `Kt, Vt, Bt, B, parts, lb, l2_eps, norm_eps, K^-0.5, 1/V` (floats as uint32 bits);
  reader: `Kt, Vt, Bt, B, H, Wq_t, Wg_t, HK_t, HV_t, parts` + `TensorAccessorArgs` of qkv, conv_state, taps, gates,
  chan_consts, o_norm_w, selectors, recurrent; writer: `Kt, Vt, Bt, B, H, HK_t, HV_t, parts` + accessors of out,
  recurrent, conv_state.
* Runtime args per core — reader: `item0, n_items, stride, <8 buffer addresses>, conv_page_bytes`; writer:
  `item0, n_items, stride, out, recurrent, conv_state addresses, conv_page_bytes, out_page_bytes`; compute:
  `item0, n_items, stride`. Buffer addresses are baked into the args, so under trace every io tensor must be
  persistent or allocated inside the trace (the model allocates `qkv`/`gates`/`out` inside its captured decode step).
* CBs (indices 0..31, per core): see `KdaDecode.CB` and the compute kernel header; ~1 MB of L1 per core.
* Mesh: the same program on every device (SPMD `ProgramDescriptor`); heads are the device's local heads.

Measured (chip 3, real layer-4 weights, see `doc/functional_decoder/README.md`): 241 us at B=4 / 930 us at B=32 with
64 heads, 199 / 343 us with 16 heads.
