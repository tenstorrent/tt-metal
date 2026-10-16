# Sol-Attn on the MiniMax-H3 ring-joint SDPA: interface between the routing pre-pass and the attention kernel

Status: 2026-10-15. Owners: attention kernel + this contract: Sol-Attn attention port (session `sol-vid2`);
routing pre-pass producer: today `sol_prepass.py` (`SolPrepass`), being replaced by the fused proxy-scoring
kernel (`sol_prepass_fused.py`, session `sol-video`). Changing anything in sections 2-4 requires both owners.

## 1. Where the boundary sits

```
QKV projections + QK-norm + RoPE          (dense path, unchanged)
        |  q, k, v  [1, H_local, R, D] bf16 TILE, per device (SP shard of R rows, TP shard of heads)
        v
+-------------------------------------------------------------+
| routing pre-pass ("proxy scoring")  -- producer              |   per device, per layer, per step
|   block pooling, pooled K/V all-gather along SP, moments,    |
|   threshold tau, proxy scores, band/sink/prefix rules,       |
|   union to kernel granularity                                |
+-------------------------------------------------------------+
        |  sol_block_mask, sol_pooled_k, sol_pooled_v   (section 2)
        v
ttnn.transformer.ring_joint_scaled_dot_product_attention(q, k, v, ..., sol_block_mask=, sol_pooled_k=,
                                                         sol_pooled_v=[, sol_bias=])      -- consumer
        |  spatial output [1, H_local, R, D] bf16 TILE, identical layout to the dense op
        v
output projection                           (dense path, unchanged)
```

The kernel owns everything after the mask: the K/V stream (ring all-gather, intra-chip chains), the exact
flash-attention steps, the pooled-correction step and its column bias, normalisation and the output layout.
The pre-pass owns everything that decides *what* is exact. Neither side depends on the other's internals:
the kernel never reads thresholds, scores or pooled statistics other than the three tensors below, and the
pre-pass never needs the kernel's core assignment, chain layout or chunk schedule (it only needs the chunk
sizes, which come from the same `SDPAProgramConfig` the caller passes to the op).

## 2. Tensors the pre-pass produces (all per device, DRAM interleaved, allocated per call)

Geometry (`sol_prepass.SolGeometry`): `S` = SP shards (ring size), `rows` = padded rows per shard, `logical_n`
= true token count of the global sequence, `q_chunk` / `k_chunk` = the op's chunk sizes in tokens, block =
64 tokens, `nb = ceil(rows / 64)`, `nb_pad = round_up(nb, 32)`, `n_blocks_pad = S * nb_pad`,
`nqc = ceil(rows / q_chunk)`, `nqc_pad = round_up(nqc, 32)`, `nkc = ceil(rows / k_chunk)`,
`nkc_pad = round_up(S * nkc, 32)`.

| tensor | shape | dtype / layout | meaning |
|---|---|---|---|
| `sol_block_mask` (chunk mode, production) | `[1, H, nqc_pad, nkc_pad]` | `uint32`, ROW_MAJOR | `mask[0, h, qc, s*nkc + c] != 0` means: local Q chunk `qc` of head `h` computes global K chunk `(shard s, chunk c)` exactly. Entries at `qc >= nqc` or beyond `S*nkc` are ignored (write 0). |
| `sol_block_mask` (gather mode, experimental) | `[1, H, nqc_pad, n_blocks_pad]` | `uint32`, ROW_MAJOR | `!= 0` means: gather global padded block `g = s*nb_pad + b` into Q chunk `qc`'s exact K/V chunks. Only full 64-token blocks may be set. The kernel tells the modes apart by the mask width (`nkc_pad` vs `n_blocks_pad`). |
| `sol_pooled_k` | `[1, H, n_blocks_pad, D]` | `bf16`, TILE | row `s*nb_pad + b` = **mean** over the valid tokens of block `b` of shard `s` (the ragged tail block divides by its true length). Padding rows (`b >= nb`, or blocks with no valid token) are zero. |
| `sol_pooled_v` | `[1, H, n_blocks_pad, D]` | `bf16`, TILE | same blocking, **mean** V (not the paper's sum: the kernel multiplies by the token count through the bias, see section 3). |
| `sol_bias` (optional, legacy) | `[1, H, nqc_pad*32, n_blocks_pad]` | `bf16`, TILE | tile row `qc` (rows `32*qc .. 32*qc+31`, all identical) = column bias per padded block, section 3. Omit it: the kernel generates it. Kept only so the generator can be A/B-validated against it. |

Rules the kernel relies on:

* **Head/shard scoping.** All three tensors are per device and already hold the whole sequence (pooled
  K/V are all-gathered along SP by the producer; the mask row covers all `S*nkc` global chunks). The op
  does not gather them. `H` is the local head count (TP shard), in the same order as `q`.
* **Mask semantics are fixed**: one word per (local Q chunk, global K chunk) in chunk mode, or per (local Q
  chunk, global padded block) in gather/compact mode; non-zero = exact. No bit packing, no extra flags. Padding
  words are ignored.
* **At least one exact chunk per (Q chunk, shard), in every mode.** A ring iteration in which a Q chunk computes
  nothing is not a supported state of the accumulator staging protocol (compute never reaches its last-chunk save,
  the writer waits: deadlock, found 2026-10-16). Producers therefore force chunk 0 of every shard (chunk mode) or
  block 0 of every shard (gather and compact modes; `force_first_chunk=True` in `SolPrepass`, same in the fused
  producer). The reader forces block 0 itself on an all-zero compact-mode row slice as a fallback, but then that
  block is both computed and pooled (a small numerical deviation), so the producer rule is the contract. The forced
  chunk/block is part of the exact set and is included in every reported density.
* **Lifetime**: the tensors must stay allocated until the op has executed. Because ttnn executes
  asynchronously, the caller keeps the Python references at least until the next op that synchronises;
  `MiniMaxH3Attention._sol_tensors` returns them as the op's kwargs, which keeps them alive through the
  enqueue (see the open question in section 6).
* **Program cache**: the op's program hash covers the mask shape, pooled shape, chunk/gather mode and whether
  `sol_bias` is present. Values do not affect the hash; addresses are re-bound on every call.

## 3. What the kernel computes from them (semantics the producer may assume)

For a Q chunk with exact set `E` (its mask row, expanded to blocks: every block of a kept chunk is exact)
and approximated set `U` = all valid blocks not in `E`:

```
m   = running max over exact logits  QK_j^T * scale  (j in E)  and pooled logits (Q KC_b^T + bias_b) * scale (b in U)
D   = sum_{j in E} exp(QK_j^T*scale - m) + sum_{b in U} len_b * exp(Q KC_b^T*scale - m)
N   = sum_{j in E} exp(...) V_j        + sum_{b in U} len_b * exp(...) * VC_b          (VC_b = mean V)
out = N / D
```

The `len_b` weight comes from the column bias `bias_b = ln(len_b) / scale` applied inside the pooled
chunks' QK^T tile before the max/exp (the kernel evaluates `exp((QK + bias - max) * scale)`), and blocks in
`E` or padding get `bias_b = -30000` (exp underflows to exactly 0 in bf16). `len_b` is the valid token count
of the block: 64, the ragged `rows % 64` tail of a shard, or `logical_n`'s tail; it is a pure function of
`(rows, logical_n, 64)` and is derived on device from the op's live `logical_n` (the trace-safe
`logical_n` tensor when present). This is Sol-Attn eq. (D_i, N_i) with a coarser exact set (whole K
chunks) and bf16 pooled tensors; the exact set is never smaller than the pre-pass asked for.

Generated bias values (device) and uploaded bias values (`sol_prepass.build_constants`) are both
`bf16_rne(fp32(ln(len)) / scale)`; the only possible difference is a 1-ulp fp32 `log` difference between
libm and torch before rounding, which the A/B test (`SOL_GEN_BIAS=ab`) checks for.

## 4. Preprocessing ownership

| step | owner | notes |
|---|---|---|
| block pooling of q/k/v, pooled K/V all-gather along SP | pre-pass | all-gather uses the CCL manager's standard path; the ring op does not touch these buffers |
| moments, threshold `mu + tau*sigma` per (Q block, head), proxy scores, compare | pre-pass | numerics (bf16 scores, fp32 threshold) are the producer's choice; the kernel is agnostic |
| band (`|i-j| <= 1` blocks), sink (prefix K blocks), dense prefix query rows | pre-pass | folded into the mask |
| union from (Q block, K block) to (Q chunk, K chunk) | pre-pass | the kernel skips only whole chunks |
| forced first chunk / block | kernel (gather mode only) | chunk mode needs no forcing |
| column bias tiles | kernel (default) or pre-pass (`sol_bias`) | see section 3 |
| chain-union of mask rows across the cores sharing a head | kernel | derived from the same mask rows, no producer involvement |
| step / layer gating (dense first 20 % of steps, first layers) | caller (`sol_active`) | pure Python branch |

## 5. Python entry points

* Producer: `SolPrepass(mesh_device, ccl_manager, sp_axis, tau, thresh_type, band, dense_prefix_queries, mode, emit_bias)`;
  `prepare(geo, sink_tokens, scale)` once per geometry; `__call__(q, k, v, scale=) -> {"block_mask", "pooled_k", "pooled_v", "bias" | None}`.
  A replacement producer only has to return the same dict (with `bias=None`).
* Consumer: `MiniMaxH3Attention._sol_tensors(q, k, v)` turns that dict into the op kwargs and drops a `None` bias.
* Tests: `test_sol_ring_sdpa_minimax_h3.py` (synthetic, torch reference, `SOL_MODE`, `SOL_GEN_BIAS=0|1|ab`),
  `test_sol_realdata_minimax_h3.py` (captured activations, timing per tau, `SOL_GEN_BIAS=1`),
  `test_sol_prepass_minimax_h3.py` (pre-pass vs its torch mirror; a fused producer should pass the same check).

## 6. Open questions / assumptions recorded for the reviewer

1. The generated-bias path assumes `k_chunk` is a multiple of 64 and at most 64 K chunks per shard (both
   already validated by the op for chunk mode).
2. The kernel reads each Q chunk's mask row from DRAM once per (ring iteration, Q chunk) plus once per
   chain member for the union; a producer that writes the mask with a different page layout (anything but
   row-major uint32 with one row per page) breaks this.
3. The fused producer may want to emit the mask directly in the kernel's `[1, H, nqc_pad, nkc_pad]` form
   without the block-level intermediate; nothing in the kernel depends on how the row was built.
4. Tensor lifetime vs asynchronous execution (section 2) is the attention side's responsibility; if the
   cross-generation hang (DESIGN.md 8.6) turns out to be a lifetime/placement effect, the fix lands in
   `MiniMaxH3Attention`, not in the producer.
