# GDN Prefill Kernel: Seq-Major Output Layout

## Problem

The Qwen3.5-27B GDN prefill kernel (`gdn_prefill_fused`) writes its output as `[num_pairs*seq, 1, Dv]` — heads-outer, seq-inner, one valid row per tile. Downstream consumers in `gdn.py:forward_prefill` need `[1, 1, seq, num_pairs*Dv]` — seq-outer, heads-as-features, dense tiles. The host bridges the two layouts with a `reshape → permute(0,2,1,3) → reshape` chain (`gdn.py:946-950`). This causes two problems visible in the current 4×P150 prefill profile (`generated/profiler/reports/2026_04_28_00_27_47/`):

1. **The reshape pile is expensive.** Each of the two `ReshapeViewDeviceOperation` calls in the chain takes ~700 μs (TILE→TILE reshapes that cross tile boundaries force a real data shuffle), and the `TransposeDeviceOperation` between them takes ~150 μs. Per profiled GDN layer that's ~1.5 ms; extrapolated to all 48 GDN layers in the model the chain alone burns **~70 ms** of full-pass kernel time.

2. **The post-kernel `rms_norm` sees a degenerate shape.** It runs on `[num_pairs*seq, 1, Dv]` (in 4D: `[1, 24576, 1, 128]`). The rms_norm kernel parallelizes across rows in the Y dim (= 1, padded to 32) and serializes across the W dim (= 24576), so it effectively runs single-core. Each call is 1.3 ms instead of the few microseconds it should be; full-pass cost **~10 ms**.

Together: ~80 ms of avoidable full-pass kernel time, all rooted in the same layout mismatch.

## Scope

- **In scope:** A new GDN prefill compute+writer kernel pair that fuses RMSNorm into the compute and emits the seq-major dense layout `[1, 1, seq, num_pairs*Dv]`, plus the host-side cleanup in `gdn.py:forward_prefill` that consumes it.
- **Constraint:** The existing `gdn_prefill.cpp`, `writer_gdn_prefill.cpp`, and `gdn_prefill_fused()` entry point stay intact. The new variants live alongside them, selected per-call via a new entry point.
- **Out of scope:**
  - The conv1d row-major round-trip in `gdn.py:845-865` (item 1 of the broader optimization set) — separate spec.
  - The FFN `gate_up` matmul split in `fused_mlp.py` (item 7) — separate spec.
  - The decode-side equivalent. Decode writes `[B, 1, Dv]` per pair and has no large reshape pile or degenerate-shape rms_norm.

## Design

### Why both kernel writer and compute change

A correctness subtlety blocks the naive "writer-only" version: the existing post-kernel `rms_norm` normalizes along the last dim Dv (per-head, 128 features). If the kernel emits `[1, 1, seq, num_pairs*Dv]` and we run `rms_norm` on that, the last dim becomes 1536 and normalization mixes all heads' features together — different math. There is no view-only reshape between `[1, 1, seq, num_pairs*Dv]` and `[1, seq, num_pairs, Dv]` in TILE layout (their tile decompositions differ), so we cannot cheaply re-expose the per-head boundary post-kernel.

The clean fix is to **fuse the per-head RMSNorm into the GDN prefill compute kernel**. The kernel already accepts `norm_w` as an arg (`gdn_kernel_op.py:1090`, `_gdn_prefill_fused` signature) but currently ignores it; the compute path computes the recurrence's Dv-vector per token and writes it raw. Adding RMSNorm there is a few extra ops per token (one reduction over Dv, one rsqrt, one elementwise multiply with `norm_w`) — a small fraction of the per-token compute. With RMSNorm done inside the kernel, the writer is free to emit any layout we want, and the host-side `rms_norm` call goes away entirely.

This means **two kernel files change** (a new compute variant + a new writer variant), but the reader, the host op signature, and the math seen by downstream consumers are unchanged.

### Compute kernel change

Add `models/demos/qwen35_27b/tt/gdn_kernel/kernels/compute/gdn_prefill_with_norm.cpp` — a copy of `gdn_prefill.cpp` with one extra stage immediately before the per-token output is pushed to `cb_out`:

1. Reduce-sum-of-squares over Dv (the value vector for this token).
2. `rsqrt(ssq + Dv * eps)` — `Dv * eps` is precomputed in `rms_eps_tt` (already passed as a kernel arg, also currently unused).
3. Multiply the Dv-vector by the rsqrt scalar and by `norm_w[1, 1, Dv]` element-wise.
4. Push to `cb_out` as before.

Numerically equivalent to applying `ttnn.rms_norm(x, weight=norm_w, epsilon=1e-6)` with `epsilon` interpreted the same way (`rms_eps_tt` is already constructed as `Dv * 1e-6` at `gdn.py:172`).

The existing `gdn_prefill.cpp` is left untouched — `gdn_prefill_fused()` keeps using it for any caller still on the legacy path.

### Kernel writer change

Add `models/demos/qwen35_27b/tt/gdn_kernel/kernels/dataflow/writer_gdn_prefill_seq_major.cpp`. The reader is **unchanged**. Only the writer's destination tile-id math and CB consumption pattern differ.

**Current writer (`writer_gdn_prefill.cpp:38-55`).** Per `(pair, tok)`, pops `Vt` tiles from `cb_out` and writes them at `tile_id = (pair * num_tokens + tok) * Vt + vt`. Each tile contains one valid token row plus 31 padding rows.

**Seq-major writer.** Destination layout is `[1, 1, seq, num_pairs*Dv]`. In tile coordinates: `Yt = ceil(seq / 32)`, `Xt = num_pairs * Vt`. Tile-id at `(token_y_tile, pair_idx, vt)` = `token_y_tile * (num_pairs * Vt) + pair_idx * Vt + vt`. Each destination tile is fully populated (32 token rows × 32 feature columns).

The writer does its own tile repacking in L1:
- Allocate an L1 scratch region of `Vt * tile_bytes` bytes per writer core (= 4 × 2 KB = 8 KB for Dv=128, BFLOAT16). This is the "dense Y-tile" being assembled.
- For each `pair` the core owns, iterate over `ceil(num_tokens / 32)` outer blocks. Inside each block, loop up to 32 token iterations: pop `Vt` tiles from `cb_out`, copy the single valid (post-norm) row (row 0 of each tile) into row `tok % 32` of the corresponding scratch tile. After 32 tokens (or end-of-sequence), NoC-write the `Vt` scratch tiles to DRAM at `tile_id = block_idx * (num_pairs * Vt) + pair * Vt + vt`.
- If `num_tokens` is not a multiple of 32, the final partial block writes only the rows it has and leaves the remaining rows zero. Confirm in implementation that `seq` is already tile-padded by the host (chunked-prefill rounds chunk size up to 32), in which case the partial-block branch is dead code — but include it defensively.

State write (`cb_state_out`) is unchanged.

### Host op wrapper

Add `gdn_prefill_fused_seq_major(...)` in `models/demos/qwen35_27b/tt/gdn_kernel/gdn_kernel_op.py`, parallel to the existing `gdn_prefill_fused`. Same arg list, except `output` is now expected to have shape `[1, 1, seq, num_pairs*Dv]` (TILE, BFLOAT16, DRAM-interleaved). Internally points the compute kernel path at the new `gdn_prefill_with_norm.cpp` and the writer kernel path at the new `writer_gdn_prefill_seq_major.cpp`.

Path constants added next to the existing pair (`gdn_kernel_op.py:39-43`):

```python
COMPUTE_PREFILL_WITH_NORM_PATH = f"{_KERNEL_DIR}/compute/gdn_prefill_with_norm.cpp"
WRITER_PREFILL_SEQ_MAJOR_PATH = f"{_KERNEL_DIR}/dataflow/writer_gdn_prefill_seq_major.cpp"
```

`norm_w`, `rms_scale_tt`, `rms_eps_tt` were previously documented as "unused by compute, kept for API compat" (`gdn_kernel_op.py:1090-1093`). Update the docstring of the new entry point to reflect that they are now consumed by the fused-norm compute kernel.

### Host model changes

In `gdn.py:forward_prefill` (lines 909-950), three changes:

1. Change the output buffer allocation:
   ```python
   prefill_output = ttnn.from_torch(
       torch.zeros(1, 1, num_pairs * num_tokens_padded, Dv, dtype=torch.bfloat16),
       # ... existing args ...
   )
   # NEW: shape [1, 1, seq, num_pairs * Dv] instead of [num_pairs * seq, 1, Dv]
   ```
   (Use `num_tokens_padded` = round-up-to-32 of `seq_len` if not already.)

2. Call `gdn_prefill_fused_seq_major(...)` instead of `gdn_prefill_fused(...)`.

3. Delete the post-kernel rms_norm + reshape pile (lines 944-950 today):
   ```python
   # REMOVED — RMSNorm now done inside the kernel; output is already in the
   # required layout. prefill_output is the consumer-ready [1, 1, seq, value_dim_tp].
   out_f = prefill_output
   ```

A feature flag (`GDN_PREFILL_SEQ_MAJOR`, env-var) gates the new path; default off until the e2e test confirms parity. After validation the default flips on.

### Data flow

After the change:

```
qkvz_proj → conv1d → kernel (fused-norm compute + seq-major writer) → [1,1,seq,value_dim_tp] TILE
                  → ⊙ silu(z) → out_proj → all_reduce
```

vs. before:

```
... → kernel ([num_pairs*seq, 1, Dv]) → rms_norm (degenerate, single-core)
   → reshape [1, num_pairs, seq, Dv] (~700 μs)
   → permute(0,2,1,3) (~150 μs)
   → reshape [1, 1, seq, value_dim_tp] (~700 μs) → ...
```

### Numerics + correctness

The math is unchanged at the module boundary. The compute kernel folds in the same per-head RMSNorm that `ttnn.rms_norm(prefill_output, weight=norm_w, epsilon=1e-6)` was applying after the fact, with the same `epsilon` (already constructed as `Dv * 1e-6` in `rms_eps_tt` at `gdn.py:172`). The writer's L1 repacking is a pure layout transform: each output element written by the new writer equals the corresponding element of `permute(reshape(rms_norm(old_output)))`, which is what consumers received before.

There is one numerical difference to watch for: in the original path, RMSNorm runs in a separate dispatch with its own intermediate dtype. The fused compute version runs RMSNorm in whatever accumulator the GDN compute kernel uses — likely fp32 inside the math FPU but bfloat16 at the cb_out boundary. Validate that the unit test PCC is ≥ what the original path achieves; expect 1.0 (bit-exact) if both paths use the same accumulator widths.

### Testing

1. **Kernel-level unit test** — new file `models/demos/qwen35_27b/tt/tests/test_gdn_prefill_seq_major.py`. Mirrors `test_gdn_prefill_dv_split.py`. Drives both `gdn_prefill_fused` and `gdn_prefill_fused_seq_major` on identical inputs (random `conv_out`, `a`, `b`, scalars). Applies `ttnn.rms_norm(...)` followed by the existing `reshape → permute → reshape` chain to the old-path output, and asserts close-equality against the new-path output (PCC ≥ 0.999; bit-exact if accumulator widths match). Recurrence-state output (untouched by this change) is compared bit-exact.

2. **GDN-module test** — extend `test_gdn.py` (or add a focused variant) to run `forward_prefill` with the env flag both off and on; assert PCC of the GDN module output stays at the existing baseline (≥0.99 vs PyTorch reference).

3. **End-to-end test** — `test_e2e_generate.py` (already in the working tree). Run with `GDN_PREFILL_SEQ_MAJOR=1` and confirm generated tokens match the baseline run. Capture a fresh profile and verify in `analysis/prefill.csv`:
   - No `ReshapeViewDeviceOperation` calls between `GenericOpDeviceOperation` (the GDN kernel) and the silu·z `BinaryNgDeviceOperation` multiply.
   - No post-GDN `LayerNormDeviceOperation` row at all (it has been folded into the kernel).
   - Total non-GDN prefill kernel time drops by ~10 ms per profiled prefill (= ~80 ms full-pass once extrapolated to 64 layers).

### Risk + rollback

- The new writer is a parallel code path; the existing `writer_gdn_prefill.cpp` and `gdn_prefill_fused` keep working with no behavioural change. Reverting is a one-line flip of the env flag default.
- L1 scratch in the writer adds 8 KB per core. Need to confirm this fits within the writer's existing L1 budget (current writer uses ~0 L1 scratch). Inspect existing `dv_split` writer for comparable patterns.
- `num_tokens` may not be a multiple of 32 in chunked prefill (e.g. the trailing chunk in `prefill_layer_chunked`). The writer must handle the partial trailing block correctly. The unit test must cover at least one non-multiple-of-32 `seq` to exercise this branch.

## Files touched

New:
- `models/demos/qwen35_27b/tt/gdn_kernel/kernels/compute/gdn_prefill_with_norm.cpp`
- `models/demos/qwen35_27b/tt/gdn_kernel/kernels/dataflow/writer_gdn_prefill_seq_major.cpp`
- `models/demos/qwen35_27b/tt/tests/test_gdn_prefill_seq_major.py`

Modified:
- `models/demos/qwen35_27b/tt/gdn_kernel/gdn_kernel_op.py` — add `COMPUTE_PREFILL_WITH_NORM_PATH`, `WRITER_PREFILL_SEQ_MAJOR_PATH`, and `gdn_prefill_fused_seq_major(...)`.
- `models/demos/qwen35_27b/tt/gdn.py` — env-flagged switch in `forward_prefill`, allocate output as `[1, 1, seq, value_dim_tp]`, delete the post-kernel rms_norm + reshape + permute + reshape pile.

## Acceptance criteria

- Unit test: PCC ≥ 0.999 between (old kernel + reference rms_norm + reference reshape/permute/reshape) and (new kernel) on at least three `seq` values — one tile-aligned, one non-aligned (e.g. 1861), one matching the production chunk size (2048).
- Module test: PCC of `TtGatedDeltaNet.forward_prefill` output ≥0.99 vs PyTorch reference, with the new flag on.
- E2E test: generated tokens from `test_e2e_generate.py` match baseline for the same prompt and decoding settings.
- Profile diff: in a fresh `analysis/prefill.csv`, no `ReshapeViewDeviceOperation` rows between the GDN `GenericOpDeviceOperation` and the silu·z `BinaryNgDeviceOperation`, and no post-GDN `LayerNormDeviceOperation` row at all (the only LayerNorm rows present are the pre/post-AG distributed-norm layers).
- Measured per-prefill kernel-time drop on device 0 of ≥8 ms (target ~10 ms; extrapolates to ~80 ms full pass).
