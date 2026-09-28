# Sparse SDPA tiled Q and output proposal

## Goal

Let `ttnn.transformer.sparse_sdpa` accept tiled Q and optionally return tiled output for the GLM 5.2 sparse MLA path. Remove the two layout conversions around the call only if the complete path is faster. The sparse KV cache remains row major, and the current token-to-core assignment and dual-NoC KV gather remain intact.

The design priority is **KV read throughput**. Moving Q or output must not consume enough DRAM, NoC, L1, or Tensix capacity to slow the indexed KV gather.

## Current data flow

In the Galaxy warm case (5120-token chunk, SP=8, TP=4), the head-to-sequence all-to-all gives each chip approximately:

| Tensor | Per-chip logical shape | Current layout at sparse SDPA |
| --- | --- | --- |
| Q | `[1, 64, 160, 576]` | BF16 row major, converted from tile immediately before the call |
| KV cache | `[B, 1, T, 656]` | Packed scaled FP8 row major; one 656-byte row per cached token |
| Indices | `[1, 1, 160, 2048]` | UINT32 row major |
| Output | `[1, 64, 160, 512]` | BF16 row major, converted to tile immediately after the call |

The program factory divides the `S` query tokens into contiguous ranges across the full compute grid. A core processes its tokens sequentially. For each token, its reader places all 64 Q head rows into `cb_q_rm`, reads one index row, and gathers selected KV rows. Its writer helps gather KV on the second NoC, then writes all 64 output head rows. Compute uses the same gathered KV for all 64 heads. The compute kernel tilizes Q internally, performs attention, and untilizes the result for the writer.

At 2048 valid keys, the approximate payload per query token is 72 KiB of Q, 1312 KiB of KV, and 64 KiB of output. KV is about 18 times the Q payload. The dual-NoC gather is performance critical; the existing code comments also identify its response traffic as NoC-link bound. Byte totals alone do not capture transaction overhead, so both KV bandwidth and end-to-end device time need measurement.

Relevant code:

- `models/demos/deepseek_v3_d_p/tt/mla/mla.py`, `_sparse_mla`: the two `ttnn.to_layout` calls and the optional head-to-sequence all-to-all.
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/sparse_sdpa_program_factory.cpp`: token assignment, circular buffers, kernel arguments, and runtime address updates.
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/sparse_sdpa_reader.cpp`: Q, indices, and one half of the KV gather.
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/sparse_sdpa_writer.cpp`: the other half of the KV gather and output writes.
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/sparse_sdpa_compute.cpp`: existing Q tilize and output untilize.

## Proposed first implementation: tiled access at the existing workers

Keep every core assigned the same query-token range. Keep KV, indices, the gather protocol, and compute unchanged. Change only Q ingestion and output placement, selected at compile time by tensor layout.

### Tiled Q reader

Accept standard 32×32 tiled BF16 Q in DRAM with logical shape `[1,H,S,K_DIM]`; require `H` and `K_DIM` to be tile aligned. The physical tile axes are `S` and `K_DIM` **within each head**. A worker still needs one row from every head for its current token.

For each head and feature tile, read the two 16-element face-row fragments for `tok % 32` from the Q tile page into the appropriate offsets of the existing `cb_q_rm` row. The tile page is selected by `(head, tok / 32, feature_tile)`. The face-row offsets depend on whether `tok % 32` is in the upper or lower tile half. Reserve and publish `cb_q_rm` exactly as the current reader does, so compute still sees 64 contiguous row-major Q rows and its existing tilize path works without modification.

This is a deliberately small correctness prototype. At GLM geometry it replaces 64 full-row Q reads with many small face-row reads. Those transfers may be expensive even though Q has few bytes relative to KV. Do not read entire Q tiles independently for each token: that would fetch the other 31 tokens' Q data repeatedly and could exceed the KV traffic. Reuse a small L1 staging tile only if measurement shows it helps without reducing KV gather throughput.

### Tiled output writer

Add an optional `output_layout` argument, defaulting to row major. For tiled BF16 output, keep the existing `cb_out_rm` and compute untilize. For each head and output feature tile, scatter the two 16-element face-row fragments of the current token's row into its output tile page `(head, tok / 32, feature_tile)` at the correct face-row offsets.

Different token workers write disjoint bytes of a tile. Verify that BF16's 32-byte face-row writes satisfy the Blackhole DRAM/NoC write alignment and are safe when different cores target different parts of one tile. Do not use read-modify-write of a shared tile. Start with `S % 32 == 0`, which covers the Galaxy warm case; then add explicit zero filling of padded tile rows before accepting other `S` values. Keep FP8 tiled output out of the first cut because a face-row fragment is only 16 bytes and may need a different aligned write strategy.

### Host API and program cache

- Infer Q layout from `q.layout()`; expose `output_layout=ROW_MAJOR_LAYOUT` as an optional Python/C++ argument. Allow all four Q/output row-major/tile combinations for BF16 so each half can be benchmarked independently.
- Keep the KV cache and indices requirements unchanged. Reject sharded Q/output, nonstandard/transposed tiles, unsupported tile geometry, and unsupported dtype/layout combinations with explicit validation.
- For tiled Q, allow only the expected tile padding in `S`; retain the current unpadded rule for row-major Q. Make the tiled output `TensorSpec` use a tiled page config and the logical `[1,H,S,v_dim]` shape.
- Add Q layout, Q tile descriptor/padding where relevant, and output layout to the program hash. The current hash does **not** include Q layout because tiled Q is rejected on every cache hit. Update the cache-hit validation and its existing regression test accordingly.
- Keep dynamic KV cache length and `cache_batch_idx` runtime behavior intact. Do not make either a new compile-time key for plain interleaved KV.

## Why not reserve conversion rows initially?

A dedicated row of Q conversion cores and another row of output conversion cores could read and write full tiles efficiently. It would also remove attention workers from the current full-grid KV gather. At the warm shape, every query token has up to 2048 KV rows to read, so fewer gathering cores could outweigh the saved conversion time. Remote L1 handoff would add NoC traffic where the gather already uses both links; DRAM staging would add an extra read and write. Treat helper cores as a second design only if the existing-worker prototype shows fragmented Q/output transactions are the limiting factor and measured KV bandwidth leaves headroom for fewer workers.

Likewise, splitting 64 heads between two workers would gather the same token's KV twice. Keep all 64 heads together on one worker so each gathered row is reused across both 32-head compute tile rows.

## Implementation sequence

1. **Baseline:** measure the current `to_layout(Q)`, row-major sparse SDPA, and `to_layout(output)` device durations separately and as a sum. Record the complete warm forward, active core count, and KV gather throughput or a stable proxy (valid keys and gather bytes divided by sparse SDPA duration). Use the established quiet CI-style environment and the same warm case for every comparison.
2. **Tiled Q only:** add layout-aware validation, hash key, accessor and reader branch. Keep row-major output. Run correctness and profile the Q-only change. If KV throughput drops enough to offset the removed conversion, stop and retain the row-major model path.
3. **Tiled output only:** add the output layout, spec, and writer branch while Q stays row major. Verify partial tile writes and padding. Profile independently so output cost is visible.
4. **Combined path:** run tiled Q and tiled output together. Remove the two `to_layout` calls in `_sparse_mla` only after the combined op beats the old three-op device sum and improves the complete warm forward. Keep the row-major API path for other callers.
5. **Broaden support:** handle non-multiple-of-32 `S`, then FP8 Q/output if needed. Neither is required to establish the GLM BF16 win.

## Correctness checks

- Compare all four Q/output layout combinations with the same row-major reference for `H=32` and `H=64`, multiple `S` values including 32/160 and a short or partial tile after padding support, and both single- and multi-chunk KV selection.
- Exercise scaled FP8 KV, BF16 KV, sentinel tails, indexed cache batches, block-cyclic cache remapping, and the optional attention sink. These should run through the same KV and compute path.
- Alternate row-major and tiled Q of the same logical shape across program-cache hits. Check that the correct program is selected and output layout is correct. Also check changing cache batch slot and plain interleaved KV length does not require recompilation.
- Compare the model's sparse MLA output and the GLM cache correctness case with the established tolerances, including the head-to-sequence all-to-all path.

## Performance decision

The primary comparison is:

`device_time(tiled-Q → tiled-output sparse_sdpa)` versus
`device_time(Q to_layout) + device_time(row-major sparse_sdpa) + device_time(output to_layout)`.

Also compare full GLM 5.2 warm forward device and host time under the same settings. Record Q-only, output-only, and combined results. Reject the tiled path if its KV gather rate falls or the combined device/forward time does not improve beyond run-to-run noise. The existing row-major path remains the performance fallback.
