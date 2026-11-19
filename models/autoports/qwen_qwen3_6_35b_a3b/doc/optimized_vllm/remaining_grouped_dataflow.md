# Remaining grouped MoE dataflow candidates

Source-only AutoFix audit after parent-reported compact-schedule full40 PASS and async33.36545ms versus33.59963ms control (~0.70%). That result rejects mailbox compaction as useful complexity. The remaining~12ms gap to21.631ms is not attributed to any component without measurement. No production source, test harness, build or device was changed in this audit.

## First experiment: select the existing per-token score layout

`tt/multichip_decoder.py` currently passes scores to `post_combine_reduce` as ROW_MAJOR `(1,1,tokens,8,1)` around line1767. This selects the legacy one-weight-per-page path. The native operation already accepts `(1,1,tokens,8)` and explicitly selects a per-token path:

- `post_combine_reduce/device/post_combine_reduce_device_operation.cpp:65–86,103`: validates both shapes; rank-minus-one means one token’s eight weights per page.
- `post_combine_reduce_program_factory.cpp:109–111`: sets `weights_per_token` from that rank; scratchc4 stores the32 per-token pages.
- `kernels/deepseek_moe_post_combine_reduce_writer.cpp`: the `weights_per_token` branch loads32 pages once, then writes the identical BF16 scalar to each expert weight tile; the old branch issues a tile-sized read for each of32×8 slots.
- The compute kernel, local-expert order, first-active initialization, scalar multiply, BF16 pack accumulation and final tilization are identical.

**Smallest candidate:** drop only the trailing singleton from the model’s score reshape. This uses an existing native API and needs no native rebuild or adapter change. It also avoids the scalar-page reshape/repacking boundary. It removes256 legacy weight reads in favor of32 compact page reads per layer, plus the representation conversion; this is an operation count, not a speed estimate.

**Discriminator:** construct one BF16 ROW_MAJOR score tensor `[1,1,32,8]`, derive the existing singleton form, and compare both calls on the same `[1,1,32,8,2048]` combined values, UINT16 IDs and rank-local INT32 table. Require exact BF16 output bits on all chips for no/one/many local experts, equal and signed-zero scores, changing IDs/weights, and same-input replay. Include a changed-score trace replay to catch stale scalar views. Then reduced buckets/full40/serial96 and matched serving timings. Preserve the current `init_zeros=True` combine control.

## Next candidate: remove padded-token work inside post-combine reduction

The grouped carrier always has32 tokens, including logicalB2/B4 (`multichip_decoder.py:1582–1591`). `post_combine_reduce_program_factory.cpp:81–90` chooses `num_cores=min(num_tokens/32, device_cores)`: the entire decode reduction therefore runs on **one core per chip**. Its reader consumes32×8×2048×2 =1MiB from DRAM; compute scans all256 expert slots, with two expert-ID/table reads broadcast to MATH/PACK per slot (`compute.cpp:109–125`), even when route suppression has made padding entirely off-rank. Each empty token still forces the last expert through zero-weight multiply to initialize an output row. Writer still streams those weights.

A default-off native `valid_token_rows` specialization for logicalB2/B4 is a concrete next step: propagate the original logical row count through `_moe`’s padding recursion and restrict the reader, weight producer and compute token loops to that static count. Retain32-row output/tilization and identical per-valid-row arithmetic. Explicitly initialize every omitted output row to the same zero bits; do not leave stale CB bytes. A reader-owned zero-fill of the row-major scratch **before publishing the first input CB**, with its completion barrier, can order initialization before compute’s first write; this requires a deliberate CB-lifetime implementation review. An alternative compute-owned zero-pack tail is simpler but retains some packing work. B32 remains ordinary. Runtime inactive rows among the validB2/B4 rows continue through the existing path; no host activity branch is introduced.

This is preferable to changing expert reduction order or replacing the grouped numerical family. It removes padded rows, not valid non-aligned prompts. Gate against full32 output bits (including signed zeros and inactive tails), B2/B4/B32 activity transitions, nonblocking changed-input traces and dense burst latency. It is larger than the first experiment and has no measured gain yet.

## Router/topk: narrow hypothesis, with a correction

`decode_batch` calls `_moe(..., batched=True)` without `ordered_router` (`multichip_decoder.py:1989`); its default isFalse. Thus this serving path does **not** run ordered terminal scan over32 rows. It uses generic FP32 router `ttnn.linear`, generic FP32 `topk`, softmax and BF16 cast. The ordered scan in `fused_decoder.py:688` is guarded by `ordered_router && rows==1`; singleton selection uses `MeshFP32TopK` with its separately established B1 semantics.

Computing the generic router/topk/softmax on logicalB2/B4 **before** restoring the32-row carrier is a possible model-only experiment. Keep the generic router family and restore IDs/scores before grouped routing, with inactive/padded slots handled by the existing coherent route mask. However, the native topk factory sizes `Ht` from **padded_shape** (`topk_single_core_program_factory.cpp:36,78`): B2 and32 still occupy one tile row, so merely changing the logical shape need not reduce topk work. Router GEMM likewise remains a physical32-row tile. Expect limited benefit unless a row-aware selector is implemented.

A model-owned multirow extension of `MeshFP32TopK` could read only the valid logical rows and write a full32-row carrier, but it must prove exact generic-topk FP32 values **and IDs**, including ties, ±0, infinities and NaNs; then exact BF16 normalized scores. The B1 selector documents lower-ID tie-breaking and NaNs-above-numbers; that does not alone prove generic topk equivalence. Avoid moving to the B1 ordered projection or indexed expert arithmetic merely to gain speed.

## Capacity and fusion bounds

There is full initialization work, but it is on the **combined token/expert carrier**, not the2240-row dispatch capacity. `combine_program_factory.cpp:944–946` divides all output pages among initializer cores; `writer_untilize.cpp:70`/`zero_init_common.hpp` zero them. For32×8×2048 BF16 this is1MiB per layer before the subsequent1MiB read. Shrinking dispatch capacity alone leaves this work unchanged. The dispatch reader still filters rank ownership/capacity before emitting entries; its valid-region work provides no new capacity-shrink hypothesis.

Simply setting `init_zeros=False` is unsafe as an exactness argument: a rank with no local expert forces a final zero-weight multiply, and stale NaN/Inf or signed values can affect the result. Do not substitute uninitialized memory for zeros.

The existing `combine_row_index` option can avoid a dense gather only for a **ROW_MAJOR** table. Native grouped output is tiled; directly passing it is rejected. A true combine/post-reduce fusion would need either a tiled gather reader preserving the exact scalar-reduction kernel or a changed native output layout with byte-exact packing. It is substantially larger than the score-layout fix. Keep it as a contingency after measurements identify whether the single-core post-reduce work matters.

IndexedB1 uses TP-sharded gate/up/down sparse matmuls and a different reduction topology; grouped decode uses rank-owned experts with grid reductions and resident/streamed weights. Substituting indexedB1 is not a data-movement-only change. The measured mailbox and ND results do not justify attributing the gap to empty turns or bank placement again.
