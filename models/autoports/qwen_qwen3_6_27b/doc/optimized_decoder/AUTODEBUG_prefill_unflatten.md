# Remove mapping-table work from packed-input prefill unflatten

Source-only candidate over `integrated134.py`. No device was opened. Main source and existing candidates remain unchanged.

## Why this target

Parent's pair135 packed-correction candidates all pass but lose to tuned separate projection: best K8/W33/SW3 prefill 6.868 ms B1 and 28.083 ms B32 versus integrated134 6.7487/26.4616 ms. SwiGLU input-L1 also loses. These are retained rejections; this candidate does not revisit either arithmetic/config change.

`perf/final118_profile_profile_l0_b32_s33/prefill_perf_report.csv` contains six `ReshapeViewDeviceOperation` rows totaling 3.715 ms. Global call1027072 alone takes **1.6306 ms**, converting corrected BF16 packed input `[1,1056,16512]` to `[32,33,16512]` (padded sequence64). The other rows are input/output/MLP flatten or unflatten operations of widths5120/6144. The single largest reshape is a more material target than further reducing the roughly49 us fused decode affine kernel, so this scoped experiment targets only it.

This source diagnosis is supported by the exact native kernels in the profiler. `reshape_view/device/device/dataflow/reader_reshape_tiled.cpp` reads a DRAM mapping page for every output tile, waits, then iterates mapping segments and individually reads their source tiles. The writer consumes mapping entries, performs segment memmoves, waits at input-page transitions and waits again before each output write. The logical mapping here is simple arithmetic because width remains tile-aligned and only sequence padding changes between batches. A persistent DRAM map is unnecessary for this case.

## Candidate

- Helper: `candidates/prefill_unflatten_bf16.py`, function `prefill_unflatten_bf16`.
- Frozen integration: `candidates/prefill_unflatten.py`, generator `make_prefill_unflatten.py` and generation JSON.
- Policy: `prefill_unflatten_packed=True`, optional `prefill_unflatten_cores`; defaults remain disabled.

The integration changes only the final reshape in `corrected_projection_chunk_add_prefill`, only when key is `packed_gdn`, B>1 and S is not divisible by32. All existing class methods, arithmetic, weight tensors, matmul configs, intermediate BF16 boundary and recurrence are identical. B1 and aligned S retain the previous operation. Compact S1 prefill and decode do not call this chunk-add helper. The policy's S%32 check selects the optimized implementation; it does not restrict legal public lengths.

For each output tile, reader computes its user, sequence-tile and width-tile indices. It converts the first logical output row to the flat input row, reads one or two adjacent whole input tiles into private scratch, and copies the exact uint32 words representing BF16 pairs to their destination tile-face rows. Invalid output padding is explicitly zero. Full aligned32-row spans use a direct input-to-output-CB read. No arithmetic or datatype conversion occurs. It then publishes the output tile to the independent writer; the output CB has two pages, permitting read/assembly to overlap writes. Scratch is two pages reserved only by the reader and never exposed to the writer. All reads/writes are2048-byte transfers at aligned tile addresses, followed by barriers before data reuse. No compute kernel is needed or supplied. Total CB allocation is8KiB per active core.

This replaces one device op with one device op. It reduces map traffic, map interpretation and segment synchronization, rather than eliminating the compulsory repack. Persistent additional DRAM storage is zero. Output allocation and padded shape match the original reshape. Optional supplied output is supported with explicit no-alias and shape checks, allowing stable replay storage; the integrated path uses normal trace-captured allocation. Width is unchanged and must be a multiple of32; the model's16512 width satisfies that contract.

## Verification and limits

`tests/test_prefill_unflatten_source.py` executes without TTNN import. It checks108 shapes spanning B1/3/31/32, S1/17/31/32/33/65/511/512/513, and widths32/96/16512, including27,202 sampled output tiles at first/middle/last width columns. Unique raw uint32 IDs verify exact source-word mapping, every source tile bound and zero output padding. Worker partitions cover output tiles once. AST comparison proves all existing definitions are unchanged except the one guarded final movement replacement, and removing that guard restores the original helper. Result: `prefill_unflatten_cpu_checks.json`. Python compilation passes.

Authored, not executed: `tests/run_prefill_unflatten_probe.py --candidate models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/prefill_unflatten_bf16.py --output <artifact>`. It compares exact BF16 bits to both CPU logical reshape and native TTNN reshape for10 cases, including B32/S33/N16512, B3/S513 and arbitrary tails. It repeats into the same output and checks stable address and equality. Diagnostic readbacks are outside performance claims. Parent must use the normal serialized device wrapper.

If that probe passes, compare whole-layer B32/S33 warmed prefill against integrated134 with identical precision and selected configs, followed by nonaligned prefix/decode state/PCC and512-step stress. B1 must remain unchanged; B32/S512 is aligned and intentionally remains unchanged. This experiment therefore cannot explain away the larger long-prefill precision cost (integrated134188.435 ms versus fused165.624 ms); that cost still needs separate evidence-backed optimization or reporting. No speedup, device compilation, trace or watcher result is claimed here.

If this narrow helper wins, its same byte-only mapping can be considered later for the other unflatten rows; those are not modified in this candidate. Flatten requires a different padding-removal mapping and is likewise not implemented here. The duplicate64-byte reads across adjacent decode-affine heads remain a secondary hand-off opportunity, not an unmeasured claimed improvement.
