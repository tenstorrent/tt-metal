# Complete coefficient-adapter microprobe

## Root-owned device results and decision

The root-owned [complete adapter probe](compact_pair_gather_probe.json) reports exact coefficient codes for all four variants at both K policies. These timings include every conversion and scalar-coefficient reshape:

| Complete adapter | K512 µs | K1024 µs |
|---|---:|---:|
| Original TILE |93.912065|93.925046|
| Compact TILE |95.495302|92.341506|
| Original ROW_MAJOR |195.002521|194.991019|
| Compact ROW_MAJOR |56.650513|56.681107|

The compact TILE candidate remains rejected by its full-decoder comparison. The compact ROW_MAJOR adapter was advanced: [integrated measurement](../compact_rm_gather_0.json) gives78.959430ms prefill /1.506119ms decode, versus79.140272/1.544202ms for the [matched scalar-norm control](../norm_scalar_fused_0.json), atlength128/context512 with10/60 samples. Short PCC is unchanged. The [seed154](compact_rm_gather_seed154_results.json), [seed42](compact_rm_gather_seed42_results.json), and [seed913](compact_rm_gather_seed913_results.json) runs each pass8192 decode tokens plus33 continuation rows with one replay. Their minimum decode contribution PCCs are0.996095021,0.995666140,0.996216458. These qualify the candidate against those streams; they do not replace final fresh-reference three-repeat validation or independent review.

The successful adapter untilizes the normalized input, reshapes it to64 rows of32 coefficients, gathers adjacent pairs in ROW_MAJOR, reshapes to the scalar-broadcast geometry, then tilizes once. All conversions are part of the measured gain. The preparation notes below are retained as the original hypothesis and test contract; hardware claims in this section come from the linked root-owned artifacts, not a device run by this investigation agent.

## Original preparation and reproduction

Source/CPU preparation only; no device result is claimed here. Parent reports compact-pair full trace1.564091ms versus recurrent-L1-prefill1.560764ms, so the initial full-decoder candidate is rejected on measured latency. This probe distinguishes gather savings from the extra reshape/conversion costs; an isolated microbenchmark win cannot overturn the full-decoder result by itself.

`tests/compact_pair_gather_probe.py` compares four complete adapters for both K512 and K1024. Every variant starts with BF16 TILE DRAM `[1,1,2048]` and returns BF16 TILE DRAM `[2048/K,K,1,1]`:

| Variant | Operations inside the measured trace |
|---|---|
| `original_tile` | DRAM placement, gather with global `[1,1,2048]` pair-swap indices, scalar-coefficient reshape |
| `compact_tile` | DRAM placement, reshape `[1,64,32]`, gather with per-row32-column pair-swap indices, scalar-coefficient reshape |
| `original_rm` | DRAM placement, untilize, global pair gather, RM scalar-coefficient reshape, tilize |
| `compact_rm` | DRAM placement, untilize, reshape `[1,64,32]`, per-row pair gather, RM scalar-coefficient reshape, tilize |

Static UINT16 indices are uploaded once in the matching input layout. The RM variants do not measure a setup-preconverted input: untilize and the final tilize are both inside capture. All variants include the final logical-scalar coefficient tensor's physical tile expansion. No projection multiply or reduction is included, making this an adapter discriminator rather than a decoder latency claim.

The native gather contract allows BF16 input with UINT16/UINT32 indices, requires identical data/index layout, and supports TILE or ROW_MAJOR (`gather/device/gather_device_operation.cpp:79–107`). Compact indices select `c xor1`; because32 is even, global address `32*r+(c xor1)` equals `(32*r+c) xor1`. Current `reshape_view/reshape.cpp:287` handles RM reshapes through device `reshape_view`; its device operation accepts BF16 and chooses the RM factory. There is no source-level prohibition on the scalar logical output width here. Installed-binary dispatch or tilize restrictions still need the root-owned device run; failures are retained per variant with the failing stage and traceback. No host fallback is introduced to bypass them.

CPU setup creates three deterministic random BF16 vectors normalized to representative RMS, plus2048 unique finite BF16 codes that expose every incorrect permutation. Both K policies compare every logical coefficient bit against CPU global pair reversal, including eager outputs and first/last warmed trace outputs. Only exact variants are timed. Input generation, uploads, readback, and static indices remain outside timing; runtime calls are enclosed by the existing device-only guard.

Timing uses five eager warmups, five warm trace replays, and30 samples of20 queued replays with synchronization at sample boundaries. The JSON preserves all samples, medians, original-adapter ratios, exact-code differences/examples, input hashes, probe/snapshot hashes, and a normal provenance sidecar. Watcher, profiler and allocation tracking must be disabled for clean latency. It opens physical chip0 only and preserves existing output unless `--overwrite` is explicitly supplied. This is diagnostic evidence, never an acceptance waiver.

Root-owned device command, after sourcing the lane environment and ensuring exclusive device ownership:

```bash
timeout 600 env -u TT_METAL_WATCHER -u TT_METAL_DEVICE_PROFILER \
  -u TT_METAL_TRACE_ALLOC_TRACKING -u TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE \
  QWEN_DECODER=fused python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.compact_pair_gather_probe \
  --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/compact_pair_gather_probe.json
```

Run that bounded command through the installed `scripts/wait_for` as with the other root-owned device jobs. Use `--variants original_tile compact_tile` for a TILE-only discriminator, or include the RM variants as the default does. `--oracle-only --output /tmp/compact_pair_cpu.json` performs the CPU permutation proof without importing TTNN or opening a device.

Preparation validation: all eight CPU comparisons (four inputs, two K policies) are bit-exact. Probe syntax and all four adapter paths were checked with a CPU fake-TTNN tensor implementation that enforces the input/index layout and final output shape/layout/memory contract. Hardware legality, exactness and latency remain unverified.
