# Gemma-4 26B A4B optimized decoder

This stage implements the single-device optimized decoder for
`google/gemma-4-26B-A4B-it`. It owns its prefill and decode entry points and
does not dispatch through the functional decoder's public forwards. Scope
ends here; multichip, full-model, and vLLM work are not included.

## Selected topology

The functional path already provides packed QKV, explicit sharded RMSNorm,
TTNN paged-cache update/fill, SDPA, and canonical sparse expert matmuls.
The optimized path additionally:

- packs the repeated-input dense gate/up projections once at setup, executes
  one linear, and splits the result on device;
- folds the two immutable router input scales once at setup, removing one
  decode/prefill multiply;
- selects sparse expert `in0_block_w=11` for gate, up, and down, preserving
  exact top-8 routing while substantially reducing traced decode latency;
- keeps attention, dense, cache, norm, and router tensors at their
  correctness-proven precision while selecting BF8/LoFi for all three sparse
  expert projections.

There is no `torch`, `from_torch`, `to_torch`, explicit tilize/untilize,
reshard, or host fallback call in the optimized forward source. The exact
device profile contains four `UntilizeWithUnpadding` rows and one
`TilizeWithValPadding` row per decode layer (about 1.1% combined). These are
internal boundaries of the selected TTNN matmul composites: unpadding
materializes logical projection extents and value-padding supplies the
tile-height-one decode input. They are required by the legal padded batch-1
matmul contract, not extra Python-level conversions. Non-aligned public
sequence lengths remain supported by internal tile padding and output
slicing.

## Operation-topology audit

| Current operation/topology | Candidate | Action | Evidence/conclusion |
|---|---|---|---|
| Q, K, V share one input | packed QKV | retain | already packed by the functional decoder; avoids two projections |
| dense gate/up share one input | packed gate/up | select | PCC passed; batch-1 decode medians improved 3.021→2.989 ms sliding and 3.217→3.177 ms full |
| two consecutive immutable router scales | setup-time combined scale | select | removes one runtime multiply; exact functional PCC before packing |
| residual/RMSNorm/dense chain | coherent L1 width-sharded dense candidate | reject | on current source, BFP8/HiFi4 DRAM-sharded dense measured 2.093/2.296 ms at b1 and 39.034/38.817 ms at b32; slower than the identical sparse-11 control |
| paged attention | TTNN SDPA composite | retain | functional path already uses `paged_scaled_dot_product_attention` with an explicit program config |
| composite matmul layout boundaries | eliminate explicit conversion calls; retain composite-internal logical unpadding/tile-height padding | retain | exact profiler shows four untilize-with-unpadding plus one tilize-with-value-padding rows, about 1.1% combined; source has no explicit conversion, and the rows implement logical projection extents and the legal padded batch-1 input contract |
| sparse MoE | `in0_block_w={2,11,22,44}` | select width 11 | width 11 gives 2.048/2.241 ms b1 and 38.955/38.758 ms b32; width 2 is slower, width 22 is slightly slower at b1 and exceeds L1 at b32, and width 44 exceeds physical L1 at b1 |
| BF8 projection weights | group-local BF8 | select expert-only | sparse expert BF8 with explicit LoFi passes every representative PCC case and wins b1/b32; unrelated dense and attention BF8 policies remain rejected by their recorded evidence |
| BF4 projection weights / LoFi | executable sparse expert gate/up/down BF4/LoFi | reject | exact adapted real-weight PCC is below 0.995 for both layer kinds: sliding prefill/decode 0.993716/0.994384 and full 0.992966/0.993189 |
| BF8 compute fidelity | executable sparse expert BF8 HiFi2 versus LoFi | select LoFi | both pass the real-weight bar; LoFi is faster at b1 and b32 for both kinds, and Tracy rows prove BF16×BF8 `SparseMatmulDeviceOperation` with `MATH FIDELITY=LoFi` |
| DRAM-sharded decode matmul | BF16 and BFP8/HiFi4 packed dense/down across legal geometries | reject | 8/6-core width-11 BF16 and BFP8 pass both PCC cases but lose whole-layer b1/b32 timing; wider 4/3-core width-22 and 2/2-core width-44/33 BFP8 candidates require 2,864,128 and 10,475,264 B/core versus 1,572,864 B available |
| large prefill matmul programs | explicit 6x8 packed gate/up config | reject | current-source warmed prefill is 679.074 ms sliding and 680.609 ms full versus sparse-control 679.421 and 680.398 ms; mixed sub-0.06% deltas are noise, not a repeatable win |
| cache BF8 | lower-movement/higher-capacity cache | reject | PCC and b1/b32 trace determinism pass, and tiled cache storage falls 46.875%; exact-current-source five-run medians are noise-scale, but BF8 does not beat BF16 for either primary batch-1 layer kind |

The first legacy Tracy enrichment produced invalid negative durations and
wrong architecture metadata, so its raw scratch output was removed and no
latency conclusion uses it.
A dedicated optimized `--device-trace-profiler` capture covers both layer
kinds and both batches with positive Blackhole durations. Isolated
`--op-support-count=10000` captures also produced valid current per-op
Blackhole reports for sliding and full-attention batch 1. Exact commands,
material-row conclusions, and roofline/device/host reconciliation are in
`tracy/provenance.md`.

## Correctness

The real-weight acceptance floor is PCC 0.995 for every meaningful layer
kind.

| Layer/cache case | Prefill PCC | Decode PCC |
|---|---:|---:|
| layer 0 sliding, shared physical | 0.998620 | 0.999628 |
| layer 5 full, natural physical | 0.998480 | 0.999784 |
| layer 5 full, shared physical | 0.998480 | 0.999784 |

Traced decode passed at batch 1 and serving batch 32 for both layer kinds,
with eager/replay and repeat determinism PCC 1.0. Batch-2 prefill,
non-aligned boundaries, bounded modulo-tail integrity, and shared full-cache
mapping passed. The context allocation and cache dtype/layout are unchanged,
so `doc/context_contract.json` remains valid at 262144 tokens.

One captured trace per layer kind was also replayed with mutable input
sequence A/B/A under `TT_METAL_WATCHER=10`. A1 versus A2 PCC was 1.0 while
A versus B PCC was 0.635 sliding and 0.446 full, proving the trace consumed
new input rather than stale buffers. The retained
`watcher_bfp8_lofi_final.log` shows watcher initialized with no disabled
features, all nine selected-path cases passed, and all devices detached
cleanly.

## Performance

Numbers are five-run same-process medians at sequence 1024. Prefill is warmed
host latency; decode is traced replay host latency.

| Layer | Batch | Functional baseline ms | Optimized ms | Delta |
|---|---:|---:|---:|---:|
| sliding prefill | 1 | 678.777 | 669.805 | -1.32% |
| sliding decode | 1 | 3.021 | 1.848 | -38.82% |
| sliding decode | 32 | 68.945 | 32.076 | -53.48% |
| full prefill | 1 | 680.550 | 671.090 | -1.39% |
| full decode | 1 | 3.217 | 2.039 | -36.62% |
| full decode | 32 | 68.740 | 31.938 | -53.54% |

The primary batch-1 decode target beats the best correct baseline for both
representative layer kinds. Serving batch 32 also does not regress. Prefill
improves modestly after reducing expert weight traffic. Machine-readable
whole-layer candidates are under
`candidates/whole_layer/`, with exact grids, shards, block widths, code hashes,
commands, hardware, PCC artifact paths, and timing payloads.

The earlier large-prefill artifacts reported approximately 2.98/3.20 ms
decode while sparse width 11 reported approximately 2.05/2.24 ms. That was
not a prefill program affecting decode: their recorded source hashes showed
the large-prefill run predated the optimized width-11 sparse MoE override.
The current-source coherent rerun explicitly records
`inherited_sparse_in0_block_w=11` for every non-sparse candidate. It measures
large-prefill decode at 2.069/2.244 ms b1 and 38.955/38.734 ms b32, within
run noise of the current sparse-11 control at 2.051/2.240 ms and
38.983/38.738 ms.

The exact-shape BF8 KV-cache candidate passed real-weight PCC and traced
determinism at batches 1 and 32. A final matched rerun on optimized-decoder
SHA `5604d3ec748b` and test SHA `a0ad898e5f66` measured BF16→BF8 decode
medians of 2.050574→2.050694 ms (+0.0059%) for sliding batch 1,
38.979322→38.969811 ms (-0.0244%) for sliding batch 32,
2.241326→2.242059 ms (+0.0327%) for full batch 1, and
38.763732→38.749591 ms (-0.0365%) for full batch 32. Prefill differences
were within 0.03%. These changes are noise-scale, but BF8 does not beat the
BF16 control for either primary batch-1 case.
BF8 would reduce tiled cache storage from 2048 to 1088 bytes per tile
(46.875%), saving 25.78125 GiB across the model's 25 sliding and five full
layer cache pairs at context 262144. Capacity does not justify regressing
the primary latency target in this decoder stage, so BF16 remains selected
and `doc/context_contract.json` is unchanged. Machine-readable calculations
and all A/B artifacts are under `candidates/kv_cache_bfp8/`.

The exact-current aggregate capture measures 1.815 ms sliding and 2.003 ms
full on device, consistent with profiler-instrumented host measurements of
1.886 and 2.084 ms and uninstrumented final medians of 1.848 and 2.039 ms.
Sparse expert matmuls remain the largest class at 41.48%/37.44%, after being
attacked with widths 2/11/22/44; dense matmuls are 19.86%/24.10% and their
coherent DRAM-sharded candidate lost whole-layer timing. Layer norms are
16.71%/15.10%, SDPA is 2.48%/2.52%, and layout/cache-update rows are
sub-percent. See `tracy/provenance.md`; invalid legacy timings are never used.

## Reproduction

```bash
GEMMA4_RANGE_DOWNLOAD=1 pytest -q \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_optimized_decoder.py

TT_METAL_WATCHER=10 GEMMA4_RANGE_DOWNLOAD=1 pytest -q \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_optimized_decoder.py::test_optimized_real_weights_prefill_decode \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_optimized_decoder.py::test_optimized_traced_decode_contract

GEMMA4_RANGE_DOWNLOAD=1 GEMMA4_OPTIMIZED_DECODER_REPEAT_PERF=1 pytest -q \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_optimized_decoder.py::test_optimized_repeated_perf
```

The precision sweep is opt-in with
`GEMMA4_OPTIMIZED_PRECISION_SWEEP=1`; expected rejected cases fall below the
acceptance bar and are recorded as candidate evidence rather than CI gates.
The executable fidelity timing sweep uses
`GEMMA4_OPTIMIZED_FIDELITY_REPEAT_PERF=1`. Five-run BF8 HiFi2→LoFi decode
medians are 1.842→1.835 ms sliding b1, 32.408→32.068 ms sliding b32,
2.035→2.026 ms full b1, and 32.302→31.925 ms full b32. Warmed prefill is
effectively unchanged at 669.749→669.828 ms sliding and
670.982→671.016 ms full. See `candidates/fidelity_decision.json`.

The winning expert BF8/LoFi policy is now the production selected default.
On final source SHA `3589c270b4c8` and test SHA `3e09e6704bc5`, its five-run
medians are 1.848 ms sliding b1, 32.076 ms sliding b32, 2.039 ms full b1,
and 31.938 ms full b32; warmed prefill is 669.805/671.090 ms. All three
real-weight PCC cases, all four b1/b32 trace cases, and all nine watcher
selected-path cases pass. This changes expert weight storage only, not
KV-cache allocation, so the context contract is unchanged.
