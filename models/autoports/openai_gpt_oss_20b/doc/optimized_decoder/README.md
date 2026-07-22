# GPT-OSS 20B optimized decoder

This stage provides the single-device optimized decoder layer in
`tt/optimized_decoder.py`. It starts from `tt/fused_decoder.py`, owns the
measured prefill and decode forwards, and does not include multichip,
full-model, or vLLM work.

## Selected path

The batch-one path uses:

- packed BF16 QKV and O projections with advisor-seeded 1D L1 layouts;
- the TTNN prefill and decode SDPA composites, with 8x8/k=32 decode SDPA;
- layer-aware attention: a 128-token window for sliding layers and no window
  for full-attention layers;
- BFP8_B persistent K/V cache with BF16 current-token K/V inputs;
- BFP8_B/LoFi routed sparse gate, up, and down expert matmuls on 9x10 cores;
- fused device-side clipped SwiGLU and an advisor-seeded L1 chain from the
  attention residual through post-attention norm, routing, sparse experts, and
  the final residual;
- automatic prefill matmul selection. Explicit S=128 8x4 and 10x4 2D configs
  were tried on real weights and rejected because they reduced PCC.

The constructor releases inherited dense expert tensors after loading sparse
weights, so batch-one cannot silently use the dense functional path. Batch two
retains the exact fused dense compatibility branch because the shared sparse
expert module supports batch one only.

## Correctness and context

Final real-weight results remain above the 0.99 functional bar:

| Layer kind | Prefill output PCC | Decode output PCC | Decode K/V PCC |
| --- | ---: | ---: | ---: |
| layer 12, sliding | 0.99024635 | 0.99563603 | 0.99981871 / 0.99980306 |
| layer 13, full | 0.99308115 | 0.99410810 | 0.99982619 / 0.99981078 |

The tests cover non-aligned S=3/17/33, synthetic S=128 prefill, both real
layer kinds, batch two, decode positions 3-18, deterministic reruns, and ten
trace replays with full cache-integrity checks. A separate real-weight
boundary test constructs a 256-entry cache, prefills 128 tokens, and validates
decode at position 128 for both sliding and full attention (0.996112 and
0.994792 output PCC respectively). The full-attention layer therefore no
longer inherits the sliding window. Removing the quadratic dense mask and
accepting caller-selected cache extents restores the HF-advertised 131072-token
logical contract without exposing the narrower hardware validation boundary as
a public cap. See `../context_contract.json` for the exact distinction between
logical support and tested accuracy.

## Performance

The final same-process gate uses real layer-12 weights, ten warmed S=17
prefills, and 100 traced decode replays:

| Path | Prefill mean / min | Traced decode mean / min |
| --- | ---: | ---: |
| fused | 7.193785 / 7.098952 ms | 6.050208 / 6.033769 ms |
| **optimized** | **4.045302 / 3.934355 ms** | **0.834213 / 0.826016 ms** |

That is 43.8% lower mean prefill latency and 86.2% lower mean traced-decode
latency. The selected L1 sparse chain improves the immediately preceding
correct DRAM-boundary result from 0.873282 to 0.834213 ms.

The final real-weight Tracy capture includes directly comparable fused and
optimized windows plus an optimized S=128 prefill window:

| Profile window | Device ops | Op gaps | Total |
| --- | ---: | ---: | ---: |
| fused S=17 prefill | 7,026.773 us | 236.111 us | 7,262.884 us |
| optimized S=17 prefill | 3,671.552 us | 612.177 us | 4,283.729 us |
| fused traced decode | 5,857.295 us | 49.365 us | 5,906.660 us |
| optimized traced decode | 768.311 us | 77.188 us | 845.499 us |
| optimized S=128 prefill | 12,866.761 us | 375.417 us | 13,242.178 us |

In optimized decode, the three active-expert sparse rows total 336.302 us and
remain the dominant family. QKV/O take 72.363/58.256 us and SDPA takes about
10 us. The routing `untilize`/`tilize` operations are required by the TopK and
sparse-matmul row-major contract; there is no host conversion or fallback in
the measured path.

## Advisor artifacts

The mandatory advisor was rerun on the rewritten dense attention+MLP graph.
Its BFP8 cache inputs are metadata-only tracer arguments, avoiding a known
host packed-conversion bug, and the unsupported cache mutation is threaded as
an identity because it has no tunable layout; runtime paged-cache semantics are
covered separately by device tests.

- `shard_advise/report.json`: 39 modeled ops, 38 choices, 24 reshards, zero
  spills, zero unfixable ops.
- `shard_advise/final_ir.mlir`: authoritative final graph, including BFP8
  cache argument types.
- `shard_advise/advise_gpt_oss.py`: reproducible dense capture harness.

Advisor attention, norm, residual, and router recommendations are applied.
The dense all-expert MLP was rejected against routed sparse execution. The
advisor-inspired continuous sparse L1 chain was corrected after a typecast
layout error, then selected because it improved traced decode.

## Validation commands and evidence

```bash
pytest -q models/autoports/openai_gpt_oss_20b/tests/test_optimized_decoder.py -s

RUN_OPTIMIZED_DECODER_PERF=1 \
OPTIMIZED_DECODER_PREFILL_REPEATS=10 \
OPTIMIZED_DECODER_TRACE_REPLAYS=100 \
pytest -q models/autoports/openai_gpt_oss_20b/tests/test_optimized_decoder.py::test_optimized_beats_fused_warmed_prefill_and_traced_decode -s
```

`work_log.md` contains the complete topology audit, command ledger, candidate
dispositions, advisor mapping, profiler interpretation, limitations, and the
completed optimize checklist. Compact human-readable and CSV profiler reports
for fused, optimized, and S=128 windows are under `tracy/final/`. Watcher and
profiler are always run in separate processes. The final full suite is 10
passed and 2 intentional opt-in skips; the separate watcher run is 6 passed
and 6 deselected with fallback exceptions enabled. Exact output is retained in
`logs/run_20260722_review_final_correctness.log` and
`logs/run_20260722_review_watcher.log`.
