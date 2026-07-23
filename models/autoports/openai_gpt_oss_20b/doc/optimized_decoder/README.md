# GPT-OSS 20B optimized decoder

This stage provides the single-device optimized decoder layer in
`tt/optimized_decoder.py`. It starts from `tt/fused_decoder.py`, owns the
measured prefill and decode forwards, and does not include multichip,
full-model, or vLLM work.

## Selected path

The batch-one path uses:

- packed BF16 QKV and O projections with advisor-seeded 1D L1 layouts;
- TTNN prefill/decode SDPA composites, with 8x8/k=32 decode SDPA on the hot
  traced path; full-attention S=128 uses exact FP32 attention, while longer
  prefill uses 128x128 SDPA chunks;
- layer-aware attention: a 128-token window for sliding layers and no window
  for full-attention layers;
- BFP8_B persistent K/V cache with BF16 current-token K/V inputs;
- BFP8_B/LoFi routed sparse gate, up, and down decode matmuls on 9x10 cores;
- BF16/HiFi2 sparse prefill for sliding layers, BF16/HiFi4 bounded sparse
  chunks for full-attention lengths above S=128, and a decoder-local BF16
  dense expert path for the sensitive full-attention S=128 point;
- bounded FP32 sink-aware attention plus HiFi2 QKV/O projections for decode
  positions at and beyond the emitted S=128 cache boundary; this prevents a
  small projection error from changing a top-4 expert at non-aligned S=129;
- fused device-side clipped SwiGLU and an advisor-seeded L1 chain from the
  attention residual through post-attention norm, routing, sparse experts, and
  the final residual;
- automatic prefill matmul selection. Explicit S=128 8x4 and 10x4 2D configs
  were tried on real weights and rejected because they reduced PCC.

The measured batch-one decode path releases inherited dense expert tensors.
The full-attention layer retains split gate/up/down tensors for its S=128
precision path; longer prefill and decode still use sparse experts. Batch two uses an
optimized-owned dense compatibility graph because the shared sparse module
supports batch one only. No selected path dispatches to a functional or fused
forward implementation.

## Correctness and context

Final real-weight results remain above the 0.99 functional bar:

| Layer kind | Prefill output PCC | Decode output PCC | Decode K/V PCC |
| --- | ---: | ---: | ---: |
| layer 12, sliding | 0.99854993 | 0.99605129 | 0.99981871 / 0.99980306 |
| layer 13, full | 0.99528336 | 0.99777805 | 0.99982619 / 0.99981078 |

The default suite covers non-aligned S=3/17/33, synthetic S=128 prefill, both
real layer kinds, batch two, decode positions 3-18, deterministic reruns, ten
trace replays with full cache-integrity checks, and real-weight boundary
prefill/decode at both aligned S=128 and non-aligned S=129. Boundary output
PCC is 0.991696/0.993322 prefill and 0.996978/0.999349 decode at S=128 for
sliding/full layers; at S=129 it is 0.991753/0.995418 and
0.996255/0.999360. The full-attention layer does not inherit the sliding
window.

Real-weight sliding/full qualification also passes at S=1024: prefill output
PCC is 0.992395/0.992225, decode-at-position-1024 is
0.996869/0.996587, and all reported cache PCC values exceed 0.995. A
progressive real-weight full-layer capacity sweep passed S=4096, 8192, 16384,
32768, 65536, and the HF-advertised S=131072 endpoint. The final endpoint run
produced finite first/last output tokens in 32.79 s. The bounded projection,
attention, and sparse-expert chunks therefore restore
`current_supported_context` to 131072; no lower public cap or divisibility
restriction is introduced. See `../context_contract.json`.

## Performance

The final same-process gate uses real layer-12 weights, 20 warmed S=17
prefills, and 200 traced decode replays:

| Path | Prefill mean / median / min | Traced decode mean / median / min |
| --- | ---: | ---: |
| fused | 7.291264 / 7.289830 / 7.099999 ms | 6.050883 / 6.050299 / 6.036739 ms |
| **optimized** | **3.876243 / 3.783550 / 3.733284 ms** | **0.834763 / 0.834537 / 0.823852 ms** |

That is 46.8% lower mean prefill latency and 86.2% lower mean traced-decode
latency. The selected L1 sparse chain improves the preceding correct
DRAM-boundary result from 0.873282 to 0.834763 ms. An equal 20/200 HiFi2 expert
control measured 0.834611 ms; that effectively tied control did not show an
accuracy gain, so LoFi remains selected on the hot path. The boundary-only
HiFi2 attention projections are not executed by the S=17 trace.

The final real-weight Tracy capture includes directly comparable fused and
optimized windows plus an optimized S=128 prefill window:

| Profile window | Device ops | Op gaps | Total |
| --- | ---: | ---: | ---: |
| fused S=17 prefill | 7,026.588 us | 312.331 us | 7,338.919 us |
| optimized S=17 prefill | 3,512.589 us | 458.704 us | 3,971.293 us |
| fused traced decode | 5,855.783 us | 49.530 us | 5,905.313 us |
| optimized traced decode | 775.810 us | 78.081 us | 853.891 us |
| optimized S=128 prefill | 12,323.612 us | 288.933 us | 12,612.545 us |

In optimized decode, the three active-expert sparse rows total 342.502 us and
remain the dominant family. QKV/O take 73.507/58.441 us and SDPA takes 9.762
us. The active-expert report models their 4/32 sparsity and shows 55.3-58.8%
of 512 GB/s. Its small-subblock advice was exercised by the legal 30-core,
subblock-3 control (0.961806 ms); the selected 90-core grid has per-core-N=1,
so a wider subblock is not legal there.

The theoretical decode floor is 0.311792 ms: 159,637,504 compulsory bytes
(BF16 QKV/O/router, 4/32 BFP8 expert weights at 1088 bytes per 1024-element
tile, biases/norms, and a 32-token K/V tile) divided by 512 GB/s. The profiled
device work is 2.49x that floor. In the same Tracy replay, device work plus op
gaps is 0.853891 ms versus 0.873356 ms host wall; the remaining 0.019465 ms is
trace-call/signpost host overhead. The uninstrumented 200-replay mean is
0.834763 ms, 2.3% below the profiled total, consistent with profiler overhead
and run variance. Required routing `untilize`/`tilize` operations follow the
TopK/sparse row-major contract; there is no host conversion or fallback in the
measured path.

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
OPTIMIZED_DECODER_PREFILL_REPEATS=20 \
OPTIMIZED_DECODER_TRACE_REPLAYS=200 \
pytest -q models/autoports/openai_gpt_oss_20b/tests/test_optimized_decoder.py::test_optimized_beats_fused_warmed_prefill_and_traced_decode -s
```

`work_log.md` contains the complete topology audit, command ledger, candidate
dispositions, advisor mapping, profiler interpretation, limitations, and the
completed optimize checklist. Compact human-readable and CSV profiler reports
for fused, optimized, and S=128 windows are under `tracy/final/`. Watcher and
profiler are always run in separate processes. Final correctness, capacity,
watcher, runner, review, and commit results are recorded in `work_log.md`.
Post-AutoFix JUnit evidence is retained as
`logs/run_20260723_post_autofix_final_correctness.junit.xml`,
`logs/run_20260723_post_autofix_final_perf.junit.xml`, and
`logs/run_20260723_post_autofix_final_watcher.junit.xml`; the latest compact
device-profiler reports are under `tracy/final/`.
