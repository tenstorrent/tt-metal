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
- a bounded FP32 sink-aware accuracy guard at decode position 127, followed
  by native TTNN decode SDPA from position 128 onward; HiFi2 QKV/O
  projections remain selected across this boundary to prevent a small
  projection error from changing a top-4 expert;
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
prefill/decode at both aligned S=128 and non-aligned S=129. It also performs
five consecutive updates at positions 127-131 for both layer kinds. Boundary
output PCC is 0.991696/0.993322 prefill and 0.996778/0.999416 decode at S=128
for sliding/full layers; at S=129 it is 0.991753/0.995418 and
0.996560/0.999379. Across repeated positions 127-131, the minimum output PCC
is 0.994946/0.998354 and all appended K/V PCC values exceed 0.99988. The
full-attention layer does not inherit the sliding window.

Real-weight sliding/full qualification also passes all metrics at S=2048:
prefill output PCC is 0.992899/0.991286, decode-at-position-2048 is
0.994725/0.999403, prefill K/V PCC is at least 0.999949, and appended-token
K/V PCC is at least 0.999897. A progressive real-weight full-layer capacity
sweep passed S=4096, 8192, 16384, 32768, 65536, and the HF-advertised
S=131072 endpoint. The final endpoint run produced finite first/last output
tokens in 32.37 s. The bounded projection, attention, cache-write, and
sparse-expert chunks therefore restore
`current_supported_context` to 131072; no lower public cap or divisibility
restriction is introduced. See `../context_contract.json`.

## Performance

The final same-process gate uses real layer-12 weights, 20 warmed S=17
prefills, and 500 traced decode replays:

| Path | Prefill mean / median / min | Traced decode mean / median / min |
| --- | ---: | ---: |
| fused | 7.113153 / 7.096621 / 7.089853 ms | 5.898925 / 5.898392 / 5.884455 ms |
| **optimized** | **3.688937 / 3.632208 / 3.614589 ms** | **0.811755 / 0.811888 / 0.801848 ms** |

That is 48.1% lower mean prefill latency and 86.2% lower mean traced-decode
latency. The selected L1 sparse chain improves the preceding correct
DRAM-boundary result from 0.873282 to 0.811755 ms. In its paired 20/200 tuning
run, a HiFi2 expert control measured 0.834611 ms versus 0.834763 ms for LoFi;
that effectively tied control did not show an accuracy gain, so LoFi remains
selected on the hot path. The boundary-only HiFi2 attention projections are
not executed by the S=17 trace.

The final real-weight Tracy capture includes directly comparable fused and
optimized windows plus an optimized S=128 prefill window:

| Profile window | Device ops | Op gaps | Total |
| --- | ---: | ---: | ---: |
| fused S=17 prefill | 7,025 us | 245 us | 7,270 us |
| optimized S=17 prefill | 3,511 us | 454 us | 3,965 us |
| fused traced decode | 5,859 us | 49 us | 5,908 us |
| optimized traced decode | 776 us | 79 us | 855 us |
| optimized S=128 prefill | 12,357 us | 352 us | 12,708 us |

In optimized decode, the three active-expert sparse rows remain the dominant
family and QKV/O take approximately 72/59 us. The active-expert report models
their 4/32 sparsity and shows 55-59% of 512 GB/s. Its small-subblock advice
was exercised by legal 45-core/subblock-2 and 30-core/subblock-3 O-projection
controls:

| O program (`per_core_N`, `in0_block_w`, subblock W) | Traced decode mean |
| --- | ---: |
| **adjacent selected 90-core control (1, 8, 1)** | **0.811668 ms** |
| 45-core (2, 8, 2) | 0.812616 ms |
| 45-core (2, 32, 2) | 0.813712 ms |
| 30-core (3, 8, 3) | 0.814129 ms |
| 30-core (3, 32, 3) | 0.816731 ms |

Both the fastest 45-core control and the extreme 30-core control pass
real-weight correctness for both layer kinds. In the targeted Tracy rows, the
45-core O matmul is 59.368 us and its required return to the 90-core residual
layout is 1.490 us, versus 58.721 us for the selected 90-core O matmul with no
O-specific reshard. The wider-subblock family therefore does not improve the
complete traced path and remains a measured rejection.

The theoretical decode floor is 0.311792 ms: 159,637,504 compulsory bytes
(BF16 QKV/O/router, 4/32 BFP8 expert weights at 1088 bytes per 1024-element
tile, biases/norms, and a 32-token K/V tile) divided by 512 GB/s. The profiled
device work is 2.49x that floor. In the final Tracy replay, device work plus
op gaps is 0.855211 ms. The uninstrumented 500-replay mean is 0.811755 ms,
5.1% below the profiled total, consistent with profiler overhead and run
variance. Required routing `untilize`/`tilize` operations follow the
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
- `shard_advise/decision_trace/decode_decision_trace.json.gz`: compressed
  advisor decision trace referenced by the report.
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
OPTIMIZED_DECODER_TRACE_REPLAYS=500 \
pytest -q models/autoports/openai_gpt_oss_20b/tests/test_optimized_decoder.py::test_optimized_beats_fused_warmed_prefill_and_traced_decode -s
```

`work_log.md` contains the complete topology audit, command ledger, candidate
dispositions, advisor mapping, profiler interpretation, limitations, and the
completed optimize checklist. Compact human-readable and CSV profiler reports
for fused, optimized, and S=128 windows are under `tracy/final/`. Watcher and
profiler are always run in separate processes. Final correctness, capacity,
watcher, runner, review, and commit results are recorded in `work_log.md`.
Final post-review AutoFix JUnit evidence is retained as
`logs/autofix_final_o_search_correctness.junit.xml`,
`logs/autofix_final_o_search_perf.junit.xml`,
`logs/autofix_final_o_search_watcher.junit.xml`, and
`logs/autofix_final_capacity_131072.junit.xml`. The repaired S=2048
all-metrics result is in `logs/autofix_native_chunked_boundary_2048.junit.xml`;
the latest compact device-profiler reports are under `tracy/final/`, the O
geometry comparison is under `tracy/o_geometry/`, and the position-129
manual/native comparison is under `tracy/boundary/`.
