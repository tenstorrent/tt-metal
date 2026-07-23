# GPT-OSS 20B optimized decoder

This stage provides the single-device optimized decoder layer in
`tt/optimized_decoder.py`. It starts from `tt/fused_decoder.py`, owns the
measured prefill and decode forwards, and does not include multichip,
full-model, or vLLM work.

## Selected path

The batch-one path uses:

- packed BF16 QKV and O projections with advisor-seeded 1D L1 layouts;
- TTNN prefill/decode SDPA composites, with 8x8/k=32 decode SDPA; the
  full-attention S=128 accuracy boundary uses an FP32 manual attention
  candidate because the composite alone missed the real-weight PCC bar;
- layer-aware attention: a 128-token window for sliding layers and no window
  for full-attention layers;
- BFP8_B persistent K/V cache with BF16 current-token K/V inputs;
- BFP8_B/LoFi routed sparse gate, up, and down decode matmuls on 9x10 cores;
- BF16/HiFi2 sparse prefill for sliding layers and a decoder-local BF16 dense
  expert path for full-attention S=128;
- fused device-side clipped SwiGLU and an advisor-seeded L1 chain from the
  attention residual through post-attention norm, routing, sparse experts, and
  the final residual;
- automatic prefill matmul selection. Explicit S=128 8x4 and 10x4 2D configs
  were tried on real weights and rejected because they reduced PCC.

The measured batch-one decode path releases inherited dense expert tensors.
The full-attention layer retains only split gate/up/down tensors for its S=128
precision path; its decode still uses sparse experts. Batch two uses an
optimized-owned dense compatibility graph because the shared sparse module
supports batch one only. No selected path dispatches to a functional or fused
forward implementation.

## Correctness and context

Final real-weight results remain above the 0.99 functional bar:

| Layer kind | Prefill output PCC | Decode output PCC | Decode K/V PCC |
| --- | ---: | ---: | ---: |
| layer 12, sliding | 0.99195677 | 0.99563603 | 0.99981871 / 0.99980306 |
| layer 13, full | 0.99436448 | 0.99410810 | 0.99982619 / 0.99981078 |

The tests cover non-aligned S=3/17/33, synthetic S=128 prefill, both real
layer kinds, batch two, decode positions 3-18, deterministic reruns, and ten
trace replays with full cache-integrity checks. A separate real-weight
boundary test constructs a 256-entry cache and real-weight qualifies S=128 for
both layer kinds. Sliding/full prefill output PCC is 0.990790/0.990509 and
decode-at-position-128 PCC is 0.996168/0.994839. The full-attention layer does
not inherit the sliding window.

The supported context is 128 tokens; the HF provenance field remains 131072.
This reduction has a hard device limit: at S=131072, the selected full-layer
accuracy path needs at least 72,477,573,120 live bytes for repeated BF16 expert
input plus gate and up tensors, versus 34,178,731,008 allocator bytes, before
weights, attention, cache, or outputs. See `../context_contract.json` for the
formula and capacity probe.

## Performance

The final same-process gate uses real layer-12 weights, 20 warmed S=17
prefills, and 200 traced decode replays:

| Path | Prefill mean / median / min | Traced decode mean / median / min |
| --- | ---: | ---: |
| fused | 7.119405 / 7.102114 / 7.094515 ms | 6.051824 / 6.050705 / 6.038011 ms |
| **optimized** | **3.953124 / 4.010050 / 3.740721 ms** | **0.834353 / 0.834215 / 0.827959 ms** |

That is 44.5% lower mean prefill latency and 86.2% lower mean traced-decode
latency. The selected L1 sparse chain improves the preceding correct
DRAM-boundary result from 0.873282 to 0.834353 ms. An equal 20/200 HiFi2 expert
control measured 0.834611 ms, so LoFi remains selected; the 0.03% difference is
small, but it does not support replacing the faster mean.

The final real-weight Tracy capture includes directly comparable fused and
optimized windows plus an optimized S=128 prefill window:

| Profile window | Device ops | Op gaps | Total |
| --- | ---: | ---: | ---: |
| fused S=17 prefill | 7,028.379 us | 260.515 us | 7,288.894 us |
| optimized S=17 prefill | 3,513.175 us | 627.634 us | 4,140.809 us |
| fused traced decode | 5,855.744 us | 49.913 us | 5,905.657 us |
| optimized traced decode | 776.196 us | 77.466 us | 853.662 us |
| optimized S=128 prefill | 12,327.276 us | 530.800 us | 12,858.076 us |

In optimized decode, the three active-expert sparse rows total 343.563 us and
remain the dominant family. QKV/O take 72.566/58.333 us and SDPA takes 9.919
us. The active-expert report models their 4/32 sparsity and shows 55.3-58.8%
of 512 GB/s. Its small-subblock advice was exercised by the legal 30-core,
subblock-3 control (0.961806 ms); the selected 90-core grid has per-core-N=1,
so a wider subblock is not legal there.

The theoretical decode floor is 0.311792 ms: 159,637,504 compulsory bytes
(BF16 QKV/O/router, 4/32 BFP8 expert weights at 1088 bytes per 1024-element
tile, biases/norms, and a 32-token K/V tile) divided by 512 GB/s. The profiled
device work is 2.49x that floor. In the same Tracy replay, device work plus op
gaps is 0.853662 ms versus 0.872583 ms host wall; the remaining 0.018921 ms is
trace-call/signpost host overhead. The uninstrumented 200-replay mean is
0.834353 ms, 2.3% below the profiled total, consistent with profiler overhead
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
profiler are always run in separate processes. Final correctness, watcher,
runner, review, and commit results are recorded in `work_log.md`. The key
rereview evidence is retained in `logs/run_20260723_rereview_boundary.log`,
`logs/run_20260723_rereview_final_perf.log`,
`logs/run_20260723_rereview_tracy.log`, and
`logs/run_20260723_rereview_shard_advise.log`.
