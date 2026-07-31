# google/gemma-4-12B optimized decoder

Repo revision: `31b45719e2ca21b695a8e7f15b5e8895bc1fb3bb`
Date: 2026-06-08
Target: single-chip TTNN autoport decoder stage for `google/gemma-4-12B`

## Scope

This directory records the optimized-decoder state for the repo-local
autoport pipeline. The implementation is limited to:

- `models/autoports/google/gemma-4-12B/tt/optimized_decoder.py`
- `models/autoports/google/gemma-4-12B/tests/test_optimized_decoder.py`
- `models/autoports/google/gemma-4-12B/doc/optimized_decoder/*`

No multichip decoder, full-model, or vLLM serving work is included.

## Optimized runtime policy

The optimized decoder keeps the functional decoder contract: same HF layer
semantics, paged prefill/decode KV-cache behavior, 2D decode RoPE, trace replay
compatibility, and representative coverage for `sliding_attention` and
`full_attention` layers.

Selected configs:

| Area | Final setting |
| --- | --- |
| Decode residuals/norms | Width-sharded L1 decode residual stream; RMSNorm1D decode in/out sharded. |
| Prefill residuals/norms | DRAM/interleaved prefill activations. |
| MLP prefill | BF16 DRAM-interleaved weights, default TTNN prefill linear path, fused GELU*up multiply, BF16 activations. |
| MLP decode | BFP8 DRAM-sharded gate/up/down weights, `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`, L1 width-sharded outputs. |
| Sliding attention prefill | BF16 DRAM-interleaved QKV/O weights with default TTNN prefill linear/SDPA path. |
| Sliding attention decode | QKV BFP8, O BF16, KV cache BF16, DRAM-sharded decode matmuls, paged SDPA decode; QKV uses HiFi3/fp32 accumulation only after the sliding-window boundary where PCC required it. |
| Full attention prefill | BF16 DRAM-sharded QKV/O weights with explicit large prefill program configs and SDPA config. |
| Full attention decode | BF16 QKV/O, BF16 KV cache, DRAM-sharded decode matmuls, paged SDPA decode. |
| MoE | Not applicable: `google/gemma-4-12B` is dense. |
| CCL/fused CCL | Not applicable for this single-chip TP=1 optimized-decoder stage. |

The optimized source audit rejects direct runtime fallback tokens in
`optimized_decoder.py`: `import torch`, `ttnn.from_torch`, `ttnn.to_torch`, and
`FunctionalDecoder`. Tracy signposted windows report `0 host ops` for both
prefill and decode.

The remaining `InterleavedToSharded`, `ShardedToInterleaved`,
`ReshardDeviceOperation`, `Tilize`, and `Untilize` device ops in the measured
decode windows are not host fallbacks. They are current TTNN op-contract
conversions around sharded residual entry, QKV head split, 2D decode RoPE
embedding/tilization, cache update, concat, and residual layout boundaries.
Trials to replace parts of this path, including 4D decode RoPE and interleaved
decode O projection, did not improve the accepted path.

## Correctness

Functional acceptance bars are preserved for every meaningful layer kind.
Synthetic rows use paged prefill followed by paged decode with permuted page
tables and rank-2 cache positions. Long rows use sequence length 1024 to cover
the sliding-window boundary.

| Layer | Seq | Functional prefill | Optimized prefill | Prefill bar | Functional decode | Optimized decode | Decode bar |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sliding_attention | 128 | 0.9973848698 | 0.9976081783 | 0.995 | 0.9933475834 | 0.9954114151 | 0.993 |
| full_attention | 128 | 0.9958167831 | 0.9970855036 | 0.995 | 0.9968326543 | 0.9953398643 | 0.995 |
| sliding_attention | 1024 | 0.9937211762 | 0.9936641457 | 0.992 | 0.9942062761 | 0.9967197443 | 0.992 |
| full_attention | 1024 | 0.9924927439 | 0.9951159032 | 0.992 | 0.9939581613 | 0.9929069739 | 0.992 |

Trace replay and determinism:

| Layer | Trace replay PCC | Replay bar | Repeated replay PCC | Determinism bar |
| --- | ---: | ---: | ---: | ---: |
| sliding_attention | 0.9954114151 | 0.993 | 1.0000000000 | 0.9999 |
| full_attention | 0.9953398643 | 0.995 | 1.0000000000 | 0.9999 |

Stress and real-weight coverage:

| Coverage | Evidence |
| --- | --- |
| Sliding stress | 3 repeated runs, all 0.9976081783 prefill / 0.9954114151 decode. |
| Full stress | 3 repeated runs, all 0.9970855036 prefill / 0.9953398643 decode. |
| Real checkpoint layer 0 | 32-token sliding layer: 0.9996062884 prefill / 0.9998653380 decode. |
| Watcher | `TT_METAL_WATCHER=10` optimized correctness run passed with `TT_METAL_WATCHER_DISABLE_ETH=1`. |

The material correctness delta is full-attention short decode versus the
functional synthetic baseline: 0.9968326543 -> 0.9953398643. It remains above
the 0.995 functional acceptance bar. The accepted full-attention path keeps
BF16 attention weights and rejects the faster default/interleaved full prefill
attention variant because that variant dropped decode PCC below the bar.

## Warmed performance

Warmed prefill and traced warmed decode were profiled with Tracy signposts and
advice-enabled `tt-perf-report`. Functional numbers are from
`doc/functional_decoder/tracy/perf_summary.json`; optimized numbers are from
`doc/optimized_decoder/tracy/perf_summary.json`.

| Layer | Mode | Functional us | Optimized us | Delta us | Speedup | Host ops | Op gap us |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sliding | prefill | 3296.464 | 3294.000 | -2.464 | 1.0007x | 0 | 14 |
| sliding | traced decode | 2881.813 | 1374.000 | -1507.813 | 2.0974x | 0 | 67 |
| full | prefill | 3655.096 | 4404.000 | +748.904 | 0.8299x | 0 | 17 |
| full | traced decode | 3198.814 | 1740.000 | -1458.814 | 1.8384x | 0 | 68 |

Full-attention prefill is slower than functional. The faster
default/interleaved full attention prefill trial was rejected because it broke
the short decode PCC bar. Keeping the accepted BF16 DRAM-sharded full prefill
path preserves semantics and still improves full traced decode by 1.84x.

## tt-perf-report conclusions

Followed:

- Decode matmuls use DRAM-sharded weights and L1 width-sharded activations.
  `tt-perf-report` marks all sliding decode matmuls optimized and marks the
  full decode MLP matmuls optimized.
- Decode is traced in the measured window. Remaining high op-to-op gap advice
  is 3-4 us, about 0.2% of the decode window.
- SDPA and paged SDPA decode are used instead of a hand-built attention path.
- MLP GELU and multiply are fused through TTNN `mul(..., GELU)`.

Rejected or ruled out with evidence:

- Global BFP4 attention failed PCC: sliding decode 0.9538122188 and full
  prefill 0.9157667111.
- Global BFP4 MLP failed PCC: sliding decode 0.9891755304 and full decode
  0.9884254788.
- BFP8 KV cache failed long sliding decode PCC at 0.6177840, so BF16 KV cache
  is retained.
- Full default/interleaved attention prefill was faster in shape, but failed
  the default short decode bar: 0.9868869619 with TT cache and 0.9942429944
  with HF cache.
- BF16 MLP for the long sliding path caused L1 pressure/OOM in the long prefill
  gate/up matmul (`CB grow to 1906080 B > 1499136 B`).
- Sliding decode required a selective precision fix at long positions:
  HiFi3/fp32 QKV only after the sliding-window boundary plus BF16 O projection.
  Global BF16 attention or HiFi4/FP32 alternatives either worsened PCC or hit
  L1 limits.
- 4D decode RoPE, interleaved decode O projection, decode-only MLP BF16,
  unfused MLP GELU, and interleaved decode norms did not improve the accepted
  PCC/perf point.

Potential `tt-perf-report` advice improvements:

- Prefill reports suggest placing large prefill input 0 in L1. For these
  sequence-sized Gemma activations, the repo LLM guidance and measured trials
  favored DRAM/interleaved prefill except where full attention needed the
  explicit DRAM-sharded path for PCC.
- Decode reports still suggest tracing could save a tiny gap even though the
  measured decode window is trace replay. It would be useful for the report to
  identify traced windows from provenance.

## Artifacts

Primary evidence:

- PCC and trace/stress records:
  `models/autoports/google/gemma-4-12B/doc/optimized_decoder/pcc_results.jsonl`
- Warmed latency summary:
  `models/autoports/google/gemma-4-12B/doc/optimized_decoder/tracy/perf_summary.json`
- Sliding Tracy raw ops:
  `models/autoports/google/gemma-4-12B/doc/optimized_decoder/tracy/sliding/ops.csv`
- Full Tracy raw ops:
  `models/autoports/google/gemma-4-12B/doc/optimized_decoder/tracy/full/ops.csv`
- Advice-backed human-readable reports:
  `tracy/sliding/prefill_perf_report.txt`,
  `tracy/sliding/decode_perf_report.txt`,
  `tracy/full/prefill_perf_report.txt`,
  `tracy/full/decode_perf_report.txt`
- Advice-backed CSV reports and stacked summaries:
  `tracy/{sliding,full}/{prefill,decode}_perf_report.csv`,
  `tracy/{sliding,full}/{prefill,decode}_perf_report_stacked.csv`,
  `tracy/{sliding,full}/{prefill,decode}_perf_report_stacked.png`
- Tracy provenance:
  `tracy/sliding/raw/reports/2026_06_08_23_10_57/*`,
  `tracy/full/raw/reports/2026_06_08_23_11_21/*`
- Watcher provenance:
  `watcher/default_disable_eth/generated/watcher/watcher.log`

