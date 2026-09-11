# B3: add direct SP8/TP1 MLA coverage and report operation priorities

Draft completion summary; not posted.

## Deliverable

The existing MLA perf rows measure TP=4, while a PP4 stage uses TP=1. B3 adds an eight-device SP8/TP1 MLA row and reports its measured operation durations directly. It does not use `approximate_mla_galaxy_perf`, which assumes TP=4.

The test runs one MLA forward with a 51,200-token KV prefix and 5,120 new tokens. It supports LoudBox and a topology-checked eight-device Galaxy column. The wrapper labels the Galaxy result separately and records results without a performance threshold.

Implementation: [worker](https://github.com/tenstorrent/tt-metal/blob/ssalice/mistral4-b3-findings/models/demos/deepseek_v3_d_p/tests/test_mla.py), [perf wrapper](https://github.com/tenstorrent/tt-metal/blob/ssalice/mistral4-b3-findings/models/demos/deepseek_v3_d_p/tests/perf/test_mla_perf.py), [measurement helper](https://github.com/tenstorrent/tt-metal/blob/ssalice/mistral4-b3-findings/models/demos/deepseek_v3_d_p/utils/perf_utils.py), and [mesh fixture](https://github.com/tenstorrent/tt-metal/blob/ssalice/mistral4-b3-findings/models/demos/deepseek_v3_d_p/tests/conftest.py).

## Local validation

The worker and perf wrapper passed on an eight-device Blackhole Galaxy column on September 11. The capture contains one measured MLA region and 24 operations on each of eight devices; all 192 device durations are finite and positive.

| Operation group | Measured sum | Share |
|---|---:|---:|
| SDPA | 7.195 ms | 83.81% |
| Matmuls | 0.738 ms | 8.59% |
| Other | 0.652 ms | 7.60% |
| **Total** | **8.585 ms** | **100%** |

This is an accumulated operation-duration metric, not elapsed MLA latency. It is one cold, untraced, random-weight forward without a numerical correctness check. All recorded operations missed the program cache. The result establishes that the row runs locally; it is not a steady-state baseline or a LoudBox calibration. Ring SDPA includes communication.

## TP=1 priorities

The historical full-layer TP=1 stage-0 capture gives a different view from the standalone MLA sample:

| Target | Share of accumulated device work |
|---|---:|
| MoE Dispatch + Combine | 35.65% |
| SDPA | 23.44% |
| All matmuls | 17.53% |
| Expert FFN | 15.77% |
| Residual collectives | 0.16% |

Start with MoE Dispatch/Combine and SDPA, then inspect projection matmuls and expert FFN. The matmul category includes operations outside MLA. These historical measurements use different inputs and a different aggregation method from the standalone sample; do not combine their absolute timings or treat their shares as expected throughput gains.

## Remaining work

B3 has the requested test row, direct local result and TP=1 ranking. A calibrated LoudBox threshold remains unverified. Useful follow-ups are repeated warm measurements across KV depths and correctness checks before changing SDPA or projection code.

## Supporting material

- [Implementation and validation](https://github.com/tenstorrent/tt-metal/blob/ssalice/mistral4-b3-findings/profiling_reports/2026-09-11/B3_IMPLEMENTATION.md)
- [Historical TP=1 ranking and method](https://github.com/tenstorrent/tt-metal/blob/ssalice/mistral4-b3-findings/profiling_reports/2026-09-11/B3_TP1_RANKING.md)
- [Direct MLA breakdown](https://github.com/tenstorrent/tt-metal/blob/ssalice/mistral4-b3-findings/profiling_reports/2026-09-11/b3-glx-column-PzSANj/DIRECT_BREAKDOWN.md) and [capture validation](https://github.com/tenstorrent/tt-metal/blob/ssalice/mistral4-b3-findings/profiling_reports/2026-09-11/b3-glx-column-PzSANj/validation.json)
- [Historical source hashes and timings](https://github.com/tenstorrent/tt-metal/blob/ssalice/mistral4-b3-findings/profiling_reports/2026-09-11/b3_evidence.json)
