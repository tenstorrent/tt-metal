# Current short and tail prefill timings

Full-attention v8 defaults, matched actual-weight fixtures against fused. These are warmed complete-prefill **host** medians, not device telemetry. Sliding v5 boundary results remain inherited through the source-delta proof. All cases pass prefill and following decode.

| Logical tokens | Fused host µs | Optimized host µs | Prefill PCC | Decode PCC |
| --- | ---: | ---: | ---: | ---: |
| 65 | 68676.759 | 5757.513 | 0.9989912333 | 0.9994158892 |
| 1023 | 705745.995 | 45255.106 | 0.9990422708 | 0.9981601235 |
| 1024 | 706077.585 | 45306.504 | 0.9990418858 | 0.9967590270 |
| 1025 | 729475.659 | 47727.450 | 0.9990396140 | 0.9991662109 |

Exact hashes: `prefill_boundary_v8_summary.json`. Commands: `v8_boundary_commands.json`; fused controls: `prefill_boundary_commands.json`.
