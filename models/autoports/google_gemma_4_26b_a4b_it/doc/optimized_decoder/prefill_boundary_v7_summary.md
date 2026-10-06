# Current full-attention short/tail prefill timings

Current v7 selected defaults at `daa82a4a5197a007ccd5d29d912f082e695fd37a596578f25fba96a6694e625b`.
Median of three warmed synchronized whole-prefill host calls; these are not device latencies.
Existing fused measurements use identical fixture bytes and logical/physical geometry but a separate earlier process.
All prefill and one-step traced decode PCC, repeat-equality and program-cache guards pass.

| Logical rows | Physical chunks | Fused host µs | Current optimized host µs | Prefill PCC | Decode PCC |
| --- | --- | ---: | ---: | ---: | ---: |
| 65 | [96] | 68676.759 | 5621.885 | 0.9989912333 | 0.9994158892 |
| 1023 | [1024] | 705745.995 | 45778.007 | 0.9990422708 | 0.9981601235 |
| 1024 | [1024] | 706077.585 | 46052.288 | 0.9990418858 | 0.9967590270 |
| 1025 | [1024, 32] | 729475.659 | 47765.071 | 0.9990396140 | 0.9991662109 |

The original [paired workload boundary study](prefill_boundary_summary.md) retains both kinds at v5.
Sliding prefill arithmetic/programs remain unchanged at v7; final-tail retention removal is separately tested.
Small differences between historical optimized and current medians are not isolated fidelity measurements;
[alternating QKV controls](minimal_pairs_v6_summary.md) provide that attribution at the4096-token target.
Current commands and source-bound report hashes are in [the journal](v7_boundary_commands.json).
The independent tight1025/cache1152 Watcher probe uses the actual native-call read-bound guard;
it is a correctness test, not a timing result.
