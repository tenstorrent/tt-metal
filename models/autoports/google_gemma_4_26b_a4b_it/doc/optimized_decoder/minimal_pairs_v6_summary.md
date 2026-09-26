# Minimal QKV paired timing controls

Runtime: `b585a21f0b66144f69a823fa2d1088b130e34928a65fc91a38fd1bc5c2526846`. Status: **paired_controls_passed_selection_pending**.

Eight alternating synchronous whole-prefill host pairs per real4096/128 control. Both variants warmed twice; samples forbid cache misses and exclude initial sync/deallocation. Final candidate only is checked against HF in each paired run; original matched baseline screen is separate. This is not per-op device timing or proof of maximum/stress accuracy.

| Layer | Candidate | Baseline median µs | Candidate median µs | Median paired delta µs | Faster pairs | Prefill PCC | Minimum decode PCC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | hifi2 | 220393.494 | 219915.530 | -473.456 | 7/8 | 0.9991513837 | 0.9954320642 |
| 0 | grid110 | 220377.061 | 220474.737 | +4.303 | 4/8 | 0.9991560943 | 0.9952574859 |
| 5 | hifi2 | 187022.342 | 186436.443 | -545.573 | 8/8 | 0.9991223369 | 0.9951056876 |
| 5 | grid110 | 186929.054 | 187105.084 | +87.481 | 2/8 | 0.9991234309 | 0.9950717323 |

Negative paired deltas favor the candidate. Every sample is retained; medians and win counts describe these eight pairs without a confidence or hardware-cause claim. Candidate precision, full source hashes, raw samples and cache counts are retained in [the JSON](minimal_pairs_v6_summary.json). No policy is adopted by this report; fidelity winners require long-context and512-step stress checks before integration.
