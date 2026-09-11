# B1 isolated dispatch/combine matrix

Only successful manager-free captures with complete signposted coverage are included. Times are milliseconds; each row is one operation in one fresh process. Each process discards its initial iteration and retains nine. Device spread is max minus min of the eight device means.

| Run | Layer | Routing | Operation | Mean device max | Median device max | Device mean spread |
|---|---:|---|---|---:|---:|---:|
| run-layer18-captured-FfQ27z | 18 | captured | Dispatch | 1.656553 | 1.656191 | 0.032294 |
| run-layer18-captured-FfQ27z | 18 | captured | Combine | 1.725759 | 1.725790 | 0.024742 |
| run-layer18-captured-oywqw8 | 18 | captured | Dispatch | 1.661606 | 1.661018 | 0.038671 |
| run-layer18-captured-oywqw8 | 18 | captured | Combine | 1.720989 | 1.722851 | 0.022382 |
| run-layer18-placement_balanced-80cPJm | 18 | placement_balanced | Dispatch | 1.627108 | 1.625150 | 0.035146 |
| run-layer18-placement_balanced-80cPJm | 18 | placement_balanced | Combine | 1.059503 | 1.059553 | 0.021723 |
| run-layer18-placement_balanced-Q6NQuD | 18 | placement_balanced | Dispatch | 1.621753 | 1.617545 | 0.028202 |
| run-layer18-placement_balanced-Q6NQuD | 18 | placement_balanced | Combine | 1.065135 | 1.063076 | 0.024846 |
| run-layer18-source_shuffled-kU8nKy | 18 | source_shuffled | Dispatch | 1.658265 | 1.659311 | 0.031583 |
| run-layer18-source_shuffled-kU8nKy | 18 | source_shuffled | Combine | 1.654330 | 1.649806 | 0.028226 |
| run-layer18-source_shuffled-udDErE | 18 | source_shuffled | Dispatch | 1.662938 | 1.661058 | 0.035961 |
| run-layer18-source_shuffled-udDErE | 18 | source_shuffled | Combine | 1.652474 | 1.651821 | 0.022048 |
| run-layer23-captured-5aZxOp | 23 | captured | Dispatch | 1.644367 | 1.638446 | 0.040323 |
| run-layer23-captured-5aZxOp | 23 | captured | Combine | 1.478881 | 1.478541 | 0.027761 |
| run-layer23-captured-fYu5qV | 23 | captured | Dispatch | 1.637837 | 1.637266 | 0.033966 |
| run-layer23-captured-fYu5qV | 23 | captured | Combine | 1.475933 | 1.475849 | 0.025913 |
| run-layer23-placement_balanced-1vo9I5 | 23 | placement_balanced | Dispatch | 1.606034 | 1.605007 | 0.034229 |
| run-layer23-placement_balanced-1vo9I5 | 23 | placement_balanced | Combine | 1.099039 | 1.101550 | 0.029894 |
| run-layer23-placement_balanced-vmwx5a | 23 | placement_balanced | Dispatch | 1.606213 | 1.605631 | 0.035302 |
| run-layer23-placement_balanced-vmwx5a | 23 | placement_balanced | Combine | 1.092841 | 1.093795 | 0.022463 |
| run-layer23-source_shuffled-8K4ANx | 23 | source_shuffled | Dispatch | 1.628298 | 1.626467 | 0.038575 |
| run-layer23-source_shuffled-8K4ANx | 23 | source_shuffled | Combine | 1.483632 | 1.484077 | 0.030396 |
| run-layer23-source_shuffled-ORvhnV | 23 | source_shuffled | Dispatch | 1.625332 | 1.622755 | 0.034902 |
| run-layer23-source_shuffled-ORvhnV | 23 | source_shuffled | Combine | 1.474195 | 1.473440 | 0.023135 |

## Fresh-process run summaries

| Layer | Routing | Operation | Runs | Mean of run means | Run-mean range |
|---:|---|---|---:|---:|---|
| 18 | captured | Combine | 2 | 1.723374 | 1.720989–1.725759 |
| 18 | captured | Dispatch | 2 | 1.659080 | 1.656553–1.661606 |
| 18 | placement_balanced | Combine | 2 | 1.062319 | 1.059503–1.065135 |
| 18 | placement_balanced | Dispatch | 2 | 1.624431 | 1.621753–1.627108 |
| 18 | source_shuffled | Combine | 2 | 1.653402 | 1.652474–1.654330 |
| 18 | source_shuffled | Dispatch | 2 | 1.660601 | 1.658265–1.662938 |
| 23 | captured | Combine | 2 | 1.477407 | 1.475933–1.478881 |
| 23 | captured | Dispatch | 2 | 1.641102 | 1.637837–1.644367 |
| 23 | placement_balanced | Combine | 2 | 1.095940 | 1.092841–1.099039 |
| 23 | placement_balanced | Dispatch | 2 | 1.606123 | 1.606034–1.606213 |
| 23 | source_shuffled | Combine | 2 | 1.478913 | 1.474195–1.483632 |
| 23 | source_shuffled | Dispatch | 2 | 1.626815 | 1.625332–1.628298 |

Placement changes destination imbalance, per-token destination fanout/locality, per-chip tile-rounded expert-region workload and expert ordering. Source shuffling preserves expert/destination totals and fanout histogram while changing source flow and per-expert ordering/batching. The host_comparison.json records tile-rounded load ratios alongside raw assignments. These controls constrain hypotheses; they do not prove imbalance alone determines time.

These are isolated eager dispatch/combine kernels with an FFN surrogate, not full-model latency or throughput. Numerical output correctness was not checked; this is not a validated production optimization. The recovered routing came from a different execution than the historical PP timings. Manager-free success after hardware recovery does not isolate manager removal from recovery as the cause of the earlier stall.

Reproduce: `python3 aggregate_timings.py --root .` from this directory.
