# Recurrent fusion rejected on complete-adapter latency

Retain native recurrence. Both fused candidates preserve the tested exact arithmetic and state contract but fail the performance gate. No integration or context-capacity change is justified.

| Complete adapter | Native warmed (ms) | Candidate warmed (ms) | Native traced (ms) | Candidate traced (ms) |
|---|---:|---:|---:|---:|
| Compile-fixed six-stage | 0.248990 | 0.911761 | 0.168848 | 0.894937 |
| Blocked state transport | 0.265286 | 0.962242 | 0.169009 | 0.944483 |

The original attempt failed NCRISC compilation because a beta-fill loop and following push shared a line (`-Werror=misleading-indentation`). The compile-fixed variant changed only a newline. Its full run passed 48 fixture/configuration combinations, 12 changing-input steps, matched evolving-state checkpoints over 305 traced replays, and restarted-trajectory determinism. The blocked variant first passed actual227 (four configurations), then passed the same complete 48/12/replay gates. Every recorded FP32 bit comparison is zero, output-alias checks pass, and all four recorded device sessions closed. These are component correctness results, not integrated decoder acceptance.

Blocked transport reduced source-issued state read and write requests separately from 32,768 to 2,048, with unchanged 2 MiB payload per direction. Both arithmetic compute strings and native reductions remained unchanged. The writer added local scalar scatter copies and temporary staging grew. This complete variant is 5.59× slower than its same-run native traced baseline; the prior exact variant is 5.30× slower. The separate runs do not establish statistical significance for their approximately 5.5% candidate-to-candidate difference, but both decisively fail the native latency gate.

There is no profile-proven bottleneck. Scalar transpose/address-generation costs, only 32 active workers, repeated register/pack feedback and synchronization, and NoC/barrier/staging costs remain plausible contributors. The experiment shows that reducing source-issued requests alone did not improve the complete adapter; it does not isolate or rule out individual transport costs. Source request counts are not measured physical wire-packet counts.

`v6_recurrent_fusion_rejection_summary.json` records exact source/artifact/log hashes, bit-comparison counts, medians, and replay checks for the original compile failure, compile-fixed full result, blocked actual227 result, and blocked full result. Those files and all frozen sources remain unchanged. No further variant was implemented, and no device or Torch import was used to prepare this summary.
