# HF20 group admission readiness note

Host-only inspection, 2026-11-15. No candidate inputs, source, commands, forecasts, or admission files changed. HF20 remains unadmitted: `admission.json` and `predecessor-gate.json` are absent. The current root roster review covers all 20 observations, the driver review explicitly says it is not admission, and host validation confirms all eight manifests and exact prompt/seed/cap identities.

The useful correction is per-group timeout budgeting. Every command currently hard-codes 18,000 seconds, and `verify_admission` correctly requires that full timeout plus 60 seconds to fit the authorized deadline, even when the group forecast is much shorter. The current sensitivity highs are 5,459–16,192 seconds. A planning candidate is `ceil(group sensitivity high) + 300s` for the child timeout, with the existing 60-second process grace kept outside it. That covers the existing high forecast plus five minutes while letting each group be admitted independently:

| Group | Seed / cap | Cases | Forecast high (s) | Candidate child timeout (s) | Minimum admitted ceiling (s) |
|---:|---:|---:|---:|---:|---:|
| 1 | 42 / 512 | 4 | 11,437.75 | 11,738 | 11,798 |
| 2 | 42 / 1024 | 1 | 5,458.66 | 5,759 | 5,819 |
| 3 | 42 / 2048 | 1 | 10,737.31 | 11,038 | 11,098 |
| 4 | 43 / 512 | 4 | 10,970.79 | 11,271 | 11,331 |
| 5 | 43 / 1024 | 3 | 16,191.08 | 16,492 | 16,552 |
| 6 | 44 / 512 | 5 | 14,018.71 | 14,319 | 14,379 |
| 7 | 44 / 1024 | 1 | 5,575.40 | 5,876 | 5,936 |
| 8 | 44 / 2048 | 1 | 10,562.20 | 10,863 | 10,923 |

These are proposed bounds, not an admission. They do not change the measured forecast or imply that the high-sensitivity sum fits the remaining phase window. The eight highs sum to about 84,952 seconds (23.60 hours), plus per-process grace; the point scenarios sum to about 34,611 seconds (9.61 hours). Use the actual root-authorized deadline at each single-group admission and stop when the next group no longer fits. Do not claim 20-observation completion unless every group is terminal and reviewed.

If root chooses this timeout policy, the exact pin refresh is: for each selected group, set `forecast.json:groups[N].timeout_seconds` to the table value; set the matching `commands.json` group `timeout_seconds` and `command[0:2]` (`timeout`, value); refresh the proposal's command/forecast hashes; and adjust `run_group.py:verify_static_bundle` to validate the per-group frozen timeout rather than literal `18000` (both the group check and argv check), with host tests covering varied values and the unchanged `timeout + 60` deadline gate. Then refresh driver-integrity, root roster/driver reviews, and root admission hashes. Manifests and source-result hashes need no change. Root must create a terminal `predecessor-gate.json` pinning successful, reviewed HF9, SKHF, and P3 completion evidence before admitting any group. Existing runtime preflight still requires no active HF/TT owners and at least 100 GiB `MemAvailable`; no automatic group advancement is authorized.
