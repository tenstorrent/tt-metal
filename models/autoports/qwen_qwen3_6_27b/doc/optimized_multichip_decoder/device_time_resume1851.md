# Stage5 device-time ledger

Snapshot: 2026-11-13T18:51:14.684211+00:00. Source: `device_time_ledger.json`.

Conservative confirmed occupation: **7021.508 seconds (1.950419 hours)**. TP4 counts once, never four chip-seconds per second.

Observed driver-open→completed-close interval union: **7337.638 seconds**. This includes initialization and teardown; the conservative count excludes both. Both require timestamp evidence and exclude unbounded tails.

Wall span from first recorded command to latest timestamped log: **14412.856 seconds**; first command to snapshot: **14414.207 seconds**. These include host preparation, profiling postprocessing and idle time, and are not device time.

The strict user stop rule is `(>=70% roofline AND last two kept improvements each <3%) OR 8 hours accumulated device time`. This ledger does not evaluate the roofline/kept-improvement branch and makes **no stop-rule firing or completion claim**. Quality gates remain required.

Native log timestamps are interpreted as UTC, matching this run's environment and UTC command sidecars. A complete interval begins at confirmed cluster readiness (or fabric readiness if needed) and ends at the first teardown marker. Driver open/close is also preserved separately. No file mtime, timeout limit, current time, reported process duration or profiler kernel sum is substituted for a missing endpoint.

Records: 634; device sessions: 630; sessions with an unknown endpoint: 1; records without sessions: 4; unmatched close markers: 0; excluded wrapper/sidecar-less logs: 203.

Individual command logs are streamed. Exact repeated event sequences retain duplicate links; the global interval union prevents duplicate/nested/profile-wrapper time from being added twice. Archived attempts remain separate from newer executions with reused output names. Every evidence path/line, interval, command and unknown record is preserved in JSON. No individual artifact was hashed or archived.

## Incomplete records

| Record | Status / limitation |
|---|---|
| `models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/conv_battery_tracking.command.json` | no_device_interval_markers |
| `models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/epilogue16_regression_beta_b32_s33.command.json` | session0: open_without_close |
| `models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/interim_profile_l0_b32_s33.command.json` | no_device_interval_markers |
| `models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/local_affine_gate_watcher.command.json` | no_device_interval_markers |
| `models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/tiled_conv_onecore_watcher8.command.json` | no_device_interval_markers |

Regenerate:

```bash
python models/autoports/qwen_qwen3_6_27b/tests/account_optimized_multichip_device_time.py
```
