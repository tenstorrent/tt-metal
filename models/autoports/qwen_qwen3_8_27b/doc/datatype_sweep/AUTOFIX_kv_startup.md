# AutoFix: pre-KV startup recovery

2026-09-13. Final independent evidence review after recovered full-model retry
completed at 23:03:57 UTC. Coordinator owns all device operations. This investigator
only read artifacts/source and authored this report.

## Hypothesis and discriminating experiment

The initialization stall is recoverable device/dispatch state, rather than a
deterministic consequence of BFP4 KV policy. Prediction: after bounded recovery,
the unchanged model and precision policy progress through `MODEL_LOADED`, cache
allocation and the original full-model check. A successful retry establishes
incident recovery; it does not establish the exact native root cause.

Starting evidence: `AUTOTRIAGE_kv_startup.md`, `kv_bfp4_evaluated.log`, and
`triage_kv_startup/host_gdb{,_main}.log`. The blocked host upload waits on a
command-queue completion before model construction finishes and before KV cache
allocation. Device triage timed out (124) with empty outputs, so no current
device kernel/NoC stop-site is known. The coordinator ended the owned stalled
process with SIGTERM; original launcher exit status is 143.

The coordinator performed bounded list/reset/list and exact Ring1x4 open/close,
then launched this discriminating retry (only experiment artifact name changes):

```bash
QWEN_PRECISION_CONFIG=models/autoports/qwen_qwen3_8_27b/doc/datatype_sweep/configs/kv_bfp4.json bash models/autoports/qwen_qwen3_8_27b/tests/run_datatype_experiment.sh kv_bfp4_recovered models.autoports.qwen_qwen3_8_27b.tests.run_datatype_candidate
```

## Equivalence checks

- The two `.source.sha256` manifests each contain 66 entries. All seven model
  Python source snapshots and both native library hashes are identical. The
  only differing entries are `tests/run_datatype_candidate.py` and
  `tests/summarize_datatype_sweep.py` (offline aggregation, outside this retry).
- The candidate snapshot diff adds a 180-second repeating Python traceback
  timer around `build_generator`, cancellation, and read-only reporting of
  already allocated tensor dtype/shape/layout after generation. It changes no
  model operation or precision policy. The timer is still a timing perturbation,
  so this is not a rigorously isolated reset-only causality experiment.
- Both `.commit` files identify `624f6352a9ca1c339870dbe517e564910bb71b15`;
  recorded environment JSONs are identical, including BFP4 config path, cache
  path, pass-through thread pool, unset watcher and unset device profiler.
- Current `configs/kv_bfp4.json` exactly equals the pre-incident
  `smoke_kv_bfp4.json` runtime policy. Its only differences from baseline config
  are `config_id` and `kv_cache_dtype`. Config JSON itself was not included in
  the original hash manifest, and the stalled full-model run wrote no runtime
  report; policy continuity is supported by that prior smoke and identical
  config path, rather than an original full-model runtime-policy snapshot.
- Policy remains decoder BFP4/LoFi, head BFP8/HiFi2, BF16 activations/residual/
  CCL, BFP4 KV and max context 262144. No higher-precision workaround was made.

## Actual results

`triage_kv_startup/recovery_actions.json` records list-before exit 1, reset exit
0, list-after exit 0 and mesh smoke exit 0. The underlying `list_before.log`
explicitly reports **NOC0 is hung on PCIe device ID 1**. `reset1.log` records
reset of PCI devices 0–3; `list_after1.log` enumerates all four Blackhole p300c
devices; `mesh_after1.log` reports Ring fabric initialized on four devices,
`MESH_SMOKE_OK` and completed device close. No second reset was needed.

This independently corroborates a recoverable device-health fault following the
stall, but does not prove whether that fault caused or resulted from the upload
stall.

The unchanged-policy retry completed: `kv_bfp4_recovered.exit_status` is 0;
the log contains `MODEL_LOADED` and completed device shutdown, with no realtime
synchronization error. `kv_bfp4_recovered.json` records `status: pass`, all 64
layers, batch 1/S203/G100, and runtime policy exactly equal to `configs/kv_bfp4.json`.
Prefill and all three teacher-forcing checks score 98/100 top1 and 100/100
top5/top100. Actual allocated key/value tensors in all 16 full-attention layers
are `DataType.BFLOAT4_B`, tiled shape `[10, 1, 32, 256]`; allocated cache capacity
is 320 tokens with 10 pages. This directly verifies the originally stalled
candidate reached and exercised BFP4 KV operations after recovery.

The last two warmed teacher-forcing samples each record 99 model trace replays,
99 sampling trace replays, no new steady-state trace captures and no host
sampling. They measure 40.177543 and 40.136699 tokens/s/user (mean 40.157121).
These measurements establish completion, not a claimed performance improvement.

## Verdict and limitations

**Recovered; recoverability hypothesis verified for this incident.** The original
full-model check now passes with the same BFP4 KV policy and model/native sources.
A deterministic claim that this KV policy prevents initialization is refuted by
the passing retry and the pre-KV stop boundary. No native source fix was proposed
or applied. A transient source/lifetime/dispatch race remains a possible
underlying mechanism, and neither reset nor the traceback timer has been isolated
as the causal remedy. The missing device triage prevents an exact native
causality claim. The run exercises capacity 320, so it does not establish full
262144-token allocation/execution or broader workload accuracy. This report is
docs-only; no build or hardware command was run by the investigator.
