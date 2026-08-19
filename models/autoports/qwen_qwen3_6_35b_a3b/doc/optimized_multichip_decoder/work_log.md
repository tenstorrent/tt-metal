# Optimized Multichip Decoder Work Log

Model: `Qwen/Qwen3.6-35B-A3B`

Branch: `vkovacevic/agentic-research/qb2-qwen36-35b-a3b`

Pre-stage SHA: `c90c9c4336956c895f7481729f66e4a866b9d678`

Stage scope: `models/autoports/qwen_qwen3_6_35b_a3b/tt/multichip_decoder.py`,
`tests/test_multichip_decoder.py`, and
`doc/optimized_multichip_decoder/*`. No full-model or vLLM work was started.

## Implementation

- Changed the default `MultichipMeshPlan.num_links` from `1` to `2`.
- Added guarded reproduction knobs:
  `QWEN36_MULTICHIP_NUM_LINKS`, `QWEN36_MULTICHIP_CCL_MODE`, and
  `QWEN36_MULTICHIP_CCL_DTYPE`.
- Added `_all_reduce()` so final public all-reduce, explicit RS/AG, and BF8 CCL
  candidates could be measured without duplicating TP/EP call sites.
- Kept the final default at public `ttnn.all_reduce`, BF16 CCL payload,
  two-link Ring topology, replicated residual boundaries, and inherited
  optimized-decoder dtype/fidelity policy.
- Extended graph-summary coverage so the no-env default asserts
  `ccl_num_links == 2`, `ccl_mode == "all_reduce"`, and
  `ccl_dtype == "bf16"`.

## Device Safety

Commands were serialized per `$tt-device-usage`. Watcher and profiler evidence
were separate runs.

```bash
tt-smi -ls --local 2>&1 | tee \
  models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_multichip_decoder/logs/tt_smi_initial.log
```

Result: four local Blackhole p300c devices visible and resettable.

```bash
./python_env/bin/python - <<'PY' 2>&1 | tee \
  models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_multichip_decoder/logs/mesh_smoke_initial.log
import ttnn
mesh = ttnn.open_mesh_device(ttnn.MeshShape(2, 2), trace_region_size=0)
ttnn.close_mesh_device(mesh)
print("MESH_SMOKE_OK")
PY
```

Result: `MESH_SMOKE_OK`.

Final health snapshot after watcher:

```bash
tt-smi -ls --local 2>&1 | tee \
  models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_multichip_decoder/logs/tt_smi_post_watcher.log
```

Result: four local Blackhole p300c devices visible and resettable.

Fused AGMM hang recovery snapshots:

```bash
tt-smi -ls --local 2>&1 | tee \
  models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_multichip_decoder/logs/tt_smi_after_fused_agmm_bf16_nonpersistent_before_reset.log
tt-smi -r all 2>&1 | tee \
  models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_multichip_decoder/logs/tt_smi_after_fused_agmm_bf16_nonpersistent_reset.log
tt-smi -ls --local 2>&1 | tee \
  models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_multichip_decoder/logs/tt_smi_after_fused_agmm_bf16_nonpersistent_after_reset.log
./python_env/bin/python - <<'PY' 2>&1 | tee \
  models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_multichip_decoder/logs/mesh_smoke_after_fused_agmm_bf16_nonpersistent_reset.log
import ttnn
mesh = ttnn.open_mesh_device(ttnn.MeshShape(2, 2), trace_region_size=0)
ttnn.close_mesh_device(mesh)
print("MESH_SMOKE_OK")
PY
```

Result: all four devices stayed visible/resettable, PCI reset completed, and
the post-reset mesh smoke printed `MESH_SMOKE_OK`.

## Operation-Topology Audit

The audit is recorded in `operation_topology_audit.md`. It covers repeated
same-input matmuls, material collectives, reshard/layout conversions,
fused/packed projections, fused matmul-CCL paths, lower-movement residual
layouts, action taken, and evidence. Dominant multi-device families are compared
as residual layout, collective placement, fused CCL+matmul, packed versus
separate projections, activation/CCL dtype, persistent buffers, and
DRAM-sharded decode matmuls.

## Baseline Performance

Baseline was the completed multichip decoder default before this pass
(`num_links=1`).

```bash
timeout 1200 env RUN_QWEN36_MULTICHIP_PERF=1 \
  RUN_QWEN36_PERF_REAL_WEIGHTS=1 \
  ./python_env/bin/python -m tracy -r -p -v --no-runtime-analysis \
  --op-support-count=5000 --check-exit-code \
  --output-folder models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_multichip_decoder/tracy/baseline_raw \
  ./python_env/bin/python -m pytest \
  --confcutdir=models/autoports/qwen_qwen3_6_35b_a3b/tests \
  --timeout=900 \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/test_multichip_decoder.py \
  -k test_perf_qwen36_multichip -s \
  2>&1 | tee models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_multichip_decoder/logs/tracy_perf_baseline_summary.log
```

Result: `4 passed, 10 deselected`. Raw ops CSV:
`tracy/baseline_raw/reports/2026_08_19_06_31_27/ops_perf_results_2026_08_19_06_31_27.csv`.

Baseline warmed multichip wall values: linear prefill `22.126 ms`, full
prefill `39.712 ms`, traced linear decode `1.400 ms`, traced full decode
`1.203 ms`.

## Candidate Evidence

Two-link public all-reduce screen:

```bash
timeout 1200 env RUN_QWEN36_MULTICHIP_PERF=1 RUN_QWEN36_PERF_REAL_WEIGHTS=1 \
  QWEN36_MULTICHIP_NUM_LINKS=2 \
  ./python_env/bin/python -m pytest \
  --confcutdir=models/autoports/qwen_qwen3_6_35b_a3b/tests \
  --timeout=900 \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/test_multichip_decoder.py \
  -k test_perf_qwen36_multichip -s \
  2>&1 | tee models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_multichip_decoder/logs/candidate_num_links_2_perf.log
```

Result: `4 passed, 10 deselected`; walls `19.459`, `31.420`, `1.326`, and
`1.084 ms`.

Explicit async RS/AG correctness:

```bash
timeout 1200 env QWEN36_MULTICHIP_NUM_LINKS=2 \
  QWEN36_MULTICHIP_CCL_MODE=explicit_rs_ag \
  ./python_env/bin/python -m pytest \
  --confcutdir=models/autoports/qwen_qwen3_6_35b_a3b/tests \
  --timeout=900 \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/test_multichip_decoder.py \
  -k 'test_synthetic_multichip_decoder_prefill_decode_against_optimized or test_synthetic_multichip_decoder_non_aligned_lengths' -s \
  2>&1 | tee models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_multichip_decoder/logs/candidate_explicit_rs_ag_correctness.log
```

Result: `4 passed, 10 deselected`; PCC matched accepted baseline for linear,
full, and non-aligned cases.

Explicit async RS/AG perf:

```bash
timeout 1200 env RUN_QWEN36_MULTICHIP_PERF=1 RUN_QWEN36_PERF_REAL_WEIGHTS=1 \
  QWEN36_MULTICHIP_NUM_LINKS=2 QWEN36_MULTICHIP_CCL_MODE=explicit_rs_ag \
  ./python_env/bin/python -m pytest \
  --confcutdir=models/autoports/qwen_qwen3_6_35b_a3b/tests \
  --timeout=900 \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/test_multichip_decoder.py \
  -k test_perf_qwen36_multichip -s \
  2>&1 | tee models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_multichip_decoder/logs/candidate_explicit_rs_ag_perf.log
```

Result: `4 passed, 10 deselected`; walls `19.619`, `34.567`, `1.540`, and
`1.095 ms`. Rejected.

BF8 CCL payload correctness:

```bash
timeout 1200 env QWEN36_MULTICHIP_NUM_LINKS=2 \
  QWEN36_MULTICHIP_CCL_DTYPE=bf8 \
  ./python_env/bin/python -m pytest \
  --confcutdir=models/autoports/qwen_qwen3_6_35b_a3b/tests \
  --timeout=900 \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/test_multichip_decoder.py \
  -k 'test_synthetic_multichip_decoder_prefill_decode_against_optimized or test_synthetic_multichip_decoder_non_aligned_lengths' -s \
  2>&1 | tee models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_multichip_decoder/logs/candidate_bf8_ccl_correctness.log
```

Result: `4 passed, 10 deselected`, but below accepted multichip baseline PCC.
Rejected before perf.

Lower-movement residual and DRAM-sharded candidates:

| Artifact | Result |
| --- | --- |
| `logs/candidate_width_sharded_residual_rmsnorm_repro.log` | first RMSNorm attempt failed with `Sharded inputs require sharded outputs`; not accepted as final rejection |
| `logs/candidate_width_sharded_residual_rmsnorm_adapted_repro.log` | adapted attempt failed tensor composer dims; not accepted as final rejection |
| `logs/candidate_width_sharded_residual_rmsnorm_adapted_repro_v2.log` | adapted sharded-output RMSNorm ran and was slower, `0.070725 -> 0.095242 ms`, PCC `0.9999903609`; rejected |
| `logs/candidate_sharded_residual_stack_probe.log` | stack-compatible width-sharded residual ran through real input RMSNorm, real mixer, residual add, real post RMSNorm, real MoE/MLP, and final residual; linear decode `3.454771 -> 4.485228 ms`, PCC `0.9998683319`; full decode `2.555675 -> 3.238674 ms`, PCC `0.9993485173`; rejected |
| `logs/candidate_dram_sharded_full_qkgv_repro.log` | sharded-only full qkgv faster, `0.065955 -> 0.050056 ms`, PCC `0.9999652109`; insufficient under current residual contract |
| `logs/candidate_dram_sharded_full_qkgv_with_convert_repro.log` | current-boundary convert/restore candidate slower, `0.069743 -> 0.107414 ms`, PCC `0.9999653433`; rejected |
| `logs/candidate_dram_sharded_linear_qkvzba_repro.log` | padded/sliced linear qkvzba candidate slower, `0.077718 -> 0.092857 ms`, PCC `0.9999652115`; rejected |

Stack-compatible sharded residual command:

```bash
timeout 1200 env PYTHONUNBUFFERED=1 RUN_QWEN36_PERF_REAL_WEIGHTS=1 \
  ./python_env/bin/python \
  models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_multichip_decoder/scripts/probe_sharded_residual_stack.py \
  --iterations 2 \
  2>&1 | tee models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_multichip_decoder/logs/candidate_sharded_residual_stack_probe.log
```

Fused matmul-CCL and persistent-buffer remediation after first stage review:

- Source-only `$autofix` subagents checked lower-movement residual,
  DRAM-sharded decode, fused AGMM, and persistent CCL feasibility without
  editing files or touching hardware.
- `logs/fused_ccl_api_source_audit.log` records the source scan. The earlier
  source-only fused rejection was corrected: `all_gather_minimal_matmul_async`
  does expose `cluster_axis`.
- `scripts/probe_fused_ccl_and_persistent.py` now has independent `--only`
  modes and flushes every probe status line so hangs leave usable evidence.
- The first fused AGMM runtime attempt hit the program-factory worker grouping
  rule. The probe was adapted to the legal 2-link TP axis 1 case by setting
  `num_workers_per_link=4`.

Adapted fused AGMM non-persistent command:

```bash
timeout 300 env PYTHONUNBUFFERED=1 \
  ./python_env/bin/python \
  models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_multichip_decoder/scripts/probe_fused_ccl_and_persistent.py \
  --only fused --fused-weight-dtype bf16 --no-fused-use-persistent \
  --iterations 1 \
  2>&1 | tee models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_multichip_decoder/logs/candidate_fused_agmm_bf16_nonpersistent_probe.log
```

Result: the probe reached `cluster_axis=1`, `force_transpose=True`,
`num_links=2`, `num_workers_per_link=4`, adapted output-sharded BF16 weights,
and then hung before producing `fused_agmm_status`. Live triage was captured:

```bash
mkdir -p models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_multichip_decoder/triage/fused_agmm_bf16_nonpersistent
timeout 180 tools/tt-triage.py --llm-output \
  --llm-output-path models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_multichip_decoder/triage/fused_agmm_bf16_nonpersistent/tt-triage.txt \
  --triage-summary-path models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_multichip_decoder/triage/fused_agmm_bf16_nonpersistent/triage-summary.txt \
  2>&1 | tee models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_multichip_decoder/triage/fused_agmm_bf16_nonpersistent/triage-console.log
```

`tt-triage.txt` shows NOC/fabric-router hang symptoms, including
`check_noc_status.py:331` and `fabric_erisc_router` callstacks beginning at
line `350`. The older fused persistent retry has the same family of evidence in
`triage/fused_agmm_persistent/tt-triage.txt` after adapting past the first API
error. The fused matmul-CCL family is rejected with adapted runtime and triage
evidence, not a first TTNN/API error.

Persistent RS/AG buffer command:

```bash
timeout 300 env PYTHONUNBUFFERED=1 \
  ./python_env/bin/python \
  models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_multichip_decoder/scripts/probe_fused_ccl_and_persistent.py \
  --only persistent-rsag --iterations 5 \
  2>&1 | tee models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_multichip_decoder/logs/candidate_persistent_rsag_probe.log
```

Result: preallocated DRAM intermediate/reduced/gathered buffers ran on TP axis
1 and EP axis 0 for `[1,1,32,2048] -> [1,1,32,1024] -> [1,1,32,2048]`, but the
path failed public all-reduce correctness at about `0.949` PCC on both axes.
TP timing was `0.171927 ms` nonpersistent vs `0.174642 ms` persistent; EP was
`0.175274 ms` nonpersistent vs `0.172148 ms` persistent. Since the model-level
nonpersistent explicit RS/AG path already passed baseline PCC but was slower
than public all-reduce, no persistent CCL path was selected.

## Final Performance

```bash
timeout 1200 env RUN_QWEN36_MULTICHIP_PERF=1 \
  RUN_QWEN36_PERF_REAL_WEIGHTS=1 \
  ./python_env/bin/python -m tracy -r -p -v --no-runtime-analysis \
  --op-support-count=5000 --check-exit-code \
  --output-folder models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_multichip_decoder/tracy/final_raw \
  ./python_env/bin/python -m pytest \
  --confcutdir=models/autoports/qwen_qwen3_6_35b_a3b/tests \
  --timeout=900 \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/test_multichip_decoder.py \
  -k test_perf_qwen36_multichip -s \
  2>&1 | tee models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_multichip_decoder/logs/tracy_perf_final_summary.log
```

Result: `4 passed, 10 deselected`. Raw ops CSV:
`tracy/final_raw/reports/2026_08_19_07_11_25/ops_perf_results_2026_08_19_07_11_25.csv`.

Final warmed multichip wall values: linear prefill `25.747 ms`, full prefill
`39.202 ms`, traced linear decode `1.346 ms`, traced full decode `1.152 ms`.

Final no-Tracy default screen:

```bash
timeout 1200 env RUN_QWEN36_MULTICHIP_PERF=1 RUN_QWEN36_PERF_REAL_WEIGHTS=1 \
  ./python_env/bin/python -m pytest \
  --confcutdir=models/autoports/qwen_qwen3_6_35b_a3b/tests \
  --timeout=900 \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/test_multichip_decoder.py \
  -k test_perf_qwen36_multichip -s \
  2>&1 | tee models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_multichip_decoder/logs/final_perf_screen.log
```

Result: `4 passed, 10 deselected`; final default walls `20.733`, `32.504`,
`1.281`, and `1.096 ms`.

One-link reproduction after the code change:

```bash
timeout 1200 env RUN_QWEN36_MULTICHIP_PERF=1 RUN_QWEN36_PERF_REAL_WEIGHTS=1 \
  QWEN36_MULTICHIP_NUM_LINKS=1 \
  ./python_env/bin/python -m pytest \
  --confcutdir=models/autoports/qwen_qwen3_6_35b_a3b/tests \
  --timeout=900 \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/test_multichip_decoder.py \
  -k test_perf_qwen36_multichip -s \
  2>&1 | tee models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_multichip_decoder/logs/baseline_num_links_1_perf_screen.log
```

Result: `4 passed, 10 deselected`; walls `19.408`, `31.912`, `1.327`, and
`1.067 ms`. This screen run is kept as noise context; the selected default uses
the Blackhole-normalized Tracy device tables and final default screen values.

## Report Generation

The raw Tracy CSVs were normalized for `tt-perf-report` because legacy rows had
blank architecture metadata. Normalized files:

- `tracy/baseline_raw/reports/2026_08_19_06_31_27/ops_perf_results_2026_08_19_06_31_27_blackhole.csv`
- `tracy/final_raw/reports/2026_08_19_07_11_25/ops_perf_results_2026_08_19_07_11_25_blackhole.csv`

Human table command shape:

```bash
./python_env/bin/tt-perf-report \
  --start-signpost MC_LINEAR_DECODE \
  --end-signpost MC_LINEAR_DECODE_END \
  --arch blackhole --active-experts 8 --no-color --raw-op-codes \
  --summary-file models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_multichip_decoder/tracy/final_reports/mc_linear_decode_summary.csv \
  models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_multichip_decoder/tracy/final_raw/reports/2026_08_19_07_11_25/ops_perf_results_2026_08_19_07_11_25_blackhole.csv \
  > models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_multichip_decoder/tracy/final_reports/mc_linear_decode_perf_report.txt
```

CSV export command shape:

```bash
./python_env/bin/tt-perf-report \
  --start-signpost MC_LINEAR_DECODE \
  --end-signpost MC_LINEAR_DECODE_END \
  --arch blackhole --active-experts 8 --no-color --raw-op-codes \
  --csv models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_multichip_decoder/tracy/final_reports/mc_linear_decode_perf_report.csv \
  --summary-file models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_multichip_decoder/tracy/final_reports/mc_linear_decode_summary.csv \
  models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_multichip_decoder/tracy/final_raw/reports/2026_08_19_07_11_25/ops_perf_results_2026_08_19_07_11_25_blackhole.csv \
  > models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_multichip_decoder/tracy/final_reports/mc_linear_decode_perf_report_csv.log
```

Both shapes were rerun after first stage review and logged in
`logs/regenerate_perf_reports.log`. They were run for baseline/final and all
eight signposted windows:
`BASE_LINEAR_PREFILL`, `MC_LINEAR_PREFILL`, `BASE_FULL_PREFILL`,
`MC_FULL_PREFILL`, `BASE_LINEAR_DECODE`, `MC_LINEAR_DECODE`,
`BASE_FULL_DECODE`, and `MC_FULL_DECODE`.

Summary artifacts:

- `logs/perf_report_family_summary.csv`
- `logs/perf_report_top_ops.csv`
- `logs/perf_wall_summary.csv`
- `logs/perf_screen_wall_summary.csv`

Raw op CSV provenance is committed as gzip parts with `SHA256SUMS` manifests:

- `tracy/baseline_raw/reports/2026_08_19_06_31_27/ops_perf_results_2026_08_19_06_31_27.csv.gz.parts/`
- `tracy/baseline_raw/reports/2026_08_19_06_31_27/ops_perf_results_2026_08_19_06_31_27_blackhole.csv.gz.parts/`
- `tracy/final_raw/reports/2026_08_19_07_11_25/ops_perf_results_2026_08_19_07_11_25.csv.gz.parts/`
- `tracy/final_raw/reports/2026_08_19_07_11_25/ops_perf_results_2026_08_19_07_11_25_blackhole.csv.gz.parts/`

Reconstruction pattern from a `.gz.parts/` directory:

```bash
cat part_*.csv.gz > file.csv.gz
sha256sum -c SHA256SUMS
gunzip -c file.csv.gz > file.csv
```

The multi-GB profiler internals under `tracy/*_raw/.logs/` and
`profile_log_device.csv` are not committed.

## Final Gates

Syntax:

```bash
./python_env/bin/python -m py_compile \
  models/autoports/qwen_qwen3_6_35b_a3b/tt/multichip_decoder.py \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/test_multichip_decoder.py \
  models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_multichip_decoder/scripts/probe_fused_ccl_and_persistent.py \
  models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_multichip_decoder/scripts/probe_sharded_residual_stack.py \
  2>&1 | tee models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_multichip_decoder/logs/py_compile_final.log
```

Result: passed.

Final correctness:

```bash
timeout 1200 env RUN_QWEN36_REAL_WEIGHTS=1 \
  ./python_env/bin/python -m pytest \
  --confcutdir=models/autoports/qwen_qwen3_6_35b_a3b/tests \
  --timeout=900 \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/test_multichip_decoder.py \
  -k 'test_multichip_decoder_graph_summary or test_multichip_runtime_fallback_audit_source or test_synthetic_multichip_decoder_prefill_decode_against_optimized or test_synthetic_multichip_decoder_non_aligned_lengths or test_synthetic_multichip_decoder_batch2_against_optimized or test_real_weight_multichip_decoder_prefill_decode_against_optimized' \
  -s 2>&1 | tee models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_multichip_decoder/logs/final_correctness.log
```

Result: `10 passed, 4 deselected`.

Post-fused-recovery final correctness:

```bash
timeout 1200 env RUN_QWEN36_REAL_WEIGHTS=1 \
  ./python_env/bin/python -m pytest \
  --confcutdir=models/autoports/qwen_qwen3_6_35b_a3b/tests \
  --timeout=900 \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/test_multichip_decoder.py \
  -k 'test_multichip_decoder_graph_summary or test_multichip_runtime_fallback_audit_source or test_synthetic_multichip_decoder_prefill_decode_against_optimized or test_synthetic_multichip_decoder_non_aligned_lengths or test_synthetic_multichip_decoder_batch2_against_optimized or test_real_weight_multichip_decoder_prefill_decode_against_optimized' \
  -s 2>&1 | tee models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_multichip_decoder/logs/final_correctness_post_fused_recovery.log
```

Result: `10 passed, 4 deselected`.

Dynamic fallback audit:

```bash
timeout 1200 env TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
  RUN_QWEN36_REAL_WEIGHTS=1 \
  ./python_env/bin/python -m pytest \
  --confcutdir=models/autoports/qwen_qwen3_6_35b_a3b/tests \
  --timeout=900 \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/test_multichip_decoder.py \
  -k 'test_multichip_decoder_graph_summary or test_multichip_runtime_fallback_audit_source or test_synthetic_multichip_decoder_prefill_decode_against_optimized or test_synthetic_multichip_decoder_non_aligned_lengths or test_synthetic_multichip_decoder_batch2_against_optimized or test_real_weight_multichip_decoder_prefill_decode_against_optimized' \
  -s 2>&1 | tee models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_multichip_decoder/logs/final_runtime_fallback_audit.log
```

Result: `10 passed, 4 deselected`.

Post-fused-recovery dynamic fallback audit:

```bash
timeout 1200 env TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
  RUN_QWEN36_REAL_WEIGHTS=1 \
  ./python_env/bin/python -m pytest \
  --confcutdir=models/autoports/qwen_qwen3_6_35b_a3b/tests \
  --timeout=900 \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/test_multichip_decoder.py \
  -k 'test_multichip_decoder_graph_summary or test_multichip_runtime_fallback_audit_source or test_synthetic_multichip_decoder_prefill_decode_against_optimized or test_synthetic_multichip_decoder_non_aligned_lengths or test_synthetic_multichip_decoder_batch2_against_optimized or test_real_weight_multichip_decoder_prefill_decode_against_optimized' \
  -s 2>&1 | tee models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_multichip_decoder/logs/final_runtime_fallback_audit_post_fused_recovery.log
```

Result: `10 passed, 4 deselected`.

Final post-recovery health snapshot:

```bash
tt-smi -ls --local 2>&1 | tee \
  models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_multichip_decoder/logs/tt_smi_final_post_recovery_gates.log
```

Result: four local Blackhole p300c devices visible and resettable.

Watcher:

```bash
timeout 1200 env \
  TT_METAL_LOGS_PATH=/localdev/vkovacevic/tt-metal/models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_multichip_decoder/watcher/final \
  TT_METAL_WATCHER=10 TT_METAL_WATCHER_APPEND=1 \
  TT_METAL_WATCHER_NOINLINE=1 TT_METAL_WATCHER_DISABLE_ETH=1 \
  RUN_QWEN36_REAL_WEIGHTS=1 \
  ./python_env/bin/python -m pytest \
  --confcutdir=models/autoports/qwen_qwen3_6_35b_a3b/tests \
  --timeout=900 \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/test_multichip_decoder.py \
  -k 'test_multichip_decoder_graph_summary or test_multichip_runtime_fallback_audit_source or test_synthetic_multichip_decoder_prefill_decode_against_optimized or test_synthetic_multichip_decoder_non_aligned_lengths or test_synthetic_multichip_decoder_batch2_against_optimized or test_real_weight_multichip_decoder_prefill_decode_against_optimized' \
  -s 2>&1 | tee models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_multichip_decoder/logs/final_watcher_correctness_disable_eth.log
```

Result: `10 passed, 4 deselected`. Filtered watcher scan:
`logs/final_watcher_failure_scan_filtered.log`.

## Stage Review

Fresh `$stage-review` subagent `01a01915-6a53-7af2-a299-e5d6432ebf9a` returned
`clean-pass`.

Residual risks noted by the reviewer:

- Final no-Tracy screen timings show the one-link reproduction slightly faster
  for full decode and prefill, while Blackhole-normalized Tracy device/CCL
  reports favor the final two-link default path; this split is disclosed above.
- Real-weight correctness coverage is seq 1; non-aligned logical length
  coverage is synthetic.
- The final path inherits DRAM/interleaved and small-subblock matmul
  inefficiencies from the prior optimized decoder stage; those remain out of
  scope for this multichip stage.
