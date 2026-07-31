# Work Log

Date: 2026-06-15

Model: `microsoft/Phi-3.5-mini-instruct`

Goal scope: optimize the completed multichip decoder in place. Do not start full-model or vLLM work.

## Baseline

Command:

```bash
PHI35_HOST_TIMING_ITERS=100 pytest --confcutdir=models/autoports/microsoft_phi_3_5_mini_instruct -q models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_multichip_decoder.py::test_multichip_dense_layer_real_weights_prefill_decode_pcc_1x8_ring -s
```

Artifact: `logs/baseline_host_timing_real_layer0.log`

Result:

- Prefill PCC: 0.9999919281757654
- Decode PCC: 0.9999935218427508
- Warmed traced decode host E2E: 580.685 us

Profiler command:

```bash
PHI35_RUN_PERF=1 python -m tracy -r -p -v -o models/autoports/microsoft_phi_3_5_mini_instruct/doc/optimized_multichip_decoder/tracy/baseline -m pytest --confcutdir=models/autoports/microsoft_phi_3_5_mini_instruct -q models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_multichip_decoder.py::test_multichip_dense_layer_real_weights_perf_profile_1x8_ring -s
```

Artifacts:

- `logs/baseline_tracy_perf.log`
- `tracy/baseline/reports/2026_06_15_14_28_56/ops_perf_results_2026_06_15_14_28_56.csv`
- `perf/baseline_prefill_perf_human.txt`
- `perf/baseline_decode_perf_human.txt`
- `perf/baseline_prefill_perf_report.csv`
- `perf/baseline_decode_perf_report.csv`

Baseline tt-perf summary:

| Phase | Ops | Device time | Op-to-op gap | Total |
| --- | ---: | ---: | ---: | ---: |
| Prefill | 50 | 798.305 us | 3719.245 us | 4517.550 us |
| Decode | 63 | 570.031 us | 575.837 us | 1145.868 us |

Baseline tt-perf advice:

- Decode O and down local matmuls used `in0_block_w=1`; try `in0_block_w>=2`.
- Output subblock size was not found.
- Decode had a large traced-profile gap before an `InterleavedToShardedDeviceOperation`.
- Prefill had large op-to-op gaps but device time was already about 798 us.

## Trials

### Explicit async all-reduce

Command:

```bash
PHI35_MULTICHIP_CCL=async_all_reduce PHI35_HOST_TIMING_ITERS=100 pytest --confcutdir=models/autoports/microsoft_phi_3_5_mini_instruct -q models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_multichip_decoder.py::test_multichip_dense_layer_real_weights_prefill_decode_pcc_1x8_ring -s
```

Artifacts:

- `logs/trial_async_all_reduce_host_timing_real_layer0.log`
- `logs/trial_async_all_reduce_tracy_perf.log`
- `tracy/async_all_reduce/reports/2026_06_15_14_33_42/ops_perf_results_2026_06_15_14_33_42.csv`
- `perf/async_all_reduce_decode_perf_human.txt`
- `perf/async_all_reduce_prefill_perf_human.txt`

Result:

- Prefill PCC: 0.999991788840335
- Decode PCC: 0.9999935080298819
- Host traced decode: 584.073 us
- tt-perf decode device/total: 572.267/1213.491 us
- Decision: rejected. It was slower than baseline and did not remove the in-layer collective cost.

### Decode local matmul `in0_block_w=2`

First attempt:

- Artifact: `logs/trial_local_matmul_in0_2_host_timing_real_layer0.log`
- Result: failed because the original decode shard width was not divisible by `in0_block_w=2`.

Second attempt with widened/aligned local decode shards:

- Artifact: `logs/trial_local_matmul_in0_2_reshard_host_timing_real_layer0.log`
- Prefill PCC: 0.9999919281757654
- Decode PCC: 0.999992733851469
- Host traced decode: 568.831 us
- Issue: planner warning for output memory config mismatch.

Final corrected attempt with explicit O/down output mem configs:

- Artifact: `logs/trial_local_matmul_in0_2_output_mem_host_timing_real_layer0.log`
- Prefill PCC: 0.999991788840335
- Decode PCC: 0.9999927771280459
- Host traced decode: 571.521 us
- Decision: accepted. It removed the planner warning and kept a measured decode improvement.

Profile artifact:

- `logs/trial_local_matmul_in0_2_tracy_perf.log`
- `perf/local_matmul_in0_2_decode_perf_human.txt`

### BF8 CCL payloads

Command:

```bash
PHI35_MULTICHIP_LOCAL_MATMUL_MIN_IN0_BLOCK_W=2 PHI35_MULTICHIP_CCL_DTYPE=bfloat8_b PHI35_HOST_TIMING_ITERS=100 pytest --confcutdir=models/autoports/microsoft_phi_3_5_mini_instruct -q models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_multichip_decoder.py::test_multichip_dense_layer_real_weights_prefill_decode_pcc_1x8_ring -s
```

Artifact: `logs/trial_ccl_bfloat8_host_timing_real_layer0.log`

Result:

- Prefill PCC: 0.9999670765792469
- Decode PCC: 0.9999759054712127
- Host traced decode: 567.444 us
- Decision: accepted for decode collectives only. Rejected for prefill.

Final CCL measurements in `perf/final_split_fidelity_decode_perf_human.txt`:

- Decode RS/AG pair 1: 61 us / 33 us, BF8 payload
- Decode RS/AG pair 2: 60 us / 34 us, BF8 payload
- Baseline RS/AG pairs were about 66/37 us and 67/36 us with BF16 payloads.

### LoFi decode fidelity

Command:

```bash
PHI35_MULTICHIP_LOCAL_MATMUL_MIN_IN0_BLOCK_W=2 PHI35_MULTICHIP_CCL_DTYPE=bfloat8_b PHI35_MULTICHIP_MATMUL_FIDELITY=lofi PHI35_HOST_TIMING_ITERS=100 pytest --confcutdir=models/autoports/microsoft_phi_3_5_mini_instruct -q models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_multichip_decoder.py::test_multichip_dense_layer_real_weights_prefill_decode_pcc_1x8_ring -s
```

Artifact: `logs/trial_lofi_host_timing_real_layer0.log`

Result:

- Prefill PCC: 0.9999558401677192
- Decode PCC: 0.9999751805658009
- Host traced decode: 559.223 us
- Decision: accepted for decode only, rejected for prefill.

Follow-up split-fidelity result:

- Artifact: `logs/final_split_ccl_fidelity_host_timing_real_layer0.log`
- Prefill PCC: 0.999991788840335
- Decode PCC: 0.999975693890481
- Host traced decode: 558.770 us

### Final default validation

Command:

```bash
PHI35_HOST_TIMING_ITERS=100 pytest --confcutdir=models/autoports/microsoft_phi_3_5_mini_instruct -q models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_multichip_decoder.py::test_multichip_dense_layer_real_weights_prefill_decode_pcc_1x8_ring -s
```

Artifact: `logs/final_default_host_timing_real_layer0.log`

Result:

- Prefill PCC: 0.999991788840335
- Decode PCC: 0.999975693890481
- Host traced decode: 559.258 us

Final profiler command:

```bash
PHI35_MULTICHIP_LOCAL_MATMUL_MIN_IN0_BLOCK_W=2 PHI35_MULTICHIP_CCL_DTYPE=bfloat8_b PHI35_MULTICHIP_MATMUL_FIDELITY=lofi PHI35_RUN_PERF=1 python -m tracy -r -p -v -o models/autoports/microsoft_phi_3_5_mini_instruct/doc/optimized_multichip_decoder/tracy/final_split_fidelity -m pytest --confcutdir=models/autoports/microsoft_phi_3_5_mini_instruct -q models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_multichip_decoder.py::test_multichip_dense_layer_real_weights_perf_profile_1x8_ring -s
```

Artifacts:

- `logs/final_split_fidelity_tracy_perf.log`
- `tracy/final_split_fidelity/reports/2026_06_15_15_01_00/ops_perf_results_2026_06_15_15_01_00.csv`
- `perf/final_split_fidelity_prefill_perf_human.txt`
- `perf/final_split_fidelity_decode_perf_human.txt`
- `perf/final_split_fidelity_prefill_perf_report.csv`
- `perf/final_split_fidelity_decode_perf_report.csv`

Final tt-perf summary:

| Phase | Ops | Device time | Op-to-op gap | Total |
| --- | ---: | ---: | ---: | ---: |
| Prefill | 50 | 798.644 us | 4059.331 us | 4857.975 us |
| Decode | 68 | 543.090 us | 93.426 us | 636.516 us |

## Performance Accounting

Accepted unperturbed host timing:

- Artifact: `logs/final_default_host_timing_real_layer0.log`
- Warmed traced decode host E2E: 559.258 us

Accepted final device timing:

- Artifact: `perf/final_split_fidelity_decode_perf_human.txt`
- Decode device time: 543.090 us
- Decode op-to-op gap: 93.426 us
- Decode total: 636.516 us

Roofline estimate:

- Per-device decode weight bytes:
  - QKV BFP8: `3072 * 1152 * 1 = 3,538,944`
  - O BFP8: `384 * 3072 * 1 = 1,179,648`
  - Gate/up BFP4: `3072 * 2048 * 0.5 = 3,145,728`
  - Down BFP4: `1024 * 3072 * 0.5 = 1,572,864`
  - KV read and norm weights at `seq_len=32`: about `36,864`
- Total estimated bytes per chip: 9,474,048.
- Using about 288 GB/s per Wormhole chip, the 1x8 aggregate lower bound is about 0.0329 ms/token.
- Actual device time is about 0.543 ms/token because small ops, CCL, and layout movement dominate this one-layer decode path.

Same-run accounting profile:

```bash
PHI35_RUN_PERF=1 PHI35_HOST_TIMING_ITERS=1 python -m tracy -r -p -v -o models/autoports/microsoft_phi_3_5_mini_instruct/doc/optimized_multichip_decoder/tracy/final_accounting_short -m pytest --confcutdir=models/autoports/microsoft_phi_3_5_mini_instruct -q models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_multichip_decoder.py::test_multichip_dense_layer_real_weights_perf_profile_1x8_ring -s
```

Artifacts:

- `logs/final_accounting_short_tracy_host_timing.log`
- `tracy/final_accounting_short/reports/2026_06_15_16_15_30/ops_perf_results_2026_06_15_16_15_30.csv`
- `perf/final_accounting_short_decode_perf_human.txt`
- `perf/final_accounting_short_decode_perf_report.csv`
- `perf/final_accounting_short_prefill_perf_human.txt`
- `perf/final_accounting_short_prefill_perf_report.csv`

Result:

- Host timed decode in same profiled run: 781.883 us. This is profiler-perturbed and is not the accepted latency number.
- tt-perf decode device/gap in same profiled run: 546 us / 639 us.
- The same-run profile proves the same optimized path and reconciles the roofline/device/host terms, while accepted performance uses the less-perturbed final host run and final profile above.

Rejected same-run attempt:

- `PHI35_HOST_TIMING_ITERS=100` was tried under Tracy in `logs/final_accounting_tracy_host_timing.log`.
- It passed and printed 647.957 us, but profiler DRAM buffers overflowed and ARC lock waits appeared.
- No usable ops report was produced. The large rejected Tracy directory was removed; the log remains as blocker evidence.

## API and Hardware Findings

Async CCL:

- `PHI35_MULTICHIP_CCL=async_all_reduce` is implemented as an opt-in path with precreated semaphores.
- Measured slower than sync, so default remains `sync_all_reduce`.
- Current `ttnn.all_reduce` already maps to async RS/AG internals for this topology, so explicit semaphore plumbing did not help.

Fused matmul-CCL:

- `ttnn/cpp/ttnn/operations/experimental/ccl/matmul_reduce_scatter_async/matmul_reduce_scatter_async_nanobind.cpp` requires persistent intermediate/output buffers, explicit semaphores, `dim`, and grid offset.
- The API returns matmul output and reduce-scatter output, not the replicated residual required by the decoder layer boundary.
- Keeping replicated residual would require an immediate all-gather. Holding sharded residual across layers would require broader distributed RMSNorm/fused all-gather matmul work outside this goal.
- Decision: rejected for this pass.

Collective placement:

- Collectives remain inside the layer after row-parallel O and down projections.
- No collective is inserted between decoder layers.

Semaphore/preallocated-buffer reuse:

- Explicit semaphore reuse was tested through async all-reduce and was slower.
- Fused CCL persistent-buffer reuse was rejected because it changes the residual contract or adds an immediate all-gather.

Residual layout:

- Boundary contract for full-model bringup: replicated BF16 residual, shape `[1, 1, T, 3072]`, tile layout, present on all 8 devices.
- Internal layer layout may temporarily use width-sharded L1 activations and BF8 CCL payloads.
- Full-model bringup should preserve this boundary and avoid rediscovering inter-layer all-gather/reshard patterns.

Activation sharding and DRAM-sharded decode matmuls:

- Decode local hidden/intermediate activations are width-sharded in L1 only inside the layer.
- O/down decode matmuls use DRAM-sharded program configs with `in0_block_w=2`.
- Explicit output memory configs are required to avoid planner mismatch warnings.

Precision/fidelity:

- Decode CCL payloads use BF8 with BF16 restore after the collective.
- Decode compute uses LoFi.
- Prefill remains BF16/HiFi2 due PCC and profile evidence.

Output subblock:

- Not applicable to the DRAM-sharded matmul program config used here. The available constructor does not expose output subblock fields.

MoE:

- Not applicable. Phi-3.5-mini-instruct is dense.

## Runtime Audits

Static fallback audit:

```bash
pytest --confcutdir=models/autoports/microsoft_phi_3_5_mini_instruct -q models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_multichip_decoder.py::test_multichip_mesh_plan_static models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_multichip_decoder.py::test_runtime_forward_fallback_audit_static -s
```

Artifact: `logs/final_static_fallback_audit.log`

Result: 2 passed.

Repeated-run coverage:

- Initial concurrent attempt failed because a watcher run owned or poisoned the mesh state.
- Boards were reset with `tt-smi -r all`.
- Serial retry passed.

Artifact: `logs/final_repeated_determinism_retry.log`

Retry result:

- Prefill repeated PCC: 0.9999947598886589
- Decode repeated PCC: 0.9999894454586629

Watcher:

- Clean final watcher run:

```bash
RUN=2026_06_15_optimized_1x8_ring_real_watcher10
BASE=models/autoports/microsoft_phi_3_5_mini_instruct/doc/optimized_multichip_decoder/watcher/$RUN
TT_METAL_WATCHER=10 TT_METAL_WATCHER_NOINLINE=1 TT_METAL_WATCHER_APPEND=1 \
TT_METAL_LOGS_PATH="$BASE" \
pytest --confcutdir=models/autoports/microsoft_phi_3_5_mini_instruct -q \
  models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_multichip_decoder.py::test_multichip_dense_layer_real_weights_prefill_decode_pcc_1x8_ring -s \
  2>&1 | tee "$BASE/pytest.log"
```

Result:

- `1 passed in 140.68s`.
- Prefill PCC: 0.999991788840335.
- Decode PCC: 0.999975693890481.
- Watcher disabled features: `None`.
- All eight devices detached cleanly.
- Ethernet retraining events were `0` for all reported cores.
- Minimum reported free stack was `456 bytes` on TRISC0 in `sdpa_flash_decode.cpp`.

Runtime scan:

```bash
rg -n -i "TT_FATAL|TT_THROW|watcher[^\\n]*error|exception|assert|out[ -]?of[ -]?bounds|stack overflow|l1[^\\n]*overflow|noc[^\\n]*(bad|error)|bad noc|retraining events: [1-9]" \
  models/autoports/microsoft_phi_3_5_mini_instruct/doc/optimized_multichip_decoder/watcher/2026_06_15_optimized_1x8_ring_real_watcher10/generated/watcher/watcher.log
```

Result: no matches.

Artifacts:

- `watcher/2026_06_15_optimized_1x8_ring_real_watcher10/pytest.log`
- `watcher/2026_06_15_optimized_1x8_ring_real_watcher10/generated/watcher/watcher.log`

Earlier watcher attempts retained as rejected diagnostics:

- `logs/final_watcher_multichip_smoke.log`: full prefill+traced-decode watcher smoke with Ethernet checks disabled; no pass marker after about 29 minutes.
- `logs/final_watcher_decode_trace_smoke.log`: reduced traced-decode watcher smoke with Ethernet checks disabled; stalled at early tilize/input-transfer kernels.
- `logs/final_watcher_decode_trace_smoke_eth.log`: reduced traced-decode watcher smoke without `TT_METAL_WATCHER_NOINLINE=1`; failed during mesh open with ERISC code-region overflow.
- `watcher/final_watcher_decode_trace_smoke_eth_watcher.log`: copied watcher-side log for the failed no-noinline attempt.
