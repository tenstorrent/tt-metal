# Optimized Multichip Decoder Work Log

Date: 2026-06-15

Scope: optimize the completed 1x8 multichip decoder for
`meta-llama/Llama-3.2-1B-Instruct` in place. Full-model and vLLM work were not
started.

Final status: complete for this stage. The final path is the multichip decoder
on the target 1x8 T3K mesh with BFP8 residual CCL payloads, W2 16-core decode,
persistent CCL buffers, fused WO all-gather matmul, clean fallback audit,
stress coverage, watcher-clean evidence, and final `tt-perf-report` artifacts.
All applicable prompt and `$optimize` checklist items were tried; no applicable
optimization is deferred.

## Code Changes

- `../../tt/multichip_decoder.py`
  - Added env helpers for reproducible optimization trials.
  - Defaulted `_all_gather_hidden` and `_reduce_scatter_hidden` payloads to
    `ttnn.bfloat8_b`.
  - Left `MD_MULTICHIP_ALL_GATHER_DTYPE` and
    `MD_MULTICHIP_REDUCE_SCATTER_DTYPE` as reproducible overrides.
  - Added persistent ping-pong CCL output buffers controlled by
    `MD_MULTICHIP_USE_PERSISTENT_CCL_BUFFERS`; default is enabled.
  - Added `MD_MULTICHIP_MLP_W1_W3_TARGET_CORES` and
    `MD_MULTICHIP_MLP_W2_TARGET_CORES`; W2 now defaults to 16 target cores.
  - Added `decode_w2_partial_output_memcfg` so W2 output sharding matches the
    chosen W2 decode grid.
  - Populated fused WO `allowed_worker_cores` explicitly.
  - Added `MD_MULTICHIP_USE_QK_FUSED_DECODE` as a rejected trial knob.
  - Updated the mesh plan to record BFP8 CCL payloads and the replicated
    inter-layer residual contract.
- `../../tests/test_multichip_decoder.py`
  - Added `MD_MULTICHIP_ARTIFACT_DIR` so final and trial artifacts can be kept
    separate from the completed multichip decoder artifacts.
  - Warmed tensors before the fallback guard so persistent-buffer allocation is
    outside the measured hot path.

## Final Commands

Compile check:

```bash
python -m py_compile models/autoports/meta_llama_llama_3_2_1b_instruct/tt/multichip_decoder.py models/autoports/meta_llama_llama_3_2_1b_instruct/tests/test_multichip_decoder.py
```

Result: passed.

Final default-path correctness, 8192 real-weight check, fallback audit, and
5-iteration stress:

```bash
MD_MULTICHIP_ARTIFACT_DIR=models/autoports/meta_llama_llama_3_2_1b_instruct/doc/optimized_multichip_decoder MD_PREFILL_SEQ_LEN=8192 MD_STRESS_ITERS=5 pytest --timeout=900 --confcutdir=models/autoports/meta_llama_llama_3_2_1b_instruct/tests -q -s models/autoports/meta_llama_llama_3_2_1b_instruct/tests/test_multichip_decoder.py::test_multichip_decoder_contract_and_runtime_fallback_audit models/autoports/meta_llama_llama_3_2_1b_instruct/tests/test_multichip_decoder.py::test_multichip_static_mesh_plan_uses_optimized_baseline models/autoports/meta_llama_llama_3_2_1b_instruct/tests/test_multichip_decoder.py::test_synthetic_multichip_paged_prefill_decode_trace_and_determinism models/autoports/meta_llama_llama_3_2_1b_instruct/tests/test_multichip_decoder.py::test_real_weights_multichip_paged_prefill_and_decode_trace models/autoports/meta_llama_llama_3_2_1b_instruct/tests/test_multichip_decoder.py::test_runtime_fallback_audit_measured_multichip_prefill_and_traced_decode models/autoports/meta_llama_llama_3_2_1b_instruct/tests/test_multichip_decoder.py::test_multichip_repeated_run_stress
```

Result: 6 passed.

Final non-profiler latency:

```bash
MD_MULTICHIP_ARTIFACT_DIR=models/autoports/meta_llama_llama_3_2_1b_instruct/doc/optimized_multichip_decoder MD_PERF_PREFILL_SEQ_LEN=8192 pytest --timeout=900 --confcutdir=models/autoports/meta_llama_llama_3_2_1b_instruct/tests -q -s models/autoports/meta_llama_llama_3_2_1b_instruct/tests/test_multichip_decoder.py::test_perf_artifact_signposted_multichip_prefill_and_decode
```

Result: passed. `perf_trace_contract.json` reports prefill 13.971094 ms and
traced decode replay 0.648592 ms.

Final watcher:

```bash
TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1 MD_MULTICHIP_ARTIFACT_DIR=models/autoports/meta_llama_llama_3_2_1b_instruct/doc/optimized_multichip_decoder pytest --timeout=900 --confcutdir=models/autoports/meta_llama_llama_3_2_1b_instruct/tests -q -s models/autoports/meta_llama_llama_3_2_1b_instruct/tests/test_multichip_decoder.py::test_runtime_fallback_audit_measured_multichip_prefill_and_traced_decode
```

Result: passed. The watcher log audit command found zero fatal/error/timeout
matches:

```bash
rg -n "ERROR|FATAL|ASSERT|TIMEOUT|Watchdog|hang|unhealthy|TT_THROW|TT_FATAL|Watcher read invalid|Timeout waiting" generated/watcher/watcher.log || true
```

The scoped `TT_METAL_WATCHER_DISABLE_ETH=1` retry was used because a prior full
watcher attempt failed before decoder execution with:

```text
idle_erisc.elf: segment[0] [0x3f10,+0x5a58) overflows region:0 limit of 0x54c0 bytes
```

Final Tracy capture:

```bash
MD_MULTICHIP_ARTIFACT_DIR=models/autoports/meta_llama_llama_3_2_1b_instruct/doc/optimized_multichip_decoder MD_PERF_PREFILL_SEQ_LEN=8192 python -m tracy -r -p -v -o models/autoports/meta_llama_llama_3_2_1b_instruct/doc/optimized_multichip_decoder/tracy/final_persistent -m pytest --timeout=900 --confcutdir=models/autoports/meta_llama_llama_3_2_1b_instruct/tests -q -s models/autoports/meta_llama_llama_3_2_1b_instruct/tests/test_multichip_decoder.py::test_perf_artifact_signposted_multichip_prefill_and_decode
```

Result: passed. Raw reports came from
`tracy/final_persistent/reports/2026_06_15_15_56_48/`.

Final `tt-perf-report` generation:

```bash
ARTIFACT_DIR=models/autoports/meta_llama_llama_3_2_1b_instruct/doc/optimized_multichip_decoder
PERF_DIR="$ARTIFACT_DIR/perf"
RAW="$ARTIFACT_DIR/tracy/final_persistent/reports/2026_06_15_15_56_48/ops_perf_results_2026_06_15_15_56_48.csv"
mkdir -p "$PERF_DIR"
cp "$RAW" "$PERF_DIR/ops_perf_results_raw.csv"
cp "$ARTIFACT_DIR/tracy/final_persistent/reports/2026_06_15_15_56_48/profile_log_device.csv" "$PERF_DIR/profile_log_device_raw.csv"
cp "$ARTIFACT_DIR/tracy/final_persistent/reports/2026_06_15_15_56_48/tracy_profile_log_host.tracy" "$PERF_DIR/tracy_profile_log_host.tracy"
tt-perf-report --no-color --no-host-ops --start-signpost PERF_MULTICHIP_PREFILL --end-signpost PERF_MULTICHIP_PREFILL_END --csv "$PERF_DIR/prefill_8192_report.csv" --summary-file "$PERF_DIR/prefill_8192_summary.csv" "$PERF_DIR/ops_perf_results_raw.csv"
tt-perf-report --no-color --no-host-ops --no-merge-devices --start-signpost PERF_MULTICHIP_PREFILL --end-signpost PERF_MULTICHIP_PREFILL_END --csv "$PERF_DIR/prefill_8192_per_device_report.csv" --summary-file "$PERF_DIR/prefill_8192_per_device_summary.csv" "$PERF_DIR/ops_perf_results_raw.csv"
tt-perf-report --no-color --no-host-ops --start-signpost PERF_MULTICHIP_DECODE --end-signpost PERF_MULTICHIP_DECODE_END --csv "$PERF_DIR/decode_trace_replay_report.csv" --summary-file "$PERF_DIR/decode_trace_replay_summary.csv" "$PERF_DIR/ops_perf_results_raw.csv"
tt-perf-report --no-color --no-host-ops --no-merge-devices --start-signpost PERF_MULTICHIP_DECODE --end-signpost PERF_MULTICHIP_DECODE_END --csv "$PERF_DIR/decode_trace_replay_per_device_report.csv" --summary-file "$PERF_DIR/decode_trace_replay_per_device_summary.csv" "$PERF_DIR/ops_perf_results_raw.csv"
tt-perf-report --no-color --no-host-ops --start-signpost PERF_MULTICHIP_PREFILL --end-signpost PERF_MULTICHIP_PREFILL_END --no-summary "$PERF_DIR/ops_perf_results_raw.csv" > "$PERF_DIR/prefill_8192_tt_perf_report.txt"
tt-perf-report --no-color --no-host-ops --no-merge-devices --start-signpost PERF_MULTICHIP_PREFILL --end-signpost PERF_MULTICHIP_PREFILL_END --no-summary "$PERF_DIR/ops_perf_results_raw.csv" > "$PERF_DIR/prefill_8192_per_device_tt_perf_report.txt"
tt-perf-report --no-color --no-host-ops --start-signpost PERF_MULTICHIP_DECODE --end-signpost PERF_MULTICHIP_DECODE_END --no-summary "$PERF_DIR/ops_perf_results_raw.csv" > "$PERF_DIR/decode_trace_replay_tt_perf_report.txt"
tt-perf-report --no-color --no-host-ops --no-merge-devices --start-signpost PERF_MULTICHIP_DECODE --end-signpost PERF_MULTICHIP_DECODE_END --no-summary "$PERF_DIR/ops_perf_results_raw.csv" > "$PERF_DIR/decode_trace_replay_per_device_tt_perf_report.txt"
```

Result: human-readable tables, CSVs, summary plots, raw profile logs, and
`perf/perf_provenance.json` exist.

Hardware health checks:

```bash
timeout 30 tt-smi -ls --local
```

Result: passed after watcher and after profiler runs.

## Correctness Evidence

| Artifact | Prefill length | Prefill PCC | Decode PCC | Repeated trace PCC |
| --- | ---: | ---: | ---: | ---: |
| `synthetic_correctness.json` | 128 | 0.9999905634530127 | 0.9999904891680687 | 1.0 |
| `real_weight_correctness.json` | 128 | 0.9999908909695797 | 0.9999914076595784 | 0.9999999999999881 |
| `real_weight_correctness_prefill_8192.json` | 8192 | 0.9999913308638281 | 0.9999914965229365 | 1.0 |

Runtime fallback audit:

- Artifact: `runtime_fallback_audit.json`.
- Status: passed.
- Guarded bridges: `ttnn.from_torch`, `ttnn.to_torch`.
- Measured passes: `prefill_forward`, `decode_forward_trace_capture_and_replay`.

Stress:

- Artifact: `stress_repeated_runs.json`.
- Iterations: 5.
- Prefill PCCs: all `0.9999904966442884`.
- Status: passed.

## Before/After Perf Trials

| Trial | Command env | Prefill 8192 host ms | Traced decode host ms | Result |
| --- | --- | ---: | ---: | --- |
| Completed multichip baseline | existing `../multichip_decoder/perf_trace_contract.json` | 15.922505408525467 | 0.6744153797626495 | Before |
| Default BF16 before cleanup | `MD_MULTICHIP_ARTIFACT_DIR=.../trials/default` | 16.001302748918533 | 0.6563551723957062 | Trial only |
| W2 output memcfg cleanup, BF16 CCL | `.../trials/partial_memcfg_default` | 15.884991735219955 | 0.6652176380157471 | Kept |
| W2 target cores 16 | `MD_MULTICHIP_MLP_W2_TARGET_CORES=16` | 16.007505357265472 | 0.662926584482193 | Later combined with BFP8 |
| All-gather BFP8 only | `MD_MULTICHIP_ALL_GATHER_DTYPE=bfloat8_b` | 14.467429369688034 | 0.676862895488739 | Rejected, decode regression |
| Reduce-scatter BFP8 only | `MD_MULTICHIP_REDUCE_SCATTER_DTYPE=bfloat8_b` | 15.041273087263107 | 0.8262209594249725 | Rejected, decode regression |
| All-gather + reduce-scatter BFP8 | both CCL dtype envs set to `bfloat8_b` | 13.916712254285812 | 0.6688199937343597 | Selected CCL policy |
| W2 target cores 16 plus BFP8 CCL | default after W2 change | 13.866011053323746 | 0.6657689809799194 | Selected W2 tiling |
| Persistent CCL buffers | `MD_MULTICHIP_USE_PERSISTENT_CCL_BUFFERS=1` | 13.87469470500946 | 0.6451494991779327 | Selected default |
| Final non-profiler selected path | root `perf_trace_contract.json` | 13.97109404206276 | 0.6485916674137115 | After |

The selected path passed real-weight 8192 PCC before and after defaulting.

## Perf-Report Evidence

Final same-run profile accounting:

| Phase | Profile host signpost ms | Device time ms | Reported gap ms | Device window ms |
| --- | ---: | ---: | ---: | ---: |
| Prefill 8192 | 14.499375 | 13.630363 | 0.022878 | 13.653241 |
| Traced decode replay | 0.617592 | 0.352362 | 0.261231 | 0.613593 |

Prefill report:

- 26 device ops, 0 host ops.
- Explicit CCL device time: `AllGatherAsyncDeviceOperation` 4564.596 us
  across 3 ops plus `ReduceScatterMinimalAsyncDeviceOperation` 1279.329 us.
- Completed multichip baseline CCL device time was 8193.324 us, so the final
  profile reduced explicit prefill CCL time by about 2.35 ms.
- Remaining top ops include SDPA 1218.337 us, WQKV 952.195 us, MLP W1/W3
  742.755/739.686 us, and W2 791.460 us.
- Advice still asks to increase two 8-core prefill grids and place large MLP
  prefill input 0 in L1. The grid advice is not actionable on this 1x8 path,
  and the L1 advice was rejected because 8192-token prefill activations are
  intentionally DRAM interleaved.

Decode report:

- 30 device ops, 0 host ops.
- Device time excluding gaps: 352.362 us. Device window including gaps:
  613.593 us.
- Decode CCLs: fused WO all-gather matmul 33.466 us, two standalone BFP8
  all-gathers 43.316 us total, and one BFP8 reduce-scatter 41.744 us.
- DRAM-sharded decode matmuls are used for WQKV, W1, W3, and W2. The final CSV
  reports L1 width-sharded input and `in0_block_w=2` for these matmuls.
- The prior W2 `in0_block_w=1` advice is resolved by the 16-core W2 target.
- The fused WO output subblock remains `1x1`. For the common `Attention1D`
  fused Ring config, hidden 2048 / TP 8 / tile 32 / `8x1` grid gives
  `do_per_core_N=1`; `_get_out_subblock_w(1)` cannot legally return 2.
- The decode gap advice suggests tracing, but the measured path is already
  `ttnn.execute_trace` replay. Same-run Tracy signposts show the replay host
  window is 0.617592 ms versus 0.613593 ms device window, so this is recorded
  as a report/window synchronization limitation rather than host fallback.

Roofline:

- Lower-bound decode bytes per layer-token at position 8192:
  35,651,584 weight bytes, 8,388,608 KV-cache bytes, and 8,192 norm bytes.
- Aggregate DRAM bandwidth assumption: 2304 GB/s.
- Lower-bound roofline: 0.019118 ms/token.
- The measured 0.613593 ms device window is limited by small kernels and CCLs,
  not by the simple DRAM streaming lower bound.

## Rejected Or Limited Options

| Option | Evidence | Result |
| --- | --- | --- |
| BFP8 all-gather only | `trials/ag_bfp8/`, `trials/ag_bfp8_real8192/` | PCC passed, prefill improved, decode regressed; not selected. |
| BFP8 reduce-scatter only | `trials/rs_bfp8/` | PCC passed, prefill improved, decode regressed materially; rejected. |
| BFP8 all-gather + reduce-scatter | `trials/ccl_bfp8_both/`, root correctness artifacts | Selected. |
| Decode W2 target cores 16 | `trials/w2_16_bfp8_default/`, final decode CSV | Selected; final W2 `in0_block_w=2`. |
| Decode W1/W3 target cores 16 | test failure output from trial | Rejected: layernorm required `block_w (2)` to equal `K / num_cores (4)`. |
| Fused QK decode | `trials/qk_fused/rejection.log` | Rejected: fused rotary requires cos/sin batch equal `q_batch + k_batch`; current helper supplies non-fused shape. |
| Fused WO larger output subblock | final decode report, `attention_1d.py` config | Rejected: local fused output has `per_core_N=1`, so subblock area >=2 is not legal. |
| Sharded residual between decoder layers | mesh plan and code inspection | Rejected for this stage: common norm and gathered-input matmul contracts require a replicated boundary. |
| Prefill L1 activation advice | final prefill report | Rejected: large 8192-token prefill activations are intentionally DRAM interleaved. |
| HiFi4 advice | final reports and PCC artifacts | Rejected: PCC is already >0.99999 and HiFi4 is a fidelity/cost increase, not a speed optimization. |
| Full watcher after ETH overflow | watcher artifacts | Full watcher ETH path overflows `idle_erisc.elf`; scoped ETH-disabled watcher passed cleanly. |

## Optimize Checklist Closure

- Functional checks still pass against the optimized path: yes.
- Prefill and decode PCC remain at acceptance: yes, all recorded PCCs exceed
  0.99999.
- Paged KV-cache and warmed trace replay behave correctly: yes.
- Runtime fallback audit remains clean: yes.
- Stress coverage matches risk: yes, 5 repeated runs after CCL/buffer changes.
- Warmed prefill and traced decode latency before/after: yes.
- Final `tt-perf-report` tables, CSVs, and provenance: yes.
- Watcher still clean: yes, with documented ETH watcher limitation and clean
  final log audit.
- Decoder path fully traced with no host fallbacks: yes.
- Decode activations generally width-sharded in L1: yes within this replicated
  boundary layer contract.
- Prefill activations generally DRAM interleaved with 2D matmul configs: yes.
- SDPA and optimized composite TTNN ops used where applicable: yes.
- Explicit memory, program, and compute configs: yes.
- Shard specs and grids divide dimensions cleanly: yes for selected grids.
- DRAM-sharded decode matmuls: yes for WQKV, W1, W3, W2.
- Fused matmul-CCL opportunities: WO fused all-gather matmul used; QK fused
  rejected with API-shape evidence.
- MoE gate-selected active experts: not applicable; this model is dense.
- LM head and sampling: not applicable to this decoder-layer stage.
- Reduced precision/fidelity experiments: BFP8 CCL payloads selected; HiFi4
  rejected with PCC/perf rationale.
- Performance accounting reconciled: yes in `perf/perf_provenance.json`.

## Artifacts

- Mesh and layout: `mesh_strategy.json`
- Latency: `perf_trace_contract.json`
- Correctness: `synthetic_correctness.json`, `real_weight_correctness.json`,
  `real_weight_correctness_prefill_8192.json`
- Fallback and stress: `runtime_fallback_audit.json`,
  `stress_repeated_runs.json`
- Perf reports and provenance: `perf/`
- Trials: `trials/`
- Watcher evidence: `watcher/watcher_clean_final_persistent_eth_disabled.log`,
  `watcher/watcher_clean_final_persistent_eth_disabled_summary.json`,
  `watcher/kernel_elf_paths_final_persistent_eth_disabled.txt`,
  `watcher/kernel_names_final_persistent_eth_disabled.txt`
