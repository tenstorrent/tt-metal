# Llama 3.1 8B Instruct Optimized Full Model

Status: complete as of 2026-06-15 for the optimized-full-model stage. No vLLM
integration work was started.

## Top-Line Metrics

Target path: `meta-llama/Llama-3.1-8B-Instruct`, T3K `1x8` Ring mesh,
`FABRIC_1D_RING`, batch 1.

| Metric | Completed full-model baseline | Optimized full-model evidence |
| --- | ---: | ---: |
| AIME24 teacher-forcing TTFT | 1094.60 ms | 648.31 ms |
| AIME24 teacher-forcing decode | 22.18 t/s/u | 49.31 t/s/u |
| AIME24 teacher-forcing e2e | 17.99 t/s/u | 37.65 t/s/u |
| Token-out prompt/generated shape | 60 / 100 | 128 / 128 |
| Token-out TTFT | 616.64 ms | 629.73 ms |
| Token-out steady traced replay | 69.21 t/s/u | 70.58 t/s/u |
| Token-out steady latency | 14.45 ms/token | 14.17 ms/token |
| Token-out sampled-token readbacks in steady path | readback path | 0 |
| Greedy sampling path | split sampling | split sampling, `force_argmax=False` |
| Watcher token-out runtime audit | not run | pass, no readbacks |
| Split greedy sampler trace | not measured | 0.514226 ms/replay |

The optimized token-out number is the no-readback serving-style path:
persistent token/current-position/RoPE/page-table inputs, `tt_out_tok`
feedback, device-side position and RoPE advance, nonblocking trace replay, one
final synchronize, and no per-token host readback.

The clean watcher run is scoped to the measured optimized token-out path. The
normal evidence run separately covers trace feedback, changed-only page tables,
and top-k/top-p-capable split sampling.

## Preserved Policy

The full model keeps the optimized multichip decoder policy:
`llama31_8b_t3k_1x8_tp8_bfp4_attn_bfp4_mlp_bfp8_act_decode_v2`.

- BFP8 activations and KV cache.
- BFP4 attention and MLP gate/up/down weights.
- LoFi MLP math.
- Async hidden all-gathers, fused decode all-gather plus WO matmul, async MLP
  reduce-scatter, and persistent decode CCL buffers.
- Replicated full-hidden residual at the inter-layer decoder boundary, with
  width-sharded L1 residuals inside each decode layer.
- BF16 token embedding and final RMSNorm, split BF8 LM head, canonical split
  sampling with `max_top_k=32`, padded vocab 128256, and `force_argmax=False`.

No broad datatype frontier search was run here; that remains owned by the
datatype-sweep stage.

## Readiness

Fresh AIME24 chat-template reference:

```bash
timeout 7200s python_env/bin/python -u -m models.common.readiness_check.generate \
  --hf-model meta-llama/Llama-3.1-8B-Instruct \
  --prompt-source aime24 \
  --chat-template \
  --gen-len 100 \
  --top-k 100 \
  --output models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_full_model/aime24_chat_template_100_top100.refpt
```

Prefill:

```bash
timeout 7200s python_env/bin/python -u -m models.common.readiness_check.run_prefill_check \
  --model-dir models/autoports/meta_llama_llama_3_1_8b_instruct \
  --reference models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_full_model/aime24_chat_template_100_top100.refpt \
  --mesh-device T3K \
  --fabric-config FABRIC_1D_RING
```

Result: top1 `90/100`, top5 `100/100`, top100 `100/100`.

Teacher forcing:

```bash
timeout 7200s python_env/bin/python -u -m models.common.readiness_check.run_teacher_forcing \
  --model-dir models/autoports/meta_llama_llama_3_1_8b_instruct \
  --reference models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_full_model/aime24_chat_template_100_top100.refpt \
  --mesh-device T3K \
  --fabric-config FABRIC_1D_RING
```

Result: top1 `92/100`, top5 `100/100`, top100 `100/100`,
TTFT `648.31 ms`, decode `49.31 t/s/u`, e2e `37.65 t/s/u`.

Autoregressive:

```bash
timeout 7200s python_env/bin/python -u -m models.common.readiness_check.run_autoregressive \
  --model-dir models/autoports/meta_llama_llama_3_1_8b_instruct \
  --hf-model meta-llama/Llama-3.1-8B-Instruct \
  --mesh-device T3K \
  --fabric-config FABRIC_1D_RING \
  --max-new-tokens 128 \
  --output-dir models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_full_model/autoregressive_story_128
```

HF and TT each produced 128 tokens. Degeneracy check:

```bash
python_env/bin/python models/common/readiness_check/check_degenerate_output.py \
  models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_full_model/autoregressive_story_128 \
  --scope autoregressive \
  --missing-artifacts critical \
  --json models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_full_model/autoregressive_degenerate_report.json
```

Result: no findings, adjacent duplication `0.0`, trigram loop fraction
`0.0297`, HF/TT token agreement `30/128`.

## Token-Out Trace Evidence

Artifact: `token_out_trace_evidence.json`.

The measured no-readback token-out run used a 128-token prompt and generated
128 tokens.

| Metric | Value |
| --- | ---: |
| TTFT | 629.731884 ms |
| First decode step with model and sampling trace capture | 536.401989 ms |
| Steady replay tokens | 126 |
| Steady replay elapsed | 1.785314 s |
| Steady replay throughput | 70.575829 t/s/u |
| Steady replay latency | 14.169157 ms/token |

Host-boundary audit for the no-readback path:

| Counter | Value |
| --- | ---: |
| Sampled-token readbacks | 0 |
| Token input host copies | 1 |
| Position host copies | 1 |
| RoPE index host copies | 1 |
| Page-table host copies | 1 |
| Position device increments | 127 |
| RoPE index device increments | 127 |

Trace feedback and sampling checks passed: `tt_out_tok` feedback updates the
persistent decode token input, current position and RoPE index advance on
device, unchanged page tables are not recopied, changed page tables are copied
only when changed, and top-k/top-p smoke keeps the same split-sampling contract.
The greedy path remains semantically greedy through split sampling with
`force_argmax=False`; no force-argmax shortcut was accepted as completion
evidence.

## Sampler Strategy Benchmark

Artifact: `sampling_strategy_benchmark.json`.

The model exposes one on-device greedy strategy through
`Llama31_8B_InstructFullModel.make_sampling_args()`: canonical split sampling
with `max_top_k=32`, `sampling_all_gather_axis=1`, and `force_argmax=False`.
`SAMPLING_AG_CONFIG.allow_force_argmax` is not present in the model args, and
the readiness generator rejects force-argmax if it unexpectedly activates for
greedy decode.

The benchmark used real full-model logits on the T3K mesh:

| Strategy | Params | ms/replay | Correctness |
| --- | --- | ---: | --- |
| Split greedy | temp 1.0, top-k 1, top-p 0.0 | 0.514226 | sampled token `11` matched row-0 argmax `11` |
| Split top-k/top-p | temp 0.7, top-k 8, top-p 0.9 | 0.514234 | traced split path, `force_argmax=False` |
| Force-argmax | unavailable in this model contract | n/a | not selected |

## Lower Bound And Attribution

Optimized multichip decoder layer latency:

| Source | Min | Avg |
| --- | ---: | ---: |
| One decoder layer | 0.397455879 ms | 0.401623140 ms |
| 32-layer stack lower bound | 12.718588114 ms | 12.851940468 ms |

Reduced one-layer full-model token-out profile:

| Metric | Min | Avg |
| --- | ---: | ---: |
| One-layer full token-out replay | 1.551530790 ms | 1.583330682 ms |
| Measured non-stack terminal work | 1.154074911 ms | 1.181707543 ms |
| Stack lower bound plus terminal work | 13.872663025 ms | 14.033648011 ms |

Measured full 32-layer no-readback token-out replay is `14.169156950 ms/token`.
That is `0.296494 ms` (`2.14%`) above the min stack-plus-terminal estimate, or
`0.135509 ms` (`0.97%`) above the avg stack-plus-terminal estimate. This is
inside the 10-15% tolerance, so there is no large avoidable full-model gap to
close before completion.

Reduced-profile token-out top device-time buckets:

| Bucket | Device time | Share |
| --- | ---: | ---: |
| Width-sharded matmuls | 544.46 us | 39.35% |
| `TopKDeviceOperation` | 154.23 us | 11.15% |
| `AllGatherDeviceOperation` | 133.84 us | 9.67% |
| `SamplingDeviceOperation` | 63.81 us | 4.61% |
| `AllGatherAsyncDeviceOperation` width-sharded | 54.84 us | 3.96% |
| `ReduceScatterMinimalAsyncDeviceOperation` | 53.54 us | 3.87% |
| `PadDeviceOperation` | 48.54 us | 3.51% |
| `AllGatherMatmulAsyncDeviceOperation` | 42.98 us | 3.11% |

The generic `TopKDeviceOperation` and vocab all-gather are visible costs, but
they do not dominate the measured no-readback token-out path and the full path
is within the stack-plus-terminal lower-bound tolerance. Sampling contract
cleanup remains a future optimization item only if post-recovery runs show this
bucket grows into the dominant gap.

## tt-perf-report

Reduced full-model profiler harness:
`doc/full_model/reduced_profile.py`.

This is a one-real-layer full-model variant with real embedding, RoPE,
page-table/KV shape, optimized multichip decoder layer, final norm, split LM
head, and no-readback split-sampling trace path.

```bash
timeout 7200s python_env/bin/python -m tracy -r -p -v --check-exit-code \
  -o models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_full_model/tracy/reduced_profile/.logs \
  models/autoports/meta_llama_llama_3_1_8b_instruct/doc/full_model/reduced_profile.py \
  --model-dir models/autoports/meta_llama_llama_3_1_8b_instruct \
  --prompt-file models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_full_model/prompt_128.txt \
  --output-json models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_full_model/reduced_profile_summary.json \
  --decode-replays 4
```

`tt-perf-report` was run over
`tracy/reduced_profile/reduced_profile_ops_perf_results.csv` using the
`PERF_REDUCED_PREFILL` / `PERF_REDUCED_PREFILL_END` and
`PERF_REDUCED_TOKEN_OUT_DECODE` / `PERF_REDUCED_TOKEN_OUT_DECODE_END`
signposts.

Reduced-profile host timings:

| Window | Host timing |
| --- | ---: |
| One-layer prefill | 85.908955 ms |
| One-layer token-out decode min / avg | 1.551531 / 1.583331 ms |

Reduced-profile stacked report highlights:

| Window | Device-time sum | Main buckets |
| --- | ---: | --- |
| Prefill | 1649.30 us | width-sharded matmuls, DRAM-interleaved matmuls, async all-gathers, layernorm |
| Token-out decode | 1383.51 us | matmuls, top-k, vocab all-gather, sampling, async CCLs |

## Watcher And Hardware Status

Full ETH watcher attempt:

```bash
timeout 7200s env \
  TT_METAL_WATCHER=10 \
  TT_METAL_WATCHER_NOINLINE=1 \
  TT_METAL_LOGS_PATH=models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_full_model/watcher/token_out_no_readback_128 \
  python_env/bin/python -u models/autoports/meta_llama_llama_3_1_8b_instruct/doc/full_model/token_out_trace_evidence.py \
  --model-dir models/autoports/meta_llama_llama_3_1_8b_instruct \
  --prompt-file models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_full_model/prompt_128.txt \
  --mesh-device T3K \
  --fabric-config FABRIC_1D_RING \
  --max-new-tokens 128 \
  --output-json models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_full_model/watcher/token_out_no_readback_128/token_out_trace_evidence_watcher.json
```

Result: failed after reaching trace capture/replay. The abort was from the
watcher server's ETH read path:
`Timeout waiting for Ethernet core service remote IO request`.

Scoped retry:

```bash
timeout 7200s env \
  TT_METAL_WATCHER=10 \
  TT_METAL_WATCHER_DISABLE_ETH=1 \
  TT_METAL_WATCHER_NOINLINE=1 \
  TT_METAL_LOGS_PATH=models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_full_model/watcher/token_out_no_readback_128_disable_eth \
  python_env/bin/python -u models/autoports/meta_llama_llama_3_1_8b_instruct/doc/full_model/token_out_trace_evidence.py \
  --model-dir models/autoports/meta_llama_llama_3_1_8b_instruct \
  --prompt-file models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_full_model/prompt_128.txt \
  --mesh-device T3K \
  --fabric-config FABRIC_1D_RING \
  --max-new-tokens 128 \
  --output-json models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_full_model/watcher/token_out_no_readback_128_disable_eth/token_out_trace_evidence_watcher.json
```

Result: failed before model execution during topology discovery:
`ETH core heartbeat check failed on device ASIC ID: 9956368389, ETH core e7-0`.

`tt-smi -ls --local` listed chips `0-7` after both failures.

Recovery:

```bash
tt-smi -r all
tt-smi -ls --local
python_env/bin/python - <<'PY'
import ttnn
from models.common.readiness_check.mesh_device import close_readiness_mesh_device, open_readiness_mesh_device
mesh = open_readiness_mesh_device("T3K", "FABRIC_1D_RING")
close_readiness_mesh_device(mesh, "FABRIC_1D_RING")
print("fabric_open_smoke=pass")
PY
```

Artifacts:

- `hardware_recovery/tt_smi_reset_all_after_watcher_failures.log`
- `hardware_recovery/tt_smi_reset_all_after_second_watcher_timeout.log`
- `hardware_recovery/hardware_status_after_reset.log`
- `hardware_recovery/fabric_open_smoke_after_reset.log`
- `hardware_recovery/hardware_status_after_clean_watcher_and_sampler.log`

A full watcher run after reset completed the no-readback token-out path and
wrote pass JSON, then later timed out in watcher ETH polling during the extra
trace-feedback/top-k probes. The completed measured-path JSON is retained under
`watcher/token_out_no_readback_128_after_reset/token_out_trace_evidence_watcher.json`.

Final scoped watcher run for the measured optimized path:

```bash
timeout 7200s env TT_METAL_WATCHER=10 TT_METAL_WATCHER_NOINLINE=1 \
  TT_METAL_LOGS_PATH=models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_full_model/watcher/token_out_only_128_after_reset \
  python_env/bin/python -u models/autoports/meta_llama_llama_3_1_8b_instruct/doc/full_model/token_out_trace_evidence.py \
  --model-dir models/autoports/meta_llama_llama_3_1_8b_instruct \
  --prompt-file models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_full_model/prompt_128.txt \
  --mesh-device T3K \
  --fabric-config FABRIC_1D_RING \
  --max-new-tokens 128 \
  --token-out-only \
  --output-json models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_full_model/watcher/token_out_only_128_after_reset/token_out_trace_evidence_watcher.json
```

Result: pass. Watcher timing is intentionally slower than non-watcher timing:
TTFT `1346.801965 ms`, steady replay `10.319454 t/s/u`,
`96.904350 ms/token`, sampled-token readbacks `0`, and
token/position/RoPE/page-table host copies `1/1/1/1`. The watcher log ended
with `Watcher thread stopped watching...`, detached devices `0-7`, and reported
zero Ethernet retraining events. The log-only runtime failure signature scan
found no matches.

Runtime fallback audit for `token_out_trace_evidence_stdout.txt`,
`runs/sampling_strategy_benchmark.log`, and the scoped watcher run found no
fallback signatures. Artifact: `runtime_fallback_audit.txt`.

## Known Limitations

- The first sampling trace capture still emits TTNN's allocator warning about
  allocating device buffers while a model trace is active. The trace counters
  and no-readback path are correct, but this should be cleaned up.
- Same-process attempts to run readback token-out and no-readback token-out
  back-to-back can hit a later prefill L1 circular-buffer clash. The measured
  optimized path runs no-readback token-out in a fresh process.
- The clean watcher evidence is scoped to the measured optimized token-out path;
  trace-feedback and top-k/top-p probes are covered by the normal evidence run.

## Artifacts

- `perf_summary.json`
- `aime24_chat_template_100_top100.refpt`
- `runs/generate_aime24_chat_template_100_top100.log`
- `runs/run_prefill_check_aime24.log`
- `runs/run_teacher_forcing_aime24.log`
- `runs/run_autoregressive_story_128.log`
- `autoregressive_story_128/hf_completion.txt`
- `autoregressive_story_128/tt_completion.txt`
- `autoregressive_story_128/autoregressive_meta.json`
- `autoregressive_degenerate_report.json`
- `token_out_trace_evidence.json`
- `token_out_trace_evidence_stdout.txt`
- `runtime_fallback_audit.txt`
- `sampling_strategy_benchmark.json`
- `runs/sampling_strategy_benchmark.log`
- `reduced_profile_summary.json`
- `tracy/reduced_profile/reduced_profile_ops_perf_results.csv`
- `tracy/reduced_profile/reduced_prefill_perf_report.{txt,csv}`
- `tracy/reduced_profile/reduced_prefill_perf_report_stacked.{csv,png}`
- `tracy/reduced_profile/reduced_token_out_decode_perf_report.{txt,csv}`
- `tracy/reduced_profile/reduced_token_out_decode_perf_report_stacked.{csv,png}`
- `tracy/reduced_profile/reduced_token_out_decode_perf_report_per_device.{txt,csv}`
- `tracy/reduced_profile/reduced_token_out_decode_perf_report_per_device_stacked.{csv,png}`
- `watcher/token_out_no_readback_128/watcher_run.log`
- `watcher/token_out_no_readback_128/generated/watcher/watcher.log`
- `watcher/token_out_no_readback_128/hardware_status_after_watcher_timeout.log`
- `watcher/token_out_no_readback_128_disable_eth/watcher_run.log`
- `watcher/token_out_no_readback_128_disable_eth/hardware_status_after_disable_eth_heartbeat_failure.log`
- `watcher/token_out_no_readback_128_after_reset/token_out_trace_evidence_watcher.json`
- `watcher/token_out_no_readback_128_after_reset/watcher_run.log`
- `watcher/token_out_only_128_after_reset/token_out_trace_evidence_watcher.json`
- `watcher/token_out_only_128_after_reset/watcher_run.log`
- `watcher/token_out_only_128_after_reset/generated/watcher/watcher.log`
- `watcher/token_out_only_128_after_reset/runtime_failure_signature_scan_logs_only.txt`
- `hardware_recovery/*`
