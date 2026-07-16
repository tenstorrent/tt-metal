# Gemma 4 31B optimized vLLM work log

## Scope and baseline

Started from main Stage 09 commit `e07e401794d` and nested vLLM plugin commit
`91c467d6fc1`. The selected datatype configuration remained
`lm_head_bfp8_hifi2`; the serving context remained 113280, max sequences 32,
block size 64, `1x4` P150b mesh, async scheduling, trace mode `all`, and device
sampling mode `all`.

The preserved primary baseline was 127 actual input / 128 output / one request /
concurrency 1: TTFT 992.586 ms, TPOT 38.023 ms, ITL P50/P99
29.348/29.739 ms, 21.974 output tok/s, and 26.300 TPOT-derived t/s/u.
The preserved secondary 99 actual input/request / 100 output/request / 32-burst
baseline was 201.070 output tok/s and 12.924 TPOT-derived burst t/s/u.

## Operation-topology audit and change

The model and greedy sampler graphs were already split, traced, and replayed
nonblocking. Greedy LM-head sampling was already the full-model TP4 split
sampler, so graph fusion, datatype retuning, a force-argmax sampler, and a
generic sampler replacement were rejected.

The remaining avoidable work was in the adapter boundary. On every steady
token it parsed stale host token/position tensors and converted/compared sixty
hybrid page tables even though the async scheduler had already proved no reset,
batch change, or page-table change. The plugin now communicates that proof via
`reuse_device_decode_inputs`; only that branch reuses all persistent device
inputs without host inspection. Unproven and changed state retains the original
validation/refresh/recapture behavior.

An autofix audit caught and repaired an initial overbroad proof before commit:
`reset_batch` reports row/layout changes but an existing request may receive
`new_block_ids` at a normal 64-token KV boundary without changing layout. Two
independent source reviews reproduced the stale-table risk. The final plugin
uses a separate scheduler-derived `page_tables_changed` marker, drains pending
overlap before the mutation, disables reuse for that step, refreshes table
inputs without recapture, and resumes reuse on the next unchanged step.

The fresh-run allocator anomaly required two evidence-driven repairs. First,
the sampler path redundantly executed its exact workload immediately before
capture; capture is now capture-only and follows exact prewarm plus model
capture. A subsequent warning occurred only when the first long batch-1 request
crossed its 64-token KV boundary. That isolated the remaining first-use program
allocation to the persistent page-table `ttnn.copy`. `initialize_trace_state`
now prewarms each distinct source/target copy pair before recording identities,
generations, or any trace. The final full runner crossed the same boundaries
with zero unsafe allocator warnings. See `anomaly_ledger.md` and
`evidence/final_server.log`.

Rejected options:

- aligned-only prompt specialization: violates serving capability; 149-token
  non-aligned evidence remains mandatory;
- lower max length or smaller benchmark/eval context: violates the context
  contract and is unnecessary;
- host argmax/full-logits readback: violates on-device sampling and async;
- force-argmax or generic eager sampling: slower than the selected split sampler;
- unconditional stale-input reuse: unsafe across scheduler/cache/page-table
  changes;
- profiling during live serving: prohibited for this stage and unnecessary
  given benchmark JSON plus focused contract evidence.

## Device record

Initial `tt-smi -ls --local` listed four P150b boards. The first bounded `1x4`
mesh smoke failed before model execution when device 0 Ethernet core 31-25
could not resume and its heartbeat was static. Only the failed smoke PIDs
472292/472293 were killed. A single `timeout 180 tt-smi -r` reset all four
boards; the subsequent list and bounded `1x4` mesh open/close smoke passed.
After the boundary fix, the first final-run attempt hit the same pre-model
device-0 core 31-25 heartbeat failure in EngineCore PID 478079 (server PID
477931). The runner terminated it; a second bounded list/reset/list recovery
completed. An initial smoke command then failed only because its Python path
omitted `tools` and could not import `tracy`; the corrected full-environment
`1x4` mesh smoke passed. The resumed complete runner then passed and terminated
cleanly. After the final runner all four boards listed healthy. No live vLLM, API server,
runner, or EngineCore remained; only historical PID-1 zombie records existed.
Firmware 19.9 emitted the known warning relative to tested bundle 19.5.
The final fresh server log contains zero unsafe allocator warnings, tracebacks,
or error matches. It does emit nanobind reference-leak diagnostics at
interpreter shutdown. All requests and gates passed, the device mesh closed,
and process/device-holder audits were clean; the shutdown diagnostic did not
select a fallback or leave live runtime state.

## Validation commands and results

Adapter and stale-input contracts:

```bash
MPLCONFIGDIR=/tmp/mplconfig \
PYTHONPATH=$PWD/vllm:$PWD:$PWD/ttnn:$PWD/tools \
LD_LIBRARY_PATH=$PWD/build/lib:/opt/openmpi-v5.0.7-ulfm/lib \
python -m pytest -q \
  models/autoports/google_gemma_4_31b/tests/test_vllm_adapter_contract.py \
  models/autoports/google_gemma_4_31b/tests/test_full_model_contract.py \
  --junitxml=models/autoports/google_gemma_4_31b/doc/optimized_vllm/evidence/adapter_contract.xml
```

Result: 49 passed across the adapter and full-model contract files. This
includes stale changed token/current-position,
changed/unchanged page tables, reset, dynamic batch, cache ownership, no host
feedback, explicit plugin handshake coverage, exact capture ordering, and
pre-trace page-table copy prewarm coverage.

Plugin scheduler/lane contracts:

```bash
MPLCONFIGDIR=/tmp/mplconfig \
PYTHONPATH=$PWD/vllm:$PWD/vllm/plugins/vllm-tt-plugin/src:$PWD:$PWD/ttnn:$PWD/tools \
LD_LIBRARY_PATH=$PWD/build/lib:/opt/openmpi-v5.0.7-ulfm/lib \
python -m pytest -q \
  vllm/plugins/vllm-tt-plugin/tests/test_lane_model_runner.py \
  vllm/plugins/vllm-tt-plugin/tests/test_lane_scheduler.py \
  --junitxml=models/autoports/google_gemma_4_31b/doc/optimized_vllm/evidence/plugin_lane_contract.xml
```

Result: 22 passed, including unchanged versus newly allocated page-table
scheduler steps and the corresponding async reuse/overlap decision.

The complete real-serving runner command is in `README.md` and was identical to
the Stage 09 baseline workload/config. It exited 0 after:

- 149-token non-aligned completion pass;
- exact repeated/cross-position top-20 logit pass;
- 113279 input + 1 output context-ceiling pass;
- 72 passed, 1 skipped full sampling suite;
- six greedy plus six sampled raw qualitative outputs;
- primary single-user benchmark and CI burst benchmark;
- clean server/EngineCore shutdown.

The scoped degeneracy checker command was:

```bash
MPLCONFIGDIR=/tmp/mplconfig TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
PYTHONPATH=$PWD/vllm:$PWD:$PWD/ttnn:$PWD/tools \
LD_LIBRARY_PATH=$PWD/build/lib:/opt/openmpi-v5.0.7-ulfm/lib \
python models/common/readiness_check/check_degenerate_output.py \
  --hf-model /home/odjuricic/.cache/huggingface/hub/models--google--gemma-4-31B/snapshots/d77cb0be8ad40327cc1c6b70eff4b3f0be35bee3 \
  --missing-artifacts critical --scope vllm
```

Result: zero findings, exit 0.

## Final metrics

Primary, 127 actual input (128 requested), 128 output, one request,
concurrency 1, temperature 0, ignore EOS:

- before: TTFT P50/P99 992.586/992.586 ms; TPOT mean/P99
  38.023/38.023 ms; ITL P50/P99 29.348/29.739 ms; 21.974 output
  tok/s; 26.300 TPOT-derived t/s/u;
- after: TTFT P50/P99 494.656/494.656 ms; TPOT mean/P99
  37.531/37.531 ms; ITL P50/P99 29.330/32.840 ms; 24.328 output
  tok/s; 26.645 TPOT-derived t/s/u.

Secondary CI burst, 99 actual input/request (100 requested), 100
output/request, 32 requests, burst concurrency up to 32, temperature 0,
ignore EOS:

- before: TTFT P50/P99 8485.248/8488.457 ms; TPOT mean/P99
  77.373/127.442 ms; ITL P50/P99 55.807/687.715 ms; 201.070 output
  tok/s; 12.924 burst-derived t/s/u;
- after: TTFT P50/P99 7956.575/7960.168 ms; TPOT mean/P99
  75.800/125.780 ms; ITL P50/P99 55.831/631.891 ms; 210.102 output
  tok/s; 13.193 burst-derived t/s/u.

CI burst is secondary capacity evidence, never the headline rate.

## Optimize checklist and artifacts

- [x] Same-harness baseline and final primary measurements.
- [x] Same-harness baseline and final CI burst measurements.
- [x] Operation-topology and adapter-boundary audit.
- [x] Dedicated split sampler preserved; no force-argmax/generic regression.
- [x] Persistent inputs, nonblocking traces, async boundary, and stale-state
  tests.
- [x] No-host-fallback and external-cache audits.
- [x] Non-aligned and maximum-context gates.
- [x] Sampling and qualitative gates.
- [x] Cleanup and post-run device audit.
- [x] `perf_summary.json`, `runtime_fallback_audit.md`, before/after JSON, and
  JUnit reports.
- [x] No Tracy, tt-perf-report, live-server device profiler, adapter profiler,
  or ReadDeviceProfiler collection.

Independent `$stage-review` verdict: `clean-pass`, with no required work. The
report is `stage_review.md`. Stage-owned implementation/evidence commit SHAs
are recorded below; no push was performed.

## Local commits

- Nested vLLM plugin: `44b7853d448f3f8c5db7ed068a4f82ebfcd1065d`
- Main tt-metal implementation and Stage 10 evidence: recorded by the local
  checkpoint commit containing this file.
