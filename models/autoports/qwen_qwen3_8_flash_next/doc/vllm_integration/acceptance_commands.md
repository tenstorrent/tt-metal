# Stage 9 acceptance commands

Prepared by CPU-only source inspection on 2026-11-16. These commands are a run
plan, not passing evidence. Run device work serially. The approved prefill-mask
interface, reduced-target checks, and async correctness must pass before this
sequence. No command changes checkpoint generation defaults.

## Common client environment

```bash
cd /home/ttuser/qwen38-flash/tt-metal
source /home/ttuser/qwen38-flash/vllm-env/bin/activate
source /home/ttuser/qwen38-flash/lane-env.sh
export TT_VISIBLE_DEVICES=0,1,2,3
export TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/p300_x2_mesh_graph_descriptor.textproto
export VLLM_TT_PLUGIN_ROOT=/home/ttuser/qwen38-flash/vllm-tt-plugin
export PYTHONPATH="$TT_MODEL_BRINGUP_ROOT/runtime:$PYTHONPATH"
export M=models/autoports/qwen_qwen3_8_flash_next
export CKPT=/home/ttuser/qwen38-flash/hf/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/812d96e10a0dd82b69d749e68693698cf6c36daa
export SERVER=http://localhost:8000
```

Use C262144/B2 throughout. B32 physical impossibility and B2 measured capacity
are in `capacity_full_v1.json`; do not label the 32-request client bursts as
32 simultaneously resident sequences. Their client workload remains unchanged.

## Controls before occupying the devices with a server

The prepared control script requires the original `python_env` (HF's qwen4_exp
implementation); the serving environment has a different transformers version.
The TT command below owns all four devices and cannot overlap a server.

```bash
source python_env/bin/activate
source /home/ttuser/qwen38-flash/lane-env.sh
export TT_VISIBLE_DEVICES=0,1,2,3
export TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/p300_x2_mesh_graph_descriptor.textproto
timeout 14400 python "$M/tests/run_vllm_quality_controls.py" --backend hf \
  --max-new-tokens 256 --output-dir "$M/readiness_vllm/quality_hf_v1" \
  > "$M/doc/vllm_integration/logs/quality_hf_v1.log" 2>&1
timeout 7200 python "$M/tests/run_vllm_quality_controls.py" --backend tt \
  --max-new-tokens 256 --output-dir "$M/readiness_vllm/quality_tt_v1" \
  > "$M/doc/vllm_integration/logs/quality_tt_v1.log" 2>&1
```

Both controls produce `manifest.json`, `prompts.json`, `outputs.json`, and 12
per-prompt/profile JSON files. The suite is six shared prompts, greedy and
sampled, seed42, thinking off, 256 new tokens. A quality anomaly needs a further
10-seed blind matched comparison before declaring a checkpoint limitation.

## Tracked acceptance server

After confirming device holders are empty, run the existing launcher in its
own managed execution session (its timeout is 86400 seconds):

```bash
TRACKING=1 bash "$M/doc/vllm_integration/run_server.sh" "$M/readiness_vllm/acceptance_tracking"
```

Use the common client environment above in a second shell. Set
`LOG="$M/readiness_vllm/acceptance_tracking/server.log"`. Reuse this server for
the final mixed-shape/lifecycle script and the following probes, without
resetting generator counters. The mixed check must retain baseline captures,
actual starts/active rows, sampling transitions, correctness, and release counts.

```bash
timeout 7200 python -m readiness_check.run_harness_requests \
  --server-url "$SERVER" --model-dir "$M" --hf-model "$CKPT" --server-log "$LOG" \
  --users 1,2 --load-users 2 --output-tokens 128 --timeout 1800 \
  > "$M/doc/vllm_integration/logs/P2_final.log" 2>&1
timeout 10800 python -m readiness_check.run_chat_conformance \
  --server-url "$SERVER" --model-dir "$M" --hf-model "$CKPT" --server-log "$LOG" \
  --thinking-on-kwargs '{"enable_thinking":true,"reasoning_effort":"low"}' \
  --thinking-off-kwargs '{"enable_thinking":false}' --timeout 1800 \
  > "$M/doc/vllm_integration/logs/P3_final.log" 2>&1
timeout 43200 python "$M/tests/run_vllm_long_context_recorded.py" \
  --record-dir "$M/readiness_vllm/P5_accuracy_raw" -- --part accuracy \
  --server-url "$SERVER" --model-dir "$M" --hf-model "$CKPT" --server-log "$LOG" \
  --lengths 8192,32768,131072 --depths 0.1,0.5,0.9 --max-tokens 512 \
  --no-thinking --request-timeout 14400 \
  > "$M/doc/vllm_integration/logs/P5_accuracy_final.log" 2>&1
```

The model-local P5 evidence wrapper delegates the unchanged shared CLI and scoring, retaining complete streamed responses and incremental progress under `readiness_vllm/P5_accuracy_raw/`. It sends no extra requests and never retries. Use a fresh evidence directory only when a justified new run is needed. Estimated P5 runtime5.5–8.2h and maximum-context2.8–4.1h require sufficient remaining server lifetime; per-request14400s/outer43200s are intentional. See `long_context_preparation_audit.md`.

Outputs go to `doc/eval_readiness/harness_requests.json`,
`chat_conformance.json`, `long_context_accuracy.json` and corresponding raw
subdirectories. P5's nine requests do not reach advertised maximum context.

Run one exact maximum-context acceptance request separately. This uses a valid
non-aligned 262136-token prompt plus eight forced output tokens, reaches total
262144, and preserves the full response. It is a continuation stress test, not
a text-quality verdict. Do not repeat this after a successful acceptance unless
an affected implementation change invalidates it.

```bash
timeout 21600 python "$M/tests/check_vllm_max_context.py" \
  --server-url "$SERVER" --model-dir "$M" --checkpoint "$CKPT" \
  --output "$M/readiness_vllm/max_context_acceptance.json" \
  > "$M/doc/vllm_integration/logs/max_context_acceptance.log" 2>&1
```

The helper reads `supported_context` from `doc/context_contract.json`, verifies
served `max_model_len`, and reserves
`readiness_vllm/max_context_acceptance.attempt.json` before HTTP. It refuses an
existing attempt marker or output, even when a different output filename is
selected. The request timeout is21000 seconds. It preserves complete responses
and failed evidence, validates exact usage/eight token IDs against checkpoint
vocabulary, and never retries automatically.

If this demonstrates a hard physical limit, preserve it and use only one
bracketing pass/fail pair; a model/cache error is repair work, not permission to
reduce context. Client timeout alone is not evidence of a physical capacity limit.

## Sampling, quality, and untracked performance

Once tracked correctness is complete and the server has no running/waiting
requests, stop its owning runner gracefully and confirm device release. Restart
with tracking disabled at the gate's root, preserving tracked evidence:

```bash
TRACKING=0 bash "$M/doc/vllm_integration/run_server.sh" "$M/readiness_vllm"
```

In the client shell set `LOG="$M/readiness_vllm/server.log"`. Preserve the
launcher's `vllm_run_summary.json` while it is live; attached invocations overwrite
that filename, and launch shutdown also writes it. Copy each completed summary
to a descriptive sibling before the next invocation. The actual server command
remains in `run_server.sh`, launch output, and launcher summary.

```bash
timeout 21600 python -m readiness_check.run_vllm_server \
  --stages sampling --sampling-profile full --max-num-seqs 2 --max-model-len 262144 \
  --server-url "$SERVER" --server-log "$LOG" --model-dir "$M" --hf-model "$CKPT" \
  > "$M/doc/vllm_integration/logs/sampling_full_runner.log" 2>&1
timeout 7200 python -m readiness_check.run_vllm_server \
  --stages benchmark,prefill_burst --max-num-seqs 2 --max-model-len 262144 \
  --server-url "$SERVER" --server-log "$LOG" --model-dir "$M" --hf-model "$CKPT" \
  > "$M/doc/vllm_integration/logs/benchmark_runner.log" 2>&1
```

The benchmark defaults are primary128/128/1 at concurrency1, greedy;
CI100/100/32 without a client concurrency cap, greedy; and
prefill-burst32x128 at client concurrency32, 32 output tokens. Do not shrink
these client workloads because B2 queues admission. Files are
`sampling_tests.log`, `vllm_result.json`, `vllm_benchmark.json`,
`vllm_benchmark.log`, `vllm_ci_serving_result.json`,
`vllm_ci_serving_benchmark.json`, `vllm_ci_serving_benchmark.log`,
`vllm_prefill_burst_result.json`, `vllm_prefill_burst_benchmark.json`, and
`vllm_prefill_burst_benchmark.log`. With batched prefill disabled, the burst is
the serial baseline; do not invent an enabled arm or a speedup ratio. A later
enabled arm needs equality/watcher evidence and uses
`--prefill-burst-serial-result <preserved-serial-summary>`.

The shared qualitative runner currently hardcodes256 tokens, sends no
`chat_template_kwargs`, and errors if `message.content is None`. Its recipe is
the checkpoint's thinking-on/xhigh default, whereas the prepared matched
controls are thinking-off. There is no runner CLI for request kwargs. **Do not
change served defaults to match the controls.** Use a targeted shared-suite
client with explicit request-level `chat_template_kwargs={"enable_thinking":false}`,
seed42, greedy `{temperature:0,top_p:1,top_k:1}` and sampled
`{temperature:.7,top_p:.9,top_k:20}`, max_tokens256. Reuse all six prompts and
record full API responses, rendered/tokenized prompt parity, and the same
`vllm_qualitative_outputs.json` fields (`prompt`, `prompt_mode`,
`greedy_completion`, `sampled_completion`) with a provenance sidecar. The
targeted client is `tests/check_vllm_qualitative.py`; the controls alone are not served
quality evidence. Run it in the original tokenizer environment:

```bash
source python_env/bin/activate
timeout 7200 python "$M/tests/check_vllm_qualitative.py" \
  --server-url "$SERVER" --controls-dir "$M/readiness_vllm/quality_hf_v1" \
  --output-dir "$M/readiness_vllm/quality_served_v1" \
  --publish-output "$M/readiness_vllm/vllm_qualitative_outputs.json" \
  > "$M/doc/vllm_integration/logs/quality_served_v1.log" 2>&1
```

The client preserves any existing shared-runner JSON as
`vllm_qualitative_outputs.before_targeted.json` before publishing and refuses
to overwrite that backup. It checks all six prompts against the HF control,
the served `/tokenize` result and the actual returned generation input IDs.
The skill permits a targeted prompt-correct request when the
packaged runner cannot express the needed format/settings. Its raw default
attempt can be kept separately, without treating truncated thinking as wrong
model output or a successful quality gate.

Prefix caching has the documented `recurrent_state` exclusion. Do not run a
false passing prefix-cache test: leave an explicit excluded/skipped artifact
under `readiness_vllm/prefix_cache_check.json` linked to the serving contract;
the check script accepts this exclusion and rejects a contradictory pass.

## Remaining evidence interpretation

- P2 top-k40 necessarily selects the explicit host compatibility path at the
  current device ceiling32. Its acceptance must pass; measure the critical
  2-user ratio rather than predicting it. The selected checkpoint top-k20
  default already fits the device sampler, so no default override is needed.
- P3's low-thinking request kwargs are an existing per-request capability,
  not a served-default change. Keep c01/c02, all tool/history cases, and their
  raw responses. Do not skip hard cases to achieve a pass.
- Scan full sampling and bursts for out-of-vocabulary token IDs and host/device
  forward routes. Unsupported-feature host sampling is optional compatibility;
  headline and CI greedy metrics must use canonical device token-out sampling.
- Read generated quality outputs for coherence, topic, repetition, gibberish,
  language drift, and contamination against the matching controls. Then run
  `python "$TT_MODEL_BRINGUP_ROOT/runtime/readiness_check/check_degenerate_output.py" --hf-model Qwen/Qwen3.8-Flash-Next --missing-artifacts critical --scope vllm`
  and
  `python "$TT_MODEL_BRINGUP_ROOT/scripts/check_eval_readiness.py" --model-dir "$M" --stage vllm`.
- Preserve P50/P99 TTFT/ITL, mean/P99 TPOT and output throughput from normalized
  and raw JSON. Headline t/s/u is1000/meanTPOT for128/128/1 only; CI TPOT includes
  B2 admission/queueing effects and is secondary.
- These are finite client timeouts, not permission to kill a busy engine.
  Follow device skill timeout/recovery rules and inspect running/waiting
  metrics before graceful final runner shutdown. Audit EngineCore/API processes
  and device holders, then finalize docs and request the independent review.

## Final sampling allowance experiment (2026-11-16)

The three focused chat nodes passed before the full rerun, using the existing
optional extra allowance1528. Completed native trajectories motivate this
conditional allowance; the native combined control remains8/9/failed. See
`hf_reasoning_budget_outcome.md` and `AUTOFIX_reasoning_budget.md`. The following
full command is running against `acceptance_tracking_v3` with no node selection:

```bash
export PYTHONPATH=/home/ttuser/qwen38-flash/tt-metal:$TT_MODEL_BRINGUP_ROOT/runtime:$PYTHONPATH
export PYTEST_ADDOPTS='--tt-reasoning-token-budget=1528 -p models.autoports.qwen_qwen3_8_flash_next.tests.record_sampling_chat --sampling-chat-record-dir=/home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/readiness_vllm/sampling_chat_full1528_v1'
timeout 21600 "$VIRTUAL_ENV/bin/python" -m readiness_check.run_vllm_server \
  --stages sampling --sampling-profile full --mesh-device P300x2 --block-size 32 \
  --max-num-seqs 2 --max-model-len 262144 \
  --tt-config '{"trace_region_size":0,"fabric_config":"FABRIC_1D_RING","trace_mode":"all"}' \
  --server-url http://localhost:8000 \
  --server-log "$M/readiness_vllm/acceptance_tracking_v3/server.log" \
  --model-dir "$M" --hf-model "$CKPT" \
  > "$M/doc/vllm_integration/logs/sampling_full_budget1528_v1.log" 2>&1
```

The observer records responses only; the shared tests and assertions remain
authoritative. A passing served run does not retroactively pass the native
control. Initial failures remain in `sampling_tests.full_initial_failure.log`.

Stage-cap instruction received during the full sampling run. Remaining commands above run once each; existing passing evidence is reused. No further optional optimization or native diagnostic run is planned.
