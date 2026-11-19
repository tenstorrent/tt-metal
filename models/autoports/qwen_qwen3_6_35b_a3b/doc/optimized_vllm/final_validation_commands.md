# Serving validation commands and execution records

These commands describe the campaign against this checkout and installed bringup runtime 0.1.24. Actual execution records under `readiness_vllm/optimized_acceptance_*` are authoritative: core serving validation and the final eval campaign are complete. The measured speed shortfalls remain explicitly owner-accepted limitations; see the dated work log and round-3 review. Do not rerun completed gates after the user cap. Production remains frozen. Use one device owner at a time. No profiler, Tracy, adapter profiling, or broad process termination is involved.

Run shell blocks from the checkout. The server runs in an owner terminal; HTTP clients run in another terminal with the same setup. Required substitutions are `FINAL_TAG`, the current passing `CURRENT_STANDALONE` JSON, each owned readiness runner PID at shutdown, and the documented **32-user per-user** floor bound for P4. Never substitute historical standalone outputs for the current candidate.

```bash
set -euo pipefail
source python_env/bin/activate
source ../lane-env.sh
export VLLM_TT_PLUGIN_ROOT="$REPO/../vllm-tt-plugin"
export MODEL_DIR=models/autoports/qwen_qwen3_6_35b_a3b
export D="$MODEL_DIR/doc/optimized_vllm"
export R="$MODEL_DIR/readiness_vllm"
export FINAL_TAG=REPLACE_WITH_UNIQUE_FINAL_RUN_TAG
export FINAL_DIR="$R/$FINAL_TAG"
export URL=http://localhost:8000
export CURRENT_STANDALONE=REPLACE_WITH_CURRENT_SOURCE_VALIDATED_STANDALONE_JSON
export MOD=models.autoports.qwen_qwen3_6_35b_a3b.tests
unset TT_METAL_DEVICE_PROFILER TT_METAL_ENABLE_PROFILER TT_METAL_WATCHER
# Once, in the owner terminal only; refuse to overwrite an earlier campaign.
test "$FINAL_TAG" != REPLACE_WITH_UNIQUE_FINAL_RUN_TAG
test "$CURRENT_STANDALONE" != REPLACE_WITH_CURRENT_SOURCE_VALIDATED_STANDALONE_JSON
test ! -e "$FINAL_DIR"
mkdir -p "$FINAL_DIR"
git rev-parse HEAD > "$FINAL_DIR/tt_metal_head.txt"
git diff --binary > "$FINAL_DIR/tt_metal_worktree.patch"
git -C "$VLLM_TT_PLUGIN_ROOT" rev-parse HEAD > "$FINAL_DIR/plugin_head.txt"
cp "$MODEL_DIR/doc/context_contract.json" "$FINAL_DIR/context_contract.json"
```

`run_vllm_server.sh` fixes the real TT plugin adapter `tt/generator_vllm.py`, mesh P300x2, max sequences 32 (argument below), `max_model_len=262144`, block size 32, trace region 805306368, FABRIC_1D, decode-only tracing, pinned HF/tokenizer revision, and checkpoint-default thinking. It does not shorten the context contract. Do not append a server-wide thinking-off override. The context lengths below are probe workloads, not reductions of the supported context.

## Same-config primary and capacity measurements

Owner terminal (fresh performance server):

```bash
export SERVER_LOG="$FINAL_DIR/perf_server_2048/server.log"
QWEN_VLLM_ASYNC=1 QWEN_VLLM_AUDIT=0 \
TT_METAL_TRACE_ALLOC_TRACKING=0 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
PYTEST_ADDOPTS='-p vllm' \
bash "$MODEL_DIR/tests/run_vllm_server.sh" "$FINAL_DIR/perf_server_2048" 32 2048 --stages serve
```

After readiness, client terminal:

```bash
export SERVER_LOG="$FINAL_DIR/perf_server_2048/server.log"
timeout 3700 python "$MODEL_DIR/doc/vllm_integration/run_final64_perf.py" \
  --server-log "$SERVER_LOG" --server-url "$URL" \
  --output-dir "$FINAL_DIR/perf_2048" --execute
```

The collector output directory must not already exist. This is the existing collector, retaining baseline commands/workloads and audit checks. It invokes `run_vllm_server` benchmark/prefill-burst against the existing server. Report primary warmed **128 input / 128 output / 1 request / concurrency 1 / temperature 0 / ignore EOS** TTFT, TPOT, ITL, output throughput, and `1000 / TPOT_ms` t/s/u against the same-config baseline. CI **100 input / 100 output / 32 requests / default burst concurrency** is capacity evidence. The prefill burst is **128 input / 32 output / 32 requests / concurrency 32**. Retain its burst ratio. Do not relabel CI aggregate throughput as single-user t/s/u. The collector also retains its matched B2 supplemental comparison; it does not replace primary serving evidence or the like-work full-model traced comparison. Preserve `optimized_before` and all collector baseline inputs.

## Owned cleanup between every server configuration

Find the exact Python `readiness_check.run_vllm_server` PID for the current `SERVER_LOG` output directory, not the shell/timeout/EngineCore PID. Substitute it below. The cleanup helper verifies ownership and idle state, signals that runner, and records remaining owned processes/device holders. Inspect its result before starting another owner.

```bash
python "$D/stop_idle_server.py" \
  --runner-pid REPLACE_WITH_OWNED_READINESS_RUNNER_PID \
  --server-log "$SERVER_LOG" --server-url "$URL" --idle-timeout 180
```

## Tracked mixed-shape and current standalone comparison

After cleaning up the performance server, owner terminal:

```bash
export SERVER_LOG="$FINAL_DIR/tracked_async_2048/server.log"
QWEN_VLLM_ASYNC=1 QWEN_VLLM_AUDIT=1 \
TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
PYTEST_ADDOPTS='-p vllm' \
bash "$MODEL_DIR/tests/run_vllm_server.sh" "$FINAL_DIR/tracked_async_2048" 32 2048 --stages serve
```

Client terminal, sequentially after readiness:

```bash
export SERVER_LOG="$FINAL_DIR/tracked_async_2048/server.log"
python "$D/run_probe.py" --name lifecycle --server-log "$SERVER_LOG" \
  --server-url "$URL" --timeout 7200 --output-dir "$FINAL_DIR/records" -- \
  python -m "$MOD.run_vllm_lifecycle" --url "$URL" --four-row-bucket \
    --long-prompt-length 4101 --sampled-top-k 40 --output-tokens 96 \
    --output "$FINAL_DIR/lifecycle_requests.json"
python -m "$MOD.summarize_vllm_lifecycle" \
  --requests "$FINAL_DIR/lifecycle_requests.json" --server-log "$SERVER_LOG" \
  --output "$FINAL_DIR/lifecycle_summary.json"
test -f "$CURRENT_STANDALONE"
python "$D/run_probe.py" --name http_logits --server-log "$SERVER_LOG" \
  --server-url "$URL" --timeout 7200 --output-dir "$FINAL_DIR/records" -- \
  python -m "$MOD.run_vllm_logit_determinism" --mode http \
    --baseline "$CURRENT_STANDALONE" --url "$URL" --server-log "$SERVER_LOG" \
    --four-row-bucket --output "$FINAL_DIR/http_logits.json"
python "$D/run_probe.py" --name growth_async --server-log "$SERVER_LOG" \
  --server-url "$URL" --timeout 7200 --output-dir "$FINAL_DIR/records" -- \
  python -m "$MOD.run_vllm_growth_control" --url "$URL" \
    --output "$FINAL_DIR/growth_async.json" --async-scheduling on \
    --decode-buckets 2 4 32 --max-num-batched-tokens 2048
```

Require lifecycle summary success, retained trace identities across B2/B4/B32, no traffic recapture/release, real supported chunk offsets for the non-aligned 4101-token prompt, greedy/native sampled-path reuse, and mixed-traffic latency evidence. Current HTTP logit/token comparisons must pass against the current full40 standalone result (including the saved independent control comparison). HTTP logprobs use a diagnostic host-sampling path; they are not device-sampling or speed evidence.

The growth fixture uses prompts 129/130/131 and output caps 160/112/80 in two admission orders. Review request records **and** runtime audit for actual page growth, changed/unchanged page-table handling, persistent token/position/sampler inputs, and deferred async read behavior; matching HTTP tokens alone cannot prove these contracts.

Clean up the tracked async server, then launch this otherwise identical nonasync server:

```bash
export SERVER_LOG="$FINAL_DIR/tracked_sync_2048/server.log"
QWEN_VLLM_ASYNC=0 QWEN_VLLM_AUDIT=1 \
TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
PYTEST_ADDOPTS='-p vllm' \
bash "$MODEL_DIR/tests/run_vllm_server.sh" "$FINAL_DIR/tracked_sync_2048" 32 2048 --stages serve
```

Client terminal:

```bash
export SERVER_LOG="$FINAL_DIR/tracked_sync_2048/server.log"
python "$D/run_probe.py" --name growth_sync --server-log "$SERVER_LOG" \
  --server-url "$URL" --timeout 7200 --output-dir "$FINAL_DIR/records" -- \
  python -m "$MOD.run_vllm_growth_control" --url "$URL" \
    --output "$FINAL_DIR/growth_sync.json" --async-scheduling off \
    --decode-buckets 2 4 32 --max-num-batched-tokens 2048 \
    --compare "$FINAL_DIR/growth_async.json"
```

## Current native qualitative and recall controls

Run only after verified idle-server cleanup and with no other TT owner. Keep the completed tracked server log for the actual cache geometry. Use a fresh output path; the harness refuses overwrites.

```bash
env -u TT_METAL_DEVICE_PROFILER -u TT_METAL_ENABLE_PROFILER -u TT_METAL_WATCHER \
  TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
  QWEN_VLLM_AUDIT=0 timeout 7200 python -m "$MOD.run_vllm_quality_controls" \
    --backend tt --thinking default --served-geometry "$SERVER_LOG" \
    --output "$FINAL_DIR/native/tt_outputs.json" \
    --recall-fixtures "$MODEL_DIR/readiness_vllm/optimized_acceptance_recall/recall_fourteen_fixtures.json" \
    --recall-max-tokens 24 --recall-thinking off
```

Require twelve shared outputs, fourteen unique completed recall answers, six trace releases, and no remaining device holder. Read all outputs. Native generation uses one public whole-prompt prefill call and solo traced decode; scheduler chunking and async ownership are established separately by the real serving checks. Shared quality is a bounded 256-token reasoning smoke, not completed-answer accuracy or a broad quality waiver.

## Full sampling and checkpoint-default qualitative outputs

Clean up nonasync, then restart the **same tracked async 2048 configuration** into fresh `$FINAL_DIR/sampling_async_2048` (substitute this directory in the tracked async launch). Set `SERVER_LOG` accordingly in both terminals.

```bash
export SERVER_LOG="$FINAL_DIR/sampling_async_2048/server.log"
QWEN_CHAT_TEST_REQUEST_LOG="$FINAL_DIR/explicit_off_requests.jsonl" \
PYTEST_ADDOPTS="-p vllm -p $MOD.checkpoint_default_test_requests --tt-chunked-prefill-budget=2048 --tt-recall-reasoning-effort=low --tt-recall-reasoning-token-budget=0 --junitxml=$FINAL_DIR/sampling_full.xml" \
python -m readiness_check.run_vllm_server \
  --stages sampling --sampling-profile full --model-dir "$MODEL_DIR" \
  --hf-model "$HF_MODEL" --mesh-device P300x2 --max-num-seqs 32 \
  --max-model-len 262144 --block-size 32 \
  --tt-config '{"trace_region_size":805306368,"fabric_config":"FABRIC_1D","trace_mode":"decode_only"}' \
  --server-url "$URL" --server-log "$SERVER_LOG" --output-dir "$FINAL_DIR/sampling_full"
python "$D/run_probe.py" --name qualitative --server-log "$SERVER_LOG" \
  --server-url "$URL" --timeout 7200 --output-dir "$FINAL_DIR/records" -- \
  python -m "$MOD.run_vllm_qualitative" --url "$URL" --thinking default \
    --output-dir "$FINAL_DIR/qualitative_default"
python -m "$MOD.check_vllm_checkpoint_default" --url "$URL" \
  --output "$FINAL_DIR/checkpoint_default.json"
python "$TT_MODEL_BRINGUP_ROOT/runtime/readiness_check/check_degenerate_output.py" \
  "$FINAL_DIR/qualitative_default" --scope vllm --missing-artifacts critical \
  --json "$FINAL_DIR/qualitative_degenerate.json"
```

This loads the entire full sampling profile, without test selection or reduction. The chunk budget option enables the plugin's supported chunk-prefill tests. The existing model-owned request fixture sets thinking off **only** for `test_bad_words`, `test_dp1_full_capacity_mixes_structured_and_plain_requests`, `test_dp1_reuses_slots_with_mixed_requests`, and the three canonical `test_chunked_prefill.py` recall tests; it preserves assertions, caps, seeds and concurrency and logs actual requests/responses. Label this as full-suite evidence with six explicit request-mode overrides, not unchanged-default execution of those six tests. The recall tests use the existing chat option and zero extra reasoning-token budget; their original 24-token cap is unchanged. All other tests retain checkpoint behavior. Qualitative runs six shared prompts in greedy and sampled modes with checkpoint-default thinking; retain reasoning/content and review all 12 outputs against matching HF/native controls. A mechanical degeneracy pass is not a substitute for review. Do not invoke the old `run_final64_gates` plan, which forces qualitative thinking off.

## Eval budget sweep, then P5 timing and P4 long answers

The installed probes write raw artifacts under `doc/eval_readiness`, including when `--output` is provided. Before new measurements, preserve the four old timing/load paths by moving them into this campaign. Keep other existing eval contracts/findings intact. Do this once, with no probe running:

```bash
python - <<'PY'
import os
import shutil
from pathlib import Path
model = Path(os.environ['MODEL_DIR']) / 'doc/eval_readiness'
archive = Path(os.environ['FINAL_DIR']) / 'previous_eval_readiness'
archive.mkdir(parents=True, exist_ok=False)
for name in ('long_context_timing', 'long_context_timing.json',
             'long_answer_load', 'long_answer_load.json'):
    source = model / name
    if source.exists():
        shutil.move(str(source), str(archive / name))
PY
```

Run each budget **512, 2048, 4096** sequentially. The separate tracked lifecycle/growth gates establish allocation safety; do not charge their per-replay garbage-collection instrumentation to production ITL/forecast. For each, clean up the previous server, set `BUDGET` to the next value, then launch a fresh async owner with allocation tracking disabled for latency measurements and audit counters enabled:

```bash
export BUDGET=512
export SERVER_LOG="$FINAL_DIR/eval_server_${BUDGET}/server.log"
QWEN_VLLM_ASYNC=1 QWEN_VLLM_AUDIT=1 \
TT_METAL_TRACE_ALLOC_TRACKING=0 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
PYTEST_ADDOPTS='-p vllm' \
bash "$MODEL_DIR/tests/run_vllm_server.sh" "$FINAL_DIR/eval_server_${BUDGET}" \
  32 "$BUDGET" --stages serve
```

Client terminal (set matching `BUDGET` and `SERVER_LOG`):

```bash
python "$D/run_probe.py" --name "chunk_${BUDGET}" --server-log "$SERVER_LOG" \
  --server-url "$URL" --timeout 10800 --output-dir "$FINAL_DIR/records" -- \
  python -m readiness_check.run_long_context --part chunk --server-url "$URL" \
    --model-dir "$MODEL_DIR" --server-log "$SERVER_LOG" \
    --max-num-batched-tokens "$BUDGET" --itl-bound-ms 1000 \
    --chunk-prompts 4096,32768 --decode-users 31 --decoder-max-tokens 16384 --decode-warmup-s 180
cp "$MODEL_DIR/doc/eval_readiness/long_context_timing/chunk_${BUDGET}.json" \
  "$FINAL_DIR/chunk_${BUDGET}.json"
```

After all three complete, choose the largest value with no request errors and worst mixed-traffic ITL at most 1000 ms. An empty eligible set is work, not permission to silently select a failing budget. This CPU-only selection uses the same installed helper as P5:

```bash
PICKED_BUDGET=$(python - <<'PY'
import os
from pathlib import Path
from readiness_check.run_long_context import chunk_sweep_rows, pick_chunk
rows = chunk_sweep_rows(Path(os.environ['MODEL_DIR']) / 'doc/eval_readiness/long_context_timing')
assert {r['max_num_batched_tokens'] for r in rows} == {512, 2048, 4096}, rows
picked = pick_chunk(rows, 1000)
assert picked is not None, rows
print(picked)
PY
)
export PICKED_BUDGET
```

Serve `PICKED_BUDGET` for the remaining eval. If the last server already has that budget it can remain; otherwise clean up and launch the eval async pattern (audit on, allocation tracking off) into fresh `$FINAL_DIR/eval_selected_${PICKED_BUDGET}` with that budget. Set `SERVER_LOG` to the actual selected server. Do not silently change the 2048 performance comparison to this selected budget.

```bash
python "$D/run_probe.py" --name p5_timing --server-log "$SERVER_LOG" \
  --server-url "$URL" --timeout 18000 --output-dir "$FINAL_DIR/records" -- \
  python -m readiness_check.run_long_context --part timing --server-url "$URL" \
    --model-dir "$MODEL_DIR" --server-log "$SERVER_LOG" \
    --max-num-batched-tokens "$PICKED_BUDGET" --itl-bound-ms 1000 \
    --lengths 1024,8192,32768,131072 --decode-tokens 32 --ttft-reps 3 --request-timeout 10800
cp "$MODEL_DIR/doc/eval_readiness/long_context_timing.json" "$FINAL_DIR/p5_timing.json"
```

P5 includes warmed 127-versus-128 prompt checks as well as all specified context lengths. Preserve non-aligned serving. P4 needs its own batch32 useful-work basis. `eval_batch32_necessary_floor.json` now supplies a conservative necessary bound4.764924390924285t/s/u, derived from the maximum possible expert union and upper first-bucket cache position. `eval_prompt_tokenization.json` verifies all32served prompt IDs, maximum74tokens. This does not claim measured active-expert coverage: falling below the bound proves failure; passing it would leave the actual-work floor unvalidated. Check the final load prompt counts remain at most74 before applying it. The B1 floor and CI aggregate throughput are not substitutes.

```bash
export BATCH32_NECESSARY_FLOOR_TPS=4.764924390924285
python - <<'PY'
import os
assert float(os.environ['BATCH32_NECESSARY_FLOOR_TPS']) > 0
PY
python "$D/run_probe.py" --name p4_long_answer --server-log "$SERVER_LOG" \
  --server-url "$URL" --timeout 14400 --output-dir "$FINAL_DIR/records" -- \
  python -m readiness_check.run_long_answer_load --server-url "$URL" \
    --model-dir "$MODEL_DIR" --server-log "$SERVER_LOG" \
    --hf-model "$HF_MODEL" --revision "$HF_REVISION" \
    --profile lm_eval_reasoning --users 32 --max-tokens 6000 \
    --functional-floor-tps "$BATCH32_NECESSARY_FLOOR_TPS" \
    --eval-plan "$TT_MODEL_BRINGUP_ROOT/runtime/readiness_check/eval_plan_defaults.json" \
    --trace-capture-regex 'QWEN_TRACE_EVENT.*"event"\s*:\s*"capture"'
cp "$MODEL_DIR/doc/eval_readiness/long_answer_load.json" "$FINAL_DIR/p4_long_answer.json"
python "$TT_MODEL_BRINGUP_ROOT/scripts/check_eval_readiness.py" \
  --model-dir "$MODEL_DIR" --stage optimized-vllm
```

P4 retains the installed eval profile (thinking on, temperature 1, top-p 0.95, top-k 40, seed 42), 32 users, 6000 generated tokens/user, 1000-token measurement buckets, its 256-token plain-versus-eval sampling comparison, default eval forecast plan, and default ignore-EOS behavior. Do not add `--no-compare`, `--stop-at-eos`, or shorter token caps. The custom regex matches this generator's actual capture log lines, including captures before the load; zero matches in the entire log is not proof of no recapture. Review per-bucket throughput/ITL, cache return to idle, preemptions, errors, capture count, host-sampling cost, and forecast, not only process exit status.

Finally run the owned cleanup command and retain its no-leftover-process/device-holder evidence. Reconcile all failures and advisory findings into work and, where required by the skill, `doc/eval_readiness/limitations.json`; a limitations entry does not automatically satisfy the user's acceptance gates. Update README/work log with fresh paths, workload shapes, same-config before/after primary and secondary capacity metrics, full-model comparison, and forecast. Stage review and local stage-owned commits remain separate required steps; do not push.
