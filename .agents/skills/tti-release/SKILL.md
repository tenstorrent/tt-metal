---
name: tti-release
description: Run the tt-inference-server model release workflow for the completed generated autoport TTNN/vLLM model and produce the customer-facing readiness markdown report. Use after optimized-vLLM or at model readiness handoff time, especially when Codex must evaluate the generated autoport through an already-running OpenAI-compatible vLLM server, run small TTI smokes before expensive release workflows, or use Docker only as an explicit fallback.
---

# TTI Release

## Overview

This skill runs the Tenstorrent `tt-inference-server` release workflow for the generated `models/autoports/<model>` implementation whose vLLM serving path is already complete. The goal is a copied-back markdown release report plus a short local handoff note with exact commands, versions, recovery actions, and release-readiness status.

Separate release workflow success from release readiness. `run.py` exiting `0`, API requests completing, or a report being copied back proves the release harness ran. It does not prove the model is ready if the report contains failed accuracy, API conformance, or benchmark target rows.

The release stage is only valid when the release workflow evaluates the just-brought-up autoport model. Do not run a stock `tt-transformers`, `models/demos`, or other packaged implementation for the same Hugging Face model. That measures a different model and must be treated as a failed release-stage artifact, even if `run.py` exits `0`.

The release stage must evaluate the autoport at the context length recorded in `models/autoports/<model>/doc/context_contract.json`. Do not cap context, LongBench, benchmark prompt/completion lengths, or API limits to hide a model context bug. A reduced supported context is valid only when earlier stages recorded evidence that a hard physical device limit prevents the advertised context from fitting or running and proved the largest feasible value. Only adjust a request that is mathematically invalid because prompt plus completion exceeds the real supported context, and record that as a harness issue.

Treat a valid TTI prompt length as a logical request length. If TTI sends a prompt whose length is within the supported context and the autoport fails because the length is not divisible by an internal chunk, tile, block, page, or trace size, that is an autoport model bug. Do not waive it, lower the context, or change the benchmark to an aligned length. Fix the model/generator/adapter padding, chunking, masking, or output slicing path and rerun the failing item.

## Topology

Assume the agent usually starts inside an IRD reservation/Codex container:

```text
reservation container:
  has tt-smi and the experiment/model context
  has the generated tt-metal autoport checkout and optimized-vLLM evidence
  use it for device health, tt-smi reset, autoport vLLM serving, and local model evidence

physical loudbox host:
  has Docker
  use only if the chosen TTI path truly needs Docker
```

Prefer the client/server topology:

```text
1. Start the generated autoport vLLM server from the tt-metal reservation checkout.
2. Run TTI workflows as a client against that server's OpenAI-compatible port.
3. Do not pass --docker-server.
```

This deliberately keeps server-side vLLM, tokenizer, plugin, and autoport imports inside the already-working tt-metal environment. Do not try to fix server import/API mismatches by mixing the TTI Docker image's vLLM packages with the autoport server. TTI still needs a working benchmark client environment, but that client should only send OpenAI-compatible HTTP requests to the generated server.

Use Docker only when the external autoport server path is unavailable after investigation, or when the user explicitly asks for the packaged Docker path. If Docker is used, make sure the container evaluates the generated autoport, not a stock implementation.

## Preflight

1. Identify:

```text
HF model id
model autoport directory
physical host name, for example wh-lb-80, only if Docker or host-level recovery is needed
TTI device name, usually t3k for T3K loudboxes
experiment evidence directory, if separate from models/autoports/<model>/doc/
```

If Docker is needed and the prompt does not give the physical host, infer it from reservation metadata or the reservation container hostname only when obvious. Otherwise stop and ask for the host; a wrong host can use the wrong hardware.

The model autoport directory is the target implementation. Keep its exact relative path, for example `models/autoports/meta_llama_llama_3_1_8b_instruct`, and use that path in later spec checks.

2. Check devices in the reservation container:

```bash
tt-smi -ls --local
```

If Docker will be used, also check Docker on the physical host:

```bash
ssh "$PHYSICAL_HOST" 'docker ps >/dev/null && echo DOCKER_OK'
```

If `docker ps` fails on the physical host, do not continue with `--docker-server`. Use the external-server topology instead, or report `physical-host Docker unavailable` if Docker is truly required.

3. Confirm vLLM readiness artifacts exist before this phase:

```text
models/autoports/<model>/doc/optimized_vllm/README.md
models/autoports/<model>/doc/optimized_vllm/work_log.md
models/autoports/<model>/readiness_vllm/
```

If optimized-vLLM is blocked only by a recoverable ARC/reset error, recover and resume optimized-vLLM first. Do not jump to TTI release ahead of a recoverable earlier-stage hardware blocker.

4. Confirm that the TTI release plan can target the autoport implementation:

- The TTI model spec or runtime spec must point at the target `models/autoports/<model>` code path.
- The launched server must import or otherwise use the target autoport vLLM implementation, usually `models/autoports/<model>/tt/generator_vllm.py`.
- The serving max context must match `doc/context_contract.json`.
- A built-in TTI model name such as `Llama-3.1-8B-Instruct` is not enough. Built-in model names commonly select stock `models/tt_transformers` implementations.
- If the only available `tt-inference-server` path selects `models/tt_transformers`, `models/demos`, or another stock implementation, stop and fix the release integration. Do not benchmark the stock model as a substitute.

## Checkout And Version Selection

Clone `tt-inference-server` on the physical host under a per-model work directory:

```bash
WORK_ROOT=/localdev/$USER/tti-release/<run-name>
ssh "$PHYSICAL_HOST" "mkdir -p '$WORK_ROOT'"
ssh "$PHYSICAL_HOST" "cd '$WORK_ROOT' && test -d tt-inference-server || git clone https://github.com/tenstorrent/tt-inference-server.git"
```

The required repo tag is model-specific. Do not assume `main` or `v0.9.0`.

Use one of these evidence-backed methods:

- Run the command on the default checkout and follow the version mismatch error if `run.py` says to checkout a matching release tag.
- Or inspect `model_spec.json` / docs for the model's Docker image tag and checkout `v<version>` from the tag prefix, for example image tag `0.9.0-25305db-6e67d2d` implies repo tag `v0.9.0`.

After checkout, run `python3 run.py --help` and use that checkout's CLI flags. Older releases use `--device`; newer docs may show `--tt-device`.

## Autoport Model Selection

Find or create the TTI model spec that corresponds to the target autoport implementation. Prefer a temporary model spec JSON under the TTI work root and pass it through the checkout's supported `--model-spec-json` or equivalent flag. If the checkout requires editing its local model spec registry, make the smallest local edit and record it.

The selected spec must identify the generated code path:

```text
impl.code_path = models/autoports/<model>
hf_model_repo = HF model id
inference_engine = vLLM
device = target TTI device
```

The spec may reuse benchmark, eval, and API-test definitions from the matching stock model, but it must not reuse the stock implementation path.

For the external-server path, create or edit a temporary TTI spec whose embedded `cli_args` are already correct:

```text
cli_args.workflow = release, benchmarks, evals, tests, spec_tests, or reports
cli_args.docker_server = false
cli_args.local_server = false
cli_args.service_port = the running autoport server port, usually 8000
cli_args.model_spec_json = path to this temporary spec
```

Do not rely on command-line flags alone to override a custom `--model-spec-json`. Some TTI checkouts load the JSON's embedded `cli_args` and do not apply the current CLI args to the loaded spec. Before running a workflow, inspect the run spec that TTI writes under `workflow_logs/run_specs/` and confirm `docker_server=false`, the expected `service_port`, and `impl.code_path=models/autoports/<model>`.

If exact matching is unclear, inspect `model_spec.json` only to find the benchmark/eval recipe and to see how the local checkout names fields:

```bash
python3 - <<'PY'
import json
specs=json.load(open("model_spec.json"))
hf="HF_MODEL_HERE"
for item in specs:
    if item.get("hf_model_repo")==hf or item.get("model_name")==hf.split("/")[-1]:
        print(item.get("model_name"), item.get("hf_model_repo"), item.get("device"))
PY
```

Do not stop at a matching TTI model name. Before launching, print the selected spec path and check that it contains `models/autoports/<model>`. After the run, inspect the copied run spec and release report data and confirm the same autoport path is present.

Known TTI issue: older `tt-inference-server` checkouts accept a custom `--runtime-model-spec-json` or `--model-spec-json`, then still validate the model/device or benchmark setup against the built-in `MODEL_SPECS` / `BENCHMARK_CONFIGS`. Symptoms include a valid autoport spec failing with `model:=... does not support device:=...` or `not found in BENCHMARKS_CONFIGS`. This is fixed in `tt-inference-server` by tenstorrent/tt-inference-server#4345, merged 2026-06-26 as `6e396b4`. Before debugging this failure, confirm the checkout includes that merge or a later commit. If it does not, update the checkout to include the merged fix. Do not work around this bug by switching to a stock TTI model spec or stock implementation.

## External-Server Smoke

Do not start with the full release workflow. First prove the topology with a small smoke that cannot run for hours.

1. Start the generated autoport vLLM server from the tt-metal checkout using the same serving command and flags proven in optimized-vLLM. A typical LLM pattern is:

```bash
cd "$TT_METAL_HOME"
source python_env/bin/activate
MODEL_DIR=models/autoports/<model>
MAX_MODEL_LEN=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["max_model_len"])' "$MODEL_DIR/doc/context_contract.json")
export PYTHONPATH="$TT_METAL_HOME:$VLLM_CHECKOUT:${PYTHONPATH:-}"
export TT_LLAMA_TEXT_VER=<autoport selector if required>
python -m models.common.readiness_check.run_vllm_server \
  --stages serve \
  --model-dir "$MODEL_DIR" \
  --hf-model "$HF_MODEL_OR_LOCAL_WEIGHTS" \
  --mesh-device T3K \
  --max-num-seqs 32 \
  --max-model-len "$MAX_MODEL_LEN" \
  --tt-config '{"sample_on_device_mode": "all"}' \
  --additional-server-args "--served-model-name $HF_MODEL"
```

Match the optimized-vLLM stage's actual command where it differs. The important points are: the server imports the generated autoport, preserves the context contract, and serves the HF model name that TTI will request.

2. Verify the server before invoking TTI:

```bash
curl -fsS "http://127.0.0.1:$SERVICE_PORT/health"
curl -fsS "http://127.0.0.1:$SERVICE_PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$HF_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply in one short sentence.\"}],\"max_tokens\":8,\"temperature\":0}"
```

3. Run a tiny TTI client workflow against the live server. Use a temporary no-Docker spec with one 8-token-in / 8-token-out benchmark request, `disable_trace_capture=true`, and very loose performance targets. This proves that TTI can load the autoport spec, see the external server, send a request, write benchmark output, and run reports.

```bash
cd "$TTI_WORK_ROOT/tt-inference-server"
export CACHE_ROOT="$SMOKE_EVIDENCE_DIR/tti_cache"
export SERVICE_PORT=8000
python3 run.py \
  --model "$TTI_MODEL" \
  --model-spec-json "$AUTOPORT_SMOKE_SPEC" \
  --device "$TTI_DEVICE" \
  --workflow benchmarks \
  --no-auth \
  --skip-system-sw-validation \
  --disable-trace-capture
```

The smoke passes only if `run.py` exits `0`, the TTI-written run spec has `docker_server=false` and `impl.code_path=models/autoports/<model>`, and the benchmark JSON records `completed=1` with `failed=0`.

If the smoke fails before a request is sent because the TTI benchmark client entry point is missing, fix the TTI client environment. For example, create/install the checkout's `BENCHMARKS_VLLM` venv or point the expected `vllm` client command at the already-installed vLLM CLI. Record this as a TTI setup fix. Do not respond by switching to `--docker-server` or a stock implementation.

## Run The Release Workflow

Run the full workflow only after the smoke passes. Keep the generated autoport vLLM server running and run TTI as a client. Never print tokens.

```bash
cd "$WORK_ROOT/tt-inference-server"
export HF_TOKEN="$(cat ~/.cache/huggingface/token)"
export MODEL_SOURCE=huggingface
export HOST_HF_HOME=/localdev/$USER/huggingface
export HF_HOME=/localdev/$USER/huggingface
export PERSISTENT_VOLUME_ROOT="$WORK_ROOT/persistent_volume"
export SERVICE_PORT=8000
python3 run.py \
  --model "$TTI_MODEL" \
  --model-spec-json "$AUTOPORT_MODEL_SPEC" \
  --device "$TTI_DEVICE" \
  --workflow release \
  --no-auth \
  --skip-system-sw-validation
```

Notes:

- Do not pass `--docker-server` on the external-server path.
- `JWT_SECRET=dummy` may still be needed on older tags even with `--no-auth`; set it only if the checkout requires it.
- `--skip-system-sw-validation` is acceptable because `tt-smi` health is validated from the reservation container.
- If the checkout supports `--tt-device` instead of `--device`, use the checkout's help output.
- If the checkout does not support `--model-spec-json`, use the checkout's supported mechanism to point the workflow at the autoport spec. If no such mechanism exists, the stage is blocked on release integration; do not fall back to a built-in stock implementation.
- If Docker is used as an explicit fallback, make sure the container can see the generated autoport code path. Mount, copy, or build from the current tt-metal checkout as needed. A Docker image that only contains stock `models/tt_transformers` cannot validate this stage.
- If a Hugging Face cache location is already provided by the experiment, use it rather than re-downloading.

Run long commands in `tmux` or another durable session. Tee stdout/stderr to a timestamped log under `WORK_ROOT`.

## Recovery Policy

If the workflow fails during server/device initialization with ARC, ERISC, remote Ethernet, or `tt-smi` reset symptoms:

1. Stop only the server/session for this run:

```bash
tmux kill-session -t tti-release-<run-name> 2>/dev/null || true
tmux kill-session -t autoport-vllm-<run-name> 2>/dev/null || true
ssh "$PHYSICAL_HOST" 'docker ps -aq --filter "name=tt-inference-server" | xargs -r docker rm -f'  # only if Docker was used
```

2. Follow `$tt-device-usage` reset recovery from the reservation container. At minimum run the bounded list/reset/list sequence, retry reset once if devices or Ethernet links do not all return, and verify a mesh open/close before relaunching:

```bash
timeout 60 tt-smi -ls --local
timeout 180 tt-smi -r
timeout 60 tt-smi -ls --local
```

3. Relaunch the same TTI workflow. Do not clear a valid completed TT cache after the server has already started and served requests. Only clear a tiny/partial stale TT cache when the failure occurred during first initialization and the cache is clearly incomplete.

If recovery fails or requires authority this agent does not have, report that host-level reboot/reacquire is needed rather than marking model readiness blocked.

## Context And Harness Integrity

Before changing release specs, eval configs, benchmark configs, or server launch flags, compare them to the context contract. Do not make a failing model pass by lowering context. Examples of invalid fixes:

- setting `max_model_len` below the context contract;
- lowering LongBench or other long-context eval limits because the model implementation cannot handle them;
- shortening benchmark prompt or completion lengths to avoid an L1, KV-cache, or trace bug;
- marking a context failure as a harness issue when the request fits inside the HF-advertised context.

Valid harness fixes are narrower: wrong autoport path, wrong tokenizer/chat template, host-sampling-only tests that need an explicit host-sampling compatibility mode, or requests whose prompt plus completion exceeds the true supported context. If the model cannot meet the context contract for any reason other than a hard physical device limit, fix the model path or report a readiness gap; do not weaken release coverage.

Use `$qualitative-check` for prompt-based release checks. Release smokes, qualitative/API requests, eval harness configuration, and report interpretation must record the prompt-format decision and must not use raw-completion output from an instruct model, or invented chat prompts for a base model, as release-readiness evidence.

## Failing Release Tests

If the release workflow exits nonzero because `spec_tests`, `tests`, API parameter conformance, eval harness execution, or benchmark harness execution failed, use `$autofix` before declaring the TTI release stage blocked. Give `$autofix` the exact failed command, model, device, physical host, workflow log path, server log path, report/test output path, and the smallest local repro command that preserves the failure.

Use `$autofix` for report-marked API/spec/test failures even when `run.py` exits 0, because the release wrapper may still generate a report with failed conformance rows. After a fix, rerun the failed workflow or the full release workflow as needed, then regenerate and copy back the final report.

Treat report generation as aggregation, not evidence collection. `release` runs expensive data-collection workflows such as `evals`, `benchmarks`, `spec_tests`, and `tests`, then runs the cheaper `reports` workflow over their outputs. If evals, benchmarks, spec tests, and tests already produced valid raw outputs, and the remaining failure is report generation, report copy-back, waiver text, stale report-input state, or the stage check not finding `report_*.md`, preserve the existing `workflow_logs` and rerun only `run.py --workflow reports` or the exact failing `workflows/run_reports.py` command after fixing the report/report-input state. Rerun expensive workflows only when their raw outputs are missing, stale for the current autoport/spec/commit, invalid, or affected by a model/spec/server change that can change the results. Record the reason for any expensive rerun.

Do not use `$autofix` for pure infrastructure failures such as missing Docker, Hugging Face auth, SSH problems, or ARC/reset hangs; recover those with the topology and reset policy above. For accuracy or performance target failures, first decide whether the evidence points to an implementation/test bug or a real readiness gap. Use `$autofix` for the former. Record the latter in `RUN_NOTES.md` and the final response.

## Release Readiness Failures

Parse the final release report and report data. Classify every failed accuracy, benchmark target, API conformance, and missing/incomparable metric row as one of:

- `fixed`: the issue was fixed and the release report was regenerated;
- `issue-waived`: a current linked issue or release note shows the same row fails for the correct canonical implementation, or proves the eval/benchmark target is invalid for reasons unrelated to this autoport;
- `readiness-fail`: this autoport is below the expected release bar;
- `external-blocker`: a gated-dataset access/auth failure (for example a GPQA token) or a harness/spec mismatch (for example an eval `max_length` above the served context) — a config/environment issue, not this autoport's model quality. Record it distinctly and fix the config (clamp the request to the served context; supply the dataset token) rather than the model.

Disclosure is not a waiver. A row is not `issue-waived` merely because it is called out in `RUN_NOTES.md`. Include the issue URL, affected rows, canonical/control behavior, and why the waiver applies.

For text LLMs, treat `meta_ifeval` and `meta_gpqa_cot` as mandatory quality gates unless a current linked issue proves the correct canonical implementation fails the same eval in the same way. These failures usually indicate a real model or serving bug, such as stale token/position feedback, async decode reset corruption, sampling/seed handling, chat-template mismatch, or prefill/decode mode-switch corruption. Use `$autofix` and rerun the affected evals before accepting the stage.

LongBench and other long-context rows may have legitimate release-harness issues, but they still need row-specific evidence. For example, a current issue may waive `longbench_code_e` or `longbench_fewshot_e` for one model because the canonical release path is using the wrong chat-template setting. Do not generalize that waiver to unrelated eval rows or models.

The final status must say one of:

- `release-readiness-pass`: all required rows passed or have row-specific issue waivers;
- `release-workflow-pass/readiness-fail`: the release workflow ran, but one or more required rows failed without a valid waiver;
- `release-workflow-fail`: the release workflow itself did not complete.

## Report And Copy-Back

The release workflow writes the final customer markdown under:

```text
workflow_logs/reports_output/release/report_<report_id>.md
```

Copy back small handoff artifacts to:

```text
models/autoports/<model>/doc/tti_release/
```

Include:

- final release markdown;
- release report data JSON;
- per-section markdown and summary CSV/JSON files;
- successful run log;
- run spec and runtime model spec JSON;
- small benchmark JSON files;
- a `RUN_NOTES.md` with commands, versions, server mode, host/session, reset actions, report path, and pass/fail summary.

Also include the run spec or report data that proves the implementation path. `RUN_NOTES.md` must have an "Autoport implementation check" line showing the target `models/autoports/<model>` path and whether the copied TTI artifacts matched it.

Do not copy:

- `.env`;
- Hugging Face model cache;
- Docker or persistent TT cache volume;
- model weights;
- generated tensor dumps or reference tensors (`*.tensorbin`, `*.pt`, `*.refpt`);
- raw profiler/op CSV bulk such as `tracy_ops_times.csv`, `profile_log_device.csv`, `ops_perf_results.csv`, and `*_decode_ops.csv`;
- large raw eval sample dumps unless explicitly requested.

After copy-back, remove any `.env` left in the TTI checkout and stop the finished tmux session. Do not release the reservation unless the user or monitor asks.

## Completion Criteria

Done means:

- `run.py --workflow release` exited `0`, or a terminal blocker is documented with exact evidence.
- The copied run spec or release report data proves that the evaluated implementation path is the target `models/autoports/<model>` directory.
- No copied final report or run spec identifies the evaluated implementation as stock `models/tt_transformers`, `models/demos`, or another packaged implementation for the same HF model.
- The copied run spec, server launch, and report data preserve the supported context from `doc/context_contract.json`.
- `$qualitative-check` evidence shows prompt-based release checks used the HF-declared prompt format, and `RUN_NOTES.md` records the tokenizer/chat-template decision plus the TTI/eval setting or rendered prompt evidence.
- Valid non-aligned prompt lengths either pass, or the stage records the exact model bug and remains not ready.
- Any failing release tests or API conformance rows were either fixed with `$autofix` and rerun, or explicitly classified as non-test readiness gaps with evidence.
- Any failed accuracy, benchmark target, API conformance, or incomparable metric row is classified as `fixed`, `issue-waived`, or `readiness-fail`. A `readiness-fail` means the stage is not clean-pass.
- `meta_ifeval` and `meta_gpqa_cot` pass for text LLMs, or each failure has a current linked issue proving the correct canonical implementation fails the same eval in the same way.
- Final release markdown is copied under `models/autoports/<model>/doc/tti_release/`.
- `RUN_NOTES.md` records the exact server mode, host/session, repo tag, Docker image/version if Docker was used, command, env variables that mattered, reset/retry actions, copied artifacts, release-readiness status, failed rows, and waiver issue links where applicable.
- The report is skimmed and the README/RUN_NOTES call out any failing accuracy, benchmark target, or API conformance checks with the classification above.
- There is no leftover autoport vLLM server, TTI release tmux session, or `tt-inference-server` Docker container from this run.
- The final response names the release report path and whether a Pushover or other requested notification was sent.
