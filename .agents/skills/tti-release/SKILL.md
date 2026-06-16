---
name: tti-release
description: Run the tt-inference-server model release workflow for a completed TTNN/vLLM model and produce the customer-facing readiness markdown report. Use after optimized-vLLM or at model readiness handoff time, especially from an IRD reservation container where Codex must SSH to the physical loudbox host for Docker while using the reservation container for tt-smi health/reset.
---

# TTI Release

## Overview

This skill runs the Tenstorrent `tt-inference-server` release workflow for a model whose vLLM serving path is already complete. The goal is a copied-back markdown release report plus a short local handoff note with exact commands, versions, recovery actions, and residual readiness gaps.

## Topology

Assume the agent usually starts inside an IRD reservation/Codex container:

```text
reservation container:
  has tt-smi and the experiment/model context
  may not have Docker
  use it for device health, tt-smi reset, and local model evidence

physical loudbox host:
  has Docker
  run tt-inference-server here with --docker-server
  copy workflow reports back to the model doc/evidence directory
```

Do not try to run `--docker-server` inside the reservation container unless Docker socket access is deliberately mounted and `docker ps` works there. Prefer the physical host path.

## Preflight

1. Identify:

```text
HF model id
model autoport directory
physical host name, for example wh-lb-80
TTI device name, usually t3k for T3K loudboxes
experiment evidence directory, if separate from models/autoports/<model>/doc/
```

If the prompt does not give the physical host, infer it from reservation metadata or the reservation container hostname only when obvious. Otherwise stop and ask for the host; a wrong host can use the wrong hardware.

2. Check Docker on the physical host and devices in the reservation container:

```bash
ssh "$PHYSICAL_HOST" 'docker ps >/dev/null && echo DOCKER_OK'
tt-smi -ls --local
```

If `docker ps` fails on the physical host, do not continue with `--docker-server`. Report `physical-host Docker unavailable`.

3. Confirm vLLM readiness artifacts exist before this phase:

```text
models/autoports/<model>/doc/optimized_vllm/README.md
models/autoports/<model>/doc/optimized_vllm/work_log.md
models/autoports/<model>/readiness_vllm/
```

If optimized-vLLM is blocked only by a recoverable ARC/reset error, recover and resume optimized-vLLM first. Do not jump to TTI release ahead of a recoverable earlier-stage hardware blocker.

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

## Model Name Selection

Find the TTI CLI model name that corresponds to the HF model. Prefer an exact instruct/chat variant. If exact matching is unclear, inspect `model_spec.json`:

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

For Llama 3.1 8B instruct on T3K, the working TTI name was `Llama-3.1-8B-Instruct`.

## Run The Release Workflow

Run from the physical host. Never print tokens.

```bash
cd "$WORK_ROOT/tt-inference-server"
export HF_TOKEN="$(cat ~/.cache/huggingface/token)"
export MODEL_SOURCE=huggingface
export HOST_HF_HOME=/localdev/$USER/huggingface
export HF_HOME=/localdev/$USER/huggingface
export PERSISTENT_VOLUME_ROOT="$WORK_ROOT/persistent_volume"
export JWT_SECRET=dummy
python3 run.py \
  --model "$TTI_MODEL" \
  --device "$TTI_DEVICE" \
  --workflow release \
  --docker-server \
  --no-auth \
  --skip-system-sw-validation
```

Notes:

- `JWT_SECRET=dummy` may still be needed on older tags even with `--no-auth`.
- `--skip-system-sw-validation` is acceptable when running from the physical host because `tt-smi` health is validated from the reservation container.
- If the checkout supports `--tt-device` instead of `--device`, use the checkout's help output.
- If a Hugging Face cache location is already provided by the experiment, use it rather than re-downloading.

Run the command in `tmux` or another durable session on the physical host. Tee stdout/stderr to a timestamped log under `WORK_ROOT`.

## Recovery Policy

If the workflow fails during server/device initialization with ARC, ERISC, remote Ethernet, or `tt-smi` reset symptoms:

1. Stop only the TTI release server/session on the physical host:

```bash
ssh "$PHYSICAL_HOST" 'tmux kill-session -t tti-release-<run-name> 2>/dev/null || true; docker ps -aq --filter "name=tt-inference-server" | xargs -r docker rm -f'
```

2. Reset devices from the reservation container:

```bash
timeout 180 tt-smi -r
timeout 60 tt-smi -ls --local
```

3. Relaunch the same TTI workflow. Do not clear a valid completed TT cache after the server has already started and served requests. Only clear a tiny/partial stale TT cache when the failure occurred during first initialization and the cache is clearly incomplete.

If reset hangs or devices do not return, report that host-level reboot/reacquire is needed rather than marking model readiness blocked.

## Failing Release Tests

If the release workflow exits nonzero because `spec_tests`, `tests`, API parameter conformance, eval harness execution, or benchmark harness execution failed, use `$autofix` before declaring the TTI release stage blocked. Give `$autofix` the exact failed command, model, device, physical host, workflow log path, server log path, report/test output path, and the smallest local repro command that preserves the failure.

Use `$autofix` for report-marked API/spec/test failures even when `run.py` exits 0, because the release wrapper may still generate a report with failed conformance rows. After a fix, rerun the failed workflow or the full release workflow as needed, then regenerate and copy back the final report.

Do not use `$autofix` for pure infrastructure failures such as missing Docker, Hugging Face auth, SSH problems, or ARC/reset hangs; recover those with the topology and reset policy above. For accuracy or performance target failures, first decide whether the evidence points to an implementation/test bug or a real readiness gap. Use `$autofix` for the former. Record the latter in `RUN_NOTES.md` and the final response.

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
- a `RUN_NOTES.md` with commands, versions, host/session, reset actions, report path, and pass/fail summary.

Do not copy:

- `.env`;
- Hugging Face model cache;
- Docker or persistent TT cache volume;
- model weights;
- large raw eval sample dumps unless explicitly requested.

After copy-back, remove any `.env` left in the physical-host repo and stop the finished tmux session. Do not release the reservation unless the user or monitor asks.

## Completion Criteria

Done means:

- `run.py --workflow release` exited `0`, or a terminal blocker is documented with exact evidence.
- Any failing release tests or API conformance rows were either fixed with `$autofix` and rerun, or explicitly classified as non-test readiness gaps with evidence.
- Final release markdown is copied under `models/autoports/<model>/doc/tti_release/`.
- `RUN_NOTES.md` records the exact physical host, repo tag, Docker image/version, command, env variables that mattered, reset/retry actions, copied artifacts, and residual readiness gaps.
- The report is skimmed and the README/RUN_NOTES call out any failing accuracy, benchmark target, or API conformance checks.
- The physical host has no leftover `tt-inference-server` Docker container from this run and no leftover release tmux session.
- The final response names the release report path and whether a Pushover or other requested notification was sent.
