# Stock CI dry/release route investigation

Outcome: operator decision 8 removes the local-Docker requirement. The existing dispatch supports immutable per-suite CI runs and a separate benchmark run without spec tests. Two controls needed for the exact final contract are not exposed by the granted dispatch: benchmark targets only, and a dry-run stop with retained task evidence. No source or workflow changed; no device used.

## Supported dispatch

Use the existing `manual-tt-shield-dispatch.yml` in `tenstorrent/tt-agentic-bringup-qb2`, with `model=gemma-4-31B-it`, `runner-label=bh-qb-ge`, `device-type=p300x2`, `impl-of-model=gemma4-31b-autoport-v3`, immutable `tt-metal-git-ref`, immutable per-suite `inference-server-git-ref`, `vllm-git-ref=ca7a18246e4111629ed6cb4a7d1192002749fccf`, `run-full-evals=false`, `run-ai-summary=false`, `create-issue-comment=false`.

Set `workflow=evals` for GPQA, `workflow=agentic` for each tool suite, and `workflow=benchmarks` for the benchmark matrix. Keep the other suites' ci-nightly limit zero in each per-suite TTI commit as operator instructions prescribe. Use the identical SHAs, workflow, limits and tasks for that arm's dry and release dispatch. Do not use `workflow=release`: it includes spec tests unconditionally.

Evidence: `setup/dispatch-workflow.yml` input definitions/forwarding; `setup/shield-run-tests.yml:333-355` constructs run.py with workflow, Docker, dev catalog, implementation and ci-mode. TTI `workflow_module/workflows.py:538-539,620-642` hardcodes release evals/benchmarks/spec_tests and conditionally appends agentic. `test_module/llm_tests/agentic_eval_tests.py:96-126` separates agentic from common eval tasks.

## Benchmark matrix limitation

`reference_config/benchmarking/benchmark_config.py:651` omits generic sweeps only when the client environment has nonempty `ONLY_BENCHMARK_TARGETS`. The manual dispatch has no such input or general environment/additional-arguments input. Its downstream test workflow exports this variable only for a pull_request event (`setup/shield-run-tests.yml:294-297`), not workflow_dispatch.

Setting the variable in model server env does not solve this: `workflows/workflow_dispatch.py:373-382` forwards only TT_RUN_COMMAND as client environment overrides; server env is used by Docker setup. A host-only resolution experiment against the present autoport config found 23 rows without the variable and exactly `(ISL=128, OSL=128, C=1, prompts=8)` with it. The other 22 rows are generic sweeps, not the stage-11 frozen matrix. The process-local experiment setting device_model_spec.env_vars confirmed `_engine_env(spec)` does not include ONLY_BENCHMARK_TARGETS; it did not modify files.

Thus the exact matrix requires a supported client-side control outside the current dispatch surface (for example operator runner environment provisioning), or explicit authorization of a generic release-route repair. Do not hide a client environment mutation in model source, alter context to filter rows, or claim the 23-row run equals the one-row matrix.

## Dry-run evidence limitation

The granted workflow has no readiness-only/dry-run input. run.py's `--print-docker-cmd` only prints a command and exits (`run.py:405-409`); `--list-tests` applies to spec tests (`run.py:512-515`). Neither launches every task environment, concurrent sandboxes and model replies as the goal requires.

Cancelling a stock CI job is not a proved substitute: both report upload and workflow-log artifact upload are guarded `always() && !cancelled()` (`setup/shield-run-tests.yml:623-624,676-683`). Cleanup stops and removes Docker containers (`:685+`). Docker log tails and run specs have always() steps (`:633,669`), but these are not the complete per-task evidence required to prove every frozen environment built and planned concurrency received replies. No mid-run stop operation or live task-evidence retrieval was found in the granted interface. The agentic runner checkpoints completed suite Blocks (`test_module/llm_tests/agentic_eval_tests.py:211-217`), but this does not override cancelled artifact upload.

A fully completed identical-SHA rehearsal would preserve logs and demonstrate more execution, but would produce scores and is not literally the prescribed stop-after-start/no-score dry run. Do not silently substitute it. The coordinator should obtain a concrete operator decision or supported route control while continuing the already authorized smoke/nightly work; this finding does not invalidate those rungs.

## Validation

Host only: `python -m pytest tests/workflows/test_gemma4_31b_autoport_v3.py -q --disable-warnings`: 6 passed in 1.79 s. Direct `get_llm_configs` resolution: 23 rows stock, 1 targets-only; model env not forwarded to client. Source snapshots inspected locally; no remote mutation, dispatch, hardware access or generic/model source edits. Used device hours: 0. Problem remains open for exact final dry/benchmark control; source evidence alone does not prove a live route.

## Follow-up: local benchmark arm resolves the benchmark control

Operator decision 8 retains local serving measurements without task containers. Extended release has a separate benchmark arm. Consequently the exact benchmark can use stock TTI client mode against the local running server, without any CI workflow change:

```bash
ONLY_BENCHMARK_TARGETS=1 python run.py \
  --dev-mode --model google/gemma-4-31B-it --impl gemma4-31b-autoport-v3 \
  --device p300x2 --workflow benchmarks \
  --server-url http://127.0.0.1:8000 --disable-trace-capture --no-auth
```

The coordinator supplies the actual server port/authentication, bounded client timeout, persistent workflow logs and queue entry. `--no-auth` matches an unauthenticated local server only. No `--docker-server` or `--local-server` is passed: the existing-server client does not own server lifecycle. `workflows/README.md:382-448` documents this supported mode, and `run.py:1380-1400` adds a server command only for explicitly managed Docker/local servers. `run.py:970-988` makes HF_TOKEN optional for benchmark-only clients.

Host-only parser+resolver confirmation used the exact argv above and TTI checkout HEAD `bb321bf0f066dd472f51bd0b3f8875d47a0b3cfb`: resolved `benchmarks`, Docker=false, local_server=false, model_id `id_gemma4-31b-autoport-v3_gemma-4-31B-it_p300x2`; `get_llm_configs` returned exactly `(128,128,1,8)`. This did not contact a server or run a benchmark. Final dry/release must use the same chosen TTI SHA (current HEAD is only the inspected candidate) and matching server batch/settings.

Compared all 14 published runtime files at tt-metal SHA `08ee8790bcbcd9e85cb799593b2bbaf40b785134` with the execution tree by SHA256: zero mismatches. The coordinator still needs live server evidence that it imports those paths, uses the pinned HF/tokenizer revision and stock plugin runtime, and has full60 layers. Static hash parity is not that runtime proof.

The benchmark-control problem is therefore resolved at the route-design level by the local arm, pending its normal dry/live measurement. CI task dry-run stopping/evidence limitations above remain unresolved. A completed identical rehearsal with scores ignored remains only a proposed explicit operator exception, not an authorized substitution.

## Prepared local preflight (not launched)

From TTI checkout at `bb321bf0f066dd472f51bd0b3f8875d47a0b3cfb`, with the coordinator's B1 server already healthy at port 8000, the stock command is:

```bash
ONLY_BENCHMARK_TARGETS=1 \
CACHE_ROOT=/home/mvasiljevic/tt-metal-gemma4-31b-pipeline/bringup/artifacts/stage12-local-tti-preflight \
timeout 900 python run.py \
  --dev-mode --model google/gemma-4-31B-it --impl gemma4-31b-autoport-v3 \
  --device p300x2 --workflow benchmarks --tools vllm \
  --server-url http://127.0.0.1:8000 --no-auth
```

Coordinator owns endpoint verification, B1 configuration, bounded server lifetime and launch. The 900-second client bound matches its 0.25-device-hour preflight box; this is a host client timeout against an independently bounded server, not termination of an outstanding hardware job. Capture stdout/stderr in the same persistent artifact directory, and preserve exit status. Do not count dependency download time as a benchmark measurement. If dependency setup consumes the slot, finish setup host-only before a fresh measured job.

`--disable-trace-capture` is removed. Search of current TTI source finds its runtime use only while constructing managed local/Docker/multihost server launch arguments (`workflows/run_local_server.py:236`, `workflows/run_docker_server.py:821`, `workflows/multihost_orchestrator.py:557`). Existing-server mode creates no such launch. It does not transmit an instruction to the already-running server or change its canonical traced sampler/decode.

Required stock benchmark venv: `.workflow_venvs/.venv_llm_vllm`, Python 3.11, `requirements/llm-vllm.txt` (vllm==0.13.0 and client dependencies), with `requirements/llm-vllm-overrides.txt` forcing transformers5.x. `launchers/run_llm_bench.py:23-45` selects this venv for `--tools vllm`; `_launcher_common.py:42-58` installs idempotently and re-execs the engine there. This path does not require the much larger separate WORKFLOW_RUN_SCRIPT venv. Inspection found only `.venv_bootstrap_uv` present, so first invocation still needs stock dependency setup. No dependencies were installed in this investigation.

One unresolved preflight detail is client tokenizer pinning: `llm_module/drivers/vllm.py` builds ordinary benchmark argv with the HF model name but no revision argument; `--tokenizer` is emitted only for token-timing mode. Existing server revision settings do not prove the client's tokenizer revision. Before measuring, coordinator must prove the client's resolved tokenizer bytes match revision `842da3794eaa0b77d5f08bae87a17459d91ff475` (e.g. a validated offline cache restricted to that snapshot), or seek a generic harness correction if explicit client revision forwarding is required. Do not silently rely on current Hub main or change benchmark protocol to token-timing merely for this flag. This is source-based uncertainty; no wrong tokenizer was observed because no client ran.

### Final prepared command: pinned client tokenizer via stock custom weights

The supported `--custom-weights` route resolves the tokenizer concern without source changes. `derive_custom_weights_spec` (`workflows/model_spec.py:1723-1769`) preserves the base device/performance config and changes its API/model identity to the custom weights path. `run.py:1177-1192` applies it after resolving the autoport. Pass the existing immutable local snapshot, which the coordinator's server exposes as an additional served-model name. Do not pass `--host-weights-dir`: that is unnecessary for an existing server and rewrites the internal model to a Docker mount path.

```bash
ONLY_BENCHMARK_TARGETS=1 \
CACHE_ROOT=/home/mvasiljevic/tt-metal-gemma4-31b-pipeline/bringup/artifacts/stage12-local-tti-preflight \
timeout 900 python run.py \
  --dev-mode --model google/gemma-4-31B-it --impl gemma4-31b-autoport-v3 \
  --custom-weights /home/bringup/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/842da3794eaa0b77d5f08bae87a17459d91ff475 \
  --device p300x2 --workflow benchmarks --tools vllm \
  --server-url http://127.0.0.1:8000 --no-auth
```

Host-only confirmation: snapshot directory exists; exact parser/resolver and `get_llm_configs` return one row `(128,128,1,8)`; ordinary (token_timing=false) `build_vllm_bench_serve_argv` emits `--model` with the absolute pinned snapshot. `test_module/llm_tests/llm_performance_tests.py:79-83` derives that driver model from the resolved spec's hf_model_repo. The tokenizer defaults to that model directory in the ordinary vLLM client; no mutable Hub main ref is needed. Live preflight should retain the emitted argv and resulting tokenizer identity as confirmation. No endpoint contacted.

Provision the missing **separate client venv** before the device preflight with the stock setup API, in this TTI checkout (host-only, bounded; not executed here):

```bash
timeout 1800 python - <<'PY'
from workflows.bootstrap_uv import bootstrap_uv
bootstrap_uv()
from workflows.workflow_venvs import VENV_CONFIGS
from workflows.workflow_types import WorkflowVenvType
cfg = VENV_CONFIGS[WorkflowVenvType.LLM_VLLM]
assert cfg.setup(model_spec=None)
print(cfg.venv_python)
PY
```

This installs into `.workflow_venvs/.venv_llm_vllm` only, not the active device-server environment. It uses the pinned stock requirements/override files as discussed above. Record setup elapsed time separately; do not spend the 0.25-device-hour preflight box on this download/install. No dependency installation performed by this effort.
