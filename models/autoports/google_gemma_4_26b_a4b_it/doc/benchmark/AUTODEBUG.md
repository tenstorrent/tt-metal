# Benchmark setup AutoDebug

Source/environment inspection only, 2026-09-27. No hardware commands, dependency installations, server launches, or implementation edits were performed.

## Failing contract

The final benchmark requires upstream lm-evaluation-harness (the selected runner expects 0.4.13), two live serving profiles, and a full-phase host-wall collector before accuracy starts. The supplied checkout does not currently satisfy these setup prerequisites.

## Findings and discriminating checks

1. **No provisioned harness was found.** `importlib.util.find_spec('lm_eval')` returns `None` in both `/home/container_app_user/tt-metal/python_env/bin/python` and `/opt/venv/bin/python`. The active environment has vLLM and datasets; `/opt/venv` has neither an installed harness nor datasets. Bounded filesystem searches under `/opt`, `/workspace`, `/home/container_app_user`, the active environments, and user caches found no `lm_eval` package or lm-evaluation-harness checkout. The only matching directory is `vllm/.buildkite/lm-eval-harness`, which contains CI caller scripts, not the upstream harness. This proves absence in the inspected provisioned environments, not universal filesystem absence.
2. **The container build fallback is unavailable here.** `shutil.which('docker')` and `shutil.which('podman')` both return `None`; `/var/run/docker.sock` is absent. The user's AGENTS instructions prohibit installing dependencies. Installing a fresh harness is therefore not an authorized setup repair. A preprovisioned compatible client or an explicit change to that restriction is needed.
3. **Stage 10 does not leave a working server running.** `doc/optimized_vllm/README.md` explicitly records cleanup with no owned serving processes remaining. Process inspection finds no `vllm.entrypoints`, `run_vllm_server`, or `EngineCore` process. A bounded TCP connect to `127.0.0.1:8000` returns `111` (connection refused). Handoff docs identify localhost only; inspection found no existing remote serving endpoint. The recorded launch is recoverable from `doc/optimized_vllm/server_command.json`; its first new launch must count inside the benchmark runner clock.
4. **A matching collector is not present in the inspected implementation.** Searches of the autoport TT implementation and tests found no `roofline_command` or full-phase prefill/decode host timing implementation. Existing decoder roofline artifacts measure a different single-layer scope and are not valid full-model serving accounting. `tt/generator_vllm.py:298` records an event after asynchronous token readback; this supplies a possible completion boundary, but does not itself time complete phases or calculate executed-shape work. A collector must instrument actual consumption/completion boundaries without extra synchronization, then validate both server profiles. No such measurement is established by source inspection.

## Recovery and limits

Provision an upstream harness client and its required scorer dependencies through an authorized existing environment, or change the no-install constraint. Then implement and verify full-phase collection before accuracy; launch the 32-slot server within the runner clock, collect its phase evidence before shutdown, and repeat for the one-slot configuration. Preserve precision, four-chip mesh, 30 layers and 262144 context. Stage 10's existing headline request uses 32 server slots despite concurrency 1, so it cannot stand in for the required one-slot profile.

No hardware failure has been observed and reset/recovery is not indicated. These are setup/integration findings, not model accuracy findings. No performance or accuracy measurements were generated in this investigation.
