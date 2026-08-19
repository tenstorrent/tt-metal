# Shield CI onboarding: Gemma 4 31B autoport on QB2 (p300x2)

Date: 2026-08-14 UTC

How this autoport is dispatched through `tenstorrent/tt-shield`, what had to change
to make that possible, and what is verified versus still open.

## Dispatch history

| Run | Workflow | Outcome |
| --- | --- | --- |
| [31801765378](https://github.com/tenstorrent/tt-shield/actions/runs/31801765378) | `spec_tests` | build succeeded; tests failed, no applicable suites |
| [31807380436](https://github.com/tenstorrent/tt-shield/actions/runs/31807380436) | `benchmarks` | image reused; **server came up in the container**, failed on the missing chat template |
| [31824560569](https://github.com/tenstorrent/tt-shield/actions/runs/31824560569) | `benchmarks` | chat template resolved; died on the 5s metal op watchdog |
| [31832502684](https://github.com/tenstorrent/tt-shield/actions/runs/31832502684) | `benchmarks` | rebuild with `DISABLE_METAL_OP_TIMEOUT=1`; aborted by the runner itself — host disk at 81%, `631G /data/mgiermakowski`. Not our failure |
| [31858340006](https://github.com/tenstorrent/tt-shield/actions/runs/31858340006) | `benchmarks` | **success** — 17 sweep points, 0 failed requests, but every block `NA (ungraded)`. See `shield_ci_results_audit.md` |
| [32006778390](https://github.com/tenstorrent/tt-shield/actions/runs/32006778390) | `release` | queued 3h+ behind other tenants, never started; **cancelled** once the watchdog and eval fixes landed, because `resolve-shas` pins SHAs at dispatch time and it would have tested the superseded config |
| [32028283074](https://github.com/tenstorrent/tt-shield/actions/runs/32028283074) | `release` | failed `resolve-shas` in 39s — dispatched with the stale `vllm-git-ref=dev`. Never reached a runner, no device time. See the `main` note above |
| [32028455474](https://github.com/tenstorrent/tt-shield/actions/runs/32028455474) | `release` | watchdog restored, eval config in place, `vllm-git-ref=main`. **Image build failed**: `error: pathspec 'c127c17' did not match any file(s) known to git`. `resolve-shas` resolves the SHA from `vllm-tt-plugin`, but this branch's Dockerfile still ran `git clone https://github.com/tenstorrent/vllm.git` and checked that SHA out there. Fixed by merging `origin/main`, which had already migrated the Dockerfile to clone `vllm-tt-plugin` |
| [32031270481](https://github.com/tenstorrent/tt-shield/actions/runs/32031270481) | `release` | dispatched on the merged branch; resolve-shas and determine-server-type passed and the image build started. **Cancelled deliberately** when the decision was taken to work locally only, so the shared `bh-qb-ge` runner was not held |

Both `dev` and `main` fail until the branch carries main's Dockerfile: `dev`
fails SHA resolution (no such branch in `vllm-tt-plugin`) and `main` resolves a
SHA that does not exist in the old `tenstorrent/vllm` the stale Dockerfile
clones. Merging `origin/main` is the fix, not a different ref.

### `bh-qb-ge` is a single host, so release jobs queue behind each other

Run 32006778390 sat `queued` with no steps started for over three hours because
`bh-qb-ge` is a **single** p300x2 host and three release jobs were contending for
it: `gemma-4-31B-it` (vvukoman), ours, and `diffusiongemma-26B-A4B-it` (zni).
Nothing needed fixing on our side for that.

One thing to carry forward: the stock **instruct** model's `release` lane was
failing on that same runner at the same time, so if ours fails, the shared-lane
explanation must be ruled out before it is attributed to the autoport.

Worth noting the PR these dispatches run under is *"switching from vllm-fork to
vllm-tt-plugin (main branch and new repo)"* — the platform is moving to exactly
the plugin mechanism this onboarding used, which is independent support for having
avoided a `tenstorrent/vllm` patch. That same migration is what invalidated the
`vllm-git-ref=dev` recipe.

### Run 31807380436 proved the container path end to end

This is the run worth reading. Inside the Shield-built image, before it failed:

```text
Set EXTRA_MODELS_DIR to /home/container_app_user/tt-metal/models/autoports/vllm_bundles
Registered TT model TTGemma4ForConditionalGeneration -> models.autoports.google_gemma_4_31b...
GPU KV cache size: 103,872
Maximum concurrency for 113,280 tokens per request: 1.00x
```

So in the container: `TT_METAL_HOME` resolved to `/home/container_app_user/tt-metal`
and the entrypoint derived the bundle path from it, the bundle registered the
autoport rather than `models/demos/gemma4`, the cwd-independent model-dir fix
held (no `context_contract.json` error), and the KV pool matched the 103,872
tokens measured locally. Weights access and `HF_TOKEN` on the runner also
worked. Four of the risks flagged before dispatching cleared here; only the
prompt contract remained.

### The base checkpoint needs a chat template, not just a caveat

Run 31807380436 then failed every request with

```text
Failed to load AutoTokenizer chat template for google/gemma-4-31B
ChatTemplateResolutionError: As of transformers v4.44, default chat template is no longer allowed
ERROR serving.py:311] Error in preprocessing prompt inputs
```

TTI's trace capture posts to `/v1/chat/completions` unconditionally
(`utils/prompt_client.py::call_chat_inference`), so a checkpoint with
`chat_template=None` cannot serve it. The spec now passes the autoport's own
`doc/vllm_integration/chat_template.jinja`, the same file the recorded Stage 09
serving run used: `bos_token` plus newline-joined message content, preserving
base completion semantics rather than inventing an instruct format. The path is
repo-relative and anchored to `TT_METAL_HOME` by
`run_vllm_api_server.resolve_repo_relative_vllm_args`, because the checkout sits
at a different absolute path in the container than in a local tree and spec
values are not variable-expanded.

### The default hang watchdog aborts a cold-cache run

Run 31824560569 resolved the chat template
(`Resolved vLLM --chat-template to /home/container_app_user/tt-metal/...`), served
requests, and then EngineCore died in `multichip_decoder` at
`ttnn.linear(activated, self.down_prefill)`:

```text
TT_THROW: TIMEOUT: device timeout in fetch queue wait, potential hang detected
tt_metal/impl/dispatch/system_memory_manager.cpp:702
```

`run_vllm_api_server.set_metal_timeout_env_vars` sets
`TT_METAL_OPERATION_TIMEOUT_SECONDS=5.0`. The same log shows `JIT cache stats:
0/636 hits` and `init engine ... took 73.48 seconds` against 3.80 s warm locally,
so a first-compile prefill matmul simply outran the 5 s limit.

**Resolved 2026-08-17 by raising the threshold, not disabling the watchdog.** The
spec briefly set `DISABLE_METAL_OP_TIMEOUT=1`, which suppressed hang detection
*and* the automatic tt-triage capture wired through
`TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE` — so a pass obtained that way
could not distinguish "did not hang" from "hangs are no longer detected". It now
sets `TT_METAL_OPERATION_TIMEOUT_SECONDS: "120"` instead. Detection stays on: a
wedged device never completes, so it still trips and still triages. No TTI code
change was needed, because `set_runtime_env_vars` applies spec `env_vars` *after*
`set_metal_timeout_env_vars`, so the spec value wins while the triage hook stays
wired. 120 s matches the value `tt-media-server/scripts/sp_env_sample.sh` already
ships.

The threshold was measured rather than guessed. On this QB2 host, with
`TT_METAL_CACHE` pointed at an empty directory (`JIT cache stats: 0/829 hits`),
the multichip prefill+decode layer bench completed **at the stock 5.0 s limit
with no timeout**, twice — 124 s and 125 s wall, identical results. So the
cold-compile cost is bounded at layer scale; the raise is headroom for the full
60-layer, 1836-kernel first compile that tripped this run, not a blanket escape
from the check. `tests/test_gemma4_31b_autoport_spec.py` in tt-inference-server
now asserts the disable cannot return.

### Changing TTI code or specs requires a rebuild

`run_docker_server.py` bind-mounts `vllm-tt-metal/src`, `utils`, `tests`,
`reference_config`, and the runtime spec JSON **only under `--dev-mode`**.
Without it the container uses the image's baked copies, so a change to the
entrypoint or to a model spec needs a fresh image; only passing different
dispatch inputs can reuse one. That is why runs 31824560569 and 31832502684
each rebuilt after a spec or entrypoint change.

### Do not use `spec_tests` for this model

Run 31801765378 reached the hardware runner and then failed with

```text
No spec test suites match model='gemma-4-31B' device='p300x2' — skipping spec_tests.
⏭  task=spec_tests no-op rc=0
No blocks accumulated — cannot generate report.
❌ command=workflow rc=1 error=no_blocks
```

`test_module/test_suites/llm.json` defines spec-test matrices for only
`qwen3_32b`, `llama_3_1_8b`, `llama_70b_family`, and `gpt_oss_20b`. **No Gemma
variant has spec-test suites**, including `gemma-4-31B-it`, which is already in
nightly CI on this device. The workflow treats zero selected suites as a failure
rather than a skip, so `spec_tests` cannot pass for this model family until
suites are added. Use `benchmarks`, `evals`, or `release`.

### What run 31801765378 did establish

- `determine-server-type` resolved `gemma-4-31B` with `impl-of-model=default` to
  the `gemma4_31b_autoport` impl via the P300X2 spec's `default_impl`. This
  confirms the prod spec entry was both necessary and sufficient for model
  resolution.
- **`build-tt-inference-server` succeeded**, building tt-metal at this branch's
  head even though the branch is ~1,900 commits behind main. The branch-age
  concern did not materialise for the image build.
- The built image is
  `ghcr.io/tenstorrent/tt-shield/vllm-tt-metal-src-dev-ubuntu-22.04-amd64:0.19.0-e4297d3bc2f10056adf81f50fd94fbd08cd3f5e1-bf98d55-94771201217`.
  It encodes the **dispatch-resolved** commits (this branch's tt-metal head and
  vLLM `dev` = `bf98d55`), not the prod spec's `vllm_commit: 6b4a3a7`. So the
  dispatch inputs drive what is built and the spec pins affect image naming and
  validation only. Convenient: `bf98d55` is exactly the vLLM commit everything
  was verified against locally.
- Passing that image as `-f docker-image=...` on a re-dispatch skips
  `resolve-shas` and the build entirely, turning a 1-4 hour run into just the
  hardware phase.

## What tt-shield is

tt-shield owns the CI that builds the inference-server image and runs tests
against it on real hardware; tt-inference-server supplies the model specs, test
modules, and eval/benchmark config. The two are coupled by a documented
cross-repo naming contract in `tt-inference-server/utils/model_naming.py`.

`on-dispatch.yml` chains: `determine-server-type` → `resolve-shas` →
`build-tt-inference-server` → `run-tests`. It builds a Docker image, so anything
the model needs must be in the image or set by the in-image entrypoint.

## Dispatch command

```bash
gh workflow run on-dispatch.yml --repo tenstorrent/tt-shield \
  -f custom-model=gemma-4-31B \
  -f model=Llama-3.1-8B-Instruct \
  -f runner-label=bh-qb-ge \
  -f device-type=p300x2 \
  -f workflow=benchmarks \
  -f impl-of-model=default \
  -f tt-metal-git-ref=mvasiljevic/fast-models-fast/gemma4-31b \
  -f inference-server-git-ref=mvasiljevic/fast-models-fast/gemma4-31b \
  -f vllm-git-ref=main
```

> **`vllm-git-ref` must be `main`, not `dev`.** tt-shield now resolves vLLM from
> **`tenstorrent/vllm-tt-plugin`** (`workflow_resolve-shas.yml`), not
> `tenstorrent/vllm`. That repo has no `dev` branch, so a dispatch with `dev`
> fails `resolve-shas` in ~39s with `HTTP 422` from
> `/repos/tenstorrent/vllm-tt-plugin/commits/dev` — observed on run
> [32028283074](https://github.com/tenstorrent/tt-shield/actions/runs/32028283074).
> `on-nightly.yml` uses `main`. The `EXTRA_MODELS_DIR` bundle mechanism this
> autoport is selected through is present in the new repo
> (`src/vllm_tt_plugin/platform.py::_iter_extra_model_bundles`), so the migration
> does not change how the model registers.

### `release` is the intended lane, not `benchmarks`

`on-nightly.yml` sets `setup-vars.outputs.workflow: "release"` and passes
`schedule: "nightly"`. So a model registered in `models-ci-config.json` under
`ci.nightly` is scheduled by the nightly cron, and that cron dispatches the
**`release`** workflow. This model's registration (`gemma-4-31B`,
`ci.nightly.devices: [P300X2]`, `inference_engine: vLLM`) mirrors the stock
`gemma-4-31B-it` entry exactly, so `release` on P300X2 *is* its intended
coverage.

`release` runs **evals and benchmarks** (`workflow_dispatch.py`:
`_ENGINE_EVAL_WORKFLOWS` and `_ENGINE_BENCHMARK_WORKFLOWS` both contain
`WorkflowType.RELEASE`). A `benchmarks`-only pass therefore covers half the
intended lane.

- `custom-model` is the documented way to test a model absent from the dropdown
  ("⭐ use for testing new models"); `model` is a required placeholder it
  overrides.
- `bh-qb-ge` is the QB2 runner label, taken from real passing runs rather than
  guessed (`Qwen3-Embedding-4B | bh-qb-ge | release | main | main`, success).
- `impl-of-model=default` because `gemma4_31b_autoport` is not in tt-shield's
  dropdown; the P300X2 spec sets `default_impl: true`. Confirmed working:
  `determine-server-type` resolved on the first dispatch.
- `release` is the lane the nightly cron actually dispatches (see above), so it
  is the one that must pass. `benchmarks` is still useful as a fast smoke of the
  serving path, since it skips the eval half. Do **not** use `spec_tests`: no
  Gemma variant has spec-test suites, so it fails with `error=no_blocks`
  (see below).
- Omit `docker-image` — the autoport must be baked in, so the pinned public
  image (`0.18.0-c49bb76-6b4a3a7`) cannot be reused: `c49bb76` predates
  `models/autoports/google_gemma_4_31b`.

## What had to change, and why

Registration, in `tt-metal`:

- `models/autoports/vllm_bundles/gemma4_31b_autoport/vllm_metadata.json`.
  The TT vLLM plugin registers bundles under `EXTRA_MODELS_DIR` ahead of its
  built-in map, so the autoport is selected with **no patch to
  `tenstorrent/vllm`**. Without it, `Gemma4ForConditionalGeneration` resolves to
  `models.demos.gemma4.tt.generator_vllm`: the server starts cleanly and serves
  the wrong implementation, which Stage 11 rules invalid even when `run.py`
  exits 0.
- `tt/generator_vllm.py` now anchors a relative `GEMMA4_31B_AUTOPORT_DIR` to
  `TT_METAL_HOME` instead of `os.getcwd()`. TTI launches the API server from
  `vllm-tt-metal/src`, so the previous `Path.resolve()` produced
  `.../vllm-tt-metal/src/models/autoports/...` and startup died on a missing
  `doc/context_contract.json`.

In `tt-inference-server` (branch `mvasiljevic/fast-models-fast/gemma4-31b`):

- `workflows/model_spec.py`: `gemma4_31b_autoport` impl with `code_path
  models/autoports/google_gemma_4_31b`. A dedicated impl, not a reuse of
  `tt_transformers` or `tt_vllm_plugin`, because the release report records
  `impl.code_path` and that is what makes a Stage 11 report provably about the
  generated implementation.
- `workflows/model_specs/dev/llm.yaml` and `prod/llm.yaml`: P300X2 device spec.
  **The prod entry is the one that matters** — tt-shield sets
  `MODEL_SPECS_ENV` nowhere, so TTI defaults to `prod` and a dev-only entry is
  invisible; the dispatch would reject the model as unknown.
- `vllm-tt-metal/src/run_vllm_api_server.py`: sets `EXTRA_MODELS_DIR` from
  `TT_METAL_HOME` in `register_tt_models()`, beside the existing
  `TT_LLAMA_TEXT_VER`/`TT_QWEN3_TEXT_VER` selection. This entrypoint serves both
  the container and `run_local_server.py`, so it is the single correct home.
- `.github/workflows/models-ci-config.json`: nightly on P300X2, as a separate
  entry from `gemma-4-31B-it`. Distinct `weights` ids, so the two coexist and
  either can be run alone.

Build pins in the prod spec: `tt_metal_commit e4297d3bc2f` (this branch's head;
the validated `c49bb76` cannot contain the autoport) and `vllm_commit 6b4a3a7`
(the same validated pin the `-it` entry uses, verified to already carry the
`EXTRA_MODELS_DIR` mechanism, so nothing diverges from the validated pair on the
vLLM side).

## Verified before dispatching

All on this host, against pristine upstream vLLM `dev` `bf98d55`:

| Check | Evidence |
| --- | --- |
| Bundle registers the autoport | `Registered TT model TTGemma4ForConditionalGeneration -> models.autoports.google_gemma_4_31b.tt.generator_vllm:Gemma4ForCausalLM (from EXTRA_MODELS_DIR/gemma4_31b_autoport)` |
| Entrypoint sets the path | `Set EXTRA_MODELS_DIR to <tt-metal>/models/autoports/vllm_bundles` |
| Autoport is the class in use | `Prefix caching is not supported ... for models.autoports.google_gemma_4_31b.tt.generator_vllm`; KV sizing reports the autoport generator |
| Spec resolves under `prod` | `P300X2 / gemma4_31b_autoport / models/autoports/google_gemma_4_31b`, both pins carried |
| `run.py` accepts the model | `--help` lists `gemma-4-31B` beside `gemma-4-31B-it` |
| End-to-end serving via TTI `run.py --workflow server --local-server` | `health=200`; `"The capital of France is"` → `" a city that needs no introduction. Paris is one of the"` |
| Test suites | 101 tt-inference-server tests, 15 in the entrypoint module |

## Open, and expected to surface in CI

- **The image build is unexercised.** Everything above ran against a local
  checkout, never a Shield-built container.
- **`HF_TOKEN` and weights on the runner.** The base checkpoint is gated, and
  the `server` workflow requires `HF_TOKEN` (`run.py::handle_secrets`); the
  runner must supply it and reach the weights.
- **`on-pr-spec-sync.yml`** ("Validate Models Spec") may want a tt-shield-side
  spec entry as well.
- **No release lane.** `models-ci-config.json` registers nightly only; a
  `"release": {"devices": ["P300X2"]}` block is a separate opt-in.
- ~~**Unexplained concurrency figure.**~~ **RESOLVED 2026-08-17 — expected, not a
  defect.** Both the Shield run and a local `--workflow release` run report
  `GPU KV cache size: 103,872 tokens` and `Maximum concurrency for 113,280 tokens
  per request: 1.00x`, with `max_num_batched_tokens=113280` matching
  `max_model_len`, so the earlier guess that TTI derives
  `max_num_batched_tokens` from `max_context` is wrong.

  The figure is a **per-cache-group** number. `get_max_tokens_all_users` sizes the
  pool as five 10-layer sliding groups plus one 10-layer global group:

  ```text
  sliding_blocks/group = ceil(113280/64) + 1        = 1771
  global_blocks        = ceil(113280/128)           =  885
  required_pool_blocks = 5*1771 + 885               = 9740   (623,360 tokens)
  per group            = 9740 // 6 = 1623 blocks    = 103,872 tokens   <-- printed
  concurrency          = 103,872 / 113,280          = 0.917  -> "1.00x"
  ```

  So the pool is deliberately sized for **exactly one** full-context request, and
  `~1.00x` is the correct reading. Note this means `103,872 < 113,280` is *not* a
  capacity shortfall: it compares a per-group token count against the full context
  length, which is not apples-to-apples for a hybrid cache. It also confirms the
  spec's `max_concurrency: 32` is a short-context figure, as the audit says.

  The one-off `8.62x` from an early direct `api_server` launch remains
  unexplained, but it is a single outlier against two reproducible runs and should
  not be relied on.
- **CORRECTION (2026-08-19): the `MODEL_SPECS_ENV nowhere` claim above does not
  hold for the new dispatch path.** `tt-agentic-bringup-qb2` ->
  `tt-shield/on-dispatch.yml@vvukoman/enable-on-dispatch-cross-repo-trigger` sets
  `MODEL_SPECS_ENV: dev`, so TTI loads `workflows/model_specs/dev/llm.yaml` and a
  **prod-only entry is the invisible one**. Dev specs must also omit
  `tt_metal_commit`/`vllm_commit`/`version`. See `agentic_bringup_ci_dispatch.md`.
- **Branch age.** This branch is ~1,900 commits behind tt-metal main, and 13 of
  those commits touch `models/demos/gemma4` modules the autoport's serving path
  imports (`attention/operations.py`, `attention/decode.py`, `layer.py`,
  `attention/kv_cache.py`). Rebasing is deferred until the integration is green,
  and would require re-measuring the perf and PCC evidence.
- **Stage 11's accuracy gate is untouched** and unreachable from this host; see
  `tti_release/STAGE11_PREREQUISITES_p300x2.md`. TTI's `device: GPU`
  bring-your-own-server path is the plausible route to the canonical reference.
