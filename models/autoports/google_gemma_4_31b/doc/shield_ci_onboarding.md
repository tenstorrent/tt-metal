# Shield CI onboarding: Gemma 4 31B autoport on QB2 (p300x2)

Date: 2026-08-14 UTC

How this autoport is dispatched through `tenstorrent/tt-shield`, what had to change
to make that possible, and what is verified versus still open.

## First dispatch

```text
run:    https://github.com/tenstorrent/tt-shield/actions/runs/31801765378
name:   gemma-4-31B | bh-qb-ge | spec_tests | mvasiljevic/fast-models-fast/gemma4-31b
                                            | mvasiljevic/fast-models-fast/gemma4-31b
```

Progress at time of writing: `resolve-shas` success, `determine-server-type`
success, `build-tt-inference-server` in progress. Comparable QB2 runs take
1h25m to 3h52m.

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
  -f workflow=spec_tests \
  -f impl-of-model=default \
  -f tt-metal-git-ref=mvasiljevic/fast-models-fast/gemma4-31b \
  -f inference-server-git-ref=mvasiljevic/fast-models-fast/gemma4-31b \
  -f vllm-git-ref=dev
```

- `custom-model` is the documented way to test a model absent from the dropdown
  ("⭐ use for testing new models"); `model` is a required placeholder it
  overrides.
- `bh-qb-ge` is the QB2 runner label, taken from real passing runs rather than
  guessed (`Qwen3-Embedding-4B | bh-qb-ge | release | main | main`, success).
- `impl-of-model=default` because `gemma4_31b_autoport` is not in tt-shield's
  dropdown; the P300X2 spec sets `default_impl: true`. Confirmed working:
  `determine-server-type` resolved on the first dispatch.
- Escalate `workflow` in order: `spec_tests`, `benchmarks`, `evals`, `release`.
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
- **Unexplained concurrency figure.** TTI reported `Maximum concurrency for
  113,280 tokens per request: 1.00x` where a direct `api_server` launch reported
  `8.62x` on an identical 103,872-token KV pool. Probably TTI deriving
  `max_num_batched_tokens` from `max_context`, but it is not confirmed and would
  affect any batching claim.
- **Branch age.** This branch is ~1,900 commits behind tt-metal main, and 13 of
  those commits touch `models/demos/gemma4` modules the autoport's serving path
  imports (`attention/operations.py`, `attention/decode.py`, `layer.py`,
  `attention/kv_cache.py`). Rebasing is deferred until the integration is green,
  and would require re-measuring the perf and PCC evidence.
- **Stage 11's accuracy gate is untouched** and unreachable from this host; see
  `tti_release/STAGE11_PREREQUISITES_p300x2.md`. TTI's `device: GPU`
  bring-your-own-server path is the plausible route to the canonical reference.
