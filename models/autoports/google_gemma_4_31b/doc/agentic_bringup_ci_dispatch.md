# Dispatching the autoport through tt-agentic-bringup-qb2

Date: 2026-08-19 UTC. Host: `qb2-120-p04t03` (BH QuietBox 2, 2x p300c, 4 chips,
11x10 grid) — a *different* QB2 node from the `p02t03` used for all prior work.

Onboarding the autoport into the new agentic-bringup dispatch path, deliberately
**without** pre-adjusting the harness so that CI reports the model's real
failures instead of hiding them. This document records the mechanism, the
decisions, and the corrections to earlier docs.

## The dispatch path

`tenstorrent/tt-agentic-bringup-qb2` (created 2026-08-18) holds one workflow,
`tt-shield-dispatch.yml`: a `workflow_dispatch` front-end that calls
`tt-shield/.github/workflows/on-dispatch.yml@vvukoman/enable-on-dispatch-cross-repo-trigger`
with `permissions: write-all` and `secrets: inherit`. It exists so bring-up work
can drive hardware CI without write access to tt-shield. The cross-repo trigger
it calls is itself still on an unmerged tt-shield branch.

```bash
gh workflow run tt-shield-dispatch.yml --repo tenstorrent/tt-agentic-bringup-qb2 \
  -f model=gemma-4-31B -f runner-label=bh-qb-ge -f device-type=p300x2 \
  -f workflow=benchmarks -f impl-of-model=default \
  -f tt-metal-git-ref=mvasiljevic/fast-models-fast/gemma4-31b \
  -f inference-server-git-ref=mvasiljevic/gemma4-31b-autoport-on-release-flow \
  -f vllm-git-ref=main
```

The image build runs on a generic cloud runner (`tt-ubuntu-2204-large-*`); only
`run-tests` occupies `bh-qb-ge`, which resolved to `120-qb2-p03t02`. A 1-4 h
build therefore does **not** hold the hardware.

### Exact refs under test (resolved by `resolve-shas` at dispatch)

Run [32245795692](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/32245795692),
`benchmarks`, dispatched 2026-08-19 11:04 UTC, hardware job on `qb2-120-p03t06`:

| Repo | Ref requested | Resolved commit |
| --- | --- | --- |
| `tenstorrent/tt-metal` | `mvasiljevic/fast-models-fast/gemma4-31b` | `dbab2c270141fea8c4abcee95990d5754c1d296e` |
| `tenstorrent/tt-inference-server` | `mvasiljevic/gemma4-31b-autoport-on-release-flow` | `09fc4fd64b6de7c30978ce1fb54078cd0e9a5ede` |
| `tenstorrent/vllm-tt-plugin` | `main` | `bd150c7e9d7526e181bfc25dc4379c65f2ba5371` |

The tt-metal commit is code-identical to `c54dca6b8bf`, the state that produced
the green local `rc=0` release run of 2026-08-18 (the four commits between are
documentation only). `resolve-shas` pins these at dispatch time, so a later push
to any of the three branches does not affect a run already in flight.

**Why it failed:** the benchmark sweep measured 0 of 17 points. Every point died
on `vllm bench serve`'s pre-flight probe with HTTP 400, because the driver posts
to `/v1/chat/completions` and the base checkpoint's tokenizer defines no chat
template (`ChatTemplateResolutionError`). Model resolution, image build, server
bring-up and autoport registration all succeeded. Detail below.

## Correction: the dispatch reads the *dev* catalog, not prod

`shield_ci_onboarding.md` states that tt-shield sets `MODEL_SPECS_ENV` nowhere,
so TTI loads `prod` and a dev-only entry is invisible. That was true of the older
dispatch path. This caller sets it explicitly:

```text
MODEL_SPECS_ENV: dev
ERROR: Unable to determine server type: No default impl for gemma-4-31B on p300x2
```

(run 32242168349, `determine-server-type`, with the entry present in `prod` only).
Two consequences: the entry must live in `dev/llm.yaml`, and dev specs must
**omit** `tt_metal_commit`/`vllm_commit`/`version` — `ProdModelSpecTemplate`
requires them, the base `ModelSpecTemplate` rejects them.

## Why a tt-inference-server change is irreducible

`get_runtime_model_spec` (`workflows/model_spec.py:1352-1380`) filters
`MODEL_SPECS` by `model_name` and `device_type` and then needs a `default_impl`.
No dispatch input injects a spec or env vars. So a model that is not declared in
the catalog cannot be dispatched at all — there is no zero-change path.

Branch: `mvasiljevic/gemma4-31b-autoport-on-release-flow`, one commit on top of
`vvukoman/add-8-models-to-release-flow` (`60f80c4b`). **32 code lines, 2 files,
0 deletions**: the `dev/llm.yaml` entry plus a 7-line `ImplSpec` and one registry
line, following the `muse_glimmer` precedent of one impl per autoport with
`code_path` under `models/autoports/`.

## What was deliberately NOT set

Counting other users of each setting across the 58 templates in `dev/llm.yaml`:

| Setting | Other entries using it | Kept? |
| --- | ---: | --- |
| `chat-template` | 0 | no |
| `TT_METAL_OPERATION_TIMEOUT_SECONDS` | 0 | no |
| `async-scheduling` | 0 | no |
| `enable_model_warmup` | 0 | no |
| `trace_mode` | 1 (DeepSeek-R1-0528) | no |
| `override_generation_config` | 1 (AFM-4.5B) | no |
| `EXTRA_MODELS_DIR` | 0 | **yes** |
| `sample_on_device_mode` | 23 | yes |
| `fabric_config` | 24 | yes |
| `MESH_DEVICE` | 13 | yes |
| `trace_region_size` | 46 | yes |

Five of the six dropped settings were used by no other model in the catalog. Each
existed because a specific failure was already known, which means each one
converted a CI finding into an invisible workaround and made a green run assert
much less than it appeared to.

`EXTRA_MODELS_DIR` is kept because it is not of that kind: it does not change how
the model behaves, it decides *which code runs*. Without it the arch resolves to
`models/demos/gemma4` and the run reports on a different implementation. The
plugin registers bundles under it (`platform.py:476-545`, feature added in
`vllm-tt-plugin` `3d978e4e`, 2026-07-22) ahead of its built-in map.

Note the autoport is the only catalog entry using `EXTRA_MODELS_DIR`; every peer
alternative-implementation model uses a `TT_*_TEXT_VER` selector instead
(all three Qwen3.6/3.8 entries). Gemma4 has no such selector — `platform.py:707`
hardcodes `_gemma4_target = "models.demos.gemma4.tt.generator_vllm:Gemma4ForCausalLM"`.
Adding `TT_GEMMA4_TEXT_VER` upstream (~10 lines, mirroring the llama block at
`platform.py:583`) is the idiomatic end state. It is the same one-line plugin gap
`muse_glimmer` is blocked on, which is documented in its own spec entry as
"⚠️ NOT RUNNABLE ... registered on NO vllm-tt-plugin branch (all 30 checked)".

## What `EXTRA_MODELS_DIR` actually points at, and why it is in the spec

The whole directory is one 138-byte file plus a README. No code:

```json
// models/autoports/vllm_bundles/gemma4_31b_autoport/vllm_metadata.json
{
  "arch": "Gemma4ForConditionalGeneration",
  "main_class": "models.autoports.google_gemma_4_31b.tt.generator_vllm:Gemma4ForCausalLM"
}
```

It is a single arch -> class mapping. The adapter itself lives in
`tt/generator_vllm.py:48`; the bundle only points at it by dotted path, which the
plugin resolves lazily against `PYTHONPATH` (TTI puts `TT_METAL_HOME` first).

**This mapping's natural home is the plugin repo, not here.** `platform.py:707`
already hardcodes the competing one:

```python
_gemma4_target = "models.demos.gemma4.tt.generator_vllm:Gemma4ForCausalLM"
```

registered under all six `Gemma4*` aliases. Every peer alternative implementation
got its mapping into the plugin as a `TT_*_TEXT_VER` selector (all three Qwen
entries). Gemma4 has no selector, so the choices were a plugin PR or a bundle.
`gh api repos/tenstorrent/vllm-tt-plugin` reports `push: false, maintain: false,
admin: false` for this account, so the plugin route needs someone else's merge.

Two facts keep the bundle from being a workaround:

- `EXTRA_MODELS_DIR` was added for exactly this case -- plugin commit `3d978e4e`
  (2026-07-22), "[feature] Register models dynamically from 'EXTRA_MODELS_DIR'",
  docstring: *"Any distribution tool can drop a bundle folder here and have it
  registered with no source edit to this plugin."*
- `muse_glimmer` is the counterfactual: it took the plugin route and its spec
  entry records ⚠️ NOT RUNNABLE, "registered on NO vllm-tt-plugin branch (all 30
  checked)". Without a bundle an autoport simply cannot serve.

The `../../tt-metal/...` prefix is also not invented: 15 paths in `dev/llm.yaml`
already use it (e.g. `TT_MESH_GRAPH_DESC_PATH`), because the server runs with cwd
`vllm-tt-metal/src` and the container places `tt-metal` as a sibling.

**End state.** Upstream `TT_GEMMA4_TEXT_VER` to the plugin (~10 lines mirroring
the llama block at `platform.py:583`, defaulting to `demos` so upstream behaviour
is unchanged). The spec line then becomes
`TT_GEMMA4_TEXT_VER: gemma4_31b_autoport`, consistent with the Qwen entries, and
the same PR unblocks `muse_glimmer`. Until that merges, the bundle is the only
route that does not depend on a repo this account cannot push.

## Expected failures, and where each one belongs

### 1. `ChatTemplateResolutionError` — harness defect, not fixable in tt-metal

`llm_module/drivers/vllm.py:58-64` hardcodes `--backend openai-chat --endpoint
/v1/chat/completions` for every model, with no conditional.

Both Gemma 4 31B checkpoints have `chat_template: None` in
`tokenizer_config.json`; the **-it** checkpoint additionally ships a separate
`chat_template.jinja` (the current HF convention), and the **base** checkpoint
ships neither. So `-it` resolves a template from the checkpoint and the base
model has nothing to resolve.

This cannot be fixed from tt-metal: the serving tokenizer is built in the API
server process from the checkpoint path, outside any TT code, and the bundle
schema carries only `arch` and `main_class`. The three possible sources are the
checkpoint (Google's, and deliberately absent for a base model), vLLM's
`--chat-template` flag (i.e. the spec — what the flag is for), or the driver.

The driver is the real gap, and the harness already models the distinction on the
eval side: `EvalTask` has `use_chat_api` and `apply_chat_template`, and
`meta_ifeval`/`meta_gpqa_cot` set `apply_chat_template=False`. The benchmark
driver has no equivalent switch, so **any** base checkpoint in the catalog hits
this.

### 2. Non-greedy request kills EngineCore — genuine model defect

`_require_semantic_greedy` raises at both prefill (`tt/generator_vllm.py:297`)
and decode (`:359`). `vllm bench serve` no longer sends `temperature==0`, so the
server-side default decides. The adapter advertises
`sample_on_device_policy: "greedy_only"` expecting the plugin to route non-greedy
requests to host sampling — a hook that was never implemented. Advertising a
contract nobody honours and then hard-failing is ours to fix: either implement
the fallback or stop advertising it.

### 3. 5 s metal-op watchdog on the cold first compile

TTI's `run_vllm_api_server.set_metal_timeout_env_vars` sets
`TT_METAL_OPERATION_TIMEOUT_SECONDS=5.0`. Shield run 31824560569 died in
`multichip_decoder`'s prefill `ttnn.linear` with a completely cold JIT cache
(0/636 hits, engine init 73.48 s against 3.80 s warm). Not specific to this
model; `tt-media-server` already ships 120 in its own sample env.

## Related, unfixed rigidity in the adapter

`tt/generator_vllm.py:125` raises unless `max_model_len` **exactly equals** the
`context_contract.json` value — not "at most". The KV-pool sizing immediately
below (`:167-169`) *derives* from `max_model_len`, so other values would compute
fine. The refusal is a policy assert with no mechanical necessity, and it is why
`max_context: 113280` is mandatory rather than tunable. Relaxing it to `<=` would
not change behaviour at the validated value.

## Mistakes made while doing this

| Mistake | Consequence | Correction |
| --- | --- | --- |
| Amended with `git commit --amend -m` (no `-a`) after `git checkout <base> -- dev/llm.yaml` had reset **index and** working tree, then editing only the working tree | The pushed commit carried `model_spec.py` but **not** the spec entry, so CI saw an empty catalog for this model and failed `determine-server-type` a second time (run 32245014913) | Verify the **committed tree**, not the working tree: `git diff --stat <base> HEAD` and `git show HEAD:<path>`. A pushed branch must also be checked from the remote (`gh api .../contents/<path>?ref=<branch>`) |
| Validated the fix with a local script run against the working tree | Passed locally while the commit was broken -- the repro read files the commit did not contain | Reproduce from a pristine `git worktree add --detach <path> HEAD` with zero modified files |
| Read author identity from branch commit metadata instead of the GitHub account | Attributed the work to the wrong first name, and authored commits under it | `gh api user --jq .name`. Note the tt-metal branch's own 34 commits carry the same wrong name from a misconfigured repo-local `user.name` on `p02t03` |
| Dispatched a run seconds before being asked not to | Run 32242168349 started; it self-terminated at `determine-server-type` before any build, so no hardware or build time was consumed | Confirm before dispatching to shared CI, and check for in-flight instructions before firing |

## Misleading error message in tt-shield

`determine_server_type.py:128` raises `No default impl for <model> on <device>`
from the `impl == "default"` branch, which is reached when
`len(matching_specs) != 1`. That includes **zero** matches, so the message does
not distinguish:

- the model is absent from the catalog entirely (both of our failures), from
- the model is present but no `device_model_spec` sets `default_impl: true`.

Reproduce tt-shield's exact resolution locally before dispatching -- it costs
seconds and needs no hardware:

```bash
gh api "repos/tenstorrent/tt-shield/contents/.github/scripts/determine_server_type.py?ref=main" \
  --jq .content | base64 -d > /tmp/dst.py
mkdir -p /tmp/repro && cd /tmp/repro
git -C <tti-checkout> worktree add -q --detach tt-inference-server <commit>
MODEL_SPECS_ENV=dev python /tmp/dst.py gemma-4-31B default p300x2   # -> tt-inference-server
```

Beware a stray `platform.py` in the working directory: the script inserts the
checkout on `sys.path`, and a shadowing module surfaces as
`partially initialized module 'platform' has no attribute 'system'`.

## Outcome of the unmodified run (32245795692)

Furthest the autoport has reached on this path. Everything up to the request layer
worked:

| Stage | Result |
| --- | --- |
| `determine-server-type` | success -- resolved to impl `gemma4-31b-autoport` |
| `build-tt-inference-server` | success, ~25 min (warm layer cache) |
| server bring-up on `qb2-120-p03t06` | success -- `healthy at http://127.0.0.1:8000/health` |
| implementation actually served | **the autoport** -- `Prefix caching is not supported in TT backend for models.autoports.google_gemma_4_31b...` |
| benchmark sweep | **0/17 points measured** |

The registration line is the important one: the 138-byte bundle plus
`EXTRA_MODELS_DIR` worked inside a Shield-built container on a QB2 node that had
never run this model. The run is demonstrably testing the autoport and not
`models/demos/gemma4`.

### The failure

Server-side, per request:

```text
Failed to load AutoTokenizer chat template for google/gemma-4-31B
ChatTemplateResolutionError: As of transformers v4.44, default chat template is
no longer allowed, so you must provide a chat template if the tokenizer does not
define one.
```

Client-side, per sweep point:

```text
ValueError: Initial test run failed - Please make sure benchmark arguments are
correctly specified. Error: Bad Request
[ERROR] llm_module.runner: vllm exited 1 on sweep point N/17
```

`exited 1` is the `vllm bench serve` child process's exit code
(`llm_module/runner.py:137`), not a test verdict. The failure is on the
**pre-flight probe**, so no measurement was attempted at all -- there is no TTFT
or throughput data, and this is distinct from defect #1 in
`defects_found_by_release_flow.md` where 8 requests "succeeded" with 0 tokens.
The server stayed healthy throughout (`Called check_health` immediately after each
failure), which is why all 17 points could attempt in sequence.

Note the asymmetry in diagnosability: the client only reports `Bad Request`, whose
message blames the benchmark arguments. The real cause appears only in the server
log.

### Why no other model hits it

Checked against the real checkpoints on this host:

| Checkpoint | Chat template |
| --- | --- |
| `google/gemma-4-31B` (this model) | **none anywhere** |
| `google/gemma-4-31B-it` | `chat_template.jinja` (18,683 bytes, Google's canonical format) |
| `Qwen3.8-2.4T-A95B` | `tokenizer_config.json` **and** `chat_template.jinja` |
| `Llama-3.1-8B-Instruct` | `tokenizer_config.json` |

The catalog is overwhelmingly instruct models, and even nominally base
checkpoints from Meta and Qwen ship a template anyway. Google ships one only on
`-it`. This is the first entry with no chat format at all, so it is the first to
expose a hardcoded chat endpoint -- a latent bug that needed a base model to find.

### Fix pushed for review

tt-inference-server branch `fix/benchmark-completions-endpoint-for-base-models`
(`llm_module/drivers/vllm.py`, +3 tests): pick the endpoint from the tokenizer's
chat capability rather than hardcoding chat. Chat models are unaffected; base
checkpoints get `/v1/completions`, which is also semantically correct for them,
because wrapping a completion model in an invented chat template changes the
prompt being measured. Detection mirrors the check
`utils.prompt_generation.template_prompt` already makes, is `local_files_only`
(no network egress -- proven by running the tests under `HF_HUB_OFFLINE=1`),
`lru_cache`d across the 17 points, and fails open to the chat endpoint so unknown
capability preserves today's behaviour. Verified: `google/gemma-4-31B` -> False,
`google/gemma-4-31B-it` -> True.

The rejected alternative is a `chat-template` line in the spec pointing at this
branch's 124-byte passthrough template
(`doc/vllm_integration/chat_template.jinja`, `bos_token` + newline-joined
content). It works today and needs no code change, but it is a per-model
workaround that changes the measured prompt and leaves the next base model to
rediscover the bug.

### Corrections to the predictions above

- **The 5 s metal-op watchdog did not fire.** Zero `TT_THROW`, zero
  `device timeout`, zero `fetch queue` hits; the only timeouts in the log are
  GitHub's `timeout-minutes: 1080` and the harness's own 1200 s health wait. The
  cold compile completed inside the default on this node, so
  `TT_METAL_OPERATION_TIMEOUT_SECONDS: 120` was not needed and should stay out.
  The Shield failure it was added for (31824560569) did not reproduce here.
- **The greedy guard was never exercised.** `_require_semantic_greedy` does not
  appear in the log; requests never reached the model. It remains an untested
  defect, hidden behind the chat-template failure rather than cleared.

### Harness observation: no early abort on a uniform failure

The runner works through all 17 points even though point 1 failed on a
shape-independent pre-flight error, spending ~10 minutes per point (~2.8 h of QB2
time) re-proving the same HTTP 400. Bailing out when the initial test run fails
identically on consecutive points would return the same information in ten
minutes.

## Sampling: the adapter was stricter than the device

Reached by asking why a non-greedy request cannot be served, rather than adding
`override_generation_config` to make CI pass.

### What was wrong

`generator_vllm.py` passed `top_k=1, top_p=0.0, temperature=1.0` to both sampler
call sites and raised `_require_semantic_greedy` on anything else. The generator
underneath already implements non-greedy on-device sampling: `_sample_eager`
branches on `_is_semantic_greedy`, seeds RNG via `_initialize_non_greedy_rng` and
calls the sampler with k/p/temp from `_make_sampling_params`; and
`prepare_token_out_decode` takes the same three values for the traced path. The
decode trace key was even already shaped `("greedy", batch)` with `must_prepare`
firing on key change. So "greedy-only" described the adapter, not the device.

### Two corrections to earlier conclusions in this document

- **top_k wider than the sampler does NOT need host sampling.** The platform
  clamps: `format_sampling_params` maps `k < 1` ("unrestricted", vLLM's default)
  to 32 and caps `k > 32` at 32. A default `vllm bench serve` request is served
  on device as top-32 sampling -- the same approximation every tt_transformers
  model on this platform already makes. An earlier note here claimed it was
  inexpressible; that was wrong.
- **The predicted greedy-guard failure for the next run no longer applies**, since
  a default request now resolves rather than raising.

### A latent numerical bug, found by reusing instead of hand-rolling

`ttnn.sampling`'s compute kernel multiplies the top-k values by `temp`
(`ttnn/cpp/ttnn/operations/reduction/sampling/device/kernels/compute/sampling.cpp:465`,
`mul_block_bcast_scalar_inplace`), so `temp` must be **1/T**.
`format_sampling_params` inverts; the autoport's `_make_sampling_params` performs
no transformation. Any implementation that plumbed a raw temperature through --
including the first version of this change -- would have applied `T` where `1/T`
was required. Greedy concealed it completely, because `T=1` is its own reciprocal.

### Canonical placement

`format_sampling_params` is called **5 times in `models/tt_transformers/tt/generator.py`
and 0 times in `models/tt_transformers/tt/generator_vllm.py`**. The platform's
division of labour is: the adapter forwards `sampling_params`, the generator
normalises. Two earlier attempts put the logic in the adapter before this was
checked; the final version has `Gemma4Generator.resolve_sampling` beside
`_is_semantic_greedy`, with the adapter only forwarding and deriving the trace
label.

Neither neighbour was a template: the sibling `qwen_qwen3_4b` autoport carries the
identical `greedy_only` / `_force_argmax_sampling` limitation, and
`models/demos/gemma4` threads a `greedy_only` constructor flag. `MllamaFor-
ConditionalGeneration` takes the third route, declaring
`supports_sample_on_device: False` and host-sampling.

Placement also decided testability: `generator.py` does not import vLLM, so
`tests/test_sampling_resolution.py` runs on a bare host -- **9 tests, 2.6 s** --
whereas `tests/test_vllm_adapter_contract.py` cannot even be collected without
vLLM installed (`ModuleNotFoundError: No module named 'vllm'`).

Still open: heterogeneous per-slot parameters raise, because
`_make_sampling_params` and `prepare_token_out_decode` take scalars; wiring them
means accepting sequences. The adapter also still advertises
`sample_on_device_policy: "greedy_only"`, now imprecise, left alone because the
plugin hook it names does not exist.

## Run 32271862302: bring-up exceeded the health budget

Refs: tt-metal `mvasiljevic/gemma4-31b-nongreedy-sampling` @
`589f30d44c291a92556f198a7960f6a36ac6183a`, tt-inference-server
`mvasiljevic/gemma4-31b-autoport-on-release-flow` @
`5352535a9a28b40fa96248976c3dc088804e8200`, vllm-tt-plugin `main` @
`bd150c7e9d7526e181bfc25dc4379c65f2ba5371`. Hardware job on `120-qb2-p03t02`.

Resolution, image build (~60 min, full rebuild), mesh open, fabric init and
autoport registration all succeeded. The server never became healthy.

### The clock

`llm_module/runner.py:64` sets `wait_healthy_timeout_s = 1200.0`.

| Run | Health wait starts | Healthy | Elapsed |
| --- | --- | --- | --- |
| 32245795692 | 12:35:16 | 12:52:16 | **17m 00s** (3 min spare) |
| 32271862302 | 16:47:18 | never | timed out at **20m 00s** |

The difference is where in-container setup falls relative to the clock. In the
healthy run the container started at 12:26:46 and the health wait only began at
12:35:16 -- setup happened before the clock. In the failed run the wait began
16:47:18, five seconds after container start, and the engine's first log line was
16:55:25, so **8m 12s of in-container setup was inside the budget**, leaving
~11m 45s for model load against the ~17 min it needs.

The engine reached `TTModelRunner: trace_mode=all, sample_on_device_mode=all,
enable_model_warmup=True` at 16:55:32 and logged nothing further until
`Got Keyboard Interrupt` at 17:07:18. No traceback, no `TT_THROW`, no watchdog
trip, no OOM in the full server log.

**On whether it was loading or blocked -- the logs cannot settle it.** State the
evidence precisely:

- *No positive evidence of progress.* After `model_runner.py:164` at 16:55:32 the
  failed run logs nothing at all until `Got Keyboard Interrupt`. Zero lines.
- *Silence is consistent with a normal load.* The healthy run behaves identically:
  engine logs at 12:35:34, nothing until 12:52:04 (16m 30s), healthy 12 s later.
  So a silent gap of this length is what a successful load looks like. That makes
  an incomplete-but-progressing load plausible -- interrupted 11m 46s into a
  ~16m 30s job, roughly 71% through, about 4m 45s short -- but plausible is not
  proven.
- *A device-side hang is unlikely.* The failed run ran with
  `TT_METAL_OPERATION_TIMEOUT_SECONDS=5.0` (TTI's default, since the spec override
  was removed) and `TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE` wired. Any
  single device operation blocking for more than 5 s should have aborted with a
  `TIMEOUT` and run the triage hook. It never fired.
- *A host-side block is not excluded.* That watchdog only covers device
  operations. CPU-bound weight conversion, file I/O, an HF fetch or a lock would
  be invisible to it and would look exactly like this.

So: consistent with a load that needed ~5 more minutes, unlikely to be a stuck
device op, and a host-side block cannot be ruled out from these logs. The root
cause of the *failure* is still the time budget, but "it was fine, just slow" is
an inference, not a finding.

The reason this is undecidable is itself the defect: the load phase emits no
progress output for ~16 minutes, so a single run cannot be told apart from a hang.
A progress log in `build_generator` -- even one line per N layers or per cache
write -- would make the next occurrence diagnosable from one run.

### The applied config was as intended

From the run's `runtime_model_specs/*.json` artifact, not inferred: `impl
gemma4_31b_autoport`, ctx 113280, conc 32, env `MESH_DEVICE=P150x4`,
`GEMMA4_PAGE_BLOCK_SIZE=64`, `GEMMA4_31B_AUTOPORT_DIR`, `EXTRA_MODELS_DIR`;
`override_tt_config` = fabric_config/trace_region_size/sample_on_device_mode only;
`vllm_args` carrying `chat-template` and no `override_generation_config`. So the
stripped entry behaved exactly as designed, and the sampling change cannot be
implicated: the engine went quiet during model load, before any request.

### Where to look next time

`gh api repos/.../actions/runs/<id>/artifacts` -> the
`workflow_logs_benchmarks_*` zip holds what the job log does not:

- `docker_server/vllm_*.log` -- the engine's own output, 35 KB
- `run_logs/run_*.log` -- the harness side, with health-wait timestamps
- `runtime_model_specs/*.json` -- the config actually applied

The job log is an extract ("Extracted 43/556 lines"), so diagnosing from it alone
is unreliable. Both earlier diagnoses in this document were made without opening
these artifacts.

### Mistakes

| Mistake | Consequence | Correction |
| --- | --- | --- |
| Quoted "~16 min bring-up" from one run and treated it as headroom | It was 17 min against a 20 min cap, a 15% margin; the next run lost the margin to container setup and timed out | Measure the health-wait window explicitly (`Waiting for inference server` -> `is healthy`) before predicting a run |
| Bundled a documentation commit into the tt-metal branch used for CI | Changed the SHA, forced a 60 min rebuild, and produced a fresh container whose setup fell inside the health window. The only functional change (sampling) cannot affect bring-up | Keep the CI-dispatched ref stable; land docs separately, or dispatch the pre-existing SHA |
| Diagnosed two runs from the truncated job log without downloading artifacts | Floated a cold-tensor-cache theory the evidence did not support | Download `workflow_logs` first; the server log and runtime spec answer most questions directly |

### Consequence for the harness

Bring-up for this model is ~17 min of model load on top of ~8 min of container
setup. Whether it fits depends on which side of the health clock the setup lands,
which is not something the model controls. That is a fourth finding of the same
kind as the chat-endpoint default: `wait_healthy_timeout_s = 1200` is marginal for
a 31B model with a cold cache, and a run can fail for reasons unrelated to the
model under test.

## Run history from this host

| Run | Lane | Refs (tt-metal / TTI) | Outcome |
| --- | --- | --- | --- |
| [32240588764](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/32240588764) | release | branch / `...-minimal` | cancelled in image build, by request |
| [32242168349](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/32242168349) | release | branch / `...-on-release-flow` (prod-only entry) | **failed** `determine-server-type`: `No default impl` — the `MODEL_SPECS_ENV=dev` discovery above |
| [32245014913](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/32245014913) | benchmarks | branch / `...-on-release-flow` @ `0474b2c8` | **failed** `determine-server-type` in 20 s: the commit was missing the spec entry (see mistakes above). No build, no hardware |
| [32245795692](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/32245795692) | benchmarks | branch / `...-on-release-flow` @ `09fc4fd6` | resolution + build + serving all **passed**; benchmark sweep 0/17 measured (chat-template), 15 points failed identically before the run was cancelled. See outcome above |
| [32271862302](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/32271862302) | benchmarks | `...nongreedy-sampling` @ `589f30d` / `...-on-release-flow` @ `5352535a` | **failed**: server not healthy within 1200 s. Registration and mesh init fine; 8m12s of container setup inside the budget left too little for model load. See above |
