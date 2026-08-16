# TTI release — run notes

Model: `meta-models/Muse-Glimmer-30B`
Autoport under evaluation: `models/autoports/meta_models_muse_glimmer_30b`
Stage input: the completed optimized-vLLM stage (`doc/optimized_vllm/`, tt-metal
`7db0eca646f`, its shipped configuration — prefill tracing off).

**Release-readiness status: `release-readiness-pass`** — the unrestricted release
suite ran, every eval and benchmark row passed, and the single failing API
parameter-conformance row is `issue-waived` with row-specific evidence that
vLLM's own float32 reference implementation fails the same assertion more often
than this autoport does. `run.py --workflow release` exited **1** solely because
of that row; see *Release readiness* below for the exact classification.

## Topology

| what | value |
|---|---|
| server mode | **external autoport vLLM server**; TTI ran as a pure HTTP client. No `--docker-server`, no `--local-server`. |
| Docker | **not used**, and not needed — the external-server path worked first time. No `tt-inference-server` image was pulled, built or referenced, and no container was created. |
| host / context | `tt-quietbox`. This deployment has no separate IRD reservation container: the tt-metal checkout, `tt-smi`, the devices and the TTI client all live on this one host, so there is no "physical loudbox host" to fall back to and `$tt-device-usage`'s reservation-container rules were applied here directly. |
| device | 4-die Blackhole **P300_X2**, mesh `(1, 4)`, `FABRIC_1D_RING`, 2 links. TTI device name `p300x2`. |
| server session | `tmux` session `tti-release-muse-glimmer-30b` (stopped at end of stage) |
| service port | 8000 |

## Versions

| component | value |
|---|---|
| tt-metal | branch `agentic-research/hous/muse-glimmer-30b`, stage input `7db0eca646ff58d033e5515ba4ae0aa3dc662a35` |
| tt-inference-server | `82777a2382e20e7411ae3bd93e1e4b7e8521268c` = `v0.10.0-1099-g82777a238`, `VERSION` 0.20.0 |
| TTI checkout | `/home/ttuser/dev/muse-glimmer/tti-release/muse-glimmer-30b/tt-inference-server`, cloned from the local mirror `/home/ttuser/.local/lib/tt-inference-server` (no network clone needed) |
| tenstorrent/tt-inference-server#4345 (`6e396b4`) | **present** — `git merge-base --is-ancestor 6e396b4 HEAD` succeeds. The runtime-spec-vs-built-in-registry validation bug `$tti-release` warns about does not apply to this checkout, and was not hit. |
| Docker image | none |
| vLLM | the tt-metal venv's `vllm` 0.24.0 + `vllm-tt-plugin` (server side). TTI's client venvs are separate and only send HTTP. |

CLI spellings are this checkout's own, from its `run.py --help`: `--tt-device`
(not `--device`) and `--runtime-model-spec-json` (not `--model-spec-json`).

## Autoport implementation check

**Target path: `models/autoports/meta_models_muse_glimmer_30b`. Confirmed.**

| evidence | value |
|---|---|
| `run_spec/runtime_model_spec_2026-08-16_02-39-22_*.json` → `runtime_model_spec.impl` | `{"impl_id": "muse_glimmer_30b_autoport", "impl_name": "muse-glimmer-30b-autoport", "repo_url": "https://github.com/tenstorrent/tt-metal", "code_path": "models/autoports/meta_models_muse_glimmer_30b"}` |
| same file, `code_link` | `https://github.com/tenstorrent/tt-metal/tree/7db0eca/models/autoports/meta_models_muse_glimmer_30b` |
| `report/report_*_2026-08-16_04-04-24.md` → metadata `model_impl` | `muse-glimmer-30b-autoport` |
| stock implementations in the copied run spec | `models/tt_transformers`, `models/demos` and `tt_vllm_plugin` are **absent** (string search over the whole JSON). The one non-autoport string in the file is `"docker_image": "ghcr.io/tenstorrent/tt-media-inference-server:0.20.0-7db0eca"`, which TTI synthesises for every catalog entry from `version` + `tt_metal_commit`. **No Docker was used and no image was pulled** — the field is unread on the external-server path. `impl.code_path`, `code_link` and `model_impl` all name the autoport. |
| stock implementations in the copied report markdown | `tt_transformers`, `models/demos`: **absent** |
| server-side import proof | `logs/server_excerpt.log`: `models.autoports.meta_models_muse_glimmer_30b.tt.generator_vllm:initialize_vllm_model` builds the generator and `...tt.model:from_pretrained` builds all 52 layers |

The spec is *derived* from the catalog entry rather than hand-written
(`bench/export_runtime_spec.py`), and that script asserts both
`impl.code_path == models/autoports/meta_models_muse_glimmer_30b` and
`max_context == doc/context_contract.json:current_supported_context` before it
writes anything, so a drifted spec cannot reach `run.py`.

## Serving configuration

Byte-identical to the optimized-vLLM stage's shipped arm apart from one addition
(the reasoning parser, below). `bench/serve_release.sh`:

```bash
python -m models.common.readiness_check.run_vllm_server \
  --stages serve \
  --model-dir models/autoports/meta_models_muse_glimmer_30b \
  --hf-model meta-models/Muse-Glimmer-30B \
  --mesh-device P300x2 \
  --max-num-seqs 32 \
  --max-model-len 131072 \
  --port 8000 --server-timeout 2400 \
  --additional-server-args="--reasoning-parser-plugin <repo>/models/autoports/meta_models_muse_glimmer_30b/tt/reasoning_parser.py --reasoning-parser muse_glimmer" \
  --tt-config '{"trace_region_size": 400000000, "fabric_config": "FABRIC_1D_RING",
                "fabric_packet_payload_bytes": 8192, "l1_small_size": 6144,
                "trace_mode": "decode_only"}'
```

Env that mattered: `TT_METAL_HOME` / `PYTHONPATH` = the tt-metal checkout;
`MUSE_GLIMMER_VLLM_PREFILL_TRACE=0` (the optimized-vLLM stage's shipped default,
set explicitly so the arm's identity is inside its own log);
`HF_HOME=/home/ttuser/.cache/huggingface`; `CACHE_ROOT` and
`PERSISTENT_VOLUME_ROOT` under the TTI work root. `HF_TOKEN` is read from
`$HF_HOME/token` by `bench/run_tti.sh` and is never echoed, logged or copied.

`max_model_len` is **131072**, read at launch from `doc/context_contract.json`'s
`current_supported_context`. Nothing in this stage caps context, request length,
benchmark ISL, or eval `max_length`.

## Commands

```bash
# 1. server (tmux tti-release-muse-glimmer-30b)
bash models/autoports/meta_models_muse_glimmer_30b/doc/tti_release/bench/serve_release.sh

# 2. the $qualitative-check shared suite
bash models/autoports/meta_models_muse_glimmer_30b/doc/tti_release/bench/qualitative.sh

# 3. no-Docker smoke: tiny benchmark, trace capture disabled
bash models/autoports/meta_models_muse_glimmer_30b/doc/tti_release/bench/run_tti.sh smoke

# 4. 1%-sampled evals: proves the lm-eval chat path before committing to the full suite
bash models/autoports/meta_models_muse_glimmer_30b/doc/tti_release/bench/run_tti.sh evalsmoke

# 5. the release workflow (evals + benchmarks + spec_tests), unrestricted
bash models/autoports/meta_models_muse_glimmer_30b/doc/tti_release/bench/run_tti.sh release

# 6. copy the small artifacts back
bash models/autoports/meta_models_muse_glimmer_30b/doc/tti_release/bench/copy_back.sh
```

Step 5 resolves to:

```bash
cd /home/ttuser/dev/muse-glimmer/tti-release/muse-glimmer-30b/tt-inference-server
python3 run.py \
  --model Muse-Glimmer-30B \
  --runtime-model-spec-json <work_root>/specs/muse_glimmer_30b_autoport_release.json \
  --tt-device p300x2 \
  --workflow release \
  --service-port 8000 \
  --no-auth \
  --skip-system-sw-validation
```

`--skip-system-sw-validation` is used because device health is validated
directly on this host with `tt-smi` (see *Hardware*).

### Embedded `cli_args`

`bench/export_runtime_spec.py` writes them into the spec before the run, so the
loaded JSON is already right rather than depending on command-line flags to
override it. The run spec TTI wrote back confirms both the spec's `cli_args` and
its own `runtime_config` agree:

```
workflow=release  docker_server=false  local_server=false  service_port=8000
device=p300x2     server_url=null      limit_samples_mode=null  no_auth=true
```

(In this checkout `run.py::populate_model_spec_cli_args` back-fills `cli_args`
from its RuntimeConfig, so CLI flags do apply to a loaded JSON — the two are set
to agree rather than one silently winning.)

## Eval sampling: unrestricted, not `ci-nightly`

No `--limit-samples-mode` was passed. Both tasks ran their full sets —
`ifeval` 541/541, `aime25` 30/30 — so every accuracy number here is **full-set**,
not a CI subset.

Projected runtime from the optimized-vLLM stage's measured serving throughput
(43.4 t/s/u single-user, ~700–1200 tok/s aggregate at concurrency 32):
`ifeval` ≈ 13 min, `aime25` ≈ 35 min, benchmark sweep ≈ 24 min, spec tests ≈ 6 min.
Measured, from the graded run's log: `ifeval` 18.1 + `aime25` 37.6 + sweep 23.3 +
spec tests 6.2 = **85.0 min** (02:39:23 -> 04:04:24 local). `ifeval` is longer
than the earlier 13.8 min because its generation budget was raised, which is the
cost of the fix below. Well inside the window, so the
`ci-nightly` exception `$tti-release` allows was not needed and was not used.

## Results

### Accuracy (full set)

| task | samples | score | reference | tolerance | ratio | check |
|---|---|---|---|---|---|---|
| `ifeval` | 541/541 | **94.45** | 77.0 | 0.05 | 1.227 | ✅ PASS |
| `aime25` | 30/30 | **90.00** | 94.7 | 0.10 | 0.950 | ✅ PASS |

`ifeval` is graded on **`prompt_level_strict_acc` alone** — `score_task_single_key`
takes `result_keys[0]` — which is the strictest of IFEval's four metrics: a
prompt counts only if *every* instruction in it is satisfied under the strict
parser. (The eval config listed two keys in the first two runs, which read as if
a mean were intended; it now lists one, so the config states what it does.)
`aime25` is `exact_match` = 27/30.

Reproducibility across the four release runs this stage made, on the **graded**
metric — the three harness configurations are not interchangeable, so they are
labelled rather than presented as one series:

| run | harness | `ifeval` `prompt_level_strict_acc` | `aime25` `exact_match` |
|---|---|---|---|
| 1 | pre-fix parser (`content=None` on truncation), ifeval 8192 | 94.09 (6 empty responses) | 86.67 (`aime25` at 32768, 4 empty) |
| 2 | shipped parser, ifeval 8192 | 95.38 (3 turns graded on analysis) | 90.0 |
| 3 | shipped parser + seeding fix, ifeval 8192 | 94.64 (3 turns graded on analysis) | 90.0 |
| 4 | **graded run** — ifeval 32768 | **94.45** (1 turn graded on analysis) | **90.0** |
The `aime25` bar is 85.23 at the configured tolerance and 89.965 at the default
0.05 — the reported 90.0 clears **both**, so the loosened tolerance is not what
makes this run pass; it is there for run-to-run robustness, and it is what would
have covered the 86.67 run.

**Residual measurement caveat on `ifeval`, measured not assumed.** A turn that
exhausts `max_gen_toks` inside the model's analysis channel has no reply in it,
and the reasoning parser returns it unsplit, so the harness grades reasoning. At
the first budget (8192) that happened on **3 of 541** prompts. Raising the budget
to 32768 — the same fix `aime25` got, for the same reason — reduced it to **1 of
541** (`evals/ifeval_sample_health.json`, doc_id 162, key 1880). That one is not
a budget problem: the prompt is *"write a short article about the morphology of
the Ukrainian language, 200 words or less; make sure the letter c appears at
least 60 times"*, and the model's analysis channel degenerates into a literal
run of `c` characters, 261,038 of them, until the cap. It scores `False`, which
is also what a run of `c`s would have scored had it been emitted as the reply.

So the graded 94.45 is a **floor**, understated by at most 1/541 = 0.18 points,
and the direction is conservative. `evals/ifeval_sample_health.json` and
`evals/aime25_sample_health.json` record per document the response length, the
score and whether the turn reached the visible channel, so this is checkable from
the evidence tree rather than only from the uncopied `samples_*.jsonl`.
**Follow-up:** the degenerate-output gate scans qualitative artifacts, not eval
samples; pointing it at the eval sample health files would catch this class
automatically.

**Why these two tasks and not `meta_ifeval` / `meta_gpqa_cot`.** Those two are
Llama-family-only: llama-cookbook's `prepare_meta_eval.py` builds their datasets
from `<hf_model_repo>-evals`, and `meta-models/Muse-Glimmer-30B-evals` does not
exist — the HF API returns 404 for it with a valid token while
`meta-llama/Llama-3.1-8B-Instruct-evals` returns 200. Their prompts are also
pre-rendered in Llama 3's chat format and fed with `apply_chat_template=False`,
so pointing them at this checkpoint's tokenizer would violate
`$qualitative-check`'s prompt-format rule. `ifeval` and `aime25` are the
model-appropriate equivalents of the same two gates (instruction following;
zero-shot chain-of-thought reasoning), and are what every non-Llama entry in
`reference_config/evals/eval_config.py` uses.

**`gpqa_diamond_cot_zeroshot` was the first choice for the reasoning gate and
could not run**: every lm-eval GPQA task reads `Idavidrein/gpqa`, a gated Hub
dataset. This host's HF account has not accepted its terms, so the task fails at
dataset download with `DatasetNotFoundError` before a single request reaches the
model (`logs/gpqa_dataset_gated.log`). Accepting a third-party dataset licence on
the user's Hugging Face account is not something this stage may do on its own, so
the reasoning gate uses `aime25` (`math-ai/aime25`, ungated), which the model card
also publishes a number for. **Follow-up: grant the account GPQA access and add
the row.**

**On the reference scores.** No Tenstorrent GPU control run exists for this
checkpoint, so `gpu_reference_score` is the vendor-published score from the model
card in both rows, and each row records why the substitution is sound:

* `ifeval` ← IFBench 77.0. IFBench is a strictly harder instruction-following
  benchmark than lm-eval's IFEval (it was built because IFEval saturated), so a
  model at 77.0 on IFBench is expected to be at or above that on IFEval. This is
  a conservative floor, not an equivalence claim. Measured 94.45 clears it by
  23 %.
* `aime25` ← AIME 2026 94.7 (High Reasoning; the server runs the chat template's
  default reasoning strength, which is `high`). Tolerance is 0.10 rather than the
  default 0.05 for two reasons that are both about the reference: the published
  figure is a different contest year, and a 30-problem set at one sample each
  only moves in 3.33-point steps, so 0.05 would put the bar at 27/30 and a single
  unlucky sample at the model card's own `temperature=1.0` recipe would fail it.

These are **vendor-published references, not measured GPU references**, and the
report labels them as such in the `gpu_reference_score_ref` column.

### Benchmarks

Perf-reference point, ISL 128 / OSL 128 / concurrency 1 / 8 requests:

| metric | measured | functional target | check |
|---|---|---|---|
| mean TTFT | **72.1 ms** | ≤ 88.3 ms | ✅ PASS |
| decode t/s/u | **43.4** | ≥ 11.33 | ✅ PASS |
| aggregate decode tput | **42.7 tok/s** | ≥ 11.33 | ✅ PASS |

Model status is `FUNCTIONAL`, so the `functional` tier is the enforced one; the
`complete` (50 %) and `target` (100 %) tiers are computed and reported and both
fail, which is expected for a first bring-up and is what those tiers are for.
**`EXPERIMENTAL` was deliberately not used**: at that status TTI masks every
performance tier *and* eval accuracy to informational, which would have made this
release report unable to fail at all.

The targets come from a new entry in
`reference_config/benchmarking/benchmark_targets/model_performance_reference.json`.
The `theoretical` figures are the DRAM-bandwidth roofline carried from
`doc/optimized_full_model/perf_summary.json`: per-device weight traffic
4,520,382,464 B ÷ 512 GB/s = **8.829 ms**, i.e. `ttft_ms 8.83` for a 128-token
prefill (which reads the weights once) and `tput_user 113.3` = 1000/8.829 for
decode. This is the memory-side bound only — no compute term — so it is an
*optimistic* bound and the tiers derived from it are correspondingly strict, not
lenient.

Full 18-point sweep, ISL 127 → 65535 at concurrency 1 and 16–32:
**18/18 points completed, 0 failed requests**, up to 65535-token prompts. Those
rows are ungraded (no targets configured for them) and are reported for
information.

### API parameter conformance (`spec_tests`)

**21 of 22 passed.** The suite did not run at all in the first release attempt —
`test_module.dispatch` logged *"No spec test suites match model='Muse-Glimmer-30B'
device='p300x2' — skipping spec_tests"* — so the model was registered in the TTI
checkout's `test_module/server_tests_config.json` and `test_module/test_suites/llm.json`
to make it run. Skipping it silently would have left the release with no API
conformance coverage at all.

The one failure and its classification are in *Release readiness*.

### This is a text-only release of a multimodal checkpoint

`meta-models/Muse-Glimmer-30B` is `MuseGlimmerForConditionalGeneration`: its HF
config carries a `vision_config`, an `image_token_id` and a `video_token_id`, and
the model card advertises interleaved text-and-image input through a ~1.8B
ViT-G/14 perception encoder. **The autoport implements the text stack only. The
vision tower is not ported by any stage of this bring-up**, and
`doc/functional_decoder/README.md` recorded it as out of scope at the time.

The release spec therefore sets `supported_modalities: ["text"]`, which is what
stops TTI generating the image benchmark sweep. Image and video input are
**unsupported** in this release; every number here is a text-only number. This is
the largest capability caveat of the released artifact and is repeated in
`README.md`'s *Limitations*.

### `$qualitative-check`

Prompt-format decision is recorded in
`qualitative/prompt_format_tti_release.json` and
`qualitative/qualitative_prompt_format.json`:

* the checkpoint ships a non-empty `chat_template.jinja`, so it is chat/instruct,
  and **every release check renders through it**. No raw-completion output is
  used as a quality verdict anywhere in this stage.
* release evals use lm-eval's `local-chat-completions` against
  `/v1/chat/completions` with `--apply_chat_template`; the vLLM server renders
  the messages with the checkpoint's own template. Nothing here re-implements or
  overrides it. A rendered example and its token ids are recorded.
* the shared suite's verdict arm posts the **pinned token ids** the full-model
  stage rendered, so this stage, the previous serving stage, the standalone TT
  model and the HF control all ran the identical input.

Results: chat arm coherent on all 6 prompts, `replacement_char_fraction` 0.0000
everywhere, worst adjacent-duplication 0.0 against a critical threshold of 0.10;
`qualitative_vllm_vs_datatype_sweep_chat.json` reports `first_divergence: 2` on
all six prompts, which is the OpenAI API stripping the `<|message|>` token the
standalone text carries — the same artifact the optimized-vLLM stage
characterised, not a new divergence. Shared degenerate-output gate over both arms:
`No degenerate output detected`, exit 0.

### Context contract and non-aligned prompt lengths

| check | result |
|---|---|
| served `max_model_len` vs `doc/context_contract.json` | **131072 = 131072**, no reduction |
| run spec `max_context` | 131072 |
| benchmark sweep reach | ISL 65535 (the sweep's 131072 point is filtered out by TTI because 131072 + 128 > max_context — a mathematically invalid request, the only kind the skill allows rejecting) |
| eval `max_length` | 131072 on both tasks |
| non-aligned prompt lengths, dedicated probe | **9/9 pass** (`non_aligned_probe.json`: 56, 56, 127, 129, 1023, 2049, 4097, 8193, 12345 rendered prefill tokens; none divides the 32-token tile, the 64-token page block or the 8192-token prefill chunk) |
| non-aligned prompt lengths, as TTI actually sent them | **18/18 sweep points**, every one at an odd ISL (127, 1023, 2047, 4095, 8191, 16383, 32767, 65535), 0 failed requests |

No request was aligned, shortened or waived to make anything pass.

## Two bugs found and fixed by this stage

Both were found *because* the TTI conformance suite was made to run, and both are
in code shared across autoports rather than in this model's own files.

### 1. Concurrent seeded decode was not reproducible (`models/common/sampling/tt_sampling.py`)

`test_non_uniform_seeding` fires 32 concurrent requests, 16 of them with
`seed=0`, and requires those 16 responses to be byte-identical. One or two
diverged on every run.

Root cause: `ttnn.manual_seed` installs the per-token RNG state as a **register
on each core**, which `ttnn.sampling`'s compute kernel then advances — nothing
carries that state in a tensor. The shared sampler called `manual_seed` and then
ran `_adjust_values_for_tiebreak` (~17 elementwise ops) before `ttnn.sampling`.
That chain contains `ttnn.typecast(int32 → bfloat16)`, which destroys the RNG
state on the cores it runs on, so the users mapped to those cores drew a
different random number from an identical seed. On this mesh the affected batch
slots were exactly `{0, 11, 22}`, and a request diverged exactly when a `seed=0`
request landed on one of them.

Fix: move the `ttnn.manual_seed` call so it is the last op before
`ttnn.sampling`. One file, one moved call, plus a comment recording why.

Evidence and refutations in `AUTOFIX_seeding.md` and `seeding/evidence.json`,
including an op-level bisect (`manual_seed`+`sampling` alone: clean over 40
seeds; with the tie-break chain between them: 19 of 20 seeds disagree; the first
breaking op is index 13, `typecast int32→bfloat16`; `exp` also breaks it, while
`add`, `abs`, `max`, `eq`, `typecast→int32` and `untilize` do not).

**This also resolves the limitation both earlier stages recorded.**
`doc/vllm_integration/README.md` *Limitations 1* and
`doc/optimized_vllm/README.md` *Sampling suite* describe "seeded reproducibility
at batch > 1" as a known limitation with 6–7 failing plugin tests. With the fix,
`test_seeding_and_variety.py` + `test_request_isolation.py` run **29 passed**.
Those two sibling documents now overstate the limitation; this stage supersedes
them rather than editing another stage's committed evidence.

**Residual risk, recorded not fixed:** the kernel behaviour is *avoided*, not
fixed. Any op inserted between the seed call and `ttnn.sampling` will silently
re-break seeded reproducibility with no error. `ttnn.manual_seed` has no way to
express "this state must survive". Worth raising upstream; a unit test that puts
a `typecast → bfloat16` between the two ops and asserts 32 equally-seeded users
agree would catch a regression at op level.

### 2. The OpenAI API returned the model's analysis channel as its answer

Muse Glimmer is a *channelled* model: its chat template ends the assistant prompt
at `<|start|>assistant` and the model writes its own channel header, so a turn is
`to=self` (analysis) then `to=user` (the reply). With no reasoning parser
configured, vLLM returns both concatenated as `choices[].message.content` and
leaves `reasoning_content` null. Every eval harness then reads the analysis
channel as part of the answer — for an instruction-following eval that is fatal,
and it is also non-conformant for a reasoning model.

Fix: `tt/reasoning_parser.py`, registered with
`--reasoning-parser-plugin ... --reasoning-parser muse_glimmer`. It is API-layer
text routing only — same sampling, same generator, same tokens on device. The
control is in `smoke/reasoning_control_unparsed.json` vs `smoke/reasoning_parsed.json`:
the same greedy request produces **the identical 618 completion tokens** either
way, and `reasoning_content + content` reconstructs the unparsed string exactly,
minus the two channel headers. In the unparsed arm the response violates the
prompt's "all lowercase" instruction; in the parsed arm `content` obeys it.

The parser **never removes information**: a turn cut off inside the analysis
channel (`max_tokens` exhausted, or a `stop` string matching inside the analysis)
is returned *unsplit*, exactly as an unparsed server would return it, so
`content` is a string for every response this server can produce. Returning
`content=None` there — which is what vLLM's `<think>`-style parsers do — broke
four conformance rows with `TypeError: argument of type 'NoneType' is not
iterable` before this was fixed. 13 host-only unit tests in
`tests/test_reasoning_parser.py` pin the behaviour, including that the coherence
guard's own sentence survives a truncated turn.

**Scoped limitation:** streaming cannot offer the same guarantee — a delta has to
be labelled when it is emitted, and whether a visible channel ever arrives is not
known until the turn ends — so a truncated turn streams as reasoning deltas with
no content, which is what every other vLLM reasoning parser does. Every eval,
benchmark and conformance path in this release runs non-streaming.

**Also noted:** the server logs
`Auto-initialization of reasoning token IDs failed. Please check whether your
reasoning parser has implemented reasoning_start_str and reasoning_end_str.`
That flag (`reasoning_config.enabled`) gates exactly one feature in this vLLM —
the per-request `thinking_token_budget` sampling parameter
(`vllm/v1/engine/input_processor.py:105`) — and nothing else reads it. No release
path uses `thinking_token_budget`. Classified: feature unavailable, no
correctness impact.

## Release readiness

`run.py --workflow release` exited **1**. Everything that TTI grades passed
except one parametrization of one conformance test:

| row | status | classification |
|---|---|---|
| `ifeval` accuracy (94.45, a floor — see above) | ✅ PASS | — |
| `aime25` accuracy | ✅ PASS | — |
| benchmark target, functional tier (TTFT / tput_user / tput) | ✅ PASS | — |
| benchmark target, complete + target tiers | ❌ FAIL | informational at `FUNCTIONAL` status by design; this is a first bring-up at 38 % of the memory-side roofline |
| 18 benchmark sweep points | ungraded | 18/18 completed, 0 failed requests |
| `spec_tests` — 21 of 22 conformance parametrizations | ✅ PASS | — |
| `spec_tests` — `test_penalties[presence_penalty-1.2-repeat_trap]` | ❌ FAIL | **issue-waived**, below |

### `test_penalties[presence_penalty-1.2-repeat_trap]` — issue-waived

The test sends "Write a very repetitive story." twice at `temperature=0.1,
max_tokens=1024, seed=1234`, once with `presence_penalty=1.2`, and asserts
`unique_ratio(penalty) >= unique_ratio(base) * 0.90` where
`unique_ratio = len(set(words))/len(words)`. Measured, deterministically, 3/3
identical texts on both arms: base 252 words / 38 unique / ratio 0.1508,
penalised 202 words / 25 unique / ratio 0.1238, ratio **0.8207** → FAIL.

The waiver rests on measurement, not on disclosure. Full workings in
`AUTOFIX_presence_penalty.md` and `presence_penalty/`:

1. **The device computes vLLM's rule, on vLLM's token set.** Rebuilding
   `argmax(raw_logprob − 1.2·[token already generated])` from the host sampler's
   pre-penalty logprobs reproduces vLLM's own emitted token at **256/256** steps.
   Scored against the *device's* greedy tokens over the 160 steps to and
   including the divergence: presence-in-bf16 **160/160**; presence-in-fp32
   159/160; by count (frequency's rule) 137/160, first contradiction at step 30;
   prompt ∪ output (repetition's set) 158/160, first contradiction at step 88;
   no penalty at all 145/160, first contradiction at step 9. Every wrong-rule
   candidate is refuted by hundreds of tokens.
2. **The one greedy device-vs-host divergence is bf16 quantisation, in closed
   form.** Without a penalty the two paths are **byte-identical over 1172
   characters**. With `presence_penalty=1.2` they agree to character 725 and then
   differ: `"Probably"` (already generated) sits at logit `L`, `"Perhaps"` (never
   generated) at exactly `L − 1.25`; `bf16(1.2) = 1.203125` and the logits are on
   the 0.125 bf16 grid, so `bf16(L − 1.203125) = L − 1.25` — an exact tie,
   resolved by the sampler's lowest-global-id tiebreak.
3. **Falsifiable prediction, zero falsifications.** If that is the mechanism, a
   penalty that is exact on the grid must make the two paths identical:
   **0.5, 1.25 and 2.0 are byte-identical over 1024 greedy tokens, 3 of 3**,
   while unaligned 1.2 and 1.1 diverge. A wrong formula cannot be cured by
   choosing a rounder penalty.
4. **The assertion is not a property of the penalty implementation.** Running the
   row's own comparison against **vLLM's own float32 host sampler** — zero
   Tenstorrent code in the sampling path — the reference fails the same assertion
   **more often than the device does**: 1 pass / 4 for the reference against
   2 pass / 4 for the device, and in the greedy trial where there is no RNG at
   all the reference scores 0.3585 (FAIL) against the device's 0.9725 (PASS).
   The test file itself already exempts `presence_penalty` from its
   repetition-reduction assertion on this exact prompt (line 313), which is the
   same asymmetry showing up in its own authors' hands.

Classification: **`issue-waived`** — the correct canonical implementation fails
the same row in the same way, and worse, for reasons unrelated to this autoport.
There is no upstream issue URL because filing one is outside this stage's
authority; the waiver rests on (4) above, which is a control against the
reference implementation rather than an appeal to "it's only a heuristic".
`frequency_penalty` and `repetition_penalty` pass on all three of the suite's
prompts, including this one — the failure is specific to presence.

**Follow-up recorded, not fixed:** the device applies the three penalty terms as
presence → frequency → repetition; vLLM applies repetition → frequency →
presence. Unobservable in this suite (frequency 0 and repetition 1.0 are
bit-exact identities) but it will matter if anyone combines
`repetition_penalty != 1.0` with a presence or frequency penalty in one request.
`models/common/sampling/tt_penalties.py` has no test file today; a torch-reference
unit test for it is the obvious next step.

## Local edits to the tt-inference-server checkout

The TTI clone is scratch (outside the tt-metal repo, not committed). Its full
diff is committed here as
`tti_local_edits/tt_inference_server_local_edits.patch` (366 lines):

| file | edit | why |
|---|---|---|
| `workflows/model_spec.py` | add the `muse_glimmer_30b_autoport` `ImplSpec` + registry entry | pins `impl.code_path` to the generated autoport instead of stock `models/tt_transformers` |
| `workflows/model_specs/prod/llm.yaml` | add the `meta-models/Muse-Glimmer-30B` P300X2 template | `EVAL_CONFIGS` is built by iterating `MODEL_SPECS`, so a runtime spec the catalog has never heard of gets **no eval tasks at all**. Also carries `max_tokens_all_users_override: 1050624`, the KV pool the autoport actually allocates — without it the pool is inferred as `max_context` and benchmark concurrency is understated 8× |
| `reference_config/evals/eval_config.py` | add the `ifeval` + `aime25` `EvalConfig` | the eval recipe, with the reasoning above recorded inline |
| `reference_config/benchmarking/benchmark_targets/model_performance_reference.json` | add the P300X2 perf targets | appended textually, not via a JSON round-trip: the file has three duplicate keys (`p300x2`, `Mistral-7B-Instruct-v0.3`, `Llama-3.3-70B-Instruct`) that `json.load` would silently collapse, deleting entries |
| `test_module/server_tests_config.json`, `test_module/test_suites/llm.json` | register the model for the vLLM parameter-conformance suite | without it the release skips API conformance entirely |

No TTI test was edited, relaxed or skipped.

## Two things that look like gaps and are not

* **`sample_on_device_mode: all` appears in the run spec's `override_tt_config`
  but not in `bench/serve_release.sh`'s `--tt-config` string.** That is not
  drift: `run_vllm_server` injects it, and the launched command in
  `logs/server_excerpt.log` is
  `--additional-config '{"tt": {"sample_on_device_mode": "all", "trace_region_size": 400000000, "fabric_config": "FABRIC_1D_RING", "fabric_packet_payload_bytes": 8192, "l1_small_size": 6144, "trace_mode": "decode_only"}}'`,
  with `TTModelRunner: trace_mode=decode_only, sample_on_device_mode=all` on the
  next line. The spec and the server that was actually validated agree exactly.
* **`qualitative/qualitative_prompt_format.json` self-labels
  `"stage": "vllm_integration"`.** That string is baked into the shared runner
  (`doc/vllm_integration/bench/qualitative_vllm.py`), which this stage re-ran
  rather than copied — the file's mtime is inside this stage's window and its
  outputs are this stage's server's. `qualitative/prompt_format_tti_release.json`
  is the stage-11 prompt-format record and covers the paths the shared runner
  does not (the release evals and the conformance suite).

## Hardware

`timeout 60 tt-smi -ls --local` at stage start: 4 Blackhole `p300c` boards, all
present, no leftover `vllm`/`EngineCore` processes, `/dev/tenstorrent/{0,1,2,3}`
free.

**No resets, no hangs, no ARC / ERISC / remote-Ethernet events, and no
`tt-triage` capture was needed at any point in this stage.** Devices were
serialized one job at a time throughout; the only device-facing job was the
autoport vLLM server, and TTI, the evals and the benchmarks were all HTTP clients
of it. The server was stopped and relaunched four times (adding the reasoning
parser, taking the parser fix, and twice by the `$autofix` subagents); each time
the launcher and `VLLM::EngineCore` were killed, `/dev/tenstorrent/*` was
confirmed free, and the next launch opened the mesh normally.

## Cleanup

* No autoport vLLM server left running; `tmux` session `tti-release-muse-glimmer-30b`
  killed.
* No `tt-inference-server` Docker container was ever created.
* `run.py` left an empty (0-byte) `.env` in the TTI checkout; removed after
  copy-back. It was never copied into the evidence tree.
* Raw `server.log` files (60–140 MB each) are deleted; `logs/server_excerpt.log`
  and `server/server_log_size.txt` are committed instead, and
  `doc/tti_release/server/server*.log` is re-ignored in the autoport `.gitignore`.
* Per-token `itls` and per-request `generated_texts` arrays were trimmed out of
  the copied benchmark JSON (6.8 MB → 0.1 MB); every metric is kept and the drop
  is recorded in each file under `_trimmed_for_evidence`.
* Not copied: `.env`, the Hugging Face cache, the persistent volume, weights,
  tensor dumps, profiler CSVs, and the per-sample eval dumps.
* The IRD reservation was not released (no monitor asked).

## Commit

Stage-owned changes are committed locally and **not pushed**, per the bringup
contract.

| repo | branch | commit |
|---|---|---|
| `tt-metal` | `agentic-research/hous/muse-glimmer-30b` | `ec69581f5d2a28d2eb8a3bf3c90be3e5ccc2a1ab` |

105 files: `doc/tti_release/` (2.8 MB of evidence; raw `server.log` re-ignored,
`logs/server_excerpt.log` committed instead), the new
`tt/reasoning_parser.py` + `tests/test_reasoning_parser.py`, the shared seeding
fix in `models/common/sampling/tt_sampling.py`, and `.gitignore`. No unrelated
dirty state was swept in; the worktree is clean at this SHA.

The `tt-inference-server` clone is scratch, outside this repo and not committed;
its full local diff is committed here as
`tti_local_edits/tt_inference_server_local_edits.patch`.

Two pre-commit hooks acted on the staged tree and are worth noting because they
touched evidence: `trailing-whitespace`/`end-of-file-fixer` normalised committed
`*.log` and `*.json` artifacts (cosmetic; no number or token changed — the run
spec, the eval health files and all 18 benchmark JSONs were re-checked after),
and `black`/`isort`/`autoflake` reformatted the stage's Python. The 13
reasoning-parser tests were re-run after the reformat and pass.

## Report path

```
models/autoports/meta_models_muse_glimmer_30b/doc/tti_release/report/report_id_muse-glimmer-30b-autoport_Muse-Glimmer-30B_p300x2_2026-08-16_04-04-24.md
```
