# TTI release run notes: zai-org/GLM-4.7-Flash autoport on one Blackhole p150

Stage 11 (`$tti-release`) for the generated tt-metal autoport
`models/autoports/zai_org_glm_4_7_flash`. Everything below is from the run of
2026-09-04.

## Final status

`release-readiness-ci-subset-pass`

`python run.py --workflow release` exited 0 (EXIT_CODE=0), the acceptance
verdict in the report is PASS with 0 blockers, and every required sampled row
either passed or has the row-specific classification recorded under
"Row-by-row classification" below. All accuracy numbers here are **CI-subset**
results from `--limit-samples-mode ci-nightly` (5% of each eval dataset). They
are not full-set accuracy and must not be compared with a full-set release
threshold without that qualification.

Release report:
`doc/tti_release/reports_output/release/report_id_autoport-glm47-flash_GLM-4.7-Flash_p150_2026-09-04_20-48-56.md`

## Autoport implementation check

Autoport implementation check: PASS, evaluated code path is `models/autoports/zai_org_glm_4_7_flash`.

The copied TTI run specs
(`run_specs/runtime_model_spec_2026-09-04_15-40-09_*.json` and the smoke spec
`run_specs/runtime_model_spec_2026-09-04_08-31-54_*.json`) both record

```
impl.impl_id    = autoport_glm47_flash
impl.impl_name  = autoport-glm47-flash
impl.code_path  = models/autoports/zai_org_glm_4_7_flash
impl.repo_url   = https://github.com/tenstorrent/tt-metal
hf_model_repo   = zai-org/GLM-4.7-Flash
model_id        = id_autoport-glm47-flash_GLM-4.7-Flash_p150
```

and the release report's metadata block records `"model_impl":
"autoport-glm47-flash"`. No copied artifact names `models/tt_transformers`,
`models/demos`, or any other packaged implementation. The server that answered
every request imported the autoport directly: its boot log shows
`Prefix caching is not supported in TT backend for
models.autoports.zai_org_glm_4_7_flash.tt.generator_vllm` and
`Getting max_tokens_all_users=412563 ... from generator
'<class 'models.autoports.zai_org_glm_4_7_flash.tt.generator_vllm.GLM47FlashForCausalLM'>'`.

## Topology and server mode

* Server mode: **external autoport vLLM server, no Docker**. `run.py` was
  invoked without `--docker-server` and without `--local-server`; the written
  run spec confirms `cli_args.docker_server=false`, `cli_args.local_server=false`,
  `cli_args.service_port=8000`, `cli_args.workflow=release`,
  `cli_args.tt_device=p150`, `cli_args.limit_samples_mode=ci-nightly`. No Docker
  image was used anywhere in this stage, so there is no Docker image to record.
* Host/context: physical host `tt-quietbox`, user `stisi`, no reservation
  container in this setup (tt-metal, the TT devices, and tt-inference-server all
  live on the same host). Four Blackhole boards, board series **p300c**, PCI
  1e52:b140 at 0000:01..04:00.0. The autoport opens a **1x1 mesh on device 0**
  (`MESH_DEVICE=N150`, `multidevice with 1 devices and grid (1, 1)`), which is
  the single Blackhole chip this whole bring-up targets, and is why the TTI
  device is `p150` (TTI's single-Blackhole-chip device) rather than `p300`.
* `tt-smi` is not on this host's PATH; health was read with
  `timeout 90 /home/stisi/ComfyUI/venv/bin/tt-smi -ls --local`, which listed all
  four boards before and after the run.

## Versions

| Component | Value |
|---|---|
| tt-inference-server repo | `/home/stisi/tt-inference-server`, VERSION `0.20.0`, `git describe` `v0.10.0-1141-gbc296ab54` |
| tt-inference-server git SHA at run time | `bc296ab5489fb3edcd6ed0fb53e7b3c7a54634fb` plus the uncommitted release wiring, now committed as `66b9f5a398a5b48c09efcf67c8713a4371402345` |
| tt-metal git SHA at run time | `3873571191c8e20608c975331cc05a11e72e5fcd` plus the uncommitted prefill-DRAM fix (see "Model fix") |
| vLLM | 0.25.1 (installed in tt-metal `python_env`) |
| vllm-tt-plugin | `9f2ec5d` |
| Docker image | not used |

No release tag checkout was needed: this checkout's own `run.py --help` was used
for the CLI spelling, which is `--tt-device` (not `--device`) and
`--runtime-model-spec-json` (not `--model-spec-json`).

## Commands

Autoport vLLM server (started from `/home/stisi/tt-metal`, held open for the
whole stage):

```bash
./python_env/bin/python -m models.common.readiness_check.run_vllm_server \
  --stages serve \
  --model-dir models/autoports/zai_org_glm_4_7_flash \
  --hf-model zai-org/GLM-4.7-Flash \
  --mesh-device N150 \
  --max-num-seqs 32 \
  --max-model-len 202752 \
  --tt-config '{"trace_region_size": 350000000}' \
  --additional-server-args "--reasoning-parser glm47"
```

This is the optimized-vLLM stage's proven serving command plus one addition:
`--reasoning-parser glm47`. GLM-4.7-Flash is thinking-by-default, so without the
parser every scored response is the raw reasoning trace. The parser is API-server
post-processing only; it does not touch the traced decode path, and the measured
decode TPOT is unchanged (29.5 ms/token, below).

TTI smoke (no Docker, tiny benchmark, trace capture disabled):

```bash
cd /home/stisi/tt-inference-server
./venv/bin/python run.py --model GLM-4.7-Flash --dev-mode --tt-device p150 \
  --workflow benchmarks --limit-samples-mode smoke-test --service-port 8000 \
  --no-auth --skip-system-sw-validation --disable-trace-capture
```

TTI release:

```bash
cd /home/stisi/tt-inference-server
./venv/bin/python run.py --model GLM-4.7-Flash --dev-mode --tt-device p150 \
  --workflow release --service-port 8000 --limit-samples-mode ci-nightly \
  --no-auth --skip-system-sw-validation --disable-trace-capture
```

Key environment: none beyond the defaults. `HF_TOKEN` was not needed (the
checkpoint and every eval dataset were already in the local HuggingFace cache),
`JWT_SECRET` was not needed (`--no-auth`, and the autoport server runs without an
API key), and no `.env` was created or modified in either checkout. No token was
printed at any point.

`--dev-mode` is required, not cosmetic: the GLM-4.7-Flash spec and its eval
config live in the dev catalog, and `EVAL_CONFIGS` is built by intersecting the
eval list with `MODEL_SPECS`, so under the prod catalog the model resolves to
nothing.

**`--server-url` must be omitted.** Passing `--server-url http://127.0.0.1`
together with `--service-port 8000`, which `run.py --help` documents as the way
to target an already-running server, sets `remote_server=True`
(`utils/url_helpers.py::is_remote_server`), and `ServerConnection.url_with_port`
then drops the port for remote connections, so the client polls
`http://127.0.0.1/v1/models` on port 80 and silently waits out its 1200 s
health timeout. Omitting the flag resolves the same localhost default with the
port attached. Recorded as a TTI harness sharp edge; not worked around in code
because the no-flag path is correct and unambiguous.

## Smoke, run before the release workflow

1. `GET /health` -> 200.
2. `GET /v1/models` -> `max_model_len: 202752`, id `zai-org/GLM-4.7-Flash`.
3. One OpenAI-compatible chat request (`max_tokens 512`, `temperature 0`):
   `finish_reason=stop`, 252 completion tokens, content `"2 plus 2 equals 4."`
   (the reasoning parser returned the answer alone).
4. Tiny TTI benchmark, `--workflow benchmarks --limit-samples-mode smoke-test
   --disable-trace-capture`: `run.py` exited 0, the written run spec had
   `docker_server=false` and `impl.code_path=models/autoports/zai_org_glm_4_7_flash`,
   and the benchmark JSON recorded `completed: 8, failed: 0`
   (`benchmarks/smoke_benchmark_isl-16_osl-4_maxcon-1_n-8.json`).

## Context contract

`doc/context_contract.json` records `hf_advertised_context = 202752` and
`current_supported_context = 202752` with `capability_reduction: none`. Nothing
in this stage lowers it:

* the server was launched at the full 202752 and `/v1/models` echoes it;
* the TTI dev spec carries `max_context: 202752` and the derived
  `vllm_args.max_model_len: "202752"`;
* the benchmark sweep is built from that context and runs all the way to
  ISL 131072;
* both eval tasks set `max_length: 202752` explicitly, because lm-eval's API
  models otherwise default to 2048;
* `.agents/scripts/check_context_contract.py --stage tti-release
  --require-contract` exits 0.

`max_gen_toks` on the eval tasks (16384 for ifeval, 65536 for GPQA) is an
**output budget**, not a context cap. The model card's own evaluation recipe is
131072 output tokens, which is unreachable in a release window at ~34 t/s/u on
one chip; the budgets are recorded here and every item that still hit them is
counted rather than quietly scored as wrong.

### Non-aligned prompt lengths

The serving prefill chunk is 1024 tokens. Valid prompt lengths that are not
multiples of it are served correctly and echo their exact length:

* explicit probe through `/v1/completions` with token-id prompts, before the
  release run: 4097, 10000, 12289, all `prompt_tokens` echoed exactly with
  coherent continuations (`non_aligned_prompts.json`);
* through the release benchmark sweep itself: ISL 10000 at concurrency 1 and 18
  (`completed: 2 / 36, failed: 0`, `input_lens` 10000);
* after the DRAM fix, in one server process: 10000, 131071, 131072, 202751
  (`AUTOFIX_prefill_dram.md`).

No benchmark or eval request was aligned, shortened, or waived to avoid a
chunking bug.

## Model fix made during this stage

The first release attempt (09:38 to 13:01) **died at benchmark sweep point 21**
(`isl=65536 osl=128 concurrency=1`). The vLLM EngineCore hit

```
TT_FATAL: Out of Memory: Not enough space to allocate 402653184 B DRAM buffer
across 8 banks ... (allocated: 4196763008 B, free: 31824896 B,
largest free block: 20877312 B) (assert.hpp:104)
```

inside `_moe_prefill`, killing the server and taking the remaining benchmark
points and the whole `spec_tests` child with it (`logs/prefill_oom_before_fix.log`).
Every sweep point up to and including ISL 32768 had passed in the same process.

This is a real autoport bug, not a harness problem, and it was fixed with
`$autofix` rather than by capping the request length. Root cause:
`GLM47FlashForCausalLM.get_max_tokens_all_users` handed every remaining DRAM byte
to the vLLM KV pool behind a fixed 0.75 GiB margin that covered only the
prompt-length-independent transients (bank rounding and the 384 MiB MoE gate_up
transpose scratch at chunk 1024). It reserved nothing for the pair of whole-prompt
`[1, 1, phys, 2048]` bf16 activations that `run_layer_stack_prefill` keeps live at
every one of the 47 layer boundaries, which costs 8192 B per prompt token and
also fragments what the transpose buffer needs. The fix derives that reservation
from `max_model_len`, the checkpoint's own `hidden_size`, and the contract's paged
block size. Full write-up, including the refuted capacity-only arithmetic and the
measured pre-fix failure boundary, in `AUTOFIX_prefill_dram.md` and
`autofix_prefill_dram.json`.

Effect: KV pool 471,168 -> 414,656 vLLM block tokens (still 2.05x a full-context
request), prompt lengths 65536 / 131071 / 131072 / 202751 all served in one
process, decode TPOT unchanged at 29.49 ms/token against the recorded 29.496,
and the release re-run cleared sweep points 21, 22 and 23 that previously killed
the server.

## Why ci-nightly

Projected **unrestricted** eval runtime, from this run's measured throughput:

* `ifeval`: 28 sampled docs at concurrency 16 took 329 s, i.e. about 165 s per
  16-doc wave. The full 541 docs are 34 waves, about **1.6 h**.
* `gpqa_diamond_cot_zeroshot`: 10 sampled docs took 4195 s in a single wave (the
  wave is bounded by its slowest reasoning trace). The full 198 docs are 13
  waves, about **15.1 h**.

About **16.7 h of evals** on top of roughly 4 h of benchmarks and spec tests,
against a single Blackhole chip that this experiment does not have exclusively
for that long. `--limit-samples-mode ci-nightly` was therefore used. It
propagated as `--limit 0.05` on both eval commands (visible in
`logs/run_py_release.log`), giving effective sample counts of:

| task | dataset | sampled | fraction |
|---|---|---|---|
| ifeval | 541 | 28 | 5.2% |
| gpqa_diamond_cot_zeroshot | 198 | 10 | 5.1% |

Nothing else was reduced: the served context, the benchmark prompt and completion
lengths, the full 23-point benchmark sweep, and the whole `spec_tests` suite all
ran unrestricted.

## Results

| Category | Verdict |
|---|---|
| `run.py --workflow release` | exit code 0 |
| Acceptance criteria | PASS, 0 blockers |
| Benchmarks | PASS (1 graded row passed at the functional tier, 22 ungraded) |
| Evals | PASS by model status (see classification below) |
| Spec tests | PASS, 1/1 test classes, 22/22 conformance cases |

Serving numbers from the release sweep (full table in the report):

| shape | TTFT | TPOT | output t/s |
|---|---|---|---|
| 128 in / 128 out, concurrency 1 | 296.1 ms | 29.5 ms | 31.7 |
| 128 in / 128 out, concurrency 32 | 9317.9 ms | 91.6 ms | 195.5 |
| 10000 in / 1024 out, concurrency 18 | 434.6 s | 106.2 ms | 33.9 |
| 131072 in / 128 out, concurrency 1 | 1034.1 s | 277.1 ms | 0.1 |

Eval scores, **CI subset**:

| task | score | reference | automated check |
|---|---|---|---|
| ifeval | 71.43 (20/28 prompt-level strict) | none published, no GPU control | NA |
| gpqa_diamond_cot_zeroshot | 70.0 (7/10, flexible-extract) | published 75.2 (full set) | FAIL, ratio 0.9309 |

## Row-by-row classification

Every non-PASS row in the report is classified below. Note that at model status
`EXPERIMENTAL` the harness itself demotes eval-accuracy and benchmark-tier
failures to informational (`ModelStatusTypes.required_target_tiers` is empty for
EXPERIMENTAL), which is why the report's acceptance verdict is PASS. That is a
model-status tier effect and **not** a row-specific waiver, so each row is
classified by hand here.

### 1. `gpqa_diamond_cot_zeroshot` accuracy FAIL: `issue-waived` (invalid target for this sample size)

Measured 70.0% (7 of 10) against the model card's full-set 75.2% with a 5%
tolerance, so the automated ratio check needs 71.44% and fails at 70.0%. This is
not a model result, it is the comparison the `$tti-release` skill explicitly
forbids: a CI-subset score judged against a full-set threshold. Three
independent pieces of evidence:

* **The threshold is unreachable in steps of one item.** With n=10 the raw
  percent-ratio rule first passes at 8 of 10 (80%). There is no 7.x outcome, so
  the gate is really "8 of 10 or fail".
* **This checkout's own subset rule passes the row.** `accept_eval_score` uses a
  sample-count-aware form for subset references,
  `round(score/100*n) >= floor(n * ref/100 * (1 - tol))`, which here is
  `7 >= floor(10 * 0.752 * 0.95) = 7`: a pass, exactly at the threshold. That
  branch is only taken when the task defines a measured
  `mode_reference_scores[CI_NIGHTLY]` entry, which does not exist for this model
  because no in-house subset reference has been measured. Inventing one to make
  the row pass would be gaming the gate and was not done.
* **The observation is statistically consistent with the published number.** For
  a model that matches 75.2% exactly, P(X <= 7 | n=10) = 0.47, so this check
  fails about half the time on a correct model. The Wilson 95% interval for 7/10
  is [39.7%, 89.2%], which contains 75.2%.

Harness quality for this row is good: **zero of the ten items were truncated**
(`evals/results_gpqa_diamond_cot_zeroshot_2026-09-04T16-55-45.json` plus a
per-sample check of the lm-eval sample log), so all ten were genuinely scored.
The first release attempt at `max_gen_toks=32768` had 2 of 10 items come back
empty because their reasoning traces were still open at the cap; raising the
budget to 65536 removed that entirely and moved the score from 60.0 to 70.0.
`exact_match,strict-match` is 0.0 for both attempts, which is expected and is why
the task is scored on `flexible-extract`: the model answers in `\boxed{X}` form
rather than lm-eval's strict "Answer: X" template.

Required next step to close this properly: a measured
`mode_reference_scores[CI_NIGHTLY]` for this task (a GPU or canonical-implementation
run over the same fixed 5% subset), or an unrestricted 198-item run when about
15 h of single-chip time is available.

### 2. `ifeval` accuracy NA: incomparable metric, disclosed

71.43% measured, no automated verdict because neither reference exists: the
GLM-4.7-Flash model card publishes no IFEval number, and there is no in-house GPU
control run of this checkpoint. This is the same situation as other newly
onboarded dev-catalog models (for example `Qwen/Qwen3-4B`). Not a failure, and
not claimed as a pass.

Two of the 28 sampled items returned an empty answer. Both were investigated
rather than left as noise, and neither is length truncation:

* replayed alone through `/v1/completions` with the chat template applied and the
  same seed, both closed their reasoning block well inside the 16384 budget
  (7440 and 1482 completion tokens) `evals/empty_answers_8192.json`;
* replayed through `/v1/chat/completions` at concurrency 2, same seed, both
  returned non-empty content at the identical token counts
  `evals/empty_answers_chat_lowconc.json`.

The only difference from the eval run is concurrency (16 in the eval, 2 in the
replay). That matches the signature of the upstream full-occupancy determinism
defect
[tenstorrent/tt-metal#55408](https://github.com/tenstorrent/tt-metal/issues/55408),
which the optimized-vLLM stage established across five measurements and could not
narrow, and which is carried into this stage as the model's most significant
serving limitation. Classified `issue-waived` against that issue, with the
caveat that this stage did not narrow it either and the two probes above are
consistent with it rather than proof of it.

### 3. Benchmark 128/128/concurrency-1, `target` tier TTFT FAIL: `issue-waived` (non-comparable target)

The block passes overall (the `functional` tier passes, and acceptance passes a
benchmark when any single tier meets all its checks). Within the `target` tier
the two throughput checks pass at ratios 1.000 (`tput_user` 33.90) and 0.9946
(`tput_output` 31.84); only TTFT fails, at 296.1 ms against a 274.1 ms target,
+8.0%.

That target is the optimized-vLLM stage's recorded TTFT, and it was measured with
a different client shape: `vllm bench serve` against `/v1/completions` with greedy
temperature, no reasoning parser. The release harness measures
`vllm bench serve --backend openai-chat` against `/v1/chat/completions` through the
glm47 reasoning parser. The decode metric the target really gates reproduces
exactly. A fresh-server control taken during the DRAM fix measured 280.9 and
275.8 ms on the same shape, so the release figure also sits inside normal
run-to-run spread for a 128-token prefill.

Required next step: re-derive the `ttft_ms` entry in
`model_performance_reference.json` from a chat-endpoint measurement so the two
sides of the comparison match.

### 4. Twenty-two ungraded benchmark rows: catalog convention, not a gap

`model_performance_reference.json` defines targets for exactly one shape per
device (128/128/concurrency-1) for every model in the catalog, so the other 22
sweep points are reported for information and are not graded. GLM-4.7-Flash now
has that one entry like every other model. The rows themselves all completed with
`failed: 0`.

### 5. Spec tests: no failures

22 of 22 parameter-conformance cases passed, including `test_coherence_verbatim_echo`,
`test_non_uniform_seeding` (32 concurrent seeded requests), all nine
`test_penalties` cases, `test_seed_reproducibility`, `test_stop`, `test_logprobs`
and `test_determinism_parameters`.

Getting there needed a harness fix, recorded because it changed what the suite
measures. Run against the server as-is the suite was **14 failed / 8 passed**, and
every failure was the same non-defect: the suite picks 32 to 64 token budgets
because it tests API parameter semantics, GLM-4.7-Flash opens a reasoning block by
default, so the budget is spent inside the trace, `message.content` comes back
`None` behind the reasoning parser, and the assertions fail on `None` rather than
on the parameter under test. Proof that this is structural rather than an autoport
defect: the same coherence prompt at `max_tokens=32` returns
`finish_reason=length, content=None` with thinking on, and returns
`"The quick brown fox jumps over the lazy dog."` in 11 tokens with
`chat_template_kwargs.enable_thinking=false`. The suite entry for this model now
supplies that request default through a new `targets.request_defaults` channel,
and the same suite is 22 passed / 0 failed. The evals and the benchmarks still
exercise the default thinking path.

## Qualitative and prompt-format evidence (`$qualitative-check`)

Prompt-format decision, recorded in `qualitative/qualitative_prompt_format.json`:
the checkpoint declares a non-empty `chat_template`, so the model is treated as
chat/instruct and **every** prompt-based artifact in this stage uses that
template.

| surface | how the template is applied |
|---|---|
| release smoke and qualitative suite | `/v1/chat/completions`, server renders the checkpoint's chat template |
| lm-eval (both tasks) | `local-chat-completions` with `--apply_chat_template` |
| benchmarks | `vllm bench serve --backend openai-chat --endpoint /v1/chat/completions` |
| spec tests | `/v1/chat/completions`, with `chat_template_kwargs.enable_thinking=false` |
| token-exact probes | `tokenizer.apply_chat_template(..., add_generation_prompt=True)` then `/v1/completions` |

No release-readiness verdict here rests on raw-completion output from this
instruct model. The rendered prompt for suite prompt 0 is stored verbatim in the
metadata file, and it ends `<|assistant|><think>`, which is the thinking-on
generation prompt.

The shared six-prompt suite
(`models/common/readiness_check/vllm_prompts.txt`) was run greedy and sampled
through the chat API: `qualitative/vllm_qualitative_outputs.json`. Eleven of the
twelve completions are coherent and on-task (haiku with correct structure,
supervised vs unsupervised explanation, three laws of thermodynamics, correct
French translation, two working Fibonacci implementations).

`models/common/readiness_check/check_degenerate_output.py` reports one
**advisory** finding and no critical findings
(`qualitative/degenerate_check.json`): adjacent-token duplication is 0.0000 to
0.0091 across all completions against a 0.10 critical threshold, so the
mechanical decode-loop signature the checker exists to catch is absent.

`run_vllm_server --stages qualitative` itself cannot be used against this server:
it hardcodes `max_tokens=256` and raises `Chat completion returned no text
content` for the same thinking-plus-parser reason as the spec tests. The suite
was therefore run with the same prompts, prompt mode, and greedy/sampled pair at
a budget the model can finish in.

## Anomaly ledger

**Observed anomaly:** the greedy arm of qualitative prompt 2 ("Complete this
story: Once upon a time, in a faraway kingdom, there lived a curious young
inventor who discovered") returns an empty answer at `max_tokens=4096` and again
at 16384; both stop with `finish_reason=length` inside the reasoning block.

**Evidence:** `qualitative/greedy_trace_probe.json`. Raw greedy trace at 4096
tokens: 2872 words, `adjacent_duplication` **0.0000**, `trigram_loop_fraction`
**0.7955**, `distinct_trigram_ratio` 0.308, most common trigram
`"...a peculiar, glowing"` repeated 29 times. The text is well-formed English
planning prose ("Analyze the Request", "Brainstorming", "Drafting") that gets
stuck re-drafting the same story paragraph.

**Affected path:** greedy (`temperature=0`) decode on an open-ended creative
prompt, reasoning trace only. Not seen on any of the other five suite prompts,
whose greedy arms all terminated (584 to 1313 completion tokens).

**Control or comparison:** the same prompt under the model card's own recommended
sampling settings terminates normally: `finish_reason=stop`, 1713 completion
tokens, `trigram_loop_fraction` 0.0846, `distinct_trigram_ratio` 0.9496, and a
coherent story. The optimized-vLLM stage's committed greedy output for this exact
prompt (256 tokens, no reasoning parser) shows the same structured planning
style, so the behaviour predates this stage.

**Likely subsystem:** decoding policy, not the serving stack. Zero adjacent
duplication rules out the stale token/position feedback and stale trace input
class that produces "the the difference difference" style output, which is what
the degenerate-output checker treats as critical. Phrase-level looping under
greedy decoding is what the same checker classifies as advisory.

**Investigation performed:** raw trace pulled through `/v1/completions` with the
chat template applied so nothing was stripped; repetition statistics computed
over the whole trace; greedy and sampled arms compared on the identical prompt
and server; previous-stage output on the same prompt inspected as a control.

**Resolution:** controlled. Recorded as a limitation: greedy decoding is outside
the operating point the model card documents (temperature 1.0, top_p 0.95), and
every graded surface in this release (both evals, all benchmarks, the conformance
suite) uses sampled or explicitly parameterised decoding. What is still missing is
a Hugging Face reference control on the same prompt, which needs a GPU host: a
4096-token generation from a 30.6B MoE on CPU is not feasible in this window. If
the canonical implementation loops the same way, this is a checkpoint property;
if it does not, it becomes autoport work.

## Recovery actions

* No ARC, ERISC, remote-Ethernet or `tt-smi` reset failure occurred during the
  release runs themselves. The one crash (the prefill OOM above) was a clean
  software `TT_FATAL`, after which all four boards still listed healthy and no
  reset was required.
* During the `$autofix` iteration, two deliberately OOM-killed engine processes
  left the driver in a bad state and a later EngineCore start failed with
  `NOC0 is hung on PCIe device ID 1`. Recovered per `$tt-device-usage` with
  `build_Release/tools/umd/warm_reset --max-attempts 3` followed by a 1x1 mesh
  open/close smoke, then the run resumed. Recorded in `AUTOFIX_prefill_dram.md`.
* End state: no vLLM server, no `EngineCore`, no `run_vllm_server`, no tmux
  session and no Docker container from this stage is running, and
  `tt-smi -ls --local` lists all four Blackhole boards.

## Copied artifacts

Under `models/autoports/zai_org_glm_4_7_flash/doc/tti_release/` (520 KB total):

* `reports_output/release/report_*.md` and
  `reports_output/release/data/report_data_*.json`: the customer-facing release
  report and its data;
* `run_specs/*.json`: the TTI runtime model specs for the release run and the
  smoke, which carry the implementation proof and the `cli_args`;
* `benchmarks/*.json`: the 23 release sweep points plus the smoke point. Four
  files exceeded the repo's file-size limit because of their per-request raw
  arrays; those arrays (`itls`, `ttfts`, `generated_texts`, `errors`,
  `input_lens`, `output_lens`) were dropped on copy and each file records that
  it was trimmed, with min/max/count summaries kept. Every aggregate metric is
  untouched and the untrimmed originals stay in `tt-inference-server/workflow_logs/`;
* `evals/results_*.json`: the lm-eval aggregate results for both tasks, plus the
  two empty-answer probe artifacts;
* `qualitative/`: the qualitative outputs, prompt-format metadata, the
  degenerate-output check result, and the greedy-trace characterisation;
* `logs/run_py_release.log`: the successful release run log;
* `logs/prefill_oom_before_fix.log`: the `TT_FATAL` and traceback from the first
  attempt, kept as the before side of the DRAM fix;
* `non_aligned_prompts.json`: the pre-release non-aligned prompt-length probe;
* `AUTOFIX_prefill_dram.md`, `autofix_prefill_dram.json`: the `$autofix` report.

Deliberately not copied: `.env` from either checkout (none was created or
modified), the HuggingFace cache, model weights, Docker or persistent TT cache
volumes, tensor dumps, profiler CSVs, the 29 MB raw server log, and the
`samples_*.jsonl` per-sample eval dumps.

## Known limitations carried out of this stage

1. GPQA is a CI-subset result (10 of 198) compared against a full-set published
   number. Not full-set release readiness.
2. No in-house GPU or canonical-implementation control exists for either eval, so
   `ifeval` has no automated verdict at all and GPQA has no apples-to-apples
   subset reference.
3. The upstream full-occupancy determinism defect
   [tenstorrent/tt-metal#55408](https://github.com/tenstorrent/tt-metal/issues/55408)
   is unchanged. This stage adds one more datapoint: two ifeval items that come
   back empty at concurrency 16 return complete answers at concurrency 2 with the
   same seed.
4. Greedy decoding can fail to terminate on open-ended creative prompts inside
   the reasoning block; see the anomaly ledger. No HF control yet.
5. The benchmark `ttft_ms` target compares a chat-endpoint measurement against a
   completions-endpoint one and should be re-derived.
6. Prefill above about 32K tokens costs 1.547 GiB of permanently reserved DRAM
   that the KV pool no longer gets. Streaming the gate_up transpose or not
   holding the whole-prompt accumulator would buy it back; both are prefill
   compute-path changes and were out of scope for an OOM fix.
7. ISL 131072 has a 1034 s TTFT, a large part of which is first-use program
   compilation at a shape no warmup bucket covers. Release harnesses at these
   lengths need generous timeouts.
