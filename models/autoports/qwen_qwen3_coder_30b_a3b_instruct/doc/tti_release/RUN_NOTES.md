# TTI release handoff — RUN_NOTES

**Status: the `run.py --workflow release` run completed successfully. `run.py`
exit code: 0 (`EXIT_CODE=0`).** This file is the handoff record for that run and
for the registration/smoke work that preceded it.

The headline is that this model is **not** the thing that failed in the first
three attempts. Every failure on the way here was in the harness or in the
client-side eval configuration copied from a sibling model, and each one is
written up below with its mechanism, its evidence and its fix. Two rows in the
task list still come back failed; both are a documented TTI limitation that
cannot be satisfied by any non-Llama model, they carry a `known_issues` waiver
naming that root cause (a waiver labels the failure; it does not hide it — see
"The `meta_*` waiver" under Finding 2), and a generic replacement pair was
registered alongside them precisely so the stage is not left with zero
measurement.

## Result summary (read this first)

| Row | Result | Graded? | Judgement |
|---|---|---|---|
| `mbpp_instruct` | **77.2 %** pass@1 | ungraded — no reference | **model, good.** 386/500. See the truncation finding below: the same model scored 27.6 % under the inherited `max_gen_toks=256` and that number was a harness artifact, not a model result. |
| `humaneval_instruct` | **92.7 %** pass@1 | ungraded — no reference | **model, good.** 152/164. |
| `meta_ifeval` | **FAILED** (exit 1) | not graded | **harness.** Cannot run on any non-Llama model. Mechanism and source lines below. Waived in the catalog with that root cause as the reason; still reported FAILED. |
| `meta_gpqa_cot` | **FAILED** (exit 1) | not graded | **harness.** Same cause as `meta_ifeval`. Same waiver, same disposition. |
| `ifeval` | **81.1 %** prompt-level strict / **87.1 %** instruction-level strict | ungraded — no reference | **model, good.** 541/541. This row exists only because `meta_ifeval` cannot run; it is the stage's sole IFEval measurement. |
| `gpqa_diamond_cot_zeroshot` | **56.1 %** (re-run) | ungraded — no reference | **model, good.** 111/198. Failed in the workflow run on a *gated dataset* (access, not the model — Finding 4); access was granted and it was re-run against the same live server. Its `strict-match` filter reports 0.0 % — a task-definition defect, explained in the judgement. |

"Ungraded — no reference" means TTI has no `published_score` or
`gpu_reference_score` to compare against, so it reports the row for information
and does not pass/fail it. That is a deliberate registration choice, not an
omission: Qwen publishes no MBPP/HumanEval/IFEval/GPQA number for this model
(their card reports agentic-coding benchmarks such as SWE-bench Verified), and
no GPU reference run exists yet. Inheriting a sibling model's numbers was
rejected — a fabricated reference is worse than an honest blank. **The missing
reference is the reason these rows are ungraded; it is not evidence of a
problem with the model.** ~10 other entries in the shipped catalog use the same
`published_score=None` pattern.

**One model-side defect was found, and it is a performance one.** Every
benchmark point completed, but `isl=131072` took a **94.4-minute** TTFT — 3.05x
worse per input token than `isl=65536`. Long-context prefill on this port is
severely superlinear. It is served correctly and uncapped; it is simply
impractically slow at the top of the range. That is **Finding 5**, it is ours,
and it is disclosed rather than waived. The four tool-side findings below are
not our defects; this one is.

**Both GPQA rows failed in the workflow run, for two entirely different
reasons** — `meta_gpqa_cot` because the Meta eval harness is Llama-only
(Finding 2, permanent), and `gpqa_diamond_cot_zeroshot` because the dataset is
gated and the account lacked access (Finding 4, since resolved). Only the second
was fixable: access was granted and the task was re-run against the same live
server, giving **56.1 %**. So the dual registration ultimately covered both
IFEval and GPQA, and the `meta_*` pair remains permanently unrunnable here.

## The release run

| | |
|---|---|
| Started | 2026-08-18 22:35:11 UTC |
| Workflow | `release` (evals + benchmarks + report) |
| Server mode | **external / no-Docker** — our own autoport vLLM server; TTI ran as a pure client via `--server-url`. No `--docker-server`, no `--local-server`, no Docker, no container. `run.py` started no server of its own. |
| Host | `qbge-devex-01`, bare metal, 1x4 Blackhole **P300_X2** (4 dies) |
| tt-inference-server | branch `raahem/qwen3-coder-30b-a3b-tti`, git SHA **`c4d1e9d42`**, `VERSION` file = **0.20.0** |
| tt-metal | `/home/raahem/tt-metal`, branch `raahem/qwen3-coder-30b-a3b`, git SHA **`24092f5381f`** |
| vLLM plugin | `/home/raahem/vllm-tt-plugin` @ **`bc4af2d`** — untouched |
| Run log | `logs/tti_release.log` (JWT redacted) |
| Runtime spec | `run_specs/runtime_model_spec_release.json` — byte-identical to the spec TTI wrote for this run |

> **Version caveat, read this before quoting a version.** The `VERSION` file says
> **0.20.0** and `run.py` logs `TT-Inference version: 0.20.0` / `TT-Inference SHA:
> c4d1e9d42033`. But `git describe --tags` on this checkout returns
> `v0.10.0-1113-gc4d1e9d42`: the nearest tag *reachable* from this branch is
> `v0.10.0`, **1113 commits behind**. A `v0.20.0` tag does exist in the repo but
> is not an ancestor of this branch, so tag-based tooling silently reports a
> version ten minor releases stale. **Quote the `VERSION` value 0.20.0 and the
> SHA `c4d1e9d42`; never the git tag.**

**Command** (`EXIT_CODE` echoed by the wrapper, `scratch: rel/run_release.sh`):

```bash
cd /home/raahem/tt-inference-server
python3 run.py \
  --model Qwen3-Coder-30B-A3B-Instruct \
  --impl qwen3-coder-30b-a3b-autoport \
  --tt-device p300x2 \
  --workflow release \
  --dev-mode \
  --server-url http://127.0.0.1:8100 \
  --no-auth \
  --skip-system-sw-validation
echo "EXIT_CODE=$?"
```

Key environment: `TT_METAL_HOME=/home/raahem/tt-metal`,
`PYTHONPATH=/home/raahem/tt-metal`. `--dev-mode` selects the dev catalog
(`workflows/model_specs/dev/llm.yaml`); the prod catalog was not touched and
does not expose this model. `--skip-system-sw-validation` is required because
the host runs a bare-metal tt-metal build rather than TTI's pinned system
software, and `--no-auth` because the server is local and unauthenticated.

**Server** — started before the run and left untouched for its whole duration
(startup to `/health` 200: ~3 min 10 s):

```bash
cd /home/raahem/tt-metal
source python_env/bin/activate
export EXTRA_MODELS_DIR=$PWD/models/autoports/qwen_qwen3_coder_30b_a3b_instruct/vllm_bundle
python -m models.common.readiness_check.run_vllm_server \
  --model-dir <scratch> \
  --hf-model Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --mesh-device P300x2 --max-num-seqs 32 --max-model-len 262144 \
  --block-size 32 --port 8100 --stages serve \
  --tt-config '{"trace_region_size": 50331648, "fabric_config": "FABRIC_1D_RING"}' \
  --additional-server-args "--generation-config vllm"
```

Port 8100 because 8000 is held by an unrelated process. `--max-num-seqs 32`
matches the catalog's `max_concurrency: 32`.

### Nothing capped the context or the request length

This is worth stating explicitly because it is the single easiest thing for a
release run to get quietly wrong.

| Layer | Value | Evidence |
|---|---|---|
| Context contract | 262144, `capability_reduction: false` | `doc/context_contract.json` |
| Catalog entry | `max_context: 262144` | `run_specs/runtime_model_spec_release.json` |
| Server | `--max-model-len 262144` | server command above; `/v1/models` reports `max_model_len: 262144` |
| Eval client | `max_length=262144` | every `lm_eval` command line in the run log |
| Benchmark sweep | tops out at isl 131072 | a *sweep point*, not a cap — TTI generates the sweep from `max_context` and its largest text point is half the contract |

**What "262144 is served" does and does not mean — read this before planning
against the top of the range.** The contract stands and nothing here lowers it:
262144 is allocated (KV cache for 263168 tokens), page-tabled, advertised by
`/v1/models`, and prefilled through **one** layer in **192.23 s** in the
contract's own probe sweep. But **the largest prompt ever pushed through all 48
layers is 131072, and it took 94.4 minutes** (Finding 5). **No full-model 262144
prefill exists, at any duration.** Treat 262144 as served-and-uncapped but
unmeasured end-to-end; the actionable consequence is that anyone planning to use
the top of the range should budget a measurement for it first, because the naive
48-layer lower bound is 2.56 hours and the measured superlinearity above 65536
says the real number is worse.

The eval client value is not cosmetic. lm-eval's `api_models` defaults
`max_length` to **2048**, which with `max_gen_toks=2048` would have left zero
prompt budget; the first attempt's log shows `Using max length 2048 - 1`. It is
pinned to the model's real context in `model_kwargs` so the client never
truncates below what the server serves. This run's log shows
`Using max length 262144 - 1`.

### Non-aligned prompt lengths

Stages 01-09 verified 16 / 37 / 131 / 333 / 1025 / 4097. Nothing in this run
required an aligned length and no alignment workaround was applied anywhere. The
smoke benchmark ran a **16**-token prompt asking for 4 tokens (8/8 completed),
and the eval tasks send whatever length the dataset produces — MBPP, HumanEval,
IFEval and GPQA prompts are arbitrary natural-language and code lengths, none of
them chunk-, page-, tile- or trace-aligned, and 500 + 164 + 541 of them
completed without a single length-related failure. There is no evidence of an
alignment constraint at any layer of this port.

### Recovery actions — three attempts preceded this one

| # | Started | Outcome | Cause | Action |
|---|---|---|---|---|
| 1 | 20:32:33 | aborted during `humaneval_instruct` (`rc=-15`, SIGTERM) | `mbpp_instruct` had just scored **27.6 %** at the inherited `max_gen_toks=256`; the run was stopped to diagnose rather than let it publish that number | diagnosed the fence-truncation mechanism (Finding 1); ran the 16-sample control at 2048 |
| 2 | 21:17:21 | abandoned | **21 `TimeoutError` retries** from the 512-prompt client oversubscription (Finding 3) | `batch_size` 16 → 1; `timeout` → 7200 s; `max_length` pinned to 262144 |
| 3 | 22:35:11 | **this run — completed, exit 0** | — | — |

Attempt 1 logged **0** `TimeoutError`s and attempt 2 logged **21**; the
difference between them was that attempt 2 was the first to raise
`max_gen_toks` to 2048, which lengthened every request enough for the
pre-existing 16× oversubscription to start tripping the client timeout. The two
bugs were independent, and fixing only the first one exposed the second.

Log copies: `logs/release_attempt_2026-08-18_20-32-33.log`,
`logs/release_attempt_2026-08-18_21-17-21.log`.

## Autoport implementation check — `models/autoports/qwen_qwen3_coder_30b_a3b_instruct`

The autoport implementation check for this stage confirms that everything TTI
evaluated came from `models/autoports/qwen_qwen3_coder_30b_a3b_instruct` and
from nothing else.

The runtime spec TTI itself wrote for this release run
(`run_specs/runtime_model_spec_release.json`, byte-identical to
`workflow_logs/runtime_model_specs/runtime_model_spec_2026-08-18_22-35-11_id_qwen3-coder-30b-a3b-autoport_Qwen3-Coder-30B-A3B-Instruct_p300x2_ZIJqG3Un.json`)
contains:

```
"code_path": "models/autoports/qwen_qwen3_coder_30b_a3b_instruct"
```

with `impl.impl_name = "qwen3-coder-30b-a3b-autoport"` and model id
`id_qwen3-coder-30b-a3b-autoport_Qwen3-Coder-30B-A3B-Instruct_p300x2`. No
`models/tt_transformers` and no `models/demos/…` path appears in any copied
artifact. The served weights came from the autoport vLLM bundle
(`EXTRA_MODELS_DIR=.../vllm_bundle`), so the implementation under test is the
generated autoport model rather than a stock one.

## The report says `Acceptance: ✅ PASS`. Read that carefully.

TTI's own verdict on this run is:

```
Acceptance status: ✅ PASS
Model status: EXPERIMENTAL
Benchmarks: 🟨 NA (0/19 passed, 19 NA)
Evals:      ✅ PASS (0/6 passed, 3 waived, 3 NA)
Spec Tests: 🟨 NA (no blocks present)
run.py exit code: 0    (EXIT_CODE=0)
```

**`PASS` here does not mean the evals passed. Zero of six passed.** Three were
ungraded for want of a reference score, and three *failed* and were then
demoted to "waived".

**The green verdict is not ours.** `known_issues` is `[]` in the runtime spec
this run loaded, and no waiver was configured in the catalog entry at the time.
(A `known_issues` waiver for the two `meta_*` rows was added to the catalog
*afterwards* — catalog edit 4, and "The `meta_*` waiver" under Finding 2. It
attaches a reason string and changes nothing about the verdict; the copied spec
and report predate it and are unedited.) The demotion here is automatic and
tier-based: `report_module/acceptance_criteria.py:434` does

```python
if not evals_enforced:
    waived[block_key] = f"{message} {_informational_suffix(model_status)}"
    continue
```

and `_evals_enforced()` (line 43) returns the tier's policy. Because this model
is registered as **`EXPERIMENTAL`**, every eval failure is masked to
informational. The waiver strings in the report data read literally:

```
"Accuracy check failed. (informational: model status=EXPERIMENTAL)"
```

So all three failing rows — `meta_ifeval`, `meta_gpqa_cot` and (in the workflow
run) `gpqa_diamond_cot_zeroshot` — were converted into a green verdict by the
model's tier alone.

This is the single most important thing to know before quoting this report. The
green tick is a statement about the EXPERIMENTAL tier's policy, not about the
model's accuracy. The row-by-row judgement below is the real result, and every
failing row is named there with its cause.

**And the report travels without this file.** The copied report is byte-identical
to TTI's original and is deliberately not edited, so on its own it still leads
with an unannotated `✅ PASS`. A sibling annotation —
**`reports_output/README.md`** — sits next to it saying that PASS does not mean
the evals passed, that zero of six did, what the three failures actually were,
and pointing back here. Anyone handed just the `reports_output/` directory gets
the caveat with it. (`spec/README.md` does the same job for the configuration
files: which of them describe the current tree and which describe the run.)

**Combined with the finding that a per-eval-task crash never reaches the
workflow exit code** (individual `lm_eval` invocations run with `check=False`, so
`task_failure_blockers` never sees them), there are *two independent mechanisms*
in this release path that turn a failed eval into a green report. Neither is a
bug we introduced, and both are disclosed here rather than relied upon.
## Finding 1 — the `mbpp_instruct` score was a harness artifact, not a model result

This is the most important thing in this document, because the default
configuration would have published a catastrophic score for a model that was
never at fault.

**Mechanism.** `mbpp_instruct` pre-fills the assistant turn with an *opening*
` ```python ` fence, and the task's `extract_code` filter needs the matching
**closing** fence to extract anything at all. A completion that is still inside
the code block when generation stops is filtered to the empty string, and an
empty string cannot pass. So the metric does not measure "did the model write
correct code" — it measures "did the model finish writing before the token
ceiling", and only then whether the code was correct. This model writes a
docstring and its own `assert` checks before closing the block, so it is
systematically penalised by a low ceiling.

**What the inherited setting did.** The `EvalConfig` was copied from
`Qwen/Qwen2.5-Coder-32B-Instruct`, the closest code-instruct model already in
the catalog, which sets `max_gen_toks=256`. At 256 tokens the first release
attempt scored **27.6 %**: 336 of 500 completions were still mid-block at the
ceiling and were filtered to empty. Of the 164 that did close the fence, **138
passed — 84.1 %**. The 27.6 % was measuring the token ceiling.

**Control.** A 16-sample control at `max_gen_toks=2048`, run against the same
server, scored **0.75** versus **0.276** for the full 500-sample run at 256
(`evals/control_16sample_2048_results.json`).

**Fix.** `max_gen_toks` raised 256 → 2048 for both coding tasks. The full run in
this handoff scored **77.2 %**. Note carefully what kind of limit this is: it
raises a *generation* ceiling. It does not cap context, and it is not a
weakening of the eval.

**The mechanism is re-derivable from this run's own artifacts, not just from the
failed attempt.** Decoding is greedy (`do_sample=false`, `seed=42`), so the
first N tokens of a 2048-token completion are exactly the tokens a
`max_gen_toks=N` run would have produced. Thresholding this run's completions at
256 generated tokens therefore *replays* the 256-token run rather than modelling
it:

| Quantity | At the 2048 ceiling (this run) | Replayed at a 256 ceiling |
|---|---|---|
| Completions that close the fence | **482** / 500 (96.4 %) | **166** / 500 (33.2 %) |
| Completions filtered to empty | **18** | 334 |
| `pass@1` | **0.7720** (386/500) | **0.2800** (140/500) |

The replay lands on 0.280 against the 0.276 that the 256-token run actually
scored, and on 166 closures against the 164 observed — a two-sample difference
from stop-sequence and tokeniser-boundary effects. The mechanism is confirmed
from an artifact that still exists.

Two honest caveats. First, **the 27.6 % / 336 / 164 / 138 / 84.1 % figures from
the original 256-token run have no surviving artifact at all**: lm-eval writes
every task's samples to the same `--output_path`, so the 20:32 run's
`samples_mbpp_instruct_*.jsonl` was overwritten by this run's. **The copied
attempt-1 log does not carry the score either** — grep it for `27.6`, `0.276` or
`pass_at_1` and you get nothing; the run was stopped before lm-eval printed a
results table. What that log *does* preserve is the **configuration and the
timings**: the verbatim `lm_eval` command line with
`--gen_kwargs max_gen_toks=256,...` and `--batch_size 16`, and the per-task
timestamps. So the log is evidence for *what was run*, not for *what it scored*.
The 27.6 % survives only as prose here plus the 0.280 replay above, and the five
figures are declared `UNCOVERED` in `probes/uncovered.json` on exactly that basis
rather than being presented as re-derived.

Second, **18 completions are still truncated at 2048** — 3.6 % of the set is
still measuring the ceiling rather than the model, so 77.2 % remains
a mild *under*-estimate. Raising the ceiling further would cost proportionally
more wall-clock for a shrinking correction; 2048 is where that trade was struck,
and the residual is disclosed rather than hidden.

## Finding 2 — `meta_ifeval` and `meta_gpqa_cot` cannot run on any non-Llama model

**Mechanism.** `workflows/workflow_venvs.py:444` builds the dataset name by
string-concatenation from whatever model is being registered:

```python
config["evals_dataset"] = f"{_model_name}-evals"
```

For this model that is `Qwen/Qwen3-Coder-30B-A3B-Instruct-evals`, which does not
exist. `prepare_meta_eval.py:280-287` then rejects anything outside the
Llama-3.1/3.2 Evals collection:

```
ValueError: The evals dataset is not valid, please double check the name,
must use the name in the Llama 3.1 or 3.2 Evals collection.
```

**TTI warns and continues rather than failing loudly.** The dataset-prep step is
invoked with `check=False`, so `workflow_venvs.py:456` logs
`Failed to prepare meta eval datasets for: ... continuing...` and the run
proceeds. The consequence surfaces much later and much less legibly: the
per-model `work_dir` is empty, so the `--include_path` handed to lm-eval
contains no task YAML, and lm-eval exits 1 with `Tasks were not found:
meta_ifeval` / `Tasks were not found: meta_gpqa_cot`. Two failures 72 minutes
apart, with the actual cause logged as a warning at t+2 s.

### The `meta_*` waiver — and a correction to what an earlier draft claimed

**These rows are now waived, and an earlier draft of this document gave a false
reason for not waiving them.** That draft said a waiver "would remove the rows
from the acceptance verdict and leave a reader of the report with no idea that
two of six eval rows never executed." **That is not what the code does.** The
waiver path in `report_module/acceptance_criteria.py:428-432` is:

```python
task_name = data.get("task_name") if data is not None else None
reason = _find_waiver(known_issues, "EVALS", task_name)
if reason is not None:
    waived[block_key] = f"{message} (waived: {reason})"
    continue
```

The only thing it changes is the **reason string** attached to the block. The
row still lands in the same `waived` map that the `evals_enforced` branch two
lines below (`:434-435`) writes to, and the accuracy table is built from the
Blocks, so the row keeps showing ❌ FAIL either way. This report already
demonstrates it: three rows are in `waived` today purely through the
EXPERIMENTAL-tier branch, and all three still print `❌ FAIL` in the report's
accuracy table (lines 41, 42 and 44 of the copied report). Declining the waiver
removed nothing from the reader's view; it only meant the report carried **no
machine-readable reason** for the two failures.

**So the waiver was added** — `workflows/model_specs/dev/llm.yaml`,
`device_model_specs[P300X2].known_issues`, catalog edit 4 in "TTI catalog
edits" near the end of this document:

```yaml
known_issues:
  - workflow_type: EVALS
    task_name: meta_ifeval
    reason: "Structural TTI limitation, not a model result: workflow_venvs.py:444
      builds evals_dataset as f'{model_name}-evals' and prepare_meta_eval.py:280-287
      accepts only the Llama-3.1/3.2 Evals collection, so this task cannot execute
      for any non-Llama model. Root cause and reproducer: tt-metal
      models/autoports/qwen_qwen3_coder_30b_a3b_instruct/doc/tti_release/RUN_NOTES.md,
      Finding 2. Upstream issue NOT YET FILED - pending filing by the project owner.
      Generic 'ifeval' is registered alongside and does measure this model."
  - workflow_type: EVALS
    task_name: meta_gpqa_cot
    reason: "… identical cause to meta_ifeval … Generic
      'gpqa_diamond_cot_zeroshot' is registered alongside and does measure this
      model."
```

Three things about it, stated plainly:

- **It postdates the release run.** The copied
  `run_specs/runtime_model_spec_release.json` still carries `known_issues: []`,
  because that is what was loaded at 22:35:11. The waiver takes effect on the
  next run. The report in `reports_output/` is likewise unchanged and still
  shows the tier-based `(informational: model status=EXPERIMENTAL)` strings.
- **The upstream issue is NOT YET FILED.** No issue number or URL is invented
  here. The reason strings cite the in-repo reproducer (Finding 2, and the
  source lines it names) and mark the filing as owed. **Owed action: file an
  issue against `tenstorrent/tt-inference-server` for the Llama-only
  `prepare_meta_eval` path and replace the `NOT YET FILED` marker in both reason
  strings with its URL.**
- **The waiver is a label, not a pass.** It does not make the rows green and it
  is not a claim that the model was measured. The two rows are still listed as
  FAILED in the summary table at the top of this document.

Note also the shape of the risk, which the waiver does not address: because the
eval runner treats an individual task failure as recoverable, these two crashes
do not propagate into the `evaluation` task's exit code, so the acceptance
verdict can read PASS while two rows silently did not run. That is exactly the
laundering this section exists to prevent.

**This is why the generic pair was registered alongside them.** `ifeval` and
`gpqa_diamond_cot_zeroshot` were added to the task list specifically because the
`meta_*` pair cannot execute here. Without them the stage would have **zero**
IFEval and **zero** GPQA measurement for this model. The two pairs are kept
side by side rather than the `meta_*` pair being deleted, so that the
limitation stays visible in the report instead of disappearing from it.

The `meta_*` and generic variants therefore **did not diverge as measurements** —
the `meta_*` pair produced no measurement at all to diverge from. Their scores
are not comparable in principle either: the Meta variants use Meta's own
prompt formatting, few-shot construction and answer-extraction regexes, which is
why a `meta_*` number and a generic number for the same benchmark are not
interchangeable even when both run.

## Finding 3 — client-side oversubscription, and the operator rule that follows

**Mechanism.** lm-eval's `api_models` creates one asyncio task per *batch* up
front and gates them with `Semaphore(num_concurrent)`. Prompts in flight is
therefore `num_concurrent × batch_size`, not `num_concurrent`. The inherited
config set `batch_size=16` with `num_concurrent=32`, i.e. **512 prompts in
flight against a 32-slot server** — a 16× oversubscription. Each request's
`ClientTimeout` clock starts when the request is *issued*, not when the server
starts working on it, so a request burns its entire timeout budget sitting
behind ~496 others that are queued ahead of it. The server is healthy and
making progress the whole time; the client times out anyway and retries, which
adds still more load. The second release attempt logged **21 `TimeoutError`
retries** this way.

**TTI's own code already warns about exactly this**, at
`reference_config/evals/eval_config.py:189-194`:

```python
# Note: batch_size is set to 1 because max_concurrent is set to 32
# this means that 32 requests are sent concurrently by lm-eval / lmms-eval
# for clarity, the client side eval scripts cannot control the batch size
# so setting just multiplys the max_concurrent which is misleading
batch_size: int = 1
max_concurrent: int = 32
```

The `Qwen2.5-Coder-32B-Instruct` entry that this config was copied from
overrides that default to 16. Copying a working sibling entry propagated the
bug.

**A false alarm worth ruling out**, because it looks identical from the outside:
inspecting the shipped config shows the two `meta_*` tasks with
`batch_size = 32`, which reads like a 32× version of the same mistake. It is
not. `EvalTask._infer_data` (`reference_config/evals/eval_config.py:238-241`)
*deliberately* moves `max_concurrent` into `batch_size` and clears it for any
`EVALS_META` task, because the meta venv pins lm-eval 0.4.3, which has no
`num_concurrent`. Prompts in flight there is 32, not 1024. Nothing to fix — and
those two rows never executed anyway (Finding 2).

**Fix.** `batch_size` left at the `EvalTask` default of 1, giving
`1 × 32 = 32` prompts in flight — exactly the server's configured width. This
run logged **0 `TimeoutError`s** across all four executable eval tasks.

**Why this matters more on this port than it would elsewhere, and the operator
instruction.** Pair the above with the configured-width characteristic measured
in stage 09: a 32-wide decode graph costs roughly the same per step whether 1
row or 32 rows are live (~229 ms/step at 1 live row of 32, ~269 ms at 32 —
`doc/optimized_vllm/README.md`). Per-step cost tracks the *configured* slot
count, not the active one. Two consequences for anyone running evals against
this port:

1. **Match client concurrency to server width, and do not multiply it with
   `batch_size`.** Oversubscribing buys nothing — the extra prompts cannot be
   worked on any faster — and it converts a healthy server into a wave of
   client-side timeouts.
2. **Set the client timeout from the slowest legitimate response, not from a
   generic default.** At full occupancy a correct 2048-token answer legitimately
   takes ~9 minutes. lm-eval's default is 1800 s, which would cut off correct
   answers; this config pins `timeout=7200`. That raises a client patience
   limit — it caps nothing about the model or the request.

## Finding 4 — GPQA was blocked by a gated dataset, not by anything we built (now resolved)

This is the same *class* of problem as the other three — a tool precondition
failing in a way that looks, in the report, exactly like a model problem — but
its cause lay outside both the model and TTI, and unlike Finding 2 it was
fixable.

**What happened.** On the first pass `gpqa_diamond_cot_zeroshot` exited 1 with:

```
datasets.exceptions.DatasetNotFoundError: Dataset 'Idavidrein/gpqa' is a gated
dataset on the Hub. Visit the dataset page at
https://huggingface.co/datasets/Idavidrein/gpqa to ask for access.
```

lm-eval raised this inside `ConfigurableTask.__init__` → `self.download(...)`,
i.e. **before a single request reached the server**. The task generated nothing,
so nothing about that row reflected the model, the port, the adapter or the
serving configuration.

**It was an account permission, not a missing token.** This is worth stating
precisely, because the obvious guess — "the host has no `HF_TOKEN`" — is wrong,
and acting on it would have sent the next person looking in the wrong place:

- `HF_TOKEN` is present in `/home/raahem/tt-inference-server/.env`.
- `run.py` calls `handle_secrets()` unconditionally at `run.py:993`, which calls
  `load_dotenv()` (`workflows/utils.py:337-353`) and sets `os.environ["HF_TOKEN"]`.
  For `--workflow release` the token is a **required** env var (RELEASE is not in
  `client_side_workflows`, `run.py:797-816`), and `handle_secrets` *asserts* it
  is set. **The run would have aborted at step 2 had the token been missing — it
  did not.** So the token was in the environment the eval subprocess inherited.
- `handle_secrets` runs at step 2, *before* logging is configured at step 3, which
  is the only reason no `loaded env var from .env file: HF_TOKEN` line appears in
  the run log. Its absence is a logging artefact, not evidence of a missing token.
- Correspondingly, the run log contains **zero** `unauthenticated requests to the
  HF Hub` warnings — the marker you do get when the token really is absent.

The token was therefore valid and visible; the account simply had not been
granted access to a gated dataset. The three datasets that did work
(`google-research-datasets/mbpp`, `openai/openai_humaneval`, `google/IFEval`)
are all ungated, which is exactly why they were unaffected.

**Resolution.** Access to `Idavidrein/gpqa` was granted to the account, and the
task was re-run against the same live server with the same settings the other
evals used. Verified before launching, rather than assumed:

```
whoami: raahemnabeel | type: user
gpqa dataset_info OK. gated: auto
files: ['.gitattributes', 'README.md', 'eval.yaml', 'gpqa_diamond.csv', ...]
load_dataset("Idavidrein/gpqa", "gpqa_diamond") -> {'train': 198}
```

That check was run **inside `.venv_evals_common`**, the same interpreter lm-eval
uses, specifically because a silent auth failure would have produced an error
identical to the original gated one.

**Outcome.** The re-run completed: **198/198, exit 0, 56.1 %**
(`exact_match,flexible-extract`, 111 correct). It used the identical settings the
other evals used — `batch_size 1`, `num_concurrent 32`, `timeout 7200`,
`max_gen_toks 4096`, `max_length 262144` — against the same live server, and
logged **0 `TimeoutError`s**. Artifacts: `evals/results_gpqa_diamond_cot_zeroshot.json`
and the derived per-sample summary in `evals/eval_samples_derived.json`.

Because the release report was generated before the re-run, **the report's
`gpqa_diamond_cot_zeroshot` row still reads FAIL**. That is correct for the
workflow as executed and has deliberately not been edited; the corrected result
lives here and in the copied artifacts.

**Why this is worth writing down.** A reader scanning the report sees two GPQA
rows with no score and concludes the model cannot do GPQA. Neither row said
anything about the model, and the two say *different* things about the setup —
one is a permanent structural block, the other was a permission that has since
been granted. Keeping them apart is the whole point.

## Finding 5 — long-context prefill is severely superlinear (model side, and ours)

The other four findings are tool-side. **This one is ours**, and it is the only
finding in this handoff that describes a defect in the port itself.

**It is not a failure.** The `isl=131072` sweep point **completed**: 1/1, 0
failed, 131072 input tokens in and 128 valid output tokens back, on a run that
ended `EXIT_CODE=0`. It is served correctly and uncapped. What is wrong is how
long it takes.

| | isl=65536 | isl=131072 |
|---|---|---|
| TTFT | 926.8 s | **5662.5 s = 94.4 min** |
| per input token | 14.14 ms | **43.20 ms** |

Doubling the input costs **6.11x** the wall clock — **3.05x worse per token**.

**Measured against our own context contract.** `doc/context_contract.json`
records `prefill_sweep_seconds`, but those are **single-layer probe** numbers;
the only 48-layer prefill ever verified end-to-end was a **128-token** prompt
(the contract's own note: *"stage 05 shipped with prefill unprofiled and
disclosed it as a gap; stage 06 closes it. One verified 48-layer prefill of a
128-token prompt"*). Scaling the single-layer sweep by 48 layers gives a naive
expectation to compare against:

| isl | single-layer probe | x48 layers | measured | measured / (48 x probe) |
|---|---|---|---|---|
| 65536 | 42.05 s | 2018.4 s | 926.8 s | **0.46** |
| 131072 | 85.85 s | 4120.8 s | 5662.5 s | **1.37** |

65536 comfortably beats naive layer scaling; 131072 badly misses it — a **2.99x
degradation** relative to the contract's own numbers between those two lengths.

**What this means for the advertised context.** 48 x the contract's 262144
single-layer probe (192.23 s) is **2.56 hours as a lower bound**, and that bound
ignores the superlinearity above, so a full 262144 prefill would plausibly take
several hours. **It has never been measured end-to-end and no such number should
be quoted.** This is a gap to disclose against the contract, **not** a reason to
lower the advertised context — 262144 is served, the KV cache is allocated for
263168 tokens, and nothing here is capped.

**Mechanism — hypothesis, explicitly unproven.** `moe_prefill_optimized`
(`tt/optimized_decoder.py`) walks the sequence in `EXPERT_CHUNK_SIZE = 32`
chunks and appends one device tensor per chunk to `outputs` before a single
`ttnn.concat` — **4096 live tensors at 131072 versus 2048 at 65536**, on top of
a KV cache already sized for 263168 tokens. That is consistent with allocator
pressure. What actually exists is `py-spy` stacks showing the loop executing;
**there is no measurement isolating the allocator**, so this remains a lead for
whoever fixes it rather than a diagnosis.

**Corroboration from the sweep itself.** For every input length the sweep runs
`max_concurrency=1` **first** and the concurrent point second. From `isl=32768`
the single request becomes *slower* than three or seven served concurrently
(1.50x at 32768, 3.35x at 65536), which is only possible if the first request at
a shape pays a one-time cost the second does not.

**The `py-spy` observation, stated with its window.** `py-spy --locals` was
attached to the EngineCore during the 131072 prefill and read `layer_idx` out of
the `prefill_hidden` frame. **Every sample it took showed `layer_idx = 0`, and
the last such sample was taken 73 minutes after the request was issued.**
Sampling stopped there; the request's measured TTFT was 94.4 minutes. So the
observation is a **lower bound**: layer 0 alone accounts for **at least 73 of
the 94.4 minutes (≥ 77 %)**. The residual **~21.4 minutes** is *unattributed* —
it contains whatever was left of layer 0, all 47 remaining layers, the final
`ttnn.concat` and the first decode step, in a split that was never measured
because the window ends at 73 min.

An earlier draft of this document wrote "the remaining 47 layers in roughly 5
minutes". **That figure is withdrawn**: no surviving artifact supports it, the
raw sampling timeline was never copied into this directory, and 73 + 5 does not
reach 94.4 in any case. Nothing load-bearing depends on it — the claim that
matters is that per-layer cost is *not* uniform, and a ≥ 77 % lower bound on
layer 0 establishes that on its own. That is precisely why superseded conclusion
(3), which multiplied layer 0's sampled rate by 48, overestimated the total by
roughly 25x.

Full detail, including three superseded conclusions and why each was wrong, is
in `stall/isl131072_stall_evidence.txt`.

**Judgement: our defect, disclosed, not waived.** A valid in-contract prompt
length must be handled, and it is — correctly, but at 94 minutes for one
request. Fixing it means changing how `moe_prefill_optimized` accumulates
chunks; that is model work beyond this stage and is recorded here for it.

## Eval rows, judged one by one

For every row: what it did, and whether the outcome is the **model**, the
**harness**, the **environment**, or a **missing reference score**.

### `mbpp_instruct` — 77.2 %, ungraded

386 of 500 passed. **Model, and the model is fine.** Ungraded only because there
is no reference score to grade against (missing reference, not a defect). The
important caveat is the one in Finding 1: 18 completions (3.6 %) are still
truncated at the 2048-token ceiling and score 0 regardless of correctness, so
77.2 % is a slight **under**-statement of the model's MBPP ability. Among the
482 completions that were allowed to finish, the pass rate is 80.1 %.

### `humaneval_instruct` — 92.7 %, ungraded

152 of 164 passed. **Model, and the model is fine.** Ungraded for the same
missing-reference reason. HumanEval solutions are short, so this task is not
materially exposed to the truncation mechanism that distorted MBPP — which is
itself a useful cross-check: the two coding tasks disagree by 15.5 points in
exactly the direction the truncation analysis predicts.

### `meta_ifeval` — FAILED (exit 1), not graded

**Harness.** Not a model result and not gradeable. Full mechanism in Finding 2:
the Meta eval path constructs a `<repo>-evals` dataset name and only Llama-3.1 /
3.2 models have one. Nothing about this row can be improved from our side; it
would need an upstream change in TTI. **Waived in the catalog with that root
cause as the reason — the waiver attaches a machine-readable reason and does not
remove the row from the report, which still shows `❌ FAIL`.**

### `meta_gpqa_cot` — FAILED (exit 1), not graded

**Harness.** Identical cause to `meta_ifeval`, same disposition.

### `ifeval` — 81.1 % / 87.1 %, ungraded

541 of 541 evaluated. **Model, and the model is fine.**

| Metric | Score |
|---|---|
| Prompt-level strict | **81.1 %** |
| Instruction-level strict | **87.1 %** |
| Prompt-level loose | **84.3 %** |
| Instruction-level loose | **89.3 %** |

The configured `result_keys` are the two *strict* accuracies, so 81.1 % / 87.1 %
are the published pair; the loose pair is recorded here because the 3-point
strict-to-loose gap is small, which is the normal signature of a model that is
genuinely following instructions rather than one that is scraping past the
checker on formatting technicalities. Ungraded for the missing-reference reason
only.

**This row is the whole justification for registering the generic pair.** With
`meta_ifeval` unable to run, deleting it and stopping there would have left the
release with no instruction-following measurement at all.

### `gpqa_diamond_cot_zeroshot` — **56.1 %** (`exact_match,flexible-extract`), ungraded

**Environment, and it was fixable.** On the first pass this exited 1 at dataset
download: `Idavidrein/gpqa` is gated on the HF Hub and the account had not been
granted access, so the task died before issuing a single request. It was **not**
a missing token — see Finding 4 for the proof. Access was subsequently granted
and **the task was re-run against the same live server** with the identical
settings the other evals used (`batch_size 1`, `num_concurrent 32`,
`timeout 7200`, `max_gen_toks 4096`, `max_length 262144`). It ran clean: **198/198**, exit 0, **111 correct = 56.1 %**.

**Model, and the model is fine** — with one harness caveat that must be read
before anyone quotes the row. The task reports two filters and they disagree
completely:

| Filter | Score |
|---|---|
| `exact_match,flexible-extract` (the configured `result_key`) | **56.1 %** (111/198) |
| `exact_match,strict-match` | **0.0 %** (0/198) |

The 0.0 % is a **defect in the task definition, not in the model.** The same YAML
that scores the answer also writes the prompt, and the prompt says *"put your
final answer (only the letter A, B, C, or D) within `\boxed{}`"*. The model obeys
it: **192 of 198 responses (97.0 %) end in a `\boxed{A-D}`**. But `strict-match`
extracts with the regex `(?<=The answer is )(.*)`, looking for a phrase the
prompt never asks for and which appears in only **38 of 198 responses (19.2 %)**.
Every strict-match extraction returned `[invalid]`. A model that followed the
instruction exactly cannot score above zero on that filter.

This is the same class of problem as Finding 1 — a scoring filter that measures
output *formatting* rather than correctness — and the registration already
avoided it: `result_keys` was set to `exact_match,flexible-extract`, which uses
the `boxed_choice` filter and reads the answer the prompt actually requested. The
strict-match figure is recorded here so that nobody rediscovers it as a model
failure.

### The two GPQA blockers are different in kind — do not collapse them

This matters more than it might look. A reader who sees two GPQA rows without
scores will conclude the model cannot do GPQA. Neither row said anything about
the model, and the two say *different* things about the setup:

| Row | Cause | Kind | Can it ever run here? |
|---|---|---|---|
| `meta_gpqa_cot` | Meta eval harness builds a `<repo>-evals` dataset name and accepts only the Llama-3.1 / 3.2 Evals collection | **Structural** | **No — permanent for this and every non-Llama model.** Needs an upstream TTI change. |
| `gpqa_diamond_cot_zeroshot` | `Idavidrein/gpqa` is a gated dataset; the account lacked access | **Access permission, since resolved** | **Yes — it runs.** Access granted, re-run against the same live server: **56.1 %**, 198/198, exit 0. |

The same distinction applies to the IFEval pair, and there it is even starker:
`meta_ifeval` is permanently blocked for exactly the same structural reason,
while the generic `ifeval` ran fine and scored **81.1 % / 87.1 %**. In both
pairs the `meta_*` row is a statement about TTI's Llama-only eval path, and the
generic row is the one that says something about this model.

### Did the `meta_*` and generic variants diverge?

**They could not be compared, because neither `meta_*` row ever executed.** The
scoreboard:

| Benchmark | Meta variant | Generic variant | Net measurement |
|---|---|---|---|
| IFEval | `meta_ifeval` — failed, structural (Llama-only) | `ifeval` — **81.1 % / 87.1 %** | **covered** |
| GPQA | `meta_gpqa_cot` — failed, structural (Llama-only) | `gpqa_diamond_cot_zeroshot` — **56.1 %** after the access grant | **covered** (on the re-run) |

Registering the generic pair alongside the `meta_*` pair is what saved both
measurements: without it the structural Llama-only block would have left this
release with no IFEval and no GPQA number at all.

Even had the `meta_*` rows run, their numbers would not have been directly
comparable with the generic ones: the Meta variants use Meta's own prompt
formatting, few-shot construction and answer-extraction regexes, and are scored
on different `result_keys` (`meta_gpqa_cot` on `exact_match,strict-match`,
`gpqa_diamond_cot_zeroshot` on `exact_match,flexible-extract`).
## Benchmark rows, judged

All **19** sweep points completed every request they issued (19/19 points,
0 failed requests). TTI grades none of them: no perf targets are configured for
these shapes, so every row is `NA (ungraded)` and the Benchmarks category reports
`🟨 NA (0/19 passed, 19 NA)`. That is a missing-target condition, not a defect.

### The 6 structured-output points never ran, and now there is a mechanism

The 6 structured-output points that `get_benchmark_config` generates did **not**
run: the release workflow emitted `✅ task=llm_benchmark blocks=19` and moved
straight to `spec_tests`. Only the text sweep executed.

**This is a harness finding, not a disclosure.** `llm_module/benchmark_configs.py:57-61`
filters the whole parameter set down to text before anything else happens:

```python
text_params = [
    params
    for task in benchmark_config.tasks
    for params in task.param_map.get(device, [])
    if params.isl is not None
    and params.osl is not None
    and params.task_type == "text"
]
```

Every subsequent structure in the function — `targets_by_shape`, the dedup
`seen` set, the `LLMRunConfig` list — is built from `text_params` alone, so a
param carrying `task_type="structured_output"` can never reach a run config.

Both sides are verifiable without a device. `get_benchmark_config` for this
model's spec returns **3 tasks** whose params are **19 `text` + 6
`structured_output`** — the 6 come from `BenchmarkTaskStructuredOutput`
(`reference_config/benchmarking/benchmark_config.py:77`), built at `:690-707`
one per `STRUCTURED_OUTPUT_PAIRS` entry. `tests/test_benchmark_config.py:88`
asserts the 3-task shape, and the structured-output script is still fetched into
the benchmark venv (`workflows/workflow_venvs.py:487-500`). So generation,
testing and venv setup all still expect structured output; only the config
builder silently drops it.

Nothing about this is specific to our model: **the filter drops structured-output
params for _every_ model in TTI 0.20.0.** It belongs alongside Findings 2 and 3
as a harness defect that a release run will not tell you about — the report
simply says `blocks=19` and no row is ever missing, because no row was ever
created. Recorded, not waived, and not ours to fix.

### The largest point is a completed row with a bad number — see Finding 5

The largest point, `isl=131072`, **completed** (1/1, 0 failed, valid output) with
a **94.4-minute TTFT** (5,662,485.6 ms). That is a real result, not an artefact,
and it is the subject of **Finding 5**: long-context prefill on this port is
severely superlinear — 2x the input costs 6.11x the wall clock, 3.05x worse per
token than `isl=65536`. It is a model-side performance defect, disclosed there
rather than explained away here.

Two things about *how to read the number* belong with the table, and neither
excuses it.

**First, it is a first-touch measurement with no warm counterpart.** For every
input length the sweep runs `max_concurrency=1` **first** and the concurrent
point **second**:

| isl | c=1 TTFT (cold, runs first) | concurrent TTFT (warm) | ratio |
|---|---|---|---|
| 128 | 0.4 s | 5.4 s (c=32) | 0.07x |
| 1024 | 5.0 s | 31.0 s (c=32) | 0.16x |
| 2048 | 9.4 s | 60.6 s (c=32) | 0.16x |
| 4096 | 14.3 s | 121.7 s (c=32) | 0.12x |
| 8192 | 49.9 s | 242.4 s (c=31) | 0.21x |
| 16384 | 100.2 s | 249.6 s (c=15) | 0.40x |
| **32768** | **394.2 s** | **263.5 s (c=7)** | **1.50x** |
| **65536** | **926.8 s** | **276.9 s (c=3)** | **3.35x** |
| **131072** | **5662.5 s** | **— none in sweep —** | **n/a** |

At small inputs the single-request point is far *faster* than the concurrent
one, as expected when 32 requests queue against 32 slots. From `isl=32768` the
relation **inverts**: one request alone becomes slower than three or seven
served together. A single request cannot be slower than three unless it pays
something the later run does not, and the later run's only advantage is running
second at the same shape. `isl=131072` is the only point in the sweep with no
concurrent counterpart, so **the steady-state cost at that length was never
measured** — the 94.4 min is the cost of the first request at a previously
unseen shape.

**Second, that does not make it acceptable.** Even the warm points are
superlinear, and 94.4 minutes is what a user issuing their first 131072-token
request actually experiences. The honest summary is: *131072 is served correctly
and uncapped; a first request at that length takes 94 minutes; the warm cost is
unmeasured.* Quoting either "94 minutes" as the steady-state latency or
dismissing it as "just warm-up" would be wrong.

### Decode is flat and matches stage 09

TPOT is essentially constant across the whole sweep — **230.0-289.7 ms** from
isl=128 to isl=131072, and 251.3-275.3 ms at concurrency. That reproduces the
stage-09 configured-width characteristic (a 32-wide decode graph costs ~230 ms
per step with one row live and ~269 ms with 32) and confirms decode cost tracks
the configured slot count rather than context depth. Peak decode throughput is
**120.9 tok/s** at isl=128/osl=1024/c=32.
## Gates

Both stage gates were materialised from
`origin/agentic-research/hous/multigoal-claude`
(`ecd7c64d0ff7f733a4fb3da852b8a137877aa617`) into a scratch root whose `models`
entry is a symlink to the real tree, so the relative `MODEL_DIR` the checks
expect resolves correctly **and no `.agents/` directory was ever created in the
tt-metal worktree**. Runner: `gates/run_gates.sh` (`GATES_ROOT` selects the
scratch root); output archived at `gates/gate_output.txt`.

**Re-run in full after the stage-10 review fixes**, from a freshly materialised
root, with no device involved.

```
SUMMARY: stage_check=0 context_contract=0 published_figures=0
```

| Gate | Result | Key line |
|---|---|---|
| `10-tti-release.check.sh` | **exit 0** | `Autoport implementation check passed: models/autoports/qwen_qwen3_coder_30b_a3b_instruct found in 1 copied TTI artifact field(s).` and `TTI release evidence present ... (1 release report(s))` |
| `check_context_contract.py --stage tti-release --require-contract` | **exit 0** | `Context contract OK for models/autoports/qwen_qwen3_coder_30b_a3b_instruct: target=262144, supported=262144 (full HF context).` |
| `probes/check_published_figures.py` (stage-local) | **exit 0** | 163 checks passed, 0 failed; coverage boundary leaves **0** numeric tokens undeclared across all three scanned documents (`RUN_NOTES.md`, `stall/isl131072_stall_evidence.txt`, `reports_output/README.md`) |

The context-contract checker emits `ADVISORY CONTEXT CAP` lines. Every one with
a real source points at **stage 08/09** documents
(`doc/vllm_integration/work_log.md`, `doc/optimized_vllm/logs/…`,
`adapter_contract_probe.py`) quoting the 2-layer adapter probe's
`--max-model-len 4096`. They are advisory, they pre-date this stage, and nothing
in `doc/tti_release/` caps context — this stage's own artifacts all carry
262144. No critical `CONTEXT CAP` was raised.

Two of them cite `doc/tti_release/RUN_NOTES.md` — that is *this paragraph*,
which mentions the string `4096`; the checker matches text, not configuration.
The archive itself is a third such source, and a self-feeding one: the checker
scans every document under the model directory, `gates/gate_output.txt`
included, so each archived advisory reappears as a new advisory next run and the
file grows without bound. **40 such self-referential lines are elided from the
archive**, which says so in its own header; every advisory with a real source is
kept verbatim.

### The published-figures checker

`probes/check_published_figures.py` follows the same coverage-boundary mechanism
as the stage 06-09 checkers: every figure-shaped number in the scanned documents
is either re-derived from an artifact or named in `probes/uncovered.json` with a
reason, and anything in neither set fails the gate. It re-derives, among others,
every eval score from its `results_*.json`, the whole mbpp truncation analysis
and its 256-token replay from `evals/eval_samples_derived.json`, every benchmark
TTFT and the entire cold/warm ratio table from the 19 `bench/*.json`, the
acceptance verdict from the copied report, the shipped eval settings and the
`known_issues` waiver from `spec/tti_catalog_edits.patch`, and the `cli_args`
key count from the copied runtime spec.

**Three boundary defects were found in review and fixed; they are worth more
than the numbers they guarded.**

1. **The boundary scanned only `RUN_NOTES.md`.** `stall/isl131072_stall_evidence.txt` —
   the primary evidence for the one model-side defect in this handoff — sat
   entirely outside it, and so did the report annotation. The checker now scans
   **all three** documents (`SCANNED` at the top of the file names them), and
   the stall file's own figures are re-derived in section 8c.
2. **A declared token defeated the boundary by collision.** `uncovered.json`
   carried `"90": "seconds between the two --locals samples"` — a reason that
   matched nothing in any current document. It was nonetheless silently covering
   the *checker's own pass count* in the gate table below, which is how "90
   checks passed" survived a rewrite that had long since carried the real count
   past it. Two fixes:
   a declared token that appears in **none** of the scanned documents is now a
   **gate failure**, not a `[note]`; and the checker now prints its own
   `N checks passed, M failed` line and asserts that the gate table quotes it,
   so that number can no longer be hand-maintained.
3. **The checker hard-coded a wrong figure to make itself agree.** The
   live-chunk-tensor count at isl=131072 was published as `4095`, and the check
   read `n = (isl // 32) - (1 if isl == 131072 else 0)` — a special case whose
   only purpose was to reproduce the published number. The correct count is
   **4096** (`ceil(131072 / EXPERT_CHUNK_SIZE)`, and `EXPERT_CHUNK_SIZE` is now
   read out of `tt/optimized_decoder.py` rather than typed in); `4095` was also
   inconsistent with the `2048` quoted for 65536 in the same sentence. Figure
   corrected, special case deleted.

It had already earned its place twice before that: it caught the meta_* timing
figure being written as "~70 minutes" when the logs said 72, and it caught the
concurrent TPOT floor being published as 259.6 ms when the artifacts said
251.3 ms. Both were fixed rather than annotated.

The `UNCOVERED` list is deliberately explicit about the weakest evidence in this
document — the five figures from the original 256-token mbpp run (27.6, 336,
164, 138, 84.1) are declared uncovered *because lm-eval overwrote their samples
file and the attempt-1 log never recorded the score*, not because checking them
was inconvenient.
### The repo's own pre-commit hooks edited the copied evidence

An evidence-handling finding, and the gate caught it rather than a person.

`doc/tti_release/` publishes thirteen files that are verbatim copies of
tt-inference-server's own output. The point of copying rather than paraphrasing
is byte-identity: it proves we published TTI's artifact and not our retelling of
it, and this checker pinned a sha256 on some of those copies to keep it that
way.

On the first commit attempt, tt-metal's own `.pre-commit-config.yaml` hooks
rewrote them before the commit could be made:

* **`end-of-file-fixer`** appends a final newline to any file lacking one. TTI
  writes its reports, specs and eval JSONs without a trailing newline, so ten of
  the thirteen copies gained one byte each. (One was already
  newline-terminated and is untouched.)
* **`trailing-whitespace`** strips trailing blanks from every line. It reached
  the remaining two copies — the release run logs, which each lost a single
  trailing space — and four of this stage's own captured logs.
* **`black`** reformatted the checker itself.

The checker failed the commit and named the copies whose digests it had pinned.
It did exactly what it was built to do. But the ones it named were only the ones
that happened to be pinned: ten further copies had been rewritten with nothing
watching them. That is the part worth recording.

**What was verified before anything was re-pinned.** Every affected file was
diffed against its original under
`/home/raahem/tt-inference-server/workflow_logs/`, with trailing whitespace and
the final newline normalised away on both sides. All thirteen copies are
identical to TTI's originals in content — the hooks changed whitespace and
nothing else. The same comparison was run across this stage's own derived
benchmark and eval JSONs and its captured logs: in every case the change is a
final newline or a stripped trailing blank, and no value, key, ordering or line
moved.

**How the guarantee was restated rather than dropped.**
`.pre-commit-config.yaml` is a core repo file and not ours to edit, and
suppressing the hooks with `--no-verify` would trade a real guarantee for a
green commit. So the claim was narrowed to what is actually true, and the check
was made stronger rather than weaker:

* The copies are no longer claimed to be byte-identical to TTI's output. They
  are claimed to be **identical to TTI's original except for trailing
  whitespace normalised by this repo's hooks** — and that is the claim the gate
  now tests, not a disclaimer sitting next to an untested one.
* The pinned digests were re-taken from the post-hook bytes and are still
  asserted on every run, so any later edit to a copy — by a person or by a tool
  — still fails the gate.
* On top of that, when the originals are reachable the checker compares each
  copy against its original modulo that same whitespace transformation. A copy
  whose *content* was altered now fails **even if someone re-pinned its
  digest**. That comparison stays silent when it succeeds: the originals live on
  the machine the release was run from, and a pass count that depended on where
  the gate ran would break the count assertion above.
* The pinned set grew from three copies to all thirteen, plus the catalog patch.
  The ten silent rewrites are the whole argument for that.

**One file was repaired rather than accepted.** `spec/tti_catalog_edits.diff`
is a unified diff, and a unified diff encodes a blank context line as a line
containing a single space. `trailing-whitespace` ate three of them. Both
`git apply` and GNU `patch` still accept the result, so nothing substantive was
lost — but a patch file whose bytes have been quietly rewritten is precisely the
artifact that should not be. `.pre-commit-config.yaml` **already** excludes
`\.patch$` from both offending hooks, so the file was renamed to
`spec/tti_catalog_edits.patch` and its original bytes restored. That reuses the
repo's own existing escape hatch instead of asking for a new one, and it is why
this one copy is still byte-exact.

## TTI friction found by the release run itself

The registration-stage frictions are listed further down. These are the ones
that only appeared once a full `--workflow release` actually ran, and they are
the ones most likely to bite the next non-stock model.

- **`meta_ifeval` / `meta_gpqa_cot` are Llama-only by construction and fail
  quietly.** Full mechanism in Finding 2. The friction is not that they fail —
  it is that the *cause* is logged as a warning at t+2 s and the *symptom* is an
  unrelated-looking `Tasks were not found` 72 minutes later. Any non-Llama model
  registered with the default task list will hit this and will find it hard to
  diagnose. A useful upstream fix would be to skip the `meta_*` tasks outright
  when `prepare_meta_eval` rejects the dataset, rather than letting them run
  into an empty `--include_path`.

- **A failing eval task does not reach the workflow exit code.** Individual
  `lm_eval` invocations run with `check=False`, so both `meta_*` crashes were
  logged and stepped over, and the `evaluation` task still reported success.
  `report_module/acceptance_criteria.py:115-138` (`task_failure_blockers`) exists
  specifically to stop a crash being "laundered into a silent PASS", but it only
  sees *task-type* exit codes, not per-eval-task ones — so a crash inside the
  evaluation task is invisible to it. The consequence is that **a release report
  can read PASS while a third of the registered eval rows never executed.**
  The `known_issues` waiver added afterwards does not close this hole either —
  it labels the two rows, it does not make the exit code see them. They are
  named as FAILED in the summary table at the top of this document.

- **Copying a sibling `EvalConfig` propagates its overrides, including ones that
  contradict TTI's own documented default.** The `Qwen2.5-Coder-32B-Instruct`
  entry sets `batch_size=16`, directly against the comment at
  `reference_config/evals/eval_config.py:189-194` explaining why the default is
  1. "Copy the nearest existing model" is the obvious registration strategy and
  it is what propagated both Finding 1 and Finding 3 into this model.

- **lm-eval writes every task's samples to the same `--output_path`, with no
  run-scoped subdirectory.** Re-running a task overwrites the previous run's
  `samples_*.jsonl` and `results_*.json` in place. This destroyed the primary
  evidence for the 27.6 % attempt: by the time the fixed run finished, the
  256-token samples were gone. Anyone diagnosing an eval regression should copy
  the samples out *before* re-running. (This stage recovered by replaying the
  256-token ceiling from the 2048-token completions — see Finding 1 — but that
  only worked because decoding was greedy.)

- **lm-eval's client-side `max_length` defaults to 2048** regardless of what the
  server advertises. With `max_gen_toks=2048` that leaves zero prompt budget.
  It is pinned in `model_kwargs` here; a model registered without that pin would
  silently truncate long prompts on the client.

- **`run.py` mints a debug JWT and echoes it in command lines even under
  `--no-auth`.** Redacted in every log copied into this directory.

## Cleanup

Performed after the GPQA re-run finished, at 05:2x UTC.

| Check | Result |
|---|---|
| vLLM server (launcher, API server, EngineCore) | **stopped** — SIGTERM to the launcher (`run_vllm_server`, pid 4085084); all four processes exited within 20 s |
| Processes holding `/dev/tenstorrent/*` | **none** |
| Residual `vllm` / `EngineCore` / `run_workflows` / `run.py` / `lm_eval` / `bench serve` | **none** |
| tmux sessions | **none** (`/tmp/tmux-1001/default` does not exist) |
| Port 8100 | **not listening** |
| Containers started by this work | **none** — no Docker was used at any point; the run was external/no-Docker throughout |

**A co-tenant on this host, and a correction.** An earlier draft of this
document said five `tt_studio_*` containers "predate this work". **That is
false, and it matters, because one of them maps the device and the window it
covers contains the isl=131072 outlier.** `docker inspect` says:

| Container | Created (UTC) | Started (UTC) | `/dev/tenstorrent` mapped |
|---|---|---|---|
| `tt_studio_chroma_dev` | 2026-08-18T22:35:43.579Z | 22:35:47.270Z | no |
| `tt_studio_backend_api_dev` | 2026-08-18T22:35:44.958Z | 22:35:57.839Z | **yes** |
| `tt_studio_frontend_dev` | 2026-08-18T22:35:44.974Z | 23:37:51.023Z | no |
| `tt_studio_agent_dev` | 2026-08-18T22:35:44.975Z | 23:37:51.022Z | no |
| `tt_studio_litellm` | 2026-08-18T22:35:44.976Z | 23:37:51.022Z | no |

The release run started at **22:35:11**. All five were created **32-33 s
after** it, and three of them were (re)started an hour into it, at 23:37:51 —
inside the measurement window. So they are contemporaneous with this run, not
prior to it.

**They were not created by `run.py`.** The run was external/no-Docker:
`cli_args.docker_server` and `cli_args.local_server` are both `false` in the
copied runtime spec, and `run.py` only enters `setup_host`/`ServerLaunchSpec`
when one of them is set (`run.py:1035-1060`). No Docker command appears in the
run log. Something else on the host brought TT-Studio up at that moment.

**The device-mapped one almost certainly never opened the board.** Two pieces of
evidence:

- `tt_studio_backend_api_dev` has `/dev/tenstorrent` in `HostConfig.Devices` and
  the four character devices `0`-`3` are visible inside it, but its
  `/tt_studio_persistent_volume` holds only `backend_volume`, `chroma` and
  `openwakeword_models` — **there is no `model_envs` directory**, which is where
  the backend writes a deployment's environment. It never deployed a model, so
  it had no reason to open the device.
- A `/proc/*/fd` scan **inside** the container finds no descriptor pointing at
  `/dev/tenstorrent/*`, and a host-wide `/proc/*/fd` scan finds **zero** holders
  overall.

**Honest limit of that evidence.** Both scans are point-in-time, taken during
cleanup. Nothing sampled the container's descriptors *during* the run, so a
transient open earlier in the window cannot be excluded from these artifacts
alone; the never-deployed state is the stronger argument. **A co-tenant with the
device mapped went unclassified across the measurement window that contains the
isl=131072 outlier, and that is recorded here as a limitation of this stage's
isolation rather than argued away.** The containers belong to a separate
workstream, were not started by this work, and were deliberately left running.

Nothing was committed and nothing was pushed, in either repository. The
tt-inference-server registration edits remain uncommitted on branch
`raahem/qwen3-coder-30b-a3b-tti`, `/home/raahem/vllm-tt-plugin` is byte-identical
at `bc4af2d`, and branch `raahem/qwen2.5-coder-32b-p300x2` and the prod catalog
were never touched.
## TTI catalog edits (all local, uncommitted; full patch in `spec/tti_catalog_edits.patch`)

1. `workflows/model_spec.py` (+10 lines, purely additive): new
   `qwen3_coder_30b_a3b_autoport_impl` `ImplSpec`
   (`impl_id=qwen3_coder_30b_a3b_autoport`, `code_path=models/autoports/qwen_qwen3_coder_30b_a3b_instruct`,
   `repo_url=https://github.com/tenstorrent/tt-metal`) plus one `_IMPL_REGISTRY`
   entry. No existing impl touched. There was no pre-existing `ImplSpec` with an
   `models/autoports/` code path — every one of the 13 existing impls points at
   `models/tt_transformers`, `models/demos/…`, `models/experimental/…`, or a
   plugin/media path.
2. `workflows/model_specs/dev/llm.yaml` (+36 lines, appended): new template for
   `Qwen/Qwen3-Coder-30B-A3B-Instruct`, `impl: qwen3_coder_30b_a3b_autoport`,
   `inference_engine: VLLM`, one `device_model_specs` entry `device: P300X2`,
   `max_concurrency: 32`, **`max_context: 262144`** (the full HF-advertised
   context from `doc/context_contract.json`, `capability_reduction: false` — not
   capped), `override_tt_config: {trace_region_size: 50331648,
   fabric_config: FABRIC_1D_RING}` and `MESH_DEVICE: P150x4`, mirroring the
   measured shipped serving config in `doc/optimized_vllm/README.md`.
   `min_disk_gb`/`min_ram_gb` are pinned (see "TTI friction" below).
   **The prod catalog (`workflows/model_specs/prod/llm.yaml`) was not touched.**
3. `reference_config/evals/eval_config.py` (+330 lines, additive): new
   `EvalConfig(hf_model_repo="Qwen/Qwen3-Coder-30B-A3B-Instruct", …)` with
   **six** tasks — `mbpp_instruct`, `humaneval_instruct`, `meta_ifeval`,
   `meta_gpqa_cot`, `ifeval`, `gpqa_diamond_cot_zeroshot`. **This is the shipped
   state, and it is what the release run executed.** The two coding tasks were
   originally copied verbatim from the existing
   `Qwen/Qwen2.5-Coder-32B-Instruct` EvalConfig (the closest code-instruct model
   already in the catalog); Findings 1 and 3 then changed three of the copied
   settings, and the shipped values are:

   | Setting | Inherited from the sibling (attempts 1-2) | **Shipped (this run)** | Why |
   |---|---|---|---|
   | `batch_size` | `16` | **`EvalTask` default `1`** | Finding 3 — `1 × 32 = 32` prompts in flight, matching server width |
   | `max_gen_toks` | `"256"` | **`"2048"`** (coding + `ifeval`), **`"4096"`** (`gpqa_diamond_cot_zeroshot`) | Finding 1 — the 256 ceiling truncated MBPP to 27.6 % |
   | `model_kwargs.max_length` | unset (lm-eval defaults to 2048) | **`262144`** | pinned to the context contract so the client never truncates |
   | `model_kwargs.timeout` | unset (lm-eval defaults to 1800 s) | **`7200`** | Finding 3 — a correct 2048-token answer legitimately takes ~9 min at full occupancy |

   The `meta_*` pair runs in `WorkflowVenvType.EVALS_META` with
   `include_path="work_dir"` and `apply_chat_template=False`; the other four run
   in `EVALS_COMMON` with `apply_chat_template=True`, and the coding pair adds
   `allow_code_execution=True`. Reference scores: Qwen publishes no
   MBPP/HumanEval/IFEval/GPQA number for this model (their card reports
   agentic-coding benchmarks), so `published_score`/`published_score_ref` are
   `None` (a pattern ~10 other entries already use) rather than invented, and
   there is no GPU reference run yet. Llama-3.3-70B's `meta_*` numbers were
   deliberately **not** inherited with the task definitions. This is the one edit
   with no dev-catalog escape hatch: `EVAL_CONFIGS` is built in
   `reference_config/evals/eval_config.py` regardless of `MODEL_SPECS_ENV`.
4. `workflows/model_specs/dev/llm.yaml`, `known_issues` on the P300X2
   `device_model_spec` (**added after the release run**, see "The `meta_*`
   waiver" under Finding 2): an `EVALS` waiver for `meta_ifeval` and `meta_gpqa_cot`
   citing the Finding 2 root cause. Because it postdates the run, it is **not**
   in the copied `run_specs/runtime_model_spec_release.json`, which still
   records `known_issues: []` — that spec is the record of what executed and was
   not edited.

## Wiring verification (no device involved)

Re-run against the **current** tree, so this block reflects the shipped catalog
rather than the state at first registration:

```
$ cd /home/raahem/tt-inference-server && MODEL_SPECS_ENV=dev python3 -c '...'
model_id: id_qwen3-coder-30b-a3b-autoport_Qwen3-Coder-30B-A3B-Instruct_p300x2
  model_name: Qwen3-Coder-30B-A3B-Instruct
  device: DeviceTypes.P300X2
  impl.impl_id: qwen3_coder_30b_a3b_autoport
  impl.code_path: models/autoports/qwen_qwen3_coder_30b_a3b_instruct
  max_context: 262144
  max_concurrency: 32
  in EVAL_CONFIGS: True
  eval tasks: ['mbpp_instruct', 'humaneval_instruct', 'meta_ifeval', 'meta_gpqa_cot', 'ifeval', 'gpqa_diamond_cot_zeroshot']
  known_issues: [('EVALS', 'meta_ifeval'), ('EVALS', 'meta_gpqa_cot')]
total specs: 300
```

The six-task list and the four post-Finding settings above are what the release
run executed; the `known_issues` pair is the only line here that postdates the
run. `spec/tti_catalog_edits.patch` is regenerated from this same tree and
matches it.

`model_name in EVAL_CONFIGS` is the assert at `workflows/validate_setup.py:203-208`
that fails fast for `--workflow release`; it now passes. The default prod catalog
still imports cleanly (225 specs) and does **not** expose the model.

`reference_config.benchmarking.get_benchmark_config` (the other RELEASE fail-fast)
generates 3 tasks and a 19-point text sweep from `max_context=262144`, up to
isl 131072 — no extra benchmark-catalog entry is needed.

## Server command (exactly as `doc/optimized_vllm/README.md` documents)

```bash
cd /home/raahem/tt-metal
source python_env/bin/activate
export EXTRA_MODELS_DIR=$PWD/models/autoports/qwen_qwen3_coder_30b_a3b_instruct/vllm_bundle
python -m models.common.readiness_check.run_vllm_server \
  --model-dir <scratch> \
  --hf-model Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --mesh-device P300x2 --max-num-seqs 32 --max-model-len 262144 \
  --block-size 32 --port 8100 --stages serve \
  --tt-config '{"trace_region_size": 50331648, "fabric_config": "FABRIC_1D_RING"}' \
  --additional-server-args "--generation-config vllm"
```

`--max-num-seqs 32` (the README's CI-burst/serving value) is used rather than the
single-user `1`, to match the catalog's `max_concurrency: 32`. Port **8100**
(8000 is held by an unrelated process). `--model-dir` is a scratch tree, as in
stage 09. Startup to `/health` 200: ~3 min 10 s. Server log:
`logs/vllm_server.log`.

## Smoke results (the three the goal requires, in order)

1. **Health check** — `GET /health` → **HTTP 200**. `GET /v1/models` reports
   `max_model_len: 262144`, id `Qwen/Qwen3-Coder-30B-A3B-Instruct`.
   (`smoke/smoke_1_health_2_openai.txt`)
2. **One OpenAI-compatible request** — `POST /v1/chat/completions`, greedy,
   `max_tokens=48`. Returned a coherent one-sentence answer, `finish_reason: stop`,
   24 completion tokens. Chat template applied server-side by vLLM (instruct model
   asked through the chat endpoint — no raw-completion evidence used).
   (`smoke/smoke_1_health_2_openai.txt`)
3. **One small TTI benchmark with `disable_trace_capture=true`** — exit code **0**.

```bash
cd /home/raahem/tt-inference-server
python3 run.py \
  --model Qwen3-Coder-30B-A3B-Instruct \
  --impl qwen3-coder-30b-a3b-autoport \
  --tt-device p300x2 \
  --workflow benchmarks \
  --dev-mode \
  --server-url http://127.0.0.1:8100 --service-port 8100 \
  --limit-samples-mode smoke-test \
  --disable-trace-capture \
  --no-auth \
  --skip-system-sw-validation
```

`--limit-samples-mode smoke-test` reduces the generated sweep to a single point
(`select_smoke_test_benchmark_config`) **without touching the spec or the context
contract** — the loaded spec still carries `max_context: 262144`.

| | |
|---|---|
| Sweep point | isl=16, osl=4, max_concurrency=1, num_prompts=8 |
| `completed` | **8** / 8, 0 failed |
| Median TTFT | 294.82 ms |
| Median TPOT | 230.51 ms |
| Acceptance | `PASS (0 blocker(s))`; the block is `NA (ungraded)` — no perf target exists for a 16/4 shape |
| `run.py` exit | **0** |

Benchmark JSON, benchmark markdown report and report-data JSON: `smoke/benchmarks/`.
Run log (JWT redacted): `logs/tti_smoke_benchmarks.log`.

**Non-aligned prompt length:** TTI's smoke shape sends a **16-token** prompt and
asks for 4 tokens — not a chunk/page/tile/trace-aligned length, and shorter than
any length verified in earlier stages (37/131/333/1025/4097). It ran clean,
8/8 completed. No alignment workaround was applied and none was needed.

## TTI friction (things that resisted a non-stock model)

- **No autoport `ImplSpec` existed.** Every built-in impl points at a stock or
  packaged path, so a new `ImplSpec` in `workflows/model_spec.py` was unavoidable.
  `--impl` and `--model` choices are both generated from the catalog, so once the
  YAML entry and the impl exist, the CLI accepts the model normally.
- **`_eval_config_map` has no dev/prod split.** `reference_config/evals/eval_config.py`
  is a single Python list shared by both catalogs, so registering a model for
  `--workflow release` always requires editing a file that also serves prod. The
  edit here is strictly additive.
- **`ModelSpec.infer_param_count` mis-reads MoE names.** It takes the *last*
  `<n>B` in the repo id, so `Qwen3-Coder-30B-A3B-Instruct` infers **3 B** and
  derives `min_disk_gb=26`, `min_ram_gb=7.5` for a 30.5 B checkpoint. There is no
  `param_count` field on `ModelSpecTemplate`, so `min_disk_gb: 80` / `min_ram_gb: 75`
  are pinned in the YAML instead. (Upstream bug — `google/gemma-4-26B-A4B-it`
  has the same shape.)
- **`--server-url` without an explicit port silently targets port 80.**
  `ServerConnection.url_with_port` (`llm_module/config.py:68-77`) returns the base
  URL unchanged when `is_remote` is true, so `--server-url http://127.0.0.1
  --service-port 8100` polled `http://127.0.0.1/v1/models` and sat in
  `_wait_for_remote_openai_ready` until timeout — `--service-port` is ignored on
  the remote path. **Workaround: always put the port in `--server-url`.** No TTI
  code was changed for this.
- **`cli_args` is dead in this version, and its `service_port` is misleading.**
  The goal text asks for `cli_args.docker_server=false` / `local_server=false` /
  `service_port` in the spec. `ModelSpec.cli_args` is marked *"DEPRECATED - only
  used by tt-media-server"* (`workflows/model_spec.py:514-515`) and server mode
  is decided entirely by the CLI (`--server-url` is mutually exclusive with
  `--docker-server`/`--local-server`, `run.py:653`), so no `cli_args` value was
  set for this model. **Correction to an earlier draft, which claimed the copied
  runtime spec carries an *empty* `cli_args` — it does not.** `run.py` dumps its whole
  parsed argument namespace there: the copied spec's `cli_args` has **63 keys**,
  and they read

  ```
  "docker_server": false,   "local_server": false,
  "server_url": "http://127.0.0.1:8100",
  "service_port": "8000"        <-- argparse default; the run used 8100
  ```

  The two the goal names are correctly `false`, and the deprecation argument
  stands unchanged — but `service_port` is the untouched argparse default, not
  the port the run used, because `--service-port` is ignored on the remote path
  (see the `--server-url` bullet above). **Do not read a port out of this
  field.** What actually proves no server was started is the code path, not the
  spec: `run.py` only enters `setup_host`/`ServerLaunchSpec` when `docker_server`
  or `local_server` is set (`run.py:1035-1060`), and both are `false`.
- **First `--workflow benchmarks` invocation spends ~9 min** building the
  `.venv_llm_vllm` client venv (vllm 0.13.0 + torch, client-side only). Subsequent
  runs reuse it.
- `run.py` mints a debug JWT and logs it in the `vllm bench serve` command line
  even under `--no-auth`. It is redacted in the copied log.

### Cleanup after the step-1 smoke run (historical)

Server stopped with `pkill -f readiness_check.run_vllm_server`, then
`pkill -9 -f VLLM::EngineCore` and `pkill -9 -f vllm.entrypoints`. Verified after:
no vLLM/EngineCore/entrypoints processes, `fuser -v /dev/tenstorrent/*` reports no
holders, port 8100 free. No tmux sessions, no Docker containers (Docker was never
used). No device reset was needed; no ARC/ERISC/Ethernet errors occurred.

### Forward look written *before* the release run (historical — one prediction was wrong)

> Kept as written. Its prefill estimate is the mistake Finding 5 corrects: it
> read the context contract's **single-layer** `prefill_sweep_seconds` (~86 s at
> isl 131072) as if it were the end-to-end cost. The measured 48-layer prefill
> at that length was **94.4 minutes**. Multiplying a single-layer probe by
> nothing, and multiplying one layer's sampled rate by 48, were the two ends of
> the same error made twice in this stage.


Both RELEASE fail-fast asserts now pass and the client topology is proven. The
remaining risks are in the workload, not the wiring: the generated benchmark sweep
runs to isl 131072 (prefill alone is ~86 s at that length per the context
contract), `mbpp_instruct`/`humaneval_instruct` need the `EVALS_COMMON` venv and
code execution, and the release report will contain ungraded rows because this
model has no `perf_targets_map` and no published/GPU reference scores. Nothing
found so far requires a prod-catalog or vLLM-plugin edit.
