# AutoFix Report — `test_penalties[presence_penalty-1.2-repeat_trap]`

**Verdict: (c) the on-device presence-penalty implementation is correct — it computes
vLLM's own rule, on vLLM's own token set, and differs from vLLM's float32 reference only
by the bfloat16 quantization of the logits it operates on. No code change.** The
conformance row is a behavioural heuristic about *the model's answer to this prompt*, and
the arithmetic below is what settles that, not an appeal to "it's just a heuristic".

Two independent measurements carry it:

1. **The device's greedy token stream is reproduced 160/160 by vLLM's presence rule done
   in bfloat16**, including the one step where vLLM's float32 reference picks a different
   token. Every alternative rule — penalise by count, penalise prompt tokens too, penalty
   never lands — first contradicts the device at step 30, 88 and 9 respectively.
2. **A presence penalty that is an exact multiple of the logit's bf16 spacing makes the
   device bit-identical to vLLM's host sampler over 1024 greedy tokens**, 3 penalties out
   of 3. A wrong formula cannot be cured by choosing a rounder penalty.

And on the behavioural half of the question:

3. **vLLM's own float32 implementation fails the same assertion, on the same model and the
   same prompt, more often and more severely than the device path** — 1 pass in 4 against
   the device's 2 in 4, and on the greedy trial where no RNG is involved at all, the
   reference scores **0.3585** where the device scores **0.9725**.

## Starting Evidence

* Original failing check, in
  `/home/ttuser/dev/muse-glimmer/tti-release/muse-glimmer-30b/tt-inference-server`:

  ```
  .workflow_venvs/.venv_workflow_run_script/bin/python -m pytest llm_module/test_vllm_chat_completions.py \
    --output-path /tmp/ppfix --task-name vllm_chat_completions \
    --endpoint-url http://127.0.0.1:8000/v1/chat/completions \
    --model-name meta-models/Muse-Glimmer-30B -q
  ```

  `test_penalties[presence_penalty-1.2-repeat_trap-messages0]` — the last failing
  parametrization after `AUTOFIX_seeding.md`. It sends `"Write a very repetitive story."`
  twice at `temperature=0.1, max_tokens=1024, seed=1234`, once plain and once with
  `presence_penalty=1.2`, and asserts
  `unique_ratio(penalty) >= unique_ratio(base) * 0.90` where
  `unique_ratio = len(set(text.lower().split())) / len(text.lower().split())`.

* No fresh `$autodebug` pass. The caller had already established the two facts this
  report builds on, and neither is re-measured here:
  * the failure is **perfectly deterministic**, 3/3 identical texts per arm
    (`doc/tti_release/presence_penalty_repeats.json`): base 252 words / 38 unique /
    ratio 0.1508, penalty 202 / 25 / 0.1238, assertion ratio **0.8207**;
  * a greedy device-vs-host equivalence check
    (`doc/tti_release/presence_penalty_greedy_equivalence.json`): with no penalty the two
    are **byte-identical**; with `presence_penalty=1.2` they split at character 725.

* Prior stages' position on presence penalty, read and not repeated:
  `doc/vllm_integration/README.md` (the `TestPresencePenalty` bullet) and
  `doc/vllm_integration/AUTOFIX.md` round 2 — those concern a *different* test, whose
  prompt has a 3.0 logit margin no legal presence penalty can close. Treated as
  background.

* New evidence artifacts: `doc/tti_release/presence_penalty/`.

## The reference, stated exactly

`vllm/model_executor/layers/utils.py::apply_penalties` (vLLM 0.24.0), reached from
`vllm/v1/sample/sampler.py::Sampler.apply_penalties`:

```python
_,   prompt_mask = get_token_bin_counts_and_mask(prompt_tokens_tensor, ...)
obc, output_mask = get_token_bin_counts_and_mask(output_tokens_tensor, ...)
apply_repetition_penalties(logits, prompt_mask, output_mask, repetition_penalties)
logits -= frequency_penalties.unsqueeze(dim=1) * obc          # scales with count
logits -= presence_penalties.unsqueeze(dim=1) * output_mask   # flat, generated tokens only
```

with `logits = logits.to(torch.float32)` immediately before. So: presence is a **flat**
subtraction on tokens that appear **at least once in the generated output**, prompt tokens
excluded, in **float32**.

`models/common/sampling/tt_penalties.py::apply_penalties` computes, in order,
`logits -= output_mask * presence`, `logits -= output_counts * frequency`, then the
repetition scaling over `prompt_mask + output_mask`. Same token sets: `output_mask` is
`counts_sliced > 0` over `output_counts`, which `reset_output_tokens()` zeroes at every
prefill and `update_output_tokens()` scatter-adds one sampled token into per decode step —
generated tokens only. `prompt_mask` is used by the repetition term alone, as in vLLM.
`models/common/sampling/generator.py:329-330` applies penalties before sampling, matching
vLLM's penalties-then-temperature-then-top-k order.

Two differences from the reference, both examined below: the **dtype** (bf16, not fp32)
and the **order** of the three terms (vLLM does repetition first, the device does it last).

## Hypothesis Experiments

### H1 — the device applies a different penalty rule (wrong token set, or count instead of presence)

**Prediction.** If the rule is wrong, the device's token stream is explained by some rule
other than vLLM's, and it is wrong from early on — not for the first time at character 725.

**Experiment**
(`presence_penalty/presence_penalty_argmax_probe.py` → `argmax_probe.json`,
`presence_penalty/presence_penalty_arithmetic_probe.py` → `arithmetic_probe.json`).
`logprobs: true` routes a request to vLLM's host sampler on this 4-die mesh, and vLLM's
`logprobs_mode` default is `raw_logprobs` — logprobs computed *before* penalties. Two
greedy requests, same prompt, `presence_penalty=1.2`, `max_tokens=256`: one with logprobs
(host sampler, plus the raw top-20 per step and `token_ids`), one without (device
sampler). Then, teacher-forced on the host sequence, rebuild each step's decision under
five rules and score each against the **device's own** tokens.

**Result — the reference model first validates itself.** Rebuilding
`argmax(raw_logprob − 1.2·[token already generated])` reproduces the token vLLM's host
sampler emitted at **256/256** steps (`argmax_probe.json → reference_model_check`). So the
reconstruction is vLLM's arithmetic, confirmed against vLLM.

**Result — rule scoring against the device tokens** (`arithmetic_probe.json →
rule_vs_device_tokens`, 160 scored steps up to and including the divergence at step 159):

| rule | matches device | first contradicts device at |
|---|---|---|
| `presence` in bf16, device tiebreak | **160 / 160** | — |
| `presence` in fp32 (vLLM's own) | 159 / 160 | step 159 |
| `count` (frequency's rule) | 137 / 160 | **step 30** |
| `prompt ∪ output` (repetition's token set) | 158 / 160 | **step 88** |
| no penalty at all | 145 / 160 | **step 9** |

**Verdict: refuted.** Every wrong-rule candidate contradicts the device long before the
observed divergence — 129, 71 and 150 steps earlier — and disagrees with it 2 to 23 times
overall. The only rule that survives to step 159 is vLLM's own; the only rule that survives
step 159 is vLLM's own done in bf16.

### H2 — the divergence at step 159 is bfloat16 rounding at a near-tie

**Prediction.** If it is precision, the two candidates are separated by a fraction of one
bf16 ULP of the logits, and the device's choice is what the *same* rule gives when the
subtraction is rounded onto the logit's bf16 grid.

**Experiment.** Same probe. `logprob = logit − logsumexp`, so within-step logprob
*differences* are logit differences and reveal the logits' bf16 spacing directly; at step
159 every difference among the top candidates is an exact multiple of **0.125**, the bf16
ULP on the `[16, 32)` binade (`arithmetic_probe.json → logit_grid`: 172/256 steps at
rank 2, 235/256 at a 0.0625 grid — the top logits sit in `[8,32)`).

**Result — the whole flip, in closed form** (`arithmetic_probe.json →
at_the_divergent_step._bf16_trace`). Write `L` for the logit of `"Probably"` (id 180574),
generated once already. `"Perhaps"` (id 89665) has never been generated and sits at
exactly `L − 1.25`.

| | penalty applied to `"Probably"` | `"Probably"` | `"Perhaps"` | winner |
|---|---|---|---|---|
| vLLM, fp32 | 1.2 | `L − 1.2` | `L − 1.25` | **Probably**, by 0.05 |
| device, bf16 | 1.203125, then rounded | `L − 1.25` | `L − 1.25` | **exact tie** |

* `float32(1.2)` stored in the bf16 penalty buffer becomes `1 + 26/128 = 1.203125`.
* `L` is a bf16 grid point, so `bf16(L − 1.203125) = L − round_to_grid(1.203125, 0.125)`.
  `1.203125 / 0.125 = 9.625 → 10`, so the **effective device penalty is exactly 1.25**.
* `L − 1.25` is precisely where `"Perhaps"` already was. The two candidates tie *exactly*.
* Greedy ties on device are broken by lowest global token id
  (`TTSampling._adjust_values_for_tiebreak`, the `k == 1` path). `89665 < 180574`, so the
  device emits `"Perhaps"`. That is the observed token.

The general statement: the device applies `round_to_grid(bf16(P), ULP)` rather than `P`,
so the penalty is quantized to the logit's own bf16 spacing, with error at most half an
ULP — ≤ 0.0625 here. For `P = 1.2` on the `[16,32)` binade the error is exactly **+0.05**,
which is exactly the margin the flip needed.

**Verdict: verified.**

### H3 — falsifiable prediction: a grid-aligned penalty must make the two paths identical

H2 is an emulation, so it was made to stick its neck out. If the only difference is
rounding, then a penalty that is an exact multiple of the logit ULP makes `logit − penalty`
exactly representable, the device's arithmetic exact, and the two paths **must** agree
token for token. A wrong formula predicts nothing of the kind — it would be just as wrong
at 1.25 as at 1.2.

**Experiment** (`presence_penalty/presence_penalty_grid_prediction.py` →
`grid_prediction.json`): greedy, `max_tokens=1024`, device vs host, six penalties.

| presence_penalty | bf16 value | multiple of 0.125 | device vs host |
|---|---|---|---|
| 0.5 | 0.5 | yes | **identical, 1024/1024 tokens** |
| 1.25 | 1.25 | yes | **identical, 1024/1024 tokens** |
| 2.0 | 2.0 | yes | **identical, 1024/1024 tokens** |
| 0.7 | 0.69921875 | no | identical (no near-tie met in 1024 tokens) |
| 1.1 | 1.1015625 | no | diverges at step **323** |
| 1.2 | 1.203125 | no | diverges at step **159** |

**Verdict: verified, 0 falsifications.** The prediction is one-sided by construction — an
unaligned penalty *can* diverge, it does not have to, because rounding only flips a
decision when a near-tie falls inside the rounding window; 0.7 met none in 1024 tokens.
The falsifiable half is the aligned half, and it held 3/3 over 3072 greedy tokens.

### H4 — the term ordering (device: presence, frequency, repetition; vLLM: repetition, frequency, presence)

This *is* a real difference from the reference, and it is recorded here as a finding rather
than left implicit. It cannot affect this row: the failing request sets only
`presence_penalty`, so `frequency_penalty = 0` makes its term an exact `subtract(logits, 0)`
and `repetition_penalty = 1.0` makes its term an exact `multiply(logits, 1.0)` — both
bit-exact identities in bf16. Ordering is only observable when repetition is combined with
presence or frequency in one request, which no row of this conformance file does, and
`TestRepetitionPenalty` and `TestFrequencyPenalty` both pass. Left unchanged: changing it
is a semantic change to shared code across every model that uses
`models/common/sampling/`, with no failing test to justify it.

**Verdict: real, out of scope for this row, recorded as follow-up.**

### H5 — the conformance assertion is a property of the *penalty*, so vLLM's own implementation must satisfy it

**Prediction.** If `unique_ratio(penalty) >= unique_ratio(base) * 0.90` is a property of a
correctly implemented presence penalty, then vLLM's float32 reference — reached by adding
`logprobs: true`, which routes the request to the host sampler where
`vllm/model_executor/layers/utils.py::apply_penalties` does the work and no Tenstorrent
sampling or penalty code runs at all — must satisfy it on this model and this prompt.

**Experiment** (`presence_penalty/presence_penalty_reference_behaviour.py` →
`reference_behaviour.json`). The conformance row's own payload — `temperature=0.1`,
`max_tokens=1024`, `"Write a very repetitive story."` — plus a greedy `temperature=0`
trial that removes RNG entirely, run on both samplers, scored with the test file's own
`repetition_stats`.

**Result.** `unique_ratio(penalty) / unique_ratio(base)`; the assertion needs ≥ 0.90:

| trial | device sampler (TT penalty, bf16) | host sampler (vLLM `apply_penalties`, fp32) |
|---|---|---|
| greedy, `temperature=0` (no RNG at all) | **0.9725 — pass** | **0.3585 — FAIL** |
| `seed=1234, temperature=0.1` (the row's own payload) | 0.8207 — FAIL | 2.6006 — pass¹ |
| `seed=1, temperature=0.1` | 2.7619 — pass | 0.2822 — FAIL |
| `seed=42, temperature=0.1` | 0.5364 — FAIL | 0.7231 — FAIL |
| **total** | **2 pass / 4** | **1 pass / 4** |

¹ that arm returned 17 words with `finish_reason: length`; a near-empty answer has a
trivially high type-token ratio and passes for the wrong reason. Recorded rather than
dropped.

**Verdict: the assertion is not a property of the penalty — verified.** vLLM's own float32
reference implementation fails the same assertion on the same model and the same prompt
**more often and more severely than the device path does** (1/4 vs 2/4; 0.3585 vs 0.9725
on the greedy trial, which is the clean head-to-head — same model, same prompt, same
penalty, no RNG anywhere, the sampler arithmetic the only difference). The row is also a
coin flip within each arm: the device passes at seed 1 with ratio 2.76 and fails at seed 42
with 0.54. The metric is a type-token ratio, so it tracks which of the model's answer modes
a sample lands in — a long varied story scores 0.27, a short repetitive one 0.06 — far more
strongly than it tracks the penalty.

### H6 — why presence is the one of the three penalty rows that fails

`test_penalties` runs the same prompt through `presence_penalty=1.2`,
`frequency_penalty=1.2` and `repetition_penalty=1.5`; only presence fails. That is a
structural property of the three definitions, not a property of the TT implementation
(`presence_penalty/penalty_strength.json`, logit shifts at a top logit of 20):

| occurrences so far | presence 1.2 | frequency 1.2 | repetition 1.5 |
|---|---|---|---|
| 1 | −1.25 | −1.2 | −6.67 |
| 2 | −1.25 | −2.4 | −6.67 |
| 5 | −1.25 | −6.0 | −6.67 |
| 20 | −1.25 | −24.0 | −6.67 |

Presence is the only one of the three whose strength does **not** grow with the number of
repetitions, and it is the weakest at every repetition count. On a prompt whose failure
mode is an unbounded sentence-level repeat loop, it saturates after the first occurrence of
every word in the loop and cannot break it — it can only change *which* loop the model
enters.

The conformance file's own source already says this. Line 313:

```python
if prompt_name == "repeat_trap" and penalty_param != "presence_penalty":
    ...assert most_common_penalty <= most_common_baseline, "Penalty didn't reduce repetition..."
```

The suite exempts `presence_penalty` from its "heavy repetition should decrease" assertion
on this exact prompt. The `unique_ratio` assertion at line 307 is a proxy for the same
property, and is not exempted.

## What was *not* done, and why

A unit test of `tt_penalties.apply_penalties` against a torch reference on synthetic logits
would confirm the op's semantics directly. It was not run: it needs the 4 Blackhole devices,
which the release server holds, and it can only confirm what H1 already establishes
end-to-end and at the token level — the device's *emitted tokens* are reproduced 160/160 by
vLLM's rule in bf16 and contradicted by every other rule within 88 steps. Spending a server
restart on a weaker form of the same evidence was not worth it. It remains the obvious
follow-up if anyone wants the op covered by a durable regression test; there is currently
no test file for `models/common/sampling/tt_penalties.py`.

## A real numerical property, recorded (not a bug, and not fixed)

The one thing this report does *not* claim is that the device and vLLM agree bit for bit.
They do not, and the exact statement of the difference is:

> The on-device penalty is applied to bfloat16 logits and stored back into them, so the
> **effective** penalty is the requested one snapped to the logits' own bf16 spacing:
> `effective = round_to_grid(bf16(P), ULP(logit))`. The error is at most half a ULP —
> ≤ 0.0625 on the `[16,32)` binade these logits occupy, and exactly **+0.05** for
> `P = 1.2`. `presence_penalty/bf16_quantization.json` confirms this in plain torch
> bfloat16, with no device and no server: 1.2 → 1.25, 0.7 → 0.75, 1.1 → 1.125, while
> 0.5 → 0.5, 1.25 → 1.25 and 2.0 → 2.0 are exact.

That is not fixable at the penalty op. The subtraction's *accumulator* precision is
irrelevant — the result is written into a bf16 logits tensor that `TTSampling` then
top-k's, so it must land on the bf16 grid however it is computed. Removing the
quantization means carrying float32 logits through top-k, the all-gather and sampling on
every model that uses `models/common/sampling/`, which is a mesh-wide precision and
bandwidth change. No failing test motivates it: the flip it causes is a coin toss at an
exact tie, the same class of event as the greedy tiebreak that `TTSampling` already exists
to make deterministic, and it moves 1 token in 160 on the worst prompt found here and 0 in
1024 at three of the six penalties tried.

## Artifacts

| what | path |
|---|---|
| vLLM's decisions rebuilt from raw logprobs; the char-725 divergence in token/logit terms | `presence_penalty/argmax_probe.json`, `presence_penalty/presence_penalty_argmax_probe.py` |
| five candidate rules scored against the device's own tokens; the bf16 trace of the flip | `presence_penalty/arithmetic_probe.json`, `presence_penalty/presence_penalty_arithmetic_probe.py` |
| grid-aligned vs unaligned penalties, device vs host, 1024 greedy tokens | `presence_penalty/grid_prediction.json`, `presence_penalty/presence_penalty_grid_prediction.py` |
| the conformance comparison run on both samplers, 4 trials each | `presence_penalty/reference_behaviour.json`, `presence_penalty/presence_penalty_reference_behaviour.py` |
| bf16 quantization of the penalty, in plain torch, no device | `presence_penalty/bf16_quantization.json` |
| why presence is the weakest of the three penalties | `presence_penalty/penalty_strength.json` |
| final conformance run | `presence_penalty/conformance_after.log` |
| caller's prior evidence, not re-measured | `presence_penalty_repeats.json`, `presence_penalty_greedy_equivalence.json`, `presence_penalty_host_control.json`, `presence_penalty_row.json`, `presence_penalty_control.json` |

## Final Status

**No code change. Verdict (c): the implementation is correct; the conformance row is a
behavioural heuristic this model legitimately violates on this prompt.**

`git status` is unchanged from the start of this investigation apart from the new evidence
directory: the only modified tracked files remain `models/common/sampling/tt_sampling.py`
(the seeding fix from `AUTOFIX_seeding.md`, kept) and
`models/autoports/meta_models_muse_glimmer_30b/.gitignore`. `tt_penalties.py` is untouched.

Commands that prove the final state:

```bash
# the arithmetic (server must be up; no device exclusivity needed beyond the running server)
source /home/ttuser/dev/muse-glimmer/muse-glimmer_pyenv/bin/activate
cd /home/ttuser/dev/muse-glimmer/tt-metal
python models/autoports/meta_models_muse_glimmer_30b/doc/tti_release/presence_penalty/presence_penalty_arithmetic_probe.py \
  --out /tmp/arith.json          # presence_bf16_device: 160/160 vs device; count: 137; prompt+output: 158
python models/autoports/meta_models_muse_glimmer_30b/doc/tti_release/presence_penalty/presence_penalty_grid_prediction.py \
  --max-tokens 1024 --out /tmp/grid.json   # 3/3 grid-aligned penalties identical, 0 falsifications
python models/autoports/meta_models_muse_glimmer_30b/doc/tti_release/presence_penalty/presence_penalty_reference_behaviour.py \
  --seeds 1234 1 42 --out /tmp/ref.json    # host (vLLM fp32) 1 pass/4, device 2 pass/4

# the conformance file
cd /home/ttuser/dev/muse-glimmer/tti-release/muse-glimmer-30b/tt-inference-server
.workflow_venvs/.venv_workflow_run_script/bin/python -m pytest llm_module/test_vllm_chat_completions.py \
  --output-path /tmp/ppfix --task-name vllm_chat_completions \
  --endpoint-url http://127.0.0.1:8000/v1/chat/completions \
  --model-name meta-models/Muse-Glimmer-30B -q
```

Outcome, `presence_penalty/conformance_after.log`, run after all of the above:

```
FAILED llm_module/test_vllm_chat_completions.py::test_penalties[presence_penalty-1.2-repeat_trap-messages0]
  - AssertionError: Test failed: Penalty unexpectedly reduced diversity.
1 failed, 21 passed in 367.67s (0:06:07)
```

**21 passed, 1 failed — unchanged, and that is the reported result, not a shortfall.** No
code changed, so nothing could have moved; `test_penalties` stays 8/9 and the file stays
21/22. Raising it to 9/9 and 22/22 would require either editing the test (out of scope, and
explicitly forbidden for this task) or changing model behaviour to satisfy a type-token
ratio that vLLM's own implementation of the same penalty fails harder. The qualitative
regression check (`bench/qualitative.sh`) was not re-run: with no edit under `models/`
there is nothing for it to regress against, and running it would need the release server
taken down.

The release server was left running and healthy in tmux session
`tti-release-muse-glimmer-30b`, launched from
`doc/tti_release/bench/serve_release.sh`, never restarted during this investigation.

## Remaining risks

* The bf16 emulation in H2/H3 has to assume a logit ULP, because absolute logit magnitudes
  are not recoverable from logprobs (a softmax is shift-invariant; the *spacing* survives,
  the offset does not). It assumes 0.125 for the top candidates, which the measured
  spacings support on 172/256 steps at rank 2 and which holds exactly at the divergent
  step. On steps whose top logits sit in `[8,16)` the true ULP is 0.0625; the emulation
  still reproduced 160/160, and H3 does not depend on the assumption at all — 0.5, 1.25 and
  2.0 are exact on every binade in range. Low risk.
* No device-side unit test of `tt_penalties.apply_penalties` against a torch reference (see
  above for why). The op has no test file today; adding one is the obvious follow-up.
* The term-ordering difference (H4) is real and untested by anything. It becomes observable
  only when `repetition_penalty != 1.0` is combined with a non-zero presence or frequency
  penalty in the same request. Worth a regression test before anyone relies on that
  combination.
* Four trials per arm in H5. Enough to show the assertion is a coin flip and that the
  reference fails it too; not enough for a pass-rate estimate. The greedy row, where the
  reference fails 0.3585 against the device's 0.9725 with no RNG anywhere, is the part that
  does not depend on sample size.
