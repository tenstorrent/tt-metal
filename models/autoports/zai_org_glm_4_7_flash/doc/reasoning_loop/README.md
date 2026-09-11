# GLM-4.7-Flash reasoning loop: investigation report

Target: `zai-org/GLM-4.7-Flash` (`Glm4MoeLiteForCausalLM`, 30.6B total / ~3.6B
active, 47 layers, routed experts at `bfloat4_b`), **one Blackhole p150-class
chip**, served through the published v5.1 container
`tt-model/glm-4-7-flash-p150:1bfc93a8a0ce` (`sample_on_device_mode: all`,
`--reasoning_parser glm47`, max_model_len 202752, max_num_seqs 32).

**This is an investigation report, not a stage report.** No source was changed.
Nothing here is a fix; it is the evidence a fix would be built on. The
motivating symptom was a field report of roughly 10% of sampled requests
returning degenerate output, plus one empty sample in a 3-sample IFEval smoke
eval run through `tt-inference-server`.

**Since this investigation, a mitigation has shipped** (`c57266658df` on
`ttmodelmanager/glm47-flash-reasoning-off`, published as
`tt-model/glm-4-7-flash-p150:c10b7c9eaed8`): reasoning is off by default
(`--default-chat-template-kwargs '{"enable_thinking": false}'`) and every
response is capped at 32,768 tokens
(`--override-generation-config '{"max_new_tokens": 32768}'`). This closes the
failure for the default path, because there is no `<think>` block left to loop
in. It does not touch the underlying defect: opt-in reasoning
(`chat_template_kwargs.enable_thinking=true`) still reaches it, now measured on
a second prompt class. See "GPQA cross-validation and the shipped cap" below.

**Headline finding, and it moved twice during the investigation.** The first
conclusion (batch-dependent sampler corruption) was wrong and is retracted
below with the evidence that killed it. The second conclusion (the loop is the
checkpoint's own attractor, so nothing on our side causes it) is **also too
strong**: the bf16 reference completes this exact prompt correctly under greedy
decoding, where the device loops forever. What survives is narrower and is
stated in "Where this leaves the diagnosis".

## Headline

| | value | source |
|---|---|---|
| Device, greedy, IFEval doc0 | **loops from ~token 460, runs to the 16384 cap, `message.content` empty** | `logs/greedy_identity.log` |
| Device greedy alone vs in a batch of 3 | **byte-identical** (same 70,667-char trace, same hashes) | `logs/greedy_identity.log` |
| HF bf16 reference, greedy, same prompt, CPU | **completes: 2973 tokens, `</think>` at 2389, EOS, valid answer** | `logs/hf_freerun_4000.log` |
| Teacher-forced agreement, device's first 804 greedy tokens | 85.6% top-1, in family with the bring-up baseline (0.79 to 0.85) | `logs/teacher_forced.log` |
| Device tokens outside the reference top-p 0.95 nucleus | **1 / 804** | `logs/teacher_forced.log` |
| Cheapest mitigation that works under both greedy and sampling | `enable_thinking=false`: 420 to 465 tokens, all constraints met, 32.5 tok/s | `logs/loop_breakers*.log` |
| Cost of the on-device penalties path | **32.0 to 23.5 tok/s, a 27% decode regression** | `logs/penalty_perf_ab.log` |
| GPQA, reasoning enabled, 10 docs, paired against reasoning-off on the same docs | **2/10 hit the same empty-at-cap failure**; scored 5/10 vs reasoning-off's 6/10 | `logs/gpqa_reasoning_on.log`, "GPQA cross-validation" below |
| The shipped 32,768 cap (`--override-generation-config`) | **unconditional**: `vllm/entrypoints/openai/chat_completion/serving.py`'s `get_max_tokens()` takes `min()` over it and the client's own `max_tokens`, so a request asking for 65,536 is silently truncated | verified in vLLM source, see below |
| Full ifeval + gpqa, shipped default, 739 samples total | **0 empty (the specific fix holds), 39/739 (5.3%) degenerate repetition in non-empty content (the defect does not)** | "Full-scale confirmation" below |

## The defect

On constraint-heavy prompts, GLM-4.7-Flash enters a verbatim draft loop inside
its `<think>` block, never emits `</think>`, and runs to whatever token cap the
request carries. Because the server runs `--reasoning_parser glm47`, the whole
generation is classified as reasoning, so `message.content` comes back **empty**
and `finish_reason` is `length`. A client sees a request that burned its entire
budget and returned nothing.

The reproducer is IFEval `doc_id 0`:

```
Write a 300+ word summary of the wikipedia page
"https://en.wikipedia.org/wiki/Raymond_III,_Count_of_Tripoli".
Do not use any commas and highlight at least 3 sections that has titles in
markdown format, for example *highlighted section part 1*, *highlighted
section part 2*, *highlighted section part 3*.
```

Three simultaneous constraints (length floor, a forbidden character, a
structural requirement) drive the model into a plan/draft/self-check cycle. The
trace opens normally, drafts a summary, reaches a self-check, and then re-enters
the same draft. The loop body is stable and literal:

```
    *   *Okay, I will just write a very long sentence structure without commas
        to get the word count up.*

    *   *Final Attempt:*
        Raymond III of Tripoli was a prominent nobleman who served as the Count
        of Tripoli from 1081 until his death in 1105. ...
        *He participated in the First Crusade.* He was a close ally of Godfrey
        of Bouillon. He fought bravely at the Battle of Antioch. ...
```

A 120-character window from that paragraph recurs **10 times** in the trace's
first sampling pass and the trace's tail is still inside it at token 16,384. The
first occurrence is at character 1,729 of the trace, roughly generated token
460, about 2% of the way into the eventual output.

Under greedy decoding on device this happens **100% of the time** on this
prompt: two independent runs produced byte-identical 70,667-character traces.

## What was ruled out, with the evidence

### Batching and the on-device sampler (this retracts an earlier conclusion)

The first hypothesis, recorded here because it was wrong and someone will
re-derive it, was batch-dependent corruption. It came from a real observation:
the same prompt at `temperature 1.0, top_p 0.95, seed 42` returned a good
1,287-character answer when sent alone (twice, byte-identical) and returned
nothing when sent concurrently with two other IFEval prompts.

Running the same comparison **under greedy** killed it:

| run | finish | tokens | content | `content_sha256[:12]` | `reasoning_sha256[:12]` |
|---|---|---|---|---|---|
| greedy doc0 **alone** | `length` | 16384 | 0 chars | `e3b0c44298fc` | `9b1201f4489c` |
| greedy doc0 **in a batch of 3** | `length` | 16384 | 0 chars | `e3b0c44298fc` | `9b1201f4489c` |

Identical to the byte, 70,667 characters of reasoning both times. The forward
pass is batch-invariant. The earlier sampled result was **two different random
streams**, not corruption: a seeded request's device stream shifts when the
decode batch layout changes, and one of those streams escaped the attractor
while the other did not. That behaviour is real and is tracked separately (see
"Adjacent issues"), but it is not a correctness defect and it is not the cause
of the empty responses.

### The eval harness payload shape

lm-eval's `LocalChatCompletion._create_payload` sends messages as
`[{"role": "user", "content": ..., "type": "text"}]` with `stop: []`, which is
not the shape a hand-written client sends. Replaying doc0 with lm-eval's exact
payload and with a plain payload produced **identical** results (7,440
completion tokens, 1,287-character content, same content head) at
`temperature 1.0, top_p 0.95, seed 42`. Not the harness.

### Token-budget exhaustion

The first theory for the empty eval sample was that IFEval's 16,384-token budget
was simply too small for this model's traces, the same failure the eval config
had already documented and fixed for GPQA by raising its budget to 65,536. That
is not it: the same prompt and parameters, sent as an isolated request,
**finished at 7,440 tokens** with `finish_reason: stop` and a complete answer.
The budget is adequate; the generation is pathological.

### The per-request seed path

`SamplingGenerator` / `SeedManager` (`models/common/sampling/generator.py`) and
the device draw (`models/common/sampling/tt_sampling.py`) were read end to end
looking for a state leak. Two facts settle it for this defect. First, the greedy
A/B above is deterministic and identical, and the greedy path does not consult
the seed stream in any way that could differ between the two runs. Second, the
failure reproduces at `temperature 0`, where `format_sampling_params` rewrites
the row to `(temperature=1, top_k=1, top_p=0)` and the draw is a pure argmax.
There is no randomness left to corrupt.

Note for whoever reads the earlier session notes: **vLLM 0.25 returns the
reasoning trace as `message.reasoning`, not `message.reasoning_content`.**
Several early readings in this investigation reported "0 characters of
reasoning" purely because of that key mismatch. The traces were always there.

## Teacher-forced reference check

**Question:** is the device rendering the reference model faithfully, or is it
producing tokens the reference would never produce?

**Method:** take the device's greedy trace for doc0, feed prompt plus the first
3,200 characters (804 generated tokens) to HF `AutoModelForCausalLM` in bf16 on
CPU in a single forward pass, and ask at every generated position whether the
reference's argmax equals the token the device actually emitted. Script:
`repro_scripts/teacher_forced_ref.py`. Prompt rendering was verified identical
at 80 tokens on both sides.

**Result: 688 / 804 = 85.6% top-1 agreement.** The bring-up baseline for this
model is top-1 0.79 (prefill check) to 0.85 (teacher forcing) against the same
reference, so this is in family and not a regression.

| token band | top-1 agreement | mean p_ref(device token) | mean p_ref(top-1) |
|---|---|---|---|
| 0 to 100 | 0.910 | 0.883 | 0.908 |
| 100 to 250 | 0.800 | 0.690 | 0.747 |
| 250 to 400 | 0.847 | 0.727 | 0.766 |
| 400 to 550 | 0.840 | 0.732 | 0.777 |
| 550 to 804 | 0.882 | 0.785 | 0.825 |

The loop entry is at roughly token 460. **There is no agreement dip there**: the
250-to-400 band (0.847) and the 400-to-550 band (0.840) sit at the run average.
Whatever steers the device into the loop, it is not a burst of bad tokens at the
entry point.

Every disagreement is a low-confidence near-tie, not a wild pick:

| pos | device token | reference token | p_ref(device) | p_ref(top-1) |
|---|---|---|---|---|
| 15 | `' Material'` | `':**'` | 0.377 | 0.621 |
| 19 | `' for'` | `' "'` | 0.438 | 0.438 |
| 34 | `'Length'` | `'Task'` | 0.243 | 0.661 |
| 71 | `' using'` | `' in'` | 0.262 | 0.711 |
| 73 | `' ('` | `' format'` | 0.235 | 0.639 |
| 81 | `'*'` | `' part'` | 0.181 | 0.813 |
| 91 | `' Content'` | `' Material'` | 0.325 | 0.535 |
| 95 | `' Sandbox'` | `' Outline'` | 0.327 | 0.327 |
| 96 | `'/S'` | `'/'` | 0.093 | 0.253 |
| 108 | `' ('` | `' of'` | 0.069 | 0.837 |
| 109 | `'also'` | `'Ray'` | 0.158 | 0.429 |
| 128 | `'108'` | `'112'` | 0.093 | 0.154 |

Position 19 is an exact tie in the reference (0.438 both ways). Position 95 is
another (0.327 both ways).

Two distribution-level checks close it out:

- **1 / 804** device tokens fall outside the reference's own top-p 0.95 nucleus.
- **0 / 804** device tokens are ranked worse than 32 by the reference, so the
  `max_top_k = 32` candidate window never dropped the token that was chosen.

**Measured:** the device samples inside the reference's distribution at every
position but one. **Inferred:** the ~15% of positions where bf4 numerics flip a
near-tie are the only mechanism available on our side to change the trajectory.
That inference is supported but not proven by the free-run below.

## Reference free-run: the reference completes this prompt

**Question:** does the reference model loop on this prompt at all?

Two runs, HF bf16 on CPU, greedy, same 80-token rendered prompt, script
`repro_scripts/hf_freerun.py`:

| run | tokens | `</think>` emitted | EOS | max repeated 120-char window | verdict |
|---|---|---|---|---|---|
| 1,500-token cap | 1500 | no | no | 1x | still drafting, no loop yet |
| 4,000-token cap | **2973** | **yes, at token 2389** | **yes** | **2x** | **completed normally** |

The 4,000-token run is the decisive one. The reference finished on its own,
closed its reasoning block, and produced an answer that satisfies **all three**
IFEval constraints:

| | reference answer | requirement |
|---|---|---|
| words | 474 | 300+ |
| commas | 0 | 0 |
| `*highlighted*` spans | 3 | 3+ |

It exits the trace cleanly, having done exactly the self-check the device gets
stuck on:

```
    *Word Count Check:* This looks to be around 350 words.
    *Comma Check:* None found.
    *Markdown Check:* Three sections.

    Ready to output.
```

The device, given the identical prompt and the same greedy rule, was still
inside a verbatim repeat of its draft paragraph 13,400 tokens later.

There is also a shorter, sharper comparison from the teacher-forced script. Free-
running the reference greedily for 200 tokens **from the device's own loop-entry
point** (character 1,727, 460 generated tokens in) shows the reference moving
forward through new material (`*Rise to Power:*`, `*Marriage and Politics:*`)
where the device re-entered the paragraph it had already written.

**Honest bounds on this result.** It is one prompt and one reference run per
budget. It does not establish that the reference never loops on any prompt, nor
that it would not loop on this prompt from a different sampled trajectory. It
does establish, for this prompt under greedy decoding, that the loop is **not
where the reference model goes**, which is a stronger statement than the
1,500-token run alone could support.

## Where this leaves the diagnosis

Ordering the evidence:

1. The forward pass is batch-invariant and the sampler is not corrupting
   anything (byte-identical greedy A/B).
2. The device stays inside the reference's distribution at 803 of 804 checked
   positions, and its top-1 agreement matches the bring-up baseline.
3. The reference nevertheless **completes** this prompt under greedy decoding
   while the device loops indefinitely.

Points 2 and 3 are both true and they constrain the answer from opposite sides.
The device is not broken in any way a PCC or top-k gate would catch, and it is
also not reproducing the reference's behaviour on this prompt. The mechanism
that fits is trajectory divergence: ~15% of positions are near-ties, bf4 expert
quantization flips some of them, and on a prompt whose trace contains a
self-reinforcing draft cycle, one early flip is enough to land in the cycle and
stay there. Greedy decoding has no escape, which is why greedy fails 100% of the
time and sampling sometimes recovers.

This is a hypothesis with support, not a proven cause. What would confirm it is
in "What is not yet known".

The practical consequence is unchanged either way: **the loop is reachable in
production and there is no mechanism in the serving path that stops it.**

## Config-knob trials

All on IFEval doc0, single isolated request, `seed 42`, against the running
container. No code changes. Constraint columns are the three IFEval checks
(300+ words / 0 commas / 3+ highlighted spans).

**Greedy rows are deterministic and reproducible. Sampled rows are n=1
anecdotes** and are marked as such; a single sampled run that finished proves
nothing about the rate, because the unmodified seed-42 sampled run also
finished.

| # | setting | determinism | finish | tokens | content | words / commas / highlights | tok/s |
|---|---|---|---|---|---|---|---|
| 1 | greedy (baseline) | deterministic | `length` | 16384 | **empty** | fails, no answer | 22.0 |
| 2 | greedy + `repetition_penalty 1.05` | deterministic | `stop` | 2583 | 2726 ch | **389 / 0 / 3** pass | 21.0 |
| 3 | greedy + `presence_penalty 0.5` | deterministic | `stop` | 2377 | 2011 ch | 328 / 0 / **2** fail | 23.2 |
| 4 | greedy + `enable_thinking=false` | deterministic | `stop` | 465 | 2059 ch | **370 / 0 / 3** pass | 32.5 |
| 5 | t1.0 / p0.95 + `repetition_penalty 1.05` | n=1 | `length` | 16384 | 47266 ch | **catastrophic**, see below | 17.8 |
| 6 | t1.0 / p0.95 + `enable_thinking=false` | n=1, but see note | `stop` | 422 | 1907 ch | **331 / 0 / 3** pass | 32.6 |
| 7 | t1.0 / p0.95 + `top_k 20` | n=1, **not evidence** | `stop` | 2892 | 1840 ch | 301 / 0 / 3 pass | 30.6 |
| 8 | t0.7 / p1.0 (card's Terminal-Bench setting) | n=1 | `stop` | 5986 | 7571 ch | 1284 / **3** / **225** fail | 28.1 |

Reading the rows that matter:

- **Row 5 is the important negative result.** A repetition penalty rescues
  greedy (row 2) and is catastrophic under production sampling. The output is
  16,384 tokens of word salad: 47,266 characters, 5,163 words at a **93%
  unique-word ratio**, 13 sentence-ending periods in the whole thing, drifting
  into unrelated token soup (`ExecStart= RestartRestartSec auto fstab mount
  umount fsck chkconfig SysV init scripts Upstart job manager cron daemon`).
  The penalty pushes the distribution off the manifold once the temperature is
  already flattening it. **Do not ship a repetition-penalty default.**
- **Rows 4 and 6 are the only mitigation that holds under both decode modes.**
  Disabling the thinking block removes the structure the loop lives in, so there
  is nothing to loop inside. Both finish in 420 to 465 tokens with all three
  constraints met, the answers are factually correct (correct dates, correct
  regent), and they are the fastest rows in the table.
- **Row 7 is not evidence.** `top_k 20` finished, but so did the unmodified
  seed-42 sampled run. One trial cannot distinguish the knob from the draw.
- **Row 8 degrades quality without fixing anything.** The card recommends
  `temperature 0.7, top_p 1.0` for Terminal-Bench and SWE-bench. On this prompt
  it produced 1,284 words (over 4x the requested length), 3 commas where zero
  were allowed, and 225 italic spans where 3 were asked for, at a 48%
  unique-word ratio. It finished, but the answer is worse than the baseline's.
- Row 3 breaks the loop but loses a constraint, so a presence penalty is
  strictly worse than a repetition penalty for the greedy case.

## Cost of the on-device penalties path

Rows 2 and 5 above both route through `TTPenalties`
(`models/common/sampling/tt_penalties.py`), which was never perf-characterised
during bring-up. Clean same-prompt A/B on IFEval doc1 (a prompt that terminates
normally), greedy, single user, `seed 42`:

| arm | tokens | wall | tok/s |
|---|---|---|---|
| no penalty | 1512 | 47.3 s | **32.0** |
| `repetition_penalty 1.05` | 1303 | 55.5 s | **23.5** |
| no penalty, repeated | 1512 | 47.3 s | **32.0** |

The repeat of the control reproduces the first arm exactly, so the delta is the
penalty path and not drift or thermal state. **The on-device penalties path
costs 27% of decode throughput.** `apply_penalties` runs seven extra elementwise
device ops per step (two typecasts and a subtract for presence, the same for
frequency, then a mask add, typecast, and the repetition divide), and it also
selects a different sampler trace slot, since `SamplingGenerator` keys its trace
on `(penalties, log_probs, force_argmax, bucket)`.

This is a second, independent reason not to reach for penalties as the default
mitigation.

## GPQA cross-validation and the shipped cap

Run after the mitigation shipped, against the published image
`tt-model/glm-4-7-flash-p150:c10b7c9eaed8`. Two questions this answers: does the
loop recur on a prompt class other than IFEval, and what does turning reasoning
back on actually cost.

**Method.** The same 10 `gpqa_diamond_cot_zeroshot` docs from a `ci-nightly`
eval run (seed 42, `temperature=1.0`, `top_p=0.95`) were replayed twice: once as
the harness ran them (server default, reasoning off), once identical in every
other respect but with `chat_template_kwargs.enable_thinking=true` added to the
request. Both scored with lm-eval's own `BoxedChoiceFilter`, not a
reimplementation, so the methodology matches the harness exactly.

| doc | target | no-think (shipped default) | think (opt-in) |
|---|---|---|---|
| 0 | (C) | (C) correct | `[invalid]`, answered with a value ("5e-15 J") instead of a letter |
| 1 | (A) | (D) wrong | (B) wrong |
| 2 | (B) | (B) correct | (B) correct |
| 3 | (D) | (C) wrong | (D) correct |
| 4 | (C) | (C) correct | (C) correct |
| 5 | (B) | (B) correct | **empty, hit the 32,768-token cap**: 88,141 chars of reasoning, `</think>` never emitted |
| 6 | (D) | (D) correct | (D) correct |
| 7 | (B) | (A) wrong | (D) wrong |
| 8 | (D) | (C) wrong | **empty, hit the 32,768-token cap**: 134,545 chars of reasoning, `</think>` never emitted |
| 9 | (B) | (B) correct | (B) correct |

**no-think: 6/10 (60%). think: 5/10 (50%).**

This directly answers open question 3 below: **the loop is not IFEval-specific**.
Two of ten GPQA docs hit the identical failure signature (empty `content`,
`finish_reason: length`, reasoning that never closes) on a completely different
task and prompt shape. That is a second, independent prompt class, and it
strengthens fixes 1 and 2 below rather than leaving them resting on one
reproducer.

It also answers open question 4, on the same small-sample caveat as everywhere
else here: n=10, one draw per condition, not repeated seeds, so 50% vs 60% is
directionally informative and not a precise number. What it does establish is
that reasoning-on is not a free upgrade even where it survives: doc 0's
think-mode answer got the physics right in its reasoning and then answered with
a value instead of the requested letter, a distinct instruction-following
failure that reasoning was supposed to help with.

**A discovery about the cap itself.** Both capped responses stopped at exactly
32,768 tokens, not the eval task's requested `max_tokens: 65536`. Checked in the
served image's vLLM source: `ChatCompletionServing.__init__`
(`vllm/entrypoints/openai/chat_completion/serving.py`) sets
`self.override_max_tokens` from `override_generation_config["max_new_tokens"]`
when `generation_config` is `"auto"` (the default, and what this image runs
with). `get_max_tokens()` then returns
`min(model_max_tokens, client's max_tokens, override_max_tokens, platform_max_tokens)`
over whichever of those are not `None`. Because `override_max_tokens` is set,
it participates in that `min()` for **every** request, regardless of what the
request itself asks for. This is stronger than the manifest comment claimed
("bounds any single runaway" undersold it): the cap is a hard ceiling, not a
fallback for the omit-`max_tokens` case. The corollary is that any legitimate
caller wanting more than 32,768 output tokens is silently truncated on this
image, not just runaways. That is worth stating plainly in the model card,
since nothing there currently says a request's own `max_tokens` can be
overridden by the server.

## Full-scale confirmation: the fix closed one signature, not the defect

Run 2026-09-10 against the published image, both `ifeval` and
`gpqa_diamond_cot_zeroshot` at their real full sizes (541 and 198 samples, no
`--limit`, confirmed by `n-samples.effective == n-samples.original` on both,
which is the marker that distinguishes a real full run from the
`--eval-samples` reporting artifact noted earlier in this doc).

**Scores.** ifeval 70.1% prompt-level strict / 77.3% instruction-level strict.
gpqa_diamond_cot_zeroshot 47.0% flexible-extract (published 75.2, ratio 0.62,
accuracy check FAIL, waived at `EXPERIMENTAL`). Both well below the earlier
small-sample reads (ifeval smoke-test 66.67 on n=3; gpqa ci-nightly 60 on
n=10), which is the expected direction: small samples were optimistic draws.

**Zero empty responses.** 0 of 739. The specific defect this doc chased and
the shipped mitigation targets, IE `<think>` never closing so
`message.content` comes back empty, does not reproduce at this scale. That
part of the fix holds.

**But 39 of 739 (5.3%) show unambiguous degenerate repetition inside the
visible answer.** Not empty. Not truncated by the parser. Real, non-empty
`content` that is garbage: `((((((((...` for 53,178 characters (5,787 repeats
of one window), `major major major major...` for 97,625 characters, a
bulleted list whose every bullet reads `Bullet point 3` (386 times), a
self-apology repeated hundreds of times, matrix/LaTeX fragments looping,
emoji spam, escaped-character spam. Scanned with the same repeated-120-
character-window detector used earlier in this doc, at a **tightened**
threshold after two false positives were caught and excluded by hand
(a forum-thread prompt that legitimately repeats speaker tags, and a
bullet-list answer with genuinely distinct bullets, both initially
over-flagged by a looser threshold). The 39 that remain were spot-checked and
the repeated span in every one is unambiguous garbage, not structure the
prompt asked for.

**`doc_id 0` is the exact reproducer this whole document is built around.**
Same prompt (Raymond III of Tripoli, 300+ words, no commas, three highlighted
sections). Under the shipped default, IE sampled, reasoning off, it looped
again: 95,873 characters ending in "I apologize for the previous response
which was quite a bit confused and contained significant hallucinations"
repeated 711 times. Different literal text than the `<think>`-block loop this
doc opened with, same phenomenon: the model enters a short cycle and cannot
exit it before the token budget runs out. The mitigation moved this reproducer
from failing silently (empty `content`) to failing loudly (garbage
`content`). Progress, not resolution.

**Read the earlier "cheapest mitigation" line in the Headline table in this
light.** "`enable_thinking=false`: 420 to 465 tokens, all constraints met" was
true on that one measured trial. It is not true in general: doc_id 0, same
prompt, same mitigation, sampled instead of greedy, still fails. The
mitigation reduces the failure rate on this specific reproducer, it does not
eliminate the underlying behaviour.

**One structural finding worth carrying forward: this loop no longer requires
a `<think>` block to happen in.** Every measurement earlier in this doc found
the loop specifically inside reasoning. This run has reasoning off entirely
(`--default-chat-template-kwargs enable_thinking=false`) and it still
happens, in ordinary answer text. That widens the failure class from
"reasoning self-check cycle" to plain autoregressive repetition collapse,
which is a documented phenomenon in language models generally (Holtzman et
al., "The Curious Case of Neural Text Degeneration") under low-diversity
decoding on prompts that stack constraints, independent of any hardware or
quantization choice. This matters for attribution: some or all of the 5.3%
could be this checkpoint's baseline behaviour under any precision, not a
consequence of anything in this stack. See the next section.

**Is this the bring-up's precision choice, or is it separate?** Both remain
live, and this run cannot separate them; it only shows the failure is bigger
and broader than believed at the point the mitigation shipped.

The evidence FOR a precision link, restated with its actual weight: an n=1
comparison (this same reproducer, greedy) where the bf16 CPU reference
completed and the bf4-expert device did not; 85.6% teacher-forced token
agreement on that one trial, meaning roughly 1 in 7 positions is a different
token at a near-tie; and a measured operator-level accuracy cost from the
bring-up's own precision choice (full MoE block PCC 0.984 at `bfloat4_b`
experts versus 0.9999 at `bfloat8_b`, `doc/probe/README.md`). All real, none
of it at the sample size this section's finding is measured at.

The evidence for a precision-independent cause: the newly measured 5.3% rate
is now the largest sample in this document by a wide margin (739 versus n=1
or n=10 everywhere else), and repetition collapse under constraint-stacked
prompts is a known LLM failure mode unrelated to hardware.

**The next test, in progress:** run the bf16 CPU reference at comparable
scale and compare its own degenerate-repetition rate against 5.3%. A literal
full 739-prompt reference run is not tractable on this box: no GPU (`nvidia-
smi: command not found`), and the single-sequence CPU reference rate measured
earlier in this doc (1.5 tok/s) puts a full 739-prompt run at the device's own
response lengths at roughly 8.9 days of continuous CPU time. The tractable
design instead targets the 39 confirmed-flagged prompts directly: if the
bf16 reference reproduces the loop on most of them, the precision link is
weak or absent and this is the checkpoint's own behaviour; if it reproduces
on few or none, the precision link is the more likely explanation and the
bf8-on-two-chips experiment already proposed below becomes the next step
rather than an optional one. Results land in `logs/reference_at_scale/` when
the run completes.

## Candidate fixes, ranked

### 1. Thinking budget (cleanest, upstream-aligned)

vLLM 0.25 already has the mechanism. `SamplingParams` carries
`thinking_token_budget` (validated at `vllm/sampling_params.py:35`), and
`ThinkingBudgetStateHolder` in `vllm/v1/sample/thinking_budget_state.py` tracks
each request's reasoning section and forces the reasoning end tokens once the
budget is exceeded. It derives its start and end token IDs from `ReasoningConfig`
(`vllm/config/reasoning.py`), which the server already populates because we run
with `--reasoning_parser glm47`.

**The blocker is that it is applied in vLLM's own sampler**
(`vllm/v1/sample/sampler.py:381`, `sampling_metadata.thinking_budget_state_holder`),
and the TT device-sampling path never reaches that code. The plugin's
`check_perform_device_sampling` (`vllm_tt_plugin/model_runner.py:1850`) routes
the batch to the device sampler, which draws with `ttnn.sampling` and returns
tokens, not logits. A grep of the plugin for `thinking` or `budget` in a
sampling context returns nothing.

Two ways to close it:

- **Port it plugin-side.** Track per-request generated-token counts inside the
  reasoning section on the host (the plugin already owns `req_ids` and
  positions), and when a request exceeds its budget, force its next token to the
  reasoning end token. The device sampler already writes into a persistent token
  tensor, so overriding one row is a host write, not a new device op.
- **Add `thinking_token_budget` to the always-host-only list.** The plugin
  already falls back to host sampling for `min_p`, `bad_words`, `logit_bias`,
  `allowed_token_ids`, `min_tokens`, structured outputs, and logprobs on a
  single-chip mesh. Adding one more entry is a two-line change, but it costs the
  whole batch its device sampling whenever any request sets a budget.

The first is more work and keeps decode on the device sampler. Prefer it.
Upstream parity is the reason to rank this first: budgets are how Qwen3-class
reasoning models are already served, so clients will expect the parameter to
exist and behave.

This is no longer resting on one reproducer. "GPQA cross-validation and the
shipped cap" above found the identical failure signature on 2 of 10 GPQA docs
under opt-in reasoning, a completely different prompt class from the IFEval
constraint prompt this section was written against. A fix that only helps
IFeval-shaped prompts would not be worth much; this one is general.

### 2. Loop guard (targeted, catches what a budget does not)

A budget truncates every long trace, including legitimate ones. A loop guard
truncates only the pathological ones.

**Detection:** maintain a rolling window of the last N generated token IDs per
row and flag a repeat when a window of length W recurs K times consecutively.
The evidence here is generous: the observed cycle repeats a 120-character span
at least 10 times, so W in the low hundreds of tokens and K of 3 is far inside
the margin. **Action:** force `</think>` (token 154842) on the next step for that
row, exactly as the budget path would.

**Where it lives:** the plugin's decode path, or the GLM adapter's
`decode_forward` in `tt/generator_vllm.py`. The device path is the right place
for both candidates, because it is the only place that sees every generated
token: under `sample_on_device_mode: all` the tokens never round-trip through a
host sampler, and `read_decode_tokens` is already called every step to return
them. **State needed:** per-row ring buffer of recent token IDs, a per-row
"inside `<think>`" flag (set when the reasoning start token is seen, cleared on
the end token), and the counters. All host-side, all cheap, no new device ops.
It must survive a batch condense, so it has to be keyed by request rather than
by physical row, the same lesson `SeedManager.apply_slot_remap` already encodes.

Do both 1 and 2. The budget bounds the worst case; the guard catches the actual
failure earlier and without penalising well-behaved long traces.

### 3. Precision (real lever, expensive, and not a complete fix)

The device diverges from the reference at ~15% of positions, all near-ties, and
the reference completes this prompt where the device does not. Moving routed
experts from `bfloat4_b` to `bfloat8_b` would cut that divergence: the one-layer
probe (`doc/probe/README.md`) measured full MoE block PCC of **0.984 at bf4
versus 0.9999 at bf8**, with per-expert error tightly clustered (0.9809 to
0.9819), so it is uniform across experts and selective mixed precision would not
help.

**The blocker is capacity, and it is a hard one.** The deployment contract is
explicit (`tt/optimized_decoder.py:23`, `doc/probe/README.md`): the 30.6B model
fits one 32 GB p150 **only** with bf4 routed experts, at about 18 GB of weights.
bf8 experts are roughly 32 GB of expert weights alone and do not fit one card.
That makes this the tensor-parallel path across two or more chips, which was
assessed separately and rejected on throughput grounds: decode is at ~18% of the
DRAM roofline and explicitly launch-bound, so TP2 would recover only a couple of
milliseconds per token while adding ~94 collectives.

It is worth restating that the accuracy case for bf4 was made honestly and still
holds on its own terms: the full-model stage's top-5 gate (>= 0.98) passes at
1.000, and the bring-up degenerate-output check passes. **This defect is
invisible to both gates**, because it is a trajectory effect on a specific
prompt class rather than a distribution error, and because the degenerate-output
check ran on a 256-token autoregressive sample that never reaches the loop.

Rank this third. It is a genuine lever on the divergence rate, it is a
multi-week hardware change, and it would reduce rather than eliminate the
problem: near-ties still exist at bf8, and the trace cycle is a property of the
model's behaviour on this prompt, not of our numerics. Fixes 1 and 2 bound the
damage regardless of precision and should ship first.

### Not recommended

- **A `repetition_penalty` server default.** Rescues greedy, catastrophic under
  sampling (row 5), and costs 27% of decode throughput.
- **`--generation-config vllm`.** This was previously logged as a deferred fix
  for the shipped manifest. It is now **withdrawn**: it makes server defaults
  greedy, and greedy loops 100% of the time on this prompt class.
- **`temperature 0.7, top_p 1.0` as a default.** Row 8. Finishes, but the answer
  quality is worse than baseline.

## What is not yet known

Stated as the questions a follow-up has to answer, with what each needs.

1. **The loop rate under production sampling.** Everything quantified here is
   greedy. The field report of roughly 10% is consistent with what was seen but
   was not reproduced under controlled conditions. Measuring it needs many
   trials at a realistic budget: the device's one successful sampled run took
   7,440 tokens, so trials must run to at least 8K before they can be scored,
   which is roughly 6 minutes each on one chip. A hundred trials is about 10
   hours of device time. The CPU reference is not an option at 1.5 tok/s.
2. **Whether the reference loops on this prompt at all, ever.** The 4,000-token
   greedy run completed, which is strong for greedy. It says nothing about
   sampled reference trajectories. Settling it needs sampled reference trials at
   >= 8K tokens, which needs a GPU.
3. **~~Whether this is prompt-class-specific.~~ Answered, partially.** A
   follow-up run (see "GPQA cross-validation and the shipped cap" above) replayed
   10 `gpqa_diamond_cot_zeroshot` docs with reasoning force-enabled and got the
   identical failure signature (empty content, `finish_reason: length`,
   `</think>` never emitted) on 2 of 10 -- a different task, different prompt
   shape, no shared constraint structure with the IFEval reproducer. That kills
   the narrow version of the prompt-class hypothesis (this is not specific to
   multi-constraint instruction-following) without yet telling us the true rate
   or trigger on GPQA-style prompts; n=10 with no repeats is not enough for
   either. An earlier 3-sample GPQA run (referenced in the withdrawn note this
   replaces) completed cleanly and was read as weak evidence against
   prompt-class specificity; that reading did not survive a larger sample.
   Still untested: code and agentic prompts. Still open: what actually triggers
   it, beyond "some prompts drive the model into a self-check/redraft cycle it
   can get stuck in.
4. **~~What reasoning-off costs on accuracy.~~ Partially answered.** A paired
   10-doc GPQA comparison (see "GPQA cross-validation and the shipped cap"
   above), same docs, same seed, same sampling, only `enable_thinking` differs:
   no-think scored 6/10, think scored 5/10. Two of the think-mode losses were
   the loop hitting the 32,768-token cap, not a reasoning quality gap as such,
   so the clean comparison on docs that actually completed is closer to 6/10 vs
   5/8. Either way, reasoning did not clearly help on this sample, and it cost
   an average of 13,340 completion tokens per doc versus no-think's much
   smaller footprint. This is n=10, one draw per condition, not repeated seeds:
   directionally useful, not a number to publish as a benchmark score. The
   card's 75.2 remains a think-mode figure measured by the checkpoint's authors
   under conditions we have not reproduced (no `bfloat4_b` experts, presumably
   no cap, unknown sample size), so it is not the right baseline for either
   number here. On the one instruction-following prompt measured earlier in
   this doc, no-think was also better: correct dates and correct regent, where
   the think-mode answer hallucinated. Two prompts now point the same
   direction; still not enough to generalise past "reasoning is not a
   free win here and is not obviously worth its cost."
5. **Whether the divergence hypothesis is actually the mechanism.** The cheap
   test is a bf8-expert arm on two chips running the same greedy prompt. If it
   completes, the precision link is established; if it still loops, the cause is
   elsewhere and fix 3 can be dropped from consideration entirely.
6. **Whether the 5.3% full-scale degenerate-repetition rate is precision-driven
   or the checkpoint's own behaviour.** See "Full-scale confirmation" above.
   The reference-at-scale run against the 39 confirmed-flagged prompts is in
   progress; a literal full-739 reference run was assessed as ~8.9 days on this
   box's CPU-only hardware and is not being attempted.

## Adjacent issues considered and excluded

None of these causes the loop. Each was considered and is recorded so the next
reader does not re-open it.

- **tenstorrent/tt-metal#33492** (stable top-k unreliable). The device's top-k
  network can order exact-value ties by all-gather and device order, which
  varies run to run. `_adjust_values_for_tiebreak`
  (`models/common/sampling/tt_sampling.py:673`, applied at `:1053`) works around
  it for `k == 1` rows. **Excluded:** the greedy A/B is byte-identical across
  runs, so tie ordering was stable here, and the teacher-forced check shows
  0/804 device tokens ranked worse than 32 by the reference, so the candidate
  window never dropped the chosen token.
- **tenstorrent/tt-metal#55408** (order-dependent sampling-state contamination).
  The original suspect when the symptom looked batch-dependent. **Excluded:** the
  byte-identical greedy A/B rules out any batch-order effect on this defect.
  The issue remains open on its own merits.
- **tenstorrent/tt-metal#51981** (seeded stream versus batch-layout changes,
  referenced at `models/common/sampling/generator.py:980`). A seeded request's
  device stream can shift when vLLM changes the decode batch layout, because the
  counter is anchored to absolute position and re-registration happens on
  `reset_batch`. **This is what actually explains the original sampled
  observation** (same seed, different stream, different outcome). It is a
  reproducibility property, not a correctness defect, and it is not what makes
  the trace loop.

## Files

- `README.md` (this file).
- `repro_scripts/teacher_forced_ref.py`: teacher-forced agreement of a device
  trace against HF bf16 on CPU, plus the free-run-from-loop-entry comparison.
  Takes a saved response dump.
- `repro_scripts/hf_freerun.py`: reference greedy and sampled generation of
  doc0 on CPU, reporting `</think>` closure, EOS, and a repeated-window metric.
- `repro_scripts/greedy_identity.py`: the alone-versus-batched greedy identity
  test that retracts the batching hypothesis.
- `repro_scripts/loop_breakers.py`, `repro_scripts/loop_breakers2.py`: the
  config-knob trials.
- `repro_scripts/penalty_perf_ab.py`: the penalties decode-rate A/B.
- `repro_scripts/gpqa_reasoning_on.py`: replays a `ci-nightly` GPQA samples
  file with `chat_template_kwargs.enable_thinking=true` forced on, for a
  paired comparison against the harness's own reasoning-off run on the same
  docs.
- `repro_scripts/scan_degenerate_repetition.py`: the repeated-120-character-
  window detector used to find the 39 confirmed-flagged docs in the
  full-scale run, with the tightened threshold and the two manually-excluded
  false positives noted in "Full-scale confirmation" above.
- `repro_scripts/reference_at_scale.py`: runs the bf16 CPU reference against
  a named list of prompts (the 39 flagged docs, or any other set) under
  matching sampled settings, for the precision-attribution comparison.
- `logs/`: stdout from every run above, plus
  `hf_freerun_greedy4000.json`, the completed reference generation,
  `gpqa_reasoning_on.log` / `gpqa_reasoning_on_results.json` from the
  post-ship GPQA cross-validation, and `reference_at_scale/` for the
  full-scale precision-attribution comparison.

All device scripts expect an OpenAI-compatible server on `127.0.0.1:8000` and
take the eval sample file as their first argument, for example
`workflow_logs/reports_output/evals/GLM-4.7-Flash_p150_evals/eval_id_autoport-glm47-flash_GLM-4.7-Flash_p150/zai-org__GLM-4.7-Flash/samples_ifeval_*.jsonl`
under a `tt-inference-server` checkout. The CPU scripts need
`python_env/bin/python` from tt-metal and about 62 GB of RAM for the bf16
reference.
