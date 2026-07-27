# Device Gumbel restored: what was fixed, what was not, and how to tell them apart (#48291)

**Status 2026-07-26: the RNG-driven canvas collapse is fixed and `DG_VLLM_GUMBEL_MODE=device` is
the default again.** The residual collapses are a different mechanism with a different owner —
they are the #48291 fidelity gap, not the sampler. This document exists mostly so that the second
mechanism is not mistaken for the first a third time.

## 0. Why device at all

`host` Gumbel is correct but does not meet the throughput bar: **~53.6 vs ~36.3 tokens/block/s**
measured at 4096 on a matched 4-seed A/B (~1.48x). Fixing the device path was therefore the task,
rather than routing around it.

## 1. Two defects, both measured, both fixed

### 1a. `ttnn.rand` is a sliding window over one stream

Mapping one 32x32 rand tile element by element: **94 distinct values in 1024 slots**. The PRNG is
per-lane (32/32 distinct inside one 32-lane SFPU vector), but only 20 of a tile's 32 vector draws
were distinct, with the exact relation

    (face f, vector 2k) == (face f+1, vector 2k+1)

which in tile coordinates is "column c is byte-identical to column c-24 for c % 32 >= 24". So
element `(read t, lane i)` carries `stream[t + i]`: one window advancing about one element per read
while all 32 lanes read overlapping positions.

Fix: consume several PRNG values per stored element
(`tt_metal/hw/ckernels/blackhole/.../ckernel_sfpu_rand.h`).

| | tile distinct | distinct vector draws | byte-identical rows | max abs r | max argmax mult |
| --- | --- | --- | --- | --- | --- |
| before | 94/1024 | 20/32 | 64/256 | 1.00000 | 11 |
| after | 214/1024 | 32/32 | **0/256** | 0.618 | 5 |
| host IID | — | — | 0/256 | 0.035 | 2 |

Distribution untouched: mean 0.4994, std 0.2887, top decile 0.0996 over 524288 samples.

**Measured dead ends — do not repeat.** NOP spacing after the PRNG read (0 through 32: byte-identical
output, so not a pipeline hazard, and the Wormhole kernel's extra NOPs are irrelevant here);
xorshift32 over two draws (no gain — any combination of reads is still a function of `t + i`);
`SFPTRANSP` across four draws with an XOR fold (modest, and it re-introduced duplicate rows). One
trap: holding `scale`/`from` in lreg4/lreg5 across a transpose silently breaks the output range to
mean 0.35 / std 0.64, which passes the correlation metrics by destroying the distribution instead.

### 1b. DG was drawing the noise on the wrong axis

`sample_gumbel_noise_with_permuted_vocab` keeps vocab off the rand innermost axis by collapsing
every OTHER axis into it — and for the production shape `(1, 1, 256, vocab)` that is the 256 canvas
positions. It moved the degeneracy from an axis where it biases WHICH token a position picks onto
the axis where it makes DIFFERENT POSITIONS pick the same token, which is what collapses a canvas.

Measured at the production geometry (canvas 256, vocab 262144), with the kernel already fixed:

| layout | max abs r | distinct flat-logit winners | max multiplicity |
| --- | --- | --- | --- |
| permuted (was production) | 0.598 | 154/256 | 8 |
| **vocab innermost (now)** | 0.350 | **253/256** | **2** |
| host IID control | 0.009 | 256/256 | 1 |

## 2. The two collapse mechanisms, and the telemetry that separates them

Both end with a committed canvas dominated by one token, so the token counts alone cannot tell
them apart. The halt telemetry can, and they have different owners:

| | CONVERGED | NON-CONVERGED |
| --- | --- | --- |
| entropy trajectory | 4.3–5.1 → **~0.01** | 3.6 → **2.4**, stays high |
| `halt_mismatch_final` | **0** (argmax settled) | **109** of 256 still flipping |
| `halt_blocking_gate` | `entropy` | `both` |
| meaning | converged onto a confident degenerate fixed point | step cap hit mid-churn |
| cause | correlated noise across positions | fidelity gap vs the reference |
| owner | **fixed here** | **#48291** |

## 3. The configuration is faithful, so NON-CONVERGED is not a settings bug

Every knob matches the released `generation_config.json` exactly, checked item by item:

| reference | DG |
| --- | --- |
| `max_denoising_steps: 48` | `max_denoise_steps = 48` |
| `confidence_threshold: 0.005` | `entropy_stop_threshold = 0.005` |
| `stability_threshold: 1` | `stable_steps_to_halt = 1` |
| `t_max: 0.8` / `t_min: 0.4` | `temperature_start = 0.8` / `temperature_end = 0.4` |
| `sampler_config.entropy_bound: 0.1` | `entropy_budget = 0.1` |

The CUDA reference converges within the same 48 steps on the same prompts and never collapses
(0/198). Same budget, same thresholds, same schedule, different outcome — that is a numerical
fidelity difference, i.e. #48291.

## 4. The CUDA reference arm (what is inherent, what is ours)

Full GPQA-Diamond, thinking, through the CUDA vLLM server — same model, no TT sampler anywhere:

| | @8192 | the 45 unanswered, retried @16384 |
| --- | --- | --- |
| accuracy | 124/198 (62.6%) | 24/45 rescued → **148/198 (74.7%)** |
| answered | 153/198 (77.3%) | 40/45 (88.9%) |
| COLLAPSE (>=20 identical consecutive tokens) | **0/198 (0.0%)** | — |
| tail repetition | 44/198 (22.2%) | 22/45 (48.9%) |
| no answer at the cap | 45/198 (22.7%) | 5/45 (11.1%) |

Three things follow:

* **Truncation and tail repetition are inherent.** The GPU truncates 22.7% at 8192 and repeats in
  22.2% of answers. Calling those "TT degeneration" was a category error.
* **Canvas collapse is NOT inherent** — the GPU never does it. That mode was ours, and it is the
  one fixed above.
* **Roughly a third of the "unanswered" set is noise, not budget.** Of the 40 rescued at 16384,
  **16 answered in under 8192 tokens** — the first run simply took a looping trajectory. vLLM's
  continuous batching is numerically batch-dependent, so even at temperature 0 the same question
  can go either way. Any truncation rate measured with batching carries that variance.

## 5. Device gate

Matched 4-seed A/B, shipped serving configuration, GPQA doc0 in thinking mode: 8/8 correct on both
arms, guard never fired, device ~1.48x faster. The two documents that previously failed hardest —
doc6 and doc9, both of which emitted ZERO characters because the guard fired on block 0 — now
generate normally, and doc9 answers `B`, matching the CUDA reference.

A full 198-question run at 16K is in progress (`tmux gpqa198` on the QB2 box; resumable, skips
questions whose metrics JSON already exists). Interim at 42/198: the CONVERGED mechanism appears
**once**, stable across every checkpoint (n=11, 22, 30, 36, 42), while NON-CONVERGED accounts for
the other 17. That split is the result this work is accountable for; the final aggregate numbers go
here when the run lands.

Note a real trade-off visible at 16K: truncation goes to 0%, but the collapse rate rises (30% at
8192/12 blocks vs ~43% at 16K/60 blocks) because a larger budget gives the NON-CONVERGED mode far
more blocks in which to churn. Raising the context is not free until #48291 is addressed.

## 7. CORRECTION: the residual was MoE math fidelity, not an untouchable ceiling

An earlier revision of this document attributed the NON-CONVERGED collapses to #48291 and called
HiFi4 a narrowing that "does not close" it. **That was wrong**, and the single-variable control says
so unambiguously. Same prompt (q007), same seed, same reveal span, same traced up-front path, same
noise mode -- only `DG_SPARSE_MOE_HIFI4` differs:

| block 0 of q007 | HiFi2 | HiFi4 |
| --- | --- | --- |
| steps run | 48 | **30** |
| halted | **False** | **True** |
| blocking gate | both | none |
| final mean entropy | **2.4278** | **0.0035** |
| positions still flipping | **109 / 256** | **0** |
| blocks emitted | 1 (guard refused block 0) | 4, clean |

The reveal span is not involved: block 0 is bit-identical at spans 2048 / 4096 / 8192 / 16384
(steps=30, H_final=0.003495, mismatch=0 at all four). The earlier misreading was mine, not the
data's -- the HiFi4 arm's q007 run emitted 8 blocks and I recorded that the guard fired without
checking that it had fired at block 9 rather than at block 0.

### HiFi4 is not uniformly better, and the reason matters

| block 0 | HiFi2 | HiFi4 |
| --- | --- | --- |
| q007 | 48 steps, not halted, mismatch 109 | **30 steps, halted, mismatch 0** |
| q012 | 45 steps, halted, mismatch 0 | **48 steps, not halted, mismatch 12** |
| q013 | 16 steps, halted | 18 steps, halted |

q012 was converging on step 45 of 48 -- three steps from failure. HiFi4 changes the numerical
trajectory, so it rescues a block whose gap was large and tips over a block that was already on the
edge. **Many blocks land right at the step cap**, which is why any perturbation flips some of them
in either direction. Judging a fidelity change on one prompt is therefore not safe; it needs a set.

### More steps do not rescue the plateau cases

Run eagerly with the cap raised to 96 (the eager path is not bound by the trace's K=48):

* q007 halts at **34** steps, accepted positions climbing 1 -> 252 of 256;
* q012 does **not** halt at 96 either. Its accepted count plateaus at ~183-195 of 256 from step ~75
  onward and stops improving, so ~60 positions never clear the entropy budget and the mean entropy
  floors at 0.183 -- 36x the 0.005 bar.

So q012 is not a "needs a few more steps" case: the entropy-bound rule reaches a fixed point that
leaves a subset of positions permanently uncertain. Raising the traced K would spend time for
nothing on this class.

### What the reference does with the same prompts

| | CUDA reference | TT under HiFi4 |
| --- | --- | --- |
| q007 | answers C, correct, 7596 tokens | block 0 converges in 30-34 steps |
| q012 | @8192 hits the cap with no answer; @16384 answers A, correct, 8357 tokens | block 0 does not converge at 96 |
| q013 | answers D (wrong, gold B) | block 0 converges |

Note what the reference does NOT have: a guard. At its own cap it commits the unsettled positions
anyway, shipping a canvas with some uncertain tokens rather than stopping -- which is what the 22.2%
tail-repetition rate in section 4 is. So the difference on TT is not that positions stay uncertain,
it is that the uncertain ones collapse onto ONE token and form a wall, which the guard then refuses.
Noise is ruled out as the cause of that collapse (fresh noise reproduces it byte-for-byte), so what
still distinguishes TT is the logits distribution over those unsettled positions.

## 8. CORRECTION: the collapse is NOT mostly inherent -- 94% of it is TT-specific

Sections 4 and 6c read the aggregate rates and concluded that truncation and tail repetition were
inherent to the model, because the CUDA arm showed 22.7% unanswered and 22.2% tail repetition at
8192. That conclusion came from comparing rates instead of comparing questions, and it is wrong.

The GPU's 22.7% unanswered at 8192 was mostly a BUDGET artefact: re-running those 45 questions at
16384 answered 40 of them (§4). Taking the union of the reference's two runs -- the fairest reading
of "could the reference do this question at all" -- and cross-tabulating question by question
against the TT arm over the 131 questions where both have results:

| | answered | correct |
| --- | --- | --- |
| CUDA reference | **126/131 (96%)** | 97 (74%) |
| TT (HiFi2, 16K, shipped config) | **64/131 (49%)** | 58 (44%) |

| | TT clean | TT collapsed |
| --- | --- | --- |
| reference answered | 66 | **60** |
| reference failed | 1 | 4 |

**60 of TT's 64 collapses (94%) are on questions the reference answers.** Only 4 questions
(28, 36, 69, 79) are hard for both -- that is the genuinely inherent set, and it is 3% of the set,
not 23%.

Nor are the 60 all long-reasoning questions that TT merely lacks room for. The reference's token use
on them is median 6744, min **1561**, max 14647, and TT's truncation rate at 16K is 0% -- the
shortest of them takes the reference about six blocks' worth of tokens, well inside TT's budget, and
TT still collapses.

So the gap is a TT defect surface covering 46% of GPQA-Diamond, not a model ceiling. Treating
#48291 as an acceptable ceiling was wrong: the reference demonstrates 96% is reachable on this
checkpoint, with the same 48-step budget, the same thresholds and the same schedule (§3).

What that changes about where to look: a fix does not need to invent new capability, it needs to
close a numerics gap on prompts that already work elsewhere. The 60 questions are a large, cheap
reproducer set -- the shortest ones (by reference token use) are the right place to start, since
they collapse without needing a long context.

## 9. ROOT CAUSE of the 60: TT's FIRST denoise forward produces flatter logits than HF-bf16

Everything above narrowed the residual to "the logits are too flat at the unsettled positions".
That is now measured at its source, and it is neither an iteration problem nor a precision ceiling.

### The mechanism, end to end

The entropy-bound acceptance rule accepts the k lowest-entropy positions whose exclusive prefix sum
of entropies is <= `entropy_bound` (0.1 nats, absolute). On q106 the per-step accepted count on TT is

    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...

**exactly one position per step.** Forty-eight steps therefore settle at most 47 of 256 positions;
the other ~209 stay renoised with uniform-random tokens for the whole block. Taking the clean argmax
over a canvas that is four-fifths random noise returns the unigram prior -- which is precisely the
observed collapse set: `-` (14 of 64 collapses), `\n` (11), `1` (7), `' the'` (4), `0`, `,`, `.`,
`2`, `' '`. Decoding the vocab neighbourhood confirms it: ids 236743-236780 are the single-character
tokens ordered by English letter frequency, i.e. the highest-prior, lowest-information tokens.

So the collapse is arithmetic, not mysterious: **accept count 1 => the canvas never fills => argmax
returns the prior.** A converging block looks completely different -- q007 ramps 1, 2, 3, 5, 6, 7,
10, 16, 23, 51, 75, 121 ... 252 and halts.

### Where it starts: step 1, before any feedback exists

Mean per-position entropy on q106, same prompt, same thinking contract:

| path | step-1 entropy |
| --- | --- |
| **HF reference, bfloat16, CPU** | **3.7495** |
| TT traced, sparse MoE, HiFi2 | 5.1022 |
| TT eager, sparse MoE | 5.1022 |
| TT eager, self-conditioning disabled | 5.3681 |
| TT traced, dense-128 MoE | 5.0913 |

Every TT path clusters at 5.09-5.37; the reference sits at 3.75, **in bf16**. The gap is 1.35 nats on
the FIRST forward, before self-conditioning exists and before any canvas feedback.

That rules out, in one table, the things this investigation spent the most time on:

* **not bf16 precision** -- the reference reaches 3.75 in bfloat16;
* **not the sparse token-gather MoE** -- the reference dense-128 expert path gives 5.0913, the same;
* **not traced vs eager** -- both give 5.1022 to five decimals;
* **not self-conditioning** -- step 1 has none, and disabling it makes the trajectory worse, not
  identical, so it is active;
* **not the iteration or the step budget** -- the divergence exists at step 1.

And it contradicts the "bf16 architectural floor" framing of #48291: a bf16 reference on the same
checkpoint, with the same 48-step budget, the same thresholds and the same schedule, converges on
q106 in **12 steps** (entropy 3.75 -> 0.0042, mismatch 175 -> 0) and answers it.

### The reference trajectory harness

`hf_reference_trajectory.py` records the reference's per-step trajectory in TT's own telemetry units
(mean per-position entropy, argmax mismatch against the previous step, distinct argmax count,
dominant-token share) by wrapping `StableAndConfidentStoppingCriteria`, which already receives the
processed logits and the argmax canvas every step. It needs no model changes and runs in **53
seconds** on the QB2 host CPU in bf16, so any prompt's reference curve is a minute away.

### Reproducer set

Seven questions collapse on **block 0**, so one block reproduces them, and the reference answers all
seven: **q106, q096, q122, q095, q007, q090, q064**. q106 is the cheapest -- the reference spends
1561 tokens on it. The right next step is a layer-by-layer comparison of that first denoise forward
against the reference, since the discrepancy is now known to live in a single forward pass.

## 10. The 56 late-block collapses are HF sliding-layer key retention (#51080), already built

Section 9 measured the first-forward entropy gap on the SEVEN questions that collapse on block 0.
That is 7 of 64 collapses. Locating the other 57 needed a different question: *where* in the
generation does each collapse happen, relative to anything that changes with the committed prefix?

### The correlation

`promptlen_vs_collapse.py` tokenises every prompt under the same thinking contract and crosses the
prompt length against the baseline's collapse block. Two things fall out:

| | n | min | median | max |
| --- | --- | --- | --- | --- |
| collapsed prompt_len | 64 | 110 | 258 | 2428 |
| clean prompt_len | 67 | 100 | 205 | 584 |

Prompt length itself separates nothing (median 258 vs 205; only 1 of 64 collapsed prompts is even
past 1024 tokens). But the *committed prefix* grows 256 tokens per block, and:

    collapse block histogram: 0:7  1:1  3:1  7:1  9:3  10:6  11:4  12:10  13:12  14:15  15:3  17:1

    56 of 64 collapses happen AT OR AFTER the block where the committed prefix crosses 1023.
     8 of 64 happen before it (7 of those are the block-0 set of section 9).

### Why 1023 is the number

DiffusionGemma is **25 sliding_attention layers and 5 full_attention layers** (`layer_types`,
window 1024) — 83% sliding, not the 1:1 interleave the plan text suggests. HF's *mask* over the
canvas is deliberately full ("DiT module doesn't need a sliding mask and has to attend fully to prev
context and itself", modeling_diffusion_gemma.py), but the *cache* still evicts: sliding layers hold
only the last `sliding_window - 1` committed keys.

Measured on the reference rather than taken from the docstring (`ref_sliding_retention.py`, A100,
1500-token prompt):

    layer  type                cached keys   get_mask_sizes(kv_len, kv_off)
        0  sliding_attention          1023                    (1279, 477)
        5  full_attention             1500                    (1756, 0)

    retained on sliding layers = 1023 == sliding_window - 1     (not 1024, not the full prompt)

TT's production denoise path is maskless all-attend, so past a 1023-token committed prefix those 25
layers attend to keys the reference does not have, and the excess grows every block:

| committed prefix | HF sliding keys | TT sliding keys | excess |
| --- | --- | --- | --- |
| block 4 (~1290) | 1023 | 1290 | 267 (21%) |
| block 9 (~2570) | 1023 | 2570 | 1547 (60%) |
| block 13 (~3600) | 1023 | 3600 | 2577 (72%) |

That growth curve is the shape of the collapse histogram: nothing before the crossing, a slow onset
around blocks 9-11, and the mass at 12-14.

### The fix was already written and left switched off

`denoise_forward.denoise_sliding_window_enabled` (#51080) implements exactly this retention as
reveal-mask content, per layer type, and its own docstring says why it shipped OFF:

> Default OFF because it is decision-CHANGING above `prompt_len = 1024` and its gate is a
> decision-agreement run against fp32 HF (not against today's TT output, which is the defect being
> corrected). **Flip the default once the fp32 HF agreement run lands.**

So this is not new code, it is a missing gate — and the collapse histogram above is the evidence the
gate was waiting for. 95 host tests already cover the mask semantics
(`test_denoise_sliding_window.py`, `test_hf_sliding_window_reference.py`, `test_attention_mask.py`,
`test_paged_prefix_reveal_mask.py`).

### What the arm does and does not prove

`DG_DENOISE_SLIDING_WINDOW=1` is a SINGLE-arm change against the baseline command: same Gumbel mode,
thinking, EOS stop, degeneracy policy, seed, context and block budget. It cannot fix the block-0
seven, because below a 1023-token prefix the window never binds and the mask is bit-identical to
today's — those 7 stay as the negative control.

## 11. CORRECTION to section 9: 5.1 nats is not a TT-wide offset, it is per-block

Section 9's table is titled as if TT's first denoise forward always lands at 5.09-5.37 while the
reference lands at 3.75. All five of those TT rows are the same prompt and the same block --
q106, block 0. Generalising from them was wrong.

A HEALTHY TT block, from the sliding-window arm's q017 (block 12, halt telemetry):

    entropy per step: 3.7167, 3.6057, 3.4377, ... 0.0062, 0.0033   halted after 24 steps
    mismatch:           251,    141,    139, ...      10,      0

**Step-1 entropy 3.7167 -- the reference's value.** TT does reach reference-quality first forwards;
5.1 is a property of the blocks that fail, not a constant TT penalty. Section 9's mechanism (accept
count pinned at 1 => canvas never fills => argmax returns the unigram prior) still stands for the
block-0 seven; only the "every TT path" framing was too broad.

### Two controls that keep the block-0 gap real

Both were missing when section 9 was written.

*Canvas luck.* HF and TT each draw their own random initial canvas, so the 3.75-vs-5.10 comparison
was never separated from canvas variance. `ref_seed_sweep.py` runs the reference on q106 over 8
initial canvases (A100, one model load, ~3 s each):

    step-1 entropy: 3.5311 .. 3.8775   (spread 0.35 nats)
    converged: 8/8, in 9-15 of 48 steps

The gap is 4x the canvas-variance spread and convergence is 8/8 against TT's 0/5, so canvas luck
does not explain block 0.

*Prompt padding.* `cache_len` is `prompt_len` rounded up to a 32-multiple, and all seven block-0
prompts carry 7-31 padding tokens, which looked like a lead. It is not: over all 131 questions the
CLEAN group has MORE padding on average (18.2 vs 16.3), and eight clean questions sit at the maximum
29-31 while running 4-17 blocks. Refuted before it was implemented.

## 12. CORRECTION to sections 9 and 11: the mean entropy is the WRONG statistic

Sections 9 and 11 both lead with mean per-position entropy -- TT 5.10 against the reference's 3.75 --
and treat the 1.35-nat gap as the thing to explain. `EntropyBoundSampler` never reads the mean. It
accepts the k lowest-entropy positions whose EXCLUSIVE prefix sum of entropies is <= `entropy_bound`
(0.1 nats, absolute), so k is decided entirely by the LOW TAIL.

Computing the reference's actual step-1 distribution from the dumps (`first_forward_stats.py`) shows
how misleading the mean is here, because the distribution is strongly bimodal:

| | q106 | q096 | q095 |
| --- | --- | --- | --- |
| entropy mean | 3.7535 | 4.0771 | 3.4311 |
| entropy **median** | **4.2624** | **4.2541** | **3.9226** |
| entropy max | 4.8179 | 4.9567 | 4.1219 |
| entropy min | 1.7e-05 | 2.9e-04 | 4.3e-05 |
| positions below 0.1 nats | 5 | 5 | 6 |
| **accept count at step 1** | **6 / 256** | **5 / 256** | **7 / 256** |
| logit_std | 7.4553 | 8.1133 | 7.4228 |

The reference's *typical* position is at 4.25 nats, not 3.75; the mean is dragged down by four to six
positions that are essentially CERTAIN (min 1.7e-05). Those few positions are the entire accept
budget: **the reference also starts at only 5-7 accepted of 256.** It converges because that count
RAMPS as the canvas fills -- the shape TT shows on a healthy block (1, 2, 3, 5, 6, 7, 10, 16, 23, 51,
75, 121 ... 252).

So "TT is 1.35 nats flatter on average" is not the mechanism. The mechanism is that TT has **no
position under the bound at all**, which pins k at 1 -- k=1 is always accepted because its exclusive
prefix sum is 0 -- and a ramp that starts at 1 and stays at 1 settles at most 47 of 256 positions in
48 steps.

Two consequences for the localisation:

* The target is the accept count and the tail that produces it, not the mean. A change that lowered
  TT's mean entropy without creating sub-0.1-nat positions would fix nothing.
* One number now discriminates the hypotheses cleanly: **logit_std, 7.455 on the reference**. TT's
  mean of 5.10 is above the reference's *most uncertain* position (4.82), which is what a uniform
  reduction in logit scale looks like -- every position flattens and the confident ones stop being
  confident. If TT's logit_std comes in materially below 7.455 this is a scale defect; if it matches,
  the flatness is distributional and the per-layer hidden RMS table is the next cut.

`first_forward_stats.py` computes these for either side and prints them paired, with the per-layer
hidden RMS ratio, so the same statistic is never implemented twice.

### Diagnostic gap: the accept count is not in the halt telemetry

Section 9 quotes TT's per-step accepted count on q106 as 1, 1, 1, ... That came from a dedicated
probe, and it is NOT reproducible from a normal run: the `DG_TRACE_METRIC` halt payload carries
`halt_entropy_per_step` and `halt_mismatch_per_step` but no accept count (keys checked on the
baseline q106 log). Given that the accept count is the statistic that actually decides whether a
block converges, this is the wrong thing to be missing.

Closing it looks cheap -- `write_halt_scalars` already reads back two `[1,1,1,1]` fp32 scalars per
step and the accept mask exists on device at that point, so a third scalar rides the same readback --
but it adds a trace-write target, which must be preallocated before `begin_trace_capture` and warmed
once eagerly like `canvas_buf`. That is a change to the traced hot path and belongs behind a device
run, not ahead of one.

For reference, what the baseline telemetry DOES show on q106 block 0 is a mean entropy that declines
far too slowly to halt: 5.102, 5.573, 5.445, 5.502, 5.461, 5.214, ... 4.077, 3.983 by step 28.

## 13. What the reference is confident ABOUT, and a block-0 RoPE-offset lead (UNTESTED)

Section 12 established that the accept budget comes from a handful of near-zero-entropy positions.
Which positions those are turns out to be the useful question.

### The accept budget is the thinking-template prefix

Decoding the reference's sub-0.5-nat positions on the three dumped questions:

| position | q106 | q096 | q095 | token |
| --- | --- | --- | --- | --- |
| 0 | 4.3e-04 | 2.9e-04 | 4.3e-05 | `<|channel>` (100) |
| 1 | 1.7e-05 | 5.4e-04 | 1.8e-04 | `thought` (45518) |
| 2 | 0.011 | 0.054 | 0.0016 | `\n` (107) |
| 3 | 0.0091 | 0.042 | 0.00096 | `*` (236829) |
| 4 | 0.0042 | 0.044 | 0.00045 | `   ` (139) |
| 5+ | 0.25, 0.47, ... | — | 0.011, 0.25 | first content tokens |

**Positions 0-4, the same five token ids, on all three prompts.** These are the thinking-template
prefix that structurally must follow the generation prompt, which is why the model is certain about
them while every content position sits at ~4.3 nats. The whole block bootstraps from them.

So the block-0 question is sharper than "why are TT's logits flat": **is TT confident at positions
0-4?** Those tokens are determined by the prompt's ending, not by reasoning. `tt_first_forward_dump.py`
now prints positions 0-7 with their entropy, argmax and top1-top2 margin against these reference
values.

### The lead: the canvas is positionally detached from the prompt at block 0

The reference places canvas position 0 immediately after the prompt:

    decoder_position_ids = arange(cache_seq_length, cache_seq_length + canvas_length)

with `cache_seq_length = past_key_values.get_seq_length()`, and HF prefills the prompt UNPADDED, so on
q106 that is 270 — canvas position 0 at absolute 270, distance 1 from the last real prompt token.

TT pads the prompt to a tile multiple before writing K/V (`_pad_prompt_tokens_for_prefill` appends the
padding: `torch.cat([prompt_tokens, padding], dim=1)`), reports `prompt_len=270 cache_len=288`, and
then threads the PADDED length everywhere: `generate_from_prompt_tokens` passes
`prompt_len=prefill.cache_len`, and the adapter does `self.q_rope_offset = prompt_len`. So TT's canvas
position 0 sits at absolute **288**, with the last real prompt token at 269 — a gap of 18 positions.

The coupling that causes it is incidental rather than intended: the reveal mask requires a 32-aligned
`prompt_len` (`update_reveal_mask_buffer` raises otherwise), so the padded length is threaded through,
and `q_rope_offset` rides the same variable. The two are independent — the mask span needs alignment,
the RoPE offset needs the true prompt length.

**This is block-0-only by construction.** At block 1 the committed 256 real tokens occupy 288-543, so
canvas position 0 at 544 is adjacent to real content again. Every later block is contiguous. That
matches the observation that these seven questions collapse on block 0 and the other 57 collapses are
late-block.

### Why this is a lead and NOT yet a cause

Padding size does not separate collapsed from clean: all seven block-0 questions have 7-31 padding
tokens, but so do most clean questions, and the clean group averages MORE padding (18.2 vs 16.3, §11).
So an offset gap cannot be sufficient on its own; at most it is necessary-and-marginal, tipping cases
that are already close. Two facts do keep it interesting: no clean question has zero padding (min 2),
and the positions it would damage are exactly the positions the accept budget depends on.

The test is two device runs on one block-0 prompt, no code change for the first:

1. Run `tt_first_forward_dump.py`. If TT's positions 0-4 carry the reference's ids at near-zero
   entropy, the lead is dead and the per-layer RMS table is the next cut.
2. If TT is uncertain there, decouple the RoPE offset from the mask span (true prompt length for
   `q_rope_offset`, padded span for the reveal mask) and re-run. Confidence returning at positions
   0-4, and the accept count rising above 1, would be the demonstration.

## 14. The RoPE offset MEASURED: 0.50 of the 1.31 nats, and not the collapse

Section 13 recorded the offset as an untested lead. It is now measured, by injecting TT's geometry
into the REFERENCE -- shifting only the canvas position ids, leaving the cached prompt K/V where they
were -- so the mechanism is tested without TT in the loop at all
(`ref_first_forward_dump.py --position-shift`, `hf_reference_trajectory.py --position-shift`).

### At step 1 the shift reproduces TT's signature exactly

| | accept count | positions < 0.1 nats | mean entropy | positions 0-4 argmax |
| --- | --- | --- | --- | --- |
| q106 reference | **6** | 5 | 3.7535 | 100, 45518, 107, 236829, 139 |
| q106 **+18** | **1** | **0** | 4.0075 | 107, 236829, 236829, 209535, 15526 |
| q096 reference | **5** | 5 | 4.0771 | 100, 45518, 107, 236829, 139 |
| q096 **+27** | **1** | **0** | 4.1477 | 818, 2934, 19565, 573, 506 |
| q095 reference | **7** | 6 | 3.4311 | 100, 45518, 107, 236829, 139 |
| q095 **+25** | **1** | **0** | 3.6482 | 140, 236829, 139, 23258, 609 |

The template prefix is wiped on all three: positions 0-4 go from ~1e-05..0.05 nats to 1.0-2.3 nats,
every sub-bound position disappears, and the accept count drops to 1 -- TT's exact signature. Note how
little the MEAN moves (3.7535 to 4.0075) while the accept count collapses 6 to 1, which is section 12's
point restated by experiment.

### But the block still converges, so the offset is not the collapse

Letting the shifted reference run the whole block:

| | step-1 H | trajectory | steps |
| --- | --- | --- | --- |
| q106 reference | 3.794 | 3.79, 3.26, 2.42, 1.66, 1.12, 0.44, 0.11, 0.012, 0.001 | **9** |
| q106 **+18** | 4.295 | 4.29, **4.51**, 3.93, 3.84, 3.55, 3.23, 2.92, 2.25, 1.27, 0.64, ... 0.002 | **17** |
| TT (baseline) | 5.102 | 5.10, **5.57**, 5.44, 5.50, 5.46, 5.21, ... 3.02 at step 48 | **never** |
| q000 reference | 3.765 | ... | 20 |
| q000 **+3** | 3.884 | ... | **11** |

The shift reproduces the SHAPE, including the step-2 entropy RISE that TT shows (4.29 to 4.51 against
TT's 5.10 to 5.57), and it roughly doubles the step count. It does not reproduce the failure: the
block converges in 17 of 48 steps. And with only 3 padding tokens (q000) the shift is inside the
noise -- 11 steps against the baseline's 20, i.e. FASTER -- so the harm scales with the gap rather
than being triggered by any padding at all.

### What this settles

Decomposing q106's step-1 entropy:

    reference                     3.79
    + TT's padded RoPE offset     4.29   (+0.50)
    TT actual                     5.10   (+0.81 unexplained)

* The offset is a **real fidelity bug worth fixing on its own**: it is a divergence from the reference
  (which places the canvas adjacent to an unpadded prompt), it wipes the anchor the accept budget
  bootstraps from, and it costs roughly 2x the denoise steps on q106 -- which is throughput, not just
  fidelity, since every block pays for the steps it takes.
* It is **not sufficient for the collapse**. About 0.81 nats and the failure to converge remain
  unaccounted for, so section 13's "lead" is confirmed as a contributor and refuted as the cause.
* It corrects the mechanism story in sections 9 and 12: **accept = 1 at step 1 is not fatal.** The
  shifted reference also starts at 1 and still converges, because the count RAMPS. What is fatal is a
  ramp that never starts, so "pinned at 1 for 48 steps" is the signature -- not "1 at step 1".

The remaining 0.81 nats is what the per-layer hidden RMS comparison is for, and it is now a smaller
and better-posed target than the 1.31 nats section 9 started from.

## 15. The FULL TT prefill geometry: +0.63 to +1.47 nats, and a quantitative prediction

Section 14 measured a position SHIFT, which is only half of what TT does. TT pads the prompt to a tile
multiple, writes K/V for the pad tokens, and reveals `[0:padded_len]` -- so its canvas also ATTENDS 18
pad-id-0 keys, which in RoPE terms are its nearest neighbours. Injecting that whole geometry into the
reference (`hf_reference_trajectory.py --pad-prompt-to 32`, appending pad-id-0 exactly as
`_pad_prompt_tokens_for_prefill` does) is the faithful reproduction:

| question | arm | step-1 H | step-2 H | steps to halt |
| --- | --- | --- | --- | --- |
| q106 | reference | 3.7945 | 3.2568 | **9** |
| q106 | position shift only | 4.2946 | 4.5087 | 17 |
| q106 | **full TT geometry** | **4.4181** | 3.5479 | 15 |
| q096 | reference | 4.0771 | — | ~13 |
| q096 | **full TT geometry** | **4.7251** | 4.8339 | **33** |
| q095 | reference | 3.4311 | — | ~12 |
| q095 | **full TT geometry** | **4.8974** | 4.4654 | 26 |
| q106 | TT actual | 5.1022 | 5.5731 | **never (48)** |

So the geometry contributes **+0.63 nats on q106, +0.65 on q096, +1.47 on q095** -- on q095 that is
almost the entire gap to TT. Section 14's +0.50 understated it by testing the gap without the attended
pad keys. (Those keys are roughly neutral on their own: 15 steps with them against 17 without, since a
canvas adjacent to garbage is no worse than a canvas adjacent to nothing.)

It also pushes q096 to **33 of 48 steps**, close enough to the cap that the mechanism is visibly the
right kind of thing to explain a failure at 48.

### The prediction this makes

The relationship between step-1 entropy and step count is steep in this range: 4.42 -> 15 steps,
4.73 -> 33 steps, and TT at 5.10 -> more than 48. If the remaining ~0.68 nats on q106 is ordinary
bf16/TP/MoE backbone drift (the known logits PCC ~0.877), then **removing the geometry error alone
should drop TT's step-1 entropy by 0.6-1.5 nats and bring the block back under the convergence
threshold** -- without touching the MoE numerics that #48291 has been stuck on.

That is falsifiable: fix the geometry, re-run the block-0 seven, and either they converge or they do
not.

### The fix shape

Do not thread the padded length into the denoise geometry. Three things currently share one
`prompt_len` that has different requirements:

| consumer | needs | today gets |
| --- | --- | --- |
| prefix KV read span | tile-aligned (`cache_len`) | 288 ✓ |
| canvas RoPE offset (`q_rope_offset`) | TRUE prompt length | 288 ✗ (should be 270) |
| reveal mask content | reveal `[0:true_len]`, hide the rest | reveals `[0:288]` ✗ (pads visible) |

`generate_from_prompt_tokens` passes `prompt_len=prefill.cache_len` and the adapter does
`self.q_rope_offset = prompt_len`; `PromptPrefill` already carries BOTH values (`prompt_len=270`,
`cache_len=288`), so the information is there and only the threading conflates them. Both corrected
consumers keep their buffer SHAPES (the RoPE mats and the mask are same-shaped, different content), so
this stays trace-safe -- it is a content change, like the retention mask in section 10.

## 16. FOUND: the canvas attends the prefill pad keys. Seeded, and the fix is a mask

Sections 13-15 got the framing wrong twice. Both errors are worth stating because each was caused by
a specific sloppiness.

**Error 1 — unseeded step counts.** `hf_reference_trajectory.py` did not seed the initial canvas, so
every arm drew a different one. `ref_seed_sweep.py` had already measured 9-15 steps on q106 across 8
canvases, i.e. more variance than most effects under test, which is why section 14 concluded the
geometry "does not reproduce the collapse". With the canvas seeded (seed 0, identical across arms;
q106's baseline reproduces the sweep's 12 steps for seed 0, so the seeding is doing what it claims):

| question | baseline | **pads attended (TT today)** | **pads hidden** | pads moved before the prompt |
| --- | --- | --- | --- | --- |
| q106 | 12 | **35** | **12** | 11 |
| q096 | 18 | **48 = the cap, i.e. FAILS** | **20** | 13 |
| q095 | 10 | **35** | **11** | 15 |

TT's prefill geometry takes the REFERENCE from 18 steps to the 48-step cap on q096. The collapse
reproduces in the reference implementation, on the reference's own numerics, from geometry alone.

**Error 2 — there is no positional gap.** Sections 13-15 described the canvas as detached from the
prompt by the pad count. In RoPE terms the sequence is contiguous: prompt at 0-269, pads at 270-287,
canvas from 288. Nothing is skipped. The defect is that **the 18 tokens immediately preceding the
canvas are garbage** (pad id 0), so the canvas's nearest context is noise -- which is what destroys
the template-prefix anchor of section 13, since those predictions depend on what directly precedes
them. The distance from the last REAL token is a red herring: hiding the pads leaves the canvas 19
positions from the prompt's end and converges in 12 steps, exactly baseline.

### The fix: hide the pad slots in the reveal mask

`_pad_prompt_tokens_for_prefill` right-pads to a tile multiple and prefill writes K/V for the pads;
the reveal predicate `j < prompt_len` is then evaluated with the PADDED length, so those keys are
revealed. Hiding them is a **content-only change to a same-shaped buffer**, so it is trace-safe in the
same way the retention mask of section 10 is, and it needs no RoPE change at all.

Two alternatives were considered and rejected:

* **Left-padding** (`torch.cat([padding, prompt_tokens])`) also works on the reference -- 11/13/15
  steps -- and would be a one-line change. It is rejected because the prefix KV cache reuse decision
  (APC, #47466) is built on "real tokens then zero-pad": `PrefixKVCache.plan` matches ALIGNED token
  sequences, so with left padding a shared real-token prefix no longer yields a shared aligned prefix
  and prefix reuse would essentially stop matching. `tests/test_prefix_cache.py` encodes that
  assumption in its `_aligned` helper.
* **Moving the RoPE offset to the true prompt length** was the section 15 proposal. Rejected as both
  unnecessary (the offset is not the defect) and invasive: committed blocks live at tile-aligned slots,
  so a true-position RoPE offset would permanently diverge from the slot index and the mask would need
  a hole anyway.

Landed here: `build_canvas_reveal_denoise_mask(..., hidden_prefix_span=(lo, hi))`, which intersects
with the committed and retention predicates (all three are per-KEY, no query dependence). It is INERT
until a caller passes a span -- byte-identical for every existing caller, which the tests pin -- so
this commit changes no behaviour. Mutation-checked: stubbing the one line that applies the span fails
`test_hidden_span_hides_exactly_those_slots` and `test_hidden_span_composes_with_the_retention_window`.

### A cross-check that could have falsified the two-defect split

If attended pad keys are the block-0 cause, then a question whose prompt is ALREADY a 32-multiple has
no pad slots and must not be able to collapse on block 0. Five of the 64 collapses have zero padding:

    q013 prompt_len=160 collapse_block=9    q102 352 -> 12    q075 224 -> 13
    q022 256 -> 14                          q104 256 -> 15

All five collapse LATE (blocks 9-15), none on block 0, and none of the seven block-0 collapses has
zero padding (their pads are 7, 15, 18, 21, 25, 27, 31). So the two mechanisms partition the failures
the way the model says they should: zero-padding questions can only fail through the sliding-window
regime, and two of those five (q022, q104) are already confirmed fixed by the retention arm.

This was a real test rather than a restatement -- a single zero-padding block-0 collapse would have
refuted the pad mechanism as the block-0 explanation.

Still to do, and it is the part that needs the device: thread `(true_prompt_len, cache_len)` from
`PromptPrefill` -- which already carries both values -- to the adapter's mask build, then re-run the
block-0 seven. The prediction is explicit: q106/q096/q095 should go from never-converging to roughly
baseline step counts, without touching MoE numerics.

## 17. GATE RESULT: the retention mask, 64 collapsed questions, and the default flip

`gate/gpqa_sw_arm.sh` ran `DG_DENOISE_SLIDING_WINDOW=1` as the ONLY variable against the shipped
baseline command, over exactly the 64 questions that collapsed in that baseline. Baseline on these 64
was 0 answered, 64 collapsed.

### Outcome, 61 of 64 (three were re-run after a mid-arm mistake, see below)

| | count | |
| --- | --- | --- |
| **cleanly fixed** (no guard fire) | **49** | |
| block-0 set | 7 | the OTHER defect's target (section 16); this arm's negative control |
| reference also fails | 2 | q028, q069 — never this fix's target |
| reference rambles to a wrong answer | 1 | q126 (reference: 11233 tokens, wrong) |
| residual after both fixes | 2 | q127, q128 |

Against the CUDA reference on the same 61 questions:

| | TT (retention on) | reference |
| --- | --- | --- |
| answered | **47 / 61** | 57 / 61 |
| correct | **28 / 61** | 38 / 61 |

and on the 47 TT answered: TT correct 60%, reference correct 64%, **agreement 68%**, median token ratio
TT/reference **0.91**. So TT is not merely emitting something — it is spending the reference's token
budget and landing near the reference's accuracy, having previously emitted nothing at all on every
one of these.

The 10-question gap in "answered" (47 vs 57) decomposes exactly: 7 block-0 + 2 residual + 1 shared
rambling. If the pad fix lands the block-0 seven, this arm's set goes to ~54 of 61 answered.

### The negative control held on all seven

The seven block-0 collapses have prompt_len 167-481, far below the 1023 where retention binds, so the
mask content is identical and their behaviour must not change. All seven fired at the **same block as
the baseline** (`moved +0`), and the collapse signatures match exactly — q007 for instance is
`19/256 distinct ids, top id 236770 covers 67.6%, longest run 30` in BOTH arms.

That rules out a whole class of alternative explanations. If the flag were helping by some diffuse
route -- less attention noise, a different accumulation order -- it would have moved these too.

### Dose-response, not just on/off

q127 is the only one of the 64 whose PROMPT alone exceeds 1023 (2428 tokens), so retention binds for
it from block 0 and its baseline excess-key exposure was the largest in the set. It also gained the
most: collapse moved from block 3 to block 33, ten times the output. The two other late residuals
moved +20 and +15. Improvement tracking exposure is stronger evidence for the mechanism than a binary
fixed/not-fixed count.

### Throughput: the same fix is 1.53x

Blocks that cannot settle burn all 48 denoise steps and still commit an unsettled canvas, so the
fidelity defect was also a throughput defect. Over 22 paired questions:

| | baseline | retention on |
| --- | --- | --- |
| blocks that halt | 238/330 = **72%** | 518/520 = **100%** |
| steady denoise steps/block | — | **0.717x** |
| steady per-block latency | — | **0.652x → 1.53x faster** |
| block throughput | 11.6 tok/blk/s | **17.6 tok/blk/s** |

It also unlocks `DG_DENOISE_SLIDING_SPAN`, which refuses to engage without the retention mask and cuts
SDPA key rows per step by 2.43x bit-identically -- so this gate opens a second perf lever that could
not previously be evaluated at all.

### What is NOT fixed, stated plainly

* **q128** is the one question where retention did nothing: baseline block 9, arm block 8, and the
  reference answers it correctly in 9663 tokens. Not explained by either defect.
* **q127** improved hugely but still collapses at block 33.
* q126/q127/q128 all collapse into the SAME token, `\n` (107), 1-3 distinct ids on the canvas. The
  block-0 set collapses into different tokens (`1`, `*`, ...). Two different degeneracies: the block-0
  set never fills its canvas, while these three fall into a newline loop deep in a long generation —
  a regime where the reference is also weak (q126: 11233 tokens to a wrong answer; q069 and q028: the
  full 16384 with no answer at all).

### A mistake in the run, and what it cost

Three questions (q091, q110, q129) exited 1 with `TypeError: ... unexpected keyword argument
'true_prompt_len'` because I edited `tt/denoise_forward.py` WHILE the arm was running and each
question launches a fresh process that imports current source. The tree was restored, those three
results deleted, and they are re-run by the same resumable script. The rule that follows: never edit
source the running experiment imports — the arm's later stages now wait on an explicit sentinel file
so source edits happen with the device idle.

## 6. Reproduce

```bash
# RNG health, kernel level (fails on the pre-fix kernel)
DG_RUN_DEVICE=1 pytest tests/ttnn/nightly/unit_tests/operations/rand/test_rand_independence.py -s

# canvas-position independence of the DG draw
DG_RUN_DEVICE=1 pytest models/experimental/diffusion_gemma/tests/test_device_gumbel_position_correlation.py -s

# which halt gate blocked, per block, on any traced run
grep DG_TRACE_METRIC <log> | grep upfront_replay   # halt_blocking_gate, halt_entropy_*, halt_mismatch_*

# degeneracy statistics per committed canvas
grep DG_DEGENERACY <log>
```
