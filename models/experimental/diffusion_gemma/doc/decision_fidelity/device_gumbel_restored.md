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
