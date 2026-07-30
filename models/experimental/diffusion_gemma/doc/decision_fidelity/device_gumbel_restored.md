# Device Gumbel restored, and the two TT-specific collapse defects (#48291)

Status: current. Three defects were found and fixed here — the `ttnn.rand` PRNG, the axis DG drew
Gumbel noise on, and (later) two geometry defects in the denoise mask. The 2026-07-26 header this
file used to carry — "the residual collapses are the #48291 fidelity gap, not the sampler" — is
**superseded by its own §5/§6 below**: the residual block-0 collapses were the prefill pad geometry.
Owns: the `ttnn.rand` defect + kernel fix + measured dead ends, the permuted-vocab layout defect and
its refuted workarounds, the two collapse mechanisms and the halt telemetry that separates them, the
CUDA reference arm, and both landed mask fixes with their gate results.
See also: [refuted list](../REFUTED.md) · [decision fidelity](README.md) · [degeneracy guard](degenerate_output_fix.md) · [plan](../../plan.md)

Over the 200-line target: everything below is a refutation, a gate result, a measurement trap or a
reproduction path.

## 1. The RNG defect and its kernel fix

### 1a. `ttnn.rand` is a sliding window over ONE stream

Mapping one 32×32 rand tile element by element: **94 distinct values in 1024 slots**. The PRNG is
per-lane (32/32 distinct inside one 32-lane SFPU vector), but only 20 of a tile's 32 vector draws were
distinct, with the exact relation

    (face f, vector 2k) == (face f+1, vector 2k+1)

i.e. in tile coordinates "column `c` is byte-identical to column `c-24` for `c % 32 >= 24`". Element
`(read t, lane i)` carries `stream[t + i]`: one window advancing about one element per read while all
32 lanes read overlapping positions.

**Fix:** consume several PRNG values per stored element, in
`tt_metal/hw/ckernels/blackhole/.../ckernel_sfpu_rand.h`. Kernels are JIT — edit the header and clear
`~/.cache/tt-metal-cache`; no rebuild.

| | tile distinct | distinct vector draws | byte-identical rows | max abs r | max argmax mult |
|---|---|---|---|---|---|
| before | 94/1024 | 20/32 | 64/256 | 1.00000 | 11 |
| after | 214/1024 | 32/32 | **0/256** | 0.618 | 5 |
| host IID | — | — | 0/256 | 0.035 | 2 |

`max abs r = 1.00000` does not mean "highly correlated" — it means whole noise **rows are identical**.
The uniform distribution is untouched: mean 0.4994, std 0.2887, top decile 0.0996 over 524288 samples.

**Residual, stated plainly:** the fix DILUTES the lane/stream degeneracy rather than removing it —
cross-position max |r| is 0.618 against 0.035 for host IID. A full fix needs a counter-based RNG keyed
on each element's own position, which this instruction sequence has no lane index to build from.

Five measured dead ends and one trap (NOP spacing, xorshift32, `SFPTRANSP`+XOR fold, the two DG-local
layout workarounds, the ttnn per-core seed) are one-liners in the
[refuted list](../REFUTED.md#sampling-rng-and-decision-fidelity). The trap worth repeating inline:
**holding `scale`/`from` in lreg4/lreg5 across a transpose silently breaks the output range to mean
0.35 / std 0.64 — which PASSES the correlation metrics by destroying the distribution instead.**

### 1b. DG was drawing the noise on the WRONG AXIS

`sample_gumbel_noise_with_permuted_vocab` keeps vocab off `ttnn.rand`'s innermost axis by collapsing
every OTHER axis into it — and for the production shape `(1, 1, 256, vocab)` that is **the 256 canvas
positions**. It moved the degeneracy from an axis where it biases WHICH token a position picks onto
the axis where it makes DIFFERENT POSITIONS pick the same token, which is what collapses a canvas.
(The docstring claiming the permuted path "avoids that correlation" was measured false and corrected
in `tt/sampling.py`.)

Measured at production geometry (canvas 256, vocab 262144), with the kernel already fixed:

| layout | max abs r | distinct flat-logit winners | max multiplicity |
|---|---|---|---|
| permuted (was production) | 0.598 | 154/256 | 8 |
| **vocab innermost (now)** | 0.350 | **253/256** | **2** |
| host IID control | 0.009 | 256/256 | 1 |

### The blind spot, the metrics, and the upstream attribution

The only gate on this path, `tests/test_device_canvas_sampling_dist.py`, **cannot see cross-position
correlation**: it averages over a sample axis into per-position marginals, and correlation *between*
positions leaves every marginal correct. Independence across positions was never tested until
`tests/test_device_gumbel_position_correlation.py`.

Two metrics, both calibrated against a host torch-Gumbel IID control at the same shape:
**exact-duplicate rows** (how many of the 256 position rows are byte-identical to another) and
**flat-logit winner multiplicity** (with flat logits the winner at each position is the argmax of that
position's noise row; over this vocab 256 IID winners essentially never collide, so
`distinct_winners ≈ 256` and `max_mult ≈ 2`). The second is the **functional** metric — exactly the
"synchronized same-token burst" texture, worst where the logits are flattest.

Pre-fix arms (canvas 256, vocab 16384, one seed; 262144 identical where measured) as
unique-rows / distinct-winners / max_mult / max |r|: host IID 256/256, 255/256, 2, 0.035 · `permuted`
(then the default) 192/256, **119/256**, 11, 1.00000 · `chunked` 1024 160/256, 156/256, 6, 1.00000 ·
`chunked` 2048 160/256, 155/256, 6 · `plain` vocab-innermost 160/256, 157/256, 4.

Root cause, confirmed element-for-element: the duplicate set is **exactly** `{i : i % 32 >= 24}` (8 of
every 32 rows, 64 of 256); every duplicate row equals row `i-24`; the pattern is **independent of the
other axis extent**; and it is present in the **raw** `ttnn.rand((vocab, 256))` output before DG's
permute/reshape, so DG's layout code is exonerated. The vocab-innermost layout showed the same class
of defect with a different constant (offset 17, 96 of 256 duplicated), so it is a property of
`ttnn.rand`'s row-stream assignment, not of which semantic axis is innermost. The reverted
per-core-seed experiment bought the attribution: the defect is **inside a single core's tile**, in the
SFPU PRNG path `compute_uniform.cpp` → `ckernel_sfpu_rand.h` (8 SFPU draws per face, 4 faces per
tile), along the tile's WIDTH axis. Filed upstream as
`tests/ttnn/nightly/unit_tests/operations/rand/test_rand_independence.py` — two `xfail(strict)` tests
pinning the width-axis properties plus a passing cross-tile control that is the arm ruling out the
seeding hypothesis; `strict=True` flips to a failure the day the op is fully fixed, and the
duplicate-column half already passes. Scan scripts live beside this file:
`probe_gumbel_dup_structure.py`, `probe_gumbel_tile_fix.py`, `probe_gumbel_chunked_arm.py`.

**Device gate:** matched 4-seed A/B, shipped serving configuration, GPQA doc0 in thinking mode —
8/8 correct, guard never fired. doc6 and doc9, which previously emitted ZERO characters because the
guard fired on block 0, now generate normally, and doc9 answers `B`, matching the CUDA reference.

## 2. The two collapse mechanisms, and the telemetry that separates them

Both end with a committed canvas dominated by one token, so token counts alone cannot tell them apart.

| | CONVERGED | NON-CONVERGED |
|---|---|---|
| entropy trajectory | 4.3–5.1 → **~0.01** | 3.6 → **2.4**, stays high |
| `halt_mismatch_final` | **0** (argmax settled) | **109** of 256 still flipping |
| `halt_blocking_gate` | `entropy` | `both` |
| cause | correlated noise across positions | the fidelity gap vs the reference |
| owner | **fixed in §1** | the defects in §4–§5 |

**"Entropy converged" cannot distinguish a finished answer from a collapsed canvas** — both report
convergence. That is why the degeneracy guard measures committed tokens instead
([degeneracy guard](degenerate_output_fix.md)).

Every generation knob matches the released `generation_config.json` item by item, so NON-CONVERGED is
not a settings bug — see [plan §1](../../plan.md#1-model-and-generation-procedure).

## 3. The CUDA reference arm (what is inherent, what is ours)

Full GPQA-Diamond, thinking, through the CUDA vLLM server — same model, no TT sampler anywhere:

| | @8192 | the 45 unanswered, retried @16384 |
|---|---|---|
| accuracy | 124/198 (62.6%) | 24/45 rescued → **148/198 (74.7%)** |
| answered | 153/198 (77.3%) | 40/45 (88.9%) |
| COLLAPSE (≥20 identical consecutive tokens) | **0/198 (0.0%)** | — |
| tail repetition | 44/198 (22.2%) | 22/45 (48.9%) |
| no answer at the cap | 45/198 (22.7%) | 5/45 (11.1%) |

**Canvas collapse is NOT inherent — the GPU never does it. That mode was ours.**

- **TRAP:** of the 40 questions rescued at 16384, **16 answered in under 8192 tokens** — the first run
  simply took a looping trajectory. vLLM's continuous batching is numerically batch-dependent, so even
  at temperature 0 the same question can go either way; **any truncation rate measured with batching
  carries that variance.**
- **Refuted (2026-07-26): "the collapse is mostly inherent."** Cross-tabulating question by question
  over the 131 questions where both arms have results: reference answered 126/131 (96%), correct 97
  (74%); TT (HiFi2, 16K, then-shipped config) answered 64/131 (49%), correct 58 (44%). **60 of TT's 64
  collapses are on questions the reference ANSWERS**; only 4 (q028, q036, q069, q079) are hard for
  both — 3% of the set, not 23%. Nor are the 60 merely long-reasoning: the reference's token use on
  them is median 6744, min **1561**, max 14647, and TT's truncation rate at 16K is 0%.
- **TRADE-OFF at larger context:** truncation goes to 0% at 16K but the collapse rate rises (30% at
  8192/12 blocks vs ~43% at 16K/60 blocks), because a larger budget gives the NON-CONVERGED mode more
  blocks in which to churn.
- Note what the reference does NOT have: a guard. At its own cap it commits the unsettled positions
  anyway — that is the 22.2% tail repetition. TT's difference is that the uncertain positions collapse
  onto ONE token and form a wall, which the guard then refuses.

## 4. The mechanism, and defect A: sliding-layer key retention (late blocks)

**The arithmetic.** The entropy-bound rule accepts the k lowest-entropy positions whose EXCLUSIVE
prefix sum of entropies is ≤ `entropy_bound` (0.1 nats, absolute). On q106 TT's per-step accepted count
is pinned at exactly **1**, so 48 steps settle at most 47 of 256 positions and the clean argmax over a
four-fifths-random canvas returns the **unigram prior** — precisely the observed collapse set: `-` (14
of 64), `\n` (11), `1` (7), `' the'` (4), `0`, `,`, `.`, `2`, `' '` (ids 236743–236780 are the
single-character tokens ordered by English letter frequency). A CONVERGING block looks completely
different: q007 ramps 1, 2, 3, 5, 6, 7, 10, 16, 23, 51, 75, 121 … 252 and halts.

Step-1 mean entropy on q106 ruled out five hypotheses in one table — HF reference bfloat16 CPU
**3.7495** vs TT traced sparse MoE HiFi2 **5.1022**, TT eager 5.1022, TT self-conditioning disabled
5.3681, TT traced dense-128 MoE 5.0913. So the gap is **not** bf16 precision (the reference reaches
3.75 *in bfloat16*), **not** the MoE variant, **not** traced-vs-eager, **not** self-conditioning and
**not** the step budget. *(Provenance: the two TT MoE rows are denoise paths deleted 2026-07-29;
neither can be re-run and any absolute number from them is void as a current result.)*

- **The accept budget is the thinking-template prefix.** Decoding the reference's sub-0.5-nat
  positions gives the SAME five token ids at positions 0–4 on all three dumped prompts:
  `<|channel>` (100), `thought` (45518), `\n` (107), `*` (236829), `   ` (139) — while every content
  position sits at ~4.3 nats. The whole block bootstraps from them.
- **The mean is the wrong statistic** (refuted, see the [refuted list](../REFUTED.md)). The
  reference's step-1 distribution is bimodal: q106 mean 3.7535 but **median 4.2624**, min 1.7e-05,
  only 5 positions below 0.1 nats, accept count **6/256**, `logit_std` 7.4553 (q096:
  4.0771/4.2541/2.9e-04/5/5, 8.1133; q095: 3.4311/3.9226/4.3e-05/6/7, 7.4228). **The reference also
  starts at only 5–7 accepted of 256** — it converges because the count RAMPS. What is fatal is a ramp
  that never starts.
- **UNMEASURED DISCRIMINATOR (open):** `logit_std` is 7.455 on the reference and has never been
  measured on TT. Materially below ⇒ a scale defect; matching ⇒ the flatness is distributional and the
  per-layer hidden RMS table is the next cut. `first_forward_stats.py` computes both sides and prints
  them paired with the per-layer hidden RMS ratio.
- **DIAGNOSTIC GAP (open):** the accept count is **not** in the halt telemetry — the `DG_TRACE_METRIC`
  payload carries `halt_entropy_per_step` and `halt_mismatch_per_step` but no accept count, so the
  "1,1,1,…" signature is not reproducible from a normal run. Closing it looks cheap (a third fp32
  scalar on the existing `write_halt_scalars` readback) but adds a trace-write target that must be
  preallocated before `begin_trace_capture` and warmed eagerly.

**Where the other 57 collapses were.** `promptlen_vs_collapse.py` shows prompt length separates
nothing (collapsed median 258 vs clean 205), but the *committed prefix* grows 256 tokens per block:

    collapse block histogram: 0:7  1:1  3:1  7:1  9:3  10:6  11:4  12:10  13:12  14:15  15:3  17:1
    56 of 64 collapses happen AT OR AFTER the block where the committed prefix crosses 1023.

**Why 1023.** DiffusionGemma is 25 `sliding_attention` + 5 `full_attention` layers (window 1024) — 83%
sliding. HF's *mask* over the canvas is deliberately full, but the *cache* still evicts: sliding layers
retain only `sliding_window - 1` = **1023** committed keys. Measured on the reference rather than taken
from a docstring (`ref_sliding_retention.py`, A100, 1500-token prompt): layer 0 sliding cached keys
1023, `get_mask_sizes` (1279, 477); layer 5 full 1500, (1756, 0). TT's maskless all-attend denoise
therefore attends keys the reference does not have, and the excess grows every block — block 4 ~1290
vs 1023 = 267 (21%); block 9 ~2570 = 1547 (60%); block 13 ~3600 = 2577 (72%). That growth curve is the
shape of the histogram.

**The fix was already written and left switched off.** `denoise_forward.denoise_sliding_window_enabled`
(#51080) implements the retention as reveal-mask content per layer type and shipped OFF because it is
decision-CHANGING above `prompt_len = 1024` and its gate was an agreement run against fp32 HF. The
collapse histogram is the evidence that gate was waiting for. 95 host tests already covered the mask
semantics (`test_denoise_sliding_window.py`, `test_hf_sliding_window_reference.py`,
`test_attention_mask.py`, `test_paged_prefix_reveal_mask.py`).

### Gate result (`gate/gpqa_sw_arm.sh`, `DG_DENOISE_SLIDING_WINDOW=1` as the ONLY variable)

Over exactly the 64 questions that collapsed in the baseline (baseline: 0 answered, 64 collapsed), 61
scored: **49 cleanly fixed**; 7 are the block-0 set (defect B's target, this arm's negative control);
2 the reference also fails (q028, q069); 1 the reference rambles to a wrong answer (q126, 11233
tokens); 2 residual (q127, q128). Against the reference on those 61: TT answered **47/61** vs 57/61,
correct **28/61** vs 38/61; on the 47 TT answered, TT correct 60% vs reference 64%, **agreement 68%**,
median token ratio TT/reference **0.91**.

- **The negative control held:** all seven block-0 questions (prompt_len 167–481, far below 1023) fired
  at the SAME block as baseline (`moved +0`) with matching signatures — q007 is `19/256 distinct ids,
  top id 236770 covers 67.6%, longest run 30` in BOTH arms. That rules out a diffuse-help explanation.
- **Dose-response:** q127 is the only one of the 64 whose PROMPT alone exceeds 1023 (2428 tokens), so
  retention binds from block 0; its collapse moved from block 3 to block 33, ten times the output. The
  two other late residuals moved +20 and +15.
- **Throughput: the same fidelity fix is 1.53×.** Over 22 paired questions, blocks that halt
  238/330 = 72% → **518/520 = 100%**; steady denoise steps/block **0.717×**; steady per-block latency
  **0.652×**; block throughput 11.6 → **17.6 tok/blk/s**. It also unlocked the bounded sliding read
  (2.43× fewer SDPA key rows per step, bit-identical), which refuses to engage without the retention
  mask; that read has been unconditional since 2026-07-29.

### Regression arm and verdict: repairs 52, regresses 1, default stays ON

On the 67 already-clean questions with retention ON: **1 new collapse (q103)**, answered 65/67,
correct **56/67** (reference 58/67 on the same set), agrees with the reference 58/67, blocks that halt
447/453 = **99%**. Verdict: repairs **52** of the 64 collapsed, regresses **1** of the 67 clean, net
**+51**. Across all 131 questions TT's correct count goes **58 (44%) → 86 (66%)** against the
reference's 97 (74%).

**The one regression, characterised and not rounded away.** q103 was correct under the baseline
(answer B, gold B, 15 blocks, 14/15 halting) and collapses at block 8 with retention on (1/256 distinct
ids, all token `1`). Its first three blocks are step-for-step identical (24, 20, 19) and divergence
begins at block 3 — exactly where its committed prefix crosses 1023. So this is the mechanism acting,
not a bug in it: on this question the reference's own geometry sends the trajectory somewhere worse.
The reference answers it in 2374 tokens. It reproduced identically across a machine reboot.

## 5. Defect B: the canvas attends the prefill pad keys (block 0)

`_pad_prompt_tokens_for_prefill` right-pads the prompt to a tile multiple, prefill writes K/V for the
pads, and the reveal predicate `j < prompt_len` is evaluated with the PADDED length — so those keys are
revealed. **There is no positional gap:** in RoPE terms the sequence is contiguous (prompt 0–269, pads
270–287, canvas from 288). The defect is that **the 18 tokens immediately preceding the canvas are
garbage (pad id 0)**, so the canvas's nearest context is noise, which destroys the template-prefix
anchor the accept budget bootstraps from. Hiding the pads leaves the canvas 19 positions from the
prompt's end and converges in 12 steps, exactly baseline.

Measured by injecting TT's geometry into the REFERENCE, so the mechanism is tested with no TT in the
loop (`ref_first_forward_dump.py --position-shift`, `hf_reference_trajectory.py --position-shift
| --pad-prompt-to 32`):

- **A position shift alone reproduces TT's step-1 signature exactly** — accept count 6→1 (q106, +18),
  5→1 (q096, +27), 7→1 (q095, +25); all sub-bound positions disappear; the template prefix is wiped.
  Note how little the MEAN moves (3.7535 → 4.0075) while the accept count collapses.
- **But the shifted block still converges** (q106 in 17 of 48 steps), so the offset is a contributor,
  not the cause. Decomposition on q106: reference 3.79, + TT's padded RoPE offset 4.29 (+0.50), TT
  actual 5.10 (**+0.81 unexplained**). This also corrects the mechanism story: **accept = 1 at step 1
  is NOT fatal** — the shifted reference also starts at 1 and converges because the count ramps.
- **The FULL TT prefill geometry** (pads attended too) contributes +0.63 nats on q106, +0.65 on q096,
  **+1.47** on q095, and pushes q096 to 33 of 48 steps.
- **ERROR acknowledged, and the rule it produced:** `hf_reference_trajectory.py` did not seed the
  initial canvas, so every arm drew a different one — and `ref_seed_sweep.py` had already measured
  9–15 steps of canvas variance, more than most effects under test. **SEED THE CANVAS.** With seed 0
  held across arms: q106 baseline 12 / pads attended **35** / pads hidden 12; q096 18 / **48 = the cap,
  i.e. FAILS** / 20; q095 10 / **35** / 11. TT's prefill geometry takes the REFERENCE to its cap on
  q096, on the reference's own numerics, from geometry alone.

**The fix:** `build_canvas_reveal_denoise_mask(..., hidden_prefix_span=(lo, hi))` — a CONTENT-ONLY
change to a same-shaped buffer, so it is trace-safe like the retention mask, and it needs no RoPE
change. It intersects with the committed and retention predicates (all three are per-KEY, no query
dependence) and is INERT until a caller passes a span. Mutation-checked: stubbing the one line that
applies the span fails `test_hidden_span_hides_exactly_those_slots` and
`test_hidden_span_composes_with_the_retention_window`. Two alternatives were rejected (left-padding;
moving the RoPE offset to the true prompt length) — see the [refuted list](../REFUTED.md).

**A cross-check that could have falsified the two-defect split.** If attended pad keys are the block-0
cause, a prompt that is ALREADY a 32-multiple has no pad slots and must not collapse on block 0. Five
of the 64 collapses have zero padding — q013 (160) → block 9, q102 (352) → 12, q075 (224) → 13, q022
(256) → 14, q104 (256) → 15 — and **all five collapse LATE, none on block 0**; none of the seven
block-0 collapses has zero padding (7, 15, 18, 21, 25, 27, 31). A single zero-padding block-0 collapse
would have refuted the pad mechanism.

### Gate result (`gate/block0_padfix_arm.sh`, both arms of `DG_DENOISE_HIDE_PREFILL_PADS`)

The prediction was written into the script BEFORE the run. **Seven of seven block-0 collapses fixed**;
block 0 halts in every case, where six of seven previously ran the full 48 steps and committed an
unsettled canvas. Guard fired on none.

| question | pads attended: step-1 H / steps / halted | pads hidden: step-1 H / steps / halted |
|---|---|---|
| q106 | 5.1022 / 48 / no | **4.0866 / 15 / yes** |
| q096 | 5.1080 / 48 / no | **4.0478 / 17 / yes** |
| q090 | 4.8282 / 48 / no | **4.4389 / 36 / yes** |
| q122 | 4.5197 / 48 / no | **4.0363 / 24 / yes** |
| q095 | 4.1141 / 36 / yes | **3.7101 / 19 / yes** |
| q064 | 4.1127 / 48 / no | **3.9742 / 21 / yes** |
| q007 | 3.6367 / 48 / no | **3.3900 / 16 / yes** |

The control arm reproduced the original measurement **exactly**: 5.102173 against the 5.1022 recorded
before the fix. And **q007 is the proof that mean entropy is the wrong proxy**: its step-1 entropy was
already 3.6367 and hiding the pads moved it only −0.25, yet block 0 went from 48 steps without halting
to 16 with. A fix chosen by watching mean entropy would have looked useless.

## 6. What is still NOT fixed

> **OPEN CONTRADICTION (unexplained):** q128 is the one question where retention did nothing —
> baseline block 9, retention arm block 8 — while the CUDA reference answers it correctly in 9663
> tokens. Neither landed defect accounts for it. Not explained.

- q127 improved hugely but still collapses at block 33.
- q126/q127/q128 all collapse into the SAME token, `\n` (107), with 1–3 distinct ids on the canvas — a
  newline loop deep in a long generation, a different degeneracy from the block-0 set (which never
  fills its canvas) and a regime where the reference is also weak (q126: 11233 tokens to a wrong
  answer; q069 and q028: the full 16384 with no answer at all).
- The +0.81 nats unexplained on q106 after the geometry is removed, and the unmeasured `logit_std`
  discriminator (§4).

**Cheap committed reproducer sets:** the seven block-0 questions q106, q096, q122, q095, q007, q090,
q064 — q106 is cheapest, the reference spends 1561 tokens on it; and the 60 questions the reference
answers and TT collapsed, ordered shortest-by-reference-token-use first.
`hf_reference_trajectory.py` records the reference's per-step trajectory in TT's own telemetry units by
wrapping `StableAndConfidentStoppingCriteria`; it needs no model changes and runs in **53 seconds** on
the QB2 host CPU in bf16.

## 7. Process rules earned in these runs

- **Never edit source the running experiment imports.** Three questions (q091, q110, q129) exited 1
  with `TypeError: unexpected keyword argument 'true_prompt_len'` because `tt/denoise_forward.py` was
  edited WHILE the arm ran, and each question launches a fresh process that imports current source.
  The arm's later stages now wait on an explicit sentinel file.
- **A device fault is distinguishable from a real failure.** `SIGBUS` / "Non-existent physical address"
  inside UMD `write32_to_device` shows up as exit code **135**, a stack inside UMD rather than model
  code, and `window_active=0` — the marker saying the run never reached the flag it was testing.
  `tt-smi -r` clears it; such runs are excluded and re-run.
- **Recovery:** a box reboot cleared `/tmp` and destroyed the 131-question baseline, both arms' results
  and the rendered prompts. Prompts were regenerated from the A100's `gpqa_run.py` and verified 198/198
  against the surviving gold oracle; the recovery script and oracle are committed.
- **A fidelity change must be judged on a set, never one prompt.** HiFi4 rescues q007 but tips q012
  (45 steps halted → 48 not halted, mismatch 12): many blocks land right at the step cap, so any
  perturbation flips some in either direction. *(Provenance: both HiFi arms ran on the token-gather
  MoE deleted 2026-07-29, so those step counts are not a re-runnable A/B — the rule is.)*

## 8. Reproduce

```bash
# env: see plan.md
# RNG health at the kernel level (fails on the pre-fix kernel)
DG_RUN_DEVICE=1 pytest tests/ttnn/nightly/unit_tests/operations/rand/test_rand_independence.py -s

# canvas-position independence of the DG draw, and the production 262144 geometry
DG_RUN_DEVICE=1 pytest models/experimental/diffusion_gemma/tests/test_device_gumbel_position_correlation.py -s
DG_RUN_DEVICE=1 DG_GUMBEL_CORR_FULL_VOCAB=1 pytest \
  models/experimental/diffusion_gemma/tests/test_device_gumbel_position_correlation.py -s -k production_vocab

# which halt gate blocked, per block, on any traced run
grep DG_TRACE_METRIC <log> | grep upfront_replay   # halt_blocking_gate, halt_entropy_*, halt_mismatch_*

# degeneracy statistics per committed canvas
grep DG_DEGENERACY <log>
```
