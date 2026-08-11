# The cell's real defect: v3 shards the norm in decode only, and reads a cache built without it

**One line of guard decides gemma-4-26B `-onA`'s sliding kind — −5,919 µs/model, 39 % of v2's corpus total.**

Measured on device 2026-08-11 with the model's own oracle (`test_optimized_real_weights_prefill_decode`, layer 0,
`shared_physical_cache=true`, real weights via `GEMMA4_RANGE_DOWNLOAD=1`, bar 0.995), in two isolated worktrees —
one at v2's tag `ad3ca71d8`, one at the v3 stage tag. Nothing varied but the core count and the guard.

| sliding cores | v2 tree | v3 tree + v2-style guard | v3 tree **as shipped** (decode-only) |
|---:|---:|---:|---:|
| 0 — interleaved | 0.9996280142258483 ✅ | — | 0.9996280142258483 ✅ |
| 11 | 0.9943331194625922 ❌ | **0.9943331194625922** ❌ | 0.9945729603715616 ❌ |
| 22 | 0.9942874693564726 ❌ | **0.9942874693564726** ❌ | 0.9944099795374435 ❌ |
| 44 | 0.9941146130802025 ❌ | **0.9941146130802025** ❌ | 0.9945729603715616 ❌ |
| **88** | **0.9996293363224806 ✅** | **0.9996293363224806 ✅** | 0.9943716809625597 ❌ |

*(v2's tree only admits multiples of 11 that divide 2816 evenly, so 11/22/44/88 is its whole legal ladder;
33/55/66/77 raise on its own width check.)*

## What this says

1. **The two trees are numerically identical to sixteen digits at every rung, once the guard matches.** So the
   difference I chased for two rounds — v2 leaves the norm weight interleaved, v3 reshards it — **does not
   matter**, exactly as the isolated op test said.
2. **The single difference that decides the cell is the guard.**

   | | v2 | v3 |
   |---|---|---|
   | condition | `x.shape[-2] > TILE_SIZE` → skip | `self._executing_decode` → else skip |
   | effect | sharded norm in **prefill *and* decode** | sharded norm in **decode only** |

   At seq_len 32 v2's shape test passes in prefill too; v3's phase test does not. So **v3 builds its KV cache with
   interleaved norms and then reads it with sharded ones** — the two phases disagree about how the residual stream
   is normalised.
3. **That inconsistency costs ~5 × 10⁻³ of layer PCC at every grid, and at 88 cores it turns a pass into a fail.**
   v3's shipped tree scores 0.9943717 at 88 where v2 scores 0.9996293.
4. **88 cores is genuinely special, and it is the only rung that passes.** At 2816/88 = 32 — **exactly one tile per
   core** — the reduction is a pure cross-core tree with no intra-core sequential accumulation, and it is *as
   accurate as interleaved* (0.9996293 vs 0.9996280) while being 13 % faster on the layer. Every other rung has
   ≥ 2 tiles per core, mixes intra- and cross-core accumulation, and lands at 0.9941–0.9946. Prefill shows the same
   shape: **0.9988100 at 88 against an unsharded 0.9986203**, versus 0.9978658 / 0.9981087 / 0.9981995 at
   11/22/44. **The ladder is non-monotonic, with the optimum at the maximum** — which the advisor advised.

## Why the isolated sweep missed it

[`NORM-GRID-SWEEP`](ADVCHAL-V3-NORM-GRID-SWEEP.md) ran 79 configurations and found the op grid-insensitive to
7.3 × 10⁻⁷. That was true, **and it used the decode-shaped input throughout — `[1,1,1,2816]`, one real row and
thirty-one of padding.** The effect lives in the **prefill** norm, `[1,32,2816]`, thirty-two real rows, which the
reconstruction never ran. → [`PITFALLS`](ADVCHAL-V3-ANALYST-PITFALLS.md) ERROR 15.

## So v3 lost this cell to two of its own defects, independently

| | |
|---|---|
| **88 was not on v3's ladder** — it swept 2/4/8/11/22/44 and stopped | the only passing rung was the one never measured |
| **v3's guard is decode-only** | even with 88 on the ladder, v3's tree scores 0.9943717 there and fails |

Either defect alone loses the cell. **The veto was correct for the tree it was applied to, and the tree was
wrong.** And because the verdict was hardcoded as `passed = kind == "full_attention"` with no oracle log committed,
none of this was visible from the artefacts.

## Actions

1. **Change v3's guard to shard the norm in prefill and decode** — v2's condition, phase-consistent. One line, and
   worth **−5,919 µs/model** on this cell alone.
2. **Put the advised grid on the ladder.** 88 was both the advised value and the only passing rung.
3. **Any placement knob gated on execution phase must have its cross-phase consistency asserted**, because the KV
   cache carries the disagreement into the measurement. Generalises beyond this op.
4. **`oracle_passed` computed from a parsed, provenanced oracle artefact** — unchanged, and it is what would have
   surfaced 1–2 during the run instead of a week later.
5. **Re-examine every other cell for the same phase asymmetry.** phi and north-mini also gained decode-only knobs
   in v3, and the same class of defect would be invisible in exactly the same way.


---

# The isolation, continued: mechanism confirmed, defect localised to ONE call, and the win measured

Three further experiments on 2026-08-11, same oracle, same worktree method.

## 1. The mechanism is phase *mismatch*, in either direction

`NORM_PHASE` parameterised so the sharded norm can be applied in prefill only, decode only, or both:

| cores | phase | decode PCC | prefill PCC | |
|---:|---|---:|---:|:--|
| 88 | **decode only** *(v3 as shipped)* | 0.9943716809625597 | 0.9986202564547553 | ❌ |
| 88 | **prefill only** | **0.9943996010696218** | 0.9988099561825600 | ❌ |
| 88 | **both** | **0.9996293363224806** | 0.9988099561825600 | ✅ |
| 11 | decode only | 0.9945729603715616 | 0.9986202564547553 | ❌ |
| 11 | prefill only | 0.9942125864050061 | 0.9978658021847042 | ❌ |
| 11 | both | 0.9943331194625922 | 0.9978658021847042 | ❌ |

**Sharding either phase alone costs ~5.2 × 10⁻³ — the same, in both directions.** Only *agreement* between the
phases recovers it. That is the mismatch hypothesis measured rather than inferred, and it needed the prefill-only
arm to be a test rather than a story: had prefill-only passed, the explanation would have been "sharding the decode
norm is harmful", which is a different defect with a different fix.

**And 88 needs both conditions.** Phase agreement alone is not enough — 11/22/44 fail at `both` too, because
≥ 2 tiles per core is genuinely less accurate. Only **88, with agreement**, clears the bar.

## 2. `full_attention` does not suffer it — which is why nobody noticed

| kind | cores | phase | decode PCC | |
|---|---:|---|---:|:--|
| full (layer 5, natural cache) | 8 | decode only *(v3 shipped this)* | 0.9997999978731844 | ✅ |
| full | 8 | both | 0.9997928485426096 | ✅ |
| full | 88 | decode only | **0.9997999978731844 — identical to 8 cores** | ✅ |
| full | 88 | both | 0.9997872958305739 | ✅ |

**The mismatch costs `full_attention` 7 × 10⁻⁶ and `sliding_attention` 5.2 × 10⁻³ — a factor of 740.** So the cell
shipped the mismatched guard on the kind where it is free and was vetoed on the kind where it is not, and the
`full` result that *passed* is itself sitting on the same latent defect with an accidental margin.

Note also that at `decode only` **full is identical at 8 and 88 cores to sixteen digits** — the op really is
grid-insensitive, exactly as [`NORM-GRID-SWEEP`](ADVCHAL-V3-NORM-GRID-SWEEP.md) measured. Sliding's grid
sensitivity comes from downstream discontinuity, not from the reduction.

## 3. One of eight norm sites carries the entire cost

`residual_norm_drop_index` leaves a single site interleaved. At 88 cores, decode-only (the failing config):

| dropped site | decode PCC | |
|---|---:|:--|
| none | 0.9943716809625597 | ❌ |
| **1** | **0.9996226684246500** | ✅ **recovered** |
| 2 | 0.9944871257200655 | ❌ |
| 3 | 0.9943649965713889 | ❌ |
| 4 | 0.9943964814206211 | ❌ |
| 5 | 0.9943716809625597 | ❌ |
| 6 | 0.9943819709107029 | ❌ |
| 7 | 0.9943574112542729 | ❌ |
| 8 | 0.9943952829240805 | ❌ |

**Site 1 is `self._rms_norm(hidden_states, self.weights.input_ln)`** — the first call in `_decode`, feeding
`_attention_decode`, i.e. the QKV projection and **the KV cache write**. Leaving that one site interleaved recovers
the whole 5.2 × 10⁻³; the other seven are numerically free.

That is the mechanism nailed shut: prefill filled the cache from an **interleaved** `input_ln`; decode computes the
new token's K/V from a **sharded** one; the new K/V is inconsistent with every cached entry. Nothing else in the
layer cares.

## 4. Both fixes measured, and there is no correctness/latency trade

`test_optimized_repeated_perf`, layer 0 sliding, batch 1, `decode_trace_host_ms`, 5 samples:

| config | decode PCC | bar | ms/layer | vs incumbent | µs/model (25 L) |
|---|---:|:--|---:|---:|---:|
| incumbent, interleaved | 0.9996280142258483 | ✅ | 1.835632 | — | — |
| **88 cores, phase = both, all 8 sites** | **0.9996293363224806** | ✅ | **1.610552** | **−12.27 %** | **−5,627** |
| **88 cores, decode-only, drop site 1** | **0.9996226684246500** | ✅ | **1.625231** | **−11.47 %** | **−5,260** |
| 11 cores, decode-only — *v3's vetoed best* | 0.9945729603715616 | ❌ | 1.597167 | −13.00 % | — |

**The best passing configuration is more accurate than the incumbent *and* 12.3 % faster.** `0.9996293` against the
incumbent's `0.9996280` — so the earlier framing of this as a *trade* ("spend 5 × 10⁻³ of PCC to buy 13 % of
latency") was wrong too: **nothing has to be spent.** The trade only appeared to exist because every configuration
v3 measured had the mismatch baked in.

**−5,627 µs/model recoverable on this cell**, against v2's booked −5,919 and v3's shipped 0. And the alternative fix
— drop site 1, keep the decode-only guard — is worth **−5,260 µs** and needs **no change to the prefill path at
all**, only a policy field.

## 5. ⚠ The carrier is the KV cache — and v2's guard only fixes it at the oracle's own prefill length

The mechanism above says **site 1's output is the only one that outlives the step**, via the cached K/V. That
predicts the mismatch cost should depend on how much cached history the decode step attends over. Swept the
oracle's prefill length (`seq_len`, otherwise hardcoded to 32):

| prefill seq | interleaved | 88c decode-only | 88c **both** *(v2's guard)* | 88c decode-only **`drop_index=1`** |
|---:|---:|---:|---:|---:|
| 4 | 0.9992558364256461 | 0.9991617114442755 | 0.9991991629585012 | — |
| 8 | 0.9991848566633527 | 0.9990645601269287 | 0.9991853489542124 | **0.9991015554868861** |
| **32** | 0.9996280142258483 | **0.9943716809625597** | **0.9996293363224806** | **0.9996226684246500** |
| 64 | 0.9937787693607536 | 0.9939853731224421 | **0.9939853731224421 — identical** | 0.9938753149590233 |

**Two findings, and the second overturns action 1 as I first wrote it.**

**(a) The mismatch cost grows steeply with prefill length** — 9.4 × 10⁻⁵ at seq 4, 1.2 × 10⁻⁴ at seq 8, and
**5.3 × 10⁻³ at seq 32.** A 4× length change for a 44× cost change, so it is not proportional to the number of
cached entries; it behaves like a threshold. *(My prediction was the opposite — that a shorter prefill would be
worse, since the newly-appended inconsistent entry holds a larger share of the attention. Wrong.
→ [`PITFALLS`](ADVCHAL-V3-ANALYST-PITFALLS.md) ERROR 18.)* The direction that did hold is consistent with the
**query** also coming from site 1: with more cached history, more attention mass sits on (Q, K) pairs computed by
different arithmetic paths.

**(b) v2's guard stops firing once prefill exceeds one tile row.** Its condition is `x.shape[-2] > TILE_SIZE →
skip`, so at seq 64 it does **not** shard prefill — and `phase=both` returns **exactly** the decode-only number,
`0.9939853731224421`, to sixteen digits. **v2's configuration is phase-consistent only for prefill ≤ 32 tokens,
which is precisely the length its oracle tests.** At any production prefill length it silently becomes the
mismatched configuration this file is about.

So **v2's passing 0.9996293 does not generalise** — not because the oracle was faked (it was not, see
[`REMEASURE`](ADVCHAL-V3-REMEASURE.md)), but because the oracle's single test point is the one length at which the
guard fires.

**`drop_index=1` is the fix that generalises.** It tracks the interleaved baseline at every length tested —
within 8.3 × 10⁻⁵ at seq 8, **5.3 × 10⁻⁶ at seq 32**, 9.0 × 10⁻⁵ at seq 64 — because it leaves site 1 interleaved
in decode, matching a prefill that is interleaved at *any* length. It needs no prefill change and no guard change,
only a policy field.

⚠ **The seq 64+ rows are not evidence about the model.** The interleaved baseline itself reads 0.9937788 at
seq 64 — below the 0.995 bar — and 0.5466 at seq 256. The oracle hardcodes `seq_len = 32` and its cache capacity,
page table and mask construction are all built around that, so longer lengths leave the regime the test supports.
Those rows are used here **only** for the code-path fact that `phase=both` collapses onto `phase=decode`, which is
independent of whether the model is in a supported regime. **Whether the model is genuinely correct at production
prefill lengths is untested by this oracle at all, and that is its own finding.**

## Revised actions

1. **Ship 88 cores with `drop_index=1`, decode-only.** Measured **−11.47 %/layer, PCC 0.9996227, −5,260 µs/model**,
   and it is the only configuration that holds at every prefill length tested. One policy field, no guard change,
   no prefill change.
2. ~~Ship 88 cores with a phase-consistent guard (v2's condition)~~ — **withdrawn.** It measures better at seq 32
   (−12.27 %, PCC 0.9996293) but **only fires when prefill ≤ 32 rows**, so it does not fix production. A guard that
   shards prefill at *all* lengths is a larger, untested change.
2b. **Extend the oracle to more than one prefill length.** Every conclusion in this file that needed correcting
   needed it because `seq_len` was pinned at 32 — including v2's original result and my first fix.
3. **Audit `full_attention`'s shipped config for the same latent mismatch.** It passes by a 7 × 10⁻⁶ margin that is
   accidental, not designed.
4. **The ladder must include the advised grid.** 88 was the advised value and the only passing rung, on both fixes.
5. **A phase-gated placement knob needs a cross-phase consistency assertion.** The KV cache carries the
   disagreement into the measurement, and nothing in the artefacts showed it. Generalise to phi and north-mini,
   which also gained decode-only knobs in v3.


---

# 6. The full causal chain, end to end — every step measured

Asked whether the decode grid's relation to prefill had been traced fully. **It had not**: §1–§5 established
*where* and *when*, and left the amplification unexplained — how a 1-ULP perturbation in one norm becomes
5.2 × 10⁻³ in the layer output, and why phase *agreement* cancels it rather than accuracy. That is now traced, and
it corrects a published claim on the way.

## 6a. ⚠ Correction: the grids are **not** identical on the real activation

[`NORM-GRID-SWEEP`](ADVCHAL-V3-NORM-GRID-SWEEP.md) reported every grid bit-identical. That was measured on
**synthetic** inputs. Dumping the **real** site-1 activation out of the live layer
(`(1,1,1,2816)`, absmax 3.281, std 0.972) and re-running with the real `input_layernorm.weight` (absmax 30.62):

| config | PCC vs float64 | vs interleaved |
|---|---:|---|
| interleaved | 0.999998582598 | — |
| 2c / 4c / 8c / 11c / 22c / 44c | **0.999998666122** | **514/2816 differ** — all identical *to each other* |
| **88c** | 0.999998626433 | **1022/2816 differ** — and differs from the group above |

So the grids *do* differ on real data, and 88 differs most. **But all of them are equally accurate** — within
1.4 × 10⁻⁶ of float64, and every sharded variant is very slightly *more* accurate than interleaved. **No grid is
wrong. They are 1 bf16 ULP apart from each other in 18–36 % of channels, and that is all.**
→ [`PITFALLS`](ADVCHAL-V3-ANALYST-PITFALLS.md) ERROR 15, third instance.

## 6b. The amplifier: a top-8-of-128 expert selection flips

Dumped `ttnn.topk`'s output during the oracle's decode step in three configurations:

| configuration | selected experts | |
|---|---|:--|
| interleaved (incumbent) | 47, 53, 124, 61, 122, 121, **104**, 50 | reference behaviour |
| **decode-only sharded** | 47, 53, 124, 61, 122, 50, 121, **123** | **expert 104 → 123** |
| both phases sharded | 47, 53, 124, 61, 122, 121, **104**, 50 | **identical to interleaved** |

And the two numbers that make it inevitable:

| | |
|---|---|
| logit gap between the **8th and 9th** expert | **0.015625** — one bf16 ULP at that magnitude |
| max \|Δlogit\| caused by the 1-ULP norm difference | **0.046875 — 3× the gap** |

**The perturbation is three times the margin that decides the selection.** So the flip is not bad luck; at this
gap it is the expected outcome.

## 6c. The chain, complete

1. Sharded and interleaved `rms_norm` differ by **1 bf16 ULP in 18–36 % of channels** on the real activation. Both
   are accurate to ~1.4 × 10⁻⁶ against float64 — **neither is wrong** (§6a).
2. Site 1 is `input_ln`, feeding QKV. **Prefill wrote the cached K/V from the interleaved path; decode computes the
   new Q/K/V from the sharded one** (§3).
3. Attention over that **hybrid** cache is perturbed, and the perturbation flows through the residual into the
   **router logits**: max \|Δlogit\| = **0.046875**.
4. The 8th-vs-9th expert gap is **0.015625**. The perturbation exceeds it.
5. **The selection flips — expert 104 out, 123 in** (§6b).
6. A different expert set is a different function, so the layer output moves by **5.2 × 10⁻³** — three orders of
   magnitude more than any arithmetic error involved.
7. Sharding **both** phases removes the hybrid: everything sits on one footing, the logits land back inside the
   gap, the selection returns to the reference's exact eight experts, and PCC returns to **0.9996293**.

**That is why *agreement* is what matters and accuracy is not.** Both paths are equally accurate; only their
*mixture* is fatal, because the mixture is what shifts a logit across a decision boundary.

## 6d. Four earlier observations that this explains

| observation | explanation |
|---|---|
| cost grows with prefill length: 9.4 × 10⁻⁵ @4, 1.2 × 10⁻⁴ @8, **5.3 × 10⁻³ @32** (§5) | more cached entries → more of the hybrid in the attention output → a larger logit perturbation. Below seq ~8 it stays under the 0.0156 gap and **no flip occurs**; at 32 it exceeds it |
| `full_attention` is **740×** less sensitive (§2) | no flip there — its logit margins are not crossed by the same perturbation |
| grid barely changes the layer number (0.99437 @88 vs 0.99457 @11) | different grids give different 1-ULP *patterns*, so slightly different post-flip values — the same flip either way |
| **north-mini's oracle cannot see it** ([`CORE-ISSUE`](ADVCHAL-V3-CORE-ISSUE.md)) | it decodes at position 0 against an **empty** cache. No cached entries → no hybrid → no logit shift → no flip. Structurally undetectable |

## 6e. What is still not established

- **That the flip is the *only* amplifier.** A flip is measured and its magnitude is sufficient, but the
  attention-path contribution has not been separated from the MoE-path contribution. Ablating the MoE (route to
  the interleaved expert set while keeping the sharded attention) would settle it.
- **Why the perturbation grows with prefill length rather than shrinking.** Measured, not derived; the sign was the
  opposite of my prediction (ERROR 18).
- **Whether `full_attention`'s margin is wide or its perturbation small.** Its logit gap was not dumped.


---

# 7. The two open items closed: path decomposition, and a dense control that needs a different explanation

## 7a. The routing flip is the amplifier, and the attention path contributes 0.6 %

§6e left open whether the flip was the *only* amplifier. Decomposed it by dumping the final `routing` vector and
pinning it, so each path can be varied alone:

| | decode PCC | 1 − PCC | added over baseline |
|---|---:|---:|---:|
| **A** interleaved (baseline) | 0.9996280142258483 | 3.720 × 10⁻⁴ | — |
| **B** 88c decode-only — full effect | 0.9943716809625597 | 5.628 × 10⁻³ | 5.256 × 10⁻³ |
| **C** 88c decode-only **+ interleaved routing pinned** | **0.9995947698880301** | 4.052 × 10⁻⁴ | **3.3 × 10⁻⁵ — 0.6 %** |
| **D** interleaved **+ sharded routing pinned** | **0.9943362923697102** | 5.664 × 10⁻³ | **5.292 × 10⁻³ — 100 %** |

**The router path alone reproduces the entire drop** — D is even marginally worse than the full effect B — and the
attention path contributes **0.6 %**. So the chain in §6c is complete and the flip is not merely *a* mechanism, it
is *the* mechanism. Freezing the routing to the incumbent's expert set recovers 99.4 % of the loss while leaving
every sharded placement in force.

**Which makes the fix menu wider than §5 suggested.** Anything that stops the selection moving works: phase
consistency, `drop_index=1`, or pinning the routing — and the last is a hint that the real defect may be a
**router that is not robust to 1 ULP**, since its 8th/9th logit gap is 0.015625 against a perturbation of 0.046875.

## 7b. A dense model with a bigger drop — so the flip cannot be the general cause

Checked the corpus's fully dense model. **phi-3.5-mini is `Phi3ForCausalLM`, no experts, no router, no `topk`** —
and it carries the corpus's largest PCC drop. Ran its own oracle
(`optimized_decoder_perf.py::test_profile_traced_decode`, batch 32, `PHI35_REAL_WEIGHTS=1`, weights cached
locally):

| tree / policy | PCC | mean_ms | |
|---|---:|---:|:--|
| v3, `final` (rope L1 off) | 0.9989930042363637 | 0.789394 | ✅ incumbent |
| **v3, `advisor_rope_l1` (as shipped-off)** | **0.9849538521359096** | *(failed before perf)* | ❌ |
| **v2, `advisor_rope_l1_chain` ON** | **0.9989930042363637** | **0.749102** | ✅ **−5.2 %, bit-identical PCC** |
| v2, same tree, chain OFF | 0.9989930042363637 | 0.790101 | ✅ |
| **v2's implementation ported into v3's tree** | **0.9989930042363637** | **0.749474** | ✅ **−5.1 %, recovers the win** |

**v2's rope L1 chain is a genuinely free win**: engaged (proved by the 0.749 vs 0.790 timing, not inferred),
**−5.2 %**, and **bit-identical PCC with the knob on and off**. v3's implementation of the same idea costs
**1.4 × 10⁻²** of PCC and fails.

So the dense case has a **third, distinct cause**: not a routing flip (no router), not the phase asymmetry (both
versions are decode-only here), but **v3's rope code**. And porting v2's 25 lines into v3's tree recovers the
entire win — **−1,284.9 µs/model, phi `nofuse-noadvise`, v3's second-largest loss.**

## 7c. ⚠ RETRACTED: the "two lines" explanation for phi

[`OP-BY-OP`](ADVCHAL-V3-OP-BY-OP-VS-V2.md) §2.3 attributed phi's 0.917 to two specific lines — the key returned in
the query's memory config, and interleaved rather than sharded arithmetic — and said *"the second is the
correctness suspect and is one line."* **Both were patched individually and both are no-ops: PCC stayed
0.9849538521359096 to sixteen digits.** The difference is in the *combination*, and the untested element was v3's
extra `value = ttnn.to_memory_config(value, ttnn.L1_MEMORY_CONFIG)` at the top of `apply_l1` together with
multiplying by **interleaved** `cos`/`sin`, where v2 keeps `value` in the query's 32-way height shard and reshards
`cos`/`sin` to match. Most likely a broadcast that resolves differently sharded versus interleaved — **not
isolated to a single line, and I am not claiming one.** What is established is that **v2's implementation is
correct and free and v3's is not**, and the remedy is a known-good replacement rather than a line edit.

Also noted: my re-run of v3's config gives **0.9849539** where the cell recorded **0.9173130**. Both fail, but they
are not the same number, so the cell's oracle differed from this one in some way its artefacts do not record —
which is the [`PCC-DROP-ISOLATION`](ADVCHAL-V3-PCC-DROP-ISOLATION.md) provenance gap again, on a second model.

## 7d. Recoverable total

| cell | fix | measured | µs/model |
|---|---|---|---:|
| gemma-4-26B `-onA` sliding | 88 cores + `drop_index=1` | PCC 0.9996227, −11.47 %/layer | **−5,260** |
| phi-3.5 `nofuse-noadvise` | port v2's rope L1 chain | PCC 0.9989930, −5.1 %/layer | **−1,285** |
| | | | **−6,545** |

Against v3's shipped −6,769 and its 8,408 µs shortfall to v2, **two known-good changes recover 78 % of the gap**,
and neither is a search or judgement improvement — both are defects in code v3's own cells wrote.


---

# 8. Four more tests from the open register

## 8a. qwen3.6 — the counter-example that fixes the rule

The one shipped win nobody had audited (**−1,130 µs, 17 % of v3's output**). It ships
`advisor_plan="mlp_product_only"`, a 109-core width-sharded MLP intermediate — and the knob sits inside
**`_mlp_decode`**, so it is **decode-only, exactly like the others.** Yet:

| kind | incumbent PCC | candidate PCC |
|---|---:|---:|
| `linear_attention` | 0.9981887732142846 | **0.9981887732142846** |
| `full_attention` | 0.9980950192619897 | **0.9980950192619897** |

**Bit-identical on both kinds, with a real −1.0 % (−1,130 µs).** Engaged — the timing moved.

**So decode-only gating is not the defect, and [`CORE-ISSUE`](ADVCHAL-V3-CORE-ISSUE.md)'s framing needs
sharpening.** The MLP's output goes into the residual for that token and is never persisted. The precise condition:

> **A decode-only knob is unsafe iff its output flows into the KV cache write.** Otherwise the phases cannot
> disagree about anything that outlives the step.

That rule classifies **every** cell correctly: gemma site 1 (`input_ln` → QKV) unsafe *and it broke*; gemma sites
2–8 (post-attn, MLP, router, post-ff) safe *and measured harmless*; north-mini `decode_norm_cores` (→ QKV) unsafe
by the rule; phi `input_norm_cores` (→ QKV) and phi's rope (→ K) unsafe by the rule; **qwen `mlp_product_only`
safe by the rule and measured bit-identical**; north-mini `decode_topk_cores` (routing, within-step) safe. It is a
one-line static question — *does this op's output reach the cache?* — and it is checkable before any measurement.

## 8b. `ladder_88`: no evidence it ever ran

The cell's `README` claims *"2/4/8/11/22/44/**88** were fresh processes"* and `build_evidence.py` writes
`legal_ladder: [1,2,4,8,11,22,44,88]`, while `measurements/` holds only 2/4/8/11/22/44. Searched **every commit on
the run and cell branches**: **zero `ladder_88` files at any point in history.** So the rung was not written and
later pruned — there is no artefact of it anywhere. Combined with the later measurement that 88 in v3's tree scores
**0.9943717 (fail)**, the README's claim is both unbacked and would have been consequential if true.

## 8c. Why `full_attention` is 740× less sensitive — the boundary is *inside* the set

Dumped its router output too. It is **not** that the perturbation is smaller:

| | `sliding_attention` | `full_attention` |
|---|---|---|
| logit gap, 8th vs 9th | 0.015625 | **0.015625 — the same** |
| max \|Δlogit\| | 0.046875 | **0.035156 — also exceeds the gap (2.25×)** |
| experts, interleaved | 47, 53, 124, 61, 122, 121, **104**, 50 | 36, 111, 112, 19, 115, 125, **9, 14** |
| experts, sharded | 47, 53, 124, 61, 122, 50, 121, **123** | 36, 111, 112, 19, 115, 125, **14, 9** |
| what changed | **membership — 104 out, 123 in** | **order only — 9 and 14 swapped** |

**Both kinds cross a boundary. On `full` the crossing permutes two experts that are *both already selected*, and
`scatter` places each weight at its own expert index, so the output is unchanged.** On `sliding` the crossing is at
the **8th/9th cut**, so the set changes and a different function runs.

**The 740× is therefore not a property of the kind — it is which pair of adjacent logits happens to sit at the
selection cut.** Which means `full_attention`'s shipped −1,198 µs passes by luck of input, not by structural
margin: a different token could put the crossing at the edge and produce the same 5 × 10⁻³. That strengthens the
case that a whole-layer PCC on this model has a discontinuous floor and cannot gate a placement change.

## 8d. phi's defect isolated — a coupled pair, and it cannot be split

Bisected from the working v2 port:

| variant | PCC | |
|---|---:|:--|
| v2 port, as v2 | **0.9989930042363637** | ✅ |
| v2 port **+ `value → L1_INTERLEAVED`** (v3's extra line) | **0.9989930042363637** | ✅ — harmless on its own |
| v2 port but **cos/sin + arithmetic in `L1_MEMORY_CONFIG`**, value left sharded | **`TT_FATAL` — does not run** | — |
| v3 as shipped (both together) | 0.9849538521359096 | ❌ |

**The two elements are inseparable**: interleaved arithmetic *requires* the value conversion to run at all, and the
value conversion alone is harmless. So the causal difference is a **design choice**, not a typo:

> **v3 does the RoPE multiply/add in an L1-*interleaved* layout; v2 does it in the query's own 32-way height
> shard, resharding `cos`/`sin` and `rotated` to match.** The sharded form is bit-identical to the incumbent and
> 5.2 % faster; the interleaved form costs 1.4 × 10⁻² of PCC.

And it explains why every single-line patch was a no-op: each was applied *inside* v3's interleaved structure,
and the structure is the cause. The remedy stays a replacement — **−1,285 µs, PCC 0.9989930.**
