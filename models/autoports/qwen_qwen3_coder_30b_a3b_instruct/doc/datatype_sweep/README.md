# Stage 07 — the datatype sweep

Find the fastest precision config that still clears the full-model accuracy
gate, select it, and show the evidence.

**Headline.** Twenty-six configs were measured end to end at 48 layers. The
winner is not a dtype at all: it is the pair of fields that live in the
precision config but are *scheduling* choices — the two expert matmul **inner
block widths**. Taking both to their full-K ceilings buys **+2.83% traced decode
(42.34 → 43.54 t/s/u) at bit-identical accuracy** — top-1 0.990, top-5 1.000,
top-100 1.000, unchanged.

**Exactly one *dtype* lever cleared every clause of the selection rule on its
own**: `R06_attn_bfp4`, both attention projections to `bfloat4_b`, at **+0.45%**
— beyond the 0.368% band — for one top-1 point (0.990 → 0.980, exactly on the
declared floor, so it clears it). Every other dtype and fidelity lever either
regressed, landed inside the band, or hit a TTNN blocker.

**But it does not compose, and that was measured rather than assumed.** `R06`
and the block widths are orthogonal — attention projection weights against
expert matmul scheduling — so the two were stacked and run as a full 48-layer
row, `R26_attn_bfp4_bw64_24`, and then repeated three times because the answer
turned out to be close. Stacked, the attention gain **disappears into the width
gain**:

| | traced decode | top-1 |
| --- | --- | --- |
| `R25` widths only, 3 repeats | 43.46 / 43.38 / 43.54, mean **43.46** | 0.990 ×3 |
| `R26` widths **+** attention bfp4, 3 repeats | 43.46 / 43.52 / 43.55, mean **43.51** | 0.980 ×3 |

The two ranges **overlap**; the means differ by 0.12%, a third of the measured
band. So the attention dtype buys nothing on top of the widths and costs a top-1
point every time. `R25` is selected.

So the answer to "which precision policy is fastest" is: **the dtype policy we
already ship**. One dtype lever is a real, if marginal, win against the
stage-06 baseline in isolation; none of them is a win against the config this
stage selects. The win came from co-tuning, which is exactly the interaction
stage 02 warned about, arriving from the opposite direction.

---

## 1. The gate, and the selection rule

**Formal accuracy gate (from the stage contract): `top-5 >= 98%` and
`top-100 = 100%`.** Top-1 is reported everywhere and is *not* gated.

That asymmetry is the whole shape of this stage. The shipped model sits at
top-5 = 1.000 and top-100 = 1.000 — **maximum margin on both gated metrics** —
while top-1 is 0.990. A rule that only reads the contract is therefore free to
spend unlimited top-1, and would.

### The rule actually used, stated before the results

> **Rank on traced teacher-forcing decode t/s/u among configs that clear the
> formal gate, and select the fastest one that also (a) stays within 1 top-1
> point of the shipped default and (b) beats the default by more than the
> measured run-to-run band. If nothing does, keep the default.**

**Clauses (a) and (b) are additional to the goal's formal gate and are a
deliberate judgment call.** They exist because the formal gate alone cannot
tell a good config from a bad one here: with top-5 and top-100 both pinned at
1.000, a rule that reads only the contract is free to spend unlimited top-1.
`R04_qkv_bfp4` is the demonstration — it spends top-1 0.990 → 0.960, three
tokens of first-token agreement, to buy **+0.33%**, and passes the gate
honestly. Stacking two or three rows like it would ship a materially worse
model for under one percent of throughput. Deviating is called out here rather
than applied quietly.

To be precise about what the clauses do and do not change. The literal rule —
"the fastest evaluated config that satisfies the gate" — does **not** select
`R04`: `R04` is twelfth fastest at 42.48 t/s/u and was never in contention for
selection. What the clauses stop is `R04` **outranking the default**: without
clause (a) it would be a legitimate improvement on the shipped policy, to be
stacked with the winner or shipped if the winner had not existed. With clause
(a) it is not. Clause (b) then disposes of it a second time — +0.33% is inside
the 0.368% band, so it was never a demonstrated win either.

The literal rule selects **`R26_attn_bfp4_bw64_24`**, the fastest measured row
at 43.58 t/s/u, which spends a top-1 point. Clause (c) below is what stops it,
and it stops it on the *band*, not on the top-1 floor — `R26` clears the floor.

### Clause (c), added after the results were in

> **Among the eligible, rows within one measured band of the fastest are
> tied, and a tie is broken on the simpler and safer config — the one moving
> fewest dtype/fidelity fields off the shipped default — then on top-1, then
> decode, then TTFT.**

This clause was written **after** seeing the row it decides, and saying so is
the point of putting it here. `R26` leads `R25` by 0.09% on the point estimate —
roughly a quarter of the 0.368% band. Ranking on that would be ranking on a
difference this document has already called unmeasurable twice over ("any win
below the band is the same number measured twice"). Applying the band when
comparing a candidate to the *default* but not when comparing two *candidates*
is the inconsistency; clause (c) is the removal of it, not a rescue of the
incumbent.

**What breaks the tie is the governing `datatype-sweep` skill's own rule**, not
a preference invented here:

> *"If two configs are within measurement noise, prefer the simpler and safer
> one."*

`R25` changes **no dtype and no fidelity at all** — only two expert matmul inner
block widths, a scheduling choice that is bit-identical by construction. `R26`
takes the attention QKV and W_O weights to `bfloat4_b`, four mantissa bits on
every attention projection in all 48 layers. Between two configs the measurement
cannot separate, `R25` is unambiguously the simpler and the safer one, and the
skill's tie rule selects it **regardless of what the accompanying top-1 point
does or does not mean**. `R28_kv_bfp8_bw64_24`, the other tied row, moves one
field (`kv_cache_dtype`) and loses the same way.

Top-1 is kept as the **secondary** ordering, and it agrees — but it is not what
the argument rests on, deliberately. Teacher forcing is deterministic per
config, so `R25`'s `0.990 ×3` against `R26`'s `0.980 ×3` is one token
re-observed three times, not three pieces of evidence; and §1(a) and limitation
3 both say a one-point top-1 difference on a 100-token reference is not signal
(`R05_wo_bfp4` scores **1.000**, *above* the baseline, which is the proof).
Breaking the tie on top-1 alone would have contradicted them.

It was also checked rather than argued: `R26` was repeated three times, the same
treatment `R00` and `R25` got, and its range **overlaps** `R25`'s (43.46–43.55
against 43.38–43.54) with means 0.12% apart. Clause (c) changes no *eligibility*
verdict — only the ordering among rows already eligible.

**(a) the top-1 floor — one point.** The reference is 100 tokens, so one point
is one token; a floor of 0.980 is the tightest meaningful bound this reference
can express. Top-1 also cannot support finer ranking than that: in this sweep
`R05_wo_bfp4` scored top-1 **1.000**, *above* the 0.990 baseline, and so did
`R13` and `R14`. A reduced-precision config beating the baseline is a direct
demonstration that ±0.01 here is noise, not signal.

**(b) the noise band — measured, not assumed.** `probes/repeats.py` re-ran three
identical configs three times each, fresh process every time:

| config | samples (t/s/u) | spread | stdev | mean | top-1 |
| --- | --- | --- | --- | --- | --- |
| `R00_default` | 42.34, 42.41, 42.34 | 0.165% | 0.040 | 42.36 | 0.990 ×3 |
| `R25_gateup64_down24` | 43.46, 43.38, 43.54 | **0.368%** | 0.080 | 43.46 | 0.990 ×3 |
| `R26_attn_bfp4_bw64_24` | 43.46, 43.52, 43.55 | 0.207% | 0.046 | 43.51 | 0.980 ×3 |

**The band is 0.368%**, taken as the widest of the three. Any "win" below that
is the same number measured twice. Nine rows are rejected on exactly this basis
— including `R04`'s +0.33%, which is *inside* the band and so was never a
demonstrated win in the first place, quite apart from its top-1 cost.

`R00` and `R25`'s ranges are **disjoint** — [42.34, 42.41] vs [43.38, 43.54] —
so the selected config's advantage over the default does not depend on which
samples you pick. `R25` and `R26`'s ranges **overlap** — [43.38, 43.54] vs
[43.46, 43.55] — which is clause (c)'s whole justification: between those two
there is no advantage to pick samples for.

---

## 2. Measurement regime

**Ranking metric: traced teacher-forcing decode t/s/u.** Every row is one
invocation of `models.common.readiness_check.run_teacher_forcing`, whose runner
*requires* `enable_trace=True` of the `generate()` it accepts and always passes
it. **No eager or untraced decode number is produced anywhere in this sweep**,
so there is nothing that could leak into the ranking — the safest reading of
"eager or untraced decode numbers are not valid for Pareto ranking or final
selection". Stage 06 is why this matters: an eager sampler delta of −0.224 ms
understated the traced −0.619 ms by 3x.

One row = one **subprocess**, with `QWEN3_PRECISION_CONFIG` pointing at that
row's JSON. Per-process isolation is required, not stylistic: `ccl_dtype` keys
the persistent CCL buffer cache, so a process visiting two CCL dtypes allocates
both and measures the second on a differently loaded device.

* Hardware: 1x4 Blackhole P300_X2, `FABRIC_1D_RING`, 4 dies
* Workload: AIME24 chat reference, 158 prompt tokens, 100 generated, batch 1
* Layers: **48 for every Pareto row.** The 2-layer tier is used only for
  structural checks and produces no ranked number.

### Two tiers, and why the cheap one earned its place

**Tier A (`probes/structural_probe.py`, 2 layers, ~10 s each)** builds every
candidate and reads back `fallback_audit`. It found **three candidates that
cannot be built at all** — which at 48 layers would have been three wasted
3-minute runs — and confirmed that **no requested `in0_block_w` was silently
clamped**, which is the failure mode that would have mislabelled the winning
rows.

**Tier B (`probes/sweep_runner.py`, 48 layers, ~3 min each)** is every number
quoted below.

---

## 3. The candidate set, and why each row is there

Decode at batch 1 is bound on weight bytes pulled per token, so the set is
ordered by how many bytes each group moves. Full rationale per row lives in
`probes/candidates.py`.

| group | rows | why |
| --- | --- | --- |
| lm_head | R01–R03 | 2048×37984 per die, read in full every token — the biggest non-expert read |
| attention | R04–R08 | qkv and wo priced **separately**; wo feeds the all-reduce |
| experts | R09–R15 | already bfp4, so these are block-width rows plus reverse-direction bfp8 |
| CCL / activations | R16–R18 | halve the wire payload twice per layer × 48 |
| KV cache | R19 | the only row that could move the context contract |
| embedding / norm / terminal | R20–R22 | small, measured rather than assumed |
| stacked | R23–R25 | the block-width winners combined, and gate_up at full-K |
| stacked | R26–R28 | the *orthogonal* stacks: the one eligible dtype row on top of the widths, its LoFi pair, and bfp8 KV on top of the widths |

**Every row is a delta from the stage-06 policy, pinned as
`candidates.BASELINE_PRECISION` rather than read off `DEFAULT_PRECISION`.** That
distinction became load-bearing the moment this stage moved the default: a
candidate set written against "the default" silently redefines every row when a
selection lands, so re-running tier A after the fact would have reported
`R00_default` at the *selected* widths, contradicting the 48-layer row every gain
here is quoted against. `candidates._assert_baseline_is_stage06` cross-checks the
pinned baseline against `configs/R00_default.json`, which was written at sweep
time and is a byte-level record of what `R00` actually ran.

### The orthogonal stack — measured, not inferred

§5 argues that gains do not add and that a stacked row has to be *run*. That
argument was originally applied only to the two width rows, which is half an
argument: `R06_attn_bfp4` was the one dtype row that cleared every clause, and
`R06` and `R25` touch disjoint ops. So `R26` composes them and `R27` is its
required LoFi pair. `R28` does the same for bfp8 KV, which is the row the
context contract needs priced against the config we ship rather than against one
we do not.

### The BFP4 + LoFi obligation

Required: for every material BFP4 matmul group considered or selected, a
BFP4+LoFi candidate for that group, or an exact blocker with evidence.

| group | BFP4 row | **BFP4+LoFi pair** | control |
| --- | --- | --- | --- |
| experts | shipped default | **shipped default is bfp4 + LoFi** | — |
| lm_head | `R01` 42.39 | **`R02` 42.18** | `R03` (LoFi at bfp8) 42.41 |
| attention | `R06` 42.53 | **`R07` 42.43** | `R08` (LoFi only) 42.32 |
| attention, at the selected widths | `R26` 43.58 | **`R27` 43.15** | `R08` (LoFi only) 42.32 |

The last row exists because `R26` is the row the literal selection rule picks
(§1), so the obligation applies at *its* widths and not only at the stage-06
ones. `R27` is 0.99% slower than `R26`, beyond the band — the same direction as
the other two pairs, and larger.

**Stage 02's "LoFi is free on bfloat4_b" did not reproduce as a *speed* win
here.** On all three pairs the BFP4+LoFi row was *slower* than BFP4 at the
shipped fidelity (lm_head 42.18 vs 42.39; attention 42.43 vs 42.53; attention at
the selected widths 43.15 vs 43.58), inside or near the band for the first two
and beyond it for the third. LoFi is free in the sense stage 02 meant — accuracy is unchanged — but
it buys nothing, because these matmuls are bandwidth-bound rather than
fidelity-bound at batch 1.

`R08` exists because **`attention_fidelity` has no measured default**: it is
`None`, meaning "the op picks". `R08` sets LoFi with no dtype change and lands
at 42.32 vs the 42.34 baseline — so the op default is already at least as good
as LoFi, and any delta in `R07` belongs to the dtype. Without `R08`, `R07` would
have been uninterpretable.

### Block widths co-tuned wherever a dtype changed

Stage 02 proved sweeping dtype or width alone finds the wrong optimum (expert
packing looked like 1.66x against an untuned baseline and was 1.09x against a
tuned one). So `R13`/`R14`/`R15` carry the widths *their* dtype wants, not
bfloat4_b's.

**The two ceilings are not symmetric, and that is arithmetic.**
`_tuned_sparse_matmul_config` requires `in0_block_w` to divide K in tiles:

* gate_up: K = `hidden_size` = 2048 = **64 tiles** → legal 1, 2, 4, 8, 16, 32, 64.
  The shipped 16 is a quarter of full-K, so there were two rungs above it.
* down: K = `moe_intermediate_size` = 768 = **24 tiles** → legal 1, 2, 3, 4, 6,
  8, 12, 24. **`R11_down_bw24` is already down's full-K row** — there is no
  wider width and no `down` analogue of the bw64 row.

---

## 4. Results

Ranked by the ranking metric. `Δtop-1` is cost relative to the default
(positive = worse). Full data in `sweep_results.json` / `.csv`.

| config | delta from default | top-1 | Δtop-1 | top-5 | top-100 | t/s/u | gain | TTFT ms | gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `R26_attn_bfp4_bw64_24` | qkv+wo bfp4, bw 64/24 | 0.980 | +0.010 | 1.000 | 1.000 | 43.58 | +2.93% | 3337 | yes |
| **`R25_gateup64_down24`** ← selected | gate_up bw 64, down bw 24 | 0.990 | +0.000 | 1.000 | 1.000 | **43.54** | **+2.83%** | 3288 | yes |
| `R28_kv_bfp8_bw64_24` | kv bfp8, bw 64/24 | 0.980 | +0.010 | 1.000 | 1.000 | 43.45 | +2.62% | 7792 | yes |
| `R24_gateup32_down24` | gate_up bw 32, down bw 24 | 0.990 | +0.000 | 1.000 | 1.000 | 43.31 | +2.29% | 3387 | yes |
| `R23_gateup_bw64` | gate_up bw 64 | 0.990 | +0.000 | 1.000 | 1.000 | 43.23 | +2.10% | 3289 | yes |
| `R27_attn_bfp4_lofi_bw64_24` | qkv+wo bfp4 + LoFi, bw 64/24 | 0.980 | +0.010 | 1.000 | 1.000 | 43.15 | +1.91% | 3256 | yes |
| `R11_down_bw24` | down bw 24 | 0.990 | +0.000 | 1.000 | 1.000 | 42.99 | +1.53% | 3479 | yes |
| `R09_gateup_bw32` | gate_up bw 32 | 0.990 | +0.000 | 1.000 | 1.000 | 42.94 | +1.42% | 3542 | yes |
| `R15_down_bfp8_only` | down bfp8 + bw 24 | 0.990 | +0.000 | 1.000 | 1.000 | 42.62 | +0.66% | 3577 | yes |
| `R06_attn_bfp4` | qkv+wo bfp4 | 0.980 | +0.010 | 1.000 | 1.000 | 42.53 | +0.45% | 3373 | yes |
| `R04_qkv_bfp4` | qkv bfp4 | 0.960 | **+0.030** | 1.000 | 1.000 | 42.48 | +0.33% | 3871 | yes |
| `R21_norm_hifi2` | norm HiFi2 | 0.990 | +0.000 | 1.000 | 1.000 | 42.44 | +0.24% | 3311 | yes |
| `R05_wo_bfp4` | wo bfp4 | 1.000 | −0.010 | 1.000 | 1.000 | 42.43 | +0.21% | 3705 | yes |
| `R07_attn_bfp4_lofi` | qkv+wo bfp4 + LoFi | 0.980 | +0.010 | 1.000 | 1.000 | 42.43 | +0.21% | 4159 | yes |
| `R03_lmhead_lofi` | lm_head LoFi | 0.990 | +0.000 | 1.000 | 1.000 | 42.41 | +0.17% | 3359 | yes |
| `R01_lmhead_bfp4` | lm_head bfp4 | 0.990 | +0.000 | 1.000 | 1.000 | 42.39 | +0.12% | 3472 | yes |
| `R13_experts_bfp8_cotuned` | experts bfp8 + bw 32/24 | 1.000 | −0.010 | 1.000 | 1.000 | 42.38 | +0.09% | 3361 | yes |
| `R00_default` | — (baseline) | 0.990 | +0.000 | 1.000 | 1.000 | 42.34 | — | 3251 | yes |
| `R08_attn_lofi` | attention LoFi | 0.990 | +0.000 | 1.000 | 1.000 | 42.32 | −0.05% | 4019 | yes |
| `R19_kv_bfp8` | kv bfp8 | 0.980 | +0.010 | 1.000 | 1.000 | 42.29 | −0.12% | 7843 | yes |
| `R22_logits_sampling_bfp8` | logits+sampling bfp8 | 0.990 | +0.000 | 1.000 | 1.000 | 42.21 | −0.31% | 3369 | yes |
| `R02_lmhead_bfp4_lofi` | lm_head bfp4 + LoFi | 0.990 | +0.000 | 1.000 | 1.000 | 42.18 | −0.38% | 3475 | yes |
| `R14_gateup_bfp8_only` | gate_up bfp8 + bw 32 | 1.000 | −0.010 | 1.000 | 1.000 | 41.80 | −1.27% | 3306 | yes |
| `R12_down_bw6` | down bw 6 | 0.990 | +0.000 | 1.000 | 1.000 | 41.67 | −1.58% | 3419 | yes |
| `R16_ccl_bfp8` | ccl bfp8 | 0.980 | +0.010 | 1.000 | 1.000 | 41.47 | −2.06% | 5731 | yes |
| `R10_gateup_bw8` | gate_up bw 8 | 0.990 | +0.000 | 1.000 | 1.000 | 41.33 | −2.38% | 3519 | yes |

### Not evaluated — TTNN blockers

| config | blocker |
| --- | --- |
| `R17_act_bfp8`, `R18_act_ccl_bfp8` | `nlp_create_qkv_heads_decode_device_operation.cpp:41` — *"Unsupported data format"*, asserting `input_tensor.dtype() == FLOAT32 \|\| input_tensor.dtype() == BFLOAT16`. With `activation_dtype=bfloat8_b` the fused QKV projection output is bfp8, and the decode head-split op takes only FP32 or BF16. |
| `R20_embed_bfp8` | `py_to_tt_tensor.cpp:399` — *"Layout must be Layout::TILE for bfloat8_b or bfloat4_b!"*. The embedding table is ROW_MAJOR for the gather; block-float requires TILE. |

`R19_kv_bfp8` used to be listed here as "our integration defect, not a dtype
rejection". It no longer is: the defect is fixed and the row is measured — §6.

Both blockers above are the **current** ones. `R17`/`R18` were originally blocked
at `paged_fill_cache_device_operation.cpp:36`; `tt/functional_decoder.match_cache_dtype`
(§6) cleared that barrier, the tier-A probe was re-run on the fixed tree
(`logs/structural_probe.log`, `2026-08-16T09:46:59`), and the model now gets
further before failing somewhere else. `probes/check_published_figures.py`
re-derives this table from `probes/structural_probe.json` on every run, so a
blocker that moves again cannot be published stale.

---

## 5. Pareto interpretation

Charts: **`top1_perf_pareto.png`** and **`top5_perf_pareto.png`**. Both plot all
26 evaluated configs, draw the frontier, mark the selected point in **red**, and
show a vertical dotted line at **that axis's own threshold** — the formal gate
(top-5 ≥ 0.98) on the top-5 chart, the selection rule's top-1 floor (0.980) on
the top-1 chart. The two thresholds are numerically equal, which is why the
top-1 line was originally drawn at the gate value and captioned as
non-binding; it was right by coincidence, and the floor *is* enforced. Both
charts also shade the **measured run-to-run band** around the default, so a
reader can see directly which apparent gains are inside the measurement's own
resolution.

**The two charts tell opposite stories, and that is the finding.**

* **top-5**: every single evaluated config sits at **1.000** — one vertical
  column. Nothing this sweep tried moved the gated metric *at all*, which means
  the gate was never the binding constraint. All that separates the candidates
  is speed. The frontier here is therefore a **single point** — only the fastest
  row is non-dominated — and it is drawn as a ringed marker with its own legend
  entry. It used to be omitted: the plotting code drew the frontier as a line
  and skipped it when it had fewer than two points, so the chart read as "no
  frontier" rather than as the finding.
* **top-1**: real spread, 0.960 → 1.000, and a genuine frontier with three
  non-dominated points — `R26` (fastest, top-1 0.980), `R25` (selected, top-1
  0.990) and `R05_wo_bfp4` (top-1 1.000 but only +0.21%, inside the band). This
  is where any accuracy cost shows up, and it is why top-1 is a first-class
  column rather than a footnote.

The practical reading: **the gate has so much margin that it does not
discriminate**, so ranking on it alone would have been ranking on a constant.
The top-1 axis is what stops "fastest passing config" from quietly degrading the
model, which is why the floor exists.

### Why the winner wins

`R25` changes **no dtype and no fidelity** — only the two expert matmul inner
block widths, from 16/12 to their full-K ceilings 64/24. A block width is a
*scheduling* choice, not a numerical one, so accuracy is bit-identical by
construction and the measurement confirms it (0.990/1.000/1.000, and top-1
stable at 0.990 across all three repeats).

Both brackets came back monotonic to the ceiling:

```
gate_up (K = 2048 = 64 tiles):   8 → 41.33   16 → 42.34   32 → 42.94   64 → 43.23
down    (K =  768 = 24 tiles):   6 → 41.67   12 → 42.34   24 → 42.99  (24 is full-K)
```

The shipped 16/12 were inherited from **single-chip stage-02 tuning**. Expert
parallelism has since cut per-die N four-fold, which changed which blocking the
matmul wants — so the old comment that "16 wins at LoFi" was true when written
and stale by stage 06. This is the sweep's most useful structural finding.

**The gains do not add, and measuring the stacked row was necessary:**

| | gain vs default |
| --- | --- |
| gate_up 64 alone (`R23`) | +0.89 t/s/u |
| down 24 alone (`R11`) | +0.65 t/s/u |
| naive sum | +1.54 → would predict 43.88 |
| **measured together (`R25`)** | **+1.20 → actual 43.54** |

Inferring the combination from the singles would have overstated it by 0.34
t/s/u. Stage 02's lesson, reproduced.

The 32 → 64 step (+0.29) is smaller than 16 → 32 (+0.60), which is the expected
shape: at full-K there is no inner-dimension blocking left to remove, so the
curve flattens rather than jumping. A large gain at full-K would have been more
suspicious than a small one. The width was verified **resolved, not requested**
— `fallback_audit` reports `[64, 24]` for exactly the `[64, 24]` asked for, so
this is a genuine full-K measurement and not a clamp.

### Rejected, with numbers

| config | why rejected |
| --- | --- |
| `R04_qkv_bfp4` | clears the formal gate but costs **0.030 top-1** for +0.33% — and that +0.33% is *inside* the 0.368% band, so it is not even a demonstrated win |
| `R10_gateup_bw8` | −2.38%, a real regression beyond the band |
| `R16_ccl_bfp8` | −2.06% **and** TTFT 5731 vs 3251 ms — the cast in/out of the collective costs far more than the halved wire payload buys |
| `R12_down_bw6` | −1.58%, real regression |
| `R14_gateup_bfp8_only` | −1.27%, real regression; doubles gate_up bytes per die |
| `R02_lmhead_bfp4_lofi` | −0.38%, marginally beyond the band |
| `R01`, `R03`, `R05`, `R07`, `R08`, `R13`, `R19`, `R21`, `R22` | inside the ±0.368% band — indistinguishable from the default. **Nine rows**, matching §1 |

### Eligible, and not selected

Three rows clear every eligibility clause and are still not the selection.
`selection_reasons.json` lists them under `eligible`, with their own reasons —
they are *not* in the band-rejection row above, which is where `R06` wrongly sat
before this was corrected.

| config | t/s/u | top-1 | why not selected |
| --- | --- | --- | --- |
| `R06_attn_bfp4` | 42.53 | 0.980 | +0.45% beats the band, and it is **the one dtype lever that does** — but it is 2.3% slower than `R25`, and stacked with `R25` (as `R26`) it adds nothing |
| `R26_attn_bfp4_bw64_24` | 43.58 | 0.980 | the fastest row measured, by 0.09% over `R25` — inside the band, ranges overlap across three repeats each. Clause (c): tied on speed, and `R25` is the simpler and safer of the two (`R26` moves attention QKV and W_O to bfp4; `R25` moves no dtype at all). Top-1 agrees |
| `R28_kv_bfp8_bw64_24` | 43.45 | 0.980 | 0.21% behind `R25`, inside the band, and moves `kv_cache_dtype` where `R25` moves nothing. Its real subject is capacity, not speed — §6 |

`R09`, `R11`, `R15`, `R23`, `R24` and `R27` are also eligible and are simply
slower than `R25` by more than the band; they are rungs on the way to it rather
than rivals to it.

`R13_experts_bfp8_cotuned` deserves its own line: this is the row that **prices
the shipped bfp4 expert choice**, co-tuned at 32/24 so it is a fair comparison.
It lands at 42.38 (+0.09%, inside the band) with top-1 1.000, while doubling
per-die expert bytes. Against bfp4 at those same widths (`R24`, 43.31) it is
**0.93 t/s/u slower for one token of top-1**. The shipped bfp4 expert weights
are confirmed correct.

---

## 6. bfp8 KV — a silent-corruption bug, found, fixed and then measured

This started as the worst kind of result and ended as a real number. It is
written out in full because the failure mode — *a documented field of the
precision config this stage introduced, accepting a legal value that silently
fills the KV cache with NaN* — is the most serious thing the sweep found.

### What the first run showed

`R19_kv_bfp8` measured top-1 / top-5 / top-100 all at **0.010** — chance — with
decode 28.86 t/s/u and TTFT 8108 ms. **That is not what precision loss looks
like:**

1. Losing mantissa bits degrades ranking *gracefully*. `bfloat4_b` expert
   weights are far more aggressive and hold top-5 at 1.000 in this same sweep. A
   merely-imprecise cache does not put the correct token outside the **top 100**
   in 99% of positions.
2. A pure dtype reduction should be **faster** — fewer bytes. Slower *and*
   broken indicates a different code path.
3. `bfloat8_b` KV is used widely on Tenstorrent (`tt_transformers` relies on it
   for capacity). A model-specific collapse points at our integration.

### The op-level diagnosis, and the surprise in it

`probes/kv_bfp8_diagnosis.py` allocates a paged cache, writes a known tensor
through each writer and reads it back — no model, no decode loop, seconds rather
than minutes. **The two cache writers turn out to have opposite contracts**,
which is the whole answer and is why the fix is asymmetric:

| op | cache dtype | input dtype | round-trip |
| --- | --- | --- | --- |
| `paged_fill_cache` (prefill) | bfloat16 | bfloat16 | **PCC 1.0** (control) |
| `paged_fill_cache` | **bfloat8_b** | **bfloat16** | **NaN** ← what `R19` ran |
| `paged_fill_cache` | bfloat8_b | bfloat8_b | **PCC 1.0** |
| `paged_update_cache` (decode) | bfloat16 | bfloat16 | **PCC 1.0** (control) |
| `paged_update_cache` | **bfloat8_b** | **bfloat16** | **PCC 0.999969** |
| `paged_update_cache` | bfloat8_b | bfloat8_b | **rejected by the op** |

`paged_fill_cache` wants the input cast **to** the cache dtype: its input
validation is a permissive `OR` (`input==FP32 || input==BF16 || cache==BF8 ||
cache==BF4`) that a mismatch satisfies, and it then writes NaN.
`paged_update_cache` wants the input left **as bfloat16**: it converts into the
cache itself and hard-rejects a block-float update at
`paged_update_cache_device_operation.cpp:296` — *"Data type of input tensor for
update cache must be FLOAT32 or BFLOAT16"*.

So the model's `bfloat16` K/V were correct at the decode writer and catastrophic
at the prefill writer, the cache was NaN from the first prefill write, and that
is exactly why top-100 sat at chance from token zero. **A symmetric "cast
everywhere" fix would have traded silent corruption for a hard crash** — the
review that asked for this fix suggested casting at all six sites, and the
measurement above is why only four of them are cast.

### The fix

`tt/functional_decoder.match_cache_dtype` takes the dtype off the **allocated
cache tensor** rather than off `precision.kv_cache_dtype`, so the thing the write
must agree with is the thing it is compared against and the two cannot drift. It
is applied at the two `paged_fill_cache` sites and the two contiguous
`fill_cache` sites, and deliberately *not* at the two `paged_update_cache` sites,
which carry a comment saying why. **It is a no-op in the shipped configuration**
(`kv_cache_dtype == activation_dtype == bfloat16`), so the selected model's graph
is unchanged.

The alternative the review offered — reject the mismatched combination in
`PrecisionConfig.__post_init__` — was not taken, because it would have closed the
hole by making a working configuration illegal. The cast makes it work.

### What bfp8 KV actually scores

Two rows, both post-fix, both full 48-layer teacher forcing in the same regime as
every other row:

| row | delta from | top-1 | top-5 | top-100 | t/s/u | vs its comparator | TTFT ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `R19_kv_bfp8` | stage-06 baseline | 0.980 | 1.000 | 1.000 | 42.29 | −0.12% vs `R00` 42.34 | 7842.6 |
| `R28_kv_bfp8_bw64_24` | the selected widths | 0.980 | 1.000 | 1.000 | 43.45 | −0.21% vs `R25` 43.54 | 7792.1 |

**bfp8 KV clears the accuracy gate and is decode-neutral.** Both deltas are
inside the 0.368% band; the accuracy cost is one top-1 point and nothing at all
on either gated metric. It is rejected for the ordinary reason — it does not
make the model faster — and not for the dramatic one.

**The one thing it does cost is TTFT: 2.4x, 7.8 s against 3.3 s** on the
158-token gate prompt, in both rows. That is not the cast: the *pre*-fix run had
the same 8.1 s TTFT while the cache was full of NaN, so the cost predates the fix
and is unattributed. It is the largest unexplained number in this stage and is
recorded as limitation 9 rather than guessed at.

What bfp8 KV *does* buy is capacity, and that is now priced — §7.

---

## 7. Context contract

**No capability is reduced.** Advertised and supported context both remain
**262144**, and every KV figure describing the *shipped* model is unchanged —
the selected config keeps `kv_cache_dtype = bfloat16`.

The selected change is two matmul block widths. It moves no allocation: block
width is a program-config field, not a tensor dtype or shape, and
`device_expert_bytes_per_die` is identical (84 934 656 B) before and after. That
is recorded in `doc/context_contract.json` as `stage07_note`.

### The bfp8-KV candidate, priced

`doc/context_contract.json` gained one new block, `stage07_kv_bfp8_candidate`.
KV is **6.443 GB of the 11.759 GB/die** the running 48-layer model holds at full
context, so this is the model's single largest capacity lever, and now that §6's
defect is fixed it can be quoted rather than guessed:

| | bfloat16 (shipped) | bfloat8_b (candidate) |
| --- | --- | --- |
| B/token/layer/die | 512 | **272** |
| B/token/layer (all 4 dies) | 2048 | **1088** |
| KV at 262144 tokens, per die | 6 442 450 944 B (6.443 GB) | **3 422 552 064 B (3.423 GB)** |
| headroom released per die | — | **3.020 GB** |

`bfloat8_b` is 1.0625 B/elem once each 16-element block's shared exponent byte is
counted, which is why the saving is 47% rather than 50%.

**It is not taken, and taking it would not have been a capability question
anyway** — the lever moves capacity *upward*, so it could never have forced a
reduction, and the advertised 262144 is already met at bfloat16 with 22.119
GB/die free. It is priced in the contract so that a future stage wanting a larger
batch or a longer served context finds the number already measured, together with
the TTFT cost it would have to explain first (§6).

### Non-aligned prompt lengths

Preserved. No KV-cache or trace-buffer dtype or layout changed — the selection
touches only `in0_block_w` inside the expert matmul program configs, which does
not participate in chunking, padding or the page table. `moe_prefill_optimized`'s
pad-to-chunk and slice-back are untouched. Re-validated anyway by the full suite
below, which includes `test_full_model.py::test_non_aligned_prompt_lengths`
(1, 31, 33, 100, 127, 128, 129, 257, 1000) and the non-aligned prefill gates in
`test_multichip_decoder.py`.

---

## 8. Proof the selected config is consumed **by default**

The goal requires the selection to be consumed by default by the construction
path the measurements use, and says a JSON field ignored by hard-coded model
code does not satisfy it.

`DEFAULT_PRECISION` in `tt/precision.py` **now carries the selected values**, so
`default_precision_config.json` and `selected_precision_config.json` are
byte-identical — the default *is* the selection.

`probes/selection_proof.py` proves it on device: it clears
`QWEN3_PRECISION_CONFIG` from the environment, builds the real 48-layer model
via `build_generator(model_dir, mesh)` with **no precision argument**, and
generates four tokens through the traced decode path, and compares
`selected_precision_config.json` against **device readback** — `fallback_audit`'s
dtypes read off uploaded tensors, the block widths `_tuned_sparse_matmul_config`
actually resolved to, the fidelities the compute-kernel-config objects carry, and
the dtypes the terminal path *produced*. Nothing reads `model.precision`, which
would only prove the dataclass round-trips. Exits non-zero on any mismatch.
Result in `selection_proof.json`.

**All 21 checks pass** (16 before the stage-07 review), including
`experts_gate_up_in0_block_w = 64` and `experts_down_in0_block_w = 24` as
*resolved* widths — the two fields this stage moved — with
`QWEN3_PRECISION_CONFIG` confirmed `None`.

### The four fields the proof could not see, and the bug that hid behind them

`logits_dtype`, `sampling_dtype`, `lm_head_fidelity` and `norm_fidelity` had **no
audit entry at all** before this review. The consequence was not cosmetic: it
made `R03_lmhead_lofi`, `R21_norm_hifi2` and `R22_logits_sampling_bfp8` produce
`device_audit` blocks *byte-identical* to `R00`'s, so **"this lever does nothing"
and "this lever is not wired up" were indistinguishable**.

Adding them found that for `norm_fidelity` it was the second.
`decode_residual_norm` built its compute config from the **module default** and
never saw `self.precision`, so the field was a documented knob with no effect and
`R21_norm_hifi2`'s original +0.35% was the baseline measured twice. The
threading is fixed (`multichip_decoder.decode_residual_norm` now takes
`precision`; the audit reports `norm_math_fidelity`, which reads `HiFi2` for
`R21` and `HiFi4` everywhere else) and **`R21` was re-measured for the first
time**: 42.44 t/s/u, +0.24%, still inside the band, but now inside it for a
reason. `norm_fidelity` reaches only the *decode* residual norms; the prefill
norms pass no compute config and still take the op default.

`ccl_dtype` is checked as a **resolved config value, not a readback** — the
collectives keep no dtype-tagged tensor to read back — and is labelled that way
in the output rather than counted as proof.

One honest wrinkle from the original stage, closed rather than papered over: the audit's
`kv_cache_dtype` reads `model.kv_cache`, but the generator keeps its cache in
`Qwen3CoderGenerator._kv_cache`, so through this path the audit reports
`kv_cache_dtype_source == "config_not_yet_allocated"` and falls back to the
configured value. That is disclosed rather than disguised — and the proof
additionally inspects the **allocated cache tensor directly**
(`kv_cache_dtype (allocated tensor)`), so the KV dtype is device-verified too.

---

## 9. Performance

All post-selection numbers below come through the **normal default construction
path** — no precision argument, `QWEN3_PRECISION_CONFIG` unset.

### Ranking metric — traced teacher-forcing decode

| | before (stage 06 / `R00`) | after (selected) | change |
| --- | --- | --- | --- |
| traced teacher-forcing decode | 42.34 t/s/u | **43.48 t/s/u** | **+2.69%** |
| top-1 / top-5 / top-100 (decode) | 0.990 / 1.000 / 1.000 | **0.990 / 1.000 / 1.000** | unchanged |
| top-1 / top-5 / top-100 (prefill) | 0.980 / 1.000 / 1.000 | **0.980 / 1.000 / 1.000** | unchanged |

The final confirmation run measured 43.48 — slightly below the 43.54 the sweep
row recorded, and inside the 0.368% band of it. (This is a re-run: the original
stage-07 confirmation measured 43.63, and every `tt/` change made during the
review is a no-op in the shipped configuration, so the 43.48/43.63 spread is the
band and not a regression. Both are re-measured here because `tt/` changed at
all.)

### Post-selection token-out — recorded separately from teacher forcing

The stage-06 warmed no-readback benchmark, re-run through the selected-config
path (`probes/perf_full_model.py`, prompt 128 / generate 128 / batch 1,
`perf_full_model_selected.json`). **This is a different measurement from the
teacher-forcing figure above and is not mixed with it.**

| metric | before | after | change |
| --- | --- | --- | --- |
| **`token_out`** (model + sampling trace, on-device feedback, no readback) | 19.686 ms / 50.797 t/s/u | **19.213 ms / 52.049 t/s/u** | **+2.47%** |
| `model_trace` (logits only) | 19.561 ms / 51.122 t/s/u | 19.096 ms / 52.368 t/s/u | +2.44% |
| `token_out_readback` | 19.701 ms / 50.760 t/s/u | 19.231 ms / 52.001 t/s/u | +2.44% |
| warmed TTFT (prompt 128) | 126.114 ms | **129.941 ms** | **+3.03%** |

Both sampler legs return token 16, before and after.

**This benchmark was re-run at the end of the re-review, after every `tt/`
change this stage made.** The first stage-07 measurement of it predated the
review fixes, so it described a tree that no longer existed — 19.217 ms /
52.039 t/s/u / TTFT 129.910 ms against the 19.213 / 52.049 / 129.941 above. The
two agree to 0.02% on `token_out` and 0.03% on TTFT, which is what the claim
that `match_cache_dtype` early-returns at the shipped configuration predicts,
now measured rather than argued. Nothing in the selection, the charts or the
band moves.

Two independent benchmarks agree on the decode win: teacher forcing +2.69% and
token-out +2.47%, against a measured band of 0.368%.

### The prefill cost, stated plainly

**TTFT got worse: +3.03% (126.114 → 129.941 ms).** That is outside TTFT's
documented warm spread (~0.55%), so it is a real regression and not noise. The
wider expert blocks that help the batch-1 decode matmul cost a little in the
prefill matmul, which runs at a different M.

That claim rests on the warm benchmark alone, which is the regime with a
documented spread. **The teacher-forcing TTFT does not corroborate it and is not
cited as if it did**: `repeats.json` has `R00` at 3277.95 / 3496.98 / 3463.80 ms
and `R25` at 3248.42 / 3500.34 / 3439.81 ms — fully overlapping ranges whose
mean moves *down*. That measurement is one cold prefill per run and cannot
resolve 3%; it neither supports nor contradicts the warm number.

It is still clearly the right trade, and the arithmetic is the argument rather
than the assertion:

```
TTFT cost   : +3.827 ms, once
per token   : -0.474 ms, every token
breakeven   : 8.1 generated tokens
```

| workload | before | after | change |
| --- | --- | --- | --- |
| prompt 128, generate 128 | 2.626 s | **2.570 s** | **−2.15%** |
| prompt 128, generate 1024 | 20.265 s | **19.784 s** | **−2.37%** |

Any request generating more than ~8 tokens is net faster. The stage's ranking
metric is decode, and decode is what dominates every realistic serving profile;
a config that helped TTFT at decode's expense would have been the worse choice.

### Test suite

`pytest tests/ -m "not models_performance_bare_metal" -q` — **158 passed, 16
deselected** (`logs/pytest_selected.log`). The stage-07 plumbing baseline was
157; the extra test is
`test_precision_config.py::test_fidelity_and_terminal_dtypes_reach_the_device`,
the regression guard for the dead `norm_fidelity` field (§8). Includes
`test_non_aligned_prompt_lengths` and the non-aligned prefill gates. The perf
tests were **not** run — they rewrite committed CSVs.

**Both readiness gates hold, re-run after every `tt/` change in this stage:**

| gate | top-1 | top-5 | top-100 | required |
| --- | --- | --- | --- | --- |
| prefill (`run_prefill_check`) | 0.980 | 1.000 | 1.000 | 0.980 / 1.000 / 1.000 |
| decode (`run_teacher_forcing`) | 0.990 | 1.000 | 1.000 | 0.990 / 1.000 / 1.000 |

`probes/check_published_figures.py` passes: every figure in this document and in
`work_log.md` re-derives from `sweep_results.json`, `repeats.json`,
`selection_reasons.json` and the perf JSONs.

---

## 10. Limitations

1. **`activation_dtype` cannot leave `bfloat16`** on this path, and the barrier
   has *moved* during this stage. `R17`/`R18` were first blocked at
   `paged_fill_cache_device_operation.cpp:36`. `match_cache_dtype` (§6) cleared
   that one — the cast it adds makes the fill writer accept a bfp8 activation —
   and both rows were re-probed on the fixed tree. They now build further and
   die at **`nlp_create_qkv_heads_decode_device_operation.cpp:41`**, whose rule
   is `input_tensor.dtype() == FLOAT32 || input_tensor.dtype() == BFLOAT16`.
   That is a **different op with an unrelated rule**: the decode head-split that
   slices the fused QKV projection into per-head tensors, nothing to do with the
   KV cache. It has no cast escape hatch the way the fill writer did — a bf16
   activation is what it wants, so honouring it would mean casting back to bf16
   at the one point in the decode step where `activation_dtype=bfloat8_b` was
   supposed to be saving bytes. **The conclusion is unchanged: on this path
   `activation_dtype` stays `bfloat16`**, and so `R18`, the only configuration
   in which a bfp8 CCL would have been free of casts, stays unmeasured. The
   follow-up is now a decode-attention question, not a KV one.
2. **`embedding_dtype` cannot be block-float** — ROW_MAJOR layout requirement.
3. **Top-1 has ±0.01 resolution** on this 100-token reference; three configs
   scored *above* the baseline. Top-1 differences of one point are not signal,
   and the floor is set at exactly that resolution rather than tighter. Nor does
   repetition help: teacher forcing is deterministic per config, so `R25`'s
   0.990 ×3 and `R26`'s 0.980 ×3 are one token re-observed, not three
   observations. This is why clause (c) breaks its tie on **simplicity** and
   only then on top-1 — a tiebreak resting on top-1 alone would be resting on
   the axis this limitation says cannot carry it.
4. **The band is measured from three configs × 3 repeats**, not from every row.
   A row whose true variance is much wider than `R25`'s could be misplaced by up
   to its own spread; the nine rows rejected as "inside the band" are all well
   inside it, so this does not affect the selection.
5. **TTFT is noisier than decode** and is reported but not ranked on — it is one
   prefill per run, not a median over 100 traced tokens. The `R16` TTFT
   regression is large enough (+76%) to be safely outside that noise.
6. **`num_experts_per_tok = 8 of 128`**, so expert *weight bytes per token* are
   sparse; the block-width result is about scheduling the sparse matmul, and
   would not necessarily transfer to a dense model.
7. The sweep did not explore `in0_block_w` **interactions with dtype above 32**
   — `R13`/`R14`/`R15` co-tune at 32/24, not 64/24, because the bw64 result
   arrived after them. A bfp8-expert row at 64/24 is unmeasured.
8. **`norm_fidelity` reaches only the decode residual norms.** The prefill norms
   pass no `compute_kernel_config` and take the op default, so `R21`'s number
   prices the decode path only — which is the ranked path, but the field is
   named as if it were global.
9. **bfp8 KV's 2.4x TTFT cost is unexplained.** 7.8 s against 3.3 s on the
   158-token gate prompt, in both `R19` and `R28`, and present in the pre-fix run
   too — so it is not the cast this stage added. Nothing in this stage profiles
   it. It is the number a future capacity stage would have to chase first, and it
   is recorded rather than guessed at.
10. **The pre-fix `R19` row's own log did not survive.** Its numbers (top-1 /
    top-5 / top-100 all 0.010, decode 28.86 t/s/u, TTFT 8108 ms) are preserved in
    §6, in `sweep_results.json` as `prior_measurement_invalid`, and in
    `work_log.md`, but the re-run overwrote `logs/rows/R19_kv_bfp8.log` in place.
    The op-level diagnosis that explains it (`probes/kv_bfp8_diagnosis.json`) is
    intact and is the load-bearing evidence.

---

## 11. Artifacts

| path | what |
| --- | --- |
| `sweep_results.json` / `.csv` | one row per evaluated config: id, dtype policy, fidelity policy, top-1/5/100, Δtop-1, TTFT, traced decode t/s/u, gain, band verdict, regime, command, hardware, mesh, pass/fail |
| `selected_precision_config.json` | **the winner** — all 20 fields |
| `default_precision_config.json` | `DEFAULT_PRECISION`; byte-identical to the above |
| `selection_reasons.json` | the rule, the band, eligible rows, per-row rejection reasons |
| `selection_proof.json` | device readback vs selected config, built with no precision argument |
| `top1_perf_pareto.png`, `top5_perf_pareto.png` | the two required charts |
| `repeats.json` | run-to-run band samples |
| `probes/candidates.py` | the candidate set with per-row rationale |
| `probes/structural_probe.py` + `.json` | tier A: constructibility and resolved widths |
| `probes/sweep_runner.py` | tier B driver |
| `probes/repeats.py` | noise-band probe |
| `probes/kv_bfp8_diagnosis.py` + `.json` | the bfp8 KV op-level diagnosis |
| `probes/analyze_sweep.py` | Pareto analysis, selection, charts |
| `probes/selection_proof.py` | the default-consumption gate, 21 device-verified fields |
| `probes/check_published_figures.py` | re-derives every figure in this README and `work_log.md` from the artifacts above; exits non-zero on drift |
| `probes/perf_full_model.py` + `perf_full_model_selected.json` | post-selection token-out |
| `logs/` | every run, each with command, git state, date, hardware in its header |
| `work_log.md` | stage-07 narrative, including the plumbing phase |

### Commands

```bash
# every published figure, re-derived from the artifacts (no device needed)
python doc/datatype_sweep/probes/check_published_figures.py

# tier A: constructibility + resolved block widths (2 layers)
python doc/datatype_sweep/probes/structural_probe.py --layers 2

# tier B: every Pareto row, 48 layers, one subprocess each
python doc/datatype_sweep/probes/sweep_runner.py
python doc/datatype_sweep/probes/sweep_runner.py --include-stacked \
    --only R23_gateup_bw64,R24_gateup32_down24,R25_gateup64_down24
python doc/datatype_sweep/probes/sweep_runner.py --include-stacked \
    --only R26_attn_bfp4_bw64_24,R27_attn_bfp4_lofi_bw64_24,R28_kv_bfp8_bw64_24

# the run-to-run band
python doc/datatype_sweep/probes/repeats.py \
    --ids R00_default,R25_gateup64_down24,R26_attn_bfp4_bw64_24 --n 3

# is bfp8 KV broken or imprecise?
python doc/datatype_sweep/probes/kv_bfp8_diagnosis.py

# analysis, selection, charts
python doc/datatype_sweep/probes/analyze_sweep.py

# proof the selection is the default, then post-selection perf
python doc/datatype_sweep/probes/selection_proof.py
python doc/datatype_sweep/probes/perf_full_model.py --layers 48 \
    --prompt-len 128 --gen-len 128 --context 8192 --tag _selected
```
