# advchal-v2 — experiments run to test the analysis

Eight experiments run on hardware on **2026-08-03, 16:07–17:03 UTC**, after the corpus was complete, to test
claims the corpus data alone could not settle.

| | outcome |
|---|---|
| confirmed a published claim | E1, E3 |
| **refuted a hypothesis of mine** | E2, E4 (partly), **E8 (refuted one of my own action points)** |
| **found a win the corpus missed** | E4 (north-mini, −264 µs/model), E5 (gemma-4-26B onA, −375 µs/model), **E7 (gemma-4-26B B, −3,918 µs/model)** |
| **corrected the mechanism behind a published conclusion** | E6 |
| **validated a proposed fix by prediction** | E7 — a static, zero-device-time check flagged the cell before the run |

> **Where the later experiments live.** This file holds E1–E8. E9–E23 are in
> [`COUNTERFACTUALS`](ADVCHAL-V2-COUNTERFACTUALS.md); **E24 is retracted there** (my "the advisor advised an
> unrunnable shard" claim — the error was mine, shard (32,64) not (32,48)); **E25** (the advisor's plan verbatim:
> −10.43 % vs the −4.88 % shipped, PCC 1.0) is in COUNTERFACTUALS; **E26** (apply the advised plan together:
> −17.84 %) and **E27** (the advised matmul placement is neutral) are in
> [`ADVICE-FAITHFULNESS`](ADVCHAL-V2-ADVICE-FAITHFULNESS.md) §7 and §10.
>
> **And a correction that touches E-numbers here:** advised core counts quoted in this file come from
> `advised_cores`, which is understated on 58.3 % of ops — `report.json`'s `cores=` prints only the first range
> of a multi-range `CoreRangeSet`. Corrected values: `advchal-v2-corrected-advice.json`. This does not change
> any measured time in E1–E8; it changes the *label* on what was measured (e.g. phi's rope advice is 32 cores,
> not 22). → [`ADVICE-FAITHFULNESS`](ADVCHAL-V2-ADVICE-FAITHFULNESS.md) §1.

## Method

Isolated git worktrees off the cells' own run branches, so nothing touched any other agent's checkout:

| worktree | branch | model |
|---|---|---|
| `/home/mvasiljevic/_exp-wt` | `1527e0a1298` (phi FN run branch) | phi-3.5-mini |
| `/home/mvasiljevic/_exp-llama` | `fc8f549c3b0` (llama-8B run branch) | llama-3.1-8B |
| `/home/mvasiljevic/_exp-nm` | `433053a393a` (`skillexp-cell/advchal-v2/nmFN`) | north-mini |
| `/home/mvasiljevic/_exp-g26` | `ad3ca71d89b` (g26 onA run branch) | gemma-4-26B |

`ttnn` resolved from the built tree (`TT_METAL_HOME=/home/mvasiljevic/tt-metal`), `models.*` from the
worktree via `PYTHONPATH`. **Each cell's own unmodified `harness.py` was used** — same protocol as the
original runs: ≥10 untimed warm-ups, 5 timed blocks, each the mean of 50 traced decode replays, one
configuration per process. Raw logs and JSON: `~/skillexp-logs/exp-advisor-probe/`.

Where a candidate's grid list was hardcoded in a cell's own source, I edited that list **inside my
worktree only** and say so below.

---

## E1 — phi FN's discarded candidate: reproduce it, and ask the right correctness question

**Claim under test.** phi FN measured a −13.24 % combined candidate and discarded it on a differential
oracle with the bar at 0.999999. Was the win real, and was the rejection justified?

### Timing (fresh control, same harness)

| candidate | median ms | vs control | cell's original |
|---|---|---|---|
| control | **0.808757** / 0.808144 | — | 0.807152 |
| rope `query_key` only — **what shipped** | 0.769096 | −4.90 % | 0.768100 |
| rope `query_key` + norm 11 — **discarded** | **0.700431** | **−13.39 %** | 0.700267 |
| rope `query_key` + norm 32 | 0.702842 | −13.09 % | never tested |

Reproduces the cell to within **0.02 %**.

### Correctness, asked three ways

The cell's own `scripts/oracle.py`, re-run verbatim, reproduces its veto bit-for-bit:

```
oracle_reference: "frozen incumbent with identical real weights and inputs"
oracle_pcc_bar:   0.999999
oracle_message:   0.9999910666979231
oracle_passed:    false
```

Then I mirrored the **model's own** real-weight test
(`tests/test_optimized_decoder.py::test_optimized_decode_real_reference_and_determinism`) exactly —
same seed, inputs, positions, page table, reference and PCC helper — varying **only** the policy.
The model's own bar is `_assert_pcc(..., threshold=0.995)` against the HuggingFace reference:

| policy | PCC vs HF reference | passes 0.995 | deterministic |
|---|---|---|---|
| incumbent (frozen) | 0.9989018932660865 | yes | yes |
| shipped (rope only) | 0.9989018932660865 | yes | yes |
| **discarded combined** | **0.9990440055776042** | **yes** | yes |

### Result

**The discarded candidate is 13.39 % faster, deterministic, passes the model's own bar, and is *more
accurate against the real reference* than the configuration that shipped** (0.99904 vs 0.99890). It was
rejected only for not being bit-identical to the incumbent.

Per model: 32 dense layers × 108.3 µs = **−3,466 µs/model available**, against **−1,267 µs/model**
shipped.

---

## E2 — phi's norm grid: my own hypothesis, refuted

**Claim under test (mine).** phi FN swept 11/12/24 and never reached 32, the exactly-tile-dividing grid
that won on north-mini. I predicted 32 would be better.

phi hidden = 3072 = **96 tiles**. `advisor_norm_cores=N` builds a `gx×gy` width shard with
`block_w = ceil(96/N)`.

| cores | grid | tiles/core | median ms |
|---|---|---|---|
| 1 (control) | — | 96 | 0.808757 |
| 11 (advised, cell's best) | 11×1 | 9 — **uneven**, 99 > 96 | **0.746800** |
| 12 | 6×2 | 8 exact | 0.749000 *(cell)* |
| 16 | 8×2 | 6 exact | 0.748313 |
| 24 | 8×3 | 4 exact | 0.748500 *(cell)* |
| **32 (new)** | 8×4 | 3 exact | 0.748842 |
| **48 (new)** | 8×6 | 2 exact | 0.749709 |

**Refuted.** From 11 to 48 cores this is a **plateau within ~3 µs**, and the *uneven* 11-core grid is
marginally the best of all. Exact tile division buys nothing here. **phi FN's 11/12/24 sweep was
adequate and missed nothing** — its only loss was the oracle bar (E1).

*Correction to my earlier write-up: I had published that phi FN's sweep "never reached the tile-dividing
grid" as a weakened finding. That finding is now dead — measured, not inferred.*

---

## E3 — is llama-8B's zero real?

**Claim under test.** llama-8B shipped nothing and I called it an "honest zero on an already
well-placed decoder". Its norm was already on 32 cores and the advisor wanted 22.

**First finding, from source.** llama's *only* advisor knob is `LLAMA31_ADVISOR_RESIDUAL_CORES`, and it
sets **both** the residual chain geometry **and** the RMSNorm decode grid
(`optimized_decoder.py:225-226`). **The cell could not have isolated the norm** — no knob exists for it.
So its `dense_geometry_64` candidate (+4.22 %) was a *combined* norm+residual change, and a norm-only
effect could not have been separated even in principle.

**Second finding, from source.** `_find_grid(n_tiles=128, target, max_rows=8, max_cols=8)` only accepts
core counts that **exactly divide 128** within an 8×8 grid: `{1, 2, 4, 8, 16, 32, 64}`. The advisor's
recommended **22 is not expressible**; the knob rounds it to 16.

### Sweep of the whole achievable ladder

| setting | median ms | vs control_confirm 0.667737 |
|---|---|---|
| control (default) | 0.665237 *(first process, floor 11.838 µs)* | — |
| control_confirm | **0.667737** *(floor 0.196 µs)* | — |
| 16 (≈ the advised 22) | 0.667376 | −0.05 % — **within the floor** |
| 32 (the shipped grid) | 0.667559 | −0.03 % — within the floor |
| 64 | 0.692968 | **+3.78 %** |
| 8 | 0.742611 | **+11.21 %** |

My `64` reproduces the cell's `dense_geometry_64` (0.692968 vs its 0.6931).

### Result

**Confirmed: llama-8B's zero is real.** Nothing on the achievable ladder beats the default; the
advisor's advice is *neutral* (within the noise floor), not a missed win. There is no hidden norm win.

**Bonus finding.** The first process of the session recorded a noise floor of **11.838 µs**; the
identical configuration in a later process recorded **0.196 µs** — a **60× difference** from JIT-cache
warmth *between processes*, which the per-process warm-up cannot remove. Any cell whose control was the
first thing it ran carries an inflated floor. See [`ADVCHAL-V2-READ-THIS.md`](ADVCHAL-V2-READ-THIS.md) §8 D4.

---

## E4 — north-mini: the best-swept cell in the corpus still left a win

**Claim under test.** north-mini FN swept 22/32/64 and shipped 32 for −10.23 %/model. Was the ladder
exhausted?

north-mini hidden = 2048 = **64 tiles**. I extended the cell's own policy list
(`optimized_decoder.py:281`, `for cores in (22, 32, 64)`) in my worktree only.

### First: most grids are illegal, and the cell's three were nearly the whole legal set

`ttnn/cpp/ttnn/operations/normalization/shard_spec_validation.cpp:104` requires the padded shard width
to exceed the tensor width by **less than one shard width**:

| cores | block_w | padded | pad | legal? |
|---|---|---|---|---|
| 22 | 3 (96) | 2112 | 64 < 96 | ✓ |
| **32** | 2 (64) | 2048 | 0 | ✓ exact |
| 40 | 2 (64) | 2560 | 512 ≥ 64 | ✗ `TT_FATAL` |
| 44 | 2 (64) | 2816 | 768 ≥ 64 | ✗ `TT_FATAL` |
| 48 | 2 (64) | 3072 | 1024 ≥ 64 | ✗ `TT_FATAL` |
| 55 | 2 (64) | 3520 | 1472 ≥ 64 | ✗ `TT_FATAL` |
| 64 | 1 (32) | 2048 | 0 | ✓ exact |
| 88 | 1 (32) | 2816 | 768 ≥ 32 | ✗ |

So **nothing between 32 and 64 is legal**, and 44/88 — which I had suggested the cell should have tried —
**cannot run at all**. That part of my earlier recommendation was wrong.

### Then: the legal ladder, against the true frozen 1-core baseline

| cores | tiles/core | median ms | vs frozen |
|---|---|---|---|
| **1 (frozen incumbent)** | 64 | **0.577971** / 0.577993 | — |
| 4 | 16 exact | 0.569837 | −1.41 % |
| 8 | 8 exact | 0.518230 | −10.34 % |
| 11 | 6 + pad | 0.526071 | −8.98 % |
| **16** | **4 exact** | **0.512764** | **−11.28 %** ← best |
| 22 (**advised**) | 3 + pad | 0.543079 | −6.03 % |
| **32 (shipped)** | 2 exact | 0.518022 | −10.37 % |
| 64 | 1 exact | 0.573559 | −0.76 % |

The curve is **non-monotonic and bimodal**: minima at 16 and at 8/32, local maxima at 22 (the advised
value) and 64.

### Confirmation, interleaved fresh processes, both MoE layer kinds

| | 16 cores | 32 cores (shipped) | gap | floors |
|---|---|---|---|---|
| layer 1, sliding-attention MoE | 0.512764 / 0.512898 / 0.512636 | 0.518022 / 0.518192 / 0.518384 | **5.4 µs** | 0.32–1.16 µs |
| layer 4, full-attention MoE | 0.514273 / 0.513835 | 0.519719 | **5.7 µs** | 0.28–0.97 µs |

Every 16-core run beats every 32-core run on both kinds — the stage's non-overlap rule holds.

### Correctness at 16 cores

Mirrored `tests/test_optimized_decoder.py::test_advisor_full_moe_policy_real_weights_against_functional`
exactly (official layer-1 tensors remapped to layer 4, real weights, model's own 0.995 bar), varying only
the candidate:

| candidate | PCC vs reference | passes 0.995 |
|---|---|---|
| 1 core (frozen) | 0.9995916558301281 | yes |
| **16 cores** | **0.9995141137226552** | **yes** |
| 32 cores (shipped) | 0.9995256482940138 | yes |

The 32-core figure reproduces the cell's reported `0.999526` exactly, which validates the whole probe.

### Result

**A further win of ~5.5 µs/layer across 48 MoE layers ≈ −264 µs/model**, taking north-mini from the
shipped −10.23 % to roughly **−11.3 %**. It is correct at the model's own bar.

**And the reason it was missed is not laziness.** The skill instructs cells to *"never sweep only at or
below an advised core count; always measure at least one exactly-dividing grid"* — so the cell swept
**at and above** 22 (22, 32, 64). The optimum is **16, below the advised value**, and also
exactly-dividing. The rule pointed the sweep in exactly the wrong direction for this model.

---

## What these four experiments change

| earlier claim | status after measurement |
|---|---|
| phi FN's −13.24 % was real and shippable | **confirmed**, and it is *more* accurate than what shipped |
| phi FN should have tested the 32-core exactly-dividing grid | **refuted** — 11→48 is a plateau, 11 is best |
| llama-8B's zero is honest | **confirmed** — whole ladder measured, nothing beats the default |
| llama's norm was never screened | **confirmed, and it was not screenable** — one knob drives norm + residual together |
| north-mini should have tried 44 and 88 cores | **refuted** — both are illegal (`TT_FATAL`) |
| north-mini's sweep was thorough | **refuted** — 16 cores is 1 pp better and was never tried |
| "sweep above the advised core count" | **wrong as stated** — the optimum was *below* it. Sweep the whole legal ladder. |

---

## E5 — gemma-4-26B onA: the corpus's biggest win shipped the wrong grid

**Claim under test.** g26 onA shipped a 1→88-core hidden-norm re-grid for −12.98 %/layer, the largest
shipped layer win in the corpus. Was 88 the best legal grid?

### The legal ladder is four rungs, and the cell measured one

`HIDDEN_SIZE = 2816` = 88 tiles. The cell's own code (`tt/optimized_decoder.py:192-202`) enforces two
constraints:

1. `HIDDEN_SIZE % cores == 0` **and** `(HIDDEN_SIZE // cores) % TILE_SIZE == 0` → cores must divide 88
2. `cores % 11 == 0` and `cores // 11 <= 10` → **multiples of 11 only**

So the cell's own legal set is **{11, 22, 44, 88}**, and its error message asserts "88 cores is the maximum
legal width-sharded grid". It measured **only 88**.

⚠ Constraint 2 is the *cell's* assumption, not the hardware's — north-mini ran 16- and 32-core norms
successfully on the same device. Without it, {1, 2, 4, 8} are also tile-aligned here.

### Sweep of the cell's own legal ladder — sliding attention

| cores | median ms | vs frozen 1.823508 | floor µs |
|---|---|---|---|
| 1 (frozen incumbent) | 1.824282 / **1.823508** | — | 1.354 / 1.163 |
| 11 | 1.574171 / 1.573104 | −13.72 % | 13.783 / 8.897 |
| 22 | 1.579826 | −13.35 % | 5.962 |
| **44** | **1.574985 / 1.575744 / 1.574808** | **−13.62 %** | 0.859 / 3.665 / 1.232 |
| **88 (shipped)** | 1.587511 / 1.587481 / 1.587275 | −12.94 % | 2.145 / 11.865 / 4.168 |

### Full attention

| cores | median ms | vs frozen 2.012718 |
|---|---|---|
| 1 (frozen) | 2.012718 | — |
| **44** | **1.763661 / 1.763577** | **−12.37 %** |
| 88 (shipped) | 1.776045 | −11.75 % |

**44 beats the shipped 88 by 12.2 µs/layer (sliding) and 12.4 µs/layer (full)**, with every 44-run below
every 88-run on both kinds. 11 cores is marginally lower still on sliding but its floors (8.9–13.8 µs) are
as large as the difference, so **44 is the defensible choice** — tightest floor, cleanest separation.

Model-level: 25 sliding + 5 full layers × ~12.3 µs ≈ **−375 µs/model** beyond what shipped.

### Correctness: 44 is numerically equivalent to what already shipped

The real gemma-4-26B weights are **not on this host** — the HF cache holds `config.json` only (28 KB), and
the model's real-weight test fails identically for the **shipped 88-core default**, so it is not evidence
about 44. Using the cell's own synthetic-weight harness state instead:

| comparison | PCC | passes 0.995 |
|---|---|---|
| 44 vs shipped 88, sliding | **1.0** | yes |
| 44 vs shipped 88, full | 0.9999984316907383 | yes |
| 1-core frozen vs shipped 88, sliding | **0.9822515179837364** | **no** |
| 1-core frozen vs shipped 88, full | 0.9998214001052961 | yes |

44 is **bit-identical** to the shipped configuration on sliding attention, so it inherits the real-weight
oracle the cell already passed at PCC 0.999629 / 0.999787.

### The row that indicts the oracle rule

The third row is the important one. **gemma's shipped 1→88 norm change moves a differential PCC by
0.0177** — enough to fail a 0.995 bar outright. phi FN's 1→11 norm change moved a differential PCC by
**0.0000089** and was **rejected for it**.

| cell | same class of change | differential PCC movement | oracle built | outcome |
|---|---|---|---|---|
| gemma-4-26B onA | 1 → 88 cores | **0.0177** | absolute, vs HuggingFace, bar 0.995 | **shipped** (−12.98 %) |
| phi-3.5 FN | 1 → 11 cores | **0.0000089** | differential, vs frozen incumbent, bar 0.999999 | **rejected** |

The cell whose change perturbed the output **~2,000× more** shipped it. The cell whose change perturbed it
least did not. **The oracle construction decided both outcomes; the numerics decided neither.**

*Caveat, stated: gemma's differential is measured on synthetic weights with `BFLOAT8_B` experts and phi's on
real weights, and they are different models — so the two PCC deltas are not a controlled comparison. The
asymmetry in outcome, however, does not depend on that.*

### Result

- A **third** cell shipped a grid that is not the best legal grid. Running total of value left on the table
  by grid choice alone: **−264 µs/model** (north-mini) **−375 µs/model** (gemma-4-26B onA).
- Across the three cells with a low-core reduction, **every one whose ladder had more than one legal rung
  shipped the wrong rung.** Only phi — whose curve is a plateau — happened to ship the best one.

---

## E6 — gemma-4-26B FN: why the "same candidate" regressed, and what it means for §5

**Claim under test.** g26 FN measured the 88-core norm at **+0.43 %** (a regression) while g26 onA measured
it at **−13.03 %**. I published this as the corpus's cleanest evidence that a large advisor win can be the
size of the starting point's defect. Was that reading right?

### It was the right conclusion for the wrong reason

Reading g26 FN's source (`tt/optimized_decoder.py:386`):

```python
# Advisor-challenger candidate knob.  Eight cores remains the frozen
# incumbent unless the stage explicitly selects another legal grid.
advisor_norm_cores = int(os.environ.get("GEMMA4_ADVISOR_NORM_CORES", "8"))
```

**g26 FN's frozen incumbent already places the norm on 8 cores.** So its measurement was **8 → 88**, not
**1 → 88**. The two arms never ran the same experiment. (g26 onA's knob defaults to 88 and its frozen
incumbent leaves the norm unsharded — effectively 1 core.)

### The full ladder on g26 FN, sliding attention

`HIDDEN_SIZE = 2816`, so cores must tile-divide 88.

| cores | median ms | vs incumbent 1.318274 |
|---|---|---|
| 1 | **cannot run** — `TT_THROW: Statically allocated circular buffers … clash with L1 buffers on core range [0-0 - 0-0]` | — |
| 2 | 1.364733 | +3.52 % |
| 4 | 1.330788 | +0.95 % |
| **8 (frozen incumbent)** | **1.318539 / 1.318274** | — |
| 11 | 1.317930 | −0.03 % |
| **22** | **1.316251** | **−0.15 %** (2.0 µs; floors 2.19 / 1.22 µs — **not separated**) |
| 44 | 1.316590 | −0.13 % |
| 88 | 1.324489 | +0.47 % |

### Result: §5's conclusion confirmed, and its mechanism corrected

Putting the two arms' ladders side by side:

| cores | 1 | 2 | 4 | 8 | 11 | 22 | 44 | 88 |
|---|---|---|---|---|---|---|---|---|
| **onA** (incumbent at 1) | **1.8235** | — | — | — | 1.5736 | 1.5798 | **1.5750** | 1.5875 *(shipped)* |
| **FN** (incumbent at 8) | can't run | 1.3647 | 1.3308 | **1.3183** | 1.3179 | **1.3163** | 1.3166 | 1.3245 |

**The norm response curve is flat from ~8 to ~44 cores, and essentially all of the available win is in
getting off 1 core.** onA was on 1 and gained 13.7 %. FN was already on 8 — already on the flat part — and
has **nothing left that clears its noise floor**.

So the earlier framing ("the same candidate won on the slow arm and regressed on the fast arm, therefore the
win was the size of the starting point's defect") reaches the right conclusion, but the mechanism is more
specific and more useful:

- ❌ Not "the two arms responded differently to the same change."
- ✅ **The two arms had different norm placements to begin with (1 core vs 8 cores), and the entire win is
  the first step off 1 core.** FN's stage-02 arm had already taken it.

And note the trap this creates for the stage: **the same env knob name means different things in the two
arms** (`GEMMA4_ADVISOR_NORM_CORES` defaults to 88 in one and 8 in the other), so a cross-arm comparison of
"the 88-core candidate" is comparing two different deltas. Nothing in the stage records the incumbent's own
grid for the op under test, which is exactly the field that would have made this visible.

---

## E7 — the static predictor's prediction, tested: g26 B left a −12.4 % win behind a knob it built itself

**Where the prediction came from.** A one-line static check over the corpus's own per-op data — *shipped
grid ≤ 2 cores **and** the advisor wants strictly more **and** the op is ≥ 2 % of the window* — flags 5 of
14 cells:

| cell | flagged | largest actionable low-core op | what actually happened |
|---|---|---|---|
| g26 onA | ✓ ×8 | `rms_norm` 1→88c, 44.7 µs, 2.5 % | shipped −12.98 %/layer |
| **g26 B** | **✓ ×8** | **`rms_norm` 1→88c, 44.5 µs, 3.7 %** | **never screened it** |
| nm FN | ✓ ×2 | `rms_norm` 1→22c, 26.1 µs, 5.0 % | shipped −10.37 %/layer |
| nm onA | ✓ ×3 | `rms_norm` 1→22c, 26.1 µs, 3.2 % | could not screen — sparse MoE untraceable |
| phi FN | ✓ ×2 | `rms_norm` 1→11c, 44.5 µs, 6.1 % | measured −13.4 %, discarded on the oracle |
| the other 9 cells | ✗ | — | **none produced a double-digit layer win** |

Every double-digit win in the corpus sits in a flagged cell; no unflagged cell produced one. So the check
predicted that **g26 B has a large unscreened win**. This experiment tests that.

### g26 B built the candidate and shipped it disabled

`tt/optimized_decoder.py:58`:

```python
_RESIDUAL_SHARD_GEOMETRIES = {
    0: None,               # <- the shipped default: norm unsharded
    11: (11, 1, 256, 8, 4),
    22: (11, 2, 128, 4, 4),
}
```

The knob is `GEMMA4_OPT_RESIDUAL_SHARD_CORES`. The cell wrote both geometries, shipped `0`, and its
measurement list contains no residual-shard candidate at all.

### Measured

| kind | control R=0 | R=11 | R=22 |
|---|---|---|---|
| **sliding** (25 layers) | 1.259101 / **1.258327** / 1.258247 / 1.258111 | 1.132508 (−10.0 %) | **1.101768 / 1.101644 / 1.102353 / 1.101011** (**−12.44 %**) |
| **full** (5 layers) | 1.261566 | 1.350694 (**+7.07 %**) | 1.320076 (**+4.64 %**) |

Floors on the sliding runs: 0.402–4.154 µs against a **156.7 µs** gap. Every R=22 repeat beats every R=0
repeat by more than two orders of magnitude of the floor — non-overlap is not close.

Per the stage's own product rule the ship would be **R=22 for sliding, R=0 for full**:

```
25 sliding layers x 156.7 us  =  3,918 us/model
```

against the **−147.9 µs/model** the cell actually shipped — a **26× larger** win, and about **−10.8 %** of
its 36,224 µs full-model estimate.

### Correctness: NOT established, and that is itself the finding

gemma-4-26B's real weights are **not on this host** (HF cache = `config.json`, 28 KB), so the absolute
oracle cannot be run. Using the cell's own harness state and its own `shipped_policy`, the differential
comparison is:

| comparison | differential PCC | vs a 0.995 bar |
|---|---|---|
| sliding: R=22 vs frozen R=0 | **0.9832233095682822** | **fails** |
| sliding: R=11 vs frozen R=0 | 0.9830912129626893 | fails |
| full: R=22 vs frozen R=0 | 0.9996136713325892 | passes |
| full: R=11 vs frozen R=0 | 0.9996100242015759 | passes |

**Circumstantial evidence that the sliding figure is benign reassociation, not a bug:** g26 onA's *shipped*
change moved its differential PCC by almost exactly the same amount on the same layer kind of the same
model (0.98225), and it then **passed the absolute real-weight oracle at PCC 0.999629**. Synthetic weights
with `BFLOAT8_B` experts amplify reassociation, which is what both figures look like.

But that is inference, not measurement. **Stated plainly at the time: the −12.4 % timing win on g26 B is solid
and reproduced four times; its correctness was unverified.**

✅ **Settled afterwards in [`COUNTERFACTUALS`](ADVCHAL-V2-COUNTERFACTUALS.md) §E9.** Against the model's own
bfloat16 `FunctionalDecoder` the candidate scores **0.99931** and the **shipped incumbent 0.98347** — on sliding
attention the incumbent is the one that fails the 0.995 bar. The differential number above was flagging the
*better* configuration.

And that is exactly the ambiguity action point **A1** exists to remove. With only a differential number I
cannot distinguish "reassociated" from "wrong", and the stage's rule would say *reject* — which is how a
26×-larger win would be lost a second time.

### Result

- **The static predictor works.** It flagged the cell, and the win is there.
- **Running total left on the table across the corpus: ≈ 8.0 ms/model** — phi FN 3.47 ms, g26 B 3.92 ms
  (correctness pending), gemma-4-26B onA 0.38 ms, north-mini FN 0.26 ms.
- **The single cheapest fix in this whole analysis is a static check that needs no device time**, computed
  from data `reconcile.py` already has. See action point **C1b**.

---

## E8 — does tightening the harness rescue an overlapping candidate? No — and it makes the floor worse

**Claim under test (mine).** Action point B3 proposed: *"if a candidate's median beats the control but its
blocks overlap, re-measure once at ≥4× `ITERS` before recording a rejection."* phi exp17 rejected
`rope_l1_tail` exactly that way — better median (1.100683 vs 1.100939), overlapping repeats — while holding
the corpus's largest ceiling (83.6 µs/layer). The skill's own `not_measurable` guidance says to "tighten the
harness (more replays per timed block)".

### Measured, with `CHALLENGER_WARMUP=20`

| protocol | control | candidate `rope_l1_tail` | floors |
|---|---|---|---|
| **5 blocks × 50** (the stage default) | 1.100367 | 1.100293 | 0.708 / 0.447 µs |
| **9 blocks × 200** (4× replays, 1.8× blocks) | 1.100427 / 1.100372 | 1.100077 / **1.100667** | 1.344 / 2.991 / 2.964 / 2.001 µs |

### Two results, both negative for the proposal

**1. It does not separate.** Under the tightened protocol the candidate's two medians *straddle* the
control's: 1.100077 < 1.100372 < 1.100667. The effect is genuinely below what this setup resolves at any
replay count tried. **phi exp17's rejection was correct**, and B3 as written would have spent device time to
reach the same answer.

**2. Tightening made the noise floor 3–4× *worse*.** Going from 250 replays per measurement (5×50) to 1,800
(9×200) took the floor from **0.4–0.7 µs to 1.3–3.0 µs**.

That contradicts the reasoning the protocol is built on. `harness_template.py` argues:

> `ITERS >= 50` — Each timed block reports the MEAN of ITERS replays, so the spread between blocks is the
> spread of means, roughly `sqrt(ITERS)` tighter than single-shot timing.

`sqrt(ITERS)` tightening assumes the noise is i.i.d. within a run. It is not: a longer measurement window
picks up slow drift (clock, thermal, allocator state), and drift does not average down. **Past some point,
more replays per block measures the drift instead of the operation.** The corpus's own protocol
(50 replays/block) appears to sit near the sweet spot; 200 is past it.

### Corrected recommendation

**B3 as originally written is wrong.** Replace it with:

> On an overlap with a favourable median, **do not** assume more replays will resolve it — measured on this
> hardware, 4× replays made the block spread 3–4× *worse*. Record `not_measurable` with the arithmetic, and
> report the unrealised ceiling. If you do re-measure, re-measure **more independent processes at the same
> block size**, not longer blocks — the cross-process term (§E3, 60×) is the one that actually dominates.
