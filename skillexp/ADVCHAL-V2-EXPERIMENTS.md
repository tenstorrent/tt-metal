# advchal-v2 — experiments run to test the analysis

Four experiments run on hardware on **2026-08-03, 16:07–16:23 UTC**, after the corpus was complete, to
test claims that the corpus data alone could not settle. Two confirmed a claim, **one refuted one of my
own hypotheses**, and one **found a win the corpus missed**.

## Method

Isolated git worktrees off the cells' own run branches, so nothing touched any other agent's checkout:

| worktree | branch | model |
|---|---|---|
| `/home/mvasiljevic/_exp-wt` | `1527e0a1298` (phi FN run branch) | phi-3.5-mini |
| `/home/mvasiljevic/_exp-llama` | `fc8f549c3b0` (llama-8B run branch) | llama-3.1-8B |
| `/home/mvasiljevic/_exp-nm` | `433053a393a` (`skillexp-cell/advchal-v2/nmFN`) | north-mini |

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
