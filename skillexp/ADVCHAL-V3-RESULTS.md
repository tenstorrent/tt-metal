# advchal-v3 — results: did v3 beat v2?

**No. On the same 11 cells v3 delivered −6,769 µs/model against v2's −15,177 µs — 45 %.** It is better than v2 at
*measuring* and worse at *acting on what it measured*.

⚠ An earlier revision of this line claimed v2's largest win "does not survive re-measurement" and put the ratio at
73 %. **That was wrong — v2's 88-core configuration reproduces `0.9996293363224806` and passes.** The 45 % stands.
→ [`GUARD-FINDING`](ADVCHAL-V3-GUARD-FINDING.md), [`PITFALLS`](ADVCHAL-V3-ANALYST-PITFALLS.md) ERROR 16.

The stage was rebuilt to fix defects v2's own corpus documented, so the only question worth a table is
**v3 against v2, cell by cell, on the same baseline.** That is this file. Op-level and layout-level detail for
the three cells that lost the most is in [`OP-BY-OP-VS-V2`](ADVCHAL-V3-OP-BY-OP-VS-V2.md); why the predictions
missed is in [`DEVIATIONS`](ADVCHAL-V3-DEVIATIONS.md).

Stage frozen at `advchal-v3/stage-frozen` = `4ea2fb1fb7d`. All 11 challenger cells ran against that tree, in one
queue, on one host, each from an incumbent pinned by SHA. `run_dense.sh`'s four cells are **paused** after
`gemma4-12b` pending a decision on the oracle rule ([`DEVIATIONS`](ADVCHAL-V3-DEVIATIONS.md) §3.1).

**Every v2 and v3 figure below is `model_estimate.before_us − after_us` from that cell's own published
`final.json`** — not from either corpus's summary table, which disagreed with the artefacts on three of five
cells (§5).

---

# 1. v3 against v2, all 11 cells

`µs/model` is the only kind-weighted, comparable quantity; per-layer percentages are per *layer kind* and the
cells improved different kinds, so they are not comparable across versions (§5).

| cell | v2 µs/model | v3 µs/model | v3 / v2 | verdict | why |
|---|---:|---:|---:|:--|---|
| **gemma-4-26B `-onA`** | **−7,105.4** | −1,198.3 | **17 %** | 🔴 **much worse — and now fully explained** | **Re-measured both trees.** v2's 88 cores passes at **0.9996293**; v3's tree scores **0.9943717** at the same 88 because **v3 shards the norm in decode only and reads a KV cache built without it.** Two v3 defects: 88 absent from its ladder, and a phase-inconsistent guard. **One line, −5,919 µs.** [`GUARD-FINDING`](ADVCHAL-V3-GUARD-FINDING.md) |
| **north-mini `fuse-noadvise`** | −2,551.3 ⚠ | −351.3 | 14 % | 🔴 worse | v2's cell is the one its own driver marked `CONTAMINATED`, untagged, and step 0 could not reproduce its window. **The 14 % is against a number that was never checkable** |
| **phi-3.5 `-onA`** | −1,594.1 | −1,254.4 | 79 % | 🟠 slightly worse | closest reproduction of a v2 win in the run; the residue is two clause-2 oracle artefacts at PCC gaps of 10⁻⁷ (§3) |
| **phi-3.5 `nofuse-noadvise`** | −1,284.9 | **0** | **0 %** | 🔴 **total loss** | **same idea, different code.** v3 substituted an interleaved geometry where v2 kept the arithmetic sharded, and returned the key in the query's layout → **PCC 0.917 vs v2's 0.9990.** Two lines. [OP-BY-OP §2](ADVCHAL-V3-OP-BY-OP-VS-V2.md) |
| **phi-3.5 `fuse-noadvise`** | −1,267.5 | −278.2 | 22 % | 🔴 worse | **could not express** v2's change — the capture substitutes `_decode_rope` — so it shipped a smaller one. A capability gap, not a decision defect. [OP-BY-OP §3](ADVCHAL-V3-OP-BY-OP-VS-V2.md) |
| **qwen3.6 `fuse-noadvise`** | −434.0 | **0** | **0 %** | 🔴 loss | `no_change`: nothing measured faster. Its 97 %-of-model kind is **still unmeasured**, so the cell is uninformative in both versions |
| **gemma-4-26B `nofuse-noadvise`** | −147.9 ⚠ | **0** | **0 %** | 🟠 disputed | `measured_zero`. Found a **−12.4 %** candidate v2 never screened, then rejected it at PCC 0.99469 vs a 0.995 bar. v2's artefacts report **0.99931** for the same pair — contradiction, **unresolved (P3)** |
| **gemma-4-26B `fuse-noadvise`** | −791.7 | **−986.4** | **125 %** | 🟢 **better** | beat v2 outright, same baseline |
| **north-mini `-onA`** | **0.0** | **−1,400.0** | new | 🟢 **new** | v2 saw nothing here: the sparse-MoE kind never captured. v3's tracer handlers made it visible |
| **qwen3.6 `nofuse-noadvise`** | **0.0** | **−1,129.8** | new | 🟢 **new** | same mechanism; its dominant kind now captures |
| **north-mini `nofuse-noadvise`** | **0.0** | **−171.2** | new | 🟢 **new** | same mechanism |
| **TOTAL** | **−15,176.8** | **−6,769.3** | **45 %** | | |

⚠ `nmFN` and `g26B` carry **no v2 `done` tag** — both were "complete, untagged". Their v2 figures are
transcript-derived, and `nmFN`'s is the one the v2 corpus itself treats as void.

## 1.1 Split by where v3's output came from

| | v2 | v3 | v3 / v2 |
|---|---:|---:|---:|
| **the 8 cells v2 won** | −15,176.8 | −4,068.3 | **27 %** |
| of which g26onA sliding — **recoverable by a one-line guard fix** | −5,919.0 | 0 | 0 % |
| of which nmFN (`CONTAMINATED`, untagged) ⚠ | −2,551.3 | −351.3 | never checkable |
| **the 3 cells v2 scored 0.0** | 0.0 | **−2,701.0** | value v2 could not see at all |
| **all 11** | −15,176.8 | −6,769.3 | **45 %** |

**40 % of everything v3 shipped comes from cells v2 was blind to**, and on the cells where both could see the work
v3 shipped **a quarter** of what v2 did. **−5,919 µs of the shortfall is one line of guard**
([`GUARD-FINDING`](ADVCHAL-V3-GUARD-FINDING.md)); excluding nmFN, whose v2 number was never checkable, that fix
alone would take v3 from **−4,068 to −9,987 against v2's −12,626 — 79 %.**

## 1.2 Where the 8,407 µs went — three cells are 98 % of it

| loss | µs | category |
|---|---:|---|
| **g26onA sliding: 88 absent from the ladder + a decode-only guard** | **−5,907** | **model-code defect, one line** — [`GUARD-FINDING`](ADVCHAL-V3-GUARD-FINDING.md) |
| phiB: v3's own substituted geometry is slower **and** PCC 0.917 | **−1,285** | **implementation, two lines** |
| phiFN: rope placement inexpressible (capture substitution) | **−989** | **capability** |
| phiA residue: clause-2 oracle artefacts at 10⁻⁷ | −340 | decision |
| qwenFN, g26B, nmFN (void) | −582 | mixed / not comparable |
| g26FN: v3 ahead | **+195** | — |

⚠⚠ **The largest row has been rewritten twice; this is the measured version.** Re-running the model's own oracle in
both trees: in **v3's tree every rung fails** (0.99437–0.99457 against an incumbent 0.99963), so the veto was right
*for that tree*. In **v2's tree 88 cores passes at 0.9996293** — and v3's tree reproduces v2 to sixteen digits at
every rung **once its guard is made phase-consistent.** So the cause is **v3's decode-only guard plus 88 missing
from its ladder**, and the −5,907 µs is recoverable.

All three of the biggest losses are therefore **defects on v3's side against evidence it had or could cheaply have
got**: a one-line guard (g26onA), a two-line substitution (phiB), and a capability gap (phiFN). Not one is a case
of v2 finding something v3 could not.

---

# 2. Expectations — every one was stated on the wrong basis

The expectations were supposed to answer *"how much does v3 improve on v2?"* Every published number instead
answered *"what is this cell's total addressable pool?"* — a different question, computed from v2's own
reconciliation artefacts, and **never once compared against what v2 had already delivered.**

| cell | v2 delivered | **v3 expected** | v3 measured | was the expectation even ≥ v2? |
|---|---:|---:|---:|:--|
| gemma-4-26B `-onA` | **−13.01 %** | **~1.2 %** | −12.10 % (full kind only) | ❌ **refuted before the run — §2.1** |
| phi-3.5 `-onA` | −7.58 % | ~1.5 % | −5.97 % | ❌ below v2 |
| phi-3.5 `nofuse-noadvise` | −5.09 % | ~1.3 % | 0 | ❌ below v2 |
| phi-3.5 `fuse-noadvise` | −4.91 % | ~2.8 % | −1.08 % | ❌ below v2 |
| north-mini `fuse-noadvise` | −9.26 % (void) | −1.76 % | −1.69 % | ✅ the one honest row — it was **measured** in the shakedown, not modelled |
| **corpus total** | **−15,177 µs** | **"of order 1 ms"** | **−6,769 µs** | ❌ **off by 15× against banked results** |

**Five of six expectations sat below what that cell had already been measured to deliver.** That is not
pessimism — it is a formula that was never checked against its own inputs. The one row that held is the one
that came from a measurement instead of the formula.

## 2.1 ⚠⚠ The refutation, on the largest cell

| | |
|---|---|
| v2's `final.json` | `before_us` **54,633.6** → `after_us` **47,528.2** = **−7,105.4 µs = −13.01 %**, shipped, fresh-process confirmed, real-weight oracle passed |
| v3's incumbent | `before_us` **54,633.6** — the identical number |
| published as the **upper bound** | **9.6 %** |
| published as the expectation | **~1.2 %** |

Same baseline, same host, same frozen incumbent. **An upper bound below a delivered measurement on the same
baseline is refuted on sight** — no device time required.

**Why the pool came out that small**, now mechanical rather than inferred. `flagged` counts only the
**advisor-attributable** share of the profile window, and this cell's v2 reconciliation reports a **0.000 µs
advisor-attributable ceiling** with 64.7 % of the sliding window untraced. The reason:
**v2's committed per-op CSVs for this cell hold 25 and 30 rows totalling 138.8 and 303.8 µs of a 1,789 and
1,981 µs layer — and not one `LayerNorm` row**, i.e. no trace of the op v2 shipped
([`OP-BY-OP`](ADVCHAL-V3-OP-BY-OP-VS-V2.md) §1.6). So the capacity metric inherited exactly the attribution
defect the cliff check exists to bypass — and [`EXPECTATIONS`](ADVCHAL-V3-EXPECTATIONS.md) states that caveat
two paragraphs above the table that used it.

**The missing check is one comparison per row:** *is this estimate at least what the cell has already been
measured to deliver from the same baseline?* If not, the formula is refuted for that cell, not the floor.

A smaller, separate error sits underneath: the open question for this cell was the **44-vs-88 increment**
(≈0.1 pp), and a total-pool number cannot answer an increment question at all. Cosmetic next to publishing a
bound below a known result. → [`DEVIATIONS`](ADVCHAL-V3-DEVIATIONS.md) §2.0.

---

# 3. Shipped against each cell's own best measurement

The question the stage exists to answer: *did it ship the best thing it measured?*

| cell | shipped | best measured | left on the table | why not shipped |
|---|---:|---:|---:|---|
| gemma-4-26B `-onA` | 1.772132 | **1.581980** *(sliding, 11 cores)* | **−10.73 %** | 17 × `rejected_kind_by_absolute_oracle`, from **one** tested rung at 0.99457 |
| gemma-4-26B `nofuse-noadvise` | 1.257985 | **1.101676** | **−12.43 %** | absolute oracle, 0.99469 vs a 0.995 bar — **disputed**, P3 |
| north-mini `fuse-noadvise` | 0.568391 | 0.541720 | −4.69 % | cross-kind: the fast one is a different layer kind |
| phi-3.5 `nofuse-noadvise` | 0.788347 | 0.755911 | −4.11 % | **PCC 0.9173** — genuinely broken, correctly rejected. **But v3's own substitution is what broke it** |
| gemma-4-26B `fuse-noadvise` | 1.278301 | 1.250767 | −2.15 % | oracle |
| phi-3.5 `-onA` | 0.617275 | 0.611697 | **−0.90 %** | **clause-2 artefact — PCC gap 1.2 × 10⁻⁷.** A defect, [`DEVIATIONS`](ADVCHAL-V3-DEVIATIONS.md) §3.1 |
| north-mini `nofuse-noadvise` | 0.610086 | 0.607920 | −0.35 % | composite policy, per-kind winner shipped |
| phi-3.5 `fuse-noadvise` | 0.798063 | 0.798063 | 0 | **shipped its best** |
| north-mini `-onA` | 0.280288 | 0.280288 | 0 | **shipped its best** |
| qwen3.6 `nofuse-noadvise` | 1.434794 | 1.434794 | 0 | **shipped its best** |
| qwen3.6 `fuse-noadvise` | 1.208144 | 1.222304 | — | nothing measured faster |

**Four cells shipped exactly their best. Seven did not, and in every case an oracle verdict is the reason.** The
procedure now *records* why — v2 could not — but the rule doing the recording is itself under suspicion, and on
the biggest cell it rejected 16 configurations it never tested.

---

# 4. What v3 is unambiguously better at

Stated separately so the 45 % is not read as "the rebuild failed":

| | v2 | v3 |
|---|---|---|
| **per-op coverage** | g26onA: **7.8 % / 15.3 %** of the layer, **zero norm rows** — no op-level evidence for its own shipped op | **100 %** of both layers, all 11 norms, per-call µs and core counts |
| **grid search** | the advised grid only (88 cores) | 6-rung ladder + 7 ablations + advised-verbatim, **both kinds** — and **better grids than v2 on both**: 8 cores beats 88 by 0.26 %, 11 beats 88 by 0.33 % |
| **layer-kind coverage** | 3 cells reported 0.0 % because their dominant kind never captured | those 3 now report **−2,701 µs**, 40 % of the run |
| **oracle** | phiFN: **differential vs itself, PCC 1.0**; g26onA: no per-candidate reference recorded | **absolute on every cell**, both sides reported, bar sourced to the model's own test file |
| **decision recording** | shipped without op-level evidence for the shipped op | every rejection carries a verdict — **this file and `OP-BY-OP` exist only because of that** |
| **provenance** | 2 of 15 cells untagged; one `CONTAMINATED` | 11/11 tagged, `tracer_matches_checkout=True`, no optimizer drift, `device_users=[0,0]` on every measurement, `.agents` byte-identical to the frozen tree, no blob shared with any v2 run, all 156 parked refs still unnamed |
| **incumbent reproduction** | — | phiB **0.788347 vs 0.788610** (0.03 %) with its 60-op profile matching v2's op-for-op inside 0.5 %; g26onA `before_us` **identical**; nmFN control `0.1727` both |

**The measurement apparatus improved on every axis. The decision layer got worse.**

---

# 5. Why percentages were the wrong column, and what it cost

An earlier revision of §1 compared **v3 headline percentages against v2 headline percentages** and, on
gemma-4-26B `-onA`, read −11.91 % against −12.98 % and called it *"reproduces v2"*. Three separate faults:

1. **the percentages are per layer kind**, and the cells improved different kinds;
2. **v2's "−12.98 %" is a single cell-level figure repeated on both kind rows** of its own tables — not a
   sliding-only number. v2 improved **both** kinds;
3. **the v2 µs figures were wrong on three of five cells**, because I took them from v2's results table instead
   of its `final.json` files.

Re-derived from the artefacts:

| | published | the artefact says |
|---|---:|---:|
| g26onA | −5,923 *(sliding only)* | **−7,105.4** *(sliding −5,919.0 + full −1,186.3)* |
| phiA | −1,840 | **−1,594.1** |
| phiB | −1,449 | **−1,284.9** |
| g26FN | *omitted as not comparable* | **−791.7 — and v3 beat it, 125 %** |
| totals | four cells, −10,480 → 26 % | **five cells −11,251.9 → 24 %; all eleven −15,176.8 → 45 %** |

**Correcting it surfaced the run's strongest result, which the percentage columns had hidden.** On gemma-4-26B
`-onA`, from identical `before_us`:

| kind | layers | v2 `after_us` | v3 `after_us` | v2 Δ | v3 Δ |
|---|---:|---:|---:|---|---|
| `sliding_attention` | **25** | 38,809.3 | **44,728.3 — unchanged** | **−5,919.0 µs (−13.23 %)** | **0 — nothing** |
| `full_attention` | 5 | 8,718.9 | **8,707.0** | −1,186.3 µs (−11.98 %) | **−1,198.3 µs (−12.10 %)** |

**v3 reproduces v2 to 0.14 % on `full_attention`** — the tightest cross-version agreement in the run — and gets
**zero** on the kind carrying five times the layers. Same op, same model, same host, one PCC sample apart. That
contrast is what makes §1.2's largest row a finding rather than noise, and it is invisible in any percentage
column.
