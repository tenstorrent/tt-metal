# advchal-v3 — results: 11 cells, expected against measured

Stage frozen at `advchal-v3/stage-frozen` = `4ea2fb1fb7d`. All 11 challenger cells ran against that tree, in
one queue, on one host, each from an incumbent pinned by SHA. `run_dense.sh`'s four cells are paused after
`gemma4-12b` pending a decision on the oracle rule ([`DEVIATIONS`](ADVCHAL-V3-DEVIATIONS.md) §3.1).

Expectations are from [`EXPECTATIONS`](ADVCHAL-V3-EXPECTATIONS.md); v2 figures are that corpus's own results
table. Δ is per layer on the cell's dominant kind; µs/model is the cell's own `model_estimate` delta.

## 1. Every cell

| cell | v2 | expected | **measured** | µs/model | band | established? |
|---|---|---|---|---:|---:|---|
| gemma-4-26B `-onA` | −12.98 % *(cell-wide; **−7,105 µs**)* | ~1.2 % ⚠⚠ *(refuted pre-run — §1b)* | **−12.10 % on `full_attention` (5 L); `sliding` +0.00 %** | −1198 | 334 | yes |
| phi-3.5 `-onA` (phiA) | −8.75 % | ~1.5 % | **−5.97 %** | −1254 | 30 | yes |
| north-mini `-onA` | **0.0 %** | ~0.5 % | **−3.94 %** | −1400 | 47 | yes |
| gemma-4-26B `fuse-noadvise` | −2.04 % | unknown | **−2.16 % sliding (25 L), −4.04 % full (5 L)** | −986 | 64 | yes |
| north-mini `fuse-noadvise` | −10.23 % *(void)* | −1.76 % *(measured in shakedown)* | **−1.69 %** | −351 | 42 | yes |
| phi-3.5 `fuse-noadvise` | −4.91 % | ~2.8 % | **−1.08 %** | −278 | 31 | yes |
| qwen3.6 `nofuse-noadvise` | **0.0 %** | ~0.02 % | **−0.12 % on `linear_attention` (48 L), −1.16 % full (16 L)** | −1130 | 372 | yes |
| north-mini `nofuse-noadvise` | **0.0 %** | ~0.1 % | **−0.59 %** | −171 | 116 | yes |
| gemma-4-26B `nofuse-noadvise` | −0.34 % | uncalibrated | **`measured_zero`** | 0 | 26 | — |
| phi-3.5 `nofuse-noadvise` (phiB) | −5.74 % | ~1.3 % | **`no_change`** | 0 | 12 | — |
| qwen3.6 `fuse-noadvise` | inside band | ~0.4 % | **`no_change`** | 0 | 385 | — |

**8 of 11 shipped a change, and all 8 are outside their own uncertainty band.** Corpus total
**−6,769.6 µs/model ≈ 6.8 ms**.

## 1a. ⚠ CORRECTION — the percentage column is not comparable to v2's, and one row was wrong

An earlier revision of this table put v2 at −12.98 %, an expectation of ~1.2 % and a measured −11.91 % on the
same row for gemma-4-26B `-onA`, and called it *"reproduces v2"*. **All three numbers are on different footings
and the verdict was false.** Corrected here.

**The percentages are per *kind*, and cells improved different kinds.** For that cell:

**v2 improved *both* kinds** — from its `model_estimate.per_kind`, not from its percentage column:

| kind | layers | v2 `after_us` | v3 `after_us` | v2 Δ | v3 Δ |
|---|---:|---:|---:|---|---|
| `sliding_attention` | **25** | 38,809.3 | **44,728.3 — unchanged** | **−5,919.0 µs (−13.23 %)** | **0 — nothing** |
| `full_attention` | 5 | 8,718.9 | **8,707.0** | −1,186.3 µs (−11.98 %) | **−1,198.3 µs (−12.10 %)** |
| | | | | **−7,105.4** | **−1,198.3** |

Both start from the identical `before_us` (44,728.325 / 9,905.275), so these are the same measurement twice.
Two things follow, and the first was not visible in the percentage column at all:

- **on `full_attention`, v3 reproduces v2 to 0.14 %** — 8,707.0 against 8,718.9. That is the strongest
  cross-version agreement in the run, and it is what makes the other row a finding rather than noise.
- **on `sliding_attention`, v3 got nothing where v2 got −5,919 µs**, on the kind carrying five times more layers.
  Why: it tried **one** rung (1 → 11 cores), hit PCC 0.99457 against a 0.995 bar, and stopped searching that
  kind — never trying the 88 cores v2 shipped. [`PCC-BY-GRID`](ADVCHAL-V3-PCC-BY-GRID.md) §2,
  [`DEVIATIONS`](ADVCHAL-V3-DEVIATIONS.md) §3.2a.

At model scope: **−1,198 µs against v2's −7,105 µs — 17 %.** "Reproduces v2" was true of one kind and false of
the cell, and I produced it by comparing a v3 headline percentage against a v2 percentage without checking they
described the same layer kind — the error v2's own corpus warns about twice (*"state the scope on every
number"*, *"per-layer ranking picks the wrong candidate across kinds"*).

**Only `µs/model` is kind-weighted and therefore comparable.** On that basis, for the four cells where v2's
control and layer counts are both recorded:

| cell | v2 µs/model | v3 µs/model | v3 as % of v2 | baselines agree? |
|---|---:|---:|---:|---|
| gemma-4-26B `-onA` | **−7,105.4** | −1,198.3 | **17 %** | **exactly** — `before_us` 54,633.6 both |
| phi-3.5 `-onA` | −1,594.1 | −1,254.4 | 79 % | 18,210.9 vs 18,268.3 (+0.3 %) |
| phi-3.5 `fuse-noadvise` | −1,267.5 | −278.2 | 22 % | 23,205.6 vs 23,162.7 (−0.2 %) |
| phi-3.5 `nofuse-noadvise` | −1,284.9 | **0** | **0 %** | 22,392.6 vs 22,302.8 (−0.4 %) |
| gemma-4-26B `fuse-noadvise` | −791.7 | **−986.4** | **125 %** | 38,887.6 vs 39,227.3 (+0.9 %) |
| **those five** | **−12,043.6** | **−3,717.3** | **31 %** |  |

Against that, the three cells v2 scored 0.0 % give v3 **−2,701 µs/model**. So the honest summary is:
**v3 recovers about a third of what v2 delivered on the cells v2 won — beating it on one — and adds roughly the
same amount again from cells v2 could not see at all.** It is a more trustworthy 6.8 ms, not a larger one.

**Three corrections inside this table itself,** all from re-reading v2's `final.json` rather than its results
table: g26onA's v2 figure was the **sliding-only** −5,919 µs where the cell shipped **both** kinds for −7,105 µs;
phiA was −1,840 (should be −1,594.1) and phiB −1,449 (should be −1,284.9); and **g26FN was omitted although it
is comparable — and it is the cell where v3 beat v2.** Every v2 number here is now
`model_estimate.before_us − after_us` from that cell's own published `final.json`.

## 1b. ⚠⚠ And the ~1.2 % expectation should never have been published — it was refuted by v2's own data

Asked plainly: *why would we ever expect less than v2 got?* We would not, and there is no defence.

**v2's `final.json` for this cell: `before_us` 54,633.6 → `after_us` 47,528.2 = −7,105.4 µs = −13.01 %,
shipped, fresh-process confirmed, real-weight oracle passed. v3's incumbent profiles the same cell at
`before_us` 54,633.6 — the identical number.** Same baseline, same host, same frozen incumbent. So a
**~1.2 % expectation sat below a 13.0 % result already on file from the same starting point**, and the 9.6 %
figure I called an *upper bound* was **below a delivered measurement**. A bound that excludes an observation
drawn from its own source data is refuted on sight — no device time required.

**Why the pool came out that small, which is the mechanism:** `flagged` is the **advisor-attributable** share of
the profile window, and this cell's v2 reconciliation reports a **0.000 µs advisor-attributable ceiling** with
**64.7 % of the sliding window untraced**. The capacity metric therefore inherited the exact attribution defect
the cliff check was built to bypass — and
[`EXPECTATIONS`](ADVCHAL-V3-EXPECTATIONS.md) states the caveat two paragraphs above the table that uses it.

**The check that was missing is one comparison per row:** *is this estimate at least what the cell has already
been measured to deliver from the same baseline?* If not, the formula is refuted for that cell, not the floor.
Applied corpus-wide it kills the headline too: "of order 1 ms" against **12.0 ms** banked across v2's five
comparable cells.

There is a *separate*, smaller error underneath it — the open question for this cell was the **44-vs-88
increment** (≈0.1 pp), and a total-pool number cannot answer an increment question at all. But that one is
cosmetic next to publishing a bound below a known result.

For scale: v2 claimed "≈9.2 ms/model still on the table"; my revised expectation was "of order 1 ms"; **12.0 ms
was already delivered and 6.8 ms was measured.** Both predictions were wrong and **the revision was the worse
of the two** — [`DEVIATIONS`](ADVCHAL-V3-DEVIATIONS.md) §2.0.

## 2. The three coverage zeros, which are the clean win

| cell | v2 | v3 | mechanism |
|---|---|---|---|
| north-mini `-onA` | 0.0 % | **−1400 µs/model** | tracer handlers: the sparse-MoE kind became visible |
| qwen3.6 `nofuse-noadvise` | 0.0 % | **−1130 µs/model** | same; its dominant kind now captures |
| north-mini `nofuse-noadvise` | 0.0 % | **−171 µs/model** | same |

**2.7 ms/model — 40 % of the run's total — comes from three cells that previously reported nothing.** This is
the one prediction that held in kind and beat its size, and it is the coverage argument, not the placement
argument: these cells were not screened badly, the advisor was never shown the layer.

## 3. Shipped against each cell's own best measurement

The question the stage exists to answer is *"did it ship the best thing it measured?"*

| cell | shipped | best measured | left on the table | why not shipped |
|---|---:|---:|---:|---|
| gemma-4-26B `nofuse-noadvise` | 1.257985 | **1.101676** | **−12.43 %** | absolute oracle, 0.99469 vs 0.995 bar — **disputed**, P3 |
| gemma-4-26B `-onA` | 1.772132 | **1.581980** | **−10.73 %** | 14 × `rejected_kind_by_absolute_oracle` |
| north-mini `fuse-noadvise` | 0.568391 | 0.541720 | −4.69 % | cross-kind: the fast one is a different layer kind |
| phi-3.5 `nofuse-noadvise` | 0.788347 | 0.755911 | −4.11 % | **PCC 0.9173** — genuinely broken, correctly rejected |
| gemma-4-26B `fuse-noadvise` | 1.278301 | 1.250767 | −2.15 % | oracle |
| phi-3.5 `-onA` | 0.617275 | 0.611697 | **−0.90 %** | **clause-2 artefact — PCC gap 1.2 × 10⁻⁷.** A defect, §3.1 of DEVIATIONS |
| north-mini `nofuse-noadvise` | 0.610086 | 0.607920 | −0.35 % | composite policy, per-kind winner shipped |
| phi-3.5 `fuse-noadvise` | 0.798063 | 0.798063 | 0 | **shipped its best** |
| north-mini `-onA` | 0.280288 | 0.280288 | 0 | **shipped its best** |
| qwen3.6 `nofuse-noadvise` | 1.434794 | 1.434794 | 0 | **shipped its best** |
| qwen3.6 `fuse-noadvise` | 1.208144 | 1.222304 | — | nothing measured faster |

**Four cells shipped exactly their best. Seven did not, and in every case an oracle verdict is the reason.**
So the decision procedure now *records* why — which v2 did not — but the rule doing the recording is itself
under suspicion.

## 4. What `apply_all` did, per cell

F5's bound, measured rather than assumed:

| verdict | cells |
|---|---|
| `hard_error` (no knob for the advised placements) | 6 — phiFN, phiA, phiB, g26B, nmFN, nmOnA |
| applied and measured, in some form | **5** — g26onA (`measured`), g26FN and nmB (`…maximal_expressible_subset`), nmB again (`…then_ablated`), qwenB (`hard_error_then_maximal_expressible_subset_measured`) |
| `rejected` | 1 — qwen |

So *"F5 is unexecutable"* — which I wrote after one cell — is **too strong**: five cells applied an expressible
subset, and the cell with the largest single-kind gain (g26onA, −12.10 % on `full_attention`) is the cell that
applied the advised plan and measured it. The bound is real and partial, not absolute.

## 5. Provenance, uniform across all 11

`tracer_matches_checkout=True`, `optimizer_files_changed_since_pin=[]`, advisor at `97724a1170` against the
pin, `device_users=[0,0]` on every measurement, `.agents` byte-identical to the frozen tree on every cell, no
blob shared with any v2 run, all 156 parked refs still unnamed at the end of every cell. One publish failure,
no measurement lost — [`RUN-LOG`](RUN-LOG.md) P6.
