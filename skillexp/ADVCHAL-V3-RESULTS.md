# advchal-v3 — results: 11 cells, expected against measured

Stage frozen at `advchal-v3/stage-frozen` = `4ea2fb1fb7d`. All 11 challenger cells ran against that tree, in
one queue, on one host, each from an incumbent pinned by SHA. `run_dense.sh`'s four cells are paused after
`gemma4-12b` pending a decision on the oracle rule ([`DEVIATIONS`](ADVCHAL-V3-DEVIATIONS.md) §3.1).

Expectations are from [`EXPECTATIONS`](ADVCHAL-V3-EXPECTATIONS.md); v2 figures are that corpus's own results
table. Δ is per layer on the cell's dominant kind; µs/model is the cell's own `model_estimate` delta.

## 1. Every cell

| cell | v2 | expected | **measured** | µs/model | band | established? |
|---|---|---|---|---:|---:|---|
| gemma-4-26B `-onA` | −12.98 % *(sliding, 25 L)* | ~1.2 % ⚠ | **−12.10 % on `full_attention` (5 L); `sliding` +0.00 %** | −1198 | 334 | yes |
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

| kind | layers | v2 | v3 |
|---|---:|---|---|
| `sliding_attention` | **25** | **−12.98 %** | **+0.00 % — nothing** |
| `full_attention` | 5 | — | −12.10 % |

**v3 missed the kind v2 won**, on the kind carrying five times more layers, and improved the minority kind
instead. At model scope that is **−1,198 µs against v2's ≈−5,923 µs — 20 %.** "Reproduces" was the opposite of
the truth, and I produced it by comparing a v3 headline percentage against a v2 percentage without checking
they described the same layer kind — the error v2's own corpus warns about twice (*"state the scope on every
number"*, *"per-layer ranking picks the wrong candidate across kinds"*).

**Only `µs/model` is kind-weighted and therefore comparable.** On that basis, for the four cells where v2's
control and layer counts are both recorded:

| cell | v2 µs/model | v3 µs/model | v3 as % of v2 |
|---|---:|---:|---:|
| gemma-4-26B `-onA` | −5,923 | −1,198 | **20 %** |
| phi-3.5 `-onA` | −1,840 | −1,254 | 68 % |
| phi-3.5 `fuse-noadvise` | −1,268 | −278 | 22 % |
| phi-3.5 `nofuse-noadvise` | −1,449 | **0** | **0 %** |
| **those four** | **−10,480** | **−2,731** | **26 %** |

Against that, the three cells v2 scored 0.0 % give v3 **−2,701 µs/model**. So the honest summary is:
**v3 recovers about a quarter of what v2 claimed on the cells v2 won, and roughly the same amount again from
cells v2 could not see at all.** It is a more trustworthy 6.8 ms, not a larger one.

**And the ~1.2 % expectation was a category error**, independent of the scope mistake: the capacity formula
estimates a cell's *total addressable pool*, but the open question here was an *increment* on a win v2 had
already established. Applying a total-pool estimate to an increment question is meaningless — and the measured
−12.10 % **exceeds the 9.6 % pool I called an upper bound**, which falsifies the metric as I applied it,
because that pool was computed from v2 artefacts in which **64.7 %** of this cell's dominant kind was untraced.

For scale: v2 claimed "≈9.2 ms/model still on the table"; my revised expectation was "of order 1 ms". Both
were wrong, in opposite directions, and the revision was wrong by more —
[`DEVIATIONS`](ADVCHAL-V3-DEVIATIONS.md) §2.1.

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
