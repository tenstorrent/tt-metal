# advchal-v3 — results: 11 cells, expected against measured

Stage frozen at `advchal-v3/stage-frozen` = `4ea2fb1fb7d`. All 11 challenger cells ran against that tree, in
one queue, on one host, each from an incumbent pinned by SHA. `run_dense.sh`'s four cells are paused after
`gemma4-12b` pending a decision on the oracle rule ([`DEVIATIONS`](ADVCHAL-V3-DEVIATIONS.md) §3.1).

Expectations are from [`EXPECTATIONS`](ADVCHAL-V3-EXPECTATIONS.md); v2 figures are that corpus's own results
table. Δ is per layer on the cell's dominant kind; µs/model is the cell's own `model_estimate` delta.

## 1. Every cell

| cell | v2 | expected | **measured** | µs/model | band | established? |
|---|---|---|---|---:|---:|---|
| gemma-4-26B `-onA` | −12.98 % | ~1.2 % | **−11.91 %** | −1198 | 334 | yes |
| phi-3.5 `-onA` (phiA) | −8.75 % | ~1.5 % | **−5.97 %** | −1254 | 30 | yes |
| north-mini `-onA` | **0.0 %** | ~0.5 % | **−3.94 %** | −1400 | 47 | yes |
| gemma-4-26B `fuse-noadvise` | −2.04 % | unknown | **−2.11 %** | −986 | 64 | yes |
| north-mini `fuse-noadvise` | −10.23 % *(void)* | −1.76 % *(measured in shakedown)* | **−1.69 %** | −351 | 42 | yes |
| phi-3.5 `fuse-noadvise` | −4.91 % | ~2.8 % | **−1.08 %** | −278 | 31 | yes |
| qwen3.6 `nofuse-noadvise` | **0.0 %** | ~0.02 % | **−1.01 %** | −1130 | 372 | yes |
| north-mini `nofuse-noadvise` | **0.0 %** | ~0.1 % | **−0.59 %** | −171 | 116 | yes |
| gemma-4-26B `nofuse-noadvise` | −0.34 % | uncalibrated | **`measured_zero`** | 0 | 26 | — |
| phi-3.5 `nofuse-noadvise` (phiB) | −5.74 % | ~1.3 % | **`no_change`** | 0 | 12 | — |
| qwen3.6 `fuse-noadvise` | inside band | ~0.4 % | **`no_change`** | 0 | 385 | — |

**8 of 11 shipped a change, and all 8 are outside their own uncertainty band.** Corpus total
**−6,769.6 µs/model ≈ 6.8 ms**.

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
subset, and the largest gain in the run (g26onA, −11.91 %) is the cell that applied the advised plan and
measured it. The bound is real and partial, not absolute.

## 5. Provenance, uniform across all 11

`tracer_matches_checkout=True`, `optimizer_files_changed_since_pin=[]`, advisor at `97724a1170` against the
pin, `device_users=[0,0]` on every measurement, `.agents` byte-identical to the frozen tree on every cell, no
blob shared with any v2 run, all 156 parked refs still unnamed at the end of every cell. One publish failure,
no measurement lost — [`RUN-LOG`](RUN-LOG.md) P6.
