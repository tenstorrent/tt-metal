# advchal-v3 — start here

15 documents and a long correction history. This is the entry point: **what the run found, what is settled, what
is still open, and which claims have been retracted.** v2 had the same two-tier structure and it should not have
taken until now to write v3's.

## The result in four lines

| | |
|---|---|
| **v3 vs v2, same 11 cells** | **−6,769 µs/model against −15,177 — 45 %** ([`RESULTS`](ADVCHAL-V3-RESULTS.md)) |
| **why** | **v3's defect was detected and v2's was not, and v3 could not attribute what it detected — so it discarded the win instead of fixing the bug.** Both versions ship the same cross-phase defect; only v2's fixture hid it ([`WHY-WORSE`](ADVCHAL-V3-WHY-WORSE.md), which supersedes part of [`CORE-ISSUE`](ADVCHAL-V3-CORE-ISSUE.md)) |
| **how much is explained** | **87 % of the 11,303 µs lost (64 % with a measured fix, 23 % explained without one). 13 % is still partial** — remaining holes are phiFN 989 µs, phiA 340 µs, g26B 148 µs ([`WHY-WORSE`](ADVCHAL-V3-WHY-WORSE.md) §7c) |
| **the second systematic defect** | **the absolute oracle's clause 2** — reject if worse than the incumbent — now has **three measured instances** costing −0.90 %/layer on phiA and −137.5 µs/model on nmFN, every candidate clearing the model's own bar by 30–60× the margin it misses the incumbent by. **It is a rule I wrote** ([`WHY-WORSE`](ADVCHAL-V3-WHY-WORSE.md) §7a) |
| **the agent's actual error** | it read a diagnostic as a verdict — a placement change that preserves arithmetic to 10⁻⁶ cannot cost 5 × 10⁻³ at layer scope, and that 1000× gap was the signal ([`WHAT-THE-AGENT-GOT-WRONG`](ADVCHAL-V3-WHAT-THE-AGENT-GOT-WRONG.md)) |
| **recoverable now** | **−6,545 µs** from two measured fixes, i.e. 78 % of the shortfall — neither is a search or judgement improvement, both are code defects |

## Reading order

1. [`WHAT-THE-AGENT-GOT-WRONG`](ADVCHAL-V3-WHAT-THE-AGENT-GOT-WRONG.md) — the reasoning error and the five checks that would have caught it. **Start here if you read one file.**
2. [`RESULTS`](ADVCHAL-V3-RESULTS.md) — v3 against v2, cell by cell, in kind-weighted µs/model.
3. [`CORE-ISSUE`](ADVCHAL-V3-CORE-ISSUE.md) — the systemic account, and the audit across all three models.
4. [`GUARD-FINDING`](ADVCHAL-V3-GUARD-FINDING.md) — the full causal chain, measured, including the path decomposition and the two fixes.
5. [`ANALYST-PITFALLS`](ADVCHAL-V3-ANALYST-PITFALLS.md) — 19 errors, 11 from building v3 and 8 from analysing it. Read it to calibrate everything above.
6. Supporting: [`OP-BY-OP-VS-V2`](ADVCHAL-V3-OP-BY-OP-VS-V2.md), [`NORM-GRID-SWEEP`](ADVCHAL-V3-NORM-GRID-SWEEP.md), [`PCC-DROP-ISOLATION`](ADVCHAL-V3-PCC-DROP-ISOLATION.md), [`REMEASURE`](ADVCHAL-V3-REMEASURE.md), [`PCC-BY-GRID`](ADVCHAL-V3-PCC-BY-GRID.md), [`DEVIATIONS`](ADVCHAL-V3-DEVIATIONS.md), [`EXPECTATIONS`](ADVCHAL-V3-EXPECTATIONS.md), [`STEP0`](ADVCHAL-V3-STEP0.md), [`CHANGES`](ADVCHAL-V3-CHANGES.md), [`RUN-LOG`](RUN-LOG.md).

## What is settled, by measurement

| finding | evidence |
|---|---|
| the norm op is accurate at every grid; sharded vs interleaved differ by 1 bf16 ULP in 18–36 % of channels | 79 synthetic configs + the real captured activation, all within 1.4 × 10⁻⁶ of float64 |
| gemma sliding's 5.2 × 10⁻³ is a **top-8-of-128 routing flip** (expert 104 → 123): Δlogit 0.046875 against a 0.015625 gap | routing dumped in three configurations |
| the routing path is **100 %** of it; the attention path **0.6 %** | pin-the-routing decomposition |
| it is a **phase mismatch**, symmetric: prefill-only fails, decode-only fails, both passes | `NORM_PHASE` sweep |
| **one of eight** norm sites carries all of it — site 1, `input_ln` → QKV → KV cache write | 8 single-site ablations against the oracle |
| the fix: **88 cores + `drop_index=1`** → PCC 0.9996227, **−11.47 %/layer, −5,260 µs/model**, holds at prefill 8/32/64 | oracle + `repeated_perf` |
| v2's guard is **not** the fix — it stops firing above 32 prefill rows | seq 64: `phase=both` == `phase=decode` to 16 digits |
| the oracle is exactly deterministic — 16 digits across a week and a different device | incumbent re-measure |
| phi is **dense** and its drop is a **third cause** — v3's rope code. v2's implementation is free (−5.2 %, bit-identical PCC) and porting it recovers **−1,285 µs** | phi oracle, both trees, plus a transplant |
| north-mini's win **stands** — its divergence *shrinks* with cache depth | prefill sweep, knob genuinely toggled |
| **decode-only gating is not the defect** — a decode-only knob is unsafe **iff its output flows into the KV cache write**. Classifies all 7 audited knobs correctly | qwen ships a decode-only MLP knob, **bit-identical PCC on both kinds** with a real −1.0 % |
| `full_attention`'s 740× lower sensitivity is **boundary luck, not structural** — same 0.015625 gap, perturbation also exceeds it, but the crossing permutes two *already-selected* experts instead of changing membership | router dumped for both kinds |
| phi's defect is a **coupled pair** that cannot be split — interleaved RoPE arithmetic *requires* the value conversion; v2's sharded form is bit-identical and 5.2 % faster | 4-variant bisect |
| **no `ladder_88` artefact exists at any point in either branch's history**, while the README and `legal_ladder` both claim it ran | full history search |
| `gemma-4-12B exp11` shipped **−11.5 µs on a 62 ms model (0.018 %)** with `oracle_pcc: None` — a *shipped non-result* | its own `final.json` |

## What is still open — the honest register

**Not tested at all**

| # | item | why it matters |
|---|---|---|
| ~~1~~ | ~~qwen3.6 never audited~~ — **done, clean**: decode-only but MLP-only, bit-identical PCC both kinds | closed |
| 2 | the dense cells — `exp11` **done (−11.5 µs)**; phi `exp17` **in flight**; llama-3.1-8B `exp17` (**the control**) and llama-3.2-1B `exp17` queued behind a re-placed sentinel | predictions pre-registered in [`DENSE-PREREG`](ADVCHAL-V3-DENSE-PREREG.md) |
| 3 | phi `-onA` (−1,254 µs) and north-mini `fuse-noadvise` (−351 µs) audited structurally, never measured | same pattern class |
| 4 | **the two fixes have not been re-run through their cells** | staged, then paused for the frame question — see below |

**Diagnosed, not resolved**

| # | item |
|---|---|
| ~~5~~ | ~~full_attention's logit gap never dumped~~ — **done**: same gap, perturbation also exceeds it, crossing is *inside* the selected set. Boundary luck, so its −1,198 µs is not structurally safe |
| 6 | why the gemma perturbation **grows** with prefill length (measured; my prediction had the sign backwards) |
| ~~7~~ | ~~phi's defect not isolated~~ — **bisected**: a coupled pair (interleaved arithmetic + value conversion) that cannot be split; the structure is the cause, not any line |
| 8 | my phi re-run reads **0.9849539** where the cell recorded **0.9173130**; both fail, neither explains the other |
| ~~9~~ | ~~ladder_88 contradiction~~ — **resolved as far as artefacts allow**: zero `ladder_88` files at any commit in either branch's history |
| 10 | **P3**: v2 and v3 report 0.99931 vs 0.99469 for the same g26B pair; unresolved since the run |
| 11 | whether `drop_index=1` holds at production prefill (1024). Tested to 64; beyond that the fixture leaves its supported regime |
| 12 | gemma-4-26B real weights are **absent from this host** (28 KB, config only) while every gemma cell reports `oracle_weights: real`; layer-0 tensors are range-fetchable, so the string is unverified rather than false |
| 12b | **llama-3.1-8B's weights were a 20 KB config stub and llama-3.2-1B's were absent entirely** until fetched at 11:05 on 2026-08-11, yet v2's `exp17` cell records `oracle_weights: real`. Its `oracle_scope` says *"unchanged frozen incumbent"*, i.e. it cited the incumbent's existing artefact rather than running one — defensible for a cell that shipped nothing, but the `real` string is still unverified |

**Stage changes designed, not implemented**

| # | item |
|---|---|
| 13 | **gate the op on the op** — the change that would have caught this cell, phi, north-mini and any future instance. This is why the re-run is paused: relaunching against the current gate reproduces the blind spot |
| 14 | `oracle_passed` computed from a parsed, provenanced artefact — not a literal, not a directory name, not an empty `exact_command` |
| 15 | every claimed measurement must have a file in `measurements/` |
| 16 | every oracle runs prefill → decode at **≥ 2 prefill lengths**; north-mini's runs none |
| 17 | a placement knob that changes a K/V producer must declare its cross-phase behaviour, asserted by the gate |
| 18 | a shipped winner must be expressible — and **disableable** — through the model's policy surface. north-mini's is hardcoded in its constructor |

## Retracted claims — read these before quoting anything

| claim | status |
|---|---|
| "candidate tt-metal bug: `rms_norm` non-monotonic in core count, three independent reproductions" | **retracted** — one policy measured three times; no kernel bug |
| "the cell tried one rung and stopped searching" | **retracted** — the ladder was fully swept; the *verdict* was extrapolated |
| "v2's oracle ran with the sharding inactive; strike its −5,919 µs win" | **retracted** — v2's 88-core oracle reproduces exactly, sharding engaged |
| "every grid is bit-identical" | **corrected** — true for synthetic inputs, false on the real activation |
| "the untested rungs may well pass" | **falsified** — all fail in that tree |
| "phi's 0.917 is one line — the key's memory config" | **retracted** — patched, no-op; it is the combination |
| "north-mini's win is untestable by its own gate and at risk" | **overstated** — its divergence shrinks with depth; the win stands |
| "v3 = 73 % / 96 % of v2 with wins struck" | **withdrawn** — 45 % stands |
| corpus expectation "of order 1 ms" | **refuted before the run** by 12.0 ms already banked |

**Nine retractions.** That count is the most decision-relevant number in this corpus, and
[`ANALYST-PITFALLS`](ADVCHAL-V3-ANALYST-PITFALLS.md) is where each one is accounted for.
