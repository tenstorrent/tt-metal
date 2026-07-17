# Picker v3 — fallback improvement + M-scaling table (Mt≤8), and the next kernel optimization

Acts on the Mt≤8 re-baseline campaign (see `REGIME_A_MT8_CAMPAIGN_REPORT.md`). **Production change**,
committed after a strict no-regression gate. Commits `e270726f52c` (picker) / `7950fcae84b` (tooling).

## What changed (regime_a_matmul_config.cpp `auto_select_config`)

**1. Lookup table — 23 measured M-scaling winners (+3…+32%).** The M-scaling shapes use the cost-model
fallback, not the 20-shape production table; the campaign measured their true winners. Added them directly
(stability / exhaustive-expand / validate confirmed). Also updated the existing `{8,64,32}` (256×2048×1024)
entry nsb2→nsb4 (+5%). Highlights: 256×15360×1536 −24%, 256×2048×512 −20%, 32×15360×768 −19%,
256×2304×6144 −18%, 128×15360×1536 −17%.

**2. Fallback cost model — Sm>1 enumeration + reduction cost + narrow-N hysteresis.** The old fallback
enumerated Sm=1 only and had no reduction/forwarding term. New:
- Enumerate Sm>1 candidates.
- `pick_cost_v3` = deployed cost + split-K reduction penalty `rk·(Pk−1)·(Mblk·Nown)` (kRk=0.8).
- **Narrow-N-guarded hysteresis**: anchor on the deployed Sm=1 pick; switch to the best Sm>1 candidate
  only when (a) the shape is narrow-N (`Nband=cdiv(Nt,8) ≤ 2`, where N-split cannot supply parallelism so
  M-split is the lever) and (b) its reduction-aware cost beats the anchor by `margin=0.03`.
- Because the anchor is the unchanged deployed Sm=1 pick, the Sm=1 ranking is **byte-identical** and the
  fallback is **zero-regression by construction**; Sm>1 fires only for very-narrow-N (N≤512) unseen shapes.

**Why not a pure global cost model?** A global analytic model **cannot** capture the Sm>1 win condition
without regressions — its Sm>1 cost predictions are *anti-correlated* with reality (it fires on wide-N
losers like 128×2304×6144 −18% and misses the narrow-N winners). Trained on the campaign with a 1/3
validation holdout and a lexicographic (min-regressions → max-wins) objective, the only zero-regression
configuration is the narrow-N guard. The clean physical discriminator is **N-width**: narrow-N → M-split is
the only parallelism lever; wide-N → N-split (Ns) handles it and M-split only adds forwarding. Trainer:
`tools/mm_sweep/picker_v3.py`.

## Gated validation (all correctness/perf gates passed)

Full 60-shape corpus re-run through the rebuilt picker (`regime_a_campaign.py rerun`, fresh product path,
diffed vs the deployed baseline):

- **21 shapes improved >3%** (up to −24.3%), **0 regressions >3%**, **0 PCC < 0.999** (all ≥0.99999).
- Corpus total 4811→4650 µs (**−3.3%**; concentrated in the low-AI Mt≥4 shapes, −8 to −24%).
- Public suite **20/20** pass; **6/6** diag gtests (placement/ring/pipelined/progressive) pass.
- Pre-finalize validation: 256×15360×1536 nsb6 **+30.4%**, 256×2048×1024 nsb4 **+4.7%** (relaunch A/B, PCC
  exact); 256×2048×512 confirmed exhaustively over all 371 feasible configs.

## The next kernel optimization (re-ablation of the new winners)

Re-ran the causal ablations on the bottom shapes' **new** winning configs (`regime_a_campaign.py reablate`).
The picker fix **relieved the reduction bottleneck** (was the dominant cost under deep split-K); on the new
M-split/lower-Pk winners the reduction ablation is now minor:

| shape (new cfg) | full µs | skip in1 | skip in0 | skip fwd | no reduce | place_current | bank_ring |
|---|--|--|--|--|--|--|--|
| 256×2048×512 [1,4,3,2,2] | 17.2 | −14% | −3% | −18% | −9% | +5% | +15% |
| 128×2048×512 [1,4,2,2,2] | 11.7 | −26% | −6% | −10% | −10% | +12% | +5% |
| 256×2048×1536 [1,4,3,2,3] | 30.4 | −21% | −3% | −4% | −8% | +16% | +7% |
| 64×2048×512 [2,4,1,2,1] | 8.8 | −18% | −5% | −7% | −13% | — | +14% |
| 32×15360×768 [1,6,1,2,3] | 51.6 | **−70%** | −3% | −1% | −3% | — | +2% |
| 256×2304×6144 [4,3,1,1,3] | 88.1 | −10% | −9% | −8% | +1% | — | +17% |

**Conclusion — next lever is in1 delivery / read.** `skip_in1_read` is now the largest ablation everywhere
(−14 to −26% on the low-AI Mt shapes, −70% on deep-K 32×15360×768). Two sub-levers:
1. **in1 read-volume/bandwidth** for deep-K read-bound shapes (32×15360×768 is intrinsically read-bound;
   burst-size / dedicated-reader work applies — CB depth won't help a shape whose critical path *is* the read).
2. **Revisit CB1-depth for the M-split low-AI shapes.** In the pre-fix ablations CB1-depth was *not*
   warranted because reduction dominated; now that reduction is relieved and in1 delivery is exposed *with
   compute present* (128×2048×512, 256×2048×1536), a diagnostic CB1-depth (2/4/8 blocks) may hide the in1
   read behind compute+forward. Worth a compile-gated diagnostic-only experiment (not a production default).

Also confirmed high-value on the new winners: **in1_near placement** (`place_current` +5…+16% = reverting
costs that much) and **PARETO ring** (`bank_ring` +5…+17%) — keep both.

## Pinned M-split follow-ups — all CLOSED
The write→signal→flush vs write→flush→signal A/B, the bounded Sm>1 search for 256×2048×1024, and the
stale-comment + rejected-diagnostic cleanup are all done — see the "Follow-up experiment" section below and
commits `2106e6fd6ee` (source-lifetime fix) and `f392b62ee71` (bounded search + cleanup).

## Follow-up experiment — in1 delivery (acted on the "next optimization"), committed c0d5257e21e

Combined the pinned M-split forwarding follow-up with optimization #5 (in1 backpressure) as three
compile-gated diagnostics (`regime_a_diag_suite.py in1exp`), measured independently then in combination on
the new picker-winning configs. Primaries: 128×2048×512, 256×2048×512, 256×2048×1536, 256×2048×1024.
Controls: 32×15360×768 (bandwidth-bound negative control), two wide-N Sm=1/Sm=2, one deep-K in0-forward-heavy.

**Source-lifetime correction (review finding).** The first forward-signal-first implementation deferred the
flush to a single kernel-exit barrier. That was **unsafe**: the per-block `noc_async_writes_flushed()` does
more than order destination visibility — it guarantees the async write has *departed the source CB slot*
before that slot is pushed, wrapped (after 4 blocks) and overwritten by a later block. Same-NoC
write-before-signal protects the destination, not the source buffer, and an exit-only barrier is too late.
The corrected default keeps the flush **per-block but after the valid signal**: early signaling still
releases the slave without waiting on the reader's flush, while the flush restores source lifetime. Numbers
below are the corrected (safe) protocol vs the old flush-before-signal.

| lever | Sm>1 primaries | controls | verdict |
|---|---|---|---|
| forward-signal-first (write→signal→**per-block** flush) | **+1.0…+2.3%** | wide-N Sm2 −0.0%, Sm1 no-op | **adopted (safe)** |
| coalesced contiguous read | +0.3…+3.8% | all −0.1…+0.3% | **adopted** |
| CB1 depth 2 / 8 | neutral (−0.5…+0.8%) | +1.3…+1.5% (slight hurt) | **rejected** (keep depth 4) |

All PCC-exact (public 20/20 random-operand + 6 diag gtests pass; **10× relaunch stress on the highest-
forwarding shape 256×2048×512 Sm3 → every run max_rel_err=0**; watcher-clean). The bandwidth-bound negative
control (32×15360×768) shows ~0 across every lever — confirming the wins are genuine in1-delivery gains, not
noise, and that the deep-K read-*volume*-bound shape is (as predicted) not helped by buffering or
flush-ordering. **CB1 depth is decisively not a lever** — in1 backpressure is not the bottleneck; the wins
come from moving the per-block flush off the slave-release critical path and from coalescing the
physically-contiguous block read. The unsafe exit-barrier variant measured a larger ~4% fwd gain; the honest
per-block-flush cost brings that down to the ~1–2% above. Old behaviour retained as A/B diagnostics
(`DIAG_FWD_FLUSH_FIRST`, `DIAG_NO_COALESCE`); the rejected CB1-depth diagnostic (`DIAG_CB1_D2/D8`) was
**removed** from the enum/factory/gtest/diag-suite after the experiment (commit `f392b62ee71`). The bounded
Sm>1 search for 256×2048×1024 on the final kernel confirmed the current `[1,4,2,2,4]` is best (0.0%;
runner-up `[1,4,3,2,4]` +0.3%, tied) — no picker change. All three pinned M-split follow-ups are now closed.

## Definitive post-change wide sweep (full Mt≤8 corpus, config=None + kernel A/B)

`regime_a_campaign.py perfstatus`, fresh cache, resident-input/8-iter methodology, 4 interleaved relaunches.
Two measurements per shape: (1) **config=None** — Picker v3 selects, mask 0 = final signal-first + coalesced
kernel (production path); (2) **kernel A/B** on the same picker-selected config — mask 0 (new) vs
`DIAG_FWD_FLUSH_FIRST | DIAG_NO_COALESCE` (old in1).

- **All 60 shapes OK, 0 PCC<0.999.** Final auto-picker **median 86% %512**.
- **Kernel improvement (new vs old in1, picker config held fixed): 0 regressions on any shape.** Sm=1 shapes
  are a no-op (fwd is Sm>1-only; median ~0 across the full 60). The gain is concentrated on the **11 Sm>1
  shapes: median +1.3%, best +6.1%** — 256×2048×512 +6.1%, 256×15360×768 +3.7%, 128×2048×512 +3.5%,
  256×2048×1024 +3.5%, …, down to +0.1% on the big wide-N Sm2 shapes (where forwarding is a negligible
  fraction). Corpus-sum +0.2% (dominated by the Sm=1 majority + large wide-N shapes).
- **Total vs the original pre-picker-v3 baseline (picker + kernel combined): corpus-sum −3.6%**, concentrated
  in the low-AI Mt≥4 shapes (−8…−24% each; the big wide-N shapes were already near-optimal). Raw:
  `regime_a_perfstatus.json`.

Net: the picker v3 changes deliver the large per-shape wins (up to −24% on low-AI Mt≥4), and the in1 kernel
change adds a further +1–6% on the Sm>1 shapes, all regression-free and PCC-exact.
