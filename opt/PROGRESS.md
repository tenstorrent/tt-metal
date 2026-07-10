# LTX 1080-high → 6s : live progress (follow this file)

**Goal:** 1080-high E2E 8.2s → 6.0s ("6s in 6s"). High quality (bf16, 8+3 steps). Autonomous grind, every lever.
**Worktree:** `.claude/worktrees/ltxperf-tip` · **opt branch:** `smarton/optimizer/ltx-sdpa-s2-2026-07-10`
**Device runs:** tt-device-mcp broker, owner `[claude]smarton`, timeout ≤300s (per-op/per-block only), profile off-reservation.
`tail -f opt/PROGRESS.md` to follow. Updated every step.

## Baseline (measured, gen#1 warm, with audio-trace fix)
| stage | high | note |
|---|---|---|
| S1 denoise | 2.9s | 8 steps @ 544×960 |
| upsample | 0.2s | |
| S2 denoise | 3.5s | 3 steps @ 1088×1920 (N=38760) |
| VAE decode | 1.0s | |
| audio | 0.5s | traced (shipped fix) |
| **TOTAL** | **8.2s** | denoise = 6.4s = 79% |

## SHIPPED (done, verified)
- Audio-trace default on origin/ltx-rt → 1080 high 8.5→8.2s, medium 5.5→5.1s. PUSHED.
- Reservation-hog fixed (tt-host/tt-profile skills + lessons.md).
- ltx-perf tip built + measured = 8.1s (parity; "7.9" was a one-off sample).

## KEY MEASURED FINDING (corrects the prior assumption)
Profiled the ring-joint SDPA at production S2 shape (unit test, PCC 0.9997, 71s<300s):
**SDPA = 4.85ms/op = ~20% of the 24.3ms/block.** It's 61% of MACs but 20% of time = compute-EFFICIENT.
⇒ The denoise is **distributed-overhead bound**, NOT one-kernel bound. No single-kernel opt reaches 6s.

## LEVER PLAN (ranked; status)
| # | lever | expected | status |
|---|---|---|---|
| A | audio trace default | −0.4s | ✅ SHIPPED |
| B | full-block per-op profile (rank ALL ops) | picks target | 🔄 recovering OFF-device from surviving 1.9GB raw log (PID 40785) |
| C | FFN weight bf8 (all_bf8_lofi) | — | ❌ NULL (measured −0.04s, PCC 0.876) — overhead-bound, not compute-bound |
| D | K/V all-gather CCL overlap (num_links/topology) if gather not hidden | ? | ⏳ after B |
| E | adaLN + RMSNorm epilogue FUSION (cut small-op dispatch + HBM round-trips) | ? | ⏳ after B — the only no-finetune lever hitting the real (overhead) bottleneck |
| F | SDPA chunk sweep + exp_ring_joint_sdpa variant (≤0.7s ceiling) | ≤0.7s | ⏳ low priority (SDPA=20%) |
| G | fewer transformer blocks (layer prune) — quality-gated | large | ❌ infra-dead this run (full-pipeline cold-compile >270s @4x8; needs prewarm wrapper). Quality hit is a foregone conclusion w/o finetune |
| H | sparse attn / QAD-bf4 / DMD fewer-steps — all need FINETUNE | large | ⏳ multi-day ML (the real 6s path per blog) |
| I | VAE conv3d micro-opt (1.0s) | ~0.1s | ⏳ |

## BLOG CROSS-CHECK — fal.ai "sub-second Ideogram V4" (2026-07-10), mapped to BH physics
Every technique they used, honestly scored against my MEASURED bottleneck (denoise = distributed-overhead bound, SDPA=20%):
- **CFG folding (their 2× lever)** → ✅ ALREADY DONE. LTX distilled runs `_denoise_no_guidance` (single conditional branch, no cond/uncond 2-pass). Captured, no more gain here.
- **FP4/weight quant (their 6× lever)** → ❌ NULL on BH. `all_bf8_lofi` (weight-only bf8) already measured −0.04s. FP4/quant shrinks *compute*; BH-Galaxy denoise is *distributed-overhead* bound (CCL+dispatch), so it doesn't move the wall. This is why NVIDIA's #1 lever does NOT transfer to a 32-chip mesh.
- **RMSNorm/adaLN/SwiGLU epilogue fusion (their #2/#3)** → ⭐ THE one transferable no-finetune lever — it attacks *overhead* (small-op dispatch + HBM round-trips), which IS my bottleneck. FFN is gelu_tanh (no SwiGLU gate to fuse), but adaLN modulation (`addcmul` shift/scale/gate around each attn/FFN) + RMSNorm are fuse candidates. Worth-it gate = lever B (how much time the small ops actually cost).
- **QAD (bf4+STE finetune) / DMD (fewer-step distill)** → the blog's actual quality-preserving unlock, and it CONFIRMS my honest verdict: sub-second without quality loss needs *retraining*, not a kernel micro-opt. Multi-day ML.

## HONEST 6s VERDICT (reinforced by the blog)
6s at high quality needs STRUCTURAL cuts that require FINETUNE (QAD-bf4, DMD fewer-steps, or sparse-attn+finetune) —
the blog proves the same. No-finetune levers left: op-FUSION (lever E, attacks overhead) + CCL overlap (lever D). Both
capped by the ~53% overhead structure. Banking every real gain; reporting the true floor with receipts. No faked numbers.

## LOOP (reboot-resilient, armed 2026-07-10 ~00:53Z)
`supervise.sh` daemon (pid 58706, nohup, interval 1200s, label ltx-opt) + Monitor on `opt/supervise.log`. Halts ONLY on
`STOP`/`DONE` at worktree root. Broker-native. Replaces the old session-only cron. `supervise.sh status --label ltx-opt` for a snapshot.

## LIVE LOG (newest last)
- 2026-07-10 00:13Z — SDPA baseline: 4.85ms/op (see finding above).
- 2026-07-10 00:24Z — full-block profile 001940-36 PASSED (129.9s); session teardown left ops report empty.
- 2026-07-10 00:47Z — RECOVERED: raw device log survives (1.9GB, 13M zone-rows). Re-aggregating OFF-device (`process_ops_logs.py --device-only`, PID 40785) — no reservation cost. Lever B data incoming.
- 2026-07-10 00:50Z — prune experiment (004013-1 ref / 004029-2 pruned) BOTH failed: full-pipeline cold-compile @4x8 exceeds the 270s pytest timeout. Lever G is infra-dead on raw broker; needs the 3-stage prewarm wrapper. Not the pruning's fault.
- 2026-07-10 00:52Z — pulled tt-workflows improvements (never-stop.md, tt-opt Opus-4.8 policy, tt-host cold-start floor); rebased my reservation-budget skill-fixes on top.
- 2026-07-10 00:53Z — BLOG cross-check (fal.ai) done: CFG already folded; FP4/quant null on BH (overhead-bound); op-fusion = the one transferable no-finetune lever; QAD/DMD confirm 6s needs finetune.
- 2026-07-10 00:54Z — reboot-resilient supervise.sh loop armed (pid 58706).
