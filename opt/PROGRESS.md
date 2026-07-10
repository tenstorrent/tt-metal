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

## KEY MEASURED FINDING — per-op S2 block profile (lever B) — ⚠️ CORRECTED (was cold, re-measuring warm)
**METHODOLOGY ERROR I made, caught by adversarial worker a9dc1ceb + CSV-verified:** the recovered profile (job
001940-36) is a **COLD, non-traced, num_layers=1 capture** — PROGRAM CACHE HIT=False on all 302 ops, METAL TRACE ID null.
My first read divided the single cold block by 2 (mis-assumed 2 iterations; it's ONE AV block = 2 RingJointSDPA video+audio
self + 4 SDPA cross). And the "14% Tilize" is a **one-time load preamble** (seq 0–106 = weights RM→TILE at load); the real
block compute (seq 107–301) has **ZERO tilizes**. ⇒ **Tilize lever is DEAD (artifact).**

CORRECTED per-block (compute phase seq 107–301, cold FW — warm may differ, esp. CCL fabric-setup): **20.56 ms/block**
| bucket | ms/block | % | note |
|---|---|---|---|
| **CCL-collective matmul (SP)** (AllGatherMatmul + MatmulReduceScatter) | **11.57** | **56.3%** | matmul compute FUSED with all-gather/reduce-scatter |
| **RingJoint SDPA** | 6.13 | 29.8% | 4.62ms video-self + 1.30ms audio-self; capped |
| adaLN elementwise (addcmul shift/scale/gate) | 1.79 | 8.7% | small ceiling |
| reshape/heads + RoPE + non-collective matmul | 1.06 | 5.2% | |

**Wall reconciliation (CORRECTED):** 144 S2 forwards × 20.56ms = **2.96s device-FW** vs S2 wall 3.5s ⇒ **~15% inter-op gap,
NOT 50%.** The denoise is **compute/CCL-bound, not overhead-bound** — my earlier "50% overhead" was the ÷2 error.
⇒ 86% of the block is collective-matmul + SDPA, both near pure-compute floor (bf8 null). No-finetune runway is thin.
**Open (needs WARM/TRACED profile):** is cold CCL FW inflated by fabric setup? Real warm per-block + real gap %. Re-measuring.

## ⭐ MASTER PLAN + HONEST FLOOR (Workflow wi7rr4cw5 synthesis, 5 Opus/xhigh digs — full text: opt/MASTER_PLAN_raw.txt)
**No-finetune levers CANNOT reach 6.0s — they floor at ~7.2–7.5s.** Only step-reduction distillation clears 6.0s, and
100% of that training is OFF-DEVICE / OUT-OF-REPO (this repo is forward-only ttnn: zero autograd/optimizer/DMD infra).
Path to 6.0s (verified in-repo): **6+2 step-distilled checkpoint (−1.89s) + VAE-decode trace (−0.35s) ≈ 5.96s.**

Verified linchpins: VAE video decode is the ONE stage left untraced (pipeline_ltx.py:1180) → **~0.35s pre-built win**
(kevinmi branch np-halo-fabric-mux, decode_device() split, 435ms device measured). Step-cut = env-var flip
(LTX_S1_SIGMAS/LTX_S2_SIGMAS, pipeline_ltx_distilled.py:39-51) but the CURRENT checkpoint fails few-step (7-step min-PCC
0.36) → needs a NEW checkpoint. num_links BH=2 vs WH=4 (pipeline_ltx.py:537-542) = 1-kwarg A/B, ~0.1-0.2s uncertain.
**B1a HF check (2026-07-10): MISS** — HF has only the same 8+3-step distilled family (distilled / distilled-1.1 /
distilled-lora-384); NO published fewer-step/turbo checkpoint. So no 1-day drop-in; 6.0s needs the ~1-3wk DMD2 train.

REFUTED (this session's earlier claims): "50% inter-op gap recoverable via CCL overlap" — every TP/SP collective is
ALREADY fused into its consumer (all_gather_minimal_matmul_async / matmul_reduce_scatter / ring_joint_sdpa, ~55% fabric
util, no barrier). CCL-overlap banks ~0.15-0.35s, not 1.75s. QAT/quant DEAD (overhead-bound). Tilize DEAD (cold artifact).

### No-finetune bank (autonomous, PCC-gated) — ⚠️ VAE trace DIED on measurement → floor is ~7.9s, not 7.5s:
| # | lever | predicted | MEASURED | status |
|---|---|---|---|---|
| A1 | VAE decode trace (port kevinmi decode_device) | ~0.35s | **0.19ms (DEAD)** | ✅ verified: PCC=1.0 but decode is device-bound (async CQ already hides dispatch); untraced 552.08ms = traced 551.88ms. Refactor kept on branch (65a3a1c), NOT worth shipping. |
| A2 | num_links 2→4 + fabric sweep | 0.1-0.2s (uncertain) | ? | ⏳ device A/B (watched, hang risk) — the one untested >0.1s lever |
| A7 | adaLN to_out fusion (a2v/v2a/attn2) | ~0.008s | ? | ⏳ trivial python-only |
| A3/A5 | RMSNorm stat-gather merge + scale/shift fold | 0.07-0.17s | ? | ⏳ kernel change, needs warm-gap attribution first |

**CORRECTED no-finetune floor: ~7.8–8.0s.** The VAE trace (biggest predicted no-finetune win, 0.35s) is measured-DEAD;
the VAE stage is device-bound (552ms decode + ~400ms host I/O, neither trace-addressable). Remaining levers (num_links
+ adaLN + RMSNorm-merge) sum to a *predicted* ~0.2–0.4s and are unproven. **The only real win this whole effort was the
shipped audio-trace fix (8.5→8.2s).** Every predicted kernel lever (VAE-trace, CCL-overlap, tilize, quant) is measured
DEAD or already-done. **6.0s is 100% gated on out-of-repo step-distillation — a training-investment decision, not a
kernel grind.** Continuing to chase sub-0.4s levers against a 2.2s gap is polish, not progress.

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
- 2026-07-10 00:58Z — LEVER B COMPLETE: definitive per-op S2 block profile (see table above). Named targets: CCL-matmul 47.5%, SDPA 25%, Tilize 14%, adaLN 7%. Inter-op gap ~50% of wall.
- 2026-07-10 01:00Z — dispatched worker a9dc1ceb (Opus/xhigh): investigate Tilize elimination (44 ops/block) → ranked removability plan. In flight.
- 2026-07-10 01:16Z — TTSUP_TICK #1 (heartbeat healthy). Ultracode on. Launched Workflow wi7rr4cw5 (4 parallel Opus/xhigh digs + max synthesis): CCL-overlap/inter-op-gap (highest ceiling ~1.75s S2), adaLN-fusion, VAE-decode (1.0s unexplored stage), finetune-path scoping (DMD/QAD honest 6s route). All read-only. In flight alongside tilize worker.
- 2026-07-10 01:30Z — Workflow DONE. Master plan synthesized (opt/MASTER_PLAN_raw.txt). Honest floor delivered: no-finetune ~7.2-7.5s, 6.0s needs out-of-repo DMD2 step-distill. CCL-overlap refuted (already fused). B1a HF check MISS. Committed becc22a.
- 2026-07-10 01:40Z — dispatched VAE-trace port worker a333a87b (kevinmi np-halo-fabric-mux → decode_device trace, ~0.35s, patch-only).
- 2026-07-10 01:45Z — A0 warm profile 014542-5 PASSED (145s, LTX_PROFILE_ITERS=4). Retires the cold-capture error: warm rows now exist. Processing 42.8M-row raw log off-device (PID 106265) → warm per-block FW + warm inter-op gap (dispatch vs sync).
- 2026-07-10 01:56Z — TTSUP_TICK #3. Device free; NOT firing num_links (fabric-hang risk on shared broker) fire-and-forget — will test it watched. VAE verification is the next device job.
