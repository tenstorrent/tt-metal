# t103: E2E time budget for t48, LTX-2.5 1080p/145f (6 s video), target < 7 s

Off-device analysis, no device jobs. Branch base: ttp/t48-ltx25-integrated @ 83c11ee2b34.

## Bottom line

- The only measured 4x8 e2e for LTX-2.5 is job 879 (t20 code, blx03, 2026-09-30): **8.76 s** warm gen #1.
- Two fixes on t48 remove most of the 1.76 s gap: the Gemma encode trace is captured at warmup
  (#40: 1.91 -> 0.23 s), and the mp4 encode is faster and overlapped (#13/#18: 0.80 -> ~0.15-0.30 s).
  With the VAE and V2A changes, **t48 projects to 6.1-6.6 s (mid ~6.3 s). That is under 7 s, but
  not yet measured on 4x8.** 4x8 runs are barred, so this is the biggest uncertainty.
- Cross-check: LTX-2.3 on 4x8 with the same export and encode-trace fixes measured 6.23 s steady
  (job 587, g15blx02), with a slower VAE (0.68 s) and S1+S2 about 0.1 s slower than 2.5. That is consistent.
- Denoise (S1 + S2) is ~4.75 s, about 75% of the projected e2e. The VAE is ~0.55 s (~9%).
- VAE tasks #97-#100 (~85-160 ms together) are 1.3-2.5% of e2e. They are not needed to get under
  7 s if the projection holds. They only matter if 4x8 lands above ~6.85 s. The denoise folds
  below are worth ~2x more.

## Per-stage table (4x8, 1080p/145f, warm, traced, new prompt)

| Stage | Job 879 gen #1 (s) | t48 projected (s) | Change | Source | Confidence |
|---|---|---|---|---|---|
| Gemma encode (new prompt) | 1.91 | 0.23-0.26 | -1.66 | 879: one-time trace capture landed on gen #1. #40 job 028 (2x4 TP=4): 0.226-0.237 s after the fix. 879 gen #0 eager TP=8: 0.26 s | Medium-high (fix verified on 2x4 only) |
| S1 denoise (8 steps) | 2.28 | 2.26-2.28 | ~0 | 879: 8 x ~276 ms steps + 57 ms init + ~16 ms loop. #85: V2A skip is noise on S1 | High |
| Latent upsample | 0.15 | 0.15 | 0 | 879 | High |
| S2 denoise (3 steps) | 2.50 | 2.46-2.50 | <= -0.04 | 879: 3 x ~823 ms + 9 ms init + ~20 ms. #85 (2x4): V2A skip -0.48%/block, <= 42 ms | High |
| Conv VAE decode + YUV + readback | 0.73 | 0.50-0.60 | -0.13 to -0.23 | #96 job 435 (2x4, same per-chip shard, blocking 4,8): decode 445 ms, 517 ms with upload + output. 4x8 reads back 4x the bytes (454 MB); that cost on 4x8 is unmeasured | Low-medium |
| Audio decode | 0.37 | 0.37 | 0 | 879 (4x8: mel 31 + vocoder/BWE 337 ms). #83's 553 ms is a 2x4 measurement and is not used here | High |
| mp4 export (tail after audio) | 0.80 | 0.15-0.30 | -0.50 to -0.65 | #13 job 587 (2.3, 4x8): 0.154 s steady. #27 jobs 922/923 (blx03): ~0.3 s. t48 has both #13 and #18 | Medium |
| Host gaps between stages | ~0.03 | ~0.03 | 0 | 879 timestamps: 12 ms after export, ~11 ms latent stats. In-stage host time (inits, loop) is already inside S1/S2 above, ~0.10 s | High |
| **E2E wall** | **8.76** | **6.1-6.6 (mid 6.3)** | **-2.2 to -2.6** | Sum of rows | Medium |

Note: blx03 ran ~0.14 s slower than g15blx02 on the same code (#27 jobs 922/923: 6.80 s vs job 610: 6.66 s).
The goal is set on g15blx02.

## What each stage must give up to get under 7 s

From the measured 8.76 s, we need >= 1.77 s:

| Stage | Must give up | Status |
|---|---|---|
| Gemma encode | 1.66 s | On t48 (#40), verified on 2x4 |
| mp4 export | 0.50-0.65 s | On t48 (#13 + #18), verified on 4x8 for 2.3 |
| VAE decode | 0 (it gives 0.13-0.23 s anyway) | On t48. #97 exact-shard (-44 to -52 ms) is opt-in |
| S1 / S2 denoise | 0 | V2A skip is on t48. Gate + norm-AdaLN folds (~-0.3 s) are opt-in |
| Audio, upsample, gaps | 0 | none |

If the 4x8 run comes in at the high end (6.6 s plus ~0.15 s blx03 noise, so ~6.75 s), we still pass
with ~0.25 s to spare. If the encode or export fix does not hold on 4x8, denoise cannot cover it:
1.7 s is 36% of denoise. So the first thing to do is the t48 4x8 e2e (t104 smoke, ready) once 4x8 is
allowed, not more VAE work.

## Top 3 denoise / host-gap cuts that a short 2x4 job can measure

All three open the full mesh, then call create_submesh(2,4), using the #85 block harness
(tmp/t85/test_denoise_trims_ab.py, Linear, sp axis 1 / tp axis 0, real block-0 weights).

1. **Gate fold forward (LTX_FUSE_GATE_ON_DEVICE=1), stacked with LTX_FUSE_NORM_ADALN=1.** Expected
   -0.2 to -0.3 s e2e. Gate: block -3.7% S1 / -4.9% S2 (t51, ~-0.20 s). Norm-AdaLN: -1.9%/block on
   2x4 (#70 job 037, ~-0.10 s). #70 showed the on-device fold gives the same bits as the fused
   cache. The fused forward has never run on 2x4, because the #70 test used Ring. One job with
   arms base / gate / adaln / both: ms/block S1+S2 and block PCC (>= 0.99999). Not bit-exact, so a
   5-seed eval follows. This is the biggest cut that is ready to measure.
2. **S2 ring-joint SDPA chunk sweep.** SDPA is ~27% of denoise (t14 profile: 1.30 of 4.79 s;
   ring-joint 25 points). S2 is 2.47 s of step time. A q/k chunk sweep of the S2 block's joint
   attention on 2x4 is a few short arms with the same math. Expected -50 to -150 ms. Low
   confidence: the 2x4 ring has 4 devices, not 8, and per-chip Q length differs from 4x8 (9728 vs 4864).
   Confirm the winner on 4x8 later.
3. **Denoise host overhead.** In 879, S1 init is 57 ms, of which prompt staging is 51 ms (29 ms on gen #0).
   The step loops add ~16-20 ms per stage, plus the S1->S2 handoff (audio latent read to host for
   stats, upsample). Total ~0.10-0.13 s, of which ~50-80 ms looks removable (stage both prompts from
   the encode trace output, skip host stats in serving). Measure with LTX_TIME_STAGES on the
   pipeline at 544x960/145f on a 2x4 submesh (host work is the same as 4x8). Expected -50 to -80 ms.
   Medium-low confidence.

Not measurable on 2x4: AGMM K=2048 (only at TP=4 Ring on 4x8), and S1 CCL overhead. S1 on 4x8 runs
5.8 ms/block, against 14.0 ms on 2x4 with 4x the per-chip work, so it is latency-bound, not compute-bound.

## Sources

- Job 879 log: tt-project/baselines/t20/run.log, lines 587-680 (gen #1 stage table, STEP_MS, timestamps).
- #40 Gemma: run 245 result (job 028). #85: branch ttp/t85-small-exact-denoise-trims-v2a-pad-mask-m NOTES.md (job 161).
- #96: tt-project/worktrees/t96/NOTES.md (job 435). #83: run 309 result. #13: run 58 (job 587). #18: run 80 (job 610).
- #27: run 174 (jobs 922/923). #70: run 270 (job 037). t14 SDPA share: run 95. VAE plan: t48 tt-project/t93/PLAN.md.
