# t166 — cut the LTX-2.5 4x8 fresh-process warmup under 400 s

Branch ttp/t166-cut-ltx-2-5-4x8-fresh-process-warmup-und (from origin/ttp/t48-ltx25-integrated b81bb403d86;
the NOTES.md it carried was t140's). Code commit: 0ee3d31bda9 `ltx: add opt-in LTX_WARMUP_T2V_ONLY ...`
(host tests: test_ic_trace_generate.py 22 passed).

## Before (cold JIT cache): where the warmup goes
blx01 job 621 (bf7db12a149, cold cache 0/3653 hits): process 660 s, warmup 551 s, gen#0 36.5 s, gen#1 6.031 s.
setup ~66 s (183 compiles) | gemma encode 79 s (501) | image encoder 63 s (288) | s1 95 s (820) | s1_i2v 12 s (73)
| upsample 13 s (93) | s2 41 s (245) | s2_i2v+transition+decode 26 s (129) | audio eager 211 s (1303 compiles,
622 s compile CPU) | audio capture 10 s (18).
blx03 t138: process 530 s, warmup 429 s, 985/3554 hits.
=> most of the warmup is in-window cold JIT compile; a warm cache on an unchanged tree removes it.

## After: blx01 job 625 (submitted 2026-10-06 21:09 UTC, -t 600)
Script /var/tmp/fasth3/t166/job.sh (TAG=a), overlay /var/tmp/fasth3/t166/tree (hardlinked t48 models/ + t166
python files; TT_METAL_HOME stays /var/tmp/fasth3/t48 so the job-621 JIT cache is reused).
Env: LTX_WARMUP_T2V_ONLY=1 LTX_WARMUP_ENCODERS=0 LTX_E2E_SEEDS=0,1,2,3,4 + t159 env, TT_METAL_LOG_KERNEL_COMPILE=1.
Output/log: g15blx01:/var/tmp/fasth3/t166/out/a/{run.log,*.mp4}.

## Next
1. When 625 ends: pull run.log; section times + JIT stats; process wall; E2E_WALL_S per gen.
2. Bit-identical: md5 of decoded frames (ffmpeg -f framemd5) of out/a/ltx_av_fast_1920x1088_{0,1}.mp4 vs
   /var/tmp/fasth3/t159/out/ same names.
3. Land 0ee3d31bda9 on t48 (cherry-pick onto <branch>-land from origin/t48, ttp push --detach).
4. result.json: SHA, env, wall, -t recommendations (measured +50%, <= 600).

## Drop 1 (light wake 21:13 UTC)
Job 625 killed by broker device recovery at ~21:11 UTC on g15blx01 (chips 16-23 left the PCIe bus, tray 3; Runtime 94 s, exit -9, our job,
during warmup stage 1). Log ended 21:10:51. Not re-queued by the broker. Next: when blx01 health check passes,
resubmit the same job.sh (TAG=a); if it drops again, move to g15blx02.

## Standard wake 21:14 UTC
blx01 broker still in recovery: its health-gate job 629 (galaxy reset) running; its bridge-reset jobs 627/628 for chips 16-23 failed (exit 8).
No submit while recovery runs. Probe: tmp/probe_blx01.sh (exit 0 once nothing is running on the blx01 broker).
On wake: check `tt-device-mcp status` on g15blx01 shows the recovery/fabric-check completed OK and no other job of ours runs,
then resubmit: ssh g15blx01 "tt-device-mcp run-bg 'env TAG=a bash /var/tmp/fasth3/t166/job.sh' -w /var/tmp/fasth3/t48 -e /var/tmp/fasth3/t159/env.yaml -t 600".
If recovery failed or chips 16-23 stay missing: move to g15blx02 (needs an overlay tree under ~/fasth3, python files only).

## Light wake 21:22 UTC
blx01 recovered (power-cycle 632, health-gate 637 OK, fabric-check 638 OK, hold 639 ended "ready for tenants").
Resubmitted as blx01 job 640 (TAG=a, -t 600) at 21:22:37 UTC. Next: same steps as "Next" above with job 640.

## Standard wake 21:26 UTC: moved to g15blx02
Drop 2: blx01 job 640 (ours, same config) killed -9 at 21:25:17 UTC after 160 s (during warmup stage 2 capture);
chips 16-23 (tray 3/4 in broker's numbering "trays [4]") left PCIe again; bridge resets 642/643 failed, broker escalating.
Same config dropped twice in a row on blx01 => skipped there (charter rule). Moved to g15blx02.

Warm-cache "before" (g15 job 399, bf7db12a14, 3653/3653 JIT hits): process 162 s, warmup 81.7 s =
setup+gemma encode 16.5 | image encoder 11.6 | s1 7.9 | s1 per-token (i2v) 4.6 | upsample 0.8 | s2 8.0 |
s2 per-token+transition+vae 8.2 | audio eager 14.3 | audio capture 9.9. gen#0 35.0 s, gen#1 6.101 s.
=> the 530-661 s fresh-process numbers are cold JIT compile (job 621: 0/3653 hits). With a warm cache the
whole 1-seed process is 162 s. The audio eager 211 s on blx01 was compile, not dispatch.

g15 overlay: data/g15/t166/tree (cp -al of t158 models/ + the 6 t166 models files, OVERLAY_COMMIT 0ee3d31bda9).
Job script tmp/t166/job_g15.sh (TAG=a, seeds 0-4, T2V_ONLY=1, ENCODERS=0). Output data/g15/t166/out/a/.
Submitted g15 job 406 at 21:26:59 UTC, -t 600 (unmeasured config), queued behind ltx-host 405.
Next on wake: section times from out/a/run.log; process wall; E2E_WALL_S per gen; framemd5 of
out/a/ltx_av_fast_1920x1088_1.mp4 vs data/g15/out/ltx_av_fast_1920x1088_1.mp4 (job 399), and _0 too;
then land 0ee3d31bda9 on t48; result.json with -t recommendations.
