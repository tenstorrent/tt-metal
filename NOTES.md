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
Job 625 killed by broker device recovery at ~21:11 UTC on g15blx01 (wedge; Runtime 94 s, exit -9, our job,
during warmup stage 1). Log ended 21:10:51. Not re-queued by the broker. Next: when blx01 health check passes,
resubmit the same job.sh (TAG=a); if it drops again, move to g15blx02.
