# t305: repeat of t301 (main 80b1cd689d0 vs ttp/ltx23-main-pr df9e5ecaac6), standard LTX-2.3 8+3 e2e, blx01 4x8

Same scripts as t301 (t301/run301.sh, env.yaml on blx01 /var/tmp/fasth3/t301), kept builds and JIT caches.
- 02:35:09 UTC: main arm = broker job 214 (-t 570), submitted by hand. Broker idle and healthy before (fabric-check 212 ok).
- 02:36:55 UTC: drv305.sh started on blx01 (/var/tmp/fasth3/t305, setsid, pgid 672445): waits for 214, then submits
  the PR arm (-t 570), one rerun per arm on a drop. Logs: t305/run_<arm>_job<id>.log. Marker: t305/drv305.done.
- Next on wake: read drv305.done + drv305.done.log; quote both gen #2 tables verbatim (grep the 3rd PERFORMANCE table
  in each run log), per-row deltas vs t301 (main 6.41, PR 6.24). Check no run301/pytest left on blx01.
  Cleanup after: git -C /var/tmp/fasth3/t48 worktree remove --force /var/tmp/fasth3/t301/pr; rm -rf t301/jit-* t301/out_*.

## Code reading / log findings (done while job 214 ran)
- mel-VAE trace (05401286709) DOES run in the standard test: traced=True (test log "traced=True: forcing warmup"),
  _trace_gates sets mel use_trace=True, and gen #0 audio decode captures 4 traces on PR vs 3 on main (t301 logs,
  02:29:34-02:29:43 vs 02:10:24-02:10:33). But gen #2 audio decode is 374 ms main vs 372 ms PR (log timestamps
  "VAE decode (forward)" -> "Audio decoded on device"). The ltx-rt claim (~0.88 -> ~0.48 s mel-VAE+BWE) does not
  hold on main: main's eager audio decode is already 0.37 s total. Gain in this path: ~2 ms = dead.
- Seeded noise prefetch (bfc93527e92): keys match by code reading. Prefetch keys (seed, ((1,s1_n,C),(1,audio_n,C)))
  and (seed, ((1,s2_n,C),)); stage 1 calls _seeded_noise(seed, v_shape, audio_shape) with initial_audio_latent None,
  stage 2 calls _seeded_noise(seed, v_shape) (seed=seed both), s1_n/s2_n from latent_grid with the same h/w, audio
  frames depend on duration only. So it is served from the thread (no silent fallback). Expected ~-46 ms S1, ~-92 ms S2.

## Run 2 (2026-10-09 08:39 UTC wake)
- Job 214 (main) completed exit 0, warm gen #2 Total 6.19 s (Enc 0.19, S1 2.14, Up 0.11, S2 2.54, VAE 0.82, Audio 0.39).
  Log: blx01 t305/run_main_job214.log (note: t301 main was 6.41 with Enc 0.45; without Encoder 5.96 vs 6.00 now).
- DROP: 2026-10-09 02:45:36 UTC, blx01, broker job 215 (our PR arm, run301.sh pr), chips 16-23 left PCIe (bridge-reset
  216-218 failed, power cycle 221 at 02:51 UTC, device recovered 02:55 UTC, fabric-check 227 ok). Broker status
  broker-kill, not re-queued. drv305.sh died with the host reboot (uptime from ~02:52 UTC); drv305.done never written.
- 08:40:46 UTC: PR arm resubmitted by hand as broker job 242 (-t 570), queued behind ltx-host 241.
  Script t305/run305.sh = run301.sh + df/du caps (lint), outputs in t305/out_pr. Same builds/JIT caches.
- Next: when 242 ends, quote its gen #2 table (3rd PERFORMANCE box in t305/out_pr/run.log), compare. If 242 drops too:
  second drop of the PR config on blx01 -> skip and report with t301 data only.

## Run 3 (2026-10-09 08:48 UTC wake)
- Job 242 (PR df9e5ecaac6) completed exit 0, gen #2 Total 7.83 s (Enc 0.26, S1 2.83, Up 0.14, S2 3.20, VAE 0.92,
  Audio 0.47). INVALID: all 32 chips "AICLK failed to settle ... observed 900, clamped by max-arbiter index 10 at 900 MHz"
  (expected 1350). Main job 214 (02:35 UTC) and t301 jobs 210/213 had 0 such warnings. Not a drop (no rerun counted).
- The clamp is box state: every device-opening broker job on blx01 since job 230 (03:59 UTC) shows it, incl. ltx-host
  live-service jobs 231-243. Started after the 02:51 power cycle (jobs 221-229 clean). We never reset; not ours to fix.
- Code reading: pipeline_ltx_distilled.py sets last_timings (line ~856) BEFORE export_video_audio(_yuv), so the two
  export commits (5565232f12c AAC overlap, df9e5ecaac6 x264 ultrafast) are invisible in the standard test's table.
  The mel-VAE trace (05401286709) is dead here (~2 ms, see run 1). Those three were most of the expected ~-0.4 s.
- Valid data now: main 6.41 (210) / 6.19 (214); PR 6.24 (213). Excluding Encoder (0.19-0.47 s noise): main 5.96/6.00,
  PR 5.77 -> -0.19..-0.23 s; Stage 2 2.55/2.54 -> 2.40 (-0.15), Stage 1 2.11/2.14 -> 2.08 (-0.03..-0.06).
- Next: probe tt-project/t305/clkprobe.sh (exit 0 = newest device job on blx01 ran unclamped, or deadline
  2026-10-10 02:00 UTC). On pass: lint, then resubmit PR arm (bash /var/tmp/fasth3/t305/run305.sh pr, -w t305,
  -e t301/env.yaml, -t 570), wait, quote its table. If still clamped at deadline: report with t301 PR sample only.

## Run 4 (2026-10-10 02:24 UTC wake)
- Clamp still on: every ltx-host job through 348 (01:09 UTC) shows "AICLK failed to settle" on 32 chips; no reboot
  (uptime 23:33). clkprobe deadline passed.
- Decision: rerun the MAIN arm under the same 900 MHz clamp, to pair with PR job 242 (7.83 s) at matching clocks.
  This gives a second same-conditions main/PR pair (relative delta only; absolute numbers are not comparable to 1350 MHz).
- 02:25:29 UTC: main arm = broker job 349 (bash t305/run305.sh main, -w t305, -e t301/env.yaml, -t 570), lint ok.
  Output: blx01 /var/tmp/fasth3/t305/out_main/run.log. /var/tmp/fasth3 was 145G before (cap 150G), df / 53%.
- Next: when 349 ends, check its clamp (expect 32 warnings), quote its gen #2 table, compare with 242.
  Then final report; cleanup t301/pr worktree, t301/jit-*, t301/out_*, t305/out_* (keep run logs).
