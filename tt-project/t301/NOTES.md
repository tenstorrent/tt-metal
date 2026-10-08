# t301: standard LTX-2.3 e2e 8+3 on 4x8, ttp/ltx23-main-pr (df9e5ecaac6) vs origin/main (80b1cd689d0)

Box: blx01 (blx03 unreachable at 19:33 UTC: "No route to host"). Everything under /var/tmp/fasth3/t301 there.
- setup301.sh (started 19:37:57 UTC, setsid, pgid 208076): worktree of the t48 clone at df9e5ecaac6 -> t301/pr,
  lean `build_metal.sh --release`. setup.rc = 0 means it built cleanly (this is the C++ build check #298 could not do).
  Log t301/setup.log.
- Main arm reuses t293's clean origin/main build /var/tmp/fasth3/t293/main @ 80b1cd689d0 (t293's setup was still
  building at 19:30 UTC). run301.sh checks commit and _ttnn.so before running.
- drv301.sh (pgid 208175): waits for both builds, then main arm, then PR arm, one broker job each (-t 600, unmeasured),
  only when no hold/health/reset/upgrade/smarton job is on the broker. Drop -> one rerun. Then cmp301.py PCC/PSNR main
  vs PR per gen -> t301/cmp.txt and still_main_vs_pr_gen2_*.png.
- Env both arms: unmodified test, 8+3 default, bf16 default, seed 10, RUN_VBENCH=0 RUN_CLIP=0, TT_DIT_CACHE_DIR unset
  (no DiT cache written), JIT cache t301/jit-<arm>. Warm table = gen #2 (pure replay) in out_<arm>/run.log.
- Marker: /var/tmp/fasth3/t301/drv301.done; per-step log drv301.done.log (job ids, statuses).

## Attempt 2 (2026-10-08 ~19:55 UTC, blx03 still "No route to host")
- First driver run: both broker submits refused ('Python env not found at .../t301/tt-metal/python_env'). Fix: env.yaml
  (PYTHON_ENV_DIR=t48 python_env, HOME/TMPDIR under /var/tmp/fasth3) passed with -e.
- Job 122 (main): 7 deselected, exit 5. main's test ids differ from ltx-rt: ltx-rt 'bh_4x8sp1tp0_ring' ((4,8) sp1 tp0
  nl2 ring fsdp0, ring_trace_params) = main/PR '4x8sp1tp0nl2_ring_is_fsdp0'. Test file identical in both arms. Own queued
  PR job 125 cancelled. Old outputs in t301/prev1.
- run301.sh now runs pytest under setsid with an EXIT/TERM trap that kills the whole group; the driver logs any
  leftover pytest/run301 process after each job.
- Driver restarted 19:58:40 UTC; main arm = job 126 (queued behind ltx-host 124).

Next: when the marker exists, read it. DONE -> quote both gen #2 timing tables from out_main/run.log and out_pr/run.log
verbatim, per-row deltas, check ~-0.4 s across Stage1+Stage2+Audio decode (VAE decode unchanged), cmp.txt, md5 lines.
mp4 md5 always differs between arms (PR export is x264 ultrafast crf 20); PCC/PSNR on decoded frames is the check.
PR_BUILD_FAILED -> read setup.log. Cleanup after: t301/pr worktree (git -C /var/tmp/fasth3/t48 worktree remove),
jit-main, jit-pr, out_* (keep small logs/stills copied here).
