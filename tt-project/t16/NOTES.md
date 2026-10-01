# t16 notes: LTX-2.5 5-seed reference clips (blx03)

Attempt 2 (2026-09-30 ~19:40 UTC, after the 19:18 reboot):
- The t16 worktree on g15blx02 was half-checked-out by the reboot (empty files, broken index); restored with `git reset` + `git checkout -- .`.
- 3c13c445c79 adds LTX_SEEDS to test_pipeline_ltx_distilled.py: after capture gen#0 and replay gen#1, one replay per seed, same default prompt, written as ltx_av_fast_1920x1088_seed<N>.mp4. One job, one load, one capture.
- blx03: python-only worktree ~/fasth3/t16 (detached at 3c13c445c7; build_Release, runtime, _ttnn.so symlinked to ~/fasth3/tt-metal), env tmp/blx03/env16.yaml (shared kernel cache /var/tmp/fasth3/cache/tt-metal-cache).
- Detached driver on blx03: ~/fasth3/drive16.sh, log ~/fasth3/drive16.log (prints JOB=<id>, ends DRIVE16_DONE). It waits for other smarton jobs (879, t20) to finish, then queues
  `W=~/fasth3/t16 bash tmp/blx03/run25.sh seeds5 LTX_SEEDS=0,1,2,3,4 LTX_FRESH_PROMPTS=0 PYTEST_TIMEOUT=1140` (timeout 1200). Output: blx03:~/fasth3/out/ltx25_1080p_6s/seeds5/.

Attempt 3 (19:42): job 880 failed in 45s: `_ttnncpp.so` not found. _ttnn.so's RUNPATH is $ORIGIN/../../build/lib and ~/fasth3/t16 had no `build` symlink (t17 has one). Added `build -> build_Release`; import + pytest --collect-only pass on CPU. drive16.sh relaunched (old log: drive16.880.log); it waits for t17 job 884 first.

Next, once `ssh g14blx03 grep -q DRIVE16_DONE ~/fasth3/drive16.log`:
1. Check the job passed (drive16.log status, run.log ends " passed").
2. `python3 tmp/t16/post.py <job>` → tt-project/baselines/ltx25_1080p_6s/ref_dv145/{seed0..4.mp4, meta.json, raw/run.log}.
3. Stills: ffmpeg -ss 3 -frames:v 1 per seed into ref_dv145/stills/.
4. VBench (g15blx02 venv): `python -m models.tt_dit.tests.models.ltx.tools.ltx_eval batch --cand-dir <B> --out <B>/eval --vbench subject_consistency,background_consistency,motion_smoothness,imaging_quality,dynamic_degree --jobs 5` (detached; no ref exists yet, so this scores the refs themselves).
5. Clean up: blx03 ~/fasth3/out/ltx25_1080p_6s/seeds5 mp4s after copy, blx03 worktree ~/fasth3/t16 once done.

Attempt 4 (20:10): job 886 passed (900s budget, exit 0). Post done:
- Refs: ~/fasth3/tt-metal/tt-project/baselines/ltx25_1080p_6s/ref_dv145/seed{0..4}.mp4 (1920x1088, 145f, 6.04s, with audio), meta.json, raw/run.log, stills/seed{0..4}.jpg (frame at 3s).
- seed0 is byte-identical to the gen#1 replay (same md5), so replay is deterministic.
- Per seed (replay): S1 2.2-2.3s, S2 2.5s, DiffVAE 11.5s, compute 17.0-17.2s, E2E wall 18.05-18.44s.
- blx03 cleanup done: worktree ~/fasth3/t16 removed, seeds5 outputs removed, driver logs in ~/fasth3/logs.
- VBench: detached, log ref_dv145/eval.log (ends VBENCH16_DONE), results ref_dv145/eval/summary.json.

Next on wake (VBench done): read ref_dv145/eval/summary.json + eval.log QUALITY lines, write ref_dv145/README.md (paths, per-seed timings, VBench means), prune eval/*.png down to one per seed if large, note + result.json done. Still for the user: ref_dv145/stills/seed0.jpg.
