# t104: 10-minute 4x8 e2e smoke of t48 (NOT submitted)

Script: tmp/t104/run_smoke.sh (derived from t95 tmp/t86/run_eval.sh). One broker job runs baseline then
'all' (LTX_FUSE_GATE_ON_DEVICE=1 LTX_FUSE_NORM_ADALN=1 LTX_VAE_CONV_FIDELITY=LoFi), SEED=0, 1080p/145f,
RUN_WARMUP=1 (gen#0 capture + gen#1 warm), no extra replays, no VBench, LTX_TIME_STAGES=1,
LTX_CONV3D_BLOCKING_MESH=4,8. Per config: mp4 + still at 2 s (<mp4>_still.png). Stops after a failed config.
Output: /var/tmp/fasth3/smoke104/{baseline,all}/, log smoke.log, marker line "SMOKE_DONE rc=N".

Dry run (g15blx02, no device): `W=$PWD DRY_RUN=1 bash tmp/t104/run_smoke.sh` -> RUN_EXIT 0 for both configs.
Test module at t48 tip py_compiles; full import needs a ttnn build (none in this worktree; no new builds allowed).

Caveats found:
- t48 tip = ttp/t48-ltx25-integrated @ 83c11ee2b34. blx03 ~/fasth3/t48 is at b43f3ea63a (older); it must be
  synced to the t48 tip (and rebuilt if C++ changed) before launch. No t95/t104 tree exists on blx03.
- LTX_CONV3D_BLOCKING_MESH has no reader in t48 pipeline code (only in t96/t97 2x4 A/B harnesses). On a
  real 4x8 mesh the production blockings apply anyway; the var is set per spec and is harmless.
- LTX_SEEDS from run_eval.sh has no reader in the test either; SEED=0 drives the single seed.
- 10 min for two 4x8 configs assumes warm caches in /var/tmp/fasth3/cache; cold compile will hit the cap.

Exact broker command (run on blx03, ONLY after the user allows 4x8 runs):
  scp tmp/t104/run_smoke.sh g14blx03:/var/tmp/fasth3/smoke104_run.sh
  ssh g14blx03 'cd ~/fasth3/tt-metal && tmp/blx03/submit.sh 600 env W=/home/smarton/fasth3/t48 bash /var/tmp/fasth3/smoke104_run.sh'
Check: ssh g14blx03 'grep SMOKE_DONE /var/tmp/fasth3/smoke104/smoke.log'
