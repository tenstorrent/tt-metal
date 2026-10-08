# t295 notes

## Done (run 1168, 2026-10-08 ~19:05 UTC)
- Not reverted anywhere before: t48 tip 9e20d905481 still had `_DEFAULT_S2_SIGMAS = [0.909375, 0.421875, 0.0]`;
  #293 only overrides via LTX_S2_SIGMAS in its run script.
- Code commit f6547442b30 "ltx distilled: restore the 3-step stage-2 schedule as the default" LANDED on
  origin/ttp/t48-ltx25-integrated (ttp push, pushed f6547442b304d744711e80e2281f6bd368291673).
  Sigma block now byte-identical to b21f12b93a2's parent (a1da2e19509). LTX_S2_SIGMAS override kept.
  Tests: test_ltx_quality (3-step default; 2-step and FAST still selectable), test_ltx_euler_tail (11 steps);
  both failed before the fix, 19 pass after. README_euler_tail / test_euler_tail_trace docstring / ltx.py
  high-tier comment back to 8+3. medium/fast tiers (FAST_S2_SIGMAS=0.909375,0.0) left as is: explicit opt-in tiers.

## Device md5 check: NOT RUN yet
blx03 (No route to host) and blx01 (ssh timeout) both down at 19:00 UTC. g15blx02 has no LTX setup (/home full).

Reference: tt-project/data/g15/ref_t48_f6b8/seed0-4.mp4 (job 710; t188 job 755 matched it byte-for-byte):
  seed0 b894579d77d8f57bf92d8db79cfd9b92  seed1 e72baf2eb9d393c486b561acb14cb9c1  seed2 395b25ecbbafaea1f6bc3d5f02151ddc
  seed3 0347a6649be40a3a0af41fd397d4d0a1  seed4 a884491baa50155bfa6597c5f2db0531
Env = LTX_VERSION=2.5 (LTX-2.3 ckpt + LTX-2.5 split ckpt root), dit-ltx25 cache, t48 build bf7db12a149 on blx01.
The ref used LTX25_ROOT=/mnt/MLPerf/... -> FORBIDDEN now (job 086 D-state hang). Need a LOCAL LTX-2.5 root.

## Run 1172 (2026-10-08 19:08-19:15 UTC)
- blx03: still 'No route to host'. blx01: answered at 19:08 UTC (up 3 min after a reboot; broker in
  hold-deadline-escalate fabric check, broker job 099; a non-t295 smarton job 091 `bash /tmp/ttb151/job.sh` queued),
  then 'No route to host' again by ~19:12 UTC. No t295 job submitted, so no drop of ours to log.
- blx01 data as of 19:08: /var/tmp/fasth3 = 159G (over the 150 GB cap already; add nothing big).
  cache/dit-ltx25 is GONE (only dit-h3hf, tt-metal-cache*); models/ has no LTX-2.5 split root.
  So the LTX_VERSION=2.5 reference env (job 710/755) cannot be reproduced on blx01 without new caches.
  It still has t220/cache/dit-ltx23 (113G, LTX-2.3 cache from t220, contents not checked: bf8 medium tier
  certain, bf16 transformer unknown) and t48 build bf7db12a149.
- g15blx02: /home 1.5T free, but no LTX build/ckpts/caches; a fill alone would break the 600 s cap. Not used.
- Fallback plan if blx03's 2.5 env is gone too: pure LTX-2.3 (LTX_VERSION unset) A/B in ONE job on a box with a
  bf16 LTX-2.3 DiT cache: arm A = overlay f6547442b30 at defaults; arm B = overlay 9e20d905481 (pre-change tip) +
  LTX_S2_SIGMAS=0.909375,0.725,0.421875,0.0. md5 A == B per seed proves the default revert is exact. Seeds 0-1
  per arm if 5 do not fit -t 600.
- Wake probe: tmp/t295/probe.sh (box up >=20 min, broker not holding/resetting).

## Next step (when blx01 or blx03 answers ssh)
1. On the box: df -h /; ls /var/tmp/fasth3/{t48,cache,models}; find a local LTX-2.5 split root
   (e.g. /var/tmp/fasth3/models/ltx-2.5*, ~/.cache/ltx-checkpoints/ltx-2.5, /home/sulphur/hf/...). None -> copy only the
   files the 2.5 path reads (check size vs 150 GB blx01 limit) or skip the box. Check cache/dit-ltx25 still exists.
   Check $W=/var/tmp/fasth3/t48 build commit: tip imports neighborhood_sdpa (C++ after bf7db12a149) - make sure ttnn import works.
2. `git archive f6547442b30 models/tt_dit conftest.py pytest.ini pyproject.toml | gzip > tmp/t295/py.tar.gz`,
   scp tarball + setup295.sh + run295.sh to /var/tmp/fasth3/t295/ (tmp name + mv), run setup295.sh (no device).
3. Health check (tt-device-mcp status), then tt-device-mcp run-bg -w /var/tmp/fasth3/t48 -t 240
   -- bash /var/tmp/fasth3/t295/run295.sh s5 <local LTX25_ROOT>   (job 755: 161.6 s at 8+3).
4. Compare md5 of ltx_av_fast_1920x1088_{1..5}.mp4 (= seeds 0-4) with the ref above. If they differ, later t48
   commits changed numerics: run a 2nd job with the pre-change tip 9e20d905481 overlay + LTX_S2_SIGMAS=0.909375,0.725,0.421875,0.0;
   it must match this run's md5s (that is the real identity check of the revert).
5. Clean up /var/tmp/fasth3/t295 (keep run.log + md5s here).

## Run 1178 (2026-10-08 19:16-19:21 UTC): fallback LTX-2.3 A/B submitted via driver on blx01
- blx03 still 'No route to host'. blx01 up again (rebooted ~19:11 UTC, broker in idle-relift health check).
- blx01 /var/tmp/fasth3/t48 build bf7db12a14 exists; t220/cache/dit-ltx23 has the bf16 LTX-2.3 cache (t283 job 077
  ran the standard e2e test on it: 139 s process wall, warm). No LTX-2.5 root, so the job-755 ref md5s
  (LTX_VERSION=2.5 env) cannot be reproduced; doing the A/B identity check instead.
- /var/tmp/fasth3/t295 on blx01: treeA = t48 models + f6547442b30 files (S2 default 4 sigmas), treeB = treeA with
  the 6 changed files at 9e20d905481 (S2 default 3 sigmas). Host import check: A default and B+LTX_S2_SIGMAS both
  give [0.909375, 0.725, 0.421875, 0.0]; B default gives the old 2-step list.
- job295.sh (one broker job, -t 520, -w t48): run295b.sh A then B, each its own pytest process, standard test
  test_pipeline_distilled -k bh_4x8sp1tp0_ring unmodified, SEED=0 LTX_E2E_SEEDS=0,1,2,3,4 (gen0 = cold seed 0,
  gen1..5 = seeds 0..4), RUN_VBENCH=0 RUN_CLIP=0, bf16, gate fold default. Prints per-gen md5 A vs B MATCH/DIFF.
- drv295.sh (pgid 19426 on blx01, started 19:19 UTC) waits for uptime >=15 min and a clean broker, submits once,
  waits, copies `tt-device-mcp logs` to job<ID>.log, writes drv295.done ("DONE job=<id> status=<s>").
- Wake: `ssh blx01 test -e /var/tmp/fasth3/t295/drv295.done`.

## Next step on wake
1. cat drv295.done; read outA/run.log, outB/run.log tails and the "[t295] genN A=.. B=.. MATCH" lines (job log or
   rerun the md5 compare from outA/outB). All 6 MATCH + both T295_EXIT=0 -> revert verified; report job id + md5s.
   DIFF -> check whether gen0 vs replays differ run-to-run (nondeterminism) before blaming the code.
2. A drop -> log UTC/box/job/chips, rerun drv295.sh once (it submits fresh).
3. Copy run logs + md5 lists to tmp/t295/res/, then rm -rf /var/tmp/fasth3/t295 on blx01 (outputs, trees).
