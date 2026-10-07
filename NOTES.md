# #189 notes: LTX_AUDIO_REPLICATE (denoise PLAN task 4)

## State (2026-10-07 ~05:05 UTC)
- Code: 5673d92a566 (opt-in LTX_AUDIO_REPLICATE, default off). Design: per audio block two "islands"
  (self-attn, FFN): TP all_gather of x -> local RMSNorm + addcmul modulation -> replicated-weight compute
  -> gated residual -> mesh_partition. Replicated weights are built on device after cache load
  (post_load_hook, before fold_gates_on_device); cache key unchanged; ~+4.8 GB DRAM/chip.
- CPU test on blx01 (t48 venv, t189 overlay): test_audio_replicate_layout.py 3 passed.
  (g15blx02 `ttp checks` fails on missing _ttnncpp.so: no build on g15, env issue.)
- blx01 staging: /var/tmp/fasth3/t189/{files,tree,mkoverlay.sh,run_ab.sh,res}. tree = t171/tree (t48
  f6b806516cc) + the 4 changed files; OVERLAY_COMMIT 5673d92a566.
- NOT submitted yet: blx01 broker was HELD recovering tray 3 (chips 16-23).

## Submitted
- 2026-10-07 05:36 UTC: blx01 broker job 754 (run_ab.sh ab1, -t 600). On wake: check res/ab1/job.log for T189_DONE, then steps 3-5 below.

## Drop log
- 2026-10-07 04:59 UTC (06 21:59 LA), blx01, broker job 732 (smarton, #188's s5 config), chips 16-23
  (tray 3) left PCIe; broker-killed, held, reset (jobs 733-736). Not a #189 job.

## Next step (exact)
1. `ssh blx01 tt-device-mcp status` shows no broker hold/health-gate RUNNING and no smarton job running/queued.
2. Submit (one job, both arms, ON first then OFF, seeds 0-4 each, DIFFVAE_MEM_LOG=1, LTX_CONV3D_BLOCKING_MESH=4,8):
   `ssh blx01 "tt-device-mcp run-bg 'bash /var/tmp/fasth3/t189/run_ab.sh ab1' -w /var/tmp/fasth3/t48 -e /var/tmp/fasth3/t159/env.yaml -t 600"`
   (ON_S=330, OFF_S=220 pytest timeouts inside; baseline 5 seeds took ~160 s.)
3. Results: /var/tmp/fasth3/t189/res/ab1/{on,off}/run.log, *.mp4, *_t3s.jpg; summary res/ab1/job.log
   (E2E_WALL_S, [dram], T189_EXIT/T189_DONE).
4. Score PCC/PSNR per seed vs tt-project/baselines/ltx25_1080p_6s/ref_t48_f6b8 (DEFAULT prompt, seeds 0-4).
5. Clear win (>= ~20 ms e2e, output equivalent) -> default-on commit, `ttp push` to ttp/t48-ltx25-integrated
   (code only). Else keep opt-in, `ttp push --own --detach`.

## Push blocker (2026-10-07 05:37 UTC)
- `ttp push --own` fails its check twice (exit 4): ~/fasth3/tt-metal/build -> build_Release, which no
  longer exists on g15blx02, so `import ttnn` in the root conftest fails (_ttnncpp.so). t188's 04:53 push
  hit the same. Not a code failure; the task branch stays local until the g15 build is back.

## ab1 result (blx01 job 754, 05:36-05:40 UTC)
- OFF arm OK: warm gen#1-5 E2E 5.726/5.697/5.785/5.721/5.721 s (mean 5.730); peak DRAM 20.15 GiB/chip; no drop.
- ON arm FAILED at 86 s: TT_FATAL minimal_matmul_split.cpp:46 `N_per_chunk % TILE_WIDTH == 0`
  (replicated audio QKV/FFN fused split matmul: chunk width not tile-aligned when N is not divided by TP).
  Log: blx01 /var/tmp/fasth3/t189/res/ab1/on/run.log line 1039. Needs a code fix (standard tier), then rerun.

## Fix + ab2 (2026-10-07 ~05:50 UTC)
- Code fix 8e6ba4d72a2: unsharded minimal_matmul_split needs equal tile-aligned chunks; the gate (32 cols)
  no longer shares the QKV split, it runs its own ttnn.linear. QKV is a 3-way split of 2048.
  CPU test on blx01 (t189 overlay): 5 passed (new test_qkv_split_chunks_fit_minimal_matmul_split).
- blx01 overlay rebuilt (OVERLAY_COMMIT 8e6ba4d72a2). #188's job 755 was running, so a detached submitter
  /var/tmp/fasth3/t189/sub_ab2.sh (pid 125838, log res/sub_ab2.log) waits for no smarton job/hold, then
  run-bg `run_ab.sh ab2` (-t 600) and writes res/ab2.marker (rc, job id, status) when the job ends.
- Next: read res/ab2.marker + res/ab2/job.log; score PCC/PSNR per seed vs ref_t48_f6b8; decide default-on
  (>= ~20 ms e2e win, output equivalent) or keep opt-in; push.
