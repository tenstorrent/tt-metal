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
