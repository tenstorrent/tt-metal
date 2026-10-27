# t511 notes (device A/B LTX25_S2_ANCESTRAL off vs on, then ltx-rt push)

- Code: 923fffb239d = cherry-pick of a22dc027ee2 onto origin/ltx-rt f6547442b30 (clean). CPU test
  models/tt_dit/tests/unit/test_ltx25_s2_ancestral.py: 4 passed.
- Box: blx03 (only box with LTX-2.5 weights: /var/tmp/fasth3/t503/ltx25; blx01's copy is gone).
  Tree ~/fasth3/t511/tt-metal on blx03 = `cp -al` of t503's tree + runtime/hw/lib from g15blx02 t313
  (t503's copy lacked runtime/hw/lib/blackhole/*.o: its job died at fabric JIT link) + models/ at 923fffb239d.
- Jobs (blx03 serial runner, FIFO behind t503-ab-r2):
  1. t511-e2e-r1: standard LTX-2.5 e2e, test defaults (pre-push gate, also fills JIT /var/tmp/fasth3/t511/jit).
     Cold JIT may not fit 575 s (t373 on blx01: cold 545 s timeout, warm 293 s): then requeue as -r2.
  2. t511-ab-r1: SEED=0, LTX_E2E_SEEDS=0,0+, LTX_E2E_AB_ENV=LTX25_S2_ANCESTRAL=1:
     out_ab/ltx_av_fast_1920x1088_1.mp4 = off, _2.mp4 = on (both pure trace replays).
- Probe: ssh g14blx03 'bash ~/fasth3/runner/probe.sh <ID>'; marker /var/tmp/fasth3/runner/done/<ID>.done
- Next: check logs (noise_seed=20000 only in gen #2), quote Stage 2 denoise for gen #1/#2, copy mp4s to
  g15blx02, run ltx_eval video --vbench-ref, look at stills; if ok, ttp push to ltx-rt (ff only).
- Cleanup at end: ~/fasth3/t511 and /var/tmp/fasth3/t511 on blx03.
- 2026-10-15 04:13 PDT: queued; runner restarted pid 49491 (blx03 rebooted ~03:51 PDT), t503-ab-r1/-r2 ahead
  (they stop early on t503's last_fail guard). ttp checks started in run 1958 (checks.out/.rc there).
- 2026-10-15 05:58 PDT: r1 jobs all failed in ~12 s, not drops: ValueError "missing Torch state keys: conv_in.weight, ..."
  (broker jobs 432, 434 e2e; 436, 450 ab). The 2.5 conv VAE split file has bare decoder./encoder. keys; ltx-rt's loaders read
  only vae.* (fix 63e54d98036 is on t48 only). The runner logged them as drops (chips=unknown) and blocked config
  t511-ltx25-e2e: cleared it. Wrote /var/tmp/fasth3/t511/vae/ltx-2.5-video-vae-conv-bf16-vaeprefix.safetensors on blx03
  (same 170 tensors + metadata, keys prefixed vae.), run511.sh points there. Queued t511-e2e-r2, t511-ab-r2.
- 2026-10-15 08:45 PDT: r2 drops on blx03 tray 2, all our jobs, all during load/compile:
  454 (06:05 PDT, chip 8), 473 (07:23 PDT, chips 8+12), 482 (08:28 PDT, chips 8-15; blx03 then rebooted ~08:35 PDT).
  t511-e2e-r2 skipped on blx03 (config t511-ltx25-e2e dropped twice: 454, 473) -> pre-push e2e must run on another box
  (g15blx02 or blx01 standard LTX e2e) once the A/B passes. t511-ab-r2 dropped once (482), but the runner logged 482 twice
  (drop + broker-kill after the reboot) in configs/t511-ltx25-ab, which would skip it after one real drop: deduped to one line.
  Runner (pid 12726) retries t511-ab-r2. If it drops again: skip blx03 for the A/B (weights exist only on blx03).
- 2026-10-27 00:00 PDT (run 2142, blx04): ltx-rt moved 23 commits (models/ only) since f6547442b30. Re-cherry-picked a22dc027ee2
  onto origin/ltx-rt 522ef26916b: clean, 3f3ce168ce8 on local branch ttp/t511-ltxrt-land; CPU test 4 passed.
  blx04 tree ~/fasth3/t511/tt-metal checked out at 3f3ce168ce8 (same C++ as the t554 build). Conv VAE full sha256 matches 685b06ee...
  A/B now runs one arm per process (memory #244: in-process env toggles can replay cached programs/traces):
  tt-project/t511/{run511b4.sh off|on|e2e, spec-*.txt, blx04-submit.sh (guard: only FastH3 jobs count; TTP_NO_DEVICE=0 is not "no device"),
  chain511b4.sh}. Chain (ttp detach chain511, run dir 2142): e2e (JIT fill + pre-push standard e2e) job 310, then off, then on; no retries.
  Outputs: ~/fasth3/data/t511/out_{e2e,off,on}/ (run.log, ltx_av_fast_1920x1088_1.mp4 = replay gen, *_t3s.png).
- Next: off vs on: grep noise_seed=20000 (only in out_on), Stage 2 denoise lines, md5 differ; ltx_eval video --vbench-ref; stills.
  If ok: ttp push --detach from ttp/t511-ltxrt-land (ff onto origin/ltx-rt); e2e job 310 is the standard e2e on that commit.
- Update 00:00 PDT (tt-buddy 2e437e49): report quality only (PCC/PSNR, VBench, stills), no speed/denoise-time quotes.
  On a tray/chip drop or broker degrade: no retry (chain already stops), `ttp note --to tt-buddy` with job id + PDT time,
  hand off blocked with waits_on `when:timeout 50 tt-project/harness/bin/fasth3-box-health blx04 >/dev/null 2>&1 || exit 1`.
- 2026-10-27 00:12 PDT (run 2149): job 310 (e2e.1) ended exit 124 at 00:09:30 PDT: cold JIT still compiling
  (model load began 00:08:34 PDT after ~8 min of kernel builds); no drop signs, not a device fault. JIT cache persists
  in ~/fasth3/data/t511/jit (tt-metal-cache<build_key>/kernels, 96 kernels after 310). chain511b4.sh now allows up to
  3 e2e passes, continuing only on exit 124/timeout (root commit 5910d21e3a9). Relaunched as `ttp detach chain511c`
  (run dir 2149): adopted 310, e2e.2 submit waiting on blx04's post-job broker gate (chainlib retries).
  Probe: ttp detach --check /home/smarton/fasth3/tt-metal/tt-project/state/runs/2149/chain511c.rc
- 2026-10-27 00:28 PDT (run 2151): host rebooted 00:25 PDT; chain511c died before e2e.2 was submitted (no job after 310;
  all e2e.2 submits were refused by blx04's broker gate). JIT cache in ~/fasth3/data/t511/jit survived. blx04 broker in
  post-boot health gate (fabric check). Relaunched the same driver as `ttp detach chain511d` (run dir 2151); it adopted 310
  and retries the e2e.2 submit. Probe: ttp detach --check /home/smarton/fasth3/tt-metal/tt-project/state/runs/2151/chain511d.rc
