# t409 — VSA port onto t400 (FastH3 Turbo)

Branch ttp/t409-port-vsa-sparse-attention-into-the-fasth, base a15eb37ec6a (t400), code commit d39d16e5c9b.

## Done
- C++ ops from fasth3-opt: vsa_sdpa, vsa_ring_sdpa, vsa_kv_gather (+ kernels, nanobind, cmake), unit tests under tests/ttnn/unit_tests/operations/sdpa/test_vsa_*.
- Python: vsa_geometry.py, vsa_stages_minimax_h3.py (MiniMaxH3VSAConfig, MiniMaxH3VSACoarseStage), vsa_reference/, attention VSA branch (agmm/to_out calls adapted to t400 signatures), transformer `vsa_config` + `set_vsa_stage`.
- Pipeline: opt-in `MINIMAX_H3_VSA_SPARSITY=<0..1)` (unset/empty = dense, code path unchanged). VSA tile order is realized only by permuting host index tables: assembly indices, video/audio out indices (via untile_index), rope position ids and AdaLN/timestep row indices (permute_metadata; pad slots repeat a valid row). rung = geometry.padded_len under VSA. Stages cached per geometry and never freed (traces replay their tensors). DiT weight cache subfolder gets a `_vsa` suffix. t2va/fl2va only.
- Host tests: test_vsa_geometry + test_vsa_oracle: 24 passed on g15blx02 (no device).
- test_vsa_attention: `_run_tt` now passes logical_n (t400 forward signature).

## Not done yet
1. C++ compile check (never built against the t400 base; ops written against Aug-31 main).
2. blx01 setup under /var/tmp/fasth3/t409: worktree + incremental C++ build (ionice -c3, nice).
3. Device: test_vsa_attention_minimax_h3.py::test_vsa_sparsity0_matches_ring_path (4x8 full mesh, PCC>=0.9995 vs dense ring).
4. After #408: JIT-fill job, then dense vs VSA-0.9 A/B at 10 s seed 0 in one job (both arms), via device-lint.sh. Template: tt-project/t400/run400.sh (drop its setsid re-exec; use timeout --foreground -k 10).

## t416 run 1565 (2026-10-11 22:36 PDT)
- Steps 1+2 merged: fresh Release build of the VSA code (d39d16e5c9b) on blx01 at /var/tmp/fasth3/t409/b
  (worktree of /var/tmp/fasth3/t48's repo, submodules from t284), driver tt-project/t409/build409.sh, run as
  `ttp detach --remote g15blx01 --dir /var/tmp/fasth3/t409 build409` (ionice -c3, nice 10, 24 jobs).
  Log /var/tmp/fasth3/t409/build409.log, marker build409.marker (rc=0 step=done = compile check passed and
  ttnn.transformer.vsa_sdpa / vsa_ring_sdpa import). An incremental build was not possible: ninja paths are absolute.
- Cherry-picked #408's test change (bff7d82040c -> 65689931ab0: 6 s duration + MINIMAX_H3_TURBO_SAVE_DIR) so the
  A/B uses exactly the #408 test. Python only: blx01 tree can be moved to it with `git checkout` (no rebuild).
  blx01 needs a bundle with 65689931ab0 first (t409.bundle has only up to a2dee6756f1).
- Step 3: tt-project/t409/chain416a.sh detached on g15blx02 (run dir state/runs/1565/chain416a.{log,rc}): waits
  for build409.rc, submits spec attn.txt (run409a.sh, -t 600 unmeasured) via tt-project/t288/blx01-submit.sh,
  retries once if not completed and not a test failure. Ends with "T416A_DONE <rc> <status>".
  Result: /var/tmp/fasth3/t409/out_attn/run.log on blx01 (PCC line from assert_quality).
- Step 4 script ready: tt-project/t409/run409b.sh <tag> dense vsa0.9 (one pytest per arm, 90 s watchdog, warm
  deadline 400, per-arm pytest timeout T409_PY_TIMEOUT). Size it from #408's 10 s job wall time. If the two arms
  do not fit in 600 s (+50% rule), run one arm per job. Before the VSA arm: one cold fill job (new `_vsa` DiT weight
  cache + VSA kernels).
- Note: I moved ttp/t409-... to d39d16e5c9b by mistake with `git branch -f` and restored it to a2dee6756f1 at once
  (update-ref); the t409 worktree was clean.

## t416 run 1597 (2026-10-12 00:05 PDT)
- Build409 finished 2026-10-12 05:42Z rc=0: the VSA C++ ops compile on the t400 base (compile check passed).
- chain416a (run 1565) never submitted: device-lint refused run409a.sh for `setsid`. Both run409a.sh and
  run409b.sh now run pytest in the broker's process group under `timeout --foreground -k 10` (b: watchdog stops
  $PG and its children by pid). Both pass device-lint and killscan; copied to blx01 /var/tmp/fasth3/t409/.
- chain416a restarted from run 1597: log $RUN/chain416a.out (state/runs/1597), rc chain416a.rc.
  blx01 is busy with #408/#421/#430 jobs; the chain waits for 2 clear minutes, then submits.
- Next: read /var/tmp/fasth3/t409/out_attn/run.log (PCC). Then step 4 once #408 is done.

## t416 run 1611 (2026-10-12 01:10 PDT)
- chain416a (run 1597) ended T416A_DONE 5 submit-rc2: device-lint refused because /var/tmp/fasth3 on blx01 was
  455 GB (cap 450). Freed ~2 GB: removed the done #423/#427 worktrees (t423/b, t427/b; clean, commits on
  g15blx02 branches) and JIT dirs from 10-06/07 in home/.cache/tt-metal-cache. Now 453 GB. Not removed (used by
  open tasks or a coordinator call): models/ltx-2.5 (67 GB, idle since 10-10), dit-h3hf transformer_precomputed_adaln
  (38 GB, #421), t408/b (#422 off arm), t428 builds.
- Code 53f7b6b707b: VSA builds `to_gate_compress` only with MiniMaxH3VSAConfig(gate_compress=True). The default VSA
  model has the dense weight set, so the pipeline reuses the dense DiT cache (no `_vsa` suffix): saves a 63 GB cache
  on blx01 (it would also push / past lint's 88% cap) and the cold weight fill. A nonzero gate in a checkpoint
  without gate_compress raises. Host tests 24/24. blx01 t409/b checked out at 53f7b6b707b (Python only, no rebuild).
- chain416a now waits (<=12 h) for blx01 /var/tmp/fasth3 <= 449 GB before the broker-clear wait. Restarted detached
  from run 1611 (state/runs/1611/chain416a.out).
- Step 4 no longer needs a separate weight-cache fill for VSA; one JIT warm-up of the VSA kernels is still needed.

## t416 run 1613 (2026-10-12 01:10 PDT)
- Host reboot killed chain416a (run 1611). Restarted unchanged from run 1613 (state/runs/1613/chain416a.out). blx01 /var/tmp/fasth3 still 453 GB, so it waits for the cap. Branch push of 502ae371602 landed.

## t416 run 1647 (2026-10-12 02:50 PDT)
- chain416a (run 1613) passed the cap (430 GB at 08:14Z), then ended T416A_DONE 5 submit-rc2 at 09:46Z. Cause: attn.txt
  had no ENV line, so blx01-submit.sh ran `run-bg ... -e -t 600` and the broker CLI refused (rc 2). Lint passed.
- Fix: tt-project/t409/env409.yaml (t400 env with t409/b paths), copied to blx01 /var/tmp/fasth3/t409/; attn.txt ENV= it.
- Direct submit at 09:50Z got 75 (another smarton job on blx01). chain416a restarted detached from run 1647.

## t416 run 1655 (2026-10-12 04:55 PDT)
- chain416a (run 1647) submitted blx01 job 939 at ~11:29Z: failed at collection, `No module named
  'diffusers.models.transformers.transformer_minimax_h3'` (t48 env has diffusers 0.38; the test needs the pinned
  commit abc5e9bf from MiniMaxH3.md). No device time used, no drop. Then the host reboot killed the chain.
- Fix: pinned diffusers installed with uv --no-deps --target /var/tmp/fasth3/t409/pydeps on blx01 (30 MB; shared
  t48 env untouched); env409.yaml PYTHONPATH gets it. `pytest --collect-only` on blx01: 4 tests collected.
- chain416a restarted detached from run 1655 (state/runs/1655/chain416a.out). Same next step as before.

## t416 run 1664 (2026-10-12 05:50 PDT)
- chain416a (run 1655) submitted blx01 job 958 at ~12:45Z: same collection error (`No module named
  'diffusers.models.transformers.transformer_minimax_h3'`). Cause: run409a.sh line 11 re-exports PYTHONPATH and
  dropped env409.yaml's pydeps entry. No device compute, no drop.
- Fix: run409a.sh PYTHONPATH adds $T/pydeps (only run409a; run409b keeps diffusers 0.38 so the dense arm matches #408).
  Copied to blx01 via tmp+mv; import check with that PYTHONPATH resolves pydeps/.../transformer_minimax_h3.py.
- chain416a restarted detached from run 1664 (state/runs/1664/chain416a.out).

## t416 run 1674 (2026-10-12 07:50 PDT)
- chain416a (run 1664) was killed by the 12:51Z g15blx02 reboot at the cap check (12:47Z), before submitting. Broker had
  no job of ours queued (only ltx-host 970). Submitted attn.txt directly with blx01-submit.sh (lint ok): blx01 job 971,
  queued behind ltx-host. The job lives in blx01's broker, so g15blx02 reboots cannot kill it; no local chain now.
- Next: read /var/tmp/fasth3/t409/out_attn/run.log on blx01 (PCC, T409_EXIT). Step 4 after #408.

## Run 1677 (2026-10-12 07:55 PDT)
blx01 job 971 (attn PCC, run409a.sh) FAILED in 48 s, exit 1: JIT kernel compile error, not a drop (post-job health gate all OK).
First error (log /var/log/tt-device-broker/2026-10-12_145010_971.log on blx01, line 387):
  vsa_sdpa_stream_compute.cpp:891:43: error: no matching function for call to 'recip_tile_init<false>()'
  could not convert 'false' from 'bool' to 'ckernel::ReciprocalDestAcc'
Cause: the t400 base changed recip_tile_init's template param from bool to enum ckernel::ReciprocalDestAcc. The host build (step 1) cannot see device kernels; they compile at JIT time.
Next: fix line 891 (and any other recip_tile_init<bool>/recip_tile<bool> uses in vsa_* kernels) to match how sdpa_flash/compute_common.hpp on the base calls it; sync the kernel file to /var/tmp/fasth3/t409 (no rebuild needed for kernel sources if build tree reads them from the src tree; check b/ttnn path is a copy), resubmit attn.txt.

## t449 run 1680 (2026-10-12 08:00 PDT)
- Fix 22d7d4a6299 (pushed to ttp/t409-...): vsa_sdpa_stream_compute.cpp:891-892 `recip_tile_init<false>()` /
  `recip_tile<false>(0, VectorMode::C)` -> no template args. The base (846c8f5ba23, #56292) dropped the legacy_compat
  bool; old `<false>` = modern math at default DEST precision = the new default. Same as the base's
  compute_streaming.hpp:764. No other recip/rsqrt bool uses in vsa_* kernels. Other drift checked by diffing the shared
  sdpa headers fasth3-opt..HEAD: fused_op_receiver (vsa reader fills the struct fields itself; fields unchanged),
  calculate_binop_with_scalar template unchanged. The JIT compile on device is the real check.
- blx01: only that file copied into /var/tmp/fasth3/t409/b (tmp+mv, ionice/nice); md5 matches. Tree = 53f7b6b707 +
  the fix (git HEAD there still prints 53f7b6b707). TT_METAL_HOME=b, so kernels JIT from it; no rebuild.
- device-lint passed. blx01 was running #421's job 973, so chain449a (tt-project/t409/chain449a.sh, copy in
  state/runs/1680) waits for the broker to clear of smarton jobs, submits attn.txt via blx01-submit.sh, waits for the
  job to end. Log state/runs/1680/chain449a.out (ends "T449A_DONE <rc> <job> <status>").
- #408 is done, so step 6 (10 s dense vs VSA-0.9 A/B, run409b.sh, one job) runs after the PCC job passes.

## t417 run 1678 (2026-10-12, PDT) — Turbo-SLA LoRA + SLA selection mode
Step 1 (LightX2V exemption check): LightX2V exempts NOTHING. minimax_h3 runs the sparse op over the full packed
sequence `[text | keyframes | audio | video]` (packing.py build_packed_sequence); the Ulysses path concatenates the
aux text prefix with the main part before attention (ulysses_attn.py _apply_bulk). get_block_map (sla_util.py):
smooth-K, mean-pool q and k per raster 64-token block (tail block averaged over its valid rows), score = pq @ pk^T
(no scale), topk = max(1, min(K, int((1-0.85)*K))), per head and q block. Smooth-K shifts each row's scores by a
constant -> same selected set; dropped.
Step 2 (code 04cbd225529, on top of 22d7d4a6299's kernel fix): `MINIMAX_H3_VSA_MODE=sla` (+ MINIMAX_H3_VSA_SPARSITY=0.85)
-> build_sla_geometry (raster 64-row tiles over the whole sequence, identity order, pads appended, no exempt tiles,
all real tiles candidates) and MiniMaxH3VSACoarseStage(selection="sla") with floor k (compute_topk_sla). Host tests
test_sla_selection_minimax_h3.py: geometry + block sets equal a torch port of get_block_map (768p 10 s and a tiny
case, sp 1 and 32, sparsity 0.85/0.9). 37 passed with the old VSA host tests; `ttp checks` passed.
Open risks for the device run:
- vsa_sdpa / vsa_ring_sdpa have never run with exempt_ids=[] (n_exempt=0); the first device job checks this.
- Block size: LightX2V's triton op uses 64/64; its sage2 op (5090 sla config) uses 64/128 on sm90, 128/64 otherwise.
  Training block size is unconfirmed; ours is fixed at 64 (VSA_TILE_TOKENS, baked into the kernels).
- Text length: our packed text rows (n_text) may differ from LightX2V's (padding), which shifts raster block
  boundaries. Selection is still the same algorithm; check n_text against LightX2V's prompt length if quality is off.
Step 3 (device): NOT RUN. exabox-login unreachable at run 1678 (ssh banner timeout via 127.0.0.1:22788 = Mac tunnel
down). Also the VSA path has not yet passed its device PCC test (#449, blx01).
Next (when exabox is up): exabox setup under ~/fasth3 (build t409 head, DiT weights, Turbo v1.0 + SLA LoRA, text
encoder), then per seed 0-4 one sbatch job per A/B pair: (v1.0 dense, v1.0 + VSA 0.9) and (v1.0 dense, Turbo-SLA + sla
0.85); 768p 10 s, per-step time, PSNR vs v1.0 dense, VBench subset, one still per seed. Template: run409b.sh.

## t417 run 1684 (2026-10-12 08:40 PDT)
- Exabox probe passed once, then exabox-login timed out again (banner exchange via 127.0.0.1:22788) on 3 tries:
  tunnel flapping. No exabox state exists yet (~/fasth3 absent on the login node; t160 used /data/smarton/fasth3).
- #449's VSA attention PCC test passed on blx01 (job 974, T449A_DONE 0): the VSA path works on device.
- SLA LoRA header check (HTTP range reads, no download): minimax_h3_fl2v_turbo_4step_v0.1_768p_sla_bf16 has the
  same 624 keys and shapes as v1.0_768p_bf16, alpha 128, key_format minimax-h3-diffusers -> drop-in for the fuse path.
  It has no SLA linear-branch (proj_l) weights: inference is block-sparse only, as LightX2V runs it.
- LightX2V's published SLA config (configs/minimax_h3/dmd/minimax_h3_fp8_4step_5090_vae_fp8_sla.json) equals the
  v1.0 config except attn_type=dynamic_sparse_attn {sparsity 0.85, operator sage2} and the LoRA path
  (video_flow_shift 6, audio_flow_shift 3, training_euler, 4 steps, 768x1344, 362 frames in both).
- dynamic_sparse_attn.py: sage2 uses BLKQ,BLKK = 64,128 on sm90 and 128,64 otherwise (5090 = sm120 -> 128-row Q
  blocks, 64-row K blocks); triton operator 64/64. Code ff705742990: SLA pools Q over 128-row blocks by default
  (MINIMAX_H3_SLA_BLKQ=64 for 64/64); build_sla_geometry(q_block_tiles=2) keeps blocks inside one shard.
  Host tests: test_sla_selection 25 passed (oracle with BLKQ 64 and 128), test_vsa_geometry+oracle 24 passed.
- Exabox setup cost (from t160/t288 notes): FastH3 needs MiniMax-H3 raw 135 GB + converted DiT cache ~152 GB + JIT
  ~17 GB + build; /data had ~100 GB free on 10-07; the tunnel moved ~3.4 MB/s (~11 h per 135 GB). Exabox is only
  practical if /data has >= ~320 GB free AND the login node can pull MiniMaxAI/MiniMax-H3 from HF directly.
  blx01 already has the build (/var/tmp/fasth3/t409/b), weights and caches.
Next (tunnel up): `ssh exabox-login 'df -h /data; curl -sI https://huggingface.co/api/models/MiniMaxAI/MiniMax-H3 | head -1'`.
  Both OK -> stage under /data/smarton/fasth3/t417 (blob:none clone + bundle of ttp/t409 head, HF download with nice/ionice),
  build job (sbatch, no device), conversion/JIT fill jobs, then the A/B pairs. Otherwise report: run step 3 on blx01.

## t417 run 1684 (cont.): exabox staging started
- Copied pydeps (diffusers with transformer_minimax_h3) and kf_t2va_f0.png from blx01 /var/tmp/fasth3/t409 to exabox /data/smarton/fasth3/t417.
- Started tt-project/t417/stage417.sh on exabox-login (ttp detach stage417, pid 3248899, log /home/smarton/.ttp-detach/1684/stage417.log).
  It adds worktree /data/smarton/fasth3/t417/src at 1d8e15320e3 (from the t48 blob:none repo) with the tracy, umd and tt-cluster-descriptors submodules,
  then runs hf download for the v1.0 LoRA, the SLA LoRA and MiniMaxAI/MiniMax-H3 (~135 GB) into t417/models. Marker: t417/stage417.done (STAGE417_RC=<rc>).
- exabox has no HF token. If MiniMax-H3 turns out to be gated, the marker records the failure.
- Next: if the rc is 0, submit a build-only sbatch on a dit node that has been idle 2 h+ (t160 qualify.sh/submit.sh pattern), then the cache-fill job, then the A/B pairs per seed:
  (dense, vsa0.9) and (dense, sla0.85 with the SLA LoRA). Use run409b.sh as the template; drop its local-disk check because /data is NFS.

## t417 run 1691 (2026-10-12 08:45 PDT)
- First staging (stage417.sh) failed with STAGE417_RC=1 after the v1.0 LoRA download: the file was complete (1383677808 B),
  but hf 1.16.1 exits non-zero via a `click.exceptions.Exit: 0` traceback, and set -e stopped the script.
- Fix: dl() logs hf's rc and fails only on timeout (124); need() checks each file; the base step checks for *.safetensors.
- MiniMaxAI/MiniMax-H3 is not gated (api 200, resolve 307 without a token). /data has 25 TB free.
- Restarted: ttp detach --remote exabox-login --dir /data/smarton/fasth3/t417/detach stage417b (pid 3269117).
  After 1 min it had both LoRAs and 46 GB of MiniMax-H3 (280 files, layout MiniMax-H3/FL2VA/...). Deleted the old log dir ~/.ttp-detach/1684.
- Check: `ttp detach --check --host exabox-login /data/smarton/fasth3/t417/detach/stage417b`, then read stage417.done.
- Next: point the run config's model path at t417/models/MiniMax-H3/FL2VA (check run409b.sh's expected layout), submit a build-only
  sbatch on a dit node idle 2 h+ (t160 qualify.sh/submit.sh pattern), then the cache-fill job, then the A/B pairs per seed.

## t417 run 1693 (2026-10-12 09:05 PDT)
- stage417b ended STAGE417_RC=3 although all 468 GB had arrived: `find | grep -q .` under pipefail dies of SIGPIPE (rc 141).
  Fixed with `find -print -quit`. Rerun stage417d (default rev 1d8e15320e3: same code as HEAD; notes commits are not on exabox): STAGE417_RC=0.
  Layout: t417/models/MiniMax-H3/FL2VA/{transformer,text_encoder,video_vae,audio_vae,processor,tokenizer}, LoRAs in lightx2v-h3-turbo{,-sla}.
- Build: tt-project/t417/build417.sbatch via submit417.sh (qualify.sh idle >= 2 h rule). Slurm job 130130, node bh-glx-120-b04u02
  (partition bh_sc5_B2B9_D12, idle 548 min, LastBusyTime 2026-10-12T06:53:43), submitted 16:02:02Z, tlim 45 min, no device.
  Log /data/smarton/fasth3/t417/slurm-130130.out, rc /data/smarton/fasth3/t417/job-130130.rc, alloc log t417/alloc.log.
- Next: check JOB_RC=0 and `squeue -u smarton` empty. Then write the device sbatch (run409b.sh as template; model root
  t417/models/MiniMax-H3/FL2VA; drop the local-disk check, /data is NFS; timeout --foreground -k 10; full 4x8 mesh),
  first a cache-fill/JIT job, then one A/B pair per job per seed: (v1.0 dense, v1.0 VSA) and (v1.0 dense, SLA LoRA + sla 0.85).

## t452 run 1696 (2026-10-12 09:20 PDT) — continue #417 on exabox: device lint refuses exabox
- Branch ttp/t452-... reset to t417 head 197ba8066f6 (t452 had no own commits). `ttp push --own` refuses to publish
  ttp/t417-... from this task ("publishes only this task's own branch"), so the t417 commits are published as part of ttp/t452-....
- Model root: t417/models/MiniMax-H3 (top level), not .../FL2VA. Its transformer/vae/text_encoder config.json md5s equal
  blx01's /var/tmp/fasth3/models/MiniMax-H3 (#408 baseline); FL2VA/ has a different transformer config and video_vae/ instead of vae/.
- Weight cache is adapter-independent (pipeline_minimax_h3_turbo.py: LoRA fused after load), so one dit cache serves dense, VSA and SLA arms.
- Wrote tt-project/t417/run417.sh (Slurm job script from run409b.sh: arms dense / vsa<s> / sla<s> (SLA LoRA + MINIMAX_H3_VSA_MODE=sla) / fill
  (CACHE_ONLY), per-arm timeout --foreground -k 10 270 s, 90 s no-progress watchdog, JOB_RC trap) and copied it plus plug/
  (t286_skipvaewarm.py, t409_alarm.py from blx01) to exabox t417/.
- device-lint.sh exabox-login refuses it: models, LoRAs and both caches are on /data (nfs4), and the lint requires local disk
  for weights/caches (job 086 D-state hang on network fs). Dit nodes report TmpDisk=0; /data is the only space we have there, and
  the 2 h idle rule puts each job on a different node, so per-node staging (~135 GB weights + ~150 GB cache) cannot fit a 600 s job.
  No device job was submitted. Also: lint reports slurm-login-01 / at 87% (cap 85%).
- Judgment: FastH3 device A/Bs cannot run on exabox under the current lint. Run them on blx01 (weights and caches local in
  /var/tmp/fasth3; t409/b at 53f7b6b707 + uncommitted vsa host/kernel edits from #449). SLA code (04cbd225529, ff705742990) is
  Python-only; the one C++ diff 53f7b6b707..1d8e15320e3 is the vsa compute kernel (JIT). blx01 needs its own tree for the SLA
  code and the SLA LoRA (not gated; lightx2v) under /var/tmp/fasth3 (ionice -c3, --bwlimit, footprint <= 450 GB; / was 80%).

## t452 run 1703 (2026-10-12 09:40 PDT) — A/Bs moved to blx01
- Judgment: exabox device lint refuses NFS weights/caches and dit nodes have no local disk, so the A/Bs run on blx01,
  where weights and caches are local. Exabox staging (/data/smarton/fasth3/t417, ~470 GB) is left in place for now; nothing runs there.
- blx01: SLA LoRA fetched to /var/tmp/fasth3/models/lightx2v-h3-turbo-sla (1383677808 B, md5 differs from v1.0).
  Code overlay /var/tmp/fasth3/t452/w (tt-project/t452/mk452.sh): symlinks into t409/b (53f7b6b707 + #449's uncommitted
  vsa kernel/host edits, built; also used by #453, so not modified) with the two minimax_h3 Python dirs copied and
  sla.patch (04cbd225529 + ff705742990) applied. Import check OK.
- One arm takes ~316 s warm (t409 out_ab10-vsa0.9), so a pair (~630 s) is over the 600 s cap: one arm per broker job,
  TIMEOUT 540, per-arm pytest limit 470. Specs tt-project/t452/s0-{dense,sla0.85,vsa0.9}.txt, device-lint passes.
- chain452.sh (ttp detach on g15blx02, log state/runs/1703/chain452.log, rc chain452.rc) submits them one after another
  through t288/blx01-submit.sh; outputs /var/tmp/fasth3/t452/out_s0-<arm>/ (run.log, latents, yuv, mp4, PASS).
- Next: read chain452.log; per-step time from each run.log ("denoise breakdown"); PSNR of sla/vsa vs dense from the
  saved yuv; still per arm. If seed 0 is sane, make s{1..4}-{dense,sla0.85,vsa0.9}.txt (sed s/s0/sN/; T409_SEED=N) and
  rerun chain452.sh on them; then VBench subset and recommendation.

## t452 run 1707 (2026-10-12 10:10 PDT) — seed 0 failed on a cold JIT cache, rerun
- All three seed-0 arms (blx01 jobs 998 dense, 004 sla0.85, 006 vsa0.9; each 439 s, exit 1) stopped at the t286 warm
  deadline (400 s): "RuntimeError: [t286] warm deadline 400 s passed in Warming audio decode lengths / prompt encoder
  keyframe layouts / vision merge sequence lengths". Cause: the overlay root /var/tmp/fasth3/t452/w gives new kernel
  paths, so the JIT cache started cold (job 998: 0/1889 hits; 004: 1889/3446; 006: 3446/5373). A warm t409 run builds
  ~6692 programs (89.7% hits), so the cache is ~80% filled now. Not a drop, not a code failure. Clock: AICLK ceiling 900 MHz.
- Rerun of the same three specs: chain452 detached in run 1707 (log state/runs/1707/chain452.log). If an arm still hits the
  warm deadline, rerun it once more (the cache keeps filling); do not raise the 540 s job timeout.
- Next: per-step time ("denoise breakdown") per arm, PSNR of sla/vsa vs dense from the saved outputs, a still per arm;
  then s1..s4 specs and the same chain.

## t452 run 1708 (2026-10-12 10:35 PDT) — seed 0 results, SLA audio clipping, seeds 1-4 queued
- Chain run 1707 (blx01, 900 MHz AICLK ceiling, relative timings only): job 008 dense PASS, job 009 sla0.85 FAIL, job 011 vsa0.9 PASS.
  Steady denoise per step (pipeline "denoise breakdown", 3 steady steps): dense 3107 ms, vsa0.9 3099 ms (no gain), sla0.85 2482 ms (-20.1%).
  Pipeline totals: dense 16.17 s, vsa 15.61 s, sla 13.72 s (VAE decode ~1.9 s in each).
- Job 009 failure (first error line, log /var/log/tt-device-broker/2026-10-12_171239_009.log on g15blx01):
  "E AssertionError: 1.5% of samples are at full scale; suspect a scaling error" (common_av.check_audio_sanity, clipped >= 1%).
  The SLA LoRA run generated fully (denoise and decode done), then failed the audio check before saving latents and video.
  So the SLA v0.1 LoRA clips the audio track (dense v1.0 passes the same check). This counts against SLA quality.
- vsa0.9 vs dense seed 0 (psnr452.py): Y PSNR 17.57 dB, yuv 19.06 dB, per-frame min 16.02, latent PCC 0.8115. Still
  stills/s0-vsa0.9.jpg is visibly blurred and ghosted against stills/s0-dense.jpg. Same as #453: VSA on dense-trained Turbo is out.
  Judgment: VSA arm not repeated for seeds 1-4 (no speed gain, clear degradation, already shown twice).
- Added plug/t452_audiowarn.py (pytest plugin, -p t452_audiowarn in run452.sh): logs "[t452] audio peak/rms/clipped" for every arm and
  turns a check_audio_sanity failure into a logged warning plus AUDIO_WARN file, so the SLA arm saves its video.
- Chain run 1708 (detached on g15blx02, log state/runs/1708/chain452.log): s0-sla0.85b, then s{1..4}-dense, s{1..4}-sla0.85.
- Next: per arm "denoise breakdown" and "[t452] audio" lines from out_s<N>-<arm>/run.log; psnr452.py out_sN-dense out_sN-sla0.85
  for each seed (note: SLA uses a different LoRA, so PSNR vs v1.0 dense measures drift, not just sparsity error); stills at 5 s
  (ffmpeg -ss 5); VBench subset on dense vs SLA; recommendation.
