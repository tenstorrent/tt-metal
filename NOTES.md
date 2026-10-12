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
## t449 run 1681 (2026-10-12 08:10 PDT)
- Attention PCC PASSED: blx01 job 974 (chain449a, 14:58Z), test_vsa_sparsity0_matches_ring_path[4x8sp1tp0nl2_ring_is_fsdp0]:
  `PCC = 99.9654 %, CCC = 99.9648 %, RMSE/σ₁ = 2.7 %` (>= 0.9995 gate). 2 WH-only params skipped. No drop.
- A/B sizing: #408's warm 10 s dense run (out_t10, seed 0, 900 MHz clamp) took 405 s process wall (344 s construction
  warm-up, 3110 ms/step steady). Two arms do not fit one 600 s job, so one arm per job. Our tree = #408's commit
  bff7d82040c + the VSA port (dense path unchanged when MINIMAX_H3_VSA_SPARSITY is unset), so the VSA-0.9 arm is
  compared with #408's out_t10 (same box, seed, inputs, clock; latents saved there). Dense rerun on t409/b only if needed.
- Copied kf_t2va_f0.png from t408 to t409 (md5 match; run409b.sh needs it there).
- Spec tt-project/t409/ab10vsa.txt (run409b.sh ab10 vsa0.9, T409_PY_TIMEOUT=560, -t 600 unmeasured). Lint ok.
  chain449a (T449_OUT=out_ab10-vsa0.9) detached from run 1681 waits for #421's job 975, then submits.
  Result: /var/tmp/fasth3/t409/out_ab10-vsa0.9/run.log on blx01 (denoise breakdown, timing table, latents, mp4).

## t449 run 1682 (2026-10-12 08:20 PDT)
- chain449b submitted blx01 job 977 (15:11Z): FAILED, not a drop. Last lines: `RuntimeError: [t286] warm deadline 400 s
  passed in Warming audio decode lengths`, process wall 420 s, T409_EXIT=1. Cause: cold JIT cache for the t409/b tree
  (JIT cache stats 96/1889 hits, 5.1%; #408's out_t10 had 78.6%). Audio decode warm took ~60 s/length vs 4 s in #408;
  only 6 of 12 lengths done. Prompt-encoder and denoise warm (incl. VSA kernels) never started. Kernels compiled so far
  stay in TT_METAL_CACHE, so reruns get further (the t286 plugin is built for this).
- ab10vsa.txt now ID t449-ab10-vsa0.9-r2 with T286_WARM_DEADLINE_S=530 (compiles more per job) and T286_WARM_LATE_S=400
  (fails cleanly if warmup ends too late to generate). Submitted directly: blx01 job 979 (15:20Z).
- Next: read /var/tmp/fasth3/t409/out_ab10-vsa0.9/run.log on blx01. PASS file there = arm done (compare denoise
  ms/step with #408 out_t10: 3110 ms/step dense). Warm-deadline/late failure again = resubmit the same spec with a new ID.

## t449 run 1690 (2026-10-12 08:40 PDT)
- blx01 job 979 (VSA-0.9 10 s arm, 15:20Z) FAILED, exit 124: killed by its 560 s pytest timeout, process wall 571 s.
  Not a drop. All 12 audio lengths compiled this time (+2180 programs, lengths 7-12 ~60 s each cold). Then it was
  killed inside the first of 3 prompt-encoder layouts (started 15:27:40Z). The t286 deadline is only checked between
  warm items, and a single cold item outlasted the time left.
- Box event: blx01 fabric-check 980 (15:30Z, right after 979) FAILED: `Device 0: Timed out while waiting for
  active ethernet core 27-25 to become active again`. Broker hold 981 then "device HELD (degraded): eth/fabric fault",
  and the broker is recovering on its own. Likely cause: our python was killed by TERM with the mesh open (977's clean
  failure was followed by a passing fabric check, 978). We did not reset anything.
- Fix: tt-project/t409/plug/t409_alarm.py (-p t409_alarm, T409_ALARM_S = T409_PY_TIMEOUT-35 = 525 s by default)
  raises RuntimeError in the main thread, so pytest teardown closes the mesh before timeout's TERM. CPU check: a
  sleeping test fails with the alarm and its finally/teardown runs. run409b.sh adds $T/plug to PYTHONPATH. Both
  copied to blx01 via tmp+mv (md5 match).
- Spec ab10vsa.txt ID -> t449-ab10-vsa0.9-r3 (same CMD). Lint ok. chain449a (detached, run 1690) waits for blx01 to
  leave HELD and for its queue to have no smarton job (job 982 `true`, not ours, was queued), then submits.
- Next: read the T449A_DONE line in state/runs/1690/chain449a.out. Warm-deadline/alarm failure = resubmit (new ID);
  each job compiles more. PASS = compare denoise ms/step with #408 out_t10 (3110 ms/step dense, seed 0, 900 MHz).

## t449 run 1692 (2026-10-12 08:46 PDT)
- blx01 job 986 (VSA-0.9 10 s arm r3, 15:37Z) FAILED, not a drop, not a timeout. Warm-up finished (JIT 71.3% hits,
  wall 391 s); first denoise step died: `TT_FATAL @ vsa_sdpa_device_operation.cpp:96: attrs.exempt_ids.size() <= 32`,
  `at most 32 exempt block ids (got 71)`. The 10 s fl2va prefix (text + 2 condition frames + audio) is 71 tiles.
  The alarm plugin did its job: clean pytest teardown, mesh closed.
- Fix 6d8503e9b71: cap is now `exempt_ids.size() <= n_kv_blocks` in vsa_sdpa and vsa_ring_sdpa validate. The reader
  reads the ids from runtime args (no fixed array); tt-metal checks the RT-arg size itself (kernel.cpp, 4096/L1 fit).
  New test test_vsa_ring_sdpa.py::test_vsa_ring_sdpa_many_exempt (71 exempt ids, 13 blocks/shard, torch reference).
- blx01: the 2 .cpp + test copied into t409/b (tmp+mv, md5 match); incremental Release build build449
  (tt-project/t449/build449.sh, detached remote, log /var/tmp/fasth3/t409/build449.log, rc build449.rc).
- chain449c (detached here, log state/runs/1692/chain449c.out, ends "T449C_DONE <rc> <step>"): waits build449.rc=0,
  then spec tt-project/t449/u449.txt (run449u.sh: many_exempt + odd_blocks on nl2, out_u449/run.log), then if it
  passes spec t409/ab10vsa.txt r4 (out_ab10-vsa0.9/run.log).
- Next: read chain449c.out. ab10 PASS = compare denoise ms/step with #408 out_t10 (3110 ms/step dense, seed 0,
  900 MHz), then PSNR of VSA latents/video vs out_t10, look at a frame.

## t449 run 1694 (2026-10-12 09:10 PDT)
- chain449c done, no drops. Unit job blx01 989 (15:52Z): test_vsa_ring_sdpa_many_exempt and _odd_blocks PASSED
  (4x8sp1tp0nl2). 10 s VSA-0.9 arm r4: blx01 job 990 (15:55Z) PASSED, process wall 316 s, JIT 89.7 % hits.
  Tree on blx01 = commit 53f7b6b707b (before t417's SLA commits).
- Speed: `denoise breakdown: ... steady 9.3s over 3 steps (3099 ms/step)`, Denoising (per step) 3.1837 s.
  Dense #408 out_t10 (same box, seed 0, 900 MHz clamp): 3110 ms/step, 3.1481 s. VSA-0.9 is NOT faster (-0.4 % steady).
- Quality vs dense out_t10: latents PCC 0.8116, PSNR 25.67 dB; decoded video (yuv420, 243x1152x1344) PCC 0.685,
  PSNR 19.06 dB. Frames 120/240 (tt-project/t449/vsa09_vs_dense_f120_f240.jpg; full PNG on blx01 /var/tmp/fasth3/t409/cmp449/grid.png, left dense, right VSA): heavy blur,
  ghosting, colour smear. Clearly degraded. Expected in part: v1.0 Turbo adapter was trained dense; LightX2V runs
  sparse only with the SLA LoRA (t417 notes). The missing speedup is a separate problem: attention share of the
  10 s step and the VSA kernel's per-block cost need a profile.
- Push of run 1692 was refused (exit 2): #417 had pushed 1d8e15320e3 to the t409 branch. Rebased (NOTES.md union).

## t453 (2026-10-12): why VSA-0.9 gives no speedup and degrades on Turbo fl2va 10 s

CPU read (no device):
- 10 s fl2va geometry (prefix 1460/1008/405, grid 72x24x42, sp 8, interleaved): 1240 tiles (6 pad), padded_len 79360,
  46 exempt tiles (not 71: the 71 came from a warmup layout), 1188 candidates, k=119. Sparse (q,kv) tile pairs are
  0.166 of dense; exempt (dense) rows are 22.5% of the sparse work. The selection really is sparse.
- fasth3-opt's 2x was on the 50-step base, 15 s, partly from clock throttling (dense 975 MHz median vs VSA 1268 MHz).
  Its VSA was never quality-checked vs dense (VSA_SCOPE.md: "v0's bar is numerical parity").
- Quality: dense-trained checkpoint + dense-trained Turbo LoRA, no gate_compress weights. LightX2V runs sparse only
  with the SLA LoRA (t417).

Device timing job: test models/tt_dit/tests/models/minimax_h3/test_vsa_timing_minimax_h3.py (commit e0ad5c7d6d2),
copied to blx01 /var/tmp/fasth3/t409/b; run script tt-project/t453/run453t.sh -> /var/tmp/fasth3/t409/run453t.sh,
spec tt-project/t453/t453.txt. blx01 broker job 992 (-t 600, py timeout 540, alarm plugin). Output
/var/tmp/fasth3/t409/out_t453/run.log; grep 'T453' for totals and per-op breakdowns (dense, vsa0.9_two_op,
vsa0.9_ring).
Next: read the breakdown; if a fix is clear, commit it and rerun one VSA arm vs dense, seed 0.

## t453 run 1700 (2026-10-12 09:20 PDT)
- blx01 job 992 failed at collection (exit 2, 11 s, no device work): `ModuleNotFoundError: No module named
  'diffusers.models.transformers.transformer_minimax_h3'` (blx01 venv has stock diffusers 0.38.0). Not a drop.
- Fix 7710e5c41bf: timing test builds its own random attention state dict and setup, no diffusers import.
  Copied to blx01 t409/b (tmp+mv, md5 83ca5c37...). Collect-only on blx01: 2 tests (two_op, vsa_ring) for 4x8sp1tp0nl2.
- Resubmitted as spec ID t453-attn-timing-r2: blx01 broker job 994 (-t 600). Log /var/tmp/fasth3/t409/out_t453/run.log.
- Next: grep 'T453' in that log; compare dense vs vsa0.9 totals and the per-op split (coarse stage, all-gathers,
  vsa_sdpa / vsa_ring_sdpa vs ring_joint_sdpa). Then push the branch (ttp push --own --detach).

## t453 run 1701 (2026-10-12 09:30 PDT)
- blx01 job 994 PASSED (16:15Z, wall 49 s, no drops). One attention layer at the 10 s shape (random weights, 900 MHz
  clamp, relative only), best of 3: dense module 47.2 ms (ring_joint_sdpa 39.7 ms); VSA-0.9 two-op 53.8 ms
  (vsa_sdpa 30.1, K/V all-gathers 2x4.5, coarse 2.9, AG 0.6); vsa_ring 48.7 ms (vsa_ring_sdpa 35.4, coarse 3.8, AG 0.9).
- BUT job 994's test built the geometry with build_vsa_geometry's default placement (identity): all 46 exempt rows on
  shard 0 -> shard 0 does 0.39x dense work (others 0.13x). The pipeline (job 990) uses interleaved (6/6/6/6/6/6/5/5,
  max shard 0.167x dense). Fix ae7b83c6ff6: the test uses MiniMaxH3VSAConfig().placement.
- Resubmitted as blx01 broker job 995 (spec t453-attn-timing-r3, -t 75, alarm 62 s). Same log path out_t453/run.log.
- Next: grep T453; if interleaved vsa_sdpa is still far above 0.17x of dense, the fine kernel's per-pair cost at this
  shape (k=119 + 46 exempt per row, 10 dealer-dense rows/shard) is the cause; plus 12 ms/layer fixed overhead
  (2 K/V all-gathers + coarse) that dense ring SDPA hides inside its ring.
- blx01 job 995 PASSED (16:18Z, wall 37 s, 900 MHz clamp, no drops). Interleaved placement (pipeline's), one layer,
  best of 3: dense 47.0 ms (ring_joint_sdpa 39.0); VSA-0.9 two-op 42.7 ms (-9%: vsa_sdpa 19.2, K/V all-gathers
  2x4.7, coarse 3.7, AG 0.8); VSA-0.9 vsa_ring 37.2 ms (-21%: vsa_ring_sdpa 23.6, coarse 3.5, AG 0.7).
- Root cause of no speedup (speed): (a) fine kernel is only 2.0x faster than dense ring SDPA for 0.167x of the tile
  pairs (per-pair cost ~3x dense: per-row gathered KV, 46 exempt ids + k=119 per row, dense rows); (b) the two-op path
  (pipeline default, ring=False; job 990) adds ~14 ms/layer of non-overlapped work (two blocking K/V all-gathers 9.4,
  coarse 3.7, AG 0.8), eating 2/3 of the 19.8 ms saving. Predicted two-op e2e: 50 x 4.3 = ~215 ms/step (~7%); job 990
  saw 11 ms, so ~200 ms/step more is lost elsewhere in the e2e (padded_len 79360 vs dense bucket 76800 = +3.3% rows
  for all non-attention ops is only ~25 ms of it; unexplained rest).
- Quality: not a kernel bug (unit tests and oracle pass); dense-trained Turbo checkpoint/LoRA run sparse without the
  SLA LoRA. Not fixable by placement or ring (same selection).
- e2e check: blx01 job 996 = 10 s VSA-0.9 with VSA_RING=1, seed 0 (spec tt-project/t453/ab10ring.txt, -t 480).
  Output /var/tmp/fasth3/t409/out_ab10ring-vsa0.9/run.log. Dense ref: /var/tmp/fasth3/t408/out_t10 (3110 ms/step).
- Next: grep 'denoise breakdown' and 'Denoising (per step)' in job 996's log; ring e2e should be ~-490 ms/step if the
  per-layer gain carries over. PSNR/PCC of latents and video vs t408/out_t10 (same as cmp449). Then ttp push --own --detach.
