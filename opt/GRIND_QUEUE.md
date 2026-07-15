# GRIND QUEUE — concrete unmeasured experiments (the loop's never-empty work source)

**The loop NEVER concludes "exhausted."** Each lap: if no experiment is in flight, dispatch the next
`[ ]` item below (in-budget, completes, real data), mark it `[~]` in flight, and on completion record the
number here + PROGRESS LIVE LOG and tick it `[x]`. When ALL are `[x]`, generate the next batch by
re-profiling the current dominant op and enumerating its config space — do not stop.

All device runs via broker, owner `[claude]smarton`, `timeout_sec<=240`, in-budget (block/op harness, NOT
full pipeline — full pipeline cold-compiles >280s and times out even with prewarm; measure denoise deltas
per-block, VAE via prof_vae_ltx.py). Env: TT_METAL_HOME/PYTHONPATH=worktree, caches tt-*-ltxrt, PINNED=0.

## Batch A — SDPA chunk sweep (SDPA = 30% of block = 6.13ms; baseline chunk (192,512)=4.85ms; NEVER swept)
Measure RingJointSDPA FW (largest-FW op in the profile) per chunk. Test:
`test_ring_joint_attention.py::test_ring_joint_sdpa -k "ltx_s2 and bh_glx and 8rpx4up and no_trace_no_check and <qID> and <kID>"` under `tracy -p -r`.
**⚠️ HARNESS BUG — the REAL no-data cause (2026-07-11 21:56Z, from job 215000-35 FULL log): `tracy` DESTROYS a
multi-word `-k`. `tools/tracy/__main__.py:402` does `osCmd=" ".join(originalArgs)` then `:415` re-runs it under a 2nd
`shell=True` → the quotes around `-k 'ltx_s2 and bh_glx and …'` are lost → pytest sees `and` as a positional →
`ERROR: file or directory not found: and` → `collected 0 items`. This hits the driver's runs IDENTICALLY (also ~6.1s /
0 tests) ⇒ every prior Batch-A job (driver's + my 215000-35) collected NOTHING = ZERO data. My earlier "wh_glx wrong mesh"
theory was NOT the operative cause (both meshes fail at collection first). FIX = select by exact node-id BRACKET (no spaces,
survives tracy's join), e.g. `::test_ring_joint_sdpa[bh_glx-line-no_trace_no_check-k512-q128-8rpx4up-ltx_s2]` (exact id
being confirmed via collect-only job 215404-37). Secondary correctness (only matters once collection works): use `bh_glx`
=[(8,4),2] (num_links=2, has ltx_s2), NOT `wh_glx`=[(8,4),4] (num_links=4 HW-capped on BH, no ltx_s2). `SDPA_SWEEP_TAG` is a
no-op; run_test_ring_joint_sdpa prints no FW → extract SDPA device FW from newest `generated/profiler/reports/<ts>/
ops_perf_results_*.csv`. Do NOT `rm -rf .logs` (cold tracy recompile → 220s timeout, killed 213135-31). ALL prior Batch-A VOID.**
EXACT node-id format (from collect-only 215404-37, 384 ltx_s2 cells): `test_ring_joint_sdpa[blackhole-<mesh>-line-<check>-<kID>-<qID>-<parallelID>-ltx_s2]`.
Target cell dispatched: `...[blackhole-bh_glx-line-no_trace_no_check-k512-q128-8rpx4up-ltx_s2]`.
- [x] q128 k512  → **SDPA FW median 4.74ms / mean 4.91ms** (n=32, job 215546-39, JIT 173/175 warm, PASSED). vs 4.85ms baseline = within ±2.3% = **NO WIN** (>5% gate not met). Env-file dispatch: broker param is `env:` (not `env_file`), `opt/env_sdpa.yaml`.
- [x] q256 k512  → **SDPA FW median 5.335ms / mean 5.359ms** (n=32, tight 5.30–5.55, job 220638-41, JIT 170/175 warm, PASSED). vs 4.85ms baseline = **+10% SLOWER = NO WIN**. Trend now clear: q128 4.74 < baseline(q192) 4.85 < q256 5.34 ⇒ **smaller q is better; q256 direction is dead**.
- [x] q64  k512  → **SDPA FW median 7.866ms / mean 7.886ms** (n=32, tight 7.79–7.98, job 220953-43, JIT 170/175 warm, PASSED). vs 4.85ms baseline = **+62% SLOWER = DEAD**. Small-q direction REFUTED: q64 over-chunks (dispatch overhead dominates). U-shape confirmed → **q128 is the minimum** (7.87 q64 ≫ 4.74 q128 < 4.85 base < 5.34 q256). Small-q reorder abandoned.
- [x] q128 k256  → **SDPA FW median 4.981ms / mean 5.158ms** (n=32, bimodal 16@~4.75 + 16@~5.1–6.0, job 221355-47, JIT 170/175 warm, PASSED). vs 4.85ms baseline = **+2.7% SLOWER = NO WIN**. k-axis now clear at q128: k512 4.74 < base 4.85 < k256 4.98 ⇒ **larger key-chunk better; smaller-k dead**. (q128,k512)=4.74 remains the global min (−2.3%, <5% gate).
- [x] q128 k128  → **SDPA FW median 5.884ms / mean 5.905ms** (n=32, tight 5.87–6.05, job 221731-49, JIT 170/175 warm, PASSED). vs 4.85ms baseline = **+21% SLOWER = DEAD**. k-axis at q128 fully closed: k128 5.88 > k256 4.98 > k512 4.74 ⇒ **larger key-chunk better; smaller-k dead** (mirrors q-axis U-shape). (q128,k512)=4.74 remains the global min (−2.3%, <5% gate).
- [x] q256 k256  → **SDPA FW median 5.882ms / mean 5.887ms** (n=32, tight 5.88–5.98, job 222219-51, JIT 170/175 warm, PASSED). vs 4.85ms baseline = **+21% SLOWER = DEAD**. Confirms the doubly-dead corner (q256 dir +10% × smaller-k dead). (q128,k512)=4.74 remains the global min (−2.3%, <5% gate).
- [x] q256 k128  → **SDPA FW median 7.368ms / mean 7.367ms** (n=32, very tight 7.36–7.37, job 222548-53, JIT 170/175 warm, PASSED). vs 4.85ms baseline = **+52% SLOWER = DEAD**. Closes the q256 row: k128 7.37 > k256 5.88 > k512 5.34 (same larger-k-better U-shape).
→ **BATCH A COMPLETE (full grid swept, all 7 cells).** Global min = **(q128,k512)=4.74ms, only −2.3% vs 4.85ms baseline (<5% act gate) ⇒ SDPA chunk offers NO shippable no-finetune win.** Both axes are U-shaped with baseline (q192,k512) near-optimal: q64 7.87 ≫ **q128 4.74** < q192 4.85 < q256 5.34 (q-axis @k512); k128 5.88 > k256 4.98 > **k512 4.74** (k-axis @q128). Nothing to wire into attention_ltx.py.

## Batch B — SDPA compute-config sweep (same harness, vary fidelity/accum)
- [x] SDPA LoFi + fp32-acc  → **DEAD (both counts)**. Job 223059-55, node-id `[blackhole-bh_glx-line-no_trace_check-k512-q128-8rpx4up-ltx_s2]`, env `LTX_SDPA_FIDELITY=LoFi LTX_SDPA_FP32ACC=1`. (1) **FAILS PCC gate** — `assert passing` (test:576, threshold 0.994) tripped (exact PCC at logger.debug, not captured). (2) **No speedup** — SDPA FW median **4.673ms** (n=32) vs HiFi2 q128k512 4.74ms = ~1.4%, within noise ⇒ **SDPA is NOT math-fidelity-bound** (HiFi2→LoFi barely moves FW). ⚠️ `LTX_QUANT` is a **no-op** in this test (never read); compute config hardcoded at test:384-389 → needed a temp env-gate source edit, **REVERTED** (no win + no PCC pass). fp32acc-alone = HiFi2 fidelity + extra acc cost ⇒ would PCC-pass but run *slower* ⇒ **whole fidelity axis closed, no further point worth a run.**
- [x] exp_ring_joint_sdpa variant (in-tree experimental kernel) — **DEAD: structurally incompatible with ltx_s2 at any performant chunk.** Probe **225026-62** (plain, non-tracy) crashed at **op validation** (SIGABRT, `1 failed in 42.61s`, NOT a hang/compile-timeout — post-job health-gate clean, no reset): `ExpRingJointSDPADeviceOperation::validate_on_program_cache_miss` → **"Q chunks per head (num_local=26+num_joint=0=26) exceeds SDPA grid columns (11) on device grid 12×10. Increase q_chunk_size or reduce sequence length."** The exp op maps one Q-chunk per grid column (≤11), so ltx_s2 (seq 38760 → 26 chunks at q≈192) can only run if q_chunk grows ≥~2.4× (into the ≥q454 regime). **Batch A already measured that direction monotonically SLOWER** (q256 +10%, q-axis U-shaped with q128=4.74 the min) ⇒ the exp kernel can ONLY run where it's already proven to lose. No win path; topology confound (Ring vs Linear) moot. Test-only scaffold commit fd2d0b2e1df kept (harmless, documents the attempt). **⇒ BATCH B CLOSED (both items DEAD): SDPA is neither chunk- nor fidelity- nor exp-kernel-improvable.**

## Batch C — VAE decode sub-levers (prof_vae_ltx.py, NOT full pipeline; VAE stage = 1.0s)
- [x] W-mask fold — **SCREENED: real 59ms/10.7% cost bucket, but mask is LOAD-BEARING ⇒ needs an output-preserving in-kernel fold, NOT a drop.** Two device jobs, both warm (JIT ≥404/404, plain non-tracy):
  - **COST** (job 230359-67, env-gate `LTX_VAE_SKIP_WMASK=1` skips the mul): TRACED wall **492.80ms** vs baseline **551.91ms** (job 225635-65) = **−59.1ms (−10.7%), clears the 5% gate 2×** — the FIRST live no-finetune device-compute lever this session (everything prior DEAD/marginal). This is the UPPER BOUND (mask entirely free); the real in-kernel fold win is ≤59ms.
  - **CORRECTNESS** (job 231013-69, `test_prof_vae_ltx_wmask_pcc` — TT mask-on vs mask-off gather+PCC, NO torch ref needed): **WMASK_AB_PCC=0.9588 = FAIL-loadbearing.** Dropping the mask drops output PCC to 0.959 (≪0.999) ⇒ pad-column garbage DOES leak into logical columns ⇒ naive drop is INCORRECT. Prof harness canNOT torch-PCC-gate (random weights, no host forward); the targeted `test_ltx_causal_conv3d` calls forward with `logical_w=0` so the mask never fires there — the TT-vs-TT A/B is the only in-budget correctness gate.
  - Both measurement scaffolds REVERTED (no shippable Python-level change; mask can't be dropped). Finding is the receipt.
- [~] **W-mask IN-KERNEL fold (the actual ≤59ms lever)** — DESIGN COMPLETE (2026-07-11 23:16Z lap, source dug end-to-end); needs a WARM authoring session, NOT a fire-and-exit cron lap-tail (real C++ + full `build_metal`). Fold the width pad-column zeroing INTO `neighbor_pad_async` so output is bit-identical to the pre-conv `ttnn.mul` (`vae_ltx.py:236`) without the separate full-tensor HBM round-trip.
  **Source map (all read):** vae mul → `models/tt_dit/models/vae/vae_ltx.py:230-239`; op entry `ttnn.experimental.neighbor_pad_async` w/ existing `logical_h`; plumbing chain `parallel/manager.py:331/363` → `neighbor_pad_async.cpp/.hpp` → device op `.cpp/.hpp` + `_types.hpp` (attr struct + hash + reflection @101/120) → nanobind `.cpp:61` → program factory `_program_factory.cpp:633-647,796-815` → kernels `local_copy_writer.cpp:39-42,67-79` (interior mask) + `phase2_w_reader.cpp:64-90` (W-halo boundary mask).
  **⚠️ KEY FINDING — NOT a clean `logical_h` mirror (this is the hard part):** `logical_h` masks whole ROWS by the halo-dim row index (`t = linear_row % input_halo_dim_size`, zero when `device_h_offset + t >= logical_h`). The vae W-mask is ORTHOGONAL — it zeros interior W COLUMNS by their W position *within* each row, and fires on `width_parallel.factor>1` **even when this conv doesn't halo W (or halos nothing)**. So it can't just piggyback the halo row-mask: it needs a NEW per-stick W-index mask in `local_copy_writer`'s inner `iter` loop (`iter`→global W index via a computed `device_w_offset`, zero when `device_w_offset + w >= logical_w`) AND correct handling of the no-W-halo case (the mul currently fires independent of `neighbor_pad` being called at all — folding into neighbor_pad MISSES convs that mul-mask but don't halo, so verify every W-masked conv also halos, or keep a fallback). The orthogonal stick↔W-index mapping across both halo-dim cases (H-halo: sticks=W cols; W-halo: different) is the correctness risk — needs a warm iterate-and-PCC loop, not a blind one-shot build.
  **Gate:** same TT-vs-mask-on A/B (`test_prof_vae_ltx_wmask_pcc`, PCC≥0.999) + wall delta (win in 0–59ms). Highest-value in-repo lever left; ≤59ms = ≤0.06s E2E (polish, does not move the ~7.9s floor).
- [x] depth-to-space permute: profile the 4 upsample reshape→permute→reshape; is a fused kernel cheaper? (~84ms device bucket). **DEAD (un-dispatchable in the lap budget): tracy VAE profile cold-compile TIMED OUT.** Job **234410-71** (2026-07-11 23:44Z, `test_prof_vae_ltx_trace` under `tracy -p -r`, TRACE_ITERS=5, pytest 270 / broker 290) ran the full 290.5s to the broker cap with **ZERO pytest output** and **produced no CSV** (nothing newer than 22:32 in `generated/profiler/reports`) ⇒ killed mid cold profiler-instrumented compile of the VAE decoder (~404 programs = a distinct cold `build_key`, not the warm plain-harness build). Post-job health-gate snapshot OK (32 chips), device auto-recovered clean, queue empty. Confirms the prior laps' rejection: the tracy-VAE path cold-times-out even at 270/290 headroom, and the prewarm *capture* stage also cold-times-out on a never-built tracy manifest (180626-11 precedent) ⇒ **the permute bucket is not isolable via tracy in the cron budget.** (Alt path if ever needed: an env-gated wall A/B like the W-mask screen — but permute is load-bearing/not skippable, so no clean wall A/B exists either.)

## Batch D — adaLN to_out fusion (transformer_ltx.py; a2v/v2a/audio_attn2 gated-residual into to_out epilogue)
**SOURCE MAP (all read 2026-07-11 23:58Z lap) — the fused primitive is PRE-BUILT & sibling-validated, NOT new-kernel work.**
`attention_ltx.py:295 _to_out_fused_addcmul(spatial, addcmul_residual, addcmul_gate, …)` computes `residual + to_out(x)*gate`
via `all_gather_minimal_matmul_async` (Ring, :328) or `dit_minimal_matmul_addcmul_fused` (:353). `forward()` routes to it
when `addcmul_residual`+`addcmul_gate` are passed (:598-606). **attn1 (:367), attn2 8-out (:389), audio_attn1 (:420) ALREADY
fuse** — incl. cross-attn (attn2 = video-text cross ⇒ same video-dim ColParallelLinear to_out as a2v). The 3 still-standalone
`ttnn.addcmul` sites = Batch D targets, each a clean `attn(...)` → `ttnn.addcmul(residual, out, gate)` (= `residual+out*gate`):
  · **a2v** `transformer_ltx.py:475-485` — residual=`video_1BND`, gate=`v_ca_gate`; writes VIDEO output. **Largest fold (video-seq N≈38760).**
  · **v2a** `:493-506` — residual=`audio_1BND`, gate=`a_ca_gate`; writes AUDIO output. (SP keeps video K/V sharded, passes `kv_logical_n`.)
  · **audio_attn2** `:441-444` — residual=`audio_1BND`, gate=`a_gate_ca`; writes AUDIO output (only under `cross_attention_adaln`).
**GATE — CLAIM CORRECTED 2026-07-12 (prior map was wrong on TWO source facts; baseline 235842-75 SKIPPED proved it):** the
`av`/`ckpt_dev` block harness is NEITHER runnable NOR PCC-gating on this box. (1) `av` weights are NOT self-contained randn —
`elif has_audio:` @test:739 loads a REAL 22B safetensors (`dev`→`ltx-2.3-22b-dev.safetensors`, `fast`→`ltx-2.3-22b-distilled-1.1.safetensors`);
this box has ONLY `sulphur_lora_fused_distil.safetensors` ⇒ `pytest.skip` @test:742 (`1 skipped, 39 deselected`). (2) `do_pcc =
run_pcc and not has_audio` @test:715 ⇒ for `av`, do_pcc is FORCED False ⇒ the VIDEO PCC assert @test:918 NEVER fires (only
audio-finiteness @test:924). a2v/v2a/audio_attn2 exist ONLY on the has_audio path ⇒ ALL THREE are structurally un-PCC-gatable in
this suite on this box. Predicted ~0.008s aggregate (below the ~1ms WARM_FWD_MS noise; sibling RMSNorm-QK fusion already measured
DEAD 45.08 vs 44.03) ⇒ not worth, and not possible, to measure here. Source map (fused primitive pre-built, sites `transformer_ltx.py`
a2v `:475-485` / v2a `:493-506` / audio_attn2 `:441-444`) preserved IF a runnable+gated harness ever exists.
- [x] **a2v fold — DEAD** (un-runnable: absent 22B ckpt; un-gatable: do_pcc forced False on av; below-noise ~0.008s). Baseline **235842-75** (2026-07-11 23:58Z) SKIPPED — re-verified from its log. No source edit made; nothing to revert.
- [x] **v2a + audio_attn2 folds — DEAD (same root cause).** Both audio-path, same un-runnable + un-gatable + below-noise verdict as a2v. **Batch D CLOSED.**

## Batch E — step-distill characterization (the 6s path; needs the PREWARM path to complete an E2E)
- [ ] **BLOCKED on the <300s reservation cap — NOT dispatchable as a cron fire-and-exit lap (forensic receipt: job 185309-15).**
  Make a full-pipeline run actually COMPLETE: measure the traced 6+2 E2E speed + quality. Re-verified 2026-07-12 03:47Z
  (20:47 PT) from the raw broker log of job **185309-15** (2026-07-11 18:53): that job WAS already the Batch-E run —
  `test_pipeline_distilled -k bh_4x8sp1tp0_ring`, `LTX_CHECKPOINT=/home/kevinmi/.cache/.../ltx-2.3-22b-distilled-1.1.safetensors`
  (kevinmi's HF cache; this box's own `~/.cache/ltx-checkpoints/` has only `sulphur_lora_fused_distil` + a dev ckpt, NOT
  distilled-1.1, so `default_ltx_checkpoint` would fall to an HF-download string — must pass LTX_CHECKPOINT explicitly),
  `LTX_S1_SIGMAS=1.0,0.99375,0.975,0.909375,0.725,0.421875,0.0` (7 vals = **6 steps**),
  `LTX_S2_SIGMAS=0.909375,0.421875,0.0` (3 vals = **2 steps**), `RUN_VBENCH=0`, `traced=True`, `--timeout=270`.
  **Timeline (log-verified):** model-load 32s → `warmup_buffers` 105s (traced=True FORCES it; RUN_WARMUP can't skip) →
  gen#0 (lazy trace-CAPTURE pass) started ~18:55:30 and was **still running** when pytest-timeout killed it at 272.01s
  → `1 failed … Failed: Timeout (>270.0s)`. gen#1 (the pure-replay STEADY-STATE measurement) NEVER ran. Device recovered
  clean (health-gate OK 32 chips, **no reset**). **Root cause = 3 in-process pipeline passes** (warmup + gen#0 capture +
  gen#1 replay) + VAE/audio/mp4-encode ×gens, NOT kernel compile — the kernel_prewarm banner shows 7742 targets built in
  **5253ms** (already warm) ⇒ **the prewarm wrapper CANNOT rescue this** (it moves *compile* off-device; compile isn't the
  cost). The metal TRACE is in-process device-DRAM ⇒ **cannot span two reservations** ⇒ the note's "submit twice / 2-reservation
  split" does NOT work (each fresh broker job re-captures the trace from scratch). Fitting warmup+gen#0+gen#1 needs a single
  **~320-350s** reservation vs the **<300s** cron cap ⇒ arithmetically over-budget; re-running at 290s would gain ~20s vs a
  ~30-50s shortfall = a known re-timeout, not worth the reservation (185309-15 IS that receipt). **Its SPEED result is already
  PREDICTED −1.89s** from the F0/I0 per-block receipts (S1 12.73ms/blk × 8→6 steps + S2 16.88ms/blk × 3→2 steps × N blocks),
  and its **QUALITY is pre-known-FAIL** (6+2 on the current checkpoint = min-PCC 0.36; the shipped distilled ckpt fails
  few-step — see MASTER_PLAN, HF B1a MISS). **Unblock paths (all off-cron):** (a) a broker-policy single >300s reservation
  to capture the traced number end-to-end; (b) out-of-repo step-distilled checkpoint (the real 6s path, a user resourcing
  decision). No in-budget cron variant yields a clean traced-comparable number (untraced single-gen would fit but is
  dispatch-dominated ⇒ non-comparable / misleading, so not recorded).
  NOTE: raw 4x8 full-pipeline = guaranteed timeout; this is why it's last + needs the multi-pass warm.

## Batch F — CCL-collective matmul (the DOMINANT op, 56.3% = 11.57ms/block; NEVER config-swept)
Profile ranked CCL-matmul (AllGatherMatmul + MatmulReduceScatter, fused all-gather/reduce-scatter into the
to_out / QKV / FFN matmuls) as the largest block bucket — 56.3%, ~2× the SDPA bucket that Batch A swept dead.
**Harness (in-budget, NO tracy, PCC-gated, self-contained — no 22B ckpt):** the VIDEO-only stage_2 block
`test_ltx_transformer_block[NOTSET-ckpt_fast-pcc-video-stage_2-<mesh>]`. It uses scaled-random weights via
`_convert_diffusers_video_block_to_tt` (video path never hits the absent-22B skip that killed all Batch-D `av`
runs), `do_pcc = run_pcc and not has_audio` = **True** (video PCC-gates @ pcc=0.988/rmse=0.10 on 32-dev mesh,
test:920), and emits **`WARM_FWD_MS=`** directly (test:876-889) under `LTX_PROFILE_ITERS>1` (iter 0 cold, rest
warm) ⇒ no tracy, no CSV, immune to the tracy multi-word `-k` bug (exact bracket node-id). Env `opt/env_sdpa.yaml`
(warm ltxrt caches, PINNED=0). **Enumerable config space that's dispatchable in-budget WITHOUT a source edit:**
mesh param exposes both `ring_bh_4x8sp1tp0` (Topology.Ring) AND `line_bh_4x8sp1tp0` (Topology.Line) ⇒ a CLEAN
CCL-topology A/B on the fused all-gather-matmul (num_links=2 HW-cap holds for both). Fidelity axis already probed
(all_bf8 weights −0.04s null); matmul program-config (grid/subblock) is source-computed ⇒ warm-authoring, not cron.
**⚠️ HARNESS GOTCHA (F0 job 001703-77 died 0-collect, root-caused): SELECT BY `-k`, NOT the exact bracket node-id.**
The first id slot is the ARCH — `blackhole` when collected UNDER the broker (device present), but `NOTSET` when
collected OFF-device (my local collect-only had no device fixture). So an off-device-derived bracket id
(`[NOTSET-...]`) NEVER matches under the broker (`[blackhole-...]`) ⇒ 0 items. This is plain pytest (NOT tracy), so a
single-quoted `-k 'video and stage_2 and ring_bh_4x8sp1tp0 and ckpt_fast'` survives the broker shell intact (the tracy
`" ".join`/2nd-shell bug that broke Batch A's spaced `-k` does NOT apply here) and is arch-slot-agnostic. Use `-k`.
- [x] **F0 — video-block warm-FW baseline @ ring_bh_4x8 = WARM_FWD_MS 16.88 (PCC 99.966%, gate 0.988).** First run
  (job 002052-79) ERRORED on a transient eth-core-27-25 teardown fault (device auto-reset by broker); re-run **job
  002717-83** on the reset device is CLEAN (1 passed, JIT 423/423 warm=zero cold-compile, clean UMD teardown) ⇒ the
  fault was a device flake, not a harness bug. This is the warm VIDEO per-block FW (all prior block runs were `av`→SKIP;
  the 44.77ms number was AV single-blocking). Substrate for F1's topology A/B. (supersedes 001703-77 0-collect bracket bug.)
- [x] **F1 — video-block warm-FW @ line_bh_4x8 (Topology.Line) = WARM_FWD_MS 20.97 (PCC 99.967%, gate 0.988) → DEAD.**
  Job **002930-85** (re-verified from raw log): Line 20.97ms vs F0 Ring 16.88ms = **+24% SLOWER**. Ring wins decisively;
  Linear all-gather is worse for the fused matmul at 4x8 block scale. No source edit (topology param A/B) ⇒ nothing to
  revert. ⇒ **BOTH no-edit CCL-topology cells measured at prod 4x8; Ring stays. Batch F topology axis CLOSED.**
- [x] **F2 — video-block warm-FW @ ring_bh_4x8 `LTX_NUM_LINKS=1` = WARM_FWD_MS 19.58 (PCC 99.966%, gate 0.988) →
  characterization DONE.** Job **003505-87** (re-verified from raw log: `WARM_FWD_MS=19.58 num_links=1 iters=3`,
  "PASSED block PCC: video (1,38760,4096)", 1 passed/39 deselected in 55.71s, JIT 398/424 warm). num_links=1 19.58ms
  vs F0 num_links=2 16.88ms = **+16% slower (Δ 2.70ms)**. So the 2nd link (already the BH HW cap, A2) saves 2.70ms =
  only ~16% of the block ⇒ **the dominant fused CCL-matmul is largely COMPUTE-bound, not interconnect-bound** — modest
  fabric headroom (~2.7ms) that prod num_links=2 ALREADY captures at the cap. Sharpens the honest floor: the 56.3% CCL
  bucket is matmul compute, not link BW ⇒ no interconnect win to chase. No source edit (env A/B) ⇒ nothing to revert.
→ **BATCH F CLOSED — the entire no-edit CCL config space is swept at prod 4x8:** topology (Ring 16.88 < Line 20.97),
  links (num_links=2 16.88 < 1 19.58, and 2 is the HW cap), fidelity (all_bf8 −0.04s null, DONE line). Ring+num_links=2
  is prod-optimal on every no-edit axis; the CCL bucket is compute-bound. **No remaining no-edit block-harness lever.**

## Batch G — CCL-matmul grid/subblock program-config (the ONLY remaining CCL axis; source-verified WARM-AUTHORING, not cron)
The fused CCL-matmul program config is **computed** by `get_matmul_config(M, K, N_out, core_grid)` at
`models/tt_dit/models/transformers/ltx/attention_ltx.py:320` (all_gather_minimal_matmul_async path) and `:351`
(dit_minimal_matmul_addcmul_fused path). **Source-verified there is NO env knob** (only `LTX_NUM_LINKS`, mesh-param
topology, `LTX_QUANT` — all swept in A/B/F) and the grid axis is already MAXED: `core_grid` = `CoreCoord(full_grid.x,
full_grid.y-1)` (:319, all-gather reserves 1 row for CCL workers) / `full_grid` (:350, reduce-scatter). Growing the
grid is impossible (already full device); shrinking only slows. The only tunable left is **inside `get_matmul_config`'s
subblock heuristic** — a shared util used across tt_dit models ⇒ **HIGH blast radius** (a change touches every model's
matmul), needs a warm iterate+PCC loop + `build_metal` if it dips into C++, NOT a fire-and-exit cron one-shot (a blind
subblock guess fails op-validation, cf. Batch B exp-kernel SIGABRT). Gate: video block PCC≥0.988 + WARM_FWD_MS vs F0
16.88ms. **Value ceiling: even a fully-free CCL bucket floors the block at the ~7.9s no-finetune wall.**
- [x] **G0 — matmul block-tune (MEASURED ~flat, compute-bound; CLOSED 2026-07-12 06:00Z, job 055639-7).** Prior "WARM-
  AUTHORING / build_metal" classification was WRONG: `get_matmul_config` (matmul.py:189) returns a pure-Python
  `ttnn.MinimalMatmulConfig` from lookup tables → block/subblock is a `register_matmul_configs` edit, PCC-safe, NO
  build_metal (there's even a no-build sweep harness `sweep_mm_block_sizes.py`). The `_BH_GALAXY_MAX_CORE_GRID=(11,10)`
  clamp orphaned the S2 tunings (keyed 12x9 in `grid_12_9_configs`; runtime queries 11x10 → fallback), so the dominant
  S2 to_out AGMM `(4864,4096,4096)` runs untuned 8x8x8. A/B: registered FE-pattern `(4,8,12,(1,4))` on 11x10 →
  WARM_FWD_MS 16.70 vs 16.88 baseline = **−1.1% (noise) + PCC 99.966% PASS ⇒ NO WIN** (compute-bound, matches MASTER_PLAN
  + FE "aggregate ~flat"). Reverted. Loose thread: guessed block, not the 12x9 swept-optimal `(5,8,16)`; a full 11x10
  re-sweep (`sweep_mm_block_sizes.py`, tracy, ~2h off-cron) caps at a few % = polish, not a 6.0s path.

## Batch H — CCL-matmul weight/activation QUANT sweep (the dominant op, env-only, PCC-gated, cron-in-budget)
`LTX_QUANT=<preset>` on the F0 block harness (`test_ltx_transformer_block -k 'video and stage_2 and
ring_bh_4x8sp1tp0 and ckpt_fast'`, num_links=param=2, prod) applies a `QuantConfig` preset (weight typecast +
compute config) to the block, emits `WARM_FWD_MS=` (vs F0 16.88ms) AND PCC-gates video vs the bf16 diffusers
oracle (0.988). Read at test:864-870 — the block test READS `LTX_QUANT` (the SDPA test does NOT; Batch B's
"LTX_QUANT no-op" applies only to `test_ring_joint_sdpa`). WARM_FWD_MS is logged at test:887 *before* the PCC
assert at test:923, so timing lands even if PCC fails. Presets: `quant_config.py` factories `default` (baseline),
`all_bf8_lofi`, `all_bf8_lofi_sdpa_lofi`, `all_bf8_lofi_sdpa_lofi_fp32acc`. Directly tests the compute-bound vs
overhead-bound question Batch F reopened (num_links=1 → +16% only).
- [x] **`all_bf8_lofi` (qkv/out/ffn weights+acts → bf8, LoFi; SDPA stays bf16) — NULL @ PASSING PCC.**
  Job **010011-89** (re-verified from raw log: test:870 `quantizing block`, test:887 `WARM_FWD_MS=16.69
  num_links=param iters=3`, assert_quality:48 `PCC = 99.8889 %`, test:923 `PASSED block PCC`, 1 passed/39 desel
  in 50.03s, JIT 416/455 warm, exit 0, 32 chips healthy). **16.69ms vs F0 16.88ms = −0.19ms = −1.1% = NULL**
  (<5% act gate). **PCC 99.89% PASSES 0.988** — this CORRECTS the old "all_bf8 PCC 0.876 FAIL" (that was a
  coarser stage/pipeline path, NOT this prod-4x8 video-block harness). **Resolves the compute-vs-overhead
  contradiction:** halving the dominant CCL-matmul's math precision (bf16→bf8/LoFi) moves the block only −1.1%,
  and Batch F showed the 2nd link saves only 2.7ms (+16%) ⇒ the 56.3% fused all-gather/reduce-scatter matmul
  bucket is **collective-latency / dispatch bound, NOT matmul-compute bound and NOT link-BW bound**. Neither
  quant (compute) nor num_links (BW) unlocks it. No source edit (env A/B) ⇒ nothing to revert.
- [x] **`all_bf8_lofi_sdpa_lofi` + `all_bf8_lofi_sdpa_lofi_fp32acc` — DEAD by composition (no run).** Both
  stack SDPA LoFi onto the `all_bf8_lofi` base, which just measured −1.1% NULL. Batch B already measured SDPA
  LoFi as null (4.673 vs 4.74ms = ~1.4%, SDPA is NOT fidelity-bound) ⇒ null-base + null-SDPA-fidelity composes
  to null on speed, and can only DROP PCC (adds SDPA quality risk to an already-passing base). Measuring would
  confirm null+worse-PCC — a receipt for an argument already closed by H0 + Batch B. Not worth a reservation.
→ **BATCH H CLOSED — the weight/activation quant+fidelity axis on the dominant op is swept: NULL at passing
  PCC.** Combined with Batch A (SDPA chunk U-shaped, base near-optimal), Batch B (SDPA fidelity/exp-kernel
  dead), Batch F (topology Ring-optimal, num_links=2 HW-cap, both near-flat): **every no-edit block-harness axis
  on both dominant buckets (CCL-matmul 56% + SDPA 30%) is measured, and the block is collective/dispatch-bound
  at ~16.88ms.** The only remaining in-repo lever is the source-computed matmul grid/subblock (Batch G,
  warm-authoring) — and even a fully-free CCL bucket floors at the ~7.9s no-finetune wall.

## Batch I — STAGE-1 block characterization (the 2nd denoise stage, 2.9s = 35% of E2E, NEVER block-profiled)
The whole session characterized only `stage_2` (N=38760). `stage_1` (N=9690, 4× smaller seq, id line 122) is the SAME
`test_ltx_transformer_block` re-pointed by a one-word `-k` swap (`stage_2`→`stage_1`) — no-edit, PCC-gated, WARM_FWD_MS,
cron-in-budget, warm ltxrt caches (E2E 185309-15 populated both stages). Enumerate the same no-edit axes Batch F/H swept for S2.
- [x] **I0 — S1 video-block warm-FW baseline @ ring_bh_4x8 = WARM_FWD_MS 12.73 (PCC 99.994%, gate 0.988).** Job **010823-91**
  (`-k 'video and stage_1 and ring_bh_4x8sp1tp0 and ckpt_fast'`, LTX_PROFILE_ITERS=4, env_sdpa.yaml, JIT 402/423 warm, 1
  passed/39 desel, 32 chips healthy). **First-ever S1 per-block receipt.** vs S2 F0 16.88ms: a **4× larger seq yields only
  1.33× the block time** ⇒ block dominated by a fixed ~12.7ms/block collective/dispatch floor, not seq-proportional compute —
  the dispatch-bound thesis (H0 −1.1%, F2 +16%) now confirmed across a 4× seq range.
- [x] **I2 — S1 num_links=1 @ ring_bh_4x8 = WARM_FWD_MS 12.71 (PCC 99.994%) → 0% (dispatch-bound floor).** Job **011036-93**
  (`LTX_NUM_LINKS=1`, else identical). vs I0 num_links=2 12.73ms = **Δ −0.02ms ≈ 0%** — the 2nd link makes NO difference at S1
  (contrast S2's +16%/Δ2.70ms). S1's small tensors leave ample single-link BW headroom ⇒ **S1 is PURELY dispatch/collective-
  latency-bound, ZERO link-BW sensitivity.** Sharpest evidence yet: the ~12.7ms floor is fixed setup+dispatch, not BW.
- [x] **I1 — S1 Line topology (line_bh_4x8sp1tp0) = WARM_FWD_MS 11.46 (PCC 99.994%) → TOPOLOGY CROSSOVER (char-only, not
  shippable).** Job **011213-95**. **Line 11.46 < Ring 12.73 = −10%** at S1 — INVERTS S2's F1 result (Line 20.97 > Ring 16.88 =
  +24%). Corrects F1's "Ring wins decisively" (that was S2-only). Physically consistent: at S1's small tensors the Ring's
  multi-hop collective-SETUP latency dominates and Linear's lower fixed setup wins (matches I2's zero BW sensitivity).
  **NOT a shippable lever:** (1) fabric topology is a DEVICE-INIT constant — one `FabricConfig::FABRIC_1D_RING` per process
  (`fabric_firmware_initializer.cpp`), and the pipeline builds ONE shared `CCLManager` (`pipeline_ltx.py:597`) used for both
  stages ⇒ can't run S1-Linear + S2-Ring in one reservation; (2) whole-run Line is net-WORSE (S2 +4.09ms/block ≫ S1 −1.27ms/block)
  ⇒ Ring-for-all (prod) stays optimal.
- [x] **I3 — S1 LTX_QUANT weight/act quant — DEAD by composition (no run).** H0 measured quant −1.1% null on the S2 dispatch-
  bound block; S1 is EVEN MORE dispatch-bound (I2: zero BW sensitivity) ⇒ a compute lever moves it even less. Would confirm null
  + add PCC risk. Not worth a reservation (mirrors H1/H2 dead-by-composition).
→ **BATCH I CLOSED — the SECOND denoise stage (S1, 35% of E2E) is now block-profiled + no-edit-swept for the first time:**
  baseline 12.73ms, num_links-insensitive (pure dispatch floor), Line-topology crossover (char-only, not shippable). Both
  denoise stages now have per-block receipts; the block is collective/dispatch-latency-bound at BOTH scales, MORE so at S1.
  **No no-edit lever moves either stage.** Reinforces: only cutting the NUMBER of blocks/steps (out-of-repo step-distill) helps.

## Batch J — CCL-matmul + SDPA STACKED quant (measures Batch H's unmeasured "dead-by-composition" cell)
Batch H measured `all_bf8_lofi` (bf8 matmul weights+acts) = −1.1% NULL @ PCC 99.89% PASS, and declared the SDPA-stacked
presets "dead by composition" from *reasoning* (no receipt). Same F0 harness (`-k 'video and stage_2 and ring_bh_4x8sp1tp0
and ckpt_fast'`, num_links=2, warm ltxrt caches, env_sdpa.yaml). Converts reasoning → a measured quality receipt.
- [x] **`all_bf8_lofi_sdpa_lofi` (bf8 matmul + LoFi SDPA) — DEAD: FAILS PCC gate.** Job **012439-97** (re-verified from
  raw log myself): **PCC = 98.5714 % < 98.80 % gate → test FAILED** (check.py:57), JIT **452/455 warm (99.3%)**, 1 failed /
  39 deselected in 48.40s, exit 0, clean UMD teardown (no eth fault, 250s cap unused). Stacking SDPA-LoFi onto the passing
  `all_bf8_lofi` base (99.89%) drops video-block PCC by **1.32 pts → 98.57%, below the 0.988 gate** ⇒ NOT shippable at
  quality, regardless of speed. **WARM_FWD_MS truncated** — the `tail -120` window was consumed by the PCC-failure traceback
  so the test:887 INFO line scrolled off; MOOT (dead on quality, and per the dispatch-bound thesis — H0 bf8 −1.1% null,
  Batch B SDPA-LoFi null — it wouldn't be meaningfully faster). Confirms H's prediction WITH a receipt: SDPA-LoFi adds pure
  quality risk to a null-speed base. No source edit (env A/B) ⇒ nothing to revert.
- [x] **`all_bf8_lofi_sdpa_lofi_fp32acc` — NOW MEASURED (job 015415-106, 2026-07-12 01:55Z): PCC 99.93% PASS, WARM_FWD_MS
  16.06 = −4.9% vs F0 16.88 (SUB-GATE, no shippable win).** CORRECTS the prior "dead-by-composition / fp32acc can't recover
  1.32 pts" reasoning: **fp32-dest-acc DOES recover the SDPA-LoFi PCC** — 98.57% FAIL (Batch J, no fp32acc) → 99.93% PASS
  (+1.36 pts, > all_bf8_lofi's 99.89%), exactly the preset's designed purpose (full-precision softmax max/sum under LoFi
  packed-dest). But SPEED is no win: −4.9% is under the >5% act-gate, and the −0.63ms increment vs all_bf8_lofi 16.69
  CONTRADICTS the physics (Batch B isolated SDPA-LoFi null −1.4%; fp32acc ADDS accumulation cost ⇒ increment should be
  null-or-slower) ⇒ favorable block-FW run-to-run noise, not a compute reduction. Face-value E2E ≈ −0.82ms/block × 144
  S2-fwd ≈ −0.12s = polish below the ~7.9s floor; replication can only push the median toward null (further from the gate)
  ⇒ not worth another reservation. (First two dispatches 014854-101/015114-103 died at the transient eth-core-27-25
  mesh-open flake — lever-irrelevant, device opens before quant applies; 3rd re-run clean on the recovered device.) No
  source edit (env A/B) ⇒ nothing to revert.
→ **BATCH J CLOSED — quant axis now fully MEASURED, not reasoned:** `all_bf8_lofi` is the sweet spot (passes 99.89%, −1.1%
  null speed); pushing further to SDPA-LoFi breaks the gate (98.57% FAIL). **No quant preset moves the dispatch-bound block
  at passing quality.** Combined with A/B (SDPA chunk/fidelity dead), F (topology/links flat), H (bf8 null), I (S1 same):
  every no-edit block-harness axis on both dominant buckets is measured — the block is collective/dispatch-bound.

## Batch K — VAE-decode QUANT (the ONE compute-bound bucket; A1 proved decode is compute-bound, opposite regime from the dispatch-bound denoise where quant was null)
Every prior quant sweep (H/J) hit the DISPATCH-bound denoise block → null. VAE decode is the one bucket A1 measured
COMPUTE-bound (traced 551.88 ≈ untraced 552.08ms → dispatch already hidden by async CQ) ⇒ a compute-precision lever
(bf8 conv weights) has a real mechanism to cut the 552ms here. Harness = `prof_vae_ltx.py::test_prof_vae_ltx_trace`
(plain, in-budget ~552ms/decode, NOT full pipeline). `LTXVideoDecoder` exposes a single `dtype` param (vae_ltx.py:646)
threaded to all conv weight uploads (:193 `from_torch(dtype=self.dtype)`); `_build_tt_decoder` didn't pass it.
- [x] **bf8 (bfloat8_b) conv weights via one-line dtype flip — DEAD: un-runnable (SEGFAULT at weight upload).** Job
  **013750-99** (2026-07-12 01:39Z, re-verified from raw log): env `LTX_VAE_BFP8=1`, test-only 1-line scaffold passing
  `dtype=ttnn.bfloat8_b` to the decoder ctor. **`Fatal Python error: Segmentation fault`** at `from_torch(dtype=bfloat8_b)`
  (`vae_ltx.py:193 _prepare_torch_state`), crashing on the **1st of 86 weight tensors** (6s in, log:85) — NO TT_FATAL,
  a hard segfault. bfloat8_b is a TILE block-float format; the conv weight-upload path uploads without a TILE layout ⇒
  the naive dtype flip can't produce a valid bfp8 conv weight. 57.6s, exit-0 (pipe tail masks the crash), **device
  recovered clean (health-gate OK, 32 chips, no reset)**. Mirrors Batch B's exp-kernel SIGABRT: structural incompat found
  fast + cheap. Scaffold **REVERTED** (crashed, no win; finding is the receipt).
  → **VAE-quant is NOT a cron one-shot.** A real bfp8-VAE probe needs per-conv TILE-layout + verifying each conv3d/upsample
  op accepts bfp8 weights (blast radius across the decode graph) = **WARM-AUTHORING**, joining the parked lot (W-mask
  in-kernel fold C, subblock tune G). No env-only VAE-quant cell exists.
→ **BATCH K CLOSED — the compute-bound VAE bucket resists the cheap quant lever:** naive bf8 flip segfaults at upload;
  proper bfp8-VAE is warm-authoring. The only env-only VAE cells left (W-mask skip, depth-to-space) are already closed
  (load-bearing / tracy-timeout). VAE-quant parked for a warm session alongside C/G.

## Batch M — parallelism STRATEGY A/B (SP vs TP): the LAST unmeasured no-edit block axis
Every prior block sweep (A/B/F/H/I/J) held parallelism fixed at **sp1tp0 (sequence-parallel)**. The block is
collective/dispatch-bound (H0 bf8 −1.1% null · F2 num_links +16% · I2 zero BW-sensitivity) ⇒ the one structural lever
compute/link/quant CANNOT touch is the **collective PATTERN itself**. TP (tp1) distributes along the head/feature dim
instead of sequence ⇒ a different all-gather shape and potentially FEWER/SMALLER collectives — a real, testable
dispatch-bound hypothesis. The ONLY mesh whose parametrize exposes both SP and TP is **2x4** (`test:108-115`):
`2x4sp1tp0` (SP, is_fsdp=False, Linear) vs `2x4sp0tp1` (TP, is_fsdp=True, Linear). ⚠️ **Confounds (honest):** (1) 2x4 =
8-chip NON-prod scale (prod is 4x8/32-chip; no `4x8sp0tp1` param exists ⇒ TP@prod would need a source-added param +
unverified model support); (2) is_fsdp differs (SP non-FSDP vs TP FSDP) — inherent to the config defs. **Value ceiling:
characterization of the last no-edit axis, below the closed ~7.9s floor** — a large TP<SP delta would justify escalating
to a 4x8-TP param (source edit); null confirms dispatch-bound across parallelism strategy. No source edit (mesh `-k`
swap), PCC-gated (video path, do_pcc=True), WARM_FWD_MS. ⚠️ 2x4 kernels were NEVER built (E2E warmed only 4x8) ⇒
COLD-compile risk; if it times out that IS the receipt (2x4 needs prewarm), broker auto-recovers.
- [x] **M0 — 2x4 SP baseline — INFRA-DEAD (2x4 mesh won't init fabric; reproducible across a reset).** BOTH the
  initial **020534-109** AND my re-run **020931-113** faulted identically at `mesh_device` setup: `Fabric Router Sync:
  Timeout after 10000 ms on Device 1` (chan e0-4/e0-5 stuck at STARTED, remote eth handshake never completes),
  `distributed.py:671`, `1 error in ~18s`. A broker glx_reset (020752-111) + fabric-check exit-0 (021047-114) ran
  BETWEEN the two runs ⇒ the fault **reproduces on a freshly-reset+health-verified device** ⇒ NOT the transient
  4x8 eth-core-27-25 flake (which auto-recovered so re-runs passed, cf. F0 002052-79); it is specific to the **2x4
  sub-mesh's fabric bring-up on this box.** The block forward NEVER ran (no WARM_FWD_MS, no PCC) ⇒ not a parallelism
  verdict. Did NOT thrash a 3rd run. No source edit (mesh `-k`) ⇒ nothing to revert.
- [x] **M1 — 2x4 TP — DEAD by dependency (same fabric-broken 2x4 mesh).** The parametrize (test:108-115) exposes TP
  (`tp1`) ONLY on the 2x4 mesh; M1 uses the identical mesh that just failed fabric init twice ⇒ will hit the same wall.
  ⇒ **BATCH M CLOSED — infra-blocked:** the SP-vs-TP axis is un-measurable via the no-edit harness on this box. TP@prod
  would need a source-added 4x8-TP parametrize row + unverified model support = warm-authoring, not a cron one-shot.

## Batch N — S1 (stage_1) quant (I3, promoted from dead-by-composition to a MEASUREMENT per Batch L precedent)
Batch I closed I3 (S1 + `all_bf8_lofi`) "dead-by-composition (no run)". S1 proved a DIFFERENT regime than S2 (I1
Line-topology crossover) ⇒ not a foregone re-run; and Batch L overturned a dead-by-composition call by measuring ⇒
measured it. Same F0/I0 block harness, `-k 'video and stage_1 and ring_bh_4x8sp1tp0 and ckpt_fast'`, `LTX_QUANT=all_bf8_lofi`.
- [x] **S1 + `all_bf8_lofi` — NULL @ PASSING PCC.** Job **021338-117** (re-verified from raw log): test:870 `quantizing
  block`, test:887 `WARM_FWD_MS=12.35 num_links=param iters=3`, assert_quality:48 `PCC = 99.9315 %`, test:923 `PASSED
  block PCC: video (1, 9690, 4096)`, 1 passed/39 desel in 23.72s, 32 chips healthy. **12.35ms vs I0 baseline 12.73 =
  −0.38ms = −3.0% NULL** (sub 5% gate), PCC 99.93% PASS. Mildly larger relative move than S2's H0 −1.1% (likely block-FW
  noise — S1's smaller absolute ms makes fixed-ms jitter a larger %); same verdict. No source edit (env A/B) ⇒ nothing
  to revert. **⇒ no quant preset clears the 5% gate at passing quality on EITHER denoise stage; both confirmed
  collective/dispatch-bound.** Batch N CLOSED (S1 stacked/fp32acc presets = dead-by-composition, same as the S2 finding).

## Batch O — VAE conv math-FIDELITY (the last untested compute lever on the ONE compute-bound bucket)
Every prior quant sweep hit the DISPATCH-bound denoise (H/J/N null). A1 measured VAE decode COMPUTE-bound
(traced≈untraced 552ms). Batch K closed the weight-DTYPE sub-path (bf8 → SEGFAULT at TILE-less upload) but NOT the
math-FIDELITY sub-path: `LTXCausalConv3d` builds a per-conv `compute_kernel_config` at HiFi2 (vae_ltx.py:146-154;
HiFi4 only when Blackhole+dtype==float32, and the prof decoder defaults dtype=bfloat16 → HiFi2). Dropping to LoFi
(2 passes → 1) is a pure-compute lever with a real mechanism on the one compute-bound bucket — env-gated 1-line scaffold
(`LTX_VAE_LOFI` → `MathFidelity.LoFi`), no build_metal, cron-in-budget. Harness = `prof_vae_ltx.py::test_prof_vae_ltx_trace`
(TRACED_DECODE_WALL_MS vs baseline 551.91ms, job 225635-65), env `opt/env_sdpa.yaml`. Speed-screen first (W-mask COST pattern):
if <5% it's DEAD regardless of PCC.
- [x] **conv LoFi (HiFi2→LoFi) — NULL: TRACED_DECODE_WALL_MS 542.17 vs 551.91 baseline = −9.74ms = −1.77% (<5% gate).**
  Job **022644-119** (re-verified from raw log: `TRACED_DECODE_WALL_MS=542.17 ... iters=10`, `1 passed in 45.51s`, warm —
  prewarm rebuilt the LoFi conv targets in 3548ms, no timeout, 32 chips healthy). **Cutting conv math-fidelity 2× barely
  moves the 552ms ⇒ the VAE decode is NOT matmul-math-bound; the "compute-bound" of A1 is conv3d data-movement / halo /
  reduction / dispatch, not multiply-accumulate precision.** No PCC A/B needed (dead on speed). Env-gated scaffold
  REVERTED (no win; finding is the receipt). ⇒ **the compute-precision axis on VAE is now closed on BOTH sub-paths (weight
  bf8 = segfault/warm-authoring K; math-fidelity LoFi = −1.77% null O).** No env-only VAE compute lever moves the decode.

## Batch P — S1 + `all_bf8_lofi_sdpa_lofi_fp32acc` (fill the quant-preset × stage matrix; measure-don't-reason)
Batch L measured this preset on S2 (99.93% PASS, −4.9% sub-gate); I3 marked the two SDPA-LoFi presets "S1 dead by
composition (no run)". Per the Batch J/L precedent (measurement overturned two dead-by-composition calls), converted the
one that PCC-passes on S2 to a real S1 measurement — no-edit env A/B on the I0/I3 S1 harness (`-k 'video and stage_1 and
ring_bh_4x8sp1tp0 and ckpt_fast'`, `LTX_QUANT=all_bf8_lofi_sdpa_lofi_fp32acc`, env `opt/env_sdpa.yaml`, ~24s warm).
- [x] **S1 fp32acc-quant — NULL @ passing PCC; caught + killed a false gate-crossing from cross-reservation baseline drift.**
  Two clean fp32acc samples (jobs **023548-121** 11.99ms / **023717-123** 11.78ms, mean **11.89ms**, PCC **99.8996 %** PASS,
  deterministic, re-verified from raw logs). Both landed *below* the stale I0 baseline 12.73 (−5.8%, nominally OVER the >5%
  gate) AND below I3 `all_bf8_lofi` 12.35 — physically impossible as a real fp32acc win (fp32acc ADDS accumulate cost over
  its bf8 base). Dispatched a **fresh same-session default baseline** (job **023830-124**) = **12.25ms** (PCC 99.9937 %):
  against apples-to-apples the preset is **−2.9% NULL (sub-5% gate)** — the apparent −5.8% was **~4% cross-reservation
  baseline drift** (stale 12.73 vs fresh 12.25), not compute. **Receipt of a methodological trap:** single-sample baselines
  from a prior reservation can manufacture a false gate-crossing; re-measure the baseline in-session. Mirrors Batch L (S2):
  fp32acc PCC-recovers to PASS but no shippable speed win. **Both denoise stages now confirmed: no quant preset clears the
  5% gate at passing quality ⇒ collective/dispatch-bound.** No source edit (env A/B) ⇒ nothing to revert. **Batch P CLOSED.**

## Batch Q — FSDP weight-sharding A/B (the LAST untouched no-edit collective-PATTERN axis on prod 4x8)
Every prior block sweep held **is_fsdp=False** (prod BH replicates weights, loads via dynamic_load). The mesh
parametrize (test:113) exposes **`wh_4x8sp1tp0` = (4,8) Ring SP with is_fsdp=True** — same mesh/topology/SP as the prod
`ring_bh_4x8sp1tp0`, differing ONLY in is_fsdp. is_fsdp=True sets `fsdp_mesh_axis = sequence_parallel.mesh_axis`
(attention_ltx.py:85, transformer_ltx.py:153) ⇒ weights are FSDP-**sharded** along the SP axis and all-gathered per
layer, changing the collective pattern of the dominant 56% CCL-matmul bucket. This is the structural/collective-pattern
lever Batch M chased via 2x4 SP-vs-TP (fabric-DEAD) — but reachable on the **working 4x8 mesh**. wh's default num_links=4
is BH-illegal (A2 HW-cap 2), but `LTX_NUM_LINKS=2` (test:510) overrides it ⇒ both configs run at 2 links, isolating
is_fsdp. **Drift-immune (Batch-P lesson): select BOTH in one job** so baseline + lever share the reservation/session.
Hypothesis: FSDP ADDS weight all-gathers ⇒ on a dispatch-bound block, likely null-or-slower — but measure, don't reason
(J/L/P overturned three dead-by-composition calls). ⚠️ wh is_fsdp=True programs were NOT E2E-warmed (prod=is_fsdp=False)
⇒ COLD-compile risk on the fsdp forward; generous timeout, and a timeout IS the receipt (fsdp needs prewarm). Value
ceiling: char-only (is_fsdp=True is not the prod BH path) below the ~7.9s floor; a large FSDP<repl delta would justify escalation.
- [x] **Q0 — is_fsdp A/B on prod 4x8 S2 video block — FSDP is +10.5% SLOWER = NULL/DEAD lever; prod is_fsdp=False optimal.**
  Job **024854-127** (drift-immune same-session A/B, re-verified from raw log + collect-only exec order). Collection order
  (pytest exec order, no randomizer) = **wh first, ring_bh second** ⇒ 1st WARM_FWD_MS=18.56 (wh, is_fsdp=True/FSDP), 2nd
  WARM_FWD_MS=16.79 (ring_bh, is_fsdp=False/prod). **FSDP 18.56 vs prod 16.79 = +1.77ms = +10.5% SLOWER**, PCC IDENTICAL
  99.9658% both (FSDP is numerically equivalent, only reschedules collectives). In-session baseline 16.79 ≈ F0 16.88 (0.5%,
  no drift — cross-check clean). Wall-times corroborate: wh ran first with a ~42s cold compile (its FSDP programs were NOT
  E2E-warmed — prod is is_fsdp=False), ring_bh ~32s warm. **FSDP weight-sharding ADDS per-layer weight all-gather collectives
  ⇒ on the dispatch-bound block, MORE collectives = slower.** Confirms the dispatch-bound thesis from the opposite direction:
  adding collectives HURTS (+10.5%), just as removing link/compute (F2/H0) barely helped — only cutting the NUMBER of
  collectives (fewer steps/blocks = out-of-repo distill) moves the wall. No source edit (mesh `-k` + env A/B) ⇒ nothing to
  revert. **⇒ BATCH Q CLOSED — the last no-edit collective-PATTERN axis (FSDP on/off) is measured: prod is_fsdp=False is
  optimal; FSDP is a +10.5% regression.** All no-edit block-harness axes on both dominant buckets are now swept across
  topology · links · quant(4/4) · fidelity · parallelism(SP fixed; TP infra-dead M) · FSDP — block is dispatch-bound.

## Batch R — VAE conv3d `_BLOCKINGS` degenerate-fallback (the ONE untouched harness on the data-movement-bound VAE bucket) — DEAD for 1080p (source-verified)
The harness inventory had a real gap: three LTX test files were NEVER touched by any batch. Checked all three
(2026-07-12 03:12Z lap): `test_ccl_allgather_scoped.py` = **2x4sp1tp0-ONLY** parametrize (test:56) → the same
fabric-DEAD 2x4 mesh as Batch M + it's a NoC-trace/tracy harness (cold-timeout); `test_ltx_stage_scoped.py` =
`skipif` needs the absent `ltx-2.3-22b-distilled-1.1.safetensors` (test:64-66) + builds the full `LTXDistilledPipeline`
(test:98) → SKIP + out-of-budget. The 3rd — **`test_ltx_conv3d_sweep_720p.py`** — is the genuinely-new signal: a
self-contained conv3d BLOCKING microbenchmark (1x1 submesh, no 22B ckpt, no full pipeline) whose docstring says the
deep 1024-ch decoder/upsampler convs **miss the tuned `_BLOCKINGS` table and fall to the degenerate `(Cin,32,1,1,1)`
one-pixel-per-work-unit path (~9× slower)**. This is exactly VAE decode's regime — Batch O proved VAE is DATA-MOVEMENT-
bound, then only tested math-fidelity (LoFi null) + weight-dtype (bf8 segfault); the conv3d BLOCKING config (the actual
data-movement lever) was never checked. A real gap in "VAE exhausted."
- [x] **DEAD for the 1080p north star — the 1080p decode is already tuned; the fallback is 720p-specific.** Source gate
  (`models/tt_dit/utils/conv3d.py:518-521`, authored comment): *"LTX-2.3 22B VAE decoder + latent upsampler … These
  channel combos all have swept exact `_BLOCKINGS` entries for 2x4/4x8 **1080p**; they remain here as the cross-mesh/
  cross-resolution fallback (the hardcoded full-Cin default OOMs at these widths)."* So the 1080p-high decode convs hit
  `[exact]` `_BLOCKINGS.get(blocking_key)` (conv3d.py:707), NOT `[fallback]` (:723). The degenerate `(256,32,1,1,1)`
  one-pixel entries in `_DEFAULT_BLOCKINGS` (:529 s0_up, :530 s1_up, :535 ups final_conv) are the CROSS-resolution
  fallback for **off-1080p** spatial dims (:528 "OOMs at off-1080p spatial dims"). **704×1280 (720p) is a distinct gen
  config, NOT an internal stage of 1080p-high** (which is S1 544×960 → S2 1088×1920 → VAE decode 1088×1920) ⇒ running
  the 720p sweep tunes a resolution we don't ship. No no-edit device cell advances the 1080p goal here, and no
  warm-authoring lever either (1080p `_BLOCKINGS` is already swept-exact). Closes the last harness-inventory gap; the
  VAE decode data-movement bucket is tuned at 1080p, not fallback-degraded. No source edit ⇒ nothing to revert.

## Batch S — VAE-decode num_links (the ONE untested interconnect cell on the ONE data-movement-bound bucket)
Every prior CCL/num_links sweep (F2/I2) hit the DISPATCH-bound denoise block (S2 +16%, S1 0%). VAE decode is the
opposite regime — Batch O proved it DATA-MOVEMENT-bound (LoFi fidelity −1.77% null → NOT matmul-compute-bound), but
only tested compute (fidelity/bf8-weights); its INTERCONNECT config was never swept. The VAE conv3d halo uses
`neighbor_pad_persistent_buffer(num_links=links)` where `links = min(upper_dims, ccl_manager.num_links)` (vae_ltx.py:252/261/309),
and the prof harness plumbs `NP_LINKS` env → `CCLManager(num_links=int(os.environ.get("NP_LINKS","2")))` (prof_vae_ltx.py:67).
So `NP_LINKS=1 vs 2` genuinely halves the halo link count on every halo'ing conv (upper_dims: H-halo=B·T≥2, W-halo=B·T·H≫2).
Harness `test_prof_vae_ltx_trace` (prints `TRACED_DECODE_WALL_MS=`, iters=10, ~45s warm, NO PCC gate — num_links is
bit-identical so this is a pure SPEED screen like the W-mask COST screen). num_links=2 is the BH HW cap (A2) ⇒ this can
only measure how much the 2nd link SAVES (characterization, not a new win, mirrors F2/I2 on denoise). Drift-immune per
Batch P: run BOTH NP_LINKS=2 (in-session baseline vs O's 551.91ms) and NP_LINKS=1 in ONE reservation. Answers: is the
552ms VAE data-movement halo/interconnect-bound (big Δ ⇒ BW headroom a future overlap could bank) or dispatch-bound too
(null ⇒ even the compute-bound bucket's data movement is latency-bound). ⚠️ NP_LINKS=1 neighbor_pad programs never warmed
⇒ minor partial recompile; generous timeout.
- [x] **S0 — NP_LINKS 2-vs-1: VAE halo IS interconnect-BW-sensitive (2nd link saves +16.5%) but num_links=2 already prod+HW-cap ⇒ NO win.**
  Job **042140-129** (drift-immune same-reservation A/B, re-verified from raw log): NP_LINKS=2 **TRACED_DECODE_WALL_MS=551.85**
  (in-session baseline ≈ O's 551.91, drift −0.06ms = CLEAN), NP_LINKS=1 **643.11** ⇒ **+91.26ms = +16.5% SLOWER**. Both
  `1 passed`, warm (prewarm 8194 targets 4354/3639ms, no timeout), 32 chips healthy. **The VAE decode's halo/interconnect
  IS link-BW-sensitive** — unlike S1 denoise (I2: num_links 0%, zero BW-sensitivity), the VAE looks like S2 (F2: +16%). This
  DECOMPOSES Batch O's "data-movement-bound 552ms": a real ~91ms chunk is conv3d halo INTERCONNECT transfer (`neighbor_pad`),
  not conv compute (O: LoFi fidelity −1.77% null) — the 2nd bucket besides S2 to show meaningful link sensitivity. **But NO
  shippable win:** num_links=2 is BOTH prod (create_pipeline device_configs) AND the BH HW cap (A2 — a 3rd link is HW-illegal),
  so prod ALREADY banks the 91ms; there's no headroom to add a link. Same verdict shape as F2/I2 (2nd link already at cap). No
  source edit (env A/B) ⇒ nothing to revert.
  → **BATCH S CLOSED — the last un-swept no-edit axis (VAE interconnect) is measured:** VAE's data-movement floor is part
  halo-BW (link-sensitive, HW-capped at 2) + part conv/permute/dispatch (fidelity-insensitive, O). Every no-edit config axis
  on BOTH regimes — dispatch-bound denoise (A/B/F/H/I/J/N/P/Q) AND data-movement-bound VAE (C/K/O/R/S) — is now swept with
  receipts; no no-edit lever clears the 5% gate at passing quality. Floor stays ~7.9s; 6.0s = out-of-repo step-distill.

## Batch T — S1 FSDP weight-sharding A/B (the ONE empty cell in the S1×collective-pattern matrix)
Batch Q measured is_fsdp on/off on **S2** (Q0: FSDP +10.5% slower — dispatch-bound block, more collectives hurt). The
S1 axis-sweep mirrors S2's on every axis EXCEPT FSDP: I1 topology · I2 links · N quant · P fp32acc were run on S1, but
is_fsdp was NEVER run there — the one empty cell in the S1×collective-pattern matrix (a HOLE, not a closed axis re-swept;
same shape as Batches I/N that produced real first-time S1 receipts). S1 is MORE dispatch-bound than S2 (I2 zero
BW-sensitivity vs S2's F2 +16%), so the MAGNITUDE of the FSDP penalty at S1 is genuinely unknown and sharpens the
dispatch-bound thesis. No-edit: block harness, mesh param `wh_4x8sp1tp0` (is_fsdp=True, test:113) vs `ring_bh_4x8sp1tp0`
(is_fsdp=False prod, :114) on stage_1 (:122), `LTX_NUM_LINKS=2` (:510) forces both to 2 links (wh default 4 is BH-illegal,
A2) isolating is_fsdp. Drift-immune (Batch P): BOTH cells in ONE reservation. WARM_FWD_MS (no tracy), video PCC-gated.
- [x] **T0 — S1 is_fsdp A/B (job 093451-9, resolved 2026-07-12 10:00Z): FSDP is +23.5% SLOWER on S1 = NULL/DEAD lever; prod is_fsdp=False optimal.**
  Drift-immune same-reservation A/B, both cells PASS: is_fsdp=True (wh_4x8sp1tp0) **WARM_FWD_MS 14.37** vs prod is_fsdp=False
  (ring_bh_4x8sp1tp0) **11.64**, PCC identical 99.9937%. Self-verified the A/B is clean, not confounded: source (test:113-114)
  = both Ring topology, sp1/tp0; the params differ in num_links (4 vs 2) too, but `LTX_NUM_LINKS=2` neutralizes it and the log
  confirms **both printed `num_links=2`** ⇒ is_fsdp is the sole live variable. Mirrors S2 Q0 (+10.5%) at ~2× the magnitude —
  confirms S1 is MORE dispatch-bound (I2 zero BW-sensitivity), so adding per-layer weight all-gathers hurts more. No source edit
  (param selection only) ⇒ nothing to revert. **Batch T CLOSED — the S1×collective-pattern matrix is now complete; every
  runtime/env-selectable no-edit axis on both stages has a receipt.**

## Batch U — CCL-matmul math-fidelity at bf16 (`all_lofi`): DEAD-ON-ARRIVAL, the preset does NOT exist
The 12:05Z dispatch premise was FALSE. It claimed `QuantConfig.all_lofi` (quant_config.py:100-115) and `all_weights_bf8`
(:86) are "distinct registered presets never run." **Source-verified 2026-07-12 12:33Z: they do not exist.** `QuantConfig`
(models/tt_dit/pipelines/ltx/quant_config.py) has EXACTLY 4 factory staticmethods — `default` (:78), `all_bf8_lofi` (:93),
`all_bf8_lofi_sdpa_lofi` (:133), `all_bf8_lofi_sdpa_lofi_fp32acc` (:146) — all already measured (baseline · H −1.1% · J
98.57% FAIL · L/P −4.9%). The cited lines are INSIDE existing bodies: :86 is `cross_attn_out=lc,` (a field in `default()`),
:101 is a docstring line in `all_bf8_lofi()`. The prior lap misread a field + a docstring as new factory methods and
dispatched a nonexistent preset name. The test resolves presets via `getattr(QuantConfig, LTX_QUANT)` (test:868), so the
run FAILED at `assert callable(_factory)`: **`AssertionError: LTX_QUANT='all_lofi' is not a QuantConfig preset`** — zero
device measurement, harness/config error, not a result.
Even had the preset been authored (a source edit, not a no-edit cell): `all_lofi` = LoFi math + bf16 dtype applies a strict
SUBSET of `all_bf8_lofi`'s optimizations (which stacks bf8 dtype on top), so its speedup is bounded ABOVE by the
already-measured `all_bf8_lofi` −1.1% (sub-gate) ⇒ predetermined-null within a CLOSED compute-precision axis (LoFi-math
alone already isolated null on VAE, Batch O −1.77%). Authoring it = the thrash the north star forbids. **Quant-preset space
is genuinely 4/4 CLOSED** (the "6 presets" belief was the misread). No source edited by the dispatch (env-only) ⇒ nothing to revert.
- [x] **`all_lofi` — DEAD-ON-ARRIVAL (job 120504-11, 2026-07-12 12:05Z): preset does not exist.** Test failed at the
  `getattr(QuantConfig, 'all_lofi')` factory assertion in 39.7s (`1 failed`, exit 0 masked by `| tail`). Source-verified
  the QuantConfig class has only the 4 measured presets; `all_lofi`/`all_weights_bf8` were a source-misread. No device data,
  nothing to revert. Batch U CLOSED; quant-preset axis 4/4 complete.

## Batch V — `LTX_SDPA_RING_CHUNK` env override: the LAST un-enumerated env knob — DEAD (half-wired no-op for both prod stages, source-verified)
Full source-level env-knob enumeration (2026-07-12 23:06Z lap: `grep os.environ models/tt_dit/`) surfaced ONE perf knob
never referenced in any batch or broker log AND live in the MODEL source (not a test-only var): `LTX_SDPA_RING_CHUNK`
(`attention_ltx.py:160`, comment "overrides the ring self-attn chunk for a sweep"). The other new-looking knob,
`LTX_CCL_SEQ_SCALE`, is test-only in `test_ccl_allgather_scoped.py:32` = the 2x4-only/tracy harness Batch R already
closed (fabric-DEAD + cold-timeout). So `LTX_SDPA_RING_CHUNK` was the only candidate for a genuinely-new no-edit cron cell
on the 30% SDPA bucket — Batch A swept chunk via the ISOLATED `test_ring_joint_sdpa` op (node-id params), never this
in-context block env hook.
- [x] **`LTX_SDPA_RING_CHUNK` — DEAD: half-wired no-op for BOTH production denoise stages (bit-identical A/B by
  construction, not dispatched).** Traced end-to-end: the override (`attention_ltx.py:160-163`) rewrites ONLY
  `ring_sdpa_chunk_size` → `self.ring_sdpa_program_config` (:164), the **fallback** config. But the ring self-attn forward
  (:512) selects `self._ring_pc_by_n.get(N, self.ring_sdpa_program_config)`, and `ring_sdpa_chunk_by_n` (:36-39) has
  entries for BOTH prod N — S2 `(True,8,4,38912)=(192,512)`, S1 `(True,8,4,9728)=(96,256)` — so `.get(N, ...)` HITS the
  per-N map and NEVER falls to the overridable config. ⇒ the env hook only fires on an N that MISSES the by_n map, which
  neither S1 nor S2 produce. Mirrors Batch B's `LTX_QUANT`-no-op class: authored hook, read, but bypassed on the prod path.
  Dispatching it would burn a reservation on a null-by-construction A/B (step-5 reservation-waste). The REAL S2 chunk it
  half-reaches, `(192,512)`, is EXACTLY Batch A's isolated baseline (4.85ms) — and Batch A already measured the only
  favorable direction from it, `(128,512)` 4.74ms = **−2.3% sub-gate on the isolated op** (~−0.7% of the dispatch-bound
  block) — reachable only by a SOURCE EDIT to `ring_sdpa_chunk_by_n[38912]`, i.e. a predetermined-sub-gate edit into the
  CLOSED SDPA-chunk axis = the north-star thrash. No device job, no source edit ⇒ nothing to revert. **Batch V CLOSED —
  the env-knob enumeration is now source-exhaustive: every perf-relevant knob the model reads is either swept (NUM_LINKS
  F2/I2 · QUANT H/J/N/P/U · NP_LINKS S0 · VAE/BWE/VOC_TRACE DONE) or a no-op/off-cron (SDPA_RING_CHUNK half-wired ·
  CCL_SEQ_SCALE 2x4-tracy-dead). No cron-safe no-edit cell remains.**

## DONE (measured, with the number)

## DONE (measured, with the number)
- audio-trace: SHIPPED -0.3s. VAE-trace: 0.19ms DEAD. num_links=4: HW-capped. RMSNorm QK-merge: null (45.08 vs 44.03). tilize: cold artifact. all_bf8 weights: -0.04s null.
- **all_bf8_lofi @ prod-4x8 video block: WARM_FWD_MS 16.69 vs F0 16.88 = −1.1% NULL @ PCC 99.89% PASS (job 010011-89).** CCL-matmul is collective/dispatch-bound, not compute- or BW-bound. (Old "0.876 FAIL" was a coarser path.)
- **S1 (stage_1) video block, first-ever receipts (jobs 010823-91/011036-93/011213-95):** baseline 12.73ms (Ring), num_links=1 12.71ms (0% = pure dispatch floor), Line 11.46ms (−10% crossover, char-only — fabric topology is a device-init constant, whole-run Line net-worse). 4× seq (S1→S2) = only 1.33× block time ⇒ dispatch-bound confirmed across scale.
- **all_bf8_lofi_sdpa_lofi @ prod-4x8 video block: FAILS PCC 98.57% < 98.80% (job 012439-97).** Stacking SDPA-LoFi on the bf8 base drops PCC 1.32pts below gate; speed truncated (moot — dead on quality). Quant axis fully measured: all_bf8_lofi is the sweet spot, further quant breaks the gate.
- **VAE-decode bf8 quant (job 013750-99): un-runnable — SEGFAULT at bf8 conv-weight upload (`from_torch(bfloat8_b)`, vae_ltx.py:193, 1st of 86 tensors).** The one COMPUTE-bound bucket (A1: traced≈untraced 552ms) resists the cheap quant flip; bfloat8_b needs TILE layout the upload path doesn't give. Proper bfp8-VAE = warm-authoring (parked w/ C W-mask fold + G subblock tune). Device recovered clean.
- **`all_bf8_lofi_sdpa_lofi_fp32acc` @ prod-4x8 video block (job 015415-106): PCC 99.93% PASS, WARM_FWD_MS 16.06 = −4.9% vs F0 16.88 (SUB-GATE, no win).** Closes the block-quant preset space 4/4 measured.
- **FSDP weight-sharding @ prod-4x8 S2 video block (job 024854-127, drift-immune same-session A/B): is_fsdp=True 18.56ms vs prod is_fsdp=False 16.79ms = +10.5% SLOWER, PCC identical 99.9658%.** FSDP adds per-layer weight all-gathers ⇒ MORE collectives ⇒ slower on the dispatch-bound block. Prod is_fsdp=False optimal; last no-edit collective-pattern axis closed. Adding collectives HURTS as much as removing link/compute helped little ⇒ only cutting collective COUNT (fewer steps = out-of-repo distill) moves the wall. Key finding: **fp32-dest-acc RECOVERS the SDPA-LoFi PCC** Batch J lost (98.57% FAIL → 99.93% PASS), correcting J's "can't recover" reasoning — but speed is null-with-favorable-noise (under 5% gate; increment vs all_bf8_lofi contradicts the isolated-SDPA physics). No preset clears the gate at passing quality ⇒ denoise block stays dispatch-bound.
- **S1 FSDP weight-sharding @ prod-4x8 S1 video block (job 093451-9, drift-immune same-reservation A/B): is_fsdp=True 14.37ms vs prod is_fsdp=False 11.64ms = +23.5% SLOWER, PCC identical 99.9937%.** The S1 mirror of Q0 (S2 +10.5%): FSDP's per-layer weight all-gathers cost ~2× more on S1 because S1 is more dispatch-bound (I2 zero link-BW-sensitivity vs S2's F2 +16%). Prod is_fsdp=False optimal on both stages. Closes the last empty cell in the S1×collective-pattern matrix ⇒ every runtime/env-selectable no-edit axis across S1/S2/VAE now has a receipt. (The one remaining hole-*looking* candidate, S1-SDPA-chunk, is structurally NOT a per-stage lever: SDPA chunk is a construction-time `SDPAProgramConfig` on the attention block — attention.py:45-46/transformer_block.py:43-44 — shared across both stages, already tuned on the dominant S2 bucket by Batch A; a per-stage chunk needs a source edit, not a no-edit run.)
- **VAE-decode num_links @ prod-4x8 (job 042140-129, drift-immune same-reservation A/B): NP_LINKS=2 551.85ms vs NP_LINKS=1 643.11ms = +16.5% (2nd link saves 91ms).** The VAE halo/interconnect IS link-BW-sensitive (unlike S1 denoise's 0%, like S2's +16%), decomposing Batch O's data-movement-bound 552ms into halo-interconnect (link-sensitive, HW-capped at 2) + conv/permute (fidelity-insensitive). No win — num_links=2 is prod AND the BH HW cap (A2). Closes the last un-swept no-edit axis (VAE interconnect); every axis on both regimes now has a receipt.

## Batch W — COLLECTIVE-COUNT reduction (source edit): the residual floor that program-launch does NOT explain

**Why this batch exists — the premise of Batches A–V was wrong.** Every prior batch ranked levers from an
**eager** profile. Eager is host-dispatch-bound (eager S1 2113ms ≈ eager S2 2136ms despite 4× the work) — a
different machine from the traced one that ships. The profiler had recorded **zero** traced ops the whole time:
`DEFAULT_PROFILER_PROGRAM_SUPPORT_COUNT = 1000` (profiler_state_manager.cpp:19) caps the profiler DRAM buffer at
1000 programs/RISC and LTX's eager prologue exhausts it, silently dropping 100% of every traced capture. That is
why attacking the "bottleneck" kept returning zero: it was never the bottleneck.

**Measured, traced (the real machine):** S1 (N=9,690) STEP_MS 348.3 (σ=0.1); S2 (N=38,760 = 4×N1) 1092.5 (σ=0.25).
Fit `t(N)=a+b·N+c·N²` ⇒ **work-independent floor a ∈ [100,299] ms/step = 1.1–3.3s of the 6.62s denoise.**
An A/B removing 144 programs/step (the gated-residual fold, a4173c3) moved it a real −1.32ms (S1) / −2.37ms (S2),
killing the pure-FLOPs AND pure-fixed-CCL explanations. Netting out the work-dependent memory pass (0.35ms @ S1)
leaves **0.97ms work-independent per 144 programs ⇒ ≤6.7 µs/program launch ⇒ all ~9,360 programs/step ≤ ~63 ms.**
**⇒ 37–236 ms/step of the floor is work-independent and is NOT program launch.** Prime suspect: **fixed collective
latency** (~87 collectives/block × 48 blocks; a collective's cost is largely payload-independent).

**This batch was NOT ruled out — it was never tried.** Batch Q/T concluded "only cutting collective COUNT moves the
wall" and then jumped straight to "= fewer steps = out-of-repo distill", skipping the obvious in-repo lever: **fewer
collectives PER BLOCK.** Q/T are in fact the strongest evidence FOR this batch — *adding* collectives (FSDP per-layer
weight all-gathers) cost **+10.5% (S2) / +23.5% (S1)**, the mirror image of the prediction here.

Rules: no-edit axes are exhausted, so these are SOURCE edits and each MUST carry a PCC gate (precedent:
`test_ltx_fold_gated_residual`, 7eda38e — in-process fused-vs-unfused A/B on real 22B weights with a bit-exact
noise-floor control, and a deliberately-wrong-gate mutant proving the gate bites at 99.85/6.0%). **A source edit with
no gate does not ship.** Measure on the block harness (`test_ltx_transformer_block -k 'ring_bh_4x8sp1tp0 and av'`,
`LTX_PROFILE_ITERS>1` for warm laps), not the raw pipeline.

- [x] **W0 — CENSUS + measured per-collective cost. → THE CCL-LATENCY HYPOTHESIS IS DEAD. Harness `test_ccl_census.py` (3d44dbc80d4).**
      **49 collectives/block × 48 = 2,352/step.** Slope-priced from traced replay wall-time (trace K copies, replay,
      (T64−T16)/48 = marginal cost of one more op in a traced step; cancels replay overhead, sidesteps the profiler
      entirely — so the METAL TRACE ID acid test is moot, there is no CSV). Reproduced across 2 runs, spread <1%.
      **A collective's FIXED cost is 6.6 µs (activation AG) to 9.3 µs (stat AG) — i.e. 1–1.6 program launches.
      All 2,352 collectives/step = ≈19 ms/step, against a 37–236 ms residual. Collectives are BANDWIDTH-bound**
      (video activation AG scales 105.5 → 402.2 µs for 4× tokens, ~72 GB/s, flat across stages). **There is no fixed
      collective latency to amortize. Cutting collective COUNT is NOT the lever. I predicted it was; the data says no.**
      What the floor IS, per the same census (two parts, both new):
      (a) **The audio branch is work-independent BY CONSTRUCTION.** Both stages use F=19 latent frames, so
      `audio_N=256` at S1 AND S2 → **32 rows/device = ONE TILE** either way. ~24 of the 49 collectives and roughly
      half the block's programs run on a one-tile tensor and cost the SAME at both stages ⇒ ≈30–55 ms/step, all of
      it landing in `a` by construction. **CAVEAT: this is REAL AUDIO COMPUTE, not overhead — the a+bN+cN² fit
      measures N in VIDEO tokens, so audio work is correctly but misleadingly binned into `a`. It cannot simply be
      deleted.** (It is, however, pathologically under-utilized: one tile of work spread over 32 devices.)
      (b) **Per-op fixed SETUP ≫ program launch.** `to_qkv` AG-mm carries **89.9 µs** of N-independent cost,
      `to_gate` 37.5 µs, vs a 5.83 µs bare program launch. Over ~195 ops/block that is plausibly 50–100 ms/step.
      **BONUS (the biggest lever found): `all_gather_minimal_matmul_async` DOES NOT MEANINGFULLY OVERLAP.** agmm_gate
      166.3 µs vs AG 105.5 + mm 58.3 = 163.8 ⇒ **only ~1.6% of the gather is hidden**; to_qkv hides ~23%. The gather is
      serialized before the matmul. 11 video AG-mms/block ⇒ full-overlap ceiling ≈1.0 ms/block ≈ **−1.0 s E2E**.
      **✅ CONFIRMED 2026-07-14 01:11Z — NOT RETRACTED — by an independent method (job 062, `opt/w1_links_ab.log`).**
      The objection to the line above was that `agmm ≈ AG + mm` assumes the fused op's internal compute equals the
      standalone matmul's, which need not hold (per-K-shard partials + accumulation). Valid objection; it does not
      bite. The **num_links falsifier** sidesteps it entirely — whatever the fused kernel's internal compute is, it is
      link-count-INDEPENDENT, so the swing from halving fabric BW measures purely how EXPOSED the gather is. Both link
      counts priced in ONE reservation; link-insensitive controls (`mm_qkv_video_gathered` 250.93 both runs,
      `prog_launch_ref` 5.83/5.84) are flat to <0.05 µs, so every delta below is signal, not drift.
      Standalone activation AG, num_links 2→1: 105.4 → 183.5 = **+78.1 µs** (the full BW sensitivity of one gather).
      A fused agmm inherits that swing in proportion to how exposed ITS gather is:
        `agmm_gate` 166.3 → 239.7 = +73.5 µs ⇒ **94% EXPOSED** (gate matmul is N=32 — no compute to hide behind)
        `agmm_out`  178.0 → 249.1 = +71.1 µs ⇒ **91% EXPOSED**
        `agmm_qkv`  326.4 → 379.1 = +52.7 µs ⇒ **68% EXPOSED** (widest matmul ⇒ hides most, and still only ~1/3)
      The level-decomposition and the BW-slope agree, and the level identity reproduces at BOTH link counts (1 link:
      AG 183.5 + mm 58.3 = 241.9 vs agmm_gate 239.7). It was not a coincidence. **Overlap grows with the matmul's
      compute, but even the widest projection hides only a third of its gather.**
      **REDUNDANCY: in all 6 attentions, `to_gate_logits` and `to_q`/`to_qkv` each fuse an AG of the IDENTICAL
      tensor** (attention_ltx.py:416 vs :419/:438) ⇒ **6 duplicate activation AGs/block, 288/step.**
      Also flagged (not a collective issue): `ff2` (mm+RS, 645 µs) + `ff1` (438 µs) = 1.08 ms/block = **52 ms/step
      at S1 — ~15% of the step in TWO ops.**
- [x] **W1 — CUT 1b: dedup the duplicate gate/qkv gather → MEASURED WIN, COMMITTED.** One explicit AG feeding two
      plain matmuls, replacing the two fused AG-mms that gathered the SAME activation twice. `LTX_DEDUP_GATE_GATHER`
      (default 1, `attention_ltx.py`); active only on Ring + tp>1 + gate-live (Linear topology already hoists the
      gather, so the cut is a no-op there).
      ⚠ **The W2 objection — "the fused op already overlaps, so hoisting the gather EXPOSES it and the dedup may be a
      LOSS" — is REFUTED BY MEASUREMENT.** The sign was the whole risk, and it came back positive: the gate's gather
      was already **94% exposed** (see W0), so the dedup had almost no overlap to sacrifice.
      **Traced, slope-priced per-attention A/B (job 062, `opt/w1_links_ab.log`, num_links=2, S1, drift <0.05 µs) —
      old (2 fused AG-mms) vs new (1 AG + 2 plain matmuls), each priced as ONE traced body. Every attention wins:**
        attn1_v (self, QKV) 494.0 → 425.2 = **−68.9 µs**    attn1_a (audio self)  98.5 → 66.2 = **−32.3 µs**
        attn2_v (cross, Q)  338.8 → 253.2 = **−85.7 µs**    attn2_a (audio cross) 77.1 → 47.6 = **−29.5 µs**
        a2v     (video Q)   334.8 → 228.0 = **−106.8 µs**   v2a (≈ attn2_a shape)             = **−29.5 µs**
      ⇒ **−352.6 µs/block traced** over the 6 attentions ≈ **−16.9 ms/step at S1** across 48 blocks. **Bank THIS
      number: production runs traced.**
      **✅ CONFIRMED END-TO-END IN THE TRACED PIPELINE — the only regime that ships (jobs 131/134, `STEP_MS` from
      `test_pipeline_ltx_distilled`, LTX_TRACED=1, denoise-only, SEED=10, `opt/w1_traced_dedup{0,1}.log`,
      `opt/w1_traced_dedup0_drift.log`). This is the number to quote; everything above it is bottom-up estimate.**
        S1 (vN=9728, 48 blocks)   346.85 ±0.11 (n=15) → **332.59 ±0.22 (n=15)** = **−14.26 ms/step, −4.11%** [221σ]
        S2 (vN=38912, 48 blocks) 1089.58 ±0.82 (n=5)  → **1045.02 ±0.78 (n=5)**  = **−44.56 ms/step, −4.09%** [88σ]
      ⇒ over the shipped 8×S1 + 3×S2 schedule: **−248 ms of denoise, banked.**
      **DRIFT CONTROL (the confound that would have voided this):** the OFF arm was re-run AFTER a full board reset
      that sat between the two arms. OFF pre-reset 346.66 vs OFF post-reset 346.85 = **+0.18 ms (+0.05%)** ⇒ the
      reset is worth nothing and the −14.26 ms is the flag, not a fresh board. Both decisive arms ran one identical
      source (`attention_ltx.py` mtime 01:38:32 < both runs; tree == HEAD).
      **⚠ THE EAGER HARNESS OVERSTATED THE ABSOLUTE WIN 5×, AND ONLY LOOKED RIGHT BY COINCIDENCE.** Eager said
      −1.54 ms/block; traced delivers **−0.297 ms/block** (14.26/48). ~81% of the eager win was host-dispatch that
      trace replays for free. The *percentages* happen to agree (−3.6% eager vs −4.11% traced) **only because eager's
      total is ~6× the traced total** — so a reviewer who banked the eager % got the right verdict for the wrong
      reason. On a lever with a smaller device component that same coincidence would have shipped a zero.
      **Bottom-up vs top-down:** the op-level slope sum predicted −16.9 ms/step at S1; E2E realizes −14.26 ms (84%).
      The per-attention pricing is a *ceiling* — it prices ops in isolation and does not fully transfer. Discount
      op-level sums ~15% before promising them.
      **Eager block harness (`WARM_FWD_MS`, AV, real 22B ckpt, ring_bh_4x8sp1tp0/stage_1): −1.5 to −1.8 ms/block** —
      i.e. LARGER than the traced saving, because an eager forward pays full per-op latency with no dispatch overlap.
      Do not quote the eager number as the production win. In-process A/B (job 101: one block, one weight load, flag
      flipped BETWEEN passes, 14 warm laps each, ON/OFF/ON): ON median 40.31/40.65 ms vs OFF 42.29 ms; the two ON
      passes' minima reproduce to 0.09 ms and median-drift is 0.34 ms ⇒ **−1.8 ms/block**. Use medians: the OFF pass
      carries a fat host-jitter tail (57.8 / 50.8 / 47.2 ms) that inflates its mean.
      ⚠ **The lap right after a flag flip is a COLD JIT COMPILE (1954 ms observed) — the flip selects op variants the
      program cache has not seen. Drop the first lap of EVERY pass, not just the first pass.**
      **PCC gate `test_ltx_dedup_gate_gather`** (AV-only — the per-head gate exists only with audio; real 22B weights,
      in-process A/B on one built model): dedup vs double-gather video **PCC 99.9997% / RMSE 0.2%**, audio **99.9999%
      / 0.1%**; same-path noise floor comes back bit-exact (100.0000%, RMSE 0.0%).
      **Gate PROVEN able to go RED:** `LTX_DEDUP_GATE_MUTANT=1` feeds the gate a corrupted copy of the gathered
      activation (the silent miswiring this cut could introduce) ⇒ **FAILS**, video RMSE 1.1% vs the 0.8% bound.
      ⚠ **The RMSE limb is what catches it — the mutant's PCC (99.9935%) still clears the 99.99% PCC bound. Do not
      loosen `_DEDUP_EQUIV_RMSE`; it is the limb doing the work.**
- [ ] **W2 — RE-OPENED. "The fused AG-matmul ALREADY overlaps → DEAD" is RETRACTED: REFUTED ON DEVICE.**
      The off-device kernel-source read (2026-07-13 22:30Z) is *structurally* accurate and its citations re-verify in
      tree: the gathered dim IS the matmul contraction (K) dim (`linear.py:207,212-214`;
      `..._program_factory.cpp:245-246`), K is ring-sharded, and the compute kernel streams gather→matmul per K-shard
      — local shard matmul'd with zero wait, each remote shard gated by its own `noc_semaphore_wait_min`
      (`matmul_dataflow_common.hpp:122,167`), partials L1-accumulated (`compute.cpp:329,404,405,440`), sender pushes
      each K-block to compute BEFORE mcasting (`dm_in0_sender.cpp:403-404`). No full-tensor barrier.
      **But structural streaming ≠ REALIZED overlap.** The num_links falsifier (W0, job 062) measures the gather to be
      **94% EXPOSED in `to_gate`, 91% in `to_out`, 68% in `to_qkv`**. The source told us overlap was POSSIBLE; the
      device says it is barely HAPPENING.
      **⚠ METHODOLOGICAL RECEIPT — do not re-derive "W2 is dead" from the kernel source again. Source code cannot
      answer "how much is actually hidden"; only a bandwidth-slope (num_links) measurement can. A code path that
      streams is not a path that overlaps.**
      ⇒ The lever is LIVE, and the ≈1.0 ms/block ≈ **−1.0 s E2E** full-overlap ceiling from W0 STANDS (it is a ceiling
      on a hypothetical fix, and W1 does not claim it — W1 only deletes the duplicate gather).
      Most promising next shape: give the gather real compute to hide behind by fusing gate+QKV into ONE wide AG-mm
      (concatenated weights, one gather, ~309 µs of matmul to overlap) — the natural successor now that W1 has already
      hoisted the gather out and left it standalone/fully-exposed by construction.
  - [x] **PRICED — the gate+QKV weight-concat fusion is a FUND-WORTHY WARM `attention_ltx.py` lever (job 568, harvested 2026-07-14 23:42Z).**
    Fused `agmm_gateqkv` vs post-W1 baseline `AG+mm_gate+mm_qkv`: **S1 345.46 vs 415.18 = −69.72 µs/site (−16.8%); S2 1132.35 vs
    1600.88 = −468.53 µs/site (−29.3%).** Dominant term = the MATMUL-FOLD (gate out=32 concats onto 12288-wide QKV at +0.08%:
    `mm_gateqkv_gathered 251.20 ≈ mm_qkv_gathered 251.01`), NOT gather-overlap (S1 hides only 11%, S2 74%). Beats all priced
    alternatives (`cut1b_new` 422.69, `cut1b_old` 467.64). **NOTE:** the original decision-rule keyed only on gather-hiding, which
    undercounts S1 — the matmul-fold carries the win; total-cost accounting is what decides. E2E per-step ceiling (48 sites):
    S2 22.5 ms/step (~2.1%), S1 3.3 ms/step (~1.0%). Follow-on = WARM weight-concat authoring, NOT a cron fire-and-exit.
- [x] **W3 — the ~90 µs per-op fixed SETUP cost → NO NEW LEVER (the one candidate is receipt-DEAD; the rest have no API).**
      Off-device source attribution (2026-07-13 23:17Z lap), citations re-verified in-tree. The census-priced 89.9 µs is
      **100% on-device replay command-stream time, NOT host program-setup** — traced `enqueue_trace` emits one fixed-size
      command independent of op count (`fd_mesh_command_queue.cpp:1066-1107`, `trace/dispatch.cpp:79-167`,
      `compute_trace_cmd_size:169-189`); all RTA/CB/semaphore payloads are assembled ONCE at capture (`TraceNode`,
      `trace_node.hpp:33-47`) and binaries cached — so replay re-pays only the recorded dispatch + the op's own kernels.
      **Decomposition via the qkv-vs-gate 2.4× ratio** (both are the SAME fused `all_gather_minimal_matmul_async` on the
      SAME input `spatial_1BND`, so their AG/fabric half is byte-identical; qkv N_out=3·4096=12288 vs gate N_out=32=384×
      wider — `attention_ltx.py:124` vs `:130-132`, dims `transformer_ltx.py:475-476`): **~52 µs = qkv's wide-matmul
      weight DRAM read + per-N-block LLK/CB overhead; ~31.7 µs = shared per-op fabric-mux/AG-ring handshake; ~5.8 µs =
      dispatch.** LEVER VERDICT — all DEAD: **(1) bf8 wide-matmul weights (the subagent's only "real" lever) is
      RECEIPT-DEAD — Batch H `all_bf8_lofi` (qkv/out/ffn weights+acts→bf8) already measured −1.1% NULL @ passing PCC
      (job 010011-89).** The physics says bf8 halves the 52 µs weight BW, but the WALL doesn't move ⇒ the weight read is
      hidden behind the collective/dispatch critical path (block is collective/dispatch-bound per H0/F2), OR the
      slope-priced marginal is off the replay critical path. Either way precision does not cut the floor. **(2) persistent
      fabric-mux to erase the 31.7 µs handshake — DEAD: the mux is program-scoped (`..._program_factory.cpp:841`,
      `FabricMuxConfig`), each op does a per-invocation `build_and_connect`/`close_mux` (`dm_in0_sender.cpp:157-158,
      629-645`) baked into the trace + re-run every replay; NO ttnn API hoists it cross-op (only the device-1D-EDM
      underneath is persistent, already amortized). Hoisting = new fabric arch, not a knob.** (3) L1 weight cache across
      replays — DEAD (48 blocks × 12-25 MB ≫ L1, no intra-step reuse). **⇒ W3 CLOSED: the residual floor's biggest term
      is NOT precision-attackable and has no persistent-connection API. Reconciles W0's per-op census with Batch H's null
      quant — the 52 µs is real per-op work but off the wall's critical path. Consistent with the whole-session
      collective/dispatch-bound thesis. Reinforces: 6.0s needs step-distillation, not a per-op setup cut.**
- [x] **W4 — the audio one-tile pathology → DEAD (the ~28 audio collectives/block are ARCHITECTURALLY pinned).**
      Off-device source dig (2026-07-13 23:30Z lap), citations re-verified in-tree. The audio branch issues **28
      collectives/block** (12 stat-sized TP RMSNorm AGs + 16 activation; 24 TP + 4 SP), but only ~14 are true one-tile
      audio (self-attn + text-cross + FFN); the 2 A↔V cross-attentions are **video-scale** (their sharding is dictated by
      video, not audio). The one-tile count cannot be cut: **(1) SP-replicate = the pre-priced-DEAD lever** (saves 4 audio
      SP AGs, inflates ~11 AG-mms 32→256 rows). **(2) TP-replicate audio → BLOCKED by construction:** A↔V cross-attn
      requires audio K/V TP-sharded to 8 heads/device to match video's head layout (`transformer_ltx.py:488-509`,
      `attention_ltx.py:96,487`); un-TP-sharding breaks the cross SDPA. **(3) sub-mesh / fewer-device audio → DEAD:** A↔V
      couples audio↔video EVERY block (`transformer_ltx.py:480-537` re-verified) — A→V needs audio KV visible to all 32
      video devices, V→A needs full SP-sharded video KV at the audio devices; those cross collectives are pinned by
      *video's* 4×8 layout, so a sub-mesh removes only the ~14 tiny audio collectives at the cost of a full↔sub-mesh
      entry-gather + exit-broadcast **per block**, each fully exposed + video-scale ⇒ strictly worse. No ttnn mid-trace
      submesh-dispatch API on the statically-4×8 tensors either. **(4) batch/merge → DEAD:** stat-AG merge pre-priced dead;
      TP activation gathers are sequential-dependent (Q-gather→SDPA→out-gather chain), not batchable. **Kill term:** the
      residual floor `a` is fixed per-collective latency, and audio's collective COUNT is pinned by the A↔V head-layout +
      SP-layout coupling to video — every count-reducing reshard either breaks cross-attn or adds ≥2 exposed video-scale
      reshards/block. **⇒ W4 CLOSED. Reinforces the whole-session thesis: the count is architectural, 6.0s needs
      step-distillation (fewer STEPS), not a per-block collective cut.** One lean-loss survivor spun out to W5 below.
- [ ] **W5 — A→V ring-fold (the sole un-dead crumb from W4; honestly LEAN-LOSS, run only when the tree is clean).**
      A→V currently does an EXPLICIT `all_gather_persistent_buffer` on the small audio KV (`transformer_ltx.py:494-497`)
      then a locally-tuned SDPA on the LONG video Q; V→A already ring-folds its gather (`attention_ltx.py:587-614`,
      `is_cross=True`). Lever: make A→V also hit `use_ring_cross` (pass SP-sharded audio K-rope + `kv_logical_n=audio_N`)
      → removes 1 exposed SP collective/block ×48. **Predicted sign = WASH, leaning LOSS** (INCLUDING exposed latency):
      saves ~1 fixed collective latency but pays ~1 ring-PC fixed overhead + retile risk — the ring cross-PC is tuned
      `q_chunk=64` for the SHORT audio Q (`attention_ltx.py:204-211`), badly sized for A→V's LONG video Q; the existing
      explicit/ring asymmetry looks deliberate for exactly this reason. **A/B (needs a CLEAN tree — UNBLOCKED as of 2026-07-15 01:27Z: transformer_ltx.py + attention_ltx.py are
      clean, only foreign `pipeline_ltx_distilled.py` is modified; still WARM authoring, not a cron cell — model source
      edit + new census-harness body):** add `LTX_A2V_RING_CROSS` env toggle, slope-price the A→V attention
      body both ways at S1 (q=1216) + S2 (q=4864) via the census harness, confirm STEP_MS delta. Revert unless slope drop
      beats the added ring-PC fixed cost. LOW priority — a ~1/28-collective crumb against a 2.2s gap.
      **MEASURED envelope (skip-cross ablation, nl2, ms/block):** cross-modal (a2v+v2a) = full-AV − skip-cross
      = **12.95ms @ S2** (job 408, 41.24−28.29; 31% of AV block) and **13.11ms @ S1** (jobs 380/410, 40.47−27.36);
      audio-self = skip-cross − video = 11.52 @ S2 / 15.52 @ S1.
      **⚠️ 2026-07-14 15:04Z — the "cross-modal is VIDEO-SCALE SDPA compute ⇒ LEAN-LOSS" reasoning above is REFUTED
      ON DEVICE.** Cross-modal is FLAT across a 4× video-token change: 13.11 @ S1 (9,690 tok) ≈ 12.95 @ S2 (38,760).
      If it were cross-attn SDPA on the long video Q it would scale ~4× (a2v/v2a SDPA is 256×N_video ≈ 0.66% of
      video-self's N_video², sub-ms, and video-linear); flat instead ⇒ the 13ms is **fixed collective/dispatch/adaLN
      latency in the a2v/v2a path — work-INDEPENDENT**, i.e. exactly the north-star COUNT target. **W5 UPGRADED from
      "confirmed crumb" to "un-refuted, worth the direct A/B"** (the `LTX_A2V_RING_CROSS` toggle) — magnitude unknown
      but the SDPA-dead dismissal no longer applies. (Caveat: audio-self wobbles ±~4ms wrong-way across stages =
      cross-reservation subtraction noise; far below the ~10ms drop the SDPA thesis predicts, so the refutation holds.)

### DEAD, with evidence — do not retry
- **Merging the RMSNorm stat all-gathers: DEAD, quantitatively.** 2 AGs = 30.7 µs; 1 merged (2× width) = 22.7 µs
  ⇒ saves 8.0 µs, but the merge needs ≥1 concat + 2 strided slices ≈ **≥17 µs of programs — a net loss.** On audio
  it loses outright even before the extra programs (2×4.46=8.9 → merged 12.3). **This is the quantitative reason the
  earlier Q/K-norm merge was slower: an AG's fixed cost is only ~1.6 launches, so there is nothing to amortize.**
- **`norm3` twice/block is NOT redundant** — the residual is updated between the two calls. No cut.
- **TP-replicated residual stream:** trades a 105 µs AG for an all-reduce (RS+AG ≈ 300 µs). Strictly worse.
- **num_links=4:** physically impossible on BH (2 eth channels). 1→2 already gives 184.2 → 105.5 µs. Maxed.
- **Shortening the in0 store-and-forward chain (grid.y / force_transpose routing): DEAD, measured.** Job 060
  (`test_agmm_chain_probe.py`, 2026-07-14 00:10Z) slope-priced the AG-mm gate at grid.y ∈ {9,7,5,4}: the gather
  portion (agmm−mm) is FLAT ~110µs (110.74→110.24) across a 2.25× chain-length range. The ~110µs exposed gather is
  intrinsic ring-hop fabric bandwidth, NOT on-chip chain traversal ⇒ no chain-length lever. (Consistent with W0
  bandwidth-bound ~72 GB/s.)
- ~~[ ] **W0 — CENSUS + measured per-collective cost.**~~ Exact table of every collective in one AV block (type /
      file:line / mesh axis / payload shape / count), split **stat-sized** (RMSNorm stats — tiny payload, pure
      latency) vs **activation-sized** (bandwidth-bound). Then the MEASURED fixed latency per collective from
      `cpp_device_perf_report.csv` (`OP TO OP LATENCY` + `DEVICE FW DURATION`). **ACID TEST: rows with a non-empty
      `METAL TRACE ID` must be > 0** — it was 0 in every pre-2026-07-13 CSV, which means no traced data; if it is 0
      the numbers are worthless and must NOT be reported. Deliverable gates the rest of the batch.
- [ ] **W1 — merge the tiny RMSNorm stat all-gathers.** `DistributedRMSNorm.forward` (layers/normalization.py) =
      pre_allgather → `all_gather_persistent_buffer(stats, dim=-1)` → post_allgather = 3 programs, 1 collective,
      EACH. Many norms per block. Can several norms' stats ride ONE collective? **⚠️ A naive version of this was
      already measured SLOWER:** the Q/K-norm "merge" cost 8 programs vs 6 — it ADDED programs. Understand exactly
      why before proposing a variant; a merge that raises program count is dead on arrival.
- [ ] **W2 — redundant all-gathers.** Find any tensor all-gathered twice on the same axis within a block, or gathered
      then immediately re-sharded. Each is a free cut.
- [ ] **W3 — fuse a collective into an adjacent op's epilogue** (same trick as the gated-residual fold, applied to a
      collective rather than an elementwise op).

## Batch W2 — hide the activation all-gather behind the big matmul — **CLOSED: the exposed gather is NOT a tunable, it is the fused kernel's core-assignment structure. Two named fixes REFUTED by measurement; one prescriptive fix left, and it is a program-factory redesign, not a knob.**

**The op already materializes the gathered activation — and it is 1/4 GARBAGE.** `compute_output_specs`
(`all_gather_minimal_matmul_async_device_operation.cpp:336-341`) gives output slot 0 the shape `in0 * ring_size`, and
the caller even supplies the buffer (`linear.py:214 persistent_output_buffer=ag_persistent_buffer`). The host wrapper
then throws it away (`device_operation.cpp:487-488`: `strip_count = 1 + fsdp; return {begin()+strip_count, end()}`).
**But it is not a valid all-gather.** `read_in0_block_sync` under `READ_FROM_LOCAL_INPUT` sources the device's OWN
K-slice from the unsharded input (`in3`) straight into the compute CB and never writes it back to the gather buffer,
so that slice is never written — and the buffer is `torch.empty` (`manager.py:176`), so the hole reads as
uninitialized garbage, not zeros. **MEASURED** (`test_agmm_gathered_out.py`, jobs 138/152, rows=256 and 1216):
remote K-slices **PCC = 1.000000** (bit-exact, all 96 cells), own K-slice **PCC = -0.004 .. +0.004 (pure noise)**,
matmul output **PCC 0.99999** (correct — proving the local data reaches compute via in3 and is never written back).
=> Route 2a ("just read the gathered buffer") **silently reads garbage today** and needs a local-slice writeback.

**And route 2a is not worth fixing on its own.** Its entire value is the overlap the fused op already achieves:
S1/site today `AG(105.5)+mm_gate(58.3)+mm_qkv(251.2)=415.0` vs `agmm_qkv(332.6)+mm_gate(58.3)=390.9` = **-24 us/site**
— and the local-slice writeback (a 2.49 MB L1->DRAM write per device) must be paid out of exactly that. Nets ~0.
**The whole prize is the overlap itself** (-105 us/site if the gather fully hid), which would ALSO speed up
`agmm_out` (177.6) and `agmm_ff1` (438.4). Do not ship 2a without the overlap fix.

**The overlap is ~ZERO today, not 25%** (census, job 152 post-reset): exposed gather = fused - pure_mm:
gate_s1 108.2 of 105.5 (**-3% hidden**), qkv_s1 81.0 of 105.5 (23%), gate_audio 22.5 of 12.3 (**-82%**),
qkv_audio 19.4 of 12.3 (**-57%**). On small shapes the fused op's own gather costs *more* than a standalone
all-gather. NOTE: the old `agmm_qkv_video_s2=1060.79` census row is **BAD** — it implies W1 was a wash at S2, which
contradicts W1's measured -44.56 ms/step. Back-solving W1 gives agmm_qkv_s2 ~1342 (=23% hidden, same as S1). Row is
noisy across runs (1008-1093); do not price S2 off it.

- [x] **CB depth (in0_cb depth 2, `program_factory.cpp:366`) — REFUTED.** The comment there says the depth-8 in0_cb
  was an upstream *perf* change reverted for an L1 OOM, so it looked like the fix. Depth 4 **OOMs L1 by 6,912 B**
  (1,579,776 vs 1,572,864 max — one in0 block = 64 tiles x 2048 B = 131,072 B). Depth 3 fits and **moves nothing**
  (job 142): agmm_qkv_s1 331.91->330.31 (-0.5%, noise), agmm_out +0.1%, agmm_ff1 +0.3%, agmm_gate -0.1%.
  **Why it cannot work:** the in0 loop fills exactly ONE block per iteration and then enters the blocking fabric
  send, so the reader can never run more than one block ahead *no matter how deep the CB is*. Depth is not the slack.
- [x] **Per-packet flush amortization (packet-header rotation) — REFUTED.** `forward_half_block_to_fabric_neighbor`
  calls `noc.async_writes_flushed()` after EVERY packet (`matmul_dataflow_common.hpp:262`), and it is load-bearing:
  `fabric_unicast_noc_scatter_write_with_state` mutates a shared packet header then NOC-writes payload+header
  *non-blocking*. At 2 tiles/packet (`num_tiles_to_write_per_packet = min(4, 4KB/2KB) = 2`) a sender core takes ~128
  blocking flushes per K-block-loop, ~= the whole exposed gather by arithmetic. Implemented header rotation
  (rotate N scatter headers, flush once per N, plus a mandatory flush before the ready-semaphore packet so it cannot
  overtake a payload). **Correct but worthless**: at FWD_PKT_HDRS=4 (job 153) PCC stays 1.000000/0.99999 and
  agmm_qkv_s1 330.61->334.50 (**+1.2%, if anything worse**), agmm_gate flat, agmm_ff1 +2.1%. Cutting flushes 4x moved
  zero time => the sender is **not** stalled on local NOC-flush latency; it is stalled in `wait_for_empty_write_slot()`
  on **wire backpressure**. Both knobs REVERTED (they were env-gated, default = current behavior).
  WARNING **TRAP (cost a hang + a board reset, jobs 144/146/148):** the BH packet-header pool is `144 B x (6*2*2) = 3456 B`
  -> **12 headers per RISC**, and `dm_in0_sender` allocates a full set for EACH of its two Ring directions. Allocating
  >12 silently overflows (`allocate_header` only `ASSERT`s, compiled out in release), corrupts L1 and **wedges an eth
  core** — the ARC-heartbeat health gate does NOT catch this; it takes a full board reset via the broker.
  Budget `2*(hdrs+2) <= 12`.

**=> THE MECHANISM (by elimination + source).** The 24 fabric-relay cores are also COMPUTE cores. The relay is issued
inline on the in0-feed RISC (`dm_in0_sender.cpp:520,542`; senders are chain indices `size-1`/`size-2` = grid rows
y=7,8 of each of the 12 columns), and that RISC is the ONLY thing that refills its own core's in0 compute CB. So
while it blocks on wire backpressure, its core's matmul starves — and the op's wall time is the max over cores.
Those 24 of 108 cores therefore pay (full compute share) + (wire time), which is exactly "gather adds, does not hide",
and exactly why the num_links bandwidth slope leaks into the fused op. **Not a math limit:** the gather IS along K
(the contraction), partial K-shards DO accumulate in L1, and the compute kernel already streams per-K-block
(`compute.cpp:421-459`). The structure is right; the core assignment is wrong.

- [x] **W2-next — UNBURDEN THE RELAY CORES — SHIPPED, but SMALL. The lever is RETIRED: it was NOT the biggest one.**
  Commit `agmm: shrink the fabric-relay cores' N share`. Give the 2 fabric rows the smallest N share the other 7
  can absorb; they then have less matmul to starve while the inline relay blocks on the wire.

  **THE BLOCKER WAS MIS-SCOPED — no kernel change was needed at all.** The real constraint is not
  `current_N_block_tiles` underflow, it is the **in0 mcast chain handshake**: the chain runs ALONG the N axis
  (`build_core_order_for_axis`, chain index == in1_idx == the core's N slice) and hands one block per (m,n,k) with a
  request/ack (`dm_in0_sender.cpp:422-432`), so a core that iterated a different number of N BLOCKS would leave its
  neighbour waiting on a request that never comes → deadlock. But the N **loop bound** (`N_blocks_per_core`) need not
  change to move N: only the tile count INSIDE the last block moves. That pins every core to the same loop count and
  bounds `N_i` to `((B-1)*N_block_tiles, B*N_block_tiles]`. The kernels ALREADY run a partial last N block (today
  every core does 11 of 12 tiles), so this is **program-factory only, zero kernel edits**.
  Free lunch: because `subblock_w` rounds up, the 7 absorbing rows do the SAME number of MAC columns at 12 tiles as
  at 11 → they pay **+0%** compute; the relay rows drop 33% (qkv) / 50% (out, ff1) / 88% (qkv_audio).

  **MEASURED (traced, drift-controlled, `test_pipeline_ltx_distilled.py`, jobs 200/203/205 + 209/210):**
  STEP_MS **S1 346.7 → 344.7/345.2 (−1.4 to −2.1 ms)**, **S2 1089.9 → 1086.4 (−3.5 ms)**. Drift control: the two
  bracketing baselines agreed to **0.03 ms** (S1) / 0.50 ms (S2); S1 σ=0.13 ⇒ the effect is ~11-16σ. Default ON;
  `TT_AGMM_FABRIC_N_PCT=0` restores uniform.
  Census (jobs 170/171/172, drift-controlled): every fused AG-mm site −2% to −6% (qkv_s1 332.4→325.4, out 177.8→173.2,
  ff1 438.5→417.6, ff1_v_s2 1402→1316); one small regression (`ff1_audio` +3.4%). **`gate` moved +0.02 µs (0.0%)** —
  the perfect internal control: its N is ONE tile, so the split provably has no room and falls back to uniform.

  **⇒ THE MECHANISM IN THE W2 DIAGNOSIS IS REFUTED. The relay cores do NOT pay "full compute PLUS wire".**
  The predicted −35..−45 ms/step assumed the relay core's time is `compute + wire` (additive), so halving its matmul
  should have returned half its compute. It did not: cutting relay compute by 33-50% returned only **2-6%** of the op
  (qkv_s1 −7.0 µs of the 81 µs exposed gather, not −56). The relay RISC pushes the block to the compute CB BEFORE it
  relays (`dm_in0_sender.cpp:429`, "push data to compute before mcasting"), so per K-block the core costs
  **max(relay, compute)**, not the sum — the relay core's matmul was ALREADY hidden behind the wire, and what is
  exposed is **wire time**, on BH's HARDWARE-CAPPED 2 eth links. This is the same conclusion the CB-depth-3 null and
  the packet-header-rotation null were already pointing at, and it is now confirmed by moving compute directly.
  **Corollary: gate_s1's 108 µs exposed gather (≈ a full standalone AG) is NOT recoverable by core assignment** —
  gate has no N to give away, and even with zero matmul the relay cores would still pay the wire.
  **What WOULD move it:** less wire (bf8 activation gather — `ag_activation_video_s1_bf8` is 62.3 µs vs 105.5 bf16,
  already in the census), or fewer gathers (W1's dedup), NOT better core assignment. Do not re-open this lever.

## Batch W6 — DE-FUSE the AG-matmuls whose fusion is a measured LOSS — **DEAD. Sign INVERTED end-to-end: +6.63 ms/step at S1 (+2.0%, 66σ). Reverted. And it burns the op-level census as a tool for signing a fusion cut.**

**The idea (from the W2 census).** The fused `all_gather_minimal_matmul_async` only pays when its matmul is
big enough to hide the gather behind; W2 measured `gate_s1` and the one-tile audio AG-mms costing MORE fused
than a standalone `all_gather` + a plain matmul. W1 had already harvested exactly that for `to_gate_logits` +
`to_q`/`to_qkv` (−14.26 ms/step S1). W6 extends it to every AG-mm W1 did not touch.

**Enumeration (post-W1, 10 fused AG-mm sites/block).** W1's dedup (`LTX_DEDUP_GATE_GATHER`, default on) already
de-fuses gate + Q/QKV in all 6 attentions, so those are GONE — do not double-count them. What is left:
6× `to_out` (`attention_ltx.py:341`, fuses the gated residual too), `a2v.to_kv` + `v2a.to_kv` (fused only
because their cross-attn context is TP-sharded; the text prompt is replicated so `attn2`/`audio_attn2` to_kv
is already a plain matmul), and 2× `ffn.ff1` (`linear.py:207`). `ff2` is a different op (matmul+reduce-scatter)
and its collective is an epilogue, not a prologue — not a de-fuse candidate.

- [x] **W6a — per-site fused-vs-defused pricing (job 154, `opt/census_defuse.log`).** Extended `test_ccl_census.py`
      with paired bodies (`c1c_*`): each site priced as ONE traced body per arm, so the de-fused arm carries its
      own extra program launch + intermediate buffer. Drift-clean (`prog_launch_ref` 5.84, `ag_activation_video_s1`
      105.40, `mm_gate_video_s2_gathered` 180.38 — all match prior runs to <0.1%); in-run replicate
      (`agmm_ff1_video_s1` 447.49 vs `c1c_fused_ff1_v_s1` 449.76, same op) pins within-run noise at ~0.5%.
      S1, µs/op: | site | ×/blk | pure_mm | AG | fused | AG+mm | Δ | exposure |
      | audio ff1 | 1 | 49.73 | 12.33 | 72.08 | **58.38** | −13.70 | 181% |
      | audio to_out | 3 | 25.64 | 12.33 | 40.74 | **36.09** | −4.65 | 122% |
      | video ff1 | 1 | 334.84 | 105.40 | 449.76 | **416.41** | −33.35 | 109% |
      | a2v to_kv | 1 | — | — | 52.18 | 51.89 | −0.29 | wash |
      | video to_out | 2 | 108.34 | 105.40 | **184.66** | 211.22 | +26.56 | 72% |
      | a2v to_out | 1 | — | — | **108.42** | 127.15 | +18.73 | — |
      | v2a to_kv | 1 | — | — | **179.44** | 199.62 | +20.18 | — |
      S2: video ff1 **1435.88 fused vs 1508.05 de-fused = +72.17 ⇒ THE SIGN FLIPS WITH THE MATMUL'S SIZE**;
      video to_out +54.21; a2v to_out +32.32; v2a to_kv +31.66.
      **The predictive rule is exposure = (fused − pure_mm)/AG > 100%, NOT "small matmul"** — it calls the sign
      4/4 where pure_mm is measured. **⚠ CORRECTS THE W0 BRIEF: `agmm_out` does NOT de-fuse favourably.** Its
      "91% exposed" is a num_links BANDWIDTH-SLOPE, a different quantity from the level comparison that decides a
      de-fuse; on levels it is 72% exposed and de-fusing it COSTS 26.6 µs/site. to_out stays fused.
- [x] **W6b — PCC gate: BIT-EXACT, and the gate PROVABLY BITES.** `test_ltx_defuse_small_agmm` (job 156,
      `opt/defuse_pcc.log`): 1-layer AV instrument, real 22B weights, module-global flag flipped between forwards
      in ONE process. De-fused vs fused **PCC 100.0000% / RMSE 0.0% — bit-identical** (same K-order, fp32 dest acc),
      *tighter* than W1's dedup (99.9997%/0.2%). Same-path noise floor also bit-exact. A once-per-shape log line
      proves the cut FIRED on all three shapes — a green gate on a path that never ran proves nothing.
      **Mutant (job 157, `opt/defuse_mutant.log`, `LTX_DEFUSE_AGMM_MUTANT=1` feeds the hoisted gather's consumer a
      0.5× copy): FAILS at PCC 94.84% / RMSE 34.2%.** The mutant's CONTROL arm still passed bit-exact, which also
      proves the in-process flag flip really takes effect — the failure mode that would have made the bit-exactness
      meaningless.
- [x] **W6c — TRACED END-TO-END: REFUTED, sign inverted.** `test_pipeline_ltx_distilled`, LTX_TRACED=1,
      denoise-only, SEED=10 (`opt/defuse_traced_{off,on,off_drift}.log`, jobs 159/161/165). Baseline = W1-ON (the
      shipped default), so this stacks on top of W1.
        **S1: 332.73 ±0.10 (n=15) → 339.36 ±0.10 (n=15) = +6.63 ms/step, +1.99% [66σ] SLOWER.** Predicted −2.46.
        **S2: 1046.06 ±0.38 (n=5) → 1045.84 ±0.32 (n=5) = −0.22 ms, −0.02% = NULL.** Predicted −1.11.
      **DRIFT CONTROL:** OFF re-run after the ON arm = 332.79 (+0.06 ms, +0.02% vs the first OFF) ⇒ board stable.
      **INTERNAL CONTROL (stronger, and free):** across the same three jobs S2 is FLAT (1046.06/1045.84/1045.78)
      while S1 moved +6.63 — board drift would have moved both. The regression is the flag.
      **S2 IS the audio-only arm** (the video-ff1 shape key `(4864,4096,4096)` never fires; only `(32,2048,512)`
      and `(32,2048,2048)` do) ⇒ **the audio de-fuse on its own delivers −0.22 of a predicted −1.33 ms/step = a
      NULL, not a small win.** Back-solving S1: audio ≈ −4.6 µs/blk (stage-independent) ⇒ **the video-ff1 de-fuse
      costs ≈ +143 µs/block on the real critical path, against the census's −33 µs — a 176 µs/block error on ONE
      site.** Source REVERTED (`linear.py`, `attention_ltx.py`, `test_transformer_ltx.py`); the census rows are
      kept as the receipt, carrying the warning below.

### ⚠ METHODOLOGICAL RECEIPT — THE OP-LEVEL SLOPE CANNOT SIGN A FUSION CUT (this is the durable finding)
**K back-to-back copies of a MULTI-OP body pipeline across iterations; a SINGLE fused op cannot pipeline with
itself.** Copy i's matmul overlaps copy i+1's all-gather (fabric and compute are different engines), so a
(AG + matmul) body's slope understates what that AG costs on a real dependency chain, where it has nothing to
hide behind. **The bias was visible in the census's own numbers and I discounted it:** every de-fused paired body
priced CHEAPER than its separately-priced parts summed (ff1_v 416.41 vs 440.24; out_v 211.22 vs 213.74; ff1_a
58.38 vs 62.06; out_a 36.09 vs 37.97). That −24 µs internal inconsistency is only a LOWER BOUND on the real bias,
which came back at 176 µs/block. **Use the census to RANK ops and to price a DELETION (W1 removed a gather — that
transferred at 84%). Do NOT use it to price a RE-ARRANGEMENT that moves a collective onto the critical path;
that must be measured traced, end-to-end.** This is the same lesson as the eager-harness receipt (W1: eager
overstated 5×), one level up: a harness that is honest for one class of change silently lies about another.

**~~Leading hypothesis for the extra ≈143 µs/block:~~ AG ping-pong buffer collision with W1's hoisted gathers —
KILLED BY SOURCE (Batch W7 below). No device time spent. The mechanism it needs does not exist in the op.**

## Batch W7 — AG ping-pong buffer collision ("6 gathers sharing 2 buffers serialize") — **DEAD ON ARRIVAL, source-verified. NO device time burned. The predicted ΔSTEP_MS is 0 BY CONSTRUCTION, not by measurement.**

The W6 post-mortem's leading hypothesis was that `get_ag_ping_pong_buffer` keys on SHAPE, so W1's 6 hoisted
attention gathers share one 2-buffer pair, and each gather stalls waiting for a buffer to free. **Five
independent kills, all mechanical. The lever was never implemented and never run.**

1. **THERE IS NO BUFFER-KEYED WAIT ANYWHERE IN THE OP.** This is the decisive fact and it is not an estimate.
   The writer (`minimal_default_writer.cpp`) writes into the peer's output buffer over the fabric and increments
   `out_ready_sem`; it *never reads, locks, or waits on the output buffer*. The reader's only wait is
   `noc_semaphore_wait_min(out_ready_sem, ...)` (`minimal_default_reader.cpp:292,368`) — waiting for *peer data
   to land*, which is the collective's inherent cost — then it resets the sem to 0 (`:392`). A second gather
   with the same buffer does not block; it simply writes. **"Each must wait for a buffer to free before it can
   start" describes a mechanism that is not in the code.**
2. **The one barrier that exists is COMPILED OUT on exactly this path.** `all_gather_async_default_program_factory
   .cpp:734` sets the kernel's `use_barrier_sem` = `barrier_semaphore.has_value() && !using_persistent_buffers`
   (arg map verified 1:1 against `minimal_default_writer.cpp:104-110`). `using_persistent_buffers` is true for
   every W1 gather (`all_gather_async_device_operation.cpp:277`), and `manager.py:461` passes
   `barrier_semaphore=None` on the persistent path anyway ⇒ **doubly false, barrier block (writer :224-272)
   skipped.** The persistent buffer is what *buys* the barrier-free path — it is a correctness device, not a
   scheduling resource.
3. **The dispatcher is buffer-blind, so buffer identity CANNOT change the schedule.** The address enters as a
   *runtime arg* (`output_tensor.buffer()->address()` → `writer_rt_args[0]`, patched by
   `override_runtime_arguments` :835-849). tt-metal keeps no buffer dependency graph; programs on a CQ run in
   enqueue order, and under trace the addresses are baked into the replayed command stream. Re-keying the buffers
   yields an **identical program, identical kernels, identical semaphores, identical op order** — only the
   destination address moves. **ΔSTEP_MS = 0 by construction.** Consistency check that closes it: if the shared
   buffers *did* serialize the gathers, nothing would be enforcing it — the code would be RACY, not slow. It is
   bit-exact today. Bit-exact + zero enforcement ⇒ the gathers are already ordered by op-issue order. That is
   precisely the "clean NULL" branch: **the buffers are never contended.**
4. **The fix targets the LOOSER of the two resources.** Buffer key = `("ag", shape, dim, mesh_axis, dtype)`
   (`manager.py:166`) → per-shape. But the **semaphore** key is `mesh_axis` ALONE (`manager.py:236-239`): 2 sets,
   alternating on *every* call, so it recycles every **2 CCL ops on the axis regardless of shape** — strictly
   TIGHTER than the per-shape buffer. If ping-pong depth serialized anything, the semaphore would bind first, and
   distinct *buffer* keys would not relax it by a single op. (The 2-deep sem ping-pong is what grants the 1-op
   device-skew tolerance that lets the barrier be compiled out — load-bearing, not vestigial.)
5. **The premise "all 6 have the SAME shape" is FALSE.** Of the 6 attentions, 3 are video-stream (`attn1`,
   `attn2`, `audio_to_video_attn` — `spatial_1BND` = video_normed / video_ca_input / video_q_a2v) and 3 are
   audio-stream (`audio_attn1`, `audio_attn2`, `video_to_audio_attn` — audio_normed / audio_ca_input /
   audio_q_v2a). Local N = **1216 video vs 256 audio** (independently confirmed by `opt/agbuf_forensics.log`
   rows=1216 / rows=256) ⇒ **two distinct keys, 3 gathers each. Never 6 on one pair**, and the de-fused ff1
   gather would have been the **4th** on the video key, not the 7th.

**The +6.63 ms W6 regression needs no buffer collision — the parsimonious cause was already on the table.**
`all_gather_minimal_matmul_async` **overlaps the gather with the matmul** (that is the whole point of the fused
op: gather chunks stream into the matmul as they land). De-fusing it into (explicit AG) → (plain matmul) takes a
gather that was *hidden behind compute* and exposes it as serial fabric time on the critical path. Magnitude fits
(~143 µs/block for a 1216-row video gather that previously cost ≈0 exposed), and it **explains the S2 null for
free**: S2 never fires the video-ff1 key, so only the small 256-row audio sites changed, where the exposed cost
is ≈4 µs/block = noise. The census's own internal inconsistency (every de-fused paired body priced CHEAPER than
its parts summed) is the *same* effect surfacing as a lower bound. **Occam beats the exotic mechanism; and the
exotic mechanism does not exist in the source.**

**DURABLE RULE:** the AG persistent ping-pong buffer is a **correctness device against 1-op device skew** (it is
what allows `use_barrier_sem=0`), **NOT a concurrency resource.** Adding buffer keys can never buy parallelism:
CCL ops on one device do not overlap, and nothing in the dispatcher or the kernels consults buffer identity to
schedule. Do not re-raise "give the gathers their own buffers" as a perf lever.

## Batch Q — QUANT, RE-MEASURED ON THE TRACED INSTRUMENT (Batches H/J/N were all EAGER — and eager is blind to this)
**The whole quant space was closed on `WARM_FWD_MS` (the eager block harness). That instrument is
HOST-DISPATCH-BOUND and therefore structurally incapable of seeing a device-time win.** Receipts for the
indictment: eager full-pipeline S1 2113 ms ≈ S2 2136 ms despite **4× the tokens** (work-independent ⇒ host-bound);
eager 44 ms/block vs traced 6.93 ms/block = **6.3× on identical AV blocks**; and W1 (an op-COUNT cut) measured
eager −1.54 ms/block but traced only −0.297 ⇒ **eager OVERSTATES an op-count cut 5×**. The mirror image must also
hold: a change that cuts DEVICE time but not op count is **masked**. Quant is exactly that change. So H/J/N's
"`all_bf8_lofi` = −1.1%, no win" was taken in the one regime that cannot see the win.
**The traced denoise is device/work-bound** (the precondition): S1 348.3 @ N=9,690 vs S2 1092.5 @ N=38,760 ⇒
linear fit gives fixed a≈100 ms and work-dependent b·N = 71% of S1 / 91% of S2. And W0/W6a localize it:
`mm_qkv_video_gathered` 250.93 µs is a **link-INSENSITIVE pure matmul**, and `ff1` = 334.84 µs matmul + 105.40 µs
gather ⇒ a large pure-matmul-compute bucket exists for quant to attack.

**What `all_bf8_lofi` actually does (source-verified, NOT what the docstring claims):** linear weights bf16→**bf8_b**
(to_qkv/to_q/to_kv/ff1/ff2; `to_out` stays bf16 by carve-out); `mm_`/`ff_compute_kernel_config` HiFi2→**LoFi** and
fp32_dest_acc True→**False**. **SDPA is byte-for-byte UNCHANGED** (both baseline and preset = HiFi2/fp32acc=False).
⚠ **`activation_dtype` and `_sdpa_input_dtype` are DEAD CODE — declared in the presets, never read by any consumer
(LTX or Wan).** So activations stay bf16 and **the collectives still move bf16: there is no CCL-bandwidth win.**
The win is purely matmul-internal (LoFi math phases, bf8 weight DRAM reads, freed dest registers).

**PRE-REGISTERED PREDICTION (before the run):** S1 −30 ms (range −15..−60), S2 −110 ms (range −50..−220);
`..._sdpa_lofi_fp32acc` 0..+5 ms vs `all_bf8_lofi` (Batch B measured SDPA-LoFi null at op level, and fp32acc ADDS
cost). Null criterion: |ΔS1| < 3 ms ⇒ quant is a true null and the space closes for real.

- [x] **Q0 — `LTX_QUANT` IS wired into the traced pipeline (contrary to the Batch-B-era "no-op" warning).**
      `pipeline_ltx.py:376-380` calls `_maybe_apply_quant_config()` **before** `_prime_caches` ⇒ weights typecast as
      they load and the trace is captured over the quantized ops; `_build_transformer_cache_name` is quant-tagged, and
      **the bf8 tensorbin caches already exist and are complete** (`…q-all_bf8_lofi/transformer`, 3419 files = same
      count as the bf16 baseline, 23G vs 37G) ⇒ a quant run is a cache HIT, not a 22B safetensors reload.
      The **AV oracle was NOT wired** — `test_ltx_transformer_model` never read `LTX_QUANT`. Now it does
      (`_run_inner_step`, via the same `QuantConfig` factory the pipeline uses), so the preset faces the diffusers
      AV reference at pcc 0.992 / rmse 0.15 — a tighter bar than the block test's 0.988.
- [x] **Q1 — `all_bf8_lofi` TRACED: a REAL WIN. Eager understated it ~5.7×. SHIPPED-QUALITY.**
      Traced `STEP_MS`, `LTX_PROFILE_DENOISE_ONLY=1`, SEED=10, prod 4x8 Ring (jobs 173/174/176/177, logs
      `opt/quant_{A_base,B_bf8lofi,C_bf8sdpa,A2_drift}.log`). Quant confirmed live in-log: *"Applying LTX quant config
      to 48 transformer blocks (has_audio=True) | Weight dtypes {BFLOAT16, BFLOAT8_B} | fidelities {LoFi}"*.
      | arm | S1 (n=15) | ΔS1 | S2 (n=5) | ΔS2 |
      | baseline A / A′(drift) | 332.88 / 333.01 | — | 1046.42 / 1044.56 | — |
      | **pooled baseline** | **332.94** | — | **1045.49** | — |
      | **`all_bf8_lofi`** | **312.06** | **−20.88 (−6.27%)** | **992.78** | **−52.71 (−5.04%)** |
      | `..._sdpa_lofi_fp32acc` | 316.61 | −16.33 (−4.90%) | 989.16 | −56.33 (−5.39%) |
      **DRIFT CONTROL:** baseline re-run AFTER both quant arms = S1 +0.13 ms / S2 −1.86 ms vs the first baseline ⇒
      board stable; the effects are 10–30× the drift, and σ≈0.1 makes −20.88 ms a ~200σ move.
      **INTERNAL CONTROL (stronger):** the two presets **cross over** — `all_bf8_lofi` wins S1 while
      `..._sdpa_lofi_fp32acc` wins S2. A board/clock artifact moves both stages of both arms the same way; a
      shape-specific inversion cannot be drift. Also `_ttnn.so` mtime 03:31:16 was unchanged across all four arms
      (03:59–04:15) ⇒ the sibling's rebuild did not land mid-flight.
      **E2E denoise (8×S1 + 3×S2):** 5800.0 → **5474.8 ms = −325.2 ms (−5.61%)** for `all_bf8_lofi`
      (vs −299.6 ms for the SDPA variant) ⇒ **`all_bf8_lofi` is the better E2E arm despite losing S2**, because the
      schedule is S1-heavy.
      **Mechanism check:** −20.88 ms/step over 48 blocks = **−435 µs/block**, i.e. ~26% off the ~1.7 ms/block
      matmul-compute bucket — inside the pre-registered 25–40% band. Prediction was directionally right and
      magnitude-conservative-correct (measured −20.9 vs predicted −30, inside the stated range).
      **PCC GATE — PASSES, and the gate provably BITES** (AV oracle, 1-layer, ring_bh_4x8sp1tp0, gate pcc≥0.992 /
      rmse≤0.15; jobs 201/204/206):
      | arm | stage | video PCC | video RMSE/σ | audio PCC | audio RMSE/σ |
      | baseline (0 quant hooks) | S1 | 99.9985% | 0.6% | 99.9981% | 0.6% |
      | baseline (0 quant hooks) | S2 | 99.9984% | 0.7% | 99.9982% | 0.6% |
      | **`all_bf8_lofi`** | **S1** | **99.9967%** | **2.0%** | **99.9967%** | **0.9%** |
      | **`all_bf8_lofi`** | **S2** | **99.9966%** | **2.2%** | **99.9967%** | **1.0%** |
      Quant costs real precision (RMSE/σ 0.6% → 2.2%, a 3.3× degradation — so the metric is not asleep) yet lands
      far inside the gate. ⚠ **HONEST LIMIT: the in-tree AV oracle is 1-LAYER; it cannot see 48-layer error
      compounding.** That risk is separately retired by the fact that **`all_bf8_lofi` is ALREADY the shipped
      `LTX_QUALITY=medium` quant** (`utils/ltx.py: FAST_QUANT = "all_bf8_lofi"`) on the real 48-layer pipeline.
- [x] **Q2 — `all_bf8_lofi_sdpa_lofi_fp32acc`: real, but WORSE E2E. Do not ship.** −16.33 (S1) / −56.33 (S2).
      It **loses 4.55 ms at S1 and wins 3.62 ms at S2** vs `all_bf8_lofi`. Coherent physics and it **refines Batch B**:
      SDPA-LoFi is *not* uniformly null — its saving scales with seq² so it only pays at S2, while the bundled
      fp32_dest_acc=True is a ~fixed accumulation tax that dominates at S1. Net E2E −299.6 vs −325.2 ms ⇒ KILL for
      the S1-heavy production schedule. (A per-stage preset — bf8_lofi on S1, sdpa variant on S2 — would give
      −336.0 ms, only **11 ms** beyond `all_bf8_lofi`: not worth two weight caches + two quant states.)

→ **BATCH Q: the quant space is REOPENED and WON. `all_bf8_lofi` = −325 ms/generation of denoise (−5.6%) at
  PCC 99.997% / RMSE 2.2% against a 0.992 / 0.15 gate. SHIP IT.**
  **The durable lesson (this is the point):** Batches H/J/N/K/P closed quant as "null" on `WARM_FWD_MS`. The
  measurement was real; the *instrument* was wrong. **Eager overstates op-count cuts ~5× and understates
  device-time cuts ~5×** — the two errors have opposite sign, so eager cannot be "corrected" with a fudge factor;
  it must not be used to price ANY device-time lever. **Every axis previously closed on `WARM_FWD_MS` alone
  (math fidelity, dest-acc, weight dtype — Batches H/J/N, and arguably the VAE fidelity work in K/O) is
  UN-closed and deserves re-measurement on traced `STEP_MS`.**

## Batch QA — ACTIVATION quant: the dead `activation_dtype` field was the biggest untapped lever. **SHIP.**

**The premise, verified.** `activation_dtype` (LinearQuantConfig) had **zero consumers** anywhere in `models/tt_dit/`
— declared in every preset, read by nobody. `_sdpa_input_dtype` is **NOT** dead in general: `wan2_2/attention_wan.py:378-420`
consumes it. It was dead **in LTX only** (`attention_ltx.py` never read it), and no LTX preset ever set it.
So the shipped `all_bf8_lofi` was matmul-internal only: **weights bf8, activations bf16, collectives moving bf16 bytes.**

**WHERE the cast must go (the whole lever is in the placement).** Source-verified mechanism:
- `all_gather_minimal_matmul_async_device_operation.cpp:66-72` validates act and weight dtypes **independently** ⇒ a bf8
  activation composes with the bf16-weight carve-out the fused addcmul epilogue needs (`:574` ties ternary_a's tile size
  to **in1/weight**, not in0/act). The carve-out is not an obstacle.
- `..._program_factory.cpp:365` `in0_data_format = ag_output_tensor.dtype()` (= the persistent output buffer,
  `device_operation.cpp:371-373`), and `:721` `l1_scratch_cb_page_size_bytes = in0_tile_size` ⇒ **the fabric page size IS
  the gathered dtype's tile size.** bf8_b tile 1088 B vs bf16 2048 B ⇒ **−47% fabric bytes.**
- `manager.py:152-180` keys the AG ping-pong buffer on dtype **and allocates at it**; every call site already passes
  `dtype=x.get_dtype()` ⇒ a bf8 input gets a bf8 buffer for free. No plumbing change needed.
- ⇒ cast at `ColParallelLinear.forward` (to_qkv/to_q/to_kv/ff1) + `_to_out_fused_addcmul` (to_out), and — the trap —
  **BEFORE** the explicit `all_gather_persistent_buffer` on the dedup-gate path. Casting at the linear there would hit the
  already-gathered tensor and shrink nothing. NOT wired on RowParallel/replicated Linear: their input never crosses the
  fabric, and RowParallel's input is the 4×-wide FFN intermediate ⇒ pure cost.

**⚠ THE BUG THIS EXPOSED (would have silently poisoned the residual stream).** The matmuls default their output dtype to
the **input's**: `output_dtype.value_or(in0_input_tensor.dtype())` (`minimal_matmul_device_operation.cpp:222`,
`all_gather_minimal_matmul_async_device_operation.cpp:333`). So quantizing the input pushes bf8 **downstream** — first
crash was `FusedRMSNormPreAllGather` TT_FATAL "Input tensor must be BFLOAT16, got BFLOAT8_B" (norm_q/norm_k), and left
unfixed it would have made the **residual stream bf8**. Fix = `resolve_output_dtype()`: pin the output back to bf16 when
the input is block-float. **The lever is input-side and must stop at the output.**

**PRE-REGISTERED PREDICTION (before any run):** ΔS1 −10 ms (range −4..−16), ΔS2 −40 ms (range −15..−60); null if |ΔS1|<3 ms;
regression a live outcome if the gathers are already overlapped. **Both landed inside the range.**

- [x] **QA1 — `LTX_QUANT_ACTIVATIONS=1` (linear activations bf8): −14.2/S1, −59.2/S2. SHIP.**
- [x] **QA2 — `+ LTX_QUANT_SDPA_BF8=1` (also cast ring-SDPA Q/K/V): −15.2/S1, −96.7/S2. STRICTLY BETTER. SHIP BOTH.**
      Traced `STEP_MS`, prod 4x8 Ring, SEED=10, `LTX_PROFILE_DENOISE_ONLY=1`, on top of the shipped `all_bf8_lofi`.
      Logs `opt/actq_traced_{P_ctrl,Q_act,R_actsdpa,P2_drift,P3_drift}.log`; jobs 218/225/234/230/236.
      | arm | S1 (n=15) | ΔS1 | S2 (n=5) | ΔS2 |
      | control ×3 (P/P′/P″) | 310.21 / 310.19 / 310.16 | — | 988.86 / 988.74 / 988.50 | — |
      | **pooled control** | **310.19** | — | **988.70** | — |
      | **Q: + activations** | **295.95** | **−14.24 (−4.59%)** | **929.48** | **−59.22 (−5.99%)** |
      | **R: + activations + SDPA** | **295.00** | **−15.19 (−4.90%)** | **892.02** | **−96.68 (−9.78%)** |
      **E2E denoise (8×S1 + 3×S2): 5447.6 → 5036.1 ms = −411.5 ms (−7.55%)** for R (Q alone: −291.6 ms, −5.35%).
      **DRIFT: three controls spread S1 0.05 ms / S2 0.36 ms** — the effect is ~300× the drift, and the arm ranges
      (294.9–296.5) never approach the control's (310.0–310.4). σ≈0.05–0.2 ⇒ −15.2 ms is ~75σ, −96.7 ms is ~250σ.
      `_ttnn.so` mtime **04:59:18 unchanged across all five arms**.
      **⚠ RE-BASELINE WAS MANDATORY:** the sibling's quant receipt (S1 312.06 / S2 992.78) was taken on a `.so` from
      03:31:16; a C++ rebuild landed at **04:59:18**. Measuring against their number would have manufactured a fake
      +1.9/+3.9 ms of "win". Every arm here is on the current binary.
      **MECHANISM — it really is bandwidth.** ΔS2/ΔS1 = 59.22/14.24 = **4.16× against a 4.0× token ratio** ⇒ the saving
      scales linearly with tokens, the signature of a byte-count lever (a fixed-overhead or launch lever cannot).
      The SDPA cast splits the same way and harder: it adds −37.5 ms at S2 but only −0.95 ms at S1, because ring-SDPA's
      K/V gather is the SP-axis payload and grows with seq. Magnitude check: the census's standalone bf8-vs-bf16 AG
      (62.5 vs 105.5 µs @S1, −43 µs) over 8 video TP-gathers × 48 blocks bounds S1 at −16.5 ms if fully exposed; measured
      −14.24 ms = **86% of the gather-only budget** ⇒ the win is predominantly fabric, and matmul-internal in0 reads can
      account for at most the remainder. (HONEST: I did not separate the two terms; the bound is what I can defend.)
      **PCC — PASSES, and better than weight-quant alone** (AV oracle, 1-layer, `ring_bh_4x8sp1tp0`, gate pcc≥0.992 /
      rmse≤0.15; jobs 216/231/235, logs `opt/actq_pcc_{A_act,B_sdpa,C_combined}.log`):
      | arm | S1 video PCC / RMSE | S2 video PCC / RMSE | S1 audio | S2 audio |
      | `all_bf8_lofi` (weights only) | 99.9967% / 2.0% | 99.9966% / 2.2% | 99.9967% | 99.9967% |
      | + activations | 99.9972% / 1.9% | 99.9970% / 2.1% | 99.9965% | 99.9966% |
      | + SDPA inputs only | 99.9970% / 1.9% | 99.9968% / 2.1% | 99.9968% | 99.9968% |
      | **+ activations + SDPA (R)** | **99.9973% / 1.8%** | **99.9972% / 2.0%** | **99.9967%** | **99.9967%** |
      Activation quant costs **no measurable accuracy** on top of weight quant. Coherent: with LoFi math + bf8 weights
      already in place the mantissa is truncated regardless, so bf8 activations add negligible marginal error. The gate is
      NOT asleep — it moved 0.6%→2.2% RMSE for the weight quant, and it caught the bf8-into-RMSNorm crash first.
      **The `_sdpa_input_dtype` scare was a category error:** the prior "SDPA-LoFi FAILS PCC at 98.57%" receipt is about math
      **fidelity**, a different knob. The **input dtype** is safe; fidelity is not. Both live in SDPAQuantConfig — do not conflate.
      **HONEST LIMITS:** (1) the AV oracle is 1-LAYER and cannot see 48-layer compounding — needs the 1080p decoded-video
      gate a sibling is building (coordinate, do not duplicate); (2) the OFF path is unchanged **by source argument**, not
      by measurement (both helpers are identity when `activation_dtype is None` and x is bf16, so the op sequence is
      byte-identical) — no pre-change run exists on the current `.so` to prove it; (3) `all_bf8_lofi_sdpa_bf8` as a *preset*
      is un-measurable on the pipeline: the tensorbin cache is keyed on the preset NAME, so it cache-misses and
      re-materialises the 22B checkpoint despite byte-identical weights. Hence `LTX_QUANT_SDPA_BF8` as an env flag. **A
      real cleanup is available: key the transformer cache on the weight dtypes, not the preset name** (`pipeline_ltx.py:672`).
      **Cold-JIT note:** the bf8-in0 matmul variants are new kernels. `prewarm_and_submit.sh -c` alone did NOT suffice
      (arm Q still timed out at 298s); what worked was run → offline `kernel_prewarm` → re-run (2nd run 218s, JIT 100%).
- [ ] **NEXT: make `LTX_QUALITY=medium/fast` carry both flags** once the 1080p decoded-video gate signs off — `utils/ltx.py`
      `FAST_QUANT` sets the preset but not these env flags, so the served tiers do NOT yet get this −411 ms.
- [x] **PRICED (job 596, census `sp_ag_video_kv_v2a_*`): the v2a cross-attn video-K/V SP-gather bf8 cast is a LARGE un-banked
      fabric payload — E2E ceiling ≈ 90 ms. FUND-WORTHY warm lever.** `_quant_cross_attn` never sets `_sdpa_input_dtype`
      (SDPAQuantConfig is documented self-attn-only), so the v2a ring-cross video K/V (audio Q attends the full video seq ⇒
      video K/V all-gathered across SP, seq-scaled) still gathers bf16. Standalone SP-gather bf16→bf8: **S1 227.85→136.19 =
      −91.66 µs (−40.2%); S2 881.87→500.91 = −380.96 µs (−43.2%)**, drift anchors dead-on (`prog_launch_ref` 5.84,
      `sp_ag_audio_kv` 16.20). Byte-count signature confirmed (bf16 3.87× / saving 4.16× for 4× tokens = the QA2 fabric
      fingerprint). Ceiling (v2a fires once/block): 48 blk × [8×S1 91.66 µs + 3×S2 380.96 µs] ≈ 90 ms. **⚠ UPPER BOUND** —
      standalone gather, not the ring-folded exposure (QA2 precedent says ring-SDPA bf8 realizes its byte saving, so mostly
      realizable, but un-measured in-fold). Wiring = source edit to `_quant_cross_attn` + own PCC gate (cross-attn K/V ≠
      self-attn tensor class; QA2's self-attn PCC pass does not transfer) + prewarm + traced pipeline = WARM-authoring, NOT
      cron. The a2v audio-K/V mirror is negligible (`sp_ag_audio_kv` 16.2 µs = 1 tile, not seq-scaled).
- [ ] **NEXT: make `LTX_QUALITY=medium/fast` carry both QA flags AND (once funded) the v2a cross-K/V bf8 cast** — gated on
      the 1080p decoded-video quality sign-off. The v2a cast (~90 ms ceiling) stacks on top of QA2's −411 ms.

## QUALITY GATE — the instrument, and why every earlier one was wrong

**A few flipped bits completely re-roll the generation.** Swapping `addcmul(t,t1,t2)` for the algebraically
identical `add(t, multiply(t1,t2))` — same math, only different bf16 rounding, a provable no-op — moves the
final S2 latents to **PCC 87.7 / RMSE 49.6%** and the decoded pixels to **PCC ~0.83**. The path is 48 blocks ×
11 sampler steps with feedback, so rounding-scale perturbations amplify super-linearly (~133×).

⇒ **NEVER gate on scene equality: latent PCC, pixel PCC, SSIM, PSNR, "is it the same video".** Those measure
re-roll, not quality. A correct optimization looks catastrophic on every one of them, and a gate built on them
rejects every valid change.

⇒ **Gate on REFERENCE-FREE attributes:**
| purpose | instrument | notes |
|---|---|---|
| video quality | **VBench** (`utils/vbench.py::assert_vbench_quality`) | subject/background consistency, motion smoothness, dynamic degree, imaging quality. Per-height floors already calibrated (1088: 0.92/0.93/0.955/1.0/0.645). Was dark because vbench was never installed + `RUN_VBENCH=0` everywhere. |
| prompt adherence | **CLIP** (video-vs-prompt) | reference-free. Proven power: the poisoned-cache (unconditional) render scores 6.98 vs baseline 18.58. |
| audio | **intrinsic** properties (peak, % clipped, spectral flatness, noise floor) | NOT waveform PCC — it is phase-sensitive and two valid renditions score ~0. |
| numerics (1 layer only) | in-process A/B + bit-exact noise floor + a MUTANT proving the gate goes red | RMSE/σ is the limb that catches bugs; a wrong-tensor mutant PASSED a 99.99% PCC bound. |

**Installed:** `vbench==0.1.5`, `decord==0.6.0`. ⚠️ **`pip install vbench` pins `transformers==4.33.2`, which
DELETES `transformers.models.gemma3` and breaks the pipeline.** Re-pin `transformers==5.10.2` after any
vbench install, and re-verify `from transformers.models.gemma3 import Gemma3ForCausalLM`.
