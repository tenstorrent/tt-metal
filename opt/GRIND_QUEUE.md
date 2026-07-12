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
- [ ] Make a full-pipeline run actually COMPLETE: run the prewarm capture+compile, then submit the run TWICE (first warms the trace, second measures) OR raise the run-stage timeout via a 2-reservation split. Then measure 6+2 E2E speed + quality.
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
- [~] **F0 — video-block warm-FW baseline @ ring_bh_4x8** (job **001703-77**, 00:17Z). Establishes the warm VIDEO
  per-block FW (all prior block runs were `av`→SKIP; the 44.77ms number was AV single-blocking). Substrate for F1's
  topology A/B. Extract `WARM_FWD_MS=` + confirm `PASSED block PCC`. NEXT lap: read log, tick `[x]`, dispatch F1.
- [ ] **F1 — video-block warm-FW @ line_bh_4x8 (Topology.Line)** vs F0 Ring. Node-id
  `test_ltx_transformer_block[NOTSET-ckpt_fast-pcc-video-stage_2-line_bh_4x8sp1tp0]`, same env/iters. Clean no-edit
  CCL-topology A/B: does Linear all-gather beat Ring for the fused matmul at block scale? >5% FW delta = a win to wire in.

## DONE (measured, with the number)
- audio-trace: SHIPPED -0.3s. VAE-trace: 0.19ms DEAD. num_links=4: HW-capped. RMSNorm QK-merge: null (45.08 vs 44.03). tilize: cold artifact. all_bf8 weights: -0.04s null.
