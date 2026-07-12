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
- [~] **G0 — get_matmul_config subblock tune (WARM-AUTHORING).** Scoped + source-mapped; not a cron lap-tail (shared
  util edit + possible build_metal + iterate loop). Parked alongside the W-mask in-kernel fold (Batch C) for a warm session.

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

## DONE (measured, with the number)
- audio-trace: SHIPPED -0.3s. VAE-trace: 0.19ms DEAD. num_links=4: HW-capped. RMSNorm QK-merge: null (45.08 vs 44.03). tilize: cold artifact. all_bf8 weights: -0.04s null.
- **all_bf8_lofi @ prod-4x8 video block: WARM_FWD_MS 16.69 vs F0 16.88 = −1.1% NULL @ PCC 99.89% PASS (job 010011-89).** CCL-matmul is collective/dispatch-bound, not compute- or BW-bound. (Old "0.876 FAIL" was a coarser path.)
- **S1 (stage_1) video block, first-ever receipts (jobs 010823-91/011036-93/011213-95):** baseline 12.73ms (Ring), num_links=1 12.71ms (0% = pure dispatch floor), Line 11.46ms (−10% crossover, char-only — fabric topology is a device-init constant, whole-run Line net-worse). 4× seq (S1→S2) = only 1.33× block time ⇒ dispatch-bound confirmed across scale.
- **all_bf8_lofi_sdpa_lofi @ prod-4x8 video block: FAILS PCC 98.57% < 98.80% (job 012439-97).** Stacking SDPA-LoFi on the bf8 base drops PCC 1.32pts below gate; speed truncated (moot — dead on quality). Quant axis fully measured: all_bf8_lofi is the sweet spot, further quant breaks the gate.
- **VAE-decode bf8 quant (job 013750-99): un-runnable — SEGFAULT at bf8 conv-weight upload (`from_torch(bfloat8_b)`, vae_ltx.py:193, 1st of 86 tensors).** The one COMPUTE-bound bucket (A1: traced≈untraced 552ms) resists the cheap quant flip; bfloat8_b needs TILE layout the upload path doesn't give. Proper bfp8-VAE = warm-authoring (parked w/ C W-mask fold + G subblock tune). Device recovered clean.
- **`all_bf8_lofi_sdpa_lofi_fp32acc` @ prod-4x8 video block (job 015415-106): PCC 99.93% PASS, WARM_FWD_MS 16.06 = −4.9% vs F0 16.88 (SUB-GATE, no win).** Closes the block-quant preset space 4/4 measured.
- **FSDP weight-sharding @ prod-4x8 S2 video block (job 024854-127, drift-immune same-session A/B): is_fsdp=True 18.56ms vs prod is_fsdp=False 16.79ms = +10.5% SLOWER, PCC identical 99.9658%.** FSDP adds per-layer weight all-gathers ⇒ MORE collectives ⇒ slower on the dispatch-bound block. Prod is_fsdp=False optimal; last no-edit collective-pattern axis closed. Adding collectives HURTS as much as removing link/compute helped little ⇒ only cutting collective COUNT (fewer steps = out-of-repo distill) moves the wall. Key finding: **fp32-dest-acc RECOVERS the SDPA-LoFi PCC** Batch J lost (98.57% FAIL → 99.93% PASS), correcting J's "can't recover" reasoning — but speed is null-with-favorable-noise (under 5% gate; increment vs all_bf8_lofi contradicts the isolated-SDPA physics). No preset clears the gate at passing quality ⇒ denoise block stays dispatch-bound.
