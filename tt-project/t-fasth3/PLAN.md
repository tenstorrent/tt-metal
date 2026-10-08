# FastH3 plan (task #282, 2026-10-08)

Research and planning only. No device jobs ran for this task.

## 0. Bottom line

- Everything the project measured on H3 so far was **slow H3**: the H3 base model, 50 steps, dense or VSA attention, on `fasth3-opt` (t159, t161, t209, `baselines/fl2va_768p_6s_dense_50steps`).
- The LTX 2.5 / t48 work and the DiffVAE work are **not H3 at all**. They target Lightricks LTX 2.5, a different model. They are neither slow H3 nor FastH3.
- **FastH3** means the distilled H3 family on the Turbo pipeline: the lightx2v 4-step Turbo LoRAs and, per Colman on 2026-10-08, "hyperflow in particular" (the HyperFlow 8-step adapter).
- The Slack thread names exactly one branch, `pshah/minimax-h3-hyperflow-turbo`. That branch already contains `pshah/minimax-h3-lightx2v-turbo`.
- The project has **no recorded FastH3 baseline**. The organization's numbers are below. On one 4x8 BH, a 10 s 768p t2va clip takes 10.77 s with 4-step Turbo (dense). That is already near the brief's target, and almost all of it is denoise (9.0 s, 2.2 s/step).

## 1. Slack thread and branch inventory

Thread: #dit-project, https://tenstorrent.slack.com/archives/C094N67K9R7/p1791465942342729 (2026-10-08, read through Glean):
- Rouzbeh Shirvani 13:25: "Parshwa Shah What's the fast H3 branch you are currently working on?"
- Colman Glagovich 13:26: "^ hyperflow in particular"
- Parshwa Shah 13:27: `github.com/tenstorrent/tt-metal/tree/pshah/minimax-h3-hyperflow-turbo`

| Branch (tenstorrent/tt-metal) | Head | Author, date | What it improves | Model / pipeline | Base |
|---|---|---|---|---|---|
| **pshah/minimax-h3-hyperflow-turbo** (in thread; PR #59676) | 3a4ad18f833 | Parshwa Shah, 2026-10-07 | HyperFlow 8-step adapter on the Turbo pipeline. It adds two-time `(t, r)` interval conditioning (`temb = emb_t + gate*(emb_r - emb_t)`, endpoint time embedder `MiniMaxH3TwoTime`), a fixed 9-point sigma grid read from the safetensors header (`hyperflow`, `hyperflow_version`, `hyperflow_gate`, `hyperflow_sigmas`), and float32 host-fused deltas (`h3_host_deltas`, `host_prefixes`). | H3 (MiniMax-H3) t2va/fl2va, `MiniMaxH3TurboPipeline` | main (merge-base b441f520545, 2026-10-07); 52 behind main, 13 ahead |
| **pshah/minimax-h3-lightx2v-turbo** (merged into the branch above; PR #57745) | 0b3e58d726c | Parshwa Shah, 2026-10-07 | lightx2v Minimax-h3-Turbo rank-128 LoRAs (4 or 8 NFE, `num_inference_steps = NFE+1`), fused into the weights on device (no per-step cost). Scale is alpha/rank = 0.0625, taken from the file metadata. Shift: 6/3 at 768p, 12/3 at 544p. Env: `MINIMAX_H3_LORA_PATH`, `MINIMAX_H3_VIDEO_SHIFT`, `MINIMAX_H3_AUDIO_SHIFT`. Tests: `MINIMAX_H3_TURBO_POINT` (768p/544p/hyperflow), `MINIMAX_H3_TURBO_NFE`, `MINIMAX_H3_TURBO_TASK` (default fl2va). Attention is dense. | H3 Turbo t2va + fl2va (one adapter file serves both), e.g. `minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16` | main (b441f520545) |

Hyperflow-turbo vs its main merge-base: 12 files, +1566/-16:
- new: `experimental/lora/h3_adapter_loader.py` (307), `pipelines/minimax_h3/hyperflow_minimax_h3.py` (192), `pipeline_minimax_h3_turbo.py` (226), and 4 tests (`test_h3_adapter_loader_minimax_h3.py`, `test_pipeline_turbo_minimax_h3.py`, `unit/test_minimax_h3_adapter_settings.py`, `unit/test_minimax_h3_hyperflow.py`)
- modified: `pipeline_minimax_h3.py` (66), `weights_minimax_h3.py` (57), `transformer_minimax_h3.py` (27), `scheduler.py` (15), `lora/promote.py` (5)

Non-merge commits, oldest first:
- f3f159a497d: run the lightx2v Turbo adapters
- b13341ccf98
- 7a4be3c69f9
- 9239c6e2b5e: scale fused QKV by each alpha
- 5dba607c951: adapter through env; re-merge after reload
- d674892b82b
- b76b5d01482: own pipeline class
- aefe4df83f8
- 5ae9975db15
- 1d2664cb7c2
- 7f728d90e4c: HyperFlow

There are also merges db1b84cd1b4 (origin/main) and 3a4ad18f833.

Related FastH3 branches. These are not in the thread, but they are useful sources:

| Branch | Head | Date | Content | Base |
|---|---|---|---|---|
| pshah/minimax-h3-hyperflow-on-turbo | 8a4f14aa744 | 2026-09-30 | **Precomputed AdaLN for the HyperFlow path** (fixed schedule) | main (older) |
| pshah/h3-hyperflow-vae | f3f847874b9 | 2026-09-28 | yuv device readback for the H3 VAE | main (older) |
| pshah/minimax-h3-lora-vsa | 4fdfc38f138 | 2026-09-11 | FastVideo FastH3 LoRA + VSA port of cglagovich/fast_h3_vsa@2b72da0e1ae + VAE device stitch | main 2a8253ad20a (1571 behind) |
| sadesoye/pshah_minimax-h3-lora-vsa-on-base | 3086e5929d8 | 2026-09-22 | Samuel's FastH3 deployment branch ("super WIP") | main (older) |
| sadesoye/pshah_minimax-h3-lora-vsa-on-base_rouzbeh_opt | d1fb310a229 | 2026-09-28 | Rouzbeh's optimizations on that branch | main (older) |
| cglagovich/fast_h3_vsa | c481d6b92c4 | 2026-09-17 | VSA kernels `vsa_sdpa` / `vsa_ring_sdpa` | bc294789ec3 (1751 behind main) |
| origin/fasth3-opt (ours) | 297aa2676ae | — | cglagovich/fast_h3_vsa + harness (5558743768c, 3712e859da9, 64684367fd1) | same old base as fast_h3_vsa |

tt-inference-server `sadesoye/add_h3_fl2va_ref2va` (45b8e8f4c60) adds the runner `tt-minimax-h3-fasth3`, "T2VA with a distilled LoRA, 4 steps". That is the serving definition of FastH3.

## 2. FastH3 vs slow H3

| | Slow H3 (H3 Base) | FastH3 (distilled) |
|---|---|---|
| Checkpoint | MiniMax-H3 base (blx03: `/mnt/MLPerf/tt-shield/persistent-volume/volume_id_tt_transformers-MiniMax-H3-v0.22.0/weights/MiniMax-H3`) | the same base, plus a fused distillation adapter: lightx2v Turbo LoRA (4 or 8 NFE, v1.2 768p / 544p), the HyperFlow 8-step adapter, or the FastVideo FastH3 LoRA |
| Steps | 50 (49 forwards), CFG | 4 or 8 NFE (`num_inference_steps` = NFE+1); HyperFlow uses a fixed 9-sigma grid |
| Shift | default schedule | 768p 6/3, 544p 12/3, HyperFlow 12/3 at 768x1344 |
| Attention | dense (fasth3-opt also has VSA 0.9) | dense on the Turbo branches; VSA in FastVideo FastH3; lightx2v also ships Turbo-SLA (trained 85% sparse) |
| Pipeline | `MiniMaxH3Pipeline` (`pipeline_minimax_h3.py`) | `MiniMaxH3TurboPipeline(MiniMaxH3Pipeline)` (`pipeline_minimax_h3_turbo.py`), with `hyperflow_minimax_h3.py` and `experimental/lora/h3_adapter_loader.py` |
| Entry/test | `test_pipeline_minimax_h3.py`; our harness `test_fasth3_baseline_minimax_h3.py` (BASE_STEPS=50) | `test_pipeline_turbo_minimax_h3.py` (TURBO_POINT / NFE / TASK) |
| VAE | same H3 ViT video VAE + audio VAE in both. Main's version takes 1.40 s at 10 s with yuv readback; fasth3-opt's older copy takes 8.1 s | same |

Sources:
- the hyperflow-turbo branch code and tests
- PR #57745 and PR #59676 descriptions
- Hao AI Lab "FastH3 Preview v1" (haoailab.com/blogs/fasth3-preview: T2VA, 4-step distilled LoRA + VSA, "15 s 768p in 13 s" on Blackwell)
- Ben Hall's "H3 Video Generation Speed Tracker" (gdrive 19ktLSH43iztiDwuM5PKQ4T8nojqy4nwqs_lFhNb_zgM), where FastH3 is listed as "distillation plus different attention backend"
- the Slack thread above

The name "fasth3-opt" misled us: the branch was cut from cglagovich/fast_h3_vsa (the VSA kernels), but our harness on it ran the **base** 50-step model.

### What the LTX 2.5 / t48 work transfers to FastH3

Transfers, as ideas to port (H3 is a 33B single-stream DiT: 50 layers, hidden 5376, 56 heads, packed text+audio+video tokens, ring joint SDPA):
- trace capture/replay on 4x8 (H3 traces only on 4x32 today)
- precomputed / fused AdaLN for a fixed schedule (HyperFlow already has this on hyperflow-on-turbo)
- fused gate and fused norm+AdaLN (t48 `f6b806516cc`)
- dropping per-step replay syncs and latent stats (`4194cd98852`)
- yuv/uint8 device readback and overlapping export with the next stage (`e6911ab558c` LTX_AUDIO_OVERLAP pattern)
- SDPA chunk retuning

Does not transfer:
- the LTX-2.3 conv VAE swap, conv3d halo/blocking retunes, `LTX_VAE_EXACT_SHARD`
- all DiffVAE / `neighborhood_sdpa` levers. H3's VAE is a different decoder, and main already reaches 1.40 s at 10 s with yuv.

Rejected LTX knobs (exp approx, LoFi SDPA matmul, agmm, export_async) need no retry on H3 unless profiling points at the same op.

## 3. Baseline

**None recorded by the project for FastH3.** Our H3 numbers are all slow H3:
- `baselines/fl2va_768p_6s_dense_50steps`: 158 frames, padded 52224. Denoise 75.10 s, VAE 4.75 s, audio 8.19 s, upscale 4.74 s.
- t209 (`ttp/t209-h3-e2e` @30a52415629, blx01): 10 s, 243 frames, 768x1344, VSA 0.9, 50 steps. Denoise 95.2 s, VAE 8.11 s, audio 12.31 s, host bicubic upscale to 1080p 6.92 s.

Organization numbers for distilled H3:

PR #57745: 4-step Turbo, t2va, 1344x768, one 4x8 BH, warm, yuv readback, dense:

| Clip | Padded len | Text enc | Denoise | VAE | Audio | Total |
|---|---|---|---|---|---|---|
| 5 s | 37888 | 0.25 s | 3.49 s (838 ms/step) | 0.75 s | 0.06 s | 4.57 s |
| 10 s | 73472 | 0.27 s | 9.00 s (2207 ms/step) | 1.40 s | 0.08 s | 10.77 s |
| 15 s | — | — | 16.81 s | 2.04 s | — | 19.36 s |

Speed Tracker (s, for 5 / 10 / 15 s clips):

| Variant | 1x BH Galaxy | 4x BH Galaxy |
|---|---|---|
| LightX2V 4-step FL2VA, first keyframe | 8.0 / 15.9 / 25.6 | 6.8 / 9.6 / 13.9 |
| LightX2V 4-step FL2VA, first + last keyframe | 9.1 / 16.0 / 26.6 | — |
| HyperFlow 8-step T2VA | 9 / 21 / 39 | — |
| FastH3 T2VA, 15 s clip | 19 | 11 |

Reading:
- At 10 s, 4-step denoise is 2.2 s/step. That is 2.6x the 5 s per-step time (attention is quadratic in the 73k tokens).
- The brief's "10 s in 7 s" needs denoise of about 5 s, i.e. about 1.2 s/step at 4 NFE. HyperFlow (8 NFE) would need about 0.6 s/step.
- A 6 s clip at roughly 45k tokens should already sit near 5 s with 4-step Turbo. This must be measured.
- 1080p output adds the upscale (6.9 s on host today). It needs a device upscale or a native 1080p decode.

What must be measured (task B below), on 4x8 with fl2va, 768x1344, 6 s and 10 s:
- variants: Turbo 4-step v1.2, Turbo 8-step, HyperFlow 8-step
- stage timings: encoder, keyframe, denoise per step, VAE, audio, upscale
- 5 seeds
- PCC/PSNR against the same variant's reference run, plus visuals. The quality reference for distillation is the 50-step base clip from the same prompt and seed (a visual/VBench comparison, not PCC).

## 4. Proposed follow-up tasks

Priorities: P0 first, then P1. One device job at a time per box, through its broker, timeout ≤600 s.

**A (P0). Integrate the Slack branch onto an ltx-rt project branch.**
- Create `ttp/fasth3-hyperflow` from `origin/ltx-rt` (b9f8587ce6c). Cherry-pick only. No merges into anyone else's branch, no PRs.
- Order:
  1. the main H3 commits from ltx-rt's merge-base c2a4d40104d (09-29) through b441f520545 that touch `models/tt_dit/pipelines/minimax_h3/`, `models/tt_dit/models/transformers/minimax_h3/`, `models/tt_dit/utils/` (policy, references) and the H3 VAE. Main moved +879/-318 over 9 H3 files in that window, including policy.py (250) and references.py (37).
  2. the 11 non-merge lightx2v-turbo/hyperflow commits, f3f159a497d → 7f728d90e4c.
  3. optionally, the precomputed-AdaLN commit from pshah/minimax-h3-hyperflow-on-turbo (8a4f14aa744).
  4. optionally, the yuv-readback commit from pshah/h3-hyperflow-vae (f3f847874b9), if it is not already in main.
- Conflict risk:
  - New files apply cleanly.
  - `pipeline_minimax_h3.py` has 769 lines of drift between ltx-rt and the hyperflow base. This is the main risk, and why step 1 must come first.
  - `transformer_minimax_h3.py` has 46 lines. `weights`, `scheduler` and `promote` have none.
  - ltx-rt's own LTX commits do not touch H3 files, so they should not interfere.
  - If step 1 pulls in shared ttnn/C++ changes (ring SDPA, CCL), the cherry-pick set grows. In that case fall back to a build-only check of those ops.
- Verification:
  - Host unit tests `unit/test_minimax_h3_hyperflow.py` and `unit/test_minimax_h3_adapter_settings.py` (no device).
  - A build on blx03/blx01 only if C++ changed.
  - The integrated branch's device run is task B.
- Do not bring the VSA code over from fasth3-opt here: it is 1751 commits behind main, with a pipeline drift of 2074 lines. See D.

**B (P0, after A). FastH3 e2e baseline on 4x8.**
- On blx03, blx01 or g15blx02 through the broker, one variant per job, ≤600 s, caches under /var/tmp/fasth3 (and checking /home free space first).
- Weights: the base H3 path above.
- Adapters:
  - lightx2v/Minimax-h3-Turbo from Hugging Face (`minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16` and the 8-step file).
  - The HyperFlow adapter's source is not public in anything indexed. Find the path Parshwa's test uses (`MINIMAX_H3_TURBO_POINT=hyperflow`) on the shared volume. If it is absent, run Turbo only and record the gap.
- Measure warm stage timings for fl2va 6 s and 10 s at 768x1344, 5 seeds, then 1080p via the current upscale.
- Save clips plus still frames, and VBench subset on the 5 seeds.
- Run before A's optional AdaLN/yuv picks and after them, as separate jobs.

**C (P1). Port the t48 levers that transfer.**
- Trace on 4x8 for the Turbo denoise loop, with precomputed AdaLN for the fixed 4/8-step schedule.
- Drop per-step syncs.
- Overlap VAE/audio/export.
- Profile one denoise step first (tt-debug-tools profiler) to rank the levers. Note that 4 steps leave little room for feature caching.
- Gate: PCC/PSNR vs the B reference of the same variant (md5 where possible). Use the 5% bar on e2e.

**D (P1). Sparsity on TT hardware.**
- Port `vsa_sdpa` / `vsa_ring_sdpa` (cglagovich/fast_h3_vsa c481d6b92c4) onto the integrated branch as a separate kernel port.
- Evaluate three options:
  - lightx2v Turbo-SLA: `minimax_h3_fl2v_turbo_4step_v0.1_768p_sla_bf16.safetensors`, trained with 85% sparse top-k over a 64-block map plus a linear branch. This is the best quality bet, because the sparsity is trained into the 4-step adapter.
  - FastVideo FastH3 LoRA + VSA.
  - Turbo LoRA + training-free VSA.
- Reuse the block mask across steps.
- This is the lever that makes 10 s reach under 7 s, because denoise at 10 s is attention-dominated.

**E (P1). 1080p output.**
- Replace the 6.9 s host bicubic upscale with a device upscale, or a 1080p-native decode path.
- Measure against the current host upscale for PSNR.

**F (P2). TP×SP / SDPA chunk re-sweep for 73k tokens on 4x8**, after C.

**G (P2, after the speed target). lightx2v 4-step Ref2VA adapter.**
- It uses the Ref2VA base (`transformer_ref/`), so it must not be mixed with the FL2VA LoRAs.
- The tt-inference-server branch `sadesoye/add_h3_fl2va_ref2va` shows the serving shape.

**H (housekeeping).**
- Close the DiffVAE hold: the DiffVAE work is LTX 2.5, not H3 (slow or fast). Tasks #272/#275 stay on hold or get cancelled, as the user decides. This plan does not depend on them.
- Mark t159/t161/t209 notes as "slow H3 (base 50-step)".
