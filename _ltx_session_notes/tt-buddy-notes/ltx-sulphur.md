# ltx-sulphur

## Sulphur LoRA characterized + LoRA-fused video runs e2e (modular path works)
**2026-06-15 12:31 UTC / 05:31 PT** · `tt-metal@f2a656a3331-dirty`

`sulphur_lora_rank_768.safetensors` (2304 tensors = 1152 `diffusion_model.*.lora_A/B.weight` pairs) is a **video-style** LoRA, cleanly compatible with the repo's `fuse_loras_into`.

**CPU reconstruction analysis** (best-fit scalar s minimizing ‖(sulphur_distil−lightricks_distil) − s·BA‖ per weight): on video attn/FF weights s≈0.6–0.9 with residual 0.06–0.28 (LoRA explains ~85–94% of the Sulphur delta); audio-branch weights (audio_attn*, *_to_*_attn) s≈0, residual≈1.0 — the LoRA does NOT touch them (those small Sulphur audio deltas come from distillation). So the LoRA reconstructs the Sulphur **video** style but not the merged checkpoint exactly.

**Third e2e run (modular LoRA path):** offline-fused `lightricks_distil + sulphur_lora_768` (strength 1.0, `fuse_loras_into`, 1152 weights fused) → `/home/smarton/.cache/ltx-checkpoints/sulphur_lora_fused_distil.safetensors`; ran distilled girl-singing → `sulphur_lora_fused_1920x1088.mp4` (job 121216-13, PASS 814s, valid H264+AAC). **Gotcha:** `save_file` must pass `metadata=` (the audio decoder reads `json.loads(f.metadata()["config"])` at pipeline_ltx.py:1505) — dropping it → `TypeError: NoneType not subscriptable` at audio-decoder construction.

**Final status — all 3 distilled e2e runs PASS (bh 4x8 linear, untraced, 1088x1920/145f):**
| video | checkpoint |
|---|---|
| baseline_lightricks_1920x1088.mp4 | Lightricks distilled |
| sulphur_distil_1920x1088.mp4 | Sulphur merged distil (the goal) |
| sulphur_lora_fused_1920x1088.mp4 | Lightricks distil + Sulphur LoRA (fused) |

Branch `ltx-sulphur` (3 commits on `smarton/audio-submesh-e2e`): design spec, `test_sulphur_checkpoint_compat.py` guard, outcome. MP4s are local artifacts (not committed).

## Sulphur-2-base generates video on tt-metal LTX-2.3 distilled pipeline (BH Galaxy 4x8)
**2026-06-15 11:50 UTC / 04:50 PT** · `tt-metal@51b977dbbbc-dirty`

**Result:** `SulphurAI/Sulphur-2-base` (distilled checkpoint) generates a valid AV MP4 through the existing tt-metal LTX-2.3 distilled pipeline. Branch `ltx-sulphur` (worktree), `bh_4x8sp1tp0_linear`, untraced.

**Why it's a pure weight-swap:** Sulphur is a fine-tune of `Lightricks/LTX-2.3`. `sulphur_distil_bf16.safetensors` (46 GB) is the full LTX-2.3 distilled bundle (DiT + `vae` + `audio_vae` + `vocoder` + `text_embedding_projection`) with Sulphur baked in — **5947 tensors, byte-identical key set + shapes to `ltx-2.3-22b-distilled-1.1.safetensors`, zero missing/extra/reshaped** (guard: `test_sulphur_checkpoint_compat.py`). So `LTX_CHECKPOINT=<sulphur abs path>` is the only change; no source edits to the pipeline. Gemma encoder + spatial upscaler still load from `Lightricks/LTX-2.3` (kevinmi cache).

**Runs (NO_PROMPT girl-singing DEFAULT_LTX_PROMPT, 1088x1920, 145f, seed 10, untraced linear):**
| run | checkpoint | job | result | output |
|---|---|---|---|---|
| baseline | Lightricks distilled (kevinmi cache) | 111728-10 | PASS 985s | baseline_lightricks_1920x1088.mp4 |
| sulphur | sulphur_distil_bf16 (smarton cache) | 111737-11 | PASS 819s | sulphur_distil_1920x1088.mp4 |

Both MP4s: H264 1920x1088 145f@24fps 6.04s + AAC 48kHz stereo. Stage 1 denoise 68.1s, Stage 2 36.5s (identical — same arch).

**Key gotchas (load-bearing):**
1. **Build must match source for JIT kernels.** Worktree symlinks `build`/`python_env` to the main checkout. First attempt branched off `ltx-perf` (28 commits behind main) → `reader_conv1d_depthwise.cpp` (audio) failed to JIT-compile against main's newer `TensorAccessor` headers ("Index out of range"). **Fix: rebased `ltx-sulphur` onto `smarton/audio-submesh-e2e`** (= main checkout HEAD, build built from it) so worktree kernel source == build headers. Audio path then compiled + ran clean.
2. **Distilled test skips without a LOCAL checkpoint.** `@skipif(not os.path.exists(default_ltx_checkpoint(...)))` — `LTX_CHECKPOINT` must be an absolute path that exists; the `Lightricks/LTX-2.3:...` HF-ref string fails `os.path.exists` → SKIP. (This guard is on audio-submesh-e2e, not on ltx-perf.)
3. **Checkpoint logistics:** sulphur in `~smarton/.cache/huggingface` (public, no token); Lightricks distilled + gemma + upscaler in world-readable `/home/kevinmi/.cache/huggingface` → set `HF_HOME` there, `LTX_CHECKPOINT` to the explicit sulphur abs path (resolver prefers `os.path.exists` over HF_HOME).

**Sulphur LoRA note:** `sulphur_lora_rank_768.safetensors` is the style delta for the DEV base; the distilled bundle already has it merged, so applying it to the distilled base is redundant/recipe-incorrect. The meaningful LoRA path is dev base + LoRA via the non-distilled pipeline (= the "full" variant).
