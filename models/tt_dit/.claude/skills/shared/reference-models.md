# Reference models

Read before writing any new layer, block, VAE stage or attention variant.
Nearly every hard problem here — mesh-sharded 3D conv, GroupNorm that doesn't
deadlock, ring attention, halo exchange, traced replay, on-device text
encoding — already has a working answer in the tree.

## Search first

```bash
grep -rn "class .*GroupNorm\|class .*Upsample\|class .*ResnetBlock" models/tt_dit --include=*.py
grep -rn "ttnn.group_norm\|ttnn.experimental.conv3d\|nlp_create_qkv_heads" models/tt_dit --include=*.py
grep -rln "halo\|neighbor_pad\|traced_function" models/tt_dit --include=*.py
```

**Search branches too** — several models' best reference never merged:

```bash
git log --all --oneline -- models/tt_dit/models/vae/
git branch -a --list '*flux2*' '*minimax*'
gh pr list --repo tenstorrent/tt-metal --search "tt_dit vae" --state all --limit 30
gh pr diff <N>          # use gh; do not scrape the web UI
```

Off-`main` at time of writing: `origin/apande/flux2`, `origin/friedrich/flux2`,
`origin/flux2_wh_glxy`, `origin/drohani_flux2`, `origin/cglagovich/minimax-h3`.

## What to crib from where

| Model | Files | Take |
|---|---|---|
| **LTX-2.3** | `models/transformers/ltx/`, `models/vae/vae_ltx.py`, `models/audio_vae/*_ltx.py`, `models/upsampler/latent_upsampler_ltx.py`, `tests/models/ltx/` | Spatial VAE sharding with halo exchange (H across `tp_axis`, W across `sp_axis`); **GroupNorm driven correctly at awkward spatial sizes**; per-region trace gates (`LTX_TRACED`, `LTX_VOC_TRACE`, `LTX_VAE_TRACE`); on-device text encode with disk cache. Docs: `models/LTX2.md` |
| **Wan2.2** | `models/transformers/wan2_2/`, `models/vae/vae_wan2_1.py`, `tests/models/wan2_2/` | Ring attention SP + TP with CCL overlap; `test_performance_wan.py` is the `expected_metrics` regression harness to copy; `bruteforce_conv3d_sweep.py` is the conv3d sweeper. Docs: `models/Wan2_2.md` |
| **Mochi-1** | `models/vae/vae_mochi.py`, `utils/mochi.py` | `MochiVAEParallelConfig` — independent time/H/W split. Caution: its `_valid_norm_grid` picks hanging GroupNorm grids (`known-issues.md`) |
| **Flux 1 / FLUX.2** | `models/transformers/transformer_flux1.py`, `tests/models/flux1/test_performance_flux1.py` | Clean image-DiT structure; worked `traced_function` example. FLUX.2 on branches |
| **SD3.5 / Qwen-Image / Ideogram-4 / Motif** | `models/transformers/transformer_*.py`, matching `pipelines/`, `tests/models/` | Image-DiT variants. Ideogram-4 is Blackhole-first and carries the dequant-cache pattern for quantized checkpoints |
| **MiniMax-H3** *(branch only — not on `main`; see "Search branches" above)* | `models/vae/minimax_h3/`, `models/audio_vae/minimax_h3/`, `tests/models/minimax_h3/` on `origin/cglagovich/minimax-h3` and derivatives | Tiled VAE where resolution changes tile *count* not tile *shape*; data-parallel over work units; per-frame GroupNorm via `GroupNorm3D`. Its perf test separates roundtrip quality from perf baselines in one file and projects full-clip time as per-invocation × work-unit count |

## Shared machinery

| Need | Use |
|---|---|
| Quality gate (PCC / CCC / RMSE-over-σ) | `utils/check.py::assert_quality` |
| Trace capture and replay | `utils/tracing.py::traced_function`, `Tracer` |
| Matmul configs / block sweep | `utils/matmul.py::get_matmul_config`, `register_matmul_configs`; `utils/sweep_mm_block_sizes.py` |
| Conv3d blockings, shape math, pad/unpad | `utils/conv3d.py::get_conv3d_config`, `register_conv3d_configs`, `compute_encoder_dims` |
| Sub-state-dict extraction | `utils/substate.py` |
| Weight caching | `utils/cache.py`, `TT_DIT_CACHE_DIR` |
| CCL semaphores and collectives | `parallel/manager.py::CCLManager`, `parallel/config.py::vae_all_gather` |
| Parallel config types | `parallel/config.py` |
| LoRA | `layers/lora.py`, `utils/fuse_loras.py` |
