# Wan2.2 TI2V-5B — manual bring-up (single BH Galaxy)

Parallel track to `tt_hw_planner auto-up` on `/home/ttuser/tt-metal-hw-planner`.

## Target

- Hardware: DC16-2-BG-0105-u20-48, 32-chip BH Galaxy, mesh 4x8
- Checkpoint: `Wan-AI/Wan2.2-TI2V-5B-Diffusers`
- Same Galaxy preset as 14B: SP=8 axis1, TP=4 axis0, Ring, FSDP off, 11x10 matmul grid

## What already differs from 14B A14B

| | 14B | 5B |
|---|---|---|
| Experts | 2 (MoE, `transformer` + `transformer_2`) | 1 dense (`boundary_ratio=None`) |
| dim / ffn / layers / heads | 5120 / 13824 / 40 / 40 | 3072 / 14336 / 30 / 24 |
| VAE | Wan2.1, 16-ch | Wan2.2-VAE 4x16x16, **48-ch** |
| frames / steps | 81 / 40 | 121 / 50 |
| expand_timesteps | False | True (TI2V) |

## Code landed in this clone

1. `model_type=ti2v` allows `in_channels=48`
2. `WanCheckpoint.build` now passes `num_layers` from HF config (30 vs hardcoded 40)
3. Pipeline skips `transformer_2` when `boundary_ratio is None`
4. Variant: `models/tt_dit/pipelines/wan/pipeline_wan_ti2v_5b.py`
5. Test: `models/tt_dit/tests/models/wan2_2/test_pipeline_wan_ti2v_5b.py`

## VAE (landed)

Wan2.2-VAE residual decoder path in `vae_wan2_1.py`:

- `WanResidualUpBlock` + `WanDupUp3D` shortcut (`is_residual=True`)
- `decoder_base_dim` from HF (256, not encoder `base_dim=160`)
- `patch_size=2` host `unpatchify` after decode; decoder planned at H/2 x W/2
- `first_chunk` threaded (full-T and cached frame 0)

Encoder residual downblocks (`AvgDown3D`) still TODO — T2V decode does not need them.

## Still TODO (manual)

- Conv3d blocking sweep for 5B VAE channel counts (160/256/512/1024/48)
- Sweep 5B 11x10 matmul / fused-MMRS (first-cut tables registered in pipeline_wan_ti2v_5b.py)
- I2V path + residual encoder
- Smoke run: `WAN5B_SMOKE=1 pytest ... -k bh_4x8` once ttnn is built on this tree

## Auto track

`tmux attach -t wan5b-auto` on this host — planner clone is `/home/ttuser/tt-metal-hw-planner`.
Weights cache: `/mnt/tt-data/teja/hf` and `/mnt/tt-data/teja/wan_cache`.
