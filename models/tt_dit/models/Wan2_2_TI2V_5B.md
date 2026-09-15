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
- ~~I2V path~~ **done** — see the I2V section below
- Residual VAE encoder (`AvgDown3D` / `WanResidualDownBlock`) — I2V encodes the conditioning
  frame on the host instead; worth ~1.4 s if ported
- Smoke run: `WAN5B_SMOKE=1 pytest ... -k bh_4x8` once ttnn is built on this tree

## Auto track

`tmux attach -t wan5b-auto` on this host — planner clone is `/home/ttuser/tt-metal-hw-planner`.
Weights cache: `/mnt/tt-data/teja/hf` and `/mnt/tt-data/teja/wan_cache`.


---

# I2V (image-to-video)

`WanTI2V5BI2VPipeline` in `models/tt_dit/pipelines/wan/pipeline_wan_ti2v_5b_i2v.py`.

TI2V-5B does **not** condition like Wan2.2-14B I2V. The checkpoint has
`in_channels == out_channels == 48` and `image_dim: null`, and ships no image encoder — so
there is no channel-concat and no CLIP embedding. Conditioning is **latent pinning + a
per-token timestep mask**: latent frame 0 is held at the conditioning latent while its tokens
see timestep 0 and every other token sees `t`. Reference is diffusers 0.38
`WanImageToVideoPipeline` with `expand_timesteps=True`.

## Generation time — single BH Galaxy (4x8), 1280x704, 81 frames, 40 steps, warm-traced

| section | T2V | I2V | delta |
|---|---|---|---|
| Text encoding | 0.089 s | 0.090 s | — |
| Image encode (host) | — | 1.379 s | +1.38 |
| Denoising | 12.045 s | 12.728 s | +0.68 |
| VAE decoding | 4.632 s | 4.649 s | — |
| **Total** | **16.783 s** | **18.863 s** | **+2.08** |

Measured by `test_pipeline_performance_ti2v_5b_i2v` (`tests/models/wan2_2/test_performance_wan.py`),
which mirrors the T2V gate exactly — same geometry, step count, methodology and section names.

**VAE decode is identical between T2V and I2V**, so the conv3d blockings registered in
`pipeline_wan_ti2v_5b.py` carry over with no I2V-specific work.

I2V optimization history: 21.64 s -> 18.86 s (-12.8%), via a two-row timestep MLP (-0.89 s),
bf16 autocast on the host encode (-0.68 s), and `torch.compile` on the host encode (-1.21 s).

## T2V is unaffected

3 runs of the pre-I2V commit vs 3 runs after, 720p: total 16.9072 s -> 16.8848 s, **-0.13 %**
(0.13 sigma, ranges fully overlapping). Denoise -0.20 %. The measurement noise floor on this
box is ~1.8 % on total, so single-sample comparisons below that are not resolvable.

## Validation

| check | result |
|---|---|
| Conditioning math vs diffusers | 17 exact-equality tests |
| Transformer vs torch, scalar timestep | PCC 99.9893 % |
| Transformer vs torch, per-token timestep | PCC 99.9894 % |
| Two-row timestep == scalar timestep | PCC 100.0000 % (bit-exact) |
| E2E decoded frame 0 vs seed image | PCC 0.9984 |
| T2V CLIP correctness gate | 40.45 (threshold 36.00) |

## The L1 overflow in the per-token path

It is the **timestep MLP**, not the AdaLN. Sharding the per-token timestep on the SP axis
brings M from 18688 to 2336, which is necessary but not sufficient — the shape has no
registered blocking and the `(8,8,8)` fallback still overflows:
`circular buffers ... grow to 1979392 B ... beyond max L1 size of 1572864 B` at
`layers/embeddings.py:105` (`TimestepEmbedding.linear_1`).

The fix is to stop running the embedder per-token at all. It is pointwise in the token axis
and the TI2V-5B schedule holds only two distinct values (0 on the conditioned frame, `t`
elsewhere), so embedding two rows and expanding through a mask is mathematically identical,
runs at M=32 like the scalar path, needs no blocking entries at any resolution, and removes
~1168x of redundant matmul.

## Known issues found while doing this (pre-existing, not I2V)

1. **`_register_5b_matmul_tables` never hits.** Wrong M values (registered
   `(2368, 2720, 3488, 9472, 13952)`; the model asks for 1024 at 480p and 2336 at 720p),
   `N=64` copied from 14B where 5B `proj_out` emits `48*2*2 = 192`, and only `"11x10"`
   registered when four of five shapes are looked up on `"12x9"`. Every transformer matmul
   runs on the `8x8x8` default. Measured for ff1 `(2336, 3072, 3584)`: default 401.2 us vs
   351.3 us swept (12.4 %), but only ~2-3 % of denoise end-to-end.
2. **`flow_shift` is the 14B value.** `pipeline_wan.py:433` passes `flow_shift=12.0`,
   overriding the checkpoint's own `scheduler_config.json` value of **5.0**.
3. **The CLIP gate cannot detect quality regressions.** A visibly soft 480p render scored
   37.24 while a crisp 720p render scored 36.63 — CLIP measures semantic alignment, not
   fidelity, and the 36.0 threshold is calibrated to one prompt at one resolution.
4. **480p is out of distribution.** 16x16 VAE compression plus patchify leaves 15x26 tokens
   at 832x480 versus 22x40 at 1280x704. Run correctness and quality work at 720p.
