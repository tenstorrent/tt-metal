# tencent/HunyuanVideo-1.5 — TTNN end-to-end pipeline

Real, on-device TTNN bring-up of the **HunyuanVideo-1.5** video **diffusion
transformer** (`diffusers.HunyuanVideo15Transformer3DModel`, an MMDiT / dual-stream
DiT), plus the full text→video / image→video generation pipeline (Qwen text-encode +
DiT denoise + tiled VAE decode) running end-to-end on Blackhole Galaxy.

The DiT reproduces `HunyuanVideo15Transformer3DModel.forward(...)` — the golden the
sampler calls repeatedly:

```
(noisy video latent, timestep, mllm/qwen text embeds, byT5 text embeds, image embeds)
        ->  denoised velocity / flow prediction
```

## Branch & status

- **Branch:** `sdawle/hunyuanvideo-bringup_bh_glx`
- **18/18** DiT modules **on device** (native ttnn, PCC-verified)
- **Full 121-frame video generation** validated end-to-end on a **24-chip Blackhole
  Galaxy** mesh (sp=3, TP=8×SP=3), text-encode + DiT + tiled VAE all on-device
- **Per-step DiT device time optimized `6.43 → 5.10 ms` (1.26×)** via 16 committed
  fusion/sharding wins + an L1-fit guard (tt-hw-planner `optimize`)

## What each Call does

| Call (demo)     | Regime | Conditioning                                   | Output                 |
|-----------------|--------|------------------------------------------------|------------------------|
| `demo/demo_i2v` | i2v    | dual text + **active** image embedding (all valid) | velocity `(1,32,F,H,W)` |
| `demo/demo_t2v` | t2v    | dual text; image zeroed/masked (`is_t2v` path) | velocity `(1,32,F,H,W)` |

Both Calls share the **one** pipeline (`tt/pipeline.py::HunyuanVideo15Pipeline.run`);
they differ only in whether `image_embeds` is populated. The t2v image tokens are
masked with a per-key additive attention bias — the final latent output is provably
independent of the invalid image tokens, so it matches the reference exactly.

## Layout

```
hunyuanvideo_1_5/
  tt/pipeline.py            the ONE shared chained forward (both demo + test call it)
  demo/demo_i2v.py          i2v denoise-step demo   (python -m … .demo.demo_i2v)
  demo/demo_t2v.py          t2v denoise-step demo   (python -m … .demo.demo_t2v)
  _stubs/*.py               the 18 graduated TTNN stubs (Source B)
  tests/pcc/                per-component PCC tests (18 modules, + mesh variant)
  tests/e2e/                e2e gate, real-weight PCC, generation, perf tests
  e2e_plan.json             the planner mental model (Command 1)
```

## How to run

```bash
# ---- correctness ----
# e2e gate (Gate 1 native + Gate 2 all-18-invoked + Gate 3 PCC>=0.95)
./python_env/bin/python -m pytest models/demos/hf_eager/hunyuanvideo_1_5/tests/e2e/test_e2e_pipeline.py -s
# real-weight DiT PCC (single-device, and 24-chip mesh @ threshold 0.99)
./python_env/bin/python -m pytest models/demos/hf_eager/hunyuanvideo_1_5/tests/e2e/test_real_weight_pcc.py -s

# ---- full video generation (24-chip sp=3, 480x848, 121 frames, 50 steps) ----
HY_MESH=3,8 HY_DIT_SP=1 HY_DIT_BF16=1 HY_TT_VAE=1 HY_TT_QWEN=1 HY_VAE_TILE=1 HY_VAE_TILE_PX=128 \
HY_H=480 HY_W=848 HY_FRAMES=121 HY_STEPS=50 HY_TRACE=0 \
./python_env/bin/python -m pytest \
  models/demos/hf_eager/hunyuanvideo_1_5/tests/e2e/test_stage2b_gen.py::test_stage2b_gen_qb2 -svv

# ---- perf (trace+2CQ device_ms, capped 2-layer profiling workload) ----
./python_env/bin/python -m pytest models/demos/hf_eager/hunyuanvideo_1_5/tests/e2e/test_main_perf.py::test_main_perf -svv

# ---- demos (single denoise step) ----
python -m models.demos.hf_eager.hunyuanvideo_1_5.demo.demo_i2v
python -m models.demos.hf_eager.hunyuanvideo_1_5.demo.demo_t2v
```

## Performance

### Per-step DiT device time — `device_ms`

Metric measured under **trace + 2 command queues** on the **capped 2-layer / M=32
profiling workload** (`test_main_perf.py`, `TT_PERF_LAYERS=2`). Optimized by the
tt-hw-planner `optimize` agent; **16 committed wins + 1 L1-fit guard**, all PCC-clean.

| phase | key levers | device_ms |
|---|---|---|
| baseline | — | 6.43 |
| bias→matmul fusion | fold every `Linear` bias into the `ttnn.linear` epilogue (all stubs + glue) | 6.02 |
| matmul fusion | AdaLayerNormZero 6-way modulation (C→6C, one matmul + slice); column-parallel QKV | 5.37 |
| ternary fusion | `addcmul` for AdaLN modulation + gated residuals (bake `+1` into scale bias) | 5.31 |
| LayerNorm width-sharding | `norm2` / `_adazero` / `norm_out` LN spread over a row of cores | 5.17 |
| launch dedup | row-linear bias fold, RoPE `mul+addcmul`, AdaLN `(B,1,C)` slice, hoist `silu(temb)` + RoPE cos/sin bcast | **5.10** |

**Net: 6.43 → 5.10 ms = 1.26× per-step device time.** A `d0d07d0` L1-fit guard makes
the width-shard LN fall back to interleaved LN at large M (fixes a 121f OOM — see Notes).

**Custom-kernel rungs (measured, no gain):** a hand-authored **tt-lang fused-FFN kernel**
(up-proj + GELU + down-proj in one launch, L1-resident intermediate — the fusion `ttnn`
can't express) was on-device correct (PCC 0.99999) but **+3%** — the FF matmul is
**dispatch/launch-bound at M=32, not compute/BW-bound**, so a compute kernel can't beat
the tuned `ttnn` matmul. C++ Metalium `generic_op` matmul: same result. Both reverted.

### Full 121-frame end-to-end generation

Real weights, 24-chip sp=3, 480×848, 121 frames, 50 steps, tiled VAE (128px).

| mode | steady-state s/it | denoise | **e2e total** | frames |
|---|---|---|---|---|
| **Eager** (`HY_TRACE=0`) | 4.18 | 3:50 | **8:18** | 121 ✓ |
| **Trace+2CQ** (`HY_TRACE=1`) | 4.15 | 4:19 (+~56 s one-time capture) | **9:36** | 121 ✓ |

**Eager is faster for one-shot 121f** — steady-state per-step is identical (the DiT is
compute-bound, so 2CQ input-overlap buys nothing), and trace's capture cost isn't
amortized over a single 50-step run. The `device_ms` wins above are at the tiny
dispatch-bound profiling scale and **do not materially speed up the compute-bound 121f
e2e** (which is also ~half model-load + VAE).

## PCC validation

### End-to-end PCC (2-layer reference weights, single forward)

| task | granularity | e2e PCC   |
|------|-------------|-----------|
| i2v  | composite / mid / deep | 0.999979 |
| t2v  | composite / mid / deep | 0.999979 |

**Gate 2 union invoked: 18/18. Gate 3 min PCC: 0.999979 (threshold 0.95).**

### Real-weight & mesh PCC

| test | scope | threshold |
|---|---|---|
| `tests/e2e/test_real_weight_pcc.py::test_real_weight_pcc` | real-weight DiT, single-device | 0.99 |
| `tests/e2e/test_real_weight_pcc.py::test_real_weight_pcc_mesh` | real-weight DiT, 24-chip mesh (TP=8×SP=3) | 0.99 |
| `tests/e2e/test_e2e_pipeline.py::test_e2e_gates` | Gate 1/2/3 e2e | 0.95 |
| `tests/e2e/test_vae_decoder.py` | tiled VAE decode | — |

### Per-module PCC (18/18 on device)

Each graduated stub has a per-component PCC test at
`tests/pcc/test_<module>.py::test_<module>` (+ a `_mesh` variant for the transformer
block). All pass, native ttnn:

`ada_layer_norm_continuous` · `ada_layer_norm_zero` ·
`combined_timestep_text_proj_embeddings` · `feed_forward` · `hunyuan_video15_ada_norm` ·
`hunyuan_video15_by_t5_text_projection` · `hunyuan_video15_image_projection` ·
`hunyuan_video15_individual_token_refiner` · `hunyuan_video15_individual_token_refiner_block` ·
`hunyuan_video15_patch_embed` · `hunyuan_video15_rotary_pos_embed` ·
`hunyuan_video15_time_embedding` · `hunyuan_video15_token_refiner` ·
`hunyuan_video15_transformer_block` (+ `_mesh`) · `linear_activation` ·
`pix_art_alpha_text_projection` · `timestep_embedding` · `timesteps`

### Gate 2 — all 18 modules invoked (union across granularities)

The graduated set is over-complete (composite stubs inline their leaves), so the
pipeline runs the same faithful forward at three granularities; the union == the full
set: **composite** (8) → **+mid** (6) → **+deep** (4) = 18. Every stub's output feeds
downstream (no coverage sweep).

## Generated outputs

Sample videos generated by the pipeline (stored at `/home/tt-admin/hunyuan_ab_videos/`):

| file | what it shows |
|---|---|
| `full121_16fusions.mp4` / `.gif` | **latest** — 121f, all 16 fusions + L1 guard, real weights (validation) |
| `full121_elaborate_prompt.mp4` / `.gif` | 121f, cinematic cat prompt |
| `full121_fusion.mp4` / `.gif` | 121f, default prompt |
| `before_prefusion.mp4` / `.gif` | A/B — pre-fusion baseline |
| `after_fusion.mp4` / `.gif` | A/B — post-fusion (visually identical → fusions are PCC-clean) |

All are 480×848, 24 fps, decoded through the on-device tiled VAE.

## Notes

- **L1-fit guard:** the LayerNorm width-sharding wins were validated by the optimizer at
  the 2-layer / M=32 profiling scale. At 121f the LN input is ~16k tokens, so
  width-sharding it into L1 needs ~16 MB/core (bank is 1.4 MB) → `TT_FATAL` OOM. `_wln`
  now falls back to interleaved LN when the per-core shard won't fit L1 (`d0d07d0`); the
  shard still applies where it fits.
- **Trace at 121f:** trace+2CQ does *not* OOM at 121f on the 24-chip sp=3 config (only
  the 4-chip flat-TP QB2 does). It's simply not faster for a one-shot generation.
- **Reference vs real weights:** the per-component / e2e-PCC tests use deterministic-random
  (seed 0) weights with the repeated stacks shrunk to `num_layers=2` for CPU feasibility,
  so PCC validates the op implementation. `test_real_weight_pcc*` and the generation
  pipeline use the real checkpoint (community diffusers conversion).
- The `hunyuan_video15_transformer_block` stub applies on-device interleaved RoPE to the
  latent q/k and accepts an optional additive attention bias; `freqs_cis=None`/
  `attn_bias=None` preserve the original per-component behavior.
