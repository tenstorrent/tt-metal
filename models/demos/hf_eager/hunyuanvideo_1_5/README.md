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
- **Full 121-frame video generation: 5:59 end-to-end** (32-chip sp=4, tile-sharded VAE)
  — 3.2× vs the ~19 min 8-chip baseline. DiT + tiled VAE on device (all 32 chips);
  text-encode (Qwen) on host — see "What runs where" below
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

## Directory structure

```
hunyuanvideo_1_5/
├── tt/
│   └── pipeline.py         # the ONE shared chained forward (demo + tests call it)
├── _stubs/                 # the 18 graduated TTNN stubs (native on-device modules)
├── demo/
│   ├── demo_i2v.py         # i2v denoise-step demo
│   └── demo_t2v.py         # t2v denoise-step demo
├── tests/
│   ├── pcc/                # per-component PCC tests (18 modules + mesh variant)
│   └── e2e/                # e2e gate, real-weight PCC, generation, perf tests
├── e2e_plan.json           # planner mental model
└── README.md
```

## How to run

```bash
# --- setup (once per shell) ---
source python_env/bin/activate
export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd)

PY=./python_env/bin/python
DIR=models/demos/hf_eager/hunyuanvideo_1_5

# --- correctness ---
$PY -m pytest $DIR/tests/e2e/test_e2e_pipeline.py -s      # e2e gate (Gate 1/2/3, PCC >= 0.95)
$PY -m pytest $DIR/tests/e2e/test_real_weight_pcc.py -s   # real-weight DiT PCC (single + mesh)
$PY -m pytest $DIR/tests/pcc/ -s                          # all 18 per-module PCC tests
$PY -m pytest $DIR/tests/e2e/test_vae_decoder.py -s       # tiled VAE decode

# --- 121-frame video generation (fastest: 32-chip sp=4 + tile-sharded VAE, ~5:59 e2e) ---
HY_MESH=4,8 HY_DIT_SP=1 HY_DIT_BF16=1 HY_TT_QWEN=1 HY_TT_VAE=1 HY_VAE_TILE=1 HY_VAE_TILE_PX=128 \
HY_FRAMES=121 HY_STEPS=50 HY_H=480 HY_W=848 \
$PY -m pytest $DIR/tests/e2e/test_stage2b_gen.py::test_stage2b_gen_qb2 -svv

# --- per-step DiT perf (device_ms, trace+2CQ, capped 2-layer workload) ---
$PY -m pytest $DIR/tests/e2e/test_main_perf.py::test_main_perf -svv

# --- demos (single denoise step) ---
$PY -m models.demos.hf_eager.hunyuanvideo_1_5.demo.demo_i2v
$PY -m models.demos.hf_eager.hunyuanvideo_1_5.demo.demo_t2v
```

### Generation flags (`test_stage2b_gen_qb2`)

Every flag is an env var; all are optional with the defaults below.

| flag | default | meaning |
|---|---|---|
| `HY_MESH` | (parametrized) | DiT mesh as `rows,cols` (e.g. `4,8`). With `HY_DIT_SP=1`: **rows → sp, cols → tp** |
| `HY_DIT_SP` | `0` | **Sequence parallelism** — shard the latent sequence across mesh rows (`sp=rows`, head-`tp=cols`). **Required for any multi-row mesh**; without it the mesh flattens to `tp=N_devices` and the 16-head DiT errors (`heads_total=16 not divisible by tp=32`) |
| `HY_DIT_BF16` | `0` | Load DiT weights + run block matmuls in **bf16** (faster; used for all perf numbers) |
| `HY_TT_QWEN` | `0` | Run the **Qwen text encoder on device** if a submesh can be carved; **at sp=4 there's no room (DiT fills all rows) so it falls back to host** |
| `HY_TT_VAE` | `0` | Run the **VAE decode on device** (else on host); auto **tile-shards** across the mesh when `ndev>1` |
| `HY_VAE_TILE` | `0` | Enable **tiled VAE decode** (split the latent into H/W tiles) — required at high frame counts |
| `HY_VAE_TILE_PX` | `0` | Per-tile pixel size; `0` = model default. **Use `128` at 121f** (192 px fragments DRAM) |
| `HY_FRAMES` | `13` | Number of video frames (e.g. `121`) |
| `HY_STEPS` | `50` | Number of denoise steps |
| `HY_H` / `HY_W` | `480` / `848` | Output height / width in pixels |
| `HY_TRACE` | `1` | `1` = trace-capture + 2 command queues; `0` = eager. (Per-step ~identical at 121f; capture is a one-time cost — see Performance) |
| `HY_PROMPT` / `HY_NEG_PROMPT` | cat prompt / — | Positive / negative text prompt |
| `HY_OUT` | `/tmp/hy15_stage2b_qb2` | Output dir (frames + `tt_blackhole.mp4` + `.gif`) |
| `HY_FPS` | `24` | Output video frame rate |
| `HY_QWEN_ZERO_PAD` | `1` | Zero out leaked Qwen padding tokens (correctness; `0` disables) |

`HY_TRUNC` (text truncation) is read only by the single-device `test_stage2b_gen`, **not** by
`qb2`. `HY_TRACE_REGION_SIZE` and the CQ count are set automatically by the test.

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

Real weights, 480×848, 121 frames, 50 steps. The VAE decode is **tile-sharded across the
mesh** (`_decode_batch_sharded`, `085be79`): the ~45 tiles are batched and the batch
dimension sharded so each device decodes one tile per round — **2 batched passes instead
of 45 sequential** — cutting VAE wall-clock ~mesh-fold while per-device DRAM peak stays at
a single tile.

| config | denoise | VAE | **e2e** |
|---|---|---|---|
| baseline (8-chip) | ~15:00 | — | 18:52 |
| 16-chip, sp=2 | 6:21 | ~2:30 | 10:38 |
| 32-chip, sp=4, replicated VAE | 3:05 | ~5:00 | 9:47 |
| **32-chip, sp=4, tile-sharded VAE** | 3:02 | ~2:00 | **5:59** ✅ |

**sp=4 with tile-sharded VAE is the fastest full-video config: 5:59 e2e (3.2× vs the
~19 min 8-chip baseline)**, beating 16-chip sp=2 (10:38). Both DiT and VAE weights stay
resident throughout; output is crisp and coherent (no tiling seams). Reproduced on
hardware with the exact command above (warm cache): 359 s total, denoise 2:49.

At 121 frames the VAE tiles must be **128 px, not 192 px** — the high frame count (T=31)
makes each tile's decode ~8× larger and 192 px fragments DRAM; 128 px fits with margin and
still gives the two-round win. `ndev=1` falls back to the sequential path unchanged; the
tile-shard flags default off.

#### What runs where (5:59 sp=4 config)

| stage | placement | detail |
|---|---|---|
| Qwen 2.5-VL text-encode | **host (CPU)** | one-time; at sp=4 the DiT fills all 4 mesh rows, so no Qwen submesh can be carved (`HY_TT_QWEN=1` falls back to CPU) |
| byT5 text-encode | **host (CPU)** | one-time; no TT adapter (its DiT-side projection `s_byt5` *is* on device) |
| Latent init + scheduler step | **host (CPU)** | stock diffusers: noise init once + flow-match latent update per step (cheap elementwise) |
| **DiT denoise** (50 steps) | **on device** | all 32 chips, sp=4 × tp=8 — the heavy per-step compute |
| **VAE decode** | **on device** | tile-sharded across all 32 chips, reusing the full mesh after denoise |
| Frame post-proc → mp4/gif | **host (CPU)** | one-time save |

So only the two heavy compute stages — **DiT and VAE** — run on device; both text encoders
(Qwen + byT5) and the scheduler/latent/post-proc torch ops run on host. All of the host
work is one-time or a cheap per-step elementwise update, so it barely dents the
denoise-dominated 5:59. (No image encoder / VAE-encode runs in t2v at all.) At **sp=2**
(16-chip) a spare mesh row *does* leave room to carve a Qwen submesh, so Qwen runs on
device there — but the smaller DiT (denoise 6:21) makes that config slower overall
(10:38 e2e).

**On-device text-encode at sp=4 (opt-in, `HY_TT_QWEN_SHARED=1`).** Qwen *can* run on the
DiT's own 32-chip mesh (TP=4 + FSDP across the other axis, weights co-resident, no
overlapping context) instead of CPU. It's correct (verified, identical output) but
**slower for a one-shot video — 6:58 vs 5:59** — because the 7B weight-load +
first-compile + per-layer FSDP all-gathers cost more than the one-time CPU encode they
replace (denoise itself is unchanged at ~2:50). So host is the default at sp=4; the
shared-mesh path only pays off in a **served / multi-prompt** deployment where the load
and compile amortize across many generations.

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
- **Tiled VAE at 121f:** VAE tiles must be **128 px** (`HY_VAE_TILE_PX=128`); the high
  frame count makes each tile's decode ~8× larger and 192 px fragments DRAM. Decode is
  tile-sharded across the mesh, so it reuses the full 32-chip mesh once the DiT denoise
  is done (sequential phases), which is what makes sp=4 the fastest e2e config.
- **Reference vs real weights:** the per-component / e2e-PCC tests use deterministic-random
  (seed 0) weights with the repeated stacks shrunk to `num_layers=2` for CPU feasibility,
  so PCC validates the op implementation. `test_real_weight_pcc*` and the generation
  pipeline use the real checkpoint (community diffusers conversion).
- The `hunyuan_video15_transformer_block` stub applies on-device interleaved RoPE to the
  latent q/k and accepts an optional additive attention bias; `freqs_cis=None`/
  `attn_bias=None` preserve the original per-component behavior.
