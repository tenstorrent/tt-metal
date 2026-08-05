# tencent/HunyuanVideo-1.5 — TTNN end-to-end pipeline

On-device TTNN pipeline for **HunyuanVideo-1.5** (an MMDiT / dual-stream video diffusion
transformer): **Qwen text-encode → DiT denoise → tiled VAE decode**, running full
text→video / image→video generation on a 32-chip Blackhole Galaxy. The DiT reproduces
`diffusers.HunyuanVideo15Transformer3DModel.forward(...)` — the golden the sampler calls
each denoise step. `demo/demo_t2v` and `demo/demo_i2v` share the one pipeline
(`tt/pipeline.py`); they differ only in whether `image_embeds` is populated (t2v masks the
image tokens out via a per-key additive attention bias, so it matches the reference).

## Status
- **Branch:** `sdawle/hunyuanvideo-bringup_bh_glx`
- **18/18 DiT modules on device** (native ttnn, per-component PCC-verified)
- **Full 121-frame generation: 5:09 e2e** (32-chip sp=8×tp=4 + tile-sharded VAE) — 3.7× vs
  the ~19 min 8-chip baseline. DiT + VAE on device (all 32 chips); text-encode on host.
- **Per-step DiT `device_ms`: 6.43 → 5.10 (1.26×)** via 16 fusion/sharding wins + an L1 guard
  (tt-hw-planner `optimize`, on the capped 2-layer profiling workload).

## Directory structure
```
hunyuanvideo_1_5/
├── tt/pipeline.py          # the ONE shared chained forward (demo + tests call it)
├── _stubs/                 # the 18 graduated TTNN stubs (native on-device modules)
├── demo/{demo_i2v,demo_t2v}.py   # single-denoise-step demos
├── tests/pcc/              # per-component PCC tests (18 modules + mesh variant)
├── tests/e2e/              # e2e gate, real-weight PCC, generation, perf tests
└── e2e_plan.json
```

## How to run
```bash
# setup (once per shell)
source python_env/bin/activate
export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd)
PY=./python_env/bin/python
DIR=models/demos/hf_eager/hunyuanvideo_1_5

# fastest 121-frame generation: 32-chip sp=8×tp=4 + tile-sharded VAE (~5:09 e2e)
HY_MESH=8,4 HY_DIT_SP=1 HY_DIT_BF16=1 HY_TT_QWEN=1 HY_TT_VAE=1 HY_VAE_TILE=1 HY_VAE_TILE_PX=128 \
HY_FRAMES=121 HY_STEPS=50 HY_H=480 HY_W=848 \
$PY -m pytest $DIR/tests/e2e/test_stage2b_gen.py::test_stage2b_gen_qb2 -svv

# correctness
$PY -m pytest $DIR/tests/e2e/test_e2e_pipeline.py -s        # e2e gate (Gate 1/2/3, PCC ≥ 0.95)
$PY -m pytest $DIR/tests/e2e/test_real_weight_pcc.py -s     # real-weight DiT PCC (single + mesh)
$PY -m pytest $DIR/tests/pcc/ -s                            # all 18 per-module PCC tests

# per-step DiT perf (device_ms, trace+2CQ, capped 2-layer workload)
$PY -m pytest $DIR/tests/e2e/test_main_perf.py::test_main_perf -svv
```

### Generation flags (`test_stage2b_gen_qb2`)
All optional env vars; defaults below.

| flag | default | meaning |
|---|---|---|
| `HY_MESH` | (parametrized) | DiT mesh `rows,cols`. With `HY_DIT_SP=1`: **rows → sp, cols → tp** |
| `HY_DIT_SP` | `0` | Sequence parallelism (shard the latent seq across rows). **Required for multi-row meshes** — else the mesh flattens to `tp=N` and the 16-head DiT errors |
| `HY_DIT_BF16` | `0` | bf16 DiT weights + block matmuls (used for all perf numbers) |
| `HY_TT_QWEN` | `0` | Qwen text-encode on device if a submesh can be carved; at sp=4/8 the DiT fills all rows → falls back to host |
| `HY_TT_VAE` | `0` | VAE decode on device; auto **tile-shards** across the mesh when `ndev>1` |
| `HY_VAE_TILE` / `HY_VAE_TILE_PX` | `0` / `0` | Enable tiled VAE / per-tile px. **Use 128 at 121f** (192 fragments DRAM) |
| `HY_FRAMES` / `HY_STEPS` | `13` / `50` | Frame count / denoise steps |
| `HY_H` / `HY_W` | `480` / `848` | Output height / width |
| `HY_TRACE` | `1` | `1` = trace + 2CQ; `0` = eager (per-step ~identical at 121f; capture is one-time) |
| `HY_PROMPT` / `HY_NEG_PROMPT` | cat / — | Positive / negative prompt |
| `HY_OUT` / `HY_FPS` | `/tmp/hy15_stage2b_qb2` / `24` | Output dir / video fps |

## Performance

### Full 121-frame generation (real weights, 480×848, 50 steps)
| config | denoise | VAE | **e2e** |
|---|---|---|---|
| baseline (8-chip) | ~15:00 | — | 18:52 |
| 16-chip, sp=2 | 6:21 | ~2:30 | 10:38 |
| 32-chip, sp=4, replicated VAE | 3:05 | ~5:00 | 9:47 |
| 32-chip, sp=4, tile-sharded VAE | 2:49 | ~2:00 | 5:59 |
| **32-chip, sp=8×tp=4 + persist-RS + SDPA-cfg** | 1:51 | ~2:00 | **5:09** ✅ |

The denoise is **attention/CCL-bound at 121f** (seq ~49k) — a different regime than the
dispatch-bound `device_ms` work below. Two levers, in order:
1. **sp=8×tp=4** (`HY_MESH=8,4`): sequence-parallel divides the O(seq²) attention, so
   doubling SP cut denoise **3.24 → 2.15 s/it**, e2e 5:59 → 5:13 (zero code).
2. **Two block levers** (`_stubs/hunyuan_video15_transformer_block.py`): reduce-scatter uses
   its persistent buffer in the bf16 path, and SDPA gets its own compute config with
   `fp32_dest_acc_en=False` (the fp32 dest-accum halved softmax throughput) → **2.02 s/it**,
   e2e 5:13 → 5:09.

Net denoise **2:49 → 1:51 (−34%)**, quality-preserving throughout (frame-PCC ≥ 0.99 on
matched forwards: sp8-vs-sp4 0.9971, + block levers 0.9938; output crisp). The **VAE decode
is tile-sharded** (`_decode_batch_sharded`, `085be79`): the ~45 tiles are batched + sharded
so each device decodes one tile/round (2 passes, not 45 sequential), reusing the full mesh
after denoise. *(sp=2×tp=16 / any tp=16 are impossible on the 8×4 Galaxy; CCL Ring / num_links>2
need a torus fabric FABRIC_1D doesn't provide.)*

### Per-step DiT `device_ms` (trace+2CQ, capped 2-layer / M=32 profiling workload)
| phase | key levers | device_ms |
|---|---|---|
| baseline | — | 6.43 |
| bias→matmul fusion | fold every `Linear` bias into the `ttnn.linear` epilogue | 6.02 |
| matmul fusion | AdaLN 6-way modulation (C→6C); column-parallel QKV | 5.37 |
| ternary fusion | `addcmul` for AdaLN modulation + gated residuals | 5.31 |
| LayerNorm width-sharding | `norm2` / `_adazero` / `norm_out` LN over a core row | 5.17 |
| launch dedup | row-linear bias fold, RoPE `mul+addcmul`, AdaLN slice, silu/RoPE hoist | **5.10** |

16 wins + an L1-fit guard (`d0d07d0`, width-shard LN falls back to interleaved at large M —
fixes a 121f OOM). A hand-authored **tt-lang fused-FFN kernel** was on-device correct
(PCC 0.99999) but **+3%** — the FF matmul is dispatch-bound at M=32, so a compute kernel
can't beat the tuned `ttnn` matmul. (Note: these device_ms wins are at the profiled scale;
the 121f e2e win comes from the sequence-parallel + VAE levers above.)

### What runs where (sp=8×tp=4)
| stage | placement | note |
|---|---|---|
| Qwen 2.5-VL + byT5 text-encode | **host (CPU)** | one-time; DiT fills all rows so no encoder submesh (their DiT-side projections *are* on device) |
| Latent init + scheduler step | **host (CPU)** | stock diffusers; cheap per-step elementwise |
| **DiT denoise** (50 steps) | **on device** | all 32 chips, sp=8 × tp=4 |
| **VAE decode** | **on device** | tile-sharded across all 32 chips |
| Frame post-proc → mp4/gif | **host (CPU)** | one-time save |

Only the two heavy compute stages (DiT, VAE) run on device; all host work is one-time or a
cheap per-step update, so it barely dents the denoise-dominated 5:09. On-device Qwen is
available at any sp via `HY_TT_QWEN_SHARED=1` (TP=4 + FSDP on the DiT's mesh) but is ~1 min
*slower* for a one-shot video (7B weight-load + FSDP don't amortize) — worth it only for a
served / multi-prompt deployment.

## PCC / correctness
- **Per-component (18/18):** every stub has `tests/pcc/test_<module>.py` (+ a `_mesh` variant
  for the transformer block); all pass native ttnn.
- **End-to-end gate** (`test_e2e_pipeline.py`, 2-layer seed-0 reference weights, 3
  granularities): **18/18 modules invoked, min PCC 0.999979** (threshold 0.95).
- **Real-weight DiT** (`test_real_weight_pcc.py`, single + 24-chip mesh): threshold 0.99.
- **sp-degree consistency:** validated by 1-step frame-PCC on the cached community weights
  (matched prompt/seed) — sp=8 vs sp=4 = 0.9971.

## Notes
- **Reference vs real weights:** the per-component / e2e-gate tests use deterministic-random
  (seed 0) weights shrunk to `num_layers=2` for CPU feasibility (validating the op
  implementation); `test_real_weight_pcc*` and generation use the real checkpoint (community
  diffusers conversion). *(The tencent 720p PCC weights are not cached on this box.)*
- **L1-fit guard:** width-shard LN needs ~16 MB/core at 121f (bank is 1.4 MB) → OOM; `_wln`
  falls back to interleaved LN when the shard won't fit L1 (`d0d07d0`).
