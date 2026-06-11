# Pi0.5 on BH Galaxy (`tt_bh_glx`)

Multi-chip TTNN inference for **π₀.₅** on a **Blackhole Galaxy** 8×4 mesh (32 chips). The model is sliced across **28 chips** in three pipeline stages — vision, prefill, and denoise — with on-device fabric sockets for cross-chip tensor handoff.

This is an experimental distribution path. The production single-chip implementation lives in `ttnn_pi0_5_model.py` (~65 ms/chunk with trace+2CQ). This Galaxy path trades latency for scale: it proves correctness and measures per-stage cost before trace capture and further optimization.

---

## What this experiment is testing

| Goal | Description |
|------|-------------|
| **Model parallelism** | Split SigLIP (4 chips), Gemma-2B VLM prefill (18 chips), and AdaRMS expert denoise (6 chips) so each chip holds only a slice of weights |
| **Cross-chip transport** | Move activations and KV caches between 1×1 per-chip submeshes via fabric mesh sockets (`send_direct_async` / `recv_direct_async`) instead of host bounce |
| **End-to-end correctness** | `Pi0_5GLXPipeline.sample_actions` matches the PyTorch reference (target PCC ≥ 0.95) |
| **Perf breakdown** | Per-stage wall-clock via `StageTimings` to find bottlenecks before Phase B trace work |

---

## Mesh layout (28 of 32 chips)

Physical placement on the parent 8×4 mesh (`stages.py`):

```
col→  0 1 2 3
row↓  0  V V V V    V = vision   (1×4) @ (0,0)   — 4 chips
      1  P P P D    P = prefill  (6×3) @ (1,0)  — 18 chips
      2  P P P D    D = denoise  (6×1) @ (1,3)  — 6 chips
      3  P P P D
      4  P P P D
      5  P P P D
      6  P P P D
      7  . . . D    (row 7 cols 0–2 = 3 spare chips)
```

### Per-stage layer mapping

| Stage | Chips | Layers per chip | Model component |
|-------|-------|-----------------|-----------------|
| **Vision** | 4 | chip 0: embed only; chips 1–3: 9 SigLIP layers each | SigLIP-27 + mm_projector → `(B, 256, 2048)` per camera |
| **Prefill** | 18 | 1 Gemma-2B block per chip | VLM prefix pass; per-layer K/V stays on owning chip |
| **Denoise** | 6 | 3 AdaRMS Gemma-300M blocks per chip | Expert chain with cross-attn to migrated prefix KV |

---

## Inference flow

One `sample_actions` call runs this sequence (`pipeline.py`):

```
Images (torch)
  → [Vision] 4-chip SigLIP pipeline
  → socket transport → prefill[0]
  → embed lang tokens + concat prefix
  → [Prefill] 18-chip VLM → final_hidden + per_layer_kv[0..17]
  → [KV migration] layer-paired prefill → denoise (bf16 → bf8_b typecast)
  → [Denoise loop] N Euler steps:
       embed_actions(x_t) → embed_adarms_cond(t) per chip
       → 6-chip expert chain → final adaRMS norm → action_out_proj
       → velocity to host → x_t += dt·velocity (fp32 on host)
  → slice padded actions → (1, action_horizon, action_dim)
```

Expected steady-state latency is ~500–700 ms (vs ~44 ms single-chip trace baseline). Use `StageTimings` to see where time goes.

---

## Module map

| File | Role |
|------|------|
| `stages.py` | Mesh constants, `MeshHandles`, `StageTimings` |
| `mesh_setup.py` | `open_galaxy_mesh()` — opens 8×4 parent, carves submeshes, enables `FABRIC_2D` |
| `transport.py` | `SocketTransport` — cached socket pairs + receiver buffers between chips |
| `kv_migration.py` | Layer-paired prefill K/V → denoise chips (3 layers per denoise chip) |
| `vision_slice.py` | `SigLIPEmbedSlice`, `SigLIPLayerSlice`, `SigLIPTailSlice` |
| `vlm_slice.py` | `VLMBlockSlice` — one Gemma-2B block per chip |
| `expert_slice.py` | `ExpertChunkSlice` — 3 AdaRMS expert blocks per chip |
| `suffix_slice.py` | Replicated suffix MLP (`action_in_proj`, `time_mlp`, `action_out_proj`) per denoise chip |
| `stage_vision.py` | 4-chip vision stage driver |
| `stage_prefill.py` | 18-chip prefill stage driver |
| `stage_denoise.py` | 6-chip expert-chain driver |
| `pipeline.py` | `Pi0_5GLXPipeline` — end-to-end `sample_actions`, trace hooks (Phase B) |

---

## Transport: socket (v2) vs host bounce (v1)

**Default (v2):** `SocketTransport` uses fabric mesh sockets. Tensors move on-device with no `ttnn.synchronize_device` — trace-compatible. Requires `FABRIC_2D` (cross-stage hops like vision[3]→prefill[0] span both mesh axes; `FABRIC_1D` fails on those).

**Legacy (v1):** Set `PI05_GLX_TRANSPORT=host` to use `send_via_host` (read tensor to host, re-upload). Useful for A/B debugging without rebuilding.

Socket pairs and receiver buffers are cached per `(src_chip, dst_chip, tag)` so wiring cost is paid once.

---

## Quick start

### Prerequisites

- BH Galaxy hardware (32-chip 8×4 mesh)
- Pi0.5 checkpoint at `PI05_CHECKPOINT_DIR` (default: `/home/tt-admin/pi05_cache/pi05_libero_upstream`)
- From repo root:

```bash
source _bench_runs/pi05_production.env   # recommended production knobs
export PYTHONPATH=$PWD TT_METAL_HOME=$PWD
```

### Smoke test (mesh carve + host bounce)

```bash
python_env/bin/pytest -xvs \
  models/experimental/pi0_5/tests/pcc/test_pcc_tt_bh_glx_stages.py::test_mesh_carve_smoke
```

### End-to-end PCC (correctness)

```bash
python_env/bin/pytest -xvs \
  models/experimental/pi0_5/tests/pcc/test_pcc_tt_bh_glx_e2e.py
```

### End-to-end perf (per-stage breakdown)

```bash
python_env/bin/pytest -xvs \
  models/experimental/pi0_5/tests/perf/test_perf_tt_bh_glx_e2e.py
```

### Per-stage perf (isolated stages)

```bash
python_env/bin/pytest -xvs \
  models/experimental/pi0_5/tests/perf/test_perf_tt_bh_glx_stages.py
```

### Per-stage PCC

```bash
python_env/bin/pytest -xvs \
  models/experimental/pi0_5/tests/pcc/test_pcc_tt_bh_glx_stages.py
```

---

## Programmatic usage

```python
from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig
from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader
from models.experimental.pi0_5.tt.tt_bh_glx.mesh_setup import open_galaxy_mesh
from models.experimental.pi0_5.tt.tt_bh_glx.pipeline import Pi0_5GLXPipeline

cfg = Pi0_5ModelConfig(action_horizon=50, num_denoising_steps=5)
loader = Pi0_5WeightLoader("/path/to/checkpoint")

with open_galaxy_mesh(l1_small_size=24576) as handles:
    pipe = Pi0_5GLXPipeline(cfg, loader.categorized_weights, handles)
    actions, timings = pipe.sample_actions(
        images=images,       # list of (1, 3, 224, 224) torch tensors
        img_masks=img_masks, # list of (1,) bool masks
        lang_tokens=lang_tokens,  # (1, lang_len) int32
        lang_masks=lang_masks,    # (1, lang_len) bool
    )
    print(f"total={timings.total_ms:.1f}ms  vision={timings.vision_ms:.1f}ms  "
          f"prefill={timings.prefill_ms:.1f}ms  denoise={timings.denoise_total_ms:.1f}ms")
```

Trace capture (`capture_trace` / `sample_actions_traced`) is wired for Phase B — persistent input buffers on vision[0], prefill[0], and denoise[0] avoid per-call allocations.

---

## Environment variables

| Variable | Default | Effect |
|----------|---------|--------|
| `PI05_CHECKPOINT_DIR` | `/home/tt-admin/pi05_cache/pi05_libero_upstream` | Weight checkpoint path |
| `PI05_NUM_DENOISE_STEPS` | `5` | Denoising iterations (tests) |
| `PI0_NUM_CAMERAS` | `3` | Number of camera views |
| `PI05_GLX_NUM_WARMUP` | `1` (e2e) / `1` (stages) | Warmup iterations before timing |
| `PI05_GLX_NUM_ITERS` | `3` (e2e) / `5` (stages) | Timed iterations |
| `PI05_GLX_TRANSPORT` | *(unset → sockets)* | Set to `host` for legacy host-bounce transport |
| `PI0_UPSTREAM_MASKS` | *(unset)* | When `1`, builds pad-aware attention masks + position-aware RoPE (upstream-openpi compat) |
| `PI0_ROPE_TABLES_L1` | *(unset)* | When `1`, places RoPE tables in L1 instead of DRAM |

---

## Simplifications vs single-chip `ttnn_pi0_5_model.py`

- `batch_size = 1` only
- No `keep_padded` / precomputed-mod fast paths
- Host-side Euler integration for `x_t` (re-uploaded each denoise step in eager mode)
- KV migration typecasts bf16 → bf8_b to match expert cross-attn concat dtype (~3 PCC points; follow-up is to typecast inside attention instead)

When `PI0_UPSTREAM_MASKS=1`, the pipeline builds and caches per-chip mask/RoPE artifacts (same idea as the single-chip model).

---

## Roadmap / open items

1. **Phase B trace** — capture full `sample_actions` on parent mesh CQ 0; on-device Euler with in-place `x_t` buffer
2. **KV dtype** — eliminate bf8_b typecast in migration to recover PCC
3. **Latency** — socket transport removes host-bounce overhead; trace + 2CQ should narrow the gap vs single-chip

---

## Related docs

- Parent package overview: [`models/experimental/pi0_5/README.md`](../../README.md)
- Single-chip TTNN model: `models/experimental/pi0_5/tt/ttnn_pi0_5_model.py`
- Tests: `models/experimental/pi0_5/tests/pcc/test_pcc_tt_bh_glx_*` and `tests/perf/test_perf_tt_bh_glx_*`
