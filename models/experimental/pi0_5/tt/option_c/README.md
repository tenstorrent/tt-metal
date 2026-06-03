# pi0.5 Option C — heterogeneous 3-stage pipeline (no TP within stage)

**Status**: scaffolding. Stage skeletons in place; full forward path is a
work in progress. See `docs/PI0_5_GALAXY_DEPLOYMENT_PLAN.md` §3 for the
design and the analytical perf model that picked this option as the
deployment target after Option B.

## Why Option C after Option B

Option B's measured per-VLM-layer cost (`test_option_b_benchmark.py`,
S=64): **3.27 ms / layer**, of which **1.25 ms (~38%)** was the two
all_reduces (~625 µs each, payload-floored). Option C runs 1 layer per
chip, so **no all_reduce exists at all** — the analytical model projects
0 ms of collective overhead, at the cost of less per-chip compute parallelism.

## Mapping

```
   col→  0 1 2 3
row↓  0  V V _ _    V  = vision      4 chips  shape (2,2) offset (0,0)
      1  V V _ _    _  = spare       4 chips  shape (2,2) offset (0,2)
      2  P P P D    P  = prefill    18 chips  shape (6,3) offset (2,0)
      3  P P P D    D  = denoise     6 chips  shape (6,1) offset (2,3)
      4  P P P D    Total used:     28 / 32   (4 spare)
      5  P P P D
      6  P P P D
      7  P P P D
```

| Stage     | Chips | Submesh shape | Layers per chip                                    |
|-----------|-------|---------------|----------------------------------------------------|
| vision    | 4     | (2,2)         | 9 SigLIP layers × 3 chips + 1 mm_proj/embed chip   |
| prefill   | 18    | (6,3)         | 1 VLM transformer layer per chip                   |
| denoise   | 6     | (6,1)         | 3 expert layers per chip; suffix MLP replicated    |
| spare     | 4     | (2,2)         | unused — reserved for denoise replica / batching   |

## L1-resident weights / biases / activations

Every weight, bias, and activation is placed in L1
(`memory_config=ttnn.L1_MEMORY_CONFIG`). This works for Option C
specifically because:

1. **No collectives inside a stage.** Option B's `tp_block.py` documents
   that L1-resident weights collide with `ttnn.all_reduce`'s static CB
   region — Option C has zero all_reduces, so the conflict doesn't apply.
2. **Per-chip budget fits in the 180 MB L1 cap.** Per deployment plan §3.1:
     - vision chip (9 SigLIP layers, bf8 attn):  ≈ 156.9 MB / chip (tight)
     - prefill chip (1 VLM layer):               ≈ 122.5 MB / chip
     - denoise chip (3 expert layers + suffix):  ≈  30.0 MB / chip
3. **No DRAM → L1 hop per call.** Tail activations stay in L1; the only
   off-chip traffic is the host-bounce transport at stage boundaries and
   the one-shot KV migration.

The upload helpers in this directory pass `memory_config=L1_MEMORY_CONFIG`
explicitly everywhere — see `_upload_l1_replicated` in `vlm_slice.py`.

## File map

- `__init__.py`          — re-exports `build_default_layout` / `StageLayout`.
- `stages.py`            — `StageSpec` + `StageLayout` dataclasses; the
                           canonical 4/18/6/+4-spare mapping.
- `mesh_setup.py`        — opens the 8×4 Galaxy and carves it into 3
                           heterogeneous submeshes via
                           `MeshDevice.create_submesh(shape, offset)`.
- `vision_slice.py`      — `Pi0_5OptionCVisionSlice` — SigLIP-27 split into
                           4 chunks (3 chips × 9 SigLIP layers + 1 mm_proj
                           chip). Weights L1-resident, no TP.
- `vlm_slice.py`         — `Pi0_5OptionCVLMSlice` — 1 VLM layer per chip,
                           L1-resident weights, no TP.
- `expert_slice.py`      — `Pi0_5OptionCExpertSlice` — expert layers chunked
                           across 6 chips, L1-resident, no TP.
- `suffix_slice.py`      — replicated suffix MLP on the denoise submesh.
- `transport.py`         — host-bounce activation passing between
                           heterogeneous submeshes (same fallback as
                           option_b/transport.py).
- `kv_migration.py`      — layer-paired KV migration: prefill chip i (which
                           owns VLM layer i) ships (K, V) to denoise chip
                           `i // 3` (which owns expert layer i).
- `stage_vision.py`      — Stage 0 orchestrator.
- `stage_prefill.py`     — Stage 1 orchestrator.
- `stage_denoise.py`     — Stage 2 orchestrator (runs the 10-step Euler loop).
- `pipeline.py`          — `Pi0_5PipelineC` end-to-end driver.

## What this is NOT (yet)

- Inter-stage and inter-chip transport is host-bounce (same fallback as
  Option B). Direct D2D copy is a follow-up once tt-blaze sockets land.
- Vocab sharding of `embed_tokens` (527 MB) — the language token table
  still lives on host and the resolved embeddings are uploaded as
  activations. Deployment plan §3.1 option (a).

## What landed (2026-06-03)

- **Layer-paired L1 prefill** (`vlm_slice.Pi0_5OptionCVLMSlicePaired` +
  `StagePrefill(..., layer_paired_l1=True)`): 1 VLM layer per chip across
  the 18-chip prefill submesh, weights L1-resident, no replication.
- **Layer-paired L1 denoise** (`expert_slice.Pi0_5OptionCExpertSlicePaired`
  + `StageDenoise(..., layer_paired_l1=True)`): 3 expert layers per chip
  × 6 chips, weights L1-resident. Denoise loop bounces x_t back to
  the first chip between Euler steps.
- **On-device 3-chip SigLIP** (`vision_slice.Pi0_5OptionCVisionSliceSplit`
  + `StageVision(..., device_siglip=True)`): SigLIP-27 runs across 3
  vision chips (9 layers each), mm_projector on the 4th chip. Powered by
  the new `SigLIPVisionTowerTTNN(layer_range=..., holds_patch_embed=...,
  holds_pos_embed=..., holds_post_ln=...)` knobs.
- **Option C benchmark** (`tests/test_option_c_benchmark.py`): staged e2e
  timings, replicated-vs-layer-paired comparison, prefill seq-len sweep.
  Gated by `PI0_OC_BENCHMARK=1`.

The smoke suite (`tests/test_option_c_smoke.py`) now has 11 tests:
the original 7 plus a layer-paired prefill slice, a layer-paired prefill
stage, a layer-paired expert slice, and a device-SigLIP dry-run.

## How to run the smoke test

```bash
cd /home/tt-admin/sdawle/pi0/tt-metal
TT_METAL_HOME=$PWD PYTHONPATH=$PWD \
  python_env/bin/python -m pytest \
  models/experimental/pi0_5/tests/test_option_c_smoke.py -s -v
```

The smoke test:
1. Opens the 8×4 mesh.
2. Creates the 3 heterogeneous submeshes (vision 4 / prefill 18 / denoise 6).
3. Confirms each submesh has the expected chip count and coordinates.
4. Closes everything cleanly.

No model weights are loaded, no compute is run.
