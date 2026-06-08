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

Source of truth: `stages.py::build_default_layout`. All 32 chips are used —
the earlier 4-spare layout was rejected to fit `embed_tokens` (527 MB) on the
vision submesh instead of pre-embedding on host. **No spare chips.**

```
   col→  0 1 2 3
row↓  0  V V V V    V  = vision      8 chips  shape (2,4) offset (0,0)
      1  V V V V                              (all 27 SigLIP + mm_proj + embed_tokens)
      2  P P P D    P  = prefill    18 chips  shape (6,3) offset (2,0)
      3  P P P D                              (1 VLM transformer layer per chip)
      4  P P P D    D  = denoise     6 chips  shape (6,1) offset (2,3)
      5  P P P D                              (3 expert layers per chip + suffix MLP)
      6  P P P D    Total used:     32 / 32   (no spare)
      7  P P P D
```

| Stage     | Chips | Submesh shape | Submesh offset | Layers per chip / contents                                        |
|-----------|-------|---------------|----------------|-------------------------------------------------------------------|
| vision    | 8     | (2,4)         | (0,0)          | All 27 SigLIP layers + mm_projector + embed_tokens (~66 MB embed shared across chips) |
| prefill   | 18    | (6,3)         | (2,0)          | 1 VLM transformer layer per chip + KV cache slot                  |
| denoise   | 6     | (6,1)         | (2,3)          | 3 expert layers per chip; suffix MLP replicated; runs 10-step Euler loop |

## L1-resident weights / biases / activations

Every weight, bias, and activation is placed in L1
(`memory_config=ttnn.L1_MEMORY_CONFIG`). This works for Option C
specifically because:

1. **No collectives inside a stage.** Option B's `tp_block.py` documents
   that L1-resident weights collide with `ttnn.all_reduce`'s static CB
   region — Option C has zero all_reduces, so the conflict doesn't apply.
2. **Per-chip budget fits in the 180 MB L1 cap.** Per deployment plan §3.1
   (budget figures unchanged from earlier 4-chip-vision layout; updated
   per-chip layer counts reduce vision-chip pressure further at 8 chips):
     - vision chip (all 27 SigLIP layers spread + embed_tokens shard + mm_proj): ≲ 140 MB / chip
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
- `vision_slice.py`      — `Pi0_5OptionCVisionSlice` — SigLIP-27 split across
                           the 8 vision chips (per-chip layer chunking
                           computed at build time, mm_projector co-located on
                           the last chip). Weights L1-resident, no TP.
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
- Note: `embed_tokens` (527 MB) is now placed on the vision submesh
  (`holds_embed_tokens=True` per `stages.py`), so the host-side
  pre-embed shortcut is no longer used.

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
source python_env/bin/activate && \
  export TT_METAL_HOME=$PWD && export PYTHONPATH=$PWD && \
  source _bench_runs/pi05_production.env && \
  pytest -s -v models/experimental/pi0_5/tests/test_option_c_smoke.py
```

The smoke suite has **15 tests** covering the full Option C surface:

1. `test_default_layout_shape_c` — assert `StageLayout` matches `vision (2,4)
   = 8 chips`, `prefill (6,3) = 18 chips`, `denoise (6,1) = 6 chips`.
2. `test_open_32_chip_mesh_partition_c` — open the 8×4 mesh, carve into the
   3 heterogeneous submeshes, confirm chip counts and shapes.
3. `test_vlm_slice_forward_one_layer_c` — one real VLM transformer layer
   forward on the 18-chip prefill submesh (single layer, S=64 random
   activation; validates the per-layer compute path).
4. `test_expert_slice_forward_one_layer_c` — one expert decoder layer
   forward on the 6-chip denoise submesh.
5. `test_inter_submesh_host_bounce_c` — round-trip a tensor through the
   host-bounce transport between two submeshes.
6. `test_e2e_vlm_to_expert_shrunk_c` — minimal vlm→expert dataflow at
   reduced depth.
7. `test_full_pipeline_object_dry_run_c` — construct the full
   `Pi0_5PipelineC` object across all three submeshes (no compute).
8. Layer-paired L1 variants (`test_vlm_slice_layer_paired_l1_two_layers`,
   `test_stage_prefill_layer_paired_l1_dry_run`,
   `test_expert_slice_layer_paired_l1_two_chips`, plus
   `test_vision_slice_device_siglip_split_dry_run`) — exercise the full-L1
   placement path. These currently trip the
   `pi05-mesh-close-ordering` bug (close submeshes BEFORE parent — see the
   memory note); fix is open.
9. `test_prefill_tp_2x1_submesh_carving` — carve the 18-chip prefill mesh
   into 9 × (2,1) TP=2 sub-meshes (for the parked TP=2 PCC investigation).
