# pi0.5 Option B — 4-stage pipeline (TP=8 per stage)

**Status**: scaffolding. Stage skeletons in place; full forward path is a work in
progress. See `docs/PI0_5_GALAXY_DEPLOYMENT_PLAN.md` §3 and §6 for the design and
the analytical perf model that picked this option.

## Mapping

| Stage | Chips | Submesh | Contents |
|-------|-------|---------|----------|
| 0     | 8     | 4×2     | SigLIP-27 vision tower + multi-modal projector + VLM embed_tokens |
| 1     | 8     | 4×2     | VLM PaliGemma layers 0–8 (9 layers) |
| 2     | 8     | 4×2     | VLM PaliGemma layers 9–17 (9 layers) + KV-migration emitter |
| 3     | 8     | 4×2     | Expert (Gemma-300M) layers 0–17 + Suffix MLP + denoise loop |

Total: 32 chips, full Galaxy.

## File map

- `mesh_setup.py` — opens the 8×4 mesh and slices it into 4× 4×2 submeshes via
  `MeshDevice.create_submeshes(MeshShape(4, 2))`.
- `stages.py` — `StageLayout` dataclass + `build_default_layout()` for the layer→stage
  mapping above.
- `stage_0_vision.py` — SigLIP + mm_proj + embed_tokens stage. Wraps existing
  `SigLIPVisionTowerTTNN` and `MultiModalProjectorTTNN` for a 4×2 submesh.
- `stage_vlm.py` — generic VLM half-stack stage; instantiated twice for stages 1 and 2.
  Wraps a slice of `Pi0_5PaliGemmaBackboneTTNN.vlm_blocks`.
- `stage_3_expert.py` — Expert + suffix stage. Holds all 18 expert layers + suffix MLP +
  the 10-step denoise loop.
- `kv_migration.py` — `KVMigration` helper. After stage 2 completes prefill, copies
  the full VLM KV cache (~8.9 MB at bf16, or ~4.5 MB at bf8) from stage 2's submesh
  to stage 3's submesh via cross-device tensor reshape.
- `pipeline.py` — `Pi0_5PipelineB` orchestrator. Drives stages 0→1→2→migrate→3 for
  one inference call.
- `transport.py` — inter-stage activation passing. Uses point-to-point copy via DRAM
  staging where ttnn's collective APIs don't yet support direct submesh→submesh send.

## Key APIs from tt-metal we depend on

- `ttnn.open_mesh_device(MeshShape(8, 4))` — confirmed available
  (`ttnn/ttnn/distributed/distributed.py:644`).
- `MeshDevice.create_submeshes(MeshShape(4, 2))` — slices parent into uniform tiles.
- `ttnn.shard_tensor_to_mesh_mapper(submesh, dim)` — per-stage TP=8 weight sharding.
- `ttnn.all_reduce(t, cluster_axis=...)` — collective inside a 4×2 submesh after
  row-parallel matmuls (o_proj, down_proj).
- `ttnn.all_gather(t, cluster_axis=...)` — used for KV migration and inter-stage
  payload passing where appropriate.

## What this is NOT

- This is not yet a working model. Stages 0–3 currently raise `NotImplementedError`
  on `forward()`; they exist to nail down the interfaces.
- KV migration is stubbed as "DRAM bounce via host" until we know whether a direct
  D2D copy primitive between submeshes exists (likely needs `ttnn.copy` or a custom
  reshape-via-parent-mesh, TBD).
- No perf measurements yet.

## How to run the smoke test

```bash
cd /home/tt-admin/sdawle/pi0/tt-metal
TT_METAL_HOME=$PWD PYTHONPATH=$PWD \
  python_env/bin/python -m pytest \
  models/experimental/pi0_5/tests/test_option_b_smoke.py -s -v
```

The smoke test:
1. Opens an 8×4 mesh.
2. Creates 4× 4×2 submeshes.
3. Confirms each submesh has 8 chips and the expected coords.
4. Closes everything cleanly.

No model weights are loaded, no compute is run. This validates the mesh plumbing
only.
