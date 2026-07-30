# Pipeline sweep MGDs

Ring-pipeline Mesh Graph Descriptors for stressing the topology mapper (e.g. on the SC36 mock
cluster: 36 hosts × 32 ASICs = 1152 ASICs). Every MGD is a closed ring of identical per-stage
meshes (last stage → stage 0).

The `.textproto` files are **generated on demand and intentionally not committed** — generate
them with the script below whenever you need them.

## Generating

```bash
python3 tests/scripts/multihost/gen_pipeline_sweep_mgds.py
```

This reads `tests/scripts/multihost/pipeline_sweep_config.yaml` and writes one
`sweep_<shape>_pipeline_<N>stage_mesh_graph_descriptor.textproto` per (shape, stage-count) into
this directory. Useful flags:

```bash
# custom config / output location
python3 tests/scripts/multihost/gen_pipeline_sweep_mgds.py \
  --config /path/to/pipeline_sweep_config.yaml \
  --out-dir /path/to/out

# only one shape
python3 tests/scripts/multihost/gen_pipeline_sweep_mgds.py --shape 4x4
```

## Config-driven: shape, loopback size, and pinnings

The generator is fully data-driven from the YAML config — add a shape by adding an entry, no code
change needed. Each shape controls three things the sweep cares about:

- **Shape** — `device_topology.dims` / `dim_types` (per-stage mesh geometry; ASICs/stage = product
  of dims) and `host_topology.dims` (how each stage is split across hosts).
- **Loopback size** — `loopback.channels` / `policy` sets the inter-stage ring link width (the
  closing wrap edge included); `loopback.assign_z` toggles `assign_z_direction` per edge.
- **Pinnings** — optional `pinnings.groups`: many-to-all ASIC groups keyed by `mesh_id_regex`. Any
  listed logical chip may map to any listed tray/ASIC position and the solver enforces a bijection.
  `mesh_id_regex` supports ranges (`"0-8"`) and comma lists (`"0,2,4-6"`) and expands per matched
  mesh.

Example shape entry:

```yaml
shapes:
  4x4:
    device_topology: { dims: [4, 4], dim_types: [RING, RING] }
    host_topology:   { dims: [2, 1] }
    mesh_channels:   { count: 2, policy: STRICT }
    loopback:        { channels: 8, policy: RELAXED, assign_z: false }
    stages: [8, 16, 32, 40, 48, 56, 64, 72]
    pinnings:
      groups:
        - name: even
          mesh_id_regex: "[0-9]*[02468]"
          chip_ids: [0, 1, 3, 12, 15]
          positions:
            - { tray_id: 1, asic_location: 3 }
            # ...
```

The shapes shipped in the default config (`2x4`: `[4,2]` RING,LINE, 8 ASICs/stage, single-host;
`4x4`: `[4,4]` RING,RING, 16 ASICs/stage, split-host galaxy with corner pinnings) each fill the
SC36 mock at their largest ring: 144 × 8 = 72 × 16 = 1152 ASICs.

Generated MGDs are consumed with `tools/scaleout/sweep_rank_binding_solutions.py`, which sweeps a
given MGD across all valid rank-binding solutions.
