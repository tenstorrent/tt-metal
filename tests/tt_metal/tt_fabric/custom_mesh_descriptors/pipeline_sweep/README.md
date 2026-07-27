# Pipeline sweep MGDs

Generated ring-pipeline Mesh Graph Descriptors used by the **bh-pipeline-sweep** test group in
`tests/scripts/multihost/run_fabric_cpu_only_unit_tests.sh`. They stress the topology mapper on the
SC36 mock cluster (36 hosts × 32 ASICs = 1152 ASICs).

## Shapes

| Shape | Device topology | ASICs/stage | Host topology | Pinnings | Stage counts |
|-------|-----------------|-------------|---------------|----------|--------------|
| **2x4** | `[4, 2]` RING,LINE | 8 | `[1, 1]` single-host | none | 16, 32, 64, 80, 96, 112, 128, 144 |
| **4x4** | `[4, 4]` RING,RING | 16 | `[2, 1]` split-host galaxy | all-to-all corner groups | 8, 16, 32, 40, 48, 56, 64, 72 |

Each ring closes (last stage → stage 0). The largest ring per shape exactly fills the mock:
144 × 8 = 72 × 16 = 1152 ASICs.

4x4 stages use **many-to-many corner pinnings** keyed by `mesh_id_regex` (even/odd mesh-id parity).
Any listed logical chip may map to any listed tray/ASIC position; the solver enforces a bijection.
This requires the MGD pinning regex / all-to-all support from the stacked MGD PR.

## Regenerating

Stage counts and pinnings live in the config file, not on the command line:

```bash
python3 tests/scripts/multihost/gen_pipeline_sweep_mgds.py
```

This reads `tests/scripts/multihost/pipeline_sweep_config.yaml` and writes all 16
`sweep_<shape>_pipeline_<N>stage_mesh_graph_descriptor.textproto` files here.

Override paths if needed:

```bash
python3 tests/scripts/multihost/gen_pipeline_sweep_mgds.py \
  --config /path/to/pipeline_sweep_config.yaml \
  --out-dir tests/tt_metal/tt_fabric/custom_mesh_descriptors/pipeline_sweep
```

### Config file

Edit `tests/scripts/multihost/pipeline_sweep_config.yaml` to change stage counts or pinning groups.
Example 4x4 pinning group:

```yaml
shapes:
  4x4:
    pinnings:
      groups:
        - name: even
          mesh_id_regex: "[0-9]*[02468]"
          chip_ids: [0, 1, 3, 12, 15]
          positions:
            - { tray_id: 1, asic_location: 3 }
            - { tray_id: 1, asic_location: 7 }
            # ...
```

Do not edit the generated `.textproto` files by hand.

## Running the sweep

Opt-in test group (16 MGDs × 20 rank-binding solutions = 320 runs):

```bash
tests/scripts/multihost/run_fabric_cpu_only_unit_tests.sh bh-pipeline-sweep
```

Each MGD is swept with `tools/scaleout/sweep_rank_binding_solutions.py` on the SC36 revC subtorus
aisleD mock; the workload is `ControlPlaneFixture.TestBlitzDecodePipelineBuilder`.
