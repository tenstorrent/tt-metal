# Pipeline sweep MGDs

Ring-pipeline Mesh Graph Descriptors used by the **bh-pipeline-sweep** test group in
`tests/scripts/multihost/run_fabric_cpu_only_unit_tests.sh`. They stress the topology mapper on the
SC36 mock cluster (36 hosts × 32 ASICs = 1152 ASICs).

The MGDs themselves are **not checked in** — only `gen_pipeline_sweep_mgds.py` and its config are.
The test group generates them into `generated/mgd/pipeline_sweep` before sweeping.

## Shapes

| Shape | Device topology | ASICs/stage | Host topology | Pinnings | Stage counts |
|-------|-----------------|-------------|---------------|----------|--------------|
| **2x4** | `[4, 2]` RING,LINE | 8 | `[1, 1]` single-host | none | 16, 32, 64, 80, 96, 112, 128, 144 |
| **4x4** | `[4, 4]` RING,RING | 16 | `[2, 1]` split-host galaxy | corner group + anchors | 8, 16, 32, 40, 48, 56, 64, 72 |

Each ring closes (last stage → stage 0). The largest ring per shape exactly fills the mock:
144 × 8 = 72 × 16 = 1152 ASICs.

4x4 stages use the same pinnings as the subtorus 4x4 pipeline MGDs: one **many-to-many corner
group** (chips 0/3/12/15 over both split-host location columns) plus two **1:1 orientation anchors**.
Any chip in a group may map to any position in that group and the solver enforces a bijection.
Positions absent from the assigned physical mesh are filtered out, so a single set of entries covers
every mesh — including the anchor for the location column that mesh does not have. This requires the
MGD pinning regex / all-to-all support from the stacked MGD PR.

## Generating

Stage counts and pinnings live in the config file, not on the command line:

```bash
python3 tests/scripts/multihost/gen_pipeline_sweep_mgds.py
```

This reads `tests/scripts/multihost/pipeline_sweep_config.yaml` and writes all 16
`sweep_<shape>_pipeline_<N>stage_mesh_graph_descriptor.textproto` files to
`generated/mgd/pipeline_sweep`.

Override paths if needed:

```bash
python3 tests/scripts/multihost/gen_pipeline_sweep_mgds.py \
  --config /path/to/pipeline_sweep_config.yaml \
  --out-dir /path/to/out-dir
```

### Config file

Edit `tests/scripts/multihost/pipeline_sweep_config.yaml` to change stage counts or pinning groups.
A group names its chips with either `chip_ids` (one entry per chip) or `chip_id_regex`, and each
position takes either `tray_id` or `tray_id_regex`. Regex fields accept ranges (`"1-4"`), comma lists
(`"0,3,12,15"`) and regexes (`"[0-9]*"`).

```yaml
shapes:
  4x4:
    pinnings:
      groups:
        - name: corners, both location columns
          mesh_id_regex: "[0-9]*"
          chip_id_regex: "0,3,12,15"
          positions:
            - { tray_id_regex: "1-4", asic_location: 3 }
            - { tray_id_regex: "1-4", asic_location: 2 }
        - name: orientation anchor, 3/4/7/8 column
          mesh_id_regex: "[0-9]*"
          chip_ids: [1]
          positions:
            - { tray_id: 1, asic_location: 7 }
```

Change the config and regenerate rather than editing a generated `.textproto` by hand.

## Running the sweep

Opt-in test group (16 MGDs × 20 rank-binding solutions = 320 runs):

```bash
tests/scripts/multihost/run_fabric_cpu_only_unit_tests.sh bh-pipeline-sweep
```

Each MGD is swept with `tools/scaleout/sweep_rank_binding_solutions.py` on the SC36 revC subtorus
aisleD mock; the workload is `ControlPlaneFixture.TestBlitzDecodePipelineBuilder`.
