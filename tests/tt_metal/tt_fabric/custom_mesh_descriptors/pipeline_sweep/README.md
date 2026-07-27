# Pipeline sweep MGDs

MGDs for the bh-pipeline-sweep multihost tests are **generated**, not checked in.

Generate one file:

```bash
python3 tests/scripts/multihost/gen_pipeline_sweep_mgds.py \
  --shape 4x4 --stages 48 --hosts 2 --pinnings \
  --out /tmp/sweep_4x4_pipeline_48stage_mesh_graph_descriptor.textproto
```

Regenerate the full SC36 preset set:

```bash
python3 tests/scripts/multihost/gen_pipeline_sweep_mgds.py --all-presets
```

Arguments:

- `--shape {2x4,4x4}` — mesh shape (8 or 16 ASICs per stage)
- `--stages N` — number of pipeline stages (ring always closes: last stage -> stage 0)
- `--hosts N` — hosts per stage mesh (`host_topology` width; default 1 for 2x4, 2 for 4x4)
- `--pinnings` — emit all-to-all corner pinnings (4x4 only)
