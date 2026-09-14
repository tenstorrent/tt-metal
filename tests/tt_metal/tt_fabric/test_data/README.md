# Pipeline builder CPU regression tests

The focused tests in `fabric_router/test_pipeline_builder.cpp` call the production
solver with synthetic or captured direct links. They do not open a MeshDevice,
need a cluster allocation, or import Blaze. Run from the Metal source root:

```sh
cmake --build build --target fabric_unit_tests -j 8
timeout 120s build/test/tt_metal/tt_fabric/fabric_unit_tests \
  --gtest_filter='PipelineBuilderLayoutTest.*:PipelineBuilderCapacityTest.*:PortCapacity/PipelineBuilderForkTest.*:CoreCapacity/PipelineBuilderReplayTest.*'
```

The tests cover input validation, defaults and overrides, host endpoints,
backtracking, disconnected graphs, parallel edges, forks, and 800 deterministic
comparisons against an independent brute-force oracle. Captured replays test
default/one/two-slot capacities. Positive replays validate stage sizes (when
specified), unique submesh assignments, actual link candidates, endpoint chip
membership, slot bounds, and slot uniqueness including host I/O. Negative replays
require the exact-search-exhausted error; a timeout is a test failure, never proof
of infeasibility. No brittle wall-time assertions are made inside the solver tests.

## Captures

- `llama_1x2_pipeline_connections.txt`: 40-submesh Llama 8B 1x2 pod mock capture.
  Default and two-slot configurations solve; one-slot configuration is infeasible.
- `gemma_sc20_pipeline_connections.txt`: SC20 mixed-shape, two-fabric-return fork,
  74 stages over 139 carved submeshes (640 chips). Captured on 2026-09-13 using
  `SC20_32x4_revC_subtorus_aisleC_mapping.yaml` from tt-cluster-descriptors
  `9bcf698e971a7a49df1aaf17892352c47d359d47`, TORUS_XY, and Blaze's
  `SC20_4x1_slices_mesh_graph_descriptor.textproto`. The MGD, graph shapes, and
  unchanged carve planner came from Blaze
  `afae0afcd97525761d930d24ced01802782887ec`, test
  `test_multi_rank_stages.py::test_fork_with_mixed_stage_sizes_sc20`.
  The planner covered each chip once; submeshes are sorted by mesh/coordinate,
  not successful stage placement order. Direct chip links were projected from
  the mock control plane onto the carved submeshes. Default/two slots solve;
  one slot cannot fit the four-chip router's six endpoints. The pre-capacity
  upstream algorithm also solved this same input during capture verification.

These files contain inputs, not golden placements: any valid answer is accepted.
They do not validate automapping, Blaze graph generation/stage insertion, runtime
rank ownership, physical worker-core assignment, or hardware traffic. For fresh
mock discovery, the separate `PipelineBuilderMockSweep.RingCapacity` test needs
`tt-run`, an MGD, and a mock mapping; it is not part of the focused filter above.

## File format

Whitespace-separated integers:

1. Number of submeshes.
2. For each submesh: number of chips, then `(mesh_id, chip_id, local_row, local_col)`
   for each chip.
3. Until EOF: `(source_submesh, destination_submesh, exit_row, exit_col,
   entry_row, entry_col)` for every direct link candidate.

Updating a capture requires checking its topology/graph provenance and rerunning
both positive and negative cases, rather than replacing data to match a new answer.
