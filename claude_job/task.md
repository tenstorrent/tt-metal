we have successfully run the dual_T3K_test. This proves the physical discovery, link/traffic validation, and the bigmesh test binary via `tt-run` + `mpirun-ulfm` is working.

The infrastructure has been setup, but we do not yet understand fully how the software stacks works together. Study how the Tenstorrent software layer work end-to-end, and prove it by a minimal but complete TTNN workload.

## Goal

From a Python process launched across both T3K hosts, import ttnn, open a single Big Mesh device spanning all 16 chips, create sharded/replicated tensors, run compute (add, matmul), and run collective communication (all-gather / all-reduce / reduce-scatter) — and document exactly which layer does what and how they hand off.

## Target software stacks

| Layer | Component | Role |
|---|---|---|
| Physical / firmware | tt-topology | Flash the Ethernet mesh routing layout onto each host's local Wormhole cards. Runs per host (single-host tool); both boxes flashed to the same layout. |
| Runtime interconnect | tt-fabric (tt-metal) | Control plane ingests the Mesh Graph Descriptor (MGD), discovers the physical links, builds routing tables, brings up the data plane across the two hosts. |
| Fabric lifecycle mgmt | TTFM (TT Fabric Manager) | Initializes / monitors / tears down fabric (tools/scaleout/fabric_manager, FabricManagerMode). Understand its role vs. Metal's built-in init, create connection graph. |
| Launch / rank mapping | tt-run + mpirun-ulfm | Maps MPI ranks → {mesh_id, mesh_host_rank}, sets per-rank env, spawns the workload on both hosts. |
| App API | ttnn + ttnn.fabric bindings | `ttnn.set_fabric_config`, `ttnn.open_mesh_device`, tensors, ops, CCL. |
| Orchestration (study) | tt-operator | OPTIONAL — if have time then try to run this too |

## Output

- A Python script the performs the full workflow on dual T3K.
- Documentation of the layer-by-layer work (topology→fabric→ttnn) (can be presented to the team).
- Verify the correctness of the computation and collection communication by running a torch golden as reference.
