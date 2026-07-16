# Plan: `tt_stack_walkthrough` — TT software stack, end-to-end, proven on dual T3K

## Context

The dual-T3K infra works: `dual_t3k_ops` already runs ttnn add/matmul/all_gather across the
16-chip Big Mesh (commit `3632044cfd6`, mounted framework at `/home/namvu/dual-t3k/tt-metal`).
`claude_job/task.md` asks for the *understanding* deliverable: study how the layers work
end-to-end (tt-topology → tt-fabric → TTFM → tt-run → ttnn), document which layer does what and
how they hand off, and **prove it** with one minimal Big-Mesh TTNN workload — add, matmul, and
**all-gather + all-reduce + reduce-scatter** — each checked vs a torch golden. Team-presentable.

Compute delta over the existing workload: **add all_reduce + reduce_scatter** (all_gather done).
Also **live-demo the in-repo fabric-manager** split lifecycle.

## Layer map (verified during exploration — cite these in STACK.md)

1. **tt-topology** (external pip tool, NOT in repo): `tt-topology -l mesh` flashes per-chip
   `EthCoord {x,y,rack,shelf}` + mesh eth-routing into each WH card's `NODE_INFO`; persists across
   reboot. UMD reads it back (`tt_metal/third_party/umd/device/topology/topology_discovery_wormhole.cpp:98`
   `get_local_eth_coord`) into a `ClusterDescriptor`. Stale flash → "re-run tt-topology"
   (`tt_metal/llrt/tunnels_from_mmio_device.cpp:82`). Show: `test_system_health`,
   `ttfm query-cluster-descriptors` (GUIDE §8).
2. **tt-fabric control plane** (`tt_metal/fabric/`): `ControlPlane` orchestrates `MeshGraph` +
   `MeshGraphDescriptor` (parses `.textproto` MGD) → `run_physical_system_discovery`
   (`physical_system_discovery.cpp:757`: per-host local discovery + MPI-gather +
   `generate_cross_host_connections` on rank 0) → `TopologyMapper` (logical↔physical, SAT:
   `topology_solver*.cpp`) → `RoutingTableGenerator` (intra+inter-mesh) → `FabricContext` →
   `FabricBuilder`. Trace: `control_plane.cpp:414 init_control_plane` → `metal_env.cpp:345
   initialize_fabric_config` → `fabric_firmware_initializer.cpp:311 write_routing_tables_to_all_chips`
   → `device.cpp:508` → `fabric_init.cpp:23` build+launch eth routers.
3. **TTFM (fabric lifecycle)**: (a) in-repo `run_fabric_manager` CLI
   (`tools/scaleout/fabric_manager/run_fabric_manager.cpp`; `configure_fabric_routing` in
   `utils/fabric_manager_utils.cpp:44`) + `FabricManagerMode` enum
   (`tt_metal/api/tt-metalium/experimental/fabric/fabric_types.hpp:44`:
   DEFAULT=INIT|TERMINATE, ENABLED=0 "someone else owns it", INIT_FABRIC, TERMINATE_FABRIC).
   Plain `open_mesh_device` = DEFAULT (Metal owns init+teardown via
   `FabricFirmwareInitializer::init/configure/teardown`). (b) external Docker `tt-fabric-manager`
   (read-only inspector) — already in GUIDE §8. **Live demo:** split lifecycle.
4. **tt-run + mpirun-ulfm** (`ttnn/ttnn/distributed/ttrun.py`): rank → {mesh_id, mesh_host_rank};
   Phase-1 `generate_rank_bindings` (`tools/scaleout/src/generate_rank_bindings.cpp`, cached under
   `generated/ttrun/`); Phase-2 spawns workload on both hosts, forwards `TT_*` env via `-x`.
5. **ttnn app**: `set_fabric_config(FABRIC_2D)` (`fabric.cpp:501 SetFabricConfig` →
   `metal_env.cpp:235`) sets mode + builds control plane/routing tables → `open_mesh_device`
   (`ttnn/ttnn/distributed/distributed.py:644` → `mesh_device.cpp:332 MeshDeviceImpl::create`;
   each rank builds a local `SystemMesh` view via `DistributedCoordinateTranslator`,
   `MaybeRemote::local` for its 8 chips; barrier; `initialize_fabric_and_dispatch_fw` at :431) →
   tensors (`ShardTensorToMesh`/`ReplicateTensorToMesh`) → ops. CCL runtime handoff:
   `ccl_worker_builder.cpp:~1136 append_fabric_connection_rt_args` wires worker kernels to the live
   EDM routers (`ccl/kernels/edm/erisc_datamover.cpp`).
- **tt-operator** (optional): no in-repo source; only CI GitHub Actions (`.github/actions/ttop-*`)
  hitting an external K8s allocation service. One paragraph, don't run.

## CCL signatures (verified stable ops — same pattern as the working all_gather)

- `ttnn.all_gather(t, dim, cluster_axis=1, topology=ttnn.Topology.Linear)`
- `ttnn.all_reduce(t, cluster_axis=1, topology=ttnn.Topology.Linear)` — Sum, **no `dim`**.
- `ttnn.reduce_scatter(t, dim, cluster_axis=1, topology=ttnn.Topology.Linear)` — output dim size
  = input/num_devices. (Stable ops internally call experimental async impls + manage semaphores.)
- `ttnn.Topology`: Linear (line, no wrap) for the 1×16; Ring needs the physical wrap link.
  `cluster_axis=1` = operate across the 16-extent axis of a (1,16) mesh.
- Reuse helpers from `../dual_t3k_ops/scripts/bigmesh_ops.py`: `local_coords_and_tensors`,
  `full_tensor_to_cpu`, `pcc`.

## Workload verification design (torch goldens)

- **all_reduce**: shard `(1,1,32,32*16)` on dim3 → 16 distinct `(1,1,32,32)` blocks;
  `all_reduce(cluster_axis=1)` → each chip holds `S = Σ_i block_i`; golden = torch sum of the 16
  chunks; verify each local chip == S.
- **reduce_scatter**: replicate a full `(1,1,32,32*16)` tensor → `reduce_scatter(dim=3,
  cluster_axis=1)` → sum across 16 (=16×full) then scatter dim3; chip i holds slice i of `16×full`;
  golden = `(16*full)[..., i*32:(i+1)*32]` per local chip.
- add/matmul/all_gather: keep the proven bf16 approach (report rel-L2 + PCC; all_gather = pure
  move → exact). all_reduce/reduce_scatter re-round in bf16 → expect ~0.999x, not exact.

## Execution (on hardware — reuse the dual_t3k_ops recipe)

Env: see `prompt.md` (mounted `TT_METAL_HOME`+`TT_METAL_RUNTIME_ROOT`, local `TT_METAL_CACHE`).

1. Preflight: `test_system_health` must PASS (3 tests).
2. Write `scripts/stack_workload.py`; run under tt-run with
   `tests/tt_metal/tt_fabric/custom_mesh_descriptors/dual_t3k_1x16_experimental_bigmesh_mgd.textproto`,
   `--hosts t3k-node-a,t3k-node-b`, `--tcp-interface ens18`, launching
   `python3 /home/namvu/dual-t3k/tt-metal/claude_job/tt_stack_walkthrough/scripts/stack_workload.py`.
   **Run in background + poll a log** (cold compile can exceed 10 min; a foreground timeout SIGTERM
   wedges the fabric — findings #13). Iterate to green; capture output.
3. **Fabric-manager live demo**: build `run_fabric_manager` if missing; run
   `run_fabric_manager --initialize-fabric --fabric-config FABRIC_2D --mesh-shape <shape>` under
   `mpirun-ulfm` on both hosts (writes `fabric_status.txt`, leaves fabric up) → run the workload
   with `ttnn.set_fabric_config(..., fabric_manager_mode=ttnn.FabricManagerMode.ENABLED)` (attaches
   to the pre-built fabric; expect log "Fabric initialized through Fabric Manager") →
   `run_fabric_manager --terminate-fabric`. Capture status file + attach log.
4. Write STACK.md from verified traces + captured evidence; update progress/findings.

## Cautions (reuse `../dual_t3k_ops/findings.md`; new ones here)

- FABRIC_2D (not 1D); both `TT_METAL_HOME`+`TT_METAL_RUNTIME_ROOT` mounted; `TT_METAL_CACHE` local;
  never `pkill` mid-init (wedge → `tt-smi -r` BOTH hosts concurrently, wait ~90 s for QSFP retrain,
  then `test_system_health`); background-run + poll, no foreground timeout wrapper.
- New (fabric-manager): `run_fabric_manager` forces slow-dispatch and **leaves fabric up on exit** —
  always `--terminate-fabric` after, else later default-mode runs collide. Wedge recovery = same
  `tt-smi -r` both-hosts procedure.

## Definition of done

- `stack_workload.py` prints rank-0 PASS for add, matmul, all_gather, all_reduce, reduce_scatter
  (PCC vs torch golden ≥ threshold), exit 0 under tt-run across both hosts — captured to a log.
- Fabric-manager demo: workload green under `fabric_manager_mode=ENABLED` against a fabric brought
  up by `run_fabric_manager`, with status file + attach log captured.
- STACK.md walks all 5 layers with correct file-path citations + a runtime trace, cross-links
  GUIDE.md/topology.html, embeds the captured PASS output. Optionally rendered as an Artifact.

## Notes

- Only new files under `claude_job/tt_stack_walkthrough/`; no repo source changes. Big-Mesh only.
