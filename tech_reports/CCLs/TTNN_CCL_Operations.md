# TTNN Collective Communication Operations (CCL)

A tech report for TT developers writing, using, debugging, and extending TTNN's multi-chip collective operations. Audience: new hires onboarding to TT hardware and RISC-V dataflow kernels, experienced op developers, and AI assistants consuming this as reference. Part 1 is for users; Parts 2–5 cover how CCLs work internally at the level of detail expected for hand-written kernel work; Parts 6–8 cover performance, testing (including simulation), and pitfalls.

API anchor: NCCL's [collectives guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html) — where NCCL has ranks over CUDA devices, TTNN has mesh coordinates over a `MeshDevice` joined by TT-Fabric. Architecture anchor: `tech_reports/TT-Fabric/TT-Fabric-Architecture.md`.

## Contents
- [1. User API](#1-user-api)
- [2. Hardware and fabric stack](#2-hardware-and-fabric-stack)
- [3. Anatomy of a CCL device kernel](#3-anatomy-of-a-ccl-device-kernel)
- [4. Worked example: all_gather end to end](#4-worked-example-all_gather-end-to-end)
- [5. Host side: the program factory](#5-host-side-the-program-factory)
- [6. Performance tuning](#6-performance-tuning)
- [7. Testing and simulation (craq-sim)](#7-testing-and-simulation-craq-sim)
- [8. Pitfalls reference](#8-pitfalls-reference)
- [9. File reference](#9-file-reference)

---

# 1. User API

## 1.1 Setup — the NCCL-communicator equivalent

```python
import ttnn

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)   # BEFORE open_mesh_device. T3K: FABRIC_1D_RING
mesh = ttnn.open_mesh_device(ttnn.MeshShape(2, 4))    # n300 (1,2) · T3K (2,4) · Galaxy (8,4)
# ... whole CCL phase on this one mesh ...
ttnn.close_mesh_device(mesh)
```

Opening a mesh with fabric enabled launches persistent fabric-router kernels on Ethernet cores and allocates fabric L1 regions on every chip. Hard rules, each validated the painful way:

- `set_fabric_config()` strictly before `open_mesh_device`; never switch configs on an open mesh — the next CCL hangs.
- Fabric context is per open-mesh. Open→close→open cycles end with `fabric_context_ != nullptr` failures; hold one mesh for the whole CCL phase.
- Any non-`DISABLED` fabric breaks `MeshShape(1,1)` at fabric-firmware init. Single-chip work: skip it.
- Never hold per-device `ttnn.open_device(k)` handles for chips inside an open mesh — UMD treats them as contested.

`FabricConfig` values: `DISABLED`, `FABRIC_1D`, `FABRIC_1D_NEIGHBOR_EXCHANGE`, `FABRIC_1D_RING`, `FABRIC_2D`, `FABRIC_2D_TORUS_X/Y/XY`, `CUSTOM`. Production drives even the physically-2D Galaxy with `FABRIC_1D` (independent row/column line fabrics); `FABRIC_2D` exists in the enum but is not the production-validated path.

Worker grids vary by harvesting — n300 exposes 7×8 = 56 worker cores per chip, Galaxy 6U units 8×9 = 72. Call `compute_with_storage_grid_size()` per chip; never hardcode.

Distributing tensors:

```python
tt  = ttnn.from_torch(t, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                      mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))   # or ReplicateTensorToMesh
out = ttnn.to_torch(tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
```

`cluster_axis=K` runs an independent collective along each row/column of the mesh; omitted = whole mesh.

## 1.2 Operation catalog

| TTNN op | Semantics | NCCL analog |
|---|---|---|
| `ttnn.all_gather(t, dim, *, cluster_axis, num_links, topology, memory_config)` | per-chip shards → concat along `dim` on every chip | AllGather |
| `ttnn.reduce_scatter(t, dim, *, cluster_axis, intermediate_memory_config, …)` | elementwise Sum, chip *i* keeps slice *i* | ReduceScatter |
| `ttnn.all_reduce(t, *, cluster_axis, num_links, topology, …)` | Sum across chips, full tensor everywhere | AllReduce |
| `ttnn.broadcast(t, sender_coord, cluster_axis, mesh)` | root chip → all | Broadcast |
| `ttnn.all_broadcast(t, …)` | every chip → list of all chips' tensors | — |
| `ttnn.point_to_point(t, sender_coord, receiver_coord, topology)` | one chip → one chip | Send/Recv |
| `ttnn.all_to_all_dispatch` / `all_to_all_combine` | MoE token routing to expert chips and back | AlltoAll |

Reduce / Gather / Scatter have no dedicated TTNN primitive — compose from the above.

**Reduction semantics: Sum only.** The experimental signature accepts `math_op: ReduceType` but drops it (`all_reduce_async.cpp` comments out the parameter). Min/max/mean: all_gather then local `ttnn.min/max/mean` along the gather dim — correct at `num_chips×` bandwidth. Negation does not rescue max; verify on every tt-metal bump.

**Layout.** TILE_LAYOUT + bf16/bf8_b is the validated path. ROW_MAJOR or a tile-padded gather dim silently switches stable all_gather/reduce_scatter to a composite (all-broadcast based) fallback — correct but much slower; check this before chasing perf. ROW_MAJOR + bf8_b unsupported. ROW_MAJOR/fp32 at Galaxy scale: §8.

**Experimental tier** (`ttnn.experimental.all_gather_async`, `all_reduce_async`, `reduce_scatter_minimal_async`, fused `*_matmul*` / `llama_*` / `deepseek_*`): faster, more knobs, caller-managed:

```python
sems = [ttnn.create_global_semaphore(mesh, worker_crs, 0) for _ in range(8)]
persistent_out = ttnn.from_torch(zeros, device=mesh, layout=TILE, mesh_mapper=...)
out = ttnn.experimental.all_reduce_async(x, buffer_tensor=persistent_out, cluster_axis=0,
        mesh_device=mesh, multi_device_global_semaphore=sems[i % 8],
        topology=ttnn.Topology.Linear, num_links=4)
ttnn.synchronize_device(mesh)        # before reading results
```

Ring ≥2 semaphores to pipeline successive steps. Production sequencing: `models/demos/llama3_70b_galaxy/tt/llama_ccl.py`.

---

# 2. Hardware and fabric stack

TT-Fabric extends the NoC over Ethernet: a worker issues a NoC-style command which is packetized, transported chip-to-chip, and **replayed on the destination chip's NoC**. Persistent router kernels own the Ethernet cores (Wormhole: 1 RISC-V per eth core runs the whole router; Blackhole: 2 share it); op kernels never run there.

**Layer 1/2 — TT-link (hardware).** 16 B header per Ethernet frame; tx/rx sequence numbers + Go-Back-N ARQ retransmit CRC-error frames invisibly. Contract: payload accepted ⇒ delivered exactly once. No retransmit logic belongs in op kernels.

**Layer 3 — TT-routing.** Address = `{MeshId, ChipId}` + destination NoC X,Y + offset; scales to 1024 meshes × 256 chips. Intra-mesh packets are **source-routed**: the worker writes the entire route into the header at injection — `fabric_set_unicast_route(hdr, num_hops)` ("deliver N hops down the line"), or multicast start+range. Inter-mesh traffic uses routing tables to exit nodes, route rewritten at each mesh entry. Tables are **dimension-ordered** (X then Y) to break cyclic deadlocks. Parallel links per direction (WH 4, BH 2) form independent **routing planes**, one fabric copy each — the API's `num_links` strides workers across planes.

**Layer 4 — transport.** Per router, one user VC of buffered channels. Line fabric: 2 sender channels (ch0 local workers, ch1 passthrough) + one 16-slot receiver; 2D: 4 senders. Your throughput shares the wire with neighbors' transit. One VC ⇒ **in-order delivery per direction** — "payloads then inc" needs no flush. Rings: bubble flow control (inject only if ≥2 slots free). Roadmap: TTL, reroute, status mailbox.

**Layer 5 — session.** Fire-and-forget writes; flow control is semaphore credits.

---

# 3. Anatomy of a CCL device kernel

A CCL op is ordinary Tensix kernels: same NCRISC/BRISC dataflow + TRISC compute, CBs, `get_arg_val`, `TensorAccessor`. Compute is unchanged; cross-chip behavior concentrates in dataflow kernels.

**Packet headers.** Allocate `volatile PACKET_HEADER_TYPE*` from `PacketHeaderPool::allocate_header()` (fabric L1, no host setup; CB-allocated fallback). Program route + NoC command (`to_noc_unicast_write`, scatter ≤4 chunks, `to_noc_unicast_atomic_inc`, fused, inline). State idiom: `*_set_state` once, `*_with_state{DstAddr}` per packet.

**Connection.** `FabricConnectionManager::build_from_args(idx)`; `open_finish()` **after** pre-waits; per packet `wait_for_empty_write_slot()` + `send_payload_flush_blocking_from_address()` (flush ⇒ reusable, not delivered); `close()`.

**Addressing.** Local `get_noc_addr` evaluates at destination — GlobalSemaphores/persistent buffers only; `safe_get_noc_addr(x,y,addr,0)` for explicit coords.

**Sync.** Remote inc + `wait_min(1 | ring−1 | target)`; reset before reuse — sender before own inc; receiver at end.

**Mux.** Shared sender; CT/RT block; status wait → connect → disconnect; worker 0 terminates. Coalescing/slice walks stay op-owned. Smallest ref: p2p writer (~116 lines).

# 4. Worked example: all_gather end to end

Reference: `ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/minimal_default_{writer,reader}.cpp`. Per link there are two unidirectional worker pairs (`direction` RT arg: 0 forward, 1 backward). Each pushes the local slice into the local output and along its direction, then relays neighbors' slices until everyone has all `ring_size` slices.

```
ring of 4:  chip0[A] chip1[B] chip2[C] chip3[D]
chip1 fwd:  B → chip2, then relay A → chip2 ...
chip1 bwd:  B → chip0, then relay C → chip0 ...
every output: [A B C D]   (placement by chip position, not arrival order)
```

## 4.1 Writer phases

1. **Topology as CT args.** `ring_size`, `my_chip_id`, `num_targets_forward/backward`, `topology`, `gather_dim`, tensor dims; line-unicast and barrier-mcast route blocks parsed `constexpr` (`get_line_*_route_info_from_args<idx>()`). Hop math compiles away; line endpoints become no-send branches (`valid_targets(direction)`).
2. **Connect.** Direct `FabricConnectionManager::build_from_args` + `open()`, or mux client (`build_connection_to_fabric_endpoint` → status wait → `fabric_client_connect`); `fabric_direction_connection` selects forward/backward EDM sender.
3. **Headers.** `PacketHeaderPool::allocate_header()` ×3 (scatter, unicast, inc); each `*_set_state` once; routes set once.
4. **Barrier.** Mcast atomic-inc to peers, then `wait_min(barrier_sem, ring_size−1)` + reset — bounds skew so no chip writes into a partner's still-bound output (program cache).
5. **Local slice.** `cb_wait_front`; output tile placement = chip position × slice volume on `gather_dim`; scatter packets (≤4 tiles) over fabric + plain local NoC writes; one cumulative inc per `chunks_per_sync`; `noc_async_writes_flushed()` paces CB.
6. **Relays.** Slice ids `my_chip_id ± (k+1)` mod ring; same packet path; even rings split last slice fwd/bwd half each.
7. **Teardown.** Write+atomic barriers; mux disconnect; worker 0 collects sems then terminates mux.

## 4.2 Reader phases

Fabric-free. (1) Local input → CB, full-packet reserves. (2) Relays: cumulative `wait_min(sem, ++target)`, re-read landed tiles from **output** → CB. (3) Pure-receive slices: waits only. Reset at end. Inc implies payload (ordered VC). Pacing mirrors writer or CB-deadlock.

# 5. Host side: the program factory

Same vocabulary as single-chip, run per chip with mesh context: ring index from coord; route CT blocks from neighbor coords (`get_forward_backward_line_unicast/mcast_configuration`); cores = links × 2 dirs × (mux + N workers); CB = 3 packets, `cb_page = tensor page`; `packet = get_tt_fabric_channel_buffer_size_bytes()` (must ≥ page, else FATAL), tiles/pkt = min(4, packet/page); RT = addresses, sem ids, range, `chunks_per_sync`, conn-append per direction; GlobalSemaphores + persistent outputs caller-passed (program-cache reuse).

# 6. Performance tuning

| Knob | Action |
|---|---|
| `num_links` | stride workers across routing planes (×4 WH, ×2 BH) |
| tiles/packet | fill to scatter cap 4; small pages → coalesce |
| `chunks_per_sync` | sync rarely; per-packet incs throttle |
| topology | ring halves worst hops; split-fwd halves last slice |
| pipelining | ~10 µs hop ≫ NoC; cumulative sems hide latency |

Sender ch1 shares the wire with passthrough — long lines self-throttle.

# 7. Testing and simulation (craq-sim)

Hardware: `tests/ttnn/unit_tests/operations/ccl/`, nightly T3K. `TT_METAL_DPRINT_CHIPS` (DPRINT defaults chip 0), `_ETH_CORES` for routers; Watcher catches bad cross-chip addrs; 6U reset `-glx_reset`; stale sems survive exit.
Simulation: BH-only craq-sim fork (forked umd+metal); one process; `TT_METAL_SIMULATOR`, `TT_METAL_MOCK_CLUSTER_DESC_PATH`, `TT_MESH_GRAPH_DESC_PATH`; descriptors `tt_metal/fabric/mesh_graph_descriptors/` + mock cluster yamls; plain pytest + `timeout` (`run_safe_pytest` forces slow dispatch); p2p ≈23 s, payload-bound; sweeps = descriptor pairs; never perf.

# 8. Pitfalls

Sum-only ignored `math_op` · Galaxy ROW_MAJOR/fp32 garbage (TILE+bf16) · reshape/concat L1 OOM (chunk) · composite fallback · reset before reuse · fabric before open · contested handles · sim slow-dispatch trap.

# 9. Files

p2p `device/kernels/dataflow/` · AG `minimal_default_*` + default factory · `fabric/hw/inc/{linear/api.h,packet_header_pool.h,edm_fabric/fabric_connection_manager.hpp}` · routing/sync utils `ccl/kernel_common/` · `llama_ccl.py` · mesh/cluster descriptors.
