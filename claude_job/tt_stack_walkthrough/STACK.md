# The Tenstorrent software stack, end-to-end — proven on a dual T3K

**What this is.** A layer-by-layer walkthrough of how the Tenstorrent layers cooperate to run one
program across **two cabled T3Ks as a single 1×16 MeshDevice** — from the per-chip topology flashed
on the cards, through the fabric control plane and its lifecycle manager, through the multi-host
launcher, up to a ttnn app doing compute + collectives. Every layer is cited to source, and the whole
stack is **proven** by one workload (`scripts/stack_workload.py`) that runs `add`, `matmul`,
`all_gather`, `all_reduce`, and `reduce_scatter` across all 16 chips and checks each against a torch
golden.

- **How to run it** (env, launch commands, troubleshooting): [`../dual_t3k_ops/GUIDE.md`](../dual_t3k_ops/GUIDE.md)
- **What the wiring physically looks like**: [`../dual_t3k_ops/topology.html`](../dual_t3k_ops/topology.html) (*Dual-T3K 1×16 Topology*)
- Verified 2026-07-15 on the mounted framework `/home/namvu/dual-t3k/tt-metal`, commit `3632044cfd6`,
  launcher `t3k-node-b` + remote `t3k-node-a`, NIC `ens18`.

---

## 0. TL;DR — the proof

`scripts/stack_workload.py`, launched once via `tt-run` across both hosts, opens a **single logical
1×16 MeshDevice** (8 chips per host) and verifies five operations per local shard against a torch
golden. Full capture: [`scripts/PASS_output.txt`](scripts/PASS_output.txt).

```
[rank 0] opened Big-Mesh (1, 16) = 16 chips across 2 host(s)
==============================================================================
[rank 0] PASS ADD           8 local shards, each(1, 1, 32, 32): min_pcc=0.999998 (>= 0.990000) at chip 5
[rank 0] PASS MATMUL        8 local shards, each(1, 1, 32, 128): min_pcc=0.999979 (>= 0.990000) at chip 6
[rank 0] PASS ALL_GATHER    8 local shards, each(1, 1, 32, 512): min_pcc=1.000000 (>= 0.999999) at chip 0
[rank 0] PASS ALL_GATHER(full) full(1, 1, 32, 512): pcc=1.000000 (>= 0.999999)  rel_L2_err=0.0000%  max_abs=0.000e+00
[rank 0] PASS ALL_REDUCE    8 local shards, each(1, 1, 32, 32): min_pcc=0.999982 (>= 0.990000) at chip 0
[rank 0] PASS REDUCE_SCATTER 8 local shards, each(1, 1, 32, 32): min_pcc=0.999974 (>= 0.990000) at chip 0
==============================================================================
[rank 0] ALL PASS: add, matmul, all_gather, all_reduce, reduce_scatter verified vs torch golden across all 16 chips
==============================================================================
```

`all_gather` is a pure data move, so it matches the golden **exactly** (pcc = 1.0). `add`, `matmul`,
`all_reduce`, and `reduce_scatter` re-round in bf16 (or sum in a different order than the CPU), so they
land at ~0.9999 — the check is about the multi-device **wiring**, not float precision. Every rank
verifies its own 8 local shards and raises on failure, so a clean rank-0 `ALL PASS` + exit 0 means all
16 chips agreed with torch.

---

## 1. The five layers, and who does what

```
 tt-topology   flash per-chip EthCoords onto the cards        (external pip tool; UMD reads it back)
      │  hands off: a ClusterDescriptor UMD can read from NODE_INFO
      ▼
 tt-fabric     ControlPlane: MeshGraph + physical discovery →  (tt_metal/fabric/)
      │        TopologyMapper (logical↔physical) → RoutingTableGenerator → routing tables on every chip
      │  hands off: routing tables + live EDM ethernet routers
      ▼
 TTFM          the fabric LIFECYCLE: who calls init / teardown  (FabricManagerMode + run_fabric_manager)
      │  hands off: a fabric that is UP (or a promise that someone owns it)
      ▼
 tt-run        rank → {mesh_id, mesh_host_rank}; spawn the app  (ttnn/ttnn/distributed/ttrun.py + mpirun-ulfm)
      │  hands off: N processes, each with TT_MESH_ID / TT_MESH_HOST_RANK / MGD path set
      ▼
 ttnn app      set_fabric_config → open_mesh_device → tensors → ops → CCL over the live fabric
```

Each layer is a hand-off: it produces exactly the artifact the next layer consumes. The rest of this
doc walks each one with source citations, then traces the actual runtime init sequence, then shows the
TTFM lifecycle live-demo.

---

### Layer 1 — tt-topology (per-chip identity on the silicon)

**External pip tool, not in this repo.** `tt-topology -l mesh` flashes each Wormhole card's `NODE_INFO`
with its `EthCoord {x, y, rack, shelf}` plus the mesh eth-routing, and this **persists across reboot**.
It is run once per physical layout change, not per job.

- UMD reads the flashed coords back during discovery:
  `tt_metal/third_party/umd/device/topology/topology_discovery_wormhole.cpp:98` (`get_local_eth_coord`),
  assembling a `ClusterDescriptor` that every layer above treats as ground truth.
- If the flash is stale (cards physically re-cabled but not re-flashed), you get an explicit
  "re-run tt-topology" style failure: `tt_metal/llrt/tunnels_from_mmio_device.cpp:82`.
- **How we confirm this layer is healthy:** `build_Release/test/tt_metal/tt_fabric/test_system_health`
  → `[ PASSED ] 3 tests.` (it prints every eth channel: which are QSFP inter-host, WARP, or internal
  trace, and whether the link is UP). This is the mandatory preflight before every run. See also the
  physical picture in [`topology.html`](../dual_t3k_ops/topology.html).

Hand-off ➜ a `ClusterDescriptor`: "here are the chips, here are their coordinates, here is which eth
channel connects to whom."

---

### Layer 2 — tt-fabric control plane (logical mesh ⇄ physical chips ⇄ routing tables)

Source root: **`tt_metal/fabric/`**. This layer turns "a pile of discovered chips" into "a routed mesh
that matches the logical shape you asked for."

1. **`MeshGraphDescriptor`** parses the `.textproto` MGD you pass. For the Big-Mesh that is
   `tests/tt_metal/tt_fabric/custom_mesh_descriptors/dual_t3k_1x16_experimental_bigmesh_mgd.textproto`:
   ```
   mesh_descriptors { name: "M0" arch: WORMHOLE_B0
     device_topology { dims: [ 1, 16 ] }     # one logical 1×16 mesh...
     host_topology   { dims: [ 1, 2 ] } }    # ...split across 2 hosts (8 chips each)
   top_level_instance { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
   ```
   The `host_topology 1×2` is the crux of Big-Mesh: **one** `mesh_id=0` owned jointly by host_rank 0
   and host_rank 1.
2. **`run_physical_system_discovery`** (`physical_system_discovery.cpp:757`): each host discovers its
   local chips, results are MPI-gathered, and rank 0 runs `generate_cross_host_connections` to stitch
   the two halves over the inter-host QSFP links.
3. **`TopologyMapper`** solves the logical↔physical assignment (a SAT-style solve;
   `topology_solver*.cpp`). This is why the fabric node IDs are a permutation of chip IDs, not identity
   — e.g. in the live demo below, physical chip 4 maps to fabric node `(M0, D0)`.
4. **`RoutingTableGenerator`** builds intra- and inter-mesh routing tables, `FabricContext` /
   `FabricBuilder` compile them, and they are written to **every** chip:
   `ControlPlane::write_routing_tables_to_all_chips()` (`control_plane.cpp:2068`), then the per-chip EDM
   ethernet routers are built and launched (`fabric_init.cpp`).

**Why FABRIC_2D and not FABRIC_1D:** a 1×16 line spanning two hosts cannot be routed by `FABRIC_1D`
(you get `Could not find any forwarding direction from src (M0,D3) to dst (M0,D0)`); `FABRIC_2D` routes
it. This is gotcha #2 in [`../dual_t3k_ops/findings.md`](../dual_t3k_ops/findings.md).

Hand-off ➜ routing tables on every chip + live EDM routers on the eth cores. The mesh is now
addressable end-to-end.

---

### Layer 3 — TTFM: the fabric lifecycle (who calls init and teardown)

The control plane in Layer 2 *can* build the fabric; **TTFM decides who owns that build and when it
gets torn down.** The knob is the `FabricManagerMode` enum
(`tt_metal/api/tt-metalium/experimental/fabric/fabric_types.hpp:44`), a bit-flag:

| Mode | Flags | Meaning |
|---|---|---|
| `DEFAULT` | `INIT_FABRIC \| TERMINATE_FABRIC` | **Metal owns it** — inits on `open_mesh_device`, tears down on close. |
| `ENABLED` | *(neither flag)* | "Someone else already brought the fabric up; just attach and use it." |
| `INIT_FABRIC` | init only | Bring the fabric up and **leave it up**. |
| `TERMINATE_FABRIC` | terminate only | Tear a running fabric down. |

The gating lives in one file, `tt_metal/impl/device/firmware/fabric_firmware_initializer.cpp`:

- `init()` — if `INIT_FABRIC`: log **"Initializing Fabric"** (`:296`), write routing tables, compile,
  log **"Fabric Initialized with config …"** (`:313`). Else (ENABLED): log **"Fabric initialized
  through Fabric Manager"** (`:320`) and do nothing else.
- `teardown()` — **only** tears down the on-device routers if the `TERMINATE_FABRIC` flag is set
  (`:347`); otherwise it just drops local handles and returns. This is what makes ENABLED safe: closing
  the mesh does **not** kill a fabric you don't own.

Our proof workload (`stack_workload.py`) uses **DEFAULT** — you can see both `"Fabric Initialized with
config FabricConfig::FABRIC_2D"` (init) and the clean `"Cluster destructor completed"` (teardown) in
[`scripts/PASS_output.txt`](scripts/PASS_output.txt). Metal owned the whole lifecycle.

**The split lifecycle (INIT_FABRIC → ENABLED → TERMINATE_FABRIC)** is exposed as an in-repo CLI,
`tools/scaleout/fabric_manager/run_fabric_manager.cpp` (built to
`build_Release/tools/scaleout/run_fabric_manager`), which calls `configure_fabric_routing` in
`tools/scaleout/fabric_manager/utils/fabric_manager_utils.cpp:44`. It forces slow-dispatch and writes a
`fabric_status.txt`. **We live-demoed it — see §4.**

> There is also an *external* Docker `tt-fabric-manager` (a read-only inspector); that one is covered in
> [`GUIDE.md` §8](../dual_t3k_ops/GUIDE.md).

Hand-off ➜ a fabric that is up, and a documented contract for who is responsible for tearing it down.

---

### Layer 4 — tt-run + mpirun-ulfm (turn one command into N cooperating processes)

`ttnn/ttnn/distributed/ttrun.py` is the multi-host launcher. It runs in two phases:

- **Phase 1 — `generate_rank_bindings`**: from the MGD + `--hosts`, compute each MPI rank's
  `{mesh_id, mesh_host_rank}` and cache it under `generated/ttrun/<hash>/`. For our Big-Mesh:
  ```yaml
  rank 0 → mesh_id 0, mesh_host_rank 0   (t3k-node-b, the launcher)
  rank 1 → mesh_id 0, mesh_host_rank 1   (t3k-node-a, the remote)
  mesh_graph_desc_path: …/dual_t3k_1x16_experimental_bigmesh_mgd.textproto
  ```
- **Phase 2** spawns the workload on both hosts via **`mpirun-ulfm`** (OpenMPI 5.0.7 ULFM in
  `/usr/local/bin` — *not* `/usr/bin/mpirun` 4.1.2), forwarding every `TT_*`/`ARCH_*` env var with
  `-x`, and **setting `TT_MESH_HOST_RANK` per rank** (`ttrun.py:1530`). That env var is exactly what the
  control plane reads to decide which half of the mesh this process owns
  (`control_plane.cpp:275`, `initialize_local_mesh_binding`).

This per-rank `TT_MESH_HOST_RANK` is the whole ballgame for multi-host: rank 0 binds host_rank 0, rank 1
binds host_rank 1, and together they form the single `mesh_id=0`. (It also forwards
`TT_METAL_HOME`/`TT_METAL_RUNTIME_ROOT` from the launcher, overriding the remote's stale `.bashrc` —
gotcha #10 in findings.)

Hand-off ➜ two processes, same script, each knowing "I am host_rank K of mesh 0, my 8 chips are …".

---

### Layer 5 — the ttnn app (open the mesh, place tensors, run ops + collectives)

This is `scripts/stack_workload.py`. The sequence (and its source spine):

1. **`ttnn.set_fabric_config(FABRIC_2D)`** *before* opening the mesh →
   `SetFabricConfig` (`tt_metal/fabric/fabric.cpp:501`) → `MetalEnvImpl::initialize_fabric_config`
   (`tt_metal/impl/context/metal_env.cpp:345`). This is where Layers 2–3 actually fire: control plane
   built, routing tables written, fabric mode chosen.
2. **`ttnn.open_mesh_device(MeshShape(1,16))`** (`ttnn/ttnn/distributed/distributed.py:644`). Each rank
   builds a **local** view of the mesh: it owns only its 8 chips (`get_view().is_local(coord)`), the
   ranks barrier, and `initialize_fabric_and_dispatch_fw` brings the routers + dispatch online.
3. **Tensor placement** via mesh mappers: `ShardTensorToMesh(dim=…)` splits a tensor across chips;
   `ReplicateTensorToMesh` copies it to all chips.
4. **Ops.** `add`/`matmul` are per-chip compute. The three collectives move data over the live fabric:
   - `ttnn.all_gather(t, dim, cluster_axis=1, topology=Linear)` — concatenate shards onto every chip.
   - `ttnn.all_reduce(t, cluster_axis=1, topology=Linear)` — **Sum** across chips (no `dim`).
   - `ttnn.reduce_scatter(t, dim, cluster_axis=1, topology=Linear)` — sum across chips, then scatter.
   `cluster_axis=1` = operate along the 16-extent axis of the (1,16) mesh; `Topology.Linear` = a line
   (no wrap link, so no Ring). Under the hood the CCL op wires its worker kernels to the live EDM
   routers via `append_fabric_connection_rt_args` (`ttnn/cpp/ttnn/operations/ccl/common/host/ccl_worker_builder.cpp:1139`),
   which talk to `ccl/kernels/edm/erisc_datamover.cpp`.

**How each op is verified** (torch goldens, all per-**local**-shard because no host-side float gather
exists across hosts — gotcha #6):
| Op | Placement | Golden | Expected |
|---|---|---|---|
| add | shard dim3 | `a+b` per column block | ~1.0 |
| matmul | rows shard dim2, B replicated | `A@B` per row block (fp32 accum → bf16) | ~1.0 |
| all_gather | shard dim3 → gather | full tensor on every chip | **exact 1.0** |
| all_reduce | shard dim3 (16 blocks) | `Σ blocks` on every chip | ~0.9999 |
| reduce_scatter | replicate full | `(16×full)[…, i·32:(i+1)·32]` on chip i | ~0.9999 |

> **Also learned this session (new gotcha):** on 16 chips you must **`ttnn.deallocate()` intermediate
> tensors between ops** and barrier the ranks. A first attempt that let tensors accumulate crashed the
> remote rank with `SIGBUS "Non-existent physical address"` inside `fetch_queue_write` (the host→device
> command-queue ring) partway through the run. Freeing each op's tensors before the next op fixed it.
> A hard rank death like this leaves the fabric wedged; recovery is the `tt-smi -r` procedure below.

Hand-off ➜ results on the chips, read back per local shard and checked against torch. **Stack proven.**

---

### tt-operator (out of scope, one paragraph)

There is no in-repo `tt-operator` source — it exists only as CI GitHub Actions (`.github/actions/ttop-*`)
that call an **external Kubernetes allocation service** to lease Tenstorrent hardware for a job. On a
directly-owned dual T3K there is nothing to run; `tt-run` + `mpirun-ulfm` (Layer 4) is the launcher.

---

## 2. Runtime init trace (what actually happens, in order)

Observed in [`scripts/PASS_output.txt`](scripts/PASS_output.txt) / `scripts/run.log`:

```
tt-run Phase 1: cache hit → rank_bindings.yaml  (rank0=host_rank0 t3k-node-b, rank1=host_rank1 t3k-node-a)
tt-run Phase 2: mpirun-ulfm spawns python3 stack_workload.py on both hosts, -x TT_* forwarded
  ├─ set_fabric_config(FABRIC_2D)                          fabric.cpp:501 → metal_env.cpp:345
  │    └─ ControlPlane: parse MGD → physical discovery → MPI-gather → cross-host stitch (rank 0)
  │       → TopologyMapper → RoutingTableGenerator
  ├─ open_mesh_device(1×16)                                distributed.py:644 → mesh_device.cpp
  │    ├─ each rank builds its local 8-chip view (is_local)
  │    ├─ "Initializing Fabric"                            fabric_firmware_initializer.cpp:296
  │    ├─ write_routing_tables_to_all_chips                control_plane.cpp:2068  (called at :311)
  │    ├─ "Fabric initialized on Device 0..7" ×2 hosts     device.cpp:548
  │    └─ "Fabric Initialized with config FabricConfig::FABRIC_2D"   fabric_firmware_initializer.cpp:313
  ├─ [rank 0] opened Big-Mesh (1, 16) = 16 chips across 2 host(s)
  ├─ ADD / MATMUL / ALL_GATHER / ALL_REDUCE / REDUCE_SCATTER  (deallocate + barrier between each)
  ├─ ALL PASS
  └─ close_mesh_device → "Cluster destructor completed" ×2 hosts   (DEFAULT-mode teardown)
```

---

## 3. TTFM live demo — `run_fabric_manager` split lifecycle (with a real finding)

Full transcript + logs: [`scripts/fm_demo/TRANSCRIPT.md`](scripts/fm_demo/TRANSCRIPT.md),
[`scripts/fm_demo/fabric_status.txt`](scripts/fm_demo/fabric_status.txt), and the three `0*.log` files.
Run on a **single T3K** (`t3k-node-b`, a 2×4 mesh), because `run_fabric_manager` hard-sets
`TT_MESH_HOST_RANK=0` for every process (`run_fabric_manager.cpp:107`, `set_config_vars`) — it is a
**single-host tool** and cannot bind the two different host_ranks a Big-Mesh needs.

**Phase 1 — INIT (`--initialize-fabric --fabric-config FABRIC_2D --mesh-shape 2x4`) → SUCCESS.**
Metal takes the `INIT_FABRIC` path, compiles + launches the EDM routers on all 8 chips, prints the
fabric node IDs, writes `fabric_status.txt`, and **leaves the fabric up**:
```
Slow dispatch mode: Using full logical grid (8, 8)              (run_fabric_manager forces slow dispatch)
Initializing Fabric                                             fabric_firmware_initializer.cpp:296
Fabric initialized on Device 0..7 → on 8 devices
Fabric Initialized with config FabricConfig::FABRIC_2D          fabric_firmware_initializer.cpp:313
Fabric Node IDs:  Chip 0→(M0,D2)  Chip 4→(M0,D0)  …             (TopologyMapper permutation, §Layer 2)
✓ Fabric status written to: …/fm_demo/fabric_status.txt
```
This demonstrates the **INIT half** of the split lifecycle and the whole `FabricManagerMode` mechanism
on real hardware.

**Phase 2 — ATTACH (`set_fabric_config(FABRIC_2D, RELAXED_INIT, fabric_manager_mode=ENABLED)` +
`open_mesh_device`) → FAILED on T3K, and here's why (the finding):**
```
RuntimeError: Timed out waiting for ETH heartbeat on device ASIC ID: … ETH core e1-6 … Stuck at 0xabcd7841
  tt::umd::TopologyDiscovery::eth_heartbeat_running → discover_remote_devices → Cluster ctor → SetFabricConfig
```
A T3K's chips 4–7 are **remote halves of the n300 cards, reached over ethernet**. When a fresh process
opens the device, UMD topology discovery probes those remote chips for an eth *heartbeat* — but the EDM
fabric routers the manager left running occupy those eth cores, so the heartbeat never advances and the
device won't even open. The ENABLED "Fabric initialized through Fabric Manager" path is never reached.

**Phase 3 — TERMINATE (`--terminate-fabric`) → FAILED the same way.** The teardown tool *also* opens the
device first, so it hits the identical eth-heartbeat wall (unhandled `UmdException` → `std::terminate`).
Via the CLI, on a T3K the FM fabric can be **neither re-attached nor torn down** once it is up — recovery
is a chip reset.

**Why this is the expected outcome, not a mistake:** the in-repo fabric-manager CCL tests
(`tests/scale_out/test_ccl_fabric_manager.py`) are **8×4 Galaxy-only** and still gated behind
`# TODO: Enable these tests once Fabric Manager is ready` (`tests/pipeline_reorg/galaxy_sanity_tests.yaml`).
On a Galaxy every chip is directly (PCIe) accessible, so there is no remote-over-ethernet discovery to
conflict with a running fabric. On a T3K there is.

### The fix — drive the split lifecycle in one process

The eth-heartbeat wall only bites a **fresh process** (a new `Cluster` → re-discovery). Driving the *same*
`FabricManagerMode` values in **one process** (`scripts/fm_lifecycle.py`) builds the UMD `Cluster` exactly
once — before any fabric is up — and reuses it across every mesh open/close, so re-discovery never happens.
Evidence: [`scripts/fm_demo/04_TRANSCRIPT_inprocess.md`](scripts/fm_demo/04_TRANSCRIPT_inprocess.md) +
`04_lifecycle.log`.

```
=== PHASE INIT (INIT_FABRIC) ===
"Initializing Fabric" → "Fabric Initialized with config FabricConfig::FABRIC_2D"   (fabric up, LEFT UP)
[INIT] mesh open (2, 4) = 8 chips → [INIT] mesh closed
=== PHASE ENABLED ===
"Fabric config changed FABRIC_2D→FABRIC_2D, reinitializing control plane"          (reuses the Cluster)
"Fabric initialized through Fabric Manager"   fabric_firmware_initializer.cpp:320   ← ATTACH SUCCEEDS
[ENABLED] mesh open (2, 4) = 8 chips
```
The **ENABLED attach now succeeds on the T3K** — the exact `:320` line the separate-process CLI could
never reach. Topology discovery appears only once (in INIT, before the fabric is up); the ENABLED phase
reuses the existing cluster, so the eth-heartbeat probe never runs.

**Remaining boundary (honest):** a workload *dispatched* under the in-process ENABLED reattach **hangs on
its first device write** on a T3K. On a T3K, dispatch to the remote chips (4–7) tunnels over the same
ethernet that the ENABLED-mode control-plane reconfigure (`configure_ethernet_cores_for_fabric_routers`,
run while the routers are live) disturbs. So a *fully-green* workload under ENABLED is still not achievable
on this T3K/commit — consistent with the CI gating FM to Galaxy.

**Net:** the fabric-manager *mechanism* and the **ENABLED attach** are proven on real hardware (Phase 1
INIT via the CLI + the in-process attach above); the fully-green fabric proof is the **DEFAULT-mode**
16-chip workload in §0 (DEFAULT = INIT | TERMINATE — the same lifecycle, Metal owning both ends, no
mid-life reattach).

---

## 4. Reproduce / recover

**Run the proof** (env in [`prompt.md`](prompt.md); details in [`GUIDE.md`](../dual_t3k_ops/GUIDE.md)):
```bash
cd /home/namvu/dual-t3k/tt-metal && source python_env/bin/activate
export ARCH_NAME=wormhole_b0 TT_METAL_HOME=$PWD TT_METAL_RUNTIME_ROOT=$PWD TT_METAL_CACHE=/home/namvu/.cache/tt_metal_local
./build_Release/test/tt_metal/tt_fabric/test_system_health          # preflight: expect "[ PASSED ] 3 tests."
tt-run --tcp-interface ens18 \
  --mesh-graph-descriptor tests/tt_metal/tt_fabric/custom_mesh_descriptors/dual_t3k_1x16_experimental_bigmesh_mgd.textproto \
  --hosts t3k-node-a,t3k-node-b \
  python3 $PWD/claude_job/tt_stack_walkthrough/scripts/stack_workload.py
```
Run it **backgrounded + poll a log** — a cold kernel compile on 16 chips can exceed 10 min, and a
foreground timeout SIGTERM wedges the fabric exactly like a hard kill (findings #13).

**If the fabric wedges** (a hard rank crash, or the fabric-manager demo above): reset **both hosts
concurrently**, wait ~90 s for the QSFP links to retrain, then re-check health:
```bash
( ssh t3k-node-a 'tt-smi -r' & ); tt-smi -r; wait          # both at once — sequential resets mistime the retrain
sleep 90 && ./build_Release/test/tt_metal/tt_fabric/test_system_health   # until "[ PASSED ] 3 tests." on both hosts
```

---

## 5. Cross-links

- [`../dual_t3k_ops/GUIDE.md`](../dual_t3k_ops/GUIDE.md) — how to run (env, both launch modes, troubleshooting, §8 fabric-manager inspector).
- [`../dual_t3k_ops/topology.html`](../dual_t3k_ops/topology.html) — the physical 1×16 wiring.
- [`../dual_t3k_ops/findings.md`](../dual_t3k_ops/findings.md) — the 13 hardware gotchas this all rests on.
- [`PLAN.md`](PLAN.md) — the verified layer map + file citations behind this doc.
- [`findings.md`](findings.md) — new facts from this job (deallocate-between-ops; fabric-manager T3K limitation).
