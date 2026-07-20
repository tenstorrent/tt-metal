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
golden — reporting **PCC, MSE, atol (max absolute error), rtol (max relative error)** and a bonus
`torch.allclose`. Full capture: [`scripts/PASS_output.txt`](scripts/PASS_output.txt).

```
[rank 0] opened Big-Mesh (1, 16) = 16 chips across 2 host(s)
[rank 0] PASS ADD            min_pcc=0.999998  mse=9.901e-06  atol(max_abs)=3.125e-02  rtol(max_rel)=7.812e-03  allclose(rtol=0.02,atol=0.02)=True
[rank 0] PASS MATMUL         min_pcc=0.999979  mse=5.197e-03  atol(max_abs)=6.250e-01  rtol(max_rel)=5.933e+02  allclose(rtol=0.02,atol=0.02)=False
[rank 0] PASS ALL_GATHER     min_pcc=1.000000  mse=0.000e+00  atol(max_abs)=0.000e+00  rtol(max_rel)=0.000e+00  allclose(rtol=0.02,atol=0.02)=True
[rank 0] PASS ALL_REDUCE     min_pcc=0.999982  mse=9.532e-04  atol(max_abs)=1.250e-01  rtol(max_rel)=2.300e+01  allclose(rtol=0.02,atol=0.02)=False
[rank 0] PASS REDUCE_SCATTER min_pcc=0.999974  mse=7.037e-03  atol(max_abs)=7.500e-01  rtol(max_rel)=1.732e-02  allclose(rtol=0.02,atol=0.02)=True
[rank 0] ALL PASS: add, matmul, all_gather, all_reduce, reduce_scatter verified vs torch golden across all 16 chips
```
<small>Captured 2026-07-19 under `moreh-lock`, exit 0, clean teardown. Full: [`scripts/PASS_output.txt`](scripts/PASS_output.txt). (Shapes/thresholds trimmed above for width.)</small>

`all_gather` is a pure data move, so it matches the golden **exactly** (pcc = 1.0). `add`, `matmul`,
`all_reduce`, and `reduce_scatter` re-round in bf16 (or sum in a different order than the CPU), so they
land at ~0.9999 — the check is about the multi-device **wiring**, not float precision. Every rank
verifies its own 8 local shards and raises on failure, so a clean rank-0 `ALL PASS` + exit 0 means all
16 chips agreed with torch.

**The four metrics (per op), and how to read them.** `min_pcc` is the worst single shard; `mse`/`atol`/
`rtol` are pooled over all 8 local shards (`error_metrics()` in `scripts/stack_workload.py`):
- **`mse`** — mean squared error `mean((out − golden)²)`. Tiny across the board (≤ 7e-03).
- **`atol`** — max **absolute** error `max|out − golden|`. Scales with the op's output magnitude: ~0.03 for
  `add`, but ~0.6 for `matmul` (each output sums 128 products → bigger numbers → bigger bf16 rounding).
- **`rtol`** — max **relative** error `max(|out − golden| / |golden|)`. **Can look alarming (matmul 593×,
  all_reduce 23×) — that's an artifact, not a failure:** where a golden element is ~0, dividing a tiny
  bf16 rounding error by it explodes. It says nothing about correctness.
- **`allclose`** — `torch.allclose(rtol=2e-2, atol=2e-2)` (overridable via `STACK_RTOL`/`STACK_ATOL`).
  `False` for `matmul`/`all_reduce` because a *fixed* 0.02 tolerance can't cover a 0.6-magnitude output or a
  near-zero element — again scale, not wiring.

**Takeaway:** **PCC (≥ 0.9999 everywhere) is the robust wiring check** — scale-invariant, immune to the
near-zero blow-up. `mse`/`atol` are useful magnitudes; `rtol`/`allclose` are scale-sensitive and expected
to look large/False on big-magnitude or zero-containing outputs even when the result is correct.

---

## 1. The five layers, and who does what

```
── LAUNCHER (what you actually type) ──────────────────────────────────────────────────
 tt-run --mesh-graph-descriptor <MGD> --hosts a,b  python3 stack_workload.py
   → starts your program on BOTH hosts + hands each its identity (mesh_id, mesh_host_rank),
     then WAITS. tt-run touches NO fabric and NO chips. It is NOT a rung in the stack below —
     it is the launcher that starts the process the stack runs *inside*.

── THE STACK (runs INSIDE each launched process; the ttnn app calls DOWN) ──────────────
 ttnn app     your script — the code that actually does the work           (top / most abstract)
    │         set_fabric_config → open_mesh_device → tensors → ops → CCL
    ▼  calls down: each call triggers the layer below, from inside the app
 TTFM         the fabric LIFECYCLE — who calls init / teardown             (FabricManagerMode, run_fabric_manager)
    ▼
 tt-fabric    ControlPlane → routing tables on every chip + live EDM routers  (tt_metal/fabric/)
    ▼
 tt-topology  per-chip identity (EthCoords) → ClusterDescriptor UMD reads back  (external tool; UMD reads it)
    ▼
 the chips    16 Wormhole chips · 2 hosts · QSFP cables                    (bottom / the metal)
```

Read it two ways — both are true:
- **Down (what happens at runtime):** the ttnn app calls `set_fabric_config` (→ tt-fabric builds routing),
  `open_mesh_device` (→ TTFM brings the fabric up), then ops (→ over the live fabric). *All of this happens
  inside the process tt-run launched — none of it before.*
- **Up (the dependency / hand-off order):** chips → topology → fabric → TTFM → ttnn, each producing exactly
  what the one above consumes.

Either way, **tt-run is the launcher, drawn outside the stack — not a layer the app is built on.** You type
it first because it's the *starter*, the way you `ssh a-machine 'python app.py'`: `ssh` runs first, but
`app.py` is the program. The rest of this doc walks each piece with source citations (we still cover tt-run
below as "Layer 4" — it's part of *getting a workload running*, just not a rung the app stands on), then
traces the actual runtime init sequence, then shows the TTFM lifecycle live-demo.

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

### Layer 2 — tt-fabric control plane (turns a pile of chips into a routed mesh)

Source root: **`tt_metal/fabric/`**.

> **One-sentence goal:** for *every* chip, fill in a small table that says *"to reach chip X, send in
> direction D,"* then write that table onto the chips and launch the routers that use it. Everything below
> is the machinery that produces that table. (The MGD it reads — Big-Mesh vs Multi-Mesh — is in Layer 4.)

#### The vocabulary (what · where it's defined · what it looks like · example)

**`RoutingDirection`** — a compass direction out of a chip.
- *Where:* `mesh_graph.hpp:46` · *Looks like:*
  ```cpp
  enum class RoutingDirection { N=0, E=1, S=2, W=3, Z=4, C=5 /*here*/, NONE=6 /*unreachable*/ };
  ```
- `C` = destination **is** this chip (deliver locally); `NONE` = no route; `Z` = a 3rd, out-of-plane axis
  that is **Blackhole-only** (`num_z_ports = 0` on Wormhole, `mesh_graph.cpp:935`) → **`Z` never appears on
  your T3Ks; routing uses only N/E/S/W + C.**

**`FabricNodeId`** — a chip's *address in the fabric*: `(mesh_id, chip_id)`.
- *Where:* `fabric_types.hpp:104` · *Looks like:* `class FabricNodeId { MeshId mesh_id; uint32_t chip_id; };`
- *Example (real, from the fabric-manager log):* `Chip ID: 0 → (M0, D2)` — **physical chip 0** is **logical
  node (mesh 0, device 2)**. Logical ≠ physical; they're a permutation.

**`RouterEdge`** — one connection, as data: a direction + the link(s) to a neighbor.
- *Where:* `mesh_graph.hpp:56` · *Looks like:*
  ```cpp
  struct RouterEdge { RoutingDirection port_direction; std::vector<ChipId> connected_chip_ids; uint32_t weight; };
  ```

**`IntraMeshConnectivity` / `InterMeshConnectivity`** — the "who is wired to whom" adjacency maps.
- *Where:* `mesh_graph.hpp` · *Looks like:*
  `vector<vector<unordered_map<ChipId, RouterEdge>>>` indexed `[mesh_id][src_chip] → {neighbor → wire}`.
- *Example — your 1×16 mesh `M0`* (as `MeshGraph::print_connectivity`, `mesh_graph.cpp:686`, would print):
  ```
  M0:
     D0:  1(E,0) 1(E,0)                  # D0 wired East to D1 (2 parallel links = "channels count:2")
     D1:  0(W,0) 0(W,0)  2(E,0) 2(E,0)   # D1 wired West to D0, East to D2
     ...
     D15: 14(W,0) 14(W,0)                # end of the line: only a West neighbor
  ```
  As the C++ object, the `D1` row is:
  ```cpp
  intra_mesh_connectivity_[M0][D1] = {
      D0 : RouterEdge{.port_direction=W, .connected_chip_ids={D0,D0}, .weight=0},
      D2 : RouterEdge{.port_direction=E, .connected_chip_ids={D2,D2}, .weight=0},
  };
  ```
  Intra = *inside* one mesh; Inter = *between* meshes (empty for Big-Mesh's single `M0`; it's where the
  Multi-Mesh cross-links live).

**`RoutingTable`** — THE deliverable: the per-chip lookup.
- *Where:* `routing_table_generator.hpp:20` · *Looks like:*
  ```cpp
  using RoutingTable = vector<vector<vector<RoutingDirection>>>;  // [mesh_id][chip_id][target] -> direction
  ```
- *Example (1×16 line):* `[M0][D3][D0]=W` (walk west to D0) · `[M0][D3][D7]=E` · `[M0][D3][D3]=C` (here).
  There are two: **intra-mesh** (targets are chips) and **inter-mesh** (targets are meshes).

**The producers:** `MeshGraph` (the parsed MGD, `mesh_graph.hpp:114`) · `run_physical_system_discovery`
(`physical_system_discovery.cpp:757` — each host probes its chips, MPI-gather, rank 0 stitches the two
halves over QSFP) · `TopologyMapper` (solves logical↔physical — why `(M0,D2)`≠physical 0) ·
`RoutingTableGenerator` (`routing_table_generator.hpp:23` — shortest-path over the connectivity → the
RoutingTable) · **`ControlPlane`** (`control_plane.hpp`, the orchestrator that runs all of the above and
exposes `get_fabric_node_id_from_physical_chip_id`, `write_routing_tables_to_all_chips` at
`control_plane.cpp:2068`) · **EDM router** (`erisc_datamover.cpp` — the tiny per-chip kernel that reads
the table and forwards packets).

#### Insight 1 — directions & chip ids are *logical*, not physical

The routing table is indexed by logical `(mesh_id, chip_id)` and its directions live in the MGD's logical
grid, **not** raw rack geography. `E/W` move along the column axis (`mesh_coord[1]`), `N/S` along rows
(`mesh_coord[0]`) — `control_plane.cpp:220-260`. For the 1×16 (`dims [1,16]` = 1 row × 16 cols) all routing
is **E/W along the logical line D0…D15**; the physical chips behind D0…D15 are scattered across both boxes,
so "go **W** from D3" may physically mean *hop across the QSFP cable to the other host*. The ControlPlane
owns that logical→physical mapping (`control_plane.cpp:1355`, `RoutingDirection → eth_chan_directions`).

#### Insight 2 — the connectivity graph lives on the HOST; chips get only their table

```
HOST process (ControlPlane)                          each CHIP
  ├─ MeshGraph: IntraMeshConnectivity   ── computes ──▶  its own compact routing table
  │  (full "who's wired to whom" graph, in RAM)          (target → direction; compressed_direction_table.cpp)
  └─ ONE copy (control_plane.cpp:429)                  + the EDM router that reads it
```
A chip never holds the graph — just "for this destination, which way do I send."

#### The flow (what runs, in order)

```
MeshGraphDescriptor (.textproto)  ─parse─▶  MeshGraph (blueprint in host RAM)
run_physical_system_discovery     ─probe─▶  real chips + links (both hosts, stitched on rank 0)
                └───────────────┬───────────────┘
                                ▼  TopologyMapper        logical (M0,D2) ⇄ physical chip 0
                                ▼  RoutingTableGenerator  RoutingTable[mesh][chip][target] = direction
                                ▼  ControlPlane.write_routing_tables_to_all_chips()  ──writes tables──▶ chips
                                ▼  launch EDM routers on the eth cores  ──▶ any chip can now reach any chip
```

**Why FABRIC_2D and not FABRIC_1D:** a 1×16 line spanning two hosts cannot be routed by `FABRIC_1D` (you
get `Could not find any forwarding direction from src (M0,D3) to dst (M0,D0)`); `FABRIC_2D` routes it
(findings gotcha #2).

Hand-off ➜ routing tables on every chip + live EDM routers. The mesh is addressable end-to-end.

---

### Layer 3 — TTFM: the ON/OFF switch for the fabric

> **The reframe that makes this click:** Layer 2 (tt-fabric) produces a *thing* you can point at — routing
> tables + running routers on the chips. **TTFM produces no thing.** It is the **ON/OFF switch** for those
> routers, plus a **policy** for who is allowed to flip it. Its result is a **state**, not an artifact.

#### The only two states

| | **Fabric DOWN** (boot / after reset) | **Fabric UP** (after init) |
|---|---|---|
| eth cores | idle (base firmware, heartbeat ticking) | running EDM **router** kernels + routing tables |
| chip→chip forwarding | ✗ — `all_gather` would hang | ✓ — collectives work |
| a fresh process opening the device | **can** (discovery pings eth cores, they answer) | **cannot** on a T3K (routers busy → eth-heartbeat probe times out — *the exact wall in §3*) |

TTFM's whole job is moving the chips between these two states — or deliberately *not* moving them.

#### The two enums that control it

**`FabricConfig`** — *which* routing scheme (`fabric_types.hpp:17`): `DISABLED=0, FABRIC_1D=2,
FABRIC_1D_RING=3, FABRIC_2D=4, …`.

**`FabricManagerMode`** — *who* does init/teardown, as **bit-flags** (`fabric_types.hpp:44`):
```cpp
enum class FabricManagerMode : uint32_t {
    INIT_FABRIC      = 1 << 0,                      // 0b01 = 1  → "I will turn it ON"
    TERMINATE_FABRIC = 1 << 1,                      // 0b10 = 2  → "I will turn it OFF"
    ENABLED          = (INIT_FABRIC & TERMINATE_FABRIC),  // 0b01 & 0b10 = 0b00 = 0  → NEITHER
    DEFAULT          = (INIT_FABRIC | TERMINATE_FABRIC),  // 0b01 | 0b10 = 0b11 = 3  → BOTH
};
bool has_flag(FabricManagerMode flags, FabricManagerMode test);   // is a given bit set?
```

#### init / teardown are the two *directions* of one switch — the result is a state transition

| operation | starting state | → ending state | what you observe |
|---|---|---|---|
| **init** (`INIT_FABRIC` bit) | DOWN | **UP** | routers launched; log `Fabric Initialized with config`; `all_gather` works |
| **teardown** (`TERMINATE_FABRIC` bit) | UP | **DOWN** | routers stopped; eth cores freed; a fresh process can re-open the device |
| **nothing** (`ENABLED` = no bits) | UP | UP *(unchanged)* | log `Fabric initialized through Fabric Manager`; it just *uses* what's there |

The gating is literally two `has_flag` checks in `fabric_firmware_initializer.cpp`:
```cpp
void FabricFirmwareInitializer::init(...) {
  if (has_flag(mode, INIT_FABRIC)) {                     // :295
      log_info("Initializing Fabric");                   // :296
      control_plane_.write_routing_tables_to_all_chips(); // :311  ← Layer 2 committed to chips HERE
      compile_and_configure_fabric();                    // :312  ← EDM routers launched
      log_info("Fabric Initialized with config {}", …);  // :313
  } else if (has_flag(mode, TERMINATE_FABRIC)) { … "…for fabric termination" (:315) … }
  else { log_info("Fabric initialized through Fabric Manager"); }  // :320  ENABLED: attach only
}
void FabricFirmwareInitializer::teardown(...) {
  if (!has_flag(mode, TERMINATE_FABRIC)) return;         // :347  no TERMINATE bit → DON'T tear down (safe)
  … actually tear the routers down …
}
```

#### The 4 modes as state journeys

```
DEFAULT   (open_mesh_device … close):     DOWN ─init→ UP ─(use)─ teardown→ DOWN     ← Metal owns it (our proof)
INIT_FABRIC   (run_fabric_manager --init): DOWN ─init→ UP ─(exit, NO teardown)→ stays UP
ENABLED   (a workload attaching):          UP ─(no init, no teardown, just use)→ still UP
TERMINATE_FABRIC (--terminate-fabric):     UP ─teardown→ DOWN
```

#### Why TTFM exists

Bringing the fabric UP is **slow** (write tables to 16 chips, compile + launch routers, sync). If Metal
*always* did init+teardown (DEFAULT), every job would re-pay that cost and no two jobs could share the
fabric. **TTFM decouples "who turns it on/off" from "who uses it"**, so the expensive UP state can be
created once and shared: one process `INIT_FABRIC` and leaves it up; many jobs run `ENABLED`; one
`TERMINATE_FABRIC` at the end. That is the entire value TTFM adds.

#### One full lifecycle (DEFAULT) — and where tt-fabric starts

```
A  set_fabric_config(FABRIC_2D, mode=DEFAULT)   ← HOST-ONLY. Records config+mode; (first call) opens the
                                                  UMD cluster (reads EthCoords, Layer 1) and BUILDS the
                                                  ControlPlane → routing tables in host RAM; reserves eth
                                                  cores.  ★ tt-fabric's PLAN is built here. Chips: still DOWN.
B  open_mesh_device(1×16)                        ← FabricFirmwareInitializer.init(): has_flag(DEFAULT,INIT)=✅
                                                  → write_routing_tables_to_all_chips + launch routers
                                                  → "Fabric Initialized with config FABRIC_2D".  Chips: UP.
C  add / matmul / all_gather / …                 ← CCL kernels attach to the live routers, data crosses fabric
D  close_mesh_device()                           ← teardown(): has_flag(DEFAULT,TERMINATE)=✅ → routers down.
   set_fabric_config(DISABLED)                    → clear fabric context, reset eth cores.  Chips: DOWN.
```
So **`set_fabric_config` starts tt-fabric (builds the plan in host RAM); `open_mesh_device` commits it to
the chips**, gated by the mode. `set_fabric_config` itself touches **no chip state** — it only records
"scheme = FABRIC_2D, and I (DEFAULT) will init-on-open + teardown-on-close," opens the cluster, and builds
the host-side plan. Swap `mode=ENABLED` and step B takes the `else` branch (`"Fabric initialized through
Fabric Manager"`) and **skips the write** — attaching to a fabric a `--initialize-fabric` process already
brought up.

**Analogy:** tt-fabric = the mechanic who installs the engine (visible work). TTFM = the **ignition key**:
init = turn key → engine running; teardown = key off → engine stops; `ENABLED` = engine already running,
just drive — don't touch the key. The "result" of turning the key isn't a new part; it's the engine being
**on or off**.

**The split lifecycle** is exposed as the in-repo CLI `run_fabric_manager`
(`tools/scaleout/fabric_manager/run_fabric_manager.cpp` → `configure_fabric_routing`,
`utils/fabric_manager_utils.cpp:44`) — it calls `SetFabricConfig` with `INIT_FABRIC` / `TERMINATE_FABRIC`,
forces slow-dispatch, writes `fabric_status.txt`. **Live-demoed in §3 below.** (A separate *external* Docker
`tt-fabric-manager` is a read-only inspector — see [`GUIDE.md` §8](../dual_t3k_ops/GUIDE.md).)

Hand-off ➜ a fabric in the state the app needs (UP), plus a contract for who tears it down.

---

### Layer 4 — tt-run + mpirun-ulfm (turn one command into N cooperating processes)

`ttnn/ttnn/distributed/ttrun.py` is a **launcher/wrapper around `mpirun-ulfm`** (OpenMPI 5.0.7 ULFM in
`/usr/local/bin` — *not* `/usr/bin/mpirun` 4.1.2). It runs in two phases: **Phase 1 plans** (who runs
where, owning which chips) and caches that plan; **Phase 2 executes** the plan via MPI.

#### Its first input: the mesh-graph descriptor (MGD) — the "floor-plan"

`tt-run`'s `--mesh-graph-descriptor` is the thing it plans *from*, so start there. The physical hardware
is fixed — `1 chip → n300 card (2 chips: one PCIe-direct, its twin reached over ethernet) → T3K box
(8 chips + 1 host) → dual T3K (16 chips, 2 hosts, joined by QSFP cables)`. Nothing in that metal says
*how you want to use it*.

The **MGD** is a small text file (`.textproto`) that **declares how the chips should be arranged** into a
logical shape and how that shape is split across hosts. It doesn't *run* anything — it's a floor-plan
every layer reads so they organize the hardware the same way. It exists because the identical chips +
cables can't guess whether you want *one 16-chip machine* or *two 8-chip machines passing work between
them*; you state that intent in this one file. (Phase 1 keeps a snapshot copy of it, `mgd.textproto` —
see the cache below.)

The two fields that decide everything: **`device_topology`** (the chip grid of *one* mesh) and
**`host_topology`** (how that *one* mesh is split across hosts).

```
──────────── BIG MESH ────────────            ──────────── MULTI-MESH ────────────
 (dual_t3k_1x16_…bigmesh_mgd)                   (dual_t3k_mesh_graph_descriptor)
mesh_descriptors { name:"M0"                   mesh_descriptors { name:"M0"
  device_topology {dims:[1,16]}  ← 16 chips      device_topology {dims:[2,4]}  ← 8 chips
  host_topology   {dims:[1,2] }  ← 2 hosts       host_topology   {dims:[1,1]}  ← 1 host
}                                              }
                                               graph_descriptors { name:"G0"
                                                 instances { M0 mesh_id:0 }   ← host A
                                                 instances { M0 mesh_id:1 }   ← host B
                                                 graph_topology { ALL_TO_ALL
                                                   channels{count:8 policy:STRICT} } }
top_level_instance { mesh M0 mesh_id:0 }       top_level_instance { graph G0 }
   ONE mesh — host boundary INSIDE it             TWO meshes — host boundary BETWEEN them
```

- **Big-Mesh** — `device 1×16`, `host 1×2`: **one** mesh of 16 chips, host boundary *inside* it (the
  16-wide axis cut into two 8-chip host-halves). One `mesh_id 0`, one coordinate space `0…15`; the cross-box
  cable is an *internal* link of that mesh.
- **Multi-Mesh** — `device 2×4`, `host 1×1`: each mesh is 8 chips owned wholly by one host; a
  `graph_descriptor` instantiates **two** of them (`mesh_id 0`, `mesh_id 1`) joined `ALL_TO_ALL`. Two
  separate meshes; the cross-box cable is an *inter-mesh* link.
- **`channels {count, policy}`** = ethernet links reserved between segments; `RELAXED` = proceed with
  whatever links are up, `STRICT` (the inter-mesh link, count 8) = that many are *required*.

| | Big-Mesh | Multi-Mesh |
|---|---|---|
| Logical device | **one** `MeshDevice(1×16)` | **two** `MeshDevice(2×4)`, one per host |
| Coordinate space | single, `0…15` | two separate, `mesh 0` / `mesh 1` |
| Cross-host data | **implicit** — CCL (`all_gather`/`all_reduce`/`reduce_scatter`) routes over the fabric, crossing the cable transparently | **explicit** — a `MeshSocket` send/recv hands a tensor from mesh 0 → mesh 1 |
| Feels like | one 16-chip machine (shared address space) | two 8-chip machines on a network (a pipeline) |
| Proof script | `scripts/stack_workload.py` | `../dual_t3k_ops/scripts/multimesh_pipeline.py` |

This is *why* Big-Mesh needs **FABRIC_2D** (a 1-D line can't route the 1×16 across two hosts), and how
those `device_topology`/`host_topology` fields become the per-rank `mesh_id`/`mesh_host_rank` values below.

#### From that floor-plan to running processes — the two phases

```
tt-run --mesh-graph-descriptor <MGD> --hosts t3k-node-a,t3k-node-b  python3 <script>
   │
   ▼  PHASE 1  (generate_rank_bindings)
   1. hash( MGD contents + host list + tt-run version )  → 3dc4e617e577a8f9…   (.phase1_cache_key)
   2. name a cache dir by the 16-char hash prefix:  generated/ttrun/3dc4e617e577a8f9/
   3. write the plan into it:  mgd.textproto · rank_bindings.yaml · hostfile · rankfile
   │       (next run, same inputs → same hash → "Phase 1 cache hit", skip straight to Phase 2;
   │        add --force-rediscovery only when the physical topology changed under an unchanged MGD)
   ▼  PHASE 2  (launch)
   mpirun-ulfm  --hostfile hostfile  <rankfile placement>  \
     -x TT_MESH_ID=… -x TT_MESH_HOST_RANK=… -x TT_VISIBLE_DEVICES=… -x TT_MESH_GRAPH_DESC_PATH=… \
     -x <forwarded launcher env>   python3 <script>     # one process per host, each with its own identity
```

**The Phase-1 cache directory — the plan, frozen.** Five files (`generated/ttrun/<hash>/`):

| File | Role | Key fields |
|---|---|---|
| `.phase1_cache_key` | cache-validity token | full SHA-256 of the Phase-1 inputs; the **dir name is its first 16 hex chars**. Recomputed each run → mismatch = fresh plan. |
| `mgd.textproto` | a **snapshot copy** of the mesh-graph descriptor this plan was built from | keeps the cached plan from silently drifting if the original descriptor is edited later |
| `hostfile` | MPI **host inventory** | `t3k-node-a slots=1` / `t3k-node-b slots=1` — one MPI rank per host (each rank drives its 8 chips via threads, not more ranks) |
| `rankfile` | MPI **placement** — pins each rank to a host+CPU slot | `rank 0=t3k-node-b slot=0` / `rank 1=t3k-node-a slot=0` (consumed as `--map-by rankfile:file=…` on OpenMPI 5) |
| `rank_bindings.yaml` | the **tt-metal identity** MPI doesn't know — which mesh + slice + devices each rank is | `rank → {mesh_id, mesh_host_rank, env_overrides.TT_VISIBLE_DEVICES}` + `mesh_graph_desc_path` |

**What's actually inside each file** (verbatim from the real Big-Mesh cache
`generated/ttrun/09951f5a4eb5deae/`):

```yaml
# ── hostfile ──  MPI host inventory: which machines, how many launch slots each
t3k-node-a slots=1
t3k-node-b slots=1

# ── rankfile ──  MPI placement: which rank runs on which host (+ CPU slot)
rank 0=t3k-node-b slot=0
rank 1=t3k-node-a slot=0

# ── .phase1_cache_key ──  full SHA-256 of the inputs; dir name = its first 16 hex chars
09951f5a4eb5deae78150fa495e0d4ad1d622378db5947a7fb54f4a6dd5874ec

# ── mgd.textproto ──  snapshot copy of the Big-Mesh descriptor this plan was built from
mesh_descriptors { name: "M0" arch: WORMHOLE_B0
  device_topology { dims: [ 1, 16 ] }      # one logical 1×16 mesh
  host_topology   { dims: [ 1, 2 ] }       # split across 2 hosts (8 chips each)
  channels { count: 2 policy: RELAXED } }
top_level_instance { mesh { mesh_descriptor: "M0" mesh_id: 0 } }

# ── rank_bindings.yaml ──  the tt-metal identity of each rank (what MPI doesn't know)
rank_bindings:
  - rank: 0
    mesh_id: 0
    mesh_host_rank: 0
    env_overrides:
      TT_VISIBLE_DEVICES: "0,1,2,3"        # this rank opens only its own 4 PCIe devices (= 8 chips)
  - rank: 1
    mesh_id: 0
    mesh_host_rank: 1                       # ← Big-Mesh: same mesh_id 0, the OTHER host-half
    env_overrides:
      TT_VISIBLE_DEVICES: "0,1,2,3"
mesh_graph_desc_path: /home/namvu/dual-t3k/tt-metal/tests/tt_metal/tt_fabric/custom_mesh_descriptors/dual_t3k_1x16_experimental_bigmesh_mgd.textproto
```

Reading it: `hostfile` + `rankfile` tell MPI **where** to start each process (rank 0 → node-b, rank 1 →
node-a); `rank_bindings.yaml` tells each process **what it is** — its `(mesh_id, mesh_host_rank)` and its
`TT_VISIBLE_DEVICES` (injected as `-x` env). Note both ranks share `mesh_id: 0` but differ in
`mesh_host_rank` (0 vs 1) — that's the Big-Mesh signature. `.phase1_cache_key` guards reuse (its first
16 hex chars, `09951f5a4eb5deae`, are the folder name); `mgd.textproto` is the frozen blueprint so the
plan can't drift from a later-edited descriptor.

**`mesh_id` and `mesh_host_rank` — the address of "a host's portion of a mesh."** The control plane reads
these two env vars (`control_plane.cpp:275`, `initialize_local_mesh_binding`) to decide which physical
chips a process owns. Their values fall straight out of the descriptor's structure:
- **`mesh_id`** counts **meshes** (one per mesh instance in the MGD).
- **`mesh_host_rank`** counts **hosts inside one mesh** (an index into that mesh's `host_topology`).

```
Big-Mesh  (device 1×16, host_topology 1×2, ONE mesh):   host boundary is INSIDE the mesh
  rank 0 → mesh_id 0, mesh_host_rank 0   (owns the first 8 chips of mesh 0)   ← SAME mesh_id,
  rank 1 → mesh_id 0, mesh_host_rank 1   (owns the second 8 chips of mesh 0)  ← DIFFERENT host_rank

Multi-Mesh (device 2×4, host_topology 1×1, TWO meshes):  host boundary is BETWEEN meshes
  rank 0 → mesh_id 0, mesh_host_rank 0   (owns ALL of mesh 0)   ← DIFFERENT mesh_id,
  rank 1 → mesh_id 1, mesh_host_rank 0   (owns ALL of mesh 1)   ← host_rank always 0
```
So Big-Mesh's per-rank `TT_MESH_HOST_RANK` (0 vs 1) is the whole ballgame: it's how each process learns
*which half of the single 16-chip coordinate space* it physically holds. (These two are set per-rank and
therefore **blocklisted** from env pass-through — they come only from the plan, never the parent shell:
`ttrun.py:1418` `ENV_BLOCKLIST`.)

**Env forwarding — set it once on the launcher.** Phase 2 injects each rank's env with `mpirun -x KEY=value`
(`ttrun.py:1573`). Two sources: (a) the per-rank plan values above; (b) **pass-through from your launcher
shell** of everything matching `TT_*`, `ARCH_*`, `WH_*`, `TTNN_*`, `MESH_*` (`ttrun.py:1394`
`ENV_PASSTHROUGH_PREFIXES`), **plus `PATH`, `VIRTUAL_ENV`, `PYTHONPATH`, `LD_LIBRARY_PATH`, `HOME`, `USER`**
(`ttrun.py:1517`). These forwarded values **override the remote's own `.bashrc`**, so the remote's stale
env can't win — you configure the environment once, on the launcher.

**Same absolute paths on both hosts (structural, not optional).** Because tt-run forwards things *verbatim*
and the remote executes them literally, the **script path**, `TT_METAL_HOME`/`TT_METAL_RUNTIME_ROOT`
(kernel-build roots — findings #10), the MGD path, and the interpreter (`PATH`/`VIRTUAL_ENV`) must all
resolve to the **same absolute path** on both machines. The one exception is `TT_METAL_CACHE`: same path
string, but it must point at each host's **local** disk (findings #11), or the two hosts race writing JIT
kernels into one NFS directory.

**One venv is enough here — because of the filesystem, not magic.** In this setup the whole tree, including
`python_env/`, lives on an **NFS mount both hosts see at the same path**; tt-run forwards `PATH`/`VIRTUAL_ENV`,
so the remote's `python3` resolves to that *same* shared venv over NFS (both ranks import the same `ttnn`).
`ttrun.py` says so directly: `TT_METAL_HOME`/`RUNTIME_ROOT`/`CACHE` are passed through "to support
**NFS-based distributed workloads where all MPI ranks share the same python_venv**" (`ttrun.py:1415-1417`).
MPI does not ship the venv anywhere — it trusts the same path to exist remotely. **Without a shared mount,
one venv is *not* enough:** each host needs an identical venv (and framework build) installed at the
identical absolute path.

Hand-off ➜ two processes, same script, each knowing "I am `(mesh_id, mesh_host_rank)`, my chips are …",
with a matching environment and interpreter on both hosts.

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

> **⚠️ Caveat — these fabric-manager runs did NOT hold `moreh-lock`.** These T3Ks are shared
> (`../dual_t3k_ops/GUIDE.md` §2), and a `moreh-vllm-*` job was running on the same 16 chips during
> this work. The GUIDE attributes the exact `Stuck at 0xabcd…` eth-heartbeat wedge — and the ENABLED
> dispatch hang has the same flavor — to **two jobs opening the fabric at once**. So the "T3K
> architectural limitation" reading of the ENABLED/TERMINATE failures is **not confirmed**: it may be
> contention. The intermittency (the 16-chip DEFAULT workload passed clean while the FM phases wedged)
> is itself a contention signature. **To settle it, re-run the fabric-manager demo under
> `moreh-lock`** (see `scripts/RUN.md`). The DEFAULT-mode 16-chip PASS in §0 stands regardless, but it
> too should be re-run under the lock to be authoritative.

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
