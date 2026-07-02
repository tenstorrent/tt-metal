# Sub-torus Routing — Working Doc

**Branch:** `agupta/subtorus-routing`  ·  **Status:** GREEN (8×4 single-process + 32×4 multi-rank)

Goal: **skip-link ("sub-torus") routing** — extra intra-mesh links that let packets jump `step`-wide
blocks along an axis, layered on a base torus. Declared by **pattern** in the mesh-graph descriptor,
expanded into `RoutingDirection::Z` edges, routed over when strictly shorter, and lowered onto physical
eth channels.

> **Capture instrumentation removed.** The early effort used a live-cluster PSD/node-map dump harness
> (`TT_FABRIC_DUMP_DIR` block in `tt_fabric_test_context.cpp`, `--dump-fabric-state` in
> `run_fabric_tests.sh`, `log_debug`→`log_info` markers in `control_plane.cpp`) to scope the work. That
> approach was replaced by the captured-`ClusterDescriptor` + mock-cluster replay, so all of it has been
> **reverted to main** — `control_plane.cpp`, `tt_fabric_test_context.cpp`, and `run_fabric_tests.sh`
> now have zero diff vs main. The `parse_fabric_node_map.py` / capture-yaml were never created.

---

## 1. Status — complete

| Area | File(s) | Work | State |
|---|---|---|---|
| proto + expansion | proto + `mesh_graph_descriptor.cpp` + `mesh_graph.cpp` | `skip_links` message/field; parse + expand pattern; fold into `intra_mesh_connectivity_` as `Z` `RouterEdge`s | **GREEN** |
| validation | `mesh_graph_descriptor.cpp` | legacy validator already passes `skip_links` (only `express_connections` rejected) — no change needed | **GREEN** |
| routing | `routing_table_generator.cpp` | route *over* skip links (strict-shorter policy) | **GREEN** |
| lowering | `control_plane.cpp` | plane/egress lowering for intra `Z` channels | **GREEN** (8×4 + 32×4) |

### Verified results
- **8×4 single-process (CPU-only mock cluster):** logical expansion (4 `Z` edges for `[LINE,RING]`),
  skip-aware routing (incl. base-then-Z `t[4][24]=S` and strict-shorter `8→24`=Z vs `8→16`=S), and
  physical lowering (`Z` channels bound, base directions intact).
- **32×4 multi-rank (`tt-run`, 4 ranks):** 32 `Z` edges expand; every skip endpoint routes `Z` with
  non-empty physical `Z` channels on all 4 ranks. The earlier "skip drops under tt-run" was a **stale
  lib** issue — rebuild with `build_metal`, not `cmake --target`, or the loaded lib's old proto silently
  drops `skip_links` (`AllowUnknownField(true)`).

---

## 2. Geometry rule (LOCKED)

New proto field on `MeshDescriptor`, declares skip links by **pattern**, not enumerated pairs:

```protobuf
skip_links { dim_idx: 0  pattern { start: 2  step: 4 } }
```

- Tile the chosen `axis` into consecutive `step`-wide blocks starting at `start`.
- Each skip link connects a block's **endpoints**: `b ↔ b + step − 1` (interior nodes get none).
- RING wrap: last block wraps (e.g. `[30,31,0,1]` → edge **30–1**).
- A block only forms if **all** its rows exist on the axis: on a `LINE` axis, blocks that would wrap
  are **dropped** (so `[8,4] [LINE,RING]` start=2 step=4 → only block `[2,5]` forms → 4 edges; the
  `[32,4] [RING,RING]` case forms all 8 blocks → 32 edges).
- Replicate across the **orthogonal axis** (×4 columns), `(row,col) → row*4+col`.

---

## 3. Identity fork — skip links use `RoutingDirection::Z` (LOCKED)

Z is an existing first-class direction (`N=0,E=1,S=2,W=3,Z=4,C=5,NONE=6`), already used for the
Blackhole inter-mesh-on-galaxy path. Using it for skip links:
- Skip channels get their **own bucket** `router_port_directions_to_physical_eth_chan_map_[node][Z]`,
  separate from the N/S/E/W grid.
- **No new enum value** → no `*_SKIP` firmware blast radius.
- The N/S/E/W plane **trimmer** (`control_plane.cpp`) only iterates N/S/E/W, so `Z` channels survive.

Assigned in `mesh_graph.cpp` (both ends of each bidirectional skip edge). Open: multiple skip families
(ROW + COLUMN) would currently share the single `Z` bucket — revisit if that needs separating.

## 4. Routing policy (LOCKED)

**Emit `Z` first-hop iff the skip-inclusive shortest path is STRICTLY shorter than base-ring-only;
equal-length stays on the base ring.** Distributed/per-hop (each chip independently picks its first
hop). In `routing_table_generator.cpp`, the two `get_shorter_direction_on_row_or_col` calls go through
`first_hop_along_axis`, which BFSs scoped to `{base±, Z}` on the axis being resolved: when a skip
helps, the first hop is the edge from src on a shortest skip-inclusive path — prefer `Z` if src can
skip, else the base hop *toward* the skip (handles the base-then-Z case where `Z` is a later hop).
Skip-free meshes are unaffected (`skip_dist==base_dist` defers to ring).

---

## 5. Test harness (CPU-only / mock cluster)

`RoutingTableGenerator` needs a `TopologyMapper` (→ `tt::Cluster` + PSD). Run deviceless from artifacts:
- **Mock cluster:** `TT_METAL_MOCK_CLUSTER_DESC_PATH=tt_metal/third_party/tt-cluster-descriptors/superclusters/blackhole/SC20_32x4_revC_subtorus_aisleC/SC20_32x4_revC_subtorus_aisleC_cluster_desc/SC20_32x4_revC_subtorus_aisleC_cluster_desc_bh-glx-110-c07u08.yaml`
  — a UMD `ClusterDescriptor` from the `tt-cluster-descriptors` submodule (the canonical home since #47402;
  `set_mock_cluster_desc` resolves bare filenames there). 8×4 uses `single_bh_galaxy`; the 32×4 multi-rank
  case uses the `SC4_32x4_revC_subtorus_aisleC` mapping. (Our own captured fixtures were removed once the
  MGDs were shown to embed onto these.)
- **PSD derived, not captured:** `run_physical_system_discovery` builds it CPU-only (`run_live=false`).
- **No-discovery `TopologyMapper`** with an **identity** `FabricNodeId→ChipId` map (inert to intra-mesh
  routing, which reads only MeshGraph geometry). Avoids the discovery ctor, which would reject `Z` edges
  (no physical cabling for skips).
- **Physical lowering** (`ControlPlaneFixture` tests) needs `configure_ethernet_cores_for_fabric_routers`
  (done in fixture `SetUp`) + `TT_METAL_SLOW_DISPATCH_MODE=1`, and the matching torus `FabricConfig`
  (`FABRIC_2D` flattens the torus → use `FABRIC_2D_TORUS_X` for `[LINE,RING]`, `FABRIC_2D_TORUS_XY` for
  `[RING,RING]`).
- **32×4 is multi-rank:** host topology `[4,1]` → 4 `tt-run` ranks, each its own mock cluster.

### Tests
| Test | Suite / file | Covers |
|---|---|---|
| `SkipLinks8x4` | `MeshGraphDescriptorTests` · `test_mesh_graph_descriptor.cpp` | logical expansion (8×4, shared descriptor) |
| `SkipLinks32x4` | `MeshGraphDescriptorTests` · `test_mesh_graph_descriptor.cpp` | logical expansion (32×4, shared descriptor) |
| `IntraMesh8x4Replay` | `SkipLinkRouting` · `test_routing_tables.cpp` | skip-aware routing |
| `PhysicalLowering8x4` | `ControlPlaneFixture` · `test_routing_tables.cpp` | physical lowering, 8×4 |
| `PhysicalLowering32x4` | `ControlPlaneFixture` · `test_routing_tables.cpp` | physical lowering, 32×4 multi-rank |

---

## 6. Reference

**Build (REQUIRED):** rebuild with **`build_metal`** after any `tt_metal/` or proto edit — `cmake
--build build_Release --target fabric_unit_tests` leaves `libtt_metal.so` stale and silently drops the
`skip_links` field.

**Single-process (CPU-only, logical + 8×4 lowering):**
```bash
TT_METAL_MOCK_CLUSTER_DESC_PATH=tt_metal/third_party/tt-cluster-descriptors/superclusters/blackhole/SC20_32x4_revC_subtorus_aisleC/SC20_32x4_revC_subtorus_aisleC_cluster_desc/SC20_32x4_revC_subtorus_aisleC_cluster_desc_bh-glx-110-c07u08.yaml TT_METAL_SLOW_DISPATCH_MODE=1 \
  ./build_Release/test/tt_metal/tt_fabric/fabric_unit_tests \
  --gtest_filter='MeshGraphDescriptorTests.SkipLinks8x4:MeshGraphDescriptorTests.SkipLinks32x4:SkipLinkRouting.*:ControlPlaneFixture.PhysicalLowering8x4'
```

**32×4 multi-rank (`tt-run`, 4 local ranks):**
```bash
TT_METAL_SLOW_DISPATCH_MODE=1 tt-run \
  --mock-cluster-rank-binding tt_metal/third_party/tt-cluster-descriptors/superclusters/blackhole/SC20_32x4_revC_subtorus_aisleC/SC4_32x4_revC_subtorus_aisleC_mapping.yaml \
  --rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/skip_links_32x4_rank_bindings.yaml \
  --mpi-args "--allow-run-as-root" \
  ./build_Release/test/tt_metal/tt_fabric/fabric_unit_tests \
  --gtest_filter='ControlPlaneFixture.PhysicalLowering32x4'
```

**MGD lowering pipeline:**
`textproto → proto::MeshGraphDescriptor → MeshGraphDescriptor → MeshGraph (IntraMeshConnectivity +
coords + host-rank split) → RoutingTableGenerator (per-dest RoutingDirection LUTs) → ControlPlane
(per-eth-channel egress tables, plane-preserving)`. A "plane" = the index of a channel within its
direction's list (N[0]↔S[0]↔E[0]…). Surplus physical channels beyond `channels.count` are trimmed;
`< requested` is fatal.

**Notes / gotchas:**
- `express_connections` (existing proto field) was a dead end: rejected by the 1.0-compat validator,
  never folded into connectivity, gets `routing_direction = C`. Pattern-based `skip_links` replaces it.
- Custom descriptors live in `tests/tt_metal/tt_fabric/custom_mesh_descriptors/`. Each skip descriptor
  is shared across its logical + routing/lowering tests: `skip_links_8x4` by `SkipLinks8x4` /
  `IntraMesh8x4Replay` / `PhysicalLowering8x4`; `skip_links_32x4` by `SkipLinks32x4` /
  `PhysicalLowering32x4`.
- Still to pin: `channels` (lanes per skip link).
