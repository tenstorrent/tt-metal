# Pairwise Validation Test — Design Document

## Motivation

Large-scale cluster bringup (e.g., 8 galaxies = 256 BH chips) suffers from a poor diagnostic experience when hardware is faulty:

1. An operator runs an expansive test like `all_to_all` across the cluster.
2. A faulty chip or link causes the test to **hang**.
3. The progress monitor can report "Device X appears hung" — but in a multi-hop test, the hung device is not necessarily the fault location. A broken link 3 hops away can stall many senders that route through it.
4. The operator has no actionable information — the fabric team gets involved, writes bespoke directed tests to binary-search for the issue, and this **elongates the feedback loop**, massively delaying cluster bringup.

**Core problem:** Multi-hop traffic conflates the fault location with the symptom location. There is no self-service diagnostic that datacenter operators can run to pinpoint faulty chips or links without specialized fabric expertise.

## Galaxy Topology Reference

- **Galaxy:** An 8×4 mesh of Blackhole (BH) devices — 32 chips per galaxy.
- **Links per direction:** Up to 2 physical links in each of the 4 intra-mesh directions (N, S, E, W).
- **Host:** A single host is responsible for 1 galaxy. The box can be torused using external cables.
- **Z-links:** Inter-mesh (inter-galaxy) connections along the Z direction. Not all devices have Z-link neighbors.
- **Cluster:** Multiple galaxies connected together (e.g., all-to-all), potentially across multiple hosts.

## High-Level Proposal

A **pairwise validation** test with two key properties:

### 1. 1-Hop-Only Traffic

Each chip sends traffic only to its **immediate neighbors** (wherever a neighbor exists). This means every sender-receiver pair maps to **exactly one physical link**. If a pair fails, you know *precisely* which link is bad — no ambiguity from intermediate routing.

Neighbor directions to cover:
- **N, S, E, W** — intra-mesh neighbors (standard 2D mesh directions)
- **Z** — inter-mesh neighbors (cross-galaxy links, where they exist)

For non-torus topologies, edge/corner chips have fewer neighbors (2 or 3 instead of 4 for intra-mesh). For torus topologies, every chip has neighbors in all 4 intra-mesh directions due to wraparound links.

### 2. Per-Pair Failure Reporting

When the test times out due to a hang, the output should clearly identify every hung flow with actionable physical location information. Only hung endpoints are reported — successful flows are inferred by absence:

```
HUNG FLOWS (1):

  [1] Flow src: [host1] Tray 0 / ASIC 5 / Eth Ch 6 / TRACE Port 3
      Flow dst: [host1] Tray 0 / ASIC 4 / Eth Ch 6 / TRACE Port 4
      Sender: 0/1000 packets | no progress for 35s

CLUSTER HEALTH: 1/208 flows have hung endpoints
```

This gives operators an observational view of which endpoints are hung, narrowed to physical link granularity, without requiring fabric team involvement for initial triage.

## Detailed Requirements

### R1: Multi-Link Coverage

The test should exercise **all physical links** in each direction. For a galaxy with up to 2 links per direction, each `(src_chip, dst_chip, direction, link_id)` tuple should be an independently observable sender.

**Existing infra support:** `expand_link_duplicates()` in `TestConfigBuilder` already handles this — it duplicates each sender for each `link_id` when `num_links > 1`. The pairwise pattern just needs to generate neighbor pairs; multi-link duplication happens downstream automatically.

### R2: Z-Link (Inter-Mesh) Neighbor Coverage

For multi-galaxy clusters, devices connected via Z-links to other galaxies must also be covered. Z-link neighbors exist in a different mesh (different `mesh_id` in `FabricNodeId`), and the receiver may live on a different host.

**Current gap:** The test fixture's `get_hops_to_nearest_neighbors()` and `get_directional_neighbor_pairs()` only iterate over `{N, S, E, W}`, never `Z`. Similarly, `get_mesh_adjacency_map()` only checks N/S/E/W for cross-mesh relationships. The `RoutingDirection::Z` enum value exists, and the control plane's `get_chip_neighbors()` API should support Z-direction queries.

**Work needed:**
- Add `RoutingDirection::Z` to direction enumeration in neighbor-finding functions
- Verify control plane returns Z-direction neighbors correctly
- Handle cross-host implications: sender is local but receiver might not be. The `is_local_fabric_node_id()` check in the progress monitor already handles polling only local devices, but receiver-side validation needs care.

### R3: Per-Endpoint Monitoring and Failure Reporting

The progress monitor must be extended from device-level to **per-endpoint** granularity. A single core can run multiple configs (e.g., a sender transmitting in 4 directions, or a receiver servicing multiple senders), and we need to know exactly which config stalled.

**Existing foundation:** `format_device_label()` already produces rich identifiers:
```
(mesh_id, chip_id) [hostname(Rank)/Tray/Node]
```

**Work needed:**
- Introduce a deterministic `flow_uid` per logical traffic config, used as the cross-reference key between sender and receiver endpoints (replaces the core-level `sender_id`)
- Track per-config progress via per-config packet counts written to the result buffer by kernels
- Map each config back to its flow descriptor (source, destination, direction, link_id, eth_channel) via host-side `flow_uid` lookup
- Produce two-tier failure reports (summary for operators, detailed for fabric team) listing all hung endpoints with physical location info
- Continue printing aggregate progress to stdout for live monitoring

### R4: Hang Detection and Termination Strategy

Once a hang is detected and the progress monitor has captured all hung endpoints, the test must exit the polling loop and skip the blocking `wait_for_programs()` call to reach the MPI exchange and reporting pipeline. See "Hang Exit Strategy: Control Flow on Confirmed Hang" for the concrete control flow changes required.

- **Option A (Self-terminate via `TT_THROW`):** After writing both reports, throw with a message containing the report file paths. The test framework catches the exception and reports failure. CI-friendly. **Recommended for initial implementation.**

- **Option B (Log and wait):** After writing both reports, log a "HUNG DETECTED" banner with file paths and block. The operator inspects the live system, then kills the process manually. Selected via `--wait-on-hang` flag.

These are not mutually exclusive. **This decision must be made before Phase 3 — it is a prerequisite for the reporting pipeline to work, not a deferrable policy choice.**

## Relationship to Existing Patterns

| Pattern | What It Does | What's Missing |
|---------|-------------|----------------|
| `neighbor_exchange` | All chips send to all neighbors concurrently | No Z-link coverage. No per-config visibility — progress monitor reports per-device aggregates only. No structured failure report. |
| `sequential_neighbor_exchange` | One pair per iteration | Too slow for large clusters (~800+ pairs per galaxy); still no structured failure report |
| `all_to_all` | Every chip sends to every other chip | Multi-hop traffic — a single fault stalls many unrelated senders; no fault isolation |

### Key Insight: No New Pattern Needed

The traffic pattern for pairwise validation is **the same as `neighbor_exchange`** — all chips send 1-hop traffic to their immediate neighbors concurrently. The pair generation logic is structurally identical. The only gaps in `neighbor_exchange` are:

1. **Z-links are excluded** — the direction list is hardcoded to `{N, S, E, W}`
2. **Multi-Z neighbors are silently dropped** — the control plane path takes only the first mesh/first chip
3. **Z always needs control plane lookup** — even when Ring/Torus topologies use coordinate-based neighbors for N/S/E/W, Z has no coordinate dimension and must go through the control plane

These are bugs/gaps in the existing neighbor discovery infrastructure, not reasons for a new pattern. Once fixed, `neighbor_exchange` naturally covers all physical neighbors including Z-links.

**What makes "pairwise validation" a distinct test is not the traffic pattern — it's the monitoring and reporting infrastructure:** per-endpoint progress tracking, multi-round hung detection, two-tier failure reports with physical metadata. This infrastructure is pattern-independent and activated via CLI flags (`--show-progress-detail`).

**The pairwise validation test is therefore a specific YAML config using `neighbor_exchange` + the enhanced monitoring flags.**

## Pattern Fix: `neighbor_exchange` with Z-Link Support

### How It Works (After Fix)

For every device in the mesh, and for every direction (N, S, E, W, Z), the appropriate neighbor lookup discovers all neighbors. If a neighbor exists, a directed sender is generated. Bi-directional coverage happens naturally: when iterating A's directions, we find B via East; when iterating B's directions, we find A via West. Every physical link is tested in both directions.

Devices that lack neighbors in some directions (edge/corner chips in non-torus, or chips without Z-links) simply have fewer senders — the control plane returns empty for those directions and they are skipped.

### Neighbor Discovery: `get_all_neighbor_node_ids()`

The existing `get_neighbor_node_id_or_nullopt()` returns a single `optional<FabricNodeId>` — structurally correct for N/S/E/W (at most one neighbor per direction within a mesh), but wrong for Z where a chip can have Z-links to **multiple galaxies**, each a different `mesh_id`. A new function is needed:

```cpp
// Returns ALL neighbors in a given direction, across all meshes.
// Use this when the caller must enumerate every neighbor — e.g., pairwise validation
// where every Z-link to every connected galaxy must be tested.
//
// For N/S/E/W on a standard mesh, this typically returns 0 or 1 entries.
// For Z in a multi-galaxy cluster, this can return multiple entries with different mesh_ids.
//
// Contrast with get_neighbor_node_id_or_nullopt(), which returns at most one neighbor
// and is appropriate for patterns that assume a single neighbor per direction (e.g.,
// NeighborExchange with intra-mesh traffic only).
std::vector<FabricNodeId> get_all_neighbor_node_ids(
    const FabricNodeId& src_node_id, const RoutingDirection& direction) const {
    const auto& neighbors =
        tt::tt_metal::MetalContext::instance().get_control_plane().get_chip_neighbors(src_node_id, direction);

    std::vector<FabricNodeId> result;
    for (const auto& [mesh_id, chip_ids] : neighbors) {
        for (const auto& chip_id : chip_ids) {
            FabricNodeId neighbor(mesh_id, chip_id);
            if (neighbor != src_node_id) {
                result.emplace_back(neighbor);
            }
        }
    }

    // CRITICAL: sort by (mesh_id, chip_id) for deterministic ordering.
    // get_chip_neighbors() returns unordered_map<MeshId, vector<ChipId>> —
    // MeshId iteration order is non-deterministic. Without sorting, Z neighbors
    // appear in different order on different hosts, which changes pair order,
    // config_idx assignment, and flow_uid — breaking cross-host collation.
    std::sort(result.begin(), result.end());

    return result;
}
```

The existing `get_neighbor_node_id_or_nullopt()` should be **restricted to N/S/E/W only**:
- **Fatal on Z.** The function's single-return contract is fundamentally incompatible with multi-Z topologies (a chip can have Z-link neighbors in multiple meshes). Rather than silently returning the first and losing data, callers must use `get_all_neighbor_node_ids()` for Z.
- **Tighten the assertion for N/S/E/W.** The current `TT_FATAL(neighbors.size() <= 1)` is correct for these directions on standard meshes — keep it, but scope it explicitly.

```cpp
// Returns a single neighbor in the given direction, or nullopt if none exists.
// Strictly for N/S/E/W directions where at most one neighbor per direction is
// guaranteed. For Z direction, use get_all_neighbor_node_ids() instead.
std::optional<FabricNodeId> get_neighbor_node_id_or_nullopt(
    const FabricNodeId& src_node_id, const RoutingDirection& direction) const override {
    TT_FATAL(
        direction != RoutingDirection::Z,
        "get_neighbor_node_id_or_nullopt() does not support Z direction — a chip can have "
        "Z-link neighbors in multiple meshes. Use get_all_neighbor_node_ids() instead.");

    const auto& neighbors =
        tt::tt_metal::MetalContext::instance().get_control_plane().get_chip_neighbors(src_node_id, direction);

    TT_FATAL(
        neighbors.size() <= 1,
        "Expected at most one neighbor mesh for {} in direction: {}",
        src_node_id,
        direction);

    if (!neighbors.empty()) {
        return FabricNodeId(neighbors.begin()->first, neighbors.begin()->second[0]);
    }
    return std::nullopt;
}
```

### Fix: `get_directional_neighbor_pairs()` with Per-Direction Z Handling

The existing function has two code paths: coordinate-based (Galaxy Ring/Torus) and control plane (everything else). The `use_coordinate_neighbors` flag is currently a blanket boolean applied to all directions. But Z has no coordinate dimension — it must always use the control plane, regardless of topology.

The fix makes the branching **per-direction**: N/S/E/W follow the existing logic (coordinate or control plane based on topology), Z always goes through control plane via `get_all_neighbor_node_ids()`.

```cpp
std::vector<std::pair<FabricNodeId, FabricNodeId>> get_directional_neighbor_pairs(
    const std::vector<FabricNodeId>& device_ids, bool is_galaxy) const override {
    std::vector<std::pair<FabricNodeId, FabricNodeId>> pairs;

    const bool use_coordinate_neighbors =
        is_galaxy && (topology_ == Topology::Ring || topology_ == Topology::Torus);

    for (const auto& src_node : device_ids) {
        const auto src_coord = get_device_coord(src_node);
        for (const auto& direction : FabricContext::routing_directions) {

            if (direction == RoutingDirection::Z || !use_coordinate_neighbors) {
                // Control plane path: always for Z, also for non-Ring/non-Torus topologies
                // Uses get_all_neighbor_node_ids() to handle multi-Z (multiple meshes)
                auto neighbors = get_all_neighbor_node_ids(src_node, direction);
                for (const auto& neighbor : neighbors) {
                    bool is_valid = true;
                    if (topology_ == Topology::Linear) {
                        is_valid = are_devices_linear({src_node, neighbor});
                    }
                    if (is_valid) {
                        pairs.emplace_back(src_node, neighbor);
                    }
                }
            } else {
                // Coordinate-based path: N/S/E/W only, for Galaxy Ring/Torus
                const auto neighbor_coord = src_coord.get_neighbor(
                    mesh_shape_,
                    get_step_for_direction(direction),
                    get_dim_for_direction(direction),
                    get_boundary_mode_for_dimension(get_dim_for_direction(direction)));

                if (neighbor_coord.has_value()) {
                    auto neighbor = get_fabric_node_id(neighbor_coord.value());
                    if (neighbor != src_node) {
                        pairs.emplace_back(src_node, neighbor);
                    }
                }
            }
        }
    }
    return pairs;
}
```

**Key design choices:**

1. **Per-direction branching** — Z always goes through control plane (no coordinate dimension). N/S/E/W follow the existing topology-dependent logic.

2. **Uses `get_all_neighbor_node_ids()`** for the control plane path, which iterates all meshes and all chips returned by `get_chip_neighbors()`. No neighbors are silently dropped — critical for Z-link coverage on multi-galaxy clusters.

3. **Uses `FabricContext::routing_directions`** (N, S, E, W, Z) instead of the hardcoded `{N, S, E, W}` vector. On Wormhole (no Z links), the control plane returns empty for Z — harmlessly skipped.

4. **No new pattern or pair generation function** — `neighbor_exchange` and `sequential_neighbor_exchange` both call `get_directional_neighbor_pairs()`, so both automatically gain Z coverage.

### Direction List Centralization (Cleanup)

There are 4 places in `tt_fabric_test_common.hpp` that construct a local `{N, S, E, W}` direction vector instead of using `FabricContext::routing_directions`:

| Location | Lines | Category |
|----------|-------|----------|
| `get_mesh_adjacency_map()` | 388-389 | Neighbor enumeration — **should use `FabricContext::routing_directions`** |
| `get_directional_neighbor_pairs()` | 903-904 | Neighbor enumeration — **should use `FabricContext::routing_directions`** |
| `get_hops_to_nearest_neighbors()` | 1676-1677 | Neighbor enumeration — **should use `FabricContext::routing_directions`** |
| `trace_traffic_per_boundary()` | 1722-1723 | Coordinate-based path tracing — **must stay N/S/E/W only** (Z has no coordinate dimension) |

The first three are neighbor-enumeration functions that query the control plane. Replacing their local vectors with `FabricContext::routing_directions` is a safe cleanup: the control plane returns empty for directions with no neighbors (e.g., Z on Wormhole), so no arch-specific guard is needed. This also fixes the Z-exclusion gap identified in the Z-Link Investigation section without any per-site logic.

The fourth (`trace_traffic_per_boundary`) does 2D coordinate stepping and legitimately only handles N/S/E/W. It should remain unchanged.

### YAML Config for Pairwise Validation

No new pattern enum or expander needed. The pairwise validation test is a specific YAML config using the existing `neighbor_exchange` pattern with multi-link and the enhanced monitoring CLI flags:

```yaml
tests:
  - name: "PairwiseValidation_Galaxy"
    fabric_setup:
      topology: Mesh
      fabric_tensix_config: MuxDemux
      num_links: 2
    high_level_patterns:
      - type: neighbor_exchange
    defaults:
      ftype: CHIP_UNICAST
      ntype: NOC_UNICAST_WRITE
      size: 4096
      num_packets: 1000
```

Run with monitoring flags:
```bash
./test_tt_fabric --yaml PairwiseValidation_Galaxy.yaml \
    --show-progress --show-progress-detail \
    --hung-confirmation-rounds 3
```

### What Existing Infra Handles Automatically

Once the pairs are generated by the fixed `get_directional_neighbor_pairs()`, the rest of the pipeline operates unchanged:

| Step | Handled By | Notes |
|------|-----------|-------|
| Multi-link duplication | `expand_link_duplicates()` | Generates per-link senders when `num_links > 1` |
| Core allocation | `GlobalAllocator::allocate_resources()` | Assigns sender/receiver cores, reserves mux cores |
| Cross-mesh routing (Z) | `add_traffic_config()` | Leaves `hops` as `nullopt` for cross-mesh pairs; runtime routing table handles it |
| Fabric connection setup | `register_fabric_connection()` + `generate_connection_args_for_core()` | **Requires fix:** `ConnectionKey` must include `dst_node_id` for multi-Z disambiguation (see Gap: ConnectionKey section) |
| Kernel creation | `TestDevice::create_kernels()` | Each sender gets 1 fabric connection to its neighbor |
| Progress monitoring | `TestProgressMonitor::poll_until_complete()` | Polls sender result buffers (enhanced version will track per-endpoint) |

### Scale Estimate

For a single galaxy (8×4 mesh, non-torus, 2 links):
- Interior chips (6×2 = 12): 4 neighbors × 2 links = 8 senders each
- Edge chips (non-corner, 16 chips): 3 neighbors × 2 links = 6 senders each
- Corner chips (4 chips): 2 neighbors × 2 links = 4 senders each
- Total directed pairs: ~104 (before link duplication), ~208 with 2 links
- With Z-links to another galaxy: add ~32 Z pairs per galaxy (one per edge chip with Z connectivity)

This is well within the infra's capacity — `all_to_all` on 32 chips generates 992 pairs.

## Existing Infra Components to Leverage

| Component | How It Helps |
|-----------|-------------|
| `get_directional_neighbor_pairs()` | Pair generation for `neighbor_exchange` — to be fixed for Z support |
| `get_all_neighbor_node_ids()` | New multi-Z-safe neighbor enumeration (see Neighbor Discovery section) |
| `add_senders_from_pairs()` | Converts `(src, dst)` pairs into parsed sender configs with destination device set |
| `expand_link_duplicates()` | Duplicates senders per link_id for multi-link coverage |
| `TestProgressMonitor` | Polls L1 result buffers for packet counts; has hung detection and `format_device_label` |
| `format_device_label()` | Rich formatting: `(mesh_id, chip_id) [hostname(Rank)/Tray/Node]` |
| `SenderMemoryMap::get_result_buffer_address()` | Address for reading 64-bit packet counts from sender cores |

## Z-Link Investigation

### Control Plane: Z Is Fully Supported

The control plane has first-class Z-link support. No work needed here.

**`ControlPlane::get_chip_neighbors()`** (`control_plane.cpp:1558-1574`) is direction-agnostic — it checks both intra-mesh and inter-mesh connectivity for whatever `RoutingDirection` is passed, including Z:

```cpp
// Intra-mesh: walks RouterEdges where port_direction == routing_direction
// Inter-mesh: walks inter_mesh_connectivity where routing_edge.port_direction == routing_direction
```

- **Intra-mesh** will return empty for Z (2D mesh only has N/S/E/W edges)
- **Inter-mesh** returns Z neighbors when the mesh graph has Z-labeled edges — which happens when `assign_z_direction` is set in the mesh graph descriptor (textproto), or when physical Blackhole Z channels are bound

**Key control plane Z-specific logic:**
- `control_plane.cpp:1364-1387` — maps `RoutingDirection::Z` ↔ `eth_chan_directions::Z`
- `control_plane.cpp:2623-2702` — `assign_logical_ports_to_exit_nodes()` prefers Z logical ports when `should_assign_z_direction` is true; on Blackhole, physical Z channels must use `RoutingDirection::Z`
- `control_plane.cpp:2690-2699` — Falls back to Z ports (with warning) if N/S/E/W ports are exhausted

**Mesh graph Z support:**
- `mesh_graph.cpp:501-508` — If `chip_spec_.num_z_ports > 0`, Z logical channels are registered per chip
- `mesh_graph.cpp:139-194` — `add_to_connectivity()` sets `RouterEdge.port_direction` (can be Z) on inter-mesh edges
- Routing table generator (`routing_table_generator.cpp:208-297`) is direction-agnostic — inter-mesh routes use whatever `port_direction` the connectivity says, including Z

**Verdict:** `get_chip_neighbors(node_id, RoutingDirection::Z)` works correctly for any topology where Z-links are configured in the mesh graph descriptor.

### Test Fixture: Neighbor Discovery Functions and Z

**`get_neighbor_node_id_or_nullopt()`** (`tt_fabric_test_common.hpp:1294-1315`) delegates directly to `control_plane.get_chip_neighbors()` and constructs a `FabricNodeId` from the result (potentially with a different `mesh_id`).

**Problem:** The function has `TT_FATAL(neighbors.size() <= 1)` which will **crash on valid multi-Z topologies** where a chip has two Z-link neighbors. Additionally, its single `optional<FabricNodeId>` return type is structurally incapable of representing multiple Z neighbors.

**Solution:** A new `get_all_neighbor_node_ids(direction)` function (see Pair Generation section above) returns `std::vector<FabricNodeId>` covering all meshes and all chips. The existing function is made **fatal on Z** — its single-return contract is incompatible with multi-Z topologies, so Z callers must use `get_all_neighbor_node_ids()` instead. Existing patterns (NeighborExchange, etc.) that only use N/S/E/W are unaffected.

### Test Infrastructure: Where Z Is Excluded

The following locations explicitly enumerate only `{N, S, E, W}` and would need Z added:

#### Critical for Neighbor Exchange Z Coverage

| Location | File | Lines | Impact |
|----------|------|-------|--------|
| `get_directional_neighbor_pairs()` | `tt_fabric_test_common.hpp` | 903-904 | **Direct blocker** — Z pairs never generated for `neighbor_exchange` |
| `get_hops_to_nearest_neighbors()` | `tt_fabric_test_common.hpp` | 1676-1677 | **Direct blocker** — Z neighbors never discovered |
| `get_mesh_adjacency_map()` | `tt_fabric_test_common.hpp` | 388-389 | **Indirect blocker** — `all_to_all` multi-mesh filtering misses Z-only adjacency |

**Fix for these three:** Replace the local `{N, S, E, W}` vectors with `FabricContext::routing_directions` (which includes Z). The control plane returns empty for directions with no neighbors, so no arch-specific guard is needed. In `get_directional_neighbor_pairs()`, Z must always use the **control plane path** (not coordinate-based), since Z has no 2D coordinate step. The per-direction branching is detailed in the "Fix: `get_directional_neighbor_pairs()` with Per-Direction Z Handling" section above.

#### Connection Count Constants (Correctness Improvements)

Two constants are currently hardcoded to 4 (N/S/E/W) and need updating. However, the fix is **not** a simple "4 → 5 per direction" change — once `ConnectionKey` includes `dst_node_id`, the connection count per core is no longer bounded by the number of directions. With the current Z-topology model, a chip can have up to 2 Z neighbors, so a core can need up to 6 distinct connections: N + S + E + W + Z(mesh A) + Z(mesh B).

**1. `NUM_DIRECTIONS` — Host-side allocator (`tt_fabric_test_allocator.hpp:910`)**

Used to conservatively estimate mux core reservations per device: `mux_cores_per_device = NUM_DIRECTIONS * num_links`. This feeds into `reserved_cores` which is subtracted from the worker pool before computing receiver allocation.

**Problem:** This formula assumes one connection per direction. With multi-Z, a device can have more connections than directions.

**Fix:** Replace `NUM_DIRECTIONS` with `MAX_CONNECTIONS_PER_DEVICE` that accounts for Z fan-out. Since a chip can have at most 2 Z neighbors, `MAX_Z_NEIGHBORS = 2` is the correct bound:

```cpp
// Maximum Z-link fan-out per chip.
// A chip can have at most 2 Z neighbors.
static constexpr uint32_t MAX_Z_NEIGHBORS = 2;

static uint32_t get_max_connections_per_device() {
    auto arch = tt::tt_metal::hal::get_arch();
    switch (arch) {
        case tt::ARCH::BLACKHOLE: return 4 + MAX_Z_NEIGHBORS;  // N, S, E, W + up to 2 Z destinations
        default: return 4;                                       // N, S, E, W
    }
}
```

The allocator formula becomes `mux_cores_per_device = get_max_connections_per_device() * num_links`.

**2. `MAX_NUM_FABRIC_CONNECTIONS` — Kernel-side (`kernels/tt_fabric_test_kernels_utils.hpp:23`)**

Sizes the `FabricConnectionArray` storage, `is_mux`, and `mux_cached_info` arrays in device code. Also used in `static_assert` checks in sender/receiver kernels.

- **Wormhole:** 4 connections max (1 per direction).
- **Blackhole:** 6 connections max (4 NESW + up to 2 Z destinations).

**Fix:** Use an `#ifdef` on architecture in the kernel header:

```cpp
#ifdef ARCH_BLACKHOLE
// 4 NESW directions + up to 2 Z-link destinations.
static constexpr uint8_t MAX_NUM_FABRIC_CONNECTIONS = 6;
#else
static constexpr uint8_t MAX_NUM_FABRIC_CONNECTIONS = 4;
#endif
```

The existing `static_assert(NUM_FABRIC_CONNECTIONS <= MAX_NUM_FABRIC_CONNECTIONS)` in sender/receiver kernels continues to be the compile-time guard.

#### Other Downstream Items (Not Blockers for Pairwise Unicast)

| Location | File | Lines | Impact |
|----------|------|-------|--------|
| `ChipMulticastFields2D` | `tt_fabric_test_traffic.hpp` | 86-97 | Only 4 hop fields (N/S/E/W). **Not a blocker** — pairwise validation uses unicast (1 hop, 1 direction), not multicast |
| `trace_traffic_per_boundary()` | `tt_fabric_test_common.hpp` | ~1722 | Bandwidth profiling only — **not needed** for validation mode |
| `get_step_for_direction()` / `get_dim_for_direction()` | `tt_fabric_test_common.hpp` | ~2178-2195 | Returns 0/-1 for Z — **not a blocker** since Z doesn't use coordinate stepping |

### Cross-Mesh Traffic Path in `add_traffic_config()`

When a sender's `dst_node_id` has a different `mesh_id` from `src_node_id`, the hop derivation path in `TestContext::add_traffic_config()` (`tt_fabric_test_context.cpp:610-612`) skips `get_hops_to_chip()`:

```cpp
if (src_node_id.mesh_id == dst_node_ids[0].mesh_id) {
    hops = this->fixture_->get_hops_to_chip(src_node_id, dst_node_ids[0]);
}
```

For Z-link pairs, `hops` remains `std::nullopt`. This is acceptable for **unicast** — the fabric runtime routing table (populated by the control plane) knows how to route cross-mesh traffic. The sender kernel just needs the `dst_node_id` (with the remote `mesh_id`) and the runtime handles the Z-hop.

**Key insight:** For pairwise validation, the pattern expansion should use `add_senders_from_pairs()` which sets `destination.device = dst_node` (explicit destination). This flows through `add_traffic_config` with `traffic_config.dst_node_ids` set, and the `hops` field is left empty for cross-mesh pairs. The runtime routing table handles the rest. **No new hop computation needed for Z.**

### Existing Multi-Mesh Test Precedent

Two existing YAML files already send cross-mesh traffic:
- **`test_5_galaxy_multi_mesh.yaml`** — Explicit cross-mesh unicast pairs like `device: [1, [7,3]]`
- **`test_t3k_2x2_z.yaml`** — Uses a Z-oriented mesh descriptor with `device: [0, ...]` / `device: [1, ...]` senders

These confirm the runtime can handle cross-mesh unicast. The fixed `neighbor_exchange` just needs to generate these pairs automatically via the control plane.

### Gap: `ConnectionKey` Drops Destination Identity (Blocker for Multi-Z)

`ConnectionKey` in `tt_fabric_test_device_setup.hpp` identifies a fabric connection as `{direction, link_idx, vc_id}`. This is sufficient for N/S/E/W where each direction has exactly one neighbor, but **breaks for Z when a chip has Z-link neighbors in multiple meshes** — two distinct Z connections (same direction, same link_idx, different destinations) would hash to the same key.

The problem surfaces in `generate_connection_args_for_core()` (`tt_fabric_test_device_setup.cpp:277`), which reconstructs the destination using `route_manager->get_neighbor_node_id(fabric_node_id, key.direction)`. This loses the explicit destination that `add_traffic_config()` originally provided, and for multi-Z the single-return `get_neighbor_node_id()` picks an arbitrary neighbor.

**Fix: Add `dst_node_id` to `ConnectionKey` unconditionally.**

```cpp
struct ConnectionKey {
    RoutingDirection direction;
    uint32_t link_idx;
    uint8_t vc_id = 0;
    FabricNodeId dst_node_id;  // Destination identity — required for multi-Z disambiguation

    bool use_vc2() const { return vc_id == 2; }

    bool operator==(const ConnectionKey& other) const {
        return direction == other.direction && link_idx == other.link_idx &&
               vc_id == other.vc_id && dst_node_id == other.dst_node_id;
    }
    bool operator<(const ConnectionKey& other) const {
        return std::tie(direction, link_idx, vc_id, dst_node_id) <
               std::tie(other.direction, other.link_idx, other.vc_id, other.dst_node_id);
    }
};
```

This is a clean, unconditional fix: for N/S/E/W the `dst_node_id` is redundant (there's only one neighbor) but adds no overhead — `ConnectionKey` is a host-side struct used during setup, not in hot paths. For Z, it provides the disambiguation that makes multi-Z connections correct.

The downstream changes are:

1. **`register_fabric_connection()`** — add a `FabricNodeId dst_node_id` parameter. The function currently takes `(logical_core, worker_type, connection_mgr, outgoing_direction, link_idx, vc_id)`. Add `dst_node_id` and pass it into `ConnectionKey` construction and `register_client()`. There is only one function (no overloads).

2. **All three call sites** must pass `dst_node_id`:
   - **`TestSender::add_config()`** (`tt_fabric_test_device_setup.cpp:409`): `dst_node_id` is already in scope from `config.dst_node_ids[0]` (line 388).
   - **`TestReceiver::add_config()`** (`tt_fabric_test_device_setup.cpp:470`): `dst_node_id` is `credit_info.sender_node_id` (line 465).
   - **`TestSync::add_config()`** (`tt_fabric_test_device_setup.cpp:526`): use `sender_config.dst_node_ids[0]` (same pattern as `TestSender`).

3. **`generate_connection_args_for_core()`** (`tt_fabric_test_device_setup.cpp:277`): read `key.dst_node_id` instead of re-deriving via `route_manager->get_neighbor_node_id(fabric_node_id, key.direction)`, eliminating the lossy re-derivation.

4. **`create_mux_kernels()`** (`tt_fabric_test_device_setup.cpp:723`): same fix — read `connection_key.dst_node_id` instead of re-deriving via `route_manager_->get_neighbor_node_id(fabric_node_id_, connection_key.direction)`.

### Gap: Sync Model Does Not Support Multi-Z (Deferred)

The `NeighborExchange` sync model (`get_hops_to_nearest_neighbors()`) builds a hop map as `unordered_map<RoutingDirection, uint32_t>` — one entry per direction. This models the "nearest neighbor in each direction" which works for N/S/E/W (one neighbor per direction) but collapses multi-Z neighbors into a single entry.

Additionally, `compute_destination_nodes_from_hops()` is explicitly local-mesh-only for unicast hop expansion, and the sync barrier in `NeighborExchange` assumes each direction maps to a single destination.

**Decision: Defer Z sync support.** Pairwise validation does not require device-level synchronization — it uses `poll_until_complete_or_hung()` with host-driven monitoring. Adding `TT_FATAL` guard:

```cpp
// In NeighborExchange sync setup:
if (has_z_neighbors && sync_enabled) {
    TT_FATAL(false,
        "NeighborExchange sync does not support Z-link topologies. "
        "Disable sync (--sync false) or use host-driven monitoring.");
}
```

This prevents silent misconfiguration on multi-Z clusters without blocking the pairwise validation use case, which does not need sync.

**Note on `get_hops_to_nearest_neighbors()`:** This function only feeds the `NeighborExchange` sync path via `get_sync_hops_and_val()`. Its `unordered_map<RoutingDirection, uint32_t>` return type collapses multi-Z by construction (single hop count per direction). Replacing its `{N,S,E,W}` direction list with `routing_directions` would be incorrect without also fixing the return type and all downstream consumers. Therefore, `get_hops_to_nearest_neighbors()` Z support is **strictly part of the deferred sync uplift**, not a Phase 1 Z-cleanup item.

### Summary: Z-Link Work Required

| Change | Scope | Difficulty |
|--------|-------|------------|
| Add `get_all_neighbor_node_ids(direction)` for multi-Z neighbor enumeration | New function, ~15 lines | Low |
| Fix `get_neighbor_node_id_or_nullopt()`: fatal on Z, NESW-only | ~10 lines | Low |
| Fix `get_directional_neighbor_pairs()`: per-direction Z branching + `FabricContext::routing_directions` | ~15 lines | Low |
| Replace `{N,S,E,W}` in `get_mesh_adjacency_map()` with `FabricContext::routing_directions` | 1-line replacement | Trivial |
| Replace `NUM_DIRECTIONS` with `get_max_connections_per_device()` (4 + `MAX_Z_NEIGHBORS=2`) in allocator | ~15 lines | Low |
| Replace `MAX_NUM_FABRIC_CONNECTIONS` with 6 on BH (4 NESW + 2 Z) via kernel `#ifdef` | ~5 lines | Low |
| Add `dst_node_id` to `ConnectionKey` + `register_fabric_connection()` param + all 3 call sites + `generate_connection_args_for_core()` + `create_mux_kernels()` | ~30 lines across 5 sites | Medium |
| Add `TT_FATAL` guard in NeighborExchange sync for Z-link topologies | ~5 lines | Low |
| *(Deferred)* Replace `{N,S,E,W}` in `get_hops_to_nearest_neighbors()` — only feeds sync path, and its `unordered_map<RoutingDirection, uint32_t>` collapses multi-Z by construction | Part of Z sync uplift | N/A for Phase 1 |
| No changes needed in control plane, routing table generator, or runtime | — | — |
| No changes needed in multicast fields (neighbor exchange is unicast) | — | — |

## Progress Monitor: Enhanced Endpoint-Level Tracking

### Current State

The existing `TestProgressMonitor` aggregates progress **per device** — it sums all sender packet counts on a device and compares against the total. When a device is detected as hung, the warning identifies the device but not which specific sender/receiver config stalled.

```cpp
struct DeviceProgress {
    tt::tt_fabric::FabricNodeId device_id{tt::tt_fabric::MeshId{0}, 0};
    uint64_t current_packets = 0;
    uint64_t total_packets = 0;
    uint32_t num_senders = 0;
    uint32_t num_receivers = 0;
};

struct DeviceState {
    uint64_t last_packet_count = 0;
    std::chrono::steady_clock::time_point last_progress_time;
    bool warned = false;
};
```

```cpp
DeviceProgress TestProgressMonitor::poll_device_senders(
    const MeshCoordinate& /*coord*/, const TestDevice& test_device) {
    DeviceProgress progress;
    progress.device_id = test_device.get_node_id();

    // ... read one aggregate counter per sender core ...

    for (const auto& [core, sender] : test_device.get_senders()) {
        uint32_t packets_low = result_data[TT_FABRIC_WORD_CNT_INDEX];
        uint32_t packets_high = result_data[TT_FABRIC_WORD_CNT_INDEX + 1];
        uint64_t packets_sent = (static_cast<uint64_t>(packets_high) << 32) | packets_low;

        progress.current_packets += packets_sent;
        progress.total_packets += sender.get_total_packets();
        progress.num_senders++;
    }

    return progress;
}
```

```cpp
void TestProgressMonitor::check_for_hung_devices(
    const std::unordered_map<tt::tt_fabric::FabricNodeId, DeviceProgress>& progress) {
    for (const auto& [device_id, prog] : progress) {
        if (prog.current_packets >= prog.total_packets) {
            continue;
        }

        if (is_device_hung(device_id, prog.current_packets)) {
            log_warning(
                tt::LogTest,
                "Device {} may be HUNG: no progress for {} seconds (packets: {}/{})",
                format_device_label(device_id),
                elapsed.count(),
                prog.current_packets,
                prog.total_packets);
        }
    }
}
```

### Design Goals

We need **per-traffic-config** granularity on both senders and receivers so that each hung config maps to a specific physical link/direction. A single core can run multiple traffic configs (e.g., a receiver servicing 4 senders, or a sender transmitting in multiple directions), and we need to know exactly which config stalled — not just that the core is hung.

The enhanced monitor must:

- remain generic enough for any traffic pattern (not just pairwise validation)
- track local endpoints as the primary observation unit
- report only observed data (no inferences, no recommended actions)
- exchange only sparse failure data over MPI after local monitoring concludes

### Flow Identity: `flow_uid` and `FlowDescriptor`

Each logical traffic config gets a deterministic `flow_uid`. This replaces the existing `sender_id` (computed via `get_worker_id()`) as the cross-reference key between sender and receiver endpoints.

**Why not `sender_id`?** The existing `sender_id` encodes `(mesh_id, chip_id, core.x, core.y)` into a `uint32_t` — it identifies a **core**, not a **config**. When a single core hosts multiple sender configs (e.g., sending in 4 directions), `sender_id` cannot distinguish between them. `flow_uid` is per-flow, not per-core, so it uniquely identifies each logical traffic config.

#### `TestContext` Owns the Flow Registry

`flow_uid` is an index into a per-host `flow_descriptors_` vector owned by `TestContext`.

```cpp
using FlowUid = uint32_t;

struct FlowDescriptor {
    FabricNodeId src_node_id;
    CoreCoord src_logical_core;

    std::vector<FabricNodeId> dst_node_ids;  // canonicalized copy
    CoreCoord dst_logical_core;

    uint32_t link_id = 0;
    uint8_t vc_id = 0;

    ChipSendType chip_send_type;
    NocSendType noc_send_type;
    uint32_t num_packets = 0;
    uint32_t payload_size_bytes = 0;
};
```

```cpp
class TestContext {
public:
    const FlowDescriptor& get_flow_descriptor(FlowUid uid) const { return flow_descriptors_.at(uid); }
    const std::vector<FlowDescriptor>& get_flow_descriptors() const { return flow_descriptors_; }

private:
    std::vector<FlowDescriptor> flow_descriptors_;
};
```

Notes:

- `flow_descriptors_` is host-side only
- `flow_uid` is deterministic because `add_traffic_config()` runs on every host and resolves the same logical flow before the local ownership checks
- `flow_descriptors_` should be cleared during `TestContext::reset_devices()`

#### `flow_uid` Is Threaded Into Sender and Receiver Configs

Both sender and receiver configs carry the host-only `flow_uid` so that the progress monitor can directly relate a local endpoint to its configured flow descriptor.

```cpp
struct TestTrafficSenderConfig {
    // existing fields...
    FlowUid flow_uid = 0;  // host-only, not serialized to kernel args
};

struct TestTrafficReceiverConfig {
    // existing fields...
    FlowUid flow_uid = 0;  // host-only, not serialized to kernel args
};
```

#### `add_traffic_config()` Constructs the Descriptor Before Local Filtering

The intended update to `add_traffic_config()` is:

- resolve `dst_node_ids`
- resolve `hops` when applicable
- make a canonicalized copy of `dst_node_ids` for the descriptor only
- append a new `FlowDescriptor`
- use the appended index as `flow_uid`
- stamp that `flow_uid` into the sender config and receiver config
- then proceed with the existing local sender / local receiver ownership checks

Sketch:

```cpp
std::vector<FabricNodeId> descriptor_dsts = dst_node_ids;
canonicalize_dst_node_ids(descriptor_dsts);  // sort copy only for descriptor stability

FlowUid flow_uid = static_cast<FlowUid>(flow_descriptors_.size());

flow_descriptors_.push_back(FlowDescriptor{
    .src_node_id = src_node_id,
    .src_logical_core = src_logical_core,
    .dst_node_ids = std::move(descriptor_dsts),
    .dst_logical_core = dst_logical_core,
    .link_id = traffic_config.link_id,
    .vc_id = traffic_config.vc_id,
    .chip_send_type = traffic_config.parameters.chip_send_type,
    .noc_send_type = traffic_config.parameters.noc_send_type,
    .num_packets = traffic_config.parameters.num_packets,
    .payload_size_bytes = traffic_config.parameters.payload_size_bytes,
});

sender_config.flow_uid = flow_uid;
receiver_config.flow_uid = flow_uid;
```

Important detail:

- canonicalization applies only to the descriptor copy
- it should not rewrite the runtime `dst_node_ids` used by the actual traffic config

### Per-Config Progress: Kernel-Level Granularity

Instead of tracking progress per-core (aggregate), we track progress **per traffic config** on both senders and receivers. This solves the visibility problem at the source — the kernel writes per-config packet counts to the result buffer, and the host reads and maps them.

This approach works for **all patterns**, not just pairwise validation. For example, a single sender transmitting in 4 directions where only one link is stuck — the per-config breakdown immediately reveals which direction is blocked.

#### Result Buffer Layout: Structured Per-Config Output

Define a `PerConfigResult` struct for the per-config result region. This avoids all hardcoded index arithmetic — adding new per-config fields (e.g., elapsed cycles) only requires extending the struct; all indexing adapts automatically via `sizeof`.

**Placement:** The struct and base constant are defined **locally in the test infrastructure**, not in `tt_fabric_status.h`. That header is a fabric-level shared header and has no concept of test buffer sizes or per-config layouts. Instead, each side defines the struct independently:

- **Kernel side:** `tt_fabric_test_kernels_utils.hpp` — alongside the existing result buffer helpers
- **Host side:** `tt_fabric_test_memory_map.hpp` — alongside the `CommonMemoryMap` that owns the result buffer layout

Both definitions must agree on layout and size. Cross-reference comments link them, and the host side includes a `static_assert` on `sizeof(PerConfigResult)` as a safety net.

```cpp
// Defined independently in BOTH:
//   - tt_fabric_test_kernels_utils.hpp (kernel side)
//   - tt_fabric_test_memory_map.hpp (host side)
// Layout must match exactly between the two.
struct PerConfigResult {
    uint32_t packets_low;       // Lower 32 bits of packets processed
    uint32_t packets_high;      // Upper 32 bits of packets processed
    // Future fields go here (e.g., elapsed_cycles_low, elapsed_cycles_high)
    // Array indexing via sizeof automatically adapts to new fields
};

// Word index in result buffer where per-config results begin
static constexpr uint32_t PER_CONFIG_RESULT_BASE_WORD_INDEX = 32;
```

Result buffer layout (4KB = 1024 words):

```
Words 0-1:   TT_FABRIC_STATUS_INDEX      (existing, from tt_fabric_status.h)
Words 2-3:   TT_FABRIC_WORD_CNT_INDEX    (existing aggregate — preserved for backward compat)
Words 4-5:   TT_FABRIC_CYCLES_INDEX      (existing)
Words 6-7:   TT_FABRIC_ITER_INDEX        (existing)
Words 8-31:  Reserved / existing misc
Words 32+:   PerConfigResult[0], PerConfigResult[1], ... PerConfigResult[N-1]
```

With `sizeof(PerConfigResult) = 8 bytes` (2 words) and `MAX_CONFIGS_PER_CORE_CEILING = 64`: 128 + 64 × 8 = 640 bytes total. Well under the 4KB limit.

#### Kernel-Side Helpers (`tt_fabric_test_kernels_utils.hpp`)

The key insight: cast the result buffer at the per-config base into a `PerConfigResult*` array. Then `results[i]` gives you the i-th config's entry directly — the compiler handles stride via `sizeof(PerConfigResult)`.

```cpp
// Returns typed pointer to per-config result array within the result buffer
inline tt_l1_ptr PerConfigResult* get_per_config_results(uint32_t result_buffer_base) {
    return reinterpret_cast<tt_l1_ptr PerConfigResult*>(
        result_buffer_base + PER_CONFIG_RESULT_BASE_WORD_INDEX * sizeof(uint32_t));
}

// Writes a single per-config result entry
inline void write_per_config_result(tt_l1_ptr PerConfigResult* entry, uint64_t packets) {
    entry->packets_low = static_cast<uint32_t>(packets);
    entry->packets_high = static_cast<uint32_t>(packets >> 32);
}
```

These helpers are layout-only — they know about `PerConfigResult` and the base offset, but nothing about sender or receiver config types. The loop that extracts `num_packets_processed` from traffic configs stays in each kernel, since `SenderKernelTrafficConfig` and `TrafficValidationConfigBase` are unrelated types that happen to share a field name. No duck-typing templates.

#### Kernel Changes

**Sender kernel** (`tt_fabric_test_sender.cpp`) — the per-config data already exists via `traffic_config_ptrs[i]->num_packets_processed`. Change the progress writes so they update periodically and flush once more at completion:

```cpp
auto* per_config_results = get_per_config_results(sender_config->get_result_buffer_address());

if (loop_count % PROGRESS_UPDATE_INTERVAL == 0) {
    uint64_t progress_packets_sent = 0;
    for (uint8_t i = 0; i < NUM_TRAFFIC_CONFIGS; i++) {
        uint64_t config_packets = sender_config->traffic_config_ptrs[i]->num_packets_processed;
        progress_packets_sent += config_packets;
        write_per_config_result(&per_config_results[i], config_packets);
    }
    write_test_packets(sender_config->get_result_buffer_address(), progress_packets_sent);
}

// Final flush before PASS status so per-config data matches aggregate completion.
uint64_t final_packets_sent = 0;
for (uint8_t i = 0; i < NUM_TRAFFIC_CONFIGS; i++) {
    uint64_t config_packets = sender_config->traffic_config_ptrs[i]->num_packets_processed;
    final_packets_sent += config_packets;
    write_per_config_result(&per_config_results[i], config_packets);
}
write_test_packets(sender_config->get_result_buffer_address(), final_packets_sent);
write_test_status(sender_config->get_result_buffer_address(), TT_FABRIC_STATUS_PASS);
```

**Receiver kernel** (`tt_fabric_test_receiver.cpp`) — currently has no periodic progress writes. Add periodic updates plus a final flush mirroring the sender. The per-config data exists via `traffic_config->num_packets_processed`:

```cpp
auto* per_config_results = get_per_config_results(receiver_config->get_result_buffer_address());

total_packets_received++;
if (total_packets_received % PROGRESS_UPDATE_INTERVAL == 0) {
    uint64_t progress_packets_received = 0;
    for (uint8_t i = 0; i < NUM_TRAFFIC_CONFIGS; i++) {
        uint64_t config_packets = receiver_config->traffic_configs()[i]->num_packets_processed;
        progress_packets_received += config_packets;
        write_per_config_result(&per_config_results[i], config_packets);
    }
    write_test_packets(receiver_config->get_result_buffer_address(), progress_packets_received);
}

// Final flush before final status write.
uint64_t final_packets_received = 0;
for (uint8_t i = 0; i < NUM_TRAFFIC_CONFIGS; i++) {
    uint64_t config_packets = receiver_config->traffic_configs()[i]->num_packets_processed;
    final_packets_received += config_packets;
    write_per_config_result(&per_config_results[i], config_packets);
}
write_test_packets(receiver_config->get_result_buffer_address(), final_packets_received);
write_test_status(receiver_config->get_result_buffer_address(), final_status);
```

The sender and receiver loops are structurally identical but intentionally not shared — each accesses its own config type's `num_packets_processed` field directly rather than relying on a template that duck-types across unrelated config types.

Without this final flush, a core that completes between periodic update intervals can publish final aggregate status while leaving stale per-config entries in the result buffer.

#### Result Buffer Clearing

The host already clears the **full** `get_result_buffer_size()` (4KB) when `progress_monitoring_enabled_` is true:

```cpp
// In create_sender_kernels() / create_receiver_kernels() (tt_fabric_test_device_setup.cpp)
if (progress_monitoring_enabled_) {
    addresses_and_size_to_clear.push_back(
        {sender_memory_map_->get_result_buffer_address(), sender_memory_map_->get_result_buffer_size()});
}
```

Since the full 4KB is zeroed, the per-config region is automatically cleared. No change needed here.

However, a **capacity validation** should be added during kernel creation to catch the case where `num_configs` would overflow the result buffer:

```cpp
// In CommonMemoryMap (tt_fabric_test_memory_map.hpp)
uint32_t get_per_config_region_size_bytes(uint8_t num_configs) const {
    return PER_CONFIG_RESULT_BASE_WORD_INDEX * sizeof(uint32_t) + num_configs * sizeof(PerConfigResult);
}

void validate_per_config_capacity(uint8_t num_configs) const {
    TT_FATAL(
        get_per_config_region_size_bytes(num_configs) <= result_buffer.size,
        "Per-config result region ({} bytes for {} configs) exceeds result buffer ({} bytes)",
        get_per_config_region_size_bytes(num_configs), num_configs, result_buffer.size);
}
```

This is called in `create_sender_kernels()` / `create_receiver_kernels()` when `progress_monitoring_enabled_`.

#### Config Index Ordering Invariant

**Critical invariant:** The config index `i` in the kernel's result buffer corresponds exactly to `configs_[i]` in the host's `TestSender` / `TestReceiver` vector. This is enforced by:

1. **Host serialization** (`create_sender_kernels`, line 955 / `create_receiver_kernels`, line 1068): iterates `configs_` sequentially, serializing args in order
2. **Kernel parsing** (`SenderKernelConfig` / `ReceiverKernelConfig` constructors): parses `NUM_TRAFFIC_CONFIGS` configs in a sequential `for` loop from `local_args_idx` which advances linearly
3. **No reordering** at any point in the pipeline

This invariant should be documented with a comment at both the serialization site and the result buffer read site.

#### link_id to eth_channel Resolution

The `link_id` in `TestTrafficSenderConfig` is a logical index into the active eth channels for a direction. To convert to physical eth channel:

```cpp
auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
auto eth_channels = control_plane.get_active_fabric_eth_channels_in_direction(src_node_id, direction);
chan_id_t eth_channel = eth_channels[link_id];
```

This is the same mapping used by `FabricConnection::open()`. Resolution is done once during monitor construction, not per-poll. The failure report shows the physical `eth_channel`, not the logical `link_id`.

### Endpoint-First Monitoring Model

The monitor tracks **local endpoints** as the primary observation unit — not devices and not interpreted pairs. This keeps the monitoring generic for any traffic pattern (pairwise, all-to-all, multicast, sequential).

#### Core Structs

```cpp
enum class EndpointRole : uint8_t { Sender, Receiver };

struct EndpointId {
    EndpointRole role;
    FabricNodeId node_id;
    CoreCoord logical_core;
    uint16_t config_idx = 0;

    bool operator==(const EndpointId&) const = default;
};

struct EndpointHungState {
    uint64_t last_packet_count = 0;
    std::chrono::steady_clock::time_point last_progress_time{};
    uint32_t consecutive_stall_rounds = 0;
    bool confirmed_hung = false;
    bool emitted = false;
};

struct EndpointProgressState {
    FlowUid flow_uid = 0;
    EndpointId endpoint_id;

    uint64_t packets_processed = 0;
    uint64_t packets_expected = 0;

    EndpointHungState hung;
};
```

The monitor owns:

```cpp
std::unordered_map<EndpointId, EndpointProgressState, EndpointIdHash> endpoint_states_;
std::vector<HungEndpointRecord> local_hung_records_;
```

`DeviceProgress` can still exist as a derived aggregate for stdout progress display, but it is no longer the primary hang-detection unit.

**Sender stalls are the primary observation** for pairwise validation — a sender that stops making progress is the most direct signal that something is wrong on that traffic path. Receiver data provides supplementary context: if both sender and receiver for the same `flow_uid` are hung, the stall is visible from both ends. If only the sender is hung but the receiver is not (or vice versa), that asymmetry is itself useful data — but the monitor reports only what it observes, not why the asymmetry exists.

#### Monitor Construction

The constructor scans `ctx_->get_test_devices()` and registers every local sender config and every local receiver config into `endpoint_states_`:

```cpp
for (const auto& [coord, test_device] : ctx_->get_test_devices()) {
    const auto node_id = test_device.get_node_id();

    for (const auto& [core, sender] : test_device.get_senders()) {
        for (uint16_t i = 0; i < sender.configs_.size(); ++i) {
            const auto& cfg = sender.configs_[i].first;

            EndpointId id{
                .role = EndpointRole::Sender,
                .node_id = node_id,
                .logical_core = core,
                .config_idx = i,
            };

            endpoint_states_.emplace(
                id,
                EndpointProgressState{
                    .flow_uid = cfg.flow_uid,
                    .endpoint_id = id,
                    .packets_expected = cfg.parameters.num_packets,
                });
        }
    }

    for (const auto& [core, receiver] : test_device.get_receivers()) {
        for (uint16_t i = 0; i < receiver.configs_.size(); ++i) {
            const auto& cfg = receiver.configs_[i].first;

            EndpointId id{
                .role = EndpointRole::Receiver,
                .node_id = node_id,
                .logical_core = core,
                .config_idx = i,
            };

            endpoint_states_.emplace(
                id,
                EndpointProgressState{
                    .flow_uid = cfg.flow_uid,
                    .endpoint_id = id,
                    .packets_expected = cfg.parameters.num_packets,
                });
        }
    }
}
```

This gives the monitor a complete local endpoint set up front, each with:

- a stable local endpoint identity
- an expected packet count
- a `flow_uid` lookup into `ctx_->get_flow_descriptor(uid)`

#### Polling and Endpoint State Updates

The polling path:

1. Batch-read sender result buffers per device (single-shot via `initiate_read_buffer_from_cores()`)
2. Parse per-config packet counts using the `PerConfigResult` struct
3. Update sender `EndpointProgressState`s
4. Batch-read receiver result buffers per device
5. Parse per-config packet counts
6. Update receiver `EndpointProgressState`s
7. Apply per-endpoint multi-round hang detection
8. Emit one local failure record only when a local endpoint becomes newly confirmed hung

This preserves the current aggregate progress display while moving hang detection to the endpoint level.

### Host-Side Batch Read and Parsing

**Critical: do NOT read per-core individually.** The existing `initiate_read_buffer_from_cores()` batches all cores on a device into a single `MeshBuffer` + `EnqueueReadMeshBuffer`. The progress monitor uses the same pattern:

1. **Collect** all sender cores (and receiver cores) for a device
2. **Single-shot read** of the result buffer region for all cores via batch read
3. **Parse locally** — for each core's result data, extract per-config counts using the `PerConfigResult` struct

```cpp
// Read all sender cores on this device in one shot
auto read_op = fixture->initiate_read_buffer_from_cores(
    coord, sender_cores, result_addr, result_read_size);
fixture->barrier_reads();
auto core_results = fixture->complete_read_buffer_from_cores(read_op);

// Parse per-config results for each core
for (const auto& [core, data] : core_results) {
    const auto& sender = test_device.get_senders().at(core);
    auto per_config = parse_per_config_results(data, sender.configs_.size());
    // ... update endpoint_states_ for each config
}
```

The `result_read_size` must cover the per-config region for the core with the most configs:
```cpp
uint32_t num_configs_max = get_max_configs_across_cores(test_device);
uint32_t result_read_size = PER_CONFIG_RESULT_BASE_WORD_INDEX * sizeof(uint32_t)
                          + num_configs_max * sizeof(PerConfigResult);
```

#### Parsing Layer (`tt_fabric_test_progress_monitor.hpp/.cpp`)

Uses the same array-cast pattern as the kernel side — cast the `uint32_t*` result data at the per-config base into a `const PerConfigResult*` and index directly:

```cpp
struct ParsedConfigProgress {
    uint64_t packets_processed = 0;
    // Extensible: add cycles, etc. as PerConfigResult grows
};

inline std::vector<ParsedConfigProgress> parse_per_config_results(
    const std::vector<uint32_t>& result_data, uint8_t num_configs) {
    std::vector<ParsedConfigProgress> results(num_configs);
    auto* per_config = reinterpret_cast<const PerConfigResult*>(
        result_data.data() + PER_CONFIG_RESULT_BASE_WORD_INDEX);
    for (uint8_t i = 0; i < num_configs; i++) {
        results[i].packets_processed =
            static_cast<uint64_t>(per_config[i].packets_high) << 32 | per_config[i].packets_low;
    }
    return results;
}
```

This parsing layer is the single host-side point where the result buffer layout is interpreted. If `PerConfigResult` gains new fields, only this function and the kernel-side `write_per_config_result()` need updating — all indexing adapts automatically.

### Cross-Host Device Skew: Barrier + Multi-Round Detection

In a multi-host environment, there is an inherent startup skew between hosts. Even though there's a barrier before program launch (`fixture->barrier()`), the actual kernel execution start times on different hosts can differ.

**The problem:** If the hung threshold is 30s and host B's receiver monitoring starts before host A's sender has even launched, host B might see the receiver at 0 packets for 30s and declare it hung — but the sender hasn't even started yet.

#### Solution: Post-Launch Distributed Barrier

**Insert a distributed barrier between `launch_programs()` and the start of progress monitoring.** This synchronizes all hosts so they begin polling from a common baseline.

Current flow (`test_tt_fabric.cpp`):
```
fixture->barrier();              // pre-launch sync (already exists)
test_context.launch_programs();  // async dispatch to devices
test_context.wait_for_programs_with_progress();  // monitoring + wait
```

New flow:
```
fixture->barrier();              // pre-launch sync (already exists)
test_context.launch_programs();  // async dispatch to devices
fixture->barrier();              // ← NEW: post-launch sync — all hosts have dispatched
test_context.wait_for_programs_with_progress();  // monitoring starts at ~same wall time
```

The post-launch barrier ensures that by the time any host begins polling, every other host has at least dispatched its programs. This eliminates the class of false positives where host A hasn't even launched yet while host B is already checking its receivers.

**Residual skew after barrier:** The barrier only guarantees that `launch_programs()` returned on every host. There is still a small skew between when the kernels actually begin executing on different devices (dispatch queue latency, fabric startup, etc.). This residual skew is typically sub-second — well within any reasonable hung threshold (30s).

#### Solution: Multi-Round Hung Confirmation

Instead of declaring a sender/receiver hung after a single threshold expiry, require **N consecutive rounds of zero progress** before confirming a hang. This filters out transient stalls, residual post-barrier skew, and momentary fabric congestion.

**Parameters (configurable via `ProgressMonitorConfig`):**

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `hung_confirmation_rounds` | 3 | Number of consecutive stall polls before confirming hung |
| `poll_interval_seconds` | 2 | Sleep between polls |
| `hung_threshold_seconds` | 30 | Minimum wall-clock time of no progress before hung is even considered |

**Logic per poll cycle:**

1. Read `current_packets` for each endpoint
2. If `current_packets > last_packet_count` → progress detected, reset `consecutive_stall_rounds = 0`
3. If `current_packets == last_packet_count`:
   - If wall-clock time since last progress < `hung_threshold_seconds` → skip (too early)
   - Otherwise → increment `consecutive_stall_rounds`
4. If `consecutive_stall_rounds >= hung_confirmation_rounds` → **confirmed hung**

**Example timeline** with defaults (`poll_interval=2s`, `hung_threshold=30s`, `confirmation_rounds=3`):

```
t=0s   : post-launch barrier completes, monitoring starts
t=0-28s: polls every 2s, sender shows 0 packets (but threshold not reached yet)
t=30s  : first stall round counted (round 1/3)
t=32s  : second stall round counted (round 2/3)
t=34s  : third stall round counted (round 3/3) → CONFIRMED HUNG
```

This gives a 34s total window — long enough to absorb any reasonable startup delay, and the 3-round confirmation ensures we're not reacting to a transient.

### Hang Exit Strategy: Control Flow on Confirmed Hang

The design so far describes a monitoring → MPI exchange → report pipeline, but the current `wait_for_programs_with_progress()` control flow blocks at two points that prevent the reporting pipeline from ever executing on a real hang:

1. **`poll_until_complete()` never exits on hang.** The loop condition is `!programs_complete`, which requires all devices to report full packet counts. Hung devices never reach that threshold, so the loop runs indefinitely. The multi-round hung confirmation detects the hang and records it, but has no mechanism to break out of the polling loop.

2. **`fixture_->wait_for_programs()` blocks after polling.** Even if polling exits early, the next line calls `MeshCommandQueue::finish()`, which blocks until all device programs terminate. Hung programs don't terminate.

```
// Current control flow — both block indefinitely on hang:
void TestContext::wait_for_programs_with_progress() {
    ...
    monitor.poll_until_complete();       // ← blocks: hung devices never complete
    fixture_->wait_for_programs();       // ← blocks: MeshCommandQueue::finish() waits for hung programs
}
```

**This must be resolved for the reporting pipeline to be reachable.** The required control flow change:

```
// New control flow for pairwise validation:
void TestContext::wait_for_programs_with_progress() {
    ...
    auto result = monitor.poll_until_complete_or_hung();

    if (result == MonitorResult::ALL_COMPLETE) {
        fixture_->wait_for_programs();   // safe — all programs finished
        // → no MPI exchange needed, generate PASS report
    } else {
        // HUNG DETECTED — skip wait_for_programs(), go directly to reporting
        // fixture_->wait_for_programs() is intentionally NOT called — it would block
        monitor.exchange_and_report();   // MPI exchange → collation → two-tier reports
        // → clean exit with error, log file paths in output
    }
}
```

**Decision needed: how to exit on confirmed hang.** The options are:

- **`TT_THROW` after reporting:** Write both reports, log their file paths to stdout, then throw. The test framework catches the exception and reports failure. Clean, CI-friendly, and ensures the reports are always written before exit. The catch handler in `test_tt_fabric.cpp` already exists for other failure modes.

- **Non-zero exit code without exception:** `monitor.exchange_and_report()` returns a failure code, `wait_for_programs_with_progress()` returns it, and the test's main loop checks it. More explicit control flow but requires threading a return value through several layers.

- **`--wait-on-hang` flag for interactive use:** After writing reports, instead of exiting, log a "HUNG DETECTED — reports written to {paths}, process is waiting for manual termination" banner and block. The operator inspects the live system, then kills the process. This is complementary to the above — the flag selects between exit-after-report and wait-after-report.

The `TT_THROW` approach is the simplest for the initial implementation: the existing test runner already handles exceptions, and it naturally prevents execution from falling through to `wait_for_programs()`. The throw message should include the report file paths so the operator knows where to look.

**What about device cleanup?** Skipping `wait_for_programs()` means hung device programs are still running when the host process exits. This is acceptable for a diagnostic test — the device reset that happens during `close_devices()` (or process teardown) will clean up. The pairwise validation test is not a steady-state workload; it's a one-shot diagnostic where a hung detection means the test is over.

### MPI Exchange: Two-Phase Sparse Transfer

MPI exchange happens only after local monitoring has concluded. Successful flows are not exchanged — only hung endpoint records cross the wire.

**Why not `gather()`?** The `DistributedContext::gather()` wraps `MPI_Gather`, which requires every rank to send the same number of bytes. Since each host may have a different number of hung endpoints (including zero), a single `gather()` would require padding every rank's buffer to a fixed maximum — wasting bandwidth and requiring an arbitrary ceiling constant. There is no `MPI_Gatherv` wrapper in the current `DistributedContext` API.

**Solution: two-phase transfer using `gather()` for counts + `send()`/`recv()` for data.** Both operations are already available in the `DistributedContext` API. Each rank sends only its actual hung records — no padding, no fixed-size ceilings.

#### Wire Format

The sparse failure payload carries only runtime observation data. Static flow metadata comes from the local `flow_descriptors_` lookup on rank 0.

```cpp
struct HungEndpointRecord {
    FlowUid flow_uid;
    EndpointId endpoint_id;

    uint64_t packets_processed = 0;
    uint64_t packets_expected = 0;
    uint32_t stall_seconds = 0;
    uint16_t confirmation_rounds = 0;
};
```

For MPI serialization, a flat primitive-only wire struct keeps the transport path lean and low-risk:

```cpp
struct HungEndpointWireRecord {
    uint32_t flow_uid = 0;
    uint8_t role = 0;  // EndpointRole

    uint32_t mesh_id = 0;
    uint32_t chip_id = 0;
    uint32_t core_x = 0;
    uint32_t core_y = 0;
    uint16_t config_idx = 0;

    uint64_t packets_processed = 0;
    uint64_t packets_expected = 0;
    uint32_t stall_seconds = 0;
    uint16_t confirmation_rounds = 0;
};
```

This is intentionally lean:

- no repeated source/destination metadata
- no repeated topology metadata
- no repeated physical metadata
- no interpretations

#### Two-Phase Transfer Protocol

**Phase 1: Gather counts (uniform collective).** Each rank sends a single `uint32_t` — its local hung endpoint count. This is uniform across all ranks (4 bytes each), so `gather()` works directly. After this phase, rank 0 knows exactly how many records to expect from each rank.

**Phase 2: Point-to-point data (variable-length).** Each non-root rank with hung endpoints sends its `HungEndpointWireRecord` array to rank 0 via `send()`. Rank 0 iterates ranks sequentially, calling `recv()` only for ranks that reported a non-zero count. Ranks with zero hung endpoints skip the send entirely — no mismatched send/recv, no deadlock.

```cpp
const auto& ctx = fixture_->get_distributed_context();
const int world_size = *ctx.size();
const Rank root{0};
const Tag data_tag{42};

// Convert local records to wire format
std::vector<HungEndpointWireRecord> wire_records = to_wire_format(local_hung_records_);

// Phase 1: gather counts (uniform — 4 bytes per rank)
uint32_t local_count = static_cast<uint32_t>(wire_records.size());
std::vector<uint32_t> all_counts(world_size, 0);

ctx.gather(
    {reinterpret_cast<std::byte*>(&local_count), sizeof(local_count)},
    {reinterpret_cast<std::byte*>(all_counts.data()), all_counts.size() * sizeof(uint32_t)},
    root);

// Phase 2: point-to-point data transfer (variable-length, only non-empty)
if (ctx.rank() != root) {
    if (local_count > 0) {
        ctx.send(
            {reinterpret_cast<std::byte*>(wire_records.data()),
             wire_records.size() * sizeof(HungEndpointWireRecord)},
            root,
            data_tag);
    }
} else {
    // Rank 0: collect its own local records first
    std::vector<HungEndpointWireRecord> all_records(wire_records.begin(), wire_records.end());

    // Then receive from each non-root rank that has data
    for (int r = 1; r < world_size; ++r) {
        if (all_counts[r] > 0) {
            std::vector<HungEndpointWireRecord> remote_records(all_counts[r]);
            ctx.recv(
                {reinterpret_cast<std::byte*>(remote_records.data()),
                 remote_records.size() * sizeof(HungEndpointWireRecord)},
                Rank{r},
                data_tag);
            all_records.insert(all_records.end(), remote_records.begin(), remote_records.end());
        }
    }

    // Collate and generate reports
    collate_and_report(all_records);
}
```

**Why this is safe:**

- **No deadlock:** Rank 0 only calls `recv()` for ranks where `all_counts[r] > 0`, and those ranks always call `send()`. Ranks with zero records skip both sides.
- **No padding:** Each rank sends exactly the bytes it needs. A rank with 3 hung endpoints sends `3 * sizeof(HungEndpointWireRecord)`. A rank with 0 sends nothing.
- **No fixed-size ceiling:** The count gather tells rank 0 the exact size to allocate per rank. No `MAX_HUNG_RECORDS_PER_RANK` constant needed.
- **Bounded latency:** For the data sizes involved (worst case: hundreds of records × ~36 bytes = a few KB per rank), `MPI_Send` buffers internally rather than blocking on a matching recv. Non-root ranks return immediately after the send.

#### Collation on Rank 0

Rank 0 reconstructs the global view from the gathered wire records. Since every host built the same deterministic `flow_descriptors_` registry, rank 0 can look up `flow_descriptors_[flow_uid]` to recover the full flow metadata (source, destination, link_id, etc.) for any hung endpoint.

```cpp
struct CollatedFlowState {
    FlowDescriptor descriptor;                       // from flow_descriptors_[flow_uid]
    std::vector<HungEndpointRecord> hung_endpoints;  // from gathered data
};

std::unordered_map<FlowUid, CollatedFlowState> collated_;
```

From this collated map, both reports are independent formatting passes over the same data.

### Failure Reports: Two-Tier Observational Design

Two separate report files serve different audiences. Both report only observed data — no inferences, no root-cause guesses, no recommended actions.

- **Summary** (`pairwise_validation_summary.log`) — for datacenter operators. Flow-grouped. Uses operator vocabulary: host, tray, ASIC, eth channel, port type, port ID. No config indices, no flow UIDs, no core coordinates.
- **Detailed** (`pairwise_validation_detailed.log`) — for the fabric team. Per-host, per-endpoint breakdown with flow_uid cross-references, config indices, core coordinates, packet counts, stall durations.

Both files are written to `{root_dir}/generated/fabric/` by rank 0 after MPI gather.

#### Physical Metadata Resolution

Both reports need operator-friendly physical location data for each endpoint. Two existing APIs provide this:

1. **`format_device_label(node_id)`** — already used in the progress monitor. Returns `(mesh_id, chip_id) [hostname(Rank)/Tray/Node]` using `PhysicalSystemDescriptor::get_tray_id()`, `get_asic_location()`, `get_host_name_for_asic()`.

2. **`Board::get_port_for_asic_channel(AsicChannel{asic_location, ChanId(eth_channel)})`** — from `tools/scaleout/board/board.hpp`. Returns `Port{port_type, port_id}` where `PortType` is one of:
   - `TRACE` — internal board trace (intra-galaxy)
   - `QSFP_DD` — external QSFP-DD cable
   - `WARP100` / `WARP400` — warp connectors
   - `LINKING_BOARD_1/2/3` — linking board connections

   The `PortType` is ground truth from the board definition — no inference from mesh IDs. `PortId` identifies the specific physical port on the board.

3. **`get_fabric_route(src_node, dst_node, src_eth_channel)`** — from the control plane. Returns `[(dst_node, dst_eth_channel)]` for 1-hop links. This gives us the remote side's eth channel, enabling full two-sided link descriptions.

The pattern for resolving a link's physical metadata follows `cluster_validation_utils.cpp` (`ConnectionInfo` struct, `generate_port_info()`). The same `PhysicalSystemDescriptor` + `Board` APIs are used — the pairwise validation report just needs to call them.

#### Report Generation Flow

1. **Each host generates its local data** (in-memory, not to disk): per-endpoint hung/healthy status for all local senders and receivers. Single-threaded — the monitor runs in the calling thread of `wait_for_programs_with_progress()`.

2. **Barrier** after monitoring completes (ensures all hosts have finished before gather).

3. **Two-phase transfer to rank 0:**
   - Phase 1: each host `gather()`s its hung endpoint count (uniform 4 bytes per rank)
   - Phase 2: each host with hung endpoints `send()`s its `HungEndpointWireRecord` array to rank 0; rank 0 `recv()`s only from ranks with non-zero counts
   - Rank 0 deserializes and collates by `flow_uid`

4. **Rank 0 generates both reports** from the collated data:
   - **Summary**: flow-grouped + physical metadata resolution + write to `generated/fabric/{summary_filename}`
   - **Detailed**: per-host, per-endpoint breakdown + write to `generated/fabric/{detail_filename}`
   - Directory is created if it doesn't exist (`std::filesystem::create_directories`)

**Important: rank 0 only has runtime data for hung endpoints.** For a flow where the sender is hung but the receiver completed normally (or vice versa), rank 0 sees the hung endpoint's packet counts and stall duration, plus the flow's static metadata from `flow_descriptors_`. It does **not** have the healthy counterpart's runtime progress — that data was never exchanged. Reports must reflect this: show hung endpoint data where available, and mark the counterpart as "no hung data reported" rather than fabricating success state.

#### Summary Report Format

```
================================================================
 PAIRWISE VALIDATION — LINK HEALTH SUMMARY
 Timestamp: 2026-04-02T15:30:00Z
 Result: FAIL — 3 hung flows out of 560
================================================================

HUNG FLOWS (3):

  [1] Flow src: [host1] Tray 0 / ASIC 5 / Eth Ch 4 / TRACE Port 3
      Flow dst: [host1] Tray 0 / ASIC 6 / Eth Ch 2 / TRACE Port 4
      Sender: 0/1000 packets | no progress for 45s
      Receiver: 0/1000 packets | no progress for 45s

  [2] Flow src: [host1] Tray 0 / ASIC 7 / Eth Ch 8 / QSFP_DD Port 10
      Flow dst: [host2] Tray 1 / ASIC 8 / Eth Ch 8 / QSFP_DD Port 10
      Sender: 0/1000 packets | no progress for 40s
      Receiver: 0/1000 packets | no progress for 40s

  [3] Flow src: [host1] Tray 0 / ASIC 5 / Eth Ch 6 / LINKING_BOARD_1 Port 1
      Flow dst: [host1] Tray 0 / ASIC 6 / Eth Ch 6 / LINKING_BOARD_1 Port 2
      Sender: 0/1000 packets | no progress for 35s
      Receiver: (no hung data reported)

CLUSTER HEALTH: 3/560 flows have hung endpoints
================================================================
```

When no hung endpoints are detected:

```
================================================================
 PAIRWISE VALIDATION — LINK HEALTH SUMMARY
 Timestamp: 2026-04-02T15:30:00Z
 Result: PASS — No hung endpoints detected across 560 flows
================================================================
```

#### Detailed Report Format

```
================================================================
 PAIRWISE VALIDATION — DETAILED DIAGNOSTIC REPORT
 Timestamp: 2026-04-02T15:30:00Z
 Configuration: hung_threshold=30s, confirmation_rounds=3, poll_interval=2s
================================================================

--- Host: host1 (Rank 0) ---

HUNG ENDPOINTS (4 of 560):

  [1] Sender endpoint
      flow_uid: 42
      Configured: (0,5) [host1(R0)/T0/N5] → (0,6) [host1(R0)/T0/N6]
      Core: (3,2) | config_idx: 2 | Eth Ch: 4
      Packets: 0/1000 | Stalled for: 45s | Confirmed: 3 rounds

  [2] Receiver endpoint
      flow_uid: 42
      Configured: (0,5) [host1(R0)/T0/N5] → (0,6) [host1(R0)/T0/N6]
      Core: (2,3) | config_idx: 0 | Eth Ch: 2
      Packets: 0/1000 | Stalled for: 45s | Confirmed: 3 rounds

  [3] Sender endpoint
      flow_uid: 55
      Configured: (0,5) [host1(R0)/T0/N5] → (0,4) [host1(R0)/T0/N4]
      Core: (4,2) | config_idx: 3 | Eth Ch: 7
      Packets: 500/1000 | Stalled for: 32s | Confirmed: 3 rounds

  [4] Receiver endpoint (sender on remote host)
      flow_uid: 200
      Configured: (1,0) [host2(R1)/T1/N0] → (0,6) [host1(R0)/T0/N6]
      Core: (2,3) | config_idx: 1
      Packets: 0/500 | Stalled for: 38s | Confirmed: 3 rounds

LOCAL SUMMARY: 278/280 sender endpoints OK, 279/280 receiver endpoints OK

--- Host: host2 (Rank 1) ---

HUNG ENDPOINTS (0 of 560):
  (none)

LOCAL SUMMARY: 280/280 sender endpoints OK, 280/280 receiver endpoints OK

... (other hosts) ...

================================================================
 GLOBAL SUMMARY
 Total sender endpoints:   2240 (2238 OK, 2 hung)
 Total receiver endpoints: 2240 (2239 OK, 1 hung)
================================================================
```

### CLI Options and Activation Model

#### When is per-endpoint granular monitoring active?

The existing progress monitoring has two levels:
- `--show-progress` — enables the aggregate progress line + device-level hung detection
- `--show-workers` — additionally logs per-device sender/receiver counts at startup

The new per-endpoint granular monitoring adds a third level, activated via `--show-progress-detail`. This is a pattern-independent flag — it works for `neighbor_exchange`, `all_to_all`, or any other pattern. For the pairwise validation use case, operators simply pass `--show-progress-detail` on the command line.

When `--show-progress-detail` is active, the monitor:
- Tracks per-endpoint progress (instead of aggregated per device)
- Uses multi-round hung confirmation per endpoint
- Produces the two-tier failure reports to log files
- Prints per-endpoint hung warnings to stdout

When only `--show-progress` is active (no detail), the monitor behaves as it does today: aggregate per-device progress, device-level hung detection.

#### New CLI options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--show-progress-detail` | bool | `false` | Enable per-endpoint granular monitoring |
| `--validation-summary-file` | string | `pairwise_validation_summary.log` | Filename for the operator summary report (written to `generated/fabric/`) |
| `--validation-detail-file` | string | `pairwise_validation_detailed.log` | Filename for the detailed diagnostic report (written to `generated/fabric/`) |
| `--hung-confirmation-rounds` | uint32 | `3` | Number of consecutive stall polls before confirming hung |

Both report files are written to the standard generated output directory:
```
{MetalContext::rtoptions().get_root_dir()}/generated/fabric/{filename}
```
This is the same directory used by YAML dumps (`built_tests.yaml`), bandwidth CSVs, telemetry files, etc. Options specify only the **filename**, not a full path — consistent with `--built-tests-dump-file`.

New constants in `tt_fabric_test_constants.hpp`:
```cpp
constexpr const char* DEFAULT_VALIDATION_SUMMARY_FILE = "pairwise_validation_summary.log";
constexpr const char* DEFAULT_VALIDATION_DETAIL_FILE = "pairwise_validation_detailed.log";
```

These integrate with the existing `CmdlineParser` pattern:

```cpp
// In CmdlineParser (tt_fabric_test_config.hpp / .cpp)
bool show_progress_detail();
std::string get_validation_summary_file();
std::string get_validation_detail_file();
uint32_t get_hung_confirmation_rounds();
```

And flow into `ProgressMonitorConfig`:

```cpp
struct ProgressMonitorConfig {
    bool enabled = false;
    bool show_workers = false;
    bool granular = false;                                          // ← NEW: per-endpoint detail mode
    uint32_t poll_interval_seconds = 2;
    uint32_t hung_threshold_seconds = 30;
    uint32_t hung_confirmation_rounds = 3;                          // ← NEW
    std::string summary_file = DEFAULT_VALIDATION_SUMMARY_FILE;     // ← NEW (filename only)
    std::string detail_file = DEFAULT_VALIDATION_DETAIL_FILE;       // ← NEW (filename only)
};
```

Config setup in `test_tt_fabric.cpp`:

```cpp
if (cmdline_parser.show_progress()) {
    ProgressMonitorConfig progress_config;
    progress_config.enabled = true;
    progress_config.poll_interval_seconds = cmdline_parser.get_progress_interval();
    progress_config.hung_threshold_seconds = cmdline_parser.get_hung_threshold();
    progress_config.hung_confirmation_rounds = cmdline_parser.get_hung_confirmation_rounds();
    progress_config.summary_file = cmdline_parser.get_validation_summary_file();
    progress_config.detail_file = cmdline_parser.get_validation_detail_file();
    progress_config.granular = cmdline_parser.show_progress_detail();

    test_context.enable_progress_monitoring(progress_config);
}
```

### Stdout Progress Display

The aggregate progress line (carriage-return style) continues as before but with endpoint-level counts when granular mode is active:

```
Progress: 95.2% (266/280 flows) | 12K/s | ETA: 3s | Devices: 30/32 done
```

Per-endpoint hung warnings are printed inline (with a newline) when an endpoint is **confirmed** hung (after all confirmation rounds) — with full flow context instead of device-level.

When granular mode is not active, the display is unchanged from today's device-level output.

### Implementation Summary

#### Kernel-Level Changes

| Change | Where | Scope |
|--------|-------|-------|
| Define `PerConfigResult` struct + `PER_CONFIG_RESULT_BASE_WORD_INDEX` | `tt_fabric_test_kernels_utils.hpp` | Kernel-local struct (matches host-side copy) |
| Add `get_per_config_results()` + `write_per_config_result()` helpers | `tt_fabric_test_kernels_utils.hpp` | Array-cast accessor + single-entry writer |
| Add periodic + final per-config writes to sender kernel | `tt_fabric_test_sender.cpp` | ~10 lines across progress + completion paths |
| Add periodic + final per-config writes to receiver kernel | `tt_fabric_test_receiver.cpp` | ~12 lines, mirrors sender pattern |

#### Host-Level Changes

| Change | Where | Scope |
|--------|-------|-------|
| Add `get_all_neighbor_node_ids(direction)` returning `vector<FabricNodeId>` | `tt_fabric_test_common.hpp` (`TestFixture`) | New function for multi-Z neighbor enumeration |
| Fix `get_neighbor_node_id_or_nullopt()`: fatal on Z, NESW-only | `tt_fabric_test_common.hpp` (`TestFixture`) | Prevent misuse on multi-Z topologies; callers must use `get_all_neighbor_node_ids()` for Z |
| Fix `get_directional_neighbor_pairs()`: per-direction Z branching + `FabricContext::routing_directions` | `tt_fabric_test_common.hpp` (`TestFixture`) | Z always uses control plane path; replaces `{N,S,E,W}` |
| Replace `{N,S,E,W}` in `get_mesh_adjacency_map()` with `FabricContext::routing_directions` | `tt_fabric_test_common.hpp` | Phase 1 — Z neighbor discovery |
| *(Deferred)* Replace `{N,S,E,W}` in `get_hops_to_nearest_neighbors()` | `tt_fabric_test_common.hpp` | Only feeds sync path; its `unordered_map<RoutingDirection, uint32_t>` collapses multi-Z by construction — part of Z sync uplift, not Phase 1 |
| Add `dst_node_id` to `ConnectionKey` | `tt_fabric_test_device_setup.hpp` | Multi-Z disambiguation — connections keyed by destination, not just direction |
| Add `dst_node_id` param to `register_fabric_connection()` + update 3 call sites | `tt_fabric_test_device_setup.cpp` | Sender: `config.dst_node_ids[0]`; Receiver: `credit_info.sender_node_id`; Sync: `sender_config.dst_node_ids[0]` |
| Fix `generate_connection_args_for_core()`: read `key.dst_node_id` | `tt_fabric_test_device_setup.cpp` | Eliminates lossy re-derivation via `get_neighbor_node_id()` |
| Fix `create_mux_kernels()`: read `connection_key.dst_node_id` | `tt_fabric_test_device_setup.cpp` | Same fix — mux destination was also re-derived from direction only |
| Add `TT_FATAL` guard for Z in `NeighborExchange` sync | `tt_fabric_test_common.hpp` | Prevents silent misconfiguration; sync deferred for Z topologies |
| Add `FlowDescriptor` struct + `flow_descriptors_` vector to `TestContext` | `tt_fabric_test_context.hpp/.cpp` | Per-host deterministic flow registry |
| Add `flow_uid` field to `TestTrafficSenderConfig` + `TestTrafficReceiverConfig` | `tt_fabric_test_traffic.hpp` | Host-only field, not serialized to kernel args |
| Build `FlowDescriptor` + stamp `flow_uid` in `add_traffic_config()` | `tt_fabric_test_context.cpp` | Before local ownership filtering |
| Define `PerConfigResult` struct + `PER_CONFIG_RESULT_BASE_WORD_INDEX` (host copy) | `tt_fabric_test_memory_map.hpp` | Host-local struct (matches kernel-side copy), `static_assert` on sizeof |
| Add `get_per_config_region_size_bytes()` + `validate_per_config_capacity()` | `CommonMemoryMap` in `tt_fabric_test_memory_map.hpp` | Capacity validation during kernel creation |
| Add `EndpointId`, `EndpointHungState`, `EndpointProgressState` structs | `tt_fabric_test_progress_monitor.hpp` | Endpoint-first monitoring model |
| Add `ParsedConfigProgress` struct + `parse_per_config_results()` | `tt_fabric_test_progress_monitor.hpp/.cpp` | Array-cast parsing layer for result buffer |
| Add `granular`, `hung_confirmation_rounds`, `summary_file`, `detail_file` to config | `ProgressMonitorConfig` | New fields |
| Add `DEFAULT_VALIDATION_SUMMARY_FILE` + `DEFAULT_VALIDATION_DETAIL_FILE` constants | `tt_fabric_test_constants.hpp` | New constants |
| Add `--show-progress-detail`, `--validation-summary-file`, `--validation-detail-file`, `--hung-confirmation-rounds` | `CmdlineParser` | New CLI options |
| Build `endpoint_states_` from local sender/receiver configs during monitor construction | `TestProgressMonitor` constructor | Endpoint registration scan |
| Batch read result buffers per device (single-shot) | `poll_device_senders()` / `poll_device_receivers()` | Use `initiate_read_buffer_from_cores()` + parse locally |
| Multi-round hung confirmation per endpoint | `check_for_hung_endpoints()` | Per-endpoint granularity + round counting |
| Post-launch distributed barrier | `test_tt_fabric.cpp` or `wait_for_programs_with_progress()` | 1 line: `fixture->barrier()` after `launch_programs()` |
| Change `poll_until_complete()` → `poll_until_complete_or_hung()` returning `MonitorResult` | `TestProgressMonitor` | Early exit on confirmed global hang (all active endpoints either complete or confirmed hung) |
| Change `wait_for_programs_with_progress()` control flow on hang | `TestContext` | Skip `wait_for_programs()`, proceed to MPI exchange/report, then `TT_THROW` (or block if `--wait-on-hang`) |
| Add `--wait-on-hang` CLI option | `CmdlineParser` | Selects block-after-report vs throw-after-report |
| Local failure data accumulation (single-threaded) | `TestProgressMonitor` | `std::vector<HungEndpointRecord> local_hung_records_` |
| Add `HungEndpointWireRecord` flat struct for MPI serialization | `tt_fabric_test_progress_monitor.hpp` | Primitive-only wire format |
| Two-phase MPI transfer (gather counts + send/recv data) + two-tier report generation | New methods `write_summary_report()` + `write_detailed_report()` | Uses `gather()` for counts, `send()`/`recv()` for variable-length data, `Board` API, `PhysicalSystemDescriptor` |
| Summary: flow-grouped with physical metadata | `write_summary_report()` | `PortType`, `PortId` via `Board::get_port_for_asic_channel()` |
| Detailed: per-host, per-endpoint breakdown via `flow_uid` | `write_detailed_report()` | Per-host breakdown + per-endpoint hung data |

## Open Questions

1. **Hang termination policy:** This is no longer a deferred question — it is a prerequisite for the reporting pipeline to work. The current `wait_for_programs_with_progress()` blocks indefinitely on hang at two points (`poll_until_complete()` and `MeshCommandQueue::finish()`). See "Hang Exit Strategy" section for the concrete control flow change. Recommended approach: `TT_THROW` after writing reports (CI-friendly), with a `--wait-on-hang` flag for interactive bringup. Decision must be made before Phase 3 implementation.
2. **Machine-parseable output:** The current report is human-readable text. Should we also produce JSON/CSV for automated tooling? Low priority — can be added later as a `--report-format` option without architectural changes.
3. **CI integration:** Should this test run in regular CI or be a dedicated bringup/diagnostic tool? This affects default timeout values, exit codes, and naming conventions.
4. **`EndpointIdHash` implementation:** The `std::unordered_map<EndpointId, ...>` requires a hash function. The hash needs to be collision-resistant for `(role, mesh_id, chip_id, core.x, core.y, config_idx)`. Exact implementation deferred to coding phase.
5. **Receiver-only hang collation:** The data model already supports receiver-role `EndpointId` records, and the MPI exchange gathers hung endpoints regardless of role. The remaining question is how rank 0 should present a receiver-only hang (where the corresponding sender is on a different host and did not report as hung): show it as an unpaired receiver entry in the detailed report, or attempt cross-referencing with the sender's `FlowDescriptor`. The former is simpler and consistent with the "observational only" principle; the latter adds diagnostic depth. Decision can be deferred to Phase 4 implementation without architectural changes.

## Implementation Phases

### Phase 1: Z-Link Plumbing + Neighbor Exchange Fix (No monitor changes)
- Add `get_all_neighbor_node_ids(direction)` to `TestFixture` for multi-Z neighbor enumeration
- Fix `get_neighbor_node_id_or_nullopt()`: make fatal on Z direction, restrict to NESW-only
- Fix `get_directional_neighbor_pairs()`: per-direction Z branching (Z always control plane) + use `FabricContext::routing_directions`
- Replace `{N,S,E,W}` with `FabricContext::routing_directions` in `get_mesh_adjacency_map()` (note: `get_hops_to_nearest_neighbors()` is **not** part of Phase 1 — it only feeds the sync path which is deferred for Z; see "Gap: Sync Model" section)
- Replace `NUM_DIRECTIONS` with connection-count-aware `get_max_connections_per_device()` (4 + `MAX_Z_NEIGHBORS=2`) in allocator; raise `MAX_NUM_FABRIC_CONNECTIONS` to 6 on BH
- Add `dst_node_id` to `ConnectionKey`; add `dst_node_id` param to `register_fabric_connection()` and update all 3 call sites (sender, receiver credit, sync); fix `generate_connection_args_for_core()` and `create_mux_kernels()` to read from key instead of re-deriving
- Add `TT_FATAL` guard in `NeighborExchange` sync for Z-link topologies (sync deferred for Z)
- Add pairwise validation YAML config (uses `neighbor_exchange` pattern with `num_links: 2`)
- **Testable with existing aggregate progress monitor** — validates Z neighbors are discovered and pairs are generated correctly

### Phase 2: Flow Registry + Per-Config Kernel Progress
- Add `FlowDescriptor` struct and `flow_descriptors_` vector to `TestContext`
- Add `flow_uid` field to `TestTrafficSenderConfig` and `TestTrafficReceiverConfig`
- Build `FlowDescriptor` + stamp `flow_uid` in `add_traffic_config()` (before local ownership checks)
- Define `PerConfigResult` struct on both sides (kernel utils + memory map)
- Add `get_per_config_results()` + `write_per_config_result()` kernel helpers
- Add periodic + final per-config writes to sender kernel
- Add periodic + final per-config writes to receiver kernel
- Add `validate_per_config_capacity()` to `CommonMemoryMap`
- **Testable independently** — can verify flow registry is deterministic and per-config data is written/read correctly

### Phase 3: Endpoint-Level Progress Monitor + Hang Exit Strategy
- Add `EndpointId`, `EndpointHungState`, `EndpointProgressState` structs
- Add `ParsedConfigProgress` + `parse_per_config_results()` parsing layer
- Build `endpoint_states_` from local sender/receiver configs during monitor construction
- Switch polling from per-core aggregate to batched `initiate_read_buffer_from_cores()` + per-config parse
- Update endpoint states from parsed per-config results
- Add multi-round hung confirmation logic per endpoint
- Add `--show-progress-detail`, `--hung-confirmation-rounds` CLI options
- Add post-launch distributed barrier
- Change `poll_until_complete()` to `poll_until_complete_or_hung()` — returns `MonitorResult` enum (ALL_COMPLETE vs HUNG_DETECTED)
- Change `wait_for_programs_with_progress()` control flow: skip `wait_for_programs()` on confirmed hang, proceed directly to MPI exchange/report, then `TT_THROW` with report file paths (or block if `--wait-on-hang`)
- Add `--wait-on-hang` CLI option for interactive bringup mode

### Phase 4: Two-Tier Observational Failure Reports
- Add `HungEndpointWireRecord` flat wire struct for MPI serialization
- Add `--validation-summary-file`, `--validation-detail-file` CLI options
- Implement two-phase MPI transfer: `gather()` for per-rank hung counts (uniform), then `send()`/`recv()` for variable-length wire records (only non-empty ranks)
- Implement rank 0 collation by `flow_uid` using local `flow_descriptors_`
- **Summary report** (`write_summary_report()`):
  - Flow-grouped with physical metadata via `PhysicalSystemDescriptor` + `Board::get_port_for_asic_channel()` (`PortType`, `PortId`)
  - Operator-friendly output: hostname, tray, ASIC, eth channel, port type, port ID
  - Observational only: packets processed/expected, stall duration — no inferences
- **Detailed report** (`write_detailed_report()`):
  - Per-host, per-endpoint breakdown with packet counts and stall durations
  - `flow_uid` cross-referencing for sender-receiver pairing
  - `eth_channel` resolution via control plane
  - `format_device_label()` enrichment per hung endpoint
  - Observational only — no inferences
