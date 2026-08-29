# Plan: stable physical node id `(host, tray, loc)` as a packed POD

**Status:** Implementation plan — **separate change** from FSD / downed-links
**Why split:** Fabric Manager and `generate_rank_bindings` need a stable mapper identity whether or not UMD discovery has run. That is independent of `LinkHealth`. Land this first; the FSD-backed downed-links plan ([`PLAN_fsd_solve_and_downed_links.md`](PLAN_fsd_solve_and_downed_links.md) §5.5) then consumes it.
**Related:** [tt-metal#54752](https://github.com/tenstorrent/tt-metal/pull/54752) (solver is already generic over `GlobalNode`). Production uses a POD, not a `std::tuple` / `std::string`.

---

## 1. Problem

The FSD does not encode ASIC ids. `physical_descriptor_builder` synthesizes `1..N` in `(host_id, tray, loc)` file order. Live discovery uses UMD `chip_unique_id` as `AsicID`. The topology solver erases nodes to dense indices **in iteration order**, which selects among equally valid mappings. Different id spaces → different legal placements. Overlaying UMD ids onto the FSD graph before the solve makes placement wait on discovery, which the place → recover → provision flow cannot do.

We want one value, computed from physical position, that FSD-built and live-discovered descriptors both produce for the same chip.

---

## 2. Decision

`PhysicalNodeId` is a **POD**: fixed-size canonical host bytes + tray + loc. No host hash. Same canonical host string + same tray + same loc ⇒ same id on every path. Fully reversible.

```
PhysicalNodeId = {
    host[kPhysicalHostNameLen],  // NUL-padded canonical hostname, not a hash
    tray,                        // 16 bits
    loc                          // 16 bits
}
```

Do **not** pack the host as `hash32` into a `uint64`. After tray and loc take 32 bits, only four characters would fit; `bh-glx-110-c01u02` is 18. A 128-bit pack is still only 12 host chars. The hash was the only way to stay in 64 bits, and it is the conversion this plan drops.

Do **not** use FSD `host_id` (0..N-1 in file order): live discovery has no FSD index.

Do **not** use `std::string` inside the node (heap, allocator-sensitive compare). Do **not** use `std::array` — the host is a fixed buffer we never pass around on its own; a C array is the pack.

UMD unique ids stay on `ASICDescriptor::umd_unique_id` / `Cluster::get_unique_chip_ids()` — they are not mapper identity.

**Mock + FSD is the load-bearing case.** The name that goes into `host[]` is the **exact** FSD / OS hostname. Mock filenames are not that name — §8.1. The fix is a UMD `hostname:` field — §8.2. Until assets are filled, the aisle-token fallback in §8.2. If mock and FSD pack different `host[]` bytes, the solve diverges again. Work list: §8.3.

---

## 3. Encoding

```cpp
// tt_metal/api/tt-metalium/experimental/fabric/physical_node_id.hpp
namespace tt::tt_metal {

// POSIX HOST_NAME_MAX-class. First DNS label after canonicalization.
// Today's FSD names: "bh-glx-110-c01u02" (18), "sjc1-tt-qb-01" (13).
inline constexpr std::size_t kPhysicalHostNameLen = 64;

struct PhysicalNodeId {
    char host[kPhysicalHostNameLen]{};  // NUL-padded C buffer, not std::array / std::string
    TrayID tray{0};
    ASICLocation loc{0};

    friend bool operator==(const PhysicalNodeId&, const PhysicalNodeId&) = default;
    friend auto operator<=>(const PhysicalNodeId&, const PhysicalNodeId&) = default;
};

// Lowercase, strip a trailing "_<rank>" when hosts are not unique, take the first DNS label.
std::string canonical_host_for_node_id(std::string_view host, bool hosts_unique = true);

PhysicalNodeId make_physical_node_id(
    std::string_view host,
    TrayID tray,
    ASICLocation loc,
    bool hosts_unique = true);

// host is the NUL-trimmed canonical string stored in the id.
struct PhysicalNodeFields {
    std::string host;
    TrayID tray{0};
    ASICLocation loc{0};
};
PhysicalNodeFields decode_physical_node_id(PhysicalNodeId id);

}  // namespace tt::tt_metal
```

`make_physical_node_id`:

1. `canonical = canonical_host_for_node_id(host, hosts_unique)`.
2. **Fatal** if `canonical.empty()` or `canonical.size() >= kPhysicalHostNameLen` (do not truncate).
3. **Fatal** if `tray` or `loc` does not fit in 16 bits.
4. Write `canonical` into `host[]`, leftover bytes `'\0'`. Write tray and loc.

`PhysicalNodeId{}` (all zeros) is unset. `make_*` must never return it (empty host already fatals).

C++20 defaulted `==` / `<=>` compare a C array member element-wise, so the struct stays a POD with no handwritten compare. Do not pass `id.host` as a decaying `char*` — keep the buffer inside the id; callers who need a string use `decode_physical_node_id` or `string_view{id.host}`.

Provide `std::hash<PhysicalNodeId>` that hashes the **whole POD bytes**. That is a container hash only — it is not identity. Do **not** use `std::hash<std::string>` on the host as the node id.

`make_physical_node_id` always canonicalizes. Callers must not pass a mix of FQDN and short names for the same machine — the FSD host-filter canonicalization (downed-links plan §5.2) is the same function. Put `canonical_host_for_node_id` here so there is one implementation.

**Collision:** two different triples cannot produce the same id unless they share the same canonical host bytes, tray, and loc. Duplicate `(host, tray, loc)` in a descriptor is already fatal. There is no hash-collision case.

**Sort order:** `operator<=>` is host bytes, then tray, then loc. Lexicographic hosts (`host10` before `host2`) are a locality-heuristic issue, **not** a stability issue: FSD and live pack the same string, so they still agree. Do not sort by AsicID to "fix" it. Rack order, if wanted later, is `(aisle, rack, u, tray, loc)` from the FSD.

---

## 4. Where it is used

One utility, three producers, mapper as the consumer. Do not invent a second encoding.

| Site | Today | After |
|------|--------|--------|
| `physical_descriptor_builder` | `next_id++` from 1 | `unique_id` / graph key = `make_physical_node_id(fsd_hostname, tray, loc)` |
| `physical_system_discovery` | `AsicID{umd_chip_unique_id}` as graph key | graph key = `make_physical_node_id(host_for_node_id(...), tray, loc)`; keep UMD id in `umd_unique_id` |
| Mock cluster YAML | basename as live host key; `chip_id << 32` as unique id | **host for the id is the FSD hostname** (§8), never the YAML basename |
| `TopologyMapper` adjacency / solver `GlobalNode` | `AsicID` from the descriptor | `PhysicalNodeId` — every site in §6.1 |
| `Cluster::get_unique_chip_ids()` | UMD ids | **unchanged** |

`PhysicalSystemDescriptor::get_asic_id(hostname, tray, loc)` already exists. Implement it as `make_physical_node_id` + lookup, so there is one construction path.

`host_for_node_id` is defined in §8. On silicon with no mock, it is `canonical_host_for_node_id(live_key)`. On mock + FSD it is the aliased FSD hostname.

---

## 5. Type: new strong type vs reuse `AsicID`

**New `PhysicalNodeId` struct.** `AsicID` is already the live UMD unique id in discovery, serialization, exit-node tables, and `verify_topology_mapping` against `cluster.get_unique_chip_ids()`. Overloading it to mean "position pack" silently breaks Check 1/3 (packed id ≠ UMD id). A struct also **cannot** be stuffed into `AsicID` — the earlier "store the uint64 in AsicID for a first slice" path is gone.

Every TopologyMapper map whose key is a physical chip is `PhysicalNodeId` (§6.1) — adjacency, rank maps, lookup tables, broadcast join, `generate_mesh_graph_from_physical_system_descriptor`. `MappedChipInfo` holds both:

- `physical_node_id` — **the** physical key (solver, mapper tables)
- `asic_id` — UMD unique id when known (field only, not an index; unset during FSD-only placement)

Do **not** instantiate the solver on a `tuple<string, TrayID, ASICLocation>` in production. #54752 proves it is legal; the POD is the production node.

---

## 6. File changes

**New:** `tt_metal/api/tt-metalium/experimental/fabric/physical_node_id.hpp` + `tt_metal/fabric/physical_node_id.cpp` (canonicalization + `make` / `decode` / hash). Add the header to `TT_METAL_PUBLIC_API`.

**`physical_descriptor_builder.cpp`:** drop `next_id++`. Every ASIC's graph key is `make_physical_node_id(hostname_of(host_id), TrayID{tray}, ASICLocation{loc})`. Duplicate `(host, tray, loc)` is fatal while filling `key_to_unique_id`.

**`physical_system_discovery.cpp`:** when creating `ASICDescriptor`, set the graph key from position (`host_for_node_id`, §8) and `umd_unique_id` from `chip_unique_ids`. Cross-host gather still carries UMD ids on the wire if they do today — translate to packed ids on ingest using the peer's **resolved** host + tray + loc from the payload, not by hashing the UMD id.

**`topology_mapper.cpp`:** build `PhysicalAdjacencyMap` / `hostname_to_asics` / `asic_positions` from packed ids (host/tray/loc are on the id). `verify_topology_mapping` compares **UMD** ids to `cluster.get_unique_chip_ids()`, never the packed id. ChipId backfill: match `umd_unique_id`, not packed id.

**Tests:** `tests/tt_metal/tt_fabric/fabric_router/test_physical_node_id.cpp` (offline).

No Mesh Graph Descriptor change. No FSD protobuf change. FSD still has no ASIC ids.

---

## 6.1 Code atlas — every solve site keyed by this id

The solver never sees UMD `chip_unique_id` or FSD `1..N`. It sees `PhysicalNodeId` built from `(host, tray, loc)` on **both** the FSD-built PSD and the live/mock PSD. Same keys → same `std::map` iteration → same `GraphIndexData` dense indices → same SAT/DFS → same `FabricNodeId` placement.

`AdjacencyGraph<NodeId>` and `TopologySolver<FabricNodeId, GlobalNode>` are already templates (#54752). This is a type argument change, not a new solver.

**One helper. Every insert into a solver-facing map goes through it.**

```cpp
// physical_node_id.hpp — already in §3
PhysicalNodeId node_id_from_asic_descriptor(
    const ASICDescriptor& d, bool hosts_unique = true) {
    return make_physical_node_id(d.host_name, d.tray_id, d.asic_location, hosts_unique);
}
```

FSD builder and live discovery both fill `ASICDescriptor::{host_name, tray_id, asic_location}`. That is enough. Do not read `d.unique_id` when building a mapper graph.

### Hash (container only)

```cpp
template <>
struct std::hash<tt::tt_metal::PhysicalNodeId> {
    std::size_t operator()(const tt::tt_metal::PhysicalNodeId& id) const noexcept {
        // Whole POD; value-init so padding is zero.
        return std::hash<std::string_view>{}(
            std::string_view(reinterpret_cast<const char*>(&id), sizeof(id)));
    }
};
```

`PhysicalNodeId` is a `std::map` key via defaulted `<=>`. `unordered_map` needs this hash.

---

### What changes (solver identity) vs what stays (UMD)

| File | Symbol | Today | After |
|------|--------|--------|--------|
| `topology_mapper.hpp:51` | `PhysicalAdjacencyMap` | `map<AsicID, vector<AsicID>>` | `map<PhysicalNodeId, vector<PhysicalNodeId>>` |
| `topology_mapper_utils.hpp:44` | same alias (keep **one**) | same | same |
| `topology_mapper_utils.hpp:51` | `AsicPositionMap` | `map<AsicID, AsicPosition>` | `map<PhysicalNodeId, AsicPosition>` |
| `topology_mapper_utils.hpp:100` | `TopologyMappingConfig::asic_positions` | `AsicPositionMap` | follows |
| `topology_mapper_utils.hpp:117` | `hostname_to_asics` | `map<string, set<AsicID>>` | `map<string, set<PhysicalNodeId>>` |
| `topology_mapper_utils.hpp:128-129` | `TopologyMappingResult` | `map<FabricNodeId, AsicID>` + reverse | `PhysicalNodeId` |
| `topology_mapper_utils.hpp:274` | `PhysicalExitNode::asic_id` | `AsicID` | `PhysicalNodeId physical_node_id` |
| `topology_mapper_utils.hpp:394` | `PhysicalMultiMeshGraph::mesh_adjacency_graphs_` | `map<MeshId, AdjacencyGraph<AsicID>>` | `AdjacencyGraph<PhysicalNodeId>` |
| `topology_mapper_utils.hpp:422` | `MeshPhysicalLayout::asics` | `unordered_set<AsicID>` | `unordered_set<PhysicalNodeId>` |
| `topology_mapper_utils.hpp:442+` | `build_physical_multi_mesh_adjacency_graph` | `map<MeshId, map<AsicID, MeshHostRankId>>` | `PhysicalNodeId` |
| `topology_mapper_utils.hpp:591` | `map_multi_mesh_to_physical` / `_n` | `asic_id_to_mesh_rank` | `physical_node_id_to_mesh_rank` |
| `topology_mapper.hpp:60` | `MappedChipInfo` | `asic_id` is the physical key | `physical_node_id` is the physical key; `asic_id` = **UMD field only** |
| `topology_mapper.hpp:428` | `asic_id_to_mapping_` | `unordered_map<AsicID, MappedChipInfo*>` | **replace** with `physical_node_id_to_mapping_` — no AsicID-keyed mapper table |
| `topology_mapper.hpp:365` | `build_asic_id_to_mesh_rank_mapping` | `map<MeshId, map<AsicID, MeshHostRankId>>` | `physical_node_id_to_mesh_rank` |
| `topology_mapper.hpp:439` | `rebuild_host_rank_structs_from_mapping` | `map<MeshId, map<AsicID, …>>` | `PhysicalNodeId` |
| `topology_mapper.cpp:1055` | `rebuild_lookup_maps` | indexes `asic_id_to_mapping_[info.asic_id]` | `physical_node_id_to_mapping_[info.physical_node_id]` |
| `topology_mapper.cpp:178` | `get_fabric_node_id_from_asic_id` | lookup `asic_id_to_mapping_` | `get_fabric_node_id_from_physical_node_id`; old AsicID API walks UMD field if still needed |
| `topology_mapper.cpp:198` | `get_asic_id_from_fabric_node_id` | returns PSD/UMD `asic_id` | add `get_physical_node_id_from_fabric_node_id`; keep UMD getter on the field |
| `topology_mapper.cpp:364` | `get_physical_chip_id_from_asic_id` | `asic_id_to_mapping_` | `get_physical_chip_id_from_physical_node_id` |
| `topology_mapper.cpp:799` | broadcast record key | `serialize_u64(*info.asic_id)` | host + tray + loc (or packed `PhysicalNodeId` bytes) — **not** UMD |
| `topology_mapper.cpp:1646` | `generate_mesh_graph_from_physical_system_descriptor` | `vector<AsicID>` / rank map | `PhysicalNodeId` |
| `topology_solver.hpp` | `AdjacencyGraph<AsicID>`, `MappingConstraints<FabricNodeId, AsicID>` | instantiated on `AsicID` | `PhysicalNodeId` — **no solver source change** |
| `physical_grouping_descriptor_matching.cpp` | `AdjacencyGraph<AsicID>` from PSD | UMD / `1..N` | same `node_id_from_asic_descriptor` |
| `generate_rank_bindings.cpp:186` | `config.hostname_to_asics` / `asic_positions` | `asic_id` from PSD | `node_id_from_asic_descriptor(desc)` |

**Locked: every TopologyMapper map whose key is a physical chip is `PhysicalNodeId`.** There is no second physical index on `AsicID`. UMD id is a *payload* on `MappedChipInfo` (`asic_id` / `umd_unique_id`) for Cluster ChipId and `verify_topology_mapping` Check 1. It is not a map key on the mapper.

**Not a physical-chip index** (unchanged key type):

- `fabric_node_id_to_mapping_` — logical
- `physical_chip_id_to_mapping_` — local Cluster `ChipId` only
- `mesh_host_rank_to_mpi_rank_` — `(MeshId, MeshHostRankId)`
- `Cluster::get_unique_chip_ids()`, `ASICDescriptor::umd_unique_id`
- Logical: `LogicalAdjacencyMap`, `FabricNodeId`, MGD

PSD `asic_descriptors_` / `AsicTopology` may stay `AsicID`-keyed in the first slice (FSD still `1..N`, live still UMD). The mapper **re-keys at the boundary** into `PhysicalNodeId` tables. After that, the mapper never looks up by the PSD's raw `AsicID`.

### TopologyMapper physical indexes (complete)

```cpp
// topology_mapper.hpp — after
struct MappedChipInfo {
    FabricNodeId fabric_node_id{MeshId{0}, 0};
    PhysicalNodeId physical_node_id{};     // THE physical key (host, tray, loc)
    tt::tt_metal::AsicID asic_id{0};       // UMD unique id when known; not an index
    ChipId physical_chip_id = 0;           // local Cluster ChipId
    TrayID tray_id{0};
    ASICLocation asic_location{0};
    MeshCoordinate mesh_coord{0, 0};
    MeshHostRankId mesh_host_rank{0};
    HostName hostname;
    int mpi_rank = -1;
    bool is_mapped = false;
};

class TopologyMapper {
    std::vector<MappedChipInfo> chip_topology_mapping_;
    std::unordered_map<FabricNodeId, MappedChipInfo*> fabric_node_id_to_mapping_;
    std::unordered_map<PhysicalNodeId, MappedChipInfo*> physical_node_id_to_mapping_;  // was asic_id_to_mapping_
    std::unordered_map<ChipId, MappedChipInfo*> physical_chip_id_to_mapping_;

    std::map<MeshId, std::map<PhysicalNodeId, MeshHostRankId>>
        build_physical_node_id_to_mesh_rank_mapping();

    void rebuild_host_rank_structs_from_mapping(
        const std::map<MeshId, std::map<PhysicalNodeId, MeshHostRankId>>&);

    FabricNodeId get_fabric_node_id_from_physical_node_id(PhysicalNodeId) const;
    PhysicalNodeId get_physical_node_id_from_fabric_node_id(const FabricNodeId&) const;
    ChipId get_physical_chip_id_from_physical_node_id(PhysicalNodeId) const;
    // Existing AsicID getters: resolve via umd field on MappedChipInfo, not a dedicated map.
};
```

```cpp
// rebuild_lookup_maps — physical side is PhysicalNodeId only
void TopologyMapper::rebuild_lookup_maps() {
    fabric_node_id_to_mapping_.clear();
    physical_node_id_to_mapping_.clear();
    physical_chip_id_to_mapping_.clear();
    for (auto& info : chip_topology_mapping_) {
        physical_node_id_to_mapping_[info.physical_node_id] = &info;
        if (info.is_mapped) {
            fabric_node_id_to_mapping_[info.fabric_node_id] = &info;
            physical_chip_id_to_mapping_[info.physical_chip_id] = &info;
        }
    }
}
```

Broadcast (`topology_mapper.cpp` ~799): the record identity is hostname + tray + loc (same bytes as `PhysicalNodeId`). Do **not** send UMD `asic_id` as the join key — FSD-only ranks have no UMD id and live ranks would re-split the id space. Receiver: `make_physical_node_id(host, tray, loc)` then `physical_node_id_to_mapping_.at(...)`.

`generate_mesh_graph_from_physical_system_descriptor` (~1646): the `all_asic_ids` vector and `asic_id_to_mesh_rank[MeshId{0}]` become `PhysicalNodeId`. Same for `MappingConstraints<FabricNodeId, PhysicalNodeId>` in that function.

---

### A. Builder — stop synthesizing `1..N` (or ignore those labels at the mapper)

Today (`physical_descriptor_builder.cpp` ~271–328):

```cpp
std::map<AsicKey, uint64_t> key_to_unique_id;
uint64_t next_id = 1;
for (const auto& k : asic_keys) {
    key_to_unique_id[k] = next_id++;   // 1, 2, 3, … file order
}
desc->set_unique_id(unique_id);
desc->set_host_name(hostname_of(host_id));
```

After (if this slice also writes packed ids into the PSD). Mapper does not need this if it always re-keys:

```cpp
auto id = make_physical_node_id(hostname_of(host_id), TrayID{tray}, ASICLocation{loc});
// store *id as unique_id only if the PSD stays uint64-keyed;
// the mapper still calls node_id_from_asic_descriptor and ignores unique_id
```

---

### B. Live / mock discovery — keep UMD on `umd_unique_id`

Today (`physical_system_discovery.cpp` ~680–698):

```cpp
psd.get_asic_descriptors()[src_unique_id] = ASICDescriptor{
    TrayID{tray_id}, asic_location, board_type,
    src_unique_id,          // unique_id == UMD chip_unique_id
    src_chip_id,
    hostname_key};
asic_graph[AsicID{unique_id}] = {};
```

After — graph key for the **mapper** is not this `unique_id`. Discovery can keep UMD as the PSD map key. When the mapper builds adjacency:

```cpp
const auto nid = node_id_from_asic_descriptor(desc);
flat_adj[nid].push_back(node_id_from_asic_descriptor(dst_desc));
```

If discovery later keys the PSD on `PhysicalNodeId`, set `unique_id` unused / packed and `umd_unique_id = chip_unique_ids[chip]`.

---

### C. Flat physical adjacency — **this is the solve input**

Today (`topology_mapper_utils.cpp` ~737–763):

```cpp
PhysicalAdjacencyMap build_flat_adjacency_map_from_psd(const PhysicalSystemDescriptor& psd) {
    PhysicalAdjacencyMap flat_adj;
    for (const auto& host_name : psd.get_all_hostnames()) {
        for (const auto& [src_asic_id, asic_connections] : psd.get_asic_topology(host_name)) {
            for (const auto& asic_connection : asic_connections) {
                auto dst_asic_id = asic_connection.first;
                for (const auto& eth_conn : asic_connection.second) {
                    flat_adj[src_asic_id].push_back(dst_asic_id);  // UMD or 1..N
                }
            }
        }
    }
    return flat_adj;
}
```

After:

```cpp
PhysicalAdjacencyMap build_flat_adjacency_map_from_psd(const PhysicalSystemDescriptor& psd) {
    PhysicalAdjacencyMap flat_adj;
    const bool hosts_unique = psd.get_all_hostnames_unique();
    const auto& descs = psd.get_asic_descriptors();
    for (const auto& host_name : psd.get_all_hostnames()) {
        for (const auto& [src_asic_id, asic_connections] : psd.get_asic_topology(host_name)) {
            const auto src = node_id_from_asic_descriptor(descs.at(src_asic_id), hosts_unique);
            for (const auto& [dst_asic_id, eth_connections] : asic_connections) {
                if (src_asic_id == dst_asic_id) {
                    continue;
                }
                const auto dst = node_id_from_asic_descriptor(descs.at(dst_asic_id), hosts_unique);
                for (std::size_t i = 0; i < eth_connections.size(); ++i) {
                    flat_adj[src].push_back(dst);
                }
            }
        }
    }
    return flat_adj;
}
```

`PhysicalAdjacencyMap` is `std::map` — iteration order is `PhysicalNodeId`’s `<=>` (host bytes, tray, loc). FSD-built and live-built maps with the same edges compare **equal**. That is the stability test.

`AdjacencyGraph<PhysicalNodeId>` is constructed from this map (`topology_solver.hpp` ~42, `std::map<NodeId, vector<NodeId>>`). `get_nodes()` follows that order.

---

### D. Solver erases to dense indices **in that order**

`topology_solver.tpp` ~1573–1589 — **do not change this**. It already copies `global_graph.get_nodes()` in map order. Changing the node type is what makes FSD and PSD agree:

```cpp
GraphIndexData::GraphIndexData(const AdjacencyGraph<TargetNode>& target_graph,
                               const AdjacencyGraph<GlobalNode>& global_graph) {
    for (const auto& node : global_graph.get_nodes()) {
        global_nodes.push_back(node);   // SAT variable i = this order
    }
}
```

Today `GlobalNode = AsicID` → FSD `{1,2,3,…}` vs live `{0x9a3f…, 0x15e8…}` → different `i` → different legal placement. After `GlobalNode = PhysicalNodeId` both graphs list the same nodes in the same order.

Call sites (type argument only):

```cpp
// topology_mapper_utils.cpp
AdjacencyGraph<PhysicalNodeId> global_graph(physical_adjacency);
MappingConstraints<FabricNodeId, PhysicalNodeId> constraints;
MappingResult<FabricNodeId, PhysicalNodeId> solver_result;

// PhysicalMultiMeshGraph
std::map<MeshId, AdjacencyGraph<PhysicalNodeId>> mesh_adjacency_graphs_;
```

---

### E. TopologyMapper — fill config from the descriptor, not from `asic_id`

Today (`topology_mapper.cpp` ~525–536 and `generate_rank_bindings.cpp` ~186–188) — **same bug on both paths**:

```cpp
for (const auto& [asic_id, desc] : physical_system_descriptor_.get_asic_descriptors()) {
    config.hostname_to_asics[desc.host_name].insert(asic_id);
    config.asic_positions[asic_id] = std::make_pair(desc.tray_id, desc.asic_location);
}
auto mapping_result = map_multi_mesh_to_physical(
    adjacency_map_logical_multi_mesh,
    adjacency_map_physical_multi_mesh,  // built from AsicID keys
    config,
    asic_id_to_mesh_rank,               // AsicID keys
    fabric_node_id_to_mesh_rank);
```

After:

```cpp
const bool hosts_unique = physical_system_descriptor_.get_all_hostnames_unique();
for (const auto& [_, desc] : physical_system_descriptor_.get_asic_descriptors()) {
    const auto nid = node_id_from_asic_descriptor(desc, hosts_unique);
    config.hostname_to_asics[desc.host_name].insert(nid);
    config.asic_positions[nid] = std::make_pair(desc.tray_id, desc.asic_location);
}
// asic_id_to_mesh_rank rebuilt with PhysicalNodeId keys (see F)
auto mapping_result = map_multi_mesh_to_physical(
    adjacency_map_logical_multi_mesh,
    adjacency_map_physical_multi_mesh,
    config,
    physical_node_id_to_mesh_rank,
    fabric_node_id_to_mesh_rank);
```

`generate_rank_bindings.cpp` (~186 and the second copy ~302) must use the **same** loop. That is how FM / Phase 1 and ControlPlane / Phase 2 agree.

---

### F. `asic_id_to_mesh_rank` and chip-info tables

Today (`topology_mapper.cpp` ~609–623, ~370–409):

```cpp
std::map<MeshId, std::map<AsicID, MeshHostRankId>>
TopologyMapper::build_asic_id_to_mesh_rank_mapping() {
    auto asics = psd.get_asics_connected_to_host(psd.my_host_name());
    for (const auto& asic : asics) {
        mapping[mesh_id][asic] = host_rank;   // AsicID from PSD
    }
}

for (const auto& [asic_id, asic_descriptor] : asic_descriptors) {
    info.asic_id = asic_id;
    info.tray_id = asic_descriptor.tray_id;
    info.asic_location = asic_descriptor.asic_location;
    if (unique_id == *asic_id) {              // ChipId backfill assumes unique_id == UMD
        info.physical_chip_id = physical_chip_id;
    }
}
```

After:

```cpp
std::map<MeshId, std::map<PhysicalNodeId, MeshHostRankId>>
TopologyMapper::build_physical_node_id_to_mesh_rank_mapping() {
    const bool hosts_unique = psd.get_all_hostnames_unique();
    for (AsicID raw : psd.get_asics_connected_to_host(psd.my_host_name())) {
        const auto nid = node_id_from_asic_descriptor(psd.get_asic_descriptors().at(raw), hosts_unique);
        mapping[mesh_id][nid] = host_rank;
    }
}

info.physical_node_id = node_id_from_asic_descriptor(asic_descriptor, hosts_unique);
info.asic_id = AsicID{asic_descriptor.umd_unique_id};  // UMD when known; 0 / unset on FSD-only
info.tray_id = asic_descriptor.tray_id;
info.asic_location = asic_descriptor.asic_location;
// ChipId backfill: match umd_unique_id, not physical_node_id
if (unique_id == asic_descriptor.umd_unique_id) {
    info.physical_chip_id = physical_chip_id;
}
```

Lookups — physical side is only `physical_node_id_to_mapping_`:

```cpp
for (const auto& [fabric_node, nid] : mapping_result.fabric_node_to_physical) {
    MappedChipInfo& info = *physical_node_id_to_mapping_.at(nid);
    info.fabric_node_id = fabric_node;
    info.is_mapped = true;
}
```

No `asic_id_to_mapping_`. An AsicID query (ControlPlane / Cluster) scans `chip_topology_mapping_` for `info.asic_id` / `umd_unique_id`, or goes `AsicID → (host,tray,loc) on the PSD → PhysicalNodeId → physical_node_id_to_mapping_`.

---

### G. Exit nodes and hierarchical / PGD graphs

`PhysicalExitNode` (`topology_mapper_utils.hpp` ~274) is the inter-mesh solver node. Change `asic_id` → `physical_node_id` so inter-mesh SAT uses the same identity as intra-mesh.

`build_physical_multi_mesh_adjacency_graph` / `build_hierarchical_from_flat_graph` take the flat map from **C** and partition it. Once the flat map is `PhysicalNodeId`, these follow if their signatures use `PhysicalAdjacencyMap` / `AdjacencyGraph<PhysicalNodeId>`.

PGD matching (`physical_grouping_descriptor_matching.cpp`) builds `AdjacencyGraph<AsicID>` from the PSD. When ControlPlane uses the PGD path, re-key that graph the same way or the preferred-pinning layout will be in a different id space than the solver.

---

### H. `verify_topology_mapping` — still UMD

Today (`topology_mapper.cpp` ~1781–1815) Check 1 compares `info.asic_id` to `cluster.get_unique_chip_ids()`. After the split, that check uses `info.asic_id` / `umd_unique_id` only. Skip it when `umd_unique_id` is unset (FSD-only `generate_rank_bindings`). Tray/loc checks already go through the PSD and stay valid.

---

### I. Consistency picture

```
FSD textproto                         live / mock PSD
    │                                      │
    ▼                                      ▼
builder fills ASICDescriptor          discovery fills ASICDescriptor
  host, tray, loc                       host, tray, loc
  unique_id = 1..N (ignored)            unique_id = UMD (ignored)
    │                                      │
    └──────────────┬───────────────────────┘
                   ▼
        node_id_from_asic_descriptor
                   ▼
        PhysicalAdjacencyMap / config / rank maps
                   ▼
        AdjacencyGraph<PhysicalNodeId>
                   ▼
        GraphIndexData  (same node order both sides)
                   ▼
        map_multi_mesh_to_physical
                   ▼
        same FabricNodeId per (host, tray, loc)
```

`generate_rank_bindings` (FSD, no UMD) and `TopologyMapper` (live PSD) both run the middle column. That is the whole point.

---

## 7. Tests

- Same `(host, tray, loc)` → same id, including FQDN vs short name vs case after canonicalization.
- `_rank` suffix stripped iff `hosts_unique == false`.
- Different loc, tray, or host → different id.
- `decode` restores the canonical host string, tray, and loc.
- Host that does not fit in `kPhysicalHostNameLen - 1` → fatal (do not truncate).
- Tray or loc `> 0xffff` → fatal.
- Empty / unset id is all zeros; `make_*` never returns it.
- Golden vector: one fixed host/tray/loc → exact `host[]` bytes + tray + loc.
- **Stability (the load-bearing one):** build one graph from FSD and one from a live-style descriptor with UMD-like `umd_unique_id`s; adjacency maps keyed by `PhysicalNodeId` are equal. `TopologyMapper` (or the solver on those maps) assigns the same `FabricNodeId` per position.
- **Mock + FSD (the other load-bearing one):** descriptor with `hostname: bh-glx-110-c01u02` packs the same id as the FSD builder. Filename-only (no field) must **not** equal that id. After the UMD field is filled, SC36 rank 15 packs `bh-glx-110-d10u20`, not the `120` in the filename.
- Builder unit test: QuietBox / a tiny in-memory FSD, two ASICs, ids equal `make_physical_node_id` of their FSD hostnames, not `1` and `2`.
- Discovery: graph key ≠ `umd_unique_id` on silicon when UMD ids are large; `get_asic_id(host, tray, loc)` returns the packed id.

Existing mapper tests that assert one specific mapping may need a re-baseline (iteration order follows packed ids). That is expected; it is the same cost #54752 recorded for tuple nodes.

---

## 8. Mock hostnames

### 8.1 Problem

`PhysicalNodeId.host[]` is the **exact** OS / FSD hostname (`bh-glx-110-c01u02`). Hall stays. FSD builder already has that string. Silicon `gethostname()` is that string.

Mock discovery does **not**. A UMD cluster descriptor YAML has no hostname field. `get_local_discovery_hostname()` (`physical_system_discovery.cpp` ~60) returns the **filename basename** of `TT_METAL_MOCK_CLUSTER_DESC_PATH`:

```cpp
return std::filesystem::path(mock_cluster_desc_path).filename().string();
```

So the same machine has three different strings, and only one of them is a hostname:

| Side | What it actually is | Example |
|------|---------------------|---------|
| FSD `hosts[].hostname` | the real name we must pack | `bh-glx-110-c01u02` |
| Mock YAML **filename** | asset-repo file name, not a hostname | `SC20_32x4_revAB_aisleC_cluster_desc_bh-glx-c01u02_rank_0.yaml` |
| Live PSD key today | that basename, including `.yaml` | same as the filename |
| Filename token (SC20) | hall **dropped** | `bh-glx-c01u02` ≠ FSD |
| Filename token (SC36 aisle D) | hall **wrong** | file `bh-glx-120-d10u20` vs FSD `bh-glx-110-d10u20` |
| Silicon | `gethostname()` | `bh-glx-110-c01u02` |

QuietBox already matches (`sjc1-tt-qb-01` on both sides). BH supercluster mock does not.

If we pack the basename or the filename token, FSD-built and mock-discovered `PhysicalAdjacencyMap`s have different keys. The solver sees two graphs. That is the same bug as `1..N` vs UMD ids. Parsing a hostname out of the filename cannot work: SC20 omits the hall, SC36 disagrees on it. Silicon would then miss if we stripped hall from the FSD name.

### 8.2 Solution

Put the real hostname **on the cluster descriptor** and query it. Do not parse the filename at runtime.

Full design: [`PLAN_umd_cluster_descriptor_hostname.md`](PLAN_umd_cluster_descriptor_hostname.md).

```yaml
hostname: bh-glx-110-c01u02    # exact FSD / OS name — hall included
arch:
  ...
```

```
desc.get_hostname()  →  optional<string>   // UMD

get_local_discovery_hostname(desc):
    if desc.get_hostname():  return *that          // the name we pack
    if mock env set:         return filename       // legacy, until assets are filled
    return get_host_name()
```

`node_id_from_asic_descriptor` then packs that string. FSD builder and mock PSD produce the same `PhysicalNodeId`.

**Backward compatible:** the field is optional. Old YAMLs still load (`nullopt`). Old UMD ignores an unknown `hostname:` key. Metal falls back to the basename when the field is absent — ClosetBox and existing tests stay green. Do not rename files. Do not edit FSD textprotos.

**Until FSD-paired YAMLs are filled:** if mock + FSD and `get_hostname()` is empty, aisle-token alias (testing plan §6.3) so `host[]` is still the FSD hostname. Delete the alias once those files have the field. ClosetBox / no-FSD keeps the basename.

```
host_for_node_id(desc, fsd):
    if desc.get_hostname():
        return canonical(*desc.get_hostname())
    if mock and fsd:
        return alias[basename]     // temporary
    return canonical(live_key)
```

| Path | String that goes into `host[]` |
|------|--------------------------------|
| FSD builder | `hosts[].hostname` |
| Silicon | `gethostname()` / stamped `desc.get_hostname()` |
| Mock + field set | `desc.get_hostname()` (= FSD name) |
| Mock, field empty, + FSD | alias fallback (temporary) |
| Mock, no FSD | basename (ClosetBox) |

### 8.3 What needs to be done

Work is three repos. None of this is “parse the filename in the mapper.”

**1. UMD** — [`PLAN_umd_cluster_descriptor_hostname.md`](PLAN_umd_cluster_descriptor_hostname.md)

- [ ] Add optional top-level `hostname:` to the cluster-descriptor YAML schema (`not` required).
- [ ] `ClusterDescriptor::get_hostname()` / `set_hostname()`; parse if present, `nullopt` if missing; reject empty / illegal strings.
- [ ] `serialize()` writes the key only when set (old goldens unchanged).
- [ ] Copy `hostname_` in `create_constrained_cluster_descriptor` and `apply_chip_id_remapping`.
- [ ] `TopologyDiscovery::fill_cluster_descriptor_info` stamps `gethostname()` on live silicon.
- [ ] Nanobind get/set.
- [ ] Offline tests: with key, without key (every existing YAML still parses), round-trip, validation fatals.
- [ ] Bump `tt_metal/third_party/umd` in tt-metal.

**2. tt-metal** — consume the field

- [ ] `get_local_discovery_hostname(cluster_desc)` prefers `get_hostname()`, else today’s basename / `gethostname()`.
- [ ] `node_id_from_asic_descriptor` / `make_physical_node_id` use that string for `PhysicalNodeId.host[]`.
- [ ] Every TopologyMapper **physical** map keyed by `PhysicalNodeId` (§6.1).
- [ ] Temporary aisle-token alias when mock + FSD and the field is still empty (testing plan §6.3).
- [ ] Tests: filename-only id ≠ FSD id; `hostname: bh-glx-110-c01u02` id == FSD id; SC36 packs `bh-glx-110-d10u20` not the `120` in the file.

**3. tt-cluster-descriptors** — **TODO: fill every YAML** (~230 files)

- [ ] Write `hostname:` on **all** cluster descriptors (BH, ClosetBox, wormhole, T3K, dual-host, virtu, …). Separate PR. Does **not** block the UMD PR. Partial fill is fine.
- [ ] FSD-paired BH: one-shot script — filename aisle token `c01u02` / `d10u20` → unique FSD `Host` with that `aisle`/`rack`/`shelf_u` → write that host’s `hostname` (`bh-glx-110-c01u02`).
- [ ] QuietBox: `sjc1-tt-qb-01` etc.
- [ ] ClosetBox: real host token (`metal-wh-09`), not the whole basename.
- [ ] Unknown captures: leave unset until recapture (`serialize_to_file` stamps live `gethostname()`).
- [ ] Pin the submodule when a batch is ready. Then delete the aisle-token fallback for filled FSD-paired files.

**Do not / what does not change**

- Parse / pack the filename token as a hostname.
- Drop the hall from the FSD name.
- Rename cluster-desc files or FSD textprotos.
- Require `hostname:` in the UMD schema or fatal when it is missing.
- Change ClosetBox tests in the UMD PR (they keep the basename until those files are filled).
- `LinkInfo` physical host still comes from the expected (FSD) PSD.

---

## 9. Non-goals

- Changing UMD `chip_unique_id` or Cluster ChipId
- Hash-joining pairing / `ExitNodeConnection` (rejected for `LinkHealth`)
- Downed-link reporting (separate plan)
- Sorting hosts into rack order
- Making `PhysicalNodeId` a protobuf field of the FSD
- Packing the host as `hash32` or stuffing the POD into `AsicID`
- Treating the mock filename token (`bh-glx-120-d10u20`) as a hostname

---

## 10. Sequence

1. UMD hostname field ([`PLAN_umd_cluster_descriptor_hostname.md`](PLAN_umd_cluster_descriptor_hostname.md)) — separate PR.
2. Utility + tests (no callers). Mock golden uses `desc.get_hostname()`, not the filename.
3. Builder switches off `1..N`. Offline test: FSD-only ids are the POD of the FSD hostname.
4. Mapper re-keys adjacency through the utility. FSD vs live agree when live uses `host_for_node_id` (§8.2).
5. Discovery writes packed graph keys and keeps UMD on `umd_unique_id`.
6. **TODO:** fill `hostname:` on **all** tt-cluster-descriptors YAMLs (UMD plan §7); delete the aisle-token fallback when FSD-paired files are done.
7. Downed-links / FSD ControlPlane work then diffs and maps without overlay.

Slices 2–4 are the PhysicalNodeId PR series. Slice 1 can land first or in parallel. Slice 6 is assets.
