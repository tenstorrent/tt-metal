# Control Plane ↔ Fabric Contract: Downed Links (FSD vs PSD)

**Status:** Draft design contract for review
**Umbrella issue:** [tenstorrent/tt-metal#52859](https://github.com/tenstorrent/tt-metal/issues/52859)
**Related PRs:** [#53451](https://github.com/tenstorrent/tt-metal/pull/53451) (tt-run `--factory-system-descriptor` plumbing), [#53857](https://github.com/tenstorrent/tt-metal/pull/53857) (offline FSD → PSD builder)
**Header:** `tt-metalium/experimental/fabric/control_plane.hpp` (additions)

---

## 1. Purpose

Give the Fabric team a first-class Control Plane API that answers one question:

> **Which connections were expected in the Factory System Descriptor (FSD) but are missing/down in the live Physical System Descriptor (PSD)?**

These "downed links" are the delta between the **ideal/as-built topology** (FSD — what *should* be wired) and the **live-discovered topology** (PSD — what is *actually* wired right now). Surfacing them is the pre-flight / degraded-cluster diagnostic that Fabric 2.0's rerouting is built on top of.

## 2. Background

| Term | Meaning |
|------|---------|
| **FSD** — Factory System Descriptor | The ideal / as-built adjacency graph: exactly which ethernet connections *should* exist between chips and hosts. Owned by DC bringup. |
| **PSD** — Physical System Descriptor | The live-discovered state: what is actually wired and healthy right now. Produced by hardware discovery. |


## 4. Full API interface (ground truth)

A **downed link** = one FSD-expected connection that is absent from the live PSD.

```cpp
// tt-metalium/experimental/fabric/control_plane.hpp  (additions)
namespace tt::tt_metal::experimental::tt_fabric {

// ── shared vocabulary ───────────────────────────────────────────────
enum class LinkScope { IntraMesh, InterMesh };

// One downed link = one FSD-expected connection absent from the live PSD.
// Standalone type — carries BOTH the logical (router/reroute) and physical
// (DC/bringup) identity of the failed link. No ExitNodeConnection dependency.
struct LinkInfo {
    // ---- logical (resolved via TopologyMapper + routing tables) ----
    FabricNodeId       src_node;
    chan_id_t          src_chan;
    FabricNodeId       dst_node;
    chan_id_t          dst_chan;
    RoutingDirection   direction;       // outbound direction on src_node
    LinkScope          scope;           // IntraMesh vs InterMesh
    routing_plane_id_t routing_plane;   // plane the src channel serves

    // ---- physical (from the descriptor) ----
    std::string  src_host;  TrayID src_tray;  ASICLocation src_loc;  AsicID src_asic;
    std::string  dst_host;  TrayID dst_tray;  ASICLocation dst_loc;  AsicID dst_asic;
    PortType     medium;                // QSFP_DD / WARP400 / TRACE ...

    MeshId src_mesh() const { return src_node.mesh_id; }
    MeshId dst_mesh() const { return dst_node.mesh_id; }
};

// ════════════════════════════════════════════════════════════════════
// Contract on ControlPlane
// ════════════════════════════════════════════════════════════════════

// ── full set + lifecycle ─────────────────────────────────────────────
// The stored set is GLOBALLY COMPLETE: refresh_connectivity_diff() computes
// the local FSD-vs-PSD diff, then all-gathers + merges across every host, so
// every rank's ControlPlane holds the full set of downed links for the whole
// system (local intramesh links included — not just cross-host links).
const std::vector<LinkInfo>& get_downed_links() const;   // complete set, both worlds
void                           refresh_connectivity_diff(); // recompute + all-host merge
bool                           has_downed_links() const;

// ── LOGICAL façade (routing-table native) ────────────────────────────
// point — throws std::out_of_range if (node, chan) is not an FSD-expected link
bool                      is_link_healthy(FabricNodeId, chan_id_t) const;   // false ⇒ down
std::optional<LinkInfo> find_downed_link(FabricNodeId, chan_id_t) const;  // nullopt ⇒ healthy

// per-node
std::vector<LinkInfo> get_downed_links(FabricNodeId) const;
std::vector<chan_id_t>  get_downed_eth_chans(FabricNodeId) const;

// by direction  (mirrors get_active_fabric_eth_channels_in_direction)
std::vector<chan_id_t> get_downed_eth_chans_in_direction(FabricNodeId, RoutingDirection) const;
bool                   has_downed_link_in_direction     (FabricNodeId, RoutingDirection) const;
size_t                 get_num_downed_routing_planes_in_direction(FabricNodeId, RoutingDirection) const;

// by scope  (mirrors get_intra/intermesh_facing_eth_chans)
std::vector<chan_id_t> get_downed_intramesh_eth_chans(FabricNodeId) const;
std::vector<chan_id_t> get_downed_intermesh_eth_chans(FabricNodeId) const;

// by route  (mirrors get_forwarding_eth_chans_to_chip / get_forwarding_direction)
std::vector<LinkInfo> get_downed_links_between(FabricNodeId src, FabricNodeId dst) const;
std::vector<chan_id_t>  get_downed_forwarding_eth_chans_to_chip(FabricNodeId src, FabricNodeId dst) const;

// by mesh boundary  (mirrors get_exit_fabric_node_ids_between_meshes)
std::vector<LinkInfo>   get_downed_intermesh_links(MeshId src_mesh, MeshId dst_mesh) const;
std::vector<FabricNodeId> get_exit_nodes_with_downed_links(MeshId src_mesh, MeshId dst_mesh) const;

// ── PHYSICAL façade (DC / bringup native) ─────────────────────────────
// point — throws std::out_of_range if the link is not an FSD-expected link
bool is_link_healthy(const std::string& host, TrayID, ASICLocation, chan_id_t) const;
bool is_link_healthy(AsicID, chan_id_t) const;
std::vector<LinkInfo> get_downed_links_for_host (const std::string& host) const;
std::vector<LinkInfo> get_downed_links_for_asic (AsicID) const;
std::vector<LinkInfo> get_downed_links_between_hosts(const std::string& a, const std::string& b) const;

}  // namespace
```

## 5. Usage examples

**Reroute planner (logical) — which of my EAST channels are down, and how many routing planes did I lose?**
```cpp
if (cp.has_downed_link_in_direction(node, RoutingDirection::EAST)) {
    auto chans  = cp.get_downed_eth_chans_in_direction(node, RoutingDirection::EAST);
    auto planes = cp.get_num_downed_routing_planes_in_direction(node, RoutingDirection::EAST);
    // reroute around `chans`; `planes` is the capacity hit
}
```

**Inter-mesh health — which cross-mesh links between two meshes failed?**
```cpp
for (const auto& l : cp.get_downed_intermesh_links(mesh_a, mesh_b)) {
    // l.src_node / l.dst_node  → logical endpoints
    // l.src_host/tray/loc + l.medium → physical cable to inspect
}
```

**Point health — throws if the link was never expected:**
```cpp
bool ok = cp.is_link_healthy(node, chan);            // false ⇒ down
if (auto down = cp.find_downed_link(node, chan))     // full record when down
    log("expected peer {} chan {} is down", down->dst_node, down->dst_chan);
```

**DC bringup (physical) — pinpoint the exact failing cable on a host pair:**
```cpp
for (const auto& l : cp.get_downed_links_between_hosts("hostA", "hostB")) {
    // "hostA tray2/loc1 chan3  ══(QSFP_DD)══  hostB tray5/loc0 chan7"
}
```
