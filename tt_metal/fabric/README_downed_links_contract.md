# Control Plane ↔ Fabric Contract: Downed Links (FSD vs PSD)

**Status:** Draft design contract for review
**Umbrella issue:** [tenstorrent/tt-metal#52859](https://github.com/tenstorrent/tt-metal/issues/52859)
**Related PRs:** [#53451](https://github.com/tenstorrent/tt-metal/pull/53451) (tt-run `--factory-system-descriptor` plumbing), [#53857](https://github.com/tenstorrent/tt-metal/pull/53857) (offline FSD → PSD builder)
**Header:** `tt-metalium/experimental/fabric/control_plane.hpp` (additions)

---

> ## NOTE — `get_downed_intermesh_links()` is **not** empty
>
> An FSD-expected intermesh cable that is missing from the live PSD **is a downed link**. It goes
> into `downed_` with `scope == InterMesh` and is returned by `get_downed_intermesh_links()`,
> `get_downed_links(LinkScope::InterMesh)`, and (when the ends are on different hosts)
> `get_downed_links_between_hosts`. Directions stay `NONE` — pairing never assigned that cable a
> port. Do **not** filter these records against post-pairing `inter_mesh_connectivity_`; that table
> is assigned-up only and would empty the set.
>
> Pairing is unchanged: `generate_intermesh_connectivity()` still gathers live / ETH-up cables only.
> Construct `LinkHealth` after pairing so intra directions are valid. Per-boundary requested-vs-
> resolved shortfall (a count, not a cable) is still **deferred past v1** — see
> [`PLAN_fsd_solve_and_downed_links.md`](PLAN_fsd_solve_and_downed_links.md) §4.1.
>
> There is **no** separate `get_absent_intermesh_cables()` set. Intermesh holes are downed links.
>
> ## NOTE — STRICT + FSD: do **not** fatal on down
>
> When an FSD is set, STRICT must **not** `TT_FATAL` on a missing or downed cable — intra or
> intermesh. Do **not** consult `is_link_healthy` or `cluster.is_ethernet_link_up` to abort.
> The outcome is a `LinkInfo` in `LinkHealth`. Log each skipped STRICT check at warning.
> Without an FSD, STRICT is unchanged. See §3.2.

---

## 1. Purpose

Give the Fabric team a first-class Control Plane API that answers one question:

> **Which connections were expected in the Factory System Descriptor (FSD) but are missing/down in the live Physical System Descriptor (PSD)?**

These "downed links" are the delta between the **ideal/as-built topology** (FSD — what *should* be wired) and the **live-discovered topology** (PSD — what is *actually* wired right now). Surfacing them is the pre-flight / degraded-cluster diagnostic that Fabric 2.0's rerouting is built on top of.

## 2. Background

| Term | Meaning |
|------|---------|
| **FSD** — Factory System Descriptor | The ideal / as-built adjacency graph: exactly which ethernet connections *should* exist between chips and hosts. Owned by DC bringup. |
| **PSD** — Physical System Descriptor | The live-discovered state: which connections are actually present right now. Produced by hardware discovery. |

"Present" is not "healthy": a link that is discovered but degraded stays present, and v1 does **not**
report it as down. `LinkHealth` answers presence only (FSD-expected minus live PSD).

ControlPlane then, **after** `LinkHealth` is built, checks `cluster.is_ethernet_link_up` the same
way `gather_intermesh_cables_for_exit_nodes` does — **local chips only** (`user_exposed_chip_ids`).
A local presence-missing cable with ETH down is copied into `locally_unhealthy_`. If ETH is up
while the cable is missing from the live PSD, that is a descriptor-vs-cluster disagreement: **log a
warning and keep the `LinkHealth` record — do not fatal when an FSD is set.** Remote cables cannot
be checked on this rank; they stay in `get_downed_links()`. See
[`PLAN_fsd_solve_and_downed_links.md`](PLAN_fsd_solve_and_downed_links.md) §7.1 / §7.3.

**Intermesh routing still chooses up links.** `generate_intermesh_connectivity` gathers from the
**live** PSD and skips ETH-down cables. Pairing fills MGD requested ports from that live/up subset.
The missing FSD intermesh cable is still a downed link: it is in `get_downed_intermesh_links()`
with `NONE` directions. Cross-host intermesh is visible on `get_downed_links_between_hosts`.
See the note at the top of this file and
[`PLAN_fsd_solve_and_downed_links.md`](PLAN_fsd_solve_and_downed_links.md) §4.

The two descriptors are compared by **physical position** `(canonical host, tray, loc, chan)`, not by
AsicID. FSD does not encode ASIC ids; the builder synthesizes opaque labels that must not drive the
mapper or the diff (implementation plan §5.5). `LinkInfo::src_asic` / `dst_asic` carry the **live**
UMD id when that chip exists (looked up by position after the solve). `(hostname, TrayID, ASICLocation)`
is the stable identity on every record. `medium` is the descriptor's port type for the expected edge.

## 3. No FSD, empty diff, and `fsd_rerouting_active`

Fabric 2.0 static rerouting is **off** unless Control Plane both (a) was given an FSD and (b) found at least one downed link. Gate that with `fsd_rerouting_active()`.

| Condition | Downed-link data | `is_link_healthy` | `fsd_rerouting_active()` |
|-----------|------------------|-------------------|--------------------------|
| No FSD on RTOptions | empty set (`get_downed_links()` empty, `has_downed_links() == false`) | **true** for every query (do not throw) | **false** |
| FSD present, live matches FSD | empty set | **true** for every FSD-expected link; `std::out_of_range` if the link was never in the FSD | **false** |
| FSD present, ≥1 FSD **intramesh** edge missing from live | those `LinkInfo`s | **false** for a downed expected link; **true** for an expected live link; `out_of_range` if never in the FSD | **true** |
| FSD present, ≥1 FSD **intermesh** edge missing from live | those `LinkInfo`s in `get_downed_intermesh_links()` (and in `get_downed_links()`) | **false** for the missing channel (live-PSD presence; **works on mock**) | **true** |

No FSD means there is no expected graph, so every link is treated as healthy by default and Fabric 2.0 static rerouting stays inactive. An FSD with a clean live match is the same for rerouting: healthy, inactive. Rerouting turns on when an **active** FSD-expected cable is missing — intermesh holes, or intra holes on planes fabric still uses. Intra holes on **downgraded** routing planes go to `get_unused_downed_links()` and do **not** activate rerouting. `is_link_healthy` is still `false` for those unused-plane holes (presence). See implementation plan §3.3.

`is_link_healthy` does **not** call `cluster.is_ethernet_link_up`. It works on **mock cluster
descriptors**: mock discovery builds the live PSD from the YAML; drop or omit the cable there and
the API returns `false`. On STRICT + FSD, do **not** use this API (or ETH-up) to abort init — see §3.2.

### 3.1 FSD is golden; extras in the PSD are OK; incompatible hosts and >10% down are not

The FSD is the expected graph. A live PSD may have **more** cables than the FSD (`extra_links`).
Those are **OK**: log at info, do not treat them as downed, do not fail init.

**Downed links are cable-level.** A missing *cable* is what this API reports. A missing *chip* is a
wrong allocation and fails init before `LinkHealth` is built, so it never appears as downed links.

| Condition | Action | Log |
|-----------|--------|-----|
| Live has a cable the FSD does not | OK. Not downed. | info: `FSD is golden: N live-only extra cable(s) ignored (not downed).` |
| **FSD has a whole chip live does not** | **Fatal, before the topology map is built.** A missing chip is not a downed-link case: `LinkHealth` is never constructed and no downed records are produced for its cables. Fatal in **every** reliability mode. | error: `FSD: N chip(s) expected by the factory descriptor are absent from the live cluster: <host/tray/loc …>. A missing chip is not a downed-link case — check the allocation and the boards.` |
| Hosts do not match (zero overlap / wrong FSD) | **Fatal on every rank, same message.** All-reduce first (`agree_or_throw_fsd_host_filter`). No per-rank fallback. §5.3 case 9 / testing plan §7.6. | error: `FSD host filter is not identical on every rank (local_ok_min=…, host_checksum min=… max=…, fsd_fingerprint min=… max=…). Every rank fails together with this message. No per-rank fallback to live mapping.` |
| More than **10%** of FSD-expected connections missing from live | **Error.** Fail init. | info of the fraction, then throw (see §3.3) |
| At most 10% missing, hosts match | Init continues. Records in `LinkHealth`. | info: `FSD: E expected connections, D downed (P%). Extra live-only cables ignored.` |

The 10% check is `missing_from_live / fsd_expected_count` (active **and** unused-plane holes — both
are missing cables). `fsd_rerouting_active()` uses only the active set. Pick directed or undirected
and test it consistently. Document the chosen convention next to `fsd_expected_count()`.

### 3.2 STRICT + FSD — do **not** fatal on down; document in LinkHealth

> **This is the load-bearing ControlPlane rule.** Without it, default STRICT dies at
> `configure_routing_tables_for_fabric_ethernet_channels()` (missing-connection and link-count
> fatals) before `LinkHealth` exists, and the feature never runs.

When `has_factory_system_descriptor_path()` is true:

- Do **not** `TT_FATAL` on missing connections or short link counts in STRICT
  (`configure_routing_tables_for_fabric_ethernet_channels`).
- Skip the same class of check in `validate_mesh_connections`, channel trim,
  `generate_intermesh_connectivity` (assigned < requested), and
  `validate_requested_intermesh_connections` if they fire on FSD holes.
- Do **not** `TT_FATAL` in `confirm_local_downed_links` on ETH-up-but-missing.
- Do **not** consult `is_link_healthy` or `is_ethernet_link_up` to decide whether to abort.
  Ignore `is_link_healthy` for the abort path. The API remains a presence diagnostic (and works on mock).
- **Do** put the missing cables — **including intermesh** — into `LinkHealth`
  (`get_downed_intermesh_links()` / `get_downed_intramesh_links()`). That record *is* the STRICT outcome.
- Log each skipped STRICT fatal at **warning**, naming the edge and that FSD is in use.

Without an FSD, STRICT is unchanged (fatal on missing connections).

| Mode | FSD? | Missing cable | Result |
|------|------|---------------|--------|
| STRICT | yes | intra or intermesh | init completes; record in LinkHealth; warning log |
| STRICT | no | any MGD-required connection | `TT_FATAL` (today's behaviour) |
| RELAXED | yes or no | any | init continues; with FSD the miss is in LinkHealth |

### 3.3 Required log lines

| When | Level | Text |
|------|-------|------|
| Skipped STRICT fatal because FSD is set | warning | `STRICT + FSD: skipping fatal on missing/short connection {src} chan {chan} → {dst}. FSD in use — recording in LinkHealth, not fatal.` |
| Live has extra cables the FSD does not | info | `FSD is golden: {N} live-only extra cable(s) ignored (not downed).` |
| After `LinkHealth` is built | info | `FSD: {E} expected connections, {D} downed ({P}%). Extra live-only cables ignored.` |
| After plane trim classified unused holes | info | `FSD: {N} unused downed link(s) on downgraded routing planes (ignored for rerouting / planes_lost). See get_unused_downed_links().` |
| `D/E > 0.10` | error / throw | `FSD: {P}% of expected connections are missing from live (limit 10%). See get_downed_links().` |
| Local ETH up but cable missing from live PSD | warning | `FSD-expected cable missing from live PSD but ethernet is up: {node} chan {chan} (chip {id}). FSD is golden — not fatal. Record stays in LinkHealth.` |
| Host mismatch / wrong FSD / ranks disagree on the filter | error / throw on **every rank** | `FSD host filter is not identical on every rank (local_ok_min=…, host_checksum min=… max=…, fsd_fingerprint min=… max=…). Every rank fails together with this message. No per-rank fallback to live mapping.` |

## 4. Full API interface (ground truth)

A **downed link** = one FSD-expected connection that is absent from the live PSD.

`LinkScope` / `LinkInfo` and the whole query surface live in their own module,
`tt-metalium/experimental/fabric/link_health.hpp` (class `LinkHealth`). `ControlPlane` owns a
`LinkHealth` and every method below is a one-line forwarder onto it, so the contract surface is
unchanged for callers. `ControlPlane::get_link_health()` returns the module for anyone who prefers to
hold it.

**A `LinkHealth` exists only when an FSD was provided.** On a system without one there is no object:
each forwarder returns the §3 default instead, so the table below holds either way and nothing
throws. This matters only if you use `get_link_health()` directly, which returns `nullptr` in that
case. See [`PLAN_fsd_solve_and_downed_links.md`](PLAN_fsd_solve_and_downed_links.md) §3.

```cpp
// tt-metalium/experimental/fabric/link_health.hpp (types + LinkHealth)
// tt-metalium/experimental/fabric/control_plane.hpp (forwarders below)
namespace tt::tt_fabric {

// ── shared vocabulary ───────────────────────────────────────────────
enum class LinkScope { IntraMesh, InterMesh, Unknown };

// One downed link = one FSD-expected connection absent from the live PSD.
// Standalone type — carries BOTH the logical (router/reroute) and physical
// (DC/bringup) identity of the failed link. No ExitNodeConnection dependency.
struct LinkInfo {
    // ---- logical (resolved via TopologyMapper: node ids + mesh graph) ----
    FabricNodeId       src_node;
    chan_id_t          src_chan;
    FabricNodeId       dst_node;
    chan_id_t          dst_chan;
    RoutingDirection   src_direction = RoutingDirection::NONE;  // outbound port on src_node
    RoutingDirection   dst_direction = RoutingDirection::NONE;  // outbound port on dst_node
                                        // The two ends of a cable almost never share a direction
                                        // (intra E↔W, N↔S; intermesh each side has its own logical
                                        // port, including NESW vs Z). One record carries both so a
                                        // caller does not have to find the reverse. The set still
                                        // stores two directed records (A→B and B→A) so src-side
                                        // indexes (`get_downed_eth_chans_in_direction(node, dir)`)
                                        // key on that record's src_direction.
                                        // Intra: both from intra_mesh_connectivity (available at
                                        // MGD parse). Intermesh downed: from the PSD diff after
                                        // pairing; directions stay NONE (pairing never assigned
                                        // that cable a port). Unmapped / unresolved: both NONE.
    LinkScope          scope = LinkScope::Unknown;
                                        // IntraMesh iff both nodes resolved and src_mesh == dst_mesh.
                                        // InterMesh iff both resolved and meshes differ.
                                        // Unknown iff logical_resolved is false — do not guess.
                                        // Prefer is_intramesh() / is_intermesh(); they gate on
                                        // logical_resolved so an unmapped ASIC never reports Intra.

    bool logical_resolved = false;       // false ⇒ the MGD solve never placed one of these ASICs,
                                         // so every field above this line is meaningless and the
                                         // physical fields below are the whole record. Such a link
                                         // appears in get_downed_links() and in the per-host /
                                         // per-ASIC queries, but NOT in the per-node, per-direction
                                         // or per-mesh-pair ones. Check this before reading src_node.

    bool is_intramesh() const { return logical_resolved && scope == LinkScope::IntraMesh; }
    bool is_intermesh() const { return logical_resolved && scope == LinkScope::InterMesh; }

    // No routing_plane field. A plane index is a position in the LIVE ordered channel
    // list, and a downed channel is by definition absent from it. Classify unused-plane
    // holes by (node, dir) downgrade counts (§3.3 / implementation plan §3.3), not by
    // inventing a plane. For a live channel, call ControlPlane::get_routing_plane_id.
    // For lost capacity on planes fabric still uses, get_num_downed_routing_planes_in_direction.

    // ---- physical (from the descriptor) ----
    std::string  src_host;  TrayID src_tray;  ASICLocation src_loc;  AsicID src_asic;
    std::string  dst_host;  TrayID dst_tray;  ASICLocation dst_loc;  AsicID dst_asic;
    PortType     medium;                // QSFP_DD / WARP400 / TRACE ...

    // Only meaningful when logical_resolved.
    MeshId src_mesh() const { return src_node.mesh_id; }
    MeshId dst_mesh() const { return dst_node.mesh_id; }
};

// ════════════════════════════════════════════════════════════════════
// Contract on ControlPlane (each one forwards to the LinkHealth module)
// ════════════════════════════════════════════════════════════════════

// Direct access to the module, for callers that would rather hold it.
// nullptr when no FSD was provided — there is no module in that case. Prefer the
// forwarders below, which apply the no-FSD defaults for you.
const LinkHealth* get_link_health() const;

// ── full set + lifecycle ─────────────────────────────────────────────
// The stored set is GLOBALLY COMPLETE on every rank, and no communication is
// needed to make it so: physical discovery already gathers and broadcasts a
// system-wide PhysicalSystemDescriptor, the FSD file is identical on every
// rank, and TopologyMapper resolves any ASIC in the system, not just local
// ones. So the purely local FSD-vs-PSD diff is already the whole-system
// answer, logical fields included (local intramesh links too, not just
// cross-host). Downed links on remote hosts appear here, fully populated.
// No FSD (or a clean FSD-vs-live match) ⇒ empty set; all links are healthy.
const std::vector<LinkInfo>& get_downed_links() const;   // complete set, both worlds
// NOT a collective: purely local, safe to call on one rank, cannot deadlock,
// and needs no ordering against other collectives. Idempotent.
// Invalidates the reference returned by get_downed_links().
void                           refresh_connectivity_diff(); // recompute
bool                           has_downed_links() const;

// Fabric 2.0 static rerouting is active iff an FSD was provided AND the
// *active* downed-link set is non-empty. Unused-plane holes do not count.
bool fsd_rerouting_active() const;

// Intra holes on routing planes fabric already downgraded away (row/col min + trim).
// Real unplugged cables that fabric does not route on, documented here rather than
// dropped. This is the only unused set: intermesh holes are ACTIVE downed links.
// NOT in get_downed_links(); does NOT set
// has_downed_links() / fsd_rerouting_active(); planes_lost ignores them.
// Empty when no FSD, or no downgrade. See implementation plan §3.3 / testing plan §7.11.
const std::vector<LinkInfo>& get_unused_downed_links() const;

// ── LOGICAL façade (routing-table native) ────────────────────────────
// No FSD ⇒ true / nullopt (treat every link as healthy).
// FSD present — throws std::out_of_range if (node, chan) is not FSD-expected.
// Presence vs the live PSD only — does NOT call cluster.is_ethernet_link_up.
// Works on mock cluster descriptors (mock discovery builds the live PSD from YAML).
// STRICT + FSD: do not use this (or ETH-up) to abort init — see §3.2.
bool                      is_link_healthy(FabricNodeId, chan_id_t) const;   // true ⇒ healthy OR no FSD
// nullopt ⇒ "no downed record": healthy, no FSD, OR never expected. Use
// is_link_healthy when you need to tell a healthy link from an unknown one.
std::optional<LinkInfo> find_downed_link(FabricNodeId, chan_id_t) const;

// per-node
std::vector<LinkInfo> get_downed_links(FabricNodeId) const;
std::vector<chan_id_t>  get_downed_eth_chans(FabricNodeId) const;

// by direction  (mirrors get_active_fabric_eth_channels_in_direction)
std::vector<chan_id_t> get_downed_eth_chans_in_direction(FabricNodeId, RoutingDirection) const;
bool                   has_downed_link_in_direction     (FabricNodeId, RoutingDirection) const;
size_t                 get_num_downed_routing_planes_in_direction(FabricNodeId, RoutingDirection) const;
                       // = planes_lost = count of *active* downed_ on (node, src_direction).
                       // 0 if that direction was downgraded (holes are in get_unused_downed_links()).

// by scope  (mirrors get_intra/intermesh_facing_eth_chans)
std::vector<chan_id_t> get_downed_intramesh_eth_chans(FabricNodeId) const;
std::vector<chan_id_t> get_downed_intermesh_eth_chans(FabricNodeId) const;

// All downed records of one scope. Unknown (unmapped ASIC) is in neither.
// Intermesh: FSD-expected cables missing from live. NOT empty when those cables are gone.
// Directions NONE. Cross-host intermesh is also in get_downed_links_between_hosts.
// The two named getters are the same as get_downed_links(LinkScope::IntraMesh)
// and get_downed_links(LinkScope::InterMesh). The (src_mesh, dst_mesh) overload
// below is the subset of InterMesh between those two meshes.
std::vector<LinkInfo> get_downed_links(LinkScope) const;
std::vector<LinkInfo> get_downed_intramesh_links() const;
std::vector<LinkInfo> get_downed_intermesh_links() const;

// by route  (mirrors get_forwarding_eth_chans_to_chip / get_forwarding_direction)
std::vector<LinkInfo> get_downed_links_between(FabricNodeId src, FabricNodeId dst) const;
std::vector<chan_id_t>  get_downed_forwarding_eth_chans_to_chip(FabricNodeId src, FabricNodeId dst) const;

// by mesh boundary  (mirrors get_exit_fabric_node_ids_between_meshes)
std::vector<LinkInfo>   get_downed_intermesh_links(MeshId src_mesh, MeshId dst_mesh) const;
std::vector<FabricNodeId> get_exit_nodes_with_downed_links(MeshId src_mesh, MeshId dst_mesh) const;

// ── PHYSICAL façade (DC / bringup native) ─────────────────────────────
// No FSD ⇒ true. FSD present — throws std::out_of_range if not FSD-expected.
// Same presence check as the logical overload — works on mock. STRICT + FSD: ignore for abort.
bool is_link_healthy(const std::string& host, TrayID, ASICLocation, chan_id_t) const;
bool is_link_healthy(AsicID, chan_id_t) const;
std::vector<LinkInfo> get_downed_links_for_host (const std::string& host) const;
std::vector<LinkInfo> get_downed_links_for_asic (AsicID) const;
std::vector<LinkInfo> get_downed_links_between_hosts(const std::string& a, const std::string& b) const;

}  // namespace
```

## 5. Usage examples

**Reroute planner — only when FSD static rerouting is active:**
```cpp
if (!cp.fsd_rerouting_active()) {
    // no FSD, or FSD with zero downed links — use the mapped tables as-is
}

if (cp.has_downed_link_in_direction(node, RoutingDirection::E)) {
    auto chans  = cp.get_downed_eth_chans_in_direction(node, RoutingDirection::E);
    auto planes = cp.get_num_downed_routing_planes_in_direction(node, RoutingDirection::E);
    // reroute around `chans`; `planes` is the capacity hit
}
```

**Inter-mesh health — which cross-mesh links failed, globally or between two meshes?**
```cpp
for (const auto& l : cp.get_downed_intermesh_links()) {
    // FSD-expected intermesh missing from live. NOT empty when those cables are gone.
    // l.src_direction / l.dst_direction are NONE (pairing never assigned that cable a port).
    // STRICT + FSD: this record is the outcome — init did not fatal.
}
for (const auto& l : cp.get_downed_intermesh_links(mesh_a, mesh_b)) {
    // l.src_node / l.dst_node  → logical endpoints
    // l.src_host/tray/loc + l.medium → physical cable to inspect
}
for (const auto& l : cp.get_downed_links_between_hosts("hostA", "hostB")) {
    // intra AND intermesh cross-host. Not intra-only.
}
for (const auto& l : cp.get_downed_intramesh_links()) {
    // same-mesh failures on planes fabric still uses; l.is_intramesh() == true
}
for (const auto& l : cp.get_unused_downed_links()) {
    // intra hole on a downgraded routing plane. Documented; fabric does not route here.
    // Not in get_downed_links(); planes_lost ignores these.
}
```

**Point health — `true` means healthy or no FSD; with an FSD, throws if the link was never expected:**
```cpp
bool ok = cp.is_link_healthy(node, chan);            // true ⇒ healthy OR no FSD; false ⇒ down
if (auto down = cp.find_downed_link(node, chan))     // full record when down
    log("expected peer {} chan {} is down", down->dst_node, down->dst_chan);
```
A `true` here does not by itself mean the link was checked against an expected graph — with no FSD
there is nothing to check against. Use `fsd_rerouting_active()` when that distinction matters.

**DC bringup (physical) — pinpoint the exact failing cable on a host pair:**
```cpp
for (const auto& l : cp.get_downed_links_between_hosts("hostA", "hostB")) {
    // intra or intermesh. "hostA tray2/loc1 chan3  ══(QSFP_DD)══  hostB tray5/loc0 chan7"
}
```

**STRICT + FSD — do not abort on health:**
```cpp
// WRONG when an FSD is set:
//   if (!cp.is_link_healthy(node, chan)) TT_FATAL(...);
//   if (!cluster.is_ethernet_link_up(...)) TT_FATAL(...);
// RIGHT: init continues; inspect LinkHealth.
if (cp.get_link_health() != nullptr) {  // FSD was provided
    for (const auto& l : cp.get_downed_intermesh_links()) { /* document / reroute */ }
    for (const auto& l : cp.get_downed_intramesh_links()) { /* document / reroute */ }
}
```
