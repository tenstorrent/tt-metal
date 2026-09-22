# Inter-Mesh Logical Port Pairing — Redesign

Status: draft / proposal
Owner: Riddy21
Related: [#49628](https://github.com/tenstorrent/tt-metal/issues/49628) (Z contention epic), [#40338](https://github.com/tenstorrent/tt-metal/issues/40338) (topology mapper / routing direction consolidation), [#49960](https://github.com/tenstorrent/tt-metal/issues/49960) (mixed strict/relaxed)

**Chosen approach:** the global both-sides **round-robin** allocator (§4). The min-cost-flow
sketch is kept as an alternative in [Appendix A](#appendix-a--alternative-considered-min-cost-flow).

---

## 1. Background

Inter-mesh connectivity is resolved in `ControlPlane::generate_intermesh_connectivity()`:

1. Each host proposes logical ports for its physical exit-node cables
   (`propose_port_descriptors_for_exit_nodes` → `generate_port_descriptors_for_exit_nodes`).
2. Proposals are gathered to rank 0 (`forward_descriptors_to_controller`).
3. Rank 0 pairs cables into logical connections (`pair_logical_intermesh_ports`).
4. Result is broadcast back and each host binds ports to physical cables by `connection_hash`.

A `PortDescriptor` carries:

```cpp
struct PortDescriptor {
    port_id_t port_id;          // {RoutingDirection, channel}
    std::size_t connection_hash;
};
```

The physical cable identity is `connection_hash` (hash of `ExitNodeConnection`).

## 2. Problem

The current pairing is an **online greedy first-fit** with **no backtracking**, done
**locally per mesh**. The hard part is not matching cables (an O(1) hash lookup) — it is
**assigning a logical direction class (NESW vs Z) to each cable** under coupled constraints:

- A logical port can be used at most once per mesh.
- **One neighbor mesh per `(mesh, chip, direction)`.** The inter-mesh routing table maps
  `(src_chip, dst_mesh) → a single direction`, and at the exit chip every channel of that
  direction egresses on physical cables that all go to the same neighbor mesh. So all cables
  assigned to a given `(chip, direction)` must share one neighbor mesh — otherwise a packet
  bound for mesh A could physically leave toward mesh B. This is the **same rule for NESW and
  Z**; Z just contends first because a chip has a single Z direction versus four NESW
  directions. (Today's code enforces only the Z half of this, via `chip_to_z_neighbor_mesh_id`.)
- MGD quotas (strict per exit node, relaxed per mesh pair).

Because each host proposes a direction class **locally and independently**, the two ends of
the same cable can disagree (one Z, one NESW), so rank 0 runs a **reconciliation** that
demotes Z→NESW, else promotes NESW→Z, else **drops** the cable — producing order-dependent,
hard-to-debug drops.

### Observed failure (root cause)

On the SC16 64-stage superpod ring, boundary `62↔63` resolved **zero** routers and fatal'd.
Instrumentation showed:

- The PSD cable is present on **both** ends (8 cables each way) — not a discovery/PSD gap.
- `pair_logical` dropped **nothing** (no Z/non-Z contention). The strand happened earlier:
  mesh 63 proposed **0 ports** toward mesh 62.
- `propose_port_descriptors_for_exit_nodes` fills **one neighbor at a time**, accumulating
  `assigned_port_ids` across neighbors. The wrap-around **M63↔M0** (16 cables spread over
  **all 8** edge chips) is processed first and consumes the only NESW ports on chips 2–5 —
  exactly the chips **M63↔M62** needs. M62 is processed last and finds no NESW left.

Concretely, each shared chip exposes only **2 NESW ports + 2 Z ports**, but **4 cables**
(2 for M0, 2 for M62) route through it. First-come (M0) takes both NESW; M62 gets none.

Two structural fixes follow:

1. **Fairness** — round-robin instead of "fill first boundary fully," so M62 gets a turn.
2. **Both-sides atomicity** — a cable resolves only when *both* endpoints get a port, so the
   two ends can never disagree (removing the demote/promote/drop reconciliation entirely).

## 3. Architectural change

"Assign both sides together" needs to see both endpoints of a cable at once, so assignment
moves to rank 0:

- **`propose` / `generate_port_descriptors_for_exit_nodes` → pure gather.** Each mesh emits
  its cables as `{connection_hash, src_node, dst_node, num_channels}` (both endpoints as
  `FabricNodeId`, §7.2) with **no port assignment**. Keep the eth-link-up skip and the
  deterministic sort here. Gather to rank 0 via the existing `forward_descriptors_to_controller`
  path.
- **Rank 0 sees every cable from both ends** (matched by `connection_hash`), so for each
  cable it knows `src_node` **and** `dst_node`, and assigns a port on **both** sides per
  placement.
- Output type `AnnotatedIntermeshConnections` is unchanged → downstream broadcast / cable
  binding / routing tables untouched.

### Concepts

- **Link (cable) = a 2-channel unit.** The eth channels sharing one `(src_asic, dst_asic)`
  pair are placed **together, in the same logical direction** on each side.
- **Port pools are per `(mesh, chip)`**, independent on each side.
- **Both ends must agree on Z-ness** (both NESW, or both Z). Directions within NESW may
  differ per side (a cable can be W on one end, E on the other).
- **A `(mesh, chip, direction)` serves exactly one neighbor mesh** (§2). Once a direction on a
  chip is committed to a neighbor mesh, only cables to that same neighbor mesh may take more
  channels of that direction; a cable to a different neighbor mesh must use a different free
  direction on that chip (or, for NESW overflow, fall through to Z / the next round). This is
  enforced uniformly for NESW and Z — there is **no special-case Z map**.

## 4. The algorithm (rank 0)

Three steps, all **both-sides**:

### Phase 1a — assign Z (first)

For every connection where `should_assign_z_direction(src, dst)` is **true**, assign
**Z on both sides**. Marked connections claim Z **up front**, before anything touches NESW,
and because they go straight to Z they never occupy NESW — freeing those ports for Phase 1b.

**No NESW fallback here.** A marked connection is *required* to be Z, so if a cable it needs
(up to the count budget) cannot be placed on Z, that is a **hard `TT_FATAL`** — Phase 1a never
leaves a marked connection on NESW. The failure is one of two kinds, and the error message
distinguishes them (§7.4):
- **Cross-mesh Z conflict** — that chip's Z is already owned by a *different* marked neighbor
  mesh (two `assign_z` connections contending for one Z lane). Message names both mesh pairs.
- **Z capacity exhaustion** — the boundary cannot reach its requested count on Z (not enough
  free Z channels on the eligible chips). Message reports the shortfall.

Consequently Phase 1a leaves **no marked cable on NESW**: every attempted marked cable is
either placed on Z or the run fatals.

### Phase 1b — round-robin NESW

All remaining (unmarked) connections → **NESW on both sides**, via round-robin, up to each
boundary's **count budget** (§5):

```
repeat in rounds until no progress:
  for each boundary (srcMesh, dstMesh) in deterministic order:      # round-robin meshes
    if placed[boundary] >= count_budget[boundary]: continue         # honor count (target + cap)
    pick the next unplaced eligible cable for this boundary         # round-robin chips (per-boundary cursor)
    if srcChip has a free NESW direction with >= N free channels
       AND dstChip has a free NESW direction with >= N free channels:
         place both sides; mark cable resolved; occupy both pools; placed[boundary]++
    # one cable per boundary per round -> fairness across boundaries
```

### Phase 2 — round-robin leftover Z

Connections still unplaced after 1b (NESW overflow) → **remaining Z on both sides**, also
round-robin and also bounded by the same per-boundary count budget (a boundary that already
hit `count` in 1b takes no Z here). Phase 1a's marked connections already claimed their Z, so
leftovers use what remains.

## 5. Strict vs relaxed = a cable filter + a per-boundary count budget

There is **one** allocator. Mode decides two things: which cables are *eligible* per boundary,
and how the per-boundary **count budget** is expressed. The round-robin honors that budget in
**both** modes — it keeps giving a boundary turns until it reaches `count`, and never places
more than `count`.

| | Relaxed (`requested_intermesh_connections`) | Chip-pinned strict (`requested_intermesh_ports`) |
|---|---|---|
| Eligible cables | all cables on the boundary | only cables on the MGD-**pinned exit chips** |
| Count budget | one `count` per mesh pair | per pinned exit chip (summed for the boundary) |
| Allocation | round-robin (§4) up to the budget | round-robin (§4) over the pinned subset, up to the per-chip budget |
| Dst side | cable's PSD dst chip | cable's PSD dst chip (src pins only) |

The MGD `count` is **both a target the allocator tries to fill and a cap it won't exceed**, in
either mode. The round-robin keeps handing a boundary turns until it hits `count` (or it runs
out of eligible cables / free ports), then stops. Two kinds of "strict" fold into this:

1. **Chip pinning** (via `requested_intermesh_ports`) — restrict eligible src exit chips (the
   cable filter) *and* apply the count budget per pinned exit chip.
2. **Required connection count between meshes** — expressed as the boundary's count budget the
   allocator fills toward.

The allocator does **not** decide whether the target was actually met — it just fits as close
to `count` as the physical ports allow, fairly. **Sufficiency is judged afterward** by the
existing validation step (`validate_requested_intermesh_connections` / `num_resolved_between`),
which fails a boundary that resolved fewer than required (`≥1` for relaxed, the exact
per-chip/per-pair count for strict). Keep `count`, `num_channels`, and `num_resolved_between`
in the **same unit** (channels) so the budget and the validation agree.

## 6. Worked example — mesh 63 (wrap M0 + ring hop M62)

Shared chips 2–5 each: `2 NESW ports (dir W or E) + 2 Z ports`. PSD: M63↔M0 = 16 cables on
**all 8** chips (ch8/ch9, QSFP); M63↔M62 = 8 cables on chips 2–5 (ch4/ch5, TRACE).

- **Today:** M0 (processed first, `should_assign_z=false`) takes the 2 NESW ports on each of
  chips 2–5. M62 finds only Z ports left → 0 → fatal.
- **With this design, if the wrap is marked `assign_z`:** M0 → Z in Phase 1a (frees NESW on
  chips 2–5), M62 → NESW in Phase 1b. Both resolve. Clean.
- **Even if the wrap is *not* marked `assign_z`:** Phase 1b round-robin gives M62 a turn on
  some shared chips (≥1 NESW), and its overflow falls to Z in Phase 2 — so it no longer
  strands.

**Caveat:** `should_assign_z_direction(63↔0)` currently returns `false`, even though the wrap
physically rides the Z lane (ch8/ch9). Getting the *correct* placement (wrap on Z, hop on
NESW) depends on the classifier marking the wrap — a **separate fix** from this allocator.
The allocator alone stops the fatal; the classifier fix makes the placement right.

## 7. Implementation

### 7.1 Where it hooks in

`generate_intermesh_connectivity()` keeps its shape; the internals change:

| step | today | after |
|---|---|---|
| per-mesh | `generate_port_descriptors_for_exit_nodes` **assigns** ports | **gathers** cables only (no `port_id`) |
| gather | `forward_descriptors_to_controller` sends `PortDescriptor`s | sends the gathered cable records |
| rank 0 | `pair_logical_intermesh_ports` matches + reconciles Z/NESW | **round-robin allocator** (§4) assigns both sides |
| broadcast / bind | unchanged | unchanged |

Output stays `AnnotatedIntermeshConnections`, so `forward_intermesh_connections_from_controller`,
`intermesh_chan_to_peer_`, `exit_node_directions_`, and routing-table build are untouched.

### 7.2 Gather (per mesh, no assignment)

Replace the port-assigning body of `generate_port_descriptors_for_exit_nodes` /
`propose_port_descriptors_for_exit_nodes` with pure collection. Keep: the deterministic
`exit_nodes` sort, the `is_ethernet_link_up` skip, the strict-mode `requested_exit_nodes`
filter (chip pinning). Group `exit_nodes` by `(src_asic, dst_asic)` into links. Resolve **both**
endpoints to `FabricNodeId` (`{MeshId, ChipId}`) at gather time — the src is local, and the dst
`AsicID` resolves via `topology_mapper_->get_fabric_node_id_from_asic_id(dst_asic)` (the same
call `propose_...` already makes per host), so no ASIC ids need to cross to rank 0:

```cpp
struct GatheredLink {
    FabricNodeId src_node;                 // exit-node on src mesh (local)
    FabricNodeId dst_node;                 // peer node on dst mesh (resolved from dst_asic via topology_mapper_)
    std::vector<std::size_t> connection_hashes;   // one per eth channel of this link (e.g. 2)
};
```

`should_assign_z` is **per mesh pair**, so it does not need to live on the link — rank 0 can
call `mesh_graph_->should_assign_z_direction(src_node.mesh_id, dst_node.mesh_id)`. Serialize
`GatheredLink`s in `forward_descriptors_to_controller` (extend the payload, or add a sibling
table alongside the existing one during migration).

### 7.3 Build the both-sides view on rank 0

Each physical cable appears **twice** in the gathered set — once from each endpoint mesh —
sharing a `connection_hash`. Join them:

```cpp
struct Cable {
    std::size_t connection_hash;
    FabricNodeId src_node;                 // from the src-mesh GatheredLink
    FabricNodeId dst_node;                 // from the dst-mesh GatheredLink
    uint32_t num_channels;                 // e.g. 2
    bool placed = false;
    std::optional<RoutingDirection> src_dir, dst_dir;  // filled on placement
};
// key cables by connection_hash; a cable is complete only when both endpoints are present.
// boundary key is the mesh pair (src_node.mesh_id, dst_node.mesh_id).
std::map<std::pair<MeshId,MeshId>, std::vector<Cable*>> cables_by_boundary;   // deterministic order
```

Port pools are per `(mesh, chip)`, seeded from `mesh_graph_->get_mesh_edge_ports_to_chip_id()`
minus anything already consumed. Alongside the taken-port set we track the **direction owner**:
which neighbor mesh each `(mesh, chip, direction)` is committed to. This one map replaces the
old Z-only `chip_to_z_neighbor_mesh_id` and enforces the §2 invariant for **every** direction:

```cpp
// occupied[node] = set of taken port_id_t {direction, logical_chan} on that (mesh,chip)
std::unordered_map<FabricNodeId, std::unordered_set<port_id_t, hash_pair>> occupied;

// dir_owner[(node,direction)] = the single neighbor mesh that direction serves on that node.
// Applies to NESW and Z identically (no special-case Z map).
std::map<std::pair<FabricNodeId,RoutingDirection>, MeshId> dir_owner;
```

Shared helper — find a direction on one side with enough free channels **that is not already
owned by a different neighbor mesh**:

```cpp
// returns the logical ports to use (size == need) for a usable direction on `node`
// toward `neighbor` mesh; a direction is usable if it is unowned OR already owned by `neighbor`,
// and has >= need free channels. NESW iterated in enum order when want_z==false, Z otherwise.
// nullopt if none fits.
std::optional<std::vector<port_id_t>>
find_free_dir(FabricNodeId node, MeshId neighbor, uint32_t need, bool want_z);

// both-sides atomic placement: succeeds only if BOTH ends have a usable direction.
bool place_cable(Cable& c, bool want_z) {
    auto s = find_free_dir(c.src_node, c.dst_node.mesh_id, c.num_channels, want_z);
    auto d = find_free_dir(c.dst_node, c.src_node.mesh_id, c.num_channels, want_z);
    if (!s || !d) return false;
    occupy(c.src_node, *s); occupy(c.dst_node, *d);
    dir_owner[{c.src_node, s->front().first}] = c.dst_node.mesh_id;   // record owner (both sides)
    dir_owner[{c.dst_node, d->front().first}] = c.src_node.mesh_id;
    c.src_dir = s->front().first; c.dst_dir = d->front().first; c.placed = true;
    emit_annotated(c, *s, *d);          // two entries sharing connection_hash, as today
    return true;
}
```

Because `find_free_dir` skips directions owned by another neighbor mesh, a cable to a new
neighbor simply falls through to the next free direction (or fails, letting the caller drop to
Z / the next round). No cable can ever be handed a channel of a direction already committed
elsewhere — the failure mode the old Z-only map plugged, now closed for NESW too.

### 7.4 The three phases

`count_budget[boundary]` is the per-boundary channel target from the MGD (§5): the relaxed
`requested_intermesh_connections` count, or the summed strict `requested_intermesh_ports` count
over that boundary's pinned exit chips. `placed[boundary]` tracks how much has been committed.

```cpp
// Phase 1a — marked connections MUST get Z on both sides, up to the boundary's count budget.
// There is NO NESW fallback: any attempted marked cable that cannot take Z is a hard TT_FATAL,
// either a cross-mesh conflict (that chip's Z is owned by a different marked neighbor mesh -> two
// assign_z connections fighting for one Z lane) or Z capacity exhaustion (the boundary can't hit
// its requested count on Z). fatal_assign_z_failure() always terminates.
for (auto& [boundary, cables] : cables_by_boundary)
    if (mesh_graph_->should_assign_z_direction(boundary.first, boundary.second))
        for (Cable* c : cables) {
            if (placed[boundary] >= count_budget[boundary]) break;      // count is target + cap
            if (place_cable(*c, /*want_z=*/true)) { placed[boundary]++; continue; }
            fatal_assign_z_failure(*c, boundary, placed[boundary], count_budget[boundary]);  // never returns
        }

// Phase 1b — round-robin NESW over the rest, bounded by the count budget.
for (bool progress = true; progress; ) {
    progress = false;
    for (auto& [boundary, cables] : cables_by_boundary) {              // round-robin meshes
        if (marked(boundary)) continue;
        if (placed[boundary] >= count_budget[boundary]) continue;      // stop at the target/cap
        Cable* c = next_unplaced(cables, /*advance per-boundary chip cursor*/);  // round-robin chips
        if (c && place_cable(*c, /*want_z=*/false)) { placed[boundary]++; progress = true; }  // one per boundary per round
    }
}

// Phase 2 — round-robin leftover Z for still-unplaced cables, same budget.
// place_cable(want_z=true) already refuses a Z direction owned by a different neighbor mesh
// (find_free_dir), so an unmarked overflow cable simply stays unplaced rather than stealing a
// marked connection's Z lane. No error here -- these are best-effort overflow, and any shortfall
// is caught by validate_requested_intermesh_connections afterward.
for (bool progress = true; progress; ) {
    progress = false;
    for (auto& [boundary, cables] : cables_by_boundary) {
        if (placed[boundary] >= count_budget[boundary]) continue;
        for (Cable* c : cables)
            if (!c->placed && place_cable(*c, /*want_z=*/true)) { placed[boundary]++; progress = true; break; }
    }
}
```

**`fatal_assign_z_failure` — the detailed error (always terminates).** A marked cable that
cannot take Z is unrecoverable in Phase 1a (no NESW fallback). Look at the Z direction owner on
each side: if either side's chip Z is already owned by a **different** neighbor mesh, that is a
cross-mesh conflict — name both `assign_z` mesh pairs. Otherwise the boundary simply ran out of
free Z channels before hitting its count — report the capacity shortfall. Either way, `TT_FATAL`:

```cpp
[[noreturn]] void fatal_assign_z_failure(
    const Cable& c, std::pair<MeshId,MeshId> boundary, uint32_t placed, uint32_t budget) {
    auto conflict = [&](FabricNodeId node, MeshId wanted_neighbor) -> std::optional<MeshId> {
        auto it = dir_owner.find({node, RoutingDirection::Z});
        if (it != dir_owner.end() && it->second != wanted_neighbor) return it->second;  // owner != us
        return std::nullopt;   // unowned, or owned by the same pair (-> capacity, not a conflict)
    };
    // (1) cross-mesh Z conflict on either side -> name both assign_z mesh pairs.
    if (auto other = conflict(c.src_node, c.dst_node.mesh_id))
        TT_FATAL(false,
            "assign_z conflict on M{}/chip{}: connection M{}-M{} (assign_z) wants this chip's Z "
            "lane, but it is already assigned to M{}-M{} (assign_z). A chip's Z direction can "
            "serve only one neighbor mesh. Remove assign_z from one of these two connections in "
            "the mesh graph descriptor.",
            *c.src_node.mesh_id, c.src_node.chip_id, *c.src_node.mesh_id, *c.dst_node.mesh_id,
            *c.src_node.mesh_id, **other);
    if (auto other = conflict(c.dst_node, c.src_node.mesh_id))
        TT_FATAL(false,
            "assign_z conflict on M{}/chip{}: connection M{}-M{} (assign_z) wants this chip's Z "
            "lane, but it is already assigned to M{}-M{} (assign_z). A chip's Z direction can "
            "serve only one neighbor mesh. Remove assign_z from one of these two connections in "
            "the mesh graph descriptor.",
            *c.dst_node.mesh_id, c.dst_node.chip_id, *c.dst_node.mesh_id, *c.src_node.mesh_id,
            *c.dst_node.mesh_id, **other);
    // (2) no cross-mesh owner -> Z capacity exhaustion for a required (assign_z) boundary.
    TT_FATAL(false,
        "assign_z capacity exhausted for connection M{}-M{} (assign_z): resolved {} of {} "
        "requested channel(s) on Z before running out of free Z ports (src M{}/chip{}, dst "
        "M{}/chip{}). assign_z connections have no NESW fallback -- reduce the requested count, "
        "free Z ports on these chips, or remove assign_z for this connection.",
        *boundary.first, *boundary.second, placed, budget,
        *c.src_node.mesh_id, c.src_node.chip_id, *c.dst_node.mesh_id, c.dst_node.chip_id);
}
```

For the running example the conflict case prints, e.g., `assign_z conflict on M0/chip3:
connection M0-M63 (assign_z) wants this chip's Z lane, but it is already assigned to M0-M1
(assign_z). ... Remove assign_z from one of these two connections` — the exact pair-vs-pair
message you asked for.

Mode enters via which cables populate `cables_by_boundary` **and** how `count_budget` is built:
**relaxed** = all cables, budget = the mesh-pair count; **chip-pinned strict** = only cables
whose `srcChip` is in the MGD's `requested_intermesh_ports` (applied during gather via the
existing `requested_exit_nodes` filter), budget = the summed per-pinned-chip count. In both
modes the allocator fills toward `count` and never exceeds it. Whether the target was actually
met is **not** decided here — `validate_requested_intermesh_connections` runs afterward and
fails any boundary that resolved fewer than required.

The pseudocode uses a single `placed[boundary]` counter for readability. **Relaxed** genuinely
has one budget per mesh pair, so that is exact. **Strict** budgets are per pinned exit chip, so
the real implementation keys the counter by `(boundary, srcChip)` (mirroring today's
`assigned_ports_for_src_node` / `requested_ports_for_src_node` in `pair_logical_intermesh_ports`)
and the "stop at budget" check becomes per-chip — otherwise the boundary sum can over-fill one
pinned chip while starving another.

### 7.5 Determinism

`cables_by_boundary` keyed by sorted `(srcMesh,dstMesh)`; cables within a boundary sorted by
`(srcChip, connection_hash)`; `find_free_dir` iterates ports in a fixed `(direction, channel)`
order. No `unordered_map` iteration feeds any assignment decision.

## 8. Open decisions

- Phase 1a has **no NESW fallback** — a marked connection that cannot take Z (cross-mesh
  conflict *or* Z capacity exhaustion) is a hard `TT_FATAL` (§7.4). This is settled, not open.
- Phase 1a fairness — straight assignment vs round-robin among marked boundaries. Since any
  marked-vs-marked Z contention on a chip fatals regardless, this only affects the order in
  which capacity is consumed before a possible capacity fatal.
- **No NESW reclaim round** (settled). Re-running NESW before Phase 2 is a no-op here: Phase 1b
  already iterates to a fixpoint (`repeat until no progress`), so all *reachable* NESW is filled
  before Phase 2; and with no NESW fallback, Phase 1a consumes zero NESW and Phase 2 touches only
  Z, so nothing frees NESW mid-run to reclaim. A cable overflows to Z only because *its own*
  pinned src/dst chips have no free NESW direction — free NESW on other chips is unreachable for
  a chip-pinned cable, so a reclaim round cannot move it. (This relies on Phase 1b's per-boundary
  cursor advancing *past* a cable it cannot place this round — rather than wedging on it — so the
  fixpoint truly exhausts every reachable NESW placement. That cursor behavior is a separate
  requirement still to be pinned down in §4/§7.4.)

## 9. Rollout / validation

1. Gather-only refactor + log the new both-sides view (no behavior change).
2. Add the round-robin allocator behind a flag on rank 0; diff against goldens
   (`SC16BlitzSuperpod_intermesh`, and the SC36 subtorus slices which should now resolve
   more boundaries).
3. Flip default once goldens match or improve; keep the old path one release.

---

## Appendix A — Alternative considered: min-cost-flow

An earlier proposal modeled the assignment as a global capacitated flow (per-chip open-slot
accounting → candidate graph per mesh pair → Dinic / successive-shortest-path with NESW and Z
as separate commodities). It delivers the same two structural wins but needs a flow network
and cost tuning. **Rejected in favor of the round-robin (§4)** — same fairness and structural
Z↔Z handling, far less machinery, still deterministic.
