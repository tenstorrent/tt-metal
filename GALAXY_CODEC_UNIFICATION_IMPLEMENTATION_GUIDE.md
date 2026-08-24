# Fabric Codec Unification — Implementation Guide

> **Step names:** every numbered step in this document has a short name — see
> **§0 Step index** in `GALAXY_CODEC_UNIFICATION_STATUS.md`. Prefer the names when discussing
> the work; the numbers exist for cross-referencing between these three documents.


Ordered, file-by-file execution guide. Third of three documents:

| doc | carries |
|---|---|
| `GALAXY_CODEC_UNIFICATION_COMPARISON.md` | the two 2D implementations side by side |
| `GALAXY_CODEC_UNIFICATION_PLAN.md` | scope, fork ledger, blockers, decisions, phasing, risks |
| **this document** | exact sites, exact edits, exact ordering, and the gates |

Read the comparison for *what differs* and the plan for *why*. Line numbers are against `42204c541c3`
and will drift — each step names the enclosing symbol so it stays findable.

Semantics are owned by `GALAXY_DEVICE_ROUTE_CODEC_CONTRACT.md` and
`GALAXY_DEVICE_ROUTER_KERNEL_CONTRACT.md`. When an edit here looks like it changes packet meaning,
it is wrong — stop and check the contract.

---

## How to use this guide

- Steps are numbered `<phase>.<step>`. Within a phase, steps are ordered by dependency.
- **Gate** blocks are mandatory. Do not start the next phase with a red gate.
- ⚠ marks a step where getting it wrong is silent (wrong results or a hang) rather than a build
  failure.
- No builds are run by this guide. Build when you choose to; the gates say what must pass.
- **All testing is deferred to Phase 7** (owner call, 2026-08-21). Steps that only add tests are
  skipped in place; gates keep their non-test conditions.

### Orientation: the ten files that matter most

```text
host, generation + embed
  tt_metal/fabric/control_plane.cpp                       L1 embed, ABI branch
  tt_metal/fabric/compressed_routing_path.cpp             vector packing
  tt_metal/fabric/mcast_reverse_tree.{hpp,cpp}            reverse trees + gate
  tt_metal/fabric/express_ring_topology.{hpp,cpp}         axis topology derivation
  tt_metal/fabric/routing_table_generator.cpp             ring derivation call site

host, build config
  tt_metal/fabric/fabric_context.{hpp,cpp}                defines, route-buffer tiers
  tt_metal/fabric/fabric.cpp                              worker encode defines
  tt_metal/fabric/compute_mesh_router_builder.cpp         kernel define
  tt_metal/fabric/erisc_datamover_builder.cpp             kernel CT args

shared ABI
  tt_metal/hostdevcommon/api/hostdevcommon/fabric_common.h    IndexedMeshRoutingFields, L1 layout
  tt_metal/fabric/fabric_edm_packet_header.hpp                packet header + tiers

device
  tt_metal/fabric/hw/inc/tt_fabric_api.h                              producers
  tt_metal/fabric/hw/inc/mesh/api.h                                   mesh API
  tt_metal/fabric/hw/inc/udm/tt_fabric_udm{,_impl}.hpp                UDM producers
  tt_metal/fabric/hw/inc/edm_fabric/fabric_edge_node_router.hpp        legacy edge recompute
  tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp      RX decode/admit/forward
  tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_ct_args.hpp    CT state
```

---

# Phase 0 — Fix the live single-hop ABI mismatch

Standalone. Ship separately from the refactor. This is a correctness fix for shipping Galaxy
workloads (plan §3.1 B4).

## 0.1 ⚠ Route the single-hop helpers to the indexed encoder

**File:** `tt_metal/fabric/hw/inc/tt_fabric_api.h`

`fabric_set_single_hop_unicast_route_from_direction` (≈line 48) has **no** express gate. It writes
`route_buffer[0] = single_hop_route_cmd_by_direction[dir]` with `hop_index == 0`. On an express mesh
the kernel reads `route_buffer[local_y]` / `route_buffer[y_size + local_x]`, so the packet is
routed by whatever stale byte sits at that index.

`fabric_set_indexed_single_hop_unicast_route_from_direction` (≈line 586) is already written and
correct, and has **zero callers**.

⚠ **Two ordering corrections to this step, found while doing it.**

**1. The shape-define fallback must move first.** `FABRIC_EXPRESS_MESH_Y_SIZE`'s zero-default lived at
line 151, *after* the single-hop helper at line 48. Referencing it from the helper without moving it
does not compile on an ERISC build, where the host never emits the shape defines. Hoist both the
fallback `#define` block and the `Y+X <= route_buffer` `static_assert` to just after the
`MeshRoutingFields` alias (≈line 24), before first use.

**2. Use `#if / #else`, not an early return.** The guide originally said to add the express call and
`return`, then move the pre-existing `ASSERT(next_hop_direction != Z)` below it. `#else` is better:
the legacy body then does not compile at all on an express build, so there is no dead hop-program
encode sitting in the binary and no risk of the two arms drifting.

Add a forward declaration before the helper (the indexed one is defined later in the file):

```cpp
// Defined in the indexed codec section below; the express path delegates to it.
inline void fabric_set_indexed_single_hop_unicast_route_from_direction(
    volatile tt_l1_ptr HybridMeshPacketHeader* packet_header,
    eth_chan_directions next_hop_direction,
    uint16_t dst_dev_id,
    uint16_t dst_mesh_id,
    uint8_t mesh_y_size,
    uint8_t mesh_x_size);
```

then split the body:

```cpp
#if defined(FABRIC_EXPRESS_ENABLED)
    fabric_set_indexed_single_hop_unicast_route_from_direction(
        packet_header, next_hop_direction, dst_dev_id, dst_mesh_id,
        FABRIC_EXPRESS_MESH_Y_SIZE, FABRIC_EXPRESS_MESH_X_SIZE);
    return;
#else
    ASSERT(next_hop_direction != eth_chan_directions::Z);   // legacy arm only
    ...existing body...
#endif
```

**3. Make the same-mesh precondition explicit.** The original comment said *"Do not use this helper
for Z / inter-mesh traffic"* — the inter-mesh half is still true and now load-bearing. Neither arm
encodes a boundary crossing: the legacy arm leaves `route_buffer[1..]` stale for the far side to trip
over, and the indexed arm has no valid coordinates for a chip numbered in *another* mesh's id space.
The indexed helper's existing `ASSERT(dst_dev_id < mesh_y_size * mesh_x_size)` does not catch this —
it passes whenever the foreign id happens to be in range, then pokes the wrong slot. Add:

```cpp
ASSERT(dst_mesh_id ==
       reinterpret_cast<tt_l1_ptr routing_l1_info_t*>(ROUTING_TABLE_BASE)->my_mesh_id);
```

Note this makes the helper device-only for testing purposes (it reads L1), which is consistent with
the rest of the indexed encoder family — `fabric_set_indexed_unicast_route` and
`fabric_set_indexed_mcast_route` also read `ROUTING_TABLE_BASE`.

**Callers that this fixes** (no change needed in them):

```text
models/demos/deepseek_v3_b1/unified_kernels/all_reduce.hpp:212
models/demos/deepseek_v3_b1/unified_kernels/all_gather.hpp:182
models/demos/deepseek_v3_b1/unified_kernels/broadcast.hpp:175
models/demos/deepseek_v3_b1/unified_kernels/reduce_to_one_b1.hpp:255   (non-_from_direction form)
models/demos/deepseek_v3_b1/unified_kernels/sdpa_reduce_worker.hpp:131
```

`fabric_set_single_hop_unicast_route` (the non-`_from_direction` form) delegates, so it is covered.

> **Gate 0.** deepseek `all_gather`, `all_reduce`, `broadcast`, `reduce_to_one_b1` pass on an
> express Galaxy config. Add a device test that sends one single-hop packet in each of N/S/E/W and,
> on BH, Z, and checks arrival — the express path had no such coverage.

---

# Phase 1 — Separate the gates, generalize the axis topology

Behaviour-preserving. Nothing changes about which codec runs; only *how the decision is expressed*
and *what the indexed path is capable of*.

## 1.1 Audit and record the express-guard inventory (no code change)

**Files:** none — this step produces a written classification that makes Phase 3 mechanical.

`express_routing_enabled(mesh_id)` stays exactly as it is, and **every G2-G5 consumer keeps calling
it**. `FABRIC_EXPRESS_ENABLED` also stays, under its own name, on its unchanged condition — it stops
being a *codec* selector, which is a change of consumers, not of the define.

So there is no new define and no repoint here. What Phase 1 owes is the classification, because
Phase 3 deletes the wrong thing if any site is miscategorised. Confirm each of the nine sites:

```text
CODEC  → #if defined(FABRIC_2D) in Phase 3
  hw/inc/tt_fabric_api.h:151,158,205,288            producers + shape fallback + static_assert
  hw/inc/mesh/api.h:1198                            source multi-inject (guard removed in 2.2)
  hw/inc/udm/tt_fabric_udm.hpp:482                  relay-to-mux selection
  hw/inc/udm/tt_fabric_udm_impl.hpp:107             initial-direction calculation
  hw/inc/edm_fabric/fabric_erisc_router_ct_args.hpp:432-447
                                                    express_enabled + MESH_*_SIZE + intermesh ingress
                                                    (all three payloads are 2D facts, not express)
  impl/dispatch/kernels/cq_relay.hpp:89             route-vs-unicast producer choice

EXPRESS → keeps FABRIC_EXPRESS_ENABLED, untouched
  hw/inc/edm_fabric/routing_plane_connection_manager.hpp:21
        TT_FABRIC_MAX_ROUTING_PLANE_CONNECTIONS 6 vs 4. An array bound in a type, so it cannot be a
        CT arg. Worker-side only -- linear/api.h, mesh/api.h and CCL kernels include this header;
        the router does not.
```

Also confirm the negative: every *other* express-specific device fact is already a host-derived CT
arg, so nothing else needs a guard.

```text
Z sender on this router   express_arm_is_realizable<DOWNSTREAM_EDM_SIZE, Z>()
VC0 five wide             ACTUAL_VC0_SENDER_CHANNELS
VC1 counter credits       vc1_uses_counter_credits
per-sender BFC role       SENDER_CH_i_IS_INJECTION
speedy / trim             enable_speedy_vc0, trim CT args
deadlock avoidance        enable_deadlock_avoidance
```

That is by design (kernel §5: the kernel *consumes* `SENDER_CH_i_IS_INJECTION`, it does not derive
it), so a surprise here means something drifted and Phase 3 should stop until it is understood.

Two earlier drafts of this step are recorded so they are not revived: splitting
`express_routing_enabled` into four predicates (`mesh_has_z_ports`, `express_resource_shape_enabled`,
`fabric_optimizations_restricted`), which widened the diff into the VC-shape and optimization code
this project leaves alone; and adding a `FABRIC_2D_HAS_Z_PORTS` define, which invented a second
spelling for a fact whose derivation, value, and site are all unchanged. These sites stay untouched:

```text
fabric_builder_context.cpp:45,53,98              G4 resource shape
builder/fabric_builder_config.cpp:48,56          G4
erisc_datamover_builder.cpp:793,1074-1130        G5 speedy / trim / forwarding capture
fabric_context.cpp:345                           G3 deadlock avoidance
compute_mesh_router_builder.cpp:271,294,333,368  G2/G3 wiring + injection policy
builder/router_wiring_rules.cpp (all)            G2
builder/fabric_edge_capability.cpp:88-108        G2
control_plane.cpp:1684                           G3 protected-ring query
```

The G1 sites, left for Phase 3: `fabric.cpp:392`, `control_plane.cpp:1976`,
`compute_mesh_router_builder.cpp:911`, `erisc_datamover_builder.cpp:1261`,
`fabric_context.cpp:364`, `fabric_context.cpp:373`.

## 1.2 ⚠ Add a regression test for gate separation

**File:** new, `tests/tt_metal/tt_fabric/fabric_router/test_abi_gate_separation.cpp`

Plan risk R4 is that unification silently re-enables speedy or trim on an express mesh, which
deadlocks. Since step 1.1 no longer renames the predicates, this test *is* the guard. Assert the
built router config directly rather than the predicate:

```text
for each express 2D mesh fixture:
    EXPECT_FALSE(built router has speedy_vc0 enabled)
    EXPECT_FALSE(built router uses the trim fast path / forwarding capture)
    EXPECT_TRUE(vc1_uses_counter_credits)
for each mesh with no Z ports:
    EXPECT no Z sender in the built router's sender set
```

This test must survive Phase 3 **unchanged**. If Phase 3 makes it fail, the flip leaked G1 into
G4/G5 — which is exactly the failure the dropped four-predicate refactor was meant to prevent, caught
more cheaply.

## 1.3 Remove the `X ≤ 4` bound (plan B1)

**File:** `tt_metal/hostdevcommon/api/hostdevcommon/fabric_common.h`, `IndexedMeshRoutingFields`

`MAX_INDEXED_MESH_X = 4` (≈line 299) excludes in-tree `[8,8]`, `[8,16]`, `[16,8]`, `[1,16]`.

Replace the per-axis X cap with a total-region bound. Keep `INDEXED_VECTOR_TABLE_BYTES` at its
current value (1028) — it is the *slot size*, and the `static_assert` at line 529 plus the
`routing_l1_info_t` union depend on it. Change only the *validation*:

```cpp
// The packed reverse-tree descriptor carries two 6-bit row indices, so neither axis may exceed 64.
// Beyond that the only real limit is the region the L1 slot provides.
static constexpr uint32_t MAX_INDEXED_MESH_AXIS = 64;
```

Then in `pack_indexed_route_vectors` (≈line 377) replace:

```cpp
if (y_size > MAX_INDEXED_MESH_Y || x_size > MAX_INDEXED_MESH_X || ...
```

with:

```cpp
if (y_size > MAX_INDEXED_MESH_AXIS || x_size > MAX_INDEXED_MESH_AXIS ||
    vectors_region_bytes(y_size, x_size) > INDEXED_VECTOR_TABLE_BYTES) {
    return false;
}
```

Keep `MAX_INDEXED_MESH_Y` / `MAX_INDEXED_MESH_X` as the *bound shape* the table is sized for
(`[64,4]`), since `INDEXED_VECTOR_TABLE_BYTES` is expressed in terms of them and the comment there
explains the Clang constexpr workaround. Add a comment distinguishing "bound shape used to size the
slot" from "shapes the slot admits."

Add static asserts for the newly admitted shapes next to the existing `[32,4]` / `[64,4]` ones
(≈line 533):

```cpp
static_assert(IndexedMeshRoutingFields::hybrid_region_fits(8, 8), "[8,8] hybrid layout must fit");
static_assert(IndexedMeshRoutingFields::hybrid_region_fits(8, 16), "[8,16] hybrid layout must fit");
static_assert(IndexedMeshRoutingFields::hybrid_region_fits(16, 8), "[16,8] hybrid layout must fit");
static_assert(IndexedMeshRoutingFields::hybrid_region_fits(32, 32), "[32,32] is the shape bound");
```

Also check `MCAST_ROW_BITS_WORDS = 2` (line 553) still covers the X axis: it is sized for 64 rows,
and X ≤ 64 after this change, so it does. Note that in a comment — it was previously only reasoned
about for Y.

## 1.4 Hoist X-ring derivation out of the express-only branch (plan B3)

**File:** `tt_metal/fabric/routing_table_generator.cpp` ≈lines 51-64

Today:

```cpp
for (each mesh) {
    auto rings = derive_express_ring_topology(...);
    if (!rings.has_value()) { continue; }          // ← X rings never derived for non-express
    express_rings_[m] = ...;
    auto x_rings = derive_ordinary_ring_topology(mesh_graph, MeshId{m}, 1);
    if (x_rings.has_value()) { x_rings_[m] = ...; }
}
```

Restructure so both axes are always derived:

```cpp
for (each mesh) {
    if (auto rings = derive_express_ring_topology(mesh_graph, MeshId{m}); rings.has_value()) {
        express_rings_[m] = std::make_unique<...>(std::move(*rings));
    }
    // Every mesh needs an axis topology on both dimensions: the indexed multicast encoder builds a
    // reverse tree per axis, and a null topology silently removes multicast rather than failing.
    y_axis_[m] = derive_axis_topology(mesh_graph, MeshId{m}, 0);
    x_axis_[m] = derive_axis_topology(mesh_graph, MeshId{m}, 1);
}
```

## 1.5 ⚠ Add the line-axis topology and one accessor (plan B2, D5)

**Files:** `tt_metal/fabric/express_ring_topology.{hpp,cpp}` (see plan §9 Q3 on renaming to
`axis_route_topology.*`)

This is the highest-value step in the phase: without it, unification **silently removes multicast**
on every LINE-axis 2D mesh (`single_bh_galaxy` `[8,4]`, `p150_x8` `[2,4]`, `dual_bh_lb` `[2,4]`,
`16x4_dual_bh_galaxy_2d` `[16,4]`, `quad_galaxy` `[8,16]`, `dual_galaxy` `[8,8]`, and the small
shapes).

The reverse-tree builder consumes only three things from `ExpressRingTopology`
(`mcast_reverse_tree.cpp:29-93`): `axis_len`, `axis_dim`, and `next_row(src, dst)`. So a line axis is
expressible in the existing struct.

Add:

```cpp
// The plain line along `axis`: one domain over every coordinate, no chords, next hop always one
// step toward the destination. Always defined, so it is the fallback that guarantees every 2D mesh
// has an axis topology on both dimensions. Trivially an arborescence from every root, so the
// multicast gate always passes.
ExpressRingTopology derive_line_axis_topology(const MeshGraph& mesh_graph, MeshId mesh_id, int axis);
```

Implementation mirrors `derive_ordinary_ring_topology` (`express_ring_topology.cpp:549`) except:

- no `axis_wraps` precondition — it always succeeds;
- edge validation walks `coord → coord+1` for `coord in [0, len-1)` only (no wrap edge);
- `forward_cycle` is the coordinate order, `pos_in_domain[c] = c`, `domain_of` all 0.

`next_row(src, dst)` must return `src + 1` when `dst > src` and `src - 1` when `dst < src`. Check
whether `ExpressRingTopology::next_row` (`express_ring_topology.cpp`) already yields that for a
single-domain non-wrapping cycle. If it computes distance modulo `len`, it will take the wrap
shortcut on a line — **that is a wrong route over a nonexistent edge**. Either add a `bool wraps`
member consulted by `next_row`, or give the line derivation its own `next_row`. Verify this
explicitly; it is the one place in this step where a bug is silent.

Then the single accessor (plan D5):

```cpp
// control_plane.hpp
// The route topology governing `axis`: express chord decomposition where the mesh declares chords,
// the ordinary ring where the axis wraps, else the plain line. Never null for a 2D mesh.
const ExpressRingTopology* axis_topology(MeshId mesh_id, int axis) const;
```

and a free function in the topology file that picks:

```cpp
ExpressRingTopology derive_axis_topology(const MeshGraph& mesh_graph, MeshId mesh_id, int axis) {
    if (axis != /*orthogonal*/ 1) {
        if (auto express = derive_express_ring_topology(mesh_graph, mesh_id); express.has_value()) {
            return std::move(*express);
        }
    }
    if (auto ring = derive_ordinary_ring_topology(mesh_graph, mesh_id, axis); ring.has_value()) {
        return std::move(*ring);
    }
    return derive_line_axis_topology(mesh_graph, mesh_id, axis);
}
```

Note `derive_express_ring_topology` takes no axis — it derives the mesh's express axis. Preserve the
existing convention from `ControlPlane::ring_for_direction` (`control_plane.cpp:1611-1616`): axis 1
(E/W) is the ordinary ring, the other axis carries chords. Do not change that convention here.

Finally, repoint `ControlPlane::ring_for_direction` to `axis_topology`, or keep it as a thin wrapper.
Its other callers (`control_plane.cpp:1630`, `1684`) are G2/G3 and must keep seeing *express* rings
specifically — check each before repointing. ⚠ `control_plane.cpp:1684` reads `get_express_rings`
directly for protected-ring queries; leave it alone.

## 1.6 Assert the Z-arm invariant; defer the WH admission shape (plan B7)

**File:** `tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp`

⚠ Earlier drafts called this a "re-key." It is not — there is nothing to re-key. The WH+VC1
compass-only bit-test admission exists at exactly one site, inside the **legacy** admit function:

```cpp
// fabric_erisc_router.cpp:730 — legacy can_forward_packet_completely
#if defined(ARCH_WORMHOLE) && defined(FABRIC_2D_VC1_ACTIVE)
```

`admit_express_dispatch` has no WH variant. That is harmless today (express is BH-only, so WH never
reaches the indexed path) but Phase 3 puts WH on it and Phase 4 deletes the legacy admit that holds
the exception. **That is decision Q6, resolved at Gate 3 when WH ERISC size is first observable** —
not Phase 1 work.

What *is* Phase 1 work is asserting the half that already holds. `express_arm_is_realizable<>`
(`fabric_erisc_router.cpp:675`) compares a direction's compact index against this router's actual
downstream array size, so a WH router's Z arm already collapses to `ASSERT(false); ok = false;` at
compile time. Pin that so it cannot regress silently:

```cpp
// A router with no Z downstream must never instantiate a Z dispatch arm. This already holds
// structurally -- the compact index for Z lands past the end of a cardinal-only downstream array --
// but it is load-bearing once the indexed path is the only path, including on WH where there is no
// Z sender at all.
static_assert(
    !express_arm_is_realizable<DOWNSTREAM_EDM_SIZE, eth_chan_directions::Z>() || HAS_Z_DOWNSTREAM,
    "a router without a Z downstream must not instantiate a Z dispatch arm");
```

No new CT arg is required — `DOWNSTREAM_EDM_SIZE` already carries the fact. If a spelling for
`HAS_Z_DOWNSTREAM` is wanted, derive it from `express_arm_is_realizable<>` itself rather than adding
a builder-emitted flag; the goal is a tripwire, not new state.

## 1.7 Make the one-feeder-per-row gate fatal (plan D6, decision Q4)

**File:** `tt_metal/fabric/control_plane.cpp` ≈lines 1994-2003

Today a failed gate logs a warning and leaves the reverse-tree region **zeroed**. On device that
reads back as `feeder=0, command=0` for every row, `widen(0) == 0`, so the map is entirely empty, the
worker's own row carries no directions, and the multicast **silently delivers to nothing**.

Replace with `TT_FATAL`. Q4 settled this: a row with two feeders is structurally impossible on a
chordless 2D mesh — without chords a row's only way in is from its neighbour toward the root — so the
gate is a no-op there. And express multicast works today, meaning the check already passes on shipping
express topologies. A fatal cannot reject a configuration that currently functions.

⚠ The message must name **the row and both of its feeders**, because that is the whole diagnostic.
`build_mcast_reverse_tree` already produces exactly that text
(`mcast_reverse_tree.cpp:84-91`) — propagate it rather than summarising:

```cpp
if (!embed_mcast_reverse_trees(..., &failure)) {
    TT_FATAL(false, "mesh {}: cannot encode 2D multicast: {}", *mesh_id, failure);
}
```

Keep this inside the existing express-only branch for Phase 1 so the phase stays
behaviour-preserving; Phase 3 inherits it for every 2D mesh when the branch goes away.

Also add the device-side half, since a zeroed region must never reach a producer even if the host
check is somehow bypassed (a stale L1 image, a partially initialised mesh):

```cpp
// A real entry has a nonzero feeder index or command, so an all-zero region is detectable.
ASSERT(mcast_tree_region_is_valid(vectors, y_size, x_size));
```

## 1.8 Un-skip non-express meshes in route-buffer sizing (plan H5)

**File:** `tt_metal/fabric/fabric_context.cpp`,
`get_max_2d_indexed_route_bytes_from_topology` ≈line 359

Delete:

```cpp
if (!control_plane.express_routing_enabled(mesh_id)) { continue; }
```

The rest of the function is already shape-generic. It computes `shape[0] + shape[1]` per mesh and
`compute_packet_specifications` (line 148) already takes `std::max` against the legacy hop count, so
in Phase 1 this only ever *grows* the buffer — never shrinks it below what the legacy path needs.
Behaviour-preserving in the sense that matters (no packet overruns); it may bump a tier on some
non-express platform, which is exactly the R2 cost you want measured early rather than at Phase 3.

> **Gate 1.** No functional change on any platform. Full 1D + 2D + express regression green. New
> host tests green:
> - `test_abi_gate_separation` (step 1.2)
> - shape sweep: every descriptor under `tt_metal/fabric/mesh_graph_descriptors/*.textproto` with a
>   2D shape packs vectors, fits the region, and `Y+X` fits some tier
> - axis-topology coverage: `axis_topology(mesh, 0)` and `axis_topology(mesh, 1)` non-null for every
>   2D mesh
> - arborescence sweep: every 2D shape × every root × both axes passes the gate
>
> Record any tier bump step 1.8 causes, per platform.

---

# Phase 2 — Close the indexed-path gaps

Still behaviour-preserving: the indexed path becomes *capable* of everything the legacy path does,
but is not yet selected for non-express meshes.

## 2.1 Complete the indexed UDM producers (plan P7, P8)

**Files:** `tt_metal/fabric/hw/inc/udm/tt_fabric_udm_impl.hpp` ≈line 107,
`tt_metal/fabric/hw/inc/udm/tt_fabric_udm.hpp` ≈line 482

Both already have a complete indexed arm. Verify two things rather than rewrite:

1. `calculate_initial_direction` reads `MEM_TENSIX_ROUTING_TABLE_BASE` while
   `select_relay_to_mux_connection` reads `ROUTING_TABLE_BASE`. Confirm both resolve to the same
   region on a Tensix worker; a mismatch reads zeros and returns `EAST` for everything.
2. `calculate_initial_direction`'s indexed arm can return `Z`. Confirm every caller can accept a Z
   initial direction — the legacy arm never could.

No edit if both hold; document the finding either way.

## 2.2 Make the mesh-API multicast source-inject unconditional (plan P6)

**File:** `tt_metal/fabric/hw/inc/mesh/api.h` ≈line 1198

`fabric_multicast_source_inject_noc_unicast_write` is wrapped in
`#if defined(FABRIC_EXPRESS_ENABLED)`. Drop the guard (keep the shape defines it uses, which Phase 3
makes unconditional). On a non-express mesh a multicast root has at most one eth output per axis, so
the loop over set bits degenerates to a single send — the function is correct there, just not
interesting.

Nothing forces callers to migrate. Existing single-connection multicast callers keep working; codec
§7.3.1 is explicit that express multicast through a single-connection adapter is a *scoping*
statement, not an enforced restriction.

## 2.3 ⚠ Audit `recompute_path` against the indexed landing encoder (plan B5) — **RESOLVED**

Phase 4.2 deletes `fabric_edge_node_router.hpp`. This is the audit that licenses that, done rather
than deferred to "the compiler says it's unused."

### The legacy machinery, enumerated

Two recompute triggers in `get_cmd_with_mesh_boundary_adjustment`:

```text
A   hop_cmd == NOOP, on any edge router (intermesh OR intramesh-on-edge)
B   is_intramesh_router_on_edge && dst_start_mesh_id != my_mesh_id
      && hop_cmd == FORWARD_<my_direction>          i.e. "the route says deliver here"
```

and four outcomes inside `recompute_path` → `fabric_set_unicast_route<called_from_router=true>`:

```text
1  next_direction valid                              full route to dst + prepend one hop
2  next_dir invalid && my_mesh == dst_start_mesh      FORWARD_<my_dir>, drain here
3  next_dir invalid && my_mesh != dst_start_mesh      FORWARD_<boundary>, route_buffer[1] = NOOP
M  mcast_params_64 != 0 && dst_start_mesh == my_mesh  fabric_set_mcast_route(header)
```

### Equivalence

| legacy | fires when | indexed equivalent | resolved |
|---|---|---|---|
| **B → 3** | at the exit chip | `action_is_intermesh_exit(action, dst_mesh, my_mesh)` → forward **as-is** on the `INTERMESH` egress | ✓ semantics differ (legacy *rebuilds*, indexed *forwards unchanged*) but the outcome is the same boundary hop. Safe only via the exit-chip invariant — see residual risk |
| **A → 1** | landing, dst elsewhere in this mesh | landing intercept → `widen_indexed_route_to_chip(dst_start_chip_id)` | ✓ |
| **A → 2** | landing, this chip *is* the dst | same widen, with dst == self: Y row reaches STOP, X slot gets `LOCAL_DELIVER` at own column → decode delivers locally | ✓ equivalent to "drain" |
| **A → 3** | landing on an *intermediate* mesh | landing intercept, `final_mesh != my_mesh` → widen toward this mesh's next exit | ✓ and strictly better: legacy re-enters case 3 hop by hop, indexed installs the whole next-exit route once |
| **A → M** | destination-mesh mcast landing | landing intercept, `mcast_params_64 != 0` → `encode_indexed_mcast_maps` rooted here, anchor from `dst_start_node_id` | ✓ |
| **A**, Z boundary | `set_forward` writes `NOOP` for a Z egress, so `recompute_path` *returns* `NOOP` as the new hop_cmd → admission sees `case NOOP: break` → stalls | no equivalent needed: Z is action bit 4, never a sentinel | ✓ the indexed path removes a latent stall rather than reproducing it |

### Trigger-coverage check

The one that could silently lose a case: legacy trigger **A** fires on *any* edge router, while the
indexed landing fires only on receiver channels flagged `IS_RECEIVER_CHANNEL_n_INTERMESH_INGRESS`.
Are those the same set?

Yes, by construction. The `NOOP` that triggers A is planted by case 3 *at the exit*, into
`route_buffer[1]`. The packet then crosses the boundary, so the router that trips on it is always the
far-side landing — which is by definition an intermesh ingress. There is no path that plants a `NOOP`
and then trips on it without crossing a boundary.

### Residual risks (both recorded, neither blocking)

**The exit-chip invariant becomes load-bearing.** Legacy *rebuilds* at the exit, so a sloppy incoming
route gets corrected there. Indexed forwards unchanged, so correctness now rests on the encoder having
produced exactly `LOCAL_DELIVER` and no eth bits at the exit chip (codec §4.5). That is a Phase 7 host
test (`exit-chip invariant`), and it is the one property whose safety net Phase 4.2 removes.

**VC2 is excluded from the landing intercept.** The builder emits
`IS_RECEIVER_CHANNEL_n_INTERMESH_INGRESS = (is_inter_mesh && i < 2)`, i.e. VC0 and VC1 only, with the
comment *"VC2 is excluded, staying outside the express derivation entirely."* Taken as a documented
decision by the builder's author rather than independently re-derived here; if VC2 ever carries a
carrier that lands, that flag needs widening.

**Conclusion: 4.2 is licensed.** Every legacy trigger and outcome has an indexed counterpart, and the
two divergences are improvements (no Z/NOOP overload, one-shot next-exit install) rather than gaps.

## 2.4 Add the missing device coverage

The indexed path has host unit tests (`test_mcast_reverse_tree.cpp`) but thin device coverage. Before
Phase 3 flips non-express meshes onto it, add:

| Test | Why |
|---|---|
| intermesh unicast, destination-mesh landing | exercises `fabric_set_indexed_intermesh_landing_route` unicast arm |
| intermesh unicast, **intermediate**-mesh landing | never exercised; needs a ≥3-mesh fixture |
| intermesh multicast carrier + destination landing rebuild | the 11-arg overload path from 2.3 |
| 2D multicast on a LINE-axis mesh | the whole point of step 1.5 |
| unicast on `[8,8]` and `[8,16]` | the whole point of step 1.3 |
| exit-chip invariant | host test: for every (src, dst-mesh) pair, the widened maps decode to exactly `LOCAL_DELIVER` at the exit chip and no eth bits (codec §4.5) |

## 2.5 Unify the 2D route API surface; make every consumer reach the indexed encoder

The indexed encoders are the accurate ones — they can express Z, they are extent-independent, and they
carry an explicit delivery flag. So no consumer should be able to reach a legacy encoder. Today most
cannot, but the exceptions are not obvious, because the *same function name* forks internally in some
places and is legacy-only in others.

### 2.5.1 The client-facing API layer — audit first, because this is what other teams call

⚠ An earlier version of this step audited only the *encoder* call sites. That is the wrong layer.
What other teams actually call is the API in `hw/inc/mesh/api.h`, `hw/inc/linear/api.h`, and
`hw/inc/udm/*` — the encoders are an implementation detail underneath.

Auditing that layer gives the single most important result in this whole project:

> **No public 2D fabric API exposes a legacy-shaped signature.** Every one of them is expressed in
> destination terms — `(dst_dev_id, dst_mesh_id)` for unicast, `(dst_dev_id, dst_mesh_id, ranges)` for
> multicast. None takes `direction + num_hops`. So the client surface is *already* in the indexed
> vocabulary, and the legacy encoder was always the odd one out beneath it.

**Consequence: no public API signature changes, and no consumer migrations.** That is why this step is
small, and it is a stronger reason than "the names happen to fork internally."

| layer | entry points | how it sets a route |
|---|---|---|
| `mesh/api.h` | 38: `fabric_{unicast,multicast}_noc_*` × `{plain, _set_state, _with_state}` | plain and `_set_state` call the forking `fabric_set_{unicast,mcast}_route`; `_with_state` **never touches the route** |
| `linear/api.h` | ~24 route-setting sites | has 2D paths under `#if defined(FABRIC_2D)` — see 2.5.2 |
| `udm/tt_fabric_udm_impl.hpp` | `calculate_initial_direction` → `fabric_set_unicast_route` | forking encoder; the express arm is step 3.4 |
| `mesh/api.h` `fabric_multicast_source_inject_noc_unicast_write` | 1 | the **only** public API that calls an indexed encoder directly; step 2.2 makes it unconditional |

Two properties worth confirming rather than assuming:

**`_set_state` / `_with_state` reuse is codec-agnostic.** `_set_state` sets the route once;
`_with_state` only updates payload and command fields and sends. The worker's template header is never
mutated in transit (the router mutates its own received copy), so reuse is valid under both codecs —
and *inherently* safer under indexed, where the maps are immutable by construction.

**`SetRoute=false` has no external users.** The template parameter exists on several `mesh/api.h`
entry points but every use is internal to that header, so no caller depends on pre-setting a route
itself.

### 2.5.2 ⚠ `linear/api.h` runs over 2D fabric too — and it confirms decision Q1

The "linear" API is not 1D-only. Under `FABRIC_2D` its route shims call the 2D encoders
(`linear/api.h:21-61`), and its multicast shim is structurally **one direction per call**, selected by
the connection's own tag:

```cpp
switch (static_cast<eth_chan_directions>(slot.tag)) {
    case EAST:  fabric_set_mcast_route(hdr, slot.dst_dev_id, slot.dst_mesh_id, hop, 0, 0, 0); break;
    case WEST:  fabric_set_mcast_route(hdr, slot.dst_dev_id, slot.dst_mesh_id, 0, hop, 0, 0); break;
    case NORTH: fabric_set_mcast_route(hdr, slot.dst_dev_id, slot.dst_mesh_id, 0, 0, hop, 0); break;
    case SOUTH: fabric_set_mcast_route(hdr, slot.dst_dev_id, slot.dst_mesh_id, 0, 0, 0, hop); break;
}
```

Exactly one nonzero extent, per slot. That is the per-direction contract from decision Q1, already
hard-coded — independent confirmation that Q1 matches how the API behaves rather than constraining it.

It also sharpens where Q1's guard belongs:

| layer | can express a multi-direction rectangle in one call? | Q1 assert needed? |
|---|---|---|
| `linear/api.h` over 2D | **no** — one extent per slot, by construction | no; structurally safe |
| `mesh/api.h` | **yes** — takes `ranges.e/w/n/s` together | **yes** — this is the layer the assert protects |

So put the single-output-root assert on the `mesh/api.h` path (via `fabric_set_mcast_route`), and note
that `linear/api.h` cannot trip it.

### 2.5.3 Classify the encoder surface

| entry point | external call sites | reaches indexed? |
|---|---|---|
| `fabric_set_unicast_route` (2D overload) | ~149 across ~40 files | ✓ worker path forks; **router path does not** — see 2.5.5 |
| `fabric_set_mcast_route` | ~49 | ✓ worker path forks |
| `fabric_set_single_hop_unicast_route{,_from_direction}` | 5 (all deepseek) | ✓ fixed in step 0.1 |
| `fabric_set_route` | 2 | ✗ **legacy only** — see 2.5.4 |

Because the first three fork behind a stable name, **their ~200 call sites need no change** — the fork
inside them is what Phase 3 removes. That is the whole reason this step is small.

### 2.5.4 Migrate the consumers of `fabric_set_route`

`fabric_set_route(header, direction, branch_forward, start_hop, num_hops, terminate)` is the raw
hop-program primitive. It has no indexed equivalent and never will — direction-plus-hop-count is the
legacy model. Two consumers:

| consumer | disposition |
|---|---|
| `impl/dispatch/kernels/cq_relay.hpp:90` | already gated `GALAXY_CLUSTER && !FABRIC_EXPRESS_ENABLED`; step 3.4 drops the legacy arm and keeps `fabric_set_unicast_route`. **Nothing to do here.** |
| `tests/.../kernels/test_fabric_set_unicast_route.cpp:57,74,78` | ⚠ uses `fabric_set_route` to build a **reference** encoding that the test diffs the real API against. Deleting `fabric_set_route` in 4.1 breaks the test. Either re-base its reference onto the indexed expectation, or retire the test in favour of the step-2.4 coverage. **Decide here, not at 4.1.** |

### 2.5.5 The router-side arm of `fabric_set_unicast_route` (plan B5)

`fabric_set_unicast_route<called_from_router=true>` still runs the **legacy** encode even on an express
build — the express fork is guarded `if constexpr (!called_from_router)`. It is unreachable today
because its only caller, `recompute_path`, is only invoked from the non-express kernel arm.

Confirm it is genuinely unreachable rather than merely untested, then let 4.1 delete it. If any caller
turns up, it must be routed to `fabric_set_indexed_intermesh_landing_route` instead — that is the
indexed equivalent of what router-side re-encode does.

### 2.5.6 Delete the one unreachable indexed variant

`fabric_set_indexed_single_hop_unicast_route` (the non-`_from_direction` wrapper) has **zero callers**
and is now redundant: after step 0.1 the legacy-named wrapper delegates to
`fabric_set_single_hop_unicast_route_from_direction`, which forks. Either delete it, or make it the
only wrapper and have the legacy name alias it. Do not leave two wrappers where one is dead.

Reachability of the rest, for the record — all live, none orphaned:

```text
fabric_set_indexed_unicast_route                  1 caller
fabric_set_indexed_mcast_route                    3
fabric_set_indexed_single_hop_..._from_direction  2
fabric_set_indexed_intermesh_landing_route        1
widen_indexed_route_to_chip                       4
fabric_set_indexed_single_hop_unicast_route       0   <- delete
```

> **Gate 2.5.** Every client-facing API layer (`mesh/api.h`, `linear/api.h`, `udm/*`) is confirmed to
> reach only the forking encoders, with no public signature change required. No consumer can reach a
> legacy 2D encoder except through a path Phase 3 is about to flip. `fabric_set_route`'s two consumers
> each have a written disposition. No orphaned indexed variants remain.

> **Gate 2.** Every capability the legacy 2D path provides has a green indexed-path test, run on both
> an express fixture and a non-express fixture. The 2.3 equivalence table is written down with each
> row resolved.

---

# Phase 3 — Flip (flag day)

⚠ **This phase changes behaviour on every 2D platform.** Keep the diff to gate flips only. Delete
nothing. A revert must be a single commit.

## 3.1 Retire the selector — delete it, do not rename it

`FABRIC_EXPRESS_ENABLED` becomes **fully redundant with `FABRIC_2D`** at the flip, so there is no new
name to introduce. Evidence, same file 43 lines apart:

```cpp
// compute_mesh_router_builder.cpp:869
defines["FABRIC_2D"] = "";
    ...
// :912 — inside the SAME if block
if (control_plane.express_routing_enabled(local_node_.mesh_id)) {
    defines["FABRIC_EXPRESS_ENABLED"] = "";
}
```

Worker side is the same shape: `fabric.cpp:389` sets `FABRIC_2D`, and `get_express_kernel_defines`
is called inside that same `is_2D_routing_enabled()` block. Once step 3.2 removes the express
condition, the two defines are emitted under identical circumstances.

So every `#if defined(FABRIC_EXPRESS_ENABLED)` becomes `#if defined(FABRIC_2D)` and the define stops
being emitted at all:

```text
fabric_context.cpp:380                        KEEP emitting, same condition — the Z-port site needs it (3.5).
                                              Shape defines beside it become unconditional for 2D.
compute_mesh_router_builder.cpp:906-913       delete the block entirely (FABRIC_2D already set at :869)
fabric_erisc_router_ct_args.hpp:432-447       #if defined(FABRIC_2D) — the #else arm is still needed by 1D
tt_fabric_api.h:151,158,205,288               → #if defined(FABRIC_2D)
mesh/api.h:1198                               (guard already removed in 2.2)
udm/tt_fabric_udm.hpp:482                     → #if defined(FABRIC_2D)
udm/tt_fabric_udm_impl.hpp:107                → #if defined(FABRIC_2D)
routing_plane_connection_manager.hpp:21       ⚠ GENUINE express site — keeps FABRIC_EXPRESS_ENABLED, no change
impl/dispatch/kernels/cq_relay.hpp:89         (guard removed in 3.4; no define needed)
```

Renames that *are* needed, because these carry real information and will be emitted for all 2D:

| now | after |
|---|---|
| `FABRIC_EXPRESS_MESH_Y_SIZE` / `_X_SIZE` (host define) | `FABRIC_2D_MESH_Y_SIZE` / `_X_SIZE` |
| `EXPRESS_MESH_Y_SIZE` / `_X_SIZE` (kernel constexpr) | `MESH_Y_SIZE` / `MESH_X_SIZE`, matching the CT arg names |
| `express_enabled` constexpr | **deleted in 5.3** — not renamed; the arms collapse |
| CT args `MESH_Y_SIZE` / `MESH_X_SIZE` | unchanged |
| `FABRIC_EXPRESS_ENABLED` at `routing_plane_connection_manager.hpp:21` | unchanged — genuinely about express topology; still emitted from `get_express_kernel_defines` on the same condition |

⚠ **This makes step 3.3 a hard precondition, not a tidy-up.** If `FABRIC_2D` guards the read of
`FABRIC_2D_MESH_*_SIZE`, then *every* `FABRIC_2D` compile must carry the shape defines — including
the dispatch kernels, which set `FABRIC_2D` at `dispatch.cpp:585` and `prefetch.cpp:549` and whose
`cq_relay.hpp` calls `fabric_set_unicast_route` after 3.4. Either resolve 3.3 so the shape reaches
them, or keep the zero-fallback at `tt_fabric_api.h:151-156` for compiles that legitimately have no
single mesh shape. Decide which **before** 3.2.

## 3.2 Flip the four host gates

| File | Edit |
|---|---|
| `fabric_context.cpp:371-384` (`get_express_kernel_defines`) | drop `|| !control_plane.express_routing_enabled(mesh_id)` from the early return; keep `!is_2D_routing_enabled_`. Rename the method to `get_2d_abi_kernel_defines`. |
| `fabric.cpp:388-398` | unchanged in shape — it already calls the method per mesh; only the method's gate moves |
| `control_plane.cpp:1974-2008` | delete the `if (this->express_routing_enabled(mesh_id))` branch; always take the indexed arm. Replace `ring_for_direction(N)/(E)` with `axis_topology(mesh_id, 0)/(1)` (step 1.5), which is never null, so the `if (y_rings != nullptr && x_rings != nullptr)` guard becomes unnecessary — remove it so a missing topology is a build error rather than a silent skip. Keep the `TT_FATAL` from step 1.7. |
| `compute_mesh_router_builder.cpp:906-913` | emit the define unconditionally inside the existing `is_2D` scope; drop the `control_plane` lookup |
| `erisc_datamover_builder.cpp:1258-1273` | drop `if (express_routing_enabled)`; always emit `MESH_Y_SIZE`, `MESH_X_SIZE`, and the three `IS_RECEIVER_CHANNEL_n_INTERMESH_INGRESS` args |

⚠ `control_plane.cpp:1966-1971` asserts `mesh_shape[0] <= 32 && mesh_shape[1] <= 32`. After step 1.3
the indexed encoder admits up to 64 per axis, but this assert is the one codec §4.7 flags as needing
removal for `[64,4]`. Leave it in place this phase — it is not blocking any in-tree shape — and note
it as Phase 6 work.

## 3.3 Delete the no-node overload's express fatal (plan H6)

**File:** `tt_metal/fabric/fabric.cpp` ≈lines 703-716

The no-node `get_fabric_kernel_defines(api_type)` currently refuses to run on an express fabric:

```cpp
// The 2D ABI is selected per mesh, so it cannot be resolved without knowing which mesh the
// kernel runs on. Refuse rather than return a set that silently encodes hop programs for a
// mesh whose L1 holds indexed vectors.
TT_FATAL(!any_mesh_uses_express, "...call the overload taking the kernel's FabricNodeId");
```

**Delete the fatal.** Its premise -- that the 2D ABI varies per mesh -- is exactly what the flip
removes. `api_type` selects the Linear (1D) versus Mesh (2D) *API surface*; express is a flavour of
mesh routing, not a third api_type, so nothing at this layer has an ABI choice to make. The overload
keeps returning `API_TYPE_*` + `FABRIC_2D`, which is correct and complete for what it does.

No signature change and no caller migration. The one caller of this overload is the Python binding at
`ttnn-nanobind/fabric.cpp:241`, which is a pure query.

Shape defines are not this overload's job and never were. Kernels that encode 2D routes get them from
the node-aware overload (`fabric.cpp:720`) or, for the dispatch relays, from their own resolution --
`dispatch.cpp:585` and `prefetch.cpp:549` already look up `get_fabric_node_id_from_physical_chip_id`
and pull `get_express_kernel_defines` for that mesh, with a comment saying why. A compile that has no
single mesh shape is covered by the zero-fallback in `tt_fabric_api.h`, which retargets from
`FABRIC_EXPRESS_ENABLED` to `FABRIC_2D` with everything else; the runtime
`ASSERT(y_size > 0 && x_size > 0)` then catches genuine misuse. That is the existing ERISC mechanism,
unchanged.

## 3.4 Flip the device producers

| File / symbol | Edit |
|---|---|
| `tt_fabric_api.h` `fabric_set_unicast_route` (2D, ≈288) | drop the `#if`; keep `if constexpr (!called_from_router)` **for now**. The router arm is dead under the indexed kernel (no `get_cmd_with_mesh_boundary_adjustment` call site) — Phase 4 deletes it. |
| `tt_fabric_api.h` `fabric_set_mcast_route` (≈205) | drop the `#if`. The legacy spine/branch body below becomes unreachable for `!called_from_router`; leave it for Phase 4. |
| `tt_fabric_api.h` single-hop (≈48) | drop the `#if` added in Phase 0 — always delegate. |
| `tt_fabric_api.h:151-156` | delete the `#if defined(...) && !defined(FABRIC_2D_MESH_Y_SIZE)` zero-default block if the defines are now always emitted to worker kernels; keep it if ERISC compiles still lack them (check `compute_mesh_router_builder` vs `fabric.cpp` define scopes — they are different call paths). |
| `tt_fabric_api.h:158-165` | make the `Y+X ≤ sizeof(route_buffer)` static_assert unconditional |
| `udm/tt_fabric_udm.hpp:482`, `udm/tt_fabric_udm_impl.hpp:107` | → `#if defined(FABRIC_2D)`, keeping only the indexed arm. ⚠ **Load-bearing, not cosmetic:** after the flip a non-express 2D mesh's L1 slot holds indexed vectors, so leaving these on `FABRIC_EXPRESS_ENABLED` sends non-express UDM down the *legacy* arm, which reads the slot as `intra_mesh_routing_path_t<2,true>` and gets garbage. |

⚠ **Do not "simplify away" the UDM express `static_assert`** in `fabric_edm_packet_header.hpp`'s
UDM 2D arm. It bars UDM + express because the mux fabric is cardinal-only
(`NUM_DOWNSTREAM_MUX_CONNECTIONS == 3`, no Z mux), and it composes with this step precisely because
Q2 kept `FABRIC_EXPRESS_ENABLED` express-scoped rather than promoting it to a universal ABI flag. Had
it been renamed to mean "indexed 2D", that assert would fire for *every* UDM build after the flip.
The three-way result after Phase 3 is:

```text
UDM + 2D non-express   indexed arm, cardinal-only first hops   -> supported
UDM + 2D express       refused at compile time                  -> barred
non-UDM + 2D           indexed arm                              -> supported
```
| `cq_relay.hpp:86-101` | delete the `#if GALAXY_CLUSTER && !FABRIC_EXPRESS_ENABLED` and keep only the `fabric_set_unicast_route` arm |

## 3.5 ⚠ Leave the connection-manager capacity entirely alone

**File:** `tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp` — **no edit**

```cpp
#if defined(FABRIC_EXPRESS_ENABLED) && defined(ARCH_BLACKHOLE)
#define TT_FABRIC_MAX_ROUTING_PLANE_CONNECTIONS 6
```

This is **G2**, not G1 — it sizes for Z peers, and it is the one device-side site where
`FABRIC_EXPRESS_ENABLED` genuinely means what its name says. If it follows the codec sites to
`FABRIC_2D` it becomes unconditional on Blackhole and grows every connection manager by two slots.

So: do not touch this file, and keep emitting `FABRIC_EXPRESS_ENABLED` from
`get_express_kernel_defines` on its existing `express_routing_enabled(mesh_id)` condition. The
existing comment there already explains both conditions correctly — preserve it.

What *does* change in step 3.2 is that `get_express_kernel_defines` splits into two emissions:

```cpp
// always, for 2D — the shape the indexed codec widens against
defines["FABRIC_2D_MESH_Y_SIZE"] = ...;
defines["FABRIC_2D_MESH_X_SIZE"] = ...;

// express meshes only — Z-port capacity, unchanged
if (control_plane.express_routing_enabled(mesh_id)) {
    defines["FABRIC_EXPRESS_ENABLED"] = "1";
}
```

The **router-side** emission (`compute_mesh_router_builder.cpp:906-913`) is deleted instead: the
router does not include `routing_plane_connection_manager.hpp`, so its only consumers were the codec
sites in `fabric_erisc_router_ct_args.hpp`, which move to `#if defined(FABRIC_2D)`.

## 3.6 Flip the kernel

**File:** `tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp`

`if constexpr (express_enabled)` becomes always-true. Do **not** delete the `else` arms this phase —
`if constexpr (true)` already compiles the legacy arm out, so leaving it costs nothing and keeps the
revert clean.

Sites: 1955 (`!UPDATE_PKT_HDR_ON_RX_CH && !express_enabled`), 2149, 2252, 2550.

In `fabric_erisc_router_ct_args.hpp:432-447`, the `#else` arm (`express_enabled = false`,
`MESH_*_SIZE = 0`) becomes dead for 2D but is still reached by **1D** builds, which never define the
selector. Keep the `#if` structure; only Phase 5 removes it, and only after confirming 1D never
instantiates the 2D decode.

> **Gate 3.** Full regression on every 2D platform:
>
> | Platform | Config |
> |---|---|
> | WH Galaxy (TG) | `[8,4]`, `[32,4]` |
> | BH Galaxy | `[8,4]`, `[32,4]`, `[4,4]` express |
> | BH LB | `[2,4]` |
> | dual/quad galaxy | `[8,8]`, `[8,16]`, `[16,8]` |
> | N300 2x2 / p150_x4 | `[2,2]` |
>
> Plus: 1D regression must be **untouched** — if any 1D test moves, a define leaked.
>
> **Measure and record**: Galaxy `[32,4]` packet header size and end-to-end bandwidth. Expect 112 B
> and a bandwidth drop (plan R2). That number is the justification for Phase 4 and the acceptance
> bar to beat.

---

# Phase 4 — Delete the legacy 2D codec, reclaim the header

The payoff phase. Deletions first, then the header change they enable.

## 4.1 Delete the legacy 2D producers

**File:** `tt_metal/fabric/hw/inc/tt_fabric_api.h`

| Delete | Notes |
|---|---|
| `single_hop_route_cmd_by_direction` table (≈line 25) | opposite-direction encoding, obsolete |
| legacy body of `fabric_set_single_hop_unicast_route_from_direction` | keep the name as a thin forwarder to the indexed helper, or rename the indexed one and delete this. Prefer keeping the public name stable — the deepseek kernels call it. |
| `fabric_set_route<mcast>` (≈line 70) | after 3.4 removed the `cq_relay` and mcast callers |
| legacy spine/branch body of `fabric_set_mcast_route` (≈lines 228-253) | everything after the indexed `return` |
| the `called_from_router` arm of `fabric_set_unicast_route` (≈lines 296-360) | dead after 4.2 deletes its only caller |
| `#if defined(COMPILE_FOR_ERISC)` `fabric_set_mcast_route(packet_header)` overload (≈line 255) | only caller is `recompute_path` |

Rename `fabric_set_indexed_*` → drop the `indexed_` infix once the legacy twins are gone, so the
public surface is `fabric_set_unicast_route` / `fabric_set_mcast_route` / `fabric_set_single_hop_*`
with one implementation each.

## 4.2 Delete `fabric_edge_node_router.hpp`

Whole file. Its two functions (`recompute_path`, `get_cmd_with_mesh_boundary_adjustment`) have no
remaining callers after Phase 3 flipped the kernel. Remove the include from
`fabric_erisc_router.cpp` and anywhere else it appears.

⚠ Precondition: step 2.3's equivalence table is fully resolved. Do not delete on the strength of
"the compiler says it's unused."

## 4.3 Delete the legacy L1 2D table

**Files:** `fabric_common.h`, `compressed_routing_path.cpp`, `control_plane.cpp`

| Delete | Site |
|---|---|
| `intra_mesh_routing_path_t<2, true>` from the `routing_l1_info_t` union | `fabric_common.h:1151` |
| `compressed_route_2d_t` | `fabric_common.h` ≈line 110-142 |
| the `<2, true>` specialization of `intra_mesh_routing_path_t` and its `decode_route_to_buffer` | `fabric_common.h`, `hw/inc/fabric_routing_path_interface.h` |
| `routing_encoding::encode_2d_unicast` | wherever it lives in `fabric_common.h` |
| `calculate_chip_to_all_routing_fields` for the 2D compressed path | `compressed_routing_path.cpp:~60-161` |
| `RoutingFieldsConstants::Mesh` | `fabric_common.h:189-212`. ⚠ Check 1D does not reference it before deleting. |

Update the `offsetof` / `sizeof` static asserts at `fabric_common.h:1162-1187`. The union slot stays
1028 B (`indexed_route_vectors_t`), so `sizeof(routing_l1_info_t)` should stay 2576 — verify, don't
assume.

## 4.4 ⚠ Retire the legacy packet-header fields and reclaim the tier

**Files:** `tt_metal/fabric/fabric_edm_packet_header.hpp`, `fabric_context.{hpp,cpp}`

This is what makes Galaxy fit a 96 B header again (plan D4). Do it as one change.

Remove from `HybridMeshPacketHeaderT<RouteBufferSize>` (≈lines 1247-1261):

```cpp
LowLatencyMeshRoutingFields routing_fields;   // 4 B — hop_index | branch_east | branch_west
uint8_t is_mcast_active;                      // 1 B — written, never read for routing
```

⚠ Every producer currently writes `packet_header->routing_fields.value = 0` and
`packet_header->is_mcast_active = 0`. Grep and remove all of them, including inside the indexed
producers (`tt_fabric_api.h` lines ≈432, ≈467, ≈518, ≈619) — they are vestigial there. Also remove
`ROUTING_FIELDS_TYPE cached_routing_fields` threading through the 2D kernel forward path
(`fabric_erisc_router.cpp` — `forward_payload_to_downstream_edm` takes it; check whether 1D needs
the parameter and template it out if so).

Also check the profiler / telemetry / debug consumers. Grep for `hop_index`, `branch_east_offset`,
`branch_west_offset`, `turn_point` across `tt_metal/` and `tools/`.

Then the base drops 61 → 56 B. Update `fabric_context.hpp:155-161`:

```cpp
static constexpr Routing2DBufferTier ROUTING_2D_BUFFER_TIERS[] = {
    // {24, 24},  // 80B header (56+24=80) - see plan Q5, still disabled pending re-measurement
    {40, 40},  // 96B  header - max capacity (56+40=96)
    {56, 56},  // 112B header - max capacity (56+56=112)
    {72, 72},  // 128B header - max capacity (56+72=128)
};
static constexpr uint32_t MAX_2D_ROUTE_BUFFER_SIZE = 72;
```

And the four asserts at `fabric_edm_packet_header.hpp:1295-1298`:

```cpp
static_assert(sizeof(HybridMeshPacketHeaderT<24>) == 80);
static_assert(sizeof(HybridMeshPacketHeaderT<40>) == 96);
static_assert(sizeof(HybridMeshPacketHeaderT<56>) == 112);
static_assert(sizeof(HybridMeshPacketHeaderT<72>) == 128);
```

Update the default `MESH_ROUTE_BUFFER_SIZE` at `fabric_common.h:166` from 35 to 40.

Verify `get_udm_header_size` (`fabric_context.cpp:230`) still derives from `get_2d_header_size` and
that `get_2d_header_size`'s switch (line 221) is updated to the new tier values.

Sanity table for the new 96 B tier:

| Shape | Y+X | Tier |
|---|---|---|
| `[8,8]` | 16 | 40 → 96 B |
| `[16,4]`, `[1,16]`, `[8,16]`, `[16,8]` | 20-24 | 40 → 96 B |
| `[32,4]` Galaxy | 36 | 40 → **96 B** ✓ |
| `[64,4]` future | 68 | 72 → 128 B |

> **Gate 4.** Galaxy `[32,4]` back to a 96 B header. Bandwidth at or above the Phase-0 baseline —
> not merely above the Phase-3 number. Zero occurrences of `hop_index`, `branch_east_offset`,
> `branch_west_offset`, `turn_point`, `is_mcast_active`, `compressed_route_2d_t` in any 2D path.

---

# Phase 5 — Kernel simplification

## 5.1 Delete the turn/header-mutation set (kernel §3.9)

**Files:** `fabric_erisc_router.cpp`, `fabric_erisc_router_ct_args.hpp`,
`fabric_edm_packet_transmission.hpp`

Delete together, per kernel §3.9's explicit list:

```text
is_spine_direction
TURN_STATUS_ARRAY_SIZE
get_sender_channel_turn_statuses
sender_channels_turn_status
IS_TURN
update_packet_header_before_eth_send        (2D body; keep 1D overloads)
UPDATE_PKT_HDR_ON_RX_CH  + its RX branches  (CT arg and host emission)
the HybridMesh routing_fields.value + 1 path (fabric_edm_packet_transmission.hpp)
```

⚠ `update_packet_header_before_eth_send` and `UPDATE_PKT_HDR_ON_RX_CH` are shared with 1D. Delete
only the `#if defined(FABRIC_2D)` bodies and the 2D call sites; 1D's decrement/shift/refill is
untouched (codec §4.5.1, kernel §3.6).

## 5.2 (deferred) `port_direction_table`

Not part of this project. kernel §3.9 notes it is independent of the ABI cutover — threaded through
normal and speedy receiver calls but never initialized, indexed, or read. It belongs to the deferred
dead-code discussion (plan §1). Leave it in place.

## 5.3 Collapse the `if constexpr (express_enabled)` arms

Now that they are always true for 2D, delete the `else` arms at `fabric_erisc_router.cpp` 2149,
2252, and the `!express_enabled` term at 1955. Then:

- if 1D never reaches these blocks, delete the `express_enabled` constexpr and collapse the
  `#if defined(FABRIC_2D)` block in `fabric_erisc_router_ct_args.hpp:432-447`, promoting
  `MESH_Y_SIZE` / `MESH_X_SIZE` / `receiver_channel_is_intermesh_ingress` to unconditional named CT
  args (⚠ verify the host emits them on 1D builds too, or guard on `is_2d_fabric`);
- otherwise keep a `#if defined(FABRIC_2D)` guard and drop only the express spelling.

Also delete the now-dead `can_forward_packet_completely(hop_cmd, ...)` 2D overload and the legacy
multicast switch it fed.

## 5.4 Rename express → indexed in the kernel

Mechanical, but do it — the names are load-bearing documentation and they currently lie:

```text
admit_express_combo        → admit_indexed_combo
forward_express_combo      → forward_indexed_combo
admit_express_dispatch     → admit_indexed_dispatch
forward_express_dispatch   → forward_indexed_dispatch
express_arm_is_realizable  → dispatch_arm_is_realizable
express_local_y / _x       → local_y / local_x
express_fwd_key            → fwd_key
express_local_deliver      → local_deliver
express_egress[_index]     → intermesh_egress[_index]
EXPRESS_MESH_Y_SIZE / _X   → MESH_Y_SIZE / MESH_X_SIZE
```

Keep `express` only where it genuinely means Z chords: `express_routing_enabled`,
`derive_express_ring_topology`, `express_links`, and `FABRIC_EXPRESS_ENABLED` at the Z-port site.

Same treatment for `mcast_reverse_tree.hpp`'s doc comments, which describe the trees as an express
artifact.

> **Gate 5.** ERISC binary size at or below Phase 3. No `express` identifier in the kernel that does
> not mean Z chords. 1D untouched.

---

# Phase 6 — L1 cleanup

## 6.1 Remove `intra_mesh_direction_table` (codec §2.11)

**Files:** `fabric_common.h:1144`, `control_plane.cpp:2053-2069`, `tt_fabric_api.h:35-44`

Its only device reader is `get_next_hop_router_direction` for same-mesh destinations
(`tt_fabric_api.h:38`). codec §2.11 says the replacement is a first-hop peek at the indexed vectors:

```cpp
// Same-mesh first hop is the DOR peek: the Y vector's action at my row while rows differ, else the
// X vector's action at my column. The direction table was a redundant cache of exactly this.
inline eth_chan_directions get_next_hop_router_direction(uint32_t dst_mesh_id, uint32_t dst_dev_id) {
    auto* rt = reinterpret_cast<tt_l1_ptr routing_l1_info_t*>(ROUTING_TABLE_BASE);
    if (dst_mesh_id != rt->my_mesh_id) {
        return static_cast<eth_chan_directions>(
            rt->inter_mesh_direction_table.get_original_direction(dst_mesh_id));
    }
    // ... peek y_vectors[dst_y][my_y], else x_vectors[dst_x][my_x]
}
```

`calculate_initial_direction` in `udm/tt_fabric_udm_impl.hpp:106` already implements exactly this
peek — factor it out and share it rather than writing it twice.

⚠ Keep `inter_mesh_direction_table` and `exit_node_table` (codec §2.11: both stay).

## 6.2 Reclaim the freed 96 B

Removing `intra_mesh_direction_table` moves the union slot's `offsetof` from 516 to 420. Update the
static asserts at `fabric_common.h:1162-1187`. Per codec §6.2 this is what buys headroom for a
`[64,4]` hybrid: `420 + 1160 + 1024 = 2604` → **2608 B** with alignment, vs 2704 B if the table had
stayed.

Decide whether to shrink `routing_l1_info_t` to 2480 B or keep 2576 B and bank the 96 B as
`[64,4]` headroom. Recommendation: keep 2576 and document the reservation, so a future `[64,4]`
enablement is a layout change with no memory-map move.

## 6.3 Lift the 32-per-axis validation

**File:** `control_plane.cpp:1966-1971`

```cpp
TT_ASSERT(mesh_shape[0] <= 32 && mesh_shape[1] <= 32, ...);
```

codec §4.7 flags this as one of two things blocking `[64,4]` (the other is the header, resolved in
4.4 — the 72-byte tier covers `Y+X = 68`). Raise to 64 per axis, and add a bound check that
`vectors_region_bytes(Y,X) + mcast_tree_region_bytes(Y,X)` fits the slot, which is the real
constraint after step 1.3.

Note this does not *enable* `[64,4]` — no descriptor declares it — it removes the blocker.

> **Gate 6.** `sizeof(routing_l1_info_t)` asserted and documented. `[64,4]` either fits or its exact
> shortfall is recorded. Full regression green.

---

# Appendix A — Fork site quick reference

Every site the project touches, for grep-driven progress tracking.

## A.1 `FABRIC_EXPRESS_ENABLED` / `express_enabled` (the codec fork — all removed)

```text
tt_metal/fabric/fabric_context.cpp:380                              3.2
tt_metal/fabric/compute_mesh_router_builder.cpp:906-913             3.2
tt_metal/fabric/erisc_datamover_builder.cpp:1258-1273               3.2
tt_metal/fabric/control_plane.cpp:1974-2008                         3.2
tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_ct_args.hpp:432-447   3.1, 5.3
tt_metal/fabric/hw/inc/tt_fabric_api.h:151,158,205,288              3.4
tt_metal/fabric/hw/inc/mesh/api.h:1198                              2.2
tt_metal/fabric/hw/inc/udm/tt_fabric_udm.hpp:482                    3.4
tt_metal/fabric/hw/inc/udm/tt_fabric_udm_impl.hpp:107               3.4
tt_metal/impl/dispatch/kernels/cq_relay.hpp:89                      3.4
tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp:1955,2149,2252,2550   3.6, 5.3
tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp:21   3.5  ⚠ NO CHANGE — genuine express site
```

## A.2 `express_routing_enabled(mesh_id)` (retained, narrowed)

Stays, meaning "this mesh's route generation uses chords":

```text
compute_mesh_router_builder.cpp:195,271,294,333,368     G2/G3 wiring + injection policy
builder/router_wiring_rules.cpp  (all)                  G2
builder/fabric_edge_capability.cpp:88-108               G2
builder/injection_policy.cpp:92                         G3
fabric_context.cpp:345  need_deadlock_avoidance_support G3
control_plane.cpp:1684  protected-ring query            G3
```

Also untouched (G4 resource shape, G5 optimization disablement) — these were going to be renamed by
the dropped four-predicate refactor and are now left exactly as they are, guarded by the step-1.2
test instead:

```text
fabric_builder_context.cpp:45,53,98                     G4 vc1_uses_counters, stream assignment
builder/fabric_builder_config.cpp:48,56                 G4 VC0/VC1 downstream fan
erisc_datamover_builder.cpp:793,1074,1082,1090,1129     G5 speedy / trim / forwarding capture
```

## A.3 Symbols deleted

```text
Producers      fabric_set_route, encode_2d_unicast, single_hop_route_cmd_by_direction,
               legacy fabric_set_mcast_route body, called_from_router unicast arm
L1 / ABI       compressed_route_2d_t, intra_mesh_routing_path_t<2,true>,
               RoutingFieldsConstants::Mesh, intra_mesh_direction_table
Packet         routing_fields (2D), is_mcast_active, hop_index, branch_east_offset,
               branch_west_offset, turn_point
Edge           fabric_edge_node_router.hpp (whole file): recompute_path,
               get_cmd_with_mesh_boundary_adjustment
Kernel         is_spine_direction, TURN_STATUS_ARRAY_SIZE, get_sender_channel_turn_statuses,
               sender_channels_turn_status, IS_TURN,
               update_packet_header_before_eth_send (2D), UPDATE_PKT_HDR_ON_RX_CH,
               can_forward_packet_completely (2D hop_cmd overload)

DEFERRED, not deleted by this project (plan §1): port_direction_table, PacketHeader,
RoutingFields, packet_header_t, the non-low-latency ROUTING_MODE arms,
intra_mesh_routing_path_t<1,true>, DYNAMIC_ROUTING_ENABLED
```

## A.4 Symbols added

```text
control_plane      axis_topology
axis topology      derive_line_axis_topology, derive_axis_topology
fabric_common.h    MAX_INDEXED_MESH_AXIS
CT / defines       FABRIC_2D_MESH_Y_SIZE, FABRIC_2D_MESH_X_SIZE,
tests              test_abi_gate_separation.cpp, shape sweep, axis-topology coverage,
                   arborescence sweep, exit-chip invariant
```

---

# Appendix B — Invariants to hold throughout

Restated from the contracts, because these are what a plausible-looking edit breaks.

1. **One ABI, no mixing.** A legacy producer with an indexed kernel (or the reverse) is invalid even
   when the packet does not branch (codec §4.5.1). There is no compatibility mode.
2. **Decode is `action_y != 0`, never `action_y & (N|S|Z)`.** A terminal multicast row is
   `LOCAL_DELIVER|E|W` with no Y child; testing only N/S/Z wrongly falls through to X
   (codec §4.3, §7.5).
3. **E/W-facing routers never index the Y map** (codec §4.3, §5.12).
4. **Multicast is source-reachable OR, never full-vector OR.** ORing `y_vectors[dst]` in full selects
   outputs from rows not on this tree (codec §5.6).
5. **Maps are immutable on transit.** No hop cursor, no branch offset, no per-branch rewrite. Fanout
   is a NOC copy of the same header (codec §4.3, kernel §3.6).
6. **The self-facing action bit is always invalid.** No producer routes a packet back over the link
   it arrived on; that arc is what the deadlock-freedom proof assumes absent (kernel §3.7).
7. **The exit predicate needs both halves**: decoded action is exactly `LOCAL_DELIVER` *and* the
   final mesh is elsewhere. Mesh-id inequality alone also matches a chip merely transiting toward a
   different exit (kernel §4.2, codec §4.5).
8. **Landing intercept runs before decode**, keyed on `INTERMESH` *ingress* capability; the exit
   check runs after decode. Opposite sides of the decoder (kernel §3.2).
9. **Remote BFC lives only in the sender step**, from that sender's own CT `IS_INJECTION` flag. Never
   in RX admission, never inferred from a direction letter (kernel §5).
10. **RX admission is atomic** across every selected eth output plus local relay; no copy is
    committed until all can accept (kernel §3.4).
11. **Express-Z channels are `INTRAMESH_EXPRESS`, never `INTERMESH`.** If a same-mesh Z channel were
    registered as an exit direction, a packet arriving over a chord at an exit chip would deliver
    locally instead of crossing the boundary (codec §6.4, kernel §4.2).
12. **Reverse-tree edges are serialized descendants-before-ancestors.** Wrong order silently drops
    branches on device rather than failing (codec §5.7.1).
13. **1D is untouched.** Its routing fields, decrement/shift/refill, and
    `MulticastRoutingCommandHeader` semantics do not change in any phase.
