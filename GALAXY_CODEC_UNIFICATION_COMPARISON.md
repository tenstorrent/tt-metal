# 2D Fabric: Current Implementation vs Express Implementation

Reference description of the two 2D route codecs that coexist in the tree today, and the divide that
unification closes. Measured against `42204c541c3`.

- **current** — the hop-program codec every non-express 2D mesh uses. Selected when
  `ControlPlane::express_routing_enabled(mesh_id)` is false.
- **express** — the destination-indexed action-map codec, selected when it is true, i.e. when the
  mesh declares express links. Gated device-side by `FABRIC_EXPRESS_ENABLED`.

Both carry the same 2D traffic. They are not a legacy/replacement pair in the tree's structure — they
are two live alternatives chosen per mesh, which is the thing to eliminate.

Companions: `GALAXY_CODEC_UNIFICATION_PLAN.md` (how), `GALAXY_CODEC_UNIFICATION_IMPLEMENTATION_GUIDE.md`
(what to edit). Semantics owned by `GALAXY_DEVICE_ROUTE_CODEC_CONTRACT.md` and
`GALAXY_DEVICE_ROUTER_KERNEL_CONTRACT.md`.

---

## 1. Summary

| | current | express |
|---|---|---|
| routing field | `LowLatencyMeshRoutingFields` (4 B: `hop_index`, `branch_east_offset`, `branch_west_offset`) | none — the maps are the whole state |
| route payload | `route_buffer[N]`, one **hop command** per hop | `route_buffer[Y+X]`, one **action byte** per coordinate |
| indexed by | `hop_index`, a cursor | `local_y` / `local_x`, pinned coordinates |
| transit | mutates the header every hop | immutable |
| size | extent-dependent; can overflow (§5.4) | always exactly `Y + X` |
| L1 artifact | `compressed_route_2d_t[256]` (1024 B) | dest-major 2-bit vectors + per-chip reverse trees |
| Z / express chords | not expressible | action bit 4 |
| local delivery | implicit: the receiver's own-facing bit | explicit: `ACTION_LOCAL_DELIVER` |
| multicast targets | (N **xor** S) × (anchor ∪ E ∪ W) | (N ∪ S) × (anchor ∪ E ∪ W), Z in the tree |
| multicast fanout | branch-offset cursor jump | same header NOC-copied to each output |
| intermesh trigger | `NOOP` hop command, overloaded | `INTERMESH` edge capability + mesh-id compare |
| kernel admit | 16-arm switch on a 4-bit command | dense 16-way key over packed outputs |
| Galaxy `[32,4]` header | 96 B (35 route bytes) | 112 B (36 route bytes) — see §3.2 |

---

## 2. Wire format

### 2.1 Current — hop command

`route_buffer[i]` is the action for the *i*-th chip after the source. Four bits, from
`RoutingFieldsConstants::Mesh`:

```text
NOOP           0b0000      also means "recompute" on an edge router  ← overloaded
FORWARD_EAST   0b0001      matches eth_chan_directions::EAST = 0
FORWARD_WEST   0b0010
FORWARD_NORTH  0b0100
FORWARD_SOUTH  0b1000
```

Multicast ORs them (`WRITE_AND_FORWARD_NSEW` etc. — 11 named combinations).

**Local delivery is positional, not a flag.** The bit matching the receiving router's own eth facing
means "write locally." A packet travelling north arrives at the next chip's *south*-facing router, so
a north-going route sets `FORWARD_SOUTH` to deliver there. Hence `encode_2d_unicast` ends a route with
the **opposite** direction bit.

There is no `FORWARD_Z`. `eth_chan_directions::Z = 4` exists in the enum and the kernel's
`direction_to_compact_index_map` has a Z column, but bit 4 is never emitted.

### 2.2 Express — action byte

`route_buffer[coord]` is the action for whichever router sits at that coordinate.
`IndexedMeshRoutingFields`:

```text
bit 0  ACTION_EAST            \
bit 1  ACTION_WEST             |  bits 0..4 == (1 << eth_chan_directions),
bit 2  ACTION_NORTH            |  asserted in fabric_common.h:510-524
bit 3  ACTION_SOUTH            |
bit 4  ACTION_Z               /
bit 5  ACTION_LOCAL_DELIVER
bits 6-7  reserved, must be 0 (kernel fail-stops otherwise)
```

Delivery is an explicit bit, so no opposite-direction convention and no positional inference. A
self-facing bit is always invalid (kernel §3.7 — no same-link return path exists).

### 2.3 Packet header

Both use `HybridMeshPacketHeaderT<RouteBufferSize>`, so they share the header *type*; express simply
leaves three fields dead.

```text
base = 61 B
   44  PacketHeaderBase (NocCommandFields 40, payload_size 2, noc_send_type 1, src_ch_id 1)
    4  routing_fields        ← current only; express writes 0 and never reads it
    4  dst_start_node_id       both
    8  mcast_params_64         both
    1  is_mcast_active       ← written by both, read for routing by neither
```

---

## 3. Sizing

### 3.1 How the route buffer is sized

```text
current:  max_2d_hops = (rows-1) + (cols-1)          fabric_host_utils.cpp:114
express:  Y + X                                      fabric_context.cpp:359
          → +2 bytes over the same shape's hop count
```

`compute_packet_specifications` takes the max of the two (`fabric_context.cpp:148`), then picks a
tier.

### 3.2 Tiers, and the Galaxy cliff

```text
61 + 19 =  80 B   ← DISABLED: "de-stabilized some Mesh benchmarks for 8X4 mesh"
61 + 35 =  96 B
61 + 51 = 112 B
61 + 67 = 128 B
```

Galaxy `[32,4]` needs 34 hop bytes (fits 35 → **96 B**) but `Y+X = 36` (→ next tier → **112 B**).

So express costs Galaxy a header tier today. Retiring `routing_fields` (4 B) and `is_mcast_active`
(1 B) — both dead under express — drops the base to 56 B, and `56 + 40 = 96`. A 40-byte tier at 96 B
covers every in-tree shape:

| shape | Y+X | 96 B tier? |
|---|---|---|
| `[8,8]` | 16 | ✓ |
| `[16,4]`, `[1,16]` | 20, 17 | ✓ |
| `[16,8]`, `[8,16]` | 24 | ✓ |
| `[32,4]` Galaxy | 36 | ✓ |
| `[64,4]` future | 68 | 128 B (56+68=124) |

**This is why the header-field retirement is load-bearing, not cleanup.**

### 3.3 L1

| | current | express |
|---|---|---|
| union slot | `compressed_route_2d_t[256]` = 1024 B | `indexed_route_vectors_t` = 1028 B |
| per entry | 4 B: `ns_hops:7, ew_hops:7, ns_dir:1, ew_dir:1, turn_point:7` | — |
| indexed by | destination chip | destination coordinate, per axis |
| mesh-identical? | no — per source chip | vectors **yes**; reverse trees **no** (one per root chip) |

Express layout inside the slot:

```text
[0, table_bytes(Y))                    Y vectors, destination-major, 2 bits per entry
[table_bytes(Y), +table_bytes(X))      X vectors
align4 → mcast_tree_y_offset           reverse tree T(my_y): 2 B × (Y-1)
                                       reverse tree T(my_x): 2 B × (X-1)
```

| shape | vectors | trees | total | fits 1028? |
|---|---|---|---|---|
| `[8,8]` | 32 | 28 | 60 | ✓ |
| `[16,8]`, `[8,16]` | 80 | 44 | 124 | ✓ |
| `[32,4]` | 260 | 68 | 328 | ✓ |
| `[32,32]` | 512 | 124 | 636 | ✓ |
| `[64,4]` | 1028 | 132 | 1160 | ✗ |

`[64,4]` is the only shape that does not fit, and no descriptor declares it.

---

## 4. Unicast

### 4.1 Current

Host (`compressed_routing_path.cpp`): for each destination, try each active source direction via
`get_fabric_route`, keep the shortest chip sequence, walk it consuming the NS axis then the EW axis,
store `(ns_hops, ew_hops, ns_dir, ew_dir, turn_point = ns_hops)`.

**Z is not among the candidate directions**, and one NS hop count cannot hold an alternating
`S,Z,S,Z,S` route. That is the structural reason express needed a different representation.

Device (`fabric_routing_path_interface.h:14` → `encode_2d_unicast`):

```text
emit (ns_hops - 1 + prepend) × ns_forward
emit  ew_hops × ew_forward
emit  1 × ew_opposite            ← makes the destination router's own-facing bit match
pad remainder with NOOP
```

Two `uint8_t[MESH_ROUTE_BUFFER_SIZE]` stack arrays (70 B of stack) build the direction lists first.
`prepend_one_hop` distinguishes worker callers from router callers.

### 4.2 Express

`widen_indexed_route_to_chip` (`tt_fabric_api.h:392`):

```text
for i in 0..Y:   route_buffer[i]     = widen_y(extract_2bit(y_vectors[dst_y], i))
for i in 0..X:   route_buffer[Y + i] = widen_x(extract_2bit(x_vectors[dst_x], i))
route_buffer[Y + dst_x] |= ACTION_LOCAL_DELIVER
```

Delivery goes on the **X** slot only: the Y row widens to `STOP` at `dst_y`, so decode falls through
to X. No prepend distinction — a router encoding for itself writes its own slot like anyone else.

### 4.3 Cost

Comparable. Current writes the full padded buffer (35 B) plus 70 B of stack scratch; express writes
exactly `Y+X` (36 B) with a 2-bit extract per entry and no scratch.

---

## 5. Multicast

The largest behavioural difference, and where the defects are.

### 5.1 Current

`fabric_set_mcast_route` (`tt_fabric_api.h:180-253`):

```text
if (e) mcast_branch |= FORWARD_EAST
if (w) mcast_branch |= FORWARD_WEST

if      (n) { fabric_set_route<true>(NORTH, mcast_branch, 0, n); spine = n }
else if (s) { fabric_set_route<true>(SOUTH, mcast_branch, 0, s); spine = s }
              ^^^^^^^ N and S are MUTUALLY EXCLUSIVE

if (e) { fabric_set_route<true>(EAST, 0, spine, e); spine += e }
if (w) { fabric_set_route<true>(WEST, 0, spine, w) }
```

`fabric_set_route<true>` writes, per hop in `[start_hop, start_hop+num_hops)`:

```text
route_buffer[i] = opposite_bit | (i == last ? mcast_branch : own_bit | mcast_branch)
```

and records where each branch's program begins:

```text
branch_east_offset = spine
branch_west_offset = spine + e
```

On an E/W turn the sender-side update jumps `hop_index = branch_east_offset` or
`branch_west_offset` (`fabric_erisc_router.cpp:989-1005`).

### 5.2 Express

Two host artifacts, then a reverse pass:

```text
1. clear route_buffer[0 .. Y+X)

2. y_targets = (n==0 && s==0) ? {anchor_y}
                             : {(anchor_y-k) mod Y : k∈1..n} ∪ {(anchor_y+k) mod Y : k∈1..s}
   x_targets = {anchor_x} ∪ {(anchor_x±k) mod X}

3. prune X: walk T(my_x) leaves→root; if child ∈ needed, out_x[parent] |= widen_x(output),
            needed |= {parent}
   out_x[t] |= LOCAL_DELIVER for every t ∈ x_targets

4. prune Y: same over T(my_y)

5. teeth   = out_x[encode_root_x] & (E|W)
   deliver = out_x[encode_root_x] & LOCAL_DELIVER
   for y ∈ y_targets: out_y[y] |= teeth | deliver
```

`encode_indexed_mcast_maps`, `fabric_common.h:595`. Fixed `(Y-1)+(X-1)` loop, two `uint32_t` bitmaps,
no allocation. A packed edge carries both endpoints and the parent's command, so a Z edge never has to
be followed forward.

Step 5 applies teeth to **target rows, not the root row** — which turns out to matter a lot (§7).

### 5.3 Capability

| | current | express |
|---|---|---|
| N and S together | ✗ — the `else if` | ✓ |
| E and W from one header | ✗ — both branches share `route_buffer`; W lives at offset `e`, but `hop_index` starts at 0, so a worker injecting West reads the *East* branch's first byte. Only a spine turn ever jumps to `branch_west_offset`. | ✓ |
| Z in the tree | ✗ | ✓ |
| delivery at intermediate rows | ✓ | ✓ |

### 5.4 A capacity cliff in the current path

Current needs `spine + e + w` route bytes, with `ASSERT(end_hop <= MESH_ROUTE_BUFFER_SIZE)`
(`tt_fabric_api.h:114`). But the buffer is sized from the *unicast* corner-to-corner hop count. On
`[32,4]` with a 35-byte buffer:

| request | bytes | fits? |
|---|---|---|
| unicast worst case | 34 | ✓ |
| N-spine 31 + E 3 | 34 | ✓ |
| **N-spine 31 + E 3 + W 3** | **37** | **✗ asserts** |

Express is always exactly `Y+X`, independent of extents.

### 5.5 Cost

On `[32,4]`, spine + teeth:

```text
current:  ~37 byte writes across 4 calls
express:  36 clears + 34 edge iterations + 32 target iterations  ≈ 100 ops
```

Express multicast encode is ~2-3× the work. It is per-packet worker-side setup, so it matters most
for small multicast payloads in a tight loop. Two shape-independent early-outs recover most of it:
`e=w=0` makes step 3 collapse to `out_x[root_x] = LOCAL_DELIVER`, and `n=s=0` skips step 4.

---

## 6. Decode

### 6.1 Current

```text
hop_cmd = get_cmd_with_mesh_boundary_adjustment(...)      # route_buffer[hop_index], + edge rewrite
can_forward_packet_completely(hop_cmd, ...)               # 16-arm switch, or WH+VC1 bit tests
receiver_forward_packet(..., hop_cmd)
hop_index++   (or jump to branch_*_offset on a turn)
```

Header mutation happens in one of two places: sender-side
`update_packet_header_before_eth_send`, or RX-side under `UPDATE_PKT_HDR_ON_RX_CH`. Both exist; the
CT arg picks one.

### 6.2 Express

```text
if constexpr (MY_DIR ∈ {E, W}):  action = route_buffer[Y + local_x]
else:                            action_y = route_buffer[local_y]
                                 action = action_y != 0 ? action_y : route_buffer[Y + local_x]
```

`IndexedMeshRoutingFields::decode_action`, `fabric_common.h:424`. Then:

```text
validate reserved bits and the self bit          → fail-stop if set
key = pack_fwd_key<MY_DIR>(action)               → 4 bits over FWD_DIRS[4], LOCAL_DELIVER excluded
admit_express_combo<KEY>(ld, ...)                → atomic across every selected output + local relay
forward_express_combo<KEY>(...)                  → identical full packet per output, local last
```

`local_y`/`local_x` are cached once at setup (`fabric_erisc_router.cpp:2550`), so transit does no
divide, no mod, no L1 read, and no 2-bit extract.

Two subtleties worth flagging because they look like bugs and are not:

- the Y test is `action_y != 0`, **not** `action_y & (N|S|Z)`. A terminal multicast row is
  `LOCAL_DELIVER|E|W` with no Y child; the narrower test would wrongly fall through to X.
- E/W-facing routers never index Y, even though they have a valid `local_y`. Re-entering Y would
  re-fire the spine from a tooth.

### 6.3 Fanout

| | current | express |
|---|---|---|
| mechanism | cursor jump to a per-branch program | NOC-copy the *same* header to each sender |
| per-branch header rewrite | yes | none |
| admission | per the 16-arm switch | atomic across all selected outputs before any copy |

---

## 7. The client contract that makes them interchangeable

The route-variant multicast API (`mesh/api.h:1178`) takes `const MeshMcastRange* ranges` — **one
range per connection** — and issues one operation per outgoing direction:

```cpp
PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* h, uint8_t i) {
    auto& slot = connection_manager.get(i);
    fabric_multicast_noc_unicast_write(&slot.sender, h, slot.dst_dev_id, slot.dst_mesh_id,
                                       ranges[i], ...);
});
```

So a bidirectional N+S multicast is already two operations. That is why the current path's
`else if (n) ... else if (s)` was never a bug, and why the E/W cursor collision (§5.3) never
surfaced.

Trace express under that contract on a mesh with no chords:

| request | `out_y[root_y]` | eth outputs |
|---|---|---|
| `n=k` | `N` | **1** |
| `s=k` | `S` | **1** |
| `n=k, e=j` | `N` — teeth land on *target* rows, not the root row | **1** |
| `e=j` (n=s=0) | `E \| LOCAL_DELIVER` | **1** |
| `n=k, s=m` | `N \| S` | 2 |
| `e=j, w=i` (n=s=0) | `E \| W \| LOCAL_DELIVER` | 2 |

The only two multi-output cases are exactly the two the current path also cannot do in one operation.

> **Under the per-direction contract that already exists, express multicast on a chordless mesh
> produces a single-output root, so the standard single-connection API works unchanged and no client
> migration is required.**

Source multi-inject (`fabric_multicast_source_inject_noc_unicast_write`) is needed only where a
*single-direction* request leaves on two edges — which requires Z chords, i.e. express only. That
helper exists and ships; codec §7.3.1's "express meshes ship unicast only" is stale.
Example from codec §7.3.1: from `Y=1` with `N=2` on `[32,4]`, targets are `{0, 31}`,
`R(1,0) = N` and `R(1,31) = Z,S`, so `action_y[1] = N|Z`.

---

## 8. Intermesh

| step | current | express |
|---|---|---|
| carrier to exit | `route_buffer` toward the exit chip | unicast-style maps toward the exit; `dst_start_node_id` retains the **final** destination |
| exit detection | `is_intramesh_router_on_edge && dst_mesh != my_mesh && hop_cmd == FORWARD_<my_dir>` | `action == LOCAL_DELIVER && dst_mesh != my_mesh` (`action_is_intermesh_exit`) |
| at the exit | `recompute_path` — **rebuilds** the route | forwards **as-is** on the `INTERMESH` egress |
| landing | `hop_cmd == NOOP` → `recompute_path` | `receiver_channel_is_intermesh_ingress[rx]` → landing encode **before** decode |
| intermediate vs destination mesh | implicit in recompute | explicit: compare retained final mesh with current mesh |
| Z ambiguity | `NOOP` means both "no-op" and "recompute" | none; Z is bit 4, boundary comes from edge capability |

Express relies on an encoder invariant (codec §4.5): a carrier's maps must decode to **exactly**
`LOCAL_DELIVER`, no eth bits, at the exit chip. Both halves of the predicate are load-bearing —
mesh-id inequality alone also matches a chip merely transiting toward a different exit.

Note the semantic shift: current *rebuilds* at the exit, express *forwards unchanged*. That is safe
only because of the invariant above, which is why it must be asserted rather than assumed.

---

## 9. Defects in the express implementation

Present in the tree today, independent of unification. Two of them are why express cannot simply be
switched on for every mesh.

### 9.1 ⚠ A missing reverse tree makes multicast a silent no-op

`control_plane.cpp:1994-2003` logs a **warning** and leaves the tree region zeroed when the
arborescence gate fails or when `ring_for_direction` returns null. Nothing device-side checks.

Trace an all-zero region through `mcast_prune_axis` (`fabric_common.h:563`):

```text
edge_count = Y-1                    # the loop still runs
edge = 0 for every i:
    child = 0, parent = 0, code = 0
    if row 0 ∈ needed: out[0] |= widen_y(0) = 0;  needed |= {0}
```

`route_buffer_y` ends up entirely zero, the root action is 0, the worker injects nowhere, and **the
multicast delivers to nothing** — no assert, on a release build. Codec §6.1 anticipates this ("the
multicast producer must refuse to encode against a mesh that lands here") but
`fabric_set_indexed_mcast_route` has no such check.

A valid 2-row edge is `child=1, parent=0, output≠0`, so an all-zero region is distinguishable. Needs a
validity check plus a host `TT_FATAL`.

### 9.2 ⚠ Non-express meshes have no tree source at all

`ring_for_direction(N)` returns `get_express_rings()`, **null on every non-express mesh**, so
`control_plane.cpp:1983` skips tree embedding entirely — and 9.1's silent path is what a unified build
would hit. The generic fallback `derive_ordinary_ring_topology` requires the axis to **wrap**
(`express_ring_topology.cpp:552`), so LINE axes get nothing, and `x_rings_` is derived only inside the
express-only branch (`routing_table_generator.cpp:60`, after a `continue`).

Affected in-tree descriptors: `single_bh_galaxy [8,4]`, `p150_x8 [2,4]`, `dual_bh_lb [2,4]`,
`16x4_dual_bh_galaxy [16,4]`, `quad_galaxy [8,16]`, `dual_galaxy [8,8]`, plus the small shapes.

### 9.3 A same-mesh multicast silently ignores the caller's anchor

`fabric_set_indexed_mcast_route` sets `dst_start_node_id` from `dst_dev_id`, then computes
`root_y`/`root_x` from `my_mesh_coord_*`. For a same-mesh send the extents are measured from the
**local chip**. The intermesh landing path *does* read the anchor from `dst_start_node_id`
(`tt_fabric_api.h:562`), so the two paths interpret the same field differently. The current path has
the same implicit contract, so it is not a regression — but it is an unasserted precondition the
landing depends on.

### 9.4 The single-hop helper never got an express arm

`fabric_set_single_hop_unicast_route_from_direction` (`tt_fabric_api.h:48`) has **no** express gate
and writes the legacy hop byte at `route_buffer[0]`. The express kernel reads
`route_buffer[local_y]`. Meanwhile `fabric_set_indexed_single_hop_unicast_route_from_direction`
(`:586`) exists with **zero callers**.

Callers of the broken one: `models/demos/deepseek_v3_b1/unified_kernels/` — `all_reduce.hpp:212`,
`all_gather.hpp:182`, `broadcast.hpp:175`, `reduce_to_one_b1.hpp:255`,
`sdpa_reduce_worker.hpp:131`. Exactly the Galaxy workloads express targets. This is a live ABI
mismatch, fixable standalone.

### 9.5 Also: a non-codec site keyed on the codec flag

`routing_plane_connection_manager.hpp:21` reads `FABRIC_EXPRESS_ENABLED && ARCH_BLACKHOLE` to size
`TT_FABRIC_MAX_ROUTING_PLANE_CONNECTIONS` at 6 instead of 4. That is **Z-port presence**, not the
codec. It is the one device-side site where the flag means what its name says, so it keeps both its
name and its condition — but it must not follow the codec sites to `FABRIC_2D`, or every Blackhole
connection manager grows two slots.

---

## 10. What the comparison implies

| finding | consequence for unification |
|---|---|
| express is strictly more capable (§5.3), constant-size (§5.4), and mutation-free (§6.3) | it is the right survivor; there is no capability argument for keeping the current path |
| the per-direction client contract yields single-output roots (§7) | express multicast is a **drop-in** on chordless meshes — zero client migrations |
| §9.2 — no tree source on non-express meshes | **hard blocker.** Without a line-axis topology, unification converts working multicast into a silent no-op on ~half the in-tree 2D descriptors |
| §9.1 — silent failure mode | must become loud before the flip, not after |
| §3.2 — Galaxy 96 B → 112 B | the header-field retirement is load-bearing; sequence it with or before the tier change |
| §8 — express forwards at the exit instead of rebuilding | the exit-chip `LOCAL_DELIVER` invariant must be asserted, since `recompute_path`'s safety net disappears |
| §5.5 — express multicast encode is 2-3× | measure on small multicast payloads; two early-outs available |
| §4.1 — one NS hop count cannot hold `S,Z,S,Z,S` | the original reason express exists; not reconcilable within the current representation |

One thing the comparison does **not** support: folding 1D in. 1D is a different codec on a different
header type for a single-downstream topology, where the indexed representation costs 16-48 header
bytes to save roughly one instruction. It is out of scope — see the plan's non-goals.
