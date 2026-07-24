# Fabric Device Route Codec Contract

Working reference for **device-visible routing**: host path-table generation and L1 embed →
worker/edge packet load → indexed decode → kernel forward/deliver. Equal weight on **current code
baseline** and the **cardinal N/S + Z express** target ABI.

Authority split and related documents:

- `GALAXY_WORKING_MODEL.md` — physical/logical Galaxy context and software-stack overview
- `GALAXY_CARDINAL_NS_Z_SKIP_ROUTING_ASSESSMENT.md` — topology/system assumptions, command
  semantics, canonical route oracle and routing-generation policy, CDG/SCC + VC0 BFC proof, system
  invariants, “Route representation requirements,” “Multicast semantics and safety,”
  “Alternative and extension decision record,” and Appendix A validation oracles
- `GALAXY_CONTROL_PLANE_ROUTING_GENERATION_CONTRACT.md` — detailed MGD/MeshGraph materialization,
  ring synthesis, canonical-route generation, and ControlPlane route/predicate handoff
- `GALAXY_BUILDER_ROUTING_CONFIG_CONTRACT.md` — ControlPlane ↔ FabricBuilder query surface
  (authoritative for *injection / BFC / wiring*)
- `GALAXY_DEVICE_ROUTER_KERNEL_CONTRACT.md` — ERISC baseline, admit/forward/fanout, NOOP removal,
  CT/speedy gates (authoritative for *kernel realization*)

**This file** is the delegated authority for L1/device routing artifacts, the packet/header ABI,
encode/load/decode, the multicast encoder, and the source fanout/reroot overlay. Its working ABI is
dual widened Y/X action-byte maps with per-chip reverse-tree multicast as the selected V1 device
artifact. Prefer the kernel contract for `fabric_erisc_router.cpp` mechanics and admission detail.

Status: **target working contract plus current-code baseline** (ABI decisions locked enough to
design against). Target requirements are aspirational and not implemented unless explicitly marked
otherwise; this is not a finalized public C++ API. Normative target bits are marked **required
contract**; inventory of today’s code is **baseline (code as of this writing)**.

---

## 0. How to use this document

Use this document for the current device-codec baseline, the target L1 and packet ABI, and the
host/worker/landing encode and load rules. §9 summarizes the facts exchanged with adjacent
components.

Route generation and proofs belong to the assessment; BFC effects and wiring belong to the builder
contract; full ERISC realization belongs to the kernel contract. Arbitrary-pattern VC1 safety/BFC and
a 1D codec redesign remain outside this contract; the current restricted VC1 envelope is not removed
by this codec work.

### 0.1 Preferred terminology

| Prefer | Avoid (in codec/API text) | Meaning |
|---|---|---|
| **express** | skip *(alone)* | Intramesh non-NN chord; command **Z** |
| **cardinal** | base *(for commands)* | Ordinary geometric N/S/E/W |
| **L1 2-bit vectors** | full action maps in L1 | Destination-major compressed Y/X relation in L1 |
| **route_buffer_y / route_buffer_x** | hop program; Y-only + line/counter X | Packet action-byte maps indexed by `local_y` / `local_x` |
| **widen+union (unicast)** | full dest-vector OR for mcast | Unicast: widen entire `*_vectors[dst]` |
| **source-reachable OR (mcast)** | full-vector union of all dests | Required *semantics*: union of `R(encode_root, target)` only (§5.6) |
| **root-pruned tree encoder** | independent mcast route policy | **Primary** express mcast artifact (§5.7.1); per-chip tables; V1 uniform per mesh after all-roots gate |
| **range_anchor** | encode_root *(confused)* | Chip extents are relative to; may differ from trace root |
| **encode_root** | — | Chip where path tracing begins (src or dest-mesh landing) |
| **same-header fanout** | clone with different route program | NOC-copy identical maps to each sender |
| **source multi-inject** | root RX fanout | Worker sends one identical-header packet copy through each connected canonical root output (§7.3.1) |
| **pre-inject reroot overlay** | replacement canonical route tree | Temporary same-mesh express fallback: edit two action bits before injection so one connected root child returns a copy to launch the other root subtrees; not itself a canonical §5.6 map (§7.3.1) |
| **same-link U-turn sender** | ordinary route output | Dedicated fallback-only VC0 child-RX→same-link-TX queue (kernel §3.9 / builder §3.7) |
| **canonical route U-turn** | same-link U-turn sender *(confused)* | A route-generated immediate reversal; remains forbidden and is not legalized by the infrastructure queue |

Keep **base-32** only when discussing the forbidden full Y-NN deadlock domain (assessment VC0 BFC
proof and “Alternative and extension decision record”).

### 0.2 Contract status

- **Current baseline:** compressed 2D hop programs and spine/branch multicast (§2).
- **Target ABI:** destination-major L1 vectors widened into packet Y/X action maps, with
  facing-aware indexed decode (§4).
- **Target multicast:** source-reachable action maps; a per-chip reverse tree is the primary encoder
  when every root passes §5.7.1. Source injection follows §7.3.1.
- **L1 sizing:** the `[64,4]` 1028-byte vector table fits in L1 (§2.12). The 1160-byte hybrid needs
  an in-struct growth from 2576 B to at least 2608 B after intra-table reclamation, 2704 B if that
  table remains, or external placement (§6.2); the 68-byte packet-map and current axis-limit changes
  remain target work (§4.7).
- **Implementation status:** specified, not implemented. §11 lists only the remaining codec-owned
  implementation choices and checks.

Validated route and multicast counters live in the assessment Appendix A. Kernel realization status
lives in `GALAXY_DEVICE_ROUTER_KERNEL_CONTRACT.md`.

---

## 1. Ownership (summary)

```text
RT gen / CP     → finalized canonical routes (annotated N/S/E/W/Z logical edges)
                  + mesh-identical dest-major Y/X 2-bit vectors
                  + per-chip reverse trees (§5.7.1 host sketch; mesh-wide arborescence gate)
                  embed L1; expose express enable / neighbors (builder wiring — not worker Z-step)
Worker / edge   → unicast: widen dest-major vectors (§7.1)
                  mcast (express primary): reverse-prune T(my_chip) / T(landing) (§5.7.1)
                  V1: one mcast representation per mesh/configuration (setup policy, not packet semantics)
                  same-mesh source: manager multi-inject or pre-inject reroot overlay (§7.3.1)
                  shared header; no transit per-branch map rewrite
Transit router  → N/S/Z: if route_buffer_y[local_y] != 0 use it, else route_buffer_x[local_x]
                  E/W: route_buffer_x[local_x] only (never re-enter Y; §4.3 / §5.9)
                  fanout = NOC-copy same header to each sender
                  never re-search routes; never consult persistent Y/X tables on hot path
Builder         → wires Z↔N/S VC0, emits IS_INJECTION (see builder contract)
```

Hard rule for the target ABI: **route search happens once during host route setup**. Device consumers
derive packet state and hop actions from installed tables / immutable packet overlays. The kernel does not
infer protected-domain roles from N/S/Z letters.

---

## 2. Current baseline — end-to-end pipeline

### 2.1 One-line picture (today)

```text
RoutingTableGenerator (N/S then E/W direction)
        → ControlPlane channel tables + get_fabric_route()
        → compressed_routing_path::calculate_chip_to_all_routing_fields()
              packs compressed_route_2d_t per destination (4B)
        → ControlPlane::compute_and_embed_2d_routing_path_table()
              memcpy into routing_l1_info_t.routing_path_table_2d
        → write_routing_info_to_devices() → Tensix / ERISC L1
        → worker: fabric_set_unicast_route()
              decode_route_to_buffer() → routing_encoding::encode_2d_unicast()
              fills HybridMeshPacketHeader.route_buffer[] with N/E/S/W hop bytes
        → router: route_buffer[hop_index] → hop_cmd bitmask
              can_forward_packet_completely()/receiver_forward_packet() hop_cmd switch
              hop_cmd_to_sender_channel_mask() is trim/resource-capture only
              hop_index++ / branch_*_offset for mcast turns
```

### 2.2 File / API inventory

| Layer | Path | Role today |
|---|---|---|
| Shared ABI | `tt_metal/hostdevcommon/api/hostdevcommon/fabric_common.h` | `compressed_route_2d_t`, `RoutingFieldsConstants::Mesh`, `routing_encoding::*`, `intra_mesh_routing_path_t`, `routing_l1_info_t`, `direction_table_t`, L1 base macros |
| Host path gen | `tt_metal/fabric/compressed_routing_path.cpp` | `calculate_chip_to_all_routing_fields` for 1D/2D |
| Host embed | `tt_metal/fabric/control_plane.cpp` | `compute_and_embed_{1,2}d_routing_path_table`, `write_routing_info_to_devices` |
| Header sizing | `tt_metal/fabric/fabric_context.cpp` / `.hpp` | max hops from geometry → route_buffer tier |
| Packet header | `tt_metal/fabric/fabric_edm_packet_header.hpp` | `HybridMeshPacketHeaderT`, `LowLatencyMeshRoutingFields` |
| Device decode | `tt_metal/fabric/hw/inc/fabric_routing_path_interface.h` | `decode_route_to_buffer` → `encode_2d_unicast` |
| Worker/edge API | `tt_metal/fabric/hw/inc/tt_fabric_api.h` | `fabric_set_unicast_route`, `fabric_set_mcast_route`, `fabric_set_route`, `fabric_set_single_hop_unicast_route{,_from_direction}` |
| Edge recompute | `tt_metal/fabric/hw/inc/edm_fabric/fabric_edge_node_router.hpp` | `recompute_path`, `get_cmd_with_mesh_boundary_adjustment` (NOOP / edge triggers) |
| Kernel | `tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp` | hop_cmd → sender mask; mcast branch jumps |
| Hop advance | `tt_metal/fabric/hw/inc/edm_fabric/fabric_edm_packet_transmission.hpp` | route_buffer shift/refill for long paths |

### 2.3 Persistent L1: `routing_l1_info_t`

Defined in `fabric_common.h`. Packed size **2576** bytes:

| Field | Size | Notes |
|---|---|---|
| `RouterStateManager` | 32 | host/device command handshake |
| `my_mesh_id`, `my_device_id` | 4 | local identity |
| `intra_mesh_direction_table` | 96 | `direction_table_t<MAX_MESH_SIZE>` — first hop toward dst **chip** |
| `inter_mesh_direction_table` | 384 | first hop toward dst **mesh** |
| union path table | 1024 | 1D uncompressed **or** 2D compressed (`intra_mesh_routing_path_t`) |
| `exit_node_table` | 1024 | 1 byte per mesh → exit chip |
| padding | 12 | 16-byte align |

Device bases (selected by core type):

```text
ROUTING_TABLE_BASE      → routing_l1_info_t
ROUTING_PATH_BASE_2D    → same region’s 2D path table (alias into union)
EXIT_NODE_TABLE_BASE    → exit_node_table
```

**Important:** path table and direction table are **different consumers**:

- **Path table** (`compressed_route_2d_t[256]`): recipe for expanding a full hop program into
  `route_buffer` (NS hops + EW hops + dirs + turn_point).
- **Direction tables**: first-hop `eth_chan_directions` toward a destination chip or mesh (used by
  `get_next_hop_router_direction`, single-hop helpers, edge recompute).
- **Exit table**: canonical exit chip for a foreign destination mesh (per source chip).

Neither stores express-edge identity, multi-segment Y recipes, or protected-domain effects.
Target fate of these L1 fields under indexed encoding: **§2.11**.

### 2.4 `compressed_route_2d_t` (4 bytes)

Fields (bit packing in `fabric_common.h`):

```text
ns_hops     : 7
ew_hops     : 7
ns_dir      : 1   # 0 = North, 1 = South
ew_dir      : 1   # 0 = West, 1 = East
turn_point  : 7   # hop index where NS→EW turn occurs (typically = ns_hops)
```

**Represents one monotonic NS segment then one monotonic EW segment.** Cannot encode
`S, Z, S, Z, S, E, E` without collapsing Z into an NS hop count (which loses the express edge).

Host fill (`compressed_routing_path.cpp`):

1. For each dst ≠ src, try active E/W/N/S source channels via `ControlPlane::get_fabric_route`.
2. Keep shortest **intramesh chip sequence**.
3. Walk sequence with `get_forwarding_direction`; consume NS axis then EW axis.
4. `paths[dst].set(ns_hops, ew_hops, ns_dir, ew_dir, ns_hops)` — turn_point = ns_hops.

Z direction is **not** among the candidate source directions in this loop.

### 2.5 Mesh hop commands (`RoutingFieldsConstants::Mesh`)

Per-hop command is an **8-bit byte** stored in `route_buffer[i]`, but only the low **4 bits** are
defined as direction flags:

```text
NOOP           = 0b0000
FORWARD_EAST   = 0b0001   # eth_chan_directions::EAST = 0
FORWARD_WEST   = 0b0010   # WEST = 1
FORWARD_NORTH  = 0b0100   # NORTH = 2
FORWARD_SOUTH  = 0b1000   # SOUTH = 3
```

Multicast ORs these bits (`WRITE_AND_FORWARD_*` combinations). There is **no `FORWARD_Z`** constant
today. `eth_chan_directions::Z = 4` exists in the enum; kernel `direction_to_compact_index_map`
already has a Z column/row, but the hop-byte ABI does not emit bit 4 (`0b10000`) for intramesh
express.

**Local delivery convention (today):** the bit matching the router’s own ingress-facing direction
means “write locally”; other bits mean forward. Final unicast hop often uses the **opposite**
direction bit (see `encode_2d_unicast`) so the destination router’s own-direction bit matches.

### 2.6 `routing_encoding::encode_2d_unicast`

Signature (host + device):

```text
encode_2d_unicast(ns_hops, ew_hops, ns_dir, ew_dir, buffer, max_buffer_size, prepend_one_hop=false)
```

Behavior:

- Emit `(ns_hops - 1 + prepend)` NS forward bytes, then `ew_hops` EW forwards, then one EW
  **opposite** write byte (when both axes nonempty).
- NS-only / EW-only analogous.
- Pad remainder with `NOOP`.

`prepend_one_hop=true` is used when **called from an edge/router** so the expanded program accounts
for the hop already being taken / about to be taken.

Device path: `intra_mesh_routing_path_t<2,true>::decode_route_to_buffer` reads `paths[dst]`, then
calls `encode_2d_unicast` into a temp buffer sized `FabricHeaderConfig::MESH_ROUTE_BUFFER_SIZE`,
then copies to the packet `route_buffer`.

### 2.7 Packet header overlay (today)

`HybridMeshPacketHeaderT<RouteBufferSize>` (`fabric_edm_packet_header.hpp`):

```text
LowLatencyMeshRoutingFields routing_fields;  # 4B: hop_index | branch_east | branch_west
uint8_t route_buffer[RouteBufferSize];       # hop program
dst_start_node_id / chip+mesh;               # 4B
mcast_params_64 / mcast_params[4];           # 8B  E,W,N,S extents
uint8_t is_mcast_active;                     # written; not meaningfully read for routing
```

Static asserts (base header, non-UDM):

```text
RouteBufferSize 19 → 80B header
35 → 96B
51 → 112B
67 → 128B
```

Default host compile uses **35** (`MESH_ROUTE_BUFFER_SIZE` default). Build can override via
`FABRIC_2D_PKT_HDR_ROUTE_BUFFER_SIZE`.

`FabricContext` selects the tier from **geometry max hops** ≈ `(rows-1)+(cols-1)`, not from
express-aware diameter. For 4×32 that yields 34 → 35-byte buffer / 96B header.

### 2.8 Worker load: `fabric_set_unicast_route` (2D)

`tt_fabric_api.h`:

1. Optionally rewrite dst through `EXIT_NODE_TABLE` when `dst_mesh_id != my_mesh_id`.
2. Worker path: `decode_route_to_buffer(dst, route_buffer)` (no prepend).
3. Router path (`called_from_router`): may prepend a direction byte; **Z uses `NOOP`** as the
   “forward to Z” sentinel.
4. Sets `branch_east_offset` / `branch_west_offset` from compressed `turn_point` (+1 if from router).
5. Clears `routing_fields` value then fills branch offsets; `hop_index` starts at 0.

`fabric_set_single_hop_unicast_route` /
`fabric_set_single_hop_unicast_route_from_direction` (baseline):

- Contract today: dest is exactly one EWNS physical hop; **rejects Z** (`ASSERT != Z`).
- Clears `routing_fields`, sets `dst_start_node_id`, writes **one** hop byte at `route_buffer[0]`
  via `single_hop_route_cmd_by_direction` (**opposite** write bit: EAST→`FORWARD_WEST`, …; Z→`NOOP`).
- `from_direction` takes an explicit `eth_chan_directions`; the non-`_from_direction` form calls
  `get_next_hop_router_direction` (intra/inter direction tables).
- Still hop-program shaped (`hop_index == 0`); not destination-indexed.

Target replacement: §7.1.1 (poke my map slot + dest `LOCAL_DELIVER`; real direction bits; Z allowed
when express).

### 2.9 Kernel hot path (today)

`fabric_erisc_router.cpp` (2D):

1. `hop_cmd = route_buffer[hop_index]` (after edge adjustment).
2. Own-direction bit → local write; `can_forward_packet_completely` /
   `receiver_forward_packet` switch arms handle other bits. `hop_cmd_to_sender_channel_mask` is
   trim/resource-capture only.
3. After forward, advance `hop_index` (and mcast may **jump** `hop_index` to `branch_*_offset`).

Edge (`fabric_edge_node_router.hpp`):

- `hop_cmd == NOOP` → `recompute_path()` (intermesh / mesh-boundary trigger).
- Intramesh edge routers may also recompute when continuing toward a foreign mesh on their axis.

**Z today is overloaded:** intermesh / boundary signaling via `NOOP` and `Z_ROUTER` VC1 wiring
(builder contract), not intramesh express VC0.

### 2.10 Direction tables (baseline role)

`direction_table_t`: 3-bit compressed entries mapping destination index →
`eth_chan_directions` (including compressed Z / invalid sentinels).

Used for:

- `get_next_hop_router_direction(dst_mesh, dst_dev)` — picks intra vs inter table by mesh id
- First-hop / exit selection
- Edge recompute when expanding a new segment

**Not** used as the per-hop source of truth for a multi-hop intramesh unicast once `route_buffer` is
filled. Chaining direction-table lookups every hop would require persistent L1 hits and still cannot
distinguish two edges that share one direction letter.

### 2.11 L1 direction / exit tables under indexed encoding

Y/X destination vectors are **mesh-local / intramesh-only**. The same widen → `route_buffer_y` /
`route_buffer_x` codec is used for a final same-mesh chip and for an intermesh *carrier* segment
whose temporary destination is the canonical exit chip. Vectors do **not** encode cross-mesh routes;
the loader remains mesh-aware (`dst_mesh == my_mesh` vs rewrite-to-exit).

| L1 field | Decision | Rationale |
|---|---|---|
| `intra_mesh_direction_table` | **Removable** once indexed vectors ship | First hop toward a same-mesh chip is exactly the DOR peek of `y_vectors[dst_y][local_y]` then, on Y STOP, `x_vectors[dst_x][local_x]` (or the widened action bytes after setup). The table is a redundant cache of that first action, not independent routing truth. Migrate single-hop helpers off it when vectors are always installed. |
| `inter_mesh_direction_table` | **Keep as-is** | Still used for intermesh first-hop / boundary direction toward a foreign mesh (including at exit when an intramesh segment has completed). Not replaced by Y/X vectors. |
| `exit_node_table` | **Keep as-is** | Exit chip selection for `(src_chip, dst_mesh)` is mesh-graph policy. Vectors only route *to* a chip already in this mesh; they cannot name the exit. Intermesh flow remains: lookup exit → widen/install both maps to exit → at completion take boundary hop. |

That carrier-segment rule repeats after every **intermediate-mesh** landing: use the retained final-mesh
identity to select the next local exit, then install unicast-style Y/X maps from the landing chip to
that exit. Preserve the final `dst_start_node_id` and, for multicast, `mcast_params_64`. Only a
**destination-mesh** landing installs maps for the final unicast destination or multicast target
range; an intermediate landing must not build a multicast reverse tree or begin multicast fanout.

Scope note: removing `intra_mesh_direction_table` is an L1 / API cleanup gated on indexed-vector
cutover. It is not required for express correctness on day one if the table is simply unused by the
new loader; do not keep it as a second source of truth that can disagree with the vectors.

### 2.12 `[64,4]` destination-major table footprint

Destination-major Y+X relation size at the current dimensional bound:

```text
Y_TABLE = 64 × ceil(64 / 4) = 64 × 16 = 1024 B
X_TABLE =  4 × ceil( 4 / 4) =  4 ×  1 =    4 B
total                       =           1028 B
```

Today’s 2D path-table union slot is **1024 B** (`compressed_route_2d_t[256]`), so the bound needs
**+4 B** in that region. That is no longer treated as a blocking L1 problem:

1. **`routing_l1_info_t` already has 12 B of trailing padding** (16-byte alignment). Expanding the
   path/vector region from 1024 → 1028 and shrinking padding 12 → 8 keeps
   `sizeof(routing_l1_info_t) == 2576` and 16-byte alignment:
   `516 + 1028 + 1024 + 8 = 2576`.
2. **Reclaiming `intra_mesh_direction_table` (96 B)** after §2.11 cutover frees far more than 4 B
   ahead of the path region (`offsetof` path table drops from 516). That headroom can absorb 1028 B
   vectors and/or later layout cleanup without growing the overall L1 routing blob.

Either mechanism is sufficient; the implementation may use padding or reclaim the unused intra table
without growing the overall L1 routing blob.

The separate target **128 B header / 68 B route payload** for `[64,4]` packet state (§4.7) remains
independent of this L1 table closure. It is compatibility work, not current 2D capability: current
`HybridMeshPacketHeaderT` rejects route buffers above 67 B and current 2D ControlPlane validation
rejects either axis above 32. Both limits must be updated and validated before `[64,4]` is enabled.

---

## 3. Why the baseline cannot carry express routing

| Requirement | Baseline | Gap |
|---|---|---|
| Alternating N/S and Z on Y | One NS hop count | Lossless Z identity gone |
| Terminal ex4→ex8 vs continue ex8→ex4 | Opaque hop bytes | No domain semantics in packet |
| Bare Z output | No FORWARD_Z; NOOP means recompute | Collides with intermesh sentinel |
| One logical express neighbor as Z | Discovery/path gen ignore Z/express | MGD express not in runtime path table |
| Multicast over constrained unicast tree | Cardinal spine N/S + E/W branches | Leaf transit / base-32 merge hazard |
| Suffix-stable device decode | Sequential hop_index program | OK for cardinal DOR; need vectors for express |
| Header sizing | max cardinal hops | Topology-sized `Y + X` action-byte buffers (§4.7) |

The assessment’s “Route representation requirements” and “Multicast semantics and safety” state the
system requirements; this section is the concrete code binding.

---

## 4. Target route ABI (required contract)

The assessment’s “Route representation requirements” and canonical route oracle own relation/tree
legality. The delegated working device packet ABI is decided here:

- L1: mesh-identical dest-major **2-bit** `y_vectors` / `x_vectors`; **per-chip** reverse trees for
  express mcast (§5.7.1) when that approach is enabled mesh-wide
- Packet carries **`route_buffer_y[Y]`** and **`route_buffer_x[X]`**
- Unicast setup: full-vector widen; mcast setup: **source-reachable OR** semantics (§5.6) via the
  **single** chosen device encoder (§5.7.1 primary)
- Transit decode (§4.3): N/S/Z → if `action_y != 0` use Y else X; **E/W → X only**; same-header fanout
- Rejected paths: `E_LINE` / `remaining_hops`, packed 2-bit packet overlay (§4.2.1)

### 4.1 Destination-indexed 2-bit vectors (L1 / compressed relation)

Transpose next-hop tables into destination-major vectors **written to L1**:

```text
y_vectors[dst_y][current_y] → { STOP=00, N=01, S=10, Z=11 }
x_vectors[dst_x][current_x] → { STOP=00, E=01, W=10, INVALID=11 }
```

Packing:

```text
BITS_PER_ACTION  = 2
ACTIONS_PER_BYTE = 4
Y_VECTOR_BYTES   = ceil(Y / 4)
X_VECTOR_BYTES   = ceil(X / 4)
Y_TABLE_BYTES    = Y * Y_VECTOR_BYTES
X_TABLE_BYTES    = X * X_VECTOR_BYTES
```

Fixture persistent table sizes (destination-major, byte-aligned vectors):

```text
[8,4]  → 16 + 4  = 20 B
[16,4] → 64 + 4  = 68 B
[24,4] → 144 + 4 = 148 B
[32,4] → 256 + 4 = 260 B
[64,4] compatibility bound → 1024 + 4 = 1028 B  # fits; see §2.12
```

**Target compatibility bound:** the universal codec must preserve the repository-wide Y≤64 /
mesh≤256 dimensional requirement. Current 2D code does not yet realize a 64-row axis (§2.12);
do not describe this bound as implemented 2D support.

Do **not** store per-destination full axis action maps in L1 (`Y×Y` or `X×X` bytes would blow the
table budget). Compression stays in L1; expansion is a packet-setup step (§4.2).

### 4.2 Packet setup: widen into `route_buffer_y` + `route_buffer_x`

Header holds two axis maps (contiguous storage or named fields — layout choice):

```text
route_buffer_y[Y]   # action byte per logical Y
route_buffer_x[X]   # action byte per logical X   # same ABI treatment as Y
```

Widen helper (identical shape on both axes):

```text
widen_axis_vector(vec_2bit, n) → out[n]:
    for i in 0 .. n-1:
        a2 = extract_2bit(vec_2bit, i)
        out[i] = one_hot_action_byte(a2)    # STOP→0; else one direction bit
```

**Unicast:** widen the **full** destination vector (suffix consistency; any chip can install from L1).

**Multicast:** do **not** OR full destination vectors — that is not equivalent to tracing from
`encode_root` (§5.6). Trace each target from `encode_root` and OR only source-reachable one-hots.

| Traffic | Setup |
|---|---|
| **Unicast** `(dst_x, dst_y)` | `route_buffer_y = widen(y_vectors[dst_y])`; `route_buffer_x = widen(x_vectors[dst_x])` + `LOCAL_DELIVER` at `dst_x`. Encode assert: ≤1 of N/S/Z on any Y row; ≤1 of E/W on any X column. |
| **Multicast** | Source-reachable OR (§5.6) via reverse-prune `T(encode_root)` (§5.7.1 primary). Shared header. |

```text
L1 (compressed)                    packet                            transit (N/S/Z)
y_vectors 2-bit  →  unicast widen / mcast trace-OR → route_buffer_y  →  if action_y != 0: use it
x_vectors 2-bit  →  unicast widen / mcast trace-OR → route_buffer_x  →  else action_x
                                                                     E/W facing: action_x only
```

#### 4.2.1 Rejected alternatives (not optimization paths)

**`E_LINE` / `remaining_hops` as X** — rejected as working X and **not** kept as a perf opt:

1. **Y map is Y-indexed only.** After Y completes, every chip on row `dst_y` sees the same Y byte.
   That byte cannot encode both “continue East” and “deliver at `dst_x`” without mutation, an X
   index, or a side counter. An X action map matches Y.
2. **Line counters are hop-program state**, not destination-indexed routing — they reintroduce
   `hop_index` / branch-style control we are removing.
3. **Axis parity.** `route_buffer_x` gets the same widen/decode treatment as Y (dim-flip friendly).
4. **Size.** `Y + X` vs `(Y-1)+(X-1)` → **+2 B**; offset by dropping `hop_index` + `branch_*`.

**Packed 2-bit packet overlay** — rejected for this ABI: L1 stays 2-bit; setup widens
to action bytes. Do not revive it as a silent dual hot path.

Elsewhere in this doc, prefer “see §4.2.1” over restating these rejections.

### 4.3 Runtime decode (transit)

Router caches once: `local_y`, `local_x`, `my_direction` (pinned logical coords + eth facing).

```text
# E/W routers never re-enter Y (§5.12); same local_y as the fanout row would still see spine bits
if my_direction in {E, W}:
    action = route_buffer_x[local_x]
else:
    # N / S / Z
    action_y = route_buffer_y[local_y]
    if action_y != 0:                 # LOCAL_DELIVER and/or E|W tooth bits count — not only N/S/Z
        action = action_y
    else:
        action = route_buffer_x[local_x]
```

Why `!= 0` (not `N|S|Z`): a terminal mcast row may be `LOCAL_DELIVER | E | W` with no Y child; that
must stay on the Y map. Unicast at `dst_y` leaves STOP→0 so X runs. Single-hop Y dest writes
`LOCAL_DELIVER` on the Y slot (§7.1.1) — also nonzero.

Both maps are **immutable** on ordinary transit (no hop cursors, no per-branch map rewrite). Fanout =
NOC-copy the **same** header (today’s pattern); E/W vs N/S/Z differ only in which map that router
indexes. Intermesh boundary handling uses the retained final mesh identity plus builder-provided
intermesh ingress/egress capability (§4.5); it is not inferred from a direction letter.

No L1 vector read on transit. No 2-bit extract on the hot path (widen/trace was at setup).

### 4.4 Representability gate

Destination vectors are legal only if the canonical route set is **suffix-consistent**:

```text
suffix of R(s,d) from node u == R(u,d)
```

The assessment Appendix A.4 owns the validated counters and regression oracle. Every declared V1
fixture must pass that check before its vectors are written. A failure makes the fixture unsupported;
V1 has no alternate representation, and packet widening does not create or repair routes.

The compact axis relations rely on **row/column route uniformity**:

```text
project_Y(R((x, ys), (x, yd))) is identical for every valid x
project_X(R((xs, y), (xd, y))) is identical for every valid y
```

The supported deployment guarantees these equalities: clusters and rings are homogeneous, every X
column repeats the same Y relation, every Y row repeats the same X relation, and supplied routing
planes share one logical `R`. The codec therefore does not carry an orthogonal-coordinate or plane
selector. ControlPlane / RT-gen must still compare the generated projections before vectors are
written. A mismatch rejects setup; V1 has no richer coordinate/context-indexed fallback for a
nonhomogeneous topology.

### 4.5 Packet fields and intermesh landing (target)

```text
HybridMeshPacketHeader
    route_buffer_y[Y]       # or route_buffer[0..Y)
    route_buffer_x[X]       # or route_buffer[Y..Y+X)
    dst_start_node_id       # unicast: final dest; mcast: final range_anchor; never a temporary exit
    mcast_params_64         # packed N/S/E/W extents only (§5.0)
```

No new packet routing-mode or boundary-direction field is required. A router compares the final mesh
in `dst_start_node_id` with its local `routing_l1_info_t::my_mesh_id`; builder-provided connection
metadata identifies whether the relevant ingress or egress edge has `INTERMESH` capability. Mesh-id
inequality alone says that the final mesh is still remote; edge capability identifies the actual
boundary hop.

The source or an intermediate landing installs unicast-style maps to the temporary exit. At that
exit, the retained `inter_mesh_direction_table` and selected intermesh connection identify the one
boundary egress. A boundary-facing receiver is identified by `INTERMESH` ingress capability and
intercepts the arrival before ordinary indexed-map decode.

**Landing transition:** every boundary arrival is classified as an intermediate- or destination-mesh
landing by comparing the retained final mesh with the landing router's current mesh:

- **Intermediate mesh:** select that mesh's next exit toward the final mesh and install unicast-style
  maps from `encode_root = landing chip` to the temporary exit. Preserve `dst_start_node_id` and
  `mcast_params_64`; do not build a multicast reverse tree or begin target-range fanout.
- **Destination mesh:** with `encode_root = landing chip`, widen to `dst_start_node_id` for unicast;
  for multicast, recover `range_anchor = dst_start_node_id` plus the extents in `mcast_params_64` and
  build the selected local-root multicast maps.

The kernel contract keeps both branches on VC1. Temporary exits are represented by the currently
installed maps and retained intermesh route/connection metadata; they must not overwrite the final
`dst_start_node_id`.

Removed from 2D ABI meaning: `hop_index`, `branch_east_offset`, `branch_west_offset`,
`is_mcast_active`, and line/counter X (§4.2.1).

Indexed 2D needs no packet `execution_state` for Y versus X or intra- versus intermesh state:
facing-aware decode handles the axes, while current/final mesh identity plus edge capability handles
boundaries.

#### 4.5.1 Legacy field retirement boundary

The indexed-map change is an atomic 2D ABI replacement:

```text
legacy producer:
    writes hop_index / branch_east_offset / branch_west_offset
legacy transit:
    route_buffer[hop_index], hop_index++, branch jumps

indexed producer:
    writes complete route_buffer_y/x maps + retained final destination/range fields
indexed transit:
    coordinate-indexed immutable decode; no cursor or branch mutation
```

At universal indexed-2D cutover:

- worker/edge `fabric_set_route`, `encode_2d_unicast`, and multicast setup stop writing
  `hop_index` / `branch_*`;
- ERISC stops reading/updating those fields and removes both RX-side and sender-side header-update
  modes (kernel §3.10);
- event-profiler/debug consumers stop decoding branch offsets;
- indexed 2D no longer assigns routing semantics to the former four-byte
  `LowLatencyMeshRoutingFields`; retaining it as reserved layout space or reclaiming it is an aligned
  header/UDM implementation choice, but no replacement intermesh packet field is added;
- header layout/UDM sizing and host/device typedefs move in the same feature cutover.

The indexed-2D ABI selection is deployment/cutover-wide and applies to **every** 2D producer and
consumer, including non-express configurations. The concrete CT/FabricContext selector must not be
`express_routing_enabled`: that flag controls express topology, Z, and associated BFC/reroot
artifacts, not legacy-vs-indexed packet interpretation.

No compatibility interpretation is defined. A legacy producer paired with an indexed kernel (or the
reverse) is invalid even when the packet happens not to branch. The separate 1D routing fields and
their decrement/shift/refill behavior are unchanged.

### 4.6 Action byte (both axes)

```text
bit 0  E
bit 1  W
bit 2  N
bit 3  S
bit 4  Z
bit 5  LOCAL_DELIVER
bits 6–7  must be 0
```

Widen from 2-bit:

```text
Y: STOP→0, N→bit2, S→bit3, Z→bit4
X: STOP→0, E→bit0, W→bit1
```

`LOCAL_DELIVER` is set at setup on the appropriate axis entry. For mcast, X marks actual target
columns; Y marks a target row only when the encode-root column is also targeted (§5.5). Invalid X
2-bit `11` must not appear in L1; reject at generation.

At runtime, reserved bits, an ungated self-facing bit, or an all-zero decoded action with no selected
destination follows the kernel contract's assert-and-fail-stop policy. There is no packet recovery
or retransmission interpretation for invalid state.

### 4.7 Header / route_buffer capacity (target)

#### Dual axis maps

```text
RouteBufferSize = Y + X     # action bytes; both axes
```

vs today’s hop sizing:

```text
max_hops ≈ (Y - 1) + (X - 1) = Y + X - 2
delta    = +2 bytes on the route payload
```

Control-word savings (`hop_index` + `branch_*`) offset that in the non-route header.

Fixture examples (`X = 4`):

```text
[8,4]:  8+4  = 12  → packed ~72 → align16 → 80B header
[16,4]: 16+4 = 20  → ~80 → 80B
[24,4]: 24+4 = 28  → ~88 → 96B
[32,4]: 32+4 = 36  → ~96 → 96B
[64,4]: 64+4 = 68  → ~128 → 128B
```

(Assumes ~60 B non-route after dropping `is_mcast_active`; exact packing may use a single
`route_buffer[Y+X]` array. UDM recalculated separately.)

`[64,4]` is a target compatibility calculation. Today
`HybridMeshPacketHeaderT<RouteBufferSize>` statically rejects `RouteBufferSize > 67`, while current
2D ControlPlane validation rejects an axis above 32. Enabling this shape therefore requires both a
memory-map/header update for the 68-byte payload and removal/validation of the 32-axis limit; the
1028-byte L1 table fit alone does not establish support.

Primary Galaxy `[32,4]` → **96 B** tier with a **36-byte** route payload (vs today’s 35-byte hop
tier max in that header size — bump the template size to 36; still one 96 B header).

#### 80 B tier: perf fallback to 96 B

Current `FabricContext` **disables** the 80 B / 19-byte route-buffer tier (8×4 Mesh instability).
If a shape’s `Y+X` fits an 80 B header but 80 B still tanks perf, **fall back to 96 B** and zero-pad.
Correctness only needs capacity ≥ `Y+X`.

---

## 5. Target 2D multicast

The assessment’s “Multicast semantics and safety” owns target/tree legality; encode/decode ABI and
the primary reverse-tree device artifact are owned here.

### 5.0 `range_anchor` vs `encode_root` (required)

These are different chips and must not be conflated:

| Concept | Meaning | Typical binding |
|---|---|---|
| **`range_anchor`** | Chip the client N/S/E/W **extents are relative to** | Often the worker’s chip; retained in `dst_start_node_id` for intermesh landing |
| **`encode_root`** | Chip where **path tracing begins** for the action maps | Same-mesh worker send: source chip. After dest-mesh **landing**: the landing chip. Intermesh carrier to exit: source (unicast-style to exit). |

Targets:

```text
target_ys =
    N/S offsets from range_anchor_y when N_extent + S_extent > 0
    {range_anchor_y} when N_extent = S_extent = 0

target_xs = {range_anchor_x}
             union E/W offsets from range_anchor_x

target_chips = target_ys × target_xs
```

Maps:

```text
trace/OR from encode_root → each target in the rectangle (Y paths; X teeth)
dst_start_node_id         → final dest (unicast) or final range_anchor (mcast); retained end-to-end
mcast_params_64           → N/S/E/W extents only; retained across intermesh until landing rebuild
temporary exit            → encoded by current maps; boundary egress comes from retained intermesh
                            route/connection metadata; not stored in dst_start_node_id
```

### 5.1 Client contract (preserved)

Client N/S/E/W extents remain **geometric cardinal ranges**, not ring steps, relative to
**`range_anchor`**.

Example: `range_anchor` Y=1, client `S=4` → target rows `{2,3,4,5}` (diagram: S increases logical Y).

E/W teeth remain cardinal (no E/W express in the initial cut; see the assessment’s “Alternative and
extension decision record”).

The working **2D** API has no independent `start_distance` field: N/S and E/W values are extents
from `range_anchor`. `MulticastRoutingCommandHeader::start_distance_in_hops` belongs to the separate
1D low-latency codec. Adding 2D start-distance semantics would require an explicit target-set and
packet-field extension; it is not implicit in this contract.

### 5.2 Why “cardinal-only mcast execution” is unsafe

The assessment's “Multicast semantics and safety” owns the safety argument. The codec requirement is
that multicast Y progress follows the union of constrained unicast routes from `encode_root`, with Z
available in the tree; a separate cardinal spine is not valid.

### 5.3 Selected strategy

Packets carry one Z-aware source-reachable Y/X action map (§5.6) and use same-header fanout (§5.9).
The primary encoder is the per-chip reverse tree in §5.7.1. It is used only when every root passes
the mesh-wide gate. If any root fails, the mesh/configuration is unsupported by this design. V1 has
no alternate multicast code path; selecting another representation requires a separate design pass.

### 5.4 Baseline multicast (today) — for contrast

Today `fabric_set_mcast_route` builds a serialized N/S-spine plus E/W-branch hop program using
`hop_index` and `branch_*_offset`; Z is not a first-class output. §2 records the shared baseline
structures, and the kernel contract owns the current execution details.

### 5.5 Target Y action map

```text
route_buffer_y[local_y] = action byte (E|W|N|S|Z|LOCAL_DELIVER)
```

Sizes: `Y` bytes → 8 / 16 / 24 / 32 / 64 for the fixtures / bound above.

Example `encode_root=1, S=4` (unicast routes to {2,3,4,5}):

```text
tree: 1→2 ├─S→3
           └─Z→5→N→4

action_y[1] = S
action_y[2] = LOCAL_DELIVER | S | Z
action_y[3] = LOCAL_DELIVER
action_y[4] = LOCAL_DELIVER
action_y[5] = LOCAL_DELIVER | N
```

Add the required X-root E/W bits on every **rectangle target** row (fanout signal at that row).
Transit-only rows may forward without `LOCAL_DELIVER`.

More precisely, first build `route_buffer_x` from `encode_root_x` to `target_xs`. At every
`target_y`, copy the X-root E/W outputs into the Y action. Set Y `LOCAL_DELIVER` only when
`encode_root_x ∈ target_xs`; X `LOCAL_DELIVER` remains set on every actual target column:

```text
x_root_action = route_buffer_x[encode_root_x]
for target_y in target_ys:
    route_buffer_y[target_y] |= x_root_action & (E | W)
    if x_root_action & LOCAL_DELIVER:
        route_buffer_y[target_y] |= LOCAL_DELIVER
```

This rule also handles destination-mesh landing where `encode_root_x != range_anchor_x`. Even when
`E_extent = W_extent = 0`, X must route from the landing column to the anchor column when they
differ. “Teeth requested” therefore means `target_xs` requires an X path from `encode_root_x`, not
merely that the client supplied a nonzero E/W extent.

Terminal ex4→ex8: landing entry has **no** Y child output. Its local/X-root bits follow the §5.5
target-column rule.

### 5.6 Mcast semantics — source-reachable OR

Persistent 2-bit enums **must not** be OR’d directly (`N|S` looks like `Z`). Widen to one-hot
action bytes first, then OR.

**Unicast** may `widen(y_vectors[dst_y])` in full.

**Multicast must not** OR full destination vectors. A dest vector holds actions from **every** row,
including rows not on this tree from `encode_root`; OR‑ing those in can select unrelated outputs.
Suffix consistency does **not** equate full-vector union with “trace from encode_root.”

**Required result** for the reverse-tree encoder:

```text
route_buffer_y = one-hot union of projected Y hops from encode_root_y to each target_y
               + X-root E/W outputs on each target_y
               + LOCAL_DELIVER on target_y iff encode_root_x is a target column

route_buffer_x = one-hot union of X hops from encode_root_x to each target_x
               + LOCAL_DELIVER on each target_x
```

**Vector path-trace** is the golden host-side validation reference. It is not a V1 fallback encoder:

```text
clear route_buffer_y[Y]
y0 = encode_root_y
for each target_y in target_ys:
    y = y0
    while y != target_y:
        a2 = extract_2bit(y_vectors[target_y], y)
        assert a2 != STOP
        route_buffer_y[y] |= one_hot(a2)
        y = step_y(y, a2)   # N/S: ±1; Z: express_neighbor_y[y] (must be available if used)

build route_buffer_x from encode_root_x to target_xs
apply X-root E/W and conditional LOCAL_DELIVER to each target_y (§5.5)
```

`step_y(..., Z)` needs express-neighbor identity in the validation model. The production reverse
tree (§5.7.1) never forward-follows Z.

Example fork (`encode_root` Y=1 → targets {3,4}): `route_buffer_y[2] = S | Z`.

### 5.7 Device mcast encoders

V1 uses the reverse-tree representation below for every encode-capable chip in a
mesh/configuration. If it is unavailable for any root, reject that mesh/configuration. A future
alternate representation requires its own artifact, execution, and validation contract.

#### 5.7.1 Primary device mcast encoder: source-rooted reverse tree

**Arborescence (here):** a rooted directed tree over the Y (resp. X) rows: the `encode_root` has
indegree 0; every other row has exactly one parent; there are no cycles; every row is reachable
from the root. Equivalently: exactly one path from root to each destination, so
`T(root) = union_dst R(root, dst)` has `Y-1` edges and matches the canonical route set.

When routes from one local `encode_root` to every destination form such an arborescence, provide a
**per-chip** reverse-edge list — a representation of the same canonical routes, not a new policy:

```text
T(root) = union over every dst of R(root, dst)
```

**Mesh-wide arborescence gate.** Host generation must validate the checks below for **every**
root chip that will ship a reverse tree (under the initial model: one canonical, row/column-uniform
all-pairs relation `R` per mesh). Under the V1 one-encoder policy, if **any** root fails, reject
that mesh/routing configuration rather than selecting another encoder.

Generation must reject reverse-tree unless, for each root:

```text
indegree(root) = 0
indegree(other Y rows) = 1
edge count = Y - 1
all Y rows are reachable
T(root) is acyclic
path in T(root) from root to dst == canonical R(root, dst), for every dst
```

##### Host reverse-edge generator (sketch)

Same pass that builds dest-major vectors. Y shown; X is analogous with E/W only (no Z in the
initial cut).

```text
# Inputs (host): finalized, annotated canonical R(src,dst) from routing generation;
#                shape + logical (x,y); edge commands include Z when express is on.

for each root in 0..Y-1:                          # mesh-wide gate: every root
    parent[y] = unset
    edges = []
    for each dst in 0..Y-1, dst != root:
        for each hop (p --cmd--> c) on R(root, dst):   # chip sequence → directions
            if parent[c] set and parent[c] != (p, cmd):
                FAIL  # not an arborescence
            parent[c] = (p, cmd)
            record edge (child=c, parent=p, parent_output=cmd)
    assert parent[root] unset
    assert every y != root has exactly one parent
    assert |unique edges| == Y - 1
    assert acyclic + all reachable
    assert path in T(root) to each dst == R(root, dst)   # ≡-trace
    order edges descendants-before-ancestors             # host packing assert
    embed T(root) on the chip whose encode_root_y == root
```

Derive the tree directly from finalized logical `R(src,dst)` records. If generation is separate from
RT-gen, expose the equivalent complete ordered logical route view. Current channel-conditioned
`get_fabric_route` and one-direction helpers are not that relation. The supported row/column-uniform
route input is defined in §4.4; builder domain effects are not codec inputs.

Under the initial homogeneous-cluster and homogeneous-ring model there is **one** `T(root)` per chip for
the mesh’s `R`. Multiple concurrent route relations (for example plane-specific `R`) are out of
scope; they would require separate tree/vector sets, packet context identity, and gates.

Serialize edges in **descendants-before-ancestors** order. **Host packing must assert this order**
(wrong order breaks `needed` propagation). At the `Y <= 64` bound one illustrative packed descriptor:

```text
bits  0..5   child_y
bits  6..11  parent_y
bits 12..13  parent_output  # N=01, S=10, Z=11; command issued by parent
bits 14..15  reserved = 0
```

The worker does not dynamically grow a list and does not need STL, a heap, a per-row pending-mask
array, or a Z-neighbor lookup. It uses one or two fixed `uint32_t` bitmaps:

```text
target_bits[ceil(Y/32)] = cardinal targets
needed = target_bits
clear route_buffer_y[Y]

for edge in reverse_tree_edges:          # fixed Y-1 loop; leaves toward root
    if test_bit(needed, edge.child_y):
        route_buffer_y[edge.parent_y] |= one_hot(edge.parent_output)
        set_bit(needed, edge.parent_y)

# Build/prune X separately. Then apply X-root E/W outputs and conditional local delivery
# to each original target_y as specified by §5.5. Never derive LOCAL_DELIVER from expanded needed.
```

The reverse scan selects an edge exactly when its child subtree contains a requested target.
Consequently it emits `union(R(encode_root,target))`. The descriptor contains both endpoint indices
and the parent command, so a Z edge never has to be followed forward during encoding.

Simple 16-bit edge-list sizes, including the independent four-column X tree, are:

```text
bytes = 2 * (Y - 1) + 2 * (X - 1)

[8,4]    14 + 6 =  20 B
[16,4]   30 + 6 =  36 B
[24,4]   46 + 6 =  52 B
[32,4]   62 + 6 =  68 B
[64,4]  126 + 6 = 132 B
```

The root tree is **per chip**, unlike destination-major unicast vectors:

| Artifact | Same bytes on every chip? | Why |
|---|---|---|
| Dest-major `y_vectors` / `x_vectors` (§4.1) | **Yes** (mesh-identical relation) | Indexed by destination; any chip widens the same tables |
| Reverse-tree edge list (§5.7.1) | **No** — one table **per root chip** | `T(root)` is rooted at that chip’s `encode_root`; chip A’s postordered edges ≠ chip B’s |

A same-mesh worker embeds **only** `T(my_chip)`; a destination-mesh landing embeds **only**
`T(landing_chip)`. Host generation therefore produces **N_chips** distinct Y+X reverse-tree blobs
(plus shared dest-major vectors for unicast), not one mcast table replicated everywhere. All roots
must pass the mesh-wide gate before reverse-tree is enabled for any chip.

The primary hybrid stores destination-major vectors plus the local postordered reverse tree. §6.2
owns its placement options and aggregate L1 sizes.

Device setup cost, scratch use, and alignment must be measured without introducing another packet
ABI or transit decoder.

### 5.8 Cardinal-range → target rows

Pinned logical layout (row-major `[Y,X]`), relative to **`range_anchor`**. Y target construction is
topology-aware:

```text
logical_chip_id = y * X + x

target_ys = {}

for k in 1..N_extent:
    y = range_anchor_y - k
    if Y is configured as TORUS: y = (y + Y) mod Y
    else: require 0 <= y < Y
    target_ys.insert(y)

for k in 1..S_extent:
    y = range_anchor_y + k
    if Y is configured as TORUS: y = y mod Y
    else: require 0 <= y < Y
    target_ys.insert(y)

if N_extent == 0 && S_extent == 0:
    target_ys = { range_anchor_y }
```

A range names **all** intermediate rows, not only its endpoint. Simultaneous N and S ranges form a
set union; duplicate rows are encoded once. On a non-torus Y dimension, an extent crossing an end is
invalid and must not wrap implicitly. Power-of-two torus Y may use a mask (`& 31` / `& 63`).

The initial contract fixes X as the ordinary four-chip ring. `target_xs` is explicitly enumerated as
`{range_anchor_x}` plus the modulo-X E/W offsets, with set-union deduplication. A future X-LINE
configuration would require the same bounds rule as non-torus Y.

### 5.9 E/W teeth on X (shared header)

Same **source-reachable** requirement as Y. The production path uses the **per-chip X reverse tree**
(§5.7.1), under the same mesh-wide gate as Y. The trace below is the golden validation reference
(cardinal E/W only in the initial cut — no Z step):

```text
# golden reference (X has no express in initial cut)
clear route_buffer_x[X]
for each target_x in target_xs:
    trace encode_root_x → target_x via x_vectors[target_x]
    OR one-hot E/W along that path only
    route_buffer_x[target_x] |= LOCAL_DELIVER
```

After building X, apply `route_buffer_x[encode_root_x] & (E|W|LOCAL_DELIVER)` to every target-row Y
entry as defined in §5.5. This launches the same X subtree independently from each selected Y row
without incorrectly delivering at a non-target landing column. Unicast X remains full
`widen(x_vectors[dst_x])` + `LOCAL_DELIVER` at `dst_x`. Maps live in the **shared** header; transit
does not rewrite them per E/W output.

At a **transit RX** multi-output Y action that includes E and/or W (often with
`LOCAL_DELIVER`):

1. Atomic admission across local deliver + all selected outputs.
2. Fanout = NOC-copy the **same** header to each sender.
3. Decode (§4.3): N/S/Z use Y if `action_y != 0`; E/W facing use X only.
4. Never hop cursors / `branch_*`.

Invariant (§5.12): E/W children never return to N/S/Z.

This is not the same-mesh source-chip injection policy: a worker send bypasses source RX. Source
multi-output is handled by §7.3.1.

### 5.10 Encoder lifecycle (mcast + intermesh)

```text
Worker / same-mesh mcast:
    range_anchor = src; encode_root = src
    reverse-prune T(src) on Y and X (§5.7.1)   # primary; mesh-wide gate assumed
    apply §7.3.1 source inject policy

Worker mcast with foreign final mesh:
    encode_root = src; dst_start_node_id = final range_anchor (never overwrite)
    unicast-style maps toward temporary exit; keep mcast_params_64 extents
    at exit: current mesh != final mesh; selected INTERMESH egress executes one boundary hop

On intermediate-mesh landing:
    encode_root = landing
    select the next temporary exit toward the retained final mesh
    install unicast-style maps toward that exit
    preserve dst_start_node_id and mcast_params_64
    do not run T(landing) or multicast fanout

On destination-mesh landing:
    encode_root = landing; range_anchor = dst_start_node_id
    rebuild via T(landing) using retained extents
    continue through landing RX fanout; do not apply worker source reroot

Worker / same-mesh unicast:
    full widen dest-major vectors (§7.1); or single-hop poke (§7.1.1)

Intramesh express Z:
    Z bit in route_buffer_y; INTRAMESH_EXPRESS capability, never INTERMESH capability
```

If the reverse-tree gate fails for any root, this lifecycle is unsupported for that
mesh/routing configuration. V1 does not replace `T(...)` with another encoder.

### 5.11 Validation scope

The assessment Appendix A owns the route and indexed-map regression counters. A local host-side
`[32,4]` reverse-tree check additionally covered every root, matched each canonical source trace,
and reproduced the legal-range execution result recorded in assessment Appendix A.3.

Those results are not production C++ or device evidence and do not cover other topology shapes,
packing, multiple planes, reroot execution, intermediate landing, or device performance. Every
deployed mesh/configuration still applies the §5.7.1 all-roots gate.

### 5.12 Multicast legality (must hold for encoded trees)

The assessment's “Multicast semantics and safety” owns route/tree legality; the builder contract owns
per-output `IS_INJECTION`. Codec-specific requirements are:

1. Every reverse-tree root passes §5.7.1, including exact canonical paths and
   descendants-before-ancestors serialization. Any root failure makes that representation
   unavailable for the mesh/configuration.
2. The optional §7.3.1 reroot overlay is applied only after the canonical §5.6 map passes validation.
   Complete direct-inject plus return execution remains a separate implementation check (§11.2).

---

## 6. Host generator / L1 embed / device loader (target contract)

### 6.1 What host must provide (device-facing)

Minimum device-visible artifacts after generation validation:

1. **Destination-major Y and X 2-bit tables** (sizes §4.1), indexed by **logical** coordinates under
   the MGD pinning contract.
2. A **deployment/cutover-wide indexed-2D ABI selection** shared by all 2D producers and consumers,
   including non-express configurations. This is not selected by `express_routing_enabled`.
3. **`express_routing_enabled`** (or equivalent) for express topology, materialized Z neighbors,
   express/BFC artifacts, and reroot eligibility (builder contract §4.1); it must not select old vs
   new packet ABI.
4. **Local chip’s `(my_x, my_y)`** already implied by `my_device_id` + pinned shape; router setup
   caches both for DOR indexing.
5. **Intermesh L1 (kept as-is):** `exit_node_table` + `inter_mesh_direction_table` for rewrite-to-exit
   and boundary/first-hop toward a foreign mesh. See §2.11.
6. **`intra_mesh_direction_table`:** not required by the indexed codec; removable after cutover (§2.11).
7. For express mcast under the primary approach: **per-chip** local-root reverse trees (§5.7.1)
   after the **mesh-wide** arborescence gate. Built on host from the same express-aware `R` as
   dest-major vectors (§5.7.1 host sketch), preferably in the same RT-gen pass over finalized
   canonical routes. If generated by a separate consumer, expose a generic complete canonical-route
   view; current channel-conditioned `get_fabric_route` is not that contract. Dest-major unicast
   vectors are mesh-identical; reverse-tree tables are **not** — each encode-capable chip gets
   `T(that chip)` only. L1 field placement is a host↔device embed contract. Any root failure rejects
   reverse-tree support for the complete mesh/configuration; V1 has no alternate mcast artifact.
8. Source-connection direction metadata for §7.3.1: a `RoutingPlaneConnectionManager` slot tag is
   its `eth_chan_directions` identity; a raw `WorkerToFabricEdmSender` retains/exposes the
   `edm_direction` already obtained from its connection-table entry. This is fabric connection
   metadata, not a new packet field or a value inferred from N/S/E/W multicast extents.

### 6.2 What replaces `compressed_route_2d_t` in the 2D union

Today the 1024B union slot holds `compressed_route_2d_t[256]`.

**Required for this ABI:** every chip that encodes (worker / edge landing) must provide the **full**
destination-major Y and X 2-bit tables (or an equivalent bitplane twin that can answer the same
extracts). Packet unicast widen and mcast path-trace both need `y_vectors[dst][*]` /
`x_vectors[dst][*]` complete for selected destinations — not “only hops from this source.”

| Layout | Fit? | Notes |
|---|---|---|
| Full mesh dest-major Y×Y + X×X on every encode-capable chip | `[32,4]` 260 B — OK; `[64,4]` **1028 B L1 fit** (§2.12) | +4 B over today’s 1024 B union via padding and/or intra-table reclaim; does not close current header/axis limits |
| Hybrid: destination-major vectors + local-root reverse tree | `[32,4]` 328 B — fits; `[64,4]` 1160 B — requires L1 layout growth or external placement | Primary express layout. After reclaiming the 96 B intra table, in-struct `[64,4]` placement is `420 + 1160 + 1024 = 2604 B`, rounded to **2608 B** for 16-byte alignment; grow the current 2576 B `routing_l1_info_t` by **32 B** |
| Shared read-only region + per-chip pointer | Implementation option | Must still expose **complete** destination vectors |

For `[64,4]`, the hybrid payload is `1028 + 132 = 1160 B`. The 2608-byte minimum assumes
`intra_mesh_direction_table` is removed at indexed cutover. If that 96-byte table is retained, the
in-struct minimum is `516 + 1160 + 1024 = 2700 B`, rounded to **2704 B**, which grows the current
struct by **128 B**.

Removed option: “only vectors from this source.” That fits today’s per-src compressed routes, not
destination-major widen / any-chip landing encode under this contract.

Transit routers need **neither** persistent table on the hot path if the packet carries both maps.

### 6.3 Loader responsibilities

| Actor | Unicast | Multicast |
|---|---|---|
| Tensix worker | Full unicast widen (§7.1). Single-hop poke (§7.1.1) | Reverse-prune `T(src)` (§5.7.1); then source multi-inject / reroot policy (§7.3.1). Set anchor in `dst_start_node_id`; fill extent-only `mcast_params_64` |
| Edge / landing encoder | Intermediate landing: widen from landing to the next local exit. Destination landing: widen to final `dst_start_node_id` | Intermediate landing: use the same unicast-style next-exit segment and preserve final anchor/extents. Destination landing only: build `T(landing)`; packet is already on RX, so ordinary atomic RX fanout handles the final multi-output landing root |
| Transit ERISC | No reload; decode §4.3 | Maps immutable; same-header fanout |

2-bit extract/shift happens at **setup**, not on the transit hot path. Worker vs router **prepend**
semantics of today’s `encode_2d_unicast` go away: local `route_buffer_y[local_y]` /
`route_buffer_x[local_x]` replace hop-0 accounting.

### 6.4 Intermesh vs express Z (loader rule)

```text
capability(edge) == INTRAMESH_EXPRESS  →  Z bit in route_buffer_y; stay on indexed DOR
capability(edge) == INTERMESH          →  boundary egress/landing handling; any direction letter
```

The boundary path additionally requires `current_mesh_id != final_dst_mesh_id` at egress. At landing,
the `INTERMESH` ingress capability triggers map installation before ordinary decode; current versus
final mesh identity chooses intermediate or destination behavior. **Forbidden:** `NOOP` means Z;
`direction == Z` alone means intermesh.

---

## 7. Encoder / decoder contract (target)

### 7.1 Unicast encoder (worker/edge)

Inputs: `dst` logical chip, local mesh shape, L1 Y/X 2-bit tables, and the deployment-selected
indexed-2D ABI. `express_routing_enabled` is additionally consulted only to validate/use express Z
topology and related artifacts; it does not select the packet ABI.

Outputs:

1. `route_buffer_y[0..Y)` = `widen(y_vectors[dst_y])`
2. `route_buffer_x[0..X)` = `widen(x_vectors[dst_x])` with `LOCAL_DELIVER` at `dst_x`
3. final `dst_start_node_id`, plus cleared mcast params unless this is a foreign-mesh multicast
   carrier

Must validate dst in-range; self-route → local deliver without fabric send (or minimal no-fabric path
as today). No intermesh mode/direction field is added (§4.5). Encode-time assert: ≤1 of N/S/Z per Y
row; ≤1 of E/W per X column.

### 7.1.1 Single-hop unicast helpers (target)

Replaces today’s `fabric_set_single_hop_unicast_route` /
`fabric_set_single_hop_unicast_route_from_direction`. Same client contract shape (dest is one fabric
hop away), but the encode is destination-indexed — **not** a 1-deep hop program.

Because transit only reads **this chip’s** map entry, the helper writes that entry directly:

```text
# clear both maps (or zero the slots touched)
# my_x, my_y from pinned logical id; dst_x, dst_y from dst_dev_id

if next_hop in {N, S, Z}:
    route_buffer_y[my_y]  = one_hot(next_hop)     # real bits — no opposite-direction table
    route_buffer_y[dst_y] = LOCAL_DELIVER
    # route_buffer_x stays STOP / 0 (same X)
elif next_hop in {E, W}:
    route_buffer_x[my_x]  = one_hot(next_hop)
    route_buffer_x[dst_x] = LOCAL_DELIVER
    # route_buffer_y stays STOP / 0 (same Y)
else:
    assert false

dst_start_node_id = …; mcast_params_64 = 0
# no hop_index / branch_* / route_buffer[0] hop program
```

| API | Target behavior |
|---|---|
| `…_from_direction(header, dir, dst, mesh)` | No L1 required for the direction; poke maps as above. **Z allowed** when `dir == Z` and the neighbor is intramesh express (not intermesh). |
| `…_unicast_route(header, dst, mesh)` | Resolve `dir` from first hop of L1 vectors (`y_vectors`/`x_vectors` peek at my coords) or, until removed, `intra_mesh_direction_table` / inter table — then same poke. |

Vs baseline:

| | Today | Target |
|---|---|---|
| Write site | `route_buffer[0]` + `hop_index == 0` | `route_buffer_*[my_coord]` + dest `LOCAL_DELIVER` |
| Direction encoding | opposite EWNS (`single_hop_route_cmd_by_direction`) | §4.6 one-hot action bits |
| Z | rejected / `NOOP` | Z bit on `route_buffer_y[my_y]` when express |
| Control word | clear hop/branch fields | no routing cursor fields |

`LOCAL_DELIVER` on the dest slot replaces today’s “opposite bit = local write” on the next chip.
Full unicast widen (§7.1) remains correct for single-hop too; this helper is the cheap path when the
caller already knows (or looks up) one next hop.

### 7.2 Intramesh decoder (router) — unicast and mcast

Inputs: `my_direction`, `route_buffer_y[local_y]`, `route_buffer_x[local_x]` (§4.3).

Outputs: action byte mask (§4.6). N/S/Z: use Y if `action_y != 0`, else X. E/W: X only.

### 7.3 Multicast encoder

Inputs: `encode_root`, `range_anchor`, N/S/E/W extents; artifact = per-chip reverse tree (§5.7.1)
under the mesh-wide arborescence gate. Any root failure rejects this encoder for the complete
mesh/configuration.

Outputs: `route_buffer_y` / `route_buffer_x` satisfying §5.6 semantics; `mcast_params_64` with
N/S/E/W extents, while `dst_start_node_id` retains `range_anchor`. Intermesh carriers preserve both
until landing.

### 7.3.1 Same-mesh 2D express source multi-output injection

#### Problem and scope

The canonical §5.6 map may contain several eth outputs at `encode_root`. Transit routers can clone
atomically, but a same-mesh worker injects directly into sender channel 0 on one router and bypasses
source RX. A raw `WorkerToFabricEdmSender` may expose only one connection, so “build the map” does
not by itself launch every root-child subtree.

This section applies only to **same-mesh worker-originated 2D indexed express multicast**. It does
not apply to:

- 1D low-latency or sparse multicast (no express routing in those paths);
- the intermesh carrier from source to exit, which remains unicast-style; or
- destination-mesh landing, where the packet is already on an RX path and normal atomic RX fanout
  executes the rebuilt landing-root action.

For a same-mesh source, `encode_root == range_anchor`. The current rectangular API therefore cannot
produce a mixed Y+X root action:

- when `N_extent + S_extent > 0`, `target_ys` contains only N/S offsets and excludes the source row;
  §5.5 copies E/W teeth only to those target rows, so the source outputs are a subset of N/S/Z;
- when `N_extent = S_extent = 0`, the operation is X-only and stays on the existing E/W multicast
  path. It does not use the express source-reroot fallback.

This section's new source-fanout handling is consequently for multi-output **Y roots** introduced by
express routing. Future E/W express links or a generalized target-set API cross the assessment’s
“Alternative and extension decision record” boundary and require a separate extension.

First produce the unchanged canonical map by §5.6 / §5.7.1. Then select one of the following
source-injection realizations. `encode_root` remains the source in both; “reroot” names the transport
overlay, not a change to the codec root or canonical route relation.

These realizations require disjoint canonical root-child subtrees, guaranteed when the §5.7.1
arborescence gate passes. A failed gate rejects source multi-inject/reroot together with the
reverse-tree representation.

#### Preferred: connection-manager source multi-inject

If `RoutingPlaneConnectionManager` has one open slot for every set eth bit in the canonical root
action:

1. Resolve slots by direction tag.
2. Make one packet/header copy per canonical root output; every copy carries identical canonical
   maps.
3. Inject each copy directly into its matching root-child edge. The source action is not consumed.
4. Deliver locally at the source exactly once in the worker when `LOCAL_DELIVER` is set.

The canonical reverse tree has disjoint root-child subtrees, so each injected copy can reach only
its selected subtree. Source multi-inject is the worker analogue of same-header fanout, not a set of
different route programs. Header/payload lifetime must cover every send; a header may not be reused
while another source copy is in flight.

#### Temporary fallback: pre-inject reroot overlay

If not every root output has a connection, select one connected canonical root output:

```text
canonical_root_action = route_buffer_y[encode_root_y]
inject_dir             = actual connection direction

assert popcount(canonical_root_action & ETH_BITS) > 1
assert inject_dir in {N, S, Z}
assert canonical_root_action & one_hot(inject_dir)

selected_edge = the canonical reverse-tree root edge whose output == inject_dir
child         = selected_edge.child
return_dir    = reverse_on_same_link(inject_dir)  # N↔S; Z uses self-facing Z

assert !(action[child] & one_hot(return_dir))
action[encode_root] &= ~one_hot(inject_dir)
action[child]       |=  one_hot(return_dir)
```

`action[...]` above is always `route_buffer_y[...]`. The selected reverse-tree edge supplies the
child coordinate, including Z, so the worker does not need a separate express-neighbor step table.

For Z, the bit value remains Z in both directions; it means same-link return only because the
first-hop receiver has `MY_DIR == Z` and the reroot kernel gate is enabled. It is not an ordinary
second forward-Z step.

Execution is:

1. The worker injects the original packet directly on `inject_dir`; source RX and the edited source
   action are bypassed.
2. At `child`, normal outputs/local delivery execute and a same-header copy takes the explicit
   same-link return output.
3. The returning copy reaches `encode_root`, where the edited source action launches every
   remaining canonical root-child subtree but cannot relaunch the selected one.
4. Canonical subtree disjointness prevents a remaining subtree from reaching `child` and requesting
   a second return.

The overlay is applied **once, before injection**. The resulting packet maps are immutable in
transit. The overlay map by itself is intentionally not bit-identical to §5.6: it removes one
canonical source bit and adds one non-canonical return bit. Its required equivalence is
**execution-level**:

```text
initial direct inject enters selected canonical subtree once
+ returned copy enters every remaining canonical subtree once
= canonical target set, with no duplicate delivery
```

The return output is an infrastructure U-turn, not permission for route generation to use arbitrary
canonical U-turns. Kernel handling and builder wiring/BFC are owned by the kernel §3.9 and builder
§3.7 contracts.

The dedicated U-turn queue is a full-packet **producer boundary**. Child RX waits only for local
queue capacity, commits the complete immutable copy, and releases its RX before the U-turn sender
waits for remote credit. That sender then applies the same-output worker `IS_INJECTION` rule
independently. RX's local-queue wait remains the existing atomic-fanout premise, while the incoming
receiver is not held during the reverse sender's remote-credit wait. Under these invariants the
assessment's existing VC0 proof remains applicable. Claiming reroot support requires differential
execution and concrete queue-boundary conformance, not a new reroot CDG/BFC proof.

`inject_dir` comes from connection metadata:

- `RoutingPlaneConnectionManager::ConnectionSlot::tag`, or
- retained `WorkerToFabricEdmSender.edm_direction`.

Do not infer it from `mcast_params_64`, N/S/E/W extents, or baseline `if (n) else if (s)` spine
selection. Reject the fallback if the actual connection direction is not a canonical root output.
If the canonical root has zero eth outputs, deliver locally only; if it has one, inject directly
without the overlay through the matching direction connection; reject if that connection is absent.

#### Source local delivery

Source `LOCAL_DELIVER` must execute exactly once:

- direct single-output or manager multi-inject: worker performs source-local delivery because the
  source action is bypassed;
- reroot overlay: worker suppresses source-local delivery and leaves `LOCAL_DELIVER` in the edited
  source action so the returning copy performs it.

All non-source `LOCAL_DELIVER` bits remain unchanged.

#### Worked `S|Z` root

For canonical `action_y[2] = S|Z`, a worker connected through S selects edge `2→3`:

```text
before: action_y[2] = S|Z
        action_y[3] = LOCAL_DELIVER

overlay:
        action_y[2] = Z
        action_y[3] = LOCAL_DELIVER|N

execute:
        worker --S--> 3
        3 delivers locally and returns one copy --N--> 2
        2 --Z--> the untouched Z subtree
```

The same rule supports a Z-connected sender (`Z↔Z`, interpreted as a self-facing Z return) when the
builder supplies the dedicated same-link return sender.

### 7.4 Intermesh decoder

A boundary-facing receiver is identified by builder-provided `INTERMESH` ingress capability. It
intercepts the arrival before ordinary indexed decode, compares retained final mesh with current
mesh, and installs the intermediate- or destination-landing maps from §4.5 / §5.10. No packet
intermesh state or line-state machine (§4.2.1) is used.

### 7.5 What decoders must not do

- Infer `IS_INJECTION` / BFC threshold from command letter or action bit.
- Treat NOOP as Z or as “recompute”.
- Read persistent destination vectors on transit hops.
- Allow E/W-facing routers to index `route_buffer_y`.
- Use `action_y & (N|S|Z)` as the Y-phase test (drops `LOCAL_DELIVER|E|W`; use `action_y != 0`).
- OR full destination vectors for mcast (use §5.6 semantics; primary device path §5.7.1).
- Under the V1 single-artifact policy, mix reverse-tree and bitplane/vector-trace artifacts within one
  mesh/routing configuration.
- Use rejected X / packed-packet paths (§4.2.1).
- Rewrite maps per E/W fanout branch on transit.
- Infer same-mesh source injection direction from multicast extents or add it to the packet; use
  connection metadata (§7.3.1).
- Apply the worker reroot overlay at intermesh landing; landing is already an RX fanout point.

---

## 8. Kernel boundary

This document owns the bytes the kernel consumes: decode §4.3, retained intermesh destination/range
fields §4.5, action bits §4.6, immutable maps, and explicit `LOCAL_DELIVER`. The kernel contract owns
admission, fanout, capability-aware boundary handling, self-bit handling, sender selection, BFC,
NOOP retirement, and speedy/trim policy.

Indexed transit performs no `hop_index`, branch-offset, line-counter, or per-branch header mutation.
It also performs no persistent L1 vector lookup or two-bit extraction. Any kernel implementation must
preserve those codec-visible properties.

---

## 9. Cross-dependency matrix

| Fact / artifact | RT gen | CP expose/embed | Builder | Codec/loader (this doc) | Kernel |
|---|---|---|---|---|---|
| Canonical physical routes | own (Z-aware when express) | co-located derivation or generic complete canonical-route view | trust | encode from vectors / trees | trust packet |
| Y/X 2-bit tables (L1) | own | L1 embed | — | widen at setup | — (transit) |
| Packet `route_buffer_y` / `_x` | — | — | — | unicast widen / mcast source-reachable OR | decode per codec §4.3 |
| Suffix consistency | validate | fail closed | — | assume valid | — |
| Row/column route uniformity | mandatory generated-projection check before L1 write | fail closed; preserve one logical relation | — | required by axis vectors/trees | — |
| Ring derivation / protected-ring predicates | own internal state | expose builder §4 predicates | derive IS_INJECTION | — | use flag only |
| Express neighbor as Z | materialize | neighbors + flag (existing APIs) | same-VC N/S↔Z wiring on VC0/VC1 | Z bit in `route_buffer_y` | forward Z bit |
| Intermesh capability | classify | edge meta + retained exit/direction tables | VC1 boundary template | preserve final mesh; install maps to temporary exit | select boundary egress; intercept capability-tagged landing |
| Header route_buffer size | footprint | FabricContext policy | channel buffers | fill (`≥ Y+X`) | read |
| Per-chip reverse tree `T(root)` | build from finalized `R` + all-roots gate (§5.7.1) | L1 embed; generic route view only if generation is separate | — | reverse-prune mcast | — |
| Multicast Y + X maps | validate trees mesh-wide | embed with vectors | fanout wiring | §5.7.1 primary (uniform per mesh in V1) | same-hdr fanout / admit |
| Source connection direction | — | existing connection metadata | retain sender direction / manager tag | select §7.3.1 inject edge | source sender only |
| Source reroot return | validate execution against canonical root subtrees | — | dedicated VC0 same-link queue + worker-mirror flag | two-bit pre-inject overlay | self bit outside JT (kernel §3.9) |
| Rejected X / packed-packet paths | — | — | — | **§4.2.1** | **not implemented** |
| `exit_node_table` | own | **keep as-is** | — | rewrite dst→exit | edge assist |
| `inter_mesh_direction_table` | own | **keep as-is** | — | intermesh first/boundary hop | edge assist |
| `intra_mesh_direction_table` | — | **removable** (§2.11) | — | derive from Y/X vectors | — |

Use this matrix when proposing CP field names: every new ControlPlane fact should map to a column that
actually reads it.

---

## 10. Worked traces

### 10.1 Same-ex4 unicast (assessment)

```text
physical: 1 → 2 → 5 → 6 → 9 → 10
command:      S   Z   S   Z   S
```

**Baseline:** would compress to a single NS hop count → **cannot** place Z chords.

**Target:** L1 `y_vectors[10]` is 2-bit S,Z,S,Z,S,…; packet setup widens to `route_buffer_y`;
`route_buffer_x` is STOP/`LOCAL_DELIVER` at matching X. Kernel indexes `route_buffer_y[local_y]`.

### 10.2 Leaf entry acquisition (BFC relevance; codec view)

```text
physical: 3 → 2 → 5 → …
command:      N   Z
```

Codec emits N then Z on the Y map. **Injection on Z sender** is a builder fact (S-face ingress → Z),
not something the hop letters encode.

### 10.3 Cross-X after Y

```text
(0,1) → (0,2) → (0,5) → (0,4) → (1,4) → (2,4)
         S        Z        N        E        E
```

At Y=4, `route_buffer_y[4] == 0` (STOP). N/S/Z decode (§4.3) falls through to
`route_buffer_x[local_x]` (widened from `x_vectors[2]` with `LOCAL_DELIVER` at x=2). Packet holds
**Y + X = 36** action bytes on `[32,4]`.

### 10.4 Multicast Y-only `encode_root=1, S=4`

`range_anchor = encode_root = 1`. Baseline spine `1→2→3→4→5` is **illegal** under constrained routing.

Target Y map (§5.5 / §5.6 source-reachable OR) fans at node 2 with S and Z; node 5 continues N.

### 10.5 Multicast with E/W teeth (both maps + decode)

Example: `range_anchor = encode_root = (x=1, y=1)`, extents `S=2`, `E=1` on a small logical mesh
(illustrative — commands from constrained unicasts, not cardinal walk).

Targets: rows `{2,3}` × columns `{1,2}` (anchor column + one East).

Suppose Y traces from encode_root:

```text
route_buffer_y (source-reachable OR):
  [1] = S
  [2] = LOCAL_DELIVER | S | E     # deliver + continue spine + tooth fanout
  [3] = LOCAL_DELIVER | E         # terminal row: deliver + tooth; no N/S/Z
```

X map (source-reachable OR over target_xs `{1,2}` from encode_root_x=1):

```text
route_buffer_x:
  [1] = E | LOCAL_DELIVER         # deliver on anchor col + continue East
  [2] = LOCAL_DELIVER
```

Decode walk (same header everywhere):

| Chip | Facing | action_y | Chosen action | Notes |
|---|---|---|---|---|
| (1,1) N/S/Z | — | `S` | `S` | `action_y != 0` |
| (1,2) N/S/Z | — | `LOCAL_DELIVER\|S\|E` | same | local + S + E fanout; admit all |
| E-neighbor (2,2) | W | (ignored) | `route_buffer_x[2]=LOCAL_DELIVER` | E/W X-only; no Y re-fire |
| (1,3) N/S/Z | — | `LOCAL_DELIVER\|E` | same | **not** dropped: `!= 0` test (old `N\|S\|Z` test would wrongly take X) |
| E-neighbor (2,3) | W | (ignored) | `LOCAL_DELIVER` | tooth deliver |

If the encoder had full-vector-ORed `y_vectors[3]` into the map, row 2 could pick up unrelated
actions toward other dests — forbidden (§5.6).

### 10.6 Intermesh mcast carrier

Foreign final mesh: unicast-style maps route to the temporary exit, while `dst_start_node_id`
retains the final `range_anchor` and `mcast_params_64` retains all four extents.
At an exit, current/final mesh inequality plus the selected `INTERMESH` egress identifies the
boundary hop. A capability-tagged landing intercept compares its current mesh with the retained final
mesh. An intermediate landing sets `encode_root=landing`, installs unicast-style maps to that mesh's
next exit, preserves the retained anchor/extents, and continues on VC1; it does not build a multicast
tree there. Only a destination-mesh landing recovers the anchor/extents and rebuilds both maps via
`T(landing)` (§5.7.1). Intramesh Z never uses this path.

Concrete X-root case: landing at `x=0`, anchor/only target column `x=2`. Build
`route_buffer_x[0]=E`, `[1]=E`, `[2]=LOCAL_DELIVER`. Each target-row Y entry receives `E` but **not**
`LOCAL_DELIVER`, so the landing column fans East without a false local delivery; column 2 performs
the actual delivery.

---

## 11. Implementation choices and checklist

### 11.1 Remaining implementation choices

These choices affect placement or integration, not the packet semantics defined in §§4–7:

1. Derive vectors and reverse trees beside RT-gen, or expose the same complete annotated
   `R(src,dst)` to a separate host generator. Current channel-conditioned `get_fabric_route` is not
   that relation.
2. Choose per-chip or shared placement for destination-major vectors and reverse trees, including
   aligned storage for topology shapes whose packed Y stride is not naturally `uint32_t`-aligned.
   The 1028-byte vector fit is established in §2.12; an in-struct `[64,4]` hybrid grows
   `routing_l1_info_t` from 2576 B to at least 2608 B after intra-table reclamation or 2704 B if that
   table remains (§6.2).
3. Bind the deployment-wide indexed-2D selector in CT/FabricContext. It is independent of
   `express_routing_enabled`.
4. Measure whether eligible shapes use the 80-byte header tier or the 96-byte fallback (§4.7).
5. If §5.7.1 fails for any root, reject reverse-tree multicast for the mesh/configuration. An
   alternate encoder is future design work, not a V1 fallback.

Builder sender allocation and kernel queue realization are owned by their component contracts.

### 11.2 Codec implementation checklist

- [ ] Finalized routes pass suffix consistency and the §4.4 row/column-uniformity setup check before
      vectors are written; require unicast vectors to satisfy the per-row action constraints (§7.1).
- [ ] Host, worker, edge, packet typedefs, and kernel select one indexed-2D ABI. No 2D producer or
      consumer uses `hop_index` / `branch_*`, and `express_routing_enabled` is not the ABI selector
      (§4.5.1).
- [ ] Unicast performs full vector widening; multicast emits source-reachable maps rather than
      full-vector OR (§5.6).
- [ ] Every reverse-tree root passes §5.7.1, including path equality, serialization order, and
      host/device descriptor packing; otherwise the mesh/configuration is rejected.
- [ ] Runtime decode follows §4.3, maps remain immutable, and action bit 4 represents intramesh Z
      rather than NOOP (§4.6).
- [ ] Source multi-inject/reroot follows §7.3.1 using connection direction metadata. Differential
      execution reaches each canonical subtree and local target exactly once; builder/kernel queue
      checks remain in their own contracts.
- [ ] Intermesh preserves final fields, uses unicast-style maps on every intermediate mesh, and
      installs final unicast or multicast maps only on destination-mesh landing (§4.5 / §5.10).
- [ ] L1 regions and packet/header/UDM capacity satisfy §§2.12, 4.7, and 6.2 for the supported
      dimensional bound.
- [ ] Single-hop helpers follow §7.1.1, and rejected packet paths remain absent (§4.2.1).

### 11.3 Scope boundary

Topology and command extensions remain governed by the assessment's initial-cut and extension
record. Codec-specific rejected packet paths are recorded once in §4.2.1.

---

## 12. ControlPlane and host handoff

§6 defines the device-facing inputs and §9 identifies their producers and consumers. In summary:

- derive vectors and reverse trees from the finalized annotated canonical relation, either beside
  RT-gen or through an equivalent complete route view;
- provide the selected artifacts, pinned shape/coordinates, express state, and deployment-wide ABI
  selection through the existing host/L1 setup boundaries;
- retain `exit_node_table` and `inter_mesh_direction_table`; `intra_mesh_direction_table` becomes
  removable when indexed vectors are the sole intramesh source (§2.11);
- retain connection direction metadata for §7.3.1 and size L1/header storage by §§2.12, 4.7, and 6.2.

The codec does not consume builder ENTER/REMAIN effects or internal ring/transition-policy state;
Builder consumes only the ControlPlane predicates in the builder contract.
