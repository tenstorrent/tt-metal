# Fabric Device Router Kernel Contract

ERISC contract for the current 2D router baseline and the target cardinal N/S + Z express
decode/admit/forward path.

Authority split:

- `GALAXY_DEVICE_ROUTE_CODEC_CONTRACT.md` owns packet/L1 byte semantics, indexed-map construction,
  widening, and source-injection requirements.
- `GALAXY_BUILDER_ROUTING_CONFIG_CONTRACT.md` owns router wiring, sender allocation, compile-time
  sender roles, and capacity.
- `GALAXY_CARDINAL_NS_Z_SKIP_ROUTING_ASSESSMENT.md` owns route legality, topology assumptions,
  CDG/BFC proofs, system invariants, and validation oracles.
- `GALAXY_CONTROL_PLANE_ROUTING_GENERATION_CONTRACT.md` owns pre-builder topology materialization,
  ring synthesis, canonical-route generation, and ControlPlane route/domain state.
- `GALAXY_WORKING_MODEL.md` owns the current physical and software baseline.

This document owns ERISC realization: decode integration, admission, forwarding, dense dispatch,
intermesh landing order, legacy turn/header retirement, sender-step BFC consumption, and kernel
CT/speedy/trim policy. Codec §8 is only the kernel boundary; the byte decoder and action layout are
authoritative in codec §4.3 and §4.6.

Status: target contract plus current-code inventory. Target behavior is not implemented unless
explicitly identified as current behavior. Arbitrary-pattern VC1 safety/BFC remains open; the
current two-link, predominantly linear VC1 envelope is retained.

---

## 0. Scope

The target kernel consumes codec-installed indexed maps and builder-emitted CT configuration. It
does not define a second packet ABI, construct routes, infer protected domains, or query
ControlPlane at runtime.

```text
Codec   → route_buffer_y/x, retained final destination/range fields, action bytes
Builder → sender/receiver wiring and edge capability, per-sender IS_INJECTION
Kernel  → decode realization, atomic admit/fanout, jump-table execution, sender-step BFC
```

Required kernel shape:

- default Z-capable builds use a dense 16-way admit and forward jump table;
- `LOCAL_DELIVER` stays outside the jump-table key; a self-facing action bit is invalid and fails
  closed;
- WH with VC1 keeps its compass-only bit-test admission exception;
- indexed transit forwards immutable full packets and does not consume hop cursors or branch
  offsets;
- intermesh landing is intercepted before ordinary indexed decode and remains on VC1;
- remote BFC is applied only by each concrete sender step;
- speedy and trimming paths stay disabled until they are admission-equivalent.

---

## 1. File / API inventory (current baseline)

| Path | Current role |
|---|---|
| `tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp` | Main 2D RX admit/forward path, hop-command dispatch, sender steps, trim hooks, and header-update call |
| `tt_metal/fabric/hw/inc/edm_fabric/fabric_edge_node_router.hpp` | Edge command adjustment and `NOOP`-driven `recompute_path` |
| `tt_metal/fabric/hw/inc/edm_fabric/fabric_edm_packet_transmission.hpp` | 2D `routing_fields.value + 1` header update when `UPDATE_PKT_HDR_ON_RX_CH` |
| `tt_metal/fabric/hw/inc/tt_fabric_api.h` | Worker/edge unicast and multicast setup; source injection bypasses RX dispatch |
| `tt_metal/hostdevcommon/api/hostdevcommon/fabric_common.h` | Current 2D routing fields and packet-header types |
| `tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_ct_args.hpp` | Direction, channel, BFC, injection, Z, speedy, and other kernel CT state |
| `tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_speedy_path.hpp` | Worker-only fast path with explicit incompatibility checks for deadlock avoidance and forwarded VC0 traffic |

---

## 2. Current 2D hop path

### 2.1 RX hot path

```text
packet on receiver channel
  → hop_cmd = get_cmd_with_mesh_boundary_adjustment(...)
       = route_buffer[hop_index] after optional edge rewrite
  → can_forward_packet_completely(hop_cmd, downstreams, local_relay)
  → receiver_forward_packet(..., hop_cmd)
  → wr_sent++
```

The current 2D hop command is a four-bit E/W/N/S mask. The bit matching this ERISC's facing means
local write; the other bits select downstream senders. Combinations implement line and 2D multicast.

### 2.2 Direction mapping and Z

`direction_to_compact_index_map[my_direction][downstream]` already has a Z row/column on Z-capable
builds. `hop_cmd_to_sender_channel_mask` maps non-self direction bits to compact sender channels, but
its current call site is trim/resource telemetry, not hot-path admission or forwarding.

The packet hop-byte ABI does not currently provide an intramesh Z action bit. WH has no Z port
(`num_z_ports = 0`); Z-capable behavior is a BH path.

### 2.3 `NOOP` overload

The current non-WH-VC1 admission path treats `NOOP` with `z_router_enabled` as a Z space check.
Edge handling also treats `NOOP` as a trigger for `recompute_path`.

```text
current: NOOP may mean Z forwarding or edge recompute
target:  Z is an action output; current/final mesh identity plus INTERMESH edge capability identifies
         boundary egress and landing
```

The target must remove both `NOOP` meanings. Mesh-id inequality says that the final mesh is still
remote, while edge capability identifies the actual boundary; direction alone must not classify a
hop as intermesh.

### 2.4 Header mutation and admission

Current 2D forwarding mutates route progress in one of two places:

- sender side: `update_packet_header_before_eth_send` increments `hop_index` or jumps to
  `branch_east_offset` / `branch_west_offset`;
- RX side: `UPDATE_PKT_HDR_ON_RX_CH` selects equivalent hop-index/branch mutation before copies.

Current default admission switches on the full hop command. WH with
`FABRIC_2D_VC1_ACTIVE` instead bit-tests E/W/N/S to control code size. Local delivery is encoded by
the self-facing bit rather than an explicit flag.

### 2.5 Current two-stage flow control

RX admission checks local downstream sender queues and local relay capacity. Later,
`run_sender_channel_step_impl` checks remote receiver credit, including the stronger threshold when
the sender's CT `IS_INJECTION` flag and deadlock avoidance are enabled. The target preserves this
stage split; §5 is the kernel authority for BFC consumption.

---

## 3. Target decode and admit/forward dispatch

Codec §4.3 owns indexed decode semantics and codec §4.6 owns action-byte layout. The kernel realizes
those bytes with cached coordinates, CT indexing, sender selection, and atomic dispatch.

### 3.1 Cached locals and flat route buffer

```text
local_y, local_x         # pinned logical chip coordinates; cache once
MY_DIR                   # this ERISC's eth facing; constexpr

constexpr Y_SIZE, X_SIZE
constexpr Y_BASE = 0
constexpr X_BASE = Y_SIZE
```

`Y_BASE` and `X_BASE` are fixed indices into the packet's flat `[Y | X]` route buffer. Transit does
not divide/mod, read an L1 vector table, or extract two-bit entries. Packet layout and capacity remain
codec-owned.

### 3.2 Decode realization

Kernel control flow must realize codec §4.3:

```text
decode_packet_action(packet):
    buf = packet.route_buffer
    if constexpr (MY_DIR in {E, W}):
        return buf[X_BASE + local_x]

    action_y = buf[Y_BASE + local_y]
    if action_y != 0:
        return action_y
    return buf[X_BASE + local_x]
```

E/W-facing routers use X only. N/S/Z-facing routers use Y when the complete Y action is nonzero,
including `LOCAL_DELIVER` or E/W tooth bits; testing only N/S/Z bits is incorrect. A receiver whose
builder-provided ingress capability is `INTERMESH` performs the §4.2 landing intercept before calling
this ordinary decoder. That is router context, not packet routing state or an action-byte value.
Action-bit positions and reserved-bit validation come only from codec §4.6.

The two boundary checks sit on opposite sides of this decoder and must not be confused. The landing
intercept is keyed on `INTERMESH` **ingress** capability and runs *before* decode, because the maps
still belong to the previous mesh. The §4.2 exit check is keyed on the decoded action plus mesh
identity and runs *after* decode, because it needs the action to know that the maps have run out
here. Only the arriving router evaluates either one.

### 3.3 Admit/forward dispatch

Default Z-capable builds preserve jump-table dispatch:

1. Test the self-facing bit before packing; a set self bit is invalid and fails closed (§3.7).
2. Pack the four possible eth outputs through fixed `FWD_DIRS[4]` into key `0..15`.
3. Use hand-written `case 0..15` labels with `admit_combo<KEY>` and `forward_combo<KEY>` bodies.
4. Keep `LOCAL_DELIVER` outside the key.
5. Admit atomically against local sender queues and local relay only.
6. Forward immutable full-packet copies to the selected eth outputs, then deliver locally.

A raw 6-bit/64-way table is not used. `FWD_DIRS[4]` remains fixed for the first implementation;
live-only packing is a later optimization. `hop_cmd_to_sender_channel_mask` remains outside the hot
path unless a future trim implementation is made equivalent.

| Dispatch choice | Kernel reason |
|---|---|
| Dense key over four non-self outputs | Preserves a bounded 16-arm table when Z replaces the self direction |
| Separate admit and forward switches | Preserves the current all-local-capacity-before-copy structure |
| Template bodies with hand-written labels | Keeps each selected output compile-time visible without a macro-generated ABI |
| `LOCAL_DELIVER` outside key | Avoids doubling the table and preserves eth-before-local order |
| Self bit rejected before packing | Canonical maps never set it; treating it as an output would add a same-link reversal |

Practical WH exception: when `FABRIC_2D_VC1_ACTIVE` is set, keep admission as runtime E/W/N/S
bit-tests plus explicit local-relay admission. WH has no Z sender, so this path must not emit, test,
or admit Z. It remains RX-local-capacity only. Multi-bit compass multicast is valid, and forward may
retain the existing compass jump table/non-inline split. Other Z-capable builds use the packed
16-way admit and forward tables.

**Invalid action policy:** generation and packet setup must prevent invalid actions. The runtime
fast path follows the existing fabric fail-stop style:

```text
if reserved bits are set
   or the self-facing bit is set
   or a selected direction has no wired sender on this router
   or no eth or LOCAL_DELIVER destination is selected:
    ASSERT(false)          # visible when watcher/assert support is enabled
    return false           # commit no copy; retain/stall the parent RX packet
```

There is no packet retransmission or recovery contract. An optimized build may therefore appear as
a permanent packet stall/workload failure rather than a recoverable error. The kernel does not
reinterpret an invalid or all-zero indexed action as Z, NOOP, or local delivery.

### 3.4 Pack and combo helpers

```text
// Four non-self eth directions available to this router.
// MY_DIR=NORTH → {E, W, S, Z}
// MY_DIR=EAST  → {W, N, S, Z}
// MY_DIR=Z     → {E, W, N, S}
constexpr eth_chan_directions FWD_DIRS[4] = { … };

u8 pack_fwd(u8 action):
    key = 0
    if (action & bit(FWD_DIRS[0])) key |= 1 << 0
    if (action & bit(FWD_DIRS[1])) key |= 1 << 1
    if (action & bit(FWD_DIRS[2])) key |= 1 << 2
    if (action & bit(FWD_DIRS[3])) key |= 1 << 3
    return key

template<u8 KEY>
bool admit_combo(bool ld):
    ok = !ld || local_relay_has_space
    if constexpr ((KEY >> 0) & 1):
        ok = ok && sender_for[FWD_DIRS[0]].has_local_slot()
    if constexpr ((KEY >> 1) & 1):
        ok = ok && sender_for[FWD_DIRS[1]].has_local_slot()
    if constexpr ((KEY >> 2) & 1):
        ok = ok && sender_for[FWD_DIRS[2]].has_local_slot()
    if constexpr ((KEY >> 3) & 1):
        ok = ok && sender_for[FWD_DIRS[3]].has_local_slot()
    return ok

template<u8 KEY>
void forward_combo(packet, bool ld):
    if constexpr ((KEY >> 0) & 1):
        forward_full_packet(sender_for[FWD_DIRS[0]], packet)
    if constexpr ((KEY >> 1) & 1):
        forward_full_packet(sender_for[FWD_DIRS[1]], packet)
    if constexpr ((KEY >> 2) & 1):
        forward_full_packet(sender_for[FWD_DIRS[2]], packet)
    if constexpr ((KEY >> 3) & 1):
        forward_full_packet(sender_for[FWD_DIRS[3]], packet)
    if (ld):
        local_relay_deliver(packet)
```

`has_local_slot()` is the current local downstream queue check. `forward_full_packet` is current EDM
forwarding without mutate-before-send. Admission must succeed for every selected local destination
before any copy is committed.

For a selected key, compile-time conditions must remove unselected queue checks and sends. The only
runtime boolean is `ld`, which does not change the key-to-direction mapping. Sender lookup must use
the builder-wired compact mapping for the packet's current VC.

Atomicity covers:

- every eth child selected by `KEY`;
- local relay when `ld` is set.

A failure at any one destination stalls the parent RX packet without committing another copy.
Remote receiver credit is deliberately absent here and is checked later by §5.

### 3.5 RX hot path

```text
action = decode_packet_action(packet)
if (action & RESERVED_BITS) != 0:
    invalid_action_fail_closed()

# §4.2 step 2: this chip is the exit when the maps say deliver here but the final mesh is elsewhere.
if constexpr (is_intramesh_router_on_edge):
    if (action == LOCAL_DELIVER
        && packet.dst_start_mesh_id != routing_l1_info.my_mesh_id):
        take_intermesh_egress(packet)   # §4.2; no local delivery, no map rebuild
        done

ld   = action & LOCAL_DELIVER
self = action & one_hot(MY_DIR)

if (self):
    invalid_action_fail_closed()    # canonical maps never set the self bit

key = pack_fwd(action)
if key == 0 && !ld:
    invalid_action_fail_closed()

switch (key):                       # WH+VC1 uses §3.3 bit-test admission
    case K: ok = admit_combo<K>(ld); break
if (!ok):
    stall

switch (key):
    case K: forward_combo<K>(packet, ld); break
```

The admit and forward switches use the same dense key.

Example for `MY_DIR=NORTH`: with `FWD_DIRS={E,W,S,Z}`, action
`S|Z|LOCAL_DELIVER` packs to `0b1100`. `forward_combo<12>` copies the full packet to S and Z, then
performs local delivery. No branch receives a different header.

### 3.6 Fanout and header behavior

All selected eth outputs receive the same complete packet image. Local relay runs last.

| Selected destination | Kernel action |
|---|---|
| Non-self eth output | Copy the complete packet to that direction's sender |
| `LOCAL_DELIVER` | Deliver through local relay after eth copies |
| Self-facing bit | Invalid; assert and fail-stop |
| No selected destination | Assert when enabled, then fail-stop without committing a copy |

Indexed transit does not call the 2D bodies of `update_packet_header_before_eth_send` or
`update_packet_header_for_next_hop`; it does not increment `hop_index`, jump through `branch_*`, or
rewrite per branch. The separate 1D decrement/shift/refill behavior is unchanged.

The §4.2 capability-tagged intermesh landing is the only boundary at which codec-owned setup may
replace maps before indexed forwarding resumes. It does not mutate or add packet routing-state
fields; an ordinary transit fanout never rewrites maps.

### 3.7 Excluded kernel paths

Do not add a line-counter X path, packed two-bit packet decoder, raw 64-way action table,
own-direction-as-local encoding, per-branch map rewrite, or remote BFC in `admit_combo`. Codec
§4.2.1 owns the rejected packet representations.

**No same-link return path exists.** No producer routes a packet back over the link it arrived on,
whether as a canonical route turn or as infrastructure. Such an arc is what assessment §5.7.3
assumes absent: it joins the two orientation views of a protected ring, or turns a bottom-stratum
attachment into a resource that waits on the ring hanging off it, and neither a BFC flag nor a
producer-boundary ordering repairs it. A self-facing action bit is therefore always invalid.

### 3.8 Source injection boundary

Source-chip worker injection bypasses RX dispatch:

```text
worker/codec installs packet state
  → worker opens the selected directional sender
  → worker writes one complete packet to that sender
  → source ERISC sender-step transmits it
  → first neighbor RX begins §3.2 decode and §3.3 dispatch
```

There is no source-side kernel root dispatch. The kernel consumes the selected sender and packet
image; codec §7.3.1 owns source-output selection, source multi-inject, and exactly-once
source-local-delivery policy. When a canonical multicast root has several outputs, the worker sends
one copy per output through its own connections — the kernel sees only ordinary independent
injections and does nothing special. Target unicast widening is codec §7.1 and primary multicast
reverse-tree construction is codec §5.7.1.

Intermesh source traffic remains a unicast-style send toward the exit. Destination-mesh landing is
already on RX and uses ordinary atomic fanout after its maps are installed.

### 3.9 Legacy turn/header retirement

The current sender-side dependency chain is:

```text
is_spine_direction
  → TURN_STATUS_ARRAY_SIZE
  → get_sender_channel_turn_statuses
  → sender_channels_turn_status
  → IS_TURN
  → update_packet_header_before_eth_send
       turn     → hop_index = branch_east_offset | branch_west_offset
       non-turn → hop_index++
```

`UPDATE_PKT_HDR_ON_RX_CH` is the alternate RX-side realization of the same legacy hop-program
consumption. Indexed 2D transit needs neither implementation.

At universal indexed-2D cutover, remove together from ERISC and its direct interfaces:

- `is_spine_direction`;
- `TURN_STATUS_ARRAY_SIZE`;
- `get_sender_channel_turn_statuses`;
- `sender_channels_turn_status`;
- local `IS_TURN`;
- the 2D body/call of `update_packet_header_before_eth_send`;
- `UPDATE_PKT_HDR_ON_RX_CH` and its RX-side branches;
- the HybridMesh `routing_fields.value + 1` path;
- all `hop_index` and `branch_east/west_offset` reads/writes;
- branch-specific mutation in the legacy multicast switch;
- `route_buffer[hop_index]` and `NOOP` recompute in
  `get_cmd_with_mesh_boundary_adjustment`.

Codec §4.5/§4.5.1 owns replacement of the legacy control bytes and producer-side cutover. The
indexed action dispatcher replaces the legacy multicast switch. No legacy 2D producer may reach an
indexed kernel; the separate 1D packet path remains unchanged.

The old four-byte routing-control region is replaced by codec-owned control state, not simply
deleted. Removing only ERISC mutation while a worker, edge encoder, profiler, or debug consumer
still interprets the legacy hop program is invalid. Express feature gating alone is not a safe ABI
selector because non-express 2D packets may share the same kernel.

`port_direction_table` is an independent removal: it is threaded through normal/speedy receiver
calls but is not initialized, indexed, or read. Its removal does not wait for indexed ABI cutover.

Retain:

- `MY_DIR`, `FWD_DIRS`, direction/compact-sender mappings, and router/VC connection maps;
- per-sender `IS_INJECTION` and sender-step BFC;
- intermesh edge tables and handling;
- 1D header-update overloads;
- `hop_cmd_to_sender_channel_mask` while trim/resource telemetry still consumes it.

If trim is later enabled for indexed routing, that helper must consume decoded actions.

---

## 4. Z and intermesh kernel behavior

### 4.1 Intramesh Z

On Z-capable builds, a decoded Z output selects the builder-wired Z sender for the carrier's current
VC. Same-mesh traffic normally uses VC0; a landed intermesh carrier remains on VC1. Z stays on the
indexed Y path and receives ordinary local admission plus sender-step BFC.

WH has no Z path. Neither `MY_DIR == Z`, a `NOOP`, nor a direction letter alone means intermesh.

### 4.2 Intermesh transition

The kernel sequence is:

1. Ordinary indexed maps route to the temporary exit while preserving the final
   `dst_start_mesh_id`.
2. **Exit detection.** A router recognizes that it is this mesh's exit for the packet when, on an
   edge-facing intramesh router, the decoded action requests local delivery only and the final mesh
   is elsewhere:

   ```text
   if constexpr (is_intramesh_router_on_edge):
       if (action == LOCAL_DELIVER                                  # no eth bits set
           && packet.dst_start_mesh_id != routing_l1_info.my_mesh_id):
           boundary_dir = get_next_hop_router_direction(
                              packet.dst_start_mesh_id, packet.dst_start_chip_id)
           forward the packet on the INTERMESH egress in boundary_dir
           suppress local delivery
           done
   ```

   This is the indexed-ABI form of the current `is_intramesh_router_on_edge &&
   dst_start_mesh_id != my_mesh_id && hop_cmd == FORWARD_<my_direction>` test in
   `get_cmd_with_mesh_boundary_adjustment()`: "the installed maps say deliver here, but the final
   mesh is elsewhere, so *here* is the exit." Both conditions are load-bearing. Mesh-id inequality
   alone is not sufficient, because a packet merely transiting a boundary-capable chip toward a
   different exit also satisfies it; the local-delivery-only action is what identifies this chip as
   the selected exit. The CT gate is what keeps interior routers from paying the mesh-id compare on
   every hop, so it must not be dropped.

   The egress is selected from the retained `inter_mesh_direction_table`, not from a compass letter
   in the packet, and the selected edge must carry builder-provided `INTERMESH` capability. Unlike
   the current path, no map rebuild happens at the exit — the packet is forwarded as-is.

   Codec “Packet fields and intermesh landing” guarantees the encoder-side half: an intermesh
   carrier's installed maps yield exactly `LOCAL_DELIVER` and no eth bits at the exit chip, so the
   predicate is never ambiguous. A multi-bit action at an exit chip is an encoder error.
3. On the remote boundary-facing receiver, `INTERMESH` ingress capability intercepts the landing
   before `decode_packet_action` can consume stale source-mesh maps.
4. Compare the landing router's current mesh with the retained final mesh, then run the landing encode
   boundary. Codec §4.5, §5.10, and §6.3 own whether intermediate
   unicast-style next-exit maps or destination maps are installed.
5. Resume ordinary indexed decode with the installed maps.
6. Keep the carrier on VC1 through intermediate and destination meshes; decoded cardinal or Z
   outputs select same-VC senders. There is no VC1→VC0 landing crossover.

No packet mode or boundary-direction field participates in this sequence. Intermediate landing must
not begin multicast fanout, while destination landing may expose a multi-output action immediately;
those distinctions come from current-versus-final mesh identity and the installed maps, not kernel
route classification.

The same ordered intercept applies regardless of the physical compass letter used by the intermesh
edge. Conversely, an intramesh Z output never takes the boundary path merely because its direction is
Z.

Both edge flags are per-Ethernet-channel and are already derived from the ControlPlane channel split
(`is_intramesh_router_on_edge` is set for a channel that is intramesh-facing on a chip that also owns
intermesh channels). An express-Z router on an exit chip therefore inherits the flag with no new
mechanism, **provided express-Z channels are classified intramesh-facing rather than intermesh**. If
a same-mesh express channel were still registered as an exit-node direction, a packet arriving at the
exit chip over a Z chord would decode without the step 2 check and would deliver locally instead of
crossing the boundary. Builder §4.3 owns that capability classification.

The kernel owns the ordering and carrier continuity, not map construction. Builder §3.5–§3.6 and
§4.4 own VC1 wiring and concrete sender roles. The assessment §5.7.5 owns the VC1 CDG/BFC decision
for arbitrary traffic beyond the current two-link, predominantly linear envelope, including
multi-mesh pass-through. No VC0 proof or flag transfers automatically to those broader VC1
dependencies.

---

## 5. BFC consumption

All kernel BFC rules are centralized here.

```text
Stage 1 — RX fanout admission:
    require local capacity for every selected child sender
    and local relay when requested
    commit no copy until all selected local destinations can accept

Stage 2 — each concrete sender step:
    if IS_INJECTION[i] && deadlock avoidance:
        require remote free >= 2
    else:
        require ordinary remote space (free >= 1)
    then transmit according to existing ACK/deadlock CT policy
```

The kernel consumes `SENDER_CH_i_IS_INJECTION`; it does not derive it. It must not infer injection
from direction pairs, Z, decode phase, line state, or the self bit, and it must not apply the remote
threshold inside RX admission.

If arbitrary-pattern VC1 BFC is enabled, landed VC1 cardinal/Z senders apply independently derived
VC1 flags and protected-receiver counts. The current restricted VC1 envelope may retain its existing
no-bubble behavior.

Local child capacity and first-level ACK do not replace the sender's remote protected-receiver
count. Each software sender applies only its own CT role when it attempts eth transmission; two
producers that need different roles must not silently alias one sender queue.

The same rules apply to unicast and every branch of multicast. Atomic RX admission establishes that
the packet can be cloned locally; it does not reserve remote space for all branches at once.

Builder §4.4 owns effect-to-flag derivation. Assessment §5.7.3–§5.7.4 owns the VC0 BFC and atomic
fanout proof; assessment §5.7.5 retains the arbitrary-pattern VC1 obligation.

---

## 6. Kernel CT and fast-path policy

- Cache pinned `local_x/local_y`; use CT `MY_DIR`, `Y_SIZE`, `X_SIZE`, `Y_BASE`, and `X_BASE`.
- Keep fixed `FWD_DIRS[4]` and dense default dispatch; WH+VC1 uses the §3.3 admission exception.
- Keep `LOCAL_DELIVER` outside the packed key and reject a set self bit before packing.
- Disable speedy, super-speedy, trim bypass, and channel remapping for express/protected-ring
  configurations until each preserves the same local admission, concrete sender, remote BFC, and
  queue graph.
- Allocate no producer that returns a packet over the link it arrived on, in any configuration.

---

## 7. Cross-document dependencies

| Dependency | Authority |
|---|---|
| Runtime indexed decode and action bytes | Codec §4.3 / §4.6 |
| Control overlay and intermesh landing map lifecycle | Codec §4.5 / §5.10 / §6.3 |
| Target unicast widening | Codec §7.1 |
| Primary multicast reverse tree | Codec §5.7.1 |
| Source multi-inject and injection-direction selection | Codec §7.3.1 |
| Sender allocation and ceilings | Builder §3.6–§3.7 |
| Per-concrete-sender `IS_INJECTION` | Builder §4.4 |
| VC0 BFC/atomic-fanout proof and source-injection boundary | Assessment §5.7.3–§5.7.4 / §9.5 |
| Arbitrary-pattern VC1 safety/BFC | Assessment §5.7.5 plus builder realization |
| ERISC admit, dispatch, fanout, landing order, BFC use | This document |

Kernel runtime inputs are codec-installed packet bytes and builder CT arguments. Byte changes begin
in the codec; sender-role changes begin in the builder. Codec §8 records this boundary but does not
contain kernel decode snippets.

---

## 8. Kernel implementation checklist

- [ ] Cache logical coordinates and use flat CT Y/X indices.
- [ ] Realize codec §4.3 decode exactly; reject reserved/invalid action state per codec §4.6.
- [ ] Handle invalid/self/all-zero actions with the §3.3 assert-and-fail-stop policy; never reinterpret
      them as Z or NOOP.
- [ ] Detect the exit chip with the §4.2 predicate — edge-facing intramesh router, local-delivery-only
      action, and final mesh elsewhere — and take the `INTERMESH` egress from the retained direction
      table instead of delivering locally; keep the CT gate so interior routers pay nothing.
- [ ] Intercept each boundary landing on `INTERMESH` ingress capability before ordinary decode;
      retain VC1.
- [ ] Remove `NOOP` as both Z and recompute.
- [ ] Use dense hand-written 16-way admit/forward dispatch on Z-capable default builds.
- [ ] Keep WH+VC1 admission as compass-only local-capacity bit tests; never add WH Z handling.
- [ ] Keep local delivery outside the jump-table key and fail closed on a set self bit.
- [ ] Admit atomically against all selected local queues; forward immutable full packets and deliver
      locally after eth outputs.
- [ ] Wire no producer from a receiver back to a sender on the same physical link (§3.7).
- [ ] Apply remote BFC only in each sender step from that sender's CT flag.
- [ ] Retire the complete §3.9 2D turn/header-mutation set only at universal indexed cutover.
- [ ] Remove `port_direction_table` independently.
- [ ] Disable speedy/trim paths until admission and queue-graph equivalence is established.
- [ ] Keep arbitrary-pattern VC1 expansion gated until its assessment and builder obligations pass;
      retain the current restricted VC1 envelope.
