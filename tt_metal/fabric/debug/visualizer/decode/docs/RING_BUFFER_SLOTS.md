# Ring buffer slots

[`ring_buffer_slots.py`](../ring_buffer_slots.py) decodes the packets in a router's channels. Each channel's ring buffer holds a packet per
slot: a header, then its payload. For each slot, it reads the header from the slot's L1 bytes, and works out where that
packet goes, with the route rules in [`routes.py`](../routes.py).

It reports what each slot holds. Whether the packet in it is still waiting, already sent or done isn't in the snapshot
(see "No slot state" below).

## What it takes and returns

Once per snapshot, `get_slot_context(manifest, types, route_context)` reads the packet header's struct
(`fabric_context.packet_header`), and checks that it has every member the slot rules read. It gives a `SlotContext`.

Then for each router, `get_ring_buffer_slots(snapshot, types, ctx, router, table)` takes:
- the snapshot, its types and the `SlotContext`;
- the router, as [`routers.py`](../routers.py) decodes it;
- the router's `RoutingTable`, from [`routing_tables.py`](../routing_tables.py), which a landing rebuilds a map from.
  `None` when capture didn't read it.

It gives each channel's slots, by the channel's path in the router, such as `channels/senders/vc0/ch0`. A `RingBufferSlot`:

| Field | What it holds |
| --- | --- |
| `empty` | Whether the header is all zero bytes: the slot hasn't held a packet since the router started |
| `header` | The header, as the type parser decodes it. `None` when empty |
| `route` | Where the packet goes: a `SlotRoute1D` or a `SlotRoute2D`. `None` when empty, or when a rule refuses |
| `scatter` | For a scatter write, its chunks, as `get_scatter_chunks` gives them |
| `problems` | Each rule that refused the slot's bytes, and why |

The slot's raw bytes are already in the router's `regions`, as the ring buffer's entries.

A `SlotRoute1D` is its `hops`, hop 0 first. Each `Hop1D` is the hop's name and the chip that reads it, or `None` once
the line can't be followed.

A `SlotRoute2D`:

| Field | What it holds |
| --- | --- |
| `start` | The chip whose router reads the map first |
| `entered_from` | The side the packet enters `start` from. `None` at a landing, which reads the map as a route's start |
| `walk` | The `ChipWalk2D` from `start`: what it delivers, the links it takes, where it leaves the mesh, and anything odd. `None` when `start` is in another mesh |
| `landing` | Whether the walk follows the map `start`'s router rebuilds when the packet lands, rather than the slot's |
| `rebuilt` | At a landing, whether the slot's map is already the rebuilt one. `None` elsewhere |

## Empty slots

The router zeroes every slot's header when it starts, in every sender and receiver channel
(`StaticSizedSenderEthChannel::init_impl`, `StaticSizedEthChannelBuffer::init_impl`). It never clears a slot after
that, and never clears payloads. So:

- a header of all zero bytes means the slot hasn't held a packet;
- any other header is the last packet written to the slot.

An empty slot needs its own case: a zero header isn't a route. In 1D, hop 0 is `noop`, which the router asserts on. In
2D, the map is empty everywhere.

In the three real snapshots (Wormhole T3K 2D, and Blackhole 1D and 2D, all idle), every slot is empty: 1,360, 2,560 and
3,240 slots. Some rings have nonzero bytes mid-slot: payload bytes that were in L1 before the router started.

**Only active channels have slots.** A channel no ERISC runs (`status` other than `active`: `vc_not_serviced`, `mux` or
`trimmed`) never holds a packet, so its ring holds whatever L1 held, and decode doesn't read it as headers. A channel
whose ring capture didn't read has no slots either.

## No slot state

A used slot's packet may be waiting, sent or done, and the snapshot doesn't say which:

- **Sender.** The router's read position is a variable (`next_packet_buffer_index`), and so is its producer's write
  position.
- **Receiver.** Its four positions are variables too (`ReceiverChannelPointers`).

The channels' counts say how many packets are waiting, but not which. So every used slot may be history: the last
packet that passed through it.

`src_ch_id` doesn't settle it either. A sender writes its channel there just before the Ethernet send, so a sent slot
carries it. But a forwarded packet waiting to be sent can carry the same value from the hop before.

## Where a route starts

Where a packet's route starts depends on the channel its slot is in. R is the router holding the channel.

| Slot in | 1D | 2D |
| --- | --- | --- |
| A sender channel | Hop 0 is read by R's link peer, the chip across the link | Walked from R's link peer's chip, entered from the side opposite R's direction |
| A sender channel whose link leaves the mesh | (1D stays in a mesh) | Not walked: `start` is the chip in the other mesh, and `walk` is `None` |
| A receiver channel | Hop 0 is read at R's chip: the route the packet arrived with (see below) | Walked from R's chip, entered from R's direction |
| A receiver channel at an intermesh ingress | | The landing's map, walked from R's chip as a route's start |

- **R's direction** is its `fields.my_direction`, the direction the kernel decodes the map by. Its enum,
  `eth_chan_directions`, uses the route rules' direction names.
- **1D hops are placed on chips along the line,** following each receiver's downstream channel and that router's link
  (`get_chips_1d`). A 1D line can turn at a corner, so a coordinate step would be wrong.
- **A sender whose link has no peer** gets no route, and a problem.

**Why sender slots and 2D slots are exact.** The router changes nothing in a sender slot's header but `src_ch_id`, and
a 2D route map never changes in transit. So those routes hold for the packet whether it's waiting or already sent.

**A sender slot whose link leaves the mesh** holds the map of the mesh it's leaving: the route the packet just took to
reach this exit. The exit chip's receiver forwards it to this channel as it is, without rebuilding it. That map says
nothing about the route in the next mesh, which the landing there rebuilds from its own chip's table. So decode names
the chip it goes to, and doesn't walk the map.

**1D receiver slots: a waiting packet's route is exact.** A 1D receiver shifts the route in its own slot only as it
forwards the packet (`update_packet_header_for_next_hop`, from `receiver_forward_packet`), and it forwards only once the
downstream channel has space (`can_forward_packet_completely`). A packet that can't be sent, as in a hang, keeps the
route it arrived with, and decode reads every 1D receiver slot that way.

The exception is history: a slot whose packet was already forwarded keeps the shifted route until a later packet
overwrites it, and decode can't tell it from a waiting one. Its chips come out one early: a route from C1 that forwards
once and writes reads as delivering at C2 while waiting, and at C1 once shifted. A packet that only writes isn't
shifted. A forwarded sparse multicast has its `write_idx` and `chip_idx` advanced at the same point, which the header
shows as they are.

## Landings

A receiver channel built as an intermesh ingress (`fields.intermesh_ingress`) takes packets from another mesh. Their
slots hold that mesh's map. Before reading a packet, the router rebuilds the map in place from its own chip's route
table (`fabric_set_2d_intermesh_landing_route`), then reads it row first, as a route's start (`decode_action_y_first`).

- **Decode rebuilds it the same way** (`get_landing_route_map_2d`), from the router's `RoutingTable`: toward
  `exit_node_table[dst_start_mesh_id]` for a packet bound for another mesh, or toward `dst_start_chip_id` for a
  unicast for this mesh. It only reads header fields the router never writes, so it gives the same map before and after
  the rebuild.
- **`rebuilt`** says whether the slot's map already is the rebuilt one. The router processes only the packet at the head
  of the channel, and rebuilds its map each time, before it knows whether it can send it. So the head packet holds this
  mesh's map even when it's blocked, while the packets queued behind it still hold the other mesh's. `rebuilt` means the
  packet reached the head, not that it was forwarded. In a hang, the head packet is the one the rest wait behind.
- **A multicast for this mesh** isn't rebuilt: intermesh multicast isn't supported (see "What the rules don't cover"
  in [ROUTES.md](ROUTES.md)). The slot gets a problem instead of a route.
- **No routing table:** when capture didn't read the router's routing table, a landing slot gets a problem instead of a
  route.

## Problems

Decode never raises on a slot's bytes. A hung fabric is where odd bytes turn up, so each rule that refuses a slot goes
in its `problems`, and the rest of the slot still decodes:

- a `noc_send_type` that names no send type. The header's `command_fields` then has no member;
- a scatter write the router can't carry out;
- no route: a 1D route that reaches a `noop` or forwards past its last word, or a landing that can't be rebuilt.

A 2D walk doesn't refuse: what the router wouldn't do shows in the walk's `odd`.

## The names it reads

The meaning of these names is in decode's code (case 2 of "What decode reads, and what it knows" in the
[README](../README.md)). `get_slot_context` checks the header's once per snapshot, so a renamed one fails with
`ManifestError` even when every slot is empty.

- **The header:** `fabric_context.packet_header`'s `schema` and `size`. Its members:
  - `command_fields`, a union with a `unicast_scatter_write` member holding `chunk_count`, `chunk_size` and
    `chunk_encoding`;
  - `noc_send_type` and `payload_size_bytes`;
  - in a 1D run, `routing_fields.value`, and `routing_fields.route_buffer` when the run has extension words;
  - in a 2D run, `route_buffer`, `dst_start_mesh_id`, `dst_start_chip_id` and `mcast_params`. `route_buffer` must hold
    a map for every mesh: one byte per row, then one per column.
- **The router:** `fields.my_direction`, `link.peer`, and each channel's `status` and `ring_buffer`.
- **A receiver channel:** `fields.intermesh_ingress`, in a 2D run.
- **The router's routing table,** at a landing: `route_table_2d.action_vectors` and `exit_node_table`.

## What it doesn't cover

- **Slot state:** which slots are waiting, sent or free. It needs the router to keep its positions in L1.
- **Following a packet into the next mesh.** That needs the landing chip's routing table, which may be another rank's.
- **Multicast landings** for this mesh (see "Landings" above).
- **Payloads.**

## Files cited

| File | Symbols |
| --- | --- |
| `tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_datamover_channels.hpp` | `StaticSizedSenderEthChannel::init_impl`, `StaticSizedEthChannelBuffer::init_impl`, `next_packet_buffer_index` |
| `tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_flow_control_helpers.hpp` | `ReceiverChannelPointers` |
| `tt_metal/fabric/hw/inc/edm_fabric/fabric_edm_packet_transmission.hpp` | `update_packet_header_for_next_hop` |
| `tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp` | the receiver step: the landing rebuild, `decode_action` and `decode_action_y_first`, the exit's forward as it is; `can_forward_packet_completely`, `receiver_forward_packet`; the sender's `src_ch_id` write |
| `tt_metal/fabric/hw/inc/tt_fabric_api.h` | `fabric_set_2d_intermesh_landing_route` |
| `tt_metal/hw/inc/hostdev/fabric_edm_packet_header.hpp` | `LowLatencyRoutingFieldsT`, `HybridMeshPacketHeaderT` |
