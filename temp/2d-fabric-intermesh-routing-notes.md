# 2D fabric inter-mesh routing notes

Session notes from walking through edge-node recompute, header encoding, and an A→B unicast. Paths are relative to the tt-metal repo.

Primary files:

- `tt_metal/fabric/hw/inc/edm_fabric/fabric_edge_node_router.hpp`
- `tt_metal/fabric/hw/inc/tt_fabric_api.h`
- `tt_metal/fabric/hw/inc/fabric_routing_path_interface.h`
- `tt_metal/hostdevcommon/api/hostdevcommon/fabric_common.h`
- `tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp`
- `tt_metal/fabric/erisc_datamover_builder.cpp` (`compute_edge_facing_flags`)
- `tt_metal/fabric/control_plane.cpp` (`get_intermesh_facing_eth_chans` / `get_intramesh_facing_eth_chans`)

---

## 1. Hop command encoding (2D mesh)

Each `route_buffer[]` entry is a 4-bit mask:


| Bit | Meaning       |
| --- | ------------- |
| 0   | Forward East  |
| 1   | Forward West  |
| 2   | Forward North |
| 3   | Forward South |


`NOOP` is `0b0000`. Z / inter-mesh exit is also encoded as `NOOP` in `set_forward`.

**Own-direction bit = local write** on that eth core, not “keep going that way.”

West-facing router (`fabric_erisc_router.cpp` ~1892–1898):

- `[0010]` (`FORWARD_WEST`) → deliver locally
- `[0011]` (`WRITE_AND_FORWARD_EW`) → deliver locally **and** forward east (line mcast)

When a packet is traveling east, it arrives on a **west-facing** port, so the stop command is `FORWARD_WEST`. Traveling south, stop is `FORWARD_NORTH`.

Forward vs local is implemented as `hop_cmd & ~(1u << my_direction)` in `hop_cmd_to_sender_channel_mask`.

---



## 2. Edge flags are per eth channel, not “chip is on the mesh perimeter”

Compile-time:

- `is_intermesh_router_on_edge` ← `IS_INTERMESH_ROUTER_ON_EDGE`
- `is_intramesh_router_on_edge` ← `IS_INTRAMESH_ROUTER_ON_EDGE`

Set in `compute_edge_facing_flags` (`erisc_datamover_builder.cpp`):

```text
if intermesh_chans.empty():
    return {false, false}   # not an exit node; even a geometric edge chip
is_intermesh = my_eth_channel in intermesh_chans
is_intramesh = my_eth_channel in intramesh_chans
```

`get_intermesh_facing_eth_chans`: channels in `exit_node_directions_` (links to another mesh).

`get_intramesh_facing_eth_chans`: all fabric-mapped eth on that chip **minus** the intermesh set.

So on an **exit-node chip**:

- Eth(s) that actually connect to the other mesh → `is_intermesh_router_on_edge`
- All other fabric EDMs on that chip (inward E/W/N/S) → `is_intramesh_router_on_edge`
- A given channel is not both
- Unused eth not in the fabric port map gets neither
- A perimeter chip with **no** intermesh links gets `{false, false}` on every EDM

```text
Mesh A                              Mesh B
                             Z / intermesh
[S] ---- E/W ---- [ exit chip E ] ================ [F] ---- [D]
                    |            |
                    |            +-- eth Z  ERISC  → is_intermesh_router_on_edge
                    +-- eth West ERISC             → is_intramesh_router_on_edge
                         (packet from S arrives here)
```

Each ethernet core runs its own `fabric_erisc_router` with those flags baked in.

`get_cmd_with_mesh_boundary_adjustment` only compiles the recompute logic if `is_intermesh_router_on_edge || is_intramesh_router_on_edge`. Interior routers just read `route_buffer[hop_index]`.

Inside that `if`:

1. `hop_cmd == NOOP` — tape empty. Typical: just landed on a **new mesh** on the intermesh port. Comment: “Arrive at another mesh.”
2. `is_intramesh_router_on_edge` **+** `hop_cmd == my_direction` **+** `dst_start_mesh_id != my_mesh_id` — tape says local write, but dest is still another mesh. Typical: landed on the **exit chip from inside A**. Comment: “Arrive at exit_node from its mesh when src != exit node.”

---



## 3. L1 tables (three different things)



### `routing_l1_info_t` (`ROUTING_TABLE_BASE`)

- `my_mesh_id`, `my_device_id`
- `intra_mesh_direction_table` — 3 bits per dest **chip**: first hop E/W/N/S/Z, or invalid if dest is **this** chip
- `inter_mesh_direction_table` — 3 bits per dest **mesh**: which way to leave **this chip** toward that mesh (used when you **are** the exit)

`get_next_hop_router_direction(dst_mesh, dst_dev)`:

- dest mesh == my mesh → intra direction table[chip]
- else → inter direction table[mesh]



### Compressed paths (`ROUTING_PATH_BASE_2D`)

`intra_mesh_routing_path_t<2, true>::paths[dst_chip_id]` is **not** the direction table. Each entry is `compressed_route_2d_t`: ns_hops, ew_hops, ns_dir, ew_dir, turn_point.

`decode_route_to_buffer` expands that via `encode_2d_unicast` into `route_buffer`.

Why keep the direction table if `paths` already has a route from this chip? `paths` is the full NS/EW recipe. It does not cheaply answer “which EDM do I inject into?” or “is dest myself?” (Z / INVALID). Workers pick a connection from the direction table; the router uses invalid direction to mean dest-is-this-chip.

### Exit-node LUT (`EXIT_NODE_TABLE_BASE`)

`exit_node_table[dst_mesh_id]` → chip id on **this** mesh that exits toward that dest mesh.

`fabric_set_unicast_route` rewrites “chip D on mesh B” into “exit chip E on my mesh,” then fills an **intra** path to E. Packet header `dst_start_chip_id` / `dst_start_mesh_id` stay the **true** dest (D / B).

---



## 4. Packet header is a tape, not a per-hop table lookup

Worker (or mesh-boundary recompute) writes `route_buffer` once.

Normal hops: read `route_buffer[hop_index]`, execute, then update `hop_index`:

- Along the same axis: `routing_fields.value + 1` (hop_index is the low 16 bits of that union — fast increment)
- Turn / mcast branch: `hop_index = branch_east_offset` or `branch_west_offset`

`update_packet_header_before_eth_send` (`fabric_erisc_router.cpp` ~523–554), called from the send path ~1701.

`IS_FORWARDED_TRAFFIC_FROM_ROUTER = (SENDER_CHANNEL_INDEX != 0)`. **Channel 0 is worker inject** — no hop_index bump, no hop_cmd decode as local vs forward. That EDM just pushes over eth.

So worker encoding assumes **hop 0 is executed by the first remote receiver**, not by the injecting EDM. Leaving the source chip **is** the inject; it is not an entry in the tape.

`encode_2d_unicast` with `prepend_one_hop = false`: `(n-1)` forwards + 1 opposite-direction write.

Recompute runs **on a receiver**. That core executes the new `[0]` in the **same** iteration (`get_cmd_with_mesh_boundary_adjustment` → `can_forward_packet_completely` → `receiver_forward_packet`). Hence `prepend_one_hop = true` when decoding from a router: extra forward at the front so this core forwards instead of treating a 1-hop dest as a local write.

`recompute_path` always resets `hop_index` to 0, then returns `route_buffer[0]`. Writing `[0]` after recompute is correct even if the previous send already incremented the old index.

---

