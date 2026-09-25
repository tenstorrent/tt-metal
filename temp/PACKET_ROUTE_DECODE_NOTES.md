# Packet Route Decode: Findings and Planned Work

Notes from reviewing how `decode/headers.py` interprets fabric packet headers against what the router
actually does. Each section ends with the concrete actions it implies. File and line references are to
the tree at the time of writing.

## How routers read a 2D route (background)

A `HybridMeshPacketHeaderT<R>` carries an **action map** in `route_buffer`: one byte per mesh row
followed by one byte per mesh column (`[Y | X]`, only the first `Y + X` of the `R` bytes are used).
Each byte is a flag set: E=`0x01`, W=`0x02`, N=`0x04`, S=`0x08`, Z=`0x10`, local=`0x20`.

The map is indexed by chip position, not by hop. There is **no hop count** in the header. Each router
goes straight to its own byte using three things it already knows:

- its mesh coordinate `(y, x)` from `routing_l1_info_t` (`control_plane.cpp:2263`),
- the mesh shape `MESH_Y_SIZE`, `MESH_X_SIZE` (compile-time args, `fabric_erisc_router_ct_args.hpp:443`),
- the direction its port faces (compile-time).

`Routing2DCodec::decode_action<dir>` (`hostdevcommon/fabric_common.h:301`) picks the byte:


| Arrival port faces | Byte read                                                    |
| ------------------ | ------------------------------------------------------------ |
| E or W             | column byte `route_buffer[Y + x]` only                       |
| N, S or Z          | row byte `route_buffer[y]`; column byte if the row byte is 0 |
| intermesh ingress  | row byte first, after the landing router rebuilds the map    |


Then, per the router (`fabric_erisc_router.cpp:1437-1545`, `admit_2d_dispatch` at `:836`):


| Byte contents                                                          | Router behaviour                                |
| ---------------------------------------------------------------------- | ----------------------------------------------- |
| `local`, no directions                                                 | delivers locally (normal end)                   |
| `local`, but `dst_start_mesh_id` != this mesh                          | forwards out the intermesh port; no local write |
| directions, with or without `local`                                    | forwards on each, and delivers if `local`       |
| 0, or only bits for ports this router lacks (including its own facing) | **packet consumed and silently dropped**        |


Other header fields:

- `dst_start_chip_id` / `dst_start_mesh_id`: the final destination (or multicast anchor). Kept for the
whole trip; used by edge routers to spot the mesh exit and by landing routers.
- `mcast_params[E, W, N, S]`: multicast target rectangle around the anchor. **Not read by routers
inside a mesh.** Only used when a packet lands in its destination mesh, to rebuild the tree there
(`fabric_set_2d_intermesh_landing_route`, `tt_fabric_api.h:204`). All-zero means unicast.
- `routing_fields.value`: unused in 2D.
- A Z bit means an **express link** (a shortcut within the same mesh), not an intermesh hop
(`builder/fabric_edge_capability.hpp:21`). Leaving a mesh is always "`local` + mesh mismatch", even
when the physical port used is the chip's Z port. The map is only rebuilt on intermesh landing.

1D (`LowLatencyPacketHeaderT`) is different: a list of 2-bit per-hop actions (`noop`, `write`,
`forward`, `write_and_forward`) that each router shifts off (`update_packet_header_for_next_hop`,
`fabric_edm_packet_transmission.hpp:396`). No destination chip is stored. The "unicast"/"multicast"
label the viewer shows for 1D is inferred by the decoder from that list; it is not a header field.

## 1. Decoder correctness fixes (`decode_2d_path`)

Today's walk disagrees with the router in several ways:

- It always reads the row byte first. It must select by arrival port as in the table above. Unicast
happens to agree; multicast after a split does not.
- It keys "visited" on the chip only. A router's decision depends on **(chip, arrival port)**, so the
same chip reached through a different port is a different state.
- It does not drop bits the router ignores (its own facing, ports it does not have). A stray
self-facing bit would show as a bounce that never happens.
- Its cap is `Y + X` hops. Replace it with the state bound below.

**Termination.** Because the decision depends only on (chip, arrival port), the walk has at most about
`Y * X * 5` states. It either reaches an end or revisits a state, after which it would repeat forever.
Use (chip, arrival port) as the visited key and `Y * X * 5` as a hard safety cap. No hop count is
needed.

Actions:

- [ ] Make byte selection facing-aware, mirroring `decode_action<dir>`.
- [ ] Track the arrival port per hop; key visited on (chip, arrival port).
- [ ] Mask out the self-facing bit and bits for non-existent ports.
- [ ] Replace the `Y + X` cap with the state bound.



## 2. Replace "(partial)" with explicit endings

Every early stop currently prints `(partial)`, which hides why the walk stopped. Agreed display:


| Case                                                  | Display                                                                                                                |
| ----------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| Delivered                                             | `hop 0 W · hop 1 W · hop 2 local`                                                                                      |
| Mesh exit (`local`, `dst_start_mesh_id` != this mesh) | `… hop 2 local (not destination mesh yet, exits to mesh <id>)`                                                         |
| Z express link                                        | Follow it (see section 3). Show plain `hop N Z` only if the capture lacks the link.                                    |
| Points off a non-torus edge                           | Leave as the last hop, e.g. `hop 1 W`. Debuggers can read that.                                                        |
| Revisits a (chip, arrival port) state                 | Decode as far as the bytes go, then `repeats from hop k`. No claims about staleness.                                   |
| Zero byte at the start                                | `none`                                                                                                                 |
| Zero byte partway                                     | **Open:** `hop N (y,x) no action (packet dropped)` vs. ending at the last decoded hop. The router really does drop it. |


The mesh exit needs this chip's own mesh ID in the decoder. The exit **port** comes from the router's
inter-mesh direction table, not the header, so it can't be shown unless that table is captured.

Actions:

- [ ] Return an explicit end reason (and the chip it happened at) from `decode_2d_path`.
- [ ] Render it in `viewer/js/chip.js` (currently line 307) instead of `(partial)`.
- [ ] Decide the partway zero-byte wording.



## 3. Follow Z express links to show the full path

A Z hop does not end the route: the far chip reads the same map (as a Z-facing arrival, row byte
first) and the walk continues. The header just doesn't say which chip is at the other end; the
topology does.

1. On the current chip, find the router whose direction is Z (manifest router metadata gives
  direction and `eth_chan`).
2. Find the `topology.links` entry whose `src` is that `(mesh_id, chip_id, eth_chan)`; its `dst` is the
  far chip.
3. Convert to a coordinate (`y = chip_id / X`, `x = chip_id % X`) and continue with arrival port Z.

Actions:

- [ ] Pass topology links and router directions into the walk.
- [ ] Continue across Z; stop at `hop N Z` only when the link is missing from the capture.



## 4. Export the packet header layout in the manifest

`headers.py` hard-codes every offset (`40`, `44`, `48`, `48 + R`, …) and the action/send-type tables.
Any header layout change silently breaks decode. Instead, have `fabric_manifest.cpp` (which already
includes `fabric_edm_packet_header.hpp` and writes `packet_header_size_bytes` and
`routing_2d_route_buffer_size` in `make_fabric_context_json`) export:

- `offsetof` and `sizeof` for each header field of the configured header type:
`command_fields`, `payload_size_bytes`, `noc_send_type`, `src_ch_id`, `routing_fields`, and for 2D
`route_buffer`, `dst_start_chip_id`, `dst_start_mesh_id`, `mcast_params`. Pick the type with the same
switch `fabric_context.cpp:174-188` uses for `sizeof`.
- `NocSendType` names and values.
- `Routing2DCodec::ACTION_*` bits and the `eth_chan_directions` enum.
- 1D hop codes (`RoutingFieldsConstants::LowLatency`).
- Per-send-type command field offsets (`noc_address`, `val`, `flush`, scatter/sparse fields, …).

Notes:

- Base-class fields: `offsetof(PacketHeaderBase<H>, field)` is standard-layout and already used in the
header (`fabric_edm_packet_header.hpp:431`).
- Derived fields: `offsetof(HybridMeshPacketHeaderT<N>, route_buffer)` is accurate for this shape
(single, non-virtual CRTP inheritance), but GCC warns with `-Winvalid-offsetof` and warnings are
errors (`CMakeLists.txt:198`). Wrap the export in `#pragma GCC diagnostic ignored "-Winvalid-offsetof"`.
- The manifest is built by the host compiler; the device writes the bytes with the RISC-V toolchain.
Add `static_assert`s on the key offsets next to the structs in `fabric_edm_packet_header.hpp`
(compiled by both), so any host/device layout disagreement fails a build.
- No fallback for older manifests is needed; the tool isn't in production use yet.

Actions:

- [ ] Add a `packet_header_layout` block (plus constants) to the manifest and its schema.
- [ ] Add host/device offset `static_assert`s in `fabric_edm_packet_header.hpp`.
- [ ] Make `headers.py` read offsets and tables from the manifest.



## 5. C++ golden walks and a Python parity test

Keep one source of truth for the per-hop decision without making the decoder depend on a C++ build.

The 2D codec lives in `hostdevcommon/fabric_common.h`, compiled into both the router and host tests:
`decode_action<dir>`, `decode_action_y_first`, `fwd_dirs<dir>`, `action_is_intermesh_exit`,
`encode_2d_mcast_maps`, `y_row` / `x_row` / `widen_y` / `widen_x`. A host test calling these is using the
router's own logic. Not host-callable today:

- `widen_2d_route_to_chip` (`fabric/hw/inc/fabric_2d_route_interface.h:151`): writes through
`volatile tt_l1_ptr`. About 20 lines of glue over host-callable pieces; move it into
`fabric_common.h` or repeat it in the test.
- `admit_2d_dispatch`, the mesh-exit branch in the receive loop: the only extra rule is "ignore bits for
absent ports", which the test mirrors.
- 1D `update_packet_header_for_next_hop`: move into a host-callable header to cover 1D the same way.

**New C++ test**, alongside `tests/tt_metal/tt_fabric/fabric_router/test_mcast_reverse_tree.cpp`
(which checks encoder bytes; it stays). The new test checks what routers do with those bytes:

- Build routes with the real encoders, walk them with `decode_action<facing>` per hop (switching on the
runtime facing), neighbours from the `MeshGraph` so torus wrap and express links are real.
- Assert that the chips that deliver match the intended targets.
- Write a golden JSON, e.g. `decode/tests/golden/route_walks.json`:

```json
{
  "name": "mcast 4x4 root(1,1) N1 S1 E2 W0",
  "mesh": {"y": 4, "x": 4, "torus": false},
  "route_buffer": "21 0c 21 00 00 21 21 20",
  "dst_start_chip_id": 5,
  "mcast_params": {"E": 2, "W": 0, "N": 1, "S": 1},
  "start": {"chip": [1, 1], "arrived_on": "worker"},
  "expected": [
    {"hop": 0, "chip": [1, 1], "arrived_on": "worker", "action": "0x0c", "local": false, "out": ["N", "S"]},
    {"hop": 1, "chip": [0, 1], "arrived_on": "S", "action": "0x21", "local": true, "out": ["E"]},
    {"hop": 1, "chip": [2, 1], "arrived_on": "N", "action": "0x21", "local": true, "out": ["E"]},
    {"hop": 2, "chip": [0, 2], "arrived_on": "W", "action": "0x21", "local": true, "out": ["E"]},
    {"hop": 2, "chip": [2, 2], "arrived_on": "W", "action": "0x21", "local": true, "out": ["E"]},
    {"hop": 3, "chip": [0, 3], "arrived_on": "W", "action": "0x20", "local": true, "out": []},
    {"hop": 3, "chip": [2, 3], "arrived_on": "W", "action": "0x20", "local": true, "out": []}
  ],
  "end": "delivered"
}
```

**Python test** in `decode/tests/test_headers.py`: build each header from the golden fields at the
manifest offsets, run `decode_packet_header`, and compare the walk (unordered within a hop) and the end
reason.

Cases to cover: unicast; multicast with one branch and several; a root on an edge column (where row
and column bytes differ); torus wrap; express Z (`express_links_8x4_mesh_graph_descriptor.textproto`);
mesh exit; zero byte partway.

Actions:

- [ ] Move `widen_2d_route_to_chip` (and later the 1D per-hop update) into host-callable headers.
- [ ] Add the golden-walk gtest; share fixture helpers with `test_mcast_reverse_tree.cpp` if that
  ```
  turns out to be cleaner.
  ```
- [ ] Add the Python parity test; have CI regenerate the golden file and fail on diff.



## 6. Potential: show the full multicast tree

*Not decided.* Today the walk stops at the first split. With facing-aware decode (section 1) the
decoder can expand every branch. For the 4x4 example above (`route_buffer = 21 0c 21 00 | 00 21 21 20`):

```text
hop 0 (1,1) N+S
├─ N: hop 1 (0,1) E+local → hop 2 (0,2) E+local → hop 3 (0,3) local
└─ S: hop 1 (2,1) E+local → hop 2 (2,2) E+local → hop 3 (2,3) local
delivers: (0,1) (0,2) (0,3) (2,1) (2,2) (2,3)
```

Things to settle before committing to it:

- Presentation: tree text vs. per-branch lines vs. highlighting delivering chips on the mesh grid.
- Size: large meshes produce big trees; may want collapse or a delivery-set summary only.
- A state reached from two branches is a duplicate delivery; decide how to show it.

Also relabel the 2D "Multicast hops" field as the **target rectangle** around the anchor, and note
when a multicast is still on its carrier leg to another mesh (the walk then shows a single path to the
exit chip while the rectangle is nonzero).

## 7. Considered and deferred: a C++ decoder

Rewriting decode in C++ (or binding the router's functions via nanobind) would make the router's code
the only interpreter. Deferred because:

- Only the codec is shareable as-is; the router's dispatch and 1D update are device-only hot-path code.
- The walk, Z-link lookup, loop handling, trees and plausibility checks would still live in the
adapter.
- The decoder is stdlib-only Python and runs anywhere against a capture; a C++ dependency would lose
that.

Sections 4 and 5 give most of the benefit. Revisit a one-hop nanobind binding only if the parity test
keeps catching drift and the decoder always runs where tt-metal is built.

## Other findings

- **NOC operation is per packet.** One `noc_send_type` and one `command_fields` per header, unchanged
by 2D routers, so every delivering chip does the same NOC op to the same core and address. Exception:
`NOC_SPARSE_MCAST_WRITE` (1D only) carries up to 4 addresses and per-chip `counts[]`; routers advance
`chip_idx` / `write_idx` in the header (`fabric_edm_packet_transmission.hpp:152`). The viewer could
show which addresses the current chip will write.
- **Stale slots.** Ring memory keeps old contents and the live read/write indices aren't captured, so
any decoded header may be residual. The viewer already warns; route decoding should state facts about
the bytes rather than guess.
- **Telemetry** `router_state` **goes stale while paused.** The router writes the authoritative state to
`routing_l1_info_t::state_manager.state` (`INITIALIZING`, `RUNNING`, `PAUSED`, `RETRAINING`,
`DRAINING`), but only copies it into telemetry from inside the main loop (`update_telemetry`,
`fabric_erisc_router.cpp:1075`, gated on the `ROUTER_STATE` stat bit). During pause/retrain,
telemetry keeps the last main-loop value (usually `RUNNING`). ERISC1 only mirrors ERISC0's value.
`decode/structs.py:86` currently reads only the telemetry copy; it should also decode
`state_manager.state` from the captured routing table and prefer it.
- **Telemetry and routing-table offsets are hard-coded too** (`decode/structs.py`, e.g. the per-ERISC
entries at 80/104). The same manifest layout export (section 4) should cover these structs.



## Open questions

1. Wording for a zero byte partway through a route (section 2).
2. Whether to build the full multicast tree view, and how to present it (section 6).
3. Whether to capture the router inter-mesh direction table so mesh exits can name the exit port.

