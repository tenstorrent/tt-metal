# v4 UDM mode: design note

Status: built and validated (2026-10-29; `variants/v4udm`, jobs 139-140: UDM fixtures 8/8 and 68/68 on main's router
and on v4udm; RESULTS §98). User: close out the remaining features in the order UDM, VC2, speedy path, telemetry
(performance gaps parked in V4_PERF_GAPS.md). Base: the lead, `variants/v4rs`. Sources: main's router
(`fabric_erisc_router.cpp`: `forward_to_local_destination`, `admit_2d_dispatch`, the relay setup and runtime args),
`fabric_edm_packet_transmission.hpp` (`execute_chip_unicast_to_relay`), the builders (`compute_mesh_router_builder.cpp`
`connect_to_local_tensix_builder`, `erisc_datamover_builder.cpp`), `fabric_router_relay_extension.cpp`,
V4_FEATURE_SPEC.md §7.

## 1. What UDM is, from the router's side

`FabricTensixConfig::UDM` puts two fabric kernels on a Tensix core per router direction: a mux (BRISC) that feeds the
router's worker channel, and a relay (NCRISC) that takes the packets addressed to this chip and does their NoC work
(writes, atomics, reads and their responses). The router itself changes in one place: **local delivery**. A packet
for this chip is not written to its NoC destination; the whole packet (header and payload) goes into the relay's next
slot, as a forward to another router would.

Configs (mock capture of fabric_unit_tests's `Fabric2DUDMModeFixture`, case `bh_2d_udm_gtest`, 40 routers): 2D mesh,
one ERISC (any Tensix extension turns two-ERISC mode off), Tensix mux mode off (`FABRIC_TENSIX_EXTENSION_MUX_MODE` 0:
the receivers forward to routers, sender channels stamp `src_ch_id`), `UDM_MODE` 1, `NUM_DS_OR_LOCAL_TENSIX_CONNECTIONS`
1 (the relay), `LOCAL_RELAY_NUM_BUFFERS` 4, `TENSIX_RELAY_LOCAL_FREE_SLOTS_STREAM_ID` 30, no deadlock avoidance, no VC1,
4 VC0 sender channels. The define `UDM_MODE` makes `PACKET_HEADER_TYPE` = `UDMHybridMeshPacketHeader`
(`HybridMeshPacketHeader` + `UDMControlFields`; base fields at the same offsets). The earlier synthesized case
`bh_2d_udm` (the mux case with UDM's args) carries mux mode and is not a real UDM config.

## 2. Main's behavior

- Admission: a packet that delivers here needs a free relay slot, as a forward needs a free slot downstream.
- Delivery: `execute_chip_unicast_to_relay`: one stateless NoC write of header + payload to the relay's slot, tagged
  with the receiver slot's transaction ID, on the forwarding command buffers; then -1 on the relay's free-slots register
  (remote) and on this router's copy (stream 30). The relay returns slots by incrementing the copy.
- Completions: the relay write carries the slot's transaction ID, so the slot is freed only once the write has left.
- Init: runtime args at the end (`has_local_tensix_relay_connection`, then relay buffer base, NoC x, y, registration
  address, location-info address, relay free-slots stream id, connection buffer-index address). The relay counts
  itself into `EDM_LOCAL_TENSIX_SYNC_PTR_ADDR` when ready; the router waits for 1 there, opens its downstream routers,
  then the relay (persistent; never closed).
- Main passes no teardown word to the relay connection (nullptr), so the open writes 0 to L1 address 0 (§7 of the
  spec: observed, effect not established).

## 3. v4 design

The relay is one more downstream of VC0's receiver channel (`router_downstream.hpp`):
- `Relay`: a cursor (next slot, slots known free) and a link (ring base, NoC coordinate, the relay's free-slots
  register), the same shapes as a downstream router's; admission tests the known count and re-reads stream 30 only
  when it is used up; the forward is the stateless forward of a downstream router (shared helper).
- It lives in `Downstreams<0>` and `DownstreamLinks<0>` as members that exist only in UDM (`[[no_unique_address]]`
  empty types otherwise), so the receiver's signatures and every other config's code stay as they are.
- `receive`: under UDM, a packet that delivers here is admitted only with a relay slot, and delivery is the relay
  forward instead of `deliver` (forward first, then the relay, as main).
- Init: the relay args read after the teardown semaphores; before the handshake, stream 30 set to the relay's slot
  count and the link filled; after the downstream routers are opened (the existing wait on the Tensix sync word covers
  the relay), the relay connection opened with the same open sequence as a router's. Its teardown word is the relay
  connection's buffer-index word, which the host reserves on this router for the relay and nothing else uses, so the
  open's reset lands on this router's own word rather than L1 address 0.
- Gate: UDM admitted on one-ERISC 2D routers without VC1 or deadlock avoidance (what the host builds), with one local
  Tensix connection and mux mode off.

Nothing else moves: the sender side is unchanged (the UDM mux is an ordinary producer on the worker channel), the
completion proof covers the relay write by its transaction ID, teardown waits on the same IDs.

## 4. Validation

1. Offline: the real UDM configs build; every other config's image is byte-identical to v4rs's.
2. Device (P150x8), main's router as the same-job control: `fabric_unit_tests` `Fabric2DUDMModeFixture.*` (8) and
   `NightlyFabric2DUDMModeFixture.*` (68, 2x4 mesh), hang capture armed. The ttnn UDM gtests
   (`unit_tests_ttnn_udm`: copy, eltwise, reductions) once built.
