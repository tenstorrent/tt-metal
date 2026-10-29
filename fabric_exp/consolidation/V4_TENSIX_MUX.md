# v4 Tensix mux mode: design note

Status: draft (2026-10-28). User: go ahead; VC1 with mux mode stays out of scope. Sources: the merged tree
(`fabric_router_mux_extension.cpp`, `fabric_tensix_builder{,_impl}.cpp`, `compute_mesh_router_builder.cpp`,
`erisc_datamover_builder.cpp`, `fabric_static_sized_channels_allocator.cpp`, `fabric_erisc_router.cpp`),
`variants/v4se` (V4_SINGLE_ERISC.md) and RESULTS §98 ("Standalone mux on v4", job 125).

## 1. What Tensix mux mode is

Two different muxes share the name:

| | standalone mux | Tensix mux mode |
| --- | --- | --- |
| kernel | `tt_fabric_mux.cpp` (and v2) | `fabric_router_mux_extension.cpp` |
| launched by | an op, on its own Tensix cores | the fabric, one per router (not on dispatch links), BRISC of a Tensix core |
| to the router | a worker on the worker channel | the router's only sender: everything for the link comes through it |
| fabric | unchanged (two ERISCs on Blackhole) | `FabricTensixConfig::MUX`; one ERISC forced |
| v4 | works as is (job 125, every mode) | this note |

The fabric's mux has worker channels (local workers connect to it instead of to the router) and router channels (1D
one, 2D three: one per upstream direction, persistent). It feeds its own router's worker channel as one ordinary
worker connection, a packet at a time with a flush after each, so in this mode the mux, not the router, sets the
throughput. On rings and tori it applies bubble flow control to its injection channels (two free router slots).

## 2. What the mode changes for the router (main)

1. **Senders: the worker channel only.** The allocator's MUX branch gives slots to VC0's channel 0 only, and only it
   is serviced (`erisc_datamover_builder.cpp` "MUX mode: Only worker channel serviced").
2. **Receivers forward into the downstream direction's mux**, not into the downstream router: the builder connects the
   receiver to the Tensix builder of each direction (`get_builder_for_vc_channel`: the extension takes all of VC0). The
   connection spec is the mux's router channel: its base, depth (power of two, at most 128, sized from Tensix L1),
   location-info and handshake addresses. Same protocol as router to router: the forward decrements the channel's
   free-slots stream register on the mux (the stream ids are the router's `SENDER_FREE_SLOTS` positions 1..3 by
   construction), the mux returns credit by incrementing the router's local copy, the persistent connection opens with
   the same handshake. The slot size is the router's (header + max payload, 4448 B).
3. **Credits go to channel 0.** The far router sends everything from its worker channel, so the receiver returns every
   credit to channel 0 and the sender does not stamp `src_ch_id` (`skip_src_ch_id_update`). The header's field is then
   stale (whatever the packet's previous hop wrote), so the receiver must not read it.
4. **Startup: wait for the muxes.** Each mux increments its upstream routers' `EDM_LOCAL_TENSIX_SYNC_PTR_ADDR` once its
   channels are set up; a router waits for `NUM_DS_OR_LOCAL_TENSIX_CONNECTIONS` before opening its downstream
   connections (so the open is not wiped by the mux's setup). The host clears the word before launch
   (`get_fabric_router_addresses_to_clear`), so the router only waits on it. The mux in turn waits for its router's
   `READY_FOR_TRAFFIC` before opening its worker connection; no cycle.
5. **One ERISC.** `is_fabric_two_erisc_enabled` is false with any Tensix extension ("out of stack size"): the router
   runs on ERISC 1. VC2 is off in MUX and UDM; express with MUX is a `TT_FATAL`.
6. **VC1 is not rejected, and does not work.** The VC1 decision ignores the Tensix mode (two meshes with links turn VC1
   on); VC1 is wired router to router around the mux, but the MUX slot branch gives VC1 sender channels no slots. Nothing
   tests it.

## 3. What v4 has, and what blocks it

| item | v4 (v4se) |
| --- | --- |
| one ERISC | done (V4_SINGLE_ERISC.md; ERISC 1 is the `fab1` placement, jobs 122-123) |
| sender: no `src_ch_id` stamp under `skip_src_ch_id_update` | done (`router_sender.hpp`) |
| sender: the worker channel only | the channel masks come from the host's serviced flags, so forwarding masks are empty; to confirm it compiles |
| forwards into a mux | the host's downstream args (ring base, location info, registration, NoC coordinate, depth) already describe the mux; v4's compile-time remote credit stream (direction table) equals the mux's stream id; one depth per VC holds (all muxes share one config) |
| forwarding command buffers kept programmed | off on one ERISC (`command_buffer_presets`) |
| receiver: credit to channel 0 | **blocker**: `static_assert(!skip_src_ch_id_update)` in `router_receiver.hpp` |
| wait for the downstream muxes | **missing** |
| bring-up gate | refuses `num_ds_or_local_tensix_connections != 0` |

## 4. Proposed design

### 4.1 The source channel of a packet

One helper in the receiver, used where credits are attributed (first-level acks and completions):

    source_channel(header field) = skip_src_ch_id_update ? worker channel (0) : header field

A compile-time choice, so a router with the field folds to today's code and a mux-mode router never reads the stale
field. The name says what it is (the far sender channel), not how main's flag is spelled.

### 4.2 Startup: wait for the downstream muxes

A named helper in `router_init.hpp`, `wait_for_downstream_muxes()`: when `num_ds_or_local_tensix_connections != 0`,
wait until the sync word reaches that count (or a termination signal), keeping the heartbeat/context-switch behavior of
the other startup waits. Called by the receiver role after the handshakes and before `open_downstreams`, as main.

### 4.3 Senders: the worker channel only

Nothing new: with only channel 0 serviced, `vc0_forwarding` and `vc1_forwarding` are empty, there are no upstream
routers to wait for, and the sender pass is the worker channel's. A static_assert records the expectation (mux mode
=> worker channel only), so a host change that gives a mux-mode router forwarding channels fails at build.

### 4.4 Gate

Lift `num_ds_or_local_tensix_connections == 0` for mux mode on one ERISC without VC1 and UDM:

    !tensix mux mode || (NUM_ACTIVE_ERISCS == 1 && !has_vc1 && !udm_mode)

with VC1 + mux mode a static_assert of its own ("VC1 with the Tensix mux extension is not supported"), placed before
the channel templates so that message comes first (their slot asserts would otherwise fire on VC1's zero slots). A
host-side rejection belongs with the builder change that makes v4 the router; not added to main's builder now.

### 4.5 What stays as main

- Bubble flow control on rings follows the host's injection flags (the router's one channel as the host marks it; the
  mux applies its own for its injection channels).
- Dispatch links (no mux on that router, credits per channel there): only with dispatch tunnels; not on this host.
- Throughput: not a goal here; the mux bounds it. Goal: functional in mux mode, nothing moves elsewhere.

## 5. Validation plan

1. Offline: every logged non-mux config builds byte-identical to v4se (the changes are compile-time no-ops there); if
   not, the two-ERISC perf gate (job 121's suites) decides.
2. Device, one job, hang capture armed, main's router first as the same-job control:
   - `fabric_exp/v4/val_mux.yaml`: the mux-mode cases of `test_fabric_sanity_at_least_2x2_mesh.yaml` (linear, full-ring
     and mesh multicast, 3 ntypes; the ring deadlock check; 19 runs on P150x8; main passed all in jobs 119-120);
   - `fabric_unit_tests`: `Fabric1DTensixFixture.*` (4, in the merge gate's `Fabric1D*Fixture.*`),
     `NightlyFabric1DTensixFixture.*`, `NightlyFabric2DTensixFixture.*` (1 + 1, nightly T3K);
   - `*FabricMuxV2Smoke*Fixture.*` (standalone mux v2, merge gate) on v4 as a side check.
3. Regression: v4's single-ERISC validation (val1d_m1, ring, Y-torus) and the two-ERISC gate if step 1 is not identical.

## 6. Decisions

- VC1 with mux mode: out of scope (user, 2026-10-28); v4 rejects it at build with its own message.
- UDM (`FabricTensixConfig::UDM`, the relay): a separate feature, next.
- Two ERISCs with mux mode: not now (the host forces one); revisit with the stack question if it matters.
