# v4 VC2: design note

Status: built and validated (2026-10-29; `variants/v4vc2`, jobs 141-142, 148, 152; RESULTS §98): the VC2 yaml 10/10 in both ERISC modes.
One change from §3 below, found on the device: with VC2, neither worker's credit uses the preset command buffer (the Blackhole
L1 inline write always issues on the write command buffer, so VC2's credits overwrote VC0's preset); both are whole commands. User order for the features: UDM (done, v4udm), VC2, speedy path, telemetry. Base: the lead,
`variants/v4udm`. Sources: main's router (`fabric_erisc_router.cpp` VC2 blocks, `fabric_erisc_router_speedy_path.hpp`),
the builders, V4_FEATURE_SPEC.md §6, the post-merge coverage case `bh_2d_vc2` (56 configs).

## 1. What VC2 is

`TT_METAL_ENABLE_FABRIC_VC2` (rtoption `enable_fabric_vc2`; test_tt_fabric's `use_vc2`): a third VC for neighbour
exchange on 2D Blackhole fabrics, not with UDM or the mux. Per router: one more worker channel (compact index
`VC2_LOCAL_CHANNEL_START` = VC0 + VC1 channel counts; fabric position `VC2_FABRIC_POSITION_START`), into the far router's
VC2 receiver channel (`VC2_RECEIVER_CHANNEL`: 1 without VC1, 2 with), which only delivers to its own chip (the host sets
`DISABLE_RX_CH2_FORWARDING`). Counter credits always. Workers connect through `append_fabric_vc2_connection_rt_args`;
the worker adapter's doorbell is the router's VC2 free-slots register, pinned at 30.

Configs (`bh_2d_vc2`): 2D mesh, both ERISC modes; VC0 4 sender channels + VC2 1 (channel 4, serviced on the sender ERISC,
no static connection, connection managed as channel 0's); 2 receiver channels (VC0, VC2 at index 1); VC2 free-slots
stream 30, VC2 packets-received stream 5.

## 2. Main's behavior

- Sender: `run_sender_channel_step_speedy` for the VC2 channel every pass (no bubble flow control, no first-level acks,
  credit amortization 1); the VC2 connection checked once per round, before the passes.
- Receiver: `run_receiver_channel_step_speedy` for the VC2 receiver: local delivery of every packet, completions in
  ping-pong transaction-ID batches, credited to `VC2_LOCAL_CHANNEL_START` whatever the header says.
- Two observations (not verified on hardware):
  - the VC2 receiver's transaction IDs alternate between `RX_CH_TRID_STARTS[2]` and 249 (V4_FEATURE_SPEC §6);
  - both workers' credits (VC0's and VC2's) go through one scratch word with no flush
    (`notify_worker_of_read_counter_update_src_address`, `enable_read_counter_update_noc_flush = false`), which the
    inline-write API leaves to the caller ("ensuring no outstanding inline write uses that address before issuing
    another write"): a VC0 credit still reading the word when a VC2 credit rewrites it carries VC2's count to the VC0
    worker.

## 3. v4 design

Sender:
- `Channel<CH>` learns VC2 (its far receiver, no first-level acks, one far slot per send).
- VC2's worker channel is visited every pass after VC0's and VC1's channels, by the worker visit (templated on the
  channel), with its own state (connection semaphore, counts, far ring): send, completions, credit, and the connection
  check every pass, as VC0's worker (the rule for worker connections; main checks VC2's once per round).
- Its free-slots register joins VC0's group read.
- Its credit is a whole command from its own source word (and, on routers with VC2, VC0's credit is a whole command
  too: both go on the write command buffer, so no preset can be kept there): the channel's
  `edm_read_counter` in its location-info block, which always holds the credits returned so far (what teardown writes
  there anyway), so no other credit shares it. Its connection events do not touch VC0's preset.
- The idle context switch counts VC2's sends as progress.

Receiver:
- VC2's receiver channel is a record as VC1's is, run after VC0 (and VC1): every packet is delivered here (no route, no
  admission), retired, and completed in groups as VC0's; credits go to the far channel named in the header (the far v4
  sender stamps its VC2 channel's index).
- Teardown waits on its transaction IDs.

Everything is compile-time: on routers without VC2 the new state is empty types and every other config's image should
stay byte-identical (checked offline).

Gate: VC2 on 2D routers without VC1 (what this host can run and check); VC2 with VC1 (multi-mesh) later.

## 4. Validation

1. Offline: the 56 VC2 configs build (both ERISC modes from one-ERISC captures too); every other config identical.
2. Device: `test_fabric_vc2_at_least_2x2_mesh.yaml` (data-checked VC2 traffic, VC2 with VC0 together, unidirectional),
   main's router as the same-job control; two ERISCs and one ERISC; hang capture armed.
