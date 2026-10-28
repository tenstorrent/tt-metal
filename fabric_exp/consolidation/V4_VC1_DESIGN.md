# v4 VC1 design note (draft for review, 2026-10-27)

Scope: bring VC1 (and the intermesh routing it comes with) into the v4 router, in a shape that the rest of the feature
set (VC2, express/Z, the one-ERISC program for mux, UDM and Wormhole) can join without reshaping the loops again. The
user's goal for mainlining is one router implementation with the full feature set, so this is the first of the
loop-shape features, done before further tuning (tuning a one-VC loop and then adding channels would redo it).

Sources: `V4_FEATURE_SPEC.md` §5 (intermesh, VC1), §2 (first-level ack), §4 (express), §6 (VC2); `SHAPE_NOTE.md` (v4's
principles P-A..P-E); `SHAPE_INVENTORY.md`; RESULTS §84-§95 (v3 on VC1 routers), §98 (v4); the `bh_vc1` coverage case
(mock multi-mesh, 90 router builds).

## 1. What VC1 is, where it exists

VC1 exists only in multi-mesh fabrics (2+ meshes joined by intermesh links). A packet crosses meshes on VC0 until it
reaches an intermesh router, which moves it onto VC1 (crossover); inside the destination mesh it travels on VC1, so
intermesh traffic never shares buffers with intramesh traffic (deadlock freedom across meshes).

Router kinds in the P150 two-mesh split (`bh_vc1`; every chip of a 2x2 mesh has intermesh links):

| Router | Defines / flags | Sender channels | Receiver channels | Downstreams |
| --- | --- | --- | --- | --- |
| Intramesh, on an edge chip | `FABRIC_2D_VC1_ACTIVE`, `FABRIC_2D_VC1_SERVICED`, `IS_INTRAMESH_ROUTER_ON_EDGE` | VC0: worker + 3 forwarding; VC1: 3 forwarding | 2 (VC0, VC1) | 2-3 per VC |
| Intermesh | `FABRIC_2D_VC1_ACTIVE`, `FABRIC_2D_VC0_CROSSOVER_TO_VC1`, `IS_INTERMESH_ROUTER`, `IS_INTERMESH_ROUTER_ON_EDGE`, both receivers intermesh ingress | same | 2; receiver 1 is flagged serviced (so its VC1 downstream connections are opened: crossover forwards into them), but without VC1 pass-through no VC1 traffic comes to it and main does not run it (no `FABRIC_2D_VC1_SERVICED`) | 2-3 per VC |

(Checked 2026-10-27 in main's compiled arguments on the split: 64 intramesh and 32 intermesh router builds; every E1
has `IS_RECEIVER_CHANNEL_{0,1}_SERVICED` = 1. `IS_INTERMESH_ROUTER_ON_EDGE` only turns off main's stateful forwards.)

Larger multi-mesh fabrics add intramesh routers inside a mesh (VC1 serviced, no exit) and intermesh routers that also
pass VC1 through (both receivers forwarding into the same VC1 downstreams). Express adds a fourth VC1 sender channel.

Facts that shape the design:
- **No worker on VC1.** VC1 sender channels are all router-to-router (indices `VC1_LOCAL_CHANNEL_START =
  ACTUAL_VC0_SENDER_CHANNELS` onward), so "one worker per router" holds.
- **No first-level ack, no bubble flow control on VC1** (host: VC1 ack always 0; deadlock avoidance never on intermesh
  routers).
- **Credits:** counter credits on Blackhole with two TX queues (`vc1_uses_counters = multi_txq || express`). The credit
  block (ack and completion counts per sender channel, all 7 channels) is one contiguous L1 block sent whole: one
  frame per pass can carry every receiver channel's credits.
- **Transaction IDs:** 4 per receiver channel in VC1 builds (channel 0 ids 0-3, channel 1 ids 4-7).
- **Slots (main's allocator, BH, 81 slots of 4448 B):** the VC0+VC1 tier `{4, 8, 2, 4}` plus the spare slots to the
  worker: VC0 worker 51, VC0 forwarding 4 each, VC0 receiver 8; VC1 forwarding 2 each, VC1 receiver 4. Every
  forwarding channel is tiny.
- **Opt level:** the builder compiles a router with a VC1 receiver at -Os, because main's E0 + E1 did not fit the
  26,624 B kernel config buffer at -O3 (29.6 KB). v4's VC0-only builds are 20.0-21.4 KB at -O3 today.
- **Shared downstreams:** with crossover, receiver channel 0 forwards through the VC1 downstreams; when VC1 is also
  serviced, receiver channels 0 and 1 forward through the same VC1 downstream objects (same cursors, same cached
  credits), so both must run on the same ERISC (E1).

## 2. Baseline and lessons

VC0 traffic on a VC1 router (two 2x2 meshes, cycles/packet; VC0-only routers run the same rows at ~125-132):

| Row | main-like v3a -Os | v3l -O3 (best v3) |
| --- | ---: | ---: |
| 1-hop E/W write 16 B / 4352 B | 233.7 / 258.5 | 199.1 / 224.5 |
| 1-hop N/S write 16 B / 4352 B | 236.2 / 266.3 | 200.8 / 226.2 |

- **E0 sets that rate** (RESULTS §86): it visits 7 sender channels every pass, and the VC1 channels' free-slot reads
  were not grouped and used at once. Not slots: v3l took 31 off with the same slots.
- **-Os costs ~20%, all of it E1's** (§88). VC1 routers run at -Os in production today.
- **Bugs v3 found, to stay fixed:** VC1 admission must read VC1's own downstream credits (v2/v3a read VC0's, §87);
  a channel holds a packet when its free count is anything but its ring size, never "below" (a 2-slot VC1 channel read
  "idle" forever with a one-bit test; the VC1 hangs of §90-§93).
- **Hangs** in v3 VC1 runs were rare (~3 in 110 full rounds, all at 4 KiB all-to-all) and were bugs; VC1 runs keep the
  hang capture armed (`run_vc1_cap`).
- **Inter-mesh traffic cannot be benchmarked today:** the test's BandwidthProfiler aborts on inter-mesh benchmark
  patterns, so VC1 traffic itself is only exercised by the sanity suites (§84). Intra-mesh rows on VC1 routers can be
  benchmarked.

## 3. Principles (v4's, plus three for VC1)

P-A..P-E from the shape note hold: each role is a loop function, per-round work out of line; state placement is
explicit; features are compile-time components; NoC and Ethernet only through APIs; keep measured optimizations in
their measured form. For VC1:

- **V-1. VC0 must not pay for an idle VC1** beyond ~2 cycles/packet on either ERISC. VC1 code exists only in builds
  with VC1 (compile-time channel lists), and when VC1 has no work a pass spends one grouped register read and one test
  on it.
- **V-2. One set of components per channel kind, instantiated per channel.** No VC1 copy of VC0 code, no `if (vc == 1)`
  inside a step. A receiver channel is a type with its traits (ring, VC, ack, batch, downstream VC, intermesh ingress);
  the pass instantiates the step per channel.
- **V-3. Steps are functions of explicit state, not of one loop's locals,** so the one-ERISC program (both roles in
  one loop: mux, UDM, Wormhole) composes the same steps later.

## 4. Design

### 4.1 Configuration

`router_config.hpp` (or today's `Channel<CH>` traits, extended) gives, per build:

```cpp
using Vc0Senders = ChannelList<0, 1, 2, 3>;          // worker + forwarding (1D: <0, 1>; express: <0, 1, 2, 3, 4>)
using Vc1Senders = ChannelList<4, 5, 6>;             // empty without VC1; + 7 with express
using Receivers  = ChannelList<0, 1>;                 // VC0, VC1 (VC2 later: 2)
template <size_t RX> struct ReceiverTraits {          // ring bounds, packets-received stream, trid range, VC,
    static constexpr size_t vc = ...;                 //   downstream VC (VC1 for VC0 with crossover),
    static constexpr bool first_level_ack = ...;      //   first-level ack, completion policy, intermesh ingress
    using Completions = select_completions<...>;      // batched (no acks, ring >= 8) or single slots (acks, or small ring)
    static constexpr bool intermesh_ingress = ...;    // landing rewrite before decode
};
```

All from named compile-time args the host already emits (`ACTUAL_VC1_SENDER_CHANNELS`, `VC1_RECEIVER_CHANNEL`,
`IS_RECEIVER_CHANNEL_i_INTERMESH_INGRESS`, the per-VC downstream tables).

### 4.2 Sender program (E0)

v4's pass today: one grouped read of the VC0 free-slots registers, the worker visit, and the forwarding channels
behind one test of "packets or packets in flight". VC1 adds a second VC of forwarding channels to the same shape:

```cpp
for (pass ...) {
    const uint32_t pending0 = channels_with_packets<Vc0Senders>();      // one group of 4 reads (as today)
    visit_worker(worker, pending0, far_vc0, ...);                       // as today
    if (((pending0 | table.in_flight) & vc0_forwarding) != 0) {
        visit_forwarding_channels<vc0_forwarding, VC0_RECEIVER>(table, pending0, far_vc0, ...);
    }
    if constexpr (has_vc1) {
        const uint32_t pending1 = channels_with_packets<Vc1Senders>();  // a second group: 3 reads (4 with express)
        if (((pending1 | table.in_flight) & vc1_forwarding) != 0) {
            visit_forwarding_channels<vc1_forwarding, VC1_RECEIVER>(table, pending1, far_vc1, ...);
        }
    }
}
```

- **Far ring per VC** (`FarRing far_vc1`): the far router's VC1 receiver ring, its own free count and cursor. Two
  more loop-carried values on a VC1 build.
- **Forwarding records** stay in the one table (already indexed by sender channel, `NUM_SENDER_CHANNELS` entries) with
  one in-flight mask for both VCs.
- **Idle VC1 costs one grouped read (~3-4 cycles) and one test per pass** (V-1). v3 paid ~27 per empty channel visit;
  this is where v3's VC1 routers lost 75-110 cycles/packet against VC0-only routers.
- **Upstream credits** for VC1 channels go to the upstream routers' VC1 free-slots registers (the producer address
  cached at connection, as for VC0). No worker credit on VC1.
- **Measured question:** whether reading VC1's group every pass costs more than reading it every other pass. Not
  planned unless the idle cost shows up.

### 4.3 Receiver program (E1)

Today: one receiver channel, its state as loop locals, one credit frame per pass. With VC1:

```cpp
for (pass ...) {
    [first-level-ack VCs: the group read + acks, as today]
    const bool progress0 = receive<0>(rx0, downstreams_vc0_or_vc1, ...);  // as today, VC0 state in locals
    frame_due |= rx0.completions.after_packet(progress0, ...);
    if constexpr (has_vc1_receiver) {
        frame_due |= receive_and_complete<1>(rx1_state, downstreams_vc1, ...);  // VC1 state in a record (see below)
    }
    if (frame_due) frame_due = !send_credit_frame(...);                    // one frame for both channels
}
```

- **State placement.** VC0's per-packet state stays as loop locals (as today). VC1's (position, counts) lives in a
  small record read and written once per VC1 step, explicitly (P-B), because E1's loop is already at the register
  limit and adding a second channel's live values would spill VC0's (v4p: two loop-carried values cost +12 stack loads,
  +8 cycles). If VC1 traffic is heavy this costs VC1 a few cycles per packet; it keeps VC0 untouched (V-1). Measured
  both ways before deciding.
- **Polling VC1.** Its packets-received register is read only when its known count is used up (the lower-bound rule
  VC0 uses), so an idle VC1 costs one register read per pass; grouped with VC0's read where both are read in the same
  pass.
- **One credit frame per pass for all channels.** Each channel's completions (and VC0's acks) raise the counts in the
  shared block; the pass sends one frame when any changed. Two frames per pass would double E1's TX-queue work.
- **Completions per channel, one component.** A receiver channel owns a completion component chosen at compile time:
  batched (no acks, ring of 8 or more: VC0 on a VC1 build has 8 slots, batch 2) or single slots (first-level acks, or
  a ring under 8: VC1's 4 slots). This is where the `release_completions` rewrite lands (designs A/C, discussed
  2026-10-27): separate due / proof / credit steps, one path for any batch size, the proof as a grouped read of the
  batch's transaction-ID status registers instead of the command-buffer bookkeeping. The single-slot timing rule
  (hold the newest slot or credit it at once) follows the Galaxy result of job 158562; on a 4-slot VC1 ring, holding
  one slot holds a quarter of the ring (LEDGER E1-27: VC1 rows +30 to +45 from lone held credits), so VC1 likely wants
  the eager rule regardless.
- **Downstreams per VC, shared where the protocol shares them.** E1 holds one `Downstreams<VC0>` and one
  `Downstreams<VC1>`; a receiver channel forwards into the one its traits name (VC1 for VC0 with crossover), so with
  crossover and pass-through both channels use the same VC1 object. VC1 forwards are stateless (several live
  downstreams; the stateful single-downstream path and its paired checks apply only where `forwards_are_stateful`).
  Admission reads the channel's own VC's credit registers (the v2/v3a bug cannot come back: the VC is a type parameter).
- **Intermesh routing** is part of the 2D routing policy, not branches in the step:
  - landing rewrite (intermesh ingress channels): before decode, out of line (`fabric_set_2d_intermesh_landing_route`
    rebuilds the route maps; every packet on those channels pays it, so it is measured, not assumed rare);
  - exit (intramesh routers on an edge chip): after decode, `deliver-here && destination mesh != my mesh` → one
    stateless forward toward the intermesh direction from the routing table, no local delivery; an unaddressable
    direction is an assert, as main.
- **Local delivery** code is shared by both channels (one out-of-line body for the rare send types, as today; the
  common arms inline), so VC1 does not duplicate it.
- **Transaction IDs.** Each channel's tracker uses its own range (`RX_CH_TRID_STARTS`); with 4 ids and an 8-slot VC0
  ring two slots share an id (the slot-index mask path), which only makes a completion check conservative.

### 4.4 Allocation (host)

Main's VC0+VC1 tier gives every forwarding channel 4 (VC0) or 2 (VC1) slots and the worker 51. From today's sweeps a
2D forwarding channel's credit loop is ~2,200-2,400 cycles (RESULTS §98 slot sweep), so 4 slots cap a multi-hop VC0
flow near 550-600 cycles/packet; 51 worker slots are far past what the worker loop needs (~18). The computed
allocation (receiver depths a power of 2, forwarding channels sized to their credit loop, the rest to the worker,
`TT_FABRIC_EXP_SLOTS*` to sweep) extends to VC1 builds with a per-VC split. Needs data from the multi-mesh harness
first; the intra-mesh rows on VC1 routers measure the VC0 side, VC1's side needs the benchmark gap closed (§5).

### 4.5 Opt level and code size

- Target: VC1 builds at -O3. v4 VC0-only builds are 20.0-21.4 KB (E0 + E1, largest of each) against the 26,624 B
  buffer; VC1 adds E0's second channel group and E1's second channel. Measured offline on `bh_vc1` at -O3, -O2, -Os
  before any hardware run.
- The host picks -Os for any router with a VC1 receiver (`get_kernel_opt_level`). If v4 fits at -O3 with margin
  (≥ ~1 KB), the builder picks -O3 for the v4 kernel; if not, -O2 (v3: -O2 fit, -O3 did not), and the -Os penalty is
  measured on v4 (v3's was E1's, ~20%).

### 4.6 What this keeps open for the rest of the feature set

- **VC2:** a third receiver channel (local delivery only, its own completion component) and one more worker-type
  sender channel; the same lists and steps (V-2).
- **Express / Z:** a fifth VC0 sender channel and fourth VC1 one; VC0's free-slots group splits 4 + 1 (a fifth read in
  one group costs ~14, §17), VC1's group becomes 4.
- **One-ERISC program (mux, UDM, Wormhole, BH one-ERISC):** both passes in one loop, from the same step functions
  (V-3). Its register budget is both roles together, so the per-channel records of §4.3 matter even more there.
- **Stream-register credits** (single TX queue: Wormhole, BH VC1 without two TX queues): a credit-transport component
  with the same two operations the completions use today (`accumulate`, `flush`): counters raise the shared block and
  send it once per pass; stream credits write each credit to the far register at `accumulate` and `flush` does nothing.
  Main already has both behind that interface.

## 5. Validation and measurement

0. **Harness on this host.** Already brought up on a09u32 (RESULTS §98 "Host move", jobs 024-027): same cube wiring,
   the rank bindings unchanged, VC1 34/34 at every level, v3q/v3z/v3q16 VC1 baselines. Added 2026-10-27: a mock
   mapping for this host (`vc1/a09u32_mock_mapping.yaml`, mock run placing both meshes and compiling the fabric with
   main's router) and main's router (v1, stock) on the split suite as the production reference.
1. **Offline:** `bh_vc1` (+ `_allinstr`, `_datacache`, `_nofuse`, `_1erisc` once the one-ERISC program exists)
   builds at -O3, -O2, -Os; code size against the buffer; stack check.
2. **Hardware, data-checked:** the multi-mesh sanity suite (all-to-all intra and inter, 4 packet types x 1 and 4 KiB,
   inter-mesh neighbor exchange) at the chosen opt level and at -Os, with hang capture; then the VC0-only suites
   (mesh, 1D, ring, torus validation) to confirm VC0 builds are unchanged.
3. **Performance:** intra-mesh rows on VC1 routers (1-hop E/W and N/S 16 B and 4352 B, multicast) against main and
   v3's numbers above; VC0-only suites against the current lead (V-1: within ~2 cycles).
4. **VC1 traffic performance** needs the benchmark gap closed: fix the BandwidthProfiler's inter-mesh abort in the
   test infra (study tree) or time inter-mesh flows another way. Proposed as its own small step.

## 6. Steps

1. Harness: main's router on the split suite on a09u32 (the harness itself was brought up 2026-10-26).
2. E0: VC1 forwarding channels (second free-slots group, far ring per VC, records in the shared table).
3. E1: second receiver channel (record state, shared frame, per-channel completion component with the existing
   policies), downstreams per VC, crossover; then intermesh landing and exit.
4. Offline builds and code size; hardware validation; performance against main and v3.
5. The completion component rewrite (A/C) and the single-slot timing rule, measured on VC0-only and VC1 builds.
6. Allocation for VC1 builds; the inter-mesh benchmark step.

## 6a. Implementation status (2026-10-27): `variants/v4vc`

Steps 2 and 3 as one variant (the bring-up gate admits VC1 on both ERISCs together):
- **E0:** `vc1_forwarding` channels (all forwarding), `far_vc1`, a second grouped free-slots read and one test per pass,
  records in the shared table.
- **E1:** `ChannelState<1>` record for VC1 (`receive_and_complete`), one counter credit sender for both channels (one
  frame per pass), `Downstreams<0>` and `Downstreams<1>` with crossover picking VC1's for channel 0; landing
  (`Routing2D<VC, INGRESS>::land`, out of line, Y-first decode after it) and exits (`exit_action`, out of line; the
  destination mesh read with the action bytes on edge routers; edge and intermesh routers have N/S live on E/W).
  An exit toward an unaddressable direction asserts and drops the packet (main fail-stops; dropping keeps the check on
  the rare branch).
- **Init:** VC1 runtime args; VC1 downstreams opened when receiver 1 is serviced (as main); the credit block zeroed for
  every sender channel (main and v4 zeroed `NUM_RECEIVER_CHANNELS` entries of a block indexed by sender channel); the
  per-entry init work (connect-request poll, downstream open) in one out-of-line body each and the handshakes at -Os,
  which brought VC1 E0 + E1 at -O3 from 29,972 B to 24,424 B (-O2 23,300, -Os 22,356; limit 26,624).
- **VC0-only builds:** E0's loop is v4uf's code in all 136 configs; E1's differs only in the setup before the loop.
- **Hardware (jobs 091, 092; RESULTS §98):** the VC1 split suite 34/34 at -O3, -O2, -Os; VC0-only suites within 0.1
  of v4uf. VC1-router rows at -O3: 1-hop 187-201 (main ~305, v3q 175-192), multicast 588 (main 549, v3q 558).
  PC sampling: those rows are bound by the credit round trip on the VC1 tier's small rings (VC0 receiver 8,
  forwarding 4), with E0's pass time a term in it; v4vc's E0 turns credits around ~10 cycles slower than v3q's
  (longer group-read waits, TX-queue writes in `send`). Levers: step 6 (allocation) and E0's per-pass cost.

## 7. Decisions for review

1. **VC1 state on E1 in a record** (VC0 untouched, VC1 pays a few cycles) vs both channels as loop locals (fastest if
   it fits, risks VC0). Proposed: record first, measure the locals form.
2. **Opt level for v4 VC1 builds:** -O3 if it fits with ~1 KB margin, else -O2 (host change in `get_kernel_opt_level`
   keyed on the kernel).
3. **The completion rewrite inside the VC1 work** (step 5) rather than before it.
4. **Closing the inter-mesh benchmark gap** in the test infra (study tree), so VC1 traffic itself can be measured.
5. **Allocation for VC1 builds** waits for harness data (step 6), with the computed rule as the target.
