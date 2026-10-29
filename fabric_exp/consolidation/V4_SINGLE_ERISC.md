# v4 single-ERISC: design note

Status: draft, revised after review (2026-10-28: user agrees with most; main's single-ERISC router is the reference
for what must work, not for how: re-derive its choices from the new router's principles). Nothing built yet. Sources: V4_FEATURE_SPEC.md §9 (main's single-ERISC router),
RESULTS §98 "Single-ERISC baseline" (jobs 119-120), the merged tree (`erisc_datamover_builder.cpp`,
`fabric_builder_context.cpp`) and `variants/v4gi_m`.

## 1. What single-ERISC is

One ERISC per router runs both the sender channels and the receiver channels. Wormhole always; Blackhole when:

| switch | scope | the router runs on | context switching |
| --- | --- | --- | --- |
| `TT_METAL_DISABLE_MULTI_AERISC=1` | the whole system has one active ERISC | ERISC 0, with the base firmware | the master's idle hand-off; on BH ERISC 0 a link-status check |
| `TT_METAL_DISABLE_FABRIC_TWO_ERISC=1` | fabric only; the system stays two-ERISC | ERISC 1 (`CMRB:952-958`) | none (`run_routing` compiles away there) |
| a tensix extension (mux, UDM) | forced (ERISC 0 stack size) | ERISC 1 | none |

CI runs fabric single-ERISC on Blackhole only as `TT_METAL_DISABLE_MULTI_AERISC=1 fabric_unit_tests
--gtest_filter="Fabric1D*Fixture.*"` (blackhole_e2e and fabric_sanity pipelines), plus the mux tests.

Main's router there (jobs 119-120, both switches, all clean: mesh 96, 1D 185, VC1 34/34, CI leg 24 + 7 skipped):
one-way 1-hop writes +11 to +40 cycles/packet over two-ERISC, bidirectional / multicast / 3-hop +70 to +230; suite
means 2D +114, 1D +117, VC1 +151; ERISC 0 (with the base firmware) 0-22 slower than ERISC 1.

## 2. What one ERISC changes

1. **Both roles in one loop.** One core does the sender pass and the receiver pass. One-way traffic loads each end
   with one role (the other's pass finds nothing), so it should stay near two-ERISC rates; anything that loads both
   directions or forwards through a router (bidirectional, multicast, multi-hop) puts both roles' per-packet work on
   one core. That is main's +11..40 against +130..230.
2. **One TX queue.** `sender_txq_id == receiver_txq_id == 0` (the builder sets TXQ 1 only with two ERISCs, `EDB:299-301`):
   data sends, the packets-sent register writes and the credits all wait for TXQ 0 to be free.
3. **Credits.** The host's credit plan (`fabric_builder_context.cpp:44-60`) uses counter credits (an L1 block sent as
   one Ethernet frame) for VC0 only with two TX queues, and stream-register credits (one remote register write per
   credit) otherwise. Two TX queues need counters because TXQ 1 cannot write remote registers (`ETHR:55`); one TX
   queue allows either. v4 implements counter credits only (`static_assert(receiver_channel_uses_counter_credits(CH))`).
   The counter area is reserved in L1 in every mode; the host clears it only with two TX queues (a wider clear in
   single-TXQ mode changes UMD's Ethernet commands during bring-up, `EDB:317-330`).
4. **One NoC, four command buffers (Blackhole).** `requires_forced_assignment_to_noc1()` (BH, one RISC core) puts all
   router NoC traffic on NoC 1: the base firmware on ERISC 0 may use NoC 0 for link health. Command buffers are per NoC
   (0 WR, 1 RD, 2 WR_REG, 3 AT on Blackhole), so the router gets NoC 1's four. Two-ERISC v4 spreads its users over
   two NoCs:

   | user | role | command buffer (two ERISCs) | state |
   | --- | --- | --- | --- |
   | worker credit (4-byte posted write of the count) | sender, NoC 0 | WR (`write_cmd_buf`) | preset per connection |
   | upstream credits, teardown | sender, NoC 0 | AT (inline writes) | none |
   | local writes | receiver, NoC 1 | WR (`local_chip_data_cmd_buf`) | preset (`noc_write_init_state`) |
   | forward data | receiver, NoC 1 | WR_REG | preset per downstream (VC0) |
   | forward sync (packets-sent register write) | receiver, NoC 1 | RD | preset (VC0) |

   On one NoC that is five users for four buffers, and two presets on WR. Main's single-ERISC router gives up
   stateful forwarding and command-buffer pre-programming altogether (`V3:419-424, 581`).
5. **Code size.** One binary holds both programs. v4gi today (largest per case, -O3): 2D E0 10.6 + E1 6.5 KB, 1D 8.7
   + 6.3, torus 10.9 + 6.3, VC1 13.1 + 10.8 KB; the sum is an upper bound (init code is shared), within Blackhole's
   26,624 B kernel buffer in every case. Wormhole keeps router `.text` in IRAM, where main's VC1 build already needs
   -Os: to measure before Wormhole bring-up.

## 2a. Main's single-ERISC choices: constraint or choice

Main's router is the functional reference (what single-ERISC must support), not the design reference. Each of its
single-ERISC choices, sorted into what the hardware, firmware or host forces and what main chose for its own loop
shape:

| main does | kind | v4's position |
| --- | --- | --- |
| one generic loop, channels interleaved one by one (sender 0, receiver 0, sender 1, ...), a TX-queue check per operation | choice | two role passes, each as lean as its two-ERISC loop (grouped register reads, one test for idle channels, rare work out of line); the TX-queue status read once per pass and shared by both passes |
| one TX queue | choice, by default: the builder gives TXQ 1 only to two-ERISC routers, behind a workaround for a two-TXQ hang (issue #29073, closed 2026-01-09; two TX queues run in every two-ERISC build today) | to check on hardware (a probe with hang risk: asked first): TXQ 0 for data, TXQ 1 for credit frames, as two-ERISC v4; then credit frames never wait behind a data packet's serialization |
| stream-register credits on VC0 | host coupling: counters only with two TX queues | counter frames in every mode (decision 1) |
| all router NoC traffic on NoC 1 (BH) | constraint where the base firmware runs beside the router: a router on ERISC 1 (`DISABLE_FABRIC_TWO_ERISC`, mux, UDM) next to the base firmware on ERISC 0. Main applies it to every one-ERISC BH router | a router on ERISC 0 (`DISABLE_MULTI_AERISC`) is where two-ERISC v4's sender already runs on NoC 0 beside the base firmware, redoing its presets when the firmware ran: keep v4's two-NoC layout there (sender side NoC 0, receiver side NoC 1), all presets; NoC 1 only on ERISC 1 (section 4.3) |
| no stateful forwarding, no command-buffer pre-programming | consequence of sharing one NoC's buffers | presets kept wherever a buffer has one owner |
| idle context switch: BH ERISC 0 a link-status check, WH a real switch to the base firmware | constraint (firmware) | `end_round` as today, presets redone when the firmware ran |
| Wormhole-only paths (TX-busy check, posted connection writes, IRAM) | constraint (hardware) | carried over |

The rule behind "no RX work on E0" (the sender's hot path stays lean) holds within the shared core: the receiver's
work stays in its own pass, never inside the sender's send path.

## 3. What v4 already has, and what blocks it

Has: roles come from the serviced channels (`is_sender_role`, `is_receiver_role`), so one ERISC is both; `sync_erisc`
compiles away with `NUM_ACTIVE_ERISCS == 1`; init and teardown are one skeleton with each role's part guarded by its
role, so they run both parts in sequence; per-round work (termination, router commands, heartbeat, context switch) is
out of line in `end_round`, which reports when the base firmware ran so presets can be redone (the sender does this
for the worker credit today).

Blocks: the bring-up gate (`NUM_ACTIVE_ERISCS == 2 && multi_txq_enabled && is_sender_role != is_receiver_role`);
`kernel_main` runs either `sender::run` or `receiver::run`, each its own endless loop with its state as loop locals;
counter credits only; the command-buffer layout of section 2.4.

## 4. Proposed design

### 4.1 One loop of two passes

Split each role's loop body into a pass over its state:

```
sender:   struct State {...};  noinline bool pass(State&);      // one pass of today's loop body
receiver: struct State {...};  noinline bool pass(State&);
two ERISCs (unchanged):  sender::run / receiver::run keep their own loop with the state as locals, calling the
                         pass inline — the two-ERISC binaries must come out byte-identical (checked offline)
one ERISC:               single::run: while (next_round()) { for (passes) { receiver::pass(r); sender::pass(s); }
                                                            end_round(...) once for both }
```

Each pass keeps its state in its struct (memory, passed by reference) and its hot values in registers within the
pass, so neither role's loop has to fit the other's live state: E0's 2D loop is already ~700 instructions with ~75
stack loads at -O2, and one fused loop would spill both. This is the shape note's explicit state placement (constants
compile-time, per-connection state in tables, per-pass values as locals), not a compiler-dependent inline.

### 4.2 Credits: counter frames

Keep v4's counter credits, on TXQ 0 with the data or on TXQ 1 alone (decision 7): the receiver pass sends the credit block as one frame when the
counts changed and TXQ 0 is free, else on the next pass (as today). One credit path for every build, one frame per
pass for all channels (batched completions stay worth the same), and no stream registers spent on completions (the
core has 32; express and VC2 already press on them). Host: the credit plan gives the study router (kernel v2) counter
credits on one TX queue too; the router zeroes its counter block in init, so the host's single-TXQ clear stays one
word (the UMD caution above). Alternative: stream-register credits as main (a second credit path in v4's sender
and receiver).

TXQ sharing (one queue): a data send leaves TXQ 0 busy for the packet's serialization, so on a router sending and
receiving at once the credit frame waits for the next free pass; the pass order (4.4) decides which waits. With TXQ 1
for the frames (decision 7) neither waits on the other.

### 4.3 NoCs and command buffers, by where the router runs

- **ERISC 0 (`DISABLE_MULTI_AERISC`):** two-ERISC v4's layout. Sender-side users on NoC 0 (worker credit preset on
  WR, upstream credits and teardown on AT), receiver-side users on NoC 1 (local writes WR, forward data WR_REG,
  forward sync RD), every preset kept; NoC 0's presets redone when `end_round` reports that the base firmware ran (as
  the sender does today). Host: the builder's force-to-NoC-1 rule is lifted for the study router on ERISC 0.
- **ERISC 1 (`DISABLE_FABRIC_TWO_ERISC`, mux, UDM):** the base firmware may use NoC 0 at any time from ERISC 0, so NoC
  1 only, four buffers for five users. Keep the per-packet presets (local writes, forward data, forward sync) and move
  the per-credit-group worker credit off WR: an inline 4-byte write on AT, as upstream credits already are. Cost: an
  unprogrammed command per worker credit (today 27 cycles with the preset).
- To confirm on hardware: that the base firmware on ERISC 0 does not touch NoC 1's buffers (else NoC 1's presets are
  redone after it runs too).

### 4.4 Pass order and placement

Receiver pass first: it frees ring slots and sends credits, which the far sender is waiting on, before this core
spends a pass sending. Measured both ways on the bidirectional rows before settling. Placement follows the mode:
ERISC 0 (context-switch master, heartbeat, link-status check) or ERISC 1 (none), from the same source
(`MY_ERISC_ID`, `enable_context_switch` as today).

### 4.5 Host

The builder already sets `NUM_ACTIVE_ERISCS`, the serviced-channel flags, the TX queue ids, NoC 1 and the command
buffers for one ERISC. Study changes, gated to the study router like the slot defaults: the credit plan (4.2), the
force-to-NoC-1 rule lifted for a router on ERISC 0 (4.3), and TXQ 1 for credit frames if decision 7 holds.
Slot depths stay as two-ERISC (same L1, measured the same in jobs 119-120).

## 5. Validation plan

1. Offline: re-collect the offline build commands on the merged tree (the database is pre-merge) and the coverage
   mocks' single-ERISC cases (`bh_2d_1erisc`, `bh_1d_1erisc`, `bh_vc1_1erisc`); after the pass split, every
   two-ERISC binary byte-identical to `v4gi_m`; single-ERISC builds compile for every Blackhole config, sizes in
   the kernel buffer.
2. Device, `DISABLE_MULTI_AERISC` first (CI's mode, and like Wormhole the router shares its core with the base
   firmware), then `DISABLE_FABRIC_TWO_ERISC`: mesh, 1D, ring, Y-torus, VC1 split and CI's `Fabric1D*` leg with the
   study router swapped in, hang capture armed; then the benchmarks against main's single-ERISC numbers (jobs 119-120)
   and v4's two-ERISC numbers.
3. Wormhole after Blackhole: an offline IRAM size check first, then runs on your cluster.

## 6. Decisions

Agreed (2026-10-28), then scoped by the user: on Blackhole two-ERISC is the preferred mode, so the bar is that
two-ERISC stays performant and single-ERISC is functional; single-ERISC tuning waits for Wormhole. No byte identity
between the modes (it depends on the compiler); a two-ERISC performance check on the device instead.
1. **Credits:** counter frames in every mode (one credit path, no stream registers spent on completions).
2. **Composition:** the role loop bodies are `sender::pass` / `receiver::pass` (force-inlined, run loops' locals by
   reference); the single-ERISC loop runs them out of line over state in memory, receiver first. Two-ERISC builds
   (v4se against v4gi_m): register and stack-slot allocation differences only (offline, 164 configs); checked on the
   device for performance.
3. **NoCs and command buffers on Blackhole single-ERISC:** the simplest robust form, as main: all router NoC traffic
   on NoC 1 (the builder's rule, both switches), and no command-buffer presets in single-ERISC builds (worker credit,
   local writes and forwards issued as whole commands). Section 4.3's per-placement layout and the presets are
   Wormhole-time optimizations.
4. **Pass order:** receiver first; measuring both waits for Wormhole.
5. **Modes:** both Blackhole switches validated (`DISABLE_MULTI_AERISC`, `DISABLE_FABRIC_TWO_ERISC`).
6. **Scope:** Blackhole 1D and 2D VC0, then VC1; Wormhole after (bring-up, then tuning).
7. **TX queues:** one queue on Blackhole single-ERISC; the second-queue probe (credit frames on TXQ 1) waits for the
   Wormhole tuning, where it matters.

Working method as for the two-ERISC router: NoC through the API, no compiler-dependent tricks, rare work out of line,
measured alternatives in the same job when tuning.
