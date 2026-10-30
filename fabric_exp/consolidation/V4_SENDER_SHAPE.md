# v4 sender structure cleanup: design (2026-10-29; revised after an independent review, §0)

## 0. Revision after review (supersedes §1's placement claims and §2.2-2.5 where they differ)

Checked in the DWARF of the v4trm_O3 images (independent review, confirmed here):
- Placement is the compiler's, decided by escape: on two-ERISC senders without VC2, `forwarding`, `worker`, `far_vc0`
  and the semaphore pointer have location lists (scalarized: registers or spill slots); the worker interface copy is a
  frame object (it escapes to the out-of-line connection event). On VC2 routers the whole table is a frame object
  (`fbreg -272`): VC2's interface inside it escapes, so VC0's forwarding records go to memory there too. Single-ERISC:
  everything in the state record (memory). So VC2 already changes the sender's shape, and §1's "memory table" is true
  only on VC2 routers and single-ERISC.
- Risks that would regress the hot loop: a `WorkerChannel` record holding the interface (escapes, so the per-pass worker
  state would leave registers); VC2's far ring as a loop local (two new live scalars); `visit_vc<VC>` doing nothing for
  absent VCs (dead inline calls); a one-word credit "source" (VC0 needs two words: the scratch word when connected,
  channel 0's location-info read counter when not); preset ownership resting on an implicit invariant; zero-length
  dense arrays and silent aliasing in `UpstreamCredits`; VC2 uses that are not template-dependent fail to compile on
  routers without VC2 (the dummy types are a compile requirement, not only for byte identity).

Revised steps (each its own variant, measured on its own):
- **1a `v4sh1` — unify the worker channels' code, nothing moves.** One `credit_worker<CH>` and one
  `worker_connection_event<CH>` driven by a constexpr `WorkerCredit<CH>` {preset, source word, idle word} (every
  channel's connection-info address is a compile-time constant, so VC2's no longer needs the runtime pointer);
  `if constexpr` preset at the call plus `static_assert(!(has_vc2_worker && worker_credit_preset))`; RX from
  `Channel<CH>::far_receiver`; one `Absent` type for absent features (VC2 uses stay template-dependent); the pass stays
  an explicit sequence with `if constexpr` at each call site; per-pass `Worker` state and the connection-time interface
  copy stay separate locals as today; VC2's state stays in the table. Dead code and stale comments (incl. the table's
  "memory" comment) fixed. Accept: every non-VC2 image byte-identical to v4trm (else a normalized per-function diff
  shows only layout), DWARF placement kinds unchanged on every image, VC2 images explained, then validation.
- **1b — dense forwarding records and upstream addresses** (`std::array` sized max(N,1); static_assert static channels
  are forwarding channels). Accept: normalized identity (offset permutations only), placement kinds unchanged, census.
- **1c — one placement for VC2 routers** (VC2's state out of the table, so VC0's records scalarize there as elsewhere;
  or deliberately in memory everywhere): a perf decision, census A/B.
- **2 — one register read for all VCs** (as §2.6), census on VC1 rows plus a mixed VC0+VC1 row and latency.

Status (2026-10-29):
- **1a `v4sh1`: done.** Built 1706/1706; 1650 images byte-identical to v4trm (every config without VC2: two and one
  ERISC, mux, VC1, trimming, express, telemetry), so no perf run is needed for them; the 56 VC2 images differ: DWARF
  placement kinds unchanged (`placement.py`), VC2 sender pass +2 instructions / +2 stack loads, the one-ERISC sender pass
  one stack load and store fewer. Validation (`variant_val_runs.sh v4sh1`, job 181): 13/13 runs pass (mesh 96, 1D 143,
  ring 19, Y-torus 12, mux 19, VC2 10, NeighborExchange 7, trimming 63; one ERISC mesh 96, ring 19, VC2 10,
  NeighborExchange 7, trimming 63). VC2 has no benchmark rows, so its +2 instructions are unmeasured.
- **1b `v4sh1b`: built 1706/1706**, every sender image changes (receivers identical): two-ERISC 1D, ring, VC2 and
  express loops unchanged (frames shrink where the table is a frame object), 2D sender pass +6 instructions, Y-torus -2;
  one ERISC frames 32-144 B smaller, the pass functions unchanged, the single loop reshaped on the configs without
  forwarding channels (mux, NeighborExchange). **Accepted:** validation 13/13 (job 182); census against v4trm, same
  jobs, two passes (jobs 183-190): every suite within 0.1% (2D 144.4 / 144.5, 1D 152.3 / 152.2, mix 128.7 / 128.8,
  one ERISC 2D 276.6 / 276.6, VC1 171.5 / 171.5, ...), the largest row move 0.4% (231 rows).
- **2 `v4sh2` (on v4sh1b): built.** One free-slots group read for every VC per pass, one in-flight load shared by the
  VC0 and VC1 forwarding groups (`visit_forwarding_group<MASK>`). Every image without VC1 equals v4sh1b's but for one
  commuted compare (`beq a5,s1` -> `beq s1,a5`); the VC1 sender pass +15 instructions, +4 stack loads, one fewer group
  read and fence pair. Measured on the VC1 split, inter-mesh and mixed VC0 + VC1 rows (`sh2_vc1_runs.sh`, jobs
  191-192). **Rejected:** VC1 1-hop and burst rows -0.1% to -0.2%, but forwarding rows +0.6% to +1.0% (split) and
  +0.8% to +2.2% (inter-mesh multi-hop; suite +1.3%), mixed +0.0% to +0.5%. Hypothesis (not isolated): VC1's registers
  read before VC0's visits, so a VC1 packet landing meanwhile waits a pass (the review's timing risk), and/or the VC1
  pass's +15 instructions. Kept from it: the shared `visit_forwarding_group<MASK>` with VC1's read in its old place
  (`v4sh2b`). v4sh2b against v4sh1b: same instructions at the same addresses on every image, a few with commuted operands only (`beq a5,s1` / `beq s1,a5`, `sub t1,s5,s1` / `sub t1,s1,s5` feeding only `snez`): same behavior and timing, so v4sh1b's census holds. **Lead: `v4sh2b`.** Tools: `placement.py` (DWARF location kinds per local, two builds),
  `regpress.py`.
- **1c `v4sh3` (on v4sh2b; user, 2026-10-29: arrange the sender state by channel role, not by VC).** `Workers` (VC0's
  and VC2's `Worker`: counts, connection semaphore pointer, expected connection value), `WorkerConnections` (the loop's
  copies of both interfaces: they escape to the out-of-line connection event and credit, so they stay apart from the
  per-pass state), `ForwardingTable` (VC0's and VC1's forwarding records only), `FarRings` (one per VC). Same pass order
  and the same free-slots reads. `Vc2Worker`, `NoVc2Worker`, `Vc2Interface`, `init_vc2_worker`, `visit_vc2`,
  `vc2_sent` gone; `single::SenderState` holds the same four groups. Built 1214/1214 (cov_m, every case, against a full
  rebuild of v4sh2b): 748 images byte-identical (every two-ERISC config without VC2, VC1, telemetry or express: 2D, 1D,
  ring, torus, NeighborExchange, trimming); VC1, telemetry and allinstr senders the same instructions after normalizing
  registers and stack offsets; Galaxy express senders one `li`/`mv` change in the far rings' init (same count);
  one-ERISC sender pass +2 instructions (VC0's semaphore pointer loaded at its use from the record, not once at pass
  entry); VC2 two-ERISC sender pass 631 -> 730 instructions, stack loads 72 -> 88, stores 26 -> 38 (VC0's forwarding
  records scalarize there now, and VC2's state is in registers with them: the predicted pressure).
  Why the VC2 sender grew (DWARF `placement.py`; instructions per source line, scratch `bylines.py`): in v4sh2b VC2's
  interface sat inside the table and escapes, so the whole table was a frame object on VC2 routers: VC0's forwarding
  records were loaded at use and stored when changed. That was an accident of where VC2 lived. In v4sh3 nothing in the
  table escapes, so the compiler scalarizes it there as on every other router (`forwarding`: frame -> split), and the
  2D sender already has more live values than registers (2D without VC2: 58 stack loads, 19 stores in the pass). The
  record loads go (`r.sent`, `r.completed`, the credit address: -6 each) and spills appear around the send and
  completion paths (slot advance +23 instructions / +12 stack ops, the forwarding visit test +21, completion check
  +14 / +8). Split by `v4sh3m` (measurement variant, built without asking first; not a design candidate: it brings a
  VC2-only record back): VC2's worker state and far ring in memory with its interface, VC0's records scalarized. Pass
  631 (v4sh2b) -> 706 (v4sh3m: VC0's records into registers) -> 730 (v4sh3: VC2's state into registers too). So most of
  the growth is VC0's records taking the placement they have on every router without VC2, not VC2's state. Static
  counts do not rank: whether v4sh3 is slower on VC2 routers is for the A/B. Hypothesis raised by it (not tested):
  memory-resident forwarding records may beat the scalarized, spilling table on the 2D senders generally; v4sh2b against
  v4sh3m on VC2 routers is that A/B for VC0 traffic on one config. Measurement (jobs 193-200): validation of both; VC2
  on, v4sh2b / v4sh3 / v4sh3m on the census suites that build VC2 routers (`sh3_vc2_runs.sh`, two passes); one-ERISC
  and VC1 census suites, v4sh2b against v4sh3 (two passes).
  **Results (jobs 193-200; the two passes agree within 0.1 cycles):** validation 13/13 for both. VC2 on, suite means
  v4sh2b / v4sh3 / v4sh3m: 2D 214.0 / 218.5 (+2.1%) / 215.8 (+0.8%), mix 170.0 / 176.0 (+3.6%) / 172.8 (+1.7%), burst
  +0.9% / +0.4%, Y-torus +1.3% / +0.5%, one-ERISC 2D +1.0% / +0.5%; the send-bound 1-hop writes carry it (154-160 /
  170 / 163-164), the receiver-bound scatter rows are unchanged. Without VC2: one ERISC 2D 276.6 -> 279.3 (+1.0%), 1D
  +0.9%, ring +0.7%, mux +0.1%, trimmed (identical images) 0; VC1 split and inter-mesh 1.000 (renamed registers only).
  **Not accepted as is:** VC2 routers and one ERISC regress beyond noise. Hypotheses (not isolated): VC2, VC0's records
  split instead of in memory (v4sh2b against v4sh3m, which differ only there: +0.4% to +1.7%, 1-hop writes ~+6%) plus
  VC2's own state in registers; one ERISC, VC0's semaphore pointer loaded at its use from the record rather than once
  at pass entry (the only change in that pass). Side finding: VC2 on costs VC0 traffic heavily on any router (v4sh2b,
  2D 4352 mean 144.5 without VC2 -> 214.0 with; 3-hop write 151.9 -> 304.3); cause unknown (hypothesis: VC0's buffer
  slots shrink). Next (user, 2026-10-29): control placement first (V4_PLACEMENT.md), then revisit 1c on top of it.

The original draft follows.

# (original draft)

User (2026-10-29): with VC1 or VC2 on, the sender should not change shape; VC0's and VC2's worker channels are the same
kind of channel to the router; the forwarding table should not waste entries for easy indexing. Do it without
regressing any config (perf or readability), the whole census as the judge (memory feedback-generic-over-narrow).
Base: the lead, `variants/v4trm`. Files: `router/router_sender.hpp` (most), `router/router_single.hpp`
(`SenderState`), `fabric_erisc_router_v4.cpp` (`wait_for_upstream_routers`, `UpstreamCredits`), `router_init.hpp`
(`has_vc2`, roles), `router_common.hpp` (feature facts).

## 1. Today (v4trm)

- One pass: group read of VC0's free-slots registers (+ VC2's worker) -> VC0 worker visit -> VC0 forwarding visits (one
  idle test) -> VC1: a second group read, its own idle test, its forwarding visits -> VC2 worker visit.
- State placement (deliberate; RESULTS §97-98 history):
  - loop locals of `sender::run` (registers when the allocator can): VC0 worker state (`Worker`: slot, sent,
    completed, acked, expected_connection), the loop's copy of the VC0 worker interface and its connection semaphore
    pointer, `far_vc0`, `far_vc1` (each far ring: free_slots, next_slot);
  - `ForwardingTable` (a local of `run`, memory, read and written once per visit): `records[NUM_SENDER_CHANNELS]`
    indexed by absolute channel (the worker channels' entries unused: 24 B, 48 B with VC2), `in_flight` (one word, both
    VCs), and VC2's whole worker (`Vc2Worker`: interface copy, semaphore pointer, `Worker`, its far ring);
  - single-ERISC: the same objects in `single::SenderState` (memory record), passed to the out-of-line pass.
- VC0 vs VC2 worker: same visit (`visit_worker<CH, RX>`); different credit function (`credit_worker`: preset command
  buffer when the router has only one worker channel, scratch source word; `credit_vc2_worker`: whole command, its own
  source word, the location-info `edm_read_counter`), and a duplicate connection event (`vc2_connection_event` =
  `worker_connection_event` without `preset_worker_credit`, which is a no-op on every router with VC2). The duplicate and
  the dummy types (`Vc2Interface`'s `? N : 1`, `NoVc2Worker`, defaulted CH/RX template parameters) exist to keep non-VC2
  images byte-identical when VC2 was added.
- `UpstreamCredits::addr[NUM_SENDER_CHANNELS]` (kernel_main's stack, init only): absolute index, workers unused.

## 2. Design

### 2.1 One description per VC (compile time)

A VC's sender side is: an optional worker channel, a set of forwarding channels, and one far ring (the far router's
receiver channel of that VC). Every VC fits: VC0 = worker + forwarding; VC1 = forwarding only; VC2 = worker only.

```
template <size_t VC> struct VcTraits {
    static constexpr uint32_t workers;      // mask: VC0 -> bit 0 if serviced; VC2 -> its channel if serviced; VC1 -> 0
    static constexpr uint32_t forwarding;   // mask of serviced forwarding channels of the VC
    static constexpr size_t far_receiver;   // VC0/VC1/VC2_RECEIVER_CHANNEL
    static constexpr bool present;          // workers | forwarding != 0
};
```
The existing masks (`vc0_channels`, `vc0_forwarding`, `vc1_forwarding`, `vc2_worker`) become these traits' members.

### 2.2 Worker channels: one type, one credit, one connection event

- The far ring belongs to the VC, not the worker (VC0's worker shares it with VC0's forwarding channels), so a worker
  channel's state is `Worker` (unchanged) + the loop's interface copy + its semaphore pointer:
  `template <size_t CH> struct WorkerChannel { Interface<CH> interface; volatile uint32_t* semaphore; Worker w; }`,
  with `Interface<CH>` = `StaticSizedSenderChannelWorkerInterface<worker_handshake_noc, Channel<CH>::num_slots>`.
- Credit: one `credit_worker<CH>(interface, returned, connected)`; per channel traits decide `preset` (CH == 0 &&
  `worker_credit_preset`) and the source word (CH == 0: the scratch word; VC2's: its location-info read counter).
  Generated code per channel unchanged: VC0's body is today's `credit_worker`, VC2's today's `credit_vc2_worker`.
- Connection event: one `worker_connection_event` (the preset call is already a no-op where it must be).
- `visit_worker<CH>` takes `WorkerChannel<CH>&` and the VC's far ring; RX from `Channel<CH>::far_receiver` (drop the RX
  template parameter everywhere; `Channel<CH>::far_receiver` exists and is unused today).

### 2.3 Forwarding records: dense, one table

- `records[num_forwarding]`, `num_forwarding = popcount(vc0_forwarding | vc1_forwarding)`; `record_index<CH>` =
  `popcount(all_forwarding & ((1u << CH) - 1))` (constexpr: every access stays a constant offset). `in_flight` bit
  positions stay per channel id (no change to masks or tests).
- `UpstreamCredits::addr` dense the same way (init only).
- Name: `ForwardingTable` keeps only forwarding records + `in_flight`; the secondary worker moves out of it (2.4).

### 2.4 Placement: unchanged

- Loop locals of `run` (as today): the primary worker channel (VC0's, when serviced) and each present VC's far ring.
- Memory table (as today, a local aggregate of `run` read/written per visit): forwarding records, `in_flight`, and the
  secondary worker channel (VC2's), as `SenderTable { ForwardingRecord records[...]; uint32_t in_flight; Optional<
  WorkerChannel<VC2_LOCAL_CHANNEL_START>> vc2; }`.
- Single-ERISC `SenderState`: the same objects as members (memory), as today.
- Rationale: the per-pass sweep (V4_PROFILE.md §6) shows the sender pass binds 94 of 159 rows; this step must not move
  any state between registers and memory. Only names, types and the table's indexing change.

### 2.5 The pass (step 1: same order and reads as today)

```
pending = channels_with_packets<VcTraits<0>::workers | VcTraits<0>::forwarding | VcTraits<2>::workers>();
visit_vc<0>(...);           // worker visit, then forwarding visits behind the one idle test
visit_vc<1>(...);           // its own group read and idle test, as today
visit_vc<2>(...);           // worker visit (secondary worker, from the table)
```
`visit_vc<VC>` is one template: if the VC has a worker, `visit_worker`; if it has forwarding channels, the idle test
and `visit_forwarding_channels`; nothing for an absent VC. Same instruction order as today on every config.

### 2.6 Step 2 (separate variant): one register read for all VCs

`pending = channels_with_packets<all_workers | all_forwarding>()` once at the top, `in_flight` loaded once; VC1's
section uses them. Changes behavior only on VC1 routers: VC1's registers are read before VC0's visits rather than after
(a packet that lands during VC0's visits waits one more pass). Measured on its own.

### 2.7 Removed

`Vc2Interface` / `NoVc2Worker` / `Vc2Worker`, `credit_vc2_worker`, `vc2_connection_event`, `vc2_sent`, `init_vc2_worker`
/ `visit_vc2` with their defaulted template parameters, the unused `for_each_bit`, RX template parameters of
`send`/visits (from `Channel<CH>::far_receiver`).

## 3. Verification

1. Offline (cov_m, -O3, every case incl. trimming and express): all configs build; v4sh1 vs v4trm per image:
   identical, or `codegen_diff.py` showing the same instruction stream except stack offsets (the denser table moves
   offsets); `regpress.py` loop counts equal per config. Expected identical where the table layout does not change.
2. Device, same job as v4trm (control): validation (mesh sanity, val1d_m1, ring and Y-torus validation, mux, VC2 yaml,
   NeighborExchange, trimming gtests, VC1 split validation), then the census (two passes). Accept v4sh1 only if no suite
   mean moves beyond noise (0.5%) and no row beyond 3% (the census showed passes within 3%).
3. v4sh2 the same way against v4sh1.

## 4. Known sensitivities to respect

- No closures in the loop; per-channel inlining of `send` with immediates; worker credit and connection events out of
  line; per-round work out of line (RESULTS §97-98).
- Layout of loop-local aggregates has moved register allocation before (an empty member permuted VC1 senders' stack
  slots); dead inline calls shifted allocation in spill-heavy loops (telemetry). Keep `if constexpr` at call sites.
- BH emulated L1 inline write always uses the write command buffer (the preset owner must be unique).
- VC2 credit source word must not be the shared scratch word (race; V4_VC2.md).
