# Blackhole fabric router: where the ERISC cycles go, and what DRISC offload can buy

Measured 2026-10-09 on a P150x8 LoudBox (2x4 mesh, 1 link per test, Blackhole), checkout `453ee7bdfa7` plus the
experiment instrumentation described at the end. Companion to `fabric_bh_throughput_drisc_discussion.md`.

All cycle numbers are ERISC core cycles. The ERISC `mcycle` counter and the Ethernet wall clock tick together
(measured ratio 1.000), and per-ERISC cycles/packet match the 1.35 GHz bandwidth-derived interval, so "cycles"
below are 1.35 GHz cycles. 200 Gb/s at 2 KiB needs <= 110 cycles/packet (the same packet rate as 400 Gb/s at 4 KiB).

## TL;DR

1. **Every configuration is ERISC-CPU bound** at the default channel depths: the per-packet interval is flat from
   1 KiB to 4 KiB, adding work on the critical ERISC shows up 1:1 or worse, and shrinking slots from 16-32 to 8
   costs only 0-7% on the generic paths.
2. **The bottleneck ERISC depends on the traffic.** RX is critical for multi-hop traffic (1D linear, ring,
   unidirectional linear, all scatter). TX is critical for single-hop plain writes (neighbor exchange) and shares
   it with RX on 2D mesh multicast. A 200G design has to make *both* ERISCs leaner.
3. **Offloading local delivery alone is a small lever for plain writes.** Deleting local-delivery data writes
   outright (the ceiling for any offload) gives +15-26% on RX-bound plain-write configs and nothing on TX-bound
   ones. Replacing them with a push-style handoff (one stateful full-packet NoC write) keeps only +3-6%, because a
   push costs about as much ERISC time as the plain local write it replaces. Scatter is different: ceiling +36-44%,
   realistic push +15-27%.
4. **TX bookkeeping is not worth offloading.** Batching worker/upstream credit returns and polling connection
   state 16x less often changed throughput by 0% in every configuration.
5. **The expensive RX item is forwarding, not local delivery:** about 85 cycles per downstream copy (payload write
   plus credit update), versus about 50 for a local unicast write. That, plus ETH-NIU read bandwidth at 400G
   (section 6), points at a "push once, DRISC fans out (forward + local)" design rather than local-only offload.
6. **The benchmark harness itself costs ~14 cycles/packet on TX:** it force-enables router bandwidth telemetry for
   every benchmark test. Production routers do not run it, so the goldens understate TX-bound throughput by 8-11%.
7. **One-way traffic on a link is capped by the Ethernet layer, not the ERISCs** (section 15). The TX queues run in
   resend mode and the sender gets sequence acknowledgements back only with data on the reverse data queue, or in a
   periodic update every 8000 cycles when that queue is idle; the fit is ~128 Ethernet packets per update. Each
   fabric packet is 2 Ethernet packets (payload + pointer update), 3 once payload + header exceeds the 4114 B
   Ethernet maximum, so a one-way stream cannot beat ~125-130 cycles/packet (187.5 above 4114 B) whatever the
   ERISCs do: above both the 117-cycle 400G@4KiB and the 110-cycle 200G@2KiB targets. Traffic in both directions
   (neighbor exchange: 124) does not see it. A bare-link probe (section 16) confirms it with no router at all, shows
   the acknowledgements are per queue (traffic on TXQ1 does not help TXQ0), and shows two fixes that reach the wire
   rate, 118 cycles for a 4 KiB payload + 80 B header: credits on the data queue, or data split over two queues.
   The simplest fix is one register: a keepalive of 1000-2000 cycles (`TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT`) removes
   the floor in the router, with no other change (2D 1-hop 4 KiB writes 187.6 -> 152.5 N/S, 169.8 E/W; one-way
   neighbor exchange 187.5 -> 122.9). The routers' own CPU is the limit again.

## 1. Baseline at 2 KiB

`cyc/pkt` is the effective interval `1.35e9 / avg_packets_per_s` from the bandwidth summary. "Upstream" is HEAD as
the CI benchmark runs it (BW telemetry on); "no telemetry" is the same build with `TT_FABRIC_NO_BW_TELEMETRY=1`.
Interval is flat across 1/2/4 KiB for every row (full tables in `results/base_upstream.csv`, `results/notel.csv`).

| Test (2 KiB) | Upstream GB/s | cyc/pkt | No telemetry GB/s | cyc/pkt |
| --- | ---: | ---: | ---: | ---: |
| Neighbor exchange (speedy) write | 20.03 | 138 | 22.17 | 125 |
| Neighbor exchange scatter | 18.32 | 151 | 18.42 | 150 |
| Linear multicast write | 15.53 | 178 | 15.81 | 175 |
| Linear multicast scatter | 12.22 | 226 | 12.22 | 226 |
| Unidir linear multicast write | 11.49 | 241 | 11.49 | 241 |
| Unidir linear multicast scatter | 9.50 | 291 | 9.50 | 291 |
| Mesh multicast write (2D VC0) | 13.96 | 198 | 15.31 | 181 |
| Mesh multicast scatter | 12.94 | 214 | 12.94 | 214 |
| Half-ring multicast write | 11.86 | 233 | 10.90 | 254 |
| Half-ring multicast scatter | 8.98 | 308 | 8.98 | 308 |

Ring rows vary by about +/-5% run to run. The 2D VC0+VC1 configuration targeted by the discussion doc needs a
multi-mesh system and could not be measured on this single 2x4 mesh.

## 2. Which ERISC is critical (delay injection)

`TT_FABRIC_EXP_TX_DELAY` / `TT_FABRIC_EXP_RX_DELAY` spin N extra cycles per packet on one ERISC. Entries are the
change in interval; a change >= the injected delay means that ERISC has no slack.

| Test (2 KiB) | TX +30 | TX +60 | RX +30 | RX +60 | Critical |
| --- | ---: | ---: | ---: | ---: | --- |
| Neighbor write | +45 | +75 | +1 | +25 | TX (RX has ~35 slack) |
| Neighbor scatter | +33 | +63 | +38 | +68 | both |
| Linear write | +28 | +52 | +42 | +71 | RX, TX close behind |
| Linear scatter | 0 | +16 | +45 | +74 | RX (TX ~50 slack) |
| Unidir linear write / scatter | 0 / 0 | +13 / 0 | +45 / +45 | +75 / +75 | RX |
| Mesh write | +43 | +75 | +10 | +34 | TX (RX ~20 slack) |
| Mesh scatter | +27 | +60 | +39 | +66 | both |
| Half ring write / scatter | -3 / -12 | +11 / -20 | +54 / +46 | +85 / +75 | RX |

The phase profiler's utilization view agrees: on the critical ERISC, busy + blocked is 94-99% of the active time.

## 3. Where the cycles go

### Per-phase costs (in situ, `TT_FABRIC_PHASE_PROFILE=2`, no telemetry, 2 KiB)

Cycles per occurrence; each timer adds a few cycles of its own.

| Phase | Cycles | Notes |
| --- | ---: | --- |
| TX send (`send_next_data` + bookkeeping) | 73-76 | includes 14 cycles spinning on the eth txq before the pointer update; 87-89 with BW telemetry on |
| TX completion processing + upstream credit | ~38 | ~0.9 per packet on linear/mesh, so 33-35 cycles/packet |
| TX connection polling of an unconnected channel | ~16 | 1D linear TX does 3.1 of these per packet (~50 cycles/packet) |
| RX decode + admit | 27-30 | |
| RX forward, per downstream copy | ~85 | payload write + remote credit inline write + local free-slot update |
| RX local unicast write | ~50 | |
| RX local 4-chunk scatter | 102-107 | |
| RX completion ack (eth) | 24-25 | |
| RX first-level ack (ring) | 22 | |

### Primitive costs (`TT_FABRIC_PHASE_PROFILE=3`, back-to-back, ~4 cycles of loop overhead included)

| Primitive | Cycles/op |
| --- | ---: |
| `csrr mcycle` | 7 |
| dependent L1 load | 8 |
| stream register read | 9 |
| eth txq busy read | 12 |
| NoC cmd buffer ready read | 10 |
| NoC write 64 B, stateless (7 register writes) | 23 |
| NoC write 2 KB, stateless (NIU-bound, ~54 B/cycle) | 38 |
| NoC write 64 B, stateful (two cmd-buffer ready waits: set-trid and issue) | 30 |
| NoC inline 4 B L1 write (emulated) | 24 |

Two consequences: the router's cost is dominated by MMIO register traffic and L1 loads, not computation; and the
Blackhole "stateful" NoC write API is *slower* than the stateless one on ERISC, so stateful forwarding buys nothing
today.

## 4. Offload experiments (no telemetry)

| Test (2 KiB) | Baseline cyc/pkt | Skip local data writes (ceiling) | Push emulation (realistic) |
| --- | ---: | ---: | ---: |
| Neighbor write | 125 | 123 (+1%) | n/a (speedy path) |
| Neighbor scatter | 150 | 123 (+22%) | n/a (speedy path) |
| Linear write | 175 | 152 (+15%) | 165 (+6%) |
| Linear scatter | 226 | 158 (+44%) | 179 (+27%) |
| Unidir write | 241 | 191 (+26%) | 233 (+3%) |
| Unidir scatter | 291 | 209 (+39%) | 248 (+17%) |
| Mesh write | 181 | 182 (-1%) | 185 (-2%) |
| Mesh scatter | 214 | 182 (+18%) | 186 (+15%) |
| Half ring write | 254 | 203 (+25%) | 243 (+4%) |
| Half ring scatter | 308 | 227 (+36%) | 260 (+19%) |

- *Skip local* (`EXP_FLAGS=1`) drops local delivery of plain writes and all-write scatters; atomics still go out
  so test sync works. It is the most any local-delivery offload could gain.
- *Push emulation* (`EXP_FLAGS=16`, generic receiver path only, so the speedy neighbor rows are unaffected) replaces those deliveries with one stateful NoC write of header + payload into
  a scratch buffer in the same core's L1. That is the ERISC side of a DRISC push, minus the doorbell (one more
  inline write, ~24 cycles, or less if batched). It is pessimistic for L1 bandwidth because the loopback also writes
  into ETH L1, which likely explains part of the ring gap.
- TX bookkeeping: batched credit returns (`EXP_FLAGS=4`) and 16x lazier connection polling (`EXP_FLAGS=8`) gave
  0% everywhere; combined with push they *hurt* mesh (-9% to -22%) by delaying credits to upstream routers.
- Removing dispatch overhead alone (`EXP_FLAGS=2`, bare write for unicast packets) gave 0-3%.

## 5. Credit loop and slot depth

`TT_FABRIC_EXP_MAX_RECEIVER_SLOTS` / `_SENDER_SLOTS` cap channel depth (defaults: 1D 16-32 sender / 32 receiver
slots; mesh 16-17 sender / 16 receiver; 4.4 KB per slot).

- 8 slots: generic paths lose 0-7% (half ring *gains* 8%); the speedy neighbor path loses 43-49% because its
  batched credit return needs deep buffers.
- 4 slots: every path collapses to 350-430 cycles/packet, i.e. a slot round trip of ~1.4-1.7k cycles (~1.1 us).
- At 110 cycles/packet that loop needs >= 14 slots. Today's 16 is enough but only just, so once the CPU cost
  drops, slot depth or loop latency becomes the next limit.

## 6. What this means for the DRISC plan

**Keep on the ERISC (cannot move):** eth TX (payload send + remote pointer update, ~73 cycles), eth completion and
ack sends (~24), packet detection.

**Not worth moving:** worker/upstream credit return and connection management (0% measured gain).

**Local delivery only (the current first slice):** worth it for scatter and other multi-write or flush-bearing
operations (+15-27% realistic), roughly neutral for plain writes, and no help on TX-bound configs. On its own it
does not reach 200G at 2 KiB anywhere: the best case, linear scatter, goes from 226 to ~179 cycles/packet.

**Forwarding + local together ("push once, DRISC fans out"):** the larger RX lever. Linear multicast RX today
spends roughly 27 (decode) + 34 (forward, 0.4 copies/packet) + 50 (local) + 24 (completion) cycles per packet plus
loop overhead. Replacing forward + local with one push (~30-40) would cut about 45-55 cycles. It also matters for
400G independently of CPU: forward + local reads each payload out of ETH L1 twice through one NIU. At 50 GB/s per
direction that is 100 GB/s against a ~73-86 GB/s NIU read rate, so multicast that both forwards and delivers
cannot run at 400G from the ERISC regardless of instruction count. One push to a DRISC that then fans out from
its own L1 halves the ETH-NIU read traffic. This design needs DRISC ownership of downstream router credits.

**TX has to get leaner on the ERISC itself.** TX-bound plain-write traffic (neighbor, mesh) sits at 125-181
cycles/packet with the send at ~73. ERISC-only candidates the measurements point at:
- batch remote pointer updates across back-to-back sends (each update is 3 txq writes plus a busy poll);
- stop polling unconnected sender channels every visit (~50 cycles/packet on linear TX; it matters once TX is
  critical);
- a leaner NoC write path (single ready check, fewer register writes).

## 7. Suggested next steps

1. Prototype a cheaper TX send (batched pointer updates) and lazy unconnected-channel polling. Measure on
   neighbor and mesh write, the TX-bound rows.
2. Build the push to a real free-subchannel DRISC with fan-out (forward + local) behind a flag, starting with 1D
   linear multicast, the clearest RX-bound case. Keep the push-only local-delivery variant as the control.
3. Repeat the delay-injection and skip-local runs on a multi-mesh (VC0+VC1, `-Os`) system before committing to
   the 2D design.
4. Re-run section 5's slot sweep after any CPU win, to see when loop latency takes over.

## 8. 2D (mesh, VC0) per-packet breakdown

Runs: `exp_2d.yaml` (single flows 1-hop E/W and N/S, bidirectional 1-hop, 3-hop E/W, all-to-all multicast; 2 KiB,
no BW telemetry) and `exp_2d_trim.yaml` (same tests with channel trimming: unused router channels compiled out).
Leaf phase costs come from runs that timed only that phase (`TT_FABRIC_PHASE_MASK`), minus the calibrated 4 cycles a
timer pair shows inside its own phase. The polling numbers come from the trimming ablation, which has no
instrumentation at all.

The 2D router on this mesh: the TX ERISC services 4 VC0 sender channels (the local worker plus one per other
direction) and visits every one on every loop iteration; the RX ERISC services 1 receiver channel.

**Single flow (one worker, 1 hop E/W, plain write): 223 cycles/packet, TX-bound (TX +30 -> +45, RX +30 -> +1)**

| TX ERISC work per packet | Cycles |
| --- | ---: |
| eth send (`send_next_data`, incl. ~10 waiting on the txq before the pointer update) | 62 |
| completion processing + credit write to the worker (35 x 0.74/packet) | 26 |
| the active channel's other polling + main-loop bookkeeping | ~52 |
| 3 visits to empty sender channels (~27 each) | ~83 |
| **total** | **223** |

Evidence for the polling rows: trimming the E/W router down to 2 sender channels gives 168 (still TX-bound), so
two empty visits cost 55 cycles; on N/S trimming leaves 1 channel and the interval drops to 152, where RX becomes
the bottleneck. With VC1 (up to 9 sender channels) a single flow would pay ~8 empty visits, ~220 cycles.

| RX ERISC work per packet | Local delivery (E/W) | Local delivery (N/S) | Forward only (3-hop middle) | Multicast N/S (2 copies + local) |
| --- | ---: | ---: | ---: | ---: |
| decode + admit | 17 | 32 | 31 | 54 |
| forward (per copy ~83-110) | - | - | 83 | 220 |
| local write / 4-chunk scatter | 46 / 103 | 46 / 103 | - | 48 / 100 |
| completion ack (eth) | 21 | 21 | 21 | 22 |
| polling + loop (residual) | ~53 | ~53 | ~53 | ~53 |
| **total (write)** | **~137** | **~152** | **~188** | **~400** |

Cross-checks: N/S trimmed is RX-bound at 152 (RX ~99% busy); 3-hop trimmed lands at 189 with both ERISCs critical;
scatter cases become RX-bound once TX polling is trimmed (191-206).

**Multicast all-to-all:** the heaviest E/W links carry 6 flows across 3 sender channels, so the empty-visit cost is
shared: 181 cycles/packet (send ~56, completions ~29, connection checks ~4, the rest polling and loop). Trimming
helps less here (181 -> 169) because most channels carry traffic.

**4 KiB check (`exp_2d_4k.yaml`):** the interval is unchanged at 4 KiB (220-222 cycles for every single flow), so
single-flow 2D reaches 200G at 4 KiB only without the harness's BW telemetry: 24.9-25.1 GB/s. With the harness's forced BW
telemetry it is 240-246 cycles, 22.5-23.0 GB/s (~180 Gb/s), which is what CI and the goldens see. Mesh multicast is
30.4 GB/s at 4 KiB (28.1 with telemetry).

**Against the 110-cycle target (200G at 2 KiB):** TX needs about -113 (empty-channel polling is 83 of it); RX needs
-27 to -42 for local delivery and about -78 for forwarding, where the forward copy (83 per copy) is the biggest item.

**Opt level:** with separate JIT caches, `-Os`/`-Oz` are +2-3% on single flows and -6% to -13% on multicast;
`-O1` is 5-16% slower. Note: an Ethernet kernel's JIT hash does not include the opt level, so
`TT_METAL_FABRIC_OPT_LEVEL` silently reuses whatever binary is cached; use a separate `TT_METAL_CACHE` per level.
Straight-line code runs at 1.00 cycle/instruction from L1 up to at least 8 KB, so instruction fetch is not a
limit for sequential code.

## 9. Validation against the official ubench on clean main

`test_fabric_ubench_at_least_2x2_mesh.yaml`, run exactly as CI runs it (`TT_METAL_CLEAR_L1=1`, profiler off, BW
telemetry forced on by the harness), on clean `main` (`453ee7bdfa7`, all experiment changes stashed). Summary in
`results/ubench_main_summary.csv`. Units: the CSV reports payload GB/s on the busiest link; 200 Gb/s = 25 GB/s.

| 1 link, 4 KiB | main here (2x4) | golden p150_x4 (2x2) | router telemetry (main) |
| --- | ---: | ---: | ---: |
| MeshMulticast write | 28.11 GB/s (225 Gb/s) | 26.45 (212 Gb/s) | 29.5 |
| MeshMulticast scatter | 25.64 (205 Gb/s) | 26.47 (212 Gb/s) | 26.8 |
| SingleSenderMeshUnicastAllDevices write | 22.89 (183 Gb/s) | 19.64 (157 Gb/s) | 23.7 |
| LinearMulticast write | 31.09 (249 Gb/s) | 29.07 (233 Gb/s) | 32.6 |
| NeighborExchangeUnicast write | 39.95 (320 Gb/s) | 38.38 (307 Gb/s) | 41.6 |
| FullRingUnicast write | 25.78 (206 Gb/s) | 22.43 (180 Gb/s) | 26.6 |

The independent router telemetry (Ethernet words sent / active cycles, headers included) agrees with the summary
within ~5%. Unidir-linear and half-ring are lower here than in the x4 golden because the 2x4 mesh puts more flows
on their busiest link. Fused write+atomic rows on HEAD are 2.5-4x the golden (e.g. mesh 4 KiB 26.9 vs 9.8 GB/s): the
fused-atomic penalty discussed in the original doc is gone on HEAD.

What "200G" means in these numbers:
- These are 400 Gb/s links; a reading of >= 25 GB/s means the router keeps up with 200 Gb/s worth of payload on a
  400G link. A real 200G link (BH Galaxy half rate) has ~24 GB/s of payload after framing and doubles the wire time
  per packet; whether the ERISC's per-packet work overlaps that longer wire time was not measured.
- In CI configuration, single-flow 2D at 4 KiB is ~180-183 Gb/s (official single-sender row 22.9 GB/s, my single
  flows 22.5-23.0 GB/s); it reaches 25 GB/s only with the harness's BW telemetry off. Multi-flow mesh multicast is
  above 25 GB/s at 4 KiB in both configurations. Every 2 KiB row is 70-160 Gb/s.

## 10. Neighbor exchange (speedy path) as the reference ceiling

`exp_nbr_sweep.yaml`, 1 link, plain writes, no BW telemetry (`results/nbr_notel.csv`; CI-mode in `nbr_tel.csv`).
`TT_FABRIC_EXP_LOG_CONFIG=1` confirms every router runs the speedy path with exactly one TX/RX pair: ERISC0 services
only sender channel 0, ERISC1 only receiver channel 0; sender credits are batched by 2, receiver credits by 4;
32/32 slots up to 5.4 KB packets, 16/16 at 7.6-8.7 KB, 8/8 at 15 KB. Links train at 400 Gb/s
(`read_eth_speed.py`).

| Payload | GB/s | Gb/s | cycles/packet | wire time at 400 Gb/s | limited by |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 KiB | 11.2 | 89 | 124 | 29 | TX ERISC |
| 2 KiB | 22.4 | 179 | 124 | 57 | TX ERISC |
| 3 KiB | 33.5 | 268 | 124 | 85 | TX ERISC |
| 4 KiB | 44.4 | 355 | 124.5 | 112 | TX ERISC |
| 5.4 KB | 47.6 | 381 | 154 | 149 | wire |
| 7.6 KB | 48.3 | 386 | 213 | 207 | wire |
| 8.7 KB | 48.2 | 385 | 244 | 237 | wire |
| 15.2 KB | 45.8 | 366 | 449 | 413 | wire (8 slots) |

The realizable ceiling of one link in one direction is ~48.2 GB/s of payload (386 Gb/s, ~96.5% of line rate; the
rest is the 48 B fabric header plus Ethernet framing). The lean path reaches it from ~5 KB packets; at 4 KiB it
needs ~8 fewer cycles per packet (<= ~116), at 2 KiB about half (<= ~58).

Criticality at 1-4 KiB: TX +30 -> +42..43; RX +30 -> +10..15; dropping local delivery -> no change.

| TX ERISC per packet (1-4 KiB) | cycles |
| --- | ---: |
| eth send, inlined (2 txq-busy reads, header load, 7 txq register writes, pointer bookkeeping; txq wait ~1) | 61 |
| completion processing + batched worker credit (18 x 0.45) | 8 |
| per-visit polling + loop (free-slot stream register, txq-busy check, completion poll, liveness L1 read, cache invalidate, loop) | ~55 |
| **total** | **124** |

| RX ERISC per packet (1-4 KiB) | cycles |
| --- | ---: |
| packet poll, header load | 13 |
| local NoC write | 46 |
| batched completion ack (12 x 0.25) | 3 |
| polling + loop | ~40-45 |
| **busy total** (~20 cycles of slack) | **~105** |

Implications: 400G at 4 KiB is within reach of a lean single-pair router (TX must drop ~8 cycles). 400G at 2 KiB is
not reachable by trimming alone: the eth send by itself (61) exceeds the 58-cycle budget, so the send has to do
less per packet (for example one remote pointer update per several packets), and the RX local write (46) would have
to leave the ERISC as well. 200G at 2 KiB (110) needs ~14 cycles off this TX path.

## 11. Ethernet TX queue: busy checks vs. waiting

The TX queue is configured with `ETH_TXQ_SPIN_WAIT_SEND_NEXT_DATA=0` (busy is checked in the can-send decision) and
32 data packets of accept-ahead.

- The queue is essentially never busy when checked: 0 of 2.9 M packets on the 2D generic path and <= 0.03% on the
  speedy path saw it busy before a send; the post-payload wait is 1 cycle (speedy, where the read is issued early)
  or ~10 cycles on the generic path, which is the cost of one busy read, not waiting. Even when wire-bound (5.4-8.7 KB)
  the queue is not busy; back-pressure shows up as receiver credits instead.
- A busy read costs ~8 cycles only when its result is used immediately; the core does not stall on a load until the
  value is consumed. Removing the speedy path's dead pre-send read (`EXP_FLAGS=32`) saved ~1 cycle (124 -> 122.6).
  So the TX queue costs the speedy path ~10 cycles per packet and the generic path ~18 (two immediately-consumed
  reads; hoisting the post-payload read as the speedy path does should save ~8).
- Removing the can-send busy check (`EXP_FLAGS=64`) hung the router on the first test, even though that check almost
  never sees the queue busy: it appears to be needed to order command-register writes. Not an option without
  understanding the hardware semantics.
- Round-robin across TX queues would not help at <= 4 KiB: there is no waiting to hide, each command still needs a
  check on its queue, a payload and its pointer update must share a queue, and packets on different queues can
  reorder on the wire (the receiver would need a counter per queue). Above ~5 KB the wire is the limit, and more queues
  do not add wire bandwidth.

## 12. Design ideas under discussion (not yet tested)

Ideas raised while discussing how to restructure the router as a pipeline across the two ERISCs and possibly a
DRISC (the mux v2 model: separate issue from retirement, bound outstanding work, hand work over in batches). Each
row lists the measurement in this document that bears on it. Cycle counts are per packet unless stated.

Facts that constrain every option:

- Ethernet TX-queue work (payload send, its "packet sent" pointer update, completion acks) can only run on an
  ERISC. Ordering is guaranteed only within a TX queue; only queues 0 and 1 can carry remote register writes
  (credits, pointer updates); queue 2 carries data only. There is no combined data-plus-notify command.
- A NoC write can only source from the issuing core's own L1. Both ERISCs share the Ethernet core's 512 KiB L1;
  RX forwarding and local writes use NoC1, TX worker credits use NoC0.
- Polling is the largest single cost: ~45-55 cycles per core per packet, plus ~27 per empty 2D sender channel.
  Every extra pipeline stage pays its own polling and a handoff (~10-25 cycles) unless work is handed over in batches.
- Each payload should leave the Ethernet core's L1 once at 400G: two copies (~96 GB/s) exceed one NoC port's
  read rate (~73-86 GB/s).

| Idea | What the data says | Notes |
| --- | --- | --- |
| Keep VC0 sender channel 0 (injection) and receiver channel 0 on separate ERISCs | Already the case; each alone costs ~105-125 even on the speedy path (section 10) | In 2D most traffic on busy links arrives via the forwarding sender channels, so those, bookkeeping and polling are what a rebalance would move |
| Rebalance work between the two ERISCs inside the Ethernet core | No payload copy needed (shared L1); a second NoC port (NoC0) adds egress | Respects packet locality; helps RX-bound shapes (multi-hop, scatter) where TX has 25-50 cycles of slack (section 2); no help where TX is the bottleneck |
| Second lane per VC, so the other ERISC can send forwarded traffic on its own TX queue | Two senders into one remote ring can reorder across queues | Needs a separate receiver ring and arrival counter per lane; only queues 0/1 can carry the per-packet pointer update |
| Data and credits on parallel TX queues | Already done in two-ERISC mode: data + pointer update on queue 0, completion acks on queue 1 | The per-packet pointer update cannot move to another queue: it could overtake its payload (section 11) |
| Batch pointer updates: one per k packets on the same queue | Each update is 3 TX-queue register writes plus a busy check, ~10-20 TX cycles | Keeps ordering; the receiver processes k slots per update; needed for 400G at 2 KiB, where the send alone (61) exceeds the ~58-cycle budget |
| Batch completion acks on the generic path, as the speedy path does | Generic RX acks per packet: 21-25 cycles; speedy batches by 4: ~3 | RX-only change; also cuts reverse-direction wire use (~3% unbatched at 2 KiB) |
| Piggyback credits on reverse-direction packet headers (ACK-on-data) | Saves the RX ack (21-25 generic, ~3 speedy); costs the TX ~5-10 (read the RX's count, store it into the header, fence before the send) | The L1 read lands on the RX, which already loads the header; the TX keeps polling a stream register written by its local RX. Needs a fallback explicit ack when there is no reverse traffic and a packed per-channel field for 2D (4-9 sender channels). Only a win when RX is the bottleneck and TX has slack |
| Offload only the header to a DRISC for the routing decision | Decode + admit is 13 (speedy), 17-32 (2D unicast), 54 (2D multicast, two outputs); the header push + reading the decision back costs the ERISC ~30-35 | Roughly break-even; the dominant RX costs (payload NoC writes 46 local / ~83 per forward, acks, polling) stay on the ERISC. Adds ~0.5-1 us of slot hold time, raising the slots needed at 110 cycles/packet from ~14 to ~23 |
| Push the whole packet to a DRISC that fans out (forward + local) | Push emulation: +3-6% for plain writes, +15-27% for scatter (section 4) | Breaks locality (an extra NoC hop for single-copy packets); pays off for multi-copy packets at 400G, where one NoC port cannot read the payload out twice. A forward-heavy multicast router could need more than one DRISC |
| Skip polling idle sender channels (summary doorbell or ready list) | ~27 cycles per empty channel; ~83 for a 2D single flow with 4 channels; up to ~220 with VC1 (section 8) | Trimming unused channels gave +25-46% on 2D single flows |
| Drain several packets per channel visit | Divides the ~45-55 cycles per core per packet of polling by the batch size | Pairs naturally with batched pointer updates and acks |
| Three-core pipeline (two ERISCs + DRISC) | 2D single flow today: TX 223 + RX ~140 = ~363 cycles of work; perfectly balanced over 3 cores ~121 each, ~93 after removing empty-channel polling | Rebalancing alone at most 1.5x: 4 KiB could reach ~78-98% of the ceiling, 2 KiB only ~39-48%; 2 KiB also needs less work per packet |

## 13. Placement model ("lego" model)

`fabric_exp/model/` (details in its README): a piece catalog (`catalog.yaml`, every cost tagged measured or
estimated), a static packer that searches all placements of the pieces on two ERISCs plus a DRISC, and an
event-driven model of one bidirectional link that re-runs the best placements with real credit loops, latency and
batching. Python 3 + PyYAML, no device.

- Validation: the static model matches 20 measured configurations (mean |error| 1.1%, max 3.5%), including 2D at
  4 KiB; the event-driven model matches 16 (mean 2.4%, max 5.0%), including the 2-32-slot sweep and the wire-bound
  sizes, with one fitted parameter: one-way link latency 450 cycles (~333 ns).
- **Target 400G at 4 KiB** = 117 cycles/packet (the wire time): the same core budget as 200G at 2 KiB, but each
  payload copy takes ~76 cycles of a NoC port, and both ERISCs share their tile's port (local writes and forwards
  both use NoC1 today). Placements keep receive work off E0 (SC0's ERISC). Event-driven, 32 slots:
  - 1-hop writes reach line rate (119) on two ERISCs with lean sender-channel polling (below) and batching by 2; no
    DRISC needed.
  - The 3-hop middle router stays at 132 on two ERISCs (E1-bound); the DRISC moving the data gets 118, given
    ~20-24-slot downstream channels (126 with 16).
  - 1-hop scatter stops at 126-128: one core cannot deliver scatter faster, wherever it runs. Line rate needs
    scatter split between E1 and the DRISC.
  - Multicast fan-out (local + 2 forwards): 284 on two ERISCs, 153 with the DRISC moving the forwards (the cap of its
    NoC port carrying both 4 KiB forwards).
- **Slot depth gates all of it.** With 16 slots per channel nothing reaches line rate (best 138-144). A worker's
  sender slot is freed only after the far side processes and acks the packet, so the sender channel is the first
  limit and the receiver the second: ~24 sender and ~24 receiver slots with batching by 2. 24 slots on all four 2D
  sender channels does not fit in L1 (~534 KB); deepening only the active channels fits today's ~360 KB. The
  existing first-level ack (today only with deadlock avoidance) would shorten the sender loop at the cost of an
  extra ack per packet; not modelled yet.
- **Two ERISCs + one DRISC** (`model/drisc_study.py`; the DRISC can only take RX work, so the sending ERISC stays the
  floor):
  - With measured DRISC costs (§14, lean stateful issue; catalog updated 2026-10-13): gains 10-19% on 1-hop scatter,
    11% on the 3-hop middle router and 47% on the multicast router (static; 10-19%, 5% and 34% event-driven at 32
    slots with 16-slot downstream channels). Lost if one DRISC serves two routers.
  - Header-only offload (DRISC decides, ERISC moves the payload) does not pay. Staging: 8 slots per router (~35 KB).
  - BH gives Metal 16 DRISCs for at most 10 active Ethernet cores, so one DRISC per router fits (8 of the 16 send on
    NoC0 only: fine for forwards; DRAM-bound local delivery would need an endpoint remap).
  - DRISC speed is measured equal to the ERISC's (alpha 1.0, §14). The remaining estimate is 14 cycles of header
    work in the DRISC's 45-cycle forward copy; at 60 cycles per copy the push design (P5) loses line rate on the
    3-hop router (124).
- **Receive pipeline (who decodes, who admits, who moves data; `model/pipeline_study.py`, 3-hop / multicast N/S
  router at 4 KiB, measured DRISC costs; event-driven with 24-slot downstream channels, static in parentheses):**
  - Every DRISC design reaches line rate on the 3-hop router (118 (117)), including the push design P5 (E1 polls,
    pushes every packet, relays downstream credits and retires slots; the DRISC decodes, admits, forwards on one NoC
    and delivers locally on the other), which leaves E1 the most headroom (77 static).
  - Multicast N/S is capped at ~153 by the DRISC's NoC port carrying both 4 KiB forwards; a router there moves four
    4 KiB transfers per packet (push, local, two forwards) through three NoC ports, so at 4 KiB some port carries two.
    Every design whose DRISC only forwards reaches it (P3, P4, P4d, P5L: 153). P5's DRISC also decodes, admits and
    delivers locally and is the limit (222 (248)).
  - Doing local delivery and the decode on E1 in parallel with the DRISC's forwards (P5L) takes multicast from 222
    to 153 and costs E1 76 more cycles in the static model (102 -> 178, with one credit per relay); the event-driven
    model, which relays whatever credits have arrived, is not limited by E1. On the 3-hop router (no local delivery)
    P5L only adds the decode to E1 (96).
  - 1D linear multicast (`model/validate_1d.py`): the model, with per-router send and receive rates, reproduces the 1D
    benchmarks within 2-11% (mean 5.6%, all low) and names the same critical ERISC as the delay injection. At 4 KiB
    with lean polling and batching, bidirectional 1D writes reach the wire limit without a DRISC; unidirectional
    writes (a middle router receiving and forwarding as much as it sends) go from 175 to 119 with P3s (E1 decodes,
    admits and pushes; the DRISC delivers locally on one NoC and forwards on the other), ~125 allowing for the model
    being low; other DRISC designs stop at 152-153. Scatter stays at 124-176.
  - One DRISC cannot serve two routers at 4 KiB (205 / 167 for two 3-hop routers on one DRISC), so the E/W routers
    need their own DRISCs, the NoC0-only ones.
  - Moving E0's sender-channel scan to E1 does not pay: with batching E0 is at 77-92 and saves 3-18 cycles while E1
    pays 16-21; without batching both ERISCs are over budget and the move only swaps the bottleneck.
  - Today a blocked packet is decoded again on every retry (`fabric_erisc_router.cpp:1597-1746`); that costs little
    throughput (retries happen while a downstream is full), and keeping the decoded action is a free ERISC-only fix.
- **Sender-channel polling (`model/polling_study.py`).** Every channel still has to be checked for new packets (each
  has its own producer), but today's visit makes three checks per channel: new packet, completions, connection or
  teardown. On an idle channel only the first can fire (no packets in flight means no completions; teardown is
  rare). Checking only that ("lean", no protocol change) takes E0 from 221 to 168 (4 channels) and from 356 to 225
  with VC1's 9 channels. Issuing all the packet checks as one block of reads ("fused") adds little if each read blocks
  (164 / 211) and a lot if the ERISC overlaps them (140 / 157): an untested property, one microbench. A summary
  doorbell (152 at any channel count) costs every producer an extra inline write per packet: the 3-hop forwarding
  core goes 185 -> 209. Draining up to 4 packets per visit amortizes the rest in a sustained stream: every scheme
  then reaches line rate on 4 channels, and the scheme matters again only with VC1 (scan 124, lean 117). Only the
  27-cycle total per idle channel is measured; its split into the three checks is estimated.
- **Batching only helps while packets queue (`model/burst_study.py`).** It is opportunistic (send what is queued,
  ack when idle), so in bursts of up to ~8 packets it gains nothing, and with a 150-cycle/packet producer (slower
  than the batched router, so little queues) at most 9% even at 64-packet bursts; with a fast producer, 13% at 16 packets and 25% at
  64. Short bursts are latency-bound (a lone 4 KiB packet takes ~940 cycles worker to delivery); lean polling helps
  every burst size (3-23%). Batching serves sustained streams (the 400G target), not bursty traffic.
- At 2 KiB (`--payload 2048`): batching needs 24-32 slots too (2D 1-hop: 148 at 16 slots, 81-93 at 32); the DRISC
  gains 11-16% on the 3-hop middle router, 12-19% on 1-hop scatter, nothing on 1-hop writes.

## 14. DRISC microbenchmark (measured)

Measured 2026-10-12 on a second P150x8 LoudBox (bh-lb-120-a06u28, firmware bundle 19.12.0; older bundles do not
enable DRISCs) with `tests/tt_metal/tt_fabric/drisc_microbench/`: slow dispatch, fabric off, a DRISC kernel and an
ERISC1 kernel (NoC1, as the receiving router), both at O3, 4 KiB payload. DRISC logical (0,2) (both NIUs in stream
mode), Ethernet core logical (0,4). Raw data: `results/drisc_mb.csv` / `.log` (the build these tables come from)
and `results/drisc_mb_*.{csv,log}`.

**Repeatability.** Runs of one build agree within 0.1 cycle, also on a second DRISC (3,2), a second Ethernet core
(0,7) and with wall-clock timing; only round-trip latency depends on placement, and the 4-slot pipeline is bimodal.
Rebuilding the kernels after unrelated edits moves back-to-back figures by up to ~7 cycles (code layout and register
allocation), so only compare figures from one build.

**Clock.** The DRISC's `mcycle` ticks 1.000 per wall-clock tick, and NoC writes from either core run at the same
53.5 B/cycle, so DRISC cycles are ERISC (1.35 GHz) cycles.

**Per operation (cycles, loop overhead included).** The DRISC matches the ERISC: alpha = 1.0.

| Operation | DRISC | ERISC |
| --- | ---: | ---: |
| timer read | 7.3 | 7.3 |
| L1 load, dependent | 8.2 | 8.2 |
| L1 load, 4 independent (per load) | 5.6 | 5.6 |
| stream register read | 10.3 | 10.3 |
| stream register read, 4 independent (per read) | 5.1 | 5.1 (6.1 in another build) |
| fence + L1 load (polling an L1 counter) | 14.4 | |
| NoC command-buffer ready read | 10.2 | |
| local stream register update | 3.2 | |

**NoC, back to back (cycles per operation).** "Issue" is until the core can issue the next one, with no barrier:
each payload write waits for the command buffer, which stays busy until the previous write has left, so for 4 KiB
it is the NoC's rate, not CPU work. "Drain" is until all have completed. NoC0 and NoC1 from the DRISC match within
0.4. "Stateful + trid": the stock `noc_async_write_one_packet_with_trid_with_state` (destination set once). "Lean":
the same with one command-buffer poll instead of two, the length kept in the state, and a stateful doorbell
(`lean_trid_write` / `lean_doorbell` in the benchmark).

| Operation | Issue | Drain | B/cycle |
| --- | ---: | ---: | ---: |
| DRISC write 64 B -> ERISC | 34.8 | 39.4 | |
| DRISC write 4 KiB -> ERISC | 70.4 | 76.7 | 53.4 |
| DRISC posted write 4 KiB | 70.2 | 72.1 | 56.8 |
| DRISC doorbell -> ERISC stream register | 24.8 | 29.3 | |
| DRISC forward copy (4 KiB + doorbell), stateless | 78.2 | 83.6 | 49.0 |
| DRISC forward copy, stateful + trid | 75.8 | 81.2 | 50.5 |
| DRISC forward copy, lean | 74.0 | 79.5 | 51.5 |
| DRISC read 4 KiB <- ERISC | 25.6 | 77.8 | 52.6 |
| ERISC write 4 KiB -> DRISC | 70.3 | 76.7 | 53.4 |
| ERISC doorbell -> DRISC stream register | 23.8 | 28.3 | |
| ERISC push (4 KiB + doorbell), stateless | 74.5 | 79.8 | 51.3 |
| ERISC push, stateful + trid | 75.3 | 80.4 | 50.9 |
| ERISC push, lean | 74.2 | 79.7 | 51.4 |

**One operation with the NoC idle (CPU cycles).** The wait for the previous operation's acks is not timed, and the
addresses and transaction id are computed before the timer starts, so this is what the core pays when the port is
free. Figures include one timer read (7.3).

| Operation | ERISC | DRISC |
| --- | ---: | ---: |
| 4 KiB write, stateless | 28 | |
| 4 KiB write, stateful + trid | 33 | |
| 4 KiB write, lean | 24 | |
| doorbell, stateless | 27 | 31 |
| doorbell, lean | 19 | |
| push / forward copy (write + doorbell), stateless | 53 | |
| push / forward copy, stateful + trid | 57 | 65 |
| push / forward copy, lean | 39 | 38 |

The stock stateful write costs more than the stateless one: it polls the command buffer twice (once to set the
transaction id, then again, right after that register write, to issue). One poll is enough when nothing else issues
on that command buffer in between.

**Latency.** A 4 KiB write plus waiting for its ack takes ~398 cycles from either core, on either NoC, for every
placement tried: the request and the ack travel the same NoC, so together they go once round the torus. A 4 KiB
read plus waiting for the data takes ~404. A doorbell ping ERISC -> DRISC -> ERISC (out on NoC1, back on NoC0)
takes 168-171, 276 or 330 cycles depending on placement; 335-337, 444 or 497 with 4 KiB each way.

**Pipeline.** The ERISC pushes 512 x 4 KiB packets into a ring of DRISC staging slots and waits when the ring is
full; the DRISC forwards each one back to the ERISC (payload write with a transaction id + doorbell) and returns
slots in batches once its writes have left its L1. Both issue modes run in one build (`results/drisc_mb_pipeline.*`,
two runs within 1 cycle); "lean" uses lean stateful issue on both cores, with the DRISC's payloads, landing
doorbells and credits each on their own command buffer.

| Staging slots | 1 | 2 | 4 | 8 | 12 | 16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| stateless: cycles/packet | 471 | 292 | 184 | 123 | 113 | 111 |
| stateless: ERISC busy / DRISC work per packet | 75 / 189 | 75 / 152 | 75 / 148 | 75 / 114 | 75 / 107 | 75 / 106 |
| lean: cycles/packet | 430 | 253 | 159 | 112 | 107 | 104 |
| lean: ERISC busy / DRISC work per packet | 51 / 153 | 51 / 146 | 58 / 124 | 60 / 103 | 60 / 100 | 60 / 99 |

"ERISC busy" is the time inside its pushes; "DRISC work" is the time in loop iterations that forwarded a packet or
returned credits, including two timer reads (~15 cycles) per iteration. Lean issue gains only 6% at 16 slots. With
the port free a lean push costs the ERISC ~32 cycles and a lean copy the DRISC ~31 (above), yet here each spends
~30 cycles more per packet waiting for its command buffer, so neither is CPU-bound: each core's 4 KiB writes leave
at ~40 B/cycle instead of 53.5. Unlike a router, this test sends every packet into and out of both tiles at once
(the DRISC forwards back to the same ERISC), and the slowdown fits each tile's NoC port or L1 being shared between
the two directions. Not yet tested: the same traffic in both directions without the pipeline's dependencies, and
the DRISC forwarding to a third core.

**Command buffers and NoC ports (part 3).** The DRISC writes 4 KiB packets into the ERISC's landing ring
(`results/drisc_mb_cmdbuf.*`, two runs within 4%). Each NIU has 4 command buffers, all the DRISC's own; writes use
at most 3, because the read buffer keeps a read control word that firmware sets once at boot and the read path never
rewrites (a first version used it for writes, hung the board and needed a reset).

| DRISC, 4 KiB writes | Cycles per write | B/cycle |
| --- | ---: | ---: |
| one write with the NoC idle: until its command buffer is free again | 30 | |
| back to back on one NoC, round-robin over 1 / 2 / 3 command buffers: issue | 70.6 / 69.3 / 68.2 | |
| ... drain | 76.8 / 76.7 / 76.8 | 53.3 / 53.4 / 53.3 |
| alternating NoC0 / NoC1 (1 or 2 buffers each): drain | 41.6-42.4 | 97-98 in total |
| pull (read 4 KiB from the ERISC) + forward (write 4 KiB to it), same NoC: per pair | 92-96 | 43-45 each way |
| ... pull on NoC0, forward on NoC1 | 92-95 | 43-45 each way |

- A write frees its command buffer ~20 cycles after issue when the port is free; back to back, every buffer waits
  for the port, so more buffers do not let the DRISC run ahead (68-71 cycles per write with 1-3 buffers). What
  avoids spinning is checking readiness (a 10-cycle register read) and doing other work in between.
- One DRISC drives both NoCs at once at ~98 B/cycle (1.84x one NoC).
- Reading and writing at the same time capped at ~44 B/cycle each way, ~92 cycles per 4 KiB in + out, whether or not
  the two directions use different NoCs. Part 4 below shows this is not a limit of either tile.

**Forwarding into other Ethernet cores.** The DRISC pulls from the benchmark's ERISC (A) on NoC0 and writes copies
into the L1 of other active Ethernet cores on the chip (B, C; no kernel there), as a router forwards to its
downstream routers (`results/drisc_mb_fwd.*`, two runs within 0.3 cycles; B and C received A's data).

| Per 4 KiB packet | Cycles | In / out B/cycle |
| --- | ---: | ---: |
| copy to B on NoC0 only (reference) | 77.6 | - / 52.8 |
| pull from A + copy to B, both NoC0 | 81.1 | 50.5 / 50.5 |
| pull from A on NoC0 + copy to B on NoC1 | 78.7 | 52.1 / 52.1 |
| pull + copies to B on NoC0 and C on NoC1 (multicast) | 102.4 | 40.0 / 80.0 |
| pull + copies to B and C both on NoC1 | 148.5 | 27.6 / 55.1 |

- With A only read from and B only written, the DRISC moves 4 KiB in and 4 KiB out in 79-81 cycles, close to one
  direction alone.
- A DRISC tile carried 120 B/cycle in total (multicast row). Multicast needs both NoCs: 102 cycles per packet with one
  copy per NoC, 149 with both copies on one NoC.

**Two DRISCs on one Ethernet tile (part 4).** A packet is never read from and written into the same Ethernet tile,
but when a link carries traffic both ways its tile is read from (received packets, by its own DRISC) and written
into (packets to send, by another router's DRISC) at once. DRISC 1 (0,2) pulls 4 KiB packets from ERISC A's source
buffer while DRISC 2 (1,2) writes 4 KiB packets into A's landing ring, 256 each; no kernel runs on A. Each variant
ran as its own program (`results/dbg_v*.log`).

| Variant | DRISC 1 pull: cycles/packet (B/cycle) | DRISC 2 write: cycles/packet (B/cycle) |
| --- | ---: | ---: |
| pull alone, NoC0 | 73.6 (55.7) | - |
| write alone, NoC0 | - | 72.7 (56.3) |
| both at once, NoC0 | 78.4 (52.3) | 78.9 (51.9) |
| DRISC 1 pulls and writes (one DRISC both ways), NoC0 | 79.3 per pull + write (51.7 each way) | - |
| DRISC 1 pulls on NoC0, writes on NoC1 | 74.1 per pull + write (55.2 each way) | - |

- With no kernel on A, its tile carried ~104 B/cycle in and out together, so the ~44 B/cycle cap in part 3 is not an
  Ethernet-tile limit, and one DRISC working both ways is not slower either. What differed in part 3: A's ERISC
  kernel was polling a stream register at a barrier throughout. Whether that polling causes the cap is untested.
- **Hang:** the full set of part-4 variants (including both DRISCs at once on different NoCs: pull NoC0 + write NoC1,
  and pull NoC1 + write NoC0) hung the chip twice; the host's PCIe reads then timed out. Each variant above passed on
  its own; the two split-NoC variants were not yet run alone. Resetting chip 0 alone afterwards left its card off the
  PCIe bus (7 of 8 visible), which needs a PCIe rescan or reboot.
- Design consequence: a pattern that hung the chip is ruled out, without probing its exact boundary. Today's fabric
  already writes into Ethernet tiles from different cores on both NoCs (data forwards on NoC1, credits and acks on
  NoC0, workers on their own NoC), so what was new in the hangs is a remote read of Ethernet L1 on one NoC while
  another core writes that tile on the other. Workers can write on either NoC, so a pull design cannot rule that
  out; a push design (the ERISC writes the packet into its DRISC) never reads Ethernet L1 remotely.

**What this means for the model (catalog not yet updated).**
- The model's default DRISC speed (alpha 1.0) holds. A DRISC forward copy runs at 74-84 cycles back to back
  depending on how it is issued (the NoC's rate; the model assumes 77) and costs the DRISC's CPU 38 cycles with lean
  issue when its port is free.
- A push needs no barrier: with transaction ids the ERISC retires the slot once the write has left. With the port
  free it costs the ERISC's CPU 39 cycles with lean issue (53 stateless, 57 with the stock stateful API); the model
  charges 30 + 24 per doorbell. It still occupies the Ethernet tile's NoC port for one 4 KiB transfer (~77 cycles at
  ~53 B/cycle), the same as forwarding the packet itself; the model charges that separately.
- Lean issue would also apply to the router's own stateful writes, which use the stock two-poll API today (~9 cycles
  per write here), wherever a command buffer keeps one destination.
- Fused polling: independent stream-register reads overlap, 5-6 cycles each against ~10 one at a time. The model's
  "overlapped" estimate (3) is about half that.
- One DRISC forwarding one stream of 4 KiB packets back to its ERISC reaches 104 cycles/packet with lean issue and
  16 staging slots (64 KiB of its 128 KiB L1), 107 with 12 and 112 with 8 (stateless: 111 / 113 / 123). Neither
  core is CPU-bound there, and the ERISC's kernel polls throughout, as in part 3's capped runs. Forwarding into
  another Ethernet core, the DRISC moves a packet in and out in 79-81 cycles.
- An Ethernet tile with no kernel running carried ~104 B/cycle in and out together (part 4); the ~92-cycle cap seen
  in part 3 is not a tile limit. Its cause (possibly the ERISC polling) and the split-NoC hang are open.
- Multicast routers need a DRISC that sends on both NoCs (102 cycles per packet with one copy per NoC, 149 on one).
- Not measured yet: one DRISC sending on both NoCs at once.

## 15. 2D baseline on a third host, and a step at 4 KiB

Measured 2026-10-13 on bh-lb-120-a04u24 (P150x8, firmware bundle 19.12.0, same build as sections 8-14), no BW
telemetry. Results: `results/a04_d2k*.csv`, `results/a04_d4k*.csv` (`exp_2d.yaml`, `exp_2d_4k.yaml`,
`exp_2d_trim.yaml`, `exp_2d_trim_4k.yaml`), probes `results/p*.csv`.

**Reproducibility.** Every 2D number from section 8 (first host, Oct 9) reproduces within 0.2%, and the DRISC
microbenchmark's part 1 within 2% of section 14 (second host). Cycles per packet:

| Test | 2 KiB | 4 KiB | 2 KiB trimmed | 4 KiB trimmed | model (trimmed) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1-hop E/W write | 223.0 | 221.5 | 168.0 | **187.6** | 167 |
| 1-hop E/W scatter | 218.6 | 220.8 | 191.0 | 191.1 | 191 |
| 1-hop N/S write | 223.2 | 221.9 | 152.4 | **187.6** | 149 |
| 1-hop N/S scatter | 219.4 | 221.4 | 206.4 | 206.5 | 206 |
| 1-hop E/W bidirectional write | 224.1 | 221.4 | 197.3 | 196.9 | |
| 3-hop E/W write | 222.9 | 220.9 | 188.6 | 196.4 | 185 (middle router) |
| 3-hop E/W scatter | 223.1 | 221.1 | 191.1 | 197.9 | |
| mesh multicast write | 180.6 | 181.9 | 168.8 | 168.2 | |

Untrimmed, every single flow is TX-bound at ~221 at both sizes (empty-channel polling, section 8). The 4 KiB
trimmed column is new (section 8's "trimmed" run was 2 KiB): the model predicts the same as at 2 KiB, and is right
except for the two 1-hop writes, which both stop at exactly 187.6.

**The step** (`exp_2d_size_sweep.yaml`, 1-hop write, trimmed):

| Payload (B) | 2048 | 2560 | 3072 | 3584 | 3840 | 3968 | 4000 | 4032 | 4064 | 4080 | 4096 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E/W | 168.2 | 168.6 | 169.1 | 168.3 | 169.1 | 170.0 | 170.0 | 187.6 | 187.6 | 187.6 | 187.6 |
| N/S | 152.4 | 152.4 | 152.4 | 152.5 | 152.5 | 152.5 | 152.5 | 187.6 | 187.6 | 187.6 | 187.6 |

Flat to 4000 B, then a fixed 187.6 from 4032 B: a per-packet cost, not a bandwidth limit. The 2D header is at
least 80 B (`fabric_context.cpp`, `HybridMeshPacketHeaderT`), so the step is where one Ethernet transfer (payload +
header) goes over 4096 B.

What it is not (all at 4 KiB, trimmed, unperturbed unless noted):
- **Local delivery:** skipping the receiver's NoC write (`EXP_FLAGS=1`) leaves 187.6 (at 2 KiB the same flag takes
  N/S from 152 to 140, the model's TX load, so the flag works).
- **Credit loops:** 12 sender slots change nothing (187.6) and 8 receiver slots add 1-5 cycles. Only 8 sender
  slots shows the sender loop, at ~1.7k (2 KiB) and ~1.8k (4 KiB) cycles per round trip, so 16 slots allow
  ~106-113 cycles/packet: not today's limit, but close to the 117-cycle target.
- **Configuration:** slot counts (16-17 sender, 16 receiver, 4448 B buffers) and router mode (generic, no credit
  batching) are the same at 2 and 4 KiB.
- **Either ERISC's CPU:** at 4032 B N/S, +20 cycles of TX or RX delay leaves 187.6 (both cores have slack); at +40,
  RX reaches 208.4 at both 4000 and 4032 B (no extra RX work above 4 KiB) and TX 194.6 vs 200.1 (at most ~5 more).
- **The link or TX queue rate:** neighbor exchange on this host sends 4096 + 48 B at 124.4 cycles (44.5 GB/s).
- **The TX-queue wait:** profiling shows the post-payload spin at 14 cycles at both sizes and no blocked sends,
  but every profiling mode (even counters only, no per-iteration timestamps) slows the router to 216-254 cycles,
  above the floor, so it cannot see the floor directly.

**The cause: Ethernet sequence acknowledgements on a one-way link.** `TT_FABRIC_TXQ_COUNTERS=1` (new, see
Reproducing) samples each TX queue's hardware counters and configuration from the ERISC that drives it, every 64
outer loop iterations (`txq_report.py` prints them):

| Sender's data queue (TXQ0), per fabric packet | 4000 B | 4032 B | neighbor exchange 4096 B |
| --- | ---: | ---: | ---: |
| commands (payload + remote pointer update) | 2 | 2 | 2 |
| Ethernet packets | 2.0 | **3.0** | 3.0 |
| resends | 0 | 0 | 0 |
| receiver acks on its TXQ1 | 1 | 1 | 0.25 |

Configuration on every queue: maximum Ethernet packet 4114 B, accept-ahead 32, `ctrl = 0x1` (resend mode),
remote sequence timeout 40000, local sequence update timeout 8000 cycles. So 4032 B + header goes out as two
Ethernet packets, and neighbor exchange at 4096 + 48 B does too, yet runs at 124. Experiments:

| Test (cycles/packet) | below 4114 B | above 4114 B |
| --- | ---: | ---: |
| 2D 1-hop write, trimmed, one way | 152.5 (N/S), 170.0 (E/W) | 187.6 |
| 2D 1-hop + a 256 B reverse flow on the same link (`exp_2d_reverse.yaml`) | <= 152.4 / <= 170.1 | **<= 159.4 / <= 171.3** |
| neighbor exchange, both directions | 123.7-124.1 | 124.5 |
| neighbor exchange, receiver acks every packet (`TT_FABRIC_EXP_RECEIVER_AMORT=1`) | 133.0-133.2 | 133.5 |
| neighbor exchange, one direction (`exp_nbr_oneway.yaml`) | 129.6-130.3 | **187.5** |

(Reverse-flow rows are upper bounds: the reverse flow was the slowest, and the summary reports the maximum over
flows.) Ack frequency does not matter (per-packet acks cost neighbor exchange 9 cycles of RX CPU at every size, no
step), and neither does 2D: the fast path shows the same 187.5 when it runs one way. What matters is data on the
reverse data queue. The queues run in resend mode, so the sender holds each Ethernet packet until the receiver
acknowledges its sequence number; the acknowledgements ride on reverse traffic of the same queue, and when that
queue is idle (the receiver's acks go out on TXQ1 in two-ERISC mode) only on the sequence update sent every
`local_seq_update_timeout` = 8000 cycles. Both floors fit one window of W packets per update: 3 x 8000 / 187.5 gives
W = 128, which predicts 125 for 2 packets per fabric packet (measured 129.6-130.3: the fast path's 124-cycle send
rate runs into the window at the end of each period). This was inferred from the counters and the four experiments;
section 16 confirms it on a bare link.

Consequences:
- Any one-way stream on a link (most 2D unicast, unidirectional 1D, each hop of a multi-hop route) is capped at
  ~125-130 cycles/packet up to 4114 B on the wire and 187.5 above, independent of ERISC work; traffic in both
  directions is not. The 400G-at-4-KiB (117) and 200G-at-2-KiB (110) targets are out of reach for one-way traffic
  until this changes. Levers, none tried: fewer Ethernet packets per fabric packet (one remote pointer update per
  several packets: 1.25 per packet at 4 -> ~78), keeping packets within 4114 B on the wire (2D: payload <= 4114 -
  header), sending receiver acks on the data queue so they carry the sequence acknowledgements (as single-TXQ mode
  did), or a shorter sequence-update timeout (`ETH_TXQ_LOCAL_SEQ_UPDATE_TIMEOUT` on the receiving side's TXQ0: a
  plain TX queue register, which firmware sets to 8000 at boot).
- The placement model now has it (`fabric_exp/model/README.md`, "Link"): one-way scenarios get a keepalive floor of
  (Ethernet frames per packet) x keepalive / 128, and the wire is computed from Ethernet framing (section 16). Its
  results assume the keepalive at 1000 cycles; `Options(keepalive=8000)` reproduces 187.5.
- `TT_FABRIC_TXQ_COUNTERS=1` on its own slowed the trimmed 1-hop runs by 3-6 cycles/packet below the step (2.6-3.9%),
  more than its sampling rate explains; use it for counts, not timing.
- 1D Linear with channel trimming switches the routers to the fast path with credit batching (sender 2, receiver 4),
  and the replay hung on its first test (3968 B), so that configuration is ruled out as a comparison; recovered with
  `tt-smi -r all`.

## 16. Ethernet TX queue probe (bare link, no fabric)

`test_eth_txq_probe` (`tests/tt_metal/tt_fabric/eth_txq_probe/`) runs one kernel on ERISC0 at each end of a link
between chips 0 and 1, with fabric off. The sender sends packets as the router does: payload into a landing slot, then
a pointer update on the same queue. The receiver takes them in order and may send 16 B messages back. Each row is
50000 packets, timed after the first 5000; 4176 B is a 4096 B payload plus an 80 B 2D header. Both links between
chips 0 and 1 gave the same numbers to 0.1 cycle, with zero resends and zero ordering errors in every run.

| Variant (cycles/packet) | 2048 B | 4000 B | 4176 B |
| --- | ---: | ---: | ---: |
| 1. TXQ0, nothing back | 125.0 | 125.0 | 187.5 |
| 3. TXQ0, 16 B back on TXQ1 every packet (two-ERISC credits today) | 125.0 | 125.0 | 187.6 |
| 5. as 3, sender gated on the counts as credits (16 slots) | 125.0 | 125.0 | 187.6 |
| 4. TXQ0, 16 B back on TXQ0 every 4 packets | 61.0 | 111.8 | 118.1 |
| 6. as 4, gated (16 slots) | 78.0 | 111.8 | 118.1 |
| 2. even packets on TXQ0, odd on TXQ1, nothing back | 66.3 | 111.8 | 118.0 |

- **The floor is the Ethernet layer alone.** With no router, one-way traffic gives the same 125 and 187.5. In variant
  1 the receiver's idle TXQ0 sent 782 Ethernet packets in 6.25M cycles: one sequence update per 8000 cycles, against
  128 data-side packets each. Every idle queue sends these, including the sender's unused TXQ1.
- **Acknowledgements are per queue.** Traffic back on TXQ1 does not release TXQ0 (variants 3 and 5). Traffic back on
  TXQ0 does, even one 16 B message per 4 packets (variants 4 and 6). This confirms the explanation in section 15.
- **Without the floor the link runs at wire rate.** 4176 B takes 118.1 cycles (47.7 GB/s of payload) and 4000 B
  takes 111.8. The slope between 2048 and 4000 B is 38.4 B/cycle (~415 Gb/s including framing). A 4 KiB payload with
  its 80 B header cannot go below ~118 cycles on one link, so the 117-cycle target is the wire itself.
- **Credit gating.** With 16 slots and one credit per 4 packets, gating costs nothing at 4 KiB but 17 cycles at
  2 KiB (78 vs 61): the credit round trip.
- **Two queues.** Splitting the packets over two queues also reaches wire rate at 4 KiB (118.0) with nothing coming
  back. At 2 KiB it gives 66.3: two windows of 128 per 8000 cycles make a 62.5 floor. Merging needs nothing beyond
  one counter per queue at the receiver, since packet k comes on queue k mod 2. One core drove both queues without
  trouble; that is the arrangement tt-metal#29073 hung on inside the router.
- **Counter caveat.** Here `PKT_START_CNT` and `PKT_END_CNT` undercount by up to ~20% at 4000 B and above, while
  `TRANSFER_CNT` is exact. At 2048 B they are exact (2.00 Ethernet packets per packet). For larger sizes the
  evidence for the packet counts is the timing: 125 = 2 x 62.5 and 187.5 = 3 x 62.5. (`WORD_CNT` counts 96 B units,
  not 16 B words, per the ISA documentation below.)
- **The ISA documentation agrees** ([BlackholeA0/EthernetTile/EthernetTxRx.md](https://github.com/tenstorrent/tt-isa-documentation/blob/main/BlackholeA0/EthernetTile/EthernetTxRx.md)).
  TT-link packets carry 8-bit sequence numbers, consistent with the 128-packet window. TX queue *n* talks to the
  remote RX queue *n*, and acknowledgements piggy-back on that queue's outgoing packets. Otherwise
  `ETH_TXQ_CTRL` bit 0 (keepalive) sends a sequence update every `ETH_TXQ_LOCAL_SEQ_UPDATE_TIMEOUT` cycles, unless
  another packet went out in that period. TX queue 2 cannot send MMIO (register) writes and RX queue 2 cannot
  receive them, so a third queue could carry L1 writes (counter credits) but not stream-register pointer updates.

### Shorter keepalive

The keepalive period (`ETH_TXQ_LOCAL_SEQ_UPDATE_TIMEOUT`) is an ordinary TX queue register. Probe variants 7 and 8
set the receiver's TXQ0 period for the run and restore the firmware's 8000 afterwards. The value read back matched
in every run, and every run found 8000 at start. No resends, RX drops or ordering errors occurred.

| Cycles/packet | 2048 B | 4000 B | 4176 B |
| --- | ---: | ---: | ---: |
| 1. nothing back, keepalive 8000 (firmware) | 125.0 | 125.0 | 187.6 |
| 7. nothing back, keepalive 4000 | 68.2 | 111.8 | 118.1 |
| 7. nothing back, keepalive 2000 / 1000 / 500 | 61.0 | 111.8 | 118.1 |
| 8. today's arrangement (gated credits on TXQ1), keepalive 4000 to 500 | 75.0 | 111.8 | 118.1 |

The keepalive count is exactly run time / period: 5896 at 1000 cycles over a 5.9M-cycle run. Keepalives travel in the
direction opposite the data and are skipped when that queue sends anything else. Even at 500 they cost the forward
link nothing measurable. Variant 8's 75.0 at 2 KiB is the credit round trip with 16 slots, as in variant 6.

Router knob `TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=N`: ERISC0 sets the period on both of its tile's queues before the
handshake (a trimmed receive-only router returns the acknowledgements too) and restores the firmware value in
teardown. Same build, `TT_FABRIC_NO_BW_TELEMETRY=1`:

| Cycles/packet | off | 1000 | 2000 |
| --- | ---: | ---: | ---: |
| 2D 1-hop write N/S, 4096 B | 187.6 | 152.5 | 152.5 |
| 2D 1-hop write E/W, 4096 B | 187.6 | 169.8 | 169.9 |
| 2D 3-hop write E/W, 4096 B | 196.4 | 186.5 | 186.9 |
| 2D 3-hop scatter E/W, 4096 B | 197.9 | 191.2 | 191.2 |
| 2D 1-hop write, 2048-4000 B (size sweep) | 152.4-170.0 | same | - |
| 2D 1-hop write, 4032-4096 B (size sweep) | 187.6 | 152.5 / 170.0 | - |
| neighbor exchange one way, 2048 / 4000 / 4096 B | 129.6 / 130.3 / 187.5 | 122.8 / 122.9 / 122.9 | same |
| everything else at 2 and 4 KiB (bidir, scatter 1-hop, multicast) | | within 1 | within 1 |

The step is gone and nothing else moved. With the knob off, the build matches the earlier baselines exactly. One-way
traffic is now limited by the routers' CPU again, below the bidirectional neighbor exchange (124). Not yet covered:
long runs (any firmware interaction with the keepalive period), 1D, and systems that tunnel host access over the
same queues.

For the router, two arrangements reach wire rate at 4 KiB; neither is built yet:
- **Credits on the data queue.** Today E1 sends credits on TXQ1. If one end of each link swaps queue numbers (its E0
  sends data on TXQ1, its E1 sends credits on TXQ0), each data stream is paired with the opposite direction's
  credits, and each ERISC still owns exactly one queue.
- **Data split over two queues.** E1's credits then need a third queue (TXQ2, which Metal never configures), or must
  share a queue with E0.

## 17. Sender-channel polling: what a visit costs, and which reads an ERISC can overlap

**Where an empty visit goes** (compiled untrimmed 2D router, E0, from the baseline run; disassembly with source
lines). A visit to a sender channel with no traffic is 13 instructions: load the channel's stream id from the stack,
read its free-slot stream register and compare (no packet), read its completion counter in L1 and compare (nothing
new). Router-to-router channels skip the connection check (`SENDER_CH_n_LIVE_CHECK_SKIP`), the data cache is off,
credits are L1 counters. The two reads are used at once, so ~15 of the measured ~27.5 cycles are waiting. The worker
channel adds a TX-queue can-send check (two register reads, ~13) and a teardown check (L1 read, ~12): ~53 cycles of
checks, which is the "~52 polling + loop" of section 8; the loop itself is 2 instructions (the termination and
context-switch checks run once per 32 iterations). `send_next_data` is 45 instructions (23 of them pointer and
counter bookkeeping) plus a ~12-cycle wait for the TX queue to latch the payload command before the pointer update.

**Which reads overlap** (`tests/tt_metal/tt_fabric/erisc_load_ubench`, ERISC0, fabric off, data cache off as in the
router; every timed loop is one aligned asm block, results identical on two Ethernet cores). Cycles the reads cost
beyond the same code without them; K = independent instructions between issuing the reads and using them:

| Reads | K = 0 | 2 | 4 | 6 | 8 | 16 | 48 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stream register, used at once | 7 | | | | | | |
| L1 word, used at once | 8 | | | | | | |
| stack (local data memory), used at once | 2 | | | | | | |
| stream read, then L1 read, each used at once (today's empty visit) | 15 | | | | | | |
| one stream read, K, use | 7 | 6 | 6 | 6 | 6 | 6 | 6 |
| one L1 read, K, use | 8 | 7 | 7 | 7 | 8 | 8 | 8 |
| two stream reads, K, use both | 7 | 5 | 3 | 2 | 2 | 2 | 2 |
| four stream reads, K, use all | 7 | 5 | 4 | 4 | 4 | 4 | 4 |
| stream + stack read, K, use both | 7 | 5 | 3 | 2 | 2 | 2 | 2 |
| stream + L1 read (the router's pair), K, use both | 8 | 6 | 4 | 3 | 9 | 9 | 9 |
| two L1 reads / L1 + stack, K, use both | 9 / 7 | 7 / 5 | 5 / 3 | 3 / 3 | 9 / 9 | 9 / 9 | 9 / 9 |
| stream + L1, then a stack read used at once mid-way | 10 | 5 | 8 | 8 | 8 | 11 | 11 |
| stream + L1, then a register read used at once mid-way | 10 | 10 | 10 | 10 | 10 | 16 | 16 |

- A read issued alone is not hidden, however much work follows it (6-8 cycles). A load into `x0` does not count as
  a second read (the core appears to drop it).
- Stream-register reads issued together overlap completely: two cost 2 cycles, four cost 4, from ~4-6 instructions
  on, at any distance.
- A group containing an L1 read overlaps only if used 5-7 instructions after issue (3-4 cycles); used later it costs
  9-10 (still less than 15).
- A stack reload or another register read used at once while reads are outstanding waits for them (+6 to +14).
- Per channel: a prefetched stream + L1 pair one visit ahead (~10+ instructions) costs ~9 instead of ~15. Grouping
  every channel's free-slot read (4 reads, 4 cycles instead of ~28) and reading the completion counter only on
  channels with packets in flight avoids the L1 reads that overlap poorly. The model's polling parameters
  (`fused_per_channel: {blocking: 9, overlapped: 3}`, the 11/10/6 split, `loop: 25`) predate these numbers.

**Other registers the router reads** (same microbenchmark, extended: `results/erisc_regs_overlap.log`). TX queue 1's
command and status registers, a NoC write-request status register (the receiver's "write flushed" check) and a NoC
command buffer's ready register, read only, with the TX queue and the NoC idle:

| Reads | K = 0 | 2 | 4 | 5+ |
| --- | ---: | ---: | ---: | ---: |
| TX-queue command, used at once | 7 | | | |
| TX-queue busy check (command read discarded, status used at once) | 8 | | | |
| NoC write-flushed status / command-buffer ready, used at once | 7 / 7 | | | |
| busy check issued early, status used after K | 8 | 7 | 7 | 7 |
| receive-count stream register + busy check, use both | 8 | 6 | 4 | 3 |
| receive-count stream register + write-flushed status, use both | 7 | 5 | 3 | 2 |
| all three (E1's per-iteration register reads), use all | 8 | 6 | 4 | 4 |
| four stream registers + an L1 word (E0's teardown semaphore folded into the grouped read) | 14 | 14 | 14 | 8 at K = 6-7, else 14 |
| busy check with both reads into one register (as the router's code does) | 8 | | | |
| E1's three, busy check into one register | 8 | 6 | 4 | 4 |
| four stream registers + busy check (E0's grouped read with its can-send check) | 14 | 14 | 14 | 8 at K = 5-6, else 14 |

- TX-queue and NoC registers cost the same as stream registers (7 used at once), and the busy check's two reads
  overlap each other (8, not 14). Issued early on their own they are not hidden, like any lone read.
- Issued together with a stream-register read they overlap: E1's three per-iteration register reads (receive count,
  write flushed, TX-queue busy before the ack), ~22 cycles of waiting when each is used at once, cost 4 together.
- Up to four reads in flight overlap; a fifth or sixth makes the group cost ~14 (four stream reads + an L1 word, or
  + the busy check's two reads), more than reading them separately (4 + 8). So neither E0's teardown semaphore nor
  its can-send check can join its grouped read of four, while E1's group (receive count, busy check's two reads,
  write flushed: four reads) fits. Loading the busy check's discarded read and its status read into one register
  costs nothing extra.

**In the router** (generic TX path; `TT_FABRIC_EXP_FLAGS` bit 7 `EXP_TX_LEAN_COMPLETIONS`: a per-channel sent count,
and the completion/ack counter is read only while sent != processed; bit 8 `EXP_TX_GROUPED_PACKET_CHECK`: the first
four VC0 channels' free-slot registers read back to back in one asm block at the top of each iteration, the raw values
kept in registers and masked where each channel uses them). Same build, `TT_FABRIC_NO_BW_TELEMETRY=1`,
`TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000` (so the Ethernet floor does not hide the change), cycles/packet:

| Untrimmed 2D, 4 KiB | off | lean (128) | grouped (256) | both (384) |
| --- | ---: | ---: | ---: | ---: |
| 1-hop write E/W | 221.5 | 208.9 | 198.4 | **186.2** |
| 1-hop write N/S | 222.2 | 208.8 | 200.1 | **188.1** |
| 1-hop bidirectional write | 222.0 | 207.7 | 199.8 | **187.1** |
| 3-hop write | 221.2 | 204.5 | 197.8 | **185.4** |
| 1-hop scatter E/W / N/S | 221.1 / 221.3 | 205.7 / 210.3 | 201.1 / 206.2 | **191.2 / 206.4** (E1-bound) |
| mesh multicast write / scatter | 182.0 / 216.1 | 184.1 / 216.1 | 173.3 / 216.0 | **173.6 / 216.0** |

- The two add up: -13 and -23, -34 to -36 together on single-flow writes (2 KiB: -35 to -38). Scatter stops at E1's
  scatter delivery (191 / 206), so E0 is no longer the limit there.
- Grouping gives what the microbenchmark predicts (~6 per channel). Lean gives ~13 instead of ~24: the compiler
  keeps the sent counts on the stack, so each check is a stack reload + compare (~3 cycles instead of the L1 read's
  ~10, ~7 saved per idle channel), and the busy channel pays ~6 for the increment and its own check.
- Trimmed 2D 4 KiB: the 2-channel generic routers gain too (1-hop E/W 169.8 -> 165.0, bidirectional 196.0 -> 175.5,
  3-hop 187.1 -> 171.4); single-channel routers take the speedy path and are unchanged; multicast +2.
- Payload validation passes with both flags and the keepalive at 1000: `test_fabric_2erisc_quick.yaml` (9 tests) and
  `test_fabric_mesh_sanity_common.yaml` (96 tests: Linear and 2D mesh, unicast and multicast, writes, atomics,
  fused atomics, scatter, 1-4 links). (A benchmark config switched to validation mode fails with or without the
  flags, so it is not a usable check.)

**Moving the upstream side to E1** (`TT_FABRIC_EXP_FLAGS` bit 9 `EXP_TX_UPSTREAM_ON_E1`, generic path, two-ERISC
mode): E1 turns completions into credits for the upstream worker or router and handles worker connections (checked
every 16th iteration) for the sender channels E0 services; E0 still reads the completion counters for its own
receiver-slot accounting (each core keeps its own processed count). E1 initialises the worker interfaces and caches the
static connections, and the builder gives E1 E0's per-channel connection protocol. Payload validation passes
(`test_fabric_2erisc_quick` 9 tests, `test_fabric_mesh_sanity_common` 96 tests, alone and with bits 7+8). Same build,
keepalive 1000, cycles/packet:

| | polling flags (384) | + upstream on E1 (896) |
| --- | ---: | ---: |
| untrimmed 4 KiB 1-hop write E/W / N/S | 186.4 / 188.4 | **171.9** / 187.1 |
| untrimmed 4 KiB bidirectional / 3-hop write | 187.1 / 185.4 | 201.7 / 215.1 |
| untrimmed 4 KiB scatter, multicast | 191-216 | 237-283 |
| untrimmed 2 KiB 1-hop write E/W / N/S | 188.5 / 184.5 | **171.8** / 187.0 |
| trimmed 4 KiB, all tests | 153-216 | +5 to +74 |

- In a one-way test the sending chip's E1 has nothing to receive, so it absorbs the credit work for free and E0 is
  no longer the limit: 1-hop E/W drops to ~172, now set by the receiving chip's E1.
- But every E1 now runs the upstream step every iteration, including the receiving chip's, whose own E0 sends
  nothing: four completion-counter reads used almost at once, four subtract-and-branch pairs into far code, and the
  iteration counter, ~35-45 cycles per iteration spent finding nothing to do. Wherever the receiving E1 was near its
  limit (N/S, scatter, 3-hop, multicast, bidirectional, trimmed) that cost dominates.
- Second version: an in-flight counter in stream register 24 (E0 adds one per packet sent, E1 subtracts the
  completions it credits); E1 skips the counter reads while it reads zero, except every 16th iteration for the
  connection check. Validation passes again (9 + 96 tests). Against the polling flags alone (384 -> 896):

  | | 4 KiB | 2 KiB |
  | --- | ---: | ---: |
  | untrimmed 1-hop write E/W | 186.5 -> **166.1** | 188.4 -> **165.9** |
  | untrimmed 1-hop write N/S | 188.2 -> 180.1 | 184.6 -> 179.9 |
  | untrimmed 3-hop write | 185.8 -> 203.1 | 185.5 -> 203.0 |
  | untrimmed bidirectional write | 187.1 -> 221.6 | 188.0 -> 220.9 |
  | untrimmed scatter (E1-bound) | 191-206 -> 230-244 | same |
  | untrimmed multicast write / scatter | 173.6 / 216.1 -> 234.9 / 292.6 | similar |
  | trimmed 1-hop write E/W / N/S | 165.1 / 152.6 -> 164.1 / 177.2 | |

  One-way writes gain 8-22 more than the first version; traffic in both directions loses more (the counter adds work
  on both cores when both are busy). The receiving E1 still pays ~38 cycles on one-way scatter, more than its idle
  check (~12-15): with the upstream code inlined into E1's loop, the loop grows (620 -> 749 instructions, 50 -> 70
  stack accesses on one router; 679 -> 878 and 62 -> 98 on another), so the receive path itself spills more.
- Third version: E1's loop keeps only the check (in-flight register read, iteration counter, branch) and calls the
  upstream work out of line (`noinline`) when packets are in flight or every 16th iteration. Validation passes
  (9 + 96 tests). Polling flags alone (384) / inline (896) / out of line (896):

  | | 4 KiB | 2 KiB | trimmed 4 KiB |
  | --- | ---: | ---: | ---: |
  | 1-hop write E/W | 186.5 / 166.1 / **161.5** | 188.4 / 165.9 / **161.4** | 165.1 / 164.1 / **161.5** |
  | 1-hop write N/S | 188.2 / 180.1 / **176.4** | 184.6 / 179.9 / **176.2** | 152.6 / 177.2 / 176.4 |
  | 1-hop / 3-hop scatter (E1-bound) | 191-206 / 230-244 / 213-229 | same | 191-207 / 220-234 / 214-229 |
  | 3-hop write | 185.8 / 203.1 / 206.9 | 185.5 / 203.0 / 206.8 | 171.4 / 201.0 / 206.8 |
  | bidirectional write | 187.1 / 221.6 / 230.7 | 188.0 / 220.9 / 230.7 | 175.5 / 215.0 / 225.6 |
  | multicast write | 173.6 / 234.9 / 244.7 | 173.4 / 235.4 / 244.4 | 170.4 / 230.3 / 241.9 |

- Out of line recovers ~16 of the receiving E1's ~38 cycles; it still pays ~22 per iteration with nothing to do (the
  register read used at once, the iteration counter in local memory, and the work call every 16th iteration). Where
  every iteration has work (traffic both ways, multicast) the call costs ~8-10 more than inline.
- Remaining ideas, not tried: read the in-flight register together with E1's own receive-ring stream register (two
  stream reads issued together overlap, RESULTS §17) and trigger the idle connection check from the outer loop (once
  per 32 iterations), which should take the idle cost to a few cycles. The cost where E1 has real credit work to do
  (both directions, multicast) stays until E1 sheds receive work.

## 18. Stack use on the two ERISCs

Each ERISC has 8 KB of private local memory at 0xFFB00000 (firmware and kernel data, then the stack growing down).
On ERISC0 the base firmware keeps the first 0x700 bytes and Metal's firmware and kernels run on the stack it hands
them; ERISC1 is Metal's alone. The watcher's stack check is disabled for Ethernet cores, so: the stack pointer at
kernel entry from a probe kernel (`StackAtKernelEntry` in `test_erisc_load_overlap`), and `TT_FABRIC_EXP_STACK_PAINT=1`
in the router (fills the free stack at start, rescans every 64 outer iterations, reports the lowest word written
through the phase-profile block). Over all 40 routers of the 2D mesh, untrimmed, trimmed and with BW telemetry
(identical results):

| | E0 (ERISC0) | E1 (ERISC1) |
| --- | ---: | ---: |
| stack pointer at kernel entry | 0xFFB01EF0 | 0xFFB01FC0 |
| stack available to the router (entry sp - end of its data) | 2,400-2,528 B | 4,752-4,848 B |
| kernel_main frame (largest router) | 1,296 B | 752 B |
| deepest use (frame + callees + base-firmware routines) | 1,364 B | 752 B |
| least free | **1,036 B** | **4,000 B** |

- E0's frame is larger because the sender channels' state (slot address tables, worker interfaces, pointers and
  counters) is kept on its stack; the experiment flags (§17) do not change the largest frame.
- The base-firmware routines ERISC0 calls on the router's stack (context switch, link status, Ethernet messages)
  go at most 68 B below kernel_main's frame, init-time helper included. E1 never goes below its frame while running.
- Moving worker-facing state and work from E0 to E1 has room on E1 (~4 KB free) but not much on E0 (~1 KB).
- With the knob on the router is ~1.5% slower (224.6 vs 221.5 on 1-hop E/W, 4 KiB).

## 19. Latency of one packet (round trip, `latency_test_mode`)

`exp_latency.yaml`: the test harness's latency mode (one sender, one packet at a time to a responder that answers
it; "net" = round trip minus the responder's own time), untrimmed 2D router, 1000 samples per case, keepalive 1000,
BW telemetry off. (The harness applied `TT_FABRIC_NO_BW_TELEMETRY` to benchmark mode only, so the first version of
this section measured latency with telemetry on, which adds ~15-20 ns; `test_tt_fabric.cpp` now applies it to
latency mode too, and these are the rerun numbers.) Median net round trip in ns; 1 ns = 1.35 cycles:

| Round trip | today (0) | lean (128) | grouped (256) | both (384) | + send flags (21888, §20) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1-hop E/W write 16 B | 1423 | 1393 | 1396 | 1389 | **1364** |
| 1-hop E/W write 4 KiB | 1743 | 1718 | 1724 | 1702 | **1704** |
| 1-hop E/W fused atomic inc 16 B / 4 KiB | 1463 / 1743 | 1443 / 1723 | 1461 / 1733 | 1455 / 1709 | **1423 / 1714** |
| 1-hop N/S write 16 B / 4 KiB | 1431 / 1767 | 1417 / 1720 | 1397 / 1730 | 1394 / 1710 | **1373 / 1712** |
| 1-hop N/S fused atomic inc 16 B / 4 KiB | 1482 / 1765 | 1453 / 1728 | 1512 / 1734 | 1462 / 1713 | **1433 / 1721** |
| 3-hop E/W write 16 B / 4 KiB | 3536 / 4399 | 3459 / 4355 | 3606 / 4441 | 3521 / 4364 | **3456 / 4357** |
| 3-hop E/W fused atomic inc 16 B / 4 KiB | 3563 / 4424 | 3531 / 4356 | 3666 / 4453 | 3581 / 4373 | **3519 / 4359** |

- **Latency is mostly fixed path cost.** A 1-hop round trip is ~1920-2380 cycles. The polling and send changes
  together (21888), worth 18% of single-flow throughput, take 2-4% off it (40-80 ns) and narrow the spread (1-hop
  4 KiB write p10-p90: 1685-1796 -> 1664-1751 ns).
- **Lean completions (bit 7) help latency everywhere** (-14 to -77 ns).
- **The grouped read (bit 8) adds latency on forwarded hops**: alone it costs a 3-hop round trip +29 to +103 ns. It
  reads every channel's free-slot register once at the top of the iteration, so a packet that lands on a router
  channel after that read waits for the next iteration (the size of the effect is larger than the loop-length
  arithmetic gives, so this is not the whole explanation). With bit 7 the 3-hop round trip is back at or below
  today's.
- The keepalive knob (1000 vs the firmware's 8000) did not change latency (telemetry-on runs, within 3 ns).
- Per-sample CSVs: `results/lat2_<flags>_latency.csv`; summary `results/latency_notel_summary.txt`. The
  telemetry-on runs (`lat0_*`, `lat128/256/384_ka1000`, `snd*_lat`) are kept but not comparable.

## 20. The send itself (`send_next_data`, generic path)

In the polling-flags build the ch0 send is 52 instructions, ~72 cycles (fabric_exp/timeline.html, "Inside the
send"): about 21 for the two TX-queue commands and the header, 39 for pointer and counter bookkeeping, 12 for the
TX-queue read before the pointer update. Five changes, each behind a `TT_FABRIC_EXP_FLAGS` bit:

| Bit | Change |
| ---: | --- |
| 10 (1024) | skip the worker interface's write counter update: the router never reads it (only the mux and relay kernels use their own copy) |
| 11 (2048) | issue the TX-queue busy read before the slot bookkeeping and test it before the pointer update, as the speedy path does |
| 12 (4096) | advance the cached slot address by the slot stride, wrapping at the last slot, instead of an index wrap + table lookup |
| 13 (8192) | mark the sent slot in the free-slot stream register through the compile-time stream id (constant address) |
| 14 (16384) | when a packet is ready, read its size just before the can-send TX-queue reads, so the L1 read overlaps them |

Untrimmed 2D, 4 KiB, cycles/packet, on top of the polling flags (384), keepalive 1000, no telemetry:

| | 384 | +10 | +11 | +12 | +13 | +14 | +10+12 (5504) | +10..13 (15744) | **+10+12+14 (21888)** | all (32128) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1-hop write E/W | 186.5 | 178.3 | 186.2 | 184.0 | 186.0 | 183.7 | 172.1 | 174.9 | **168.7** | 179.3 |
| 1-hop write N/S | 188.3 | 177.5 | 188.0 | 184.7 | 187.9 | 184.6 | 171.7 | 174.4 | **169.1** | 179.1 |
| bidirectional write | 187.1 | 177.7 | 187.1 | 183.9 | 186.7 | 183.7 | 173.1 | 181.8 | **169.1** | 178.3 |
| 3-hop write | 185.5 | 177.4 | 185.4 | 182.9 | 185.2 | 182.9 | 174.1 | 175.9 | **171.3** | 178.4 |
| mesh multicast write | 173.6 | 168.9 | 171.9 | 172.2 | 172.4 | 170.9 | 167.7 | 174.5 | **166.1** | 170.4 |
| scatter (E1-bound) | 191-216 | same | same | same | same | same | same | same | same | same |

(The +14 column is after a fix: the first version read the size on every channel visit, before knowing a packet
was there, and cost +32 cycles: the unused L1 read left in flight delays the next read used at once, ~10 per idle
visit, as §17 predicts.)

- **Removing the dead write counter is the big one** (-8 to -11). The stride slot address gives -3, the early size
  read -3; together -14 to -19 (1-hop E/W 186.5 -> 168.7). 2 KiB (5504): -14 to -19; trimmed 4 KiB (5504): 1-hop
  E/W 165.1 -> 152.9, bidirectional 175.5 -> 161.6, N/S unchanged (E1-bound at 152.7).
- **Issuing the TX-queue read early (11) gains nothing**: the stack reloads of the bookkeeping that follows it are
  used at once and wait for the outstanding register reads (§17). The compile-time stream id (13) gains nothing
  either, though it removes 5 instructions; plausibly they ran in the shadow of the TX-queue register writes. Both
  make the combination worse (15744 vs 5504, +3 to +9), so they are out.
- **Latency**: 21888 is the best build on every latency case (§19): -40 to -80 ns against today.
- **The send is now 35 instructions** (`disasm/e0_21888.dis`): TX-queue payload and pointer-update commands 9,
  the read before the pointer update 4, slot stride 5, far receiver pointer 3, releasing the worker's slot 7 (stream
  id still rebuilt at run time), sent count 3, header stamp / size / progress flag / free-slot count 4. The size
  read (3) sits before the can-send check.
- **Validation**: `test_fabric_mesh_sanity_common.yaml` (96 tests) passes on 5504, on 21888 and on 32128 (first
  version of bit 14); `test_fabric_2erisc_quick.yaml` (9 tests) on 32128 with the fixed bit 14. No hang anywhere.

## 21. E1's register reads issued together (`TT_FABRIC_EXP_FLAGS` bits 15, 16)

Bit 15 (`EXP_RX_GROUPED_READS`, generic receiver, two ERISCs with separate TX queues): at the start of each receiver
step, one asm block reads the packets-received stream register, TX queue 1's command and status registers (the busy
check) and the NoC write-sent status of the oldest uncompleted slot, and the step uses those values where it read them
before. A slot written during the step is checked with a fresh read. The compiled E1 loop has the four loads back to
back at its top (`disasm/e1_54656.dis`), the first used at once. 96-test sanity passes. On top of 21888, 4 KiB,
cycles/packet:

| | 21888 | + bit 15 (54656) |
| --- | ---: | ---: |
| untrimmed scatter E/W / N/S (E1-bound) | 191.2 / 206.5 | 188.3 / 202.4 |
| trimmed 1-hop write N/S (E1-bound) | 152.7 | 145.8 |
| untrimmed / trimmed 3-hop write | 171.3 / 170.2 | 172.3 / 172.5 |
| untrimmed multicast write / scatter | 166.1 / 216.0 | 166.7 / 216.8 |
| 2 KiB: scatter / multicast | 191.1-206.3 / 165.9 | 188.1-202.3 / 167.0 |

- The microbenchmark (§17) predicted ~14 cycles less waiting per iteration; E1-bound cases gain 3-7, and forwarding
  and multicast lose 1-2.
- Receive delay injection (`TT_FABRIC_EXP_RX_DELAY=30`, trimmed 4 KiB): without bit 15 every case pays the delay's
  ~42 (E1 is the critical core); with bit 15 it pays 46 (N/S write), 54 (scatter) and 68 (3-hop).
- Likely cause (not yet confirmed): the early reads are stale. The busy check at the top of a step lands a few
  instructions after the previous step's ack command, so the queue often still reads busy, and the write-sent check
  sees the previous step's 4 KiB write before it has left. Either way the step skips its completion; when E1 is the
  bottleneck it can only catch up on an iteration with no new packet, so a skipped completion costs a whole
  iteration later. A busy read that soon after a TX command is also the case the BH-55 workaround is about.
- Next version: issue the group at the end of the step, where the completion is decided (write-sent status and busy
  check as fresh as today's), together with the next step's packets-received count (used a few instructions later
  at the top of the next step).

**Second version** (bit 15 now groups at the end of the step: the oldest uncompleted slot's write-sent status and the
busy check, read where today's code reads them, plus the next step's packets-received count, kept in the receiver
pointers; `disasm/e1_54656v2.dis`, inner loop 226 instructions against 238). Bit 16 keeps the packets-received read at
the top of the step (the group has only the completion reads). 96-test sanity passes for v2, 9-test quick for bit 16.

| 4 KiB, cycles/packet | 21888 | v1 | v2 (54656) | v2 without prefetch (120192) |
| --- | ---: | ---: | ---: | ---: |
| scatter E/W / N/S (E1-bound) | 191.2 / 206.5 | 188.3 / 202.4 | 183.4 / 199.3 | 185.2 / 201.2 |
| trimmed 1-hop write N/S | 152.7 | 145.8 | 145.9 | 147.7 |
| 3-hop write, untrimmed / trimmed | 171.3 / 170.2 | 172.3 / 172.5 | 172.4 / 172.6 | 175.0 / 175.4 |
| multicast write / scatter | 166.1 / 216.0 | 166.7 / 216.8 | 166.9 / 213.9 | 168.5 / 215.6 |
| round trip 1-hop 16 B / 4 KiB (ns) | 1364 / 1704 | 1363 / 1701 | 1394 / 1724 | 1393 / 1710 |
| round trip 3-hop 16 B / 4 KiB (ns) | 3456 / 4357 | 3478 / 4373 | 3570 / 4461 | 3509 / 4401 |

- v2 removes the delay anomaly: a 30-cycle receive delay now costs ~45 everywhere (N/S write +45, scatter +45,
  3-hop +46), so the first version's stale reads were the problem.
- E1-bound receive paths gain 7-8 cycles (scatter) and 7 (trimmed N/S write): about half the microbenchmark's
  ~14. The group's first value is used at once (8 cycles for the group, not 4), and the address work and the
  stored prefetch cost a few instructions.
- But v2 costs latency: +15 to +36 ns per 1-hop round trip, +104 to +115 ns per 3-hop (~20-25 cycles per receiving
  router), and forwarding loses 1-2 cycles of throughput. Without the prefetch, about half the latency cost goes away
  and the forwarding loss grows (+4 to +5); the rest comes from grouping the completion reads, which on paper should
  not delay anything (the reads happen where today's code reads them). Not explained.
- Net: the E1 grouped reads buy ~5-8 cycles on E1-bound receive paths and cost latency on every hop and some
  forwarding throughput. Parked behind the flags.

## 22. E1 with its local delivery pushed away (emulation, 4 KiB)

On the best build (21888), benchmark mode only (the emulations drop or redirect data writes, so data-validation
suites cannot pass: running one waited for data until stopped, and the chips were reset with `tt-smi -r all`).
Bit 0 skips local delivery of data writes (the ceiling); bit 4 pushes header + payload instead, with the stock
stateful transaction-id write, into a scratch buffer in the Ethernet tile's own L1, with no doorbell; bit 17 (with
bit 4) issues that push with one command-buffer poll and adds a doorbell, a stateless register write to an unused
stream register on the tile (no command buffer is free for a stateful one: E1 uses NoC1 buffer 3 for the atomics of
local delivery). 21888 rerun after the reset: identical (152.7 / 206.8 trimmed N/S). Cycles/packet:

| 4 KiB | today | skip local (0) | stock push (4) | lean push + doorbell (4+17) |
| --- | ---: | ---: | ---: | ---: |
| trimmed 1-hop write N/S (E1-bound) | 152.7 | 131.4 | 144.2 | 169.5 |
| trimmed 1-hop scatter N/S / E/W | 206.8 / 191.2 | 131.6 / 147.0 | 160.3 / 150.9 | 185.5 / 171.2 |
| trimmed 3-hop write | 170.2 | 177.3 | 230.2 | 230.6 |
| untrimmed 1-hop write E/W / N/S (E0-bound) | 168.7 / 169.1 | 169.8 / 168.0 | 179.8 / 179.7 | 178.6 / 178.7 |
| untrimmed multicast write | 166.1 | 157.0 | 185.7 | 199.7 |

- **The loopback distorts the result wherever L1 or the NoC matters.** Each push writes 4 KiB back into the
  Ethernet tile's L1, traffic a push to a real DRISC would not create: E0-bound writes slow from ~169 to ~179, the
  3-hop write from 170 to 230, multicast by 20-34. Only the E1-bound trimmed rows say something about E1, and even
  they carry some of that contention. **Correction (§24): this reading is wrong.** Bit 4 then took an 8928 B scratch
  that cut the forwarding sender channels to 8 slots; that caused the 3-hop, multicast and E0-bound slowdowns. With a
  4352 B scratch the loopback push leaves E0-bound rows unchanged; §24 has the rerun.
- **E1 without any local delivery is ~131** on N/S (the skip-local ceiling; E0 there is the ~123 speedy path).
  (Wrong, §25: an RX delay sweep puts E1 there at ~111; 131 is another bound.)
  Delivery today costs E1 ~21 for a write and ~75 for a scatter.
- **A push saves E1 little on plain writes**: 152.7 -> 144.2 with the stock call and no doorbell, i.e. the push
  costs ~13 against the write's ~21. With the stateless doorbell it is 169.5, worse than today: the doorbell costs
  ~25 in place. On scatter the push pays (206.8 -> 160.3).
- So for writes the handoff costs about what the work it hands off did, as the model said (§13); E1's own overhead
  (~131 on N/S with nothing to deliver: decode 32, ack 21 + busy check 12, polls and loop ~44, and the rest) is
  what stays. A faithful E1 number for a push needs it to go to a real DRISC (no loopback into Ethernet L1) with its
  own doorbell command buffer.

## 23. Skipping idle forwarding channels on E0 (`TT_FABRIC_EXP_FLAGS` bits 18, 19)

A summary doorbell or work queue fed by the producers (§12, §13) is not needed on VC0 with four channels: E0 already
reads all four free-slot registers together (bit 8, ~4-8 cycles for the group), and the ~9 cycles per idle channel
are the per-channel checks that follow. (A shared register cannot carry a queue for three producers without a slot
claim per packet; packed counts fit only 3 channels at 16 slots in its 17 bits; a flag word in L1 scales but costs
producers a write per packet. Those only pay with more than four registers to read: VC1, inter-mesh, 5-channel VC0.)
Both changes use the grouped values plus bit 7's sent count; a channel with nothing to send and nothing awaiting
completion has nothing else to do (forwarding channels skip the liveness check), so skipping it misses no event:

- **Bit 18 (262144)**: one branch over the visits to channels 1-3 when the AND of their grouped values has the
  all-free bit (16 slots: bit 4) and one in-flight count for the set is zero. 6 instructions; the count costs a
  load/add/store per forwarding send and per completion batch.
- **Bit 19 (524288)**: per channel, at the top of the visit: all-free bit and sent == processed -> return. An idle
  channel costs ~4 instructions instead of 8; a busy one takes two extra taken branches before its send.

Cycles/packet, 4 KiB, keepalive 1000, no telemetry, on 21888 (rerun: within 0.1 of §20):

| 4 KiB | 21888 | +18 (284032) | +19 (546176) | +18+19 (808320) |
| --- | ---: | ---: | ---: | ---: |
| untrimmed 1-hop write E/W | 168.6 | **148.6** | 153.0 | 151.2 |
| untrimmed 1-hop write N/S | 169.1 | 152.6 | 154.2 | 152.6 |
| untrimmed bidirectional write | 169.1 | **149.0** | 155.8 | 152.7 |
| untrimmed 3-hop write | 171.3 | 181.7 | **170.3** | 170.4 |
| untrimmed multicast write | 166.1 | 170.1 | 166.2 | 168.5 |
| trimmed 1-hop write E/W | 146.7 | 147.4 | **143.6** | 145.2 |
| trimmed 1-hop write N/S (E1-bound) | 152.7 | 152.7 | 152.7 | 152.7 |
| trimmed bidirectional write | 159.6 | **148.6** | 149.9 | 153.9 |
| trimmed 3-hop write | 170.2 | 177.8 | 170.4 | 170.4 |
| trimmed multicast write | 165.9 | 169.2 | 165.7 | 167.8 |
| scatter (E1-bound) | 191-216 | same | same | same |

- **Single flows**: the gate (18) takes untrimmed 1-hop writes from ~169 to ~149 (-20), as estimated (3 idle
  channels x ~9 minus the 6-instruction gate). N/S stops at 152.6, E1's limit there. Per channel (19) gets -13 to -16.
- **Forwarding routers pay for the gate**: on the 3-hop middle router one forwarding channel is always busy or has
  packets in flight, so the gate never passes and adds its test plus the in-flight count: +10 untrimmed, +8 trimmed,
  multicast +3-4. Bit 19 does not have that cost (3-hop -1 / +0.2, multicast ~0).
- **Both together** sit in between (the gate passes on single flows, the per-channel test catches the idle channels
  when it fails), but 1-hop loses 2-3 against the gate alone and multicast +2.
- **Decision: bit 19 is in, the gate is out. The best build is now 546176 (21888 + 19).** Keeping the gate's
  in-flight count in a register (a file-scope GNU explicit register variable) would recover only part of its
  forwarding cost (~5 memory operations per forwarded packet), and it ties the router to the compiler's register
  allocation, so it was not tried.
- **Latency** (median net round trip, ns, 21888 / +18 / +19 / +18+19): 1-hop E/W 16 B 1363 / 1349 / 1358 / 1348,
  4 KiB 1702 / 1685 / 1684 / 1684; 3-hop 16 B 3456 / 3425 / 3430 / 3444, 4 KiB 4354 / 4344 / 4327 / 4351. Every
  variant is 5-31 ns better on every case (N/S and fused atomics alike).
- **Validation**: `test_fabric_mesh_sanity_common.yaml` (96 tests) passes on 284032, 546176 and 808320. No hang.
- Disassembly: `disasm/e0_284032.dis`, `e0_546176.dis`, `e0_808320.dis`.

## 24. E1 pushing into a real DRISC's L1 (`TT_FABRIC_EXP_FLAGS` bit 20)

Bit 20 (with bit 4) makes §22's push faithful: instead of the tile's own L1, E1 pushes header + payload into an
8-slot ring (8 x 4448 B at 0x8600, the HAL's unreserved DRISC L1) on a DRISC of its chip, then rings a doorbell
on that DRISC's stream 13 with a **stateful** inline write (destination and value set once; each doorbell is a
command-buffer poll and a go). The builder gives each router a different DRISC whose NoC1 NIU is in stream mode
(there a plain address lands in DRISC L1, not GDDR): (17,12), (17,17), (17,20), (17,23), (18,12), (18,15) by
eth-channel rank, so one endpoint per bank, NoC1 only. No DRISC kernel runs: the ring is overwritten and no slot
credit is checked (~7 cycles, less if grouped, not modelled).

Two setup bugs, both fixed (by the user) before the numbers below:

- **Doorbell command buffer.** The first version rang the doorbell on NoC1 buffer 3 and hung the all-to-all multicast
  test (the other trimmed tests ran; reset with `tt-smi -r all`). In 2D two-ERISC mode forwarding sends downstream
  credits with a stateless `noc_inline_dw_write<REG>` (`fabric_router_adapter.hpp:277`), which uses buffer 3, so on a
  packet that is both forwarded and delivered locally the stateful doorbell went out with forwarding's destination and
  value: a second -1 to the downstream router's credit register. A bug in the emulation, not a DRISC or NoC hang (I
  first concluded otherwise, wrongly, having checked only the forwarding adapter's own buffers). In 2D the doorbell now
  uses buffer 1 (`read_cmd_buf`): stock 2D already leaves it in inline-write state (`setup_edm_noc_cmd_buf`), nothing
  on the receiving ERISC uses it after that, and firmware re-initializes the NoC before each launch.
- **Scratch size.** Bit 4 used to take the profiler's microbench region (8928 B), which cut VC0 to 79 slots and the
  forwarding sender channels to 8 slots each. That, not traffic written back into Ethernet L1, is what slowed §22's
  3-hop (170 -> 230), multicast and untrimmed rows. The loopback push now takes 4352 B (all four VC0 sender channels
  keep 16 slots) and the DRISC push none.

Cycles/packet, 4 KiB, all on 546176, keepalive 1000, no telemetry (546176 reproduces §23 within 0.1). No hang in 12 runs.

| 4 KiB | today | skip local (0) | loopback stock push (4) | loopback lean + stateless doorbell (4+17) | DRISC stock push + doorbell (4+20) | DRISC lean push + doorbell (4+17+20) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| trimmed 1-hop write N/S (E1-bound) | 152.7 | 131.2 | 143.3 | 174.9 | 166.2 | **159.2** |
| trimmed 1-hop scatter N/S | 206.8 | 131.8 | 159.5 | 191.0 | 182.2 | **175.4** |
| trimmed 1-hop write / scatter E/W | 143.7 / 191.4 | 142.7 / 142.2 | 146.0 / 147.7 | 159.4 / 175.6 | 151.9 / 168.1 | 147.2 / 161.0 |
| trimmed bidirectional write / scatter | 150.0 / 191.5 | 149.8 / 149.5 | 151.6 / 152.4 | 159.6 / 175.7 | 152.6 / 168.3 | 151.8 / 161.2 |
| trimmed 3-hop write / scatter | 170.4 / 191.5 | 177.3 / 177.5 | 174.5 / 174.5 | 174.5 / 175.7 | 175.3 / 175.3 | 175.3 / 175.3 |
| trimmed multicast write / scatter | 165.7 / 215.9 | 151.3 / 154.0 | 160.2 / 173.6 | 193.0 / 203.4 | 179.6 / 194.4 | 173.0 / 188.1 |
| untrimmed 1-hop write E/W / N/S | 152.9 / 154.2 | 153.9 / 153.7 | 152.9 / 152.7 | 159.4 / 174.5 | 153.7 / 165.8 | 152.7 / 158.8 |
| untrimmed 1-hop scatter N/S | 206.5 | 153.6 | 159.2 | 190.6 | 181.8 | 175.1 |
| untrimmed 3-hop write | 170.3 | 177.3 | 174.4 | 174.6 | 175.3 | 175.3 |
| untrimmed multicast write / scatter | 166.2 / 215.9 | 154.7 / 157.6 | 160.6 / 173.6 | 193.8 / 203.2 | 179.4 / 194.3 | 172.9 / 188.1 |

- **Handing a write to a DRISC costs E1 more than doing the write.** Plain writes get 6.5 slower on the E1-bound row
  (trimmed N/S, 152.7 -> 159.2), multicast writes 7.3 (165.7 -> 173.0). (Corrected by §25: E1 without local delivery
  is ~111, not the 131.2 the skip-local run shows, so the write costs E1 ~42 and the lean handoff ~48, not 21.5 / 28.)
- **Scatter gains, but well short of the ceiling**: N/S 206.8 -> 175.4 (-31), bidirectional 191.5 -> 161.2, multicast
  215.9 -> 188.1. The scatter push costs 43.6 over its ceiling, ~15 more than the write push for the same bytes (the
  loopback shows the same gap, 159.5 vs 143.3); not explained yet.
- **A real destination plus the doorbell cost 22.9 over a loopback push with no doorbell** (stock: 166.2 vs 143.3);
  the lean issue saves 7.0. These runs do not split the 22.9 between the stateful doorbell and the trip to the DRAM
  column. The stateless doorbell is far dearer: ~39 (lean loopback 174.9, if the lean issue saves the same 7).
- **3-hop is not bound by the last router's delivery**: every variant, skip local included, sits at 174.5-177.3
  against 170.4 today; scatter improves to ~175 because the last router stops being E1-bound.
- ~~E1's floor with no delivery at all, ~131, is above the 117-118.5 target~~ **Wrong (§25)**: 131 is not E1's bound;
  E1 with no local delivery takes ~111.
- With the slot cut gone, the loopback push no longer slows E0-bound rows (untrimmed writes 152.9 / 152.7 vs today):
  it is a fair lower bound for a push with no doorbell.

## 25. Blind push to the DRISC (`TT_FABRIC_EXP_FLAGS` bit 21): E1 in the P5 receive pipeline

Bit 21 (the user's change) emulates E1 in P5 (§13: the DRISC decodes, admits and forwards). At the top of the generic
receiver step E1 pushes the whole receiver slot (4448 B; the length is preset once in the command buffer) to its DRISC
ring and rings the stateful doorbell (§24's assignment, NoC1, buffer 1). No route decode, admission, forwarding or
local-delivery dispatch: one 4 B header load (send type + source channel id at offset 40) issued with the two
command-buffer ready reads and 5 independent instructions before first use (§17's overlap window), then the
transaction id, the two addresses and the two go's written directly. Standalone atomic incs (the benchmark's sync)
take the normal path; the capture pass of the trimmed tests runs stock, so trimming still sees the multi-hop profile.
Nothing is forwarded, so 3-hop and multicast measure only the first router (multicast reports a meaningless 25.6),
and each push moves 4448 B instead of 4192. Reviewed before running: the register offsets, the header layout, the
OR'd ready test (READY is 0), one transaction id per slot (the stock path checks none either).

Cycles/packet, 4 KiB, all on 546176, one session (references within 1 of §24). No hang.

| 4 KiB | today | skip local (0) | DRISC lean push of local delivery (4+17+20) | blind push (21, 2643328) |
| --- | ---: | ---: | ---: | ---: |
| trimmed 1-hop write / scatter N/S | 152.7 / 206.8 | 131.3 / 131.5 | 158.3 / 174.5 | **130.9 / 130.7** |
| trimmed 1-hop write / scatter E/W | 143.7 / 191.5 | 142.6 / 142.5 | 147.1 / 160.0 | 146.8 / 146.9 |
| trimmed bidirectional write / scatter | 150.0 / 191.5 | 149.8 / 149.5 | 152.0 / 160.3 | 151.9 / 152.0 |
| untrimmed 1-hop write / scatter N/S | 154.2 / 206.5 | 153.8 / 153.6 | 157.8 / 174.0 | 153.5 / 153.4 |
| untrimmed 1-hop write / scatter E/W | 152.7 / 191.3 | 153.8 / 153.6 | 153.0 / 159.9 | 153.2 / 153.4 |
| untrimmed bidirectional write / scatter | 155.8 / 191.5 | 156.2 / 156.1 | 153.8 / 160.3 | 153.3 / 153.4 |

E1's own time, from an RX delay sweep on trimmed N/S (`exp_2d_trim_4k_ns.yaml`; the spin costs N+12, and once E1 is
the bound each added cycle adds one):

| trimmed N/S, RX delay N | 0 | 10 | 20 | 30 | 45 | 60 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| blind push, write | 130.8 | 133.1 | 139.2 | 152.0 | 165.6 | 182.2 |
| skip local, write | 131.4 | 134.7 | 140.4 | 153.3 | 165.4 | 183.3 |
| skip local, scatter | 131.6 | 150.5 | 156.6 | 169.5 | 181.3 | 200.1 |
| today, write / scatter | 152.7 / 206.8 | 177.8 / 231.8 | 183.5 / 237.8 | | | |

- **E1 under the blind push takes ~110 cycles/packet** (152.0 - 42, 165.6 - 57, 182.2 - 72), below the 117-118.5
  wire time: in P5 E1's receive side would not limit a 1-hop flow, for writes and scatter alike.
- **The ~131 on trimmed N/S is not E1** (corrects §22 and §24): E1 skipping local delivery takes ~111 for writes, so
  131 is set elsewhere (E0, the sending worker or the link; not yet identified). The local write costs E1 ~42
  (152.7 - 111), the lean DRISC push of it ~47; the handoff still loses ~6 on writes.
- **Push + doorbell cost E1 about what decode + admission cost** (blind ~110 vs skip local ~111 on N/S).
- **Scatter's extra ~16 is outside the local write**: skipping local delivery leaves scatter at ~127 vs ~111 for
  writes (hidden under 131 at N = 0); the blind push, which reads only the send type, removes it. It is in decode,
  admission or something else keyed off the scatter header; not located. This is the "scatter push costs ~15 more"
  gap of §24.
- Elsewhere the blind push changes nothing: E/W, bidirectional and untrimmed rows sit at E0's bound (143-156).

## 26. Batched completions on E1 (`TT_FABRIC_EXP_FLAGS` bit 22)

Bit 22 (the user's change) applies to the generic receiver with fused flush/completion and no first-level ack:
- Receiver channel 0 gets all 16 hardware transaction ids. Its ring has 16 slots, so the transaction id is the slot
  index, with no bookkeeping and no reuse between consecutive batches. VC1/VC2 builds keep 4/8.
- While packets keep arriving, E1 reads nothing for completions until a 5th slot has been issued.
- Then one write-sent read of the 4th slot covers the batch of 4. E1 does one TX-queue busy read, writes 4 L1
  counter updates and sends one credit frame on TXQ1.
- On a step with no new packet, the 1-4 pending slots are flushed as soon as the first and last are sent. With
  nothing pending there are no reads at all.

Review before running found no bugs for the benchmark configurations. The inference "the last slot was sent, so the
earlier ones were" assumes each slot in a batch uses the same command buffers. In two-ERISC 2D, local writes go on
NoC1 buffer 0 and forwards on NoC1 buffer 2. A mixed batch such as local, forward, forward, local leaves the middle
slots unchecked: they are safe only because ~300 cycles pass in between. Uniform streams (all-local 1-hop, uniform
multicast) are sound. A production version would need one check per command buffer.

**4 KiB results.** Cycles/packet, one session. Sanity yaml with 4740480: 96 tests, 0 failed. No hang.

| 4 KiB | today (546176) | + bit 22 (4740480) | + bit 2 instead (546180) |
| --- | ---: | ---: | ---: |
| trimmed 1-hop write / scatter N/S | 152.7 / 206.8 | **132.6 / 184.8** | 152.7 / 206.7 |
| trimmed 1-hop write / scatter E/W | 143.7 / 191.4 | **131.1 / 172.7** | 147.0 / 191.2 |
| trimmed bidirectional write / scatter | 149.9 / 191.5 | **138.1 / 172.7** | 164.9 / 191.4 |
| trimmed 3-hop write / scatter | 170.4 / 191.5 | **151.4 / 173.0** | 170.2 / 191.4 |
| trimmed multicast write / scatter | 165.6 / 215.8 | **152.0 / 199.3** | 165.0 / 216.3 |
| untrimmed 1-hop write / scatter N/S | 154.2 / 206.5 | **134.5 / 184.6** | |
| untrimmed 1-hop write / scatter E/W | 152.9 / 191.3 | **134.1 / 172.6** | |
| untrimmed bidirectional write / scatter | 155.8 / 191.5 | **139.5 / 172.7** | |
| untrimmed 3-hop write / scatter | 170.3 / 191.4 | **151.6 / 172.6** | |
| untrimmed multicast write / scatter | 166.2 / 215.9 | **153.2 / 199.3** | |

The 546176 baseline rerun at the end matched §24 within 0.1.

**Delay sweeps.** Trimmed N/S (`exp_2d_trim_4k_ns.yaml`); a spin of N costs N+12.

| trimmed N/S | delay 0 | 5 | 10 | 20 | 30 | 45 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RX delay, 4740480 write / scatter | 132.6 / 184.8 | | | 161.3 / 218.3 | 174.4 / 231.4 | 186.2 / 243.7 |
| RX delay, blind push + bit 22 (6837632) | 129.7 | | | | 140.3 | 143.4 |
| TX delay, blind push + bit 22 (6837632) | 129.8 | 135.4 | 135.6 | 141.9 | 156.2 | |
| TX delay, 4740480 write / scatter | 132.6 / 184.8 | | 137.1 / 184.8 | 142.8 / 184.8 | | |

**What the sweeps show:**
- **Batching saves E1 ~22-24 cycles/packet on every path.**
  - Stock-path writes take ~130 (161.3 - 32, 174.4 - 42, 186.2 - 57), down from ~153.
  - Scatter takes ~186, down from ~207.
  - The blind push takes ≤ 86 (143.4 - 57), down from ~110 (§25).
  - That fits a completion step (write-sent read, TX-queue busy read, credit frame) of ~30 cycles, now paid once
    per 4 packets.
- **E0 is not the ~130 floor.** With E1 at ~86 (blind push + bit 22), a TX spin of 5-10 (cost 17-22) adds only ~6.
  Every point bounds E0 at ≤ ~110 (141.9 - 32). On the stock path with bit 22 it is ≤ ~111 (142.8 - 32).
  - Coupled stages near balance lose a few cycles, so the small-delay steps overstate E0.
- **The floor is not the link either: it is the worker channel's credit loop** (depth test below). Both E1 (blind
  push) and E0 have 20+ cycles of slack, yet trimmed N/S still sits at ~130.
  - The link alone would predict ~124: one-way neighbor exchange reached 122.9 at keepalive 1000 (§16), and the 2D
    header adds ~1.3 cycles of wire time.
  - The floor barely depends on size: with bit 22, E/W writes take 126.3 at 256 B and 1 KiB, 127.1 at 2 KiB and
    131.1 at 4 KiB (table below).
- **Trimmed N/S write is now E1 and the floor together (~130 each).** Scatter is E1-bound at ~185.
- **The rows I attributed to E0 (E/W, bidirectional, untrimmed: 143-156) drop by 12-19.**
  - Bit 2, which batches E0's upstream worker credits the same way, does not reproduce this: 0 to +3, and +15 on
    bidirectional.
  - So the gain comes from receiving a quarter as many completion frames, not from fewer worker-credit writes.
  - Mechanism not located: E0's completion block now runs once per 4 packets, and a quarter as many 96 B credit
    frames cross the link. On bidirectional, credits also share the link with data (wire model: 118.5 one way,
    120.7 with acks).
- 3-hop (-19) and multicast (-14 / -17) gain as much as 1-hop: completions cost the same on forwarding routers.

**Small packets** (`exp_2d_trim_small.yaml`, trimmed 1-hop, cycles/packet). The gain is the same at every size, and
both rows are flat across size.

| trimmed 1-hop | today write / scatter | + bit 22 write / scatter |
| --- | ---: | ---: |
| E/W 256 B | 142.3 / 191.4 | 126.3 / 172.4 |
| E/W 1 KiB | 142.3 / 191.4 | 126.3 / 172.4 |
| E/W 2 KiB | 142.5 / 191.5 | 127.1 / 172.5 |
| N/S 256 B | 152.4 / 206.6 | 130.3 / 184.6 |
| N/S 1 KiB | 152.5 / 206.7 | 130.4 / 184.7 |
| N/S 2 KiB | 152.6 / 206.7 | 130.6 / 184.7 |

**Latency** (`exp_latency.yaml`, untrimmed, median round trip in ns). Nothing gets slower. A lone packet's
completion goes out on the next step without a new packet, so batching adds no wait.

| round trip (ns), today -> + bit 22 | 16 B | 4 KiB |
| --- | ---: | ---: |
| 1-hop E/W write | 1359 -> 1346 (-13) | 1685 -> 1676 (-9) |
| 1-hop E/W fused atomic inc | 1421 -> 1407 (-14) | 1696 -> 1686 (-10) |
| 1-hop N/S write | 1359 -> 1359 (+0) | 1695 -> 1685 (-10) |
| 1-hop N/S fused atomic inc | 1433 -> 1416 (-17) | 1705 -> 1693 (-12) |
| 3-hop E/W write | 3429 -> 3417 (-12) | 4327 -> 4301 (-26) |
| 3-hop E/W fused atomic inc | 3501 -> 3465 (-36) | 4338 -> 4303 (-35) |

**On top of the blind DRISC push** (6837632 = 546176 + bits 21 + 22; benchmark only, nothing delivered or forwarded,
so 3-hop measures the first router and multicast is meaningless). Cycles/packet, same session:

| | blind push (2643328) | blind push + bit 22 (6837632) |
| --- | ---: | ---: |
| trimmed 1-hop N/S write / scatter, 4 KiB | 130.9 / 130.7 | **129.7 / 129.8** |
| trimmed 1-hop E/W write / scatter, 4 KiB | 146.8 / 146.9 | **132.2 / 132.3** |
| trimmed bidirectional write / scatter, 4 KiB | 151.9 / 152.0 | **137.7 / 137.8** |
| untrimmed 1-hop N/S, E/W, bidirectional, 4 KiB | 153.2-153.5 | **133.8-137.8** |
| trimmed N/S write 256 B / 1 KiB / 2 KiB | 130.8 / 130.4 / 130.9 | **124.0 / 125.0 / 126.4** |
| trimmed E/W write 256 B / 1 KiB / 2 KiB | 146.5 / 146.0 / 146.8 | **127.4 / 127.7 / 128.8** |

Scatter equals write throughout (small sizes within 0.2).
- The blind push's own ~131 N/S floor also moves with batching: 130.8 -> 124.0 at 256 B. Batching acts on that floor
  as well, not only on E1's time.
- With batching the floor has a size term: N/S 124.0 at 256 B -> 129.7 at 4 KiB, E/W 127.4 -> 132.2. Without
  batching it is flat at ~131.

**Channel depth sets the floor** (6837632, trimmed N/S 4 KiB, `TT_FABRIC_EXP_MAX_*_SLOTS`). The receiver cap of 12
gives 8, because the slot table has no 12. Capping the receiver hands the freed space to the worker channel.

| worker sender slots / receiver slots | 17 / 16 (default) | 12 / 16 | 8 / 16 | 49 / 8 |
| --- | ---: | ---: | ---: | ---: |
| cycles/packet (write = scatter) | 129.7 | 175.6 | 257.0 | 208.9 |
| slots x cycles/packet | 2205 | 2107 | 2056 | 1671 |

- **Throughput scales with the worker channel's depth**: slots x cycles/packet stays at ~2,050-2,200 cycles
  (~1.6 us). A worker slot is freed only when the far receiver's completion comes back (no first-level ack). So 17
  slots over a ~2,100-cycle loop give ~124-130, the floor seen since §25. It is not the worker's own speed.
  - This is §13's model prediction ("slot depth gates all of it; the sender channel is the first limit").
  - My "sending worker" guess above is wrong in that sense: the worker only sits inside the loop.
- **The receiver's own loop is shorter**: ~1,670 cycles, so its 16 slots allow ~104 per packet.
- **For 117 at 4 KiB the worker channel needs >= 18-19 slots at this loop length, or a shorter loop.**
  - Only 1 spare slot is left today: the allocator gives the worker channel 16 + spare, and the trimmed 1-hop
    router still allocates its idle forwarding channels at 16.
  - Ways out: deepen the worker channel at the forwarding channels' expense, return worker credits earlier
    (first-level ack), or shorten the loop. Batching holds a completion until a 5th slot is issued, ~4 packet times.

## 27. A deeper worker channel (`TT_FABRIC_EXP_WORKER_SLOTS`)

The user's allocator knob: `TT_FABRIC_EXP_WORKER_SLOTS=W` gives the worker sender channel (VC0 ch0) W slots and
keeps the receiver at 16. The three forwarding channels share what is left: (81 - 16 - W) / 3, at most their usual
16.
- W = 24 gives 13 per forwarding channel, W = 32 gives 11.
- Adding `TT_FABRIC_EXP_MAX_SENDER_SLOTS=8` gives 8, all powers of two, so bit 19's single-bit idle test stays on.

Reviewed before running: the capacity checks cannot underflow, and 32 is far below the 127-slot cap. The log
confirmed sender=32 receiver=16. Sanity yaml with 4740480 + W = 32 + forwarding 8: 96 tests, 0 failed. No hang.

**Results.** Cycles/packet, 4 KiB trimmed unless noted, same session as §26:

| 4 KiB | 4740480 (17 / 16 fwd) | 4740480 W32 / fwd 8 | 4740480 W32 / fwd 11 | 6837632 (17) | 6837632 W24 / fwd 13 | 6837632 W32 / fwd 11 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1-hop N/S write / scatter | 132.6 / 184.8 | 131.9 / 186.0 | 131.9 / 186.0 | 129.7 / 129.8 | **124.5 / 123.9** | **124.4 / 124.4** |
| 1-hop E/W write / scatter | 131.1 / 172.7 | **124.8** / 171.5 | 129.0 / 171.6 | 132.2 / 132.3 | **127.0 / 127.3** | 131.0 / 130.9 |
| bidirectional write / scatter | 138.1 / 172.7 | **132.5** / 171.5 | 136.4 / 171.5 | 137.7 / 137.8 | 138.8 / 139.0 | 136.9 / 137.0 |
| 3-hop write / scatter | 151.4 / 173.0 | 267.6 / 275.8 | 201.6 / 208.7 | (not forwarded) | | |
| multicast write / scatter | 152.0 / 199.3 | 198.8 / 227.0 | 163.5 / 201.3 | (not forwarded) | | |
| untrimmed 1-hop N/S / E/W write | 134.5 / 134.1 | 134.9 / 135.1 | | | | |
| untrimmed bidirectional / 3-hop write | 139.5 / 151.6 | 135.6 / 269.5 | | | | |

Small packets (256 B-2 KiB, 4740480 W32 / fwd 8) moved by -2.4 to +1.2. Latency moved by -10 to +16 ns, within run
to run spread.

**What the runs show:**
- **24 worker slots remove the depth floor.** Blind push + bit 22 on N/S goes 129.7 -> 124.5 and stays at 124.4 with
  32. On the stock path, E/W writes go 131.1 -> 124.8 and bidirectional 138.1 -> 132.5.
  - Stock-path N/S is unchanged at ~132, because E1 (~130) binds there.
  - All scatter stays at E1's ~172-186.
- **The next floor is ~124 on one-way N/S, which matches the link's one-way rate at keepalive 1000.** (§28 weakens
  this: with bit 23, stock-path E/W one-way reaches 119.8, so at least the E/W links are not at 124.)
  - One-way neighbor exchange reached 122.9 (§16), and the 96 B 2D header adds ~1.3 cycles.
  - That is ~6 above the 118.5 wire time. §16's one-way acknowledgement effect is shortened by the 1000-cycle
    keepalive but not gone.
  - The fixes from §16: a shorter keepalive (not tried below 1000), credits on the data queue, or data over two
    queues.
- **E/W with blind push is not monotonic** (127.0 at W24, 131.0 at W32), and bidirectional does not move (~137).
  Neither is explained.
- **Forwarding channels are depth-bound too, so slots cannot come out of them.**
  - With 11 and 8 forwarding slots, 3-hop writes go 151.4 -> 201.6 -> 267.6 (slots x cycles ~2,200 and ~2,140,
    the same loop length as the worker channel's). Multicast goes 152.0 -> 163.5 -> 198.8.
  - At 16 slots that loop allows ~137, so once E1 forwards faster than that, 3-hop hits it too. 117 needs ~19+
    slots on the forwarding channels as well: §13's "~20-24-slot downstream channels".
  - Deepening the worker channel this way only suits routers whose forwarding channels are idle. A general
    design needs more L1 per channel or a shorter credit loop.
- **Untrimmed 1-hop does not move** (134.5 -> 134.9), even with bit 19's bit test back on. Untrimmed 1-hop is bound
  ~10 above trimmed by something other than depth; E0's extra channel visits are the likely candidate (§23).

## 28. E0 overlapping its send tail with the completion read (`TT_FABRIC_EXP_FLAGS` bit 23)

The user's change, `EXP_TX_SEND_OVERLAP_COMPLETION = 1u << 23`. It applies on BH to counter-credit channels, with
no spin-wait send, no bit 11 (early TX-queue read) and no data cache.
- After the data command, `send_next_data` returns early (`SKIP_TAIL`).
- One asm block then issues three reads back to back: the TX queue's dummy command read (BH-55), its status, and the
  L1 completion counter. Six independent ALU ops follow (packets sent + 1, free slots - 1, the local stream
  register's address, the command value). The status and the counter are consumed at K = 6, inside the
  multi-load overlap window (§17: `S + S2 + L` costs 4 at K = 6 and 10 at K = 7).
- A busy queue goes to an out-of-line spin. Then come the packets-sent register write to the far receiver and the
  local free-slot increment, and the completion delta is used without reloading anything.

Reviewed before running; no bugs found.
- It does what the stock tail does: busy is bit 16 of the status register, the command is 4 (register write),
  and the BH-55 dummy read still comes first.
- The constraints are sound: early-clobber outputs, and `"+r"` for the updated values.
- Reading the counter on every send is what bit 7's lean check would do anyway, since a completion is always
  pending right after a send.
- In the trimmed E0 build the block compiles as written. There the inputs were already in registers, so there
  were no stack loads to hoist.
- Nit: the offsets and shifts (4, 8, 12, 15, `li 4`) are literals rather than `"i"` operands from the register
  macros.

Sanity yaml with 13129088 + W32 / forwarding 8: 96 tests, 0 failed. No hang.

**Results.** Cycles/packet, all with `TT_FABRIC_EXP_WORKER_SLOTS=32 TT_FABRIC_EXP_MAX_SENDER_SLOTS=8`:

| | 4740480 | 13129088 (+ bit 23) | 6837632 (blind) | 15226240 (blind + bit 23) |
| --- | ---: | ---: | ---: | ---: |
| trimmed 1-hop E/W write / scatter | 124.8 / 171.5 | **119.8** / 171.6 | 125.3 / 125.8 | **123.7** / 124.7 |
| trimmed 1-hop N/S write / scatter | 131.9 / 186.0 | 131.9 / 185.9 | 125.6 / 126.8 | 124.6 / 125.4 |
| trimmed bidirectional write / scatter | 132.5 / 171.5 | **130.1** / 171.6 | 132.7 / 132.7 | **129.4** / 129.3 |
| trimmed 3-hop / multicast write | 267.6 / 198.8 | 268.4 / 198.8 | (not forwarded) | |
| untrimmed 1-hop E/W / N/S write | 135.1 / 134.9 | **131.1** / 133.1 | | |
| untrimmed bidirectional / 3-hop write | 135.6 / 269.5 | **131.5** / 270.2 | | |
| small E/W write 256 / 1024 / 2048 B | 126.7 / 126.6 / 124.7 | **119.8 / 120.1 / 119.7** | | |
| small N/S write 256 / 1024 / 2048 B | 131.5 / 131.5 / 131.6 | 131.5 / 131.5 / 131.7 | | |

Latency, median round trip in ns, three runs each (ranges):

| | 4740480 | 13129088 |
| --- | ---: | ---: |
| 1-hop E/W / N/S unicast write 16 B | 1344-1349 / 1353-1356 | 1356-1364 / 1373-1374 |
| 1-hop E/W / N/S fused atomic inc 16 B | 1407-1410 / 1413-1427 | 1435-1461 / 1438-1476 |
| 3-hop unicast write / fused atomic inc 16 B | 3402-3407 / 3476-3478 | 3457-3464 / 3521-3528 |
| 4 KiB rows | 1683-4320 | +0 to +6 |

**What the runs show:**
- **Bit 23 lowers every E0-bound row and leaves E1-bound rows alone.**
  - Trimmed E/W write: 124.8 -> 119.8. Bidirectional: 132.5 -> 130.1.
  - Untrimmed 1-hop: E/W 135.1 -> 131.1, bidirectional 135.6 -> 131.5. This supports §27's guess that E0 sets the
    untrimmed rate.
  - Stock-path N/S (~132) and all scatter (~172-186) are E1's and do not move. Neither do 3-hop and multicast,
    which are bound by the 8 forwarding slots.
- **E/W one-way is now 119.8 at every size from 256 B to 4 KiB.** A per-packet cost sets it, not the wire: the
  2 KiB wire time is far below that. 119.8 is 1.3 above the 4 KiB wire time (118.5) and 2.8 above the 117
  target.
  - This undercuts §27's "~124 link floor". The E/W links carry one-way 4 KiB at 119.8.
  - Blind push sits at ~124 on both directions (123.7 E/W, 124.6 N/S), above stock E/W. So the ~124 seen with
    blind push looks like a cost of that configuration, not of the link.
- **§27's E/W blind-push anomaly is mostly the forwarding depth.** Blind push at W32 gives E/W 131.0 with 11
  forwarding slots and 125.3 with 8, and bidirectional 136.9 vs 132.7.
  - Only 8 is a power of two. With 11, bit 19's single-bit idle test is off and E0 is slower.
  - W24 (13 forwarding slots, also not a power of two) at 127.0 is still unexplained.
- **Small-packet latency is worse by ~12-20 ns per hop** (16 B rows; 3-hop +55). 4 KiB rows do not move.
  - Not yet measured. The packets-sent register write is what lets the far E1 see a packet. With 16 B packets it is
    on the latency path; with 4 KiB the data transfer hides it.
  - Two candidates delay that write under bit 23: the extra L1 counter read and ALU work before it, and the
    out-of-line spin when the queue still reads busy (a far jump and back, where the stock spin is inline).

**Follow-up: bit 22's tail flush after 16 idle steps (the user's v2).** v2 adds `exp_idle_steps` to the receiver
pointers.
- It counts consecutive receiver steps with no packet and something uncompleted.
- The partial batch is flushed only after 16 such steps. Before, it was flushed on the first one (v1 is v2 with a
  threshold of 1).
- The aim: a fast E1 (blind push) sends completions 4 at a time instead of 1 at a time between packets.
- It cannot hang: a stopped stream still flushes after 16 empty polls. Sanity yaml with 13129088: 96 tests,
  0 failed.

| W32 / fwd 8, cycles/packet | 15226240 v1 | v2 | 13129088 v1 | v2 |
| --- | ---: | ---: | ---: | ---: |
| trimmed E/W write / scatter | 123.7 / 124.7 | 128.7 / 128.7 | 119.8 / 171.6 | 127.6 / 181.1 |
| trimmed N/S write / scatter | 124.6 / 125.4 | 129.9 / 129.9 | 131.9 / 185.9 | 141.5 / 195.5 |
| trimmed bidirectional write | 129.4 | 129.9 | 130.1 | 130.3 |
| trimmed 3-hop / multicast write | (not forwarded) | | 268.4 / 198.8 | 333.7 / 229.4 |
| untrimmed N/S write / 3-hop write | | | 133.1 / 270.2 | 141.4 / 333.2 |
| small E/W / N/S write, 256 B-2 KiB | | ~128.6 / ~129.7 | ~119.9 / ~131.6 | ~126.9 / ~141.1 |

Latency, three runs each: 16 B unicast +11 to +21 ns more than v1 (3-hop +12 to +26). 16 B fused atomic −15 to
−40 ns, and much steadier (1421-1422 vs 1435-1461 on E/W). 4 KiB rows: 0 to +8.

**What it shows: holding completions costs more than it saves.**
- Every row whose rate depends on how fast completions return gets slower, by 5 (blind push) to 65 (3-hop) cycles.
  Only bidirectional, which E0 binds, holds.
- My reading: the hold adds straight to a credit loop that is already the limit (§26, §27).
  - With uncompleted > 4 as the only trigger in a stream, a completion waits for the 5th packet: on average
    ~2.5 packet gaps (~300 cycles).
  - A flow that is depth-bound stops when its window is full, so a partial batch is left at the end of every
    window round, not just at the end of the stream. That batch then waits out the 16 idle steps.
  - 3-hop with 8 forwarding slots shows this most clearly: slots x cycles goes from ~2,150 to ~2,670.
- So v1's flush on the first idle step is the right policy here. Batching helps only for what is already queued.
- Reverted to v1. A trimmed rerun with 13129088 matches v1 within 0.7 (E/W 119.8, N/S 131.9, 3-hop 268.7).

## 29. 2D in GB/s at the default slot payload (4352 B)

The 2D router's slot is 4448 B: the 96 B header plus the default payload, 4352 B (four Bfp8_b tiles,
`default_mesh_packet_payload_size_bytes`). New yamls `exp_2d_4352.yaml` and `exp_2d_trim_4352.yaml` are the 4 KiB ones
with `size: [4352]`. Both sizes were run, trimmed and untrimmed, in one session (2026-10-15).
- All runs have `TT_FABRIC_NO_BW_TELEMETRY=1`.
- All except "stock, keepalive 8000" set `TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000`.
- GB/s is the harness's payload rate per flow. Bidirectional is per direction. Trimmed rows are the replay pass.

Wire ceilings per direction (model, §16 framing: frames of at most 4114 B, 56 B overhead each, plus one 84 B pointer
update per packet). In the bidirectional case each direction also carries the other flow's completion credits.
- On this build VC0 uses counter credits: each credit is a 96 B L1 write (ack and completion counter arrays), so
  152 B on the wire.
- The model used to count every credit as an 84 B minimum frame, which is right only for stream-register credits.
  The "120.7 with acks" quoted earlier comes from that.
- It now uses `credit_bytes` (2D 96 B, 1D 32 B -> 88 B), giving 122.6 at 4 KiB with one credit per packet.
- Its validation is unchanged (static mean |error| 1.5%, event-driven 2.4%): no validated scenario is limited by
  the credits on the wire.

| Payload | Ethernet frames | one way: cycles / GB/s | bidirectional, 1 credit / packet: cycles / GB/s | bidirectional, bit 22 (1 credit / 4 packets) |
| ---: | ---: | ---: | ---: | ---: |
| 4096 B | 2 | 118.5 / 46.67 | 122.6 / 45.11 | 119.5 / 46.27 |
| 4352 B | 2 | 125.4 / 46.86 | 129.5 / 45.37 | 126.4 / 46.48 |

Both sizes need two Ethernet frames, so the extra 256 B cost little wire time. The ceiling rises 0.4%, but the time
per packet rises 6.9 cycles.

Configurations:
- stock: no flags.
- 546176: the best build before §26.
- 4740480: + bit 22.
- 13129088: + bits 22 and 23.
- W32 / fwd 8: `TT_FABRIC_EXP_WORKER_SLOTS=32 TT_FABRIC_EXP_MAX_SENDER_SLOTS=8`.
- 15226240: blind push + bits 22 and 23. Benchmark only, nothing delivered, so it is an upper bound for an E1 that
  offloads its delivery.

4352 B, trimmed, GB/s, write / scatter:

| | stock, keepalive 8000 | stock | 546176 | 4740480 | 13129088 | 13129088 W32 / fwd 8 | 15226240 W32 / fwd 8 (blind) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1-hop E/W | 31.3 / 30.7 | 34.6 / 30.8 | 40.9 / 30.7 | 44.3 / 34.0 | 44.4 / 34.0 | 46.7 / 34.2 | 46.3 / 46.4 |
| 1-hop N/S | 31.3 / 28.4 | 38.5 / 28.5 | 38.5 / 28.4 | 44.1 / 31.8 | 43.8 / 31.8 | 44.5 / 31.6 | 46.6 / 46.5 |
| bidirectional (per direction) | 29.8 / 29.9 | 30.0 / 29.9 | 39.1 / 30.7 | 42.0 / 34.0 | 41.6 / 34.0 | 44.9 / 34.2 | 45.2 / 45.2 |
| 3-hop | 29.9 / 29.7 | 31.4 / 30.7 | 34.5 / 30.7 | 38.7 / 34.0 | 38.6 / 34.0 | 21.8 / 21.1 | (not forwarded) |
| multicast | 34.9 / 27.1 | 34.9 / 27.1 | 35.4 / 27.2 | 38.6 / 29.4 | 38.7 / 29.4 | 29.2 / 25.8 | (not forwarded) |

4352 B, untrimmed, GB/s, write / scatter:

| | stock, keepalive 8000 | stock | 546176 | 4740480 | 13129088 | 13129088 W32 / fwd 8 | 15226240 W32 / fwd 8 (blind) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1-hop E/W | 26.5 / 26.6 | 26.5 / 26.6 | 38.5 / 30.7 | 43.8 / 34.0 | 43.6 / 34.0 | 44.8 / 34.3 | 44.8 / 44.9 |
| 1-hop N/S | 26.4 / 26.6 | 26.4 / 26.5 | 38.1 / 28.4 | 43.6 / 31.8 | 43.2 / 31.8 | 44.2 / 31.7 | 44.1 / 44.1 |
| bidirectional (per direction) | 26.5 / 26.7 | 26.5 / 26.7 | 37.7 / 30.7 | 41.9 / 34.0 | 42.0 / 34.0 | 44.7 / 34.2 | 44.7 / 44.7 |
| 3-hop | 26.5 / 26.5 | 26.5 / 26.5 | 34.5 / 30.7 | 38.7 / 34.0 | 38.7 / 34.0 | 21.6 / 21.0 | (not forwarded) |
| multicast | 32.3 / 27.1 | 32.3 / 27.1 | 35.4 / 27.2 | 38.3 / 29.4 | 38.7 / 29.4 | 29.2 / 25.7 | (not forwarded) |

4096 B, trimmed, GB/s, write / scatter:

| | stock, keepalive 8000 | stock | 546176 | 4740480 | 13129088 | 13129088 W32 / fwd 8 | 15226240 W32 / fwd 8 (blind) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1-hop E/W | 29.5 / 28.9 | 32.6 / 28.9 | 38.5 / 28.9 | 42.2 / 32.0 | 42.3 / 32.0 | 46.2 / 32.2 | 44.5 / 43.7 |
| 1-hop N/S | 29.5 / 26.8 | 36.2 / 26.8 | 36.2 / 26.7 | 41.7 / 29.9 | 41.4 / 29.9 | 41.9 / 29.8 | 43.6 / 44.5 |
| bidirectional (per direction) | 28.1 / 28.1 | 28.2 / 28.2 | 36.9 / 28.9 | 40.0 / 32.0 | 40.0 / 32.0 | 42.5 / 32.2 | 42.7 / 42.7 |
| 3-hop | 28.2 / 27.9 | 29.6 / 28.9 | 32.4 / 28.9 | 36.5 / 32.0 | 36.4 / 32.0 | 20.6 / 19.9 | (not forwarded) |
| multicast | 32.9 / 25.6 | 32.8 / 25.6 | 33.4 / 25.6 | 36.4 / 27.7 | 36.9 / 27.7 | 27.7 / 24.2 | (not forwarded) |

4096 B, untrimmed, GB/s, write / scatter:

| | stock, keepalive 8000 | stock | 546176 | 4740480 | 13129088 | 13129088 W32 / fwd 8 | 15226240 W32 / fwd 8 (blind) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1-hop E/W | 25.0 / 25.0 | 24.9 / 25.0 | 36.1 / 28.9 | 41.2 / 32.0 | 41.5 / 32.0 | 42.2 / 32.2 | 42.2 / 42.2 |
| 1-hop N/S | 24.9 / 25.0 | 24.9 / 25.0 | 35.9 / 26.8 | 41.1 / 30.0 | 40.8 / 30.0 | 41.5 / 29.8 | 41.5 / 41.5 |
| bidirectional (per direction) | 25.0 / 25.1 | 24.9 / 25.1 | 35.5 / 28.9 | 39.6 / 32.0 | 40.0 / 32.0 | 42.0 / 32.2 | 42.1 / 42.0 |
| 3-hop | 25.0 / 25.0 | 25.0 / 25.0 | 32.5 / 28.9 | 36.5 / 32.0 | 36.5 / 32.0 | 20.4 / 19.8 | (not forwarded) |
| multicast | 30.4 / 25.6 | 30.4 / 25.6 | 33.3 / 25.6 | 36.1 / 27.7 | 36.7 / 27.8 | 27.7 / 24.3 | (not forwarded) |

**What the numbers show:**
- **At 4352 B the best trimmed 1-hop reaches the wire.**
  - 13129088 W32 / fwd 8 runs E/W writes at 46.7 GB/s (125.8 cycles/packet, 99.7% of 46.86).
  - Blind push reaches 46.3-46.6 one way and 45.2 bidirectional, writes and scatter alike. 45.2 is 99.7% of the
    one-credit-per-packet ceiling (45.37). Between packets a fast E1 flushes at the first idle step, so its batches
    are probably mostly single credits.
  - At 4096 B the same configurations reach 46.2 and 43.6-44.5. The wire time per packet grows by 6.9 cycles while
    the per-packet core cost grows by 0-2, so the 4352 B slot leaves more slack for the core than 4 KiB did.
- **The configuration that keeps multi-hop (13129088 at default slots)** reaches 43.2-44.4 GB/s on 1-hop writes,
  41.6-42.0 bidirectional, 38.6-38.7 on 3-hop and 38.7 on multicast, trimmed and untrimmed alike.
  - Over stock that is +20% to +65% untrimmed (stock 26.4-32.3) and +11% to +39% trimmed (stock 30.0-38.5).
  - Bit 23 adds nothing at default slots: these rows are bound by the 17-slot worker channel's credit loop.
- **Untrimmed 1-hop stops at ~131-133 cycles** (44.1-44.9 GB/s at 4352 B), even with W32 or blind push. That is E0's
  channel visits (§28).
- **Scatter is the gap.** On the stock receive path with bit 22, E1 caps it at 31.6-34.3 GB/s (~172-186 cycles)
  in every configuration. Only the blind push (E1's delivery offloaded) takes scatter to the wire: 46.4-46.5 trimmed, 44.1-44.9
  untrimmed.
- **W32 / fwd 8 is a 1-hop-only layout.** It halves 3-hop (21.8) and cuts multicast to 29.2.
- **Stock with the firmware keepalive (8000)** is capped at 31.3 one way when trimmed: §16's 187.5-cycle floor of
  3 frames per 8000-cycle keepalive. Untrimmed stock is core-bound at ~222 cycles whatever the keepalive. Every
  result above 31.3 one way needs the keepalive at 1000.

## 30. Mixed traffic streams: bit 22 slows atomic_inc

Source: the test-infra "streams" work (`/home/aagarwal/tt-metal/fabric_test_streams_plan.md`, P3 status). Each run
sends an ordered mix of command types or sizes from one worker, next to single-type runs of each part. Benchmark
mode, no BW telemetry, untrimmed, 1 link. Files: `fabric_exp/streams/p3_mixes.yaml`, `streams/mix_penalty.py`,
`streams/results/p3_penalty.md`.

Single-type runs, 1-hop east-west, cycles/packet (1.35 GHz / packets per s):

| | stock | 546176 | 4740480 | 13129088 | 13129088 W32 / fwd 8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| write 4352 B | 221.9 | 153.1 | 133.5 | 132.6 | 131.1 |
| scatter 4352 B | 220.8 | 191.3 | 172.6 | 172.6 | 171.5 |
| fused_atomic_inc 4352 B | 222.6 | 179.6 | 156.7 | 156.6 | 158.7 |
| atomic_inc | 374.2 | 374.8 | **509.4** | **509.6** | **507.1** |
| write 1024 / 256 / 16 B | 223.4 / 223.4 / 221.6 | 151.2 / 152.1 / 153.3 | 144.7 / 144.8 / 144.7 | 137.2 / 136.8 / 137.0 | 138.6 / 138.3 / 138.4 |

**What the runs show:**
- **Bit 22 makes atomic_inc 36% slower:** 374.8 -> 509.4 cycles/packet. North-south and the shared link of a
  forwarded pair (two flows into chip 2) show the same. Bit 23 adds nothing on top. Earlier sections measured write
  and scatter only, so this didn't show.
- **Bit 22 makes small writes slower than 4352 B writes:** 144.7 against 133.5 cycles/packet. Bit 23 brings them
  back to about 137.
- **Alternating command types costs the fast builds 3-7%** over the average of the single-type runs (1.03-1.07,
  east-west and north-south). On stock it is 6-10% cheaper (0.90-0.94).
- **Size mixes and a 7-writes-plus-signal chunk cost nothing extra at the router** (0.96-1.03 where the test worker
  isn't the limit).

## 31. §29 rerun with the faster benchmark send loop

The test infra's benchmark send loops were rebuilt for the streams work (plan "Faster benchmark send loop").
- **Why:** the old loops kept the fabric connection object in L1 and fenced on every free-slot check.
- **Cost before:** the worker was busy about 113 cycles per packet.
- **Cost after:** about 58, with the same router-visible traffic: one credit per packet, every packet into a free slot.
- **This rerun:** same yamls and settings as §29 (`fabric_exp/newloop_runs.sh`, labels `n_*`; table script
  `newloop_tables.py`). 140 runs, 0 failed, none flagged worker-bound. The router code is unchanged since §29.

4352 B, trimmed, GB/s write / scatter (change vs §29 in %):

| | stock, ka 8000 | stock | 546176 | 4740480 | 13129088 | 13129088 W32/fwd8 | 15226240 W32/fwd8 (blind) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1-hop E/W | 31.3 / 30.7 | 34.6 / 30.7 | 40.9 / 30.7 | 44.6 (+0.5) / 34.0 | 44.5 / 34.0 | 46.7 / 34.2 | 46.3 / 46.4 |
| 1-hop N/S | 31.3 / 28.4 | 38.5 / 28.4 | 38.4 / 28.4 | 44.4 (+0.7) / 31.8 | 44.3 (+1.2) / 31.8 | 44.5 / 31.6 | 46.5 / 46.5 |
| bidirectional | 29.8 / 29.9 | 30.0 / 29.9 | 39.1 / 30.7 | 42.3 (+0.8) / 34.0 | 42.0 (+0.8) / 34.0 | 45.0 / 34.2 | 45.3 / 45.3 |
| 3-hop | 29.9 / 29.7 | 31.4 / 30.7 | 34.5 / 30.7 | 38.7 / 34.0 | 38.6 / 34.0 | 21.8 / 21.1 | (not forwarded) |
| multicast | 34.9 / 27.1 | 34.9 / 27.1 | 35.4 / 27.1 | 38.6 / 29.4 | 38.7 / 29.4 | 29.2 / 25.8 | (not forwarded) |

4352 B, untrimmed, GB/s write / scatter (change vs §29 in %):

| | stock, ka 8000 | stock | 546176 | 4740480 | 13129088 | 13129088 W32/fwd8 | 15226240 W32/fwd8 (blind) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1-hop E/W | 26.5 / 26.6 | 26.5 / 26.6 | 38.5 / 30.7 | 44.0 (+0.6) / 34.0 | 44.3 (+1.6) / 34.0 | 44.9 / 34.3 | 44.9 / 44.9 |
| 1-hop N/S | 26.4 / 26.6 | 26.4 / 26.6 | 38.1 / 28.4 | 43.8 (+0.5) / 31.8 | 43.7 (+1.3) / 31.8 | 44.1 / 31.7 | 44.1 / 44.1 |
| bidirectional | 26.5 / 26.7 | 26.5 / 26.7 | 37.7 / 30.7 | 41.9 / 34.0 | 42.4 (+1.0) / 34.0 | 44.7 / 34.2 | 44.7 / 44.7 |
| 3-hop | 26.6 / 26.5 | 26.5 / 26.5 | 34.5 / 30.7 | 38.7 / 34.0 | 38.7 / 34.0 | 21.6 / 20.9 | (not forwarded) |
| multicast | 32.3 / 27.1 | 32.3 / 27.1 | 35.4 / 27.2 | 38.3 / 29.4 | 38.7 / 29.4 | 29.2 / 25.7 | (not forwarded) |

Changes of less than 0.5% are not shown.
- **Only the bit-22 builds' 1-hop write rows move,** by +0.5 to +1.6%. For example, 13129088 untrimmed E/W goes
  43.6 -> 44.3 and N/S 43.2 -> 43.7. The old loop was slightly worker-limited there.
- **Everything else matches §29,** so §29's conclusions stand: wire-level 1-hop for the best trimmed layouts,
  scatter capped at E1 by about 172-186 cycles, untrimmed 1-hop at about 131-133 cycles, and multi-hop at 38.7 for
  default slots.

**Stream mixes on the same builds** (`streams/mix_allbuilds.sh`, `streams/p3_mixes[_trim].yaml`, labels `m_*`;
tables in `streams/results/m_mix_penalty.md` and `m_mix_trim_penalty.md`).
- **Runs:** 33 tests untrimmed and 66 trimmed per build, 0 failed, none flagged worker-bound.
- **Metric:** cycles/packet (1.35 GHz / packets per s), with the mix penalty in brackets: measured ÷ the
  packet-share-weighted average of the single-type runs.

Trimmed (replay pass), 1-hop E/W:

| | stock, ka 8000 | stock | 546176 | 4740480 | 13129088 | 13129088 W32/fwd8 | 15226240 W32/fwd8 (blind) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| commands `[write, scatter, fused, atomic]` | 231.1 (0.98) | 233.5 (1.02) | 231.3 (1.04) | 257.9 (1.06) | 258.0 (1.06) | 258.0 (1.07) | 246.9 (1.11) |
| chunk `[write x7, fused]` | 187.6 (1.00) | 170.3 (1.00) | 143.8 (0.97) | 132.9 (0.98) | 132.9 (0.98) | 125.8 (0.97) | 126.9 (1.00) |
| sizes `[4352, 16]` | 169.6 (0.94) | 167.6 (0.98) | 142.2 (0.99) | 127.1 (0.98) | 126.4 (1.00) | 119.7 (0.98) | 124.5 (1.00) |
| atomic_inc alone | 374.3 | 374.3 | 374.4 | 508.2 | 508.2 | 507.4 | 506.9 |

N/S command-mix penalties are 1.02, 1.06, 1.06, 1.08, 1.08, 1.08 and 1.01 across the same columns. The forwarded
case adds no penalty on any build that forwards (0.95-1.04); blind push doesn't forward.

**What the runs show:**
- **For a command mix that includes atomic_inc, the bit-22 builds are slower than 546176:** 258 against 231
  cycles/packet E/W. The cause is atomic_inc at about 508 cycles alone, against 374 without bit 22.
- **For the collective chunk (writes plus a fused write), 13129088 wins:** 132.9 against 143.8.
- **Blind push keeps atomic_inc at about 507,** while scatter and fused writes reach the wire (about 126-128
  trimmed). So bit 22's atomic_inc cost is not E1's NoC delivery.
- **Mixing sizes costs nothing at the router** on any build (0.94-1.01).
- **Trimmed, small writes are not slower than 4352 B writes:** 4740480 runs 16 B at 126.2 against 132.3. The
  small-write slowdown under bit 22 noted in §30 (144.7 against 133.5) shows only untrimmed.

## 32. Lean VC0 sender channel (bit 24) and `[[likely]]` idle skip (bit 25)

- **Bit 24 `EXP_TX_LEAN_VC0_LOOP` (16777216):** E0's VC0 sender channels become `LeanStrideSenderEthChannel`,
  4 B each. The slot address and stride fields are dropped, and slots advance by compile-time stride. The bit-23
  send tail uses a compile-time lreg address. It needs bit 12 and is off in super-speedy mode.
- **Bit 25 `EXP_TX_IDLE_LIKELY` (33554432):** marks the ch1..3 idle early return `[[likely]]`. Nothing else
  changes.
- **Builds:** on 13129088 (stock RX) and 15226240 W32/fwd8 (blind push, benchmark only). Combos: 29906304 (+24),
  46683520 (+25), 63460736 (+24+25), 32003456 (blind+24), 65557888 (blind+24+25).
- **Runs:** `bit2425_runs.sh` (`v_*`, `b_*`, `x_*`) and `bit2425_repeats.sh` (`r1_*`, `r2_*`). Tables come from
  `bit2425_tables.py` and `bit2425_noise.py`.
- **Validation:** the four stock-RX builds pass mesh sanity (96/0) and stream sanity (10/0). No benchmark run
  failed or was flagged worker-bound.
- **E1 is unaffected.** Every E1 router binary from the bit-24/25 builds is byte-identical (`.text`) to a
  13129088 build. Only E0 changes.

**Noise.** Three runs each of untrimmed 4352 B and E/W mixes.
- **Stock RX:** results repeat within ±0.1 cycles/packet, so every stock-RX delta below is reproducible for its
  binary.
- **Blind push:** bit-24 builds are bimodal on 1-hop E/W, at about 126 or about 128.5 cycles/packet. The same
  binary landed on different levels in different runs and yamls; blind base stays at 131.0. So blind
  differences under about 2.5 cycles are not meaningful.

Small writes (16/256/1024 B), the packet-rate-bound case, cycles/packet (`x_*` mixes, single run, consistent
with the repeats):

| | base 13129088 | +24 | +25 | +24+25 |
| --- | ---: | ---: | ---: | ---: |
| untrimmed E/W | 137.0 | 147.0 (+10.0) | 144.9 (+7.8) | 147.1 (+10.0) |
| untrimmed N/S | 130.4 | 130.4 | 133.6 (+3.2) | 130.5 |
| untrimmed forwarded | 115.1 | 118.8 (+3.7) | 115.5 | 119.1 (+4.0) |
| trimmed E/W | 122.0 | 119.9 (-2.1) | 122.0 | 119.9 (-2.1) |
| trimmed N/S | 130.2 | 130.4 | 130.2 | 130.4 |
| trimmed forwarded | 115.1 | 116.5 (+1.4) | 115.2 | 115.1 |

4352 B writes, cycles/packet (scatter is unchanged everywhere, within 0.2):

| | base 13129088 | +24 | +25 | +24+25 |
| --- | ---: | ---: | ---: | ---: |
| untrimmed 1-hop E/W | 132.5 | 133.6 (+1.1) | 133.6 (+1.1) | 133.6 (+1.1) |
| untrimmed 1-hop N/S | 134.4 | 134.3 | 135.0 (+0.6) | 134.3 |
| untrimmed bidir | 138.7 | 141.1 (+2.5) | 139.6 (+0.9) | 141.3 (+2.6) |
| trimmed 1-hop E/W | 132.0 | 131.0 (-1.0) | 131.9 | 131.0 (-1.0) |
| trimmed bidir | 140.1 | 141.0 (+0.9) | 140.2 | 140.9 (+0.8) |

3-hop and multicast are within 0.3 on every build.

Untrimmed E/W mixes, mean of 3 runs: `[4352, 16]` 137.3 → 144.7 (+24) and 143.4 (+25); 4 sizes 130.1 → 134.4
and 134.2. Command mix and atomic_inc are unchanged.

Blind push W32 untrimmed, cycles/packet:
- **1-hop E/W:** 131.0 → about 126-128.
- **N/S:** 133.3 → 131.3 (+24) and 128.2 (+24+25).
- **E/W small writes:** 130.7 → 122.2 (+24) and 125.2 (+24+25).
- **Bidir:** 131.5 → 138.2 (+24) and 137.8 (+24+25), so worse.

Trimmed blind is within ±1.5, except forwarded small writes 64.2 → 61.2.

**What the runs show:**
- **Bit 24 makes E0's send path cheaper when nothing else limits it.** Blind push E/W small writes drop about
  8.5 cycles, and trimmed stock E/W drops 2.1.
- **On the untrimmed router with stock RX, the same build is 10 cycles/packet slower on E/W small writes (+7%).**
  It is also 3.7 slower forwarded and 2.5 slower on 4352 B bidir. E1's code is identical, so the loss is in
  E0's binary.
- **Bit 25 changes only branch layout, yet costs 7.8 cycles on untrimmed E/W small writes and 3.2 on N/S.** It
  gains nothing anywhere.
- **E0's untrimmed loop is very sensitive to code layout.** Measured disassembly:
  - Bit 25 turns the branchless ch1..3 idle checks into 25 branches with 1.5-2.6 KB jumps.
  - Bit 24 cuts stack loads/stores on the 4-channel router (141 → 122 loads, 121 → 108 stores) and keeps 2012
    instructions.
  - A layout or i-cache explanation fits but is not confirmed; ERISC has an i-cache, size unmeasured here.
- **Neither bit is a keeper as built.** Bit 24 is a small win only on trimmed E/W, and the untrimmed router
  is the default.

## 33. Bit 22: grouped transaction-ID flush

- **Cause of the §30 atomic_inc slowdown:**
  - Bit 22 raises `NUM_TRANSACTION_IDS` from 4 to 16.
  - atomic_inc headers flush by default (`flush = true`), and `flush_write_to_noc_pipeline` polls every ID serially,
    about 11.2 cycles each. So 12 extra reads explain 374 → 508. The test's fused writes use `flush = false`.
- **Change:** under bit 22 the flush calls `exp_flush_noc_trids_grouped`.
  - It is `noinline`, plain C++, and ORs four `NIU_MST_REQS_OUTSTANDING_ID` reads per poll.
  - It compiled as 4 back-to-back loads, then the ORs; the call costs 2 stack saves and 2 restores.
- **New ID guard:** 16 IDs only with `!enable_deadlock_avoidance && NUM_RECEIVER_CHANNELS == 1`. Every router built
  in these runs meets that, so the ID count is unchanged.
- **Runs:** `flush_runs.sh`, labels `g_*` and `gf_*`; `flush_compare.py` compares against the §32 and §31
  baselines. The older 10-15 `g_13129088_4352[_trim]` files are in `results/g_1015_backup/`.
- **Validation:** 13129088 and 4740480 pass mesh sanity (96/0) and stream sanity (10/0). No benchmark run failed or
  was flagged worker-bound.

Cycles/packet, before → after (13129088 and 4740480 agree within 0.5 unless noted):

| | E/W | N/S | forwarded |
| --- | ---: | ---: | ---: |
| atomic_inc alone | 508.7 → 432.4 | 508.4 → 432.3 | 508.5 → 432.3 |
| command mix `[write, scatter, fused, atomic]` | 257.8 → 234.1 | 265.0 → 246.4 | 240.5 → 229.5 |
| scatter 4352 B | 172.6 → 168.6 | 184.6 → 181.6 | 172.7 → 168.6 |
| small writes (16-1024 B), 13129088 | 137.0 → 135.9 | 130.4 → 129.5 | 115.2 → 116.2 |
| write 4352 B | 132.7 → 132.7 | 134.4 → 133.9 | 125.7 → 125.7 |

Trimmed replays move the same way: atomic_inc 508 → 432, command mix -20 to -24 (-7 to -9 forwarded), scatter
-3 to -4.

- **Blind push W32:** atomic_inc 507 → 433, command mix -16 to -18; everything else within 0.5.
- **Control, 546176 (bit 22 compiled out):** unchanged, atomic_inc 374.8.

**Regression, 4352 B multi-hop writes** (exp_2d_4352; repeated, ±0.1):

| | before | after |
| --- | ---: | ---: |
| 3-hop E/W, untrimmed | 151.7 | 161.8 (+10.1; 38.7 → 36.3 GB/s) |
| multicast, untrimmed | 151.7 | 157.0 (+5.3) |
| 3-hop E/W, trimmed replay | 152.1 | 161.9 |
| multicast, trimmed replay | 151.8 | 157.0 |
| 4740480 3-hop / multicast, untrimmed | 151.9 / 153.4 | 161.6 / 158.5 |

1-hop writes and bidir are flat (-0.5 to 0).

**What the runs show:**
- **The grouped flush cuts atomic_inc by 76 cycles, not back to 374.** The flush is now about 103 cycles (about
  179 for 16 serial reads, minus 76): 4 groups of 4 loads with a ~7-cycle wait each, plus 3 address adds and loop
  overhead per group, plus the call.
- **The scatter gain and the multi-hop write loss are not on the flush path.** Neither executes the flush. Only
  E1's code changed (the guard leaves the ID count at 16), so this is E1 code generation: the call changes register
  allocation across `kernel_main`.
  - Against E1 builds known to be identical to the old 13129088 ones (from the §32 bit-24 builds), the most common
    E1 config's `kernel_main` is 32 instructions shorter, with 11 more stack loads and 11 more stores.
  - Other configs got fewer.
  - Scatter (E1-bound) gains 3-4; forwarding (E1-bound in 3-hop) loses 10.
- **Not a keeper as built:** multi-hop forwarding is the main bandwidth case, and it loses 6.7%.

## 34. Bit 22: single-register ACK-counter flush

- **Change:** under bit 22 on Blackhole without deadlock avoidance, `flush_write_to_noc_pipeline` polls
  `ncrisc_noc_nonposted_writes_flushed` (`NIU_MST_WR_ACK_RECEIVED == noc_nonposted_writes_acked[noc]`). It polls the
  local-write NOC, plus the downstream NOC if that differs. `exp_flush_noc_trids_grouped` is removed.
- **Codegen:** the flush is a 2-instruction spin (`lw` of `0xffb30204`, `bne` against the counter held in a register).
  `kernel_main` has no new calls.
- **Why it is safe on this system (reviewed before running):**
  - E1's `_start` initializes its NOC1 software counters from the hardware registers.
  - Every E1 NOC1 non-posted write updates them. On Blackhole the dataflow API forces `update_counter` (#28758
    workaround). The blind-push asm adds 2 for its 2 writes, and the lean push uses a counted helper.
  - E0 touches NOC1 only for configuration registers (init and context switch) and issues no transactions on it.
  - Stock Blackhole already relies on the sibling issued-count equality on E1: flushed L1 inline writes wait on
    `ncrisc_noc_nonposted_writes_sent`.
- **Runs:** `ackflush_runs.sh`, labels `ak_*`. `ackflush_compare.py` shows stock baseline, grouped flush (§33) and
  ACK counter side by side.
- **Validation:** 13129088 and 4740480 pass mesh sanity (96/0) and stream sanity (10/0). No benchmark run failed or
  was flagged worker-bound.

Cycles/packet, stock → grouped (§33) → ACK counter. 13129088 and 4740480 agree within 0.5 unless noted.

| | E/W | N/S | forwarded |
| --- | ---: | ---: | ---: |
| atomic_inc alone | 508.7 → 432.4 → 135.2 | 508.4 → 432.3 → 150.2 | 508.5 → 432.3 → 135.1 |
| command mix | 257.8 → 234.1 → 212.1 | 265.0 → 246.4 → 223.4 | 240.5 → 229.5 → 216.4 (4740480: 236.7 → 178.6) |
| scatter 4352 B | 172.6 → 168.6 → 173.6 | 184.6 → 181.6 → 188.6 | 172.7 → 168.6 → 173.6 |
| 3-hop E/W write 4352 B | 151.7 → 161.9 → 151.7 | | |
| multicast write 4352 B | 151.7 → 157.0 → 151.7 | | |
| small writes, fused 4352 B | within 0.1 of stock | within 0.1 | within 0.1 |

- **Trimmed replays:** move the same way. atomic_inc goes to 135 E/W and forwarded, 150 N/S. First-pass trimmed
  atomic_inc is 167 / 186. Command mix is -40 to -59.
- **4740480 untrimmed 4352 B:** 3-hop 151.9 → 151.8, multicast 153.4 → 153.3, scatter N/S +4.0.
- **Blind push W32:** 4352 B is flat. atomic_inc 507 → 202-215, command mix -30 to -49. Writes move +0.8 to +1.3,
  inside the blind-push bimodal noise.

**What the runs show:**
- **The forwarding regression from §33 is gone:** 3-hop and multicast match stock to 0.1.
- **atomic_inc lands far below stock's 374, at the small-write rate.** Saving reads accounts for at most ~45 cycles
  against stock, so the old flush must also have waited on something this check skips.
  - Likely cause: Blackhole forces atomics to be non-posted (`noc_fast_atomic_increment` sets `posted = false`), so
    each one is outstanding until its response returns.
  - The router never sets the transaction-ID tag on the atomic command buffer, so atomics probably go out as ID 0,
    which the per-ID flush polls first.
  - The ACK counter covers non-posted writes only (atomics have their own `NIU_MST_ATOMIC_RESP_RECEIVED`), so atomics
    now pipeline.
  - **Semantic change:** write→atomic ordering, the purpose of `flush_write_to_noc_pipeline`, is kept. Completion of
    earlier atomics before a later one is no longer enforced. The old flush gave that implicitly, and it matters only
    for atomics to different destinations.
  - Not yet confirmed. Check: add `ncrisc_noc_nonposted_atomics_flushed`; atomic_inc should return near 374.
- **Scatter 4352 B regresses +1 E/W, +4 N/S** (reproduced on both builds, ±0.1). Pure scatter never flushes, and the
  scatter case's own code is unchanged (same instruction count, one fewer stack load). So this is E1 layout or
  register allocation outside the case, the same sensitivity as §32 and §33.
- **Latent hazards:**
  - **Different local and downstream NOCs:** the `else` branch polls a NOC whose E1 software counters are not
    initialized at kernel start. That NOC is shared with E0's NOC0 traffic, so the check would never match and would
    hang. Not compiled on this system (both NOCs are 1). A `static_assert` (both NOCs == `noc_index`) or a fallback
    to the per-ID path would close it.
  - **Counter workaround:** the check depends on Blackhole's forced counter updates. If #28758's workaround is
    removed and a path passes `update_counter = false`, the check desyncs and hangs.
- **Follow-up (user):** the guard now also requires `local_chip_noc_equals_downstream_noc &&
  edm_to_local_chip_noc == noc_index`; otherwise it falls back to the per-ID loop. The `else` branch is gone, and a
  comment documents the writes-only contract.
  - Verified: after rebuilding through mesh sanity and exp_2d_4352 (labels `ak2_*`), 120 ELFs (60 E0, 60 E1) are
    byte-identical in `.text` and `.data` to the §34 builds.
  - Mesh sanity passes 96/0, and the 4352 B numbers repeat within 0.13.

## 35. Forwarding-channel mask gate (bit 26), shared forwarding loop (bit 27), partial-batch hold (bit 28)

User-written, on top of 13129088:
- **Bit 26 (E0):** a 3-bit completion-pending mask for ch1..3. One branch skips all forwarding channels when none has
  a packet or a pending completion; each templated ch1..3 step is gated on its own bit.
- **Bit 27 (E0):** the same outer gate, then a `do { ctz } while (mask)` loop over one shared forwarding-step body,
  with per-channel state in a 32-byte struct array.
- **Bit 28 (E1, under bit 22):** partial completion batches (1..3 slots) are held for 4 consecutive idle receiver
  steps before flushing.

**Review before running:**
- **Bit 26:** matches the stock per-channel idle skip. The mask is set on every send and cleared only when
  `completions_received_and_processed == packets_sent`.
- **Bit 27's private state is safe:**
  - The upstream credit address is cached in `wait_for_static_connection_to_ready`, before the main loop.
  - Nothing after setup reads the stock ch1..3 objects.
  - The enable condition requires persistent (liveness-skip) connections, counter credits (VC0-wide), stride slots,
    lean completions, no first-level ack, no amortized credits, and no upstream-on-E1.
  - Skipped relative to stock `send_next_data`: the write counter (bit 10 skips it in stock too), perf-telemetry
    `record_packet_send`, and `exp_fwd_in_flight` (unused on that path).
  - `__builtin_ctz` compiles to the native `ctz`.
- **Bit 28:**
  - The hold is bounded and cannot deadlock with one receiver channel: either the counter reaches 4 or the backlog
    passes 4.
  - It is a wait-for-the-batch-to-fill policy, so latency and short bursts need measuring.
  - `exp_rx_idle_steps` is a single global. With 2+ receiver channels (VC1), one busy channel resets the other's
    count every step. Not compiled here (1 receiver channel).
- **Codegen:** neither new variable is in a register. `exp_fwd_comp_pending_mask` is a local-memory global:
  - It is loaded at the outer gate and again at each per-channel gate.
  - It takes a load and a store on every send and on every completion.
  - `exp_rx_idle_steps` takes a load and a store every receiver step.
- **Baselines:** with the bits off, E1 is byte-identical to §34. E0 differs only by two `bne` operand swaps (same
  instruction count and addresses). The same-tree baselines (`fmb_*`) repeat the `ak_*` ones within 0.1.

**Runs:** `fm_runs.sh` (labels `fm_*`), `fm_base_runs.sh` (`fmb_*`), compared with `fm_compare.py`. All five builds
pass mesh sanity (96/0) and stream sanity (10/0). No benchmark failed or was flagged worker-bound.

Cycles/packet (stock-RX builds; "fwd" = forwarded through an intermediate router; untrimmed unless noted):

| | 13129088 | +26 | +27 | +28 | +26+28 | +27+28 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E/W small writes (16-1024 B) | 137.1 | 131.7 | 143.2 | 129.1 | 126.5 | 131.4 |
| N/S small writes | 130.4 | 130.4 | 131.6 | 131.7 | 131.7 | 131.7 |
| fwd small writes | 115.1 | 125.2 | 149.6 | 116.6 | 122.7 | 137.2 |
| 3-hop E/W write 4352 B | 151.6 | 154.0 | 180.7 | 153.1 | 155.2 | 176.5 |
| multicast write 4352 B | 151.7 | 153.4 | 183.6 | 152.4 | 154.2 | 182.2 |
| scatter / fused / atomic, all directions | | ±0.5 | ±0.5 to +3 | +1.5 | +1.5 | +1.5 |
| E/W small writes, trimmed first pass / replay | 156.2 / 121.9 | 151.3 / 125.3 | 168.5 / 134.7 | 153.5 / ±0.5 | | |
| fwd small writes, trimmed first pass / replay | 145.7 / 115.1 | 148.4 / 117.6 | 177.1 / ~135 | 147.2 / +1.5 | | |

- **Blind push W32, 4352 B and mixes:**
  - +26: E/W 131.1 → 129.1, N/S 133.2 → 129.0, fwd 65.5 → 64.5. Only the N/S move is outside the blind-push
    bimodal noise.
  - +27: +2.5 to +5 everywhere.
- **Median net round trip (ns, 1000 samples):**
  - 3-hop writes 16 B / 4 KiB: 3435 / 4312 → +26 3460 / 4336, +27 3572 / 4427, +28 3444 / 4307.
  - 1-hop rows move -31 to +57 ns with no consistent sign between builds, including bit 28, which only changes E1.

**What the runs show:**
- **Bit 27 is a loss:**
  - Forwarding +20 to +34 cycles (-20% or worse on 3-hop and multicast), and 3-hop latency +115 to +137 ns.
  - The shared body keeps per-channel state in a memory struct indexed by a runtime channel, instead of the
    templated copies' constants and registers.
  - Even 1-hop E/W, where the forwarding channels are idle, gets +6 (layout).
- **Bit 26 helps the source router and hurts the forwarding router:**
  - With idle forwarding channels: E/W -5.4 untrimmed, -4.8 trimmed first pass, and blind push -2 to -4.
  - With active forwarding channels: forwarded small writes +10 (+2.5 trimmed) and 3-hop +2.4.
  - The trimmed replay (forwarding channels compiled out where unused) loses 3.3 on E/W, so part of the movement
    is layout.
  - The per-packet load/store of the mask global is on the forwarding router's send and completion path.
- **Bit 28 only pays where the stream has gaps:**
  - E/W small writes -8 untrimmed and -2.6 trimmed first pass. The trimmed replay (122 cycles/packet, packets
    arrive fast enough to fill batches) does not change.
  - Every other row pays a flat +1.5 for the per-step load/store of `exp_rx_idle_steps`.
  - Latency is unchanged within the noise of these runs. Bursts longer than the slot count were not measured.
- **Not keepers as built:** 27 outright; 26 and 28 only if their state moves out of per-packet memory traffic.
  26+28 gives the best E/W small writes (126.5) but keeps 26's forwarding loss.

## 36. Bits 26/27/28 reworked: state in locals

User rework of §35, same flag values:
- **Bit 26:** no mask. The outer gate is computed from the ch1..3 free-slot bits and
  `packets_sent != completions_received_and_processed` of each channel's credits. Inside the gate, the stock
  per-channel idle checks run as usual.
- **Bit 27:** a standalone `noinline` E0 loop (`exp_run_tx_main_loop_masked`). The completion mask is a local, set
  only on the 0 → 1 in-flight transition.
- **Bit 28:** per-receiver-channel local counters, passed by reference into the inlined receiver step, touched only
  when the step makes no progress and on a flush.

**Review and codegen** (current-tree builds, paired by compile-time args):
- **Correctness:**
  - Bit 26's gate is the OR of the per-channel idle checks' negations. `exp_skip_idle_supported` requires every
    serviced ch1..3 to be a skip-idle channel, so nothing is starved.
  - Bit 27 replaces the whole `run_fabric_edm_main_loop` only when no receiver channel is serviced (E0). Its setup
    matches the stock loop, minus the DRISC-push state, which is used only by the receiver. Teardown uses only the
    receiver trackers.
  - Bit 28 resets the counter only on a flush, so it counts idle polls since the last flush, not "4 consecutive"
    as the comment says. The hold is still bounded.
  - Bit 26's compile-time stream-id change only applies with bit 13, which is not in any tested flag set.
- **All three new variables are in registers:**
  - Bit 26's counters (one `beq` pair per channel).
  - Bit 27's mask (`or`/`andn` on `s7`).
  - Bit 28's counter (`s4`: 7 instructions, no stack access).
- **Bit 26's gate is long anyway:**
  - The `||` chain is if-converted into about 16 straight-line instructions per loop iteration (3 × sub/snez/or,
    two ANDs, srli/li/andn, or, beqz) on a 4-channel E/W router.
  - The stock ch1..3 idle checks it skips are about 10–12, and §35's mask gate was 7 (one `lw` plus AND/test).
  - 22 of 60 E0 configs also reload one stream id from the stack per visit (line 2475).
- **Bit 27's shared loop:** each entry spills 6 registers and each exit reloads them. Per-channel state is still read
  from the stack array by runtime index (slot address, sent/processed counts, credit address, last/first slot).
- **Bits-off baseline:**
  - E1 is byte-identical to §35.
  - E0 changes register allocation in 14 of 60 configs (+1 instruction; the remote slot stride moves to the stack,
    the base moves to a register). The same-tree baselines (`fnb_*`) match `fmb_*` within 0.13 at 4352 B; the
    command-mix rows differ by −2.7 (noisy rows).

**Runs:** `fn_runs.sh` (labels `fn_*`, baselines `fnb_*`), tables from `fn_table.py` and `fn_compare.py`.
- All five builds and the bits-off build pass mesh sanity (96/0) and stream sanity (10/0).
- No benchmark failed or was flagged worker-bound.

Delta in cycles/packet against the same-tree 13129088 baseline. Brackets give §35's delta for the same flags:

| | base | +26 | +27 | +28 | +26+28 | +27+28 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E/W small writes | 137.2 | +14.6 [−5.5] | +4.1 [+6.1] | −7.8 [−8.1] | +0.4 [−10.6] | −6.2 [−5.7] |
| N/S small writes | 130.5 | +8.3 [+0.2] | +0.8 [+1.2] | +0.2 [+1.3] | +6.7 [+1.3] | +0.5 [+1.2] |
| fwd small writes | 115.2 | +13.0 [+10.0] | +37.8 [+34.5] | +0.2 [+1.5] | +8.9 [+7.6] | +26.1 [+22.1] |
| 1-hop E/W write 4352 | 132.5 | +4.8 [0.0] | −0.3 [+0.2] | 0.0 [+0.1] | +4.7 [+0.3] | −0.3 [+0.4] |
| 3-hop E/W write 4352 | 151.7 | +2.6 [+2.4] | +28.8 [+29.1] | +5.7 [+1.4] | +6.5 [+3.5] | +26.7 [+24.9] |
| multicast write 4352 | 151.7 | +2.5 [+1.7] | +31.4 [+31.9] | +3.1 [+0.7] | +5.4 [+2.5] | +31.8 [+30.5] |
| scatter/fused/atomic, mean | 160.1 | +0.2 [+0.1] | +0.7 [+0.3] | +0.3 [+1.5] | +0.4 [+1.6] | +0.8 [+1.6] |
| E/W small, trimmed first / replay | 156.1 / 122.1 | +3.8 / 0.0 | −0.9 / +8.5 | −2.4 / −0.6 | | |
| fwd small, trimmed first / replay | 145.7 / 115.2 | +0.1 / +3.4 | +27.1 / +36.0 | +0.3 / +0.3 | | |
| 3-hop 4352, trimmed first / replay | 179.2 / 152.1 | 0.0 / +0.6 | +26.6 / +24.5 | +6.2 / +5.2 | | |

- **Blind push W32** (against 15226240):
  - +26: +2.6 to +5.2 on every row, where §35 gave −1 to −4.2.
  - +27: −0.2 to −2.8, against §35's +2.4 to +4.8.
- **Small yaml (+28):** E/W writes −2.4, E/W scatter and N/S +0.2 to +0.3 (§35: +1.5).
- **Median net round trip, 3-hop rows (ns):**
  - +26: +68 to +82 (§35: +18 to +25).
  - +27: +126 to +144 (§35: +113 to +137).
  - +28: +5 to +16 (§35: −15 to +9).
  - 1-hop rows stay within ±20 except one +27 row (+58, 1-hop N/S atomic).

**What the runs show:**
- **Bit 26 is now a loss everywhere:** +14.6 E/W, +8.3 N/S, +13 forwarded, +68 to +82 ns on 3-hop.
  - Keeping the state in registers did not help, because the gate became longer than the checks it skips.
  - §35's version kept a 7-instruction gate. Its loss came only from the per-packet mask load/store on the
    forwarding router.
- **Bit 27 is unchanged:** −20% on forwarding and +126 to +144 ns. The runtime-indexed state and the spill/reload
  around the shared body dominate. The only gains are blind-push N/S (−2.8) and 1-hop N/S (−2.7).
- **Bit 28 keeps its gappy-stream gain and loses the flat cost:**
  - E/W small writes −7.8; scatter, fused, atomic and N/S rows are now +0.2 to +0.3 instead of +1.5.
  - 3-hop and multicast get worse: +5.7 / +3.1, and +6.2 / +5.2 trimmed.
  - The hold logic is 7 register-only instructions and the counter is not touched while packets arrive. The E1
    differences outside it are register-allocation shuffles, not on the forwarding path as far as checked, so the
    3-hop cost is unexplained (layout is the likely suspect, as in §33).
  - It is still a wait-for-the-batch-to-fill policy, and long bursts are not measured.
- **Not keepers:**
  - 27 is out.
  - A bit 26 worth retrying would keep a single forwarding in-flight count as a main-loop local (add on send,
    subtract on completion) and gate on `(~(r1 & r2 & r3) & N) | count`: about 5 instructions and no branch in
    the updates.
  - Bit 28 needs the 3-hop cost explained before it can be judged.

## 37. Bit 26 as a local in-flight count, bit 28 with reset-on-progress and a compile-time forwarding cmd buf

User changes on top of §36, same flag values; bit 27 retired:
- **Bit 26:** one local `exp_fwd_in_flight_local` in `run_fabric_edm_main_loop`, passed by reference into the sender
  steps: `++` on a forwarding-channel send, `-= completions` on completion.
  - ch1..3 run only when `((~(r1 & r2 & r3) & N) | in_flight) != 0`, with bit 19's per-channel skip inside.
  - This is §23's bit 18 gate with its counter as a local instead of a global.
- **Bit 28:**
  - The idle counter is reset on progress as well as on a flush, so it now counts consecutive idle polls.
  - When bit 28 is set, `RouterStaticSizedChannelWriterAdapter::send_payload_from_address_with_trid_impl` uses
    the compile-time `receiver_channel_forwarding_data_cmd_buf_ids[0]` instead of `this->data_noc_cmd_buf`. This
    applies only if all receiver channels share one forwarding cmd buf and the mode is not UDM.

**Build fix (mine):**
- Retiring bit 27 also deleted `exp_skip_idle_channel`, `exp_skip_idle_supported` and
  `fabric_exp_tx_skip_idle_{channels,per_channel}` (bits 18/19), so no router built, bits-off included (JIT
  "not declared"). The first run attempt failed at compile time on every config; no kernel was launched.
- I restored the block verbatim from the earlier source, between `exp_read_grouped_free_slots` and
  `exp_fwd_in_flight`. The failed JIT compile commands then build cleanly offline.

**Review and codegen:**
- **Bit 26 is correct:**
  - The counter sums every forwarding channel's in-flight packets, so it is zero exactly when no channel has a
    completion pending.
  - The gate compiles to 6 instructions (`and`, `and`, `li`, `andn`, `or`, `beqz`), with the count in `s0` and one
    `addi`/`sub` per send or completion.
  - With exactly one serviced forwarding channel, the gate skips nothing the per-channel check does not, and adds
    10–17 instructions to `kernel_main`.
  - With no forwarding channels, the gate is compiled out and E0 is byte-identical to bits-off (21 configs).
- **The compile-time cmd buf is correct:** every adapter (VC0, VC1, local relay) is constructed with
  `receiver_channel_forwarding_data_cmd_buf_ids[0]` or `[1]`, the all-equal check covers VC1, and nothing writes
  the member after construction.
  - But it is a forwarding-path change unrelated to the partial hold, so bit 28's numbers mix the two.
  - In 26 of 60 E1 configs (directions 1 and 2), bit 28 adds about 86 instructions and 13 stack ops, mostly in
    the adapter send and the cmd-buf register writes. The `0xffb30000` (NoC1 cmd-buf base) loads in `kernel_main`
    go −2 in some configs and +2 in others. Statically, the "eviction removed" claim is not visible.
- **Bits-off baseline moved again (unrelated source edits):**
  - E1 changed in 22 of 60 configs: up to −39 instructions and −8 stack loads in `kernel_main`, in the
    local-delivery NoC write and atomic code.
  - Against §36's baseline: scatter −4.3 on every row, but 3-hop write +5.2 and multicast +3.0 (blind push
    unchanged). §36's bit-28 3-hop cost (+5.7) is the same size as this move, which points to E1 layout.

**Runs:** `fp_runs.sh` (labels `fp_*`, baselines `fpb_*`), tables from `fn_table.py --fp`.
- All three builds and the bits-off build pass mesh sanity (96/0) and stream sanity (10/0).
- No benchmark failed or was flagged worker-bound.

Delta in cycles/packet against the same-tree 13129088 baseline. Brackets give §36's delta for the same flags:

| | base | +26 | +28 | +26+28 |
| --- | ---: | ---: | ---: | ---: |
| E/W small writes | 137.5 | −6.1 [+14.6] | −8.3 [−7.8] | −11.4 [+0.4] |
| N/S small writes | 130.3 | 0.0 [+8.3] | +1.4 [+0.2] | +1.5 [+6.7] |
| fwd small writes | 115.0 | +5.5 [+13.0] | +1.7 [+0.2] | +4.4 [+8.9] |
| E/W MixSize2/4 | 133.4 | −4.8 [+9.9] | −3.6 [−4.1] | −5.7 [+3.6] |
| scatter/fused/atomic, mean | 158.8 | 0.0 [+0.2] | +1.2 [+0.3] | +1.2 [+0.4] |
| 1-hop E/W / N/S write 4352 | 132.0 / 134.4 | +0.2 / −0.2 | 0.0 / +0.5 | +0.4 / +0.2 |
| 3-hop E/W write 4352 | 156.9 | +0.2 [+2.6] | −6.9 [+5.7] | −5.3 [+6.5] |
| multicast write 4352 | 154.7 | +0.7 [+2.5] | −4.1 [+3.1] | −3.4 [+5.4] |
| E/W small, trimmed first / replay | 156.1 / 122.1 | −7.2 / +7.0 | −2.5 / −0.3 | −7.5 / +3.1 |
| fwd small, trimmed first / replay | 145.8 / 115.0 | +0.1 / 0.0 | +2.0 / +1.7 | +2.0 / +1.7 |
| 3-hop 4352, trimmed first / replay | 179.2 / 156.9 | +0.1 / +0.2 | −9.6 / −7.1 | −9.3 / −6.5 |

- **Absolute 3-hop write 4352:** §36 base 151.7, this base 156.9, +28 150.0, +26+28 151.6.
- **Blind push W32, +26** (against 15226240): −1.0 to −4.4 on every row (E/W −2.0, N/S −4.4, fwd −1.0). §36's
  version gave +2.6 to +5.2.
- **Small yaml (trimmed 1-hop, 256 B–2 KiB):**
  - +28: E/W writes −1.6 first pass; scatter and N/S writes +1.6 to +2.0.
  - +26+28: E/W writes −6.5 first pass / +2.8 replay.
- **Median net round trip (ns):**
  - 3-hop: +26 +19 to +30; +28 −18 to −64; +26+28 −44 to +9.
  - 1-hop 4 KiB writes: +28 +20 / +27, +26+28 +22 / +25 (E/W / N/S).
  - Other 1-hop rows stay within ±23.

**What the runs show:**
- **Bit 26 now pays on the source router:** E/W small writes −6.1 (−7.2 trimmed first pass), blind push −1 to
  −4.4, nothing on 4352 B rows.
  - It still costs the forwarding router +5.5 on small writes and 3-hop latency +19 to +30 ns.
  - The trimmed E/W replay is +7.0. That fits the one-forwarding-channel case, where the gate is pure overhead
    (not confirmed which routers that replay runs on). A gate enabled only with 2+ serviced forwarding channels
    would remove that case.
- **Bit 28 cannot be judged as one flag:**
  - The 3-hop and multicast gains (−6.9 / −4.1, latency −18 to −64 ns) most likely come from the cmd-buf change,
    and partly recover this tree's baseline regression. Absolute 3-hop is 150.0 against 151.7 in §36's tree.
  - The hold still gives E/W small writes −8.3.
  - The flat cost is back: +1.2 to +2.0 on N/S, forwarded, scatter, fused and atomic rows, and 1-hop 4 KiB
    latency +20 to +27 ns.
  - The cmd-buf change needs its own flag to be measured.
- **26+28** gives the best E/W small writes (125.1, −11.4) and keeps 3-hop at 151.6, but adds forwarded small
  writes +4.4 and N/S +1.5.

## 38. DRISC offload (bit 21, P5): hung the chip on all-to-all multicast

User-written, run 2026-10-17. Bit 21 is now `EXP_DRISC_OFFLOAD`. The blind push (§25) is gone, so §25–§37's
blind-push W32 rows can no longer be reproduced in this tree.
- **Design:**
  - Each router with one serviced receiver channel is paired with its own free-subchannel DRISC (both NIUs in stream
    mode), ranked by Ethernet channel.
  - E1 pushes each slot into a 16-slot ring in the DRISC's L1 over NoC1. The data write is stateful on cmd buf 0; the
    doorbell goes on cmd buf 1.
  - The DRISC decodes, admits against credits from the downstream E0s and forwards on NoC0. It opens those
    connections itself. It delivers locally on NoC1 and returns ring credits to E1 on NoC1, up to 4 per write.
  - E1's decode, forwarding and local delivery are compiled out.
- **Fixed before running:**
  - Two compile errors:
    - `ROUTING_MODE`: CreateKernel adds the fabric defines to data-movement and Ethernet kernels only, so the DRISC
      got the 1D header. The builder now adds them.
    - `packed.src_ch_id`: no such field.
  - Two review findings:
    - The doorbell was DRISC stream 26, but DRAM tiles have 16 stream registers (`DRAM_NOC_NUM_STREAMS`). It is now
      13, the doorbell used by the DRISC microbenchmark and §24, and a `static_assert` checks every DRISC stream id.
    - Ring credits checked only the batch's last slot, or its first and last. Forwards and local writes go on
      different NoCs, so that does not cover an earlier slot's write on the other NoC. Every slot is now checked.
- **Runs (slow dispatch):**
  - Stream sanity passed 10/0, including the 1D Linear tests. All 60 E1 ELFs carry the offload path. The DRISC kernel's
    `.text` is 5.0–6.5 KB, under its ~11 KB limit.
  - Mesh sanity: 63 tests passed (all 48 1D Linear, plus 2D Mesh 1024 B for 1 link and for 2 links up to scatter
    unicast).
  - The 64th, `FlowControlMeshLowLatency_size_1024_num_links_2_ntype_unicast_scatter_write_ftype_mcast` (all-to-all
    multicast, 10,000 packets), hung. The host's L1 barrier read timed out (MMIO 4 B load 55 ms), within ~0.1 s
    of launch.
  - The same test with 1 link passed, as did 2-link multicast of writes, atomics and fused atomics.
  - `tt-smi -r all` recovered all 8 cards, and 13129088 stream sanity then passed 10/0. No benchmark ran.
- **The signature is a wedged NoC, not a credit deadlock:**
  - A fabric credit deadlock stops traffic but leaves the NoC free for host reads. Here a host read through the
    NoC could not complete.
  - Part 4's hang (§14) had the same signature. What the two share: several DRISCs issuing on NoC0 and NoC1 at once.
  - This design has no remote read of Ethernet L1, which §14 named as what was new in part 4. That explanation of
    part 4 is therefore in doubt.
- **Ruled out as built (a configuration that hangs is ruled out); boundary not probed.** Candidate causes, none
  tested:
  - **Doorbell overtaking data at the DRISC NIU:** payload-then-doorbell ordering is relied on for Ethernet and
    Tensix receivers but has not been validated on a DRISC NIU. If the doorbell lands first, the DRISC reads a stale
    header, and a write to a garbage address can wedge the NoC. That would fit a load-dependent hang.
  - **A hardware hazard of DRAM tiles driven or targeted on both NoCs at once:**
    - The DRISC tile receives on NoC1 (ring and doorbells from E1) and on NoC0 (credits from the downstream E0s).
    - It sends on both NoCs.

**Narrower runs (user-requested, `dq_narrow_runs.sh`, labels `dqn_*`):**

| Run | Result |
| --- | --- |
| the hung test alone (2D, multicast, scatter, 2 links, 1024 B) | 1/0 passed |
| the same combination at 1024 / 2048 / 4096 B | 3/0 passed |
| the 16 2D tests at 1024 B, in mesh sanity's order | hung on the first: unicast write, unicast, 1 link (same MMIO timeout, 14 ms after launch) |

`tt-smi -r all` recovered all 8 cards again.
- **The hang does not follow the traffic:** it hit the simplest test, which passed in the original run. 2 hangs in
  about 78 offload tests, on different patterns.
- **Ordering is ruled out (user):** the data and the doorbell go from E1 over NoC1, both on VC 2
  (`forward_and_local_write_noc_vc`).
- **Likely trigger: the host's barrier reads of DRAM tiles under slow dispatch.**
  - `SDMeshCommandQueue::finish()` calls `Cluster::l1_barrier()`. On Blackhole, UMD's `l1_membar` writes and reads
    a flag on every DRAM tile as well as on Tensix and Ethernet cores, over PCIe on the default NoC (NoC0).
  - Both hangs died in that barrier's 4 B read (`set_membar_flag`), while traffic was running.
  - At those moments the DRISC tiles were being written on NoC1 by E1, so one endpoint was targeted from both NoCs.
    The BH rule (user, via Glean) says that "can drop requests or hang", and a dropped host read is an MMIO timeout.
  - §24/§25 pushed into the same tiles on NoC1 under fast dispatch without hangs. Fast dispatch does not run this
    barrier, and those runs had no NoC0 traffic into the DRISC.
- **The design targets each DRISC tile from both NoCs on its own:** E1 on NoC1, and the downstream E0s' credits on
  NoC0. Whether the rule covers stream-mode tiles, where inbound traffic ends in DRISC L1 rather than GDDR, is not
  stated.

**Fast dispatch (user's option 1, `dqf_runs.sh`, labels `dqf_*`, same-tree baselines `dqfb_*`):**
- **No hangs.** Stream sanity 10/0, mesh sanity 96/0 twice, and every benchmark, small-packet and latency run passed.
  That is over 500 offload tests, against 2 hangs in about 78 under slow dispatch.
- **This fits the host barrier as the trigger.** It does not show the design's own NoC0 + NoC1 targeting of the
  DRISC tile is safe.
- **Benchmarks:** offload is slower almost everywhere. Cycles/packet:

| Row | 13129088 | Offload | Δ |
| --- | ---: | ---: | ---: |
| 1-hop E/W write 4352 B | 132.4 | 130.5 | −2.0 |
| 1-hop N/S write 4352 B | 134.4 | 138.9 | +4.5 |
| 1-hop E/W / N/S scatter 4352 B | 173.6 / 188.6 | 184.4 / 190.7 | +10.7 / +2.1 |
| 3-hop E/W write 4352 B | 151.7 | 167.9 | +16.2 |
| multicast write / scatter 4352 B | 151.7 / 200.9 | 163.9 / 214.9 | +12.3 / +14.1 |
| atomic_inc E/W / N/S / fwd | 135.2 / 150.2 / 135.1 | 200.7 / 211.3 / 200.8 | +61 to +66 |
| fused atomic 4352 B, E/W / N/S | 156.6 / 171.6 | 170.1 / 182.9 | +13.5 / +11.2 |
| small writes (16 B–1 KiB), E/W / N/S / fwd | ~137 / ~130.5 / ~115 | ~142 / ~142 / ~125 | +5 / +12 / +10 |
| command mix E/W / N/S / fwd | 209.5 / 220.5 / 178.6 | 221.3 / 229.7 / 216.5 | +12 / +9 / +38 (see §39: fwd is bimodal) |
| W32: 3-hop write 4352 B / fwd 4352 B (mix) | 151.7 / 125.7 | 272.7 / 150.0 | +121 / +24 |
| W32: 1-hop E/W bidir write 4352 B | 138.5 | 131.5 | −7.0 |

- **Trimmed first passes are unchanged (E0-bound).** Trimmed replays move like the untrimmed rows.
- **Small yaml:** replays E/W writes +2.6, N/S writes +7, E/W scatter +10.4, N/S scatter +1.8.
- **Latency (median net round trip):** 1-hop +372 to +490 ns, 3-hop +1323 to +1641 ns. That is about +225 ns
  (~300 cycles) for each hop that goes through E1 → DRISC (a round trip crosses 2 such hops at 1-hop and 6 at 3-hop).
- The bits-off baseline matches §36's tree: 3-hop 151.7, against 156.9 in §37's tree.

**What the runs show:**
- **The DRISC's per-packet path sets the rate on every row it touches**, not the E1 push. The DRISC's clock equals
  the ERISC's (§14).
  - Forward-only (3-hop) runs at 168, against 152 for stock E1 doing the same work. The model's P5 predicted ~118.
  - Scatter delivery runs at 184 against E1's 174, and N/S writes at 139 against 134.
  - Only E/W 1-hop writes gain, and only slightly. They sit at the worker-loop floor either way.
- **atomic_inc (~200) is very likely flush-bound.** Not confirmed.
  - On the DRISC, `flush_write_to_noc_pipeline` waits for every non-posted NoC1 write to be acked.
  - That includes the DRISC's ring-credit returns to E1, which are non-posted inline writes. So every flushed atomic
    waits a NoC round trip to E1.
  - On E1, NoC1 carried only local writes. Posted credit returns would take them out of the count.
- **W32 3-hop at 272.7 is the forwarding credit loop:** 8 slots × 272.7 ≈ 2,180 cycles per slot, the same loop
  length as stock (§26). At 16 slots the DRISC's rate binds first.
- **Latency costs ~300 cycles per offloaded hop:** the push, the doorbell, the DRISC's poll and decode, and its
  write.
- Not yet looked at: where the DRISC's ~168 cycles per forward go. Disassembly or a DRISC-side timer would show it.
  Its forward uses the stateless API, `invalidate_l1_cache()` runs per packet, and each 4-slot credit batch reads
  8 transaction-ID registers.

## 39. DRISC offload v2 (user's patches) and DRISC timing

**The user's patches** (on top of §38's P5; flags 15226240 vs 13129088, fast dispatch, `dqf_runs.sh dqg dqgb`):
- Ring credits DRISC → E1 and doorbells E1 → DRISC are posted. The DRISC's atomic flush
  (`ncrisc_noc_nonposted_writes_flushed(1)`) then waits only on local data writes.
- The DRISC retires a credit batch from two registers at most. All ring-sourced writes on one NoC use one command
  buffer, so it keeps the last NoC0 and NoC1 transaction IDs of the batch (cur + pend stages) instead of reading 8.
- Stateful NoC0 forwarding on the DRISC (single live direction), with compile-time forwarding command buffers. On
  Blackhole, `send_chunk_from_address_with_trid` drops its second command-buffer poll (ERISCs too).
- E1 sends completions in batches of 4 while the DRISC ring holds packets, and a 1–3 tail only once the ring has fully
  drained. The tail's credits to the remote sender therefore wait for the DRISC. That is safe in the 2D mesh (no
  routing cycles).
- Scatter chunk writes use `forward_and_local_write_noc_vc` (ERISCs too), and `fabric_init.cpp` zeroes the DRISC
  control region only when offload is on.
- Fixed here: `fabric_init.cpp` did not compile (`fabric_experiment_env`, `EXP_DRISC_OFFLOAD` not visible there), and
  its zeroing ran 16 B into ring slot 0.

**Validation:** stream sanity 10/0, mesh sanity 96/0 twice, every benchmark, small-packet and latency run passed. The
shared-header changes move the bits-off baseline by at most ±3 (except FWD_MixCmd, below).

**Benchmarks** (cycles/packet; `dq_compare.py dqg dqgb dqf`):

| Row | 13129088 | §38 offload | v2 offload |
| --- | ---: | ---: | ---: |
| 1-hop E/W / N/S write 4352 B | 132.5 / 134.4 | 130.5 / 138.9 | 131.1 / 135.2 |
| 1-hop E/W / N/S scatter 4352 B | 173.6 / 188.6 | 184.4 / 190.7 | 169.5 / 180.9 |
| 3-hop E/W write 4352 B | 151.7 | 167.9 | 144.3 |
| multicast write / scatter 4352 B | 151.7 / 198.5 | 163.9 / 214.9 | 143.2 / 182.0 |
| atomic_inc E/W / N/S / fwd | 135.2 / 150.5 / 135.1 | 200.7 / 211.3 / 200.8 | 129.0 / 140.0 / 128.2 |
| fused atomic 4352 B E/W / N/S / fwd | 156.6 / 171.7 / 156.5 | 170.1 / 182.9 / 170.1 | 160.0 / 174.6 / 163.5 |
| small writes (16 B–1 KiB) E/W / N/S / fwd | ~137 / ~130.5 / ~115 | ~142 / ~142 / ~125 | ~129 / ~129 / ~114.5 |
| FWD_MixSize2 / FWD_MixChunk | 115.4 / 125.8 | 126.0 / 133.0 | 122.8 / 127.9 |
| W32 3-hop write / multicast write | 151.7 / 151.7 | 272.7 / 191.2 | 277.5 / 208.4 |

- **Atomics confirm §38's guess:** with posted credit returns, each flushed atomic no longer waits on them.
- **Offload now beats the baseline** on forwarding, multicast, scatter and atomics. It is still behind on fused atomics
  (+3 to +7), N/S writes (+1 to +2.5), the chunked mixes (+2 to +5) and FWD_MixSize2 (+7).
- Trimmed first passes are unchanged (E0-bound); replays move like the untrimmed rows. Small yaml: writes −5 to +4,
  scatter −13 to −17 (§38: +2 to +10).
- W32 is still bound by the forwarding credit loop (8 slots).
- **Latency is unchanged from §38:** 1-hop +380 to +490 ns, 3-hop +1.25 to +1.59 µs (~300 cycles per offloaded hop).
- **FWD_MixCmd is bimodal on both builds:** the baseline gave 175.9–179.0 in some runs and 215.6–219.1 in others,
  with no change in between. §38's "+38" for it was mostly that, not the offload. It is left out of comparisons.

**DRISC timing tool.** The ERISC phase profiler now runs on the DRISC too (`fabric_drisc_datamover.cpp`):
- `TT_FABRIC_PHASE_PROFILE=<level>` profiles the ERISCs and the DRISC. `TT_FABRIC_DRISC_PHASE_PROFILE=<level>`
  sets the DRISC's level on its own: alone, it profiles only the DRISC (compile-time arg `PHASE_PROFILE_DRISC_MODE`).
- The DRISC stages its `FabricPhaseProfile` block in its own L1 right after the ring. Every 256 loop iterations it
  copies the block into a third block of E1's profile buffer (`PHASE_PROFILE_DRISC_BLOCK`), on NoC1 through
  `read_cmd_buf`, which nothing else on the DRISC uses there. The host never reads DRISC L1.
- Phases: `rx_visit` (one packet step), `rx_decode_admit`, `rx_forward` (NoC0 issue, only when forwarded),
  `rx_local` (NoC1 issue), `rx_completion` (credit batch returned to E1). Events: `rx_blk_admit` (downstream full),
  `rx_blk_trid` (credit batch waiting on its writes). Cost when off: none. When on, the ERISC profile buffer grows
  by one 624 B block.
- Reports: `phase_report.py` (role `D`) and `drisc_report.py` (DRISC rows, timer cost taken out).
- `TT_FABRIC_EXP_DRISC_DELAY=<d>` spins d cycles per packet on the DRISC only (criticality sweeps).

**What profiling costs:**
- Profiling every core is useless here: E0 becomes the bottleneck at ~287 cycles/packet (99% busy).
- DRISC only, level 1 adds 40–75 cycles/packet and level 2 adds 90–155. A timer read costs 7.3 cycles (§14), and one
  timed open/close pair costs 4.0 inside the phase it measures and 18.2 to whatever encloses it.
- Timing one phase at a time (`TT_FABRIC_PHASE_MASK`, `TT_FABRIC_PHASE_NO_ITER=1`) still adds ~45 to the rate, but
  the phase values match the all-phases run within ~4. The exception is E/W scatter local delivery (143 one at a
  time, 153 all at once). Even 143 cannot fit the unprofiled 172-cycle rate, so read it as ~25 high.

**Delay sweep (DRISC only, d = 0/32/64/96, profiling off; `dqs_runs.sh`, `dqs_report.py`):**
- **The DRISC sets the rate on every row.** d = 32 slows every row by 38 to 60 cycles. The spin itself costs about
  40, because it polls `mcycle`. So the DRISC has at most a few cycles of slack anywhere, 1-hop writes included.
- The steps between d = 64 and 96 come out at ~1.1 per cycle of d: the polled spin quantizes, so it cannot give the
  loop time by extrapolation. A NOP-loop spin would.

**Where the DRISC's time goes** (single-phase timing, DRISC cycles net of the timer, per packet unless noted;
`dqm_runs.sh`):

| | E/W | N/S |
| --- | ---: | ---: |
| packet step (`rx_visit`): plain write / scatter / fused / atomic | 102 / 171 / 148 / 124 | 118 / 183 / 166 / 134 |
| decode + admission | 17 | 31 |
| local delivery: write (any size, 16 B–4 KiB) | 58 | 48 |
| local delivery: scatter (4 chunks) / fused / atomic_inc | 143 (~25 high) / 101 / 72 | 117 / 94 / 69 |
| forward issue (NoC0, stateful), per forward | 51 | |
| credit batch to E1, per batch of 4 (atomic: 20) | 32–36 | 32–36 |

- For a plain 4352 B write (131.8/packet): decode/admission 17, local issue ~58, the rest of the packet step ~25
  (transaction ID, header load, bookkeeping, doorbell decrement), and ~30 outside it (doorbell poll, credit batching,
  sent checks).
- **The local write issue is size-independent CPU cost.** 16 B and 4352 B both cost 58, so the stateless
  `noc_async_write_one_packet_with_trid` (all registers, a command-buffer poll, counters) is the biggest item.
  §14's lean issue is ~17 net (24 with its timer read).
- **Scatter pays that stateless issue four times**, so it is the largest DRISC cost.
- An atomic's local step is ~70: the flush plus the non-posted atomic. A fused atomic is write + flush + atomic.
- N/S decode costs 14 more than E/W, and N/S local delivery 10 less.

## 40. DRISC offload v3 (user's CPU optimizations): forced forward inline costs E/W

**The user's changes** (on top of §39; `dqf_runs.sh dqh dqhb`, `dqm_runs.sh dqmh`):
- **Deferred credit retire.** The pending credit batch is checked only once the next batch has a packet, or when the
  DRISC is idle. §39's immediate check missed every time (`rx_blk_trid` 0.25/packet).
- **Cheaper routing decisions.**
  - `Routing2DCodec::decode_action_y_first` loads both route bytes before selecting one (ERISCs too).
  - Admission and forward dispatch return early for local-only packets on multi-direction routers.
  - `forward_payload_to_downstream_edm` is `FORCE_INLINE` on the DRISC.
- **Fast local writes on the DRISC (NoC1 command buffer 0).**
  - `NOC_CTRL` and `NOC_RET_ADDR_MID` are programmed once at init, and restored after an inline write. Stock Blackhole
    writes also leave MID alone, since it is 0 for every on-chip destination.
  - Scatter reads its sizes, count and encoding with two word loads, and tags and counts once per packet. The counts
    total 2, 3 or 4 writes, matching what it issues.
  - A fused atomic loads its semaphore fields before the payload write.
- Review found nothing to fix. The DRISC code grew from ~5.5 KB to 7.7 KB (limit ~11.2 KB), and the profiled builds
  still fit.

**Validation:** stream 10/0, mesh 96/0 twice, and every benchmark, small-packet, latency and profiled run passed.

**Unprofiled results (cycles/packet).** Rows marked `noinl` are v3 with only the DRISC `FORCE_INLINE` on the forward
reverted (`dqab_runs.sh`, `ab.py` in the session scratchpad; the user's v3 files were restored afterwards):

| Row | 13129088 | v2 (§39) | v3 | v3 noinl |
| --- | ---: | ---: | ---: | ---: |
| 1-hop N/S write / scatter 4352 B | 132.8 / 176.9 | 135.2 / 180.9 | 131.0 / 173.5 | 131.0 / 172.8 |
| 1-hop E/W scatter 4352 B | 169.4 | 169.5 | 178.4 | 169.5 |
| 3-hop write / multicast write 4352 B | 157.2 / 154.6 | 144.3 / 143.2 | 144.0 / 143.0 | 144.0 / 142.2 |
| multicast scatter 4352 B | 197.6 | 182.0 | 189.1 | 181.1 |
| atomic_inc E/W / N/S / fwd | 135.2 / 138.2 / 135.1 | 129.0 / 140.0 / 128.2 | 139.6 / 140.2 / 139.5 | 131.3 / 133.5 / 131.2 |
| fused 4352 B E/W / N/S / fwd | 157.6 / 160.9 / 157.6 | 160.0 / 174.6 / 163.5 | 163.8 / 162.6 / 163.8 | 159.3 / 161.0 / 159.4 |
| N/S MixChunk / FWD_MixChunk | 135.2 / 125.8 | 141.9 / 127.9 | 132.2 / 130.6 | 132.3 / 125.7 |
| forwarded small writes (16–256 B) / FWD_MixSize2 | 115.1 / 115.4 | 114.6 / 122.8 | 123.5 / 124.1 | 118.2 / 118.3 |
| small yaml replays: E/W scatter / N/S write 256 B | 169.4 / 120.3 | 160.2 / 125.1 | 176.8 / 124.3 | 167.3 / 118.8 |

- **N/S gains everywhere,** as intended.
- **The forced inline costs the E/W (single-direction) DRISC 4–10 cycles** on scatter, atomics, fused and
  forwarded small writes. Reverting it alone gets them back, and it gives the best overall build: at or better than
  v2 everywhere except atomics (+2 to +3), forwarded small writes (+3.6) and small-yaml E/W scatter (+7 to +9).
- **The other changes, one at a time** (in the inlined build unless noted):
  - Keep the fused preload: reverting it makes fused worse (163.8 → 169.9).
  - Reverting the deferred retire in the no-inline build costs scatter and fused 9–12 and gains atomics and
    forwarded small writes 2–3.6, so the deferral is a net win.
  - The E/W DRISC binary moves ±5–10 with small code changes in both directions. Trust single-change effects
    below that level only after checking the generated code.
- **Latency is ~2–3% worse than v2:** +27 to +54 ns at 1 hop and +105 to +137 ns at 3 hops, about 27 cycles per
  offloaded hop. No single revert (inline, deferral, fast write) recovers it; each variant lands within ~20 ns of
  v3, which is about the run-to-run noise (baseline ±15 ns).
- The bits-off baseline moved again with the shared-header change: 3-hop +5.5, scatter −4 to −12. The router's
  code-layout sensitivity is known from §37.

**DRISC phase costs** (one phase timed at a time, DRISC cycles net of the timer; `dqm_compare.py dqm dqmh`):
- **Decode/admission:** N/S 31 → 20; E/W unchanged at 17.
- **Local write issue:** E/W 58 → 50, N/S 48 → 47.
- **Local delivery for E/W scatter:** 143 → 110 (N/S 117 → 112).
- **Credit-batch misses:** `rx_blk_trid` 0.25 → 0 per packet.
- **Fused local delivery reads slower in the profiled build** (101 → 115 E/W, 94 → 119 N/S), although unprofiled the
  preload helps. Profiled builds have their own layout.

## 41. DRISC offload v4 (user's NoC0/NoC1 state reuse): best offload build so far

**The user's changes** (on top of §40 with the forward inline dropped, as recommended; `dqf_runs.sh dqi dqib`,
`dqm_runs.sh dqmi`):
- **DRISC writer adapter:**
  - It keeps a current-slot address (stride `channel_buffer_size`) and a pre-shifted NoC coordinate, instead of the
    16-entry slot table, saving 224 B of stack.
  - Multi-direction routers reuse the NoC0 command-buffer state from setup. The data write sets only its coordinate,
    and the credit write only its target address and coordinate (posted).
- **Compile-time stream IDs** for downstream free-slot reads and decrements. Multi-direction routers read them back
  to back during admission.
- **Multicast forwarding is split around local delivery.** The first NoC0 forward goes out, local delivery runs on
  NoC1 while it moves, then the remaining forwards go out (`forward_2d_dispatch_first` / `_rest`).
- **Local atomics on NoC1 `write_at_cmd_buf`:** control and target MID are set at init, and target, length and data
  are staged before the flush. The atomic fires as soon as the flush completes.
- **Review found nothing to fix:**
  - The split dispatch forwards each direction exactly once, and the local-delivery decision is taken before
    `action` bits are cleared.
  - The free-slot decrement happens once per forward.
  - The staged atomic matches stock `noc_fast_atomic_increment` (control bits, opcode, `noc_nonposted_atomics_acked`;
    no return address in dedicated mode).
- DRISC code: 7.3 KB.

**Validation:** stream 10/0, mesh 96/0 twice, and every benchmark, small-packet, latency and profiled run passed.

**Results** (cycles/packet; `dqv_compare.py`):

| Row | 13129088 | v2 | v3 no-inline | v4 |
| --- | ---: | ---: | ---: | ---: |
| multicast write / scatter 4352 B | 154.7 / 197.6 | 143.2 / 182.0 | 142.2 / 181.1 | 140.7 / 179.4 |
| trimmed replays: 3-hop / multicast / multicast scatter | 157.0 / 154.6 / 197.7 | — | 143.8 / 142.2 / 189.4 (v3) | 143.3 / 139.2 / 179.6 |
| atomic_inc E/W / N/S / fwd | 135.2 / 138.2 / 135.1 | 129.0 / 140.0 / 128.2 | 131.3 / 133.5 / 131.2 | 129.0 / 131.5 / 126.2 |
| forwarded small writes / FWD_MixSize2 / FWD_MixSize4 | 115.2 / 115.4 / 115.4 | 114.6 / 122.8 / 118.3 | 118.2 / 118.3 / 118.3 | 114.5 / 115.0 / 114.7 |
| fused 4352 B E/W / N/S / fwd | 157.6 / 160.9 / 157.6 | 160.0 / 174.6 / 163.5 | 159.3 / 161.0 / 159.4 | 159.6 / 165.8 / 159.8 |
| small yaml replays E/W scatter / N/S write 256 B | 169.3 / 120.3 | 160.2 / 125.1 | 167.3 / 118.8 | 167.5 / 120.7 |

- **Against the bits-off baseline, offload now matches or beats every row of the 4352 B, mix, trimmed and W32-1-hop
  suites, except fused atomics:** +2 E/W and fwd, +4.9 N/S. Small-yaml E/W scatter is still 7 behind v2's 160 (it
  ties the baseline).
- **The pre-staged atomic is a trade-off** (A/B with the stock flush + `noc_semaphore_inc` in its place,
  `dqab4_runs.sh`, `ab4.py`, v4 restored). It gains 2–5 on plain atomics, and costs N/S fused ~3 and fused 4 KiB
  latency ~15–20 ns. The rest of N/S fused's +4.8 against v3 no-inline is from the other changes.
- **Latency:** plain writes are unchanged from v3, still ~27 cycles per offloaded hop worse than v2. Fused 4 KiB is
  +23 ns more (above).
- W32 is unchanged: bound by the forwarding credit loop (8 slots).
- **DRISC phases** (`dqm_compare.py dqmh dqmi`): fused local delivery 115/119 → 108, multi-direction admission
  25 → 23. But the fused packet step is 13–15 dearer (and scatter 6–8) outside local delivery, so per-phase gains do
  not all reach the rate.

## 42. What limits the fused atomic on the DRISC: CPU time, not the NoC

**Question (user):** what limits the fused atomic path on the DRISC (v4: 159.6 E/W, 165.8 N/S against 131.6 for a plain
4352 B write)?

**Probes** (temporary, applied on top of v4 by the scratchpad's `probe.py`; v4 restored after): mcycle timers around the
fast local write (command-buffer ready poll; register writes + send) and the staged atomic (ready poll; register
writes + optional flush + send). They were folded into spare profiler slots; `TT_FABRIC_DRISC_PHASE_PROFILE=2`,
`TT_FABRIC_PHASE_MASK=0`, `TT_FABRIC_PHASE_NO_ITER=1`, label `dqpr_15226240_mix`. Cycles per occurrence, net of the
timer's 4.0:

| | ready poll | registers + send |
| --- | ---: | ---: |
| payload write (any row) | 9–11 | 15 (8.5 per scatter chunk) |
| atomic, fused (`flush = false`) | 9 | 17 |
| atomic, plain (header default `flush = true`; nothing outstanding) | 9 | 25 |

- **Nothing waits.** Every ready poll costs what a bare command-buffer register read costs (10.2, §14), so neither
  command buffer is ever busy when the DRISC comes back to it.
  - That also holds at full rate. A 4352 B write drains in ~83 cycles, and the atomic queued behind it on the same VC
    leaves right after, both well inside the ~160-cycle packet interval.
  - The benchmark's fused packets do not flush, so there is no ack wait either.
- **The fused cost is the second NoC command plus three more header fields, on a core with no slack (§39 delay
  sweep).**
  - Unprofiled, fused costs +28 (E/W) / +34 (N/S) over a plain write.
  - The atomic issue itself is ~26 of that, including its ready poll. The rest is the semaphore address, value and
    flush flag loads.
- **The ERISC pays almost the same:** in the bits-off baseline, fused is +25 E/W / +28 N/S over a write. So the
  offload's fused deficit (+2 E/W, +4.9 N/S) is only the DRISC's slightly dearer atomic step, not a new bottleneck.
- **What could still come off:**
  - Read both command buffers' ready status back to back: independent register reads cost 5.1 each instead of 10.3
    (§14), about 5–10 per fused packet.
  - Skip rewriting the atomic's target, coordinate and length when the semaphore address matches the last one. The
    registers persist on `write_at_cmd_buf`; this saves a few stores in streams that bump one counter.
  - The bigger lever is the ~80 cycles of per-packet overhead every row shares: doorbell poll, decode/admit, transaction
    ID, bookkeeping, credit batching.
- The probed build runs much slower (187–278 cycles/packet). But the timed sections are short register sequences that
  do not depend on the rate, and the ready polls show no waiting even there.

## 43. The DRISC's per-packet overhead outside local delivery (~80 cycles)

**Question (user):** what makes up the ~80 cycles per packet that every row pays besides its NoC delivery work (§42)?

**Method:**
- A lap timer over the whole DRISC main loop, applied on top of v4 by the scratchpad's `lap.py lap2` (v4 restored
  after). It reads the cycle counter once at each section boundary and charges the difference to that section, so
  per-packet sections add up to the loop time.
- One mark costs 14.9 cycles, calibrated on the DRISC at start-up and subtracted per mark. The profiler's own
  per-packet hooks are stripped from the loop; only its packet count and the progress/blocked flags that end-of-loop
  gap detection needs remain.
- `dqlap_runs.sh` (labels `dqlap2_*`), `dqlap_report.py --p=dqlap2`.
- Check: an E/W 4352 B write sums to 140.0 against 131.6 unprofiled (DRISC-bound, §39). The ~8 left over is the
  remaining profiler counting, and it lands in the poll and bookkeeping sections.
- Ordinary instructions can move across the marks; register accesses cannot. So sections are good to ~5 cycles. The
  0.1 for "transaction ID + header load" means that work was scheduled into neighbouring sections.

**Cycles per packet, plain 4352 B write (local delivery only):**

| Section | E/W | N/S |
| --- | ---: | ---: |
| doorbell poll + loop back (incl. ~5–8 of remaining profiler counting) | 24.7 | 24.6 |
| cache invalidate + route decode + admission | 18.1 | 21.3 |
| transaction ID + packed header load | ~0 (moved) | ~0 |
| forward dispatch check (nothing to forward) | 5.2 | 5.1 |
| bookkeeping: trid record, counters, ring address, doorbell decrement, packet count | 19.9 | 19.9 |
| credit batching: retire check, promotion, credit write to E1 (1 per 4), idle check | 19.8 | 19.7 |
| **total outside local delivery** | **87.7** (≈80 without the profiler's counting) | 90.7 |
| local delivery (write) | 52.1 | 52.1 |

- **The doorbell poll's floor is one stream-register read (10.3, §14).** The rest is loop overhead plus this build's
  counting.
- **Decode/admission** is the cache invalidate and the uncached route-byte loads (~8 each when dependent, §14). N/S
  decodes one more byte.
- **Bookkeeping** is plain ALU and local-memory work plus one stream-register update (3.2). It also absorbs part of
  the moved transaction-ID and header-load work.
- **Credit batching:**
  - A credit batch is retired once per 4 packets and its retire check usually succeeds first time (`rx_blk_trid` 0
    since v3). Yet this section still costs ~20 per packet.
  - In it: the per-iteration promotion/retire logic, the transaction-ID "sent" register read (~10 each) when a batch is
    pending, and the stateful credit write, whose command-buffer poll costs 10 by itself.
- **Forwarding rows add their forward issue:** 40 per forward for 3-hop E/W (stateful) and 26 for the first
  multicast forward. Decode/admission rises to 25–31 there, because forwarding packets read downstream credits and
  sometimes retry (1.1–1.3 packet steps per packet).
- **Where cycles could come off:**
  - Overlap the iteration's register reads: the doorbell, the pending batch's transaction-ID status, and, on
    forwarding routers, the downstream credits. Four independent reads cost 5.1 each instead of ~10 (§14), worth
    ~10–15 per packet.
  - Check the pending batch only on the iteration a new batch would promote, not every iteration.
  - Drop the credit write's command-buffer poll. Its stateful command buffer is used for nothing else, and a
    4-packet interval is far longer than the write takes to leave.

**The credit-batching section, split** (`lap.py lap3`, label `dqlap3_15226240_mix`, v4 restored; plain write,
E/W and N/S alike):

| Part | per occurrence | occurrences/packet | per packet |
| --- | ---: | ---: | ---: |
| retire check of the pending batch, incl. its transaction-ID "sent" read | 29 | 0.25 | 7.2 |
| credit write to E1 (stateful posted inline write; its command-buffer poll is ~10 of it) | 15 | 0.25 | 3.8 |
| retire check skipped (condition only) | ~1 | 1.03 | 0.9 |
| promotion + idle/termination conditions (promotion itself 1 in 4) | 10 | 1.03 | 10.3 |
| **total** | | | **~22** |

- **The "sent" read is ~13 of the retire check's 29.** Plain-atomic batches have no NoC1 transaction ID to check, and
  their retire check costs 16.
- **The biggest piece is control flow that runs on every iteration.**
  - The promotion test (`pend_count == 0 && (cur_count >= 4 || (cur_count != 0 && !progress))`) and the idle test
    after it cost ~10 per loop iteration, although promotion happens on 1 in 4 and the idle path never runs in a
    stream.
  - On forwarding DRISCs it is ~14.5 per iteration, at 1.6–1.8 iterations per packet (~23 per packet).
- **Options:**
  - Promote where `cur_count` is incremented (it reaches 4 there), and reach the promotion/idle block only on
    iterations without progress. A busy iteration then costs one branch: ~8 per packet.
  - Issue the "sent" read together with the doorbell read (~3 per packet, once per batch).
  - Drop the credit write's command-buffer poll (~2.5 per packet).
- **Precision:** two back-to-back marks measured −3.9 after the 14.9 calibration, so a mark's real in-loop cost varies
  by ~4 with what surrounds it. Each part here is good to about ±4 per occurrence.

## 44. Summary: experiments so far, and bandwidth against main

**Baseline "main":** the stock router (no experiment flags), with the Ethernet sequence-update timeout left at the
firmware's 8000 and BW telemetry off.
- Run fresh in this tree as `dqs_stock_kafw_*` (`dqsum_runs.sh`). It reproduces §31's stock runs to within 0.3%;
  stock was validated against clean main in §9.
- The other columns come from the same tree: the best build without offload (`dqib_13129088_*`) and DRISC offload v4
  (`dqi_15226240_*`). Both have the keepalive fix (1000).
- The setup is 2D mesh, 4352 B payload, untrimmed, fast dispatch. Figures are payload GB/s per flow from the
  harness; bidirectional is per direction.
- Wire ceiling at 4352 B: 46.86 GB/s one way (125.4 cycles/packet) and 45.37 bidirectional.

**Experiments:**

| Experiment | Sections | Outcome |
| --- | --- | --- |
| Measurement and per-ERISC profiling | §1–§8, §15 | Every config is ERISC-CPU bound. E0 (TX) limits single-hop, E1 (RX) multi-hop |
| Ethernet keepalive (TXQ sequence-update timeout 8000 → 1000) | §15, §16 | Removes a one-way floor above 4114 B frames (trimmed 1-hop 187.6 → 152–170). Untrimmed rows don't move (E0 binds there) |
| ERISC-only optimizations (lean TX loop, overlapped register reads, batched completions, ...; best set 13129088) | §17–§37 | 1-hop 222 → 132 cycles/packet (+67%), 3-hop +41%, multicast +18% |
| Deeper worker channel (W32, 8 forwarding slots) | §27 | Bidirectional 1-hop +5%, but 3-hop becomes credit-loop bound (277). Not a default |
| Local-delivery offload emulation, pushes into DRISC L1 | §22–§25 | DRISC NoC capacity is ample. Benchmark-only |
| DRISC offload v1 (P5: E1 pushes to the DRISC, which decodes, forwards and delivers) | §38 | Hung twice under slow dispatch (host barrier). Under fast dispatch, slower nearly everywhere |
| v2: posted credits, 2-register credit retire, stateful forward, E1 completion batching | §39 | Atomics 200 → 129; beats ERISC-best on forwarding, multicast and scatter |
| DRISC timing (profiler, delay sweep) | §39 | The DRISC sets the rate on every row |
| v3: deferred retire, decode tweaks, fast local write (+ forced inline) | §40 | N/S gains. The forced inline cost E/W 4–10 and was dropped |
| v4: adapter state reuse, multicast split around local delivery, pre-staged atomics | §41 | Best build: 3-hop 143.6, multicast 140.7 |
| Fused-atomic and per-packet overhead analysis | §42, §43 | Fused is CPU-bound (+28 over a write). ~80 cycles of per-packet overhead outside delivery, with options to cut it |

**Bandwidth against main (GB/s; cycles/packet in brackets):**

| Traffic | main | ERISC-only best | DRISC offload v4 | v4 vs main |
| --- | ---: | ---: | ---: | ---: |
| 1-hop write, E/W | 26.5 (222) | 44.4 (132) | **44.9** (131) | **+70%** |
| 1-hop write, N/S | 26.5 (222) | 44.3 (133) | **44.8** (131) | **+69%** |
| 1-hop bidirectional write, per direction | 26.5 (221) | 42.4 (139) | **42.5** (138) | **+60%** |
| 1-hop scatter, E/W | 26.6 (221) | 34.7 (169) | **34.7** (169) | +31% |
| 1-hop scatter, N/S | 26.6 (221) | 33.2 (177) | **34.0** (173) | +28% |
| 3-hop write | 26.6 (221) | 37.4 (157) | **40.9** (144) | **+54%** |
| multicast write | 32.3 (182) | 38.0 (155) | **41.8** (141) | +29% |
| multicast scatter | 27.5 (214) | 29.7 (198) | **32.8** (179) | +19% |
| forwarded write through a router | 31.3 (188) | 46.7 (126) | **46.7** (126) | **+49%** (at the wire) |
| fused write + atomic, E/W | 26.6 (221) | **37.3** (158) | 36.8 (160) | +39% |
| 256 B writes, E/W | 1.55 (223) | 2.52 (137) | **2.68** (129) | +73% |
| atomic_inc, E/W (Mpkt/s) | 3.6 (374) | 10.0 (135) | **10.5** (129) | ×2.9 |
| latency, 1-hop 4 KiB round trip | 1739 ns | **1681 ns** | 2213 ns | +27% (worse) |
| latency, 3-hop 4 KiB round trip | 4422 ns | **4326 ns** | 5990 ns | +35% (worse) |

- **The ERISC-only optimizations deliver most of the gain.** On 1-hop writes they reach 95% of the wire, and the
  DRISC adds ~1% there.
- **The DRISC offload adds where E1's work is heavy:**
  - 3-hop forwarding: +9.5% over ERISC-best.
  - Multicast: +10% (write) and +10% (scatter).
  - N/S scatter: +2%.
  - Atomics: +5%.
- **The offload costs latency:** ~300 cycles per offloaded hop (+27% at 1 hop, +35% at 3 hops). The ERISC-only build
  is 2–5% faster than main.
- **Remaining limits:**
  - 1-hop writes run at 131 cycles/packet against the wire's 125.4, with E0 and the DRISC co-limiting.
  - Scatter (~170) and fused (~160) are DRISC CPU-bound (§42, §43).
  - W32 forwarding is credit-loop bound.

## 45. Neighbor exchange on main

The stock router, as in §44: no flags, BW telemetry off, keepalive at the firmware's 8000, and with the keepalive fix
(1000). `topology: NeighborExchange` (`FabricConfig::FABRIC_1D_NEIGHBOR_EXCHANGE`), not 1D Linear/Ring; 1 link,
plain writes; every router runs the speedy path (§10). Script
`dqnbr_runs.sh`, labels `dqnbr_stock_ka{fw,1000}_{sweep,oneway}`. GB/s is payload per flow.

**Exchange (both directions, `exp_nbr_sweep.yaml`):**

| Payload | main GB/s (cycles/packet) | + keepalive fix | §10 (older tree and host) |
| ---: | ---: | ---: | ---: |
| 1 KiB | 11.19 (123.6) | 11.19 (123.6) | 11.2 |
| 2 KiB | 22.37 (123.6) | 22.36 (123.7) | 22.3 |
| 3 KiB | 33.49 (123.8) | 33.51 (123.8) | 33.5 |
| 4 KiB | 44.46 (124.4) | 44.59 (124.0) | 44.4 |
| 5440 B | 47.63 (154.2) | 47.63 (154.2) | 47.6 |
| 7616 B | 48.25 (213.1) | 48.25 (213.1) | 48.2 |
| 8704 B | 48.18 (243.9) | 48.15 (244.0) | 48.2 |
| 15232 B | 45.87 (448.3) | 45.90 (448.0) | 45.8 |

**One way (one sender, `exp_nbr_oneway.yaml`):**

| Payload | main GB/s (cycles/packet) | + keepalive fix |
| ---: | ---: | ---: |
| 2 KiB | 21.35 (129.5) | 22.52 (122.8) |
| 4000 B | 41.47 (130.2) | 43.96 (122.8) |
| 4096 B | 29.49 (187.5) | 45.01 (122.9) |

- **Reproduces §10 to within 0.3%:** the TX ERISC limits it at ~124 cycles/packet up to 4 KiB, and the wire limits
  it from ~5 KB (48.2 GB/s at 7.6–8.7 KB). 15 KB drops because only 8 slots fit.
- **One way, main is held back by the keepalive:**
  - Above 4114 B a packet takes 3 Ethernet frames, and with no reverse traffic sequence acks return only every 8000
    cycles. At 4096 B payload that gives 187.5 cycles (29.5 GB/s).
  - With the timeout at 1000 it runs at 122.9 cycles (45.0 GB/s), the exchange's speed or better.
- **Context:** main's speedy 1D path already runs 4 KiB at ~44.5 GB/s, while main's 2D router runs 4352 B at 26.5
  (§44). The best 2D builds (44.4 ERISC-only, 44.9 DRISC offload) bring 2D up to this level.

## 46. Inline asm load groups replaced with plain C++ volatile loads: E0 loses its overlap

**The user's change:** the five inline-asm blocks in `fabric_erisc_router.cpp` became plain `volatile` C++ loads:
- bit 8's grouped free-slot read;
- bit 23's TX-queue CMD + STATUS + completion-counter group;
- the read groups of bits 9, 15 and 16.

Its stated premise was that volatile loads make the compiler issue them back to back without consuming any result
in between. That is not what C++ guarantees. Volatile accesses keep their order relative to each other, but ordinary
instructions, including ones that use an earlier load's result, may be scheduled between them. GCC did exactly that.

**Code generation** (E0 of 13129088, west router; ELF saved before and after, `asm_vs_cpp/` in the session
scratchpad):
- **Bit 8, main loop.**
  - asm: the four free-slot loads issue back to back (`0x9048`–`0x9054`). Three of the four addresses are first
    reloaded from the stack, because the `"r"` operands were spilled.
  - C++: the addresses come from `lui` immediates, which is better. But channel 0's load is consumed at once (`and`,
    `addi`, `snez` …), and only then are channels 1–3 loaded. On this in-order core the first load's latency is
    exposed on every loop pass.
- **Bit 23, send tail.**
  - asm: 8 instructions separate the STATUS load from its test.
  - C++: 3. Most of the ~7–10-cycle TX-queue read latency is now exposed on every send.

**Measured** (`dqf_runs.sh dqj dqjb` against `dqi` / `dqib`, same tree otherwise). All validation and benchmark runs
passed.

| Row (cycles/packet) | ERISC-only asm → C++ | DRISC offload asm → C++ |
| --- | ---: | ---: |
| small writes 16–256 B, E/W | 137.0 → 144.5 | 128.9 → 132.8 |
| small writes, trimmed replay, E/W | 121.9 → 131.2 | 120.2 → 126.2 |
| trimmed first pass, 1-hop write | 153.7 → 157.9 | 153.8 → 158.0 |
| 1-hop write 4352 B, E/W | 132.5 → 133.5 | 130.8 → 133.1 |
| scatter, 3-hop, multicast, fused | ±0.5 | −1.4 to +0.1 |
| W32 1-hop write / 3-hop | — | 131.0 → 134.4 / 279.8 → 272.2 |

- **Only rows limited by E0's loop move, and they all get slower,** which matches the code generation. Rows limited by
  E1 or the DRISC do not move.
- **Latency changes are within noise:** 1-hop 16 B +8 to +29 ns, 3-hop −10 to −21 ns.
- The W32 3-hop gain (−7.6) is on a credit-loop-bound row and sits inside the layout noise of §37.
- **Bits 9, 15 and 16** are in neither build, so their conversion changed no measured row. The same scheduling caveat
  applies if they are re-enabled.
- **Recommendation:** keep the explicit asm for bits 8 and 23. It fixes the issue order, and it does not pin
  registers or depend on allocation: the compiler still picks the registers. The asm could take its constant
  addresses as `lui`/`lw` immediates (split hi/lo operands) to drop the stack reloads the C++ version avoids.

## 47. Router kernel v2 (user's pure-C++ E0 loop): compiles after three fixes, then slower than v1

**The user's change:**
- `fabric_erisc_router.cpp` (v1) is back to the asm version. Its rebuilt E0 for 13129088 is
  instruction-for-instruction identical to the binary saved before §46. Reruns of the 4352 and mix suites
  (`dqv1*`) match `dqi`/`dqib` within 0.26 cycles/packet.
- `compute_mesh_router_builder.cpp` picks `fabric_erisc_router_v2.cpp` when `TT_FABRIC_KERNEL_VERSION=2`.
- v2 is pure C++ with a new E0 loop, `run_e0_tx_main_loop`, marked
  `noinline, optimize("no-schedule-insns,no-schedule-insns2")`:
  - per-channel state in local scalars;
  - the four free-slot reads at the top of each pass;
  - a send tail that issues CMD + STATUS + the completion counter right after the send;
  - an out-of-line handler for worker-connection changes and for upstream credit returns.
- The new loop is used only on BH E0 with two ERISCs, sender channels only, at most 4 VC0 channels, counter
  credits, and no speedy path, deadlock avoidance or first-level ack. Other routers keep the generic loop. It does
  not depend on the experiment flags.

**It did not compile** (kernels compile at test time; nothing reached the device). Fixed here:
- `continue_running_main_run_loop` was used in the new template loop before its declaration (added a declaration).
  The VC1 `static_assert` had moved outside `#if defined(FABRIC_2D_VC1_ACTIVE)` (moved back).
- v2 makes every sender channel a `LeanStrideSenderEthChannel`, which only has the compile-time slot advance. The
  generic send path still called the runtime variants when bit 24 is off, so it now always uses
  `advance_to_next_cached_buffer_slot_addr_by_stride_ct` (equivalent: slots are contiguous).
- The VC0 and VC1 `RouterStaticSizedChannelWriterAdapter` constructions had dropped v1's unused worker free-slots
  pointer argument, which shifted the rest. Restored, with the downstream stream id passed as `uint32_t`, as in v1.
- The user's file as received is kept in the session scratchpad (`fabric_erisc_router_v2.user_plus2fixes.cpp` has
  the first two fixes; `.fixed.cpp` is the tree's version).

**Code generation** (E0 of 13129088, the same router as §46):
- **Loop head:** the four free-slot loads go out back to back as `lui`/`lw` pairs (`0x8198`–`0x81b4`), with
  addresses from immediates and no stack reloads. Nothing uses them until all four are issued. This is better than
  both the asm (stack reloads) and §46's C++.
- **Send tail:** the STATUS load is first used 18 instructions after it is issued (asm: 8, §46 C++: 3).

**Measured** (`TT_FABRIC_KERNEL_VERSION=2 dqf_runs.sh dqk dqkb` against `dqi`/`dqib`):
- Validation passed: stream 10/0, mesh 96/0 twice. Every benchmark run passed too.

| Row (cycles/packet) | v1 | v2 | v2 with the scheduler on |
| --- | ---: | ---: | ---: |
| ERISC-only: 1-hop E/W write 4352 B | 132.5 | 145.3 | 135.7 |
| ERISC-only: E/W 16 B writes | 137.0 | 178.3 | 164.6 |
| ERISC-only: forwarded 16 B writes | 115.1 | 155.2 | 141.1 |
| ERISC-only: E/W atomic_inc | 135.2 | 153.2 | 135.3 |
| ERISC-only: trimmed first pass, 1-hop write | 153.7 | 185.2 | — |
| offload: E/W 4352 B write / 16 B write | 131.6 / 128.9 | 144.8 / 144.7 | 135.2 / 134.6 |
| offload: forwarded 4352 B / 16 B | 125.7 / 114.5 | 138.3 / 138.0 | 128.6 / 128.4 |
| offload: 1-hop / multicast write 4352 B | 130.8 / 140.7 | 144.8 / 155.7 | — |
| scatter (E1- or DRISC-bound) | unchanged | unchanged | unchanged |

- Latency is worse on 16 B round trips: +10 to +75 ns.
- **Disabling the scheduler costs ~10 cycles on every row where E0 sets the rate.** It turns off scheduling for the
  whole loop, sends included, not just the load group.
- **With the scheduler on**, E0 is still ~4–6 slower than v1 on writes and atomics, and ~3–14 slower on forwarded
  rows. Small writes on the ERISC-only build stay ~27 slower.
- The receive step changed only in formatting and in dropped experimental bits that neither build uses, so the
  small-write loss is not explained yet. The rows move with the E0-only attribute, so E0 is at least part of it.
- **Conclusion:** keep v1 as the reference. In v2 the load grouping got better, but the restructured loop costs
  more elsewhere.

## 48. Router kernel v2, second version: pure C++ v1 with compiler fences, on par with v1

**The user's update** to `fabric_erisc_router_v2.cpp` (`TT_FABRIC_KERNEL_VERSION=2`; v1 unchanged):
- The separate E0 loop of §47 is gone. v2 is now v1's loop structure with ~1,365 lines of unused experimental code
  removed, and no inline asm.
- **Bit 8:** the grouped free-slot read is four `volatile` loads followed by `std::atomic_signal_fence(seq_cst)`, a
  compiler barrier that GCC's scheduler does not move instructions across.
- **Bit 23:** the CMD + STATUS + completion-counter reads sit between two such fences, with a third fence after the
  local free-slot increment. The remote pointer update is written as the three TX-queue stores that
  `eth_write_remote_reg_no_txq_check` does (without its Watcher check).
- The STATUS test is `static_cast<int32_t>(txq_status_raw << 15) < 0`, which is bit 16 (`ETH_TXQ_STATUS_CMD_ONGOING_BIT`).
  Correct, though `31 - ETH_TXQ_STATUS_CMD_ONGOING_BIT` would say so.
- Lean sender channels are back to bit 24 only, which also removes the need for §47's lean-advance fix.
- It compiled as received.

**Code generation** (E0 of 13129088, the router used in §46/§47):
- **Bit 8:** the four loads issue back to back as `lui`/`lw` pairs from immediates (`0x8fac`–`0x8fc8`), with
  channel 0's mask after the fourth, the same shape as the asm minus its stack reloads.
- **Bit 23:** CMD, STATUS and the completion counter are loaded back to back (`0x9048`–`0x9050`), and STATUS is
  first used 9 instructions later (asm: 7).
- 2,842 instructions against v1's 2,909.

**Measured** (`TT_FABRIC_KERNEL_VERSION=2 dqf_runs.sh dql dqlb`, `dqk_compare.py dql`, against `dqi`/`dqib`):
- Every validation and benchmark run passed: stream 10/0, mesh 96/0 twice.
- Most rows are within ±2 cycles/packet, with no consistent sign.
  - Better: untrimmed small writes −1 to −1.7, forwarded small writes −1.3, N/S trimmed replays −1.5 to −2.9,
    offload multicast −2.0, W32 3-hop −6.
  - Worse: offload 4352 B writes +1.2 to +1.7, trimmed first passes +2.2 to +4.4, E/W trimmed small-write replays
    up to +7.3 (121.9 → 129.2).
- Latency is within noise, except ERISC-only E/W fused 16 B at +46 ns, a single row.
- These are the sizes the router's code layout moves by (§37, §40).
- **v2 is a working asm-free v1.** Its code generation is equal or better on both load groups, and its performance is
  at parity within layout noise.
- **Caveat:** that the signal fence keeps the scheduler from moving register-only instructions across it is GCC
  behavior, not a C++ guarantee. Recheck the disassembly after compiler upgrades.

## Reproducing

Host build: `./build_metal.sh --configure-only --build-metal-tests --enable-ccache --without-python-bindings`, then
`cmake --build build --target test_tt_fabric`. Router changes only need a re-run (JIT). Each experiment is:

```bash
./fabric_exp/run_exp.sh <label> fabric_exp/exp_base.yaml [ENV=VALUE ...]   # or exp_2k.yaml for 2 KiB only
./fabric_exp/run_exp.sh <label> fabric_exp/exp_latency.yaml TT_FABRIC_NO_BW_TELEMETRY=1 [ENV=VALUE ...]
#   latency mode: per-sample CSV in results/<label>_latency.csv (summary: results/latency_flags_summary.txt)
python3 fabric_exp/summarize.py ref=fabric_exp/results/notel.csv mine=fabric_exp/results/<label>.csv
python3 fabric_exp/phase_report.py fabric_exp/results/<label>_phase.csv fabric_exp/results/<label>.csv
```

| Knob | Effect |
| --- | --- |
| `TT_FABRIC_PHASE_PROFILE=1` | per-ERISC busy / blocked / idle accounting (perturbs throughput 13-26%) |
| `TT_FABRIC_PHASE_PROFILE=2` | plus per-packet phase timers (perturbs 27-45%; use the costs, not the throughput) |
| `TT_FABRIC_PHASE_PROFILE=3` | plus the primitive-cost microbenchmark (incl. instruction-fetch sweep) at router start |
| `TT_FABRIC_PHASE_MASK=<bits>` | time only the selected phases (bit per `FabricPhase`); counts are kept for all |
| `TT_FABRIC_PHASE_NO_ITER=1` | drop the per-iteration timestamp (counters-only at level 1) |
| `TT_FABRIC_EXP_TX_DELAY=N`, `TT_FABRIC_EXP_RX_DELAY=N` | spin N cycles per packet on TX / RX |
| `TT_FABRIC_EXP_FLAGS` bit 0 / 1 / 2 / 3 / 4 | skip local data writes / bare unicast write / batch upstream credits / lazy connection polling / push emulation |
| `TT_FABRIC_EXP_FLAGS` bit 5 / 6 | drop the speedy path's dead txq read / drop the can-send txq check (hangs) |
| `TT_FABRIC_EXP_STACK_PAINT=1` | fill each ERISC's free stack at start, report the deepest stack use (phase-profile output and CSV) |
| `TT_FABRIC_EXP_FLAGS` bit 7 / 8 | generic TX: read completion counters only with packets in flight / read up to four VC0 free-slot registers together per iteration |
| `TT_FABRIC_EXP_FLAGS` bit 9 | two-ERISC generic path: E1 returns upstream credits and handles worker connections for E0's sender channels |
| `TT_FABRIC_EXP_FLAGS` bit 10 / 11 / 12 / 13 / 14 | generic send: skip the unread worker write counter / early TX-queue read / slot address by stride / compile-time free-slot stream id / size read overlapped with the can-send check (§20; 21888 = 384 + 10 + 12 + 14 is the best build) |
| `TT_FABRIC_EXP_FLAGS` bit 18 / 19 | with bits 7 + 8: skip VC0 forwarding channels 1-3 while they have no packet and nothing awaiting completion, one branch for all / per channel (§23) |
| `TT_FABRIC_EXP_FLAGS` bit 17 | with bit 4: push with one command-buffer poll, plus a stateless doorbell to an unused stream register (§22; benchmark mode only) |
| `TT_FABRIC_EXP_FLAGS` bit 20 | with bit 4: push into a per-router DRISC's L1 ring plus a stateful doorbell to its stream 13 (NoC1 buffer 1 in 2D) (§24; benchmark mode only) |
| `TT_FABRIC_EXP_FLAGS` bit 21 | generic receiver: push the whole slot to the router's DRISC + doorbell at the top of the step, no decode/admission/forwarding (P5's E1; §25; benchmark mode only, multi-hop rows meaningless) |
| `TT_FABRIC_EXP_FLAGS` bit 22 | generic receiver, fused completion, no first-level ack: 16 transaction ids on receiver channel 0, completion reads skipped until a 5th slot is issued, completions acked 4 at a time in one credit frame (§26; passes the sanity yaml) |
| `TT_FABRIC_EXP_FLAGS` bit 23 | E0, counter credits: after the data command, the TX-queue busy reads and the L1 completion-counter read issued together in one asm block, consumed at K = 6 with the send bookkeeping in between (§28; passes the sanity yaml; small-packet latency +12-20 ns per hop) |
| `TT_FABRIC_EXP_FLAGS` bit 15 / 16 | generic receiver: the step's completion reads (write-sent status, TX-queue busy check) and the next step's packets-received count read together at the end of the step / without the packets-received count (§21; -5 to -8 on E1-bound paths, costs latency; parked) |
| `TT_FABRIC_EXP_MAX_SENDER_SLOTS`, `TT_FABRIC_EXP_MAX_RECEIVER_SLOTS` | cap VC0 channel depth |
| `TT_FABRIC_EXP_WORKER_SLOTS=W` | mesh: worker channel (VC0 ch0) gets W slots, receiver stays 16, forwarding channels share the rest (at most 16; with `TT_FABRIC_EXP_MAX_SENDER_SLOTS=8`, 8) (§27) |
| `TT_FABRIC_EXP_LOG_SLOTS=1` | log the chosen slot counts |
| `TT_FABRIC_EXP_LOG_CONFIG=1` | log each router's speedy flag, serviced channels, credit batching and slots |
| `TT_FABRIC_NO_BW_TELEMETRY=1` | do not force router BW telemetry in benchmark mode |
| `TT_FABRIC_TXQ_COUNTERS=1` | sample each ERISC's Ethernet TX queue counters and configuration (`txq_report.py`) |
| `TT_FABRIC_EXP_RECEIVER_AMORT=N` | fast path: completions per receiver credit return (default min(4, slots/2)) |
| `TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=N` | keepalive (sequence update) period of each tile's TX queues, restored in teardown |

Pass `TT_FABRIC_NO_BW_TELEMETRY=1` to compare with the tables here. With bandwidth telemetry on, one-way neighbor
exchange is ~7 cycles/packet slower (137 vs 130) and trimmed E/W 1-hop at 4 KiB ~9 slower (196 vs 187.6).

All knobs default off. With them off the router behaves as before: `results/p0.csv` matches
`results/base_upstream.csv` within 0.2% on every non-ring row (ring rows vary +/-5% run to run). Experiment flags other than the delays break data delivery and are
for benchmark mode only.

Files: router kernel (`fabric_erisc_router.cpp`, `fabric_erisc_router_speedy_path.hpp`,
`fabric_erisc_router_ct_args.hpp`), new headers under `tt_metal/fabric/hw/inc/edm_fabric/telemetry/`
(`fabric_phase_profile_types.hpp`, `fabric_phase_profiler.hpp`, `fabric_erisc_microbench.hpp`), builder
(`erisc_datamover_builder.{hpp,cpp}`, `fabric_static_sized_channels_allocator.cpp`), and test harness
(`tt_fabric_test_phase_profiler.{hpp,cpp}`, `tt_fabric_test_context.{hpp,cpp}`, `test_tt_fabric.cpp`).

ERISC load-overlap microbenchmark (§17; fabric off, slow dispatch):

```bash
cmake --build build --target test_erisc_load_overlap
TT_METAL_SLOW_DISPATCH_MODE=1 ERISC_LOAD_CSV=fabric_exp/results/erisc_load_overlap.csv \
  ./build/test/tt_metal/tt_fabric/test_erisc_load_overlap   # ERISC_LOAD_ETH_INDEX, ERISC_LOAD_DATA_CACHE
```

Ethernet TX queue probe (§16; fabric off, slow dispatch):

```bash
cmake --build build --target test_eth_txq_probe
TT_METAL_SLOW_DISPATCH_MODE=1 TXQ_PROBE_CSV=fabric_exp/results/txq_probe.csv \
  ./build/test/tt_metal/tt_fabric/test_eth_txq_probe   # TXQ_PROBE_VARIANTS, _PAYLOADS, _LINK, _CHIPS, _SLOTS
```

DRISC microbenchmark (§14; needs firmware bundle >= 19.12):

```bash
cmake --build build --target test_drisc_microbench
TT_METAL_SLOW_DISPATCH_MODE=1 DRISC_MB_CSV=fabric_exp/results/drisc_mb.csv \
  ./build/test/tt_metal/tt_fabric/test_drisc_microbench
```
