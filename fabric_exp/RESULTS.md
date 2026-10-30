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

**Repro note (added 2026-10-21):** "main" here means main's router behavior with the harness's BW telemetry off. That
is not what an unmodified main checkout measures.
- These runs used the study tree: router v1 at flags 0, which behaves like main (audited for `V3_INVENTORY.md`), with
  `TT_FABRIC_NO_BW_TELEMETRY=1`.
- That switch exists only in this tree and on `aagarwal/fabric-test-streams`. Main's `test_tt_fabric.cpp:205`
  forces BW telemetry on in every benchmark/latency run.
- With telemetry on, neighbor exchange at 4 KiB is 138 cycles/packet = 40.1 GB/s here (`nbr_tel.csv`), and 39.95 on
  a clean main checkout run as CI does (§9). With it off, 124.4 cycles = 44.5 GB/s.
- The same caveat applies to every "main" column in §44.
- **Goldens:** CI's golden `test_infra/golden/golden_bandwidth_summary_blackhole_p150_x4.csv` comes from
  `test_fabric_ubench_at_least_2x2_mesh.yaml` (telemetry on, 2x2 p150_x4). Neighbor exchange, 4 KiB, 1 link:
  golden 38.38 GB/s, clean main here 39.95, this tree with telemetry on 40.08, telemetry off 44.46.
  - That yaml also has `NeighborExchangeLinear` (topology Linear: generic 1D router, not the speedy path). Golden
    29.06 GB/s, clean main here 30.41 at 4 KiB. Not comparable with these numbers.

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

## 49. Router kernel v2, third version: worker/forwarding split; `[[likely]]` and the early semaphore read cost time

**The user's update** to `fabric_erisc_router_v2.cpp` (v1 unchanged):
- `run_sender_channel_step` splits into `run_worker_sender_step` (channel 0) and `run_forwarding_sender_step`
  (channels 1–3, chosen by `sender_ch_live_check_skip`).
  - The forwarding step computes `all_slots_free` once and returns early, marked `[[likely]]`, when nothing is queued.
  - It sets `has_unsent_packet = !all_slots_free`, which is exact for power-of-two slot counts.
- The worker step reads the connection semaphore as a fourth load in the bit 23 fenced group (CMD, STATUS, completion
  counter, semaphore). The close test becomes `!connected || conn_sem_raw == close_connection_request_value`, which
  is equivalent to v1's.
  - v1 also reads the semaphore every iteration, but at the end of the step, after the send. Here it moves ahead of
    the send decision.
- The STATUS test is now `txq_status_raw << (31 - ETH_TXQ_STATUS_CMD_ONGOING_BIT)`, the same instruction as before.
- **Latent:** the forwarding step drops v1's bubble flow-control check. That only matters with deadlock avoidance
  plus an injection forwarding channel, which none of our configs use.

**Code generation** (E0 of 13129088, west router): STATUS is spilled to the stack right after its load
(`lw s0,8(t0)` … `sw s0,24(sp)`), and the semaphore pointer is reloaded from the stack. The fourth load runs out of
registers.

**Full suite** (`TT_FABRIC_KERNEL_VERSION=2 dqf_runs.sh dqo dqob`, `dqv2_compare.py dqo`): all validation and
benchmark runs passed. Mean change in cycles/packet against v1 (second v2, `dql`, in brackets):

| Suite | ERISC-only | offload |
| --- | ---: | ---: |
| 4352 | +1.07 (−0.03) | +1.74 (+0.10) |
| mix | +4.99 (−0.46) | +3.71 (−0.44) |
| 4352_trim | +1.16 (+0.42) | +1.11 (+0.27) |
| mix_trim | +2.41 (+0.99) | +2.53 (+0.93) |
| small | +2.70 (+1.42) | +2.91 (+1.40) |

- Worst rows: ERISC-only untrimmed E/W 16 B write 137.0 → 152.9, 1-hop E/W 4352 B 132.5 → 138.3.
- W32 offload is the only suite that improves: 4352 −2.16, mix −1.41.
- 16 B latency +19 to +63 ns, growing with hops (ERISC-only 3-hop +56, offload 3-hop +63). 4096 B latency +2 to +48.

**A/B of the two suspects** (ERISC-only, `abv2c.py`, each variant patched into the user's file and then restored):
`nolikely` removes the `[[likely]]`; `nosem` reads the semaphore after the send as v1 does; `both` does both.

| Variant | mix | mix_trim | 1-hop E/W 4352 | E/W 16 B | E/W 16 B replay | 1-hop E/W 16 B lat | 3-hop 16 B lat |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v1 | 0 | 0 | 132.5 | 137.0 | 121.9 | 1356 | 3456 |
| dql (second v2) | −0.46 | +0.99 | 132.5 | 135.4 | 129.2 | 1356 | 3454 |
| dqo (third v2) | +4.99 | +2.41 | 138.3 | 152.9 | 122.1 | 1383 | 3513 |
| nolikely | +1.36 | +0.11 | 132.7 | 142.6 | 122.0 | 1383 | 3502 |
| nosem | +2.35 | +1.96 | — | 148.0 | 131.3 | — | — |
| both | −0.58 | +1.27 | 132.7 | 135.0 | 131.3 | 1356 | 3461 |

- **`[[likely]]` is the main cost**, as with bit 25 (§40). Removing it brings 1-hop 4352 B back to v1 (138.3 → 132.7)
  and the mix mean from +4.99 to +1.36. The 4352 B suites of both variants are at v1 (mean −0.02 / +0.05).
- **The early semaphore read is a trade-off.**
  - It brings the trimmed E/W 16 B replay to v1 (131.3 → 122.0).
  - It costs +7.6 on the untrimmed E/W 16 B write (135.0 → 142.6).
  - It costs 16 B latency: +27 ns at 1 hop, +41 ns at 3 hops (`nolikely` against `both`). The send now waits
    behind one more L1 load plus the spill on every router it crosses.
- `both` is the second v2 (`dql`) plus the split, and it measures the same.
- Nothing here beats v1 beyond layout noise (§37, §40).

## 50. Router kernel v2, fourth version: at or below v1 everywhere except the W32 offload build

**The user's update** to `fabric_erisc_router_v2.cpp` (v1 unchanged):
- `sender_channel_try_send` absorbs `send_next_data`. It owns the whole TX-queue send:
  - the space check, with bubble flow control restored for injection channels (fixes §49's gap);
  - the TX-queue busy check;
  - the send, pointer advances and the bit 23 group (CMD, STATUS, completion counter; no semaphore);
  - the free-slot increment and the remote doorbell.
- `sender_channel_process_completions` owns the completion and ack handling and the upstream credit return.
- `run_worker_sender_step` and `run_forwarding_sender_step` call those two functions.
  - The forwarding step keeps the idle early return without `[[likely]]`.
  - The worker step checks the connection at the end, after the send, as v1 does.
- The liveness-skip template argument matches v1: true for forwarding channels, false for the worker channel.
- Experiment bits 3 and 9 (lazy connection check, upstream on E1) are not handled. Earlier v2s didn't handle them
  either, and neither build sets them.

**Code generation** (E0 of 13129088, west router): CMD, STATUS and the completion counter load back to back
(`0x9058`–`0x9060`). STATUS stays in a register until its test 8 instructions later, so §49's spill is gone.
2,796 instructions (second v2: 2,842; third v2: 2,847).

**Full suite** (`TT_FABRIC_KERNEL_VERSION=2 dqf_runs.sh dqq dqqb`, `dqv2_compare.py dqq`): all validation and
benchmark runs passed (stream 10/0, mesh 96/0 twice). Mean change in cycles/packet against v1:

| Suite | ERISC-only | offload | offload W32 |
| --- | ---: | ---: | ---: |
| 4352 | +0.11 | +0.14 | **+3.36** |
| mix | −0.12 | +0.10 | **+4.54** |
| 4352_trim | −0.42 | −0.51 | |
| mix_trim | −0.62 | −0.53 | |
| small | −0.66 | −0.57 | |

- **Trimmed first passes improve 2–4 cycles on v1** in both builds, for example E/W 16 B 156.1 → 152.9 and
  1-hop E/W bidirectional 4352 B 156.4 → 152.4.
- **Trimmed E/W 16 B replay** is at v1 (122.0 against 121.9) without §49's early semaphore read. The second v2 was at 129.2.
- **Untrimmed small writes** are 0.6–0.8 below v1.
- **Latency:** 16 B within noise (−5 to +9 ns, one +9 fused row). 4096 B +2 to +9 ns at 1 hop, +19 to +31 ns at 3 hops.
- **W32 offload regresses:** E/W rows +11.2 at every size, N/S +5, 3-hop +4.
  - A rerun reproduces it (`dqq2`: E/W 142.2).
  - The second v2 rerun in the same session matches its earlier numbers (`dqlr`: 131.9 against 132.0), so the
    machine has not drifted.
  - The third v2 was the best W32 build (128.6).
  - Uniform across sizes means a per-packet loop cost on one core, most likely E0 of the W32 build (32 worker
    slots, 8-slot senders). That is unconfirmed. Code layout moves this binary by this much (§37, §40), but it has
    not been checked.

**Cleanup of the fourth version** (`dqr`, reduced suite):
- **The change:** the TX experiment flags in v2 are hard-wired to the 13129088 setting, and the `exp_` names are dropped.
  - Always on: bits 7, 8, 10, 12, 14, 19, 23.
  - Removed: bits 6, 11, 13, 24. This includes the lean sender-channel type.
  - Only bits 21 and 22 and the delay knobs still change v2.
- **Fix:** it first failed to compile. The removed `exp_local_delivery_is_data_only` is still called by
  `fabric_erisc_router_speedy_path.hpp:295`, so I restored it unchanged.
- **E0 is instruction-identical** to the fourth version (2,796 instructions).
- **Measured:** every run passed. Means against v1 match the fourth version within 0.02, and no row moved by 1 cycle
  or more: ERISC-only 4352 +0.11, mix −0.12, mix_trim −0.63, offload mix +0.09, W32 4352 +3.34. Latency is within
  ±5 ns of the fourth version.

## 51. Grouped pre-send loads: a win, but only if a busy TX queue still returns to the loop

**The user's update** to `sender_channel_try_send` (Blackhole, no spin-wait build):
- Before the send, the payload-size L1 read and the TX-queue dummy CMD and STATUS reads are issued as one fenced group.
- The payload size is computed as `excluding_header + sizeof(PACKET_HEADER_TYPE)`. That equals
  `get_payload_size_including_header()` for every header type, including the UDM header's override.
- **Behavior change:** when STATUS shows the queue busy, the channel spins until it is free and then sends.
  - Before, and in v1 with bit 14, the channel returned `false` and the loop moved on.
  - `EVENT_TX_BLOCKED_TXQ` is no longer counted on Blackhole.
- **Code generation** (E0 of 13129088): the group is `lhu` size, CMD, STATUS, and STATUS is tested 5 instructions
  later, the same shape as before. 2,808 instructions (cleanup: 2,796).

**Measured** (`dqt`, reduced suite): all runs passed. Mean change against v1, with the cleanup (`dqr`) in brackets:
ERISC-only 4352 +1.11 (+0.11), mix +4.23 (−0.12), mix_trim +1.65 (−0.63), offload mix +3.25 (+0.09),
W32 4352 +2.22 (+3.34).
- Untrimmed 16 B–1 KiB writes +13–14 (E/W 136.4 → 149.7).
- Trimmed first passes +7 to +14, offload small writes +7.7, 1-hop 4352 B +3.6.
- Latency within +9 ns.

**A/B** (`dqtyb`, ERISC-only): the update unchanged except that a busy queue returns `false` as before.

| Row | v1 | cleanup | update (spin) | update + return false |
| --- | ---: | ---: | ---: | ---: |
| 4352 mean vs v1 | 0 | +0.11 | +1.11 | +0.02 |
| mix mean vs v1 | 0 | −0.12 | +4.23 | −0.66 |
| 1-hop E/W 4352 | 132.5 | 133.0 | 136.6 | 132.8 |
| E/W 16 B write | 137.0 | 136.4 | 149.7 | 134.3 |
| N/S 16 B write | 134.1 | 133.4 | 147.5 | 132.2 |

- **The spin is the whole regression.** When the old code found the queue busy, it went back to the loop and did its
  other work while the queue drained. The spin makes that work wait until after the send.
- **The grouped loads on their own are a win:** small writes 2.1 below the cleanup and 2.7 below v1 (E/W).
- In this variant `did_something` is set before the busy check, so a yield still counts as activity. It only affects
  idle context switching, and a channel with a packet waiting is not idle.

## 52. Router kernel v2 with the yield change: best on small and forwarded writes, W32 worse again

**The user's change:** byte-identical to §51's A/B variant, where a busy TX queue returns `false`.

**Measured** (`dqu`, reduced suite): all runs passed. Mean change against v1, with the cleanup (`dqr`) in brackets:

| Suite | `dqu` | cleanup |
| --- | ---: | ---: |
| ERISC-only 4352 | +0.02 | +0.11 |
| ERISC-only mix | −0.67 | −0.12 |
| ERISC-only mix_trim | +0.00 | −0.63 |
| offload mix | −0.57 | +0.09 |
| offload W32 4352 | **+6.75** | +3.34 |

- **Repeatable:** ERISC-only 4352 and mix match §51's A/B run within 0.01.
- **Better than the cleanup:**
  - Untrimmed small writes: E/W 136.4 → 134.3, N/S 133.4 → 132.2.
  - Offload forwarded small writes: 114.7 → 112.4.
- **Worse than the cleanup:** trimmed first passes +2.2 to +3.5. They are back at v1's level, so the cleanup's gain is
  gone. Trimmed E/W small-write replays +1.5 to +1.7 (123.6 against v1 121.9).
- **W32 offload worsens again** against v1: N/S 147.1 (133.4), 1-hop E/W 144.7 (131.0), bidirectional 149.5
  (131.6).
  - The W32 mean against v1 by version: dql −1.16, third v2 −2.16, fourth v2 +3.36, cleanup +3.34, spin +2.22,
    yield +6.75.
  - That binary moves several cycles with each edit, in both directions. The cause is still unconfirmed.
- **Latency** is within ±11 ns of v1.

## 53. E1 experiments in v2: stateful E/W forwarding brings ERISC-only multi-hop to the DRISC offload's level

**The user's three E1 changes** in `fabric_erisc_router_v2.cpp`, as E1 bits 26–28. These are new; the old §35–§37
bits with the same numbers are gone from the tree.
- **Bit 26:** load header word 40 (payload size, send type, `src_ch_id`) once, after admission, and pass the values
  to forwarding and local delivery. `src_ch_id` is no longer written before admission. Only admitted slots read it.
- **Bit 27:** stateful NoC1 forwarding on two-ERISC E/W routers with one live direction.
  - Main allowed this only in single-ERISC mode.
  - E1 re-programs downstream 0's command buffers last at init, after the loop over all downstream connections.
  - E1 forwards on NoC1 buffers 2 (data) and 1 (sync). Its local writes use buffer 0 and its atomics and inline writes
    use buffer 3. E0 runs on NoC0 (the builder sets ERISC 0 to NOC_0 and ERISC 1 to NOC_1).
  - Still unverified from source: whether base firmware touches NoC1 buffers 1/2 during E0's context switch. Main's
    single-ERISC stateful path makes the same assumption.
- **Bit 28:** admission with compile-time stream IDs. Same registers as the adapter's `edm_has_space_for_packet`.
  The stream reads are issued back to back, and local-only packets return early.
- **Fix needed to compile:** `PackedHeaderWord40` took `offsetof` on the non-standard-layout header
  (`-Werror=invalid-offsetof`), failing every variant. It now takes the offsets on the header's deduced
  `PacketHeaderBase`, as `PackedPayloadAndSendType` does.

**Measured** (`fabric_exp/dqx_runs.sh`, ERISC-only, labels `dqx_<flags>_*`):
- Every variant passed mesh validation (96/0) and every benchmark.
- The new baseline with all bits off matches §52 within 0.05, so the restructure is neutral.

| Row | v1 | base | +26 | +28 | +27 | 27+28 | all |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4352 mean vs base | −0.07 | 0 | +0.83 | −0.97 | −3.22 | **−3.38** | −2.83 |
| mix mean vs base | +0.66 | 0 | +0.50 | 0.00 | +0.26 | +0.23 | +0.47 |
| 3-hop E/W write | 157.2 | 157.0 | 161.5 | 152.2 | 143.7 | **143.4** | 146.0 |
| multicast write | 154.7 | 154.5 | 152.8 | 152.1 | 142.1 | **141.6** | 141.1 |
| multicast scatter | 197.6 | 197.7 | 196.6 | 195.3 | 188.1 | **187.6** | 188.2 |
| 1-hop/forwarded scatter, atomic, fused | — | — | +1.9 | 0 | +1.0 | +1.0 | +1.9 |
| forwarded small writes | 115.1 | 115.1 | 117.1 | 115.1 | 116.1 | 116.0 | 117.1 |
| 3-hop latency 16 B (ns) | 3456 | 3466 | 3484 | 3446 | 3396 | **3387** | 3402 |
| 3-hop latency 4 KiB (ns) | 4326 | 4341 | 4345 | 4319 | 4306 | **4290** | 4316 |

- **Bit 27 carries the gain:** 3-hop −13.3 and multicast −12.4. E/W 1-hop rows don't move, as expected, since E0
  binds them.
  - At 4352 B, 3-hop is now 41.0 GB/s and multicast 41.5 GB/s.
  - ERISC-only best before this was 37.4 and 38.0 (§44). DRISC offload v4 is 40.9 and 41.8.
- **Bit 27 also costs +1.0 on scatter, atomic, fused and forwarded rows**, uniformly across sizes. That looks like a
  per-packet cost or layout on one core; not investigated.
- **Bit 28 alone:** 3-hop −4.8, with no row worse. On top of 27 it adds only −0.3 to −0.5; the two mostly overlap.
- **Bit 26 makes things worse** alone (3-hop +4.5, forwarded +2, atomics/fused/scatter +1.9) and in combination (all
  vs 27+28: 3-hop +2.6).
  - The removed loads probably overlapped other work already. Moving the word-40 load after admission likely puts
    it on the critical path. Not checked in the disassembly.
- **Offload build:** baseline and +26 are equal (+0.03 / +0.02). The DRISC sets the rate there.
- **Recommendation:** keep 27 and 28, drop 26.

## 54. E1 on 1-hop writes: ~125 cycles/packet in total, of which the timed phases are ~80

**Setup:** current best ERISC-only build, v2 at 13129088 + E1 bits 27/28 (415782272), 4352 B suite
(`fabric_exp/dqe1_runs.sh`). E1 spins `TT_FABRIC_EXP_RX_DELAY` cycles per packet. A fully E1-bound row (multicast)
gives the spin's real cost: +20 adds 23.1 cycles.

| Row | no delay | RX +20 | +40 | +60 | +80 | E1 slack | E1 per packet |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1-hop E/W write | 132.8 | 147.1 | 172.0 | 190.0 | 208.1 | ~9 | **~124** |
| 1-hop N/S write | 132.6 | 150.3 | 175.1 | 193.0 | 211.1 | ~5 | **~127** |
| 1-hop E/W bidirectional | 139.0 | 147.5 | 172.0 | 189.9 | 207.8 | ~15 | ~124 |
| 3-hop E/W write | 143.4 | 161.1 | 186.0 | 204.0 | 222.0 | ~5 | (E1-bound) |
| multicast write | 141.6 | 164.7 | 186.7 | 203.9 | 219.8 | 0 | (E1-bound) |

- **E1 has little slack on 1-hop writes:** about 124–127 cycles/packet against a 132.8 interval. Stages near balance
  lose a few cycles, so the true figure may be a little lower (§26).
- **The ~80 figure counts only E1's timed phases.** From §39's level-2 profile (v1, same E1 receive path), per
  packet:
  - decode/admit 21.0 (E/W) / 36.1 (N/S);
  - local write 50.1;
  - completion 18.8 / 17.9, from 0.36 / 0.31 batches per packet at ~52 cycles each.
    - Corrected 2026-10-21: an earlier version of this bullet said "~7 per packet (18.8 per batch)". The profiler
      reports ticks per packet.

  Raw that sums to ~90 / ~104. Minus the ~4-cycle timer self-cost per timed occurrence it is ~80 / ~95. The rest,
  roughly 45 (E/W) / 32 (N/S), is broken down in §55.
- **E1 in offload mode** (blind push to the DRISC, §26) is ≤ 86 in total, because it skips local delivery.
- **Level-1 profile** (`dqe1_prof1_*`): the timestamps stretch the loop to ~223, but both ERISCs read ~93% busy on
  1-hop writes. Both are near saturation, consistent with the slack test.

## 55. Where E1's ~125 cycles go on a 1-hop write

**Method** (`fabric_exp/dqe1s_runs.sh`, `exp_2d_4352_1hop_write.yaml`, the user's v2 with E1 bits 27/28 hard-wired,
flags 13129088):
- Each run times one segment at level 2, with no per-iteration timing (`TT_FABRIC_PHASE_NO_ITER=1`).
- The phase mask is compile-time, so the run carries exactly one timer pair. Net = raw ticks minus the calibrated
  4.0 per occurrence.
- Four extra E1 segments come from a temporary instrumented copy of v2, on phase IDs E0 never hits in this traffic.
  The user's file was restored and byte-checked.
- Level 2 still increments a counter on every phase `close`, which slows the profiled runs to 210–260 overall. That
  cost falls outside the segments: no segment contains another phase's close except the completion send inside the
  completion block, which is subtracted.

| E1 segment, per packet | E/W | N/S | What runs there |
| --- | ---: | ---: | --- |
| packet poll | 9.2 | 9.2 | packets-received stream-register read, used at once (§17: 7), plus the test |
| decode/admit (phase) | 17.1 | 20.1 | header and route decode, admission |
| post-admit | 13.1 | 13.0 | transaction-ID tracker update; the local-only pass through `forward_2d_dispatch` (loads `payload_size_bytes`); `generic_rx_local_delivery` loading header word 40 again |
| local write (phase) | 51.1 | 52.2 | the NoC write |
| post-local | 6.0 | 6.0 | sent-counter increment, packets-received stream decrement |
| completion decision | ~11 | ~11 | bit 22 bookkeeping every iteration; every 4th packet the write-flushed and TX-queue busy reads |
| completion send (phase) | ~14 | ~14 | credit frame for a batch of 4, ~57 per batch × 0.25 |
| loop | ~2 | ~2 | inner-loop branch; outer checks once per 32 iterations (§17) |
| **total** | **~124** | **~128** | slack test (§54): ~124 / ~127 |

- **The untimed ~41 is not mostly polling.**
  - The poll is 9.
  - About 13 is the gap between admission and the local write.
  - 6 is retiring the packet.
  - About 11 is bit 22's per-iteration completion bookkeeping.
  - The loop itself costs ~2.
- **Post-admit (13) is two L1 header loads that cost their full latency.** `forward_2d_dispatch` loads the payload
  size even for local-only packets, and the local-delivery wrapper then loads word 40 again.
  - This is from the source, not checked in the disassembly.
  - Bit 26 tried to remove the duplicate and got slower (§53). A version that skips the forward-dispatch load only
    when no Ethernet direction is set might keep the gain without moving a load onto the critical path. Untested.
- **N/S decode/admit fell from 36 (v1, §39) to 20**, with bit 28's compile-time admission. E/W is unchanged at 17.
- **The completion decision (~11)** includes, every 4 packets, the write-flushed and TX-queue reads (§17: up to 8 if
  not grouped).
- Idle iterations add the poll plus the completion decision, which can send a partial batch. E1 has 5–9 cycles of
  slack, so unprofiled there are few.

## 56. E1 local-delivery header loads: a faster E1 can slow E0 through bit 22's completion batching

**Premise check (disassembly, E/W and N/S E1 of the user's v2):** §55's guess was wrong.
- `forward_2d_dispatch`'s payload-size load is already sunk into the forward branches. On N/S, bit 28's early return
  skips it too, so local-only packets never execute it.
- The local path loads header word 40 three times:
  - `lbu 43` (`src_ch_id`), before admission;
  - `lhu 40`, the payload size the caller passes;
  - `lw 40` in `generic_rx_local_delivery`, for the send type, used at once by the jump-table dispatch.

**Variants** (`fabric_exp/dqpa_runs.sh`, `dqpa_compare.py`; whole-file swaps, the user's file restored and checked):
- **A:** `generic_rx_local_delivery` takes the payload size from its own packed word-40 load, so the caller's `lhu`
  is dead.
- **B:** A, plus the pre-admission `src_ch_id` byte load widened to one 32-bit word-40 load (base pointer + `offsetof`
  on the deduced `PacketHeaderBase`). 2D local delivery reuses its payload size and send type, so the dispatch no
  longer waits on a load.
  - The first two attempts at B failed `-Werror=strict-aliasing`: a `uint16_t` member's address was cast to
    `uint32_t*`, directly and then through `uintptr_t`.

**Measured** (ERISC-only, flags 13129088): every variant passed mesh validation (96/0) and every benchmark.
- **Base (the user's hard-wired file) equals the bit-27/28 flag build:** +0.02 / −0.01.

| Row | v1 | base | A | B |
| --- | ---: | ---: | ---: | ---: |
| 4352 mean vs base | +3.33 | 0 | −0.37 | **−2.72** |
| mix mean vs base | +0.42 | 0 | −0.23 | **−1.42** |
| 1-hop E/W scatter | 169.5 | 170.5 | 169.4 | 164.4 |
| E/W atomic | 135.2 | 136.2 | 135.1 | 130.4 |
| E/W fused | 157.6 | 158.6 | 158.4 | 153.4 |
| forwarded 16 B write | 115.1 | 116.1 | 116.0 | 111.5 |
| multicast write / scatter | 154.7 / 197.6 | 141.6 / 187.5 | 142.4 / 187.4 | 139.7 / 183.1 |
| 1-hop E/W 4352 B write | 132.5 | 132.8 | 132.8 | 132.1 |
| E/W 16 B / 1 KiB write (stream) | 137.0 / 137.3 | 133.5 / 133.6 | 133.6 / 133.7 | **138.1 / 138.3** |
| N/S 16 B write (stream) | 134.1 | 132.2 | 132.2 | 132.9 |

- Latency: A and B are within ±12 ns of base.
- **A** recovers the +1.0 that bit 27 cost on scatter and atomics, back to v1, and changes nothing else. 1-hop
  writes don't move, as §17 predicted: two back-to-back L1 loads cost about the same as one used at once.
- **B** gains 5–6 on every E1-bound row, but E/W 16 B–1 KiB writes get +4.6.

**Why B slows E/W small writes** (`dqpa_slack_runs.sh`, `p3_small_ew_ns.yaml`, `dqpat_*`):
- **E0's binary is instruction-identical** in base and B (all eight E0 builds compared); only E1 changed.
- **Slack test, E/W 16 B write:**

  | | rate | TX +20 | RX +20 |
  | --- | ---: | ---: | ---: |
  | base | 133.5 | 179.3 | 146.7 |
  | B | 138.1 | 179.7 | 141.7 |

  E1's slack grew (RX +20 costs +3.6 against +13.2), so E1 is faster. E0 binds.
- **TX-queue counters** (`TT_FABRIC_TXQ_COUNTERS=1`), receiving E1, 48,000 packets. One transfer per completion
  frame (§57 check: at 4352 B, 12,195 transfers = bit 22's batch of 4):
  - base: 24,107 frames (~2.0 packets per frame);
  - B: 38,856 (~1.24 per frame);
  - at 4352 B the count doesn't change (12,195 / 12,161, ~3.9 per frame).
  - Corrected 2026-10-21: an earlier version said ~4.0 and ~2.5, assuming 2 transfers per frame. The ratio, and the
    conclusion, are unchanged.
- **Mechanism:** with B's faster E1 the stream runs dry more often, and bit 22 flushes a partial batch on any idle
  iteration (`uncompleted != 0 && !progress`). E0 then processes ~60% more completion frames, and E0 pays per frame
  (§26). So an E1 speed-up turns into an E0 slow-down on E0-bound rows.
- **Next step:** pair B with a partial-batch hold that only flushes after N consecutive idle polls and resets on
  progress, like the §37 design. §37's hold was mixed when E1 had little slack; it should matter more now. Untested.

## 57. E1 word-40 reuse plus a partial-batch hold: best ERISC-only build so far

**The user's update to v2:**
- **Variant A** (§56) is folded in unconditionally.
- **E1 bit 26** = variant B: the pre-admission `src_ch_id` load widened to `PackedHeaderWord40::load` (base pointer +
  `offsetof` on the deduced `PacketHeaderBase`), with payload size and send type reused by local delivery.
- **E1 bit 28** = a partial-batch hold on top of bit 22.
  - Iterations with progress keep the stock "> 4" rule.
  - On idle iterations a full batch still flushes at once, while a 1–3 packet tail waits for 4 consecutive idle
    polls.
  - The counter is a per-receiver-channel local, increments only while something is pending, and resets on progress
    or on a send. So the hold is bounded and cannot deadlock.
- **Reviewed:** `src_ch_id` is set exactly once on 1D and 2D paths, and forwarding is unchanged. Compiled first try.

**Measured** (`fabric_exp/dqhd_runs.sh`, `dqhd_compare.py`, labels `dqhd_<flags>_*`, ERISC-only): all four values passed
mesh validation (96/0) and every benchmark. Cycles/packet:

| Row | v1 | A (13129088) | B (+26) | A+hold (+28) | **B+hold (+26+28)** |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4352 mean vs A | +3.71 | 0 | −2.38 | +0.24 | **−2.74** |
| mix mean vs A | +0.66 | 0 | −1.20 | −0.97 | **−3.21** |
| 1-hop E/W write 4352 | 132.5 | 132.8 | 132.0 | 132.7 | **130.6** |
| E/W 16 B / 1 KiB write | 137.0 / 137.3 | 133.6 / 133.7 | 138.1 / 138.3 | 128.0 / 128.2 | **128.3 / 128.3** |
| N/S 16 B write | 134.1 | 132.2 | 132.9 | 128.4 | **128.3** |
| 1-hop E/W scatter | 169.5 | 169.4 | 164.4 | 170.0 | **164.0** |
| 1-hop N/S scatter | 176.9 | 175.7 | 174.6 | 173.4 | **172.3** |
| E/W atomic / fused | 135.2 / 157.6 | 135.1 / 158.4 | 130.4 / 153.4 | 135.6 / 158.0 | **130.1 / 152.0** |
| forwarded 16 B write | 115.1 | 116.0 | 111.5 | 116.6 | **111.4** |
| multicast write / scatter | 154.7 / 197.6 | 142.4 / 187.4 | 139.8 / 183.1 | 143.6 / 188.8 | **140.9 / 184.1** |
| N/S atomic | 138.2 | 137.2 | 136.3 | 139.0 | 138.0 |
| packets per completion frame, E/W 16 B (TX-queue counters) | | 1.59 | 1.25 | 3.87 | 3.76 |

- **The hold fixes what B broke.** It restores ~3.8–3.9 packets per completion frame on small writes (4352 B rows stay
  at ~3.94).
- **The hold also helps on its own.** Variant A already left E1 enough slack to flush partial batches (1.59 per
  frame). Holding them takes small writes from 133.6 to 128.0 on E/W and from 132.2 to 128.4 on N/S.
- **B+hold keeps both gains:**
  - small writes −4 to −5;
  - scatter, atomics, fused and forwarded rows −4.6 to −6.4;
  - 1-hop E/W 4352 B write 130.6. That is 45.0 GB/s, equal to DRISC offload v4's 44.9 (§44) and 96% of the wire
    (125.4).
- **Costs of the hold:**
  - N/S atomic +0.8 against A;
  - multicast write −1.5 against A, but +1.1 against B alone;
  - 3-hop 16 B latency +23 to +27 ns (3379 → 3402 / 3406).
  - Other latency rows are within ±13 ns.
  - With a single packet in flight the hold should only delay completion credits. Why 3-hop small packets see it was
    not investigated.
- **Recommendation:** keep both (hard-wire bits 26 and 28).

## 58. Fused write + atomic with flush: 3.4× slower, and production uses it

**Our benchmarks never flush.** Every fused header the test harness builds passes `flush = false`, the same on main:
- benchmark and stream traffic, `tt_fabric_test_kernels_utils.hpp:1265` and `:1283`;
- latency sender and responder.

The harness comment explains why: the payload and the counter sit on the same core, so NoC ordering already keeps the
write before the atomic. So the fused premium in §42–§57 (~+20 over a write) is the atomic's second NoC command and
three more header fields, not a flush.

**Production does flush.**
- `NocUnicastAtomicIncFusedCommandHeader`'s constructor defaults to `flush = true` (`fabric_edm_packet_header.hpp:299`).
- Kernels that rely on that default or pass `true` explicitly:
  - `ring_reduce_scatter_minimal_async_writer.cpp:220`;
  - `high_bw_all_gather/.../unicast_common.hpp:258`;
  - `reduce_to_root` sender and root writers;
  - `tt_fabric_udm.hpp:672`;
  - `minimal_ccl_common.hpp` passes a caller's flag.

**What a flush does on E1** (`flush_write_to_noc_pipeline`, Blackhole with bit 22):
- E1 issues the payload write, then spins until every outstanding non-posted write on its local NoC is acknowledged
  (`ncrisc_noc_nonposted_writes_flushed`), then issues the atomic.
- The receiver stops while the write drains and its ack returns.

**Measured** (`fabric_exp/dqfl_runs.sh`, `streams/p3_fused.yaml`, best ERISC-only build 348673408):
- The harness's flush flag was forced on by a temporary edit, then restored and checked.

| Row | flush off | flush on |
| --- | ---: | ---: |
| E/W fused 4352 B | 152.0 (38.65 GB/s) | **522.8 (11.24 GB/s)** |
| N/S fused 4352 B | 160.3 (36.65 GB/s) | **531.2 (11.06 GB/s)** |
| forwarded fused 4352 B | 152.0 (38.66 GB/s) | **522.7 (11.24 GB/s)** |
| E/W / forwarded write 4352 B | 132.6 / 125.7 | 132.6 / 125.7 |

- A flushed fused packet costs **+371 cycles**, the drain-and-ack time of a 4352 B local write. That's 3.4× slower.
  Writes don't move.
- The rate cost per packet in production depends on how often fused packets appear. For example, one fused packet
  per 8 would add ~46 cycles per packet on average. A flush also drains every earlier write still in flight.
- This is not specific to the study builds: main's flush is the per-transaction-ID loop over the same writes.
  Not measured on main.

**Options (untested):**
- **Skip the flush when it's redundant.** When the payload and semaphore addresses name the same core and the atomic
  uses the same NoC and VC, the write is already ordered before the atomic, which is what the harness relies on. One
  compare of the coordinate bits would spare every same-core producer. Needs a hardware-ordering confirmation.
- **Don't spin.** Record the pending atomic with its write's transaction ID, keep servicing packets, and issue the atomic
  (then the completion) once that ID is flushed. The ID tracking already exists for completions.
- **Benchmark coverage:** add a flushed fused row to the stream suite so this cost shows up in our numbers.

## 59. Survey: which ops make the router flush

**Method:** `fabric_exp/flush_survey.py` (call sites in `fabric_exp/results/flush_survey.csv`).
- It parses every construction of the three header types whose `flush` makes the receiving router drain its
  outstanding local writes before the atomic, across `ttnn/`, `models/` and `tt_metal/` (tests excluded):
  - `NocUnicastAtomicIncCommandHeader`, the standalone atomic;
  - `NocUnicastAtomicIncFusedCommandHeader`, write + atomic;
  - `NocUnicastScatterAtomicIncFusedCommandHeader`, scatter + atomic.
- **All three default to `flush = true`.**
- Each call site is classified as explicit true, false, default, or an expression.
- Expressions were resolved by hand: shared helpers in `minimal_ccl_common.hpp` and `moe_utils.hpp` traced to their
  callers, local variables and `constexpr` flags read at the call site.
- Sampled default sites were checked by eye.

**Counts:**
- **182 call sites:**
  - 132 use the default (flush);
  - 13 pass explicit `true`;
  - 15 pass `false`;
  - 22 pass an expression. These are the helpers and fabric API pass-throughs (`linear/addrgen_api.h`, `mesh/api.h`,
    UDM), plus a few locals.
- **By type:** standalone atomics are 123 default, 4 true, 6 false. Fused is 7 default, 5 true, 9 false. Scatter + atomic
  is 2 default, 4 true.
- **39 ops or kernel directories** construct these headers:
  - 35 have at least one flushing site;
  - 4 never flush: `all_reduce_async`, `rms_allgather`, and DeepSeek V3 B1 `d2d_exchange` / `host_io`.

**Ops that flush on a data packet** (fused or scatter + atomic, the §58 pattern), 10:

| Op | flushing sites | notes |
| --- | --- | --- |
| `experimental/ccl/reduce_scatter_minimal_async` | fused 4, scatter 2 (default) + 21 atomics | the chunk's semaphore increment is fused onto its last data packet: one drain per chunk |
| `ccl/reduce_to_root` | fused 2 (true) + 3 atomics | |
| `experimental/ccl/reduce_scatter_minimal_direct` | fused 2 (default) + 3 atomics | |
| `experimental/ccl/ring_attention_all_gather_async` | fused 1, scatter 1 (true) + 1 atomic | |
| `experimental/ccl/deepseek_moe_reduce_scatter` | scatter 2 (true) | |
| `models/demos/deepseek_v3_b1/unified_kernels` | scatter 1 (true) + 8 atomics | `sdpa_reduce_worker`'s fused writes do not flush |
| `experimental/high_bw_all_gather` | fused 1 (true) + 3 atomics | |
| `tt_metal/impl/dispatch/kernels` | fused 1 (default) + 1 atomic | dispatch, not a CCL |
| `experimental/ccl/dit_fused_distributed_rmsnorm` | fused (true) via helper | |
| `experimental/ccl/dit_fused_norm_common` | fused (true) via helper | payload to `dram_dest`: the DRAM-then-notify case |

**Ops that flush only through standalone atomics**, 25: `all_gather_async` (14), `neighbor_pad_halo` (13),
`all_to_all_async_generic` (10, its fused write passes `false`), `ccl/all_gather` (8), `neighbor_pad_async` (8),
`ccl/broadcast` (6), `strided_reduce_scatter_async` (5), `all_gather_minimal_matmul_async` (4),
`slice_reshard_async`, `strided_all_gather_async`, `point_to_point`, `all_to_all_combine` (via `moe_utils`),
`all_gather_concat_heads_fused`, `all_to_all_async`, `llama_all_gather_matmul_async`, both `llama_reduce_scatter`
variants (their fused writes are `constexpr false`), DeepSeek prefill
`attn_res_gather_softmax` / `combine` / `combine_fabric2d` / `dispatch` / `dispatch_fabric2d` (data handshakes
`true`, init handshakes `false`), `all_reduce_create_qkv_heads`, `transformer/sdpa`, `ttnn/core/tensor/kernels`.

**Caveats:**
- These are static call sites, not runtime frequency. Many default sites are init or barrier handshakes, often with
  value 0 or 1, where nothing is outstanding and the drain costs ~nothing.
- What costs is a flush that follows data writes still in flight. That is up to the ~371 cycles of §58 for a 4352 B
  write, paid once per flushed packet: per chunk for `reduce_scatter_minimal_async`, per signal for the
  standalone-atomic ops.
- Whether the payload lands in DRAM (where the flush is needed) or same-core L1 (where NoC ordering would suffice)
  was not classified per op, apart from `dit_fused_norm_common`.

## 60. Survey: which ops could opt in to "order only my own data" flushes

**Context:** the user's direction is a new header field, off by default. With it set, the router would order a packet's
atomic only after that sender's earlier data, instead of acting as a full barrier, and would keep servicing later
packets meanwhile. This survey, a read-only code read by a subagent, checked each flushing producer's consumers for
patterns that rely on the barrier:
- **H1:** a later unflushed write or flag used as evidence that earlier data landed;
- **H2:** a buffer reused with no consumer-to-producer release;
- **H3:** unflushed increments sharing a counted semaphore.

I spot-checked the key citations against the code. Items marked (inf) are the subagent's inference.

**Router conditions every candidate needs:**
- **C1, multi-packet coverage.** Most producers send many plain packets, then one flushed signal. So the atomic must
  wait for all earlier writes on the receiver channel (an in-order "landed" pointer over slots ≤ N), not just its
  own payload.
- **C2, atomics in order.** Every consumer waits on a count.
- **C3, multicast.** The rule applies at each hop.

**Verdicts:**
- **Opt-in candidates.** No H1 or H3 pattern was found anywhere.
  - **Strongest:**
    - `deepseek_moe_reduce_scatter`: a flushed scatter+atomic on every packet, L1 on the semaphore core (inf), one
      counting consumer;
    - `reduce_scatter_minimal_async`: a flushed increment every `chunks_per_sync` chunks into the intermediate tensor,
      often DRAM; each (batch, slice) has its own region; a hot CCL;
    - `high_bw_all_gather`: several mux workers share a link, so today each flush also waits for the other workers'
      writes;
    - `cq_relay` (dispatch): flushes per page batch, and credits already guard reuse.
  - **Safe but little to gain** (once per op or iteration): `reduce_scatter_minimal_direct`, `reduce_to_root`,
    DeepSeek V3 B1 sampling, the dit ops, all-gather variants, broadcast, neighbor pad, prefill 1D combine/dispatch,
    `all_to_all_combine`.
- **Not safe:** DeepSeek prefill `combine_fabric2d` and `dispatch_fabric2d` (H2).
  - Forwarding pages are reused across launches with no release.
  - The code says so at `reader_combine_fabric2d.cpp:637-638`: "a chip far enough ahead can still overwrite pages we
    have not read". Dispatch is the same (inf).
  - The race exists today. Opt-in would only remove some throttling.
- **Same class, unverified:** ring-attention AG/halo and the dit ops reuse buffers across launches with no handshake
  inside the op. Host-side rotation may cover this.

**Possible existing ordering bugs, independent of the opt-in** (leads for the op owners; consumer side not verified):
- `all_to_all_async_generic`: on the local path the semaphore is incremented before the data writes are issued
  (`all_to_all_sender_writer.cpp:411-417`, checked). The remote path's last packet is fused with `flush = false`
  (:440-442) after scatter writes to possibly different cores.
- `ccl/all_gather` multicast with load-balanced alternate routes on even rings: no flushed increment covers the
  alternate-path writes (inf, `all_gather_multicast_factory.cpp:463-471/513-516` vs `multicast_writer.cpp:233-240`).
- `neighbor_pad_halo` corner-first mode sends the last row's middle sticks after the semaphore raise
  (`np_writer` :408-433, inf).

**Router note:** `sender_combine_fabric2d.cpp:77-78` says a fused write+atomic whose payload goes to DRAM "is
documented to hang Blackhole", so that op uses a header-only atomic after the writes (checked). An opt-in design has
to work for the standalone-atomic form (C1), because DRAM producers can't switch to fused packets.

## 61. Staged atomic: flushed fused −26 to −32, unflushed fused +3.5 to +7.6

**The user's update to v2:**
- **E1 bits 26 (word-40 reuse) and 28 (partial-batch hold) are hard-wired.** The hold's `FABRIC_2D` guard is gone, so
  1D builds on the batched-completion path now hold partial batches too. That is bounded, but not measured here.
- **The local-delivery function is copied into v2** as `v2_execute_chip_unicast_to_local_chip_impl`. It matches the
  header's ERISC path except for comments and the three atomic sites: standalone atomic, fused, and the scatter's
  semaphore chunk.
- **The atomic is staged:** `v2_local_semaphore_inc` writes the same registers and counters as the stock
  `noc_fast_atomic_increment`, but the flush wait sits between staging and the send.
  - Data-before-atomic is kept, and nothing else on E1 uses `write_at_cmd_buf` meanwhile.
  - Watcher and debug builds fall back to `noc_semaphore_inc`.
- **Fused packets launch the payload write before loading the semaphore fields,** in a fenced load group.

**Measured** (`fabric_exp/dqst_runs.sh`, `dqst_compare.py`, flags 13129088, against the same code with bits 26/28 as
flags, `dqhd_348673408`): stream sanity 10/0, mesh 96/0, and every suite and fused run passed.
- A second variant, "loads first" (`dqstlf_*`), keeps the staging but restores the original fused order: semaphore
  fields loaded before the write, no fences.

| Row (cycles/packet) | before | staged (user) | loads first |
| --- | ---: | ---: | ---: |
| fused, flush on: E/W / N/S / forwarded | 522.8 / 531.2 / 522.7 | **491.2 / 499.1 / 491.0** | 496.7 / 503.8 / 495.8 |
| fused, flush off: E/W / N/S / forwarded | 152.0 / 160.3 / 152.0 | 159.6 / 167.2 / 159.6 | 155.5 / 163.2 / 155.5 |
| standalone atomic: E/W / forwarded | 130.1 / 129.6 | 131.3 / 131.2 | 131.3 / 131.2 |
| mix mean vs before | 0 | +0.84 | +0.42 |
| 4352 mean vs before | 0 | −0.42 (3-hop −1.2, multicast −1.7 to −1.9) | — |
| fused latency 16 B, 1 hop E/W / N/S (ns) | 1413 / 1409 | 1443 / 1436 | 1435 / 1424 |

- **Staging pays where it was meant to.** With nothing left to program after the wait, a flushed fused packet costs
  32 fewer cycles (−6%).
- **The unflushed fused regression has two parts:**
  - **The write-first reorder: about 4.** Restoring the original order brings 159.6 to 155.5 and costs 5 of the
    flushed gain. Mechanism not checked: perhaps E1's header loads contend with the NoC reading the 4352 B payload
    from the same L1, or the atomic simply issues later.
  - **The hand-written atomic path itself: about 3.5,** and the standalone atomic's +1.2 to +1.6 in both variants.
    It has the same register writes as stock, so this is likely code layout or the fences in the atomic case.
    Not checked in the disassembly.
- **Suggested next variant:** stage only when `flush` is set, and take the stock `noc_semaphore_inc` path otherwise,
  with loads first. Unflushed traffic, the benchmark default and `flush = false` producers, would then run the
  original code. Untested.

## 62. Staging only when flush is set: keeps the flushed gain, unflushed back to within ~2

**The user's update** (the §61 suggestion, built as reusable API):
- **New HAL helpers** next to `noc_fast_atomic_increment` in `noc_nonblocking_api.h` (Blackhole, Wormhole, Quasar
  v1/v2/v3): `noc_fast_atomic_increment_set_state` and `noc_fast_atomic_increment_with_state`.
- **New dataflow API** in `dataflow_api.h`: `noc_semaphore_inc_set_state` and `noc_semaphore_inc_with_state`.
- **Blackhole review:** `set_state` is the stock function up to the send: ready wait, mid-address assert, return
  address in dynamic mode, all six registers with the same values. `with_state` holds the dynamic-mode counter, an
  optional ready check, the send and the atomics-acked counter. The API wrapper keeps the stock debug and sanitize
  hooks.
- **Wormhole and Quasar:** additive only and mirror each architecture's stock function. Not compiled or run here.
- **`execute_local_semaphore_inc(addr, incr, flush, rx_channel_id)`** in `fabric_edm_packet_transmission.hpp`, used by
  the standalone atomic, fused and scatter-semaphore cases:
  - with `FABRIC_ERISC_ROUTER_V2` and flush set: `set_state`, then the flush, then `with_state` (no ready check);
  - otherwise: the stock `noc_semaphore_inc`.
  - Without the define, v1 and the speedy path keep the original flush-then-increment. The DRISC branches are
    untouched.
- **v2 defines `FABRIC_ERISC_ROUTER_V2`** just before including the header, and drops its §61 copies. No header
  includes the transmission header indirectly, so the define takes effect.
- **The fused case loads its header fields before the payload write,** in the original order. The hold is
  `FABRIC_2D`-only again.

**Measured** (`fabric_exp/dqsf_runs.sh`, `dqsf_compare.py`, flags 13129088): stream 10/0, mesh 96/0, and every suite
and fused run passed.

| Row (cycles/packet) | before | §61 staged always | §61 loads first | **§62 flush-only** |
| --- | ---: | ---: | ---: | ---: |
| fused, flush on: E/W / N/S / forwarded | 522.8 / 531.2 / 522.7 | 491.2 / 499.1 / 491.0 | 496.7 / 503.8 / 495.8 | **494.7 / 502.7 / 494.0** |
| fused, flush off: E/W / N/S / forwarded | 152.0 / 160.3 / 152.0 | 159.6 / 167.2 / 159.6 | 155.5 / 163.2 / 155.5 | **153.9 / 162.4 / 153.9** |
| standalone atomic: E/W / N/S / forwarded | 130.1 / 138.0 / 129.6 | 131.3 / 139.0 / 131.2 | 131.3 / 139.0 / 131.2 | 130.2 / 139.1 / 129.8 |
| fused latency 16 B, 1 hop E/W / N/S (ns) | 1413 / 1409 | 1443 / 1436 | 1435 / 1424 | 1414 / 1414 |
| 4352 mean vs before | 0 | −0.42 | — | −0.05 |
| mix mean vs before | 0 | +0.84 | +0.42 | +0.25 |

- **Flushed fused: −28 (−5.4%).** Most of §61's −32 is kept.
- **Unflushed fused: +1.9 (E/W, forwarded) and +2.1 (N/S).** That is down from +7.6, and these packets run the stock
  increment. The remainder is probably E1 layout from the larger function. Not checked.
- **Standalone atomics and latency are back at the before values.** §61's −1 to −2 on 3-hop and multicast is gone
  too, so that was layout.
- **Recommendation: keep §62.** The HAL and `dataflow_api.h` additions are public API in shared headers. They need
  their owners' review, and a Wormhole compile, before they leave this branch.

## 63. Deferring the flushed atomic: −55 on flushed fused, +2 to +3 on every E1-bound row

**The user's update** (step 1 of "do work during the flush wait", barrier semantics kept):
- **`execute_local_semaphore_inc(..., bool* pending_atomic_flush)`:**
  - With flush set, it stages the atomic (`noc_semaphore_inc_set_state`) and checks
    `ncrisc_noc_nonposted_writes_flushed` once.
  - If the writes have already landed, the atomic fires at once, so a standalone atomic with nothing in flight pays
    nothing.
  - Otherwise it sets `*pending_atomic_flush` and returns.
- **Callers that pass no pointer keep §62's behavior:** v1, the speedy path, the DRISC.
- **`can_defer_local_atomic_flush()` allows deferral only on:** Blackhole 2D, a single receiver channel, no VC1/VC2, no
  deadlock avoidance, no first-level ack, no UDM, no DRISC offload, fused flush/completion pointer, bit 22, and local
  and downstream writes on E1's own NoC. No other channel can touch the staged `write_at_cmd_buf`.
- **Drain points** (`complete_pending_atomic_flush` = flush wait + send):
  - next packet admitted: right after its transaction ID, before any forward or local NoC operation;
  - idle or blocked iteration: before the completion block, so a partial batch cannot include slot N early;
  - end of the inner loop: before telemetry, context switch, teardown.
- **Reviewed:**
  - Between staging and send, E1 issues no NoC transaction for later packets.
  - The completion block with progress only takes `batch_target` slots when more than that are outstanding, so it never
    includes the newest slot.

**Measured** (`fabric_exp/dqpd_runs.sh`, `dqpd_compare.py`, flags 13129088): stream 10/0, mesh 96/0, and every suite and
fused run passed.

| Row (cycles/packet) | before | §62 | **§63** |
| --- | ---: | ---: | ---: |
| fused, flush on: E/W / N/S / forwarded | 522.8 / 531.2 / 522.7 | 494.7 / 502.7 / 494.0 | **439.9 / 440.4 / 440.2** |
| fused, flush off: E/W / N/S | 152.0 / 160.3 | 153.9 / 162.4 | 156.1 / 164.6 |
| scatter: E/W / N/S | 164.0 / 172.3 | 163.9 / 172.3 | 167.0 / 175.6 |
| forwarded 16 B–1 KiB writes | 111.4–111.5 | 111.4–111.5 | 113.7 |
| 1-hop E/W 4352 write | 130.6 | 130.4 | 132.2 |
| multicast write / scatter | 140.9 / 184.1 | 140.8 / 184.1 | 142.2 / 186.5 |
| 4352 / mix mean vs before | 0 / 0 | −0.05 / +0.25 | +2.03 / +1.31 |

Latency is within noise (±11 ns).

- **Flushed fused: −55 against §62, −83 (−16%) against before.** N/S gains the most (−62): its admission and decode are
  longer, so more of the wait is overlapped.
- **Every E1-bound row pays +2 to +3,** including scatter and forwarded writes that never touch an atomic.
  - E1's hot path only adds a register test (`bnez t1`, the flag stays in a register).
  - But E1 grew from 1610 to 1752 instructions: the flush loop and send are inlined at each drain site, and each atomic
    site gains a "landed yet?" branch.
  - So the cost is most likely layout and spread. Not proven.
- **Trade-off:** one flushed packet per N saves ~55 and costs ~2.5 × N on E1-bound traffic. It wins when a flushed
  signal comes at least every ~20 packets and is a net loss for ops that flush once per op.
  - §59 and §60: `reduce_scatter_minimal_async` flushes per `chunks_per_sync` chunks, `deepseek_moe_reduce_scatter` per
    packet, the standalone-atomic ops per chunk or group.
- **Next steps:**
  1. Shrink the code: one out-of-line drain body. That needs care with §33's non-leaf `kernel_main` register spills,
     so a single inlined site is better.
  2. Measure a realistic mix, 7 writes + 1 flushed fused. That needs the per-segment flush option in the stream yaml.

## 64. Flushed-signal benchmark: the deferral cuts a chunked signal's flush cost by 55–70%

**New test-infra option** (`fabric_exp/streams/port/streams_flush.patch`, made in the streams worktree first and
applied here; both trees build):
- An optional `flush` on a pattern or a stream segment, for `atomic_inc` and `fused_atomic_inc`. Unset keeps today's
  defaults: atomic true, fused false. Existing yamls are unchanged.
- **Host:** parsed, merged, carried through stream expansion and into the traffic parameters, and packed into the top two
  bits of the increment argument (bit 31 = set, bit 30 = value). Argument counts don't change. `atomic_inc_val` must
  fit in 30 bits, which is checked.
- **Kernel:** `NocUnicastAtomicIncFields::build_from_args` decodes and masks the bits. The fused header builders (setup
  and per-packet update) use `flush_or(false)`, the atomic builder `flush_or(true)`.
- Line sync, flow-control credits and the latency kernels are unchanged.
- **Benchmark:** `fabric_exp/streams/flush_mix.yaml`, 16 tests, 1-hop E/W and N/S, 4352 B, flush on/off:
  - fused on every packet;
  - 7 writes + 1 fused (chunk of 8);
  - 15 writes + 1 fused (chunk of 16);
  - 7 writes + 1 standalone atomic.
- The override reaches the device: fused on every packet gives 439.9 with flush against 156.0 without, matching §63's
  forced-flush runs.

**Measured** (`fabric_exp/dqfm_runs.sh`, `dqfm_compare.py`): three router versions, each a v2 + transmission-header pair
swapped in and restored (checked). All 16 tests passed on each.
- **before:** pre-atomic-change, flags 348673408.
- **§62:** flush-only staging.
- **§63:** deferral.

| Row (cycles/packet) | before | §62 | §63 |
| --- | ---: | ---: | ---: |
| E/W chunk 8 fused, flush / no flush | 162.6 / 134.8 | 159.3 / 134.8 | **147.6** / 135.1 |
| E/W chunk 16 fused, flush / no flush | 148.2 / 134.3 | 146.7 / 134.4 | **139.1** / 134.8 |
| E/W chunk 8 atomic, flush / no flush | 140.6 / 132.8 | 141.4 / 132.9 | 141.8 / 133.7 |
| N/S chunk 8 fused, flush / no flush | 171.9 / 135.7 | 168.5 / 135.8 | 166.7 / 136.4 |
| N/S chunk 16 fused, flush / no flush | 151.9 / 135.1 | 150.4 / 135.3 | **141.1** / 135.8 |
| N/S chunk 8 atomic, flush / no flush | 158.9 / 134.6 | 155.7 / 134.6 | 153.3 / 135.3 |
| fused every packet, E/W, flush / no flush | 522.9 / 151.9 | 494.7 / 153.9 | 439.9 / 156.0 |

**Cost of one flushed signal** (flushed − unflushed, × chunk length, cycles):

| Shape | before | §62 | §63 |
| --- | ---: | ---: | ---: |
| E/W chunk 8 fused | 222 | 196 | **100** |
| E/W chunk 16 fused | 222 | 197 | **70** |
| N/S chunk 8 fused | 289 | 262 | 242 |
| N/S chunk 16 fused | 269 | 242 | **85** |
| E/W chunk 8 atomic | 63 | 68 | 65 |
| N/S chunk 8 atomic | 194 | 169 | 144 |
| fused every packet (E/W / N/S) | 371 / 371 | 341 / 340 | 284 / 276 |

- **A signal at the end of a chunk is much cheaper to flush than a flush on every packet.** At the chunk's end most
  earlier writes have drained.
  - Before, it cost 220–290 cycles per signal.
  - With §63 it costs 70–100 (E/W chunks of 8 and 16, N/S chunk 16), a 55–70% cut, because E1 works on the next
    packets while the last write drains.
  - The flushed chunk rows improve 6–9% end to end (E/W chunk 8: 162.6 → 147.6).
- **Unexplained:**
  - **N/S chunk 8 fused barely moves (289 → 242).** Not investigated. A guess: the packet after the signal is often
    blocked, so the idle-iteration drain spins at once.
  - **Standalone atomics after 7 writes:** E/W costs ~65 per signal in every version. N/S drops 194 → 144.
- **Unflushed rows:** §63 costs +0.3 to +0.9 per packet on these chunk shapes, +4 on fused every packet, and +2 to +3
  on the E1-bound non-signal rows of §63 (scatter, forwarded).
- **Net:** in traffic that signals every 8–16 packets, the deferral saves 7–15 cycles/packet on those flows, against
  ~2–3 on E1-bound flows that never signal.

## 65. A slimmer deferral that ended up slower: E1's packet path became a taken branch

**The user's update to §63:**
1. **Always defer:** every flushed atomic is staged and marked pending. The inline "already landed" check and its send
   are gone.
2. **Non-blocking idle drain:** before the completion block, the atomic is sent if its writes have landed. On an
   iteration with no progress and the atomic still pending, completions are skipped.
3. **Drain moved:** the inner-loop-end drain moves to the main-loop exit. Deferral also requires
   `!any_sender_channel_serviced`.
- **Reviewed, correct:**
  - Admission still drains, blocking, before the next packet's NoC operation.
  - Slot N's credit cannot overtake its atomic.
  - On Blackhole the context switch is a no-op on ERISC 1, and on ERISC 0 it uses NoC0 only. So a pending atomic
    across the outer loop only adds latency.

**Measured** (`fabric_exp/dqps_runs.sh`, `dqps_compare.py`, flags 13129088): stream 10/0, mesh 96/0, and every suite
and the flushed-signal yaml passed.

| Row (cycles/packet) | before | §63 | §65 |
| --- | ---: | ---: | ---: |
| 4352 / mix mean vs before | 0 / 0 | +2.03 / +1.31 | **+6.10 / +4.96** |
| 1-hop E/W scatter | 164.0 | 167.0 | 174.1 |
| forwarded 16 B write | 111.4 | 113.7 | 120.6 |
| E/W atomic | 130.1 | 130.5 | 137.9 |
| E/W chunk 8 fused flush: cost per signal | 222 | 100 | 149 |
| N/S chunk 16 fused flush: cost per signal | 269 | 85 | 130 |
| fused every packet, flush (E/W) | 522.9 | 439.9 | 439.0 |
| 3-hop 16 B latency (ns) | 3406 | 3408 | 3436 |

**Why it is slower:**
- **Layout.** E1 grew 1752 → 1815 instructions, and GCC flipped the receive loop: after the packet poll, the
  packet-present path is now a taken branch out of line (`bnez a2, …`), where §63 fell through. Scatter and forwarded
  rows, which never touch an atomic, lose 7 more cycles.
- **Change 1** defers even atomics with nothing in flight, so standalone atomics pay the deferral (+7.4).
- **Change 2** withholds all completions while an atomic is pending on an idle iteration, including the earlier slots
  that are safe to complete. The upstream sender then waits for credits. That fits chunk 8 getting worse (E/W 147.6 →
  154.7, N/S 166.7 → 172.5).

**Recommendation:** keep §63.
- If its code size is still worth chasing, keep the inline "already landed" fast path.
- On idle iterations, complete every slot except the newest rather than none.
- Avoid steering block layout with `[[likely]]` or `__builtin_expect`. They have cost before (§32, §49), and the flip
  here is a layout effect.

## 66. New host; §65 with the fast path and partial idle completions is still ~2 slower than §63

**New host: bh-lb-120-a08u24.** It is shared, so every run goes through the tt-device-mcp broker. The runners
(`dq66*_runs.sh`) do no resets themselves, stream their output and print a heartbeat.
- **Calibration** (v1 at 13129088, whose code hasn't changed):
  - 4352: mean +0.17 cycles/packet against a04u24, largest row change 1.1 (3-hop E/W).
  - mix: +0.01, largest row change 0.3.
  - So a04u24 throughput tables compare directly.
- **Latency does not carry over between hosts:** 1-hop E/W is 5–14 ns lower here, 1-hop N/S 4–11 ns higher, and 3-hop
  50–72 ns higher. Latency is compared within one host only.
- **Links:** `read_eth_speed.py` shows every trained link at 400G, but 7 ports are down:
  - chip 0 eth (10,1), chip 2 eth (10,1), chip 7 eth (10,1);
  - chip 3 eth (6,1);
  - chip 5 eth (6,1) and (7,1).
  The links the 2D tests use are not affected.

**Router sources are now kept in `fabric_exp/variants/`.** A Claude Code restart wiped the session scratchpad and the
A/B copies with it.
- **§63 was rebuilt from the session transcript** (`variants/s63r`):
  - The transmission header is the same as §66's.
  - In the router, every deferral site lands on §63's line numbers.
  - It reproduces a04u24's §63: mix mean +1.31 on both hosts, and the flushed-signal rows within 0.1 (the noisy chunk-8
    atomic rows excepted).

**The user's update to §65** (`variants/s66`):
1. Immediate-send fast path restored: the atomic fires at once when the writes have already landed.
2. On an idle step:
   - the atomic is sent if its writes have landed;
   - otherwise the newest slot is left out of the completion batch (`uncompleted -= pending`) and the earlier slots still
     complete.
3. `[[unlikely]]` removed. The drain stays at the main-loop exit (§65).

**Reviewed:**
- Correct for two-ERISC Blackhole:
  - The pending atomic always belongs to the newest admitted slot, since the next admission drains it first.
  - With progress the batch only takes `batch_target` of more than `batch_target` outstanding slots.
  - On an idle step the newest slot is excluded, and `uncompleted` is at least 1 whenever the atomic is pending, so the
    subtraction cannot underflow.
- **But §65's `!any_sender_channel_serviced` guard is gone.** With one ERISC per link, the ERISC also runs sender
  channels:
  - `notify_worker_of_read_counter_update` issues `noc_inline_dw_write`, which goes through `write_at_cmd_buf`. That is
    the command buffer holding the staged atomic, so the later `noc_semaphore_inc_with_state<…, skip_cmdbuf_chk>` would
    send whatever the inline write left in it.
  - The context switch to base firmware has the same exposure.
  - §63 has the sender-channel hole too, because its atomic stays pending across iterations.
  - **The guard costs nothing here:** §63 with the guard (`variants/s63g`) gives byte-identical `.text` on all 80 ERISC
    ELFs (40 configs, E0 and E1). It should go back in.

**Measured** (`dq66_runs.sh`, `dq66b_runs.sh`, `dq66c_runs.sh`, `dq66d_runs.sh`, `dq66e_runs.sh`; `dq66b_compare.py`,
`dq66c_compare.py`). Every run passed: stream 10/0 and mesh 96/0 for §66 and §63r, and every suite on every variant. §66
ran three times; the runs agree to ±0.01 on the 4352 and mix means.
- §66u: §66 with `[[unlikely]]` back on the admission drain (diagnostic).
- §66d: §66 with the drain back at the end of the inner loop (diagnostic).

| Row (cycles/packet, a08u24) | §63r | §66 | §66u | §66d |
| --- | ---: | ---: | ---: | ---: |
| 4352 / mix / flushed-signal mean vs §63r | 0 / 0 / 0 | **+1.95 / +2.19 / +1.53** | +2.27 / +2.33 / +2.09 | +2.29 / +2.31 / +2.35 |
| forwarded 16 B–1 KiB writes | 113.7 | 117.0 | 117.7 | 117.1 |
| 1-hop scatter E/W / N/S | 167.0 / 175.6 | 170.3 / 180.0 | 171.1 / 179.9 | 170.5 / 182.8 |
| atomic (flushed, nothing in flight) E/W / N/S | 130.4 / 138.5 | 135.2 / 144.7 | 136.0 / 145.7 | 135.3 / 144.6 |
| fused, no flush E/W / N/S | 156.0 / 164.6 | 160.4 / 168.2 | 159.1 / 168.1 | 160.5 / 168.0 |
| fused every packet, flush E/W / N/S | 439.8 / 440.3 | 437.8 / 436.7 | 438.8 / 440.1 | 442.0 / 441.1 |
| E1 instructions (most common config) | 1742 | 1800 | 1792 | 1799 |

**Cost of one flushed signal** (flushed − unflushed × chunk length, cycles):

| Shape | §63r | §66 |
| --- | ---: | ---: |
| E/W chunk 8 / 16 fused | 99 / 70 | 114 / 64 |
| N/S chunk 8 / 16 fused | 242 / 85 | 260 / 107 |
| fused every packet E/W / N/S | 284 / 276 | 277 / 269 |

**Latency** (a08u24, against one §63r run): §66 is 0–20 ns higher. The 3-hop rows move most (+6 to +20).

**Findings:**
- **§66 is ~2 cycles/packet slower than §63 on average.** It recovers about 40% of §65's loss, but rows that never touch
  an atomic still pay +3.3 (forwarded writes, scatter).
- **Neither `[[unlikely]]` nor the drain location explains it:** §66u and §66d measure the same as §66. What remains is
  the idle-step change: the non-blocking send before the completion block, and leaving out the newest slot. E1 grows by
  ~58 instructions, and register allocation changes across the whole function (renames start in the first few
  instructions). So the cost is spread through the code, not on one path.
- **Partial idle completions don't help the flushed shapes.**
  - Only fused-every-packet improves, by 7–8 per signal.
  - N/S chunk 16 costs 107 per signal against §63's 85, and N/S chunk 8 costs 260 against 242.
  - So §64's guess that N/S chunk 8 loses time to a blocking idle drain doesn't hold: not blocking is worse.

**Recommendation:** go back to §63 and restore the guard: `variants/s63g`, byte-identical to §63 on every ERISC here.
The open item is unchanged: §63 costs +2 to +3 on E1-bound rows that never signal (§63).

## 67. Trimming the deferral: the idle drain in the completion `else` arm costs 2.5; nothing beats §63

**The user's update** (`variants/s67`), aimed at §63's +2 to +3 on rows that never signal:
1. **Idle drain moved** into the `else` (no progress) arm of the completion block's `if (progress)`. The aim was to stop
   GCC from duplicating the completion block across a second progress test.
2. **Header: only fused atomics defer.**
   - Fused atomics stage and mark pending with no "landed yet?" check, since the payload write was just issued.
   - Standalone atomics and scatter semaphore increments go back to §62's inline flush.
3. **E/W drain moved:** on routers with one live direction (stateful NoC), the admission drain moves after
   `forward_2d_dispatch`. N/S routers still drain before it.
4. The `!any_sender_channel_serviced` guard is back, and the drain stays at the main-loop exit.

**Reviewed: correct.**
- E1's forwarding uses command buffer 2 (data) and 1 (credit), both posted (`RX_CH_0_FWD_DATA/SYNC_CMD_BUF_ID` = 2/1;
  posted chunk and credit writes). So the forward neither touches the staged atomic in `write_at_cmd_buf` (3) nor the
  non-posted counter the flush waits on.
- The intermesh branch is dead on those routers: `use_2d_single_live_stateful_noc` excludes intermesh and edge routers.
- One semantic difference: on E/W, packet N+1 is now forwarded before packet N's flushed writes are confirmed. Stock
  waits for them first. Ethernet latency makes an inversion practically impossible, but the guarantee is weaker. N/S
  forwarding uses the same posted command buffers 2 and 1.

**Measured.** Every run passed: §67 stream 10/0, mesh 96/0, and every suite on every variant.
- **Baseline: §62 behavior on this host** (`variants/s62x`): §63+guard with `can_defer_local_atomic_flush()` compiled
  to false.
  - It matches a04u24's §62: mix −0.01, 4352 −0.5, the 4352 gap almost all from the host-sensitive 3-hop row.
  - E1 has 1699 instructions in the most common config.
- **§67 ran twice;** the means agree to 0.02.
- **One-change splits:**
  - §67a: §67 with one drain before the forward on every router.
  - §67b: §67 with the idle drain back at §63's place.
  - §67c: the §63+guard router with §67's header.
- Scripts: `dq67_runs.sh`, `dq67s_runs.sh`, `dq67_compare.py`.

| Row (cycles/packet, a08u24) | §62x | §63g | §67 | §67a | §67b | §67c |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4352 / mix / flushed-signal mean vs §62x | 0 / 0 / 0 | +2.05 / +1.05 / −9.40 | +4.87 / +4.15 / −6.71 | +4.32 / +3.74 / −6.72 | **+2.30 / +1.62 / −9.66** | +2.19 / +1.33 / −8.13 |
| forwarded 16 B write | 111.4 | 113.7 | 120.2 | 119.2 | 114.1 | 114.0 |
| 1-hop scatter E/W / N/S | 164.0 / 172.3 | 167.0 / 175.6 | 173.6 / 177.5 | 172.6 / 177.6 | 167.5 / 179.2 | 167.3 / 175.6 |
| atomic, flushed, nothing in flight E/W / N/S | 130.3 / 139.0 | 130.4 / 138.5 | 138.4 / 143.3 | 137.5 / 143.3 | 132.3 / 142.0 | 132.2 / 140.3 |
| E1 instructions (most common config) | 1699 | 1742 | 1764 | 1770 | 1724 | 1734 |

**Cost of one flushed signal** (flushed − unflushed × chunk length, cycles):

| Shape | §62x | §63g | §67 | §67a | §67b | §67c |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E/W chunk 8 / 16 fused | 203 / 199 | 100 / 70 | 134 / 75 | 130 / 75 | 92 / 64 | 101 / 65 |
| N/S chunk 8 / 16 fused | 262 / 244 | 242 / 84 | 256 / 103 | 253 / 104 | 240 / 87 | 242 / 84 |
| E/W / N/S chunk 8 standalone atomic | 89 / 168 | 61 / 110 | 173 / 198 | 168 / 198 | 59 / 189 | 122 / 184 |
| fused every packet E/W / N/S | 341 / 340 | 284 / 276 | 274 / 271 | 277 / 271 | 276 / 269 | 285 / 277 |

The chunk-8 standalone-atomic rows are noisy: the flushed E/W row ranges 138.7–146.5 across §63's three runs.

**Latency** (this host): §67 is +5 to +16 ns over §62x, about the same as §66.

**Findings:**
- **The idle drain in the `else` arm is what made §67 slower.**
  - Moving it back (§67b) takes 4352 from +4.87 to +2.30 and mix from +4.15 to +1.62.
  - It cuts 40 instructions: 1764 → 1724.
  - So that placement grew the code. §63's separate `!progress` test did not, and the jump-threading premise doesn't
    hold here.
  - Over §62, deferral adds 43 instructions in this config (§63g) or 25 (§67b), not the +132 (1610 → 1742) that comes
    from comparing different configs.
- **The E/W post-forward drain doesn't pay:** §67 is +0.5 slower than §67a, which drains before the forward.
- **The header change (§67c) is about neutral in throughput** (+0.14 / +0.28 against §63g) and 8 instructions smaller.
  - But standalone-atomic signals lose the deferral: N/S chunk 8 goes 110 → 184 per signal, E/W 61 → 122.
  - A standalone atomic with nothing in flight costs 1.8–1.9 more.
- **No variant gets the rows that never signal below §63g.**
  - Code size doesn't predict speed: §67b's E1 is the smallest deferral build and is still +0.25 / +0.57 over §63g.
  - The +2 (4352) / +1 (mix) that deferral costs over §62 survives every rearrangement tried (§63, §66, §67, splits),
    so it is not one misplaced block.

**Recommendation:** keep §63 + guard (`variants/s63g`).
- §67b is the only variant that ties it: flushed-signal mean −0.26, fused every packet −8. But it is +0.25 / +0.57 on
  4352 / mix and loses standalone-atomic deferral.
- Next lever for the +2: profile, rather than rearranging blocks. Phase-profile §62x against §63g on the forwarded-write
  row and find where the cycles go.

## 68. Where deferral's +2 goes: one taken jump out of the local-write arm

**Setup** (`fabric_exp/p68_runs.sh`, `p68_compare.py`, `streams/p68_profile.yaml`: the five rows deferral slows most).
§62 behavior (`s62x`) against §63 + guard (`s63g`), flags 13129088, on a08u24. Each variant ran:
- once without profiling;
- with an E1 spin of +20 and +40 cycles per packet (slack test);
- five single-phase level-2 profiles, no per-iteration timing: decode/admit, forward, local delivery, completion, and
  the whole receiver step. Net = (ticks − 4.0 × occurrences) / packets, on the E1 with the most local deliveries (and,
  for the forwarded rows, the E1 with the most forwards).

All 16 runs passed.

**These rows are E1-bound in both builds.** The spin moves them fully: +20 adds ~31 and +40 adds ~56.
1-hop E/W 4352 write is the exception: partly E0-bound, +9 / +12 at +20. So the deferral cost on these rows is E1
time.

| Row (cycles/packet) | s62x | s63g | E1 local delivery s62x → s63g | E1 receiver step s62x → s63g |
| --- | ---: | ---: | ---: | ---: |
| forwarded 16 B write | 111.4 | 113.7 | 47.0 → 49.0 | 123.1 → 128.1 |
| forwarded scatter | 163.8 | 167.0 | 103.1 → 105.1 | 177.5 → 179.5 |
| 1-hop E/W scatter | 164.0 | 167.1 | 103.1 → 105.1 | 179.2 → 180.7 |
| 1-hop E/W 4352 write | 132.7 | 133.3 | 47.1 → 49.1 | 126.3 → 127.8 |
| 1-hop E/W fused, no flush | 153.9 | 156.1 | 90.1 → 90.1 | 167.7 → 170.2 |

The other phases don't move:
- decode/admit 18.0 (29.5 on the forwarding E1);
- forward 60.0;
- completion 15.3–17.7 per packet at 0.25–0.36 batches per packet, within ±0.2.

**Local delivery is +2.0 on every write and scatter row and unchanged on fused.** In the disassembly (most common E/W
E1 config, `variants/<v>/elf/10458260907906636366/`), the send-type jump table and the unicast-write arm are the same in
both builds, instruction for instruction (registers renamed). What changes is how the arm exits:
- **s62x:** the write arm falls through into the post-delivery block (sent-counter increment, buffer index,
  packets-received decrement) that sits right after it.
- **s63g:** that block is placed earlier, at `0xd744`, and the write arm (and the scatter arm) end with
  `j d744 <.L72>`. That is one taken jump per packet.

So the deferral's own per-packet work costs nothing measurable here: the pending test sits outside every phase that
moved, and decode/admit and completion are unchanged. The +2 is block placement. The growth of the atomic arms and
drain sites makes GCC put the shared post-delivery block before the switch instead of after the write arm. This also
explains why §63, §65, §66 and §67 all kept "about +2" while their code sizes differed: any arrangement that leaves that
merge block out of line pays the jump. The fused row's +2.2 is not in local delivery. It shows only in the whole
receiver step, and is not traced.

**Options:**
1. **Accept the +2.** It is layout, not work, and may move with any unrelated change to `kernel_main`, as earlier ±5–10
   swings did.
2. **Make the write arm independent of placement:** do the post-delivery bookkeeping inside the hot arms (write,
   scatter) in source, so neither needs a jump to a shared block. This is a structural change, not a hint, but GCC may
   still merge identical tails.
3. **Shrink what deferral adds inside the switch,** for example one non-inlined drain body, so the function's shape
   returns to §62's. §33's non-leaf `kernel_main` spills make that risky.

Block-layout hints (`[[likely]]`, `__builtin_expect`, `hot`/`cold`) are ruled out: they are compiler-fragile and have
cost before (§32, §49).

**Decision:** the user accepted the +2 for now (option 1).

## 69. Opt-in flush field: §60's candidates mapped to how often they flush and which workloads run them

**Method:** a code read on top of §59/§60, no device runs.
- Flush frequency comes from each op's writer kernel and program factory.
- Workloads come from callers under `models/` (tests excluded), and CI suites from `tests/pipeline_reorg/`.
- The upper bound on the gain is §64's residual flush cost under §63's deferral. An opt-in can at most remove it, and
  only if the router keeps servicing later packets instead of spinning.

**Residual cost of a flush under §63** (§64): every packet ~280 cycles (439.9 against 156.0); a signal every 16 packets
70–85; every 8 packets 100 (E/W) or 240 (N/S); a standalone atomic after 7 writes 65 (E/W) or 144 (N/S).

**`ttnn.reduce_scatter` reaches the same kernels:** its program factory builds
`build_{ring,line}_reduce_scatter_minimal_async_program_artifacts`. It does so unless the input goes to
`reduce_scatter_minimal_direct`: ring or torus, no tuning arguments, ≤ 512 KB per device. That direct path flushes once
per op.

| Op | Flushed signal comes | Upper bound over §63 | Workloads (CI) |
| --- | --- | --- | --- |
| `deepseek_moe_reduce_scatter` | every packet: a scatter + increment per 2-tile packet (writer :238) | ~280 cycles/packet, up to ~2.8× this op's link time | DeepSeek V3 decode MoE: `deepseek_v3/tt/moe.py`, `moe_optimized.py`, `decoder_block/moe_decoder_block_2d.py`, `common/modules/moe/tt_moe_decode.py` (blackhole_e2e, BH multi-card sanity, T3K e2e, device perf) |
| `reduce_scatter_minimal_async` ring/line, and `ttnn.reduce_scatter` above 512 KB | a chunk is 4 packets (`tile_granularity` = 4 × tiles per packet). Ring dims 1–3: every 4 chunks (16 packets), or every chunk (4 packets) when a step has ≤ 4 chunks. Line: every 20 chunks (80 packets). Ring dim 0: every 80 chunks (320 packets) | ring dims 1–3 ~4–5 cycles/packet (3–4%); short steps more (chunk of 4 not measured); line and dim 0 < 1 | broadest reach: `tt_transformers` (`mlp.py` w1/w3, `ccl.py`), `deepseek_v3`, `deepseek_v3_d_p`, `gpt_oss(_d_p)`, `llama3_70b_galaxy`, `minimax_m3`, `gemma4(_d_p, _31b_qb2)`, `blackhole/qwen36`, `qwen38_27b_qb2`, `llama31_8b_qb2`, `tt_dit`, `z_image_turbo`, `t3000/falcon40b` |
| `high_bw_all_gather` | fused flushed signal after a page on granular stripes, plus standalone atomic signals; mux workers share a link, so each flush also waits on the others' writes | not quantified | prefill on BH (blaze suite, bh_sc1/sc4): `deepseek_v3_d_p`, `minimax_m3`, `llama_3p1_8b_d_p` |
| `cq_relay` (dispatch) | per page batch | not quantified | only chips that dispatch reaches over fabric; which BH systems have those was not checked |

**Safe, little to gain** (one flush per op or iteration): `reduce_scatter_minimal_direct` (decode-size
`ttnn.reduce_scatter` on rings), `reduce_to_root`, DeepSeek V3 B1 sampling, the dit ops, `ttnn.all_gather` and the
all-gather variants, broadcast, neighbor pad (`tt_dit` VAE), prefill 1D combine and dispatch, `all_to_all_combine`
(`gpt_oss`, `deepseek_v3`).

**Nothing to gain:** `ttnn.all_reduce` goes to `all_reduce_async`, whose fused writes pass `flush = false` (callers:
`gpt_oss`, `gemma4`, `llama_3p1_8b_d_p`, `deepseek_v3`). The same holds for `rms_allgather` and B1
`d2d_exchange`/`host_io`.

**Must keep the default flush:** DeepSeek prefill `combine_fabric2d` and `dispatch_fabric2d` (§60, H2). Both are part of
`deepseek_v3_d_p` prefill.

**Caveats:**
- These are static reads. The real gain also depends on the op's share of model time and on whether its links are the
  bottleneck.
- Decode-size `deepseek_moe_reduce_scatter` sends few packets per call, so a per-packet saving may show up mostly as
  latency.
- Rows not yet benchmarked:
  - a flushed signal every 4 packets (RS short steps);
  - a flushed scatter on every packet (deepseek_moe).
  Both are expressible with §64's per-segment `flush` option.

## 70. FLUSH_RELAXED opt-in: bypass works on chunked signals once the completion guard goes; per-packet flush is slower

**The user's change (variant `s69`):**
- **Header field:** `NocFlushType` (`FLUSH_NONE` 0, `FLUSH_STRICT` 1, `FLUSH_RELAXED` 2). `flush` becomes `uint8_t` in the
  three atomic headers. The default is STRICT; sizes and offsets are unchanged.
- **Pending state:** `pending_atomic_flush` becomes a single `uint32_t`:
  - 1 means a strict atomic is pending (as in §63);
  - `0x10 | trid` means a relaxed one is pending. The trid is the packet's own slot trid (fused, scatter) or the
    previous slot's (standalone).
- **At admission:** a relaxed atomic is drained unless the incoming packet is a plain unicast write (local delivery, or
  the single-live stateful NoC) and the pending trid is still unacked. The drain, `complete_relaxed_atomic_flush`,
  waits for trids s, s−1, s−2 and s−3, then sends.
- **Completion guard:** no completion credits are returned while a relaxed atomic is pending.
- **Test kernel:** `kActiveFlushMode = FLUSH_RELAXED` makes every flushed atomic the test kernel sends relaxed.

**Diagnostic `s69a`:** `s69` without the completion guard. A plain write may bypass the pending atomic only while its slot's
trid is outside the atomic's 4-trid window, which bounds bypass to 12 packets.

**Runs** (`fabric_exp/dq69_runs.sh` a/b/c, `dq69_compare.py`; broker jobs 178–180):
- `dq69`, `dq69r2`: s69 relaxed.
- `dq69s`: s69 with the test kernel sending STRICT.
- `a8s63g2`: s63g against the new header, so RELAXED is treated as STRICT. It is pooled with the earlier `a8s63*` runs.
- `dq69a`, `dq69ar2`: s69a.

Validation (stream and mesh sanity) passed on both s69 and s69a. The fused rows repeat within ±0.1. The chunk-of-8
atomic rows are bimodal on s63g (EW 138.7–146.5, NS 141.6–153.2).

| flushed row (cycles/packet) | §63g | §69 strict | §69 relaxed | §69a |
| --- | ---: | ---: | ---: | ---: |
| EW fused every packet | 439.8 | 439.4 | 571.4 | 494.2 |
| NS fused every packet | 440.3 | 442.3 | 567.7 | 487.4 |
| EW chunk 8 fused | 147.6 | 148.1 | 152.6 | **140.9** |
| NS chunk 8 fused | 166.7 | 167.7 | 152.9 | **145.9** |
| EW chunk 16 fused | 139.2 | 139.3 | 146.4 | **138.4** |
| NS chunk 16 fused | 141.0 | 141.8 | 149.2 | **140.5** |
| EW chunk 8 atomic | 142.5 (bimodal) | 137.9 | 138.5 | 139.0 |
| NS chunk 8 atomic | 150.1 (bimodal) | 151.3 | 142.0 | 141.2 |

**Cost per flushed signal** (§63g → §69a):
- EW chunk 8 fused: 100 → 44.
- NS chunk 8 fused: 242 → 75.
- EW chunk 16 fused: 70 → 55.
- NS chunk 16 fused: 84 → 76.

**Findings:**
- **The completion guard is s69's regression.**
  - With a signal on every packet, a relaxed atomic is pending after almost every step. Credits then go back only on
    idle steps: +131 per packet.
  - On chunk-16 rows it costs +7 to +8.
  - Removing it (s69a) makes every chunked row beat §63g.
- **Fused on every packet is still +47 to +54 in s69a.** The next packet is fused, so it can't bypass, and the drain
  runs at every admission, as in §63.
  - The likely difference is 4 per-trid status reads against 1 global read (not measured).
  - At admission no write has been issued since packet N, so the global counter is already exact there. Using it
    whenever nothing has bypassed would make this row match §63.
- **Strict packets on the new code** (`dq69s`) match §63g on the flushed rows.
- **Cost on unflushed rows:**
  - E1 grows by 66–100 instructions over s63g (+8 to +20 more in s69a).
  - Mean: 4352 +0.60 / mix +0.72 (s69) and +0.05 / +0.34 (s69a).
  - s69: NS scatter +4.0 (175.6 → 179.6), fused unflushed +1.8, atomics +1.5 to +1.9.
  - s69a recovers NS scatter (175.5). That is layout, as in §68.
- **Latency:** within +15 ns of §63 on this host.

**Review notes:**
1. **The 4-trid window does not guarantee C1 (§60).**
   - Acks from different destinations return out of order, so a write five or more slots back can still be unacked
     after s..s−3 are clean.
   - `reduce_scatter_minimal_async` (a signal every 16 packets over interleaved DRAM banks) is that shape.
   - For producers that signal on every packet (`deepseek_moe_reduce_scatter`), the packet's own trid alone is exact.
   - Pick one of two semantics:
     - own payload only: one trid, exact and cheap;
     - channel-wide: exact tracking. Use the global counter while nothing has bypassed, and per-trid checks back to
       the last drain only after a bypass.
   - Inline writes carry no trid, so the window never covers them.
2. **Scatter + atomic can't carry RELAXED.** The 2-bit chunk encoding uses all four values, so the header builder maps
   RELAXED to STRICT.
   - `deepseek_moe_reduce_scatter`, and `reduce_scatter_minimal_async` ring with 2-tile packets (bf16), both send
     scatter + atomic. So §69's top two candidates can't opt in as is.
   - Free room exists: the high bits of `chunk_count` (max 4), or the unused fourth chunk's encoding when
     `chunk_count` = 3.
3. **The `& 0xF` arithmetic assumes trid = slot mod 16.**
   - `can_defer_local_atomic_flush()` forces 16 trids at offset 0. With fewer than 16 receiver slots, though, trids
     wrap at the slot count (for example `TT_FABRIC_EXP_MAX_RECEIVER_SLOTS=8`).
   - Then `(trid − 1) & 0xF` and the window point at unused trids, and the real previous writes are skipped.
   - Fix: add a `static_assert` (receiver slots ≥ 16) to `can_defer_local_atomic_flush()`, or wrap by
     min(slots, 16).
4. **`kActiveFlushMode = FLUSH_RELAXED` is global in the test kernel**, so validation yamls' atomics now go out relaxed
   as well. Better: default STRICT and a per-pattern `flush: relaxed`, using a 2-bit value in the increment word's
   top bits.
5. **Readers of the header field outside the ERISC v2 path** treat any nonzero value as flush, so RELAXED degrades to
   STRICT there. That covers `api_common.h` scatter, the relay extension and DRISC. `linear/addrgen_api.h` and
   `mesh/api.h` take `bool flush`, so they cannot request RELAXED.

## 71. Exact relaxed flush: walking 16 trids costs too many status reads; a two-read check works

**The user's change (variant `s71`):**
- **Before any write bypasses** (`pending < 0x100`), the drain is the single global ack check, which is exact because
  nothing has been issued since packet N.
- **After a write bypasses,** an acked-trid pointer (bits 4–7) walks all 16 trids:
  - before a bypassing packet reuses trid t, E1 waits for t's older write and steps past it;
  - each bypass step advances the pointer over at most 5 more acked trids;
  - the atomic fires when the pointer reaches s and s is acked.
- **Completion guard removed.** Bypass is bounded to trids outside s..s−3, as in s69a.
- **`can_defer_local_atomic_flush()` now requires** 16 receiver slots and 16 trids.
- **Scatter + atomic can carry RELAXED** in bit 6 of `chunk_encoding`, the unused fourth chunk when `chunk_count` = 3.
  Both builders start from 0, and the router reads it at the seminc chunk. No test sends fused scatter, so this is
  untested on device.

**Diagnostic `s71b`:** s71 with the walk replaced by a two-read test at the first bypass.
- At that point nothing has been issued since N, so global outstanding (SW `noc_nonposted_writes_acked` − HW
  `NIU_MST_WR_ACK_RECEIVED`; BH writes always update the SW counter, issue #28758) is read first, then trid s's
  outstanding count. This repeats until the two are equal.
- Counts only fall, so equality proves every older write is on trid s. From then on only trid s is watched:
  `pending = 0x100 | s<<4 | s`, so the drain waits on s alone.
- Cost: 2 reads once, then 1 read per bypassing packet. The check also covers writes that carry no trid (inline
  writes).

**New row:** `fabric_exp/streams/flush_mix4.yaml`, a chunk of 4 (3 writes + 1 flushed fused). This is the shape of
reduce-scatter's short ring steps.

**Runs** (`fabric_exp/dq71_runs.sh` a/b/c, `dq71_compare.py`; broker jobs 181–183):
- s71 relaxed `dq71`, `dq71r2`; s71 strict `dq71s`;
- s71b `dq71b`, `dq71br2`;
- s63g `a8s63g3`, `a8s63g4`; s69a chunk of 4 `dq69a`.

Validation (stream and mesh sanity) passed on s71 and s71b. The s63g chunk-of-4 E/W row is bimodal (204.8 / 180.7).

| flushed row (cycles/packet) | §63g | §69a (4-trid, inexact) | §71 (walk) | §71b (two reads) |
| --- | ---: | ---: | ---: | ---: |
| EW fused every packet | 439.8 | 494.2 | 444.0 | 442.0 |
| NS fused every packet | 440.3 | 487.4 | 438.9 | 437.4 |
| EW chunk 4 fused | 192.8 | 162.8 | 223.5 | **162.1** |
| NS chunk 4 fused | 210.8 | 169.1 | 230.5 | **189.2** |
| EW chunk 8 fused | 147.6 | 140.9 | 170.4 | **143.9** |
| NS chunk 8 fused | 166.7 | 145.9 | 175.5 | **157.4** |
| EW chunk 16 fused | 139.2 | 138.4 | 141.8 | **137.2** |
| NS chunk 16 fused | 141.0 | 140.5 | 148.7 | **139.2** |
| EW chunk 8 atomic | 143.3 | 139.0 | 165.4 | **140.7** |
| NS chunk 8 atomic | 150.7 | 141.2 | 171.8 | **146.0** |

**Cost per flushed signal** (§63g → §71b):
- E/W chunk 4: 226 → 102.
- N/S chunk 4: 290 → 203.
- E/W chunk 8: 100 → 70.
- N/S chunk 8: 242 → 167.
- E/W chunk 16: 70 → 36.
- N/S chunk 16: 84 → 53.

**Findings:**
- **s71's walk is correct but slow.** I traced the invariant: the pointer stays at or ahead of every reused trid, so
  only older writes are checked.
  - Each NoC status read costs about 15 cycles: s69a's three extra reads per drain cost +54.
  - s71 does up to ~8 reads per bypassing packet, so every chunked row ends up worse than §63g.
- **The global check before any bypass fixes fused-every-packet** (444 / 439 against s69a's 494 / 487).
- **s71b keeps exactness for 2 + 1 reads.**
  - It beats §63g on every chunked row and matches it on fused-every-packet.
  - It is within ~3 of s69a on E/W.
  - On N/S it trails s69a by 11 to 20. The first bypass waits until the older writes are acked, which is the price
    of exactness; N/S's older writes ack late (the same effect as §64's N/S chunk-8 anomaly).
- **Unflushed rows:**
  - s71b: 4352 −0.17, mix −0.05 against §63g, although E1 grows by 134–175 instructions (layout luck, §68).
  - s71: −0.07 / +0.21.
- **This config has 16 receiver slots:** deferral stays on under the new `== 16` condition.
  - The condition is stricter than needed. Trids run sequentially mod 16 for any slot count ≥ 16 (power of two:
    idx & 15; otherwise the counter).
  - BH mesh's first slot option is {32, 32}. A config that gets 32 slots would silently lose all deferral, §63's
    included. `>= NUM_TRANSACTION_IDS` is enough.
- **The bypass bound s..s−3** only needs to keep trid s unused once tracking is exact (t ≠ s).

## 72. Where the current build stands, and the E0/E1 split per packet

**Setup** (`fabric_exp/p72_runs.sh`, `p72_compare.py`; broker job 184, all 26 runs passed):
- The tree as it stands: the user's s71, flags 13129088, a08u24. On rows without a flush it is within ±0.2 of §63g
  and s71b (§71).
- Two yamls:
  - `streams/p72_profile.yaml`: 1-hop E/W and N/S write and scatter; E/W unflushed fused, atomic and 16 B write;
    forwarded 4352 B and 16 B writes;
  - `exp_2d_4352.yaml`: adds bidirectional, 3-hop and multicast.
- Per yaml:
  - a run with no profiling;
  - +20 spin on E0 and on E1 (a fully bound ERISC moves ~+27 to +31 here);
  - level-2 single-phase profiles of every E0 and E1 phase. Net = (ticks − 4.0 × count) / packets.
- The profiled phases run ~5–10% long: a multicast forwarder's phases add to 153 against a 140.8 interval. Use them
  for attribution and the slack test for which ERISC binds.

**Bandwidth** (payload GB/s, cycles/packet). "main" is §44's stock router (a04u24; throughput carries across hosts
within ±0.2). Wire ceiling: 46.86 GB/s (125.4).

| Traffic | main | now | vs main | E0 / E1 slack response (+20) | binds |
| --- | ---: | ---: | ---: | --- | --- |
| 1-hop write E/W | 26.5 (222) | **44.2** (132.9) | +67% | +27.0 / +13.1 | E0 (E1 close) |
| 1-hop write N/S | 26.5 (222) | **44.1** (133.3) | +66% | +26.6 / +19.2 | both |
| 1-hop bidirectional, per direction | 26.5 (221) | **42.2** (139.1) | +59% | +26.7 / +7.2 | E0 |
| 1-hop scatter E/W / N/S | 26.6 (221) | **35.1 / 33.6** (167.6 / 174.8) | +32% / +26% | 0 / +31 | E1 |
| 3-hop write | 26.6 (221) | **39.5** (148.8) | +49% | +14.7 / +20.4 | both |
| multicast write | 32.3 (182) | **41.7** (140.8) | +29% | +23.5 / +22.8 | both |
| multicast scatter | 27.5 (214) | **31.6** (186.0) | +15% | 0 / +28 | E1 |
| forwarded write through a router | 31.3 (188) | **46.7** (125.7) | +49% | +16 / +20 | at the wire |
| fused write + atomic, no flush | 26.6 (221) | **37.3** (157.5) | +40% | +2 / +31 | E1 |
| atomic_inc (Mpkt/s) | 3.6 (374) | **10.26** (131.6) | ×2.8 | +28 / +31 | both |
| E/W 16 B write | | 128.1 | | +33 / +18 | E0 |

- **ERISC-only now matches DRISC offload v4 (§44)** on multicast write (140.8 against 141) and comes within 3% on 3-hop
  (148.8 against 144), without the offload's +27–35% latency.
- **Flushed traffic** (§63, §71): flush on every packet runs at 440, against 523 before the atomic work. Chunked signals
  with s71b cost 36–203 cycles per signal, against 70–290 with §63.

**E0 (TX) per packet:**

| Row | send | credit returns (frames/packet) | connection checks | all channel visits (visits/packet) | rest of visits |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1-hop E/W write | 50.2 | 7.6 (0.22) | 0.1 | 129.6 (4.24) | ~72 |
| 1-hop bidirectional | 51.3 | 8.1 (0.22) | 0.1 | 130.3 (4.25) | ~71 |
| 3-hop (busiest E0) | 47.2 | 8.4 (0.23) | 0.2 | 147.9 (5.00) | ~92 |
| multicast | 47.8 | 15.3 (0.44) | 5.9 | 140.2 (3.04) | ~71 |
| forwarded write | 48.8 | 15.5 (0.44) | 0.1 | 126.6 (2.57) | ~62 |

- The TX-queue spin is 0 everywhere.
- **E0's largest item is visiting channels, not sending.** About 72 cycles per packet go to the worker channel's poll
  and bookkeeping plus three idle forwarding channels (~1.06 loop iterations of 4 visits per packet). The send itself
  is 50.
- On 1-hop writes E0 is ~129 against a 132.9 interval: it binds.

**E1 (RX) per packet:**

| Row | decode/admit | forward | local delivery | completion (batches/packet) |
| --- | ---: | ---: | ---: | ---: |
| 1-hop E/W write | 20.1 | | 46.1 | 15.4 (0.26) |
| 1-hop N/S write | 27.4 | | 50.1 | 15.3 (0.25) |
| 1-hop E/W scatter | 20.0 | | 100.1 | 15.6 (0.26) |
| 1-hop N/S scatter | 27.1 | | 110.1 | 15.4 (0.25) |
| fused, no flush | 20.0 | | 89.1 | 15.7 (0.27) |
| atomic | 20.0 | | 64.2 | 15.4 (0.26) |
| 3-hop middle router | 30.3 | 60.0 | | 15.6 (0.26) |
| multicast, forwarding + local | 30.7 | 60.1 | 46.3 | 16.0 (0.27) |

- E1 has a further ~35–40 untimed per packet: poll, post-admit, retire, completion bookkeeping (§55).
- On 1-hop writes E1 is ~115–120 (slack ~15).
- **Local delivery decides every E1-bound row:** scatter 100–110, fused 89 and atomic 64, against 46 for a plain
  write.
- **Forwarding costs E1 ~90** (decode 30 + forward 60).

## 73. Where E0's ~70 cycles of channel-visit overhead go

**Method** (`fabric_exp/p73_runs.sh`, `p73_compare.py`; broker jobs 185–187, 33 runs, all passed). This mirrors §55
for E0.
- **Variant `e0seg`:** the tree's router (the user's s71) with timers around each piece of E0's loop. It reuses phase
  IDs that E0 never closes in this config (job 184 counted every close: the RX phases and the TX-queue spins were 0).
- **Profiler copy:** counts occurrences only for the timed phase, so a timed segment carries no other phase's counter
  update.
- **Runs:** level 2, one mask per run, no per-iteration timing. Net = (ticks − 4.0 × occurrences) / packets.
- **Unprofiled check:** `e0seg` builds to s71's instructions on all 80 router configs (same registers, addresses
  shifted). It ran 133.7 on the 1-hop E/W write against §72's 133.6, and every other row matched within 0.1.
- **Disassembly:** the 1-hop E/W sender's E0 (eth channel 11, direction 1) in the unprofiled build.

**1-hop E/W write, E0 per packet** (N/S within 0.3 on every line; 16 B, atomic and fused rows within ±2):

| Segment | cycles/packet | passes/packet | per pass | instructions (sending pass) | What runs |
| --- | ---: | ---: | ---: | ---: | --- |
| loop top | 9.1 | 1.13 | 8.0 | 8 | issue the 4 grouped free-slot stream reads |
| worker: packet + credit test | 17.6 | 1.12 | ~15.7 | 7 | wait for the grouped reads (the 0x1FFFF mask is reloaded from the stack and used at once); slot and receiver-credit test |
| worker: pre-send | 29.0 | 1.00 | 29.0 | 10 | slot address from the stack → dependent L1 load of the header's payload size; TX-queue command + status reads (BH-55); `did_something` and `src_ch_id` byte stores; busy test |
| worker: send | 50.2 | 1.00 | 50.2 | 35 | as §72 |
| worker: completion test | 5.7 | 1.13 | ~5.0 | 1 | branch on the completion delta read in the send |
| worker: credit returns | 7.6 | 0.22 | 34.2 | ~26 | as §72 |
| worker: connection check | 14.8 | 1.15 | ~13.0 | 6 | teardown semaphore: stack load → dependent L1 load, used at once |
| forwarding channels 1–3 | 18.5 | 1.16 | 16.0 | 11 | three idle tests, 4 taken branches; channel 3's sent/processed counts are on the stack (out-of-line path) |
| back-edge + outer loop | | | ~2–3 | 2 + ~25/32 | |

- **Cross-checks with one timer:**
  - The whole worker visit is 116.3 per packet; its timed parts add to 124.9 (+7%).
  - The iteration body is 146.1 (127.4 per pass); loop top + worker visit + forwarding channels add to 143.9.
  - Splitting the visit adds ~2 cycles per timer boundary that the calibration misses. Read the parts as shares.
- **Passes per packet:** unprofiled, E0 makes ~1.0–1.07 passes per packet. Profiling slows E1 as well, so the profiled
  runs show 0.12–0.16 non-sending passes per packet: ~0.07 blocked on receiver credit, the rest with no packet yet.
  That inflates the per-pass items by ~5–10% per packet. The whole body per pass (127.4) plus the back-edge is
  consistent with the unprofiled 133.7.
- **The "~72" of §72** is 69.4 in this build's own visit run: 127.3 − send 50.2 − credit returns 7.6 − connection
  body 0.1. Per sending pass it splits:

  | Item | per pass | share |
  | --- | ---: | ---: |
  | pre-send | 29 | 37% |
  | packet + credit test | ~16 | 20% |
  | forwarding channels | ~16 | 20% |
  | connection check | ~13 | 17% |
  | completion test | ~5 | 6% |

  That is ~79 per pass against ~70 measured whole, the boundary cost again.

**What it shows:**
- **The overhead is the worker channel's own per-packet work, not idle polling.**
  - The worker's test, pre-send, connection check and completion test are ~63 per pass.
  - The three idle forwarding channels are ~16.
- **Two loads behind a stack load, each used at once, make up ~40 of it:**
  - the header's payload size (pre-send);
  - the teardown semaphore (connection check).

  The TX-queue busy read in the pre-send overlaps the size load. The mask reload in the packet test waits behind the
  four grouped stream reads (§17: a stack load issued with reads outstanding waits for them).
- **The connection check runs every pass for an event that happens once per connection.**
  - It is ~13 per packet on every router that has a worker channel.
  - v1's bit 3 (`EXP_TX_LAZY_CONN_CHECK`, every 16th visit) is not implemented in v2.
  - Moving the teardown check to the outer loop (once per 32 passes) is ruled out (user, 2026-10-23). The worker's
    close would wait up to 32 passes longer, which adds to every op's cost. The check stays on every pass. The only
    way left at its ~13 is to hide the flag's load latency, without changing when teardown is seen.
- **Forwarding routers pay for the idle worker channel.**
  - The 3-hop middle router's E0 (busiest) spends 51.9 per packet in the worker visit: 1.48 passes, ~35 per pass
    (test ~14, connection ~13, completion test ~3). Its forwarding channels take 110.1 (sends included), against a
    148.7 interval.
  - On the multicast forwarder the worker visit is 50.2 per packet. There the connection body itself runs on ~0.45
    passes per packet (6.3), which is not investigated.
- **Ceiling:** on the 1-hop E/W write E1's slack is ~13 (§72). A cut on E0 larger than that shows up there only once
  E1 also gets faster. Bidirectional and 16 B writes have more E1 slack.
- **Ruled out:** cutting the stack reloads (slot address, mask, channel 3's counts) means steering register allocation,
  which the no-compiler-fragile rule excludes.

## 74. Two-slot exact check (the user's s74): the best flushed-signal build so far, N/S included

**The change** (variant `s74`, router + transmission; header, `api_common.h` and test kernel as s71):
- **At the first bypassing write:**
  - Global outstanding (SW `noc_nonposted_writes_acked` − HW `NIU_MST_WR_ACK_RECEIVED`), trid_s and
    trid_p = (trid_s − 1) & 0xF are read in that order, in one fenced group.
  - The read repeats until g == s + p. That proves every other trid is idle, whichever of s and p acks first.
  - The pending state keeps only the non-zero one(s): `0x100 | b<<4 | a`.
- **Later bypassing writes** check a, then b, narrowing to `b|b` if a acks first. The drain waits on a, then b.
- **Standalone atomic:** last write trid = `transaction_id − 1`, the previous slot, since the atomic's own slot carries
  no write.
- **Bypass bound** `((trid_s − trid) & 0xF) > 1` protects s and p; up to 14 writes bypass.
- **Slot condition:** `can_defer` takes `RECEIVER_NUM_BUFFERS_ARRAY[0] >= NUM_TRANSACTION_IDS`.

**Review:** correct, no changes needed.
- **The proof:** counts only fall and nothing is issued between the atomic and the first bypass (a non-bypassable
  packet or the end-of-loop drain clears the pending state first). So g(t1) ≥ s(t1) + p(t1) ≥ s_out(t2) + p_out(t3),
  and equality forces every other trid to 0 at t1.
- **The counter pair** is the one the production strict flush (`ncrisc_noc_nonposted_writes_flushed`) relies on.
- **A non-trid write** only makes the loop wait longer; it cannot fire the atomic early.
- **Narrowing to `b|b`** moves the bypass window to b, b−1. That is safe: b (s−1) was inside the old window, so no
  bypassing write landed on it, and writes on s are newer than the atomic.
- **Untested here:** the 32-slot case (this mesh has 16 slots) and fused scatter with RELAXED (no test sends it).

**Runs** (`fabric_exp/dq74_runs.sh` a/b, `dq74_compare.py`; broker jobs 188–189, all passed):
- stream sanity 10/10, mesh sanity 96/96;
- 4352, mix, flushmix, flushmix4, each twice;
- a same-day s71b control on flushmix / flushmix4 (`dq71br3`).

s74's two runs agree within 0.1 on 4352. The noisy rows are EW_MixCmd (excluded), E/W chunk 4 (156.8 / 154.4), and
s71b's N/S chunk-8 atomic, which is bimodal (140.2 / 151.7 / 151.7).

**Flushed rows** (cycles/packet; s71b as §71's mean / today's control):

| Row | §63g | s69a (inexact) | s71b | s74 |
| --- | ---: | ---: | ---: | ---: |
| E/W fused, flush every packet | 439.8 | 494.2 | 442.0 / 442.1 | 441.2 |
| E/W chunk 4 | 192.8 | 162.8 | 162.1 / 160.9 | **155.6** |
| E/W chunk 8 fused | 147.6 | 140.9 | 143.9 / 142.9 | **139.4** |
| E/W chunk 16 fused | 139.2 | 138.4 | 137.2 / 137.4 | **136.7** |
| E/W chunk 8 atomic | 143.3 | 139.0 | 140.7 / 140.9 | **136.4** |
| N/S fused, flush every packet | 440.3 | 487.4 | 437.4 / 437.4 | 437.9 |
| N/S chunk 4 | 210.8 | 169.1 | 189.2 / 186.2 | **159.8** |
| N/S chunk 8 fused | 166.7 | 145.9 | 157.4 / 155.7 | **143.6** |
| N/S chunk 16 fused | 141.0 | 140.5 | 139.2 / 139.2 | **138.6** |
| N/S chunk 8 atomic | 150.7 | 141.2 | 146.0 / 151.7 | **138.8** |

Cost per flushed signal ((flushed − unflushed) × chunk):

| Signal | §63g | s71b (today) | s74 |
| --- | ---: | ---: | ---: |
| E/W chunk 4 / 8 fused / 16 fused / 8 atomic | 226 / 100 / 70 / 76 | 97 / 62 / 38 / 57 | **78 / 36 / 34 / 24** |
| N/S chunk 4 / 8 fused / 16 fused / 8 atomic | 290 / 242 / 84 / 124 | 191 / 153 / 54 / 131 | **87 / 59 / 48 / 29** |

- **s74 is the fastest variant on every chunked row**, ahead even of the inexact s69a. Flushing every packet is
  unchanged (~440).
- **The gain is mostly N/S**: chunk 4 186 → 160, chunk 8 fused 156 → 144. §71 traced s71b's N/S deficit to the first
  bypass waiting for packet s−1's ack (~110 cycles). Counting trid_p removes that wait.
- **Unflushed rows are also ~1–2 faster than s71:**
  - fused 157.5 → 155.2, atomic 131.6 → 130.1, scatter 167.5 → 165.3, forwarded small writes 114.3 → 112.0;
  - N/S scatter is 1.3 slower;
  - 4352 suite −0.8 on average.

  These rows never take the changed path, so this is code layout, not the change. §62x (no deferral) is still
  2.0 / 1.1 below §63g on 4352 / mix.
- **E1 code size:** 1875–1986 instructions; s71 is 1911–1991 and s71b 1881–1952.
- **Latency** was not rerun. The unflushed path is unchanged.

## 75. Survey: fused scatter (scatter write + atomic increment)

**Method:** a code read, no device runs.
- It starts from §59's call sites and adds the fabric API helpers (`fabric_*_fused_scatter_write_atomic_inc*` in
  `linear/api.h` and `mesh/api.h`) and their callers.
- It also covers the router's scatter path in `fabric_edm_packet_transmission.hpp` and the bypass rule in s74
  (§74).

**Shape:**
- Every fused scatter has exactly 2 write chunks plus the increment
  (`NOC_SCATTER_WRITE_ATOMIC_INC_FUSED_WRITE_CHUNKS = 2`), so `chunk_count` = 3.
- The RELAXED bit (bit 6 of `chunk_encoding`, §71) therefore fits on every producer.
- On the ERISC path every scatter chunk write carries the slot's trid, as a plain write does.

**Producers outside tests:** 4 ops, 6 sites. All flush (STRICT); none passes `false`.

| Op | Site | When it sends a fused scatter | How often | Workloads |
| --- | --- | --- | --- | --- |
| `deepseek_moe_reduce_scatter` | writer :151 (state), :238 | every data packet: 2 tiles to 2 intermediate pages + increment of the op semaphore, 1 hop | every packet | DeepSeek V3 decode MoE (§69) |
| `reduce_scatter_minimal_async`, ring writer | :226 (state), :340 | a chunk's last packet when it holds 2 tiles (bf16; bfp8 packs 4 and falls back to a standalone increment, 1 tile goes as fused unicast). Only in the tiled intermediate layout: ring with scatter dim 0, or a caller-supplied intermediate that doesn't match the chunk-paged spec | dim 0: every 80 chunks (320 packets). Most model callers pass no intermediate, so ring dims 1–3 use the chunk-paged layout (plain writes + fused unicast, no scatter) | ring dim 0 users only |
| `ring_attention_all_gather_async`, neighbor-halo writer | :234 | the last packet of each (input, batch-head) when its 2 pages are not adjacent | once per batch-head | `ring_joint_sdpa` with a sliding window: `gemma4_d_p` ring prefill, `gpt_oss_d_p` dense SP, `minimax_m3` |
| DeepSeek V3 B1 sampling | `sampling.hpp` :1505 | top-K scores and indices to two slots + increment, connection opened and closed around it | once per decode step | DeepSeek V3 B1 |

**What RELAXED can do for them under s74:** nothing yet.
- s74 lets only a plain unicast write (`NOC_UNICAST_WRITE`) pass a pending relaxed atomic. Any other packet drains it
  first.
- **`deepseek_moe_reduce_scatter`:** the next packet is always another fused scatter, so each one drains the previous
  atomic. That is §63's strict deferral.
- **The reduce-scatter tiled layout:** its data packets are pure scatter writes, which drain at the first data packet
  of the next chunk.
- **Halo and sampling** flush once per batch-head or per op; there is nothing to gain.

**What would change that (untested):**
1. **Let pure scatter writes bypass.** Their chunk writes are trid-tracked like a plain write, so s74's two-slot check
   covers them unchanged.
   - Admission must read the scatter's chunk encoding to exclude fused scatters. That is one more header load, only
     while an atomic is pending.
   - A fused packet's own increment would overwrite the pending state, so fused packets must keep draining.
   - Only the rare reduce-scatter tiled layout benefits.
2. **Per-packet producers (`deepseek_moe`) need fused packets to bypass.**
   - That means more than one atomic pending, fired in order as their trids ack (C2).
   - Alternatively, "own payload only" semantics for those packets (§70 note 1), which needs no queue.
   - Both are design changes beyond s74.

**Data gap:** no test sends a fused scatter, so its cost is unmeasured, flushed or not.
- Expectation: §74's flush-every-packet cost (~285 over unflushed) plus scatter's dearer delivery on E1 (100–110 against
  46 for a write, §72).
- Measuring it needs a fused-scatter traffic type in the stream test, built in the streams worktree first, plus rows for
  every packet and for a chunk ending in a fused scatter.

## 76. 1D on the current tree: bandwidth vs main, which ERISC binds, per-packet split

**Setup** (`fabric_exp/p76_runs.sh` v/a/b, `p76_compare.py`; broker jobs 190–192, all passed):
- The tree at checkpoint `5ef745cb2a0`: v2 + s74, flags 13129088, keepalive 1000, telemetry off.
- `fabric_exp/exp_1d_4352.yaml` (4352 B, the 1D default payload as in 2D):
  - Linear 1-hop E/W, N/S, bidirectional, and 3-hop (mirrors of the 2D rows);
  - `exp_base.yaml`'s linear and unidirectional multicast, half ring and neighbor exchange;
  - write + scatter for every row.
- "main" is the same-tree stock router (v1 kernel, no flags) at the firmware keepalive, as in §44.
- **1D validation on v2:** `test_fabric_sanity_at_least_2x2_mesh.yaml` (Linear, Ring, Mesh), 181/181.
- **Channel depths,** from the router builds' template symbols:

  | Topology | worker slots | forwarding slots | receiver slots | trids |
  | --- | ---: | ---: | ---: | ---: |
  | Linear | 16 | 16 | 32 | 16 |
  | Ring (deadlock avoidance on) | 16 | 16 | 32 | 8 |
  | Neighbor exchange (speedy, one sender channel) | 32 | | 32 | 16 |
  | 2D | 17 | 16 × 3 | 16 | 16 |

**Bandwidth** (cycles/packet, payload GB/s), with the +20 spin response on E0 / E1:

| Row | main | main + keepalive | current | vs main | E0 / E1 response | binds |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| 1-hop write E/W (N/S identical) | 187.6 (31.3) | 168.4 | **137.6 (42.7)** | +36% | +9.7 / +8.5 | neither: worker credit loop |
| 1-hop scatter | 194.4 (30.2) | 194.5 | **167.1 (35.2)** | +16% | 0 / +31.5 | E1 |
| bidirectional write | 168.0 (35.0) | 168.0 | **146.9 (40.0)** | +14% | +5.1 / +3.4 | neither |
| 3-hop write | 187.6 (31.3) | 166.6 | **146.8 (40.0)** | +28% | +7.8 / +21.2 | E1 |
| linear multicast write / scatter | 163.4 / 220.8 | same | **149.2 / 196.3** | +10% / +13% | +5.5 / +20.8; 0 / +32 | E1 |
| unidirectional multicast write / scatter | 206.2 / 273.5 | same | **196.4 / 255.3** | +5% / +7% | −2 / +22.5; 0 / +32 | E1 |
| half-ring multicast write / scatter | 249.5 / 299.2 | 247.9 / 297.5 | **229.9 / 284.2** | +9% / +5% | +16.6 / +28.4; +11.6 / +25.8 | both |
| neighbor exchange write / scatter | 124.8 / 153.9 | same | **125.0 / 146.0** | 0% / +5% | +29.9 / +1.3; +10 / +35 | E0 (at the wire) / E1 |

**E0 per packet:**
- 1-hop write: send 47.6, credit returns 6.2 (0.19/packet), visits 121.8 at 2.31 visits per packet. That is two
  channels against 2D's four, so the overhead is ~68 here against 2D's ~70.
- Half ring: credit returns 27.9 at 1.7 per packet. The ring returns credits per packet.
- Unidirectional multicast forwarder: the connection body runs 1.5 times per packet (18.3). It is a router with no
  worker connected, so `check_worker_connections` runs on every pass. E0 does not bind there.
- Neighbor exchange (speedy path): send 61.4, visits 130.1.

**E1 per packet** (1-hop receiver / forwarder):

| Row | decode/admit | forward | local | completion |
| --- | ---: | ---: | ---: | ---: |
| 1-hop write | 15.2 | | 47.1 | 15.5 (0.27) |
| 1-hop scatter | 15.1 | | 100.0 | 15.2 |
| 3-hop forwarder | 28.5 | 66.0 | | 15.6 |
| linear / unidirectional multicast forwarder | 29.0 | 58.0 | 41.2 | 15.8 |
| half-ring forwarder | 27.4 | 75.0 | 49.2 | 20.0 (1.0) |

The 1-hop decode is 15 against 2D's 20 (E/W) and 27 (N/S).

**What it shows:**
- **1-hop writes are bound by the worker credit loop, not by an ERISC.**
  - E0 needs ~122 per packet and E1 ~129, against a 137.6 interval. +20 on either ERISC adds only ~9.
  - E0 is never blocked on receiver credit or the TX queue. Its extra 0.2 passes per packet find the worker channel
    empty.
  - A worker slot frees only when the far E1's completion returns, so 16 slots cap the rate at about one 4 KiB
    round trip (~2,300 cycles) / 16 ≈ 140.
  - 2D escapes this with 17 worker slots, because the Mesh allocator gives every spare slot to the worker channel.
  - Linear takes the first option that fits, {16 sender, 32 receiver} (64 slots of 80–81). Nothing gives it the ~17
    spare slots. The §27 `TT_FABRIC_EXP_WORKER_SLOTS` knob sits in the same Mesh-only branch.
  - Expected with ≥ 24 worker slots: 1-hop at E1's ~129 (45 GB/s). Untested.
- **Every forwarded, multicast and scatter row is E1-bound,** with 2D's costs. Local write 47, scatter 100, forward
  58–66 (75 on the ring), decode 28–29 on forwarders.
- **The ring costs more per packet:** forward 75, completion every packet (20, no batching), credit returns 1.7 per
  packet on E0. Both ERISCs are near their limit.
- **Neighbor exchange is at the wire** (125.0); its scatter row is E1-bound (+5% over main).
- **None of the flushed-atomic work reaches 1D.** `can_defer_local_atomic_flush()` is compiled only under
  `FABRIC_2D`, so 1D flushed atomics pay the pre-§63 cost. The s74 `>=` slot condition is not exercised by 1D
  either.

## 77. The user's s77: compile-time stream ids win; dropping the sender slot table costs up to 21 on E0

**Change (s77, the user's):** three parts on top of the committed tree (s74c, 5ef745cb2a0).
- `StaticSizedSenderEthChannel` drops the `SlotAddresses` table and buffer index. It keeps only the cached slot
  address, first / last slot and stride, and advances by stride everywhere (speedy path included). The E0 stack frame
  shrinks 304–320 bytes on every 2D config.
- `sender_channel_try_send` and both sender steps take the free-slots stream id at compile time
  (`increment_local_update_ptr_val<id>`, `get_ptr_val<id>`). The per-send update becomes `li; lui; sw` instead of a
  stack reload of the id plus `zext / slli / add` (6 → 3 instructions).
- `[[unlikely]]` removed from the pending-atomic branches (E1) and the two TXQ-status checks (E0).

**Review:** all three are behavior-preserving.
- The runtime array passed to the steps (`local_sender_channel_free_slots_stream_ids`) is a straight copy of the
  constexpr `sender_channel_free_slots_stream_ids`, and the steps' only call site indexes it with the same
  `fabric_position_for_compact_sender(sender_channel_index)`.
- No caller uses the removed members. The mux, relay and UDM kernels call `get_buffer_address(BufferIndex)` on
  `EthChannelBuffer` (the receiver-side class), which keeps its table.

**Runs** (`dq77_runs.sh` a–e, jobs 193–197, `dq77_compare.py`). To split the change:
- s77u = s77 with the hints restored;
- s77t = slot-table removal alone;
- s77s = compile-time stream ids alone;
- s77sn = s77s without the hints.

Control s74c ×3, s77 ×2, s77s ×2, others ×1; run spread ≤ 0.4 except the noisy chunk-4 flushed rows.

| Row (cycles/packet) | s74c | s77t | s77s | s77sn | s77u | s77 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mesh 1-hop E/W write 4352 | 132.3 | 147.5 | **132.0** | 132.1 | 135.4 | 140.1 |
| Mesh 1-hop N/S write 4352 | 133.3 | 149.0 | **133.2** | 134.2 | 135.8 | 138.3 |
| Mesh 1-hop E/W bidir write | **138.4** | 149.5 | 139.1 | 139.8 | 140.3 | 141.8 |
| E/W write 16–1024 B | 128.0 | 148.8 | **123.8** | **123.8** | 135.4 | 140.6 |
| N/S write 16–1024 B | 128.3 | 150.0 | **124.0** | 128.2 | 135.2 | 138.2 |
| Forwarded write 16–1024 B | 111.9 | 123.6 | 111.9 | **111.6** | 115.6 | 119.3 |
| E/W fused (E1) | 155.2 | 155.2 | 155.2 | **153.8** | 155.2 | 153.8 |
| E/W fused, flushed every packet | 441.3 | | 441.5 | **434.9** | 441.1 | 434.7 |
| N/S atomic | 138.9 | 148.8 | 138.9 | 141.4 | 138.8 | 141.4 |
| Linear 1-hop write (1D) | 137.7 | | **136.1** | **136.1** | 136.5 | 136.5 |
| Mean vs s74c: 2D 4352 / mix | | +5.3 / +11.7 | **−0.0 / −1.2** | +0.1 / −0.6 | +0.9 / +3.2 | +2.1 / +5.8 |
| Mean vs s74c: flushmix / flushmix4 / 1D | | | **−0.7 / −1.4 / −0.8** | −0.6 / −1.2 / −0.7 | +0.7 / +2.2 / −0.6 | +2.1 / −0.2 / −0.6 |

Validation:
- s77: stream 10/10, mesh 96/96, 1D sanity 181/181.
- s77s: the same three suites, all passing.
- Every benchmark suite of every variant passes.

**What it shows:**
- **As submitted, s77 is slower on every E0-bound row.** 1-hop E/W write +7.8, small writes +12.6 E/W and +9.9 N/S,
  forwarded small writes +7.4. E1-bound rows move ≤ 1.4 either way.
- **The compile-time stream ids are a clean win (s77s).**
  - Small writes: −4.2 E/W, −4.3 N/S.
  - Mix −1.2 mean, flushed mixes −0.7 / −1.4, 1D −0.8 (1-hop −1.6).
  - 4352 B 2D flat, except bidir +0.7.
  - E0 is 2–9 instructions smaller; E1 is unchanged.
- **The slot-table removal alone (s77t) costs 15 on 4 KiB writes and 21 on small writes,** though the removed code
  never ran in the loop. With the stream ids on top (s77u), the loss is still 3–7.
  - With the table gone, the 16-byte channel struct gets scalarized. Stride and bounds become constants: s77u
    advances by an immediate 4448. Each channel's slot pointer moves into a register.
  - That reshuffles register allocation across E0's whole loop:
    - s77u keeps `channel_connection_established` on the stack: a load plus a store every pass.
    - s77t spills the TXQ status two instructions after reading it.
  - Hot-path instruction counts and stack accesses stay within a few instructions of the control in all five builds
    (worker send 46–50 against 47), so the cycle loss is not explained by instruction count. No single instruction
    pattern predicts it: s77s, the fastest build, uses the TXQ status sooner than the control. Same E0 layout
    sensitivity as §32.
- **Dropping `[[unlikely]]` is mixed (s77s → s77sn):**
  - E/W fused −1.4 and E/W fused flushed −6.6 (E1);
  - N/S small writes +4.2, N/S atomic +2.5, N/S scatter and fused +1.2 to +1.3;
  - E0 +42 to +52 instructions on most 2D configs.
  - Mean slightly worse than s77s.
- **Recommendation: take s77s** (`variants/s77s/`): compile-time stream ids, slot table kept, hints kept.
  - The table removal is a footprint cleanup whose cost depends on register allocation. The no-allocation-dependent-
    changes rule rules it out as built.
  - The hint removal trades N/S for E/W fused rows.

## 78. The user's s78 (word load + lean sender channels) and a `noinline` main loop: neither beats s77s

**Change (s78, the user's):** s77s plus two parts.
- **Word load:** the pre-send payload size is read as the whole 4 B word at header offset 40 (`lw`), inside the
  fence, and masked (`& 0xFFFF`) after the TX-queue busy test. The `PacketHeaderBaseT` alias moves out of
  `#if defined(FABRIC_2D)` so 1D builds see it.
- **Lean channels:** v2's sender channels become the 4 B `LeanStrideSenderEthChannel` (cached slot address only) and
  advance with `advance_to_next_cached_buffer_slot_addr_by_stride_ct<base, channel_buffer_size>()`, in
  `sender_channel_try_send` and, under `FABRIC_ERISC_ROUTER_V2`, in the speedy path. v1 is unchanged. This is §32's
  bit-24 channel type, which §50 removed from v2.

**Review:** both parts are behavior-preserving.
- Offset 40 is 4 B-aligned in both header types; `PackedHeaderWord40` asserts it, but only in 2D. The `src_ch_id`
  byte store to offset 43 comes after the load, and the mask keeps only bytes 40–41.
- Every v2 sender channel is initialised from `channel_allocs` with `channel_buffer_size` as its stride, so the
  compile-time bounds match the runtime ones. The speedy path (VC2) was the reason v1 kept the full class there; the
  `FABRIC_ERISC_ROUTER_V2` branch covers it.
- In s77s the `zext.h` sits after the CMD and STATUS loads (`lhu; lw CMD; lw STATUS; zext.h`), not between the L1
  load and the TX-queue loads. In s78 it moves after the busy test, as intended.

**`noinline` A/B** (the user's question): `run_fabric_edm_main_loop` marked `__attribute__((noinline))` instead of
`FORCE_INLINE`, on both ERISCs (s78n, s77si). Then on E0 only (s78n0, s77si0), using an `if constexpr (MY_ERISC_ID == 0)`
dispatch to a `noinline` wrapper while E1 keeps the inline call.
- The out-of-line builds pass a stack check in the run script: the three largest frames plus 256 B leave ≥ 940 B of
  local memory free.
- In the E0-only builds, E1's instruction count changes by +8 to +28 on most configs, against +18 to +108 when both
  ERISCs go out of line. The wrapper passes the volatile termination-signal pointer by reference, so E1 is not
  byte-identical.

**Runs** (`dq78_runs.sh` a–c, jobs 198–200, `dq78_compare.py`):
- Split builds: s78w = word load alone, s78l = lean channels alone.
- Runs: s77s ×2 (this session's control plus §77's `dq77sv`), s78 ×2, others ×1. Run spread is ≤ 0.2, except the
  chunk-4 flushed rows (up to 12).

| Row (cycles/packet) | s77s | s78w | s78l | s78 | s78n | s78n0 | s77si | s77si0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mesh 1-hop E/W write 4352 | 132.0 | 132.0 | 135.4 | 131.2 | 131.6 | **131.2** | 132.5 | 132.1 |
| Mesh 1-hop N/S write 4352 | 133.2 | 133.3 | 135.8 | 133.6 | 132.7 | **131.6** | 133.5 | 132.6 |
| E/W write 16 B | 123.8 | 123.8 | 135.4 | 130.1 | 120.1 | **118.9** | 122.3 | 124.9 |
| N/S write 16 B | 124.0 | 124.0 | 135.2 | 130.3 | 125.4 | **122.7** | 125.7 | 126.0 |
| Forwarded write 16 B | **111.9** | **111.9** | 115.7 | **111.8** | 115.6 | 113.1 | 115.6 | 113.1 |
| E/W atomic | **129.4** | **129.4** | 135.0 | 130.1 | 133.1 | 130.5 | 133.0 | 130.5 |
| E/W scatter 4352 (E1) | **165.2** | **165.2** | **165.2** | **165.2** | 175.1 | 166.4 | 175.1 | 166.4 |
| 1D Linear 1-hop write | 136.1 | 136.1 | 136.5 | 136.4 | **134.2** | 135.4 | **134.2** | 135.5 |
| 1D unidir multicast write | 196.3 | 196.2 | **195.9** | 196.0 | 209.3 | 196.1 | 209.3 | 196.1 |
| 1D neighbor exchange write | **124.9** | 125.0 | **124.9** | **124.9** | 133.2 | 135.1 | 131.5 | 131.6 |
| Mean vs s77s: 2D 4352 / mix | | +0.0 / +0.0 | +0.9 / +4.4 | +0.1 / +1.7 | +4.5 / +1.7 | +0.4 / **−0.5** | +4.6 / +2.2 | +0.6 / +0.9 |
| Mean vs s77s: flushmix / 1D | | +0.0 / +0.0 | +1.5 / +0.1 | +0.6 / +0.1 | +0.4 / +1.6 | **−0.6** / +0.9 | +1.3 / +1.5 | +0.5 / +0.7 |

Small writes at 256 and 1024 B are within 2.1 of the 16 B rows in every build.

Validation:
- s78, s78n, s77si, s78n0 and s77si0 each pass stream 10/10, mesh 96/96 and 1D sanity 181/181.
- Every benchmark suite of every variant passes.

**What it shows:**
- **The word load changes nothing.** s78w is within 0.1 of s77s on every row, and E0 has the same instruction count.
  Before the change, the `zext.h` already waited on the L1 load while the STATUS load was outstanding.
- **The lean channels lose again (s78l), the third time after §32 and §77's s77t.**
  - Small writes +11.6 E/W and +11.2 N/S, forwarded +3.8, E/W atomic +5.6, 1-hop 4352 B +3.4.
  - E0 is 4–38 instructions and 128–320 B of frame smaller. E1 is unchanged.
  - The loop's stack loads barely move (34 → 32 on the hot E/W config). In s78, the compile-time last-slot address is
    materialised once and spilled, so the wrap test still reloads it from the stack.
- **s78 is +6.3 on small writes against s77s,** half of s78l's loss. 1-hop 4352 B E/W is −0.8. The word load is
  neutral on its own, but combined with the lean channel type it lands on a different allocation.
- **`noinline` on both ERISCs is ruled out.**
  - E1 grows by 18–108 instructions. Every E1-bound scatter row is +6 to +10, and 1D multicast is +7 to +13.
  - E/W small writes do improve: 120.1 (s78n) and 122.3 (s77si) against s77s's 123.8.
- **`noinline` on E0 alone depends on the base build:**
  - **On s78 (s78n0)**, it gives the best small writes so far: E/W 118.9 (−4.9 against s77s), N/S 122.7 (−1.3), and
    1-hop 4352 B −0.8 E/W and −1.6 N/S.
    - Forwarded +1.2, atomics +0.5 to +1.3 and fused +1.3.
    - 2D scatter +1.2, from E1's small code change or E0.
    - 1D neighbor exchange +10.2 write and +6.0 scatter.
  - **On s77s (s77si0)** it is worse almost everywhere: small writes +1.1 E/W and +2.0 N/S, forwarded +1.2, neighbor
    exchange +6.7.
  - Out of line, E0's loop trades stack loads for loads through the argument pointers. On the hot config, sp loads
    drop from 32–34 to 12–18 while other loads rise from 27 to 42–46, so total memory traffic is about the same.
  - The neighbor-exchange loss shows in all four out-of-line builds, so it is the one effect that does not depend on
    allocation.
- **Recommendation: stay on s77s.**
  - The word load is neutral; keep it only for readability, with an alignment `static_assert` in 1D builds too.
  - The lean channels are ruled out a third time.
  - `noinline` helps E0 only when combined with lean channels, and the sign of that change flips with the base build,
    which the no-allocation-dependent-changes rule rules out. It also costs 1D neighbor exchange 6.6–10.2 in every build.

## 79. The user's s79 (cold connection helpers + 0x7FF free-slots mask): the mask alone wins, the helpers lose

**Change (s79, the user's):** s77s plus two parts, both aimed at stack traffic in E0's loop.
- **Cold connection helpers:** `run_worker_sender_step` keeps the disconnected poll inline and moves the two
  transition bodies into `__attribute__((noinline))` `establish_worker_connection_cold` /
  `teardown_worker_connection_cold`, which take only the worker interface. They replace the inlined
  `check_worker_connections` call. The aim is to remove the per-pass `li a6,1; sw a6,12(sp)` (the merged store of
  `channel_connection_established = true`).
- **Narrow mask:** the grouped free-slots value is masked with `grouped_free_slots_imm_mask = 0x7FF` instead of the
  17-bit field mask `(1 << REMOTE_DEST_WORDS_FREE_WIDTH) - 1`. 0x1FFFF does not fit `andi`'s 12-bit signed immediate,
  so s77s keeps it in a stack slot and reloads it every pass (`lw t6,56(sp); and`). It applies in
  `channel_all_slots_free` and in both sender steps. The change carries `static_assert`s that the mask fits the field
  and every channel's `num_buffers` fits the mask.

**Review:** both parts are behavior-preserving.
- The helpers reproduce `check_worker_connections` exactly: poll `connection_live_semaphore` while disconnected,
  test for a teardown request while connected. The teardown flag is now read once per pass instead of twice (the
  outer `||` and the inner `else if` both read it before).
- `PHASE_TX_CONN` now times only the two transitions, not the disconnected poll. This matters only in phase-profiler
  builds.
- The free-slots counter stays within [0, `num_buffers`] (it starts at `num_buffers`, and a slot is taken only when one
  is free). Bits 11–16 are therefore always zero and the narrow mask gives the same value. The upper register fields
  (bit 17 and up) still need masking, and they are.

**Runs** (`dq79_runs.sh` a and b, jobs 203–204, `dq79_compare.py`):
- Split builds: s79c = connection helpers alone, s79m = mask alone.
- Runs: s77s ×3 (this job's control plus §77's `dq77sv` and §78's `dq77s3`), all others ×2. Run spread is 0.0 on
  every 2D row except the chunk-4 flushed rows (up to 12). 1D half-ring multicast had a one-run outlier of up to 6.0,
  in s79m scatter and s79c write.
- Validation (s79): stream 10/10, mesh 96/96, 1D sanity 181/181. Every benchmark suite of every variant passes.
- Stack check: every new router ELF leaves ≥ 1052 B of local memory free.

| Row (cycles/packet) | s77s | s79c | s79m | s79 |
| --- | ---: | ---: | ---: | ---: |
| Mesh 1-hop E/W write 4352 | 132.0 | **131.7** | 132.2 | 132.1 |
| Mesh 1-hop N/S write 4352 | 133.3 | **132.2** | 133.1 | 133.1 |
| Mesh 3-hop E/W write 4352 | 147.8 | 148.1 | **146.9** | 147.8 |
| E/W write 16 B | 123.8 | 124.3 | **121.7** | 127.4 |
| E/W write 1024 B | 124.0 | 124.4 | **122.8** | 127.5 |
| E/W mix of 4 sizes | 124.3 | 124.4 | **122.2** | 127.6 |
| N/S write 16 B | 124.0 | 125.5 | **123.9** | 127.5 |
| Forwarded write 16 B | **111.9** | **111.9** | **111.9** | 112.3 |
| E/W atomic | 129.4 | **129.3** | **129.3** | 129.6 |
| Mean vs s77s: 2D 4352 / mix | | −0.11 / +0.17 | −0.07 / **−0.36** | −0.06 / +1.00 |
| Mean vs s77s: flushmix / 1D | | +0.03 / +0.10 | **−0.25** / +0.16 | +0.47 / +0.03 |

The 1D means are within the half-ring outliers; every other 1D row is within 0.3 of s77s in every build. E1 is
byte-for-byte the same size in all three builds (0 instructions on every config), so every difference is on E0.

**E0 loop on the hot E/W config** (`10733501891889981293`):

| Build | Loop instructions | sp loads | sp stores | Mask | Per-pass `established` store |
| --- | ---: | ---: | ---: | --- | --- |
| s77s | 227 | 34 | 7 | `lw t6,56(sp); and` | `li a6,1; sw a6,12(sp)` |
| s79m | 226 | 33 | 6 | `andi t1,t1,2047` | gone |
| s79c | 244 | 26 | 6 | `lw a4,24(sp); and` | gone |
| s79 | 241 | 25 | 6 | `andi a5,a5,2047` | gone |

- The helpers add 25–77 instructions to E0 across configs; the mask removes 1–9.
- In s79c and s79, GCC duplicates the TX-queue issue, completion read and `did_something` code inside the loop. That
  accounts for the +17 loop instructions, despite 8 fewer stack loads. The calls themselves sit outside the loop.

**What it shows:**
- **The mask alone is the best build so far on E/W small writes (s79m).**
  - E/W 16 B −2.1, 1024 B −1.2, mix of 4 sizes −2.1, 3-hop E/W −0.9.
  - N/S and forwarded rows are unchanged.
  - It also removes the per-pass `established` store that the helpers targeted: with the mask constant no longer held
    in a stack slot, the flag no longer lands there either.
- **The helpers lose on their own (s79c).** Small writes +0.5 E/W and +1.5 N/S, against 1-hop 4352 B −0.3 E/W and −1.1
  N/S. They remove the store and 8 stack loads, but the duplicated send path costs more.
- **Together they are worse than either alone (s79):** small writes +3.6 E/W and +3.5 N/S, forwarded +0.4. Adding the
  mask on top of the helpers costs 3.1 on E/W 16 B, where on s77s it saves 2.1. As in §78, E0's cycle cost follows the
  loop's layout, not its instruction or spill counts.
- **Recommendation: take the mask alone (s79m) as the new base, and drop the helpers.**
  - The mask removes work on every pass and is safe by the counter bound, so it is not an allocation trick.
  - The size of its gain still depends on allocation (it reverses on the helper layout), so don't treat −2.1 as
    banked for later changes.

## 80. Why 3-hop E/W writes run ~16 slower than 1-hop, and a follow-up: cached downstream credits on E1 (not yet run)

**Question (the user's):** 3-hop E/W writes run at 147.8 against 132.0 for 1-hop (s77s, 4352 B). The middle routers'
E1 forwards instead of delivering locally, which is a write plus a doorbell, so why does it cost that much?

**Answer** (no new runs: §27, §54, §72, §73, plus the s79m disassembly below): the forward costs the middle router's E1
~24 more per packet than a local write. E1 has ~13 of slack on a 1-hop write, so in 3-hop it becomes the slowest
stage. Two more limits sit right behind it.

| Per packet (cycles) | 1-hop E/W (chip 1's West router delivers) | 3-hop (chips 1–2's West router forwards) |
| --- | ---: | ---: |
| E1 decode/admit (§72 profile) | 20.1 | 30.3 |
| E1 local write / forward (§72 profile) | 46.1 | 60.0 |
| E1 busy (§72, +20 spin slack test) | ~115–120 | ~140 |
| E0 busy (same test) | ~129 (binds) | ~135 |
| Credit-loop floor | 17 worker slots → ~124–130 | 16 forwarding slots → ~135–140 |
| Measured | 132.0 | 146.9–148.8 |

§72 ran on s71 (3-hop 148.8). E1's receive path has barely changed since: 3-hop is 147.8 on s77s and 146.9 on s79m.

**Why the forward costs E1 more:**
- **Admission (+10).** For a local-only packet, `admit_2d_dispatch` returns true at compile time. For a forward, it
  reads the downstream channel's free-slot stream register and branches on it at once (`lui; lw 1188(0xffb45000);
  slli; beqz`), so E1 waits out the stream-register read (~7–9, §17 / §55).
- **The forward itself (+14).** Disassembly: s79m E1, config `10856749301170935304` (chip 1's West router, eth channel
  4). This is the same binary that delivers locally in the 1-hop test; only the branch differs.
  - Local write (`0xd514–0xd578`): 26 instructions, one NoC command (one command-buffer-ready poll, 7 register
    stores, 2 counter updates).
  - Forward (`0xdcb8–0xdd64`): 44 instructions and two NoC commands, the data write and the doorbell. Each has its own
    ready poll and issued-counter update.
  - The `EdmToEdmSender` state lives in E1's stack frame, so the forward adds six stack loads and a store: slot
    index byte → slot address table (a dependent chain ahead of the data write), the index reload, wrap and store,
    both command-buffer ids, and the local credit pointer.
  - The doorbell's own register write is one store (stateful API: E/W routers have one live downstream). What it adds
    is the second ready poll and the bookkeeping.
- **The forwarding channels are one slot shallower.** The Mesh allocator gives the one spare slot to the worker
  channel (17), and the forwarding channels keep 16. A forwarding credit loop is ~2,150–2,220 cycles (§27: 8 and
  11 slots), so 16 slots floor at ~135–140 even with infinitely fast cores.
- **The middle router's E0 is no lighter than chip 0's.** It still visits the idle worker channel on every pass
  (~35 per pass, including the ~13-cycle connection poll, which stays per pass), and makes ~1.5 passes per packet
  (§73).
- **No core is at 148 by itself.** E1 is ~140, E0 ~135 and the credit floor ~135–140. That three such close limits,
  with completions coming back 4 at a time, settle ~8 above the slowest is an inference, not a measurement.

**Follow-up (deferred by the user 2026-10-24; other changes first): cache downstream credits on E1.**
- **Change.**
  - Keep a loop-carried count of known-free downstream slots per live downstream.
  - Admission tests the count. It re-reads the stream register only when the count is zero, and each forward
    decrements the count.
  - Keep the existing per-packet local decrement of the register, so the register stays the source of truth. That
    means no protocol change, and teardown and assert reads stay correct.
  - Also refresh the count on idle E1 iterations (no packet), so the read stays off the packet path after a lull.
- **Where.**
  - `admit_2d_dispatch`: the single-live E/W path and the multi-live N/S path (`free_0..3`).
  - 1D `can_forward_packet_completely` → `edm_has_space_for_packet`, which does the same read-then-test.
- **Not a wasted branch on single-flow forwarding.**
  - Credits arrive about 4 at a time. E0 returns `completions_since_last_check` as one +N write, and bit 22 acks 4
    per frame (0.23 credit frames per packet on the 3-hop middle E0, §72).
  - A reload takes everything that is free, not just the latest arrival.
  - When E1 binds, the downstream channel drains ahead of it, so one read covers many packets. When the downstream
    binds, E1 is waiting for credits anyway.
  - Worst case (a reload every packet) is today's read plus one predictable branch.
- **Expected.**
  - 3-hop: the channel runs nearly full (~15.7 of 16 slots in flight at ~140 per packet), so a reload finds about
    one batch. That spreads the read over ~4 packets and saves E1 ~5–7 per packet.
  - The 3-hop interval should move less, a few cycles rather than 16, because the 16-slot floor and E0 are close
    behind.
  - Largest gains where E1 also delivers locally (the multicast forwarder; mixed local + forwarded traffic), since
    credits pile up during local work.
  - 1D multi-hop should gain too (E1-bound, §76).
- **Optional second step.** Consume credits in bulk at reload (read V, write −V, cache V) to also drop the
  per-packet register decrement and its stack-pointer load. Anything that reads the register as "free slots"
  (teardown or drain checks, asserts) must then add the cached count back.
- **Risk.**
  - It adds one loop-carried value per live downstream on E1.
  - If GCC spills it, the test becomes a local-memory load, still cheaper than the stream-register read, so the
    gain shouldn't hinge on allocation.
  - Check E1's stack frames (the `dq79_runs.sh` stack check) and the E1 receive path in the disassembly.
- **Runs when picked up.**
  - A variant on top of s79m, with validation (stream, mesh, 1D sanity) and the 4352 / mix / flushmix / 1D suites
    against s79m.
  - Rows to watch: 3-hop E/W, multicast, the forwarded (`FWD_*`) rows and 1D multi-hop.
  - A +20 RX-delay slack test on 3-hop, to see whether the bound moves to E0 or to the 16-slot credit floor.

## 81. The user's s80 (de-templated forwarding channels in an unroll-1 loop): correct, but small writes +1.8 to +2.9

**Change (s80, the user's, on top of s79m):** E0's forwarding sender channels move out of the templated
`run_sender_channel_step` / `run_forwarding_sender_step` path.
- **State:** one `ForwardingSenderChannelState` per forwarding channel. It holds the slot cursor, first and last slot,
  stride, sent and processed counts, the completion-counter pointer, the free-slots stream base, the upstream credit
  address, the channel index, `num_buffers` and two flags.
- **Init:** filled once before the main loop by a `noinline`, `-Os` init function. The aim is to keep `-ftree-sra` from
  promoting the channels' counts into loop-carried registers.
- **Step:** one `FORCE_INLINE` `run_forwarding_sender_step_runtime`, stepped over the VC0 forwarding channels in a
  `#pragma GCC unroll 1` loop (VC1 the same way). In 1D, with one forwarding channel, it is called directly.
- **Scope:** only when every serviced forwarding channel skips the liveness check, uses counter credits and has no
  first-level ack. Otherwise the templated path remains.
- **Also:** the unused cold-helper templates left in s79m are removed, and `skip_idle_bit_test` now also yields the
  slot mask.

**Review:** the runtime step does exactly what the templated one does.
- It has the same idle test (bit test or masked compare), receiver-space test, BH pre-send TX-queue/status reads and
  busy bail-out, and the same send.
- The slot cursor wraps with the same stride (`channel_buffer_size`, as in `local_sender_channels.init`).
- The overlapped completion read, local free-slot increment and remote packets-sent register write are unchanged.
- The non-overlapped completion read and the `+N` credit write to the upstream credit stream match the s79m path
  (`notify_persistent_connection_of_free_space_impl`).
- The upstream credit address is copied after `wait_for_static_connection_to_ready` has cached it. Nothing else in the
  router reads the forwarding channels' old sender channel or credit state.
- The non-bit-test path gains an early return when the channel is idle with nothing pending. Its behavior is
  unchanged.

**Runs** (`dq80_runs.sh` a, job 205, `dq80_compare.py`):
- s79m ×3 (§79's two plus a same-job control, which matched to 0.0 on the small-write rows) and s80 ×2. Run spread
  is ≤ 0.2 on every 2D row except the chunk-4 flushed rows.
- Validation (s80): stream 10/10, mesh 96/96, 1D sanity 181/181. Every suite passes.
- Stack: the smallest margin drops from 1052 B to 908 B (E0 config `14376931258930336131`, frame 1424 → 1568).
- E1 is unchanged on every config (same instruction stream).

| Row (cycles/packet) | s79m | s80 |
| --- | ---: | ---: |
| E/W write 16 B / 256 B | 121.7 / 121.7 | 124.6 / 124.6 |
| E/W write 1024 B / mix of 4 sizes | 122.8 / 122.2 | 124.8 / 124.7 |
| N/S write 16 B / 1024 B | 123.9 / 124.9 | 125.7 / 126.1 |
| Forwarded write 16 B / 4352 B | 111.9 / 125.7 | 111.9 / 125.7 |
| Mesh 1-hop E/W / N/S write 4352 | 132.2 / 133.1 | 132.0 / 132.5 |
| Mesh 3-hop E/W write 4352 | 146.9 | 148.0 |
| Linear 3-hop E/W write 4352 | 145.7 | 147.1 |
| Mean vs s79m: 4352 / mix / flushmix / flushmix4 / 1D | | +0.02 / +0.68 / +0.43 / +0.50 / −0.03 |

**E0 code** (hot E/W config `10733501891889981293`: worker channel sending, three forwarding channels idle):

| | s79m | s80 |
| --- | ---: | ---: |
| Loop instructions | 226 | 231 |
| sp loads / stores in the loop | 33 / 6 | 42 / 10 |
| Worker channel's sent / processed counts | `32(sp)` / `28(sp)` | `16(sp)` / `8(sp)`, plus a new store and reload of `8(sp)` around the completion read |
| Idle forwarding visit | bit test, sent ≠ processed in registers | bit test, then two stack loads (sent, processed) and a compare, per channel |
| .text on 2D configs | | +9 to +23 instructions |
| Two largest frames | | +32 to +144 B |

**What it shows:**
- **The registers freed from the forwarding channels did not reach the worker channel.** Its counts stay on the
  stack, so the sending pass is no cheaper. Every idle forwarding visit now waits on two stack loads before its
  branch. On the E0-bound small-write rows that costs ~3 cycles per pass (three idle channels): E/W +2.9, N/S +1.8.
- **The `unroll 1` loop is still peeled.** The step body appears three times in the hot config, each copy with its
  own fixed stack offsets (`352/356`, `400/404`, `448/452`). So the code-size benefit of a single body did not
  happen either.
- **Forwarding gains nothing.** Forwarded writes are flat, while 3-hop (+1.1) and linear 3-hop (+1.4) are slightly
  slower: the middle router's E0 now loads its active channel's counts from memory.
- **Recommendation: stay on s79m.** This is consistent with §77/§78: moving channel state into memory to steer
  `-ftree-sra` changes which values sit in registers, but not in the direction the hot path needs.

## 82. Consolidation Phase 0: s79m's performance holds under small edits and compiler settings, but not at -Os

**Context:** first step of the consolidation work (`consolidation/PLAN.md`). The goals, in order, are a router whose
performance does not swing with unrelated edits or toolchain changes, then a mainline-worthy source, then further
cuts. Phase 0 measures how stable today's build is and builds the tools to measure it for new designs. The written
companions are `consolidation/ROUTER_MAP.md` (what each ERISC does per pass and per packet, essential vs overhead)
and `consolidation/LEDGER.md` (every experiment in §1–§81, classified). ERISC-only mode; the DRISC line is out of scope.

**Tools** (`consolidation/`):
- **Offline builds** (`offline_build.py`). The JIT's compile and link commands were logged once per opt level
  (`TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1`, fresh kernel cache) for the 440 router kernels the 4352, mix, flushmix,
  flushmix4, 1D and three validation suites use. The tool rebuilds any variant on a copy of `tt_metal/fabric` for all
  440 in ~25 s, at the logged opt level, another one, or with extra flags. It is exact: 440/440 at -O3 and 440/440 at
  -Os have the same `.text`, at the same addresses, as the hardware-built ELFs.
- **Code comparison:** `codegen_diff.py` (how many configs changed, and how many instructions differ in the hot E0 and
  E1), `textcmp.sh`. **Benign edits:** `perturb.py`.
- **ELF injection** (`p0c_runs.sh`). Offline-built ELFs copied over the router ELFs in a warm kernel cache are used as
  they are: the tree and the dependency hashes are unchanged, so the JIT does not rebuild. Checked on job 208: no
  router kernel recompiled, the cache still held the injected ELFs, and the runtime's load dumps (`.xip.elf`) carry
  the injected code. Any offline build can run on hardware this way without touching the tree.
- **Gotcha:** `TT_METAL_FABRIC_OPT_LEVEL` with a warm cache reuses the ELFs built at the other level; no router kernel
  is recompiled, so the kernel cache key does not include the opt level. Job 206's -O2 and -Os runs were therefore
  -O3 repeats (relabeled `p0o3b_`, `p0o3c_`). Each level now gets its own cache.

**Static counts do not predict cycles/packet.** On the 17 variants from s74c to s80 that have both saved ELFs and
hardware numbers, loop size and stack traffic in the hot E/W E0 (the `loopstat` heuristic) do not order the E/W 16 B
write:
- s78 and s78l have the same loop (372 instructions, 41 sp loads, 11 stores), but run at 130.1 and 135.4.
- s79m has more loop instructions and sp loads than s77s (379 / 47 vs 345 / 44) and is 2.1 faster.
- s78n0, with 16 sp loads, runs at 118.9.

So offline builds can say whether and where code moved, but only hardware says what that costs.

**Runs** (`p0a_runs.sh`, `p0b_runs.sh`, `p0c_runs.sh`; jobs 206–208; `p0_compare.py`):
- **Router:** s79mc, i.e. s79m with two unused templates removed. Its code is identical to s79m's on all 320 configs
  both have.
- **-O3 baseline:** 7 runs (§79's three plus four here). The run-to-run spread has a median of 0.06–0.14 per suite,
  except the chunk-4 flushed rows (6.2) and the half-ring rows (up to 8.5).
- **Validation:** passes at -O3, -O2 and -Os: stream 10/10, mesh 96/96, 1D sanity 181/181 at each level.
- **Stack:** smallest margin 1052 B at -O3; 1036 B with the `field` edit.

| Build (all from the same source unless noted) | Hot E0 / E1 instructions that differ | 4352 | mix | flushmix | 1D | E/W 16 B | N/S 16 B | FWD 16 B |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| -O3 (production without VC1) | — | 0 | 0 | 0 | 0 | 121.7 | 123.9 | 111.9 |
| `reorder`: two sender-channel members swapped | 17 / 0 | −0.02 | −0.00 | +0.03 | −0.11 | 121.7 | 123.8 | 111.9 |
| `field`: one member added, set at init, never read | 66 / 0 (frame +16 B) | −0.01 | −0.00 | +0.01 | −0.08 | 121.7 | 123.9 | 111.9 |
| `-fira-algorithm=priority` | 40 / 14 | −0.00 | −0.00 | +0.01 | −0.14 | 121.7 | 123.9 | 111.9 |
| `-fno-schedule-insns2` | 52 / 32 | −0.01 | +0.01 | +0.01 | −0.11 | 121.7 | 123.9 | 111.9 |
| `-fno-gcse` | 302 / 240 | −0.00 | −0.01 | +0.06 | −0.13 | 121.6 | 123.9 | 111.9 |
| `-fno-tree-loop-im` | 10 / 10 | −0.01 | −0.00 | +0.03 | −0.12 | 121.7 | 123.8 | 111.9 |
| -O2 | (whole kernel) | +3.83 | +3.94 | +2.59 | +2.55 | 125.0 | 127.7 | 117.4 |
| -Os (production with VC1) | (whole kernel) | +25.33 | +27.85 | +21.98 | +19.14 | 146.9 | 152.1 | 145.8 |

The suite columns are mean changes vs -O3 in cycles/packet; flushmix4 is left out because of its noise. The 1D
means for the edits and flags are pulled by the noisy half-ring rows; every other 1D row is within ±0.3.

More rows at -O3 / -O2 / -Os:

| Row | -O3 | -O2 | -Os |
| --- | ---: | ---: | ---: |
| Mesh 1-hop E/W write 4352 | 132.2 | 132.4 | 147.2 |
| Mesh 3-hop E/W write 4352 | 146.9 | 150.7 | 170.2 |
| Mesh multicast write 4352 | 140.7 | 143.3 | 166.5 |
| E/W scatter 4352 / fused 4352 | 165.3 / 155.2 | 170.8 / 160.8 | 199.0 / 190.1 |
| Linear 3-hop E/W write 4352 | 145.7 | 149.6 | 169.4 |
| Neighbor exchange write | 125.0 | 125.9 | 130.9 |

E0 `.text`, median (max) over its 160 configs: 8,708 (12,196) B at -O3, 7,394 (10,092) B at -O2, and 7,174 (8,936) B
at -Os.

**What it shows:**
- **s79m is not fragile to incidental changes.** Two struct edits, and four compiler settings that each change 10–302
  instructions of the hot E0, leave every suite within noise. Earlier swings came from edits that changed the main
  loop's structure:
  - the channel state shrinking until it was split into registers (§77 +21, §78 +11.5);
  - code reaching into the loop (§79);
  - the forwarding state moving to memory (§81).

  The sensitivity has steep edges near the current build rather than a smooth slope. Random benign edits did not
  find one; structural edits did.
- **The steepest edge found so far is the optimization level.**
  - -Os costs +19 to +28 per suite: E/W small writes +21% (121.7 → 146.9), forwarded writes +30%.
  - -O2 costs +2.5 to +3.9.
  - Production builds the router at -Os whenever VC1 is active (`get_kernel_opt_level`), so a VC1 router runs this
    source about 20% slower than every number in this study.
  - The -O3 code is only ~20% larger than the -Os code here. Whether VC1 configs fit at -O3 or -O2 on Blackhole has
    not been checked.
- **For Phase 1, stability is measured where the edges are.** Each candidate runs on hardware at -O3, -O2 and -Os.
  It also runs with edits of the kind that moved s79m before, such as a call reaching into the main loop or smaller
  channel state, rather than random edits. Offline builds show where the code moved.
- **Two items from the Phase 0 documents need a decision before Phase 1:**
  - The test kernel sends every flushed packet as `FLUSH_RELAXED` (`kActiveFlushMode`,
    `tt_fabric_test_kernels_utils.hpp`). So the validation yamls never exercise the default STRICT flush.
  - s79mc still carries six `[[unlikely]]` hints, five of them added by the study: on the two E0 TX-queue status checks
    and on three E1 pending-atomic branches. They are layout hints.

## 83. Consolidation Phase 1, round 0: references on s79m (hook probe, no layout hints, STRICT flushes)

**Context:** Phase 1 builds a new kernel file, v3, starting from v3a, a faithful consolidation of s79m. Branch
hints are out (the user's call), and so is the experiment machinery. Before any v3 candidate, three references on
s79mc:
- **hook:** s79mc plus `perturb.py hook`. That is a `noinline` debug call inside the main loop, on a value read once
  before the loop that is never `0x5AFE`: the kind of hook a contributor adds. Offline, it changes 1,274 of the hot
  E0's 2,523 instructions and 1,865 of the middle E1's 1,909. By itself it costs one compare per pass.
- **nohint:** s79mc without the five `[[unlikely]]` hints the study added (two E0 TX-queue status checks, three E1
  pending-atomic branches).
- **strict:** s79mc with the test kernel sending flushed packets as `FLUSH_STRICT`, the default, instead of
  `FLUSH_RELAXED` (`kActiveFlushMode`). Every validation yaml so far ran RELAXED.

New tools (`consolidation/`):
- `swap.py` swaps any variant (router, headers, test kernels) into the tree and restores it.
- `jobs_lib.sh` holds the job functions.
- `compare.py` compares any set of builds.
- Offline builds now overlay a snapshot of `tt_metal/fabric` (taken at s79mc), so device jobs that swap files cannot
  leak into them.

**Runs** (`p1r0_runs.sh`, job 209): hook and nohint ran the benchmarks at -O3; strict ran validation plus the
flushed-signal suites. Every suite passed. The STRICT validation is new: stream 10/10, mesh 96/96, 1D sanity 181/181.

| Row (cycles/packet) | s79m (7 runs) | hook | nohint | strict |
| --- | ---: | ---: | ---: | ---: |
| E/W write 16 B / mix of 4 sizes | 121.7 / 122.2 | 126.1 / 126.3 | 121.6 / 122.2 | |
| N/S write 16 B / 1024 B | 123.9 / 125.0 | 126.7 / 127.0 | 127.2 / 127.5 | |
| Forwarded write 16 B | 111.9 | 115.1 | 111.6 | |
| Mesh 3-hop E/W / multicast write 4352 | 146.9 / 140.7 | 149.3 / 142.4 | 146.8 / 140.2 | |
| Neighbor exchange write | 125.0 | 128.5 | 125.0 | |
| E/W fused 4352 flushed / unflushed | 442.2 / 155.2 | 442.3 / 158.4 | 434.8 / 153.9 | 437.9 / |
| E/W / N/S chunk-8 fused, flushed | 137.9 / 144.8 | | 137.7 / 146.4 | 146.3 / 166.5 |
| N/S chunk-8 atomic, flushed | 136.1 | | 138.3 | 153.0 |
| E/W / N/S chunk-4 fused, flushed | 156.7 / 166.0 | | | 180.3 / 210.2 |
| Suite means vs s79m: 4352 / mix / flushmix / 1D | | +0.99 / +2.16 / +1.27 / +0.79 | +0.06 / +0.47 / +0.02 / −0.08 | flushmix +3.46 |

**What it shows:**
- **The hook moves s79m by up to +4.4 (E/W small writes), about 3 of it beyond the probe's own compare.** That is
  the yardstick a v3 design has to beat: the same hook should cost it no more than the compare.
- **The hints were doing something, but not consistently.**
  - Without them, N/S rows lose 1–3.3: small writes +3.3, N/S atomics +2.5.
  - E/W rows are flat, and E/W flushed fused writes gain 7.4.
  - v3 drops them, as decided, so the N/S cost has to come back through structure.
- **STRICT flushes cost more on chunked flushed signals:** +8.4 (E/W) and +21.7 (N/S) at chunk 8, +24 and +44 at
  chunk 4. That is the RELAXED bypass of §70–§74. The router passes validation either way. From here on, v3 is
  validated with both modes.

## 84. VC1 on the P150x8 (two 2x2 meshes): it runs once the rank bindings use chip ids; -O3 does not fit

**Harness** (`fabric_exp/vc1/`, `consolidation/vc1_runs.sh`). The P150x8 is split into two 2x2 meshes, one MPI rank
each (`ttrun.py` legacy mode with a rank binding), so the routers carry VC1.
- **First three attempts (jobs 210, 211, 213) stopped in the control plane:** `SAT placement: mesh 'M0' has no PGD
  grouping and no embeddable MGD fallback`.
- **The descriptor was correct:** each face of the cube is a 2x2 ring of 2-link edges.
- **The rank bindings were the cause.** They listed each face's PCIe ids (`0,1,2,7` / `3,4,5,6`), but the mapper reads
  `TT_VISIBLE_DEVICES` as cluster-descriptor chip ids. As chip ids, `{0,1,2,7}` is not a face.
- **Found offline.** tt-metal's mock cluster (`--mock-cluster-rank-binding`, this host's cluster descriptor;
  `vc1_mock.sh`) reproduces the failure CPU-only in a second. With chip ids `0,1,2,3` / `4,5,6,7` it places both
  meshes. The board type (eight single-chip `p150` boards, every `asic_location` 0) does not matter.
- **ttrun's discovery mode cannot split one host:** it needs one host per rank.
- **The test harness aborts on inter-mesh benchmark patterns** (`BandwidthProfiler::trace_line_or_mesh_traffic_path`,
  `bad optional access`). The yaml now runs inter-mesh traffic only in the data-checked sanity tests.

**Run** (job 216; the tree's router, s79mc, run as v2; 13129088; keepalive 1000; telemetry off):

| Level | Result |
| --- | --- |
| default (builder picks -Os because VC1 is active) | Multi-mesh sanity: all-to-all (all, intra and inter), 4 packet types × 1/4 KiB, plus inter-mesh neighbor exchange × 4 types. All 12 finished on both ranks without error. Intra-mesh benchmarks ran (numbers lost: the per-test CSV was overwritten by the next level) |
| -O3 | Does not load: `Program size (30048) too large for kernel config buffer (26624) on ACTIVE_ETH` |
| -O2 | Same sanity tests, all finished. Benchmarks below |

| -O2, VC1-enabled routers | cycles/packet | VC0-only 2x4 mesh at -O2 (§82) |
| --- | ---: | ---: |
| 1-hop E/W write 16 B / 4352 B | 234.5 / 253.7 (23.2 GB/s) | 125.0 / 132.4 |
| 1-hop N/S write 16 B / 4352 B | 236.3 / 261.7 | 127.7 / 134.1 |

**Code size.** The kernel config buffer holds both ERISCs' binaries, so the limit is on E0 + E1 together. Measured
as the largest E0 plus the largest E1 over the 48 + 48 VC1 kernels, in loadable bytes:

| | -O3 | -O2 | -Os |
| --- | ---: | ---: | ---: |
| s79mc | 29,604 (does not fit) | 26,064 (fits, ~0.5 KB spare) | 24,004 |
| v3a | 29,704 | 26,152 | |

Offline builds of the 96 VC1 kernels (now in `offline/cmds.json`, hashes in `offline/vc1_hashes.txt`) match the
hardware-built ones exactly at -O2.

**What it shows:**
- **Correction (§86): E0's CPU binds these rows, not slots.**
  - A VC1 E0 has 4 VC0 and 3 VC1 sender channels and 2 receiver channels in the same L1.
  - Main's allocator gives the VC0 worker channel 51 slots (every spare slot goes to it, main's rule), the VC0
    forwarding channels 4 each, the VC1 senders 2 each, and the receivers 8 (VC0) and 4 (VC1).
  - I first read the 234–262 cycles/packet as slot-bound: 8 receiver slots at a ~2,150-cycle credit loop allow ~270.
    That was wrong. v3l (§86) runs the same rows at 199–227, so the credit loop on these 1-hop flows is shorter than
    that, and E0's per-pass work over 7 channels is what binds. The VC1 channels' free-slot reads are not grouped and
    are used at once.
- **-O3 needs about 3 KB less code to fit with VC1, and -O2 fits with only ~0.5 KB to spare.** A v3 design whose code
  does not grow with the channel count could let VC1 routers build at -O2 or -O3 instead of -Os.

## 85. Phase 1, round 2: v3u (one runtime table for every E0 sender channel) is correct and stable across opt levels, but ~30 slower on VC0 E0-bound rows; faster on VC1

**v3a** (consolidation reference, `variants/v3a`, `consolidation/V3A_NOTES.md`). It is s79mc as the new v3 kernel file,
with no experiment flags, DRISC code, profiler hooks or study-added hints. On all 440 VC0 configs at -O3, -O2 and -Os
it compiles to s79mc-without-hints instruction for instruction, so its -O3 numbers are §83's nohint row. It builds for
all 96 VC1 configs. v1 and v2 compile unchanged with its header edits.

**v3u** (agent design, `variants/v3u`, `consolidation/V3U_NOTES.md`). On two-ERISC Blackhole E0s with counter credits
(168 of 208 logged E0 configs: 1D, ring, 2D and 2D-VC1), E0 runs one `noinline` program instead of v3a's main loop.
- **Table:** every sender channel (worker, forwarding, VC0 and VC1) is a 32 B entry in a stack array.
- **Each pass:**
  - one grouped read of all free-slot registers, through a small documented inline-asm load-group primitive;
  - a work mask (unsent | pending completions), kept in a register;
  - a `ctz` loop visiting only channels with work, through one step body per VC.
- **Worker connection check:** still every pass, but its semaphore is read before the visits.
- **Unchanged:** E1 is identical to v3a.
- **Code size:** the VC1 E0 + E1 is 25,496 B at -O3 (fits the 26,624 B buffer; v3a 29,704 does not).

**Runs** (jobs 218, 220, 221, 217; `p1r2_runs.sh`, `p1vc1_runs.sh`; `compare.py`).
- **v3u validation passes:**
  - RELAXED: stream 10/10, mesh 96/96, 1D sanity 181/181;
  - STRICT: the same, plus the flushed-signal suites;
  - VC1 multi-mesh sanity at -Os, -O2 and -O3 (34 test completions over both ranks each).
- **v3a's references:** -O2, -Os and the hook probe.

| Row (cycles/packet) | v3a -O3 | v3a -O2 | v3a -Os | v3a hook | v3u -O3 | v3u -O2 | v3u -Os | v3u hook |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E/W write 16 B | 121.6 | 123.1 | 146.9 | 126.1 | 153.8 | 152.3 | 154.3 | 171.8 |
| N/S write 16 B | 127.2 | 129.3 | 156.2 | 131.9 | 153.0 | 151.2 | 156.6 | 170.9 |
| Forwarded write 16 B | 111.6 | 117.3 | 145.7 | 117.3 | 141.0 | 141.7 | 146.0 | 152.4 |
| Mesh 1-hop E/W write 4352 | 132.3 | 133.3 | 147.2 | 132.5 | 151.7 | 150.6 | 154.5 | 169.4 |
| Mesh 3-hop E/W write 4352 | 146.8 | 150.0 | 170.4 | 149.5 | 155.0 | 158.2 | 170.9 | 168.7 |
| Mesh multicast write 4352 | 140.2 | 143.0 | 166.5 | 143.2 | 167.9 | 170.2 | 176.5 | 179.7 |
| E/W scatter / fused 4352 | 164.9 / 153.9 | 170.6 / 159.6 | 198.9 / 190.0 | | 164.9 / 153.9 | 170.7 / 159.7 | 199.4 / 190.5 | 168.7 / 168.2 |
| Suite means vs v3a -O3: 4352 / mix / flushmix / 1D | | +3.5 / +3.5 / +2.2 / +2.6 | +26.1 / +28.7 / +23.4 / +19.2 | +1.8 / +3.2 / +1.4 / +0.9 | +8.6 / +19.2 / +11.8 / +3.4 | +11.7 / +19.6 / +11.8 / +5.8 | +28.7 / +30.5 / +25.1 / +22.5 | +17.6 / +32.6 / +25.9 / +6.5 |

| VC1 router, two 2x2 meshes (cycles/packet) | s79mc -Os | v3a -Os | v3a -O2 | v3u -Os | v3u -O2 | v3u -O3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1-hop E/W write 16 B / 4352 B | 233.7 / 258.7 | 233.7 / 258.5 | 234.5 / 253.9 | 227.6 / 250.9 | 221.1 / 246.9 | 219.0 / 245.3 |
| 1-hop N/S write 16 B / 4352 B | 236.0 / 267.1 | 236.2 / 266.3 | 236.2 / 261.9 | 230.9 / 255.7 | 225.1 / 249.3 | 221.9 / 246.7 |
| Intra-mesh multicast 4352 (noisy) | 297.5 | 318.5 | 290.8 | 323.3 | 294.2 | 302.5 |

**What it shows:**
- **v3u's E0 no longer depends on the opt level, but it runs at about v3a's -Os speed** at every level (E/W 16 B:
  153.8 / 152.3 / 154.3). It is stable, but on VC0 configs it is ~30 slower than v3a at -O3/-O2 and about even at -Os.
- **The hook moves v3u far more than v3a** (+18 vs +4.5 on E/W 16 B). The cold call lands in the hot program, and E0
  then becomes the bound on the E1-bound rows too. So a uniform table is not automatically robust: what the hot path
  depends on matters, not just how big it is.
- **On VC1 routers v3u is faster:** −6 to −15 at the same level, and it can build at -O3. With 7 sender channels,
  visiting only channels with work saves more than the table costs, even on slot-bound rows.
- **Next:** v3u2 (the agent's second iteration, aiming at the ~30 on VC0 while keeping channel-count-independent code
  size). The other two designs, v3l (lean specialized E0) and v3e1 (E1), are still being written.

## 86. Phase 1, round 3: v3l (lean specialized E0): best worker path and best VC1 so far; forwarding visits through a call cost 19–41

**v3l** (agent design, `variants/v3l`, `consolidation/V3L_NOTES.md`). On every non-speedy two-ERISC E0 (168 of 208
logged configs), a new `noinline` program `e0::run_main_loop` owns E0's state; v3a's init, router commands and outer
loop are kept. E1 is v3a's on every config.
- **Each pass:**
  1. VC0's fenced group of four free-slot reads.
  2. The worker channel, inline and compile-time specialized. The connection poll is one load and one compare;
     open and close are a `noinline` call.
  3. One test for all VC0 forwarding channels: the AND of their free-slot values plus one in-flight word.
  4. Only if that test fails, each forwarding channel with work is visited through one shared `noinline` body that
     takes the channel index at run time, with state in a per-channel record.
  5. VC1 the same way, after a second read group.
- **Size:** the 2D E/W hot loop is 92 instructions with 8 stack loads (v3a: 226 with 33). VC1 E0 + E1 is 26,096 B at
  -O3 (fits, 528 B spare).

**Runs** (jobs 222, 223; `p1r3_runs.sh`).
- **Validation passes:**
  - RELAXED: stream 10/10, mesh 96/96, 1D sanity 181/181;
  - STRICT: the same, plus the flushed-signal suites;
  - VC1 multi-mesh sanity at -Os, -O2 and -O3.

| Row (cycles/packet) | v3a -O3 / -O2 / -Os | v3l -O3 / -O2 / -Os | v3l hook | v3u -O3 |
| --- | --- | --- | ---: | ---: |
| E/W write 16 B | 121.6 / 123.1 / 146.9 | **117.2** / 120.4 / 146.1 | 120.8 | 153.8 |
| N/S write 16 B | 127.2 / 129.3 / 156.2 | **124.3** / 128.7 / 156.7 | 124.4 | 153.0 |
| E/W mix of 4 sizes | 122.2 / 124.4 / 147.0 | **120.1** / 123.3 / 146.2 | 124.0 | 154.1 |
| Mesh 1-hop E/W write 4352 | 132.3 / 133.3 / 147.2 | **129.6** / 131.2 / 146.3 | 131.8 | 151.7 |
| Forwarded write 16 B | 111.6 / 117.3 / 145.7 | 130.9 / 132.1 / 146.1 | 132.9 | 141.0 |
| Mesh 3-hop E/W write 4352 | 146.8 / 150.0 / 170.4 | 172.3 / 169.6 / 172.6 | 173.4 | 155.0 |
| Mesh multicast write 4352 | 140.2 / 143.0 / 166.5 | 181.5 / 182.4 / 185.5 | 183.8 | 167.9 |
| E/W scatter / fused 4352 | 164.9 / 153.9 | 165.0 / 154.0 | 168.8 / 160.7 | 164.9 / 153.9 |
| Linear 3-hop E/W write 4352 | 145.7 / 149.6 / 169.4 | 150.1 / 153.2 / 169.7 | 151.3 | 150.2 |
| Suite means vs v3a -O3: 4352 / mix / flushmix / flushmix4 / 1D | | +7.1 / +2.3 / −1.2 / +0.7 / +0.7 | +9.2 / +4.5 / +0.4 / +3.0 / +1.8 | +8.6 / +19.2 / +11.8 / +8.2 / +3.4 |

| VC1 router (cycles/packet) | v3a -Os / -O2 | v3u -Os / -O3 | v3l -Os / -O2 / -O3 |
| --- | --- | --- | --- |
| 1-hop E/W write 16 B | 233.7 / 234.5 | 227.6 / 219.0 | **202.3 / 200.5 / 199.1** |
| 1-hop E/W write 4352 | 258.5 / 253.9 | 250.9 / 245.3 | **227.6 / 225.7 / 224.5** |
| 1-hop N/S write 16 B | 236.2 / 236.2 | 230.9 / 221.9 | **217.1 / 203.1 / 200.8** |
| Intra-mesh multicast 4352 (noisy) | 318.5 / 290.8 | 323.3 / 302.5 | 297.4 / 289.5 / 287.0 |

**What it shows:**
- **The worker path is the best E0 so far.** E/W 16 B at 117.2 beats s79m's 121.7, and the hook probe costs it 3.6, as
  for v3a. A specialized inline worker visit with a small hot-path state works.
- **A call per forwarding visit is too expensive.** It costs 19–41 on forwarded, 3-hop and multicast rows. v3u's
  inlined runtime body lost less there (141 / 155 / 168).
- **-Os still costs v3l's E0 ~29**, as for v3a; v3u is flat but slow. No design yet has a fast hot path that is the same
  at -O3 and -Os.
- **VC1 routers gain 31 cycles/packet** (E/W 16 B 233.7 → 202.3 at -Os). So E0 binds there (correction in §84).
- **Next:** v3l2 (inline, compile-time specialized forwarding visits behind the same combined test, VC1 within the
  buffer at -O3, and the -Os cost explained) and v3u2 (a runtime-indexed forwarding body that costs what v3a's inline
  visits cost, possibly with v3l's worker path). v3e1 (E1) is still being written.

## 87. Phase 1, round 4: v3e1 (E1 program rewrite): forwarding and multicast faster, local delivery slower, VC1 large packets much slower; and a VC1 admission bug in v2

**v3e1** (agent design, `variants/v3e1`, `consolidation/V3E1_NOTES.md`). In two-ERISC mode, E1 runs its own program
(`e1_kernel_main`): its own part of init (same order, same ERISC syncs), its own downstream connections, and a new
receive loop. It covers 2D VC0, 1D line and VC1 including intermesh routers; ring, speedy, single-ERISC, VC2, UDM and
mux keep v3a's code. E0 is v3a's on every config.
- **Packets-received register:** read only when a cached count of present packets runs out. Still decremented once
  per packet.
- **Slot address:** a running pointer over the compile-time ring.
- **Forwarding admission:** a cached downstream free-slot count, re-read only at zero (§80's idea, first time run).
- **The forward itself:** through E1-owned per-downstream cursors with compile-time command buffers.
- **Completion check:** exact, over every slot of a batch (four trid statuses plus TXQ CMD/STATUS in one grouped
  read). v3a checks only the last slot.
- **VC1 builds:** local delivery, admission and forwarding go through one out-of-line body shared by both receiver
  channels.
- **Code size:** VC1 E0 + E1 is 26,060 B at -O3 (fits).

**VC1 admission bug in v2/v3a (study-introduced).**
- Main's `admit_2d_dispatch` asks each downstream interface for space (`downstreams_have_space`), so each receiver
  channel checks its own VC.
- The study's compile-time version (§53) reads `vc0_local_free_slots_stream_id<…>` for every receiver channel. So
  on a VC1 router, the VC1 receiver channel admits a forward when VC0's downstream has space, whatever VC1's
  downstream holds.
- s79m and v3a have it. The VC1 sanity runs pass, presumably because VC1 never backs up there.
- v3e1 uses each VC's own streams. Any v3 must carry the fix.

**Runs** (jobs 224, 225; `p1r4_runs.sh`).
- **Validation passes:**
  - RELAXED: stream 10/10, mesh 96/96, 1D sanity 181/181;
  - STRICT: the same, plus the flushed-signal suites;
  - VC1 multi-mesh sanity at -Os, -O2 and -O3.

| Row (cycles/packet) | v3a -O3 / -O2 / -Os | v3e1 -O3 / -O2 / -Os |
| --- | --- | --- |
| Mesh 3-hop E/W write 4352 | 146.8 / 150.0 / 170.4 | **144.3 / 146.9 / 163.0** |
| Mesh multicast write / scatter 4352 | 140.2 / 143.0 / 166.5; 185.1 | **136.6 / 138.7 / 158.2; 178.5** |
| Linear multicast write / scatter | 149.3 / 196.2 | **144.3 / 188.7** |
| Unidir linear multicast write / scatter | 196.3 / 255.4 | **188.6 / 228.1** |
| Linear 3-hop E/W write 4352 | 145.7 / 149.6 / 169.4 | 146.5 / 149.0 / **161.6** |
| E/W atomic / N/S atomic | 128.9 / 141.4 | 136.6 / 146.5 |
| E/W scatter / fused 4352 | 164.9 / 153.9 | 168.9 / 157.0 |
| 3-hop scatter / linear 3-hop scatter | 164.9 / 167.1 | 170.9 / 174.6 |
| Chunk-4 fused flushed E/W / N/S | 153.2 / 160.8 | 162.5 / 173.2 |
| E/W / N/S write 16 B | 121.6 / 127.2 | 122.6 / 130.7 |
| Suite means vs v3a -O3: 4352 / mix / flushmix / flushmix4 / 1D | | +0.9 / +2.7 / +2.9 / +7.4 / −0.8 (hook: +2.0 / +5.1 / +4.6 / +8.6 / +0.7) |

| VC1 router (cycles/packet) | v3a -Os / -O2 | v3e1 -Os / -O2 / -O3 |
| --- | --- | --- |
| 1-hop E/W write 16 B / 4352 | 233.7 / 234.5; 258.5 / 253.9 | 237.4 / 236.4 / 232.4; 255.3 / 321.4 / 332.2 |
| 1-hop N/S write 16 B / 4352 | 236.2 / 236.2; 266.3 / 261.9 | 245.7 / 236.6 / 233.6; 299.3 / 339.0 / 331.8 |
| Intra-mesh multicast 4352 | 318.5 / 290.8 | 415.7 / 349.0 / 349.0 |

**What it shows:**
- **The forwarding work pays.** The cached downstream credits and E1-owned forwarding state cut multicast and
  multi-hop at every level (−4 to −8 at -O3/-Os, −27 on unidirectional linear multicast scatter). That is the first
  measurement of §80's idea.
- **Local delivery got slower** (atomic +5 to +8, scatter/fused +2 to +6, chunk-4 flushed +9 to +12). Suspects: the
  exact completion check (six grouped reads per batch) and the loop shape around local delivery. Under review.
- **VC1 large packets are much slower** (E/W 4352: 255–332 vs 254–259). The benchmarks are intra-mesh, so this is
  VC0 traffic on routers that also carry a VC1 receiver channel. In VC1 builds both channels go through shared
  out-of-line bodies.
- **Next:** v3e1b, keeping the forwarding gains and the VC1 fix and recovering local delivery and VC1.

## 88. Phase 1, round 5: the whole -Os penalty is E1's; v3l2 (v3l's worker path + inline forwarding) lands near v3a and fits VC1 at -O3

**Which ERISC makes -Os slow** (job 226, `p1r5_runs.sh c`). The test uses ELF injection, so the source is the same in
every build. v3l was swapped in and its -O3 kernels built in a fresh cache. Then two mixed sets from the offline
builds were injected:
- E0 at -Os with E1 at -O3;
- E0 at -O3 with E1 at -Os.

No router kernel was recompiled, and the cache still held all 80 injected ELFs after the runs. The sets' -O3 halves
are identical to the offline v3l -O3 build on all 536 kernels.

| Row (cycles/packet) | all -O3 | E0 -Os, E1 -O3 | E0 -O3, E1 -Os | all -Os |
| --- | ---: | ---: | ---: | ---: |
| E/W write 16 B | 117.2 | **114.3** | 146.1 | 146.1 |
| N/S write 16 B | 124.3 | 124.4 | 156.5 | 156.7 |
| Forwarded write 16 B | 130.9 | **125.3** | 146.0 | 146.1 |
| Mesh 1-hop E/W write 4352 | 129.6 | 129.0 | 146.3 | 146.3 |
| Mesh 3-hop E/W write 4352 | 172.3 | **166.5** | 173.4 | 172.6 |
| E/W scatter / fused 4352 | 165.0 / 154.0 | 165.2 / 154.1 | 199.4 / 190.3 | 199.5 / 190.4 |
| Suite means vs all -O3: 4352 / mix | | −0.6 / −2.0 | +21.0 / +26.6 | +21.1 / +26.7 |

- **E1 accounts for all of the -Os penalty** (~20%, production's level whenever VC1 is active).
- **v3l's E0 is not -Os-sensitive at all;** it is faster at -Os on worker and forwarding rows.
- This also explains why v3a and v3l land within a cycle of each other at -Os despite very different E0 code: E1
  binds there.
- -Os robustness is therefore an E1 design goal (passed to the v3e1 iteration).

**v3l2** (agent design, `variants/v3l2`, `consolidation/V3L_NOTES.md` §7). It keeps v3l's worker path and its one
combined test per VC.
- **Forwarding visits:** when the combined test fails, each forwarding channel with work runs compile-time
  specialized code inline in the pass, with no call and no runtime index.
- **Out of line, off the pass path:** the upstream-router credit write, the busy spin, connection open/close and the
  static-connection wait.
- **E0 init sets up only the sender side.** The receiver-side adapters, semaphores and headers are E1's; E1's code is
  unchanged. E0's deepest stack is 800 B (v3a 1,408).
- **Code size:** VC1 E0 + E1 is 25,836 B at -O3 (fits).

**Runs** (jobs 227, 228).
- **Validation passes:**
  - RELAXED: stream 10/10, mesh 96/96, 1D sanity 181/181;
  - STRICT: the same, plus the flushed-signal suites;
  - VC1 sanity at -Os, -O2 and -O3.

| Row (cycles/packet) | v3a -O3 / -Os | v3l -O3 | v3l2 -O3 / -O2 / -Os | v3l2 hook |
| --- | --- | ---: | --- | ---: |
| E/W write 16 B | 121.6 / 146.9 | 117.2 | 120.9 / 120.9 / 146.0 | 124.7 |
| N/S write 16 B | 127.2 / 156.2 | 124.3 | 125.3 / 128.7 / 156.4 | 126.0 |
| Forwarded write 16 B | 111.6 / 145.7 | 130.9 | 119.5 / 117.5 / 145.9 | 124.0 |
| Mesh 1-hop E/W write 4352 | 132.3 / 147.2 | 129.6 | 131.5 / 131.8 / 146.1 | 132.6 |
| Mesh 3-hop E/W write 4352 | 146.8 / 170.4 | 172.3 | 149.9 / 152.7 / 170.7 | 154.0 |
| Mesh multicast write 4352 | 140.2 / 166.5 | 181.5 | 145.4 / 145.1 / 166.5 | 152.2 |
| Suite means vs v3a -O3: 4352 / mix / flushmix / flushmix4 / 1D | | +7.1 / +2.3 / −1.2 / +0.7 / +0.7 | +0.7 / +0.8 / −0.4 / +1.4 / +0.2 | hook vs v3l2: +2.5 / +2.8 / +1.8 / +2.9 / +1.0 |

| VC1 router (cycles/packet) | v3a -Os | v3l -Os / -O3 | v3l2 -Os / -O2 / -O3 |
| --- | ---: | --- | --- |
| 1-hop E/W write 16 B / 4352 | 233.7 / 258.5 | 202.3 / 199.1; 227.6 / 224.5 | 214.1 / 202.4 / 200.7; 239.7 / 224.4 / 221.5 |
| 1-hop N/S write 16 B / 4352 | 236.2 / 266.3 | 217.1 / 200.8; 241.7 / 226.2 | 224.1 / 203.5 / 202.9; 248.6 / 227.6 / 224.6 |

**What it shows:**
- **Inline forwarding visits recover most of v3l's forwarding loss** (multicast 181.5 → 145.4, 3-hop 172.3 → 149.9),
  but forwarding stays 3–8 above v3a.
- **The worker gain mostly goes back** (E/W 16 B 117.2 → 120.9). The agent attributes this to the forwarding
  channels' state now competing for registers in the same pass, so the worker's counts sit on the stack.
- **On VC0 rows v3l2 is roughly even with v3a.** Its gains are structural: it fits VC1 at -O3, VC1 rows are 30–33
  faster than v3a at -O2/-O3, E0 init is smaller, and it is as stable as v3a under the hook (+1 to +3).

## 89. Phase 1, round 6: v3u2 (runtime forwarding table, specialized worker): best worker rows at -O3, but not at -O2; multicast +29

**v3u2** (agent design, `variants/v3u2`, `consolidation/V3U_NOTES.md` §12–13). It follows the agent's own accounting
of v3u's ~30-cycle loss: ~22 from the runtime channel index and the nested visit loop, and ~11 from load groups
serialized by a spilled constant and by the size's zero-extension.
- **Worker channel:** inline through the step body with a compile-time index. The connection check is one compare.
  The semaphore is read in the send's first load group.
- **Forwarding channels:** one combined test, then one visit per loop iteration in rounds, through a per-VC working
  copy of the channel state. Swapping the copy costs ~17 instructions when the visited channel changes.
- **Code size:** 2 step bodies on VC0 configs, 3 on VC1. VC1 E0 + E1 is 26,116 B at -O3 (fits).
- **Caveat:** the agent chose among source variants that differed only in what GCC kept in registers.

**Runs** (jobs 229, 230).
- **Validation passes:**
  - RELAXED and STRICT validation (stream, mesh, 1D sanity, flushed-signal suites);
  - VC1 sanity at -Os, -O2 and -O3.

| Row (cycles/packet) | v3a -O3 / -O2 / -Os | v3l2 -O3 / -O2 | v3u2 -O3 / -O2 / -Os | v3u2 hook |
| --- | --- | --- | --- | ---: |
| E/W write 16 B | 121.6 / 123.1 / 146.9 | 120.9 / 120.9 | **116.8** / 121.2 / 146.0 | 120.2 |
| N/S write 16 B | 127.2 / 129.3 / 156.2 | 125.3 / 128.7 | **124.3** / 128.6 / 156.5 | 124.4 |
| E/W mix of 4 sizes | 122.2 / 124.4 / 147.0 | 121.5 / 123.7 | **118.6** / 123.4 / 146.1 | 122.9 |
| Forwarded write 16 B | 111.6 / 117.3 / 145.7 | 119.5 / 117.5 | 112.8 / 117.3 / 145.9 | 119.9 |
| Mesh 1-hop E/W write 4352 | 132.3 / 133.3 / 147.2 | 131.5 / 131.8 | 131.0 / 132.1 / 146.1 | 131.0 |
| Mesh 3-hop E/W write 4352 | 146.8 / 150.0 / 170.4 | 149.9 / 152.7 | 151.2 / 154.8 / 170.9 | 152.4 |
| Mesh multicast write 4352 | 140.2 / 143.0 / 166.5 | 145.4 / 145.1 | 169.6 / 174.1 / 187.7 | 175.5 |
| Suite means vs v3a -O3: 4352 / mix / flushmix / flushmix4 / 1D | | +0.7 / +0.8 / −0.4 / +1.4 / +0.2 | +3.3 / −1.1 / −0.8 / −0.4 / +0.2 | hook vs v3u2: +2.1 / +3.0 / +1.9 / +4.4 / +0.8 |

| VC1 router (cycles/packet) | v3a -Os | v3l2 -Os / -O2 / -O3 | v3u2 -Os / -O2 / -O3 |
| --- | ---: | --- | --- |
| 1-hop E/W write 16 B / 4352 | 233.7 / 258.5 | 214.1 / 202.4 / 200.7; 239.7 / 224.4 / 221.5 | 224.8 / 206.9 / 203.9; 254.2 / 233.5 / 231.8 |
| 1-hop N/S write 16 B / 4352 | 236.2 / 266.3 | 224.1 / 203.5 / 202.9; 248.6 / 227.6 / 224.6 | 226.6 / 208.0 / 205.2; 252.9 / 231.2 / 228.2 |

**What it shows:**
- **At -O3 v3u2 has the best worker rows of any design** (E/W 16 B 116.8), and forwarded 16 B is back at v3a's level.
- **The worker gain does not survive -O2** (121.2, v3a's level). It came from one allocation outcome, the kind of
  dependence the consolidation is meant to remove.
- **Multicast is +29 at every level.** That is the working-copy swap when consecutive forwarding visits alternate
  channels, and E0 binds multicast even at -Os.
- **Next:** v3u3 must remove the swap cost and prefer a worker path that is the same at -O3 and -O2.

## 90. Phase 1, round 7: v3u3 fixed v3u2's -O2 and swap costs, then hung on VC1 at -Os: ruled out

**v3u3** (agent design, `variants/v3u3`, `consolidation/V3U_NOTES.md` §14). Two changes from v3u2:
- **Free-slot group read:** feeds three plain values (worker idle bit, forwarding idle bits, AND of the forwarding
  values) instead of an array. At -O2 the array went to the stack.
- **Forwarding state:** read and written in place in the per-channel table by runtime index. There is no working
  copy and no swap.

Offline, the worker, forwarding, idle and VC1 hot paths compile line for line the same at -O3 and -O2 (register
names aside). VC1 E0 + E1 is 25,832 B at -O3.

**Runs** (jobs 231, 232).
- **Validation:** RELAXED and STRICT pass (stream, mesh, 1D sanity, flushed-signal suites).
- **VC1 at the default level (-Os) hung.** Both ranks stopped after 9 test completions, in
  `MultiMeshAllToAll_size_4096_ntype_unicast_write` (data-checked sanity), and the run hit its 600 s timeout. The
  broker then held the device for an Ethernet/fabric fault and it recovered by itself after ~4 minutes; no reset was
  needed.
- v3u2 passed the same VC1 tests at all three levels, so the cause is in v3u3's two changes. A root-cause review is
  offline, with no new variant.

| Row (cycles/packet) | v3a -O3 / -O2 | v3u2 -O3 / -O2 | v3u3 -O3 / -O2 | v3u3 hook |
| --- | --- | --- | --- | ---: |
| E/W write 16 B | 121.6 / 123.1 | 116.8 / 121.2 | **116.1 / 119.0** | 119.3 |
| N/S write 16 B | 127.2 / 129.3 | 124.3 / 128.6 | 124.3 / 128.7 | 124.3 |
| Forwarded write 16 B | 111.6 / 117.3 | 112.8 / 117.3 | 120.9 / 121.3 | 124.6 |
| Mesh 1-hop E/W write 4352 | 132.3 / 133.3 | 131.0 / 132.1 | 130.5 / 131.8 | 131.9 |
| Mesh 3-hop E/W write 4352 | 146.8 / 150.0 | 151.2 / 154.8 | 154.2 / 156.8 | 155.4 |
| Mesh multicast write 4352 | 140.2 / 143.0 | 169.6 / 174.1 | 161.4 / 162.8 | 165.0 |

**What it shows:**
- **Both fixes worked as designed.** The worker lead now mostly survives -O2 (119.0 vs v3a 123.1). Multicast
  recovered 8 of v3u2's +29, but forwarding paid ~+9.
- **A design that hangs is out (the study's rule).** v3u3 and any combination using its E0 (v3u3e1b) are not run
  again.
- **Remaining E0 candidates:** v3l2, and v3u2 if multicast is acceptable (it is not: +29).

## 91. Phase 1, round 8: v3e1b (E1 iteration): -Os penalty roughly halved, VC1 recovered and better than v3a, forwarding gains kept

**v3e1b** (agent design, `variants/v3e1b`, `consolidation/V3E1_NOTES.md` §9). It keeps v3e1's E1 program and VC1
admission fix, with five changes:
- **Cheaper exact completion check.** Each retired slot records which command buffer reads its data. A batch with no
  slot that reads data (atomics, inline writes) needs no transaction-id read. Otherwise the newest reading slot's ID
  is read, plus a second read only if an older slot used the other command buffer. The TX queue is read after.
- **One credit of 4** for a uniform full batch.
- **Plain local writes issued directly,** not through the send-type jump table.
- **VC1:** channel 0 runs inline code, as on a VC0-only router; only the VC1 channel uses the shared out-of-line
  bodies.
- **A slot-add fix for 2-slot rings:** v3a's `wrap_increment_n<2>` ignores the increment; no logged config hits it.

The agent's -O3/-Os per-packet path counts for a 16 B local write: mid-W E1 72.5 / 82.3 instructions (v3a
83.8 / 104.5), 1D 75.3 / 80.0 (v3a 78.5 / 104.0). What -Os stops doing is loop unrolling, block duplication for jump
threading, and frequency-aware allocation; the new structure depends on none of them. E0 is v3a's on every config.

**Runs** (jobs 233, 234).
- **Validation passes:**
  - RELAXED and STRICT validation (stream, mesh, 1D sanity, flushed-signal suites);
  - VC1 sanity at -Os and -O2.
- VC1 at -O3 was not run: with v3a's E0 the pair is 29,136 B.

| Row (cycles/packet) | v3a -O3 / -O2 / -Os | v3e1 -O3 / -Os | v3e1b -O3 / -O2 / -Os | v3e1b hook |
| --- | --- | --- | --- | ---: |
| E/W write 16 B | 121.6 / 123.1 / 146.9 | 122.6 / 149.6 | 121.4 / – / 147.0 | 125.9 |
| N/S write 16 B | 127.2 / 129.3 / 156.2 | 130.7 / 153.3 | 126.8 / – / 147.0 | 131.8 |
| Forwarded write 16 B | 111.6 / 117.3 / 145.7 | 114.1 / 146.3 | 112.0 / – / **130.6** | 113.4 |
| Mesh 3-hop E/W write 4352 | 146.8 / 150.0 / 170.4 | 144.3 / 163.0 | **143.4** / – / **150.7** | 144.7 |
| Mesh multicast write 4352 | 140.2 / 143.0 / 166.5 | 136.6 / 158.2 | **137.5** / – / **148.7** | 139.0 |
| E/W scatter / fused 4352 | 164.9 / 153.9; −Os 198.9 / 190.0 | 168.9 / 157.0 | 167.7 / 155.2; −Os **179.5 / 169.8** | 168.4 / 158.0 |
| E/W / N/S atomic | 128.9 / 141.4 | 136.6 / 146.5 | 132.4 / **135.8** | |
| Chunk-4 fused flushed E/W / N/S | 153.2 / 160.8 | 162.5 / 173.2 | 158.3 / 171.9 | |
| Unidir linear multicast write / scatter | 196.3 / 255.4 | 188.6 / 228.1 | 189.0 / 233.8 | |
| Suite means vs v3a -O3: 4352 / mix / flushmix / flushmix4 / 1D | -O2 +3.5 / +3.5 / +2.2 / +4.1 / +2.6; -Os +26.1 / +28.7 / +23.4 / +31.6 / +19.2 | -O3 +0.9 / +2.7 / +2.9 / +7.4 / −0.8 | -O3 +0.5 / +0.6 / −0.3 / +4.0 / −0.9; -O2 +1.7 / +1.7 / −0.3 / +2.0 / +1.3; **-Os +11.5 / +17.4 / +10.8 / +11.1 / +7.7** | |

| VC1 router (cycles/packet) | v3a -Os / -O2 | v3e1 -Os / -O2 | v3e1b -Os / -O2 |
| --- | --- | --- | --- |
| 1-hop E/W write 16 B / 4352 | 233.7 / 234.5; 258.5 / 253.9 | 237.4 / 236.4; 255.3 / 321.4 | 227.0 / 232.8; **246.2 / 247.0** |
| 1-hop N/S write 16 B / 4352 | 236.2 / 236.2; 266.3 / 261.9 | 245.7 / 236.6; 299.3 / 339.0 | 232.0 / 233.0; **250.3 / 248.6** |
| Intra-mesh multicast 4352 | 318.5 / 290.8 | 415.7 / 349.0 | 290.3 / 310.5 |

**What it shows:**
- **The E1 structure, not the opt level, was most of the -Os penalty.** v3e1b's -Os cost is about half of v3a's on
  every suite. E1-bound rows at -Os are 15–20 better than v3a's.
- **At -O3 and -O2 v3e1b is level with or better than v3a** on suite means. It keeps most of v3e1's forwarding gains
  and fixes its local-delivery and VC1 regressions, except scatter (+3) and chunk-4 flushed signals (+5 to +11).
- **E/W 16 B at -Os is still 147.** In this build E0 is v3a's, whose -Os behavior has not been measured on its own.
  The v3l2 + v3e1b combination (`variants/v3l2e1b`, an exact composition: E0 identical to v3l2 and E1 identical to
  v3e1b on every config; VC1 E0 + E1 25,268 B at -O3) is the next run.

## 92. Phase 1, round 9: v3l2 + v3e1b removes most of the -Os penalty on VC0 routers, then hangs on VC1 at -Os like v3u3

**v3l2e1b** (`variants/v3l2e1b`). A three-way merge of v3l2 and v3e1b over v3a: it merged cleanly, and the only
differing file is the router. It is an exact composition: E0 is identical to v3l2 on all 208 E0 configs, and E1 is
identical to v3e1b on all 328 E1 configs, at -O3. VC1 E0 + E1 is 25,268 B at -O3. (v3u3 + v3e1b also built as an
exact composition but is not run; see §90.)

**Runs** (jobs 241, 242).
- **Validation passes:** RELAXED and STRICT (stream, mesh, 1D sanity, flushed-signal suites).

| Row (cycles/packet) | v3a -O3 / -O2 / -Os | v3e1b -Os | v3l2e1b -O3 / -O2 / -Os | v3l2e1b hook |
| --- | --- | ---: | --- | ---: |
| E/W write 16 B | 121.6 / 123.1 / 146.9 | 147.0 | 120.8 / 119.3 / **122.6** | 124.4 |
| N/S write 16 B | 127.2 / 129.3 / 156.2 | 147.0 | **121.3 / 119.5 / 127.4** | 124.4 |
| Forwarded write 16 B | 111.6 / 117.3 / 145.7 | 130.6 | 119.4 / 118.4 / 130.8 | 124.0 |
| Mesh 1-hop E/W write 4352 | 132.3 / 133.3 / 147.2 | 147.1 | 131.7 / 131.2 / 131.5 | 130.1 |
| Mesh 3-hop E/W write 4352 | 146.8 / 150.0 / 170.4 | 150.7 | 146.9 / 147.1 / 152.1 | 149.9 |
| Mesh multicast write 4352 | 140.2 / 143.0 / 166.5 | 148.7 | 143.6 / 143.3 / 153.6 | 149.6 |
| E/W scatter / fused 4352 | 164.9 / 153.9; −Os 198.9 / 190.0 | 179.5 / 169.8 | 167.7 / 155.2; −Os 179.1 / 169.8 | 168.6 / 158.4 |
| Suite means vs v3a -O3: 4352 / mix / flushmix / flushmix4 / 1D | -Os +26.1 / +28.7 / +23.4 / +31.6 / +19.2 | +11.5 / +17.4 / +10.8 / +11.1 / +7.7 | -O3 +1.4 / +0.7 / −0.8 / +1.9 / −0.5; -O2 +2.0 / +0.7 / −1.2 / +1.7 / +1.9; **-Os +8.0 / +8.8 / +4.5 / +6.7 / +8.3** | +1.4 / +2.4 / −0.8 / +1.7 / +2.3 |

- **VC1 at the default level (-Os) hung.** Rank 1 finished `MultiMeshAllToAll_size_4096_ntype_unicast_write`; rank 0
  (mesh 0, chips 0–3) never did, after 9 completions. The run hit its 600 s timeout and the tree was restored. This is
  the same place and the same rank as v3u3's hang (§90).

**The two VC1 hangs** (all at the default level, -Os, `fabric_exp/vc1/vc1_split.yaml`):

| Run | E0 | E1 | Result |
| --- | --- | --- | --- |
| v3a, v3u2, v3l2 | v3a / v3u2 / v3l2 | v3a's (VC1 admission bug, §87) | passed (34 completions) |
| v3e1, v3e1b | v3a | v3e1 / v3e1b (own-VC admission) | passed |
| v3u3 | v3u3 | v3a's | **hung** |
| v3l2e1b | v3l2 | v3e1b | **hung** |

- **The admission bug is confirmed in the hardware binary.** The VC1 E1 that ran with v3u3 (config
  `10565308837383794061`) reads the downstream free-slot register (offset 1188) only for streams 4, 5 and 6. Streams
  10–13, VC1's downstream registers, are never read, so the VC1 receiver channel admits forwards on VC0's credits.
- **But the admission bug is not the whole story.** v3l2e1b's E1 checks each VC's own streams and still hung.
- Both hangs pair a faster E0 with an E1 and stop at the first 4 KiB all-to-all, which suggests a timing-dependent
  fault that faster routers expose.
- An offline root cause is in progress (three audits: cross-config comparison of the exact binaries, E1 VC1 and
  intermesh paths, and E0/E1 init ownership in the combination). Under the study's rule both configurations are out
  until it is found. No further VC1 runs of new designs until then.

**Root-cause audits (offline, three agents; no device runs):**
- **Init:** v3e1b's E1 does every receiver-side init write that v3l2's E0 dropped, with the same values and in the
  same phase between ERISC syncs. The one write neither does (remote receiver header zeroing) has the same addresses
  as the local zeroing in every logged config.
- **Binaries:** every binary in both hung runs also ran in a passing pairing.
  - v3l2's E0 passed with v3a's E1; v3e1b's E1 passed with v3a's E0.
  - v3a's E1 passed with four E0s and hung only with v3u3's.
  - So neither hang is a fixed property of one program.
- **Confirmed defect:** v3a's/v2's VC1 admission (§87). It explains a v3u3-style hang: a forward overwrites an unsent
  packet in a 2-slot VC1 channel, a packet is lost, and the routers stay alive. That matches the broker seeing every
  ERISC still advancing. It does not apply to v3l2e1b, whose E1 checks each VC's own credits.
- **Robustness hazard in the lean E0s** (v3l2, v3u2, v3u3). The "all slots free" test on 2-slot VC1 channels is one
  bit (`raw & 2`), where v3a uses a masked compare. If an upstream over-fills the channel, the register wraps, the bit
  reads "idle", and the channel is never served again: a permanent stall instead of v3a's wrong-but-moving behavior.
  The fix is v3a's exact predicates (v3l2b, being built offline).
- **Unexplained:** what lost a packet or credit in v3l2e1b. v3l2e1b is the first pairing in which an E1 reads the
  VC1 router-to-router credits that a lean E0 returns, but no defect was found in that path.
  - One shared condition: 4 KiB is the first test where a fabric packet spans two Ethernet packets, under the
    study's keepalive of 1000.
  - Base rate so far: 2 hangs in 24 VC1 runs reaching this test.

**What it shows (VC0 configs):**
- **A -Os-robust E0 plus the restructured E1 removes most of the -Os penalty.** E/W 16 B is 122.6 at -Os against
  v3a's 146.9, and suite means at -Os are +4.5 to +8.8 against v3a's +19 to +32.
- **-O3 and -O2 stay level with v3a,** with N/S small writes 6 better.
- That is the consolidation's stability goal on VC0 routers, pending the VC1 hang.

## 93. Phase 1, round 10: the VC1 hang has a fix (exact idle tests in the lean E0); v3l2be1b passes everything and is the lead v3 candidate

**Root cause, as far as the evidence goes.** Three offline audits are summarized in §92. Two hardware tests follow,
both repeat runs of the VC1 data-checked sanity group (`fabric_exp/vc1/vc1_sanity.yaml`) at the default level (-Os),
stopping at the first hang (`p1rep_runs.sh`, jobs 248–249).

| Router | E0 | E1 | VC1 sanity repeats |
| --- | --- | --- | --- |
| v3a (base rate) | v3a's | v3a's | 10 of 10 clean (all 3 earlier v3a-based VC1 runs also passed) |
| v3l2be1b | v3l2b: v3l2 with v3a's exact idle tests | v3e1b's | 10 of 10 clean (v3l2e1b hung on its only run) |

- **Not a shared fault.** At v3a's speed there is no sign of a below-the-router fault that would hit every design.
- **The lean E0s had a stall mechanism.** v3l2, v3u2 and v3u3 tested "2-slot VC1 channel all free" with one bit.
  When the free-slot register goes out of range (an over-fill), that bit reads "idle" and the channel is never served
  again. v3a's masked compare keeps serving it.
- **v3l2b** (`variants/v3l2b`, `V3L_NOTES.md` §10) uses v3a's exact predicates everywhere. 2D VC0-only and 1D E0s
  are byte-identical to v3l2; VC1 routers pay 1–3 instructions on idle tests.
- **v3l2be1b,** with that E0, passes where v3l2e1b hung.
- **What over-filled a channel in v3l2e1b is still unexplained.** Its E1 checks each VC's own credits. The exact
  test removes the permanent stall either way.
- For v3u3 the over-fill source is the v2/v3a VC1 admission bug (§87), which is still in the tree's v2.

**v3l2be1b** (`variants/v3l2be1b`). An exact composition: E0 identical to v3l2b on every E0 config and E1 identical to
v3e1b on all 328 E1 configs, at -O3, -O2 and -Os. VC1 E0 + E1 is 25,340 B at -O3.

**Full protocol** (jobs 250, 251).
- **Validation passes:**
  - RELAXED: stream 10/10, mesh 96/96, 1D sanity 181/181;
  - STRICT: the same, plus the flushed-signal suites;
  - VC1 sanity at -Os, -O2 and -O3 (34 completions each).

| Row (cycles/packet) | v3a -O3 / -O2 / -Os | v3l2be1b -O3 / -O2 / -Os | v3l2be1b hook |
| --- | --- | --- | ---: |
| E/W write 16 B | 121.6 / 123.1 / 146.9 | 120.8 / 119.3 / **122.6** | 124.4 |
| N/S write 16 B | 127.2 / 129.3 / 156.2 | **121.2 / 119.5 / 127.3** | 124.4 |
| Forwarded write 16 B | 111.6 / 117.3 / 145.7 | 119.4 / 118.3 / 130.5 | 124.2 |
| Mesh 1-hop E/W write 4352 | 132.3 / 133.3 / 147.2 | 131.7 / 131.2 / 131.5 | 130.4 |
| Mesh 3-hop E/W write 4352 | 146.8 / 150.0 / 170.4 | 146.9 / 147.1 / 152.0 | 149.9 |
| Mesh multicast write 4352 | 140.2 / 143.0 / 166.5 | 143.7 / 143.7 / 153.6 | 149.5 |
| E/W scatter / fused 4352 | 164.9 / 153.9; -Os 198.9 / 190.0 | 167.6 / 155.1; -Os 179.2 / 169.6 | 168.6 / 158.5 |
| Linear 3-hop E/W write 4352 | 145.7 / 149.6 / 169.4 | 146.8 / 146.7 / 152.7 | 149.3 |
| Suite means vs v3a -O3: 4352 / mix / flushmix / flushmix4 / 1D | -O2 +3.5 / +3.5 / +2.2 / +4.1 / +2.6; -Os +26.1 / +28.7 / +23.4 / +31.6 / +19.2 | -O3 +1.4 / +0.7 / −0.8 / 0.0 / −0.4; -O2 +2.0 / +0.7 / −1.2 / +3.0 / +1.6; **-Os +8.0 / +8.8 / +4.2 / +6.8 / +7.7** | +1.4 / +2.4 / −0.9 / +2.5 / +2.6 |

| VC1 router (cycles/packet) | v3a -Os / -O2 | v3l2be1b -Os / -O2 / -O3 |
| --- | --- | --- |
| 1-hop E/W write 16 B / 4352 | 233.7 / 234.5; 258.5 / 253.9 | **208.7 / 199.5 / 199.6; 230.7 / 218.4 / 219.1** |
| 1-hop N/S write 16 B / 4352 | 236.2 / 236.2; 266.3 / 261.9 | **211.8 / 201.5 / 201.3; 232.2 / 218.4 / 218.6** |
| Intra-mesh multicast 4352 | 318.5 / 290.8 | 318.3 / 279.5 / 276.9 |

**What it shows:**
- **v3l2be1b meets the consolidation's goals on what we can measure:**
  - level with v3a at -O3 and -O2;
  - about 70% of the -Os penalty gone;
  - as stable as v3a under the hook;
  - 25–40 faster on VC1 routers, which now fit at -O3;
  - the VC1 admission bug fixed;
  - correct in both flush modes and on VC1.
- **Remaining gaps vs v3a at -O3:** forwarded writes +7.8 (v3l2's E0 forwarding visits), multicast +3.5, scatter +2.7.
  These are next.

## 94. Phase 1, round 11: gaps closed (v3l3be1c beats v3a at every level), a mainline review with a latent bug, a measured fragility, and a third VC1 hang

Two tracks ran in parallel from v3l2be1b (user, 2026-10-25): close its gaps, and review it for mainline.

**Gap closing** (agent designs, offline, each confined to one ERISC; `V3L_NOTES.md` §11, `V3E1_NOTES.md` §10).
- **v3l3 (E0):**
  - No calls on the pass's path: the upstream credit write and the TX-queue wait are inline again, as in v3a.
  - The idle-context-switch flag is set only when a send yields; a round that sent is detected by its sent counts.
  - VC0's combined "any forwarding work" test is dropped (VC1 keeps it).
  - **v3l3b** is the same but keeps the combined test on VC0.
- **v3e1c (E1):** scatter, atomic and fused writes are issued directly, without the shared send-type dispatch.
  - Scatter updates the software NoC write counters once per packet instead of once per chunk.
  - **v3e1d** is a control with the scatter change only.
- **Merges** over v3l2be1b with `git merge-file`:
  - v3l3e1c and v3l3be1c are exact compositions: each ERISC's ELF is PT_LOAD-identical to its parent's on every config at -O3, -O2 and -Os.
  - VC1 E0 + E1 at -O3: 25,304 B and 25,360 B (limit 26,624).

**Runs** (jobs 252–257, 263, 264). Every variant passes RELAXED and STRICT validation.

| Row (cycles/packet) | v3a -O3 | v3l2be1b -O3 | v3l3 -O3 | v3e1c -O3 | **v3l3be1c -O3 / -O2 / -Os** | v3l3be1c hook |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| E/W write 16 B | 121.7 | 120.8 | 115.6 | 120.8 | **116.0 / 115.6 / 118.4** | 117.7 |
| N/S write 16 B | 123.9 | 121.2 | 117.9 | 121.0 | **116.9 / 116.2 / 126.0** | 118.5 |
| Forwarded write 16 B | 111.9 | 119.4 | 112.3 | 119.3 | **112.5** / 117.2 / 126.9 | 114.1 |
| Mesh 3-hop E/W write 4352 | 146.9 | 146.9 | 144.4 | 146.9 | **145.4** / 147.3 / 147.7 | 146.0 |
| Mesh multicast write 4352 | 140.7 | 143.7 | 138.5 | 143.8 | **139.0** / 142.4 / 143.6 | 142.3 |
| E/W / N/S scatter 4352 | 165.3 / 176.1 | 167.6 / 180.9 | 167.6 / – | 157.6 / 164.4 | **158.1 / 164.3** (-O3) | 154.0 / – |
| E/W / N/S atomic | 129.3 / 138.9 | 132.3 / 135.6 | 131.3 / 134.7 | 125.6 / 128.1 | **124.5 / 126.9** (-O3) | |
| Forwarded fused 4352 | 155.2 | 160.8 | 160.6 | 160.9 | 160.8 / 166.1 / 179.7 | |
| Suite means vs v3a -O3: 4352 / mix / flushmix / flushmix4 / 1D | | +1.4 / +1.2 / −0.8 / −1.9 / −0.5 | +0.3 / −1.3 / −1.7 / −0.3 / −0.9 | −3.1 / −1.1 / −3.2 / −0.7 / −1.2 | **-O3 −4.2 / −3.6 / −4.2 / −0.1 / −1.5; -O2 −2.3 / −2.1 / −4.1 / −0.6 / −1.6; -Os −0.3 / +4.7 / +0.9 / +3.9 / +1.9** | −6.4 / −3.1 / −3.5 / −0.2 / −1.9 |

v3a's own means at -O2 are +3.5 / +3.9 / +2.2 / +2.3 / +2.5, and at -Os +26 / +29 / +23 / +30 / +19.

| VC1 router (cycles/packet) | v3a -Os / -O2 | v3l2be1b -Os / -O2 / -O3 | v3l3be1c -Os / -O2 / -O3 |
| --- | --- | --- | --- |
| 1-hop E/W 16 B / 4352 | 233.7 / 234.5; 258.5 / 253.9 | 208.7 / 199.5 / 199.6; 230.7 / 218.4 / 219.1 | 197.9 / 201.3 / 201.3; 217.7 / 213.1 / 214.8 |
| 1-hop N/S 16 B / 4352 | 236.2 / 236.2; 266.3 / 261.9 | 211.8 / 201.5 / 201.3; 232.2 / 218.4 / 218.6 | 210.4 / 201.2 / 201.9; 226.5 / 215.0 / 215.2 |

- **v3l3be1c closes every gap but one, and beats v3a at all three opt levels.**
  - At -Os it is at v3a's -O3 level; v3a's -Os penalty was 19–30.
  - The hook probe does not cost it anything.
  - VC1 sanity passes at -Os, -O2 and -O3 (once each).
  - Remaining: forwarded fused writes, +5.6 at -O3 and more at -Os (E1).
- **E1's forwarding path still changes with opt level:** forwarded 16 B is 112 at -O3, 117 at -O2 (with v3a's E1 too) and 127 at -Os.
- **v3e1c's atomic and fused arms pay for themselves:** v3e1d (scatter only) gets the scatter rows but loses atomics (+8).

**Which ERISC carries the remaining -Os cost** (job 267, `p1r11_runs.sh inj`, mixed-level ELF injection as in §88, v3l3be1c).
- No router recompiles, and the injected ELFs are still in the cache after the runs.
- The fresh -O3 build matches round a's -O3 within 0.1 on every suite mean.

| Row (cycles/packet) | all -O3 | E0 -Os, E1 -O3 | E0 -O3, E1 -Os | all -Os |
| --- | ---: | ---: | ---: | ---: |
| E/W / N/S write 16 B | 116.0 / 116.9 | 114.5 / 115.6 | 121.3 / 126.4 | 118.4 / 126.0 |
| Forwarded write 16 B / fused / atomic | 112.5 / 160.8 / 125.3 | 112.4 / 160.8 / 125.3 | 126.9 / 179.7 / 140.9 | 126.9 / 179.7 / 140.8 |
| E/W fused / N/S atomic | 155.2 / 126.9 | 154.9 / 126.7 | 167.8 / 137.5 | 167.8 / 137.1 |
| HalfRing multicast write | 226.8 | 227.0 | 240.9 | 240.7 |
| Suite means vs all -O3: 4352 / mix / 1D | | +0.1 / −0.6 / +0.5 | +4.1 / +8.6 / +3.0 | +3.9 / +8.2 / +3.4 |

- **E0 is -Os-robust**, and slightly faster at -Os.
- **All of v3l3be1c's remaining -Os cost is E1's.** Along with the -O2 step on forwarded rows and v3r2's E1 sensitivity, this is the next E1 iteration (`v3t`): an E1 hot loop insulated from cold code, and per-packet paths that do not depend on opt-level transformations.

**A third VC1 hang (v3l3, job 254).**
- The hang:
  - v3l3 (with v3e1b's E1) passed VC1 at -Os and -O2.
  - At -O3, rank 1 never finished `MultiMeshAllToAll_size_4096_ntype_unicast_scatter_write`. Rank 0 finished it.
- After the job:
  - The broker's fabric check failed, so it held the device and reset it (`tt-smi -r`, about 5 minutes).
  - **This hang needed a reset; the earlier ones did not.**
- All three hangs are in the 4 KiB inter-mesh all-to-all:
  - v3u3, -Os, rank 0, write;
  - v3l2e1b, -Os, rank 0, write;
  - v3l3, -O3, rank 1, scatter.
- v3l3 has v3a's exact idle predicates, so the §93 fix is not the whole story.
- v3l3b, whose E0 differs only in VC0's idle test, passed VC1 at all three levels, once each.
- v3l3 and v3l3e1c are blocked pending cause.
- User decision: capture the hang state.
  - Re-run v3l3's VC1 sanity at -O3 up to 10 times.
  - On a hang, kill the test and read every router's credit registers and channel headers with tt-exalens (read only).
  - The tool is being built (`hang_capture.py`, `hang_decode.py`, `vc1_capture_lib.sh`).

**Hang capture (user-approved) and VC1 repeats** (jobs 268–270).
- **The tool** (`hang_capture.py`, `hang_decode.py`, `vc1_capture_lib.sh`, `p1cap_runs.sh`):
  - It reads every link's Ethernet core on all 8 chips through tt-exalens in safe mode, read only:
    - stream registers;
    - the full 512 KiB of L1;
    - the TX-queue registers;
    - the heartbeat.
  - The decoder matches each core's E0/E1 image to its config. It prints every channel's credits, slot headers and counters, and each downstream copy against its downstream register.
- **Self-test** on an idle device after a v3l3 -O3 run: 48 routers in 1.7 s, 0 read errors, every config matched, and the device stayed healthy.
- **Repeats:**
  - **v3l3 at -O3:** 10 of 10 clean, 11 counting the self-test. The hang did not reproduce.
  - **v3l3be1c:** 10 of 10 clean at each of -Os, -O2 and -O3.
- **Hang rate:** 3 hangs in about 110 VC1 runs that reached the 4 KiB all-to-all.
  - All three were in a round's full VC1 run (`vc1_split.yaml`), none in a repeat.
  - The capture is now armed in every round's VC1 runs (`round.sh b` uses `run_vc1_cap`), so the next one is recorded.
  - v3l3 stays blocked.

**Compile coverage** (review agent, `consolidation/coverage.py`, V3_REVIEW.md "Compile coverage").
- **Method:** the CPU-only mock cluster builds the router for every link and logs the commands. Types the host never emits are synthesized from the builder source. In all: 1,227 configs over 39 cases, for v3sb and v3a, at all three levels.
- **Blackhole:** v3sb builds everything reachable:
  - 1D line and ring, 2D mesh and torus;
  - telemetry, perf telemetry, code profiling, trimming capture;
  - tensix mux, single-ERISC, VC1 (two- and single-ERISC);
  - the data cache and spin-wait flags (synthesized);
  - UDM (synthesized).
- **Wormhole:** v3sb builds 1D, ring, 2D, instrumentation and tensix mux.
- **One new failure, VC2:** v3 does not compile there (v3a does). `receiver_erisc::run_kernel` is still type-checked under VC2.
  - Fix: guard it with `receiver_erisc::runs`' condition; tested, VC2 then builds 56/56.
  - It goes into v3t.
- **Inherited from main, v3a identical:**
  - `fuse_receiver_flush_and_completion_ptr == 0` does not compile; the builder never emits it.
  - Wormhole VC1 overflows IRAM at -O3/-O2 on 4–5 of 20 configs; production builds VC1 at -Os.
- Trimming replay needs a device run (job 271, logging only, no swap).

**The production keepalive** (job 272, `p1r11_runs.sh ka0`).
- Every number in this study uses a sequence-update timeout ("keepalive") of 1000 cycles, the study's knob `TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT`. Main leaves the firmware's 8000.
- With 0, the router keeps the firmware value. Run at -O3, benchmark suites:

| Row (cycles/packet) | v3a: 1000 / firmware 8000 | v3l3be1c: 1000 / firmware 8000 |
| --- | --- | --- |
| Mesh 1-hop, 3-hop E/W write 4352; scatter, fused 4352; chunked flushed fused | 132–165 / **187.6** | 131–158 / **187.6** |
| E/W / N/S write 16 B | 121.7 / 128.7; 123.9 / 131.5 | 116.0 / 125.6; 116.9 / 126.0 |
| Forwarded write 16 B | 111.9 / 125.1 | 112.5 / 125.1 |
| Mesh multicast write 4352; bidirectional neighbor exchange | 140.7 / 140.2; 125.0 / 125.0 | 139.0 / 138.9; 125.0 / 125.0 |
| Suite means vs v3a at 1000 (4352 / mix / flushmix / flushmix4 / 1D) | +20.7 / +24.5 / +40.1 / +39.1 / +12.8 | +19.7 / +23.1 / +38.6 / +39.1 / +11.3 |

- **With the firmware keepalive, one-way traffic is floored by the Ethernet sequence-update round trip** (187.6 at 4 KiB, §16). v3 and v3a are then identical on every large-packet row. v3's gains show only where the floor does not bind: small writes are 3–6 better, multicast about the same.
- **Bidirectional traffic is unaffected,** because acknowledgements ride on the reverse data.
- So the keepalive is a separate lever, larger than the router changes on one-way flows. For mainline it needs to be a real host setting (V3_REVIEW must-fix 3), and it would also benefit main's router.

**Mainline review** (agent, `consolidation/V3_REVIEW.md`).
- **Latent bug (confirmed):**
  - A deferred flushed atomic stays staged in `write_at_cmd_buf` until it fires, so nothing else may use that buffer meanwhile.
  - E1's bypass test excludes intermesh egress. The generic loop's copy does not, so an intermesh-exit packet's credit write can overwrite the staged atomic.
  - This comes from the study's deferral, so main is unaffected, but the tree's v2 has it too.
  - No logged config reaches it. Fixed in v3r.
- **Must-fix before a PR:**
  - The two init paths must make the same six ERISC syncs, which is unchecked.
  - The router reads study-only compile-time args. That includes the keepalive, `EXP_SEQ_UPDATE_TIMEOUT`, which is 1000 in every measurement; the firmware default is 8000.
  - Only Blackhole was ever compiled: no Wormhole, UDM, VC2, profiling, data cache or spin-wait builds.
  - `NocFlushType` is a packet-header API change and needs its own PR.
- **v3r:**
  - Renames: `sender_erisc` and `receiver_erisc` namespaces, applied by `v3r_renames.py`.
  - The two programs move into their own headers; the .cpp goes from 5,879 to 3,945 lines.
  - A file overview, with the unstated invariants written down.
  - Study scaffolding and dead code removed.
  - **PT_LOAD-identical to v3l2be1b on every config at all three levels.**
- **v3s and v3sb:** v3l3e1c and v3l3be1c ported onto v3r. Each is PT_LOAD-identical to its merge at all levels.

**Measured fragility: v3r2** (jobs 265, 266). v3r2 is v3r plus six semantically null cleanups, and every ELF changes:
- plain functions instead of dummy templates;
- `optimize("Os")` attributes removed;
- an inline heartbeat helper;
- an unused parameter removed;
- plain bit tests;
- plain pointers in init and one shared handshake.

| vs v3l2be1b (same code otherwise) | 4352 / mix / flushmix / flushmix4 / 1D means | Scatter rows |
| --- | --- | --- |
| -O3 | +1.1 / −0.3 / −0.1 / +2.8 / +1.9 | +4 to +6 |
| -O2 | +0.7 / +1.1 / +1.5 / +0.5 / +0.6 | +3 |
| -Os | 0.0 / 0.0 / +0.1 / +1.1 / −0.6 | 0 |

- **What moved:** E1-bound rows. E0's main loop is its own function, and its rows did not move. E1's loop is inlined into `kernel_main`, so cold init edits re-lay it out.
- **This is goal 1's failure mode, measured.**
- The structural fix to try: E1's main loop in its own function. The check is whether v3r2's edits then leave the loop's code unchanged.
- VC1 passes on v3r2 at all three levels.

## 95. Phase 1, round 12: v3t holds its gains at every opt level and under cleanup edits; the keepalive becomes a host setting

**v3t** (E1 only, on v3sb; `V3E1_NOTES.md` §11). Its aims:
- E1's main loop does not move under cold edits;
- E1's per-packet paths do not depend on -O3-only transformations.

What changed:
- `receiver_erisc::run_main_loop` is its own `noinline` function, with no multiplication by a constant that is not a
  power of two inside it.
  - GCC expands such multiplies through a cost cache shared by every function in the unit. An unrelated edit
    elsewhere can flip one from `mul` to shift + add, which renumbers registers and re-lays out the whole loop (§11.1).
- Full completion batches have a fast path.
- 1D decides once from the route value.
- 2D deliver-only packets test the forward bits once.
- The rare send types leave the loop.
- VC2 builds again (the coverage fix, §94).

Controls:
- **v3t2** = v3t plus v3r2's six null cleanups (§94), made by `v3r_renames.py join` → `derive_edits.py all` → `split`.
  Offline, v3t's loop code is unchanged under each edit at every level; only its start address moves.
- **v3tb** = v3t with the loop inlined again, which measures the function boundary.

**Runs** (jobs 277–281, `round.sh`; labels p12t, p12t2, p12tb). Every variant passes RELAXED and STRICT validation.
VC1 (full `vc1_split.yaml`, capture armed) finishes 34/34 at -Os, -O2 and -O3 for v3t and for v3t2.

| Row (cycles/packet) | v3a -O3 | v3sb -O3 / -O2 / -Os | **v3t -O3 / -O2 / -Os** | v3t2 -O3 | v3tb -O3 |
| --- | ---: | --- | --- | ---: | ---: |
| E/W write 16 B | 121.7 | 116.0 / 115.6 / 118.4 | **115.4 / 115.2 / 115.3** | 115.0 | 114.0 |
| N/S write 16 B | 123.9 | 116.9 / 116.2 / 126.0 | **117.2 / 116.1 / 123.1** | 117.0 | 119.9 |
| Forwarded write 16 B | 111.9 | 112.5 / 117.2 / 126.9 | **111.0 / 112.0 / 113.5** | 110.9 | 109.5 |
| Mesh 1-hop E/W write 4352 | 132.2 | 131.1 / 130.6 / 129.0 | **127.8 / 128.9 / 127.6** | 128.5 | 128.5 |
| Mesh 3-hop E/W write 4352 | 146.9 | 145.4 / 147.3 / 147.7 | **143.9 / 145.9 / 146.0** | 143.8 | 144.3 |
| Mesh multicast write 4352 | 140.7 | 139.2 / 142.4 / 143.6 | **138.9 / 141.9 / 140.4** | 138.8 | 139.4 |
| E/W scatter 4352 | 165.3 | 158.1 / 156.6 / 164.0 | **151.1 / 153.8 / 154.8** | 151.1 | 149.2 |
| N/S scatter 4352 | 176.1 | 164.3 / 175.1 / 170.3 | **154.8 / 155.6 / 166.7** | 154.7 | 157.1 |
| E/W / N/S atomic | 129.3 / 138.9 | 124.5 / 126.9 (-O3) | **126.6 / 127.3** (-O3); 127.6 / 134.4 (-Os) | 126.6 / 127.3 | 124.7 / 131.3 |
| Forwarded fused 4352 | 155.2 | 160.8 / 166.1 / 179.7 | **159.8 / 159.8 / 159.6** | 159.8 | 156.6 |
| Half-ring multicast write (1D) | 227.5 | 226.0 / 226.9 / 240.7 | 226.2 / 226.7 / 240.7 | 227.8 | 225.8 |

| Suite means vs v3a -O3 (4352 / mix / flushmix / flushmix4 / 1D) | -O3 | -O2 | -Os |
| --- | --- | --- | --- |
| v3sb | −4.2 / −3.6 / −4.2 / −0.1 / −1.5 | −2.3 / −2.1 / −4.1 / −0.6 / −1.6 | −0.3 / +4.7 / +0.9 / +3.9 / +1.9 |
| **v3t** | **−8.4 / −4.8 / −3.8 / −0.5 / −6.8** | **−6.6 / −4.5 / −3.9 / −1.0 / −6.3** | **−5.4 / −2.1 / −1.3 / +0.5 / −1.4** |
| v3t2 (v3t + six cleanups) | −8.4 / −4.9 / −3.9 / −2.9 / −6.6 | −6.7 / −4.5 / −3.8 / −1.5 / −6.3 | −5.3 / −2.1 / −1.3 / +0.1 / −1.8 |
| v3tb (loop inlined) | −8.2 / −4.9 / −3.7 / −1.8 / −4.4 | | |
| v3t hook | −6.0 / −2.3 / −2.7 / −1.6 / −5.2 | | |

(flushmix4 has 4 rows and a run-to-run spread of 6.2; its mean is noise at this size.)

| VC1 router (cycles/packet) | v3sb -Os / -O2 / -O3 | v3t -Os / -O2 / -O3 |
| --- | --- | --- |
| 1-hop E/W 16 B; 4352 | 197.9 / 201.3 / 201.3; 217.7 / 213.1 / 214.8 | 199.5 / 201.2 / 201.5; 214.5 / 214.2 / 214.0 |
| 1-hop N/S 16 B; 4352 | 210.4 / 201.2 / 201.9; 226.5 / 215.0 / 215.2 | 201.5 / 200.1 / 198.9; 219.3 / 213.9 / 213.7 |

- **v3t is the best build at every level.**
  - At -Os it beats v3a at -O3 on every suite but flushmix4 (+0.5, within that suite's noise).
  - v3sb's -Os cost (+4.7 on mix) is gone. Forwarded rows are now nearly level-independent: 16 B 111.0 / 112.0 / 113.5
    (v3sb 112.5 / 117.2 / 126.9); fused 159.8 at all three levels (v3sb 160.8 / 166.1 / 179.7).
  - Scatter gains another 7 to 10.
  - 1D gains 4 to 5 on means at -O3/-O2.
- **The stability test passes on hardware.** v3t2 matches v3t within 0.4 on every mean at every level (except the
  noisy flushmix4), and within 0.4 on the scatter rows. v3r2's same six edits had cost +1 to +2 on means and +4 to +6
  on scatter (§94).
- **The function boundary costs nothing on the means** (v3tb within 0.2 on 2D, mix and flushmix) and is needed for
  insulation (V3E1_NOTES §11.3). Inlining helps a few forwarding rows (FWD 16 B 109.5, forwarded fused 156.6). It
  costs 2.4 on the 1D mean and on N/S rows (N/S 16 B 119.9, N/S atomic 131.3).
- **Left over:**
  - N/S rows still lose 3 to 11 at -Os (N/S 16 B 123.1, N/S scatter 166.7, N/S fused 173.2).
  - Half-ring multicast loses 14 at -Os, as with v3sb.
  - Forwarded fused is +4.6 vs v3a at every level.
  - E/W atomic is +2.1 vs v3sb at -O3, N/S +0.4; both still beat v3a, by 2.7 and 11.6.
- **The hook probe now costs about 2.5 on 2D means** (v3sb: none).
  - The probe is a noinline call per pass in each ERISC's loop.
  - v3t2's hook costs the same as v3t's; v3tb's costs more (+2.7 to +3.6 on the 2D suites).
  - Not explained yet (no disassembly read). A call per pass is real added work, so some cost is expected. v3sb's
    free hook may have been the exception.
- **Compiler assumption to note.** The insulation rule "no constant multiplications inside the loop function" is
  written against one GCC behavior: the synth_mult cost cache. It is a rule about what to keep out of the hot
  function, not a steering trick: no attributes, pinning or hints. If a toolchain stops sharing that cache, the rule is
  harmless. If a toolchain adds another unit-global cache like it, cold edits could move the loop again; the v3t2-style
  edit test (offline, `fncmp.py`) is what catches that.
  **Superseded by §96 and §97 (the 2x2: multiplies have no effect; per-round work out of line is the insulation):** with the loop in its own function, a `t * 17` put back into the loop did not move under
  any of seven edits, and the hardware jitter band is noise on the means. The multiply rule is a documented gotcha,
  not a rule.

**Channel-trimming replay hangs on the tree's v1 too** (job 282, `trimdiag_runs.sh`).
- Setup:
  - the 1D trimming suite (`coverage/trim_1d.yaml`, deterministic all-to-all: capture, then replay of the captured
    profile);
  - kernel v1, every study flag off, the firmware keepalive, a fresh cache.
- Result: the capture finished and the replay hung. The read-only capture recorded all 48 routers (0 errors,
  `results/trimdiag_v1_hangcap`), and the device stayed healthy.
- So the replay hang (jobs 271, 276 on the tree's v2) is not the v3 router's. It is in main's trimming flow, main's
  router, or the study's host changes. Step 2 (v3l3be1c) did not run, by design.
- Trimmed builds stay covered by compile coverage only (§94: 120/120 2D, 48/48 1D).

**The keepalive becomes a host setting** (user, 2026-10-25: include it in v3).
- **Host** (in the tree, used by every router version):
  - `FabricEriscDatamoverBuilder::set_eth_txq_seq_update_timeout(cycles)`; the default 0 leaves the firmware's value.
  - The builder passes it as the named compile-time arg `ETH_TXQ_SEQ_UPDATE_TIMEOUT`, next to `SWITCH_INTERVAL`.
  - `compute_mesh_router_builder.cpp` sets 1000 in the Blackhole block that already sets the context-switch interval,
    the same pattern as main.
  - Wormhole keeps the firmware value: not measured there, and Wormhole tunnels host access over Ethernet.
  - Study-only override: when `TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT` is set, it replaces the setting, so the logged
    protocol (1000) and the ka0 runs (0 = firmware) still work. It goes in the PR's drop list with the other study
    knobs.
- **Router: v3tk** = v3t reading `ETH_TXQ_SEQ_UPDATE_TIMEOUT` instead of the study arg, with a comment that says what
  the keepalive does.
  - PT_LOAD-identical to v3t on every config at -O3, -O2 and -Os (536/536/534).
  - `offline_build.add_host_setting_args` gives logged configs the value today's host would pass.
- **What the host passes** (CPU-only mock runs of the rebuilt host without the override; `consolidation/scratch_tools/ka13/mock_ka.py`):

  | Case | Value passed |
  | --- | --- |
  | Blackhole 2D (80 configs), 1D (80), VC1 (96), single-ERISC 2D (40) | 1000 |
  | Wormhole 1D (28) | 0 |

  `EXP_SEQ_UPDATE_TIMEOUT` is then 0, so v1, v2 and v3a keep the firmware's 8000 unless the override is set.
- **Hardware:** v3tk without the override (`KA_ENV=`, jobs 283 and 284; labels p13k*).
  - The router got the host's value in every built config: `ETH_TXQ_SEQ_UPDATE_TIMEOUT` 1000 and
    `EXP_SEQ_UPDATE_TIMEOUT` 0. That covers 440 VC0 configs (-O3), 320 hook configs and 288 VC1 configs.
  - Correctness: validation passes with RELAXED and STRICT flushes, and VC1 finishes 34/34 at -Os, -O2 and -O3.
  - Performance: it reproduces v3t.
    - Suite means are within 0.25 of v3t's at -O3, 0.15 at -O2 and 0.3 at -Os, except flushmix4 (noise).
    - Key rows are within 0.5, and the VC1 1-hop rows within 0.5 except one (N/S 4352 at -O2, 217.8 vs 213.9).
  - The hook also reproduces (+2.1 / +2.5 / +1.2 / −0.5 / +1.4 vs v3t's +2.4 / +2.5 / +1.2 / −1.1 / +1.7), so the
    hook cost above is not noise.
- **So the keepalive needs no study knob any more:** the v3 router plus this host change gives the measured numbers.
  - The firmware value stays available. Pass 0 to `set_eth_txq_seq_update_timeout`, or leave the builder's default.
  - Not covered: Wormhole (left at the firmware value), and long runs with firmware interaction (§16's caveat).
    Every study run since §16 used 1000, and the VC1 runs repeat at it.

## 96. The jitter band: ordinary edits move v3tk by noise on the means; the multiply rule is not needed

**Why** (user, 2026-10-25):
- The rule "no constant multiplies in the loop" (§95) would apply to nearly all router code, since every per-packet
  path is inlined into the loop.
- Many multiplies are implicit: array indexing, GCC rewriting shift/or as a multiply, division by a constant.
- A codegen-identity gate would block toolchain updates that every kernel shares.
- So no team rules or gates. Measure instead how much ordinary edits move the lead, at -O3 and -Os.

**Edits** (`consolidation/jitter_edits.py`, variants in `variants/v3tk_jit/`). Eight changes a contributor makes
without expecting a performance effect; none changes per-packet work:
- `reorder`, `statefield` (`perturb.py`): swap two fields of E0's channel state; add an unused field to each state struct.
- `hb`: both heartbeats every 128 rounds instead of 64.
- `coldmul`: an init-time helper with constant multiplies and a division.
- `rxreorder`: swap two fields of E1's channel state.
- `cursorfield`: one more field in E1's downstream cursor (32 → 40 B).
- `init`, `templates`: two of v3r2's six cleanups (§94).

Offline, at -O3, in the hot configs:

| Effect on the hot loops | Edits |
| --- | --- |
| loop code identical, same address | `reorder`, `rxreorder`, `templates` |
| identical code, loop moved | `coldmul`, `init` |
| E0's loop code changes | `statefield`, `hb` |
| E1's loop code changes in the N/S config | `cursorfield` (field offsets; no multiply, the cursor index is compile-time everywhere) |

Each ERISC's config count changed by an edit runs from 20 (`rxreorder`, E1) to all 536 (`coldmul`, `init`).

**Runs** (jobs 285, 286; `jitter_runs.sh O3|Os`, `jitter_compare.py`).
- All benchmark suites pass. v3tk runs first and last in each job, which gives the run-to-run noise.
- Change of each suite's mean vs v3tk, in cycles/packet:

| | 4352 | mix | flushmix | 1D | flushmix4 (4 rows, noisy) |
| --- | --- | --- | --- | --- | --- |
| run-to-run, -O3 / -Os | +0.02 / −0.01 | +0.01 / −0.01 | +0.01 / −0.06 | +0.01 / −0.09 | +1.24 / +0.10 |
| 8 edits, -O3 | −0.10 .. +0.08 | −0.01 .. +0.19 | −0.03 .. +0.04 | −0.07 .. +0.36 | +0.51 .. +2.55 |
| 8 edits, -Os | −0.28 .. 0.00 | −0.01 .. +0.02 | −0.36 .. +0.03 | −0.31 .. +0.07 | −0.18 .. +1.13 |

- Single rows (608 row results per level):

| | median \|change\| | 90th percentile | largest |
| --- | --- | --- | --- |
| -O3 | 0.06 | 0.25 | 6 |
| -Os | 0.06 | 0.31 | 3 (outside two bimodal rows) |

- -O3: 9 row results moved by at least 1 cycle.
  - 1D scatter and half-ring rows moved by 3 to 6, under `coldmul` and `init`, which only move the loops: an
    alignment effect.
  - The heartbeat edit moved two forwarded rows by 2.5 to 2.8.
- -Os: the larger moves are on two flushmix N/S chunk-8 rows that were bimodal between the two v3tk runs (5 cycles
  apart), plus improvements of 1 to 3 on 3-hop scatter.

**The multiply gotcha, tested directly.**
- `cursorfield` did not put a multiply in any E1 loop: at -Os no config's loop gained a `mul`. The cursors are
  indexed with compile-time indices.
- So the deferred atomic's value was written back in its old form, `0x100 | (t << 4) | t` (edit `trid17`). GCC makes
  that `t * 17` in the W and N/S loops (optimized GIMPLE).
- Then seven edits were applied on top: `coldmul` (which itself multiplies by 17 in init), `init`, `templates`, `hb`,
  `statefield`, `reorder`, `rxreorder`. Results over W, N/S and 1D at -O3 and -Os (42 cases):
  - The loop code was identical, or identical but moved, in 39.
  - The other 3 are `rxreorder` at -Os. It reorders the loop's own state struct and changes the loop with or without
    the multiply.
- The mechanism is real: V3E1_NOTES §11.1 traced one case to it in the RTL dumps (the `templates` edit, W config,
  -O3: `x * 17` expanded as `mul` in one build and shift + add in the other).
  - But the evidence for "a function boundary is not enough" was confounded. The boundary-only draft (`vA`) still had
    its multiplies (×17, ×0x01010101, four ×144) and also the per-round work (termination check, heartbeat,
    context-switch handshake) inside the loop function. v3t changed both at once.
  - `vA`'s heartbeat edit changes code that was inside its loop, so that one is not a multiply effect at all.
  - With the loop in its own function and the per-round work out of line, the multiply did not fire in anything tried
    here.
- The insulation that matters is the function boundary, an ordinary structural choice. A ban on multiplies is not
  needed.

**Conclusions**
- **v3tk's jitter band is noise on the suite means**: within ±0.4 cycles/packet at both levels, the same as run to run.
  - Before v3t, two of these same edits (`init`, `templates`) were among v3r2's six, which moved the inlined-loop design
    by +1 to +2 on means and +4 to +6 on scatter rows (§94).
- **What is left is rows, not means.** A few rows move by up to 6 when code only shifts address (alignment). No
  source rule controls that.
  - For perf tracking: a band of about ±0.5 on suite means and about ±6 on single rows (or ±4%).
- **-Os is not more stable than -O3.** The bands match, and -Os costs +2.8 to +5.3 on the means (v3tk -Os vs -O3).
  It is not worth using -Os for stability.
- **Team guidance**:
  - Keep each ERISC's main loop a function of its own, with once-per-round and rare work in separate small functions
    (as E1 is now; §97 shows this is what insulates the loop).
  - The synth_mult effect is a documented gotcha (§95, V3E1_NOTES §11.1) with a diagnosis method (`rtlfirst.py`),
    not a rule.
  - No gates.

## 97. The 2x2: per-round work out of line is what insulates E1's loop; constant multiplies play no part

**Question** (user, 2026-10-25): §95 credited v3t's stability partly to removing constant multiplies from the loop.
§96 found that the evidence was confounded:
- The boundary-only draft also kept the per-round work inside the loop function.
- v3t changed both at once.

So this test varies each factor separately, offline (`consolidation/twobytwo.py`; cells in `variants/v3tk_2x2/`;
report in `results/twobytwo_report.txt`).

**Design**, on v3tk:
- **P, per-round work** (termination check, heartbeat, context-switch handshake): out of line, as in v3t (P0), or
  inlined into the loop with `FORCE_INLINE` (P1).
- **M, a non-power-of-two constant multiply in the loop:** none (M0), or the deferred atomic's `t * 17` (M1).
- Inlining the per-round work also brings the termination check's mailbox index (`launch[rd]`, 144-byte entries) into
  the loop. So the P1 cells read the 8-entry mailbox through an offset table, with no multiply.
- Two cells keep the natural code: `p1nat` (with its `* 144`) and `p1nat_m1` (`* 144` and `* 17`, close to draft vA).
- Each cell gets four cold edits: `coldmul`, v3r2's `init`, `templates` and `os`.
- Builds: the hot E1 configs (W, N/S, 1D, VC1) at -O3, -O2 and -Os, 90 builds in all.
- Check: `run_main_loop`'s code, against the cell's own build.
- The loop's non-power-of-two multiplies (N/S, -O3 GIMPLE) are as designed:
  - none in p0m0 and p1m0;
  - 17 in p0m1 and p1m1;
  - 144 in p1nat;
  - 17 and 144 in p1nat_m1.

**Result.** Loop code changed by at least one of the 4 cold edits, in the W, N/S and 1D configs:

| | M0: no multiply | M1: `t * 17` | natural: `* 144` | natural + `t * 17` |
| --- | --- | --- | --- | --- |
| **P0: per-round work out of line** | never (all levels) | never (all levels) | | |
| **P1: per-round work inlined** | 3 of 4 edits at -O3 and -O2; none at -Os | same | same | same |

- In P0, every edit leaves the loop's code identical, at the same or a moved address.
- In P1, `coldmul`, `templates` and `os` change it in the W, N/S and 1D configs; VC1 only moves.
- `init` never changes it.
- The P1 changes are whole-function re-layouts: about 1,860 differing lines in a loop of about 1,130 instructions.
  The three edits give the same result: N/S p1m0, 1136 → 1132 instructions.
- The set of out-of-line functions is unchanged apart from `coldmul`'s new helper, so no inlining decision flipped.
- The mechanism is not identified. It is neither multiplies nor inlining.

**Conclusions**
- **The multiply factor has no effect in any cell.** The loop with no multiply at all (p1m0) is exactly as sensitive
  as the ones with `* 17` or `* 144`, and with per-round work out of line the multiply makes no difference.
  - The synth_mult case traced in V3E1_NOTES §11.1 (v3sb, `templates`, W, -O3) is a real instance of how a cold edit
    can reach the loop. It is not why v3t is stable.
  - The "no constant multiplies" rule is withdrawn entirely, not just downgraded. §95's attribution was wrong.
- **What insulates E1's loop is that once-per-round and rare work lives in its own small functions**
  (`termination_requested`, `end_round`, `intermesh_egress_slot`), so the loop function holds only per-packet code.
  - That is ordinary structure, not a rule about what code may be written.
  - It holds at every level.
  - -Os was immune in all cells, but the hardware band (§96) showed -O3 is just as stable once the structure is right.
- **v3t's cycle gains are separate**, from the per-path changes (§95). They were never attributed to the multiply
  removal.

## 98. Round 13: v3v2 closes most E1 gaps; NoC programming through the API; a fourth VC1 hang with the fabric drained

**Designs** (E1 design agent, offline; V3E1_NOTES §12). The E1 changes on v3tk:
1. **Batch cut:** completion batches end at a record-word boundary.
2. **Preset:** local writes program the command buffer once before the loop.
3. **Decode:** the 2D route bytes are read as plain header fields.

Rule from the user, mid-round: NoC programming goes through the provided NoC APIs, wrapped in named helpers.
- Change 2 was first hand-written (six `NOC_CMD_BUF_WRITE_REG` per write; kept as `v3v_raw`).
- It is now `local_write_init_state()` / `local_write_with_state()` over `noc_write_init_state`, `noc_write_with_state`
  and `ncrisc_noc_set_transaction_id`, the stateful API the dispatch kernels use. Instruction counts match the raw form.
- Audit of the current lead, v3tk:
  - no raw command-buffer programming outside the study's DRISC-only blocks (PR drop list);
  - two raw status reads in E1, the completion check and the deferral bypass. The API has only boolean forms of
    those; count accessors are proposed as API additions (V3E1_NOTES §12.9).
  - Main's fabric code has neither.

**Variants and runs** (jobs 287–297, 303; all RELAXED and STRICT validation passed). Labels p14v, p14vb, p15v2,
p15e, p15c, p15d, and v3tk re-run p15k.

| Variant | Preset | Decode | Batch cut | NoC code |
| --- | --- | --- | --- | --- |
| v3v_raw (p14v) | yes | plain | yes | raw |
| v3vb (p14vb) | no | plain | yes | raw |
| v3vc (p15c) | yes | codec (as v3tk) | no | API |
| v3vd (p15d) | yes | plain | no | API |
| v3ve (p15e) | yes | fixed | no | API |
| **v3v2 (p15v2)** | yes | fixed | yes | API |

Suite means change vs v3t (4352 / mix / flushmix / 1D):

| | -O3 | -O2 | -Os |
| --- | --- | --- | --- |
| v3vc | −1.16 / −1.00 / +0.01 / −1.67 | −1.48 / −1.11 / −0.11 / −1.87 | −1.65 / −1.65 / +0.01 / −1.66 |
| v3ve | −0.83 / −0.74 / −0.14 / −1.69 | −1.26 / −0.91 / −0.10 / −1.87 | −2.38 / −2.72 / −0.99 / −1.64 |
| **v3v2** | **−2.08 / −1.15 / −0.25 / −3.68** | **−2.52 / −1.13 / −0.97 / −3.82** | **−3.25 / −2.93 / −1.34 / −2.23** |

| Row (-O3 / -O2 / -Os) | v3t | v3v2 |
| --- | --- | --- |
| N/S write 16 B | 117.1 / 116.1 / 123.1 | 115.2 / 114.7 / 116.0 |
| N/S fused 4352 | 162.1 / 161.6 / 173.3 | 153.3 / 152.1 / 160.8 |
| N/S scatter 4352 | 154.7 / 155.6 / 166.6 | 146.7 / 145.9 / 155.1 |
| Forwarded fused 4352 (v3a -O3: 155.2) | 159.8 / 159.8 / 159.6 | 154.9 / 155.0 / 157.5 |
| E/W scatter 4352 | 151.1 / 153.7 / 154.8 | 144.5 / 146.8 / 146.3 |
| Linear 1-hop E/W scatter | 154.3 / 154.8 / 158.6 | 142.5 / 143.4 / 151.7 |
| Mesh 1-hop E/W write 4352 | 128.1 / 128.8 / 127.5 | 130.5 / 131.1 / 130.2 |
| E/W mixed size 4 | 117.8 / 118.0 / 118.6 | 121.5 / 121.9 / 121.6 |
| Half-ring multicast write | 226.0 / 226.6 / 240.9 | 226.6 / 226.6 / 240.7 |

**Attribution, from the controls**
- **Preset (v3vc vs v3t):** scatter, fused and forwarded-fused gains.
- **Decode as first written (v3vd vs v3vc):** the N/S regressions: N/S atomic 127.3 → 134.1, N/S mixed size +3.7,
  N/S write 16 B +5.7.
  - With plain fields, GCC loads the X route byte only after testing the Y byte, so the load waits.
  - The fixed decode (v3ve) removes them.
- **Batch cut (v3v2 vs v3ve):**
  - Gains: 1D (Linear 1-hop scatter −6), N/S fused −3.
  - Costs: mixed-size rows (E/W +3.7, N/S +2.5), Linear 3-hop write at -Os +3.4, Mesh 3-hop write +1.3.
- **Unexplained:** the 4352 B one-hop write rows are +2.4 to +3.0 in every variant, including v3vc (preset only) and
  v3vb (no preset).
  - Not drift: v3tk re-run now (p15k) gives 128.1 / 128.6 on Mesh 1-hop E/W / N/S write, as at 09:00–18:00.
  - Not counting: v3tk's old write call already counted every write on Blackhole (issue #28758 workaround).
  - Back with the agent (disassembly of the one-hop write path).
- **Half-ring at -Os:** untouched. Ring routers run the generic loop, not E1's program; closing it means bringing
  rings onto E1's program.

**Fourth VC1 hang** (job 295, v3vc at -Os).
- Rank 0 never finished `MultiMeshAllToAll_size_4096 unicast_write`; rank 1 did. It is the same test family as all
  three earlier hangs.
- The armed read-only capture recorded all 48 routers (`results/p15cvc1_auto_hangcap`, decode in `decode.txt`):
  - every router alive (heartbeat advancing on 48 of 48);
  - the fabric fully drained: no packet waiting in any sender or receiver channel, every completion counter equal on
    both sides of every link, every upstream credit copy equal to its downstream register.
  - **So whatever rank 0 waits for is not held inside the routers.** Candidates: a test worker kernel waiting for a
    credit or acknowledgement it never got, or the test's cross-rank synchronisation. The capture reads only the
    routers.
- After the job:
  - the broker's fabric check failed;
  - it held the device and reset it (`tt-smi -r`, about 4 minutes);
  - the device recovered and the next jobs ran normally.
- v3v2, v3ve and v3vd passed VC1 at all levels in the same session.
- Per the standing rule, v3vc is blocked pending cause.

**v3v4 and v3v5** (jobs 304–307, labels p16v4, p16v5; all validation passed, VC1 34/34 at every level).
- **v3v4** = v3v2 with a 2D partial batch held for 8 idle passes instead of 4 (one constant).
  - The design agent's explanation was more, smaller credit frames costing the sending E0. It is refuted:
    - the one-hop 4352 B writes are unchanged (Mesh 1-hop E/W write 130.5 as in v3v2; E/W / N/S write 4352
      130.5 / 130.8);
    - its prediction for v3vd also failed: v3vd's N/S write rows are the worst of the four (132.3 / 132.5).
  - v3v4 helps elsewhere: Mesh 3-hop write −0.8 to −1.8 vs v3v2, mixed-size rows −0.3 to −1.9, small writes −0.3 to
    −1.0.
  - It is the best candidate on means at every level. Change vs v3t (4352 / mix / flushmix / 1D): -O3
    −2.11 / −1.38 / −0.23 / −3.70; -O2 −2.63 / −1.44 / −0.99 / −3.84; -Os −3.40 / −3.29 / −1.39 / −1.86.
- **v3v5** = v3v4 with the batch cut in 1D only.
  - Gains: the mixed-size rows return (E/W 117.8–118.9), and 1D keeps its gain (Linear 1-hop scatter 142.5).
  - Losses: the 2D cut gains (N/S fused 156.1 vs 153.5, E/W scatter 146.9 vs 144.5), so its 2D, mix and flushmix
    means are worse than v3v4's.
  - Linear 3-hop write at -Os stays +3.8 (152.6), so that cost is from the 1D cut.
- **Next hypothesis for the one-hop write rows:** every regressing variant has a faster E1, which spends more time
  idle-polling on these E0-bound rows. E0 and E1 share the tile's L1, stream registers and NIU.
  - Test: diagnostic-only builds with a fixed delay in E1's idle pass, `v3vc_d20/d50/d100`. Never for shipping.

**E1 idle-delay sweep** (job 309, `diag_runs.sh`; diagnostic builds only; all benchmark suites passed).
- Builds: v3vc_dN and v3tk_d50. After a pass with no progress, E1 spins N cycles on its own cycle counter.
  Same-session references: v3vc (p17c), v3tk (p17k).

| Row (-O3) | v3tk | v3vc | d8 | d20 | d50 | d100 | v3tk_d50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mesh 1-hop E/W write | 128.0 | 130.9 | 130.2 | 130.4 | 132.5 | 133.3 | 132.1 |
| N/S write 4352 | 128.6 | 131.4 | 131.0 | 130.8 | 132.0 | 133.5 | 130.5 |
| Mesh 3-hop E/W write | 143.8 | 145.2 | 146.9 | 147.5 | 149.9 | 156.2 | 148.3 |

- **Idle-polling contention is ruled out.** Polling less makes the one-hop 4352 B rows worse, not better, and v3tk
  degrades the same way.
- These rows depend on how quickly E1 reacts when a packet arrives, which points at the credit round-trip latency
  (send, receive, write, acknowledge, credit). It does not point at contention or at E1 throughput.
- The design agent's polling-period floor sketch is therefore dropped.
- The mechanism of the +2.4 to +3.0 in the v3v family is still open.

**One-packet round trip** (job 310, `lat_runs.sh`, `lat_compare.py`; exp_latency.yaml, telemetry off; all runs
passed).
- Two v3tk runs, first and last in the job, agree within about 5 ns.
- Median net round trip, change vs v3tk in ns:

| Case | v3vc | v3vb | v3v4 |
| --- | ---: | ---: | ---: |
| 1-hop E/W write 4 KiB (v3tk 1626) | +8 | +1 | +9 |
| 1-hop N/S write 4 KiB (v3tk 1645) | +10 | +11 | +10 |
| 1-hop N/S write 16 B (v3tk 1315) | +35 | +38 | +19 |
| fused, 3-hop | ±13 | ±13 | ±13 |

- **Single-packet latency does not explain the one-hop 4352 B throughput regression.**
  - v3vb's E/W 4 KiB round trip is unchanged (+1 ns), yet its E/W 4352 B throughput regresses like the others.
  - The +8 to +10 ns of the others is under 1 cycle/packet at these rates.
- So the cost appears only under sustained load.
- Working hypothesis, not tested: these rows are bound by the credit window. 17 worker slots × about 128
  cycles/packet is about 2,200 cycles, close to the measured round trip. How credits are batched and returned under
  load would then matter more than one packet's latency.

**Cause found: E1 checks a batch's completion too early** (V3E1_NOTES §12.16).
- In the variants, E1's completion check runs about 10 cycles sooner after a batch's last local write. Two causes:
  - the NoC API's write updates the software write counters before sending, where v3tk's call updated them after;
  - the batch cut removes v3tk's slower straddled-batch case.
- A 4352 B write leaves the NIU about 75–85 cycles after issue. A check landing just before that fails, and the
  credit then waits for a later pass, about 85 cycles later under load.
- The one-hop 4352 B rows are bound by the credit window, so the late credits cost throughput.

| Diagnostic (job 311, -O3) | v3tk | v3tk_cw | v3vc | v3vc_cw |
| --- | ---: | ---: | ---: | ---: |
| Mesh 1-hop E/W write | 127.9 | 126.9 | 130.9 | 126.9 |
| E/W write 4352 | 128.4 | 126.9 | 131.1 | 127.0 |
| N/S write 4352 | 128.6 | 127.7 | 131.5 | 127.9 |
| Mesh 1-hop E/W bidir write | 137.2 | 135.8 | 139.9 | 135.8 |
| Mesh 3-hop E/W write | 144.1 | 143.3 | 145.2 | 143.3 |

- `_cw` waits in place for the batch's newest write to leave before crediting; diagnostic builds only.
- **Confirmed.** It also gains about 1 on v3tk, which was paying the same cost on part of its batches.

**v3v6 = v3v4 + that wait, through the NoC API** (`ncrisc_noc_nonposted_write_with_transaction_id_sent`, capped
at 32 reads; then the batch goes to a later pass as before).
- Jobs 312–313, label p19v6. All validation passed; VC1 34/34 at every level.
- E0 is byte-identical to v3tk; E1 changes only in the 128 configs that run its program.

| Row (-O3 / -O2 / -Os) | v3t | v3v4 | **v3v6** |
| --- | --- | --- | --- |
| Mesh 1-hop E/W write | 128.1 / 128.8 / 127.5 | 130.5 / 131.0 / 130.1 | 126.8 / 127.0 / 126.6 |
| N/S write 4352 | 128.5 / 128.7 / 128.8 | 130.8 / 131.2 / 130.6 | 128.1 / 127.3 / 127.3 |
| Mesh 3-hop E/W write | 144.0 / 146.0 / 146.0 | 145.9 / 147.5 / 147.2 | 143.2 / 144.7 / 144.1 |

Suite means change vs v3t (4352 / mix / flushmix / 1D):

| | -O3 | -O2 | -Os |
| --- | --- | --- | --- |
| v3v4 | −2.11 / −1.38 / −0.23 / −3.70 | −2.63 / −1.44 / −0.99 / −3.84 | −3.40 / −3.29 / −1.39 / −1.86 |
| **v3v6** | **−3.42 / −1.70 / −1.57 / −4.06** | **−4.13 / −1.87 / −2.31 / −3.63** | **−4.84 / −3.68 / −2.59 / −2.45** |

- **v3v6 is the lead.**
- Half-ring multicast at -O2 read 235.2 (v3v4 226.3). The ring routers' code is byte-identical between v3v4 and v3v6
  at -O2 (offline PT_LOAD), so this is run-to-run variation on that row.
- Left: mixed-size rows +2.8 to +3.6 (batch cut); Linear 3-hop write at -Os +3.5 (1D cut); half-ring at -Os (rings
  are not on E1's program).

**v3v6 jitter band** (jobs 314–315, `jitter_runs.sh O3|Os v3v6 jv6`; the eight §96 edits on v3v6; all passed).
- Suite means:
  - 2D, mix and flushmix within ±0.06 at -O3 and ±0.03 at -Os (v3tk: ±0.2 and ±0.4);
  - 1D +0.27 at most at -O3;
  - 1D at -Os: −0.07 to −0.32 for every edit, while the two base runs differ by +0.35 themselves (one bimodal 1D row
    moved 6.6 between them).
- Rows: 90% within 0.15–0.17.
- **v3v6 holds the §96 stability band**, a little tighter than v3tk.

**User, 2026-10-26: the early-check effect is a timing cliff.** The credit goes back at once or about 85 cycles later,
depending on where a check lands relative to a write leaving the NIU. A stable version is preferred even at 1–2 cycles
per packet slower.
- Next: v3w, a continuous completion scheme. Every pass credits every slot whose write has left, and amortization is
  a rate rule, not a batch shape.
- Acceptance: a perturbation sweep. Inject 0 to 32 cycles before the completion check and after the write issue;
  the result must show a smooth slope with no steps.

**Perturbation sweep** (jobs 316–317, `pert_runs.sh`, `pert_compare.py`; diagnostic builds; all passed).
- **A, delay before the completion check.** Range over d = 8..32 on the one-hop 4352 B writes:
  - v3tk 1.5–2.1, v3v6 1.6–2.1, v3w4 1.5–2.1;
  - v3w 4.6–5.1;
  - no steps anywhere.
- **B, delay after a local write issue.** All four respond identically: a plateau over d = 8..16 (the spin's minimum
  is about 14 cycles), then roughly one-for-one. It measures E1's spare capacity per packet and does not separate
  designs.
- The sweep did not reproduce the early-check cliff on v3tk or v3v6. It was seen between code versions (v3vc vs
  v3tk), and v3vc was not swept.

**v3w and v3w4** (jobs 318–321; validation passed).
- **v3w (K=3) is out.** Its E1-bound rows lose 7 to 16: E/W scatter 156.8, Mesh 3-hop E/W write 148.6, Linear 1-hop
  scatter 158.9.
- **v3w4 is out.**
  - Its means are below v3v6's at every level.
  - Its VC1 rows lose 30 to 45 cycles/packet (-Os 1-hop E/W 16 B 199.4 → 244.4, 4352 B 214.5 → 257.9; the same at
    -O2/-O3). Its `uncompleted > T` rule (T = 1 below 8 slots) holds credits on the small VC1 rings.
  - The user flagged this concern beforehand (bursty and short streams, small windows).
- **The 20-slot window run was cancelled.** User: slot counts vary by config and must not be relied on.

**v3x** = v3v6 with credit holds bounded relative to the ring (V3E1_NOTES §12.20).
- T = 1 below 8 slots, otherwise min(4, slots/4).
- A sender misses at most T−1 slots because of E1 (0 on 2- and 4-slot rings).
- A partial batch goes after 4 idle passes, about 150 cycles at any window.
- Bursty-sender benchmark: stream idle segments, `{idle: <cycles>}` (streams plan v2;
  `fabric_exp/streams/port/streams_idle.patch`, applied to the tree and host rebuilt 2026-10-26), with
  `fabric_exp/exp_burst.yaml`.
- v3x's protocol and the burst runs for v3tk / v3v6 / v3w4 / v3x: jobs 328–330.

**The VC1 hangs are a test-harness race, not a router fault** (fifth hang: job 319, v3w at -O2,
`results/p22wvc1_o2_hangcap`, the first with the worker capture).
- What the capture shows, in `MultiMeshAllToAll_size_4096_ntype_unicast_write`:
  - 16 receivers on rank 1, cores (1,2)–(1,5) on chips 4–7, are still running with 0 of 100 packets validated;
  - each polls for packet 1, while its slot holds packet 65's data. The decoder flags "receiver buffer overrun";
  - each one's sender on rank 0 sent all 100 packets;
  - all routers are drained, and the rest of both ranks is done.
- **Mechanism.**
  - The test synchronises senders but not receivers. The host barrier runs before launch, and senders wait on the
    device global sync, which needs each rank's sync core running but not its receivers.
  - A receiver that starts after its sender has passed packet 64 finds packet 1's slot overwritten (64 slots of 4 KiB
    per receiver buffer, 100 packets), and waits forever.
  - At 1 KiB, 100 packets fit without wrapping. That is why every hang was in the 4 KiB all-to-all, under five router
    designs, at about 4% of full VC1 runs.
- **Scope.**
  - Confirmed for the fifth hang. The fourth (v3vc) shows the same drained-fabric signature, router capture only. The
    third (v3l3) had no capture.
  - The §93 hangs (v3u3, v3l2e1b) had a real router bug, fixed then (single-bit idle test).

**Bursty senders** (job 328, `burst_runs.sh`, exp_burst.yaml with stream idle segments; 44 tests per build, all
passed).
- The idle segments behave as designed: N1_G2000 takes 2093 cycles/packet, N4_G500 196.
- **Bursts of 1, 2 or 4 packets:** identical across v3tk, v3v6, v3w4 and v3x at both sizes. The 17-slot window
  absorbs the burst and the gap outlasts any hold.
- **Bursts of 8 with a 500-cycle gap, 16 B:**

| Build | E/W | N/S |
| --- | ---: | ---: |
| v3tk | 129.8 | 130.0 |
| v3v6 | 136.2 | 137.4 |
| v3x | 134.0 | 135.1 |

- **4352 B cells are mixed:** N/S 32-packet bursts, 500-cycle gap: v3tk 134.6, v3v6 and v3x 139.2. E/W 8-packet
  bursts, 500-cycle gap: v3tk 137.2, v3v6 134.6.
- **v3w4 is worst or tied-worst in most cells.**
- **Repeat for noise:** job 331, p25_*.

**v3x** (jobs 329–330, label p24x; all validation passed, VC1 34/34 at every level).

| VC1 row (-Os / -O2 / -O3) | v3v6 | v3x |
| --- | --- | --- |
| 1-hop E/W 16 B | 199.4 / 195.2 / 194.8 | 178.3 / 176.3 / 176.4 |
| 1-hop E/W 4352 B | 214.5 / 214.6 / 214.2 | 191.4 / 190.1 / 190.1 |
| 1-hop N/S 4352 B | 219.1 / 215.2 / 215.8 | 197.6 / 192.9 / 193.2 |
| Multicast 4352 B | 333.0 / 323.8 / 322.4 | 250.2 / 263.3 / 246.7 |

- The window-relative batch size is 2 on the 8-slot VC0 rings of VC1 routers (v3v6: 4) and 1 on the 4-slot VC1
  rings. Credits there no longer wait on half the ring.
- **2D/1D means:** 0.2 to 0.5 behind v3v6 on 4352, mix and flushmix at every level; 1D −0.6 to +0.1.
- **Jitter band** (jobs 332–333, the eight §96 edits on v3x): 2D, mix and flushmix means ±0.03 at -O3 and -Os; 1D
  within −0.38..+0.19; flushmix4 noisy as always. v3x is stable.
- **Burst repeat** (job 331, p25_*): every cell repeats within about 0.3, so the burst differences are real.
  - The v3v family (v3v6, v3x) loses 4 to 7 against v3tk on 16 B bursts of 8 with 500-cycle gaps, and 4.6 on N/S
    4352 B bursts of 32 with 500-cycle gaps.
  - It gains 2 to 4 on 4352 B bursts of 8 and on N/S bursts of 32 with 2000-cycle gaps.
  - The mechanism is with the design agent.
- **Test-race fix applied for checking:** streams_rxready.patch (receiver start gate: the host waits for every
  receiver to start, then the multi-host barrier, then releases the sync cores) and diag_launch_delay.patch
  (diagnostic only). Jobs 334–336: a deterministic 200 ms launch skew, then 50 plain repeats.
  - Results with the v1 gate (host polled every receiver): 50 of 50 plain repeats clean, with no gate warning.
    At the old ~4% rate that alone has about a 13% chance of happening anyway.
  - The deterministic check was invalid: the delay never fired. The run was not longer (5.5 s vs ~6.1 s), likely
    because each mesh is rank 0 of its own subcontext in the split. Being fixed (global launch rank, plus a log
    line per delay).
  - User nit: do not poll every receiver from the host. Gate v2 (streams_rxready_v2_delta.patch): each receiver
    increments a ready counter on its chip's sync core, and the host reads one word per chip, then the barrier,
    then the release.
  - Gate v2 with the fixed diagnostic (diag_launch_delay v2: global launch rank, one log line per launch). Job 337:
    - rank 1 logged a 200 ms delay on all 12 launches, and its launches were measured 200 ms after rank 0's;
    - both runs finished 24/24 with no gate warning;
    - world size 2 on both processes, so the barrier spans both ranks and the earlier repeats count;
    - a sender wraps a 64-slot 4 KiB buffer in about 10 µs, so without the gate this skew would overrun;
    - the negative control (the same skew, no gate) was not run: it would be a deliberate hang and a broker
      reset.
    Plain repeats on v2 (job 338): 25 of 25 clean, 24/24 finishes each, no gate warning.

**E0 audit** (agent; `consolidation/E0_AUDIT.md`; LEDGER.md now covers §1–§98 with 'applies to' and 'in the
lead?' columns; DESIGN_BRIEF.md has the standing rules).
- **A freed far slot is used one pass late.** The worker visit tests for room before it reads completions, so the
  pass that receives a credit frame credits the worker and ends; the send waits for the next pass.
  - It is a fixed lag of about 30–60 cycles per credit frame, not a timing race.
  - Inferred gain: up to 6–7 cycles/packet on the VC0 rows of VC1 routers, 2–3 on 2D. Candidate v3yc.
- **E0's per-round work is not insulated** (§97's failure mode). One extra per-round compare re-lays out the whole
  loop. v3y moves the per-round work out of line, and the same edit then only moves the loop.
- **E0 raw TX-queue and stream accesses** now go through the existing helpers. The result is PT_LOAD-identical,
  so no hardware run is needed.
- **No timing cliffs, and no credit holds,** on E0.
- **Candidates** (E0 only, E1 identical to v3v6):
  - v3yc: helpers + the worker change;
  - v3y: helpers + restructuring;
  - v3yb: v3y + the worker change;
  - v3v6l3: round 11's v3l3 E0 change.
- **Runs:** jobs 339–345 (full protocol for v3yc, v3v6l3 and v3yb; burst runs for all three).
- **Flow control on by default:** the assessment advises against it. It would route most validation traffic
  through a test mux, and the 2D torus yamls exceed the mux's receiver cap. The user decides.

**E0 candidates on hardware** (jobs 339–345; labels p29yc, p29yb, p29l3 with the usual o2 / os / s / hook / vc1_*
suffixes). Everything passed: 93 logs with 0 failures, and VC1 34/34 at every level for all three. E1 is v3v6's in all
three.
- **VC1** (cycles/packet, mean of the rows' entries; -Os / -O2 / -O3):

| VC1 row | v3v6 | v3yc | v3yb | v3v6l3 | v3x (E1 change, for scale) |
| --- | --- | --- | --- | --- | --- |
| 1-hop E/W 16 B | 199.4 / 195.2 / 194.8 | 186.9 / 186.0 / 186.4 | 188.7 / 186.1 / 185.5 | 196.4 / 198.4 / 199.9 | 178.3 / 176.3 / 176.4 |
| 1-hop E/W 4352 B | 214.5 / 214.6 / 214.2 | 208.7 / 207.3 / 207.3 | 208.3 / 208.4 / 207.6 | 218.8 / 222.8 / 222.7 | 191.4 / 190.1 / 190.1 |
| 1-hop N/S 16 B | 200.0 / 198.8 / 198.4 | 196.7 / 186.8 / 187.6 | 198.7 / 187.3 / 187.0 | 214.4 / 203.8 / 201.4 | 187.7 / 180.0 / 180.5 |
| 1-hop N/S 4352 B | 219.1 / 215.2 / 215.8 | 215.1 / 211.3 / 208.9 | 216.0 / 209.9 / 211.1 | 220.3 / 221.5 / 223.7 | 197.6 / 192.9 / 193.2 |
| Multicast 4352 B | 319.9 / 310.0 / 309.5 | 322.3 / 311.8 / 311.2 | 320.4 / 311.6 / 309.8 | 316.4 / 310.7 / 310.5 | 257.7 / 255.7 / 254.5 |

- The worker change (taking completions before the send decision) is worth 4 to 13 on the VC1 1-hop rows, as the
  audit inferred. It does not touch multicast.
- **2D / 1D means against v3v6** (4352 / mix / flushmix / 1D):
  - v3yc: -O3 −0.14 / −0.19 / −0.30 / +0.94 (HalfRingMulticast scatter +14, the noisy half-ring row); -O2 −0.36 /
    −0.02 / +0.10 / +0.39; -Os +0.97 / +0.16 / +0.41 / +0.69.
  - v3yb: -O3 +0.05 / −0.31 / −0.39 / +0.22; -O2 −0.04 / +0.28 / +0.19 / −0.17; -Os +1.18 / +1.15 / +0.67 / +0.15
    (multicast +6.5, small writes +3: the -Os cost the audit predicted for v3yb's restructuring).
  - v3v6l3: VC1 worse on every row (up to +14). Out.
- STRICT flushes: both up to 0.4 faster than v3v6. The hook probe costs v3yc and v3yb what it costs v3v6 and v3x (about +2).
- E0 barely moves the burst cells (jobs 344–345, p28_*): every cell within 1.2 of v3v6, and every move a gain
  (v3yc up to −1.2, v3yb up to −1.0, on the 16 B bursts). The bursty losses are E1's.
- **Choice: v3yb's E0.** It has v3yc's VC1 gains, a flatter 1D, and per-round work out of the loop (the §97
  insulation); its -Os cost is about 1. The jitter band of the combined build decides whether the insulation holds.

**The batch cut causes the bursty losses** (job 346, `burst_cut_runs.sh`; labels p30_v3x, p30_v3x_nocut, p30_v3x2
for exp_burst.yaml at -O3, and p30nc for v3x_nocut's 2D / 1D suites; all passed).
- v3x_nocut is v3x without the record-word batch cut: every batch reads the four records from its oldest slot,
  across two words if it must, as v3tk did. v3x2 keeps the cut only on passes that handle a packet.
- **Bursts** (cycles/packet; p30_v3x repeats p25_v3x within 0.1 in every cell):

| Cell | v3tk | v3x | v3x_nocut | v3x2 |
| --- | ---: | ---: | ---: | ---: |
| 16 B, bursts of 8, 500-cycle gap, E/W | 129.7 | 134.1 | 129.7 | 131.9 |
| 16 B, bursts of 8, 500, N/S | 130.0 | 135.2 | 130.0 | 133.2 |
| 16 B, bursts of 32, 500, E/W | 123.4 | 125.0 | 123.5 | 124.4 |
| 4352 B, bursts of 32, 500, N/S | 134.8 | 139.0 | 134.4 | 138.1 |
| 4352 B, bursts of 8, 500, E/W | 137.2 | 135.4 | 137.1 | 136.6 |
| 4352 B, bursts of 8, 500, N/S | 141.2 | 137.3 | 137.4 | 137.4 |
| 4352 B, bursts of 32, 2000, N/S | 183.0 | 180.4 | 180.6 | 180.4 |
| 16 B, steady (bursts of 1, no gap), E/W | 115.4 | 114.2 | 115.5 | 114.2 |
| 16 B, steady, N/S | 117.2 | 115.5 | 116.8 | 115.0 |

- Without the cut, every bursty loss returns to v3tk's level, including the unexplained N/S 4352 B bursts of 32
  (it follows the cut, not the route decode). v3x2 recovers only part. The bounded wait's gains on 4352 B bursts
  stay except E/W bursts of 8 (+1.7, back to v3tk's).
- **Steady suites, v3x_nocut against v3x** (-O3): 4352 +1.02, mix +0.40, flushmix +0.50, 1D +0.79. The cut is worth
  about a cycle on steady streams. It also cost the mixed-size rows: EW_MixSize4 121.6 → 117.8 without it.
- flushmix4 +4.1: NS_Chunk4Fused_Flush 153.7 → 163.8 (v3tk 162.2). v3x's 153.7 repeats in its two jitter-band base
  runs, so the cut really is worth ~10 on that row. It has 4-packet chunks with a flush, the same shape as the cut's
  word of 4 records: a gain from alignment, not one to rely on.
- Against v3tk, v3x_nocut keeps −2.0 / −1.1 / −1.0 / −2.6 (4352 / mix / flushmix / 1D).
- **Decision (robust over peak):** drop the cut. It is a quantized batch shape: worth ~1 cycle on steady streams
  and 10 on one aligned row, it costs bursty senders 4 to 5. The clean form is `v3x3` (v3x_nocut with mainline
  comments, the unused `to_word_end` removed). v3x2 is out.
- **Next:** `v3z` = v3yb's E0 + v3x3's E1 (disjoint files), checked offline for each ERISC against its parent, then
  the full protocol, VC1, the jitter band and the bursts.

**v3z = v3yb's E0 + v3x3's E1** (jobs 347–351; labels p31z*, jz3_* / jzs_*, p31_v3z with p31_v3tk as the
same-session reference).
- **Offline:** an exact composition. v3x3 is PT_LOAD-identical to v3x_nocut on both ERISCs, v3z's E1 to v3x3's and
  v3z's E0 to v3yb's, on every config at -O3, -O2 and -Os (536 / 536 / 534).
- **Correctness:** 133 logs, 0 failures (validation RELAXED and STRICT, benchmarks at three levels, hook, bursts,
  jitter builds); VC1 34/34 at every level.
- **VC1** (mean of the two meshes; -Os / -O2 / -O3). The E0 and E1 gains add up on the 1-hop rows:

| VC1 row | v3v6 | v3x | v3yb | v3z |
| --- | --- | --- | --- | --- |
| 1-hop E/W 16 B | 199.4 / 195.2 / 194.8 | 178.3 / 176.3 / 176.4 | 188.7 / 186.1 / 185.5 | 174.9 / 168.0 / 167.2 |
| 1-hop E/W 4352 B | 214.5 / 214.6 / 214.2 | 191.4 / 190.1 / 190.1 | 208.3 / 208.4 / 207.6 | 186.6 / 182.5 / 182.6 |
| 1-hop N/S 16 B | 200.0 / 198.8 / 198.4 | 187.7 / 180.0 / 180.5 | 198.7 / 187.3 / 187.0 | 184.9 / 172.8 / 174.6 |
| 1-hop N/S 4352 B | 219.1 / 215.2 / 215.8 | 197.6 / 192.9 / 193.2 | 216.0 / 209.9 / 211.1 | 192.9 / 187.5 / 187.7 |
| Multicast 4352 B | 319.9 / 310.0 / 309.5 | 257.7 / 255.7 / 254.5 | 320.4 / 311.6 / 309.8 | 272.2 / 267.6 / 267.5 |

  - Multicast: v3z is 12 to 14 behind v3x. Its two meshes differ by 8 to 15 within each run (v3x 262 / 247, v3z
    261 / 274 at -O3), so one run each does not settle that row. E1 is identical on VC1 routers (the cut never
    applied on 8- and 4-slot rings), so any real difference is E0's.
- **Steady suite means** (`compare.py`; 4352 / mix / flushmix / flushmix4 / 1D, half-ring rows left out of 1D):

| v3z against | -O3 | -O2 | -Os |
| --- | --- | --- | --- |
| v3tk | −2.0 / −1.3 / −1.1 / −0.2 / −3.0 | −2.6 / −0.9 / −1.0 / −1.1 / −3.2 | −2.6 / −2.2 / −1.1 / −0.2 / −1.7 |
| v3v6 | +1.3 / +0.4 / +0.4 / +2.7 / +1.7 | +1.5 / +0.9 / +1.4 / +1.7 / +1.6 | +2.1 / +1.5 / +1.5 / +2.1 / +1.2 |

  - The cost against v3v6 adds up as expected: about 1 from dropping the cut (job 346), plus v3yb's E0 at -Os
    (about 1).
  - flushmix4 is the NS_Chunk4Fused_Flush row (+10), the aligned row the cut won.
- **Bursts:** every cell at v3tk's level or better against the same-session v3tk (largest +0.6, E/W 4352 B bursts of
  32 at 500). v3tk repeats its job 331 cells within 0.25. The v3v family's gains on N/S 4352 B bursts stay (bursts
  of 8 at 500 −4.0; of 32 at 2000 −2.7).
- **STRICT flushed rows:** v3z reproduces the pre-cut designs (v3t, v3t2, v3tk) to within about 1.5 on every row,
  e.g. EW_Chunk4Fused_Flush 174.3 (v3tk 173.3) and NS_Chunk8Fused_Flush 155.0 (156.5). The cut had moved these by −19
  to +12. Flushed-chunk rows follow the batch shape; without the cut they are as before v3v.
- **Jitter band** (the eight §96 edits):
  - The 2D, mix and flushmix means stay within −0.18..+0.06 at both levels. With the half-ring rows left out, 1D
    stays within −0.16..+0.38.
  - The coldmul / init 1D moves (+0.27 / +0.38) are v3x's too.
  - Edit row changes (90th percentile 0.28 / 0.25) are about v3z's run-to-run noise (0.22 / 0.33). v3v6 and v3x show
    0.15–0.23.
  - Offline, on the hot configs (2D E/W, N/S, 1D line, ring, VC1) at -O3 and -Os, six of the eight edits leave both
    ERISCs' main loops the same or only moved. That includes hb on E0, which re-laid out v3v6's whole E0 loop
    (E0_AUDIT §1).
  - statefield changes only E0's prologue: the frame grows by 32 B, and the loop body is identical. cursorfield
    changes E1's cursor indexing (a 40 B cursor; up to 3 more instructions), which is that edit's purpose.
- **Verdict: v3z is the most robust v3 so far.**
  - It keeps every bursty cell at v3tk's level and gives the best VC1 1-hop rows.
  - Its codegen holds under ordinary edits on both ERISCs.
  - It runs 0.4 to 2.1 behind v3v6 on steady means: the robust-over-peak trade, inside the 1–2 cycle budget except
    the -Os 4352 mean (+2.1).
  - Adopted as the lead (user, 2026-10-26). VC1 -O3 code: E0 13,268 B + E1 11,828 B = 25,096 B (v3v6 25,116).

**VC1 multicast: repeat and attribution** (jobs 352–353; labels p32*, p33*).
- **Repeat** (job 352, 16 runs, all clean): v3z and v3x, four interleaved pairs at the builder's level (-Os) and at
  -O3. Every row repeats within about 1 cycle.
  - Multicast: v3z is 14.3 (-Os) and 12.8 (-O3) behind v3x, on both entries in every run. The gap is real.
  - 1-hop rows: v3z 2.8 to 8.9 ahead.
  - The CSV's two multicast entries are not in a fixed mesh order (one v3x run swapped them), so compare means.
- **Attribution** (job 353, -O3, two rounds, all clean). Two exact compositions, each ERISC PT_LOAD-identical to its
  parent on all 536 -O3 configs:
  - v3zy = v3y's E0 (per-round work out of line, helpers) + v3x3's E1;
  - v3zc = v3yc's E0 (helpers, the worker change) + v3x3's E1.

| VC1 row (-O3) | v3x | v3z | v3zy | v3zc |
| --- | --- | --- | --- | --- |
| 1-hop E/W 16 B | 176.2 | 167.3 | 174.9 | 168.2 |
| 1-hop E/W 4352 B | 189.7 | 182.4 | 190.5 | 183.1 |
| 1-hop N/S 16 B | 180.5 | 174.5 | 180.9 | 173.2 |
| 1-hop N/S 4352 B | 193.4 | 187.8 | 191.9 | 187.6 |
| Multicast 4352 B | 254.8 | 267.6 | 254.4 | 267.2 |

  - The worker change (E0_AUDIT §2, ledger A3: the worker visit takes completions before its send decision) gives
    both the 1-hop gains and the multicast cost. The per-round restructuring (A1) moves neither.
  - With v3v6's E1 the same change left multicast unchanged (v3yb vs v3v6), so it interacts with v3x's ring-bounded
    credit holds. The mechanism is not measured.
  - 2D at -O3 (one run each, p30nc → p31z, E0 v3v6 → v3yb): MeshMulticast write +2.7; the other rows within row
    noise. FWD_MixCmd's −37 is its known bimodality (§39): about 153 or about 197 on any build (jzs_base 199.1,
    jzs_base2 154.3).
- **Compile coverage on v3z** (`coverage.py`, 41 cases): 1,259 / 1,395 at -O3, -O2 and -Os, case by case as v3tk.
  The failures are the inherited ones (`*_nofuse`, Wormhole VC1 IRAM at -O3/-O2).

**Plain source and batching** (jobs 354–360, all clean; labels p34*, p34_*). The user asked what v3's completion and
forwarding tricks buy against main, and whether completion batching earns its place on bursty traffic. Three variants,
each through validation (RELAXED and STRICT), the suites at three levels, the hook probe, VC1 (34/34 at every level)
and bursts:
- **v3zy** (v3z without the worker change A3) as the reference.
- **v3p**: v3zy written plainly. E1 checks every slot of a batch (`transaction_flushed` per slot) and credits slot by
  slot; no read-class records, no bounded wait. E0's forwarding visit goes through small helpers (its pass code is
  v3zy's to within one instruction, in the emulator).
- **v3p_b1**: v3p without batching (every slot credited on its own, in E1's program and the generic loop).

Results:
- **v3zy against v3z:** 2D and 1D means within ±1 at every level; VC1 1-hop rows +3.5 to +8.7, multicast −12 to −14
  (as job 353).
- **v3p costs E1, not E0.** Its completion check makes four status reads per batch, each used at once (about 40
  cycles per batch), and one credit per slot (about 9 instructions each).

| v3p against v3zy (4352 / mix / flushmix / flushmix4 / 1D) | -O3 | -O2 | -Os |
| --- | --- | --- | --- |
| means | +6.2 / +6.3 / +3.9 / +9.3 / +5.5 | +9.9 / +10.5 / +7.9 / +11.5 / +7.4 | +12.4 / +14.3 / +10.9 / +13.5 / +8.8 |

  E1-bound rows lose 10 to 15 at -O3 (scatter, fused, atomics). v3zy's one-read check is load-bearing.
- **Without batching (v3p_b1 against v3p, -O3):**
  - Steady 2D small writes stop at a floor of about 133, where the far E0 handles a credit frame per packet:
    EW_PureWrite16 121.8 → 132.9 (v3zy 116.3). Means +5.5 / +7.2 / +3.3 / +1.9.
  - 1D one-hop writes are better without batching: 127.5 (v3p 138.2, v3zy 129.4). They are bound by the worker's
    credit loop.
- **Bursts** (`exp_burst.yaml`, -O3):
  - Bursts of 1 to 4 packets with 500- or 2000-cycle gaps are identical in all three designs (within 0.4). The gaps
    set the rate, and the worker (17 slots) never waits for credits, so when they come back does not matter.
  - Bursts of 8 to 32 packets: batching is better by 4 to 9 at 16 B; at 4352 B no batching is equal or better by up
    to 2. Back-to-back single 16 B packets: batching is better by 15 to 16.
- **VC1: no batching is best on every row at every level.** These routers' VC0 rings have 8 slots, so batches are 2
  slots, and holding one slot of 8 costs more than the frame it saves.

| VC1 row (-O3) | v3z | v3zy | v3p | v3p_b1 |
| --- | --- | --- | --- | --- |
| 1-hop E/W 16 B | 167.2 | 175.0 | 171.4 | 162.1 |
| 1-hop E/W 4352 B | 182.6 | 190.6 | 197.3 | 179.5 |
| 1-hop N/S 16 B | 174.6 | 181.3 | 177.0 | 170.1 |
| 1-hop N/S 4352 B | 187.7 | 192.0 | 201.6 | 183.6 |
| Multicast 4352 B | 267.5 | 254.5 | 256.6 | 241.5 |

- **v3q** is v3zy's algorithm written readably: E1's completion check states the two facts it rests on and uses named
  helpers (`in_each_byte`, `newest_slot_in`, `writes_outgoing`, `batch_slot_trid`); E0's combined idle test is one
  helper (`all_channels_free`). It is byte-identical to v3zy on every config at -O3, -O2 and -Os, so v3zy's numbers
  are v3q's. Offline edit check: as v3zy (v3p's E1 VC1 loop moved under rxreorder; v3q's does not). A first draft
  with a lambda for the slot's transaction ID changed the 1D E1 loop (−47 instructions); a plain function did not.
- **Verdict:** the specialized completion check pays 4 to 14 cycles/packet on E1-bound rows; batching pays on
  16-slot rings with steady small packets and costs on 8-slot rings; it does not hurt short bursts.

**Host move to bh-lb-120-a09u32** (2026-10-26; that host's broker jobs 024–027).
- On a08u24 the v3q16 VC1 run (job 361) failed at device open: UMD waited over 240 s on Ethernet training for one core
  of device 4, so no kernel loaded and no test ran. Another tenant then held that device.
- Host check (job 024, read-only; `consolidation/hostcheck_runs.sh`, `topo_dump.py`, `fabric_exp/a09u32_cluster_desc.yaml`):
  the same cube wiring as a08u24 (12 chip pairs, 2 links each, all trained at 400G). Chips {0,1,2,3} and {4,5,6,7}
  are the two 2x2 faces with 8 links between them, so the VC1 split's rank binding works unchanged. Two ports are down
  (chip 3 eth (10,1), chip 5 eth (6,1)); neither is a mesh link.
- Offline builds need the logged runs' generated headers, firmware ELFs and precompiled headers from their kernel
  caches; copied from a08u24 (`~/.tt-metal-cache-import`, merged into the same cache paths). v3q's offline -O3 build
  on a09u32 is byte-identical to a08u24's on all 536 configs.
- Calibration (jobs 025–026: v3q, which is v3zy's code, against a08u24's v3zy): all validation clean, VC1 34/34 at
  every level. Suite means within ±0.5 at -O3 and ±0.8 at -O2 / -Os; VC1 1-hop rows within 0.3.
- **VC1 multicast depends on the host.** a08u24's two meshes gave 247 / 262 at -O3, a09u32's give 279 / 279, with
  identical code. The row depends on how each 2x2 is placed on the chips and links, so compare it only within a host.

**Batching only on rings of 16+ slots** (v3q16; job 027 on a09u32, 18 VC1 runs, all clean). v3q16 is v3q with rings
under 16 slots credited slot by slot. Offline it differs from v3q only on the 48 VC1 E1 configs (the VC1 routers'
8-slot VC0 rings), so only VC1 was run: v3q16, v3q and v3z interleaved, two rounds, at the builder's level (-Os), -O2
and -O3. Repeats agree within 0.2.

| VC1 row (-Os / -O2 / -O3) | v3z | v3q | v3q16 |
| --- | --- | --- | --- |
| 1-hop E/W 16 B | 174.8 / 168.1 / 167.4 | 180.1 / 176.2 / 175.0 | 184.2 / 172.2 / 171.8 |
| 1-hop E/W 4352 B | 186.5 / 182.3 / 182.3 | 190.4 / 191.1 / 190.5 | 188.3 / 183.5 / 183.1 |
| 1-hop N/S 16 B | 185.0 / 172.9 / 174.6 | 188.5 / 179.8 / 181.5 | 197.1 / 182.7 / 180.8 |
| 1-hop N/S 4352 B | 193.1 / 187.4 / 187.9 | 197.8 / 192.8 / 191.9 | 198.7 / 188.9 / 188.1 |
| Multicast 4352 B | 290.2 / 287.1 / 286.7 | 282.6 / 279.7 / 278.7 | 269.8 / 265.7 / 264.3 |

- Multicast: v3q16 is the best of the three at every level, 13 to 14 below v3q and 20 to 22 below v3z.
- 1-hop 4352 B: v3q16 is back at v3z's level (within 0.2 to 1.9) and 1 to 8 below v3q, except N/S at -Os (+0.9).
- 1-hop 16 B: mixed. At -O2 / -O3 E/W is 3 to 4 below v3q; at -Os (the level VC1 builds use) v3q16 is 4.1 (E/W)
  and 8.6 (N/S) above v3q. Against v3z the 16 B rows are 4 to 12 higher everywhere.
- v3p_b1 (no batching anywhere, the plain E1 check, no bounded wait; a08u24, where VC1 1-hop rows calibrate within 0.3)
  had VC1 16 B rows about 10 below v3q16 at -Os and -O3. v3q16 keeps v3zy's bounded completion wait, which with
  one-slot batches can apply on every pass without a packet. That is a hypothesis; untested.

**Decisions (user, 2026-10-26, after the consolidated summary).**
- Lead: **v3q** (v3zy's code written readably; no worker change A3). v3z's worker change stays open, to revisit with
  the phase/cost analysis.
- v3q16 (batching only on 16+-slot rings) and further variant experiments: on hold.
- Flow control on by default in validation: not pursued. It is a test-infrastructure question, and the receiver start
  gate fixed the race.
- Profiling real VC1 (inter-mesh) traffic: on hold.

**Phase/cost analysis: PC sampling over the debug bus, stages 1 and 2** (a09u32 jobs 028–029; `consolidation/pc_sampler.py`,
`pcs_report.py`, `pcs_trial_runs.sh`, `fabric_exp/pcs_trial.yaml`).
- Method: each Ethernet core's debug bus (an observation mux read through RISCV_DEBUG_REG_DBG_BUS_CNTL_REG /
  RISCV_DEBUG_REG_DBG_RD_DATA) exposes each ERISC's decode/execute PC. A host process reads it while a test runs: one
  NoC write (select) and one NoC read per sample; the core is not halted or stepped. The device is opened as
  hang_capture.py opens it (tt-exalens, safe mode), after the test has opened it.
- Stage 1 (offline): building with `-g` leaves the loadable code byte-identical on every config at -O3 and -Os, so
  sampled PCs can be mapped to source lines without changing what runs.
- Stage 2 (device): one long 2D 1-hop E/W 4352 B row run three times (A without the sampler, B with it, C without),
  twice. All six runs 126.92–126.93 cycles/packet: no measurable perturbation. The sampler opened the device in 0.03 s
  beside the running test and read all 96 ERISCs (48 connected Ethernet cores × 2) about 78k times a second with no
  read errors; the broker's post-job health gate was clean.
- PCs map correctly once two layout facts are used:
  - kernels run from the kernel config window (E0 at 0xf480, E1 at about 0x11b50 here), not from their link address,
    so a PC maps to the ELF through the launch message's config base and text offset;
  - the image in L1 is the relocated `.xip.elf` one, byte-equal to it.
  About 87% of each router ERISC's samples fall in its kernel (the rest are outside the traffic window). Idle E0s on
  unused links alternate between tt-metal's Ethernet firmware and the base firmware (the idle context switch). The
  traffic-carrying pair (chip 6 E0 → chip 4 E1 here) touches the most distinct addresses.
- Not yet done: mapping addresses to source lines through the `-g` builds, sampling only the routers that carry the
  traffic, and the per-config section breakdown (stage 3).

**Phase/cost analysis, stage 3: the v3q cost map** (a09u32 jobs 030–031; `consolidation/ROUTER_MAP_V3.md`,
`pcs3_runs.sh`, `pcs_sections.py`, `pcs_tables.py`, `fabric_exp/pcs3/*.yaml`, results `pcs3_*`).
- 13 rows (2D: 1-hop E/W and N/S writes at 4352 B and 16 B, scatter, fused, atomic, 3-hop, multicast, forwarded,
  bidirectional; 1D: 1-hop, 3-hop, multicast), about 8 s of traffic each, every router ERISC sampled (about 5,500
  samples per ERISC per row, 0 read errors). Rates match the suites (e.g. 1-hop E/W 126.9, 3-hop 142.8, multicast
  141.1).
- Mapping: the run's own compile commands rebuilt with `-g` (all 68 builds identical in code to what ran); each
  address's inline chain classified into sections of the two main loops.
- Findings (cycles per packet; ROUTER_MAP_V3.md has the tables):
  - E0 sets the rate on 1-hop writes (no spare time; E1 polls 11–17); E1 on scatter and fused writes (polls 6 or
    less); 3-hop, forwarded and multicast routers are near their limit on both ERISCs.
  - E0's send costs 53–63 in every row; on the 1-hop sender the per-pass checks cost about 45–55 more (worker send
    check ~13–16, connection poll ~10, forwarding idle test ~11, group read ~9, completion check ~4–7) and worker
    credits ~8.
  - E1: completion credits 20–33 per packet even batched (80–130 per frame), retire 14–20, decode 18–22; local delivery
    35–38 (write), 57 (atomic), 74 (scatter), 80 (fused); forward ~51 per copy plus 12–24 of admission on forwarders.
- Not covered: multi-mesh (VC1-enabled) routers.

**Galaxy (200G links): main against v3q** (Slurm job 155240 on bh-glx-120-c07u14, single BH Galaxy 8x4, one link,
telemetry off; `fabric_exp/galaxy/`: `frs_glx.sbatch`, `glx_{mesh,linear,ring}.yaml`, `glx_report.py`; results
`results/glx155240/`).
- Configs: **main** = router v1 at flags 0 (main as of the Oct 9 merge base), firmware keepalive 8000; **mainka** = the
  same with the keepalive at 1000; **v3q** = the lead (keepalive 1000 from the host setting). Trees run from
  `/data/aagarwal/frs-glx/trees` (lean copies of the study tree, v3q's files swapped into v2's place); each was checked
  first on a09u32 (2D 1-hop 4352 B: main 221.6, v3q 127.0, as before).
- Validation: mesh sanity (data-checked) 96/96 on both routers. Every benchmark run passed (mesh 27, linear 10, ring 5
  tests per config). Device clock 1350 MHz, as on the P150s.
- At 200G a 4352 B packet takes 251 cycles on the wire (23.4 GB/s), so every 4352 B row is wire-bound for both routers
  (v3q 252.1, main 251.1). Router differences show at 16–2048 B. v3q's 1-hop floor (~121) meets the wire near 2 KiB;
  main's (~223) only near 3.8 KiB.

| Row (cycles/packet) | main | v3q | v3q vs main |
| --- | ---: | ---: | ---: |
| 2D 1-hop E/W write 16 B / 256 B / 1024 B / 2048 B | 223.5 / 223.9 / 223.1 / 222.5 | 120.5 / 120.7 / 123.4 / 128.7 | 1.85x / 1.85x / 1.81x / 1.73x |
| 2D 1-hop N/S write 16 B / 2048 B | 223.8 / 223.2 | 121.1 / 129.2 | 1.85x / 1.73x |
| 2D scatter 2048 B · fused 1024 B · atomic | 218.9 · 223.5 · 223.9 | 145.0 · 150.1 · 128.8 | 1.51x · 1.49x · 1.74x |
| 2D bidirectional 1024 B | 223.6 | 124.5 | 1.80x |
| 2D 3-hop E/W 256 B / 1024 B | 224.0 / 223.6 | 137.6 / 139.5 | 1.63x / 1.60x |
| 2D 7-hop N/S 256 B / 1024 B | 223.9 / 221.7 | 180.4 / 181.8 | 1.24x / 1.22x |
| 2D forwarded 1024 B | 159.5 | 113.9 | 1.40x |
| 2D multicast all-to-all 1024 B · 4352 B | 236.7 · 331.5 | 188.6 · 330.6 | 1.25x · 1.00x |
| 1D 1-hop 16 B / 1024 B / 2048 B | 167.4 / 164.1 / 164.8 | 121.8 / 123.9 / 130.9 | 1.37x / 1.33x / 1.26x |
| 1D 3-hop 1024 B · multicast 1024 B | 164.4 · 186.4 | 138.1 · 153.3 | 1.19x · 1.22x |
| Ring 1-hop 256 B / 1024 B | 192.6 / 190.1 | 175.9 / 176.0 | 1.09x / 1.08x |
| Full-ring multicast 1024 B | 276.0 | 269.4 | 1.02x |

- Rings gain only 2–9%: ring routers still run the generic loop (deadlock avoidance, no completion batching; ledger
  E1-31). Rings and tori are the Galaxy's production topologies, so this is the largest gap the Galaxy shows.
- 7-hop N/S (column 0 down the Galaxy) runs at ~181 against 3-hop E/W's ~138. Not yet explained (candidates: a
  longer credit round trip on some N/S links, or one slower link in the column); main hides it at 222.
- v3q's small-packet floor is ~5 above the P150's (1-hop 16 B 120.5 vs 115.7).
- Keepalive: no effect on main at router-bound sizes (main = mainka within 0.1: main's ~223 hides it). On wire-bound
  4352 B rows 1000 costs ~1 cycle (251.1 → 252.1, 0.4%); the floor it removes at 400G (§15) does not exist at 200G.
  The sweep below shows that v3q does need it (firmware keepalive: +5 to +7 at router-bound sizes).

**Galaxy size sweep: where each router meets the 200G wire** (Slurm job 155281 on bh-glx-110-a01u08, another Galaxy;
`glx_sweep_{mesh,linear}.yaml`, 1-hop writes at 1024–4352 B, bidirectional at 2048/3072/4352, 200,000 packets per row;
configs v3q, main, v3qfw = v3q with the firmware keepalive (override 0), v3qr = v3q repeated; results
`results/glx155281/`).
- Repeatability: v3q and v3qr agree within 0.1 on every row, and the rows shared with job 155240 (another node) agree
  within 0.2.
- The wire line, fitted on v3qfw's wire-bound rows (2560 / 3072 / 3584 B, residuals ≤ 0.01): cycles/packet =
  12.80 + size / 18.50 in 2D and 10.31 + size / 18.51 in 1D. That is 199.8 Gb/s, with a fixed 237 B per packet in 2D
  and 191 B in 1D (header, framing, the packets-sent update; the 46 B difference is the 2D route header). Above ~4 KiB a
  packet takes a second Ethernet frame, +3.0 cycles (4096 B: 237.2 against 234.2 on the line; §15's 4114 B frame
  limit). Keepalive 1000 stretches wire-bound rows by 0.40%.

| Share of the wire (with keepalive 1000) | 1024 B | 1536 B | 1792 B | 2048 B | 2304 B | 2560 B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| wire, 2D (cycles/packet) | 68.4 | 96.2 | 110.1 | 124.0 | 137.9 | 151.8 |
| v3q 2D E/W | 123.3 (55%) | 124.9 (77%) | 126.1 (87%) | 128.5 (96.5%) | 138.0 (100%) | 151.8 (100%) |
| v3q 2D N/S | 123.9 (55%) | 125.5 (77%) | 126.7 (87%) | 129.1 (96.0%) | 138.0 (100%) | 151.8 (100%) |
| main 2D E/W | 223.0 (31%) | 221.4 (43%) | 221.8 (50%) | 222.7 (56%) | 223.1 (62%) | 222.9 (68%) |
| wire, 1D | 65.9 | 93.7 | 107.5 | 121.4 | 135.3 | 149.2 |
| v3q 1D | 123.9 (53%) | 125.9 (74%) | 127.9 (84%) | 130.9 (92.8%) | 136.9 (98.9%) | 149.2 (100%) |
| main 1D | 164.0 (40%) | 164.3 (57%) | 164.7 (65%) | 165.0 (74%) | 165.1 (82%) | 165.2 (90%) |

- v3q saturates the link from ~2.3 KiB in 2D and ~2.5 KiB in 1D; at exactly 2 KiB it is 4.5–5 cycles short in 2D
  (96–96.5%) and 9.5 short in 1D (93%; 1D's worker channel has no spare slots, ledger "1D Linear worker slots"). Main
  meets the line only at ~3.9 KiB in 2D and ~2.8 KiB in 1D.
- Below 2 KiB v3q's own floor (~123–128) sets the rate: 1 KiB would need ~68 cycles per packet.
- Bidirectional E/W: 2048 B 137.6 (v3q) vs 223.8 (main); 3072 B 181.7 vs 223.1; 4352 B 256.9 vs 259.2. The reverse
  direction's credits share each wire: at 3072 B v3q sits 2.8 above the one-way line.
- **Firmware keepalive costs v3q 5–7 cycles at router-bound sizes** (1024 B 130.3 vs 123.3, 2048 B 133.7 vs 128.5;
  same on N/S and 1D), and gains it 0.6–1.0 on wire-bound rows. Main never shows this (its ~223 floor hides it). The
  mechanism at 200G is not investigated; the host setting (1000 on Blackhole) is the right default for v3 on Galaxy
  too.

**Shape inventory** (2026-10-26, offline; `consolidation/SHAPE_INVENTORY.md`, tools `argdump.py`, `stack_scan.py`,
`pcs_stack.py`, Galaxy cases in `coverage.py`).
- Runtime args: on every Galaxy topology and the P150x8, every runtime arg except the packed downstream NoC x
  coordinates is constant within each existing binary (most are identical on every router), so they can be
  compile-time at no extra binaries. Compile-time NoC x would add 14% (2D mesh), 33% (torus), 63% (express) binaries.
- Program coverage (v3q built for 1,435 configs at three levels): the sender and receiver programs run on BH 2D/1D VC0
  and VC1 only; express falls back to the generic loop on both ERISCs (5 VC0 channels on every router), rings and tori
  on E1, VC2, the mux, UDM, one-ERISC and all Wormhole configs entirely.
- Memory-resident loop state, from stage 3's PC samples: E0 8–12 cycles/packet on 1-hop rows and 16–21 on forwarding
  rows (the far-slot value, `ForwardingState::in_flight` and the `channels[]` records live on the stack); E1 0.0–0.2
  on local-delivery rows and 5.6–7.4 on forwarding rows (downstream cursor fields). Every main loop saves all 13
  callee-saved registers.

**v4 router bring-up, family 1** (2026-10-26; new files `fabric_exp/variants/v4/` = `fabric_erisc_router_v4.cpp` +
`tt_metal/fabric/hw/inc/edm_fabric/router/*.hpp`, 2,256 lines; design `consolidation/SHAPE_NOTE.md`; feature semantics
for later families `consolidation/V4_FEATURE_SPEC.md`).
- Scope: two-ERISC Blackhole, 1D and 2D, VC0, strict atomics; other configs fail to compile (static_assert) until their
  family is brought up. Builds: all 196 in-scope configs (P150 and Galaxy, 1D and 2D) at -O3 and -Os.
- Hardware (a09u32, jobs 033-036): validation 249/249 data-checked tests on the first run (mesh 96, stream 10, val1d
  without rings/mux/telemetry 143), no hangs.
- Rate (cycles/packet; v3q / main in brackets): 2D 1-hop E/W 4352 B 152.0 -> 141.1 -> 143.0 (127 / 222); scatter
  158.1 -> 157.7 -> 152.7 (143.9); 3-hop 175.1 -> 158.6 -> 164.5 (142.8), over three layout revisions found with PC
  sampling (`pcsv_runs.sh`, `pcs_profile.py`):
  - v4a -> v4b (sender): the free-slots group's values were spilled to the stack right after their loads (each store
    waits for its load: 20.7 cycles/packet); reduced to a one-word mask of channels with packets; per-round bookkeeping
    (yield flag, sent snapshot) moved into the round object in memory; no closure around the loop.
  - v4b -> v4c (receiver): per-packet counters out of one aggregate with arrays and references (they were stack-resident);
    Route as a one-word value; downstream cursors back in a table reached by pointer (as v3q): local-delivery rows
    better (scatter -5), forwarding rows worse (3-hop +6, admission 8.5 -> 18.7).
- Remaining gaps: register pressure. About 5 of the sender's 12 callee-saved registers hold hoisted address constants
  (credit-write paths), so the far ring cursor is stack-resident (~10 cycles/packet in the send); the receiver's
  cursor placement trades local-delivery rows against forwarding rows.

**1D against 2D** (2026-10-26, a09u32 jobs 037-039; study host switch `TT_FABRIC_EXP_LINEAR_SPARE_SLOTS` in
`fabric_static_sized_channels_allocator.cpp` gives Linear's spare slots to the worker channel, as main already does for
Mesh and Torus; `TT_FABRIC_EXP_WORKER_SLOTS` caps it).
- On v3q, 1D is not faster than 2D: 1-hop 129.2 vs 127.0, bidirectional 140.3 vs 135.1, 3-hop 140.3 vs 142.9. (Main's
  1D beats main's 2D only because main's 2D is slow: Galaxy 164 vs 222.)
- Worker-injected rows are bound by the worker credit loop (§76): Linear gives the worker channel 16 slots (2D: 17)
  and leaves ~17 slots of L1 unused. With 24 or 35 worker slots, v3q's 1D beats 2D: 1-hop 124.5 (2D 127.0),
  bidirectional 124.8 (2D 135.1), 1-hop scatter 140.4 (2D 146.7); 3-hop and multicast unchanged (receiver-bound). 24
  slots suffice. Data-checked 1D validation at 35 slots: 143/143.
- The rest of the per-packet work is the same in both modes (v3q cost map: send 53-63, local write ~35, forward ~51,
  completions ~25-32); 1D only polls two sender channels instead of four and carries a smaller header, while its
  forwarders rewrite the header every hop.
- v4's 1D: its forwards were stateless in 1D (only E/W 2D routers preset the downstream): every single-downstream VC
  now forwards statefully: 3-hop 179 -> 155, linear multicast 162.2 -> 152.0, unidirectional 215.7 -> 196.9 (v3q 140.3,
  144.6, 188.5). (Multicast figures first read with awk were wrong: the quoted "[2,4]" field shifts the columns;
  csv.DictReader gives the figures here.)
- 1D header rewrite per hop (`update_packet_header_for_next_hop`): v3q 1.75 cycles/packet on the 3-hop middle receiver
  (shift and store of the field read at decode). v4 reloaded the field just before the shift: 8.65; with the fields read
  once per packet and carried to the rewrite, 1.94 (job 040): 3-hop 155.1 -> 149.8, multicast -3, validation 96 + 143.
  Remaining 1D 3-hop gap: the middle router's sender, send 72.0 vs v3q 60.4 (the same register pressure as in 2D).
  The sender's send costs 69.6 vs v3q's 53.1 in 1D too: v4's loop held two pointer arguments in callee-saved registers
  (worker interfaces, upstream credit table) where v3q keeps the far ring; now copied into the loop's own tables.

**Sender send fix: no closures in the sender loop** (2026-10-27, a09u32 jobs 041-043; v4g, v4h, v4i are successive
edits of `variants/v4`; profiles `pcsv_v4{g,h,i}2d`; v4f reconstructed from the transcript reproduces job 040's binary
byte for byte).
- v4f -> v4g: send returns `bool` with an output parameter instead of a `Sent` struct, and both credits went out of
  line. 2D 1-hop 141.8 -> 130.5, but 3-hop 162.2 -> 192.2 and multicast 160.5 -> 178.6. v4h (upstream credit inline
  again) left 3-hop at 188.5: the credit was not the cause.
- Cause: the forwarding dispatch was a `for_each_bit` lambda capturing the loop's state by reference. In v4f GCC
  dissolved the closure; after the `Sent` edit it kept it as a stack object of pointers (`pending`, `in_flight`, the
  table, the far ring), and every channel test loaded through them: the dispatch line cost 27 cycles/packet on the
  3-hop middle sender. Static stack accesses in `sender::run` fell (v4f 140, v4h 103) while the hot path got worse;
  counts do not show it, the disassembly does.
- v4i: the loop function takes no closures. The forwarding dispatch, the free-slots group read, the per-round sent
  count and the record init recurse over their channel mask with explicit arguments (`lowest_bit`,
  `without_lowest_bit`). Validation 96 + 143.

| cycles/packet | v4f | v4h | v4i | v3q |
| --- | ---: | ---: | ---: | ---: |
| 2D 1-hop E/W 4352 B | 141.8 | 133.1 | 130.0 | 127.0 |
| 2D 1-hop bidirectional | 142.2 | 135.8 | 136.5 | 135.1 |
| 2D 3-hop | 162.2 | 188.5 | 149.8 | 142.9 |
| 2D multicast | 160.5 | 173.0 | 148.9 | 141.1 |
| 2D 1-hop scatter | 155.5 | 154.8 | 154.3 | 146.7 |
| 1D 1-hop (24 worker slots) | 129.0 | 124.5 | 124.5 | 124.5 |
| 1D 3-hop (24) | 149.1 | 148.8 | 147.5 | 139.9 |
| 1D multicast (24) | 150.6 | 151.6 | 151.6 | 147.0 |

- Credit placement no longer matters: v4i with the upstream credit out of line too (v4cuo, perf rows only) is within
  2.0 of v4i on every row (3-hop 150.6, multicast 150.9, 1-hop 130.2). v4i keeps it inline (E0-11).
- v4i 3-hop middle sender (cycles/packet): send 68.3 (v3q 62.4), forwarding dispatch 10.7 (v4h 27), visits 22.8,
  upstream credit 11.0. The 1-hop worker send is 62.4 (v3q 57.9). What remains in the send are single stack loads (far
  ring next slot, the worker slot, a header address), about 1 cycle each: register allocation, not pursued.
- Remaining v4 gaps to v3q: 2D 3-hop +7 and multicast +8 (on the 3-hop row the middle sender's send is +6, the
  receiver's sections within 3 of v3q's each), scatter rows +5 to +8 (receiver local delivery), 1D 3-hop +7.6 (not
  profiled at v4i).

**Receiver gaps: v4j** (2026-10-27, a09u32 jobs 044-046; `variants/v4i` is the frozen v4i, v4j = `variants/v4`;
profiles `pcsv_v4ir{2d,1d}`, `pcsv_v4j{2d,1d}`, `pcsv_v4js`, `pcsv_v3qs`; new sampler rows `pcs3/fwd16.yaml`,
`pcs3/mixcmd.yaml`).
- Instruction-level comparison with v3q on the same rows: the receiver's forward (both pay ~15 cycles/packet of
  command-buffer waits), retire and completion check matched v3q within an instruction; decode did not. v4i's 2D path
  from poll to the deliver test was 19 instructions (v3q ~12): the one-word Route packed the deliver bit into bit 31
  and the slots below it, then masked them apart again; the 1D one rotated the field and did the same.
- v4j: each routing mode's Route holds the routing value as read (2D action byte, 1D field) and answers forwards /
  delivers / forward slots with one mask each (2D path now 13 instructions, 1D 10); Downstreams holds its links by
  value (a forward's ring base one load from its cursor instead of two dependent loads); the receiver's per-packet
  helpers (admit, forward_all, the completion trid lookup) recurse over their masks instead of taking closures (no
  code change today: the receiver's closures were dissolved, 20 stack accesses either way). Validation 96 + 143,
  mix suite 33/33.

| cycles/packet | v4i | v4j | v3q |
| --- | ---: | ---: | ---: |
| 2D 1-hop E/W 4352 B | 130.0 | 129.3 | 127.0 |
| 2D 1-hop scatter E/W / N/S | 154.3 / 156.0 | 149.5 / 153.4 | 146.7 / 149.1 |
| 2D 3-hop | 149.8 | 149.7 | 142.9 |
| 2D multicast / scatter | 148.9 / 184.5 | 148.3 / 179.2 | 141.1 / 174.6 |
| 1D 1-hop scatter (24 worker slots) | 146.1 | 141.2 | 141.3 |
| 1D 3-hop (24) | 147.5 | 143.0 | 139.9 |
| 1D multicast / scatter (24) | 151.6 / 187.5 | 149.2 / 181.8 | 147.0 / 181.5 |
| 2D fused 4352 B (mix suite, E/W) | 157.0 | 152.9 | 151.1 |
| 2D atomic (mix, E/W) | 135.1 | 131.1 | 127.9 |

- Scatter, fused and atomic rows gain 3-7 everywhere; 1D scatter rows now equal v3q. 2D 3-hop and multicast do not
  move: their receivers were not decode-bound. On the 3-hop row the forwarding receiver is saturated and its sections
  are within 2-3 of v3q's each (forward +3, completions +2); with the middle sender's send (+4-6) that is the gap.
- Mix suite (first a09u32 baseline; `streams/p3_mixes.yaml`), v4j against v3q: 4352 B pure rows +0 to +2.4, mixed
  sizes +3.5 to +9.6, small writes (16-1024 B) +5.8 to +9.5, mixed commands +37 (E/W, N/S) to +63 (forwarded).
  - Mixed commands (write, scatter, fused atomic, atomic; job 046): v4j's receiver spends 62.6 cycles/packet spinning
    in `flush_write_to_noc_pipeline` (strict atomics wait for every earlier write's acknowledgement); v3q's deferred
    flushed atomic costs 26.9. The difference is the whole gap (203.2 vs 166.8 on the sampled row). Deferred atomics
    are off in v4 by decision until the shape is reliable.
  - Small writes: sender-bound. On forwarded 16 B (two senders through one router) the merge-point sender is ~96%
    busy: send 64.3 vs v3q 60.6, and the per-pass free-slots group read 15.0 vs 3.4. v4 turns the four registers into a
    channels-with-packets mask every pass (~5 instructions per channel; v4b's fix for spilled group values); v3q keeps
    the raw values in callee-saved registers and tests each where it is used. On 1-hop 16 B both ERISCs have slack
    (the worker credit loop sets the rate): sender +3 per send, receiver +3 to +5 over the packet.

**Shared send experiment: v4s** (2026-10-27, a09u32 job 047; `variants/v4s`, control `variants/v4j` re-run in the same
job, equal to job 045 within 0.2 except MixCmd ±1.6).
- Question: the send and the forwarding visit were templated on the channel and inlined once per channel. A diff of
  two inlined sends: 49 of 70 instructions identical up to registers; the rest carry ~6 per-channel values (header id,
  ring bounds, completion-counter address, free-slots register block, record location).
- v4s: behavior stays compile-time (template parameters per kind: worker or forwarding, the VC's far ring, flags); the
  addresses become a `SenderChannel` value. The worker's visit passes `sender_channel<0>` (folds to immediates, same
  code as before); the forwarding channels share one visit, reached by a run-time loop over the channels with work,
  with the `SenderChannel` in each record. 2D sender loop 686 -> 414 instructions (2 sends instead of 4); 1D unchanged.
  Validation 96 + 143, mix 33/33.
- Rate against the control: worker rows and all 1D rows unchanged (within 0.2; 1D has one forwarding channel);
  forwarding rows worse: 2D 3-hop +5.4, multicast +18.4 (the §89 pattern again), forwarded small writes and mixed
  sizes +2.9 to +3.5, MixCmd within its noise.
- Where (multicast sender, cycles/packet): the shared visit's checks 40.1 vs 16.5; its send 60.8 vs 67.3 (cheaper: the
  channel loads hide behind the TX-status reads). The +23.6: the record's addresses loaded per visit 12.4 (the
  completion check becomes record -> counter address -> counter, two dependent loads, where an immediate address needed
  one), the run-time loop 8.6 (ctz, mask), the in-flight bit with a variable shift ~3. A multicast pass visits several
  forwarding channels, most only for their completion check, so the per-visit cost multiplies.
- Conclusion: sharing the visit puts run-time channel addressing on the completion check of every in-flight channel on
  every pass. If code is shared, share the send only (its channel values are needed off the critical path); keep the
  per-channel checks compile-time. `variants/v4` stays v4j.

**Shared send only: v4s2** (2026-10-27, a09u32 job 048; `variants/v4s2`; control v4j re-run, within 0.2 of job 047).
- v4s2: the forwarding channels with a packet send through one send (run-time loop, `SenderChannel` in the record,
  the send does not read completions); then per-channel compile-time code takes completions (their counters read as
  one group every pass with forwarding work), credits upstream and keeps the in-flight bit. 2D sender loop 470
  instructions (v4j 686, v4s 414); validation 96 + 143, mix 33/33.
- Against the control: worker-injected rows better (1-hop 16-1024 B -3.1 to -4.5, E/W MixSize4 -3.7, MixCmd E/W -6,
  4352 B 1-hop -1.2); forwarding rows worse (2D 3-hop +17.4, 3-hop scatter +10, multicast +7, forwarded small writes
  and mixed sizes +5.5); multicast scatter -4; 1D unchanged.
- 3-hop forwarding sender (now saturated, 167): send 67.8 (v4j 67.1); run-time send loop and dispatch 19.6 (v4j
  dispatch 10.7); completions 21.9 (v4j visit checks 9.8): the group read of every forwarding completion counter (L1)
  on every pass with forwarding work, where v4j's channel took its completions from the read its send already made.
- Two arrangements, one result: a run-time-addressed forwarding path costs forwarded packets 5-18 cycles (v4s: the
  checks' dependent loads; v4s2: the loop's ctz -> record -> slot chain before the send, and a completion read that is
  no longer folded into the send). Per-channel code with constant addresses is worth that much on the forwarding
  path; the code saved is 30-40% of the 2D sender loop. The worker-row gains in v4s2 come with the smaller loop
  (allocation), not with a mechanism, and are not counted on.

**Readability round: v4k, v4m** (2026-10-27, a09u32 jobs 049-050; `variants/v4m` = `variants/v4`, controls v4j re-run in
each job). Three of the user's review points, no intended codegen change:
- Sender `channels_with_packets`: one named predicate, `holds_packet<CH>` (the free-slots count is anything but the ring
  size), a group read into `registers[CH]`, and `channels_holding_packets`. It stays `!=`, not `<`: an out-of-range
  count (an over-fill) must read as holding packets, as the generic loop reads it (LEDGER E0-10, §93). Written as a
  ternary (`holds_packet ? 1u << CH : 0u`) the 1D sender grew 328 -> 387 instructions (GCC branched on it and
  duplicated the worker visit); as `static_cast<uint32_t>(holds_packet) << CH` every sender is byte-identical to v4j
  (68 configs). `send` gained a comment naming its four channel inputs and why it stays per channel (v4s, v4s2).
- Downstreams by name: routing names a downstream by its direction in 2D (action bit d is direction d, asserted
  against the codec) and 0 in 1D; the downstream tables keep the host's compact order and `downstream_entry` maps a
  name to its entry at compile time. `action_to_slots` is gone; 2D receivers E/W 681 -> 678, N/S ~811 -> 790
  instructions. The stateful-forward preset finds its one live entry instead of assuming entry 0.
- The 1D sparse-multicast hop (write this chip's page group, advance the indices, forward): v4k computed a run-time
  "write first" flag in front of the forward and the write (one inlined forward instead of two); the 1D scatter rows
  lost 2.3 to 4.4 (1-hop scatter 141.2 -> 145.5), a register-allocation shift in the delivery path. v4m keeps v4j's
  two-way branch and moves the whole sparse case into one out-of-line function (`sparse_multicast_hop`): 1D receiver
  731 -> 684 instructions, 1D scatter rows +1.0 to +1.4 against the control, all else within noise. Accepted (robust
  over peak); the remaining cycle is allocation, not mechanism.
- Validation 96 + 143 and mix 33/33 on both. 1D sparse multicast, which no suite sends, on v4m with fabric_unit_tests
  (built on a09u32: `ninja -C build fabric_unit_tests`, 41 test objects, no library rebuild; job 051,
  `consolidation/v4m_gtest_runs.sh`, suite router settings): `Fabric1DFixture.TestSparseMcastPerPage*` 3/3 (single
  destination, two contiguous, non-contiguous); all `Fabric1DFixture` tests 10 passed, 3 skipped by the tests (two
  T3K-only, one TG gateways). The router ELFs carry `receiver::sparse_multicast_hop`, so v4m ran.

**Rings and tori: v4 family 2** (2026-10-27, a09u32 jobs 052-060; `variants/v4`).
- What rings and tori add (V4_FEATURE_SPEC §2-3): deadlock avoidance (bubble flow control: injection channels, the
  worker and on tori a channel fed from the other axis, send only with two free far slots) and first-level acks on VC0
  (the far sender returns producer slots on an ack, sent when a packet arrives; completions only free far slots). All
  ring and torus routers (P150x8 and Galaxy) have both, one receiver channel, no VC1; 8 transaction ids.
- v4 changes: the bring-up gate admits deadlock avoidance and VC0 acks; receiver `acknowledge_arrival` (the generic
  loop's protocol: one ack per pass in its own credit frame when the TX queue is free, the packets-received register
  consumed at the ack; only acked packets processed); completions one slot per frame under acks (the generic loop does
  not batch them there); stateful forwards re-program the credit address on deadlock-avoiding routers, as main does
  (main gives no reason; the forwarding sync buffer is RD_CMD_BUF, atomics use AT_CMD_BUF); the sender's ack and
  bubble paths were already written and compile now (`far_slots_needed_by` reads the untrimmed injection array: GCC
  rejects the trimmed std::array<bool, N>'s operator[] there as "used before its definition"). 2D routes never turn
  an E/W-facing router's packet onto N/S on a torus either (E/W routers read only the column's X action), so the live
  downstream rule holds. 272 configs build at -O3 and -Os (P150 and Galaxy, mesh, line, ring, torus); mesh and line
  binaries are byte-identical to v4m.
- What this P150x8 can run (cube wiring; mock-probed first, see below): ring validation `val_ring_p150.yaml`
  (FullRingMulticast, FullRingUnicast, FullRingUnicastDeadlockQuick, HalfRingMulticast) and the Y-torus all-to-all
  `val_torus_y_p150.yaml`; the Galaxy-sized Ring LinearMcast and Torus Y multicast tests abort in setup here.
  Benchmarks `v4/exp_ring.yaml` (exp_1d_4352's rows on Ring), `v4/exp_torus_y.yaml` (2D rows on Torus Y; only the N/S
  routers wrap and have deadlock avoidance).
- Incident: jobs 052-053 ran unvetted ring/torus yamls; tests that abort in setup after the fabric is up leave the
  ERISCs in the router, every later run failed ("Timed out while waiting for active ethernet core"), and the broker held
  the device degraded (~03:21-03:41) until tt_device_reset. Since then, yamls are vetted CPU-only on the mock cluster
  (a09u32's descriptor) and on main's router before v4.
- Hardware: v4 ring validation 19/19, Y-torus 12/12 (data-checked; main 19/19, 12/12).

| cycles/packet | main | v3q | v4 |
| --- | ---: | ---: | ---: |
| Ring 1-hop E/W / scatter | 182.1 / 234.3 | 176.1 / 227.6 | 183.4 / 222.3 |
| Ring 1-hop bidirectional | 182.1 | 176.1 | 183.5 |
| Ring 3-hop / scatter | 210.0 / 233.8 | 209.1 / 227.1 | 215.3 / 222.1 |
| Ring multicast / scatter | 213.0 / 264.1 | 208.9 / 259.6 | 206.2 / 250.8 |
| Ring unidirectional multicast | 272.9 | 272.0 | 251.2 |
| Half-ring multicast / scatter | 246.8 / 291.0 | 230.9 / 285.0 | 231.4 / 269.5 |
| Torus Y 1-hop N/S / scatter | 203.8 / 242.4 | 173.1 / 231.2 | 188.6 / 227.7 |
| Torus Y 1-hop E/W (no deadlock avoidance) | 132.5 | 127.0 | 129.4 |

- The ack protocol costs every router ~50 cycles/packet (ring 1-hop ~180 vs mesh ~130). v4 ring 1-hop (pcsv_v4ring):
  both ERISCs saturated. Receiver: completions 69.4 and ack 46.9 cycles/packet (two credit frames per packet, each
  behind TX-status reads ~7 cycles, plus the header's source id from L1 and the completion's bounded wait), local
  delivery 32. Sender: worker credit 33.9 (one per ack, i.e. per packet; without acks once per completion batch,
  ~8), worker visit 54.6 (ack-counter and connection loads stall ~10 each), send 62.3. v3q's sender pays the same
  credit (34.4); its receiver is the generic loop. The receiver ring has 32 slots on these configs.

**Rings: the ack protocol's cost, v4f1-v4f4** (2026-10-27, a09u32 jobs 061-064; variants kept as `variants/v4f1`..`v4f4`;
lead `variants/v4` = `variants/v4l` = v4f1, byte-identical over all 272 configs).
- v4f1, one credit frame per pass. The credit frame (counter credits) carries every sender channel's ack and
  completion counts, absolute, so acks and completions only raise the counts and the loop sends one frame per pass when
  they changed and the TX queue is free (`add_ack`, `send_credit_frame`; `release_completions` no longer checks the TX
  queue or sends). Every packet that has arrived is acked when seen (the packets-received register consumed in one
  step), not one per pass. Completions stay one slot at a time under acks.
- v4f2 = v4f1 with completions batched under acks as without (batch of 4 on the 32-slot ring); v4f3 = v4f2 with
  transaction ids in contiguous slot blocks instead of round-robin (ring: 8 ids, 32 slots); v4f4 = v4f2 with partial
  batches held at most 8 passes. All validate (mesh 96, 1D 143, ring 19, Y-torus 12, mix 33).

| cycles/packet | main | v3q | v4 (job 059) | v4f1 | v4f2 | v4f3 | v4f4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ring 1-hop E/W | 182.1 | 176.1 | 183.3 | 177.4 | 154.2 | 165.7 | 159.5 |
| Ring 1-hop scatter | 234.3 | 227.6 | 222.3 | 217.2 | 192.5 | 195.3 | 198.0 |
| Ring 3-hop | 210.0 | 209.1 | 215.3 | 208.7 | 187.8 | 190.0 | 192.5 |
| Ring multicast / scatter | 213.0 / 264.1 | 208.9 / 259.6 | 206.3 / 250.9 | 177.4 / 216.8 | 171.4 / 205.5 | 179.4 / 208.1 | 174.2 / 210.7 |
| Ring unidirectional multicast | 272.9 | 272.0 | 251.2 | 244.8 | 226.4 | 229.8 | 231.5 |
| Half-ring multicast / scatter | 246.8 / 291.0 | 230.9 / 285.0 | 228.4 / 268.2 | 213.0 / 261.5 | 417.9 / 843.0 | 430.8 / 925.2 | 411.4 / 883.5 |
| Torus Y 1-hop N/S / scatter | 203.8 / 242.4 | 173.1 / 231.2 | 188.6 / 227.7 | 182.8 / 222.9 | 169.7 / 198.2 | 169.5 / 200.7 | 170.1 / 203.4 |

- v4f1: every ring and Y-torus N/S row better than job 059's v4 (-5 to -34) and than main; mesh, 1D and mix unchanged
  (means -0.4 to +0.05). Lead.
- Batching completions under acks (v4f2) is much faster on most ring rows but half-ring multicast collapses (2-4x).
  Not transaction-id sharing (v4f3: block ids made it worse, and cost 1D multicast 9) and not trickle holds (v4f4: an
  8-pass hold limit changed nothing there and cost meshes 2-4). Profile (pcsv_v4f2r vs pcsv_v4f1r, half-ring): in v4f2
  both ERISCs of the router are slow at every L1 and register read (receiver: packets-received register 34.6 vs 13.6
  cycles/packet, header word 32.5 vs 5.9, downstream credit register 31.4 vs not hot; sender: each of its counter and
  register loads ~25), the sender spinning for far slots (visit checks 122 + 96 vs send 67) and the receiver stalled in
  admission (56) and poll (38). Mechanism not found; batching under acks stays off (the generic loop's choice too).
- Galaxy (200G) validation of v4l and main on the 2D torus and rings queued: Slurm job 155833
  (`galaxy/frs_glx_torus.sbatch`; trees /data/aagarwal/frs-glx/trees/{base,v4l}; all 173 Galaxy ring/torus tests got
  past setup on the mock Galaxy first; mock runs now use their own root, trees/mockroot, so a swapped /home tree cannot
  reach them).

**Galaxy rings and torus, v4l and main** (Slurm job 155833, bh-glx-110-a02u02, 200G links; `glx_report.py --configs`).
- Data-checked, both routers: torus deadlock stability 87/87, torus stability `_short` 56/56, ring sanity 19/19, ring
  rows 5/5.
- Torus nightly (v4l only): stopped by the step's 1500 s timeout while still running (2,138 tests passed, none failed,
  group 122 of 145; tests finishing every ~0.2 s up to the kill), not a hang. The sbatch now gives the nightly
  `NT_TIMEOUT` (3000 s). Rerun (job 155913): all 145 groups, 2604 tests passed, none failed, 27.5 min.
- Ring rows (cycles/packet, the slowest sender, device clock 1350 MHz on both):

| row | main | v4l |
| --- | ---: | ---: |
| 1-hop E/W 256 B / 1024 B / 4352 B | 192.5 / 190.2 / 248.6 | 177.4 / 177.6 / 249.6 |
| Full-ring multicast 1024 B / 4352 B | 270.2 / 263.7 | 236.6 / 313.8 |

- 4352 B rows are wire-bound at 200G (1-hop: equal). Full-ring multicast at 4352 B, where every chip multicasts and
  the links are saturated, is 19% slower on v4l (the slowest sender: 43.9 M vs 36.9 M cycles); on the 400G P150 the
  same row is faster than main (177.4 vs 213.0). Open: how v4's sender shares a saturated link between injection
  and forwarded traffic.

**Per-packet credit cost: v4u** (2026-10-27, a09u32 jobs 065-068; variants `v4u`, halves `v4ue0`/`v4ue1`, pieces
`v4ue0a`/`v4ue0b`/`v4ue1a`/`v4ue1b`; all built for the 272 configs at -O3 (v4u also -Os)). The study's question: is
batching wrong, or is each credit just expensive? Each credit made cheaper, nothing held or batched:
- Sender (E0): (a) the worker credit (an emulated L1 inline write on E0's write command buffer, which nothing else in
  the sender uses) is programmed once per connection (`preset_worker_credit`: destination, size, NoC) and again after
  router commands or the base firmware ran (`end_round` now says when); each credit stores the count and issues the
  write with that state. (b) The worker visit reads its counters (completions, acks, connection semaphore) as one
  fenced group.
- Receiver (E1): (a) under first-level acks, the pass reads the packets-received count and the TX queue status as one
  group, and the credit frame skips its busy re-read when the queue was free (only E1 sends on it). (b) A single-slot
  completion path (`completion_batch == 1`, the first-level-ack case): one record, one bounded wait, one credit.
- Validation: ring 19, Y-torus 12, mesh 96, 1D 143, mix 33, stream 10, burst 44, flushmix 16, flushmix4 4, every
  variant data-checked where run; v4u rerun (v4ur2) within 0.2 of the first run.

| cycles/packet | main | v3q | v4l | E0 (a) | E0 (b) | E0 both | E1 (a) | E1 (b) | E1 both | v4u |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ring 1-hop E/W | 182.1 | 176.1 | 177.4 | 177.8 | 177.5 | 177.6 | 172.3 | 160.7 | 156.3 | 156.7 |
| Ring 1-hop scatter | 234.3 | 227.6 | 217.1 | 216.9 | 216.0 | 215.9 | 211.6 | 199.6 | 195.4 | 195.5 |
| Ring 3-hop | 210.0 | 209.1 | 208.8 | 209.3 | 208.8 | 209.2 | 204.5 | 192.9 | 189.0 | 189.0 |
| Ring multicast | 213.0 | 208.9 | 177.4 | 177.4 | 177.5 | 177.5 | 172.4 | 163.5 | 164.6 | 156.6 |
| Ring unidirectional multicast | 272.9 | 272.0 | 244.8 | 244.7 | 244.7 | 244.7 | 240.5 | 236.1 | 234.0 | 233.9 |
| Half-ring multicast / scatter | 246.8 / 291.0 | 230.9 / 285.0 | 213.1 / 261.5 | 213.1 / 261.4 | 213.1 / 261.7 | 213.1 / 261.6 | 208.0 / 256.2 | 197.2 / 245.8 | 192.4 / 240.7 | 192.5 / 241.0 |
| Torus Y 1-hop N/S / scatter | 203.8 / 242.4 | 173.1 / 231.2 | 182.9 / 223.0 | 183.4 / 222.8 | 183.3 / 222.3 | 183.4 / 222.0 | 177.8 / 217.3 | 171.1 / 205.7 | 171.1 / 200.8 | 161.7 / 200.9 |
| 2D 1-hop E/W 4352 B | | 127.0 | 129.5 | 129.0 | 128.7 | 127.9 | 129.5 | 129.5 | 129.4 | 127.9 |
| EW_PureWrite16 | | 116.3 | 123.7 | 121.9 | 120.8 | 120.1 | 123.7 | 123.7 | 123.7 | 120.1 |
| mix mean (vs v4l) | | -4.7 | | -0.81 | -1.17 | -1.90 | +0.01 | -0.00 | -0.01 | -1.88 |

- The pieces add up. The receiver carries the ring and torus gain (rings are E1-bound): single-slot completions -17,
  grouped registers -5, together -21. The sender carries the mesh and mix gain (worker-injected small packets: -3.6 to
  -4.1 on PureWrite16; 4352 mean -0.5). Where E1 alone stops short (ring multicast 164.6, torus N/S 171.1), the
  bound moves to E0, and the sender pieces take those rows the rest of the way (156.6, 161.7). 1D unchanged (mean
  +0.00); burst mean -0.55 (no row worse than +0.01), flushmix -0.66, flushmix4 -0.61.
- v4u gets v4f2's batched-completion ring gain (ring 1-hop 154.2) without batching, and has no half-ring collapse
  (192.5; v4f2 417.9). The answer to the question: credits were expensive per event, and the robust fix is cheaper
  events, not fewer.
- Profile, ring 1-hop (`pcsv_v4f1r` vs `pcsv_v4ur`, cycles/packet). E1: completions 47.0 -> 32.7, credit frame
  20.0 -> 11.0, ack (now with the TX status read) 32.6 -> 36.7, delivery/header/retire unchanged (32.6/12.3/11.7 ->
  31.9/12.2/12.2). E0: visit checks 54.0 -> 42.5, worker credit 27.8 -> 21.8, send 61.8 -> 59.7, free-slots read
  23.9 -> 27.3 (E0's slack, E1 being the bound).
- Lead: `variants/v4` = v4u.
- Galaxy (200G, Slurm job 155963): torus deadlock stability 87/87, stability `_short` 56/56, ring sanity 19/19, ring
  rows 5/5, torus nightly 2604/2604 (all data-checked). Ring rows (cycles/packet, main / v4l / v4u): 1-hop 256 B
  192.5 / 177.4 / 156.3, 1024 B 190.2 / 177.6 / 156.6, 4352 B 248.6 / 249.6 / 249.6 (wire); full-ring multicast
  1024 B 270.2 / 236.6 / 216.1, 4352 B 263.7 / 313.8 / 303.6 (still behind main; open item above).
- Galaxy multicast size sweep (job 155971, `glx_ring_mcast_sweep.yaml`): no data. Main's run waited the whole 1500 s
  for a KMD lock that another process held on node bh-glx-110-a03u02 (an `--exclusive` job). The sbatch now stops a
  step after 180 s of lock wait with no test started and lists the processes it can see. Resubmitted without that
  node: job 156067 (bh-glx-120-c02u02), all 6/6 per router. Full-ring multicast, cycles/packet:

| size | main | v4l | v4u | v4u/main |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 270.2 | 236.5 | 216.1 | 1.25x |
| 2048 | 265.7 | 236.9 | 216.6 | 1.23x |
| 2560 | 282.3 | 237.0 | 216.8 | 1.30x |
| 3072 | 261.2 | 237.3 | 217.2 | 1.20x |
| 3584 | 261.1 | 237.6 | 218.6 | 1.19x |
| 4352 | 263.4 | 318.3 | 302.7 | 0.87x |

  v4 is router-bound and flat (~217) up to 3584 B; 4352 B is the one size whose 200G wire time (~248) exceeds v4's
  router time, and there v4 keeps the links only ~82% busy (main, router-bound at ~261 everywhere, ~94%). So the
  loss appears exactly when the wire is the bound. Suspects (untested): credit frames taking wire time (v4 sends a
  frame on any pass with new acks or completions, up to two per packet, ~14 wire cycles each at 200G, on a direction
  that also carries data) or E0's turnaround after the TX queue frees.
- Naming: this run first used the name v4g, already taken by job 041's sender-fix variant. It overwrote job 041's
  result files `v4g_13129088_{mesh,val1d,4352}` and `v4gw24_13129088_1d` (their numbers are in the sender-fix entry
  above) and `consolidation/v4g_runs.sh` (regenerated exactly from `v4f_runs.sh`). Renamed to v4u (variant, results,
  `v4u_runs.sh`, coverage builds).

**Credit frames per packet: v4p, v4p2** (2026-10-27, a09u32 jobs 069, 072; Galaxy Slurm job 157726). The Galaxy 4352 B
full-ring multicast gap's first suspect: a wire-bound link carries an ack frame and a completion frame per packet,
because with idle receiver passes between arrivals the ack (sent the pass a packet arrives) and the completion (sent
once its forward has left, a later pass) never share a frame. Both are already counts in one frame per pass (v4f1).
- v4p = v4u + under first-level acks, completions credited on a pass without a frame wait for the next ack frame, at
  most 16 passes. v4p2 = at most one held completion (a second sends the frame). Mesh and 1D code identical to v4u
  (272 configs, offline builds). Data-checked on P150: ring 19/19, Y-torus 12/12 (both).
- P150 (router-bound everywhere): every first-level-ack row slower, v4p / v4p2 vs v4u: ring writes +6.5 / +8.2,
  scatter +9.5 / +11.6, half-ring +7.0 / +9.1, torus N/S +8.0 / +9.5 (scatter +14 / +17); unidirectional ring
  multicast write -2.9 / -0.4; rows without acks unchanged. v4u twice in the same jobs: within 0.1 of job 065.
- Cause: E1's code, not the protocol. The hold's two loop-carried variables push the ring receiver loop into spilling
  (`receiver::run`: 555 -> 594 instructions, stack loads 20 -> 32, stores 36 -> 46); holding one completion (v4p2)
  costs as much as holding 16 passes' worth (v4p), so credit clumping is not it. P150 cannot answer the frame question;
  the Galaxy row can (the wire binds there, not E1).
- Galaxy (Slurm job 157929, bh-glx-120-c06u08, 200G; jobs 157726 and 157804 ran nothing: other users' processes held
  the device on the --exclusive node): ring sanity 19/19 on v4p and v4p2 (data-checked), but full-ring multicast
  collapses at every size: v4p 18,904-20,526 cycles/packet, v4p2 17,279-18,676, against v4u 216.1-304.2 and main
  261-284 (two runs each, within 1%). Ring 1-hop: +6.6 / +8.2 at 256 B and 1 KiB (E1's spills, as on P150), equal at
  4352 B (wire-bound).
- Mechanism: bubble flow control. A saturated full ring moves on its free slot (a forward needs one free far slot, an
  injection two); a held completion is that slot's credit, so the upstream sender cannot forward, no packet arrives,
  no ack frame goes out to carry the credit, and the slot is released only by the 16-pass limit: the ring advances one
  hop per limit. One held slot is enough (v4p2). Consistent in size: 16 passes x ~65 cycles x 19 hops ~ 19,800 cycles
  per packet. P150's 8-chip ring rows do not saturate the ring this way. Likely also v4f2's unexplained half-ring
  multicast collapse (completions batched under acks hide bubble credits the same way).
- So on deadlock-avoiding rings a completion must not wait for anything. v4p and v4p2 are dropped.
- The frame suspect is weak anyway: main's ring receiver sends one frame per ack and one per completion (each
  `send_ack_credit` / `send_completion_credit` sends the whole counter block, `fabric_router_flow_control.hpp`), two
  frames per packet at any load, and keeps the link ~94% busy; v4 sends at most that many. The 4352 B gap goes back to
  how v4's sender shares a saturated link between injection and forwarded traffic.

**Slot allocation sweep: v4us** (2026-10-27, a09u32 jobs 070-071, 073-075). The user's question: with no power-of-2
requirement, give channels computed depths and the leftover to the worker channel, instead of the tier tables.
- Host study switch (`fabric_static_sized_channels_allocator.cpp`): `TT_FABRIC_EXP_SLOTS_1D` (Linear, Ring) /
  `TT_FABRIC_EXP_SLOTS_2D` (Mesh, Torus) / `TT_FABRIC_EXP_SLOTS` (both) = worker,forwarding,receiver for VC0 (worker
  0 = every slot left, at most 127), checked against capacity: 83 slots of 4400 B on 1D and rings, 81 of 4448 B on 2D
  and tori. Every router gets the same depths, so the receiver depth matches at both ends of a link.
- v4us = v4u + the transaction-id tracker indexes ids by slot (no table) also when the ring is a whole number of id
  rounds (`SLOT_INDEX_TRIDS`); identical code to v4u at today's depths (272 configs).
- Data-checked: 1D validation 143/143 at 27/24/32 (worker/forwarding/receiver), ring 19/19 at 27/16/40, mesh sanity
  96/96 at 29/12/16 (its Linear tests at 55/12/16) and at 13/20/8, Y-torus 12/12 at 21/12/24.

| cycles/packet, 4352 B | today | change at |
| --- | ---: | --- |
| 1D 1-hop write / bidirectional | 131.8 / 140.8 | worker 19-51: -7.3 / -15.9 (16 slots bind; 19 suffice) |
| 1D 1-hop scatter | 149.2 | worker 35: -8.0, 27: -7.4, 19: -4.8; receiver 16 (worker 51): -3.1 |
| 1D 3-hop write | 144.2 | forwarding 20: -8.0, 24: -10.9, 28: -13.5, 32: -11.6 |
| 2D 1-hop write / bidirectional | 127.9 / 135.7 | worker 20-41: -2.1 / -8.8 (saturates at 20) |
| 2D 3-hop write | 149.7 | forwarding 15: +6.2, 14: +15.0, 12: +39.6, 10: +72.3, 8: +122.7 |
| 2D any row | | receiver 8: +40 to +50 (forwarding 16-20 cannot help); 32 (worker 13): 1-hop +32 |
| Ring, all rows | 156.7 (1-hop) | worker 19-43, forwarding 24: within 1.6; receiver 24, 40, 48: +7.5 to +8.7 on every row |
| Y-torus N/S (acks) | 161.8 | worker, forwarding: 0; receiver 24: +8.0; 32 (worker 13): -0.3 |

- Credit loops (slots x cycles/packet where a depth binds): 2D worker ~2,170 cycles, 2D forwarding (3-hop) ~2,200-
  2,400, 2D receiver ~1,420 (8 slots: 178). So at ~126 cycles/packet 2D wants worker >= 18, forwarding >= 19 per
  channel, receiver >= 12: 92 slots of 81. The 2D split is a trade; 1D has room for all.
- Rings: depth is not the limit anywhere (acks shorten the worker loop); only the receiver ring's size matters, as a
  cost.
- A receiver ring that is not a power of 2 costs ~8 cycles/packet on E1 regardless of depth (24, 40, 48 equal; 48 is
  a whole number of id rounds, so not the tracker): `ring_add` becomes a compare and a usually-taken branch, ~5 per
  packet (ack loop, retire, completion index, transaction-id lookup); the ring receiver loop is 48-72 B larger, no
  more spills. Sender channels show no such cost (worker 19, 27, 35 and forwarding 15, 20, 24, 28 are all fine).
- 2D worker/forwarding at receiver 16, stream mixes (33 rows): 20/15 mean -3.33 (every E/W and N/S row -1.0 to -7.6;
  FWD_MixCmd 217.4 -> 175.7, not repeated at 23/14, so noise), 23/14 mean -1.92; 4352 B rows: 20/15 mean -0.40
  (1-hop -2 to -3, bidirectional -8.8, 3-hop +6.2 / +9.1 scatter).
- 1D picks: 31/20/32 mean -6.07, 27/24/32 -6.46, 23/28/32 -6.60 (12 rows; 3-hop -8.0 / -10.9 / -13.5, 1-hop scatter
  -7.7 / -7.4 / -6.7).
- Results: `results/v4us_*` (2D, torus; job 071 used the one switch, its yamls are single-family), `v4usa_*` (1D,
  ring), `v4usc_*` (receiver 8, torus receiver 32), `v4usd_*` (mixes, 1D picks); `consolidation/slots_table.py
  PREFIX SUITE [TAG ...]` prints a sweep with the depths each run got.

**1D slot split in the builder** (2026-10-27, a09u32 jobs 076-077). User: keep 2D as is for now. New default in
`fabric_static_sized_channels_allocator.cpp` for Linear and Ring on Blackhole (VC0 only, no tensix extension, none of
the older study switches): the receiver keeps its tier depth (32), the sender channels share the other slots evenly,
remainder to the worker channel: 26 / 25 / 32 of 83 (was 16 / 16 / 32). `TT_FABRIC_EXP_TIER_SLOTS=1` restores the tier
depths. Mock check: Linear and Ring 26/25/32, Y-torus 17/16/16/16/16 unchanged.
- Data-checked at 26/25/32: main's router (kernel v1, flags 0, firmware keepalive) 1D 143/143, ring 19/19; v4u 143/143,
  19/19.

| cycles/packet, 4352 B | v4u tier | v4u 26/25 | main tier | main 26/25 | main 19/32 | main 32/19 | main 35/16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1D 1-hop write / bidirectional | 131.8 / 140.8 | 124.5 / 124.8 | 187.6 / 168.0 | 187.6 / 171.2 | 187.6 / 171.3 | 187.5 / 167.9 | 188.6 / 174.9 |
| 1D 3-hop write | 144.3 | 132.6 | 187.6 | 187.6 | 187.6 | 187.5 | 188.1 |
| Half-ring multicast write / scatter | 192.4 / 241.1 | 192.0 / 240.7 | 249.4 / 299.1 | 232.8 / 284.5 | 229.9 / 281.5 | 251.6 / 301.0 | 245.2 / 297.5 |
| Ring 1-hop write / bidirectional | 156.8 / 156.5 | 156.7 / 156.5 | 198.6 / 191.6 | 200.2 / 194.9 | 200.3 / 194.7 | 198.5 / 191.4 | 202.2 / 198.7 |
| Ring 3-hop write | 189.0 | 187.1 | 209.6 | 215.8 | 209.5 | 215.9 | 205.9 |
| Ring multicast write / scatter | 156.6 / 195.8 | 156.6 / 195.8 | 212.4 / 263.7 | 196.6 / 233.6 | 198.7 / 233.7 | 206.5 / 252.5 | 210.3 / 260.5 |
| Unidirectional ring multicast write | 233.9 | 233.0 | 273.0 | 279.2 | 272.8 | 279.3 | 268.4 |

(worker / forwarding; receiver 32 throughout. v4u 1D rows from `exp_1d_m1.yaml`, main's from `exp_1d_4352.yaml`; v4u at
32/19: 1-hop as 26/25, 3-hop 138.3.)
- v4u: 1D gains everywhere (1-hop -7.3, bidirectional -16.0, 3-hop -11.7, scatter -7 to -9), rings unchanged (3-hop
  -1.9). v4's sender channels have no power-of-2 path; the worker needs >= 19 slots.
- Main: deeper forwarding pays on multicast (ring multicast -16 / -30 scatter, half-ring -14 to -19), but every sender
  channel whose depth is not a power of 2 costs the flows through it 2-6 cycles/packet (`wrap_increment` and the 2N
  pointer wrap in `edm_fabric_flow_control_helpers.hpp` have power-of-2 paths): a 26- or 19-slot worker: ring 1-hop
  +1.6, bidirectional +3.1 to +3.3; a 25- or 19-slot forwarding channel: ring 3-hop +6.2, unidirectional multicast
  +6.2. 32/19 = tier on 1-hop rows; 19/32 = tier on forwarded rows. Main's best fit would be 16/32/32 (all powers of 2;
  not run), which gives up v4's 1-hop gain (16 worker slots bind v4).
- Results: `results/main1d_*`, `v4u1d_*` (`_split_`, `_tier_`, `_s<w>-<f>-<r>_`); scripts `consolidation/split1d_runs.sh`,
  `main_slots_runs.sh`.
- Kept form: the split applies only to the study's router in the v2 slot (`TT_FABRIC_KERNEL_VERSION=2`, v4); main's
  router keeps the tier depths, so its numbers are today's (mock check: kernel v1 ring 16/16/32, v2 26/25/32). Both
  pairings are the ones job 076 measured and data-checked. Routers with only the worker channel (neighbor exchange,
  the speedy path, which v4 does not run yet) keep their tier depths too.
- Main at 16 / 32 / 32 (all powers of 2, 80 of 83; job 078, with the tier depths again as the same-job control):
  data-checked 1D 143/143, ring 19/19. Every forwarded multicast row gains, nothing else moves (all within 0.4):
  ring multicast 212.4 -> 198.6 (scatter 263.7 -> 233.7), half-ring 249.2 -> 230.5 (300.2 -> 281.8), linear multicast
  163.6 -> 159.0 (221.0 -> 211.8). Neighbor exchange write 124.9 -> 142.7, but that is the study switch giving the
  single-sender router 16 worker slots instead of its tier 32, not the rule: a builder rule for main would leave it.
  Not a builder rule (the builder's computed split is for the router that replaces main); it is main's best measured
  1D allocation, the fair baseline when comparing main and v4 on 1D and ring rows.

**Is the batched completion path worth its complexity? v4sc1, v4sc2** (2026-10-27, a09u32 job 079; v4u as the same-job
control `v4uctl`). User: `release_completions` reads as complex; is it costing cycles to save cycles? The batched path
(no first-level acks: meshes, 1D, torus E/W) came from v3 (LEDGER E1-16 to E1-28), when E0's per-credit cost was high.
- v4sc1 = v4u with single-slot completions on every VC (the path rings already use); E1's mesh/1D receiver loop 2,908
  -> 2,187 B, stack loads 22 -> 17. v4sc2 = v4sc1 with the single-slot path checking a slot's writes once (a later
  pass credits it) instead of waiting up to 32 checks. Both data-checked: mesh 96, 1D 143, ring 19, Y-torus 12.

| suite (rows), cycles/packet vs v4u | v4sc1 mean (range) | v4sc2 mean (range) |
| --- | ---: | ---: |
| 2D 4352 B (10) | +12.80 (+1.2 to +19.4) | +11.77 (+0.8 to +18.8) |
| stream mixes (30, MixCmd out) | +12.91 (+4.4 to +20.8) | +12.28 (+4.6 to +18.8) |
| flushed-signal mixes (16) / chunk-4 (4) | +13.02 / +16.31 | +11.68 / +14.25 |
| bursty senders (44) | +2.18 (-0.7 to +17.2) | +1.93 (-2.0 to +16.4) |
| 1D (12) | +12.16 (0 to +22.0) | +10.93 (0 to +20.2) |
| ring (14) | -0.01 | -2.76 (-3.0 on every row) |
| Y-torus (12) | +8.57 (E/W +9 to +19.5, N/S 0) | +6.87 (N/S -3.0) |

- Batching still pays ~12 cycles/packet without acks (1-hop writes +9, scatter +16 to +20, small writes +16, 3-hop
  +19): a credit per packet costs both ERISCs (E1 a frame per completion, the far E0 a producer credit per completion).
  The complexity buys real cycles; it is not waste.
- The single-slot path's bounded wait is waste: checking once and leaving the slot to a later pass is -3.0 on every
  ring and torus N/S row (v4sc2 vs v4sc1 there). The batch path has the same bounded wait (E1-25, worth -1.6 to -4.8
  in v3); untested in v4.

**Where v4u's E1 waits: line-level profile** (2026-10-27, a09u32 jobs 080-081; `pcsv_v4uq2d` (2D), `pcsv_v4uqr` (ring);
new tool `consolidation/pcs_lines.py CACHE GDIR PREFIX ROW...`: the busiest ERISC's samples by innermost source line,
with each line's hottest instruction; a waiting instruction holds the PC, so a load's wait shows on its first use).
User's question: is there scope to restructure E1's pass for more overlap (grouped reads up front, header prefetch,
decode one packet while delivering the previous)?

| E1 wait, cycles/packet | scatter 146.0 | fused 152.6 | 3-hop 149.6 | multicast 146.8 | 16 B 119.5 | ring 1-hop 156.7 | half-ring 191.8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| command-buffer ready check before each NoC command | ~17-25 | ~20 | ~19 | ~15 | ~8 | ~8 | ~22 |
| header word, used at once | 8.7 | 8.5 | 10.6 | 9.8 | 8.9 | 8.4 | 6.3 |
| header loaded again by the ack loop (source channel) | | | | | | 8.6 | 6.9 |
| packets-received (+ TX status) read, used at once | 4.6 | 4.4 | 7.3 | | (idle) | 5.8 | 15.5 |
| downstream credit read (admission) | | | 3.6 | 5.7 | | | |
| completion: status read + bounded-wait spin | 2.3 | 2.3 | 2.1 | 2.6 | 1.9 | 12.0 | 11.7 |
| `mul` for the ack loop's header address | | | | | | 3.6 | 4.1 |

- The command-buffer checks are mostly one lone register read each (~7, `noc_cmd_buf_ready` used at once inside the
  NoC API before every command: `ncrisc_noc_set_transaction_id`, `noc_fast_write_dw_inline_with_state`); scatter and
  fused add real busy time (consecutive commands on one buffer). Grouping those reads needs NoC API variants that skip
  the internal check (`noc_write_with_state` and `noc_fast_atomic_increment_with_state` already take one;
  `ncrisc_noc_set_transaction_id` and the inline-write path do not).
- From the load-overlap microbenchmark (§17): distance hides nothing on this core (a lone load costs 6-8 at any
  distance); only reads issued together overlap, up to four. So a prefetch issued a phase early does not help, and a
  pipeline that decodes the next packet while delivering this one would add a pass of latency per hop and loop-carried
  state (spills) for no load-latency gain. The restructure is a same-pass reshape around groups of up to four reads.

**E1 pass reshape prototypes: v4x1, v4x2** (2026-10-27, a09u32 job 082; v4u as the same-job control `v4uxc`). Built from
the profile above, all data-checked (mesh 96, 1D 143, ring 19, Y-torus 12).
- v4x1 (stage 1): the next completion chosen first (`next_completion`, no loads), then one register group
  (packets-received, TX status, that completion's write-sent status), then acks, the completion (`credit_completion`)
  and the frame, then the packet; no bounded waits; `receive` no longer polls (the group's count is used every pass).
- v4x2 (stage 2): v4x1 + the packet's header word, routing fields and (one or two live downstreams) their local credit
  registers read as one group in `receive`.

| vs v4u, cycles/packet (mean, range) | v4x1 | v4x2 |
| --- | ---: | ---: |
| 2D 4352 B (10) | +5.75 (+2.6 to +7.8) | +11.05 (+4.4 to +17.1) |
| stream mixes (30) | +4.14 (-0.5 to +10.9) | +11.79 |
| flushed-signal mixes (16) / chunk-4 (4) | +6.24 / +8.23 | +13.32 / +15.97 |
| bursty senders (44) | +1.74 | +2.78 |
| 1D (12) | +7.08 | +11.21 |
| ring (14) / Y-torus (12) | +1.95 / +4.75 | +8.74 / +11.39 |
| latency 1-hop / 3-hop (ns) | -2 to +11 / +4 to +20 | +18 to +39 / +48 to +56 |

- v4x1's loss includes E0-bound rows (2D 1-hop write +5.7): credit timing, not E1's cost. The completion was chosen
  before the group with "a packet is waiting", which without acks is stale at the top of the pass (0 after the last
  retire): the newest slot was never held, the batch's proof was usually the slot issued a pass earlier (still
  leaving), and partial batches went out on the idle timer, so credits came smaller and later. Fixed in v4x3 (the
  previous pass's progress).
- v4x2's further loss fits §17: an L1 read in a group but used later costs ~9, and the credit registers were read and
  stored on every packet, also on rows that never forward. Dropped.

**E1 pass reshape: v4x3, v4x4** (2026-10-27, a09u32 jobs 083-084; same-job v4u controls `v4uxd`, `v4uxe`, within 0.1 of
job 082's). Both data-checked (mesh 96, 1D 143, ring 19, Y-torus 12).
- v4x3 = v4x1 with the completion chosen on "the previous pass handled a packet" (known before the group). v4x4 (hybrid)
  = with first-level acks v4x3's order (register group, acks + completion + frame, then the packet); without acks
  v4u's order (packet first, the count polled only when the kept count is used up, the TX queue read only when a frame
  is due), with v4x's completion code (`next_completion`, `credit_completion`, no bounded waits).

| vs v4u, cycles/packet (mean, range) | v4x3 | v4x4 |
| --- | ---: | ---: |
| 2D 4352 B (10) | +3.61 (+0.8 to +7.5) | +2.89 (+0.4 to +4.7) |
| stream mixes (30) | +3.42 | +1.14 (-0.1 to +4.5) |
| flushed-signal mixes (16) / chunk-4 (4) | +4.74 / +5.71 | +2.35 / +2.12 |
| bursty senders (44) | +0.52 | +0.82 |
| 1D (12) | +7.08 | +2.13 |
| ring (14) / Y-torus (12) | +1.01 / +3.24 | +1.02 / +2.92 |
| latency (ns) | 0 to +20 | -9.6 to +8.1 |

- None of the reshapes beats v4u. On this core grouping helps only reads a pass needs anyway: without acks v4u reads
  the count rarely (a lower bound kept between reads) and the TX queue once per frame, so reading them every pass, even
  grouped, adds cost (v4x1/v4x3 vs v4x4 there); with acks v4u already groups the per-pass reads, and sending the frame
  before the packet costs ~3-4 (v4x4 ring +1.0 against v4sc2's -2.8 for the same spin removal in v4u's order).
- The bounded wait: worth keeping in the batch path (v4x4's remaining +2 to +4 without acks, the E1-24 race: a full
  batch's newest write usually leaves just after the first check), pure cost in the single-slot path (-3, job 079).
- Left with real headroom (profile above): the command-buffer ready check (~7 per NoC command, more on scatter and
  fused), which needs NoC API variants that skip the internal check so the ready registers can be read in a group; and
  on ack VCs the second header load (~7) and the ack loop's `mul` (~3.6).

**Readable completions and the ring ack cursor: v4y, v4y3** (2026-10-27, a09u32 job 085; v4u control `v4uye`, within 0.1
of job 084's). v4y = v4u with `release_completions` split into `next_completion` (which credit is due) and
`credit_completion` (whether its writes have left, then the credit), the batch path keeping its bounded wait, single
slots checked once. v4y3 = v4y + an ack cursor on first-level-ack VCs (the ack loop steps a slot address instead of
`first_slot + index * size`). Both data-checked (mesh 96, 1D 143, ring 19, Y-torus 12).
- v4y: rings -2.74 (every row -2.9 to -3.3: the single-slot spin), Y-torus N/S -3; but scatter and fused rows +2.4 to
  +3.6 and 1D +2.25 (multicast scatter +6.9), with no change in the batch path's behavior: code layout of the split
  function. Above the 1-2 cycle tolerance: the split waits for a profile; the single-slot change alone is taken
  (v4ua below).
- v4y3: the cursor gives back ~1.2 of the ring gain (one more loop-carried value). Dropped.

**Single-slot completions without the wait, and grouped command-buffer checks: v4ua, v4ub** (2026-10-27, a09u32 job
086; v4u control `v4uyf`, within 0.2 of job 085's). Both data-checked (mesh 96, 1D 143, ring 19, Y-torus 12).
- v4ua = v4u with the single-slot completion checked once (a later pass credits a slot still being read): rings -2.73
  (every row -2.8 to -3.1 but two at -1.1/-1.2), Y-torus N/S -3.0, every mesh, 1D, mix, flush and burst row within
  0.2 (that code is unchanged). Latency -4 to +7 ns. Lead candidate.
- NoC API flags (user-approved; in the variant as an overlay of the tree's headers, so a job swaps them in only for
  itself): `skip_cmdbuf_chk = false` on `ncrisc_noc_set_transaction_id` and `noc_fast_write_dw_inline_with_state`
  (Blackhole and Wormhole), passed through `noc_inline_dw_write_with_state` (`dataflow_api.h`); defaults keep every
  caller's code. `offline_build.py` now puts a variant's copy of any include directory outside `tt_metal/fabric` first.
- v4ub = v4ua + `receive` reads the ready state of the local-write, forward-data and forward-credit command buffers in
  one group with the header loads, waits there for any not ready, and the packet's first command on each skips its
  check. Worse nearly everywhere: 2D +4.59, mixes +5.83, flushed +5.54, 1D +8.28, ring +7.18, Y-torus +5.30 (atomic
  rows +11, though they use none of those buffers); better only where a packet forwards: ring 3-hop -6.9,
  unidirectional ring multicast -10.3. Two mistakes in the shape: the wait moved before decode, so a buffer still
  draining the previous write is no longer overlapped with decode and admission; and the NIU register reads sit in
  the header's group, so decode's first use of the header waits behind them (loads complete in order, §17). A first
  build with fences around the group added 6-9 stack loads (the fence makes memory-held state reload) and an
  out-of-line wait added more (a call in the loop); both removed before the run.

**The forward's command-buffer checks paired: v4uf** (2026-10-27, a09u32 job 087; v4ua as the same-job control
`v4uac`, within 0.2 of job 086's). v4uf = v4ua + the NoC API flags (overlay) + in a stateful forward (one live
downstream: 1D, rings, E/W 2D routers) the data and credit command buffers' ready state read as a pair right before the
forward; a buffer not ready is waited for just before its command, as the API would; the two commands then skip their
own checks. Local writes, atomics and everything else as in v4ua. Data-checked: mesh 96, 1D 143, ring 19, Y-torus 12.
- Every suite within ±0.6 of v4ua except forwarded rows: ring 3-hop -6.8, unidirectional ring multicast -5.0,
  half-ring multicast -3.4, 1D 3-hop -3.1, 1D unidirectional multicast -2.0. 2D 3-hop does not move (bound by its
  forwarding slots' credit loop, not E1). Latency 3-hop -2 to -10 ns, 1-hop -4 to +7.
- Lead: `variants/v4` = v4uf (v4u + single-slot completions checked once + the paired forward checks; the NoC API
  flags travel with it as an overlay of `tt_metal/hw/inc` until they are applied to the tree).

**Galaxy check of the lead v4uf** (Slurm job 158477, bh-glx-110-a04u14, 200G; the Galaxy trees' host predates the 1D
slot split, so 1D/ring routers there keep 16/16/32). Data-checked: ring sanity 19/19, torus deadlock stability 87/87,
torus stability `_short` 56/56 (nightly in the same job).

| full-ring multicast, cycles/packet | main | v4u (two runs) | v4uf (two runs) |
| --- | ---: | ---: | ---: |
| 1024 / 2048 / 2560 / 3072 B | 276.0 / 271.0 / 282.4 / 261.4 | 216.2 / 216.6 / 216.9 / 217.2 | 207.3 / 207.9 / 208.1 / 208.7 |
| 3584 B | 261.1 | 218.8 / 218.9 | 240.1 / 235.0 |
| 4352 B | 263.8 | 303.5 / 305.2 | 345.5 / 348.9 |
| ring 1-hop 256 B / 1 KiB / 4352 B | | 156.3 / 156.6 / 249.6 | 153.3 / 153.7 / 249.6 |

- Router-bound sizes -4%, 1-hop -2%; but where the wire binds (4352 B, and now 3584 B, which v4uf's faster router
  pushes into the wire-bound regime) +8% to +14%: v4uf/main at 4352 B 0.76 (v4u 0.87).
- Suspect: the single-slot change (checked once, a later pass credits it). A saturated full ring moves on its free
  slot, so each hop's completion credit for it is on the ring's critical path; v4u's bounded wait credited it the
  moment its write left, v4uf a pass later. On the P150's rings nothing saturates, so the same change read as -3.
  The same reasoning names a candidate for the old 4352 B gap to main: v4 never credits the slot just written while
  packets keep coming (the newest slot is held until the next packet or an idle pass); main credits every slot as
  soon as its write has left. Tests: v4wt (v4uf with the bounded wait back) and v4eg (v4uf with single slots due as
  soon as one waits, main's rule).

**Single-slot completion timing: v4wt, v4eg** (2026-10-27, a09u32 job 088; v4uf control `v4ufc`, within 0.05 of job
087's; Galaxy Slurm job 158562). Both data-checked on P150 (mesh 96, 1D 143, ring 19, Y-torus 12). Also v4uf's torus
nightly on Galaxy (job 158477): 2604/2604.
- v4wt = v4uf with the single-slot bounded wait back (v4u's rule): rings +2.76, Y-torus N/S +3.0, all else within 0.4.
- v4eg = v4uf with single slots due as soon as one waits, the slot just written too (main's rule), checked once:
  rings -1.71 (-1.9 to -2.2 on 11 of 14 rows; ring multicast write +1.5), Y-torus N/S -2.0, all else within 0.4;
  latency -5 to +12 ns. Ring 1-hop ~151.6 (v4u 156.7).

**Single-slot completion timing on Galaxy** (Slurm job 158562, bh-glx-110-a04u14; v4uf, v4wt, v4eg against main, two
multicast runs each, agreeing within 0.5 except v4uf 4352 B 351.9 / 347.3). Data-checked: ring sanity 19/19 (v4wt,
v4eg), torus deadlock stability 87/87 and stability `_short` 56/56 (v4eg).

| full-ring multicast, cycles/packet | main | v4uf (checked once) | v4wt (bounded wait) | v4eg (eager, checked once) |
| --- | ---: | ---: | ---: | ---: |
| 1024-3072 B | 271.7-275.8 | 207.3-208.7 | 210.3-211.6 | 205.3-206.8 |
| 3584 B | 261.1 | 238.2 / 237.5 | 215.9 / 216.0 | 234.9 / 235.0 |
| 4352 B | 263.7 | 351.9 / 347.3 | 310.3 / 309.7 | 356.0 / 356.9 |
| ring 1-hop 256 B / 1 KiB | | 153.3 / 153.7 | 156.4 / 156.7 | 151.3 / 151.5 |

- The bounded wait is what a saturated ring needs: with it (v4wt) 4352 B is back at v4u's ~310 and 3584 B is the best
  of all (215.9); without it the free slot's credit goes a pass later at every hop. Crediting the newest slot early
  (v4eg) does not help there, so holding the newest slot is not the old gap to main (310 vs 264, still open).
- Where the router binds the wait costs ~3, on Galaxy as on P150 (v4wt +3 vs v4uf; v4eg -2 more).
- So the single-slot rule wants both: wait (bounded) on a pass without a packet, where E1 has slack, which is the
  wire-bound case; check once while packets keep coming. Goes into the completion component (V4_VC1_DESIGN §4.3).

**Main's router on VC1 routers: the reference for v4's VC1 work** (2026-10-27, a09u32 jobs 089 and 090;
`consolidation/vc1_main_runs.sh [fw|1000]`, labels `mainvc1_*` and `mainvc1ka_*`). Main's router (v1 kernel, flags 0)
on the VC1 split (`vc1/vc1_split.yaml`: the multi-mesh sanity suite, data-checked, then the intra-mesh rows on VC1
routers), hang capture armed, at the builder's level (-Os) and -O2, each in a fresh cache. All four runs clean: 34/34
finishes, no error lines. VC1 design step 1 (`consolidation/V4_VC1_DESIGN.md` §6).

| VC0 rows on VC1 routers, cycles/packet (slowest sender) | main -Os | main -O2 | v3q -Os / -O2 / -O3 | v3z -Os |
| --- | ---: | ---: | ---: | ---: |
| 1-hop E/W 16 B | 307.2 | 305.4 | 180.1 / 176.2 / 175.1 | 174.8 |
| 1-hop E/W 4352 B | 310.8 | 306.6 | 190.3 / 191.2 / 190.5 | 186.4 |
| 1-hop N/S 16 B | 309.6 | 304.9 | 188.5 / 179.8 / 181.4 | 185.0 |
| 1-hop N/S 4352 B | 303.5 | 302.8 | 197.8 / 192.8 / 191.9 | 193.0 |
| multicast 4352 B | 553.9 | 548.7 | 565.5 / 559.5 / 557.7 | 580.9 |

- Main is core-bound at ~305 on every 1-hop row, whatever the size or the keepalive: v1 reads only the study's
  override (`EXP_SEQ_UPDATE_TIMEOUT`), so `fw` ran at the firmware's 8000 and `1000` at 1000, and the rows agree within
  0.1 (as §44's untrimmed stock, then ~222 on VC0-only routers). The opt level is worth ~1%.
- v3 is ~40% faster than main on the 1-hop rows; on multicast main is 1-5% ahead of v3q and 5% ahead of v3z.
- Targets for v4 on VC1 routers: v3q's 1-hop rows (175-198) and main's multicast (549-554).

**v4vc: VC1 in v4** (2026-10-27, a09u32 job 091; `variants/v4vc`, `consolidation/v4vc_runs.sh`; design note
`consolidation/V4_VC1_DESIGN.md` §6a). v4vc = v4uf + VC1: on E0, VC1's forwarding channels (a second grouped free-slots
read and one test per pass, a far ring per VC, one shared table); on E1, the VC1 receiver channel with its state in a
record, one credit sender and one frame per pass for both channels, downstreams per VC (crossover: channel 0 forwards
on VC1's), intermesh landing (out of line, then Y-first decode) and exits (out of line; the destination mesh read with
the action bytes on edge routers). Init: VC1 runtime args, VC1 connections opened when receiver 1 is serviced (as
main), the credit block zeroed for every sender channel, and the per-entry init work in one out-of-line body each.

| | -O3 | -O2 | -Os |
| --- | ---: | ---: | ---: |
| VC1 E0 + E1, largest of each (limit 26,624 B) | 24,424 | 23,300 | 22,356 |
| first build, before the init changes | 29,972 | 27,756 | 26,356 |
| VC0-only 2D at -O3 (v4uf 20,124) | 17,692 | | |

- **Correct first time:** the VC1 split suite (multi-mesh sanity, data-checked, intra- and inter-mesh) 34/34 at -O3,
  -O2 and -Os, no error lines; every router ELF in the three caches is v4's.
- **VC0-only builds:** E0's loop is v4uf's code in all 136 configs (8 BH and Galaxy cases), E1's differs only in the
  setup before the loop. Validation (mesh 96, 1D 143, ring 19, Y-torus 12) passes; every suite's mean within 0.1 of
  the same-job v4uf control (`v4ufv`), no row beyond 1.0.
- **VC0 rows on VC1 routers** (cycles/packet, slowest sender):

| row | main -Os | v3q -Os / -O2 / -O3 | v4vc -Os / -O2 / -O3 |
| --- | ---: | ---: | ---: |
| 1-hop E/W 16 B | 307.2 | 180.1 / 176.2 / 175.1 | 194.5 / 186.2 / 187.3 |
| 1-hop E/W 4352 B | 310.8 | 190.3 / 191.2 / 190.5 | 205.6 / 198.5 / 199.0 |
| 1-hop N/S 16 B | 309.6 | 188.5 / 179.8 / 181.4 | 196.1 / 189.1 / 189.9 |
| 1-hop N/S 4352 B | 303.5 | 197.8 / 192.8 / 191.9 | 207.5 / 200.4 / 201.1 |
| multicast 4352 B | 553.9 | 565.5 / 559.5 / 557.7 | 588.4 / 586.9 / 588.2 |

  v4vc is ~40% faster than main on 1-hop, but 8-12 behind v3q (v3q from 2026-10-26) and on multicast 30 behind v3q,
  35-40 behind main. v3q uses the same completion policy on VC0's 8-slot ring (batches of 2, the newest slot held while
  packets keep coming), so that is not the difference. Next: PC sampling of both on these rows (job 092,
  `consolidation/vc1_pcs_runs.sh`, `vc1/pcs_vc1_*.yaml`: one long row each, the sampler beside the two ranks; the -g
  twins rebuild from a two-rank log: 96/96 configs).

**v4vc and v3q on VC1 routers, PC-sampled** (2026-10-27, a09u32 job 092; `consolidation/vc1_pcs_runs.sh VARIANT TAG
LEVEL ROW...`, `vc1/pcs_vc1_{compile,ew16,ew4352,mcast}.yaml`: one long row of the split suite each, pc_sampler.py
beside the two ranks, 96 targets, 155-217k samples per row; -g twins from the two-rank compile log, 96/96 and 94/94;
`pcs_lines.py --target CHIP/CHAN` added to profile the same router in both). Both at -O3, same job:

| row (cycles/packet, slowest sender's pps) | v3q | v4vc |
| --- | ---: | ---: |
| 1-hop E/W 16 B | 175.0 | 187.6 |
| 1-hop E/W 4352 B | 190.5 | 199.5 |
| intra-mesh multicast 4352 B (pps metric: half the split table's) | 279.0 | 294.0 |

- **These rows are bound by the credit round trip on the VC1 tier's small rings, and E0's pass time is a term in
  that loop.** VC1 builds get VC0 receiver rings of 8 slots, VC0 forwarding channels of 4, and the worker 51.
  - 1-hop E/W 16 B: v4vc's E0 spends 71 cycles/packet in `send` and ~104 in passes that send nothing: one or two
    passes per packet waiting for far-ring credits, so E0 is not saturated (8 slots x 187 = ~1,500 cycles per slot
    round trip; v3q 1,400). A freed far slot waits for E0's next pass, which is why a leaner E0 took 31 off at the
    same slots in §86: both the slots and E0's per-pass time set these rows.
  - Multicast: the busiest E1 spends 49 cycles/packet in admission (the downstream credit re-reads of 4-slot
    forwarding channels); the split table's multicast rows (main 549-554, v3q 558-566, v4vc 587-588) sit at the
    ~550-600 cap that 4 slots put on a forwarded flow (V4_VC1_DESIGN §4.4).
- **Same router, line level (E/W 16 B):** E0's pass has v3q's shape (VC0 group of four, worker visit, VC0 forwarding
  test, VC1 group of three and one test). v4vc's E0 waits longer:
  - group reads and their first use: ~66 cycles/packet (v3q ~50);
  - the TX-queue register writes in `send`: 22 (v3q 12).
  The two E1s look alike (packets-received polling 26 vs 29, local-write command buffer 11 vs 10, completions
  similar). On a slot-bound row these per-pass costs set how fast the loop turns a credit around.
- So the levers for VC1-router rows are the VC1 allocation (design step 6; the slot override is VC0-only today) and
  E0's per-pass cost (the ~10 to v3q on 1-hop).

**Completion design B, mechanism only: v4gi** (2026-10-27, a09u32 job 093; `variants/v4gi`, `consolidation/
v4gi_runs.sh`; v4vc as the same-job control `v4vcg`, within 0.2 of job 091 on every suite and within 0.3 on the VC1
rows). v4gi = v4vc with one transaction ID per aligned completion group (`GroupIds`: slot i carries
first + (i / T) mod ids), so one status read proves a group over both command buffers; slot records hold only the
sender channel (no command-buffer bits, no ordering argument, no ring wrap: a group of 1, 2 or 4 aligned slots lies in
one record word); `release_completions` as due / proof / credit, one path for every group size; the tracker leaves the
receiver (teardown still waits on the channel's ID range). Timing as v4vc (same due rule, a group of several slots
waits up to 32 checks, a single slot is checked once), with two deliberate differences: a group whose slots no command
read (atomics) now costs one status read, and after a partial (idle) credit the next credit is the rest of that group
(alignment). Group size now a power of 2 (rings of 12-15 slots: 2, was 3). Data-checked: mesh 96, 1D 143, ring 19,
Y-torus 12; VC1 split 34/34 at -O3 and -Os. E1 loop 713 -> 572 instructions (2D, median); VC1 E0 + E1 at -O3 23,884 B.

| suite (vs v4vc, same job) | mean | min | max |
| --- | ---: | ---: | ---: |
| 4352 | -2.84 | -7.9 | +1.9 |
| mix | -1.06 | -6.2 | +3.3 |
| flushmix | -4.08 | -9.0 | -0.4 |
| flushmix4 | -3.94 | -7.0 | +0.1 |
| burst | +0.12 | -2.4 | +4.9 |
| 1d_m1 | -3.77 | -6.7 | +0.0 |
| ring | -2.71 | -3.9 | +1.6 |
| torusy | -1.29 | -7.7 | +2.9 |

VC0 rows on VC1 routers, -O3 (-Os): 1-hop E/W 16 B -3.8 (-2.6), 4352 B -1.3 (+0.7), N/S 16 B -4.1 (-2.6), 4352 B
-0.8 (0.0), multicast -8.2 (+0.2). Latency within -18 to +7 ns.
- Gains: scatter -5 to -7 everywhere (2D, 1D, Y-torus E/W), fused -4 to -6, flushed atomics -4 to -9, every ring row
  -2.7 to -3.9 (rings keep one slot per group and the same IDs: the gain is the smaller loop and no record bits).
- Losses: pure atomics +2.0 / +3.3 (the status read for groups no command read); MixSize4 +2.1 / +3.2; bursts of 16 B
  (N8 G500) +3.4 / +3.7 and N/S N32 4352 B +4.9; Y-torus N/S scatter +2.9; 3-hop writes +1.9 / +2.1 (Mesh, Y-torus E/W);
  ring multicast write +1.6. Bursts and mixed sizes are where partial (idle) credits happen, so the alignment rule is
  the suspect there; not yet profiled.

**v4gi's burst losses, PC-sampled** (2026-10-27, a09u32 job 094; `pcsv_runs.sh`, new long rows `pcs3/burstns8.yaml`,
`burstew8.yaml`, `burstns32.yaml`: exp_burst.yaml's rows at ~20M packets; v4gi `pcsv_v4gib_*`, v4vc `pcsv_v4vcb_*`).

| row (long) | v4vc | v4gi |
| --- | ---: | ---: |
| N/S bursts of 8 x 16 B, gap 500 | 132.1 | 138.1 |
| E/W bursts of 8 x 16 B, gap 500 | 130.9 | 130.1 |
| N/S bursts of 32 x 4352 B, gap 500 | 135.1 | 139.9 |

- The N/S rows reproduce (+6.0, +4.8); E/W 16 B does not (the suite's +3.4 there was a short row).
- N/S 16 B, per packet: the busiest E0 spends more in the worker visit (+2.6) and in worker credits (+1.7: 3.8 ->
  5.5), so completion credits reach it in more, smaller pieces; the busiest E1 polls more (26.9 -> 30.5), so E1 is not
  the limit. N/S 4352 B: the same, smaller (E0 worker visit +1.6, worker credit +0.5; E1 completions +2.4).
- Cause (from the code): E0 drains the worker channel continuously, so the gaps E1 sees fall at any slot and an idle
  pass credits part of a group. v4vc then goes on in full batches from wherever it stopped; v4gi credits the rest of
  that group first (an extra, smaller credit), and that rest became due only once more than a whole group waited, so
  its slots were also held longer. Fixes in job 095: v4gi2 (due unit = the rest of the oldest group), v4gi3 (v4vc's
  unit, spanning two groups with two ID proofs), both with a READ bit in the records (no proof for atomics).

**Fixes for v4gi's losses that did not work: v4gi2, v4gi3, v4gir** (2026-10-27, a09u32 jobs 095 and 098; job 095 was
cut by the broker's 600 s default during its controls' validation, so the device was held for an eth/fabric fault and
reset by the broker; job 098 ran the controls v4gi `v4gic` and v4vc `v4vcd`, with v4gic within 0.1 of job 093's v4gi on
every suite). All three add a READ bit to the slot records (set by forwards and by deliveries that read the slot; no
proof for a group no command read). v4gi2: due unit = the rest of the oldest group, `first` and its group end computed
every pass before the due test. v4gi3: v4vc's unit (oldest T slots), spanning at most two groups, two ID proofs.
v4gir: the bit alone, v4gi's timing. Data-checked (mesh 96, 1D 143, ring 19, Y-torus 12); VC1 split 34/34 at -O3.

| suite mean vs v4gi | v4gir (job 098, same job) | v4gi2 (job 095) | v4gi3 (job 095) |
| --- | ---: | ---: | ---: |
| 4352 | +2.05 | +4.32 | +11.30 |
| mix | +1.12 | +3.47 | +6.49 |
| flushmix | +2.73 | +6.11 | +12.32 |
| flushmix4 | +2.88 | +6.77 | +15.68 |
| burst | +0.20 | +0.51 | +1.61 |
| 1d_m1 | +2.88 | +6.33 | +13.07 |
| ring | +5.36 | +8.37 | +5.44 |
| torusy | +1.55 | +3.90 | +7.39 |

VC1-router rows at -O3 (job 095): v4gi 183.4 / 197.9 / 185.8 / 200.5 / 579.8; v4gi2 +5 to +10; v4gi3 +4 to +18.
- The READ bit costs E1 +1 to +5 on every suite (rings +5.4) and saves 1.1 on the pure atomic row it was for: carrying
  the record through delivery costs E1 registers (static E1 stack loads, 2D: 6 -> 19). Dropped.
- v4gi3's spanning proof waits on the second group's ID, which also covers that group's newer slots (with groups of
  2 on VC1 builds' 8-slot rings, often the slot just issued): VC1 N/S +18. Group IDs and spanning credits do not mix.
- v4gi2's due arithmetic before the due test adds +2 to +4 on top of the bit. Next: v4gi4, the relative due as
  T - ((next_index - waiting) & (T - 1)) without the bit (rings and tori compile to v4gi's code; 2D E1 +11
  instructions, no added stack traffic); job 102.

**v4gi4: due relative to the group, no READ bit** (2026-10-28, a09u32 job 102; v4gi as the same-job control `v4gie`,
VC1 rows within 1.0 of jobs 093/095). Data-checked (mesh 96, 1D 143, ring 19, Y-torus 12); VC1 split 34/34.

| suite mean vs v4gi | 4352 | mix | flushmix | flushmix4 | burst | 1d_m1 | ring | torusy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v4gi4 | +1.71 | +1.59 | +2.91 | +2.87 | +0.19 | +2.68 | +0.01 | +0.98 |

- Long burst rows (pcsv_v4gi4b, against job 094): N/S 8 x 16 B 138.5 (v4gi 138.1, v4vc 132.1), N/S 32 x 4352 B 140.6
  (v4gi 139.9, v4vc 135.1). Not fixed.
- So the longer holds after a partial credit are not the burst loss: v4gi4 credits the rest of a group as soon as it
  and one newer slot are retired, and the rows did not move. What is left of job 094's evidence is the far E0's extra
  worker-visit and worker-credit time; E1's frame-send time per packet is the same in v4vc and v4gi (10.4 vs 10.3), so
  it is not simply more credit frames. Cause open.
- The +11-instruction due test costs +4 on every scatter, fused and atomic row (E1-bound rows) with no added stack
  traffic: E1 at the register limit again. Dropped.
- Net: v4gi stays as decision 1's result; its losses (N/S bursts +4 to +6, pure atomics +2 to +3) stand.

**v4gi's burst losses diagnosed: the worker's credit window** (2026-10-28, a09u32 jobs 103-104;
`consolidation/v4dg_runs.sh`, `v4dg2_runs.sh`, decoder `dg_report.py`). Diagnostic builds v4gidg / v4vcdg = v4gi / v4vc
plus `router_diag.hpp` counters, written into the phase-profile block that `TT_FABRIC_TXQ_COUNTERS=1` allocates and
compiled away without it (offline, 163 configs each: loops identical to v4gi / v4vc). E0: worker credit sizes, cycles in
`credit_worker`, passes where the worker had a packet and the far ring no room. E1: credit sizes (split by idle pass and
group alignment), frames, proofs, idle passes with a partial group waiting. The counters perturb the rates (steady N/S
16 B 135 with them, 121 without), so rates are compared within one setting; without them the long N/S 8 x 16 B row
reproduces job 094 in the same job (v4gi 138.08, v4vc 132.11).

| long N/S 8 x 16 B, counters on (job 103) | v4vc | v4gi |
| --- | ---: | ---: |
| cycles/packet | 139.85 | 148.12 |
| credits per packet (slots per credit) | 0.252 (3.97) | 0.291 (3.44) |
| credits of 4 / of 2 slots | 98% / 1% | 72% / 27% |
| idle passes with a partial group waiting, per packet | 0.117 | 0.321 |
| E0: cycles per `credit_worker` / passes blocked by the far ring | 27.0 / 0 | 27.1 / 0 |
| E1: proof cycles / failed proofs / frames put off | 21.3 / 0 / 0 | 18.0 / 0 / 0 |

- Not E0's cost per credit: `credit_worker` is 27 cycles in both, so v4gi's extra 0.039 credits per packet cost E0
  ~1 cycle per packet against a loss of 6 to 8. Over the burst suite, rows near link rate lose 140-220 cycles per extra
  credit; rows paced by the worker's own gaps (bursts of 1-4) get as many extra credits and lose nothing.
- Not the far ring: E0 never had a worker packet with the far ring full.
- The worker's credit window binds (job 104, counters off): with `TT_FABRIC_EXP_WORKER_SLOTS=32` (2D default 17) every
  burst row runs 7-15 faster and v4gi equals v4vc on all of them (long N/S 8 x 16 B 122.92 / 122.99, N/S 32 x 4352 B
  125.71 / 125.71; suite within 0.1 except E/W 32 x 4352 B G2000 +1.9 and the steady 16 B rows +0.6 / +0.7). At 17:
  +5.98 and +4.77 again.
- Mechanism: v4gi's groups sit at fixed ring positions; v4vc's batches restart wherever a partial batch ended, so they
  line up with the stream's gaps. At a gap (the worker waiting on credits starves E0, so E1 sees one) v4vc usually
  has a full batch waiting, credited on the first idle pass; v4gi often has a partial group, held 4 idle passes while
  the worker waits on exactly those credits (idle passes with a partial waiting: 0.321 vs 0.117 per packet).
- Partial groups on the first idle pass (v4gidg1, diagnostic) at 17 worker slots: long N/S 8 x 16 B gap +5.98 ->
  +3.07, suite N/S and E/W 8 x 16 B G500 +3.3 -> +0.6 and +3.1 -> +1.2, but N/S 32 x 4352 B unchanged (+4.9) and steady
  16 B rows +1.7 to +3.0 (+2.7 to +4.4 at 32 worker slots: 0.33-0.37 credits per packet, E0 pays per credit). The hold
  is part of the loss, not all of it; not a fix.
- So v4gi's burst loss exists only while the 17-slot worker window binds. Results `results/dg_*`, `results/dg2_*`.

**Bursts of 6 and 12** (2026-10-28, a09u32 job 105; `v4dg3_runs.sh`, new suite `exp_burst612.yaml` (`burst612`):
N = 6, 12 x G = 500, 2000 x E/W, N/S x 4352 / 16 B). exp_burst.yaml's N are all multiples of T = 4, so v4vc's batches
(restarting at a gap) end exactly at the next gap there.
- v4gi - v4vc at 17 worker slots: mean +0.37 over 16 rows; N12 G500 16 B +2.3 (E/W) / +2.0 (N/S), N6 G500 +0.2 to
  +0.9, G2000 rows equal (worker-paced). The burst suite in the same job: N8 G500 16 B +3.2 / +3.5, as jobs 103-104.
- At N = 6 v4vc leaves partial batches too: 8-10% of its credits are 2 slots (counters, `dg3_*_burst612_c17`).
- At 32 worker slots: mean +0.04 (all within 0.1 but N/S N12 16 B +0.5). The window is worth more than the design
  difference: N12 G500 16 B 131.3 -> 118.1, N/S 4352 B 138.0 -> 125.9 (both designs).
- Analysis of the problem and its levers: `consolidation/COMPLETION_CREDITS.md`.

**The worker credit loop's round trip, measured** (2026-10-28, a09u32 jobs 106-108; `v4rt3_runs.sh`, decoder
`rt_report.py`). v4girt / v4vcrt = v4gi / v4vc plus `router_rt.hpp`: sampled timestamps, each interval on one chip's
clock. E0: a worker packet's send -> E0 reading a completion count that covers it (every 61st packet). E1: the
watched slot at the head -> retired -> its group credited -> the frame carrying it sent (~1 in 64). Compiled away
without `TT_FABRIC_TXQ_COUNTERS=1` (offline: loops identical to v4gi / v4vc). Job 106's first version timed every
credit and sampled every ~5th packet on E1, which made E1 the bottleneck (+24 to +47 cycles/packet); job 107 sampled
every 64th packet on E0, always the last packet of a burst of 8 or 32, and timed E1's own bookkeeping. Job 108 is the
valid one; its hooks still cost +2 to +6 (means), +7.6 / +9.3 on the long N/S 8 x 16 B row, so absolute times are a
little high. Plain v4gi with the profile buffer allocated: +0.01 (layout is not it).

| cycles, v4gi (v4vc) | steady N/S 16 B | steady N/S 4352 B | N/S 8 x 16 B G500 | N/S 12 x 16 B G500 |
| --- | ---: | ---: | ---: | ---: |
| E0 send -> completion seen | 1319 (1529) | 1505 (1580) | 1349 (1431) | 1351 (1426) |
| E1 head -> retired | 90 (90) | 91 (91) | 90 (90) | 90 (90) |
| E1 retired -> credited (batching wait, proof) | 350 (325) | 336 (343) | 368 (344) | 356 (344) |
| E1 credited -> frame sent | 50 (57) | 50 (54) | 51 (54) | 51 (55) |
| rest: link out + link back + E0's look | 829 (1056) | 1028 (1092) | 840 (942) | 854 (936) |
| packets still in the worker channel at the send | 2.7 (1.9) | 2.0 (1.9) | 2.2 (2.1) | 2.7 (2.3) |

- The link dominates: the rest is ~63-70% of send -> seen, ~820-950 cycles at 16 B and ~970-1,090 at 4352 B. With
  the latency suite (1-hop one-way 905 cycles worker to worker at 16 B, ~150 of it router work), the link is ~400-500
  cycles each way. No router change touches it.
- Batching is the largest router term: 325-370 cycles (~25%; a sample sits ~1.6 slots from its group's oldest).
  E1's own work is ~90, the frame ~50.
- E0's queue ahead of a send (2-3 packets, ~300-400 cycles) and the credit write (27) are on the worker's side of the
  loop, outside send -> seen; the worker's loop is ~1,800 cycles in all, against 17 slots x ~120 = ~2,040: at the edge.
- The mean does not explain the v4gi / v4vc difference: v4gi's send -> seen is shorter on every row (by 50-210), yet
  N/S 8 x 16 B is slower. The loss is in which credits are late when the worker is blocked (the gaps), not in the mean.
- Window arithmetic: a ring of W slots caps a link at RT / W cycles per packet (RT: E0's send -> seen, plus its look).
  At the 2D 16 B round trip that is ~1,350-1,500 / 8 = ~170-190 for VC1 builds' 8-slot VC0 rings, close to what the
  VC1 1-hop rows run at (v4gi 183-201; their round trip was not measured here, T = 2 there); 4-slot rings ~350. No
  completion policy can lift that far: the link alone takes ~900 of the round trip.

**VC1 allocation sweep, VC0 side** (2026-10-28, a09u32 job 109; `vca_runs.sh`, table `vca_table.py`). New study switch
`TT_FABRIC_EXP_SLOTS_VC1=worker,forwarding0,receiver0,forwarding1,receiver1` (`fabric_static_sized_channels_allocator.cpp`;
0 = the rest to the worker, every router the same depths; checked on the mock first: both ranks get it and v4gi
compiles). New yaml `vc1/vc1_alloc.yaml`: vc1_split.yaml's tests plus a 2-hop unicast ([0, 0] -> [0, 3], through a VC0
forwarding channel), the 2-hop merged with a 1-hop flow from each middle chip, and bursts of 8 and 32 (G500) on 1-hop
E/W. v4gi -O3, 7 allocations x 2 passes (the second in reverse order): all 14 runs clean (sanity data checked, 50/50
finishes), pass-to-pass spread <= 0.7.

| cycles/packet, v4gi -O3 | main 51,4,8,2,4 | r16 (w43) | f8 (w39) | f8r16 (w31) | f8r16v (w21) | f12r16 (w19) | f8r32 (w15) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1-hop E/W 16 B / 4352 B | 183.5 / 198.0 | -25.7 / -40.4 | +0.2 / +0.1 | -25.8 / -40.3 | -25.7 / -40.1 | -25.6 / -40.0 | -25.5 / -37.2 |
| 1-hop N/S 16 B / 4352 B | 185.9 / 200.6 | -28.3 / -43.0 | 0.0 / +0.1 | -28.3 / -42.9 | -28.2 / -42.8 | -28.0 / -42.6 | -28.0 / -39.4 |
| multicast 4352 B | 289.4 | +7.4 | -84.3 | -119.9 | -119.0 | -149.2 | -119.8 |
| 2-hop 16 B / 4352 B | 482.4 / 530.2 | +46.2 / +42.8 | -223.2 / -250.2 | -192.0 / -219.2 | -190.0 / -217.9 | -267.7 / -308.3 | -191.8 / -219.6 |
| 2-hop merged 16 B / 4352 B | 257.6 / 283.0 | +19.8 / +19.2 | -77.6 / -86.9 | -104.1 / -119.6 | -102.0 / -118.7 | -119.1 / -144.2 | -104.0 / -119.6 |
| bursts 8 x 16 B / 4352 B | 183.4 / 197.9 | -25.8 / -40.3 | +0.3 / +0.1 | -25.5 / -40.1 | -25.5 / -39.9 | -25.4 / -39.8 | -15.6 / -15.7 |
| bursts 32 x 16 B / 4352 B | 183.5 / 197.8 | -25.5 / -40.2 | +0.1 / +0.2 | -25.7 / -40.0 | -25.4 / -39.8 | -25.4 / -39.7 | -24.1 / -34.5 |

(Allocations: worker, VC0 forwarding, VC0 receiver, VC1 forwarding, VC1 receiver; columns after main are differences.)
- The VC0 receiver binds every 1-hop row: 8 -> 16 slots is -26 to -43 (E/W 4352 B 198 -> 158). 32 slots adds nothing,
  so at ~158 those rows are bound elsewhere (E0's pass over 7 sender channels; VC0-only routers run them at ~120-130).
- The VC0 forwarding channels bind everything forwarded: 4 -> 8 halves the 2-hop row (482 -> 259), 12 takes it to 215
  (530 -> 222 at 4352 B); multicast 289 -> 140.
- The worker needs ~19: at 15 (f8r32) bursts of 8 lose 10-24 of the gain; 19 and up are the same as 51.
- **The group size works against the forwarding window.** A 16-slot receiver credits groups of T = 4. On a 4-slot
  forwarding channel that is the whole window: r16 makes the 2-hop row +46 slower although it speeds up every 1-hop
  row. On an 8-slot one it is half: f8r16's 2-hop row is 31 slower than f8's (8-slot receiver, T = 2). T comes from
  the ring, but the window that binds is the far sender channel's.
- VC1 doubled (f8r16v: VC1 forwarding 4, receiver 8, T = 2 there) runs clean and costs the VC0 rows nothing (within
  2); VC1's own traffic is not measured by these rows.

**Inter-mesh benchmark rows, and the group size against the window** (2026-10-28, a09u32 job 110; `vcb_runs.sh`).
- **The profiler fix:** `BandwidthProfiler` now traces configs without hops (inter-mesh traffic, routed by the tables)
  through the control plane's forwarding direction node by node (`trace_node_id_traffic_path`), and takes their first
  direction from it (`first_hop_direction`); before, it aborted on `hops.value()` (job 216). New yaml
  `vc1/vc1_inter.yaml`: [0, 0] to each chip of mesh 1. All 16 inter runs clean (16/16 finishes).
- **v4git** (diagnostic) = v4gi with each receiver's completion group sized to the smaller of its ring and the smallest
  sender channel of its VC (every router has the same depths, so the far sender channels'); VC0-only builds are
  byte-identical to v4gi (141 of 163 offline configs; the 22 VC1 ones differ), VC1 E0 + E1 23,564 B at -O3.
- v4gi and v4git interleaved per allocation, one run each, main's allocation first and last (drift <= 1.1). All 16
  alloc runs clean (50/50).

| v4gi -O3, cycles/packet | main 51,4,8,2,4 | f12r16 0,12,16,2,4 | f10r16v4 0,10,16,4,4 | f8r16v6 0,8,16,6,4 | f8r16v 0,8,16,4,8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| worker slots | 51 | 19 | 19 | 19 | 21 |
| 1-hop E/W 16 B / 4352 B | 183.5 / 198.0 | 157.9 / 158.0 | 158.0 / 157.9 | 158.4 / 158.0 | 157.8 / 157.8 |
| 2-hop 16 B / 4352 B | 482.2 / 530.3 | 214.7 / 222.1 | 240.5 / 258.2 | 290.4 / 310.8 | 292.4 / 312.2 |
| multicast 4352 B | 289.5 | 140.2 | 144.6 | 169.4 | 170.6 |
| inter-mesh, direct link 16 B / 4352 B | 232.5 / 236.6 | 206.4 / 207.1 | 206.4 / 207.1 | 206.4 / 207.1 | 206.4 / 207.1 |
| inter-mesh, then VC1 hops, 16 B (3 rows) | 780-853 | 778-855 | 398-446 | 310-375 | 455-497 |
| same, 4352 B | 894-966 | 901-965 | 456-500 | 371-408 | 498-547 |

- **VC1's 2-slot forwarding channels cap every inter-mesh flow that goes on inside the destination mesh** at 780-970
  cycles/packet (6.1-6.6 GB/s at 4352 B against 24.8 direct), 3-4x the direct link. 4 slots: 400-500; 6: 310-410. The VC0 slots don't
  touch these rows. Main's allocation has this cap today.
- With VC1 forwarding at 4, a 4-slot VC1 receiver (T = 1) beats an 8-slot one (T = 2) by 40-60: the same group-against-
  window effect as on VC0.
- **v4git - v4gi** (T from the smallest sender channel instead of the ring; pairs are 16 B / 4352 B):

| allocation (VC0 T: v4gi -> v4git) | 1-hop | 2-hop | multicast | 2-hop merged | inter direct | inter + VC1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| main (2 -> 1) | -7 to -11 | -47 / -32 | -15 | -18 / -16 | +17 / +16 | -2 to -13 |
| f8r16 (4 -> 2) | +9.5 | -34 / -33 | +4 | +11 / +6 | +19 | -7 to +4 |
| f10r16v4 (4 -> 2) | +9.5 | +1 / -11 | +14 | +17 / +16 | +19 | -4 to +10 |
| f12r16 (4 -> 2) | +9.5 | +22 / +17 | +16 | +17 / +17 | +19 | -7 to +5 |
| f8r16v (VC0 4 -> 2, VC1 2 -> 1) | +9.5 | -36 / -34 | +3 | +10 / +5 | +19 | -37 to -56 |

- So a smaller group helps exactly while it is the forwarding window that binds (4-slot channels; 8 for the plain
  2-hop row) and costs ~9.5 on every 1-hop row otherwise (more credits for E0, whose pass binds those rows here), and
  16-22 on forwarded rows once the forwarding channels have 10-12 slots. With forwarding sized to ~3 groups or more,
  the ring's group (v4gi) is the better rule; v4git only rescues allocations that are too small. Not pursued.

**The VC0 / VC1 forwarding split, and a mixed load** (2026-10-28, a09u32 job 111; `vcc_runs.sh`, tables
`vca_table.py`, `vcc_mix.py`). v4gi -O3, worker 19, VC0 receiver 16, VC1 receiver 4 (T = 4 and 1), the forwarding
slots (VC0 f0, VC1 f1, 3 channels each) split five ways; main's allocation first and last, a10 twice (drift <= 0.7).
New yaml `vc1/vc1_mix.yaml`: [0, 0] -> [1, 0] (crosses to [1, 1], then W on VC1) together with a VC0 flow on the same
[1, 1] W link, either [1, 1]'s worker ([1, 1] -> [1, 0]) or a forwarded one ([1, 3] -> [1, 0], N then W); one
sender per rank, so each rank's CSV row is its own flow. All 24 runs clean (50/50, 16/16, 16/16).

| cycles/packet | main 51,4,8,2,4 | 12 / 2 | 10 / 4 | 9 / 5 | 8 / 6 | 7 / 7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1-hop E/W 16 B / 4352 B | 183.4 / 198.0 | 157.9 / 158.0 | 157.8 / 158.0 | 157.8 / 158.0 | 157.9 / 158.0 | 158.2 / 158.0 |
| 2-hop 16 B / 4352 B | 482.2 / 530.0 | 214.5 / 221.8 | 240.5 / 258.2 | 264.2 / 283.0 | 290.4 / 311.1 | 328.7 / 353.6 |
| multicast 4352 B | 289.5 | 140.1 | 144.8 | 156.0 | 169.3 | 188.1 |
| 2-hop merged 16 B / 4352 B | 257.5 / 283.0 | 138.5 / 138.8 | 138.7 / 140.8 | 143.8 / 150.3 | 153.7 / 163.3 | 170.0 / 183.1 |
| inter-mesh direct 16 B / 4352 B | 232.4 / 236.7 | 206.3 / 207.2 | 206.3 / 207.2 | 206.3 / 207.2 | 206.3 / 207.2 | 206.3 / 207.2 |
| inter-mesh + VC1 hops, 16 B (3 rows) | 781-853 | 778-855 | 398-446 | 320-369 | 310-375 | 310-372 |
| same, 4352 B | 894-966 | 901-965 | 456-500 | 378-404 | 371-408 | 371-406 |
| mixed: VC0 worker / VC1 flow, 16 B | 227 / 843 | 216 / 833 | 221 / 453 | 216 / 380 | 216 / 379 | 216 / 378 |
| mixed: VC0 forwarded / VC1 flow, 16 B | 528 / 830 | 245 / 814 | 291 / 445 | 314 / 376 | 337 / 368 | 380 / 371 |

(Columns: VC0 forwarding / VC1 forwarding slots per channel; the worker gets 19 in all of them. Mixed rows: the two
flows overlapped 0.62-0.98 of the longer one's time, 0.39-0.65 at 12 / 2, where the slow VC1 flow ran on alone.)
- **VC1 forwarding has a knee at 5.** 2 -> 4 -> 5 takes the VC1 rows 780-970 -> 400-500 -> 320-400; 6 and 7 add
  nothing. At 5 the VC1 receiver (4 slots) binds instead: ~1,300 / 4 = ~325.
- **VC0 forwarding is still window-bound at 12** (2-hop 215, ~2,400 / 12 = 200) and every slot moved to VC1 costs it,
  more the smaller the channel: the 2-hop row +13 to +18 per slot from 12 to 10, +24 to +28 from 10 to 8, +38 to +43
  from 8 to 7; multicast +2 to +19 per slot.
- Under the mixed load the split moves each flow as it does alone, and the shared link costs both about the same at
  every split: the VC1 flow 40-70 slower than alone, the VC0 worker 216-225 (alone 158; main 227), the forwarded VC0
  flow 24-54 slower.
- So 9 / 5 (19, 9, 16, 5, 4: 81 slots) puts VC1 at its knee for 24 / 25 on the VC0 2-hop row and 11 on multicast
  against 10 / 4, and 74-96 faster VC1 rows. Limits: 2x2 meshes (at most 2 hops a side), one or two flows.

**The allocation from a model, checked; main's router on the same rows** (2026-10-28, a09u32 job 112; `vcd_runs.sh`).
User: do not hardcode a split; find an algorithmic way, since the split moves with the router's shape; keep the data
for the case that free depths (no power-of-2 constraint) lift other bottlenecks (`consolidation/ALLOCATION_NOTE.md`).
- **Model** (`consolidation/alloc_model.py`): a row runs at max(service bound, slowest window on its path), a window of W
  slots allowing one packet per RT0 / W + b cycles (RT0: its credit loop's round trip; b: what each packet in flight
  adds), fitted per window kind (worker, VC0 ring behind an intra- / inter-mesh link, VC0 / VC1 forwarding, VC1
  ring), completion group and size. Fit to jobs 109-111: 80 points, mean error 0.2%, max 2.0%.
- **Allocation**: receivers at T = 4 sized to stop binding at the 1-hop bound (16), the worker from its round trip (18),
  then max-min water-filling of the rest between VC0 forwarding (3-slot steps) and the VC1 path (forwarding 3-slot
  steps, ring 1-slot), weighted VC0:VC1. The weight is the only policy input.
- **Check** (predictions written before the run, `vcd_predictions.txt`): every VC0 row within 1.3% (18-slot worker
  included: bursts within 1.3 of 19), the all-power-of-2 point within 0.3%; the VC1 rows it extrapolated were
  optimistic (VC1 ring 7 / forwarding 8: 207 predicted, 290 measured): the VC1 path has its own service bound, ~290 /
  293, which nothing earlier reached. Refit with job 112: 112 points, mean 0.6%, max 3.8%; weights 1:1 -> 18,8,16,6,5,
  3:2 -> 18,9,16,5,5, 2:1 -> 19,10,16,4,4 (all three measured), 1:2 -> 20,6,16,7,6.
- All 21 runs clean (7 x 50, 14 x 16 finishes); 19,9,16,5,4 repeats job 111 within 0.4.

| cycles/packet | main's router, main's table | main's router, 25,8,16,4,4 | v4gi, 25,8,16,4,4 | v4gi 18,8,16,6,5 | v4gi 18,9,16,5,5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1-hop E/W 16 B / 4352 B | 307.2 / 310.8 | 303.3 / 306.5 | 158.1 / 157.9 | 157.9 / 158.0 | 157.9 / 158.0 |
| 2-hop 16 B / 4352 B | 454.1 / 518.4 | 305.8 / 303.4 | 290.6 / 310.6 | 292.5 / 310.9 | 266.1 / 285.0 |
| multicast 4352 B | 277.0 | 200.5 | 169.6 | 169.7 | 156.4 |
| inter-mesh direct 16 B / 4352 B | 333.3 / 333.5 | 330.3 / 330.7 | 206.3 / 207.1 | 206.4 / 207.3 | 206.4 / 207.3 |
| inter-mesh + VC1 hops 16 B / 4352 B | 795-899 / 982-1,035 | 436-459 / 500-523 | 398-446 / 456-500 | 299-326 / 326-353 | 335-373 / 377-412 |
| mixed 16 B: VC0 worker / VC1 flow | 353 / 855 | 345 / 475 | 221 / 454 | 212 / 340 | 216 / 386 |

(Main's router as shipped: kernel v1, flags 0, firmware keepalive, the builder's -Os.)
- Main's table costs main's router as well (2-hop 454 -> 306, VC1 rows ~halved with 25,8,16,4,4), until its ~305
  per-packet pass caps it.
- The same VC0 with a faster VC1, only off powers of 2: 18,8,16,6,5 against 25,8,16,4,4, VC0 rows within 2, VC1 rows
  97-130 faster (24-29%): the power-of-2 split leaves 7 slots on a worker that needs 18.

**The same rule on VC0-only routers** (2026-10-28, offline; no device job). User: check the rule for VC0-only as well.
`alloc_model.py` now has three families (vc1; 2d and 1d from the slot allocation sweep, jobs 070-075, router v4us,
4352 B suites), the same model and the same allocation steps; window kinds per traffic where the loops differ
(bidirectional and scatter rows have longer worker and ring loops). Two changes found on the way: VC0 receivers are
taken at powers of 2 (the sweep's ~8 cycles/packet on E1 for other ring sizes; the note's earlier "receivers of any
size" was wrong), and the water-filling freezes rows at their service bound (1D's multicast sits at its bound and hid
the 3-hop row, which still gains).

| family | fit | rule | measured | today |
| --- | --- | --- | --- | --- |
| 2D (81 slots) | 60 points, mean 0.3%, max 2.3% | 20 / 15 / 16 | the sweep's best 2D split (mixes mean -3.33; 4352 B rows -0.40) | 17 / 16 / 16 |
| 1D (83 slots) | 35 points, mean 0.6%, max 7.5% | 19 / 32 / 32 | mean -5.92; best measured 23 / 28 / 32 -6.60 (within ~0.7) | 26 / 25 / 32 |
| VC1 (81 slots) | 112 points, mean 0.6%, max 3.8% | 18,8,16,6,5 (1:1) .. 19,10,16,4,4 (2:1) | all measured (jobs 110-112) | 51,4,8,2,4 |

(worker / forwarding / receiver.) Rings and tori: no window binds (acks), so the rule only fixes their receivers. The
2D and 1D round trips are v4us's; using the rule for today's router needs one calibration run. Note:
`consolidation/ALLOCATION_NOTE.md` section 4.

**The builder's VC1 default for the study's router** (2026-10-28, a09u32 job 113; `vce_runs.sh`). User: keep the
allocation offline, but make the experiments run with the right slots so the window problem is not chased again.
- `fabric_static_sized_channels_allocator.cpp`: with `TT_FABRIC_EXP_SLOTS_VC1` unset, the router in the v2 slot
  (`TT_FABRIC_KERNEL_VERSION=2`) on Blackhole VC1 routers of the measured shape (4 VC0 + 3 VC1 sender channels, one
  receiver each) gets `0,9,16,5,4` (worker 19 of 81), the split measured twice (jobs 111, 112) between the model's
  3:2 and 2:1 picks, every receiver a power of 2. Not computed in the builder: the numbers come from
  `consolidation/alloc_model.py` and move with the router's shape. Main's router keeps the tier table;
  `TT_FABRIC_EXP_TIER_SLOTS=1` keeps it for the study's router too; any other slot switch disables the default.
- Mock (CPU only): kernel v2 19,9,16,5,4; kernel v1 51,4,8,2,4; v2 + `TT_FABRIC_EXP_TIER_SLOTS=1` 51,4,8,2,4; v2 +
  explicit `TT_FABRIC_EXP_SLOTS_VC1` the explicit depths. VC0-only builds unchanged by construction (VC1 channel count).
- Device, v4gi -O3, no slot switch, depths logged: VC1 routers 19,9,16,5,4 and rows as job 112's 19,9,16,5,4 (21 rows,
  mean +0.02, within 0.5); 2D 17/16/16 and 1D 26/25/32 with rows as job 093's v4gi (22 rows, within 0.3). All clean.
- So VC1 numbers before this entry (jobs up to 108, and every `main` column of jobs 109-112) are at the tier table,
  window-bound: compare VC1 rows only at the same slots (`TT_FABRIC_EXP_LOG_SLOTS=1` logs them). 2D stays at
  17 / 16 / 16 (the user's earlier call; the rule's 20 / 15 / 16 trades 3-hop for 1-hop rows), 1D at 26 / 25 / 32.

**Merging main (2026-10-28): the compiler update, offline.** User: pull main (969 commits since the merge base
453ee7b, 2026-10-09: test-infra ntype options, firmware, NoC API, main's router, sfpi 7.86.0 -> 7.91.0), but check
the compiler first: router code generation has been sensitive. sfpi 7.91.0 fetched outside the tree
(`~/.sfpi-dl/7.91.0`, sha256 as main's `tt_metal/sfpi-version`; GCC 15.1.0 in both); `offline_build.py` gained
`OFFLINE_SFPI=<root>` (rewrites the logged `.../runtime/sfpi` prefix). v4gi rebuilt for every logged Blackhole config
(2D, 1D, ring, torus, VC1; the ELFs carry the 7.91.0 compiler ID):
- -O3 (our runs, and the builder's default for VC0-only routers): all 163 configs byte-identical in every loadable
  section; a 7.86.0 control rebuild reproduces the existing builds exactly.
- -Os (the builder's default on VC1 routers): all 119 identical.
- -O2: 76 of 119 differ (2D 56, 1D 20; VC1 identical): E1's receiver loop 23-24 instructions shorter, E0's 2D loop 8
  longer, stack loads and stores within 1, code 1.1 KB smaller (2D).
- So the compiler update does not touch our router at the levels we run. What a device run still rebuilds with it
  (firmware, main's router, the test's worker kernels, `runtime/hw/lib` objects) changes with the merge's sources
  too; the device before / after (`merge_ab_runs.sh`, labels `mpre_*` / `mpost_*`) covers both, and a toolchain-only
  A/B follows only if a row moves. "Before" runs: jobs 114-115 (all clean).

**Merging main: before / after on the device** (2026-10-28, a09u32 jobs 114-118; `merge_ab_runs.sh`, `mab_runs.sh`,
`toolchain_ab.py`).
- The merge: checkpoint 8 (`22f35f8`), then `origin/main` (6d8a103). Conflicts in 7 files:
  - test infra: kept main's ntype options (`NocSendTypeSpec`); our stream-aware filters compare against it, stream
    segments parse their ntype as main's patterns do, our per-pattern / per-segment `flush` overrides main's new ntype
    option `flush` (a segment `flush` on a scatter write is now an error pointing to the ntype option);
  - main's router and speedy path: main's link-transport layer (`SenderLinkTransport` / `ReceiverLinkTransport`, a
    thin wrapper over the same TX-queue calls) adopted; the study's flag-gated hooks moved onto it;
  - flow control (our credit methods kept, on main's transport), packet transmission (our flushed scatter path).
  - Submodules (UMD, tracy, cluster descriptors) updated; host rebuilt with sfpi 7.91.0.
- `v4gi_m` = v4gi with main's changes three-way merged into the 7 tree files it overlays (no conflicts; main moved the
  packet header to `hostdev/`, so the v4 kernel's include changed by hand). v4gi stays the measured pre-merge record.
  VC1 mock: 200 router builds, no errors.
- v4gi before (v4gi, sfpi 7.86.0) / after (v4gi_m, 7.91.0): validation clean on both sides (mesh 96, 1D 143, ring 19,
  Y-torus 12); every suite within +-0.05 cycles/packet on the mean, no row over 0.4 (4352 10, mix 30, flushmix 16,
  flushmix4 4, burst 44, 1D 12, ring 14, Y-torus 12 rows); latency medians -17 to +16 ns on 1,300-4,300.
- Main's router as shipped (v1, flags 0, -O3): 2D 4352 mean +0.68 (+0.1 to +1.2), 1D +0.39 (NeighborExchange scatter
  4352 B +6.0, the rest -0.9 to +1.2); mesh sanity 96/96.
- The VC1 split changed placement: [0, 0]'s inter-mesh link lands on [1, 0] now, [1, 1] before ([0, 0] -> [1, 0]
  320 -> 209, [0, 0] -> [1, 1] 206 -> 358; the 206-209 direct-link rate moved with it). Main has placement fixes in
  the range (b67f852, 0985b0e, cdeaeee). Rows on the VC1 split are not comparable one to one: v4gi's one-VC1-hop rows
  run 356-358 on the new path (320-326 before), two hops 385 (369); intra-mesh 1-hop and bursts unchanged (158),
  2-hop -14. Main's router (-Os): 1-hop +9 to +18, 2-hop -17 to -27, multicast -3 / +14.
- Compiler or sources: the merged tree does not build with sfpi 7.86.0 (main's LLK uses `__builtin_rvtt_ttinsn`), so
  no device A/B (job 118 stopped there). Offline instead (`toolchain_ab.py`: replays a run's logged router compile and
  link commands with both toolchains; the replays reproduce the runs' own ELFs, 146 of 146 checked): main's router is
  byte-identical under 7.86.0 and 7.91.0 in every config of the after runs (2D 72, 1D 200 at -O3; VC1 74 at -Os, 6
  lines cut by the other MPI rank). So main's router's moves are its sources and the placement, not the compiler.
  v4gi_m (with the variant swapped in again): byte-identical in all 288 configs of the after runs (2D and 1D 144, ring
  72, Y-torus 72; -O3), replays matching the runs' ELFs 288 of 288. So the merge does not touch our router's code
  under the compiler change, and its rows did not move.

**Single-ERISC baseline: main's router** (2026-10-28, after the merge; a09u32 jobs 119-120; `se_runs.sh`, table
`se_table.py`). Main's router as shipped (kernel v1, flags 0, firmware keepalive; -O3 VC0, -Os VC1) in three modes:
`two` (default), `fab1` (`TT_METAL_DISABLE_FABRIC_TWO_ERISC=1`: fabric single-ERISC, system two-ERISC) and `aer1`
(`TT_METAL_DISABLE_MULTI_AERISC=1`: the whole system single-ERISC, CI's single-ERISC fabric leg).
- Placement as the survey said: `two` builds both ERISCs, `fab1` only ERISC 1 (`subordinate_active_erisc`), `aer1`
  only ERISC 0 (`active_erisc`, shared with the base firmware). Slot depths the same in every mode (2D 17 / 16 / 16,
  1D 16 / 16 / 32, VC1 51,4,8,2,4).
- All clean in all three: mesh sanity 96/96, 1D sanity (test_fabric_sanity_at_least_2x2_mesh.yaml) 185/185, VC1
  split 34/34 finishes, CI's leg (`fabric_unit_tests --gtest_filter="Fabric1D*Fixture.*"`) 24 passed, 7 skipped (the
  same 7 in every mode: route-setting / TG-gateway tests and the 5-8 chip mux variants). 2D and VC1 single-ERISC have
  no CI coverage; they run here.
- Cost of one ERISC (cycles/packet over `two`; `fab1` / `aer1`):

| suite (rows) | `two` | `fab1` | `aer1` |
| --- | --- | ---: | ---: |
| 2D 4352 B (10): mean | 1-hop write 222.6 | +113.9 | +115.7 |
| 1-hop write / scatter | 222.6 / 221.8 | +39.8 / +95.6 | +39.8 / +95.6 |
| 1-hop bidirectional write / scatter | 222.8 / 220.7 | +147.4 / +204.4 | +147.1 / +204.2 |
| 3-hop write; multicast write | 221.9; 182.0 | +77.4; +161.5 | +86.4; +165.8 |
| 1D 4352 B (16): mean | 1-hop write 187.6 | +117.3 | +121.6 |
| 1-hop write; bidirectional; 3-hop | 187.6; 168.6; 187.6 | +10.9; +133.4; +69.9 | +14.7; +138.1; +71.7 |
| VC1 split (6): mean | 1-hop 318-322 | +151.2 | +155.6 |

- So one ERISC costs main's router 11-40 on one-way 1-hop writes and 130-230 wherever both directions or forwarding
  load the same ERISC (bidirectional, multicast, 3-hop); `aer1` (sharing ERISC 0 with the base firmware) is 0-22
  slower than `fab1`. Main's two-ERISC 2D rows (~222) sit at the firmware keepalive's one-way cap (§95: the study's
  router runs the same rows at ~126), so these deltas are against a capped reference, not main's CPU limit.

**Single-ERISC v4: v4se** (2026-10-28; a09u32 jobs 121-122; design `consolidation/V4_SINGLE_ERISC.md`, variant
`fabric_exp/variants/v4se`, runs `merge_ab_runs.sh` / `se4_runs.sh`, table `se4_table.py`). v4gi_m with each role's
loop body as one pass (`sender::pass`, `receiver::pass`); `router_single.hpp` runs both on one ERISC, receiver first,
each out of line over its role's state record. One ERISC keeps no NoC command buffer programmed between commands
(`command_buffer_presets = NUM_ACTIVE_ERISCS > 1`: local writes, forwards and the worker credit issue whole; the credit
is an inline L1 write from a scratch word), runs teardown itself, and uses counter credits on its one TX queue (host:
`fabric_builder_context.cpp` gives the study router counters in every mode). Goal per the user: functional on
Blackhole, performant in two-ERISC mode; tuning on Wormhole later.
- Two-ERISC gate (job 121, same job, v4gi_m control `s1ctl_v4gi` vs v4se `s1two_v4gi`): every suite within +-0.4
  (4352, mix, flushmix, flushmix4, burst, 1d_m1, ring, Y-torus), latency within -6 / +3 ns; validation clean.
- Single-ERISC (job 122, labels `s1aer1_v4se_*`, `s1fab1_v4se_*`): placement as intended (`aer1`: 360 router builds
  for `active_erisc`, `fab1`: 360 for `subordinate_active_erisc`); mesh sanity 96/96, val1d_m1 143/143, ring 19/19,
  Y-torus 12/12 in both; `fabric_unit_tests --gtest_filter="Fabric1DFixture.*"` with the study router (36 v4 builds
  per mode, on the mode's ERISC) 10 passed, 3 skipped (the route-setting / TG-gateway tests main skips too; the
  standalone mux fixture ran in job 125, below). No hangs.
- Throughput (cycles/packet; `aer1` and `fab1` agree within 1 on every row):

| suite (rows) | v4 `two` | v4 one ERISC | v4 cost | main cost | v4 - main, one ERISC |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2D 4352 B (10): mean | 145 | 277 | +132 | +115 | -56 (-17 to -92) |
| 1-hop write / scatter | 128 / 142 | 207 / 249 | +79 / +107 | +40 / +96 | -55 / -68 |
| 1-hop bidirectional write / scatter | 135 / 144 | 291 / 333 | +156 / +189 | +147 / +204 | -79 / -92 |
| 3-hop write; multicast write | 152; 146 | 262; 327 | +110; +181 | +77..+86; +162..+166 | -37..-46; -17..-21 |
| 1D 4352 B (12, no ring): mean | 147 | 281 | +134 | +105..+109 | -22..-25 (-13 to -40) |
| 1-hop write; bidirectional; 3-hop | 124; 125; 126 | 185; 273; 244 | +61; +148; +118 | +11..+15; +133..+138; +70..+71 | -14..-17; -29..-34; -14..-15 |

- So v4 on one ERISC is faster than main on one ERISC on every row, and its cost over its own two-ERISC mode is
  larger than main's because its two-ERISC reference is not capped by the keepalive (main's `two` rows ~222 are).
- VC1 (job 123, `se4vc1_runs.sh two aer1 fab1`, labels `s1<mode>vc1[i]_v4se_default_p1_o3`, hang capture armed): the
  VC1 split (vc1_split.yaml, main's baseline rows) 34/34 finishes and the inter-mesh rows (vc1_inter.yaml) 16/16 in
  every mode, no hangs; placement as intended (`two` 240 + 240, `aer1` 240 `active_erisc`, `fab1` 240
  `subordinate_active_erisc`); slots the study default in every mode (VC0 19,9,9,9 / 16, VC1 5,5,5 / 4).

| VC1 rows (cycles/packet) | v4 `two` | v4 one ERISC | v4 cost | main cost | v4 - main, one ERISC |
| --- | ---: | ---: | ---: | ---: | ---: |
| VC1 split (6): mean | 156 | 327 | +170 | +151..+156 | -132..-137 (-116 to -146) |
| 1-hop EW / NS | 158 / 158 | 278 / 283 | +120 / +125 | +72..+81 / +102..+107 | -116..-125 / -140..-142 |
| multicast (two senders) | 156 / 148 | 419 | +263 / +271 | +269..+290 | -139..-146 |
| inter-mesh (8, v4 only): mean | 209-419 | 362-518 | +106 (+66 to +153) | - | - |

- So VC1 on one ERISC is functional in both modes and faster than main's on every row (main's two-ERISC VC1 rows
  318-322 run at the keepalive's cap and its VC1 build is -Os). `aer1` and `fab1` agree within 1 on every row (VC0
  and VC1): sharing ERISC 0 with the base firmware costs v4 nothing measurable. Single-ERISC on Blackhole: done as
  scoped (functional; Wormhole tuning later).

**Standalone mux on v4** (2026-10-28; a09u32 job 125, `mux4_runs.sh two aer1 fab1`; job 124 ran nothing: the script
lacked the results directory). Two different muxes: the standalone mux (`tt_fabric_mux`, `FabricMuxConfig`) is a
kernel ops launch on their own Tensix cores, a worker to the router, on a plain fabric (two-ERISC on Blackhole); the
fabric's Tensix mux mode (`FabricTensixConfig::MUX`) is launched by the fabric itself, one mux core per router, changes
the router (senders: only the worker channel, fed by the mux, `erisc_datamover_builder.cpp`; receivers forward into
the downstream direction's mux, `downstream_is_tensix_builder` in `compute_mesh_router_builder.cpp`) and forces one
ERISC (`is_fabric_two_erisc_enabled`: any Tensix extension, "out of stack size"); UDM (`FabricTensixConfig::UDM`)
likewise. `Fabric1DMuxFixture` is `Fabric1DFixture` (`test_basic_fabric_mux.cpp:48`): the standalone mux.
- `fabric_unit_tests --gtest_filter="Fabric1DMuxFixture.*"`, fresh caches: main's router (kernel v1, default mode,
  same-job control) 10 passed, 4 skipped; v4se in `two`, `aer1` and `fab1` 10 passed, 4 skipped each (the same four
  in every run: the 5-8 chip variants). Builds per run: 36 router configs on the mode's ERISCs (`two` both), 8
  `tt_fabric_mux`, 21 + 21 mux client kernels. So v4 carries the standalone mux as is, in every mode; the fabric's
  Tensix mux mode and UDM are the router features still to build.

**Tensix mux mode on v4: v4tm** (2026-10-28; design `consolidation/V4_TENSIX_MUX.md`; variant `variants/v4tm` = v4se
plus: the receiver credits the far worker channel without reading the header's stale `src_ch_id`
(`source_channel`), the receiver waits for the downstream muxes' sync count before opening its downstreams
(`wait_for_downstream_muxes`), the gate admits mux mode on one ERISC, VC1 with mux mode refused at build with its own
message; user: VC1 with mux mode out of scope). Main does not reject VC1 with mux mode: VC1 is wired router to router
around the muxes, but the MUX slot branch gives VC1's sender channels no slots.
- Offline (`coverage.py`, merged-tree mocks): outside mux mode v4tm loads the same image as v4se in every config,
  164 two-ERISC (1D, ring, 2D, torus, VC1) and 38 single-ERISC (`objcopy -O binary`; only debug sections differ). Mux
  mode (new mock cases `bh_1d_mux`, `bh_2d_mux`, `bh_ring_mux`, run with v4tm swapped in, as the tree's v2 slot does not
  build after the merge): every logged router config compiles (36 / 40 / 36), 48 distinct configs at -O3; router on
  ERISC 1, only the worker channel serviced.
- Device (a09u32 job 126, `tm_runs.sh`, main's router first in the same job): `fabric_exp/v4/val_mux.yaml` (the mux
  cases of `test_fabric_sanity_at_least_2x2_mesh.yaml`: linear, full-ring and mesh multicast over three send types, the
  ring deadlock check) 19/19 on both; `fabric_unit_tests` on `Fabric1DTensixFixture`, `NightlyFabric1DTensixFixture`,
  `NightlyFabric2DTensixFixture` 6 passed on both (v4: 108 router builds, all ERISC 1; 26 mux kernels); the
  standalone mux v2 smoke fixture (`*FabricMuxV2Smoke*Fixture.*`, merge gate) 16 passed on v4 (two-ERISC). No hangs.
- The one benchmark row (ring deadlock check, 1 KB, one link): main 419.6 cycles/packet (419.6 / 422.0 / 419.6 in
  jobs 119-120 and 126), v4tm 440.9 (+21, +5 %; one sample). The mux bounds this row (one packet and a flush at a time),
  so the gap is in how the router serves it (credit return to the mux, forwards into the next mux), not the router's
  own rate; not examined yet.

**Mux-mode gap: it is single-ERISC rings, not the mux** (2026-10-28; a09u32 jobs 127-128). Same job, A/B/A/B, main's
control repeating to 0.4 (127) / 0.2 on the steady rows (128):
- Mux mode (`exp_mux.yaml`, `mxb_runs.sh`: 1D, ring and 2D 1-hop / bidirectional / 3-hop / multicast and the full
  ring, 1024 and 4352 B): every unicast row equal to 0.5 (the mux sets them: 1D 313-332, ring 341-359, 2D 445-468);
  linear multicast v4tm -18 / -19; ring multicast +6 / +8; full-ring multicast +21 / +22; identical in both passes,
  independent of size.
- One ERISC without the mux (`exp_ring_se.yaml`, `rse_runs.sh`, `TT_METAL_DISABLE_FABRIC_TWO_ERISC=1`, main vs v4se):
  one-way rows v4 faster (1-hop -4, 3-hop -2, multicast -9 / -12, unidirectional multicast -7 / -8); 1-hop
  bidirectional +19 (both passes); half-ring and full-ring multicast +9 to +34 (main itself moves up to 20 between
  passes there). Single-ERISC ring rows cost the same at 1024 and 4352 B (one-way ~256-260): per-packet bound.
- Against job 122 (one ERISC, no rings): v4 bidirectional rows were 29-34 (1D) and 79 (2D) faster than main. So v4
  loses only where a router both sends data and returns first-level acks on its one TX queue: rings (first-level
  acks), with traffic both ways or forwarded. Hypothesis (not yet confirmed): under first-level acks v4's receiver
  sends a credit frame (an L1 block, completions and acks) most passes, one slot per completion group, on the TX queue
  its sender's data uses; main's single-ERISC router credits with stream-register writes. Mux mode on rings is this
  case on every router.

**Stream-register credits on one ERISC: v4sc** (2026-10-28; a09u32 jobs 129-130). On one TX queue main's credit plan
uses stream registers (the receiver adds each credit to the far sender channel's packets-completed / packets-acked
register with one remote write; the sender reads the register and lowers it by what it takes); v4 had only counter
credits, and a study switch (`fabric_builder_context.cpp`) gave it counters on one TX queue too (V4_SINGLE_ERISC.md
§4.2). User: support main's mechanism there. `variants/v4sc` = v4tm plus the second path, chosen at compile time by
the host's plan per VC: `StreamCredits` (owed counts per far sender channel, sent once per pass when the TX queue is
free, a channel's acks before its completions; a further write waits for the next pass), the sender's credit reads
through `read_*_report` / `*_reported` (absolute counts as before; with registers, taken plus the register, which it
lowers). Host: `TT_FABRIC_EXP_STREAM_CREDITS=1` gives the study router main's plan; default unchanged (counters).
- Offline: with counter credits v4sc loads the same images as v4tm in all 250 mock configs (164 two-ERISC, 38
  single-ERISC, 48 mux); with stream credits all 130 logged single-ERISC configs build (1D, 2D, ring, torus, mux 1D /
  2D / ring, VC1 split; new mock cases `*_sc`). The 1D single-ERISC mock with stream credits crashes in host dispatch
  at launch (`add_prefetch_set_ringbuffer_offset`); main's router under its own plan crashes there too: a mock
  artifact, after the compiles.
- Functional (job 129, one ERISC, stream credits): mesh 96/96, val1d_m1 143/143, ring 19/19, Y-torus 12/12, val_mux
  19/19, VC1 split 34/34, Fabric1DFixture + Tensix fixtures 16 passed, 3 skipped. No hangs.
- Throughput (job 130, `scp_runs.sh`, one ERISC, order main, counters, stream; pass b repeats rings and mux to 0.4;
  main's half/full-ring rows move up to 24 between passes):

| suite (rows) | v4 counters - main | v4 stream - main | stream - counters |
| --- | ---: | ---: | ---: |
| 2D 4352 B (10) | -54.6 | -47.2 | +7.4 |
| 1D (12) | -21.7 | -17.7 | +4.0 |
| VC1 split (6) | -132.2 | -121.3 | +11.0 |
| mux mode (22) | +0.9 | +4.7 | +3.8 |
| rings (16) | +5.9 | +63.0 | +57.1 (1-hop +12, bidirectional +25, multicasts +13 to +61, full ring +258) |

- So the hypothesis above is refuted: counter frames are not what costs v4 on rings; v4 with main's mechanism is slower
  in every suite. Even without first-level acks (2D, 1D, VC1: completions only, one register write per group, never
  deferred) registers cost 4 to 11: v4's sender reads its credits every visit, and a stream-register read and decrement
  cost more than an L1 load. On rings the second write of a pass (the completion after the ack) waits a pass as well.
  Counter credits stay v4's one-TX-queue default; the ring gap to main (bidirectional +19, half/full ring) has another
  cause, not yet found.

**Where the single-ERISC ring cycles go: PC sampling** (2026-10-28; a09u32 job 131, `pcse_runs.sh`; tools
`pcs_report.py` (now maps main's router builds too), `pcs_lines.py` (one router's samples by source line and inline
chain, through addr2line on the run cache's own -g ELF; `--dump` writes the chains)). The ring 1-hop bidirectional row
(1024 B, 40M packets, ~11 s, `fabric_exp/pcse/bidir.yaml`), one ERISC, every chip's ERISC 1 sampled (78k samples/s,
0 read errors; rates under sampling main 364.1, v4tm 383.1 cycles/packet, as without). The two routers carrying the
row (chip 0 channel 6, chip 1 channel 5) differ from the 38 idle ones (v4: sender pass 38%, receiver pass 50%, loop
3.5%, against 53 / 33 / 13% idle). Both routers are CPU-bound: no single stall; the time spreads over per-packet work.
Per packet, by activity (sample share x rate; categories matched by function name, so approximate; main inlines more
into its step functions, so its rows below are partly lumped into "header, route, retire"):

| cycles/packet (two links averaged) | main | v4 | v4 - main |
| --- | ---: | ---: | ---: |
| sender: worker credit (the spoofed inline write) | 33.4 | 30.2 | -3.2 |
| sender: data send | 53.9 | 74.4 | +20.4 |
| sender: polling, credits in, connection check | 98.6 | 74.1 | -24.6 |
| receiver: first-level acks | 10.1 | 41.3 | +31.2 |
| receiver: credit frame out | - | 13.0 | +13.0 |
| receiver: completions | 21.9 | 34.8 | +12.9 |
| receiver: local delivery | 64.4 | 34.5 | -29.9 |
| receiver: header, route, retire | 78.4 | 66.3 | -12.1 |
| loop | 3.5 | 14.8 | +11.3 |

- Correction (same job, by source line): main's first-level-ack credit code is inlined into its receiver step, so the
  name-matched rows above undercount it. By line (main's ack block, `fabric_erisc_router.cpp` 2284-2318, and its
  completion paths, 2535-2670): acks ~47, completions ~42 cycles/packet, ~89 in all, the same as v4's ~89 (acks 41,
  frame 13, completions 35). Their composition differs:

| credit bookkeeping, cycles/packet | main | v4 |
| --- | ---: | ---: |
| transport: TX-queue busy check and one remote register write per credit (main: 2292, 2313, 2652, 2663); one frame per pass (v4: 270) | ~51 | ~13 |
| attribution and counting: the header's source channel and the counts (main: 2308, 2315, 2666; v4: the ack loop 158-165, the completion's source lookup and credit, 188) | ~20 | ~50 |
| proof: the slot's writes have left (main 2645; v4 183) | ~11 | ~12 |

- So v4's credits are not heavier than main's: v4 traded main's per-credit transport for per-packet counting and
  attribution, at an equal total. The +19 on this row is outside the credits: v4's single-ERISC pass structure (loop
  ~15 against main's ~4: two out-of-line passes over state in memory, V4_SINGLE_ERISC.md) and per-pass receiver reads,
  partly offset by local delivery and sender polling (the categories there do not align cleanly between the two).
- A candidate for the Wormhole tuning (not built): credits by ring position. The far sender already knows which of
  its channels each far-ring slot came from (it sent them in ring order); acks and completions could go back as one
  in-order count per VC, attributed to channels at the sender from its own per-slot record, removing the receiver's
  per-packet header reads, per-channel counters and per-slot source lookups: the attribution half of v4's cost (~45-50
  of its ~89, estimated from the lines above), where v4's frame already did the transport half. On main it would
  remove only ~15-20 (its cost is mostly per-credit transport).

**Credits by ring position, first build: v4rp** (2026-10-28; a09u32 job 132, `rp_runs.sh`; design
`consolidation/V4_RING_CREDITS.md`; `variants/v4rp` = v4tm + ring-position credits on routers whose only VC is VC0 with
first-level acks). Receiver: acks are `acked += arrived` (no header reads), completions `completed += n` (no source
record or lookup), the frame is `[acked, completed]`, 16 bytes. Sender: a per-far-slot channel record written at the send
(no `src_ch_id` stamp), the two counts taken once per pass at the pass's start (`take_ring_credits`: far slots freed by
the completion difference, newly acked slots credited to their channels' producers in runs).
- Offline: every non-ring config loads v4tm's image (two-ERISC 1D / 2D / VC1, one-ERISC 1D / 2D, mux 1D / 2D; the
  single-ERISC state's record member is `[[no_unique_address]]`, so empty elsewhere); ring and torus configs build in
  both ERISC modes and with the mux. Images on rings: receiver -144 B, sender +224 B; tori -96 / -490 B. Loops (ring
  configs, instructions / stack loads / stack stores): receiver::run 517/19/33 -> 481/18/20, sender::run 401/36/46 ->
  457/50/69.
- Validation, data-checked: two ERISCs ring 19/19, Y-torus 12/12, mesh 96/96, val1d_m1 143/143, val_mux 19/19; one
  ERISC ring 19/19, Y-torus 12/12. No hangs.
- Throughput against v4tm, same job, two passes (each arm repeats within 0.2), cycles/packet:

| row | v4tm | v4rp | |
| --- | ---: | ---: | ---: |
| two ERISCs, ring 1-hop write / scatter (4352 B) | 150.7 / 189.5 | 193.1 / 194.1 | +42.4 / +4.6 |
| ring 3-hop write; ring multicast write | 173.5; 156.8 | 194.1; 177.5 | +20.6; +20.7 |
| ring unidirectional multicast write / scatter | 224.4 / 274.4 | 205.0 / 235.7 | -19.4 / -38.7 |
| ring half-ring multicast write / scatter | 181.9 / 234.5 | 182.3 / 195.3 | +0.4 / -39.2 |
| Y-torus N/S 1-hop write / scatter (E/W rows unchanged) | 158.7 / 200.7 | 199.2 / 199.4 | +40.5 / -1.3 |
| one ERISC, ring 1-hop (1 KB / 4352 B) | 255.6 / 255.9 | 269.1 / 272.4 | +13.5 / +16.5 |
| one ERISC, ring 3-hop; half ring; full ring (1 KB) | 314.8; 467.4; 500.6 | 282.5; 448.4; 481.5 | -32.4; -19.0; -19.1 |
| one ERISC, ring multicast / bidirectional 1-hop (1 KB) | 363.3 / 383.3 | 378.5 / 392.1 | +15.2 / +8.9 |
| mux mode, full ring / ring multicast (1 KB); 1-hop | 440.9 / 332.5; 340.6 | 415.9 / 325.0; 340.8 | -25.0 / -7.5; +0.2 |

- So the receiver side delivers where the receiver is the bound (forwarding and multicast rows -8 to -39; mux full ring
  -25), but the sender side as built costs more than it saves: its loop grows and spills (the far-send record carried
  per pass, and a credit read of its own at the pass's start), and rows where both ERISCs ran near ~150 (light one-way
  writes) move to the sender's new bound, ~194 (write rows now equal the scatter rows). Next: the sender's part reworked
  (the counts read in the worker visit's existing load group; the attribution, rare per pass, out of line; the record
  kept out of the loop's state), then the same A/B.
- Attribution (a09u32 job 133, `pcsr_runs.sh`: PC sampling of both ERISCs on the two-ERISC ring 1-hop write row, 60M
  packets of 4352 B, v4tm 150.6 / v4rp 193.1 cycles/packet under sampling; the row runs from chip 1 ERISC 0 to chip 0
  channel 6 ERISC 1). The sender is the bound in both builds (its samples are per-packet work, not polling; WRONG, see
  the v4rs entry: polling samples look like work, and the receiver is the bound in v4tm); the
  receiver's real work fell as designed (acks 34.9 -> 12.7) and its polling rose with the slower rate. Sender, cycles
  per packet: the new credit step `take_ring_credits` +51.5 (the two credit words read in their own group and used at
  once, a load-use stall, with the far-slot update ~14; the attribution walk ~12, run although only the worker sends,
  because the router has a forwarding channel; the early-exit test of three record counters ~7; checks and the call
  into the worker credit ~9; the forwarding credit's setup hoisted by the compiler onto the common path, with a stack
  load and store, ~3: code motion, not credits taken); `send` +10 (the record write, and more stack traffic around the
  TX queue writes); the worker visit -15 and the free-slots reads -4 (their credit reads moved). Tool: `pcs_lines.py
  --dump` now records each sample's ELF address.

**Ring-position credits, sender reworked: v4rq** (2026-10-29; a09u32 jobs 134-135, `rq_runs.sh`, `pcsq_runs.sh`;
`variants/v4rq` = v4rp with the sender reworked per job 133: the two credit words read in the worker visit's load group
with the connection semaphore, no early-exit test, sends recorded as runs of one channel (`SendRuns`: a send from the
newest run's channel adds one, any other starts a run; acks taken from the oldest runs), the record in the forwarding
table, empty elsewhere). Offline: non-ring configs load v4tm's images; sender::run 405/41/50 (v4tm 401/36/46, v4rp
457/50/69). Validation as job 132, all passed. Same job against v4tm, two passes (each arm within 0.2):

| row, cycles/packet | v4tm | v4rq | v4rq - v4tm | v4rp - v4tm (job 132) |
| --- | ---: | ---: | ---: | ---: |
| two ERISCs: ring 1-hop write / scatter | 150.7 / 189.5 | 181.9 / 182.7 | +31.2 / -6.8 | +42.4 / +4.6 |
| ring 3-hop write; multicast write | 173.5; 156.7 | 182.5; 176.1 | +9.0; +19.4 | +20.6; +20.7 |
| unidirectional multicast write / scatter | 224.4 / 274.4 | 197.7 / 236.2 | -26.8 / -38.2 | -19.4 / -38.7 |
| half-ring multicast write / scatter | 181.9 / 234.7 | 172.6 / 196.3 | -9.3 / -38.4 | +0.4 / -39.2 |
| Y-torus N/S 1-hop write / scatter | 158.7 / 200.8 | 195.4 / 196.9 | +36.7 / -3.9 | +40.5 / -1.3 |
| one ERISC: ring 1-hop 1 KB / 4352 B; bidirectional | 255.6 / 255.9; 383.3 | 263.1 / 256.9; 384.1 | +7.5 / +1.0; +0.8 | +13.5 / +16.5; +8.9 |
| ring 3-hop; multicast (1 KB) | 314.8; 363.2 | 291.7; 377.4 | -23.1; +14.2 | -32.4; +15.2 |
| unidirectional / half / full ring (1 KB) | 490.2 / 467.3 / 500.6 | 482.1 / 444.2 / 474.5 | -8.1 / -23.1 / -26.1 | +1.1 / -19.0 / -19.1 |
| mux mode: full ring / ring multicast (1 KB); 1-hop | 440.9 / 332.4; 340.6 | 401.8 / 320.0; 340.6 | -39.1 / -12.5; 0 | -25.0 / -7.5; +0.2 |

- Attribution (job 135, the two-ERISC ring 1-hop write row, v4rq 181.9 under sampling): the sender, the bound in both
  builds, +31: the credit step +33 (the credit word from L1 used on the next instruction, a load stall the group cannot
  hide with nothing to overlap; the ring's taken counts on the stack, loaded and stored each visit; the runs' byte
  arrays read in the attribution), the run update in `send` ~15 (two dependent byte loads from the stack, the newest
  run's count and channel, and a branch on each, then the stores; the header stamp it replaces was one store), the
  worker visit -11.5, the slot checks -5. The receiver's work as in job 133 (acks -23).
- So the receiver half pays where the receiver is the bound (forwarding and multicast rows, mux mode, single-ERISC
  multi-flow: -9 to -39), but on light two-ERISC rows the sender is the bound already (v4tm 150.6 is its own
  per-packet work; WRONG, see the v4rs entry: v4rq's sender became slower than the receiver), so moving work to it can at best tie, and any of it shows (+31 to +37 on 1-hop writes).
- Why the rework missed its estimates (credit handling expected ~11 cycles/packet, measured ~41; v4rp ~55): the
  estimates removed the cost of the code a profile line pointed at, but on this in-order core a stalled load shows at
  its first use, so changing the code moves the wait to the next dependent instruction rather than removing it. (1) The
  credit words grouped with the semaphore load hide nothing when their first use is the next instruction (§17's rule:
  grouping helps only where independent work follows). (2) Moving the record into memory to stop v4rp's spills turned
  the per-send update into a dependent chain of loads and stores (index, then the run's count and channel) where the
  header stamp was one store. (3) One-run attribution still loads the oldest run's index, count and channel. v4tm stays
  at 150 by keeping its per-packet state (the worker's counts, the far ring's free slots) in registers and reading
  credit words where the send's own work overlaps them; v4rp added register state (spills), v4rq moved it to memory
  (load stalls). The next step's target is therefore no new per-packet memory access and no added loop-carried state.

**Ring-position credits, sender state in registers: v4rr** (2026-10-29; a09u32 jobs 136-137, `rr_runs.sh`,
`pcsrr_runs.sh`, `pcsf_runs.sh`; `variants/v4rr` = v4rq with the sender's per-packet state in registers, replacing
state: the completion count taken is the worker's `completed`, the ack count taken and the newest run's channel sit with
the worker's loop locals (`RingState`, empty elsewhere); a send compares its channel with the newest run's and touches
memory only on a change (`start_run`, out of line, queueing the finished run's unacked packets); queued runs are taken
out of line (`take_queued_runs`); the worker's send reads the two credit words in its post-send load group, the visit
only on passes without a send). Offline: non-ring configs load v4tm's images; sender::run 438/49/53. Validation as
job 132, all passed. Same job against v4tm, two passes (within 0.5, one one-ERISC half-ring row 14):

| row, cycles/packet (v4rr - v4tm) | two ERISCs | one ERISC |
| --- | ---: | ---: |
| ring 1-hop write / scatter | -10.0 / -37.7 | -26.1 (1 KB) |
| ring 1-hop bidirectional | -11.6 / -38.0 | -34.3 |
| ring 3-hop write / scatter | -21.9 / -37.8 | -28.1 |
| Y-torus N/S 1-hop write / scatter (E/W rows unchanged) | -4.7 / -33.6 | |
| unidirectional ring multicast | -18.5 / -30.2 | +27.8 / +32.0 |
| ring multicast (all-to-all) | +37.9 / +26.1 | +32.9 / +36.6 |
| half-ring multicast | +24.0 / -13.9 | -0.1 / +12.6 |
| full-ring multicast | | +73.6 / +77.1 |
| mux mode: full ring / ring multicast; 1-hop | -38 / -12 to -15; 0 | |

- Single-flow rows (job 136's PC sampling, two-ERISC ring 1-hop write, 140.6; the conclusion of this item is WRONG, see
  the v4rs entry: the receiver is the bound, and the gain is the receiver's): the gain is the sender's, the bound:
  the worker visit 37.9 -> 4.7 (v4tm's visit reads the worker's ack counter in its load group and uses it at once; v4rr
  reads the credit words inside the send, whose own work hides their latency), less the new credit step and credit
  dispatch (+24). The receiver's ack saving (34.9 -> 11.1) does not count on this row.
- Mixed rows (job 137, one-ERISC full-ring multicast, router 0/6, v4tm 500.6 / v4rr 577.1): every router's sender
  interleaves its worker's packets with forwarded ones, so a run is one packet: the run bookkeeping +35
  (`take_queued_runs` 18.9, `record_send` 10.4, `start_run` 6.0) and the out-of-line calls each send and ack visit
  makes +35 more (pass and loop overhead, the registers saved around the calls). And the receiver's ack saving there
  (-23.8) reappears in `receive` (+23.7): v4tm's ack loop was the first touch of a newly arrived packet's header and
  paid its wait; without it the same wait lands on `receive`'s header read.
- So on these rows the protocol change is not what made single-flow rows faster: the placement of the credit read on
  the sender is, and it does not need ring position. Mixed traffic makes attribution per packet whichever end does it,
  and the receiver does it more cheaply (it reads the header anyway).

**Credit reads placed in the send: v4rs; and which ERISC bounds the light ring rows** (2026-10-29; a09u32 job 138,
`rs_runs.sh`; `variants/v4rs` = v4tm with, on first-level-ack VCs, each channel's ack counter read in the send's
post-send load group with its completion counter; the visit uses it when it sent and reads it only on passes without a
send; v4tm's protocol otherwise. Offline: only ring and torus sender images change; sender::run 410/38/47). Validation
as job 132, all passed. Same job, v4tm / v4rs / v4rr, two passes (each arm within 0.5):

| suite mean vs v4tm (range) | v4rs | v4rr |
| --- | ---: | ---: |
| two ERISCs, rings (14) | -0.37 (-6.0 to +0.4) | -12.80 (-38.0 to +38.0) |
| two ERISCs, Y-torus (12) | +0.01 | -6.49 (-34.0 to +0.1) |
| one ERISC, rings (16) | -3.58 (-7.2 to 0) | +4.18 (-34.4 to +76.3) |
| mux mode (22) | -0.61 (-4.9 to +0.2) | -4.64 (-38.1 to +0.2) |

- v4rs leaves every single-flow row unchanged (two-ERISC ring 1-hop write 150.7, scatter 189.5; torus N/S; one-ERISC
  1-hop and 3-hop) and gains only where the sender is busy: one-ERISC multicast rows -4 to -7, mux full ring -5,
  two-ERISC all-to-all ring multicast write -6.
- Correction (to the v4rp, v4rq and v4rr entries above): a router's loop polls without stopping, so both ERISCs'
  samples always add up to the row's rate, and a profile cannot say which ERISC bounds a row; idle passes are billed to
  whatever the loop polls (the worker visit, the slot checks). The A/Bs say it: a sender-only change (v4rs) does
  nothing on the light two-ERISC ring rows; a leaner receiver with a sender no slower (v4rr) gains -10 to -38; a leaner
  receiver with a slower sender (v4rq) loses +31. So those rows are receiver-bound in v4tm (as the v4u study had it:
  "rings are E1-bound"), v4rq's sender became slower than its receiver, and v4rr's single-flow gain is the receiver's
  (acks by count, no per-packet header read before the frame), with the sender kept cheap enough. Ring-position
  credits' receiver half is worth -10 to -38 where the receiver binds; the only problem is the sender's attribution
  under mixed traffic (per-packet runs, out-of-line calls).

**Where v4 stands; v4rs leads** (2026-10-29, from stored runs, no new job; per-suite means in cycles/packet, P150 400G;
main as shipped (firmware keepalive) unless marked; v3q = the v3 lead; v4 = v4tm, which on two-ERISC non-mux rows equals
v4gi_m / v4se within 0.4).
- Two ERISCs: 2D 4352 B main 217.2 / v3q 144.4 / v4 144.4; 1D 197.9 / 152.9 (24 worker slots) / 147.1; mix 137.0 (main
  on a08u24) / 127.0 / 128.7; bursts - / 522.5 (v3zy, a08u24) / 523.3; ring 239.6 / 230.5 / 188.9; Y-torus 181.9 (main
  at keepalive 1000) / 164.7 / 157.4; VC1 split 1-hop ~320 / 180-198 / 158. Latency against main: 1-hop within 30 ns,
  3-hop -124 to -203 ns.
- One ERISC against main: 2D 331.3 / 276.5, 1D 303.3 / 281.4, VC1 ~471 / 327; rings 375 / 378.7 (1-hop -4, bidir +19,
  half / full ring +10 to +34); mux mode 373.3 / 374.1 (full ring +21, ring multicast +7; 18 of 22 rows mux-bound,
  equal).
- Open gaps, by weight: (1) Galaxy full-ring multicast at 4352 B, the one production row below main: v4u 0.87x,
  v4uf's single-slot rule (v4gi's and so the lead's) 0.75x; the lead has not run on Galaxy. (2) One-ERISC multi-flow
  rings and mux-mode rings (above). (3) Two-ERISC ring and torus receiver headroom: v4rr showed -10 to -38 on
  single-flow rows. (4) Rows behind v3q: 3-hop write +9, multicast write +5, small writes +2 to +3 (forwarded +5),
  pure atomics +2 to +6, 16 B bursts +3 to +6, MixCmd +15 to +36 (deferred atomics off by decision).
- Decision (user): **v4rs is the lead** (`variants/v4rs`, v4tm plus the ack-counter read in the send). Ring-position
  credits (v4rp, v4rq, v4rr) are not adopted; a register bit-history version of them and a Galaxy run of the lead are
  open for the user's call.
- Method lesson from v4rp-v4rs (three device rounds spent on a wrong bound): which ERISC bounds a row, and which change
  made a gain, are decided by an isolating A/B (a change to one side only), never by PC-sample shares or by subtracting
  a profiled line's cost; until then they are hypotheses, and the next design waits for the A/B.
**Feature inventory after the merge, offline** (2026-10-29; user: close out the remaining features first, then
performance; gaps parked in `consolidation/V4_PERF_GAPS.md`). Post-merge coverage (`coverage.py`, `TT_COV_DIR=cov_m`,
`OFFLINE_BASE=~/.tt-metal-p1-base-m`; the pre-merge cases fail on every config). The feature cases were captured again
after the merge with main's router in the v2 slot (`variants/mainv2cap`, capture only; tree restored), plus a new case
`bh_neighbor` (NeighborExchange: every router has one VC0 sender channel, so the host selects the speedy path); UDM is
synthesized from the mux case as before. v4rs at -O3 builds 612 of 1,074 configs:
- Builds: BH 1D / 2D / ring / torus (both ERISC modes), mux mode, VC1 split, code profiling (56), channel-trimming
  capture (2D 56, 1D 20), Galaxy 1D / 2D / ring / torus and express (5 VC0 sender channels, Z routers; 56).
- Not brought up (the bring-up gate): dynamic telemetry (`TT_METAL_FABRIC_TELEMETRY`, 2D 56 / 1D 20), the perf
  telemetry recorder (`enable_fabric_bw_telemetry`, which test_tt_fabric turns on in its perf modes unless
  `TT_FABRIC_NO_BW_TELEMETRY`; 56), VC2 (56), UDM (28), the speedy path (NeighborExchange, 8), Wormhole (every config;
  the WH cases were not captured again). The `*_sc` cases fail by design (the parked stream-credit plan).
- Out of the list: the data-cache and TX-queue spin-wait flags (constants off in the builder).
- Built, never run on hardware with v4: code profiling, trimming capture and replay (replay configs come only from a
  device run; on this host main's replay works only for 1-hop traffic), express / Z (Galaxy only).
- `bh_vc1_allinstr` logged 24 configs (the two MPI ranks' lines cut each other; partial).

**UDM mode on v4: v4udm** (2026-10-29; a09u32 jobs 139-140, `udm_runs.sh`; design `consolidation/V4_UDM.md`;
`variants/v4udm` = v4rs + UDM). User order for the features: UDM, VC2, speedy path, telemetry. UDM
(`FabricTensixConfig::UDM`) puts a mux and a relay on a Tensix core per router direction; the router's only change is
local delivery: a packet for this chip goes whole into the relay's ring, which does its NoC work. In v4 the relay is one
more VC0 downstream (`router_downstream.hpp`: cursor and link as members that exist only in UDM; admission needs a relay
slot when the packet delivers here; the forward is the shared stateless forward; opened after the downstream routers).
Its teardown word is the relay connection's buffer-index word on this router (main passes none, so its open writes 0
to L1 address 0).
- Real UDM configs captured for the first time (mock run of `Fabric2DUDMModeFixture`, case `bh_2d_udm_gtest`): one
  ERISC, mux mode off, one local Tensix connection, 4 relay slots, relay credit on stream 30, no deadlock avoidance, 4
  VC0 sender channels. The earlier synthesized case (`bh_2d_udm`, from the mux case) kept mux mode and was not one.
- Offline: the 28 UDM configs build; the other 612 post-merge configs are byte-identical to v4rs's.
- Device, fresh caches: main's router (kernel v1, flags 0, firmware keepalive) `Fabric2DUDMModeFixture.*` 8/8,
  `NightlyFabric2DUDMModeFixture.*` (2x4) 68/68; v4udm 8/8 and 68/68 (36 router builds, every one with the relay
  path). No hangs. The ttnn UDM gtests (`unit_tests_ttnn_udm`) are not configured in this build; not run.
- Lead: **v4udm**.

**VC2 on v4: v4vc2, first device runs** (2026-10-29; a09u32 jobs 141-142, `vc2_runs.sh`; design `consolidation/V4_VC2.md`;
`variants/v4vc2` = v4udm + VC2: VC2's worker channel visited every pass like VC0's, its credits from its own source word
(the channel's location-info read counter; main shares one scratch word between both workers' credits with no flush),
its receiver channel delivers every packet locally). Offline: 84 VC2 configs build (56 two-ERISC, 28 one-ERISC from a
new capture); 646 other images byte-identical to v4udm, 22 VC1 senders identical up to a permutation of three stack
slots in `sender::run`.
- The test needed a fix first: the streams port's benchmark loop called the VC0-only `update_credits_and_slots` for
  every adapter, so any VC2 sender kernel failed to compile (`streams/port/streams_vc2_replay.patch`, also in the
  streams worktree, uncommitted; VC0 benchmark sender binaries byte-identical). Every group then vetted on the mock.
- Main's router (job 141): two ERISCs 10/10. One ERISC: does not build: main static_asserts that a VC2 sender and
  receiver are on different ERISCs ("VC2 receiver channel not serviced"), so VC2 on one ERISC is not supported on main.
- v4vc2 (job 142), two ERISCs: the first group (VC0 traffic, VC2 enabled) passed; the second (`VC2MeshNeighbourExchangeVC2`,
  VC2 and VC0 neighbour exchange with flow control) hung (capture `results/vc2_v4vc2_two_hangcap`). Every VC2 sender
  channel stopped at exactly 512 packets (sent = completed = credited to the worker; the test's flow-control window is
  min(buffer, packets) = 512 at 2 KiB), all router channels empty, VC0 flows complete. The VC2 packets in both rings
  are well formed (src_ch 4, 2048 B, to the receiver core). So the senders wait on flow-control credits, which the
  receivers return on VC0 (`register_fabric_connection(..., vc_id 0)`) and which never came: cause not found. The
  capture's worker part found no test program (post-merge launch-message layout; `newest_program: None`), so the
  workers' state is unknown. One-ERISC VC2 not run (the job stopped at the hang).
- Found (user: fix the capture, then isolate). The capture read Tensix go messages with the pre-merge layout: since main
  6d8a103 they are rings, `go_messages[9].slot[8]` (one per launch slot), `go_message_index` at +1472, and every base
  after the mailbox moved 0x100; it read slot 0 (DONE) for test cores running in slot 1. `hang_capture.py` and
  `hang_workers.py` updated (HANG_CAPTURE.md). Isolation (job 148, `vc2iso_runs.sh`, groups one at a time): VC2 alone
  and VC2 in both directions passed; a VC2 and a VC0 flow on one router hung, and the capture now showed it: the VC2
  flow complete (40000/40000), the VC0 sender stuck at 43 with credit word 0 although its router had completed all 43.
  Cause: Blackhole's `noc_inline_dw_write` to L1 is emulated and always issues on `write_cmd_buf`, ignoring its
  cmd_buf argument, so every VC2 credit (and VC2's connection notify) overwrote the VC0 worker credit's preset on that
  buffer; VC0's credits then went to VC2's worker. (The code comments said "atomics buffer": wrong, now corrected.)
  Fix: the worker credit preset only on routers with one worker channel (`worker_credit_preset`); with VC2 both
  workers' credits are whole commands, each from its own source word. Other configs unchanged (646 identical, the 22
  VC1 senders as before).
- v4vc2 (job 152): the whole yaml 10/10 on two ERISCs and 10/10 on one ERISC (main cannot build the one-ERISC case);
  the sender images carry the VC2 worker code (36 of 36 E0 / single images). No hangs. Lead: **v4vc2**.

**Speedy path on v4: v4spd** (2026-10-29; a09u32 jobs 155-156, `spd_runs.sh`, `spdv_runs.sh`; `variants/v4spd` = v4vc2
+ the host's speedy-path routers). The host selects the speedy path (`ENABLE_SPEEDY_VC0`) for VC0 worker-only routers
whose receiver never forwards, e.g. every router of a NeighborExchange fabric (1D, one VC0 sender channel, receiver
forwarding off). Main runs a separate loop for them (amortized credits, ping-pong transaction IDs). In v4 the generic
loop already specializes to one worker channel; the only change: VC0's receiver channel delivers every packet without
decode or admission on these routers (the delivery-only path written for VC2), as main's speedy receiver. Main's
generic receiver ignores the forwarding-off flags (DISABLE_RX_CHx_FORWARDING, also set by channel-trimming replay),
and v4 does too outside the speedy path. **WRONG (2026-10-29, found while reading for trimming):** main's generic
receiver honors them (`run_receiver_channel_step_impl`, `if constexpr (!is_receiver_channel_forwarding_disabled[...])`
at 6d8a103's line 1523: no forward or delivery; the packet is still consumed and credited). Only v4 ignores them
outside the speedy path; the trimming port brings that in. The speedy credit-amortization args do not apply.
- Offline: the 12 NeighborExchange configs build (8 two-ERISC; 4 one-ERISC from a new capture, `bh_neighbor_1erisc`);
  the other 752 post-merge configs are byte-identical to v4vc2's.
- Validation (new `v4/val_nbr.yaml`: write, scatter, fused at 4352 and 1024 B, atomics, over every NeighborExchange
  link, data-checked, flow control on): main's router and v4spd 7/7 each, two ERISCs and one. A first version without
  flow control failed on main's router too (DATA_MISMATCH, job 155: senders overrun the receivers' buffers).
- Throughput, same job (job 155, `exp_1d_4352.yaml` in full; main as shipped, v4spd with the study's settings),
  cycles/packet: NeighborExchange write 124.9 / 124.8, scatter 159.9 / 128.7 (-31); the other 14 rows (1D lines,
  v4spd = v4vc2 there) -19 to -67, as before.
- Lead: **v4spd**.

**Telemetry on v4: v4tel** (2026-10-29; a09u32 job 157, `tel_runs.sh`; design `consolidation/V4_TELEMETRY.md`;
`variants/v4tel` = v4spd + the host's two telemetry options). New `router/router_telemetry.hpp`: packets and words
counted in the send and receive paths, per-round work out of line after each loop's `end_round` (perf recorder window
into the L1 buffer on the sender side; dynamic bandwidth, router state and TX / RX heartbeats into the FabricTelemetry
block), the counts in one global rather than in loop locals. Every call site is guarded by `if constexpr` on its option:
with only the hooks' functions empty, dead inline calls and lambda arguments still shifted register allocation in the
spill-heavy loops (VC1 senders, VC2 and speedy receivers; offline A/B), so a telemetry-off build now carries none.
- Offline: the 240 telemetry configs build (2D / 1D dynamic stats, perf recorder, all instruments on 2D, 1D, ring, VC1);
  every telemetry-off config's image byte-identical to v4spd's (764).
- Device (telemetry-on runs are functional checks only, not scored): `test_bandwidth_telemetry_validation`
  (dynamic bandwidth within 15% of measured, several sizes; built for this) 1/1 on main's router and on v4tel, two
  ERISCs and one (telemetry 10.9-11.3 GB/s on both). test_tt_fabric with its bandwidth telemetry on: exp_2d_4352 10/10
  and exp_1d_4352 16/16 on both; the perf recorder's bandwidth over the harness's own on unicast rows 1.03-1.04 (main)
  and 1.04-1.05 (v4tel; the recorder counts busy rounds only); multicast rows compare the busiest link to the sender's
  rate and read 1.1-1.5 (main) and 1.2-1.7 (v4tel). Heartbeats and router state not read separately.
- Lead: **v4tel**. The four features the user ordered (UDM, VC2, speedy path, telemetry) are in; still open: Wormhole,
  express / Z on Galaxy hardware, trimming replay on hardware (V4 feature inventory above), and the parked
  performance gaps (`consolidation/V4_PERF_GAPS.md`).

**Express / Z on v4** (2026-10-29; Galaxy Slurm job 162896 on bh-glx-110-a03u08, `frs_glx_torus.sbatch` suites
xr / xl / xr1 / xl1; note `consolidation/V4_EXPRESS.md`). Express links are physical Y chords inside a mesh (one
Galaxy: rows 2 <-> 5 of each column; the cluster descriptor has 68 chip pairs, the 8x4 torus's 64 plus these 4), which
the host models as a fifth direction, Z: 5 VC0 sender channels on every router, `MY_DIRECTION` 4 on Z-facing ones.
Main's router has no other express code, and v4's decode, downstream tables, live downstreams and credit-stream map
already follow main's rules for Z (offline review), so no router change.
- Test shapes (vetted on the mock Galaxy, `express/probe`): the sanity yaml's groups
  (`test_fabric_express_routing_sanity_common.yaml`, 24 tests each, data-checked) split by what a descriptor admits:
  `express_links_8x4_mesh_graph_descriptor` (RING x RING) takes Mesh and Torus XY (`express/glx_express_ring.yaml`, 5
  groups; it refuses Torus X: dim 0 has express links but no end wrap); a new single-mesh copy of CI's quad-8x4 mesh
  (`express/express_links_8x4_line_ring_mesh_graph_descriptor.textproto`: dim 0 LINE with the ex4 and row 0 <-> 7
  chords) takes Torus X (`express/glx_express_line.yaml`: CI's quad-8x4 groups).
- Offline: v4tel builds every express config (312: Mesh, Torus XY, Torus X; two ERISCs and one; new coverage cases
  `glx_express_mesh`, `glx_express_1erisc`, `glx_express_line`, `glx_express_line_1erisc`).
- Galaxy (post-merge trees; `/data/.../trees/base` rebuilt from the merged tree, the old one kept as `base_pre`):

  | Suite | main two ERISCs | v4tel two ERISCs | main one ERISC | v4tel one ERISC |
  | --- | --- | --- | --- | --- |
  | xr (Mesh, Torus XY on the express 8x4) | 120/120 | 120/120 | 120/120 | 120/120 |
  | xl (Torus X, mesh multicast on the line-ring 8x4) | 72/72 | 72/72 | 72/72 | 72/72 |

  v4tel's first post-merge Galaxy run of the older suites: ring sanity 19/19, torus deadlock stability 87/87.
- v4trm's express images are byte-identical to v4tel's, so these results hold for v4trm. Not run: VC1 with express
  (needs several Galaxies, CI's quad 8x4), express benchmark rows.

**Channel trimming on v4: v4trm** (2026-10-29; a09u32 jobs 159-163, `trim_runs.sh`; design `consolidation/V4_TRIM.md`;
`variants/v4trm` = v4tel + trimming). v4tel already recorded sender channels used, packet sizes, receivers that took a
packet and local NoC send types, but not the forwarded-to masks the host uses to turn forwarding off and to call a VC0
terminal-only. v4trm:
- records forwarded-to as main (2D: the forward's compact slots through the host's packed downstream ids; 1D: channel 1);
- honors `DISABLE_RX_CHx_FORWARDING` on the generic receiver as main (packet consumed and credited, nothing forwarded
  or written);
- opens downstreams by the host's gates (`OPEN_DOWNSTREAM_VC0/VC1_CONNECTIONS`; v4tel keyed VC1 on receiver 1, which
  trimming clears on crossover routers whose receiver 0 forwards on VC1);
- runs trimmed role shapes: on two ERISCs an ERISC left with nothing to service runs rounds only (`run_idle`: router
  commands on the master, context switch, termination); one ERISC runs the single-ERISC loop whatever roles are left;
  receiver channels run only when serviced. The first device run (job 160) found these: the capture tests passed, but
  the replay scenarios' routers were refused by the bring-up gate (75 builds, every one an ERISC with no serviced
  channel).
- Offline: 1,260/1,390 post-merge configs build (the 130 `_sc` ones fail by design, as on v4tel); 1,174 images
  byte-identical to v4tel's, the 86 others the capture configs' receiver images.
- Device: main's `ChannelTrimming*` gtests (capture bitfields, fixed-override traffic, 11 capture-then-replay scenarios
  on Mesh and Torus X, multi-hop included): main 63/63 and v4trm 63/63, two ERISCs and one (3 T3K-only tests skip).
  The runs built every trimmed shape (idle ERISC 0 / ERISC 1 on two ERISCs; one ERISC with no role, sender-only,
  receiver-only). Main's replay now works multi-hop on this host (the pre-merge job 282 note is superseded).
- Replay configs for offline builds: coverage cases `bh_trim_gtest` and `bh_trim_gtest_1erisc` (ingested from the
  v4trm runs; v4trm 316/316, v4tel 224/316).
- Not measured: trimmed throughput. Lead: **v4trm**.

**Perf census after the features** (2026-10-29; a09u32 jobs 164-171, `census_runs.sh`, `census_vc1_runs.sh`,
`census_report.py`; Galaxy Slurm job 163369; notes `consolidation/V4_PERF_GAPS.md` (census section) and
`consolidation/V4_CODE_SHAPE.md`). User: tackle the perf gaps; first see where every config stands, and check whether the
features degraded the code. Same jobs, two passes (no v4trm row moved more than 3% between them), routers back to back:
main as shipped (kernel v1, flags 0: none of the study experiments; firmware keepalive), main at keepalive 1000, v3qm
(`variants/v3qm` = v3q with the post-merge header path only; mesh sanity 96/96, ring 19/19), v4trm.
- Code: every production image of v4trm equals v4rs's (two-ERISC 2D / 1D / ring / torus, one ERISC, mux, VC1
  receivers); only the 22 VC1 senders differ (stack-slot permutation) and their rows equal v4gi's. The features cost
  nothing on the measured configs. Readability findings (independent review, the key ones checked) in V4_CODE_SHAPE.md.
- Suite means (main / main ka1000 / v3qm / v4trm): 2D 217.2 / 217.2 / 144.5 / 144.4; 1D 200.5 / 196.8 / 162.0 / 152.2;
  mix 222.2 / 219.9 / 127.0 / 128.7; bursts 555.0 / 555.0 / 522.5 / 523.3; ring 239.9 / 238.6 / 227.6 / 188.6; Y-torus
  233.7 / 233.8 / 164.7 / 157.4; one ERISC 2D 331.3 / 331.3 / 245.5 / 276.6, 1D 318.3 / 318.5 / 284.5 / 305.7, ring
  375.7 / 374.2 / 360.6 / 375.1; mux 373.3 / 373.3 / 368.7 / 373.6; trimmed 2D 215.9 / 211.9 / - / 154.4; VC1 split
  332.6 / - / 155.2 / 171.5 (1-hop 158 vs v3qm 137), inter-mesh 841.0 / - / 336.4 / 342.5. The keepalive is worth
  0-4 on main here.
- Galaxy (main / v4trm, both passes): mesh and line rows below 4352 B 1.18-1.85x main; ring 1-hop 1.27-1.29x; full-ring
  multicast 1-3 KB 1.32-1.37x, 3584 B 1.07x, 4352 B 0.76x (263.7 vs 346-349); 4352 B unicast wire-bound on both.
- New against the parked list: v4 trails v3q on one ERISC (2D +12.7%, 1D +7.5%, ring +4%) and on the VC1 split (+10.5%;
  1-hop +15%, size-independent). The old standings had v3q's VC1 at 180-198, measured before the VC1 slot defaults
  changed, so this gap was hidden, not new. 13 one-ERISC rows trail main (NeighborExchange write +7.4%, half-ring and
  full-ring multicast, ring bidirectional, mux ring multicast: +1.7% to +4%).

**Profile matrix: bound side, register pressure, phases** (2026-10-29; a09u32 jobs 172-176; `consolidation/V4_PROFILE.md`,
`prof_delay_runs.sh`, `prof_report.py`, `prof_pcs_runs.sh`, `regpress.py`; user: capture the data before choosing
between readability and perf). Instruments: a fixed spin after each send or each received packet (`variants/v4trmd`, at
0 byte-identical to v4trm, 600 of 600), PC sampling (no router code), and a static scan of the ELFs. Main's on-device
phase timers were not used (they perturbed rows 13-45%).
- Bound side, rows per suite (TX / RX / both / neither): 2D 6/4/0/0, 1D 0/8/2/6 (the 1-hop writes run at the wire),
  mix 21/8/1/3 (every write TX, scatter / fused / atomics RX), ring 0/13/0/1, Y-torus 5/5/2/0, one ERISC 2D 0/8/2/0, 1D
  0/7/9/0, ring 2/6/8/0, mux 0/0/2/20 (the mux bounds 20 rows), VC1 8/1/2/2 (1-hop TX).
- Phases on the bound side: a two-ERISC sender spends 56-73 cycles per packet on the send and 50-75 on checks, the
  free-slots group read, credits and dispatch, as v3 does; ring receivers spend ~31 on first-level acks. On one ERISC the
  receiving router's core also runs an idle sender pass (~35-47 per packet) and the out-of-line pass structure (~26-39).
- Static: the bound senders have the biggest passes and spills (2D 582 instructions / 58 stack loads, Y-torus 709 / 78,
  VC1 1051 / 120; v3 515 / 43, 579 / 56, 914 / 68); receivers 4-9 stack loads (VC1 21).
  **WRONG as "spills"** (2026-10-29): the sender's stack loads include its ForwardingTable records (a local of
  sender::run, memory by design); on the 2D sender ~64 of 94 in-loop stack operations are table accesses, ~30 spills.
  **That correction is itself WRONG** (2026-10-29, DWARF of the v4trm_O3 images): without VC2 the compiler scalarizes
  the table (location lists), so these are spills; only on VC2 routers is the table a frame object (VC2's worker
  interface inside it escapes to out-of-line calls).
- Hypotheses for the gaps (A/B first): one ERISC, the idle sender pass and the pass structure; VC1, the sender pass;
  3-hop / multicast, the sender's checks and dispatch; state cutting only for the two-ERISC senders.

**What a cycle of pass cost is worth** (2026-10-29; a09u32 jobs 177-179; `consolidation/V4_PROFILE.md` §6; user: optimize
the router generically, judged on the whole matrix, not one row). A fixed spin once per sender pass or once per receiver
pass (`variants/v4trmd`, still byte-identical to v4trm at 0) over every census suite: at the margin (16 cycles per pass)
one cycle of sender-pass cost costs 1.2-2.1 cycles per packet on the 94 of 159 rows it binds (every two-ERISC write
row, small writes ~2; every one-ERISC row, where it runs on the bound core even idle; VC1 1-hop ~1.9 with the receiver
pass ~0); the receiver pass binds 108 (scatter / fused / atomics, rings, Y-torus N/S, every one-ERISC row); 27 rows
neither (mux-bound, wire-bound). Past the slack the cost is 1.0-1.2 per cycle. The 2D sender's in-loop stack traffic
(94 operations against 1D's 49) is loop-invariant reloads at the back edge, NoC / TX-queue / header addresses and the
per-channel ring cursors around each send. **WRONG as first written:** ~64 of the 94 are the forwarding table's record accesses; ~30
are reloads. **That marking is itself WRONG:** the table is scalarized there; the 94 are spills and reloads.

**Sender structure cleanup, steps 1a / 1b / 2** (2026-10-29; `consolidation/V4_SENDER_SHAPE.md`; a fresh review of the
design first, user's request). The review, confirmed in the DWARF of the images: placement is the compiler's, decided by
escape: on two-ERISC routers without VC2 the forwarding table, the worker's counts and VC0's far ring are scalarized; on
VC2 routers the table is a frame object (VC2's worker interface in it escapes), so VC2 already changes the sender's
shape. (This makes the "stack traffic is table accesses" correction to the profile matrix WRONG; marked there.)
- 1a `v4sh1` (one worker-channel credit and connection event, a constexpr credit table, RX from channel traits, dead
  code and stale comments): 1650 of 1706 images byte-identical to v4trm (every config without VC2); VC2 images: same
  placement, sender pass +2 instructions; validation 13/13 (job 181). Accepted.
- 1b `v4sh1b` (dense forwarding records and upstream addresses): every sender image changes, loops unchanged but 2D
  +6 instructions, Y-torus -2; validation 13/13; census against v4trm, two passes (jobs 183-190): every suite within
  0.1%, largest row 0.4%. Accepted: the lead's structure moves to v4sh1b.
- 2 `v4sh2` (one register read for every VC): VC1 1-hop -0.1% to -0.2%, forwarding rows +0.6% to +2.2% (jobs
  191-192). Rejected; the shared group helper kept with VC1's read in place (`v4sh2b`: the same instructions as
  v4sh1b at the same addresses, a few with commuted operands in symmetric uses). Lead: **v4sh2b**.
- 1c `v4sh3` (user: the sender state by channel role: the worker channels' state, their interfaces apart, the
  forwarding table with forwarding channels only, one far ring per VC): 748 of 1214 images byte-identical to v4sh2b
  (two-ERISC configs without VC2, VC1, telemetry or express); VC1, telemetry and express senders renamed registers or an
  init-only change; one-ERISC sender pass +2 instructions; VC2 sender pass 631 -> 730 (the table is split there now:
  `frame` -> `split`). Validation 13/13 (job 193). Census, two passes agreeing within 0.1 (jobs 195-200): VC2 on, 2D
  +2.1%, mix +3.6%, one ERISC +1.0% (1-hop writes 154-160 -> 170); without VC2, one ERISC 2D +1.0%, 1D +0.9%, ring
  +0.7%; VC1 1.000. Not accepted as is. `v4sh3m` (measurement only; VC2's state in memory with its interface) isolates
  VC0's records: split instead of in memory costs the VC2 1-hop writes ~6% (hypothesis for other configs). Side finding:
  VC2 on doubles VC0's 3-hop write cost on any router (v4sh2b 151.9 -> 304.3); cause unknown.

**Controlling placement, stage 1** (2026-10-29; `consolidation/V4_PLACEMENT.md`, `placement/`; user: find a way to
choose what goes on the stack first; DESIGN_BRIEF rule 2 amended for documented placement primitives). A compile-only
sender-like test at -O3/-O2/-Os: `asm volatile("" : "+m"(obj))` (and `volatile` fields) keep the named object a frame
object at every level, with loads and stores exactly where the source writes them, no added cost (pass loop equal or
fewer instructions), the grouped register reads undisturbed. It applies to the whole enclosing object (`in_flight` in the
table went to memory with it). Out-of-line calls in the loop limit what can stay in registers (hypothesis for the router:
the per-pass set competes for the 12 callee-saved registers). Stage 2 (the router) next.

**Controlling placement, stage 2: `v4pl1`** (2026-10-29/30; `consolidation/V4_PLACEMENT.md`). v4sh3 + `keep_in_memory()`
(router_common.hpp: an asm memory operand) on each forwarding record at the end of its visit + `in_flight` out of the
table into its own local. Offline: the table is a frame object on every sender at -O3/-O2/-Os and under three benign
edits (v4sh3: split everywhere); sender pass 2D 588 -> 500 instructions, VC1 1051 -> 894, Y-torus 707 -> 631; layout
and declaration-order edits no longer move the pass. The stack is the ERISC's local data memory (2 cycles a load, §17),
not L1. Device (jobs 201-211, two passes agreeing within 0.001): validation 13/13; against v4sh2b, VC1 split -6.2%
(1-hop 158 -> 141), one ERISC 2D -2.2%, trimmed -2.4%, mix -1.6%, 2D -1.0%, Y-torus -0.7%, 1D/ring/burst/mux flat;
VC2 on +0.5% to +1.9% (two ERISCs), one ERISC -2.1%; four rows worse by 1.2-2.6% (VC1 2-hop merge, VC1 multicast,
one-ERISC NeighborExchange write).

**Grouped local-memory loads** (2026-10-30, job 216; §17's kernel with new patterns; `consolidation/V4_STATE.md` §7):
in a group with four stream-register reads, each local-memory (stack) load adds ~1 cycle, first or last (4 -> 5 with
one, -> 7 with three), with no four-in-flight penalty (an L1 word: 4 -> 14). A header L1 read through a slot held in
local memory: +1 cycle when the slot loads with the group, +2 when it loads right before.

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
| `TT_FABRIC_EXP_SLOTS=W,F,R`, `_SLOTS_1D`, `_SLOTS_2D` | VC0-only fabrics: worker W (0 = every slot left, at most 127), every forwarding channel F, receiver R slots; `_1D` (Linear, Ring) and `_2D` (Mesh, Torus) set one family and take precedence (§98, slot allocation sweep) |
| `TT_FABRIC_EXP_SLOTS_VC1=W,F0,R0,F1,R1` | VC0 + VC1 fabrics: worker W (0 = every slot left), every VC0 forwarding channel F0, the VC0 receiver R0, every VC1 sender channel F1, the VC1 receiver R1 (§98, VC1 allocation sweep). Unset, the router in the v2 slot on Blackhole VC1 routers of the measured shape (4 VC0 + 3 VC1 senders) gets `0,9,16,5,4` (worker 19), from `consolidation/alloc_model.py`; main's router keeps the tier table 51,4,8,2,4 |
| `TT_FABRIC_EXP_TIER_SLOTS=1` | Blackhole with the router in the v2 slot: keep the tier table's depths instead of the builder's study defaults: 1D (Linear, Ring) 16 / 16 / 32 instead of the even split 26 / 25 / 32 (§98, 1D slot split), VC1 routers 51,4,8,2,4 instead of 19,9,16,5,4 (§98, the builder's VC1 default). Main's router always keeps the tier depths |
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
