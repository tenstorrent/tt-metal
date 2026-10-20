# Fabric router placement model ("lego" model)

A calibrated model for deciding how the fabric router's work should be split across the two ERISCs of an Ethernet
core and a DRISC. Pure Python 3 + PyYAML, no device needed; runs in seconds.

## Pieces

| File | Role |
| --- | --- |
| `catalog.yaml` | The blocks: per-piece cycle costs (per occurrence, per batch), where each piece may run, poll costs, connector costs between cores, link and NoC parameters, and traffic scenarios. Every number is tagged `measured RESULTS §N` or `estimated: <basis>`. |
| `fabric_model.py` | Static model: places every piece on exactly one core, adds connector costs where adjacent pieces are split, sums each core's per-packet load (work + loop/poll overhead amortized over packets per iteration), applies wire, keepalive and NoC-port floors, and searches all placements (DRISC symmetry pruned). |
| `des.py` | Event-driven model of one link in both directions: each core runs its pieces in a polling loop; sender rings, credits, receiver slots, a serialized wire per direction (data, pointer updates and acks share it), NoC ports and handoff queues are modelled, so credit loops, latency and batching dynamics show up. Forwards go to downstream channels with finite slots that drain at the downstream link's rate (optionally congested or stalling), so admission can fail: a blocked packet stays at the head of its queue and is retried. |
| `validate.py` | Static model vs. hardware measurements. |
| `explore.py` | Searches placements for each scenario × core set × lever set; `--verify` re-runs the best placements in the event-driven model. |
| `drisc_study.py` | Two ERISCs + one DRISC: gain over two ERISCs per scenario and lever (both models), sensitivity of that gain to the DRISC's estimated costs with break-even points, and DRISC staging depth. |
| `validate_1d.py` | 1D linear multicast, router by router: per-router send and receive rates from the phase profile, the slowest router against the measured interval, delay injection, and the receive-pipeline designs on that traffic. |
| `tx_split_study.py` | E1 also owning some sender channels (sending on TXQ1) once receive work is on a DRISC: both ways and one way, 1 or 4 busy channels, batching 1 or 4, with sensitivity to E1's per-iteration costs. Uses the §17 sending-side costs, not the catalog's older polling parameters. |
| `pipeline_study.py` | Receive-side pipelines: who decodes, who admits (holds the downstream credits) and who moves the data, under downstream congestion, with the unmeasured handoff costs swept. |

Core sets: two ERISCs, or two ERISCs plus one DRISC per router. BH has 24 DRISCs (8 DRAM channels x 3
subchannels); Metal gets 16 (each channel's NoC0 endpoint is reserved for system firmware,
`metal_soc_descriptor.cpp`), against at most 10 active Ethernet cores, so every router can have its own. Of the 16,
8 have both NIUs in stream mode ("free subchannel": can initiate on NoC0 and NoC1); on the other 8 NIU1 stays in
NOC2AXI for the DRAM path, so they initiate on NoC0 only (`drisc_mode.h`). DRISC fields in the catalog: `alpha`
(DRISC cycles per ERISC cycle for the same work), `share` (routers served), `egress_bytes_per_cycle`, `noc_ports`.
Both ERISCs send payload copies through their Ethernet tile's NoC port: today local writes and forwards both use NoC1
(`eth_tile_noc_ports: 1`). Placement rule (`rules.rx_on_tx_erisc: false`): no receive piece runs on E0, the sending
ERISC, so SC0 and RC0 run in parallel; transmit work may move to E1.

Link (`catalog.yaml: link`): wire time is Ethernet framing at the 400 Gb/s line rate, with no fitted parameter.
- A packet (payload + header: 96 B in 2D on this mesh, 48 B on the speedy path) goes out as frames of at most 4114 B,
  each with 56 B of framing (preamble, gap, MAC header, FCS, 18 B TT-link header).
- Pointer updates are minimum-size frames (84 B on the wire).
- Credits are counters on BH: one L1 write of the counter arrays, 96 B in 2D (152 B on the wire) and 32 B in 1D (88 B)
  (`credit_bytes`). In both-way traffic they share each direction's wire with its data.
- A 2D 4 KiB packet is two data frames and a pointer update: 118.5 cycles (116.8 with pointer updates batched by 4).

One-way scenarios (`reverse_data: false`) also get the keepalive floor: with no data coming back, the sender's
acknowledgements arrive only in the peer's keepalive, so at most 128 frames go out per keepalive period. The
catalog assumes the keepalive at 1000 cycles (`TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT`, RESULTS §16), where it never binds;
`Options(keepalive=8000)` gives today's firmware (2D 4 KiB one way: 3 frames x 62.5 = 187.5).

The study outputs under `results/` were made before 2026-10-15, with an 80 B 2D header and 84 B credits. Their wire
floors are ~0.4 cycles low one way (more for payloads within 16 B of a frame boundary) and ~2.3 low both ways with
one credit per packet.

Levers (`Options`): `idle_poll` (how the sending ERISC polls its sender channels: `scan` today, `lean`, `fused`,
`fused_overlap`; see "Sender-channel polling"), `summary_doorbell` (producers also ring one shared counter),
`b_ptr` (packets per Ethernet pointer update), `b_ack` (packets per completion ack), `k_drain` (packets handled per
queue visit), `sender_channels` (e.g. trimming), `noc_ports` (NoCs used for payload copies), `cache_decode` (a
blocked packet keeps its decoded action), `blocked_per_pkt` (admission failures per packet, static model),
`relay_batch` (credits per relay when a DRISC admits), `payload`.

Receive pieces: `rx_poll` (arrivals), `rx_decode`, `rx_admit` (downstream credit check and slot reservation, present
when the packet has forwards), `rx_local`, `rx_forward` (data movement), `rx_ack`. Handoffs between them are costed by
pattern: ERISC->ERISC descriptor, push of the whole packet to a DRISC (the decision rides along), grant of a packet
already in the DRISC, header-only push, DRISC->ERISC request (decoded output list) or decision, and credit relay when
a DRISC admits.

## Running

```bash
cd fabric_exp/model
python3 validate.py [-v]          # static model vs hardware (per-core breakdown with -v)
python3 des.py --validate         # event-driven model vs hardware
python3 des.py --calibrate        # refit the one-way link latency to the slot sweep
python3 explore.py [scenario ...] [--payload N] [--detail]   # default 4096 B, checked against line rate
python3 explore.py [scenario ...] [--payload N] --verify
python3 explore.py [scenario ...] [--payload N] --slots     # event-driven interval vs slot depth
python3 drisc_study.py [gain|sensitivity|staging] [--scenario NAME ...] [--payload N]
python3 pipeline_study.py [designs|congestion|sensitivity|sharing|scan|alpha] [--scenario NAME ...]
python3 polling_study.py [--payload N]    # sender-channel polling schemes on the sending ERISC
python3 burst_study.py [--scenario NAME] [--payload N]   # what batching is worth in bursty traffic
```

To change a hardware assumption, edit `catalog.yaml` (piece costs, core types, link, NoC) and re-run. To add a lever
or a piece, extend `fabric_model.evaluate` and `des.simulate` together.

## Validation (Blackhole P150x8, 400G links)

- Static model, 28 configurations (speedy neighbor at 1-8.7 KB both ways and at 2-4 KiB one way; 2D single flows
  E/W and N/S at 2 and 4 KiB, untrimmed and trimmed, write and scatter; 3-hop; mesh multicast heaviest link): mean
  |error| 1.1%, max 3.5%. It puts the bottleneck on the right core (TX for untrimmed 2D; RX for trimmed N/S, scatter
  and the 3-hop middle router; wire above ~4.3 KB on the speedy path). Per-packet core costs do not depend on size:
  2D intervals are the same at 2 and 4 KiB (220-222 cycles untrimmed; trimmed 152.5-206.5 at both sizes once the
  keepalive is at 1000). The wire-bound points (neighbor exchange at 5.4-8.7 KB) are within 0.1%, from framing alone.
- At the firmware keepalive (8000), 10 one-way configurations: mean |error| 1.5%, max 4.5%. The 187.5 floor at 4 KiB
  is exact; where the floor and a core limit are close the hardware is above both (3-hop 196.4 vs 187.5; one-way
  neighbor exchange 129.6-130.3 vs 125), which a max() of the two misses.
- Event-driven model, 19 configurations including the slot sweep (2-32 slots, the latency-bound regime), the
  wire-bound sizes (within 0.1%) and trimmed 2D at 4 KiB: mean |error| 2.4%, max 6.5%, with one fitted parameter
  (one-way link latency = 450 cycles; refitting with the framing-based wire leaves it at 450). The worst point is the
  trimmed E/W write at 4 KiB: 158.7 vs 169.8 measured; the model has it at 176 at 2 KiB and 159 at 4 KiB, while the
  hardware is flat at 168-170 (the old link model gives the same, so it is not the wire).
- 1D linear multicast (`validate_1d.py`, static, per-router send and receive rates): 12 configurations (bidirectional
  and unidirectional, write and scatter, 1/2/4 KiB): mean |error| 5.6%, max 10.8%, all low. The event-driven model
  assumes a router sends and receives at one rate, so it does not run these.

## Target: 400G at 4 KiB

Line rate at 4 KiB is 118.5 cycles per 2D packet (4096 B payload + 96 B header: two data frames and a pointer update;
the framing matches a bare-link probe at 4176 B, 118.1, RESULTS §16), ~117 with pointer updates batched. That is about the same core budget
as 200G at 2 KiB, but every payload copy now takes ~76 cycles of a NoC port. These results assume the keepalive fix
(RESULTS §16); at the firmware keepalive no one-way 2D link gets below 187.5 at 4 KiB. Best placements (`drisc_study.py`, event-driven with 32 slots and 16-slot
downstream channels; static in parentheses; DRISC costs measured, `results/drisc_study_gain_4k_measured.txt`):

| router | levers | 2 ERISC | + DRISC | line rate? | limit |
| --- | --- | ---: | ---: | --- | --- |
| 1-hop write (E/W, N/S) | lean poll + batch x2 | 118 (117) | 118 (117) | yes, no DRISC needed | wire |
| 1-hop scatter | lean poll + batch x4 | 138-153 | 124 (124) | no, ~5% short | the core doing scatter |
| 3-hop middle (1 forward) | lean poll + batch x4 | 132 (132) | 126 (116) | only with the DRISC, and ~20-slot downstream channels (117 with 24) | E1 / wire |
| mesh multicast heaviest link | batch x4 | n/a (117) | n/a (117) | yes (static only) | wire |
| multicast N/S router (local + 2 forwards) | lean poll + batch x4 | 284 (298) | 188 (159); 153 with a forward-only DRISC and 24-slot downstream | no, ~30% short | the DRISC's NoC port (two 4 KiB forwards) |

- **Slot depth gates everything.** With the mesh's 16 slots per channel, nothing reaches line rate: best 139-149
  (1-hop, lean poll + batch x2), 154-186 with batch x4. A worker's sender slot is freed only after the far side has
  processed the packet and acked it, so its loop is two NoC legs plus the Ethernet round trip plus remote
  processing. The sender channel is the first limit (more receiver slots alone change nothing), the receiver the
  second: line rate needs ~24 sender and ~24 receiver slots with batching by 2 (s24/r24: 118), ~32/24-32 with
  batching by 4. Today's 2D allocation is 4 sender channels x ~16 + 16 receiver slots of 4448 B (~360 KB of the
  512 KB L1); 24 on every channel would be ~534 KB. It fits if only the active sender channels are deep (e.g. 24 +
  3 x 11 + 24 receiver = 81 slots, today's budget). Freeing the worker's slot earlier with the existing first-level
  ack (enabled today only with deadlock avoidance; it costs an extra ack per packet) would shorten the sender loop;
  not modelled yet.
- **1-hop writes do not need a DRISC.** Lean polling + batching on the two ERISCs reach the wire.
- **Forwarding routers need the DRISC.** With E0 kept to transmit work, two ERISCs leave the 3-hop middle router at
  132 (E1-bound). The DRISC moving the data gets 117, given ~20-slot downstream channels (126 with today's 16): the
  extra hop lengthens their credit loop. (Moving arrival polling and decode onto E0 would reach 120 without a DRISC,
  but puts receive work on the SC0 ERISC; the placement rule excludes it.)
- **Scatter is limited by one core's delivery rate** (~103 cycles per packet plus overhead): 124 wherever it
  runs. Line rate needs scatter split across E1 and the DRISC (alternate packets, subject to ordering); the model
  does not split a piece across cores yet.
- **Multicast fan-out needs the DRISC.** On two ERISCs E1 carries decode, admission, the local write and two
  forwards (284; three 4 KiB copies also need 228 cycles of the tile's NoC port). A DRISC that only forwards gets
  153, the cap of its NoC port carrying both 4 KiB forwards (receive pipeline section below).

### DRISC at 4 KiB: what it depends on

- Gains with measured DRISC costs (static, against the best two-ERISC placement with E0 transmit-only, lean poll +
  batch x4): 10-19% on 1-hop scatter, 11% on the 3-hop middle router, 47% on the multicast router (event-driven at 32
  slots: 10-19%, 5%, 34%). Lost if one DRISC serves two routers (receive pipeline section).
- **The DRISC's NoC rate:** measured 53.5 B/cycle on both NoCs (RESULTS §14), so a 4 KiB copy takes ~76 cycles of
  a port; one DRISC sends on both NoCs at once at ~98 B/cycle.
- Header-only offload (DRISC decides, ERISC moves the payload) does not pay: the header round trip costs the ERISCs
  about what the decode saves.
- Staging: 8 DRISC slots per router (4 KiB each, ~35 KB of DRISC L1); with 4 slots throughput drops 20-30%.
- One DRISC per router fits (16 Metal DRISCs, at most 10 Ethernet cores). Pairing 10 routers uses at least 2
  NoC0-only DRISCs: fine for forwards; DRAM-bound local delivery from NoC0 needs an endpoint remap (doc §6.7).
- **DRISC costs are now measured** (`tests/tt_metal/tt_fabric/drisc_microbench/`, RESULTS §14; its README maps each
  output to a catalog field). The catalog's DRISC entries now hold them, assuming lean stateful issue: alpha = 1.0,
  DRISC NoC rate 53.5 B/cycle on both NoCs, push 20 + 12 per doorbell, doorbells and credit returns 12-13, DRISC
  forward copy 45 (31 measured + 14 estimated). Still at the old estimate: independent stream-register reads 5-6
  each measured vs `fused_per_channel.overlapped` 3 (polling study).

### At 2 KiB (200G)

`--payload 2048` reproduces the earlier 2 KiB results: batching needs 24-32 slots there too (2D 1-hop: 148 at 16
slots, 93 at 32 with batch x4); the DRISC gains 11-16% on the 3-hop middle router,
12-19% on 1-hop scatter, and nothing on 1-hop writes.

## Sender-channel polling (`polling_study.py`)

Polling cannot be removed: each sender channel has its own producer (the worker, or the router receiving from
another direction), so every channel still has to be checked for new packets. What can change is what else a visit
checks and how the checks are issued. Today a visit makes three checks per channel (`run_sender_channel_step_impl`):
new packet (free-slot stream register), completions from the far receiver, and connection or teardown request. Only
the 27-cycle total per idle channel is measured; the split (11 / 10 / 6) is an estimate from the primitive costs.

| scheme | what changes | protocol change |
| --- | --- | --- |
| scan | today | - |
| lean | idle channels get only the packet check; completions are checked only on channels with packets in flight; connection and teardown one visit in 16 | no |
| fused | lean, with every channel's packet check in one block of reads and one branch | no |
| fused, overlapped | fused, if the core keeps several MMIO reads in flight (unmeasured) | no |
| summary doorbell | producers also increment one shared counter; idle channels are skipped while it agrees with the active channels' counts | yes: +24 per packet on every producer |

E0 cycles per packet at 4 KiB (static, today's placement; the 3-hop column is E1, the forwarding core):

| scheme | 4 channels (VC0) | 9 channels (VC0 + VC1) | 3-hop middle E1 | with batch x4: 4 ch / 9 ch / 3-hop E1 |
| --- | ---: | ---: | ---: | ---: |
| scan (today) | 221 | 356 | 185 | 90 / 124 / 132 |
| lean | 168 | 225 | 185 | 77 / 92 / 132 |
| fused, reads block | 164 | 211 | 185 | 76 / 88 / 132 |
| fused, reads overlap | 140 | 157 | 185 | 70 / 75 / 132 |
| summary doorbell | 152 | 152 | 209 | 73 / 73 / 156 |

- **Lean polling gets most of it with no protocol change.** Two of the three checks on an idle channel look for
  events that cannot happen (no packets in flight, so no completions) or are rare (teardown).
- **Fused polling is worth it only if the ERISC overlaps the reads.** If each read blocks (9 cycles, as the
  microbench measures reads whose result is used at once), it saves only branches (~4 cycles on 4 channels); if
  reads overlap, 140 vs 168 at 4 channels and 157 vs 225 at 9. RESULTS §11 found the ERISC stalls on a load only when
  its value is used (a busy read costs ~8 cycles only if consumed at once), so the blocking row is a pessimistic
  bound. One microbench settles the per-read cost: N independent stream-register reads issued back to back, results
  used afterwards.
- **The summary doorbell moves the cost to producers.** In a forwarding router the forwarding core pays 24 more per
  copy: the 3-hop E1 goes 185 -> 209, and 132 -> 156 with batching. A counter also does not say which channel has
  the packet, so the sender has to compare it with the active channels' counts and fall back to a full scan.
- **Batching amortizes whatever polling is left, but only while packets queue** (next section). Draining up to 4
  packets per visit divides the per-visit polling by up to 4. In a sustained stream every scheme then reaches line
  rate on 4 channels, and the scheme matters again with VC1 (scan stays E0-bound at 124).

## Bursty traffic (`burst_study.py`)

All batching in the model is opportunistic: a visit sends what is queued (up to 4), one Ethernet pointer update
covers the packets sent in that visit, and the receiver acks what it has retired as soon as it runs out of new
packets. Nothing waits for a batch to fill, so batching acts only when packets queue. Each worker sends a burst of B
packets, then idles; burst time runs from the burst's start to the last packet's local delivery on the far side
(2D 1-hop E/W, 4 KiB, 32/32 slots; cycles per packet = burst time / B, latency included):

| B | today | lean | lean + batch x4 | batching gain | lean + batch, slow producer (150 cycles/packet) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 964 | 937 | 937 | 0% | 1049 (0%) |
| 4 | 395 | 342 | 350 | -2% | 373 (0%) |
| 8 | 292 | 242 | 241 | 0% | 261 (0%) |
| 16 | 256 | 206 | 180 | 13% | 207 (4%) |
| 64 | 234 | 181 | 136 | 25% | 164 (9%) |

- **Batching does nothing for bursts of up to ~8 packets**, and little when the producer is slower than the
  batched router (150 cycles/packet: little queues). It pays only in long bursts or sustained streams from a fast producer: the
  bandwidth benchmarks, and the 400G target.
- **Short bursts are latency-bound.** A lone 4 KiB packet takes ~940 cycles from the worker to local delivery (one-way
  Ethernet latency alone is ~450). Lower per-packet cost (lean polling) still helps every burst size, by 3-23%.
- So batching is not a substitute for a cheaper per-packet path; the two serve different workloads.

## Receive pipeline: who decodes, who admits, who moves data (`pipeline_study.py`)

Today a blocked packet is decoded again on every visit: each pass re-reads the header at the head slot, decodes it
and re-runs admission; nothing is kept when admission fails (`fabric_erisc_router.cpp:1597-1746`; intermesh
landings also re-encode the header). Measured admission failures: 0.27-0.29 per packet for mesh multicast, 0 for
the unicast benchmarks. Designs compared (all keep all-or-nothing admission and per-channel order):

| design | decode | admission + downstream credits | data movement | Ethernet slot freed |
| --- | --- | --- | --- | --- |
| today | E1, again on every retry | E1 | E1 | after the writes are sent |
| P0 | E1, once (decoded action kept) | E1 | E1 | as today |
| P3 | E1, once | E1; grant rides on the push | DRISC | after admission + push |
| P1 | DRISC (request back to E1) | E1 (grant to the DRISC) | DRISC | right after the push |
| P2 | DRISC | DRISC, with credits relayed by E1 | DRISC | right after the push |
| P4 | E1, once (E1 also delivers locally) | DRISC, with credits relayed by E1 | DRISC (forwards) | right after the push |
| P4d | E1, once (E1 also delivers locally) | DRISC, credits returned to it by downstream routers (protocol change) | DRISC (forwards) | right after the push |
| P5 | DRISC | DRISC, with credits relayed by E1 | DRISC: forwards on one NoC, local delivery on the other | right after the push |
| P5L | E1, once (E1 also delivers locally) | DRISC, with credits relayed by E1 | DRISC: forwards on one NoC | right after the push |

P5 and P5L are the push designs with a DRISC that sends on both NoCs, forwards on one and local deliveries on the
other, so no Ethernet tile is written by DRISCs on both NoCs (pull designs are ruled out: RESULTS §14). E1 relays
whatever downstream credits have arrived each time it relays.

In every DRISC design the DRISC is a dumb forwarder for the data: whoever admits also reserves the downstream slot,
and the DRISC writes the packet into it and rings the downstream doorbell (it must, so the doorbell lands after its
own write). Local delivery holds no downstream credit, so it needs only the decode and never waits for admission.

Best placement per design, 4 KiB, lean poll + batch x4, E0 kept to transmit work (event-driven: 32/32 Ethernet slots,
12 DRISC staging slots, 24-slot downstream channels; static in parentheses). Handoff and DRISC costs are the measured
lean-issue values (catalog, RESULTS §14); the DRISC forward copy is 45 cycles (31 measured + 14 estimated for the
next-hop slot address and header update). Raw output: `results/pipeline_study_p5_*.txt`.

| router | today | P0 | P3 | P1 | P2 | P4 | P4d | P5 | P5L |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3-hop middle (1 forward) | 132 (132) | 132 (132) | 117 (116) | 117 (116) | 117 (116) | 117 (116) | 117 (116) | 117 (116) | 117 (116) |
| multicast N/S (2 forwards + local) | 284 (298) | 284 (290) | **153** (170) | 164 (159) | 188 (205) | **153** (178) | **153** (153) | 222 (248) | **153** (178) |
| E1 load, 3-hop / multicast (static) | 132 / 298 | 132 / 290 | 89 / 170 | 84 / 154 | 77 / 159 | 96 / 178 | 71 / 128 | **77 / 102** | 96 / 178 |

- **With measured costs every DRISC design reaches the line rate on the 3-hop router** (117 event-driven, 116
  static: the wire with pointer updates batched by 4). The old estimates (77-cycle DRISC copies, 24-cycle doorbells) had P1 and P2 at 137-142.
- **Multicast N/S is capped at ~153 by the DRISC's NoC port**: its two 4 KiB forwards share one port (2 x 76 cycles;
  measured two copies on one NoC: 149, RESULTS §14). Every design whose DRISC only forwards reaches it. A router
  moves four 4 KiB transfers per packet here (the push, the local delivery, two forwards) through three NoC ports
  (the Ethernet tile's, the DRISC's two), so at 4 KiB some port carries two whatever the split; at 2 KiB the cap is
  ~76. 153 is 36 GB/s, against 19 GB/s today.
- **P5 vs P5L (local delivery on the DRISC or on E1).** On the 3-hop router (no local delivery) they are the same
  throughput; P5 keeps E1 at 77 (it polls, pushes, relays credits and retires slots) and P5L adds the decode (96).
  On multicast P5's DRISC does decode, admission, local delivery and both forwards (248 static, 222 event-driven);
  moving the decode and local delivery to E1 (P5L) runs them in parallel with the DRISC's forwards and reaches the
  port cap (153), at the cost of E1 going from 102 to 178 in the static model. That static figure relays one credit
  per relay (25 cycles per output per packet); the event-driven model relays whatever has arrived and is not limited
  by E1.
- **Sensitivity** (static, `pipeline_study.py sensitivity`): on the 3-hop router P5 holds the wire (116) up to a
  45-cycle DRISC forward copy and loses it at 60 (124), with stateless NoC calls (126) or a 20% slower DRISC (131);
  P5L and P3 hold 116 in all of these.
- **One DRISC cannot serve two routers at 4 KiB** (`pipeline_study.py sharing`): two 3-hop routers on one DRISC
  give 205 (P5) / 167 (P5L) event-driven. The E/W routers need their own DRISCs, i.e. the NoC0-only ones (8 of 16),
  which send forwards and local deliveries on NoC0.
- **Moving E0's sender-channel scan to E1** (static, `pipeline_study.py scan`) does not pay. With batching E0 is at
  77-92 and the move saves it 3-18 cycles while costing E1 16-21. Without batching (bursty traffic) both ERISCs are
  over budget (P5, 3-hop: E0 168-217, E1 148-190) and moving the whole scan only swaps the bottleneck (E1 208-265);
  E0's send alone is 133 there (`results/pipeline_study_p5_scan_nobatch.txt`).
- **1D linear multicast** (`validate_1d.py`; `results/validate_1d.txt`). The benchmark multicasts all-to-all along
  lines of 4 chips, so routers differ: an end chip receives 3 flows and sends 1 (local delivery only), its neighbour
  sends 3 and receives 1, a middle link carries 2 each way (local + forward). Each router is a scenario with its own
  send and receive rate (`tx_rate`, `rx_rate`, from the phase profile's packet counts); the slowest sets the interval.
  - Validation (no telemetry, 1/2/4 KiB): bidirectional write / scatter 175 / 226 measured vs 167 / 202; unidirectional
    241 / 291 vs 228 / 285. Mean |error| 5.6%, max 10.8%, all low: the model underestimates 1D receive work by
    ~5-25 cycles per packet. It names the critical ERISC the delay injection does. With the spin's own cost (N + 12;
    on the sender it overlaps the TX-queue wait) receive and send delays match within ~7 cycles, except receive
    delays on the bidirectional write (+20 / +50 vs +42 / +71), where the model leaves the receiver slack it lacks.
  - Designs at 4 KiB on that traffic (static, lean poll + batch x4): bidirectional writes reach the wire limit even
    without a DRISC (the busiest link carries 3 flows; the routers' work is spread). Unidirectional writes, where a
    middle router receives and forwards as much as it sends: 175 today, 152-153 for P3, P5 and P5L (a port with two
    copies, or a DRISC doing everything), 119 for P3s (E1 decodes, admits and pushes; the DRISC delivers locally on
    one NoC and forwards on the other). Scatter: 124-176, limited by scatter delivery on one core. With the model
    ~5% low on 1D, expect P3s on unidirectional writes nearer ~125.
  - The symmetric scenarios `1d_linear_mcast` / `_scatter` (a middle router in a long line, sending and receiving at
    one rate) give the same picture: P3s 118 static / 133 event-driven, others 146-153.
- **The duplicate decode costs little throughput.** Retries happen while a downstream is full, so the router is
  waiting anyway: under a congested downstream (drain x1.5, or 1500-cycle stalls every 4000) the other direction's
  throughput is unchanged even with E1 carrying transmit work, with or without keeping the decode (0.5-1.7 retries
  per packet). Keeping the decoded action for the head packet (P0) is still free to do.
- **Any DRISC hop lengthens the downstream credit loop** by the push: with 16-slot downstream channels the 3-hop
  router reaches 122-125 instead of 117 (P4d, whose DRISC holds the downstream credits itself, still 117).

## Limitations

- The keepalive floor is in the static model only; the event-driven model assumes the keepalive does not bind (true
  at 1000 cycles). Static output combines the floor and the core limits with a max(), which is low where they are
  close (see Validation).
- The event-driven model has the trimmed 1-hop E/W write getting faster from 2 to 4 KiB (176 -> 159); the hardware is
  flat (168-170). Not investigated.

- DRISC handoff costs and DRISC speed (alpha 1.0) are measured primitives (RESULTS §14) assuming lean stateful issue,
  not measurements of a router implementation; the DRISC forward copy includes 14 estimated cycles of header work.
  ERISC<->ERISC handoff costs are still estimates. `pipeline_study.py sensitivity` and `alpha` show how much depends
  on them.
- 1D receive work is underestimated by ~5-25 cycles per packet (`validate_1d.py`); the piece costs are 2D-calibrated.
- The split of the 27-cycle idle-channel poll into packet, completion and connection checks is estimated; only the
  total is measured (trimming ablation).
- Local delivery on a DRISC is charged like the ERISC's measured local delivery (46 cycles, stock NoC calls).
- E0's work moved to E1 (sender-channel scan) is modelled in the static model only.
- A shared DRISC is modelled by doubling its time and NoC traffic (symmetric traffic on both routers); DRISC NoC
  ingress is not serialized in the event-driven model.
- The event-driven model feeds one active sender channel per direction; the multicast heavy-link scenario is
  static-only (it uses the measured 1.4 packets per iteration).
- One link only: no multi-hop chains, no NoC contention beyond per-port serialization, VC0 only, single worker per
  direction with an estimated 40-cycle per-packet cost.
- `k_drain` assumes a deep queue at the bottleneck; the event-driven model shows when that fails (few slots).
- NoC ports assume copies spread evenly over the ports in use; Ethernet-tile L1 bandwidth and NoC ingress are not
  modelled. A piece runs on one core: splitting scatter or multicast copies across cores is not modelled yet.
- One receiver channel per router (no VC1), so a retrying core's effect on other channels is not modelled. A
  downstream is a drain at its link rate plus a fixed slot hold (2 x link latency + 200 cycles), not a full router.
- The split of today's measured decode+admit and forward costs into decode, credit check (12/output), reservation
  (6/output) and data movement (77/copy) is derived or estimated, as are the request, grant and relay costs.
- `slots_needed` in the static output is a rough estimate; confirm with `--verify`.

## Sender channels on E1 (`tx_split_study.py`)

Question: once receive work moves to a DRISC, can E1 take some of E0's sender channels (sending them on TXQ1) and
balance the two cores? 4 KiB, wire 118.5 one way and 122.6 with acks both ways (96 B header, 152 B credits; the results files were made
with 84 B credits, 120.7). Results in
`results/tx_split_study*.txt`.

- **What the DRISC frees on E1 (no batching):** local write 134 -> 123 (11), N/S 149 -> 123, scatter 191 -> 140,
  forward 185 -> 148, multicast 352 -> 217. E1's floor after the offload is ~123: loop + polls 62 (including the
  DRISC's slot-credit doorbell), push 32, ack 21, credit read 8. For plain writes the handoff costs nearly what the
  NoC write did.
- **As the router is today, moving channels to E1 loses everywhere.** Each channel moved saves E0 ~33 but costs E1
  ~64: E1 iterates once per received packet and visits its sender channel every iteration (grouped read 7 +
  completion read 20, at 0.25 packets per visit with 4 busy channels), plus the second ring (10) and the packet work
  (0.25 x 110). Both ways, 4 busy channels: local write E/W 136 (E0) -> 187 (E1) with one channel moved.
- **It needs E1's checks to be nearly free.** With E1's sender checks at ~4 per iteration (one read grouped with its
  ring read) and E1's receive loop + polls cut by 30: one way, one channel on E1 takes 136 to the wire (118 E/W, 123
  N/S), with or without a DRISC; both ways only the local write with a DRISC gains (136 -> 124), and forwarding and
  multicast get worse.
- **With batching by 4 (sustained streams) there is nothing to balance:** writes and forwards reach the wire with
  E0 owning every channel; scatter (124) and multicast (159) are set by the DRISC.
- Biggest unknown: E1's receive loop + polls (the catalog's 25 + 15 + 10, measured only as a 53-cycle residual). It
  is the largest part of E1's floor after the offload and has not been broken down the way E0's polling was.
- Not modelled: slot depth (two receive rings split the slots), and the cost of an L1-counter pointer update on TXQ1
  (assumed equal to the stream-register one).
- **What E1's "loop + polls" is** (compiled baseline E1, `fabric_exp/disasm/e1_untrimmed.dis`): an idle iteration
  is 14 instructions, ~27 cycles: the packets-received stream register read used at once (~9), the NoC
  write-flushed status read for the oldest in-flight slot plus compare (~16, read every iteration even with nothing
  pending), and the loop counter (2). A packet iteration adds slot and counter updates (4) and the TX-queue busy check
  before the completion ack (~12); the outer loop every 32 iterations adds ~1. About 44 per packet against the 53
  residual; the catalog's split (loop 25, polls 15 + 10) is an estimate fitted to that residual.
- **It is per iteration, and E1 handles one packet per iteration**, so when E1 is the limit it is paid per packet
  however fast E0 delivers. Handling every queued packet in one iteration (the stream register is a count) and acking
  per batch divides it: with the receiving core at 4 per iteration and acks per 4, sending core unchanged
  (`--k 1 --k-rx 4 --b-ack 4`, `results/tx_split_study_rxdrain4_ack4.txt`), both ways with 4 busy channels: E1 81
  (local write E/W, no DRISC) / 69 (DRISC), E0 117, interval 119 (wire). E1 then has the slack, and one channel on
  E1 balances the cores (E0 89, E1 84), but the link is already at the wire, so it adds margin on E0, not throughput.
