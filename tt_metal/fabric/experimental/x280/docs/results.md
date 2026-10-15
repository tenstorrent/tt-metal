# X280 characterization results

Consolidated results of the six benchmark families. Each section gives a summary table, the
conditions it was measured under, and the archived record each row comes from. The records are kept in the project's results archive,
not in this tree, and are cited by name. How each quantity is defined, which counter it is read
on and when a tick may be converted to time is in [measurement.md](measurement.md); each
family's design is in [benchmarks/](benchmarks/design.md), one document per family.

## Conditions common to every section

| | |
|---|---|
| Hardware | Blackhole p150b, card 0 (eth_credit: cards 0 and 1, Ethernet-linked) |
| X280 | L2CPU 0 (noc0 (8, 3)) unless a row names another; PLL 1750 MHz, set by the fixture and confirmed against ARC telemetry |
| AICLK | 1350 MHz, from each record's validated AICLK evidence |
| LIM shape | 15 ways, except packet_injection and the X280 pipeline_stage cases at 14 ways (`WAYENABLE=1`) |
| Source | the pre-port tree, final characterization pass of 2026-10-02 unless a section says otherwise |
| X280 compiler | riscv-collab gcc 15.2 (`-march=rv64gc -mabi=lp64d`) for every X280 figure below, all recorded before the tree moved into tt-metal. This tree builds the images with SFPI; on the same source SFPI reads p2p TURN 2 cycles higher (one 4-byte `ld` in the timed window where gcc emits a 2-byte `c.ld`) and matches transport bandwidth within 0.01%. A figure measured from this tree is labelled SFPI |
| Metal kernels | built by the tt-metal JIT of the time |
| Dispatch | slow dispatch for every executable launching on an idle ERISC or DRISC |
| Watcher | off |

Every figure is a raw counter reading or derived offline from one: X280 wall time is CLINT
`mtime` (50 MHz, 20 ns per tick), X280 CPU time is `rdcycle`, and a Metal core's wall time is
its own timer, converted at the record's validated AICLK only when that evidence is clean.
Medians and ranges are over the runs named in each section, not over samples inside a run.

The 2026-10-02 pass ran each configuration three times (six for the Metal-peer p2p and the
packet_injection latency configurations), interleaved in the same boot with the tree before
the benchmark de-duplication, and found every configuration within 5% of it except where a
section says otherwise. Its records are archived as `L9b-final/` (`comparison.csv` holds the
per-configuration comparison; `INDEX.md` traces each value to its attempt).

## p2p_latency

Two-phase ping-pong of a 4-byte token, `TOKEN_ONLY`, `TIGHT` poll, `WAIT_FOR_COMPLETION`.
`N=64` samples per phase; `W=8` on the X280 members, calibrated to 25 ms on the Metal-peer
members. `x280_peer_tile` pairs L2CPUs 0 and 1 (6/6 hops). Each cell is the range of per-record
p50s over every record of that participant and kind, both launch orders pooled.

| Member | Participant | Kind | Records | Wall p50, ticks (ns) | CPU p50, cycles |
|---|---|---|---:|---|---|
| `x280_same_tile` | A, B | RTT | 12 | 7 (140) | 248-250 |
| `x280_same_tile` | A, B | TURN | 12 | 2 (40) | 74-76 |
| `x280_peer_tile` | A, B | RTT | 12 | 13-17 (260-340) | 470-474 (10 records), 586 (2) |
| `x280_peer_tile` | A, B | TURN | 12 | 8-9 (160-180) | 294-296 |
| `tensix_peer_tile` | X280 | RTT | 12 | 18-21 (360-420) | 640-644 (9), 750 (3) |
| `tensix_peer_tile` | X280 | TURN | 12 | 13 (260) | 458-462 |
| `tensix_peer_tile` | Tensix | RTT | 12 | 553 (409.630) | 553 |
| `tensix_peer_tile` | Tensix | TURN | 12 | 413 (305.926) | 413 |
| `erisc_peer_tile` | X280 | RTT | 12 | 19-21 (380-420) | 644 (1), 754 (11) |
| `erisc_peer_tile` | X280 | TURN | 12 | 13 (260) | 460-464 |
| `erisc_peer_tile` | ERISC | RTT | 12 | 552-553 (408.889-409.630) | 552-553 |
| `erisc_peer_tile` | ERISC | TURN | 12 | 422 (312.593) | 422 |
| `drisc_peer_tile` | X280 | RTT | 12 | 18-19 (360-380) | 638-642 |
| `drisc_peer_tile` | X280 | TURN | 12 | 13 (260) | 460-464 |
| `drisc_peer_tile` | DRISC | RTT | 12 | 543-547 (402.222-405.185) | 544-548 |
| `drisc_peer_tile` | DRISC | TURN | 12 | 412 (305.185) | 412 |

- X280-initiated round trips to a Metal peer or to another X280 tile land at one of two levels
  116 cycles apart, and a record's p50 can fall in either; the cause is open.
- The `x280_peer_tile` placement sweep (2026-09, one run per pair of L2CPUs 0, 1 and 2, hop
  splits 6/6, 2/10 and 8/4) gave RTT p50 453-457 cycles and TURN p50 276 cycles on every pair:
  each non-posted store crosses one full 12-hop lap whatever the placement. Taken before the
  paired-read order change, so its cycles are not comparable with the table above. Its JSONL
  was not retained; the fields are transcribed in the archived pre-port results document,
  Appendix A.6.
- A posted-mode (ordering mode 3) member run earlier on the same tile collapses every later
  responder TURN on that tile to issue cost (measured: 160 ns before, 42 ns after). Every
  figure here was taken on a tile no mode-3 member had run on since reset.
- With SFPI-built images, X280 TURN reads 2 cycles higher (codegen, confirmed by a code-placement
  pad probe); RTT and every wall-tick figure are unchanged within spread.

Records: `L9b-final/campaign-tip/attempts/{x280_same_tile,x280_peer_tile,tensix_peer_tile,erisc_peer_tile,drisc_peer_tile}.*`.

## transport_bandwidth

`N=16384` issues per batch. X280 -> Tensix: `W=128`, X280 hart 0 issuing u32 posted stores
through one paired-range TLB window in ordering mode 2. Tensix -> X280: warm-up calibrated to
25 ms (1,388,309-1,389,529 issues), `noc_async_write` from Tensix core (0, 0) into L2CPU 0 LIM.
Each cell is the range over three runs.

| Member | Size | Boundary | ns/item | CPU/item (cycles) | GB/s |
|---|---:|---|---:|---:|---:|
| X280 -> Tensix | 64 B | RETIRED | 256.018 | 448.031-448.032 | 0.250 |
| X280 -> Tensix | 64 B | INJECTION | 256.003-256.005 | 448.007 | 0.250 |
| X280 -> Tensix | 256 B | RETIRED | 1024.026 | 1792.044 | 0.250 |
| X280 -> Tensix | 256 B | INJECTION | 1024.010 | 1792.016-1792.017 | 0.250 |
| Tensix -> X280 | 64 B | DRAINED | 17.797 | 24.025 | 3.596 |
| Tensix -> X280 | 64 B | INJECTION | 17.782 | 24.006 | 3.599 |
| Tensix -> X280 | 256 B | DRAINED | 17.797 | 24.026 | 14.384 |
| Tensix -> X280 | 256 B | INJECTION | 17.782 | 24.006 | 14.396 |

- The X280 -> Tensix rows measured LIM loads, not stores: the timed loop loaded each store's
  word from a stack array, and the stack is in LIM, which is uncached, so one LIM read (16 ns,
  28 cycles) per u32 store set the size-invariant 0.250 GB/s. The member now holds its 64 B
  source pattern in registers across the timed loop; the same four points measure 1.800 GB/s,
  3.89 cycles per `sw` (see the store-width sweep below). Under mode 2 the drain read does not wait for the writes, so
  RETIRED is an issue rate like INJECTION. The records label RETIRED with the earlier
  `DRAINED` boundary name.
- Tensix -> X280 is bound by Tensix issue (17.8 ns per write) at 1750 MHz. Below about 1.44 GHz
  (256 B) or 0.45 GHz (64 B) the X280 inbound sink binds instead. INJECTION sweep of 2026-09-03,
  `W=128 N=16384`:

| X280 PLL (MHz) | 64 B (GB/s) | 256 B (GB/s) | Bound by |
|---:|---:|---:|---|
| 200 | 1.607 | 2.004 | X280 sink, both sizes |
| 800 | 3.600 | 8.014 | sink at 256 B only |
| 1000 | 3.600 | 10.017 | sink at 256 B only |
| 1750 | 3.600 | 14.399 | Tensix issue, both sizes |

  The sink fits about `2.1 + size/10.9` X280 cycles per write from the two 200 MHz points, and
  predicts the 800 and 1000 MHz 256 B points within 0.003 GB/s. No record of the PLL sweep was
  retained; the figures stand as recorded in the archived pre-port results document, § 4.1.
- X280 -> Tensix publishes `mtime` and `rdcycle` from one bracket, and the firmware measures its
  own PLL (1,749,996-1,750,004 kHz against a 1,750,000 request). The two rates agree to a ratio
  of 1.0000 at both sizes and boundaries, because this loop does not stall the hart; `mtime` is
  the primary because it needs no PLL round-trip to convert. One representative run, archived
  pre-port results document § 4.3.

**Tensix -> X280 size sweep** at X280 PLL 1750 MHz, `N=16384`, warm-up calibrated, median of
three attempts per point (2026-10-08):

| Size (B) | Boundary | ns/issue | GB/s |
|---:|---|---:|---:|
| 64 | DRAINED | 17.80 | 3.596 |
| 64 | INJECTION | 17.78 | 3.599 |
| 128 | DRAINED | 17.80 | 7.192 |
| 128 | INJECTION | 17.78 | 7.198 |
| 256 | DRAINED | 17.80 | 14.384 |
| 256 | INJECTION | 17.78 | 14.397 |
| 320 | DRAINED | 18.31 | 17.481 |
| 320 | INJECTION | 18.27 | 17.518 |
| 384 | DRAINED | 21.96 | 17.484 |
| 384 | INJECTION | 21.92 | 17.516 |
| 512 | DRAINED | 29.28 | 17.488 |
| 512 | INJECTION | 29.23 | 17.514 |
| 1024 | DRAINED | 58.53 | 17.494 |
| 1024 | INJECTION | 58.49 | 17.508 |
| 2048 | DRAINED | 117.05 | 17.497 |
| 2048 | INJECTION | 109.68 | 18.672 |
| 4096 | DRAINED | 234.08 | 17.498 |
| 4096 | INJECTION | 234.01 | 17.504 |
| 8192 | DRAINED | 468.13 | 17.499 |
| 8192 | INJECTION | 468.03 | 17.503 |
| 16384 | DRAINED | 936.25 | 17.500 |
| 16384 | INJECTION | 936.09 | 17.503 |

- Issue-bound at 17.80 ns per issue through 256 B. The knee lies between 256 and 320 B, where the
  two-point sink fit put it (about 316 B).
- From 384 B every size holds 17.48-17.52 GB/s, 10.0 B per X280 cycle at 1750 MHz: about 8%
  below the fit's 19 GB/s plateau (10.9 B per cycle), so the fit's per-byte term overstates the
  sink at 1750 MHz.
- Some attempts read the sink's upper level, 18.67 GB/s, instead of 17.50: 2048 B INJECTION's
  median does, and the 2026-10-14 whole-suite pass saw it at 4096 B and 320 B DRAINED too, with
  identical firmware. Attributed to the sink; not chased.

Records: `L9b-final/campaign-tip/attempts/bw.*`; the size sweep in
`later-runs/records/tx2x-size-sweep-20261008/` (`campaign-fix/attempts/`, tabulated in
`tabulated/ab.md`); the whole-suite pass in `later-runs/records/full-suite-20261014/`
(`tabulated/size-sweep.md`).

## pipeline_stage

The routing image is emulated: the X280 runs the same instruction sequence over the same table
bytes on any card, so no row describes a live fabric. X280 cases: 14 ways, inside one
pinned-input preparation, `W=128`, `N=16384` plain and 4096 instrumented. Each cell is the
range over three runs.

**Recorded on the pre-growth region.** These X280 figures were taken with the prepared input at
4,160 bytes (routing image 2,576 B, first header at `0x0A000A40`). The region is now 4,288 bytes
(routing image 2,704 B, first header at `0x0A000AC0`), so header addresses moved and the figures
are not comparable with runs of this tree.

| Case | CPU/item (cycles) | ns/item |
|---|---:|---:|
| `route_compute` lookup, plain | 74.018-74.060 | 42.297-42.321 |
| `route_compute` lookup, instrumented | 127.161-127.686 | 72.661-72.964 |
| `route_forward` copy, plain | 168.003-168.010 | 96.001-96.005 |
| `route_forward` copy, instrumented | 247.305-252.038 | 141.318-144.019 |
| `route_forward` decide, plain | 42.008-42.009 | 24.004-24.005 |

- `copy_plain` moves with firmware code layout alone: padding the image so its loop moved,
  instructions unchanged, gave 160 or 168 cycles per item. 168 is the upper layout.
- DRISC and idle-ERISC configurations of the same members were measured in the same pass
  (lookup plain about 84 rv32 cycles per item on both; decide plain about 27). This tree carries
  no Metal-RISC configuration of pipeline_stage, so those figures are not reproducible here.

Records: `L9b-final/campaign-tip/attempts/pipeline.*`; layout probe `copy_plain-layout-probe.txt`.

## core_sync

X280 director releasing a 4 x 4 logical Tensix grid; raw debug-timer ticks of the spread of the
16 worker exit timestamps, median of three runs (min-max). Counter origins across tiles are not
qualified as comparable, so no time is derived.

| Release | Rectangle destinations | Tier-2 writes | Exit spread, ticks |
|---|---:|---:|---|
| null (barrier compiled out, X280 not launched) | 0 | 0 | 132 (131-133) |
| unicast | 0 | 16 | 11,569 (11,562-11,569) |
| multicast | 16 | 0 | 122 (118-129) |

- The null range is a fast-dispatch figure; under slow dispatch it is about 247,000 ticks (the
  launch floor), so compare it only across runs of the same dispatch mode.
- All seven `x280_core_sync_director` behavioural cases passed in the same boot, including the
  renumbered-column case at logical x 7..10 of an 11 x 10 grid and three guarded 2 x 2
  multicasts across translated and harvested columns.

Records: `L9b-final/campaign-tip/attempts/core_sync.release_skew.fixed4x4.*`,
`L9b-final/gtest/tip-s4-director-*`.

## packet_injection

### Tensix -> X280

96-byte headers in 128-byte slots, ring `S=16`, `B=1`, `T=1`, BRISC producers. Throughput
`W=N=1000`; latency `W=100`, `N=3`. Aggregate rate is P x N headers over the X280 `mtime`
interval from the measured-phase release to the final credit; median of the tip runs.

| Configuration | P | C / K | Aggregate interval, mtime ticks | M headers/s |
|---|---:|---|---:|---:|
| throughput | 1 | 1 / 1 | 36,124 | 1.384 |
| throughput | 1 | 8 / 16 | 34,314 | 1.457 |
| throughput, paired BRISC+NCRISC | 2 | 1 / 1 | 71,313 | 1.402 |

| Configuration | One-way latency p50, Tensix timer ticks (median of 6 runs) |
|---|---:|
| latency, timestamp before header construction (`include`) | 3,275 |
| latency, timestamp after header construction (`exclude`) | 2,799 (2,775-2,807) |

- `include` minus `exclude` is the cost of building one header, about 476 ticks (about 350 ns
  at 1350 MHz). The `exclude` p50 is 5% lower than before the de-duplication; attributed to the
  consumer phase being recompiled, with no single commit responsible.
- Per-worker rate falls as 1/P (S=16, B=1, C=1, recorded 2026-09): about 1.42 M requests/s at
  P=1, 719 k at P=2, 361 k at P=4 and 181 k at P=8, with workers in one shape within 0.05% of
  each other and BRISC-only and paired topologies agreeing. These do not establish aggregate
  X280 capacity or attribute the 1/P fall to a mechanism. No archived record carries these rows.
- At C=8, K=16 (S=16, B=1) the aggregate scales with producers through P=2, at about 1.456 M
  headers/s per worker with no credit stall, then plateaus at about 4.92 M headers/s (203 ns per
  header) from P=8; credit stalls appear from P=4 and double with P. At C=K=1 the plateau is
  about 1.50 M headers/s from T=4. Median of three runs, 2026-10-14 whole-suite pass:

| Family | T | P | Aggregate M headers/s | ns/header | Credit-stall episodes |
|---|---:|---:|---:|---:|---:|
| BRISC | 1 | 1 | 1.457 | 686.26 | 0 |
| BRISC | 2 | 2 | 2.911 | 343.49 | 0 |
| BRISC | 4 | 4 | 4.901 | 204.03 | 244 |
| BRISC | 8 | 8 | 4.920 | 203.27 | 494 |
| paired | 1 | 2 | 2.912 | 343.41 | 0 |
| paired | 2 | 4 | 4.784 | 209.03 | 243 |
| paired | 4 | 8 | 4.920 | 203.25 | 494 |
| paired | 8 | 16 | 4.918 | 203.34 | 991 |

- Backpressure at P=1 produces no stall sample by construction; an empty population there is
  not "no stalls".
- Three multicast release shapes (contiguous, across translated columns 8-9, across a
  harvested column) gave 133,493-133,495 mtime ticks for the throughput phase at T=4.

Records: `L9b-final/campaign-tip/attempts/packet.{throughput,latency-include,latency-exclude,backpressure-include,backpressure-exclude}.*`,
`L9b-final/gtest/tip-s7-pi-*`; the C=8, K=16 load sweep in
`later-runs/records/full-suite-20261014/tabulated/pi-c8k16-load.md`, and the T=4 capacity, C/K
isolation and header-ingress capacity sweeps beside it (`pi-t4-capacity.md`, `pi-t4-ck.md`,
`header-ingress-capacity.md`).

### ERISC -> X280 header ingress

Inactive idle ERISCs (card 0 has 8, so `writers_all` is `writers8`) publish 96-byte headers
into one X280 consumer, `S=16`, `N=1024`, warm-up calibrated. M headers/s = P x 1024 headers
over the aggregate `mtime` interval, median of three runs.

| Point | P | B | C=1, K=1 | C=8, K=16 |
|---|---:|---:|---:|---:|
| `writers1` | 1 | 1 | 1.384 | 2.549 |
| `writers2` | 2 | 1 | 1.404 | 4.525 |
| `writers4` | 4 | 1 | 1.413 | 4.802 |
| `writers8` / `writers_all` | 8 | 1 | 1.417 | 4.807 |
| `writers1` | 1 | 4 | | 4.443 |
| `writers1` | 1 | 8 | | 4.449 |
| `writers4` | 4 | 4 | | 4.810 |
| `writers4` | 4 | 8 | | 4.816 |

- At C=1, K=1 every point runs at 1.38-1.42 M headers/s with almost every header entering the
  credit wait and X280 finding the ring empty at most three times per run: X280 service bounds
  it. Its cost falls with credit writes per header, about 570 X280 cycles per credit (a linear
  fit over three settings, not a direct measurement).
- At C=8, K=16 four or more writers plateau at about 4.8 M headers/s, still X280-bound. A lone
  writer at B=1 is paced by its own per-header publication (B=4 lifts it to about 4.4).
- `writers_all` at N=4096 matches N=1024 within 0.1%, so the release is amortized.
- Credit-protocol cost against the single-writer prototype (2026-09-30): about 553 ERISC wall
  ticks per header at C=1, K=1.
- Figures are for inactive idle ERISCs; an ERISC doing live Ethernet receive and router work is
  not shown to have the same cycles left.

Records: `L9b-final/campaign-tip/attempts/packet.erisc_header.*`; the 2026-09-30 scaling
campaign and prototype comparison in `erisc-ingress-scaling/{silicon,campaign-curve3}/` and
`benchmarks/campaigns/20260929-erisc-header-ingress-scaling/`.

## eth_credit

Read/write-pointer credit flow control across one live Ethernet link between cards 0 and 1:
X280 forwarders on L2CPU 0 of each chip, E0/E1 on the linked active Ethernet core.
`throughput_sweep`, chip 0 -> chip 1, `N_LINK=8`, `N_DELIVER=8`, 20,000 packets per forwarder;
rates on the receiving forwarder's `mtime` window. One run of three shown; the three agree
within 0.5%.

| P | Slot (B) | Payload (B) | BATCH | Mpkt/s per forwarder | Slot GB/s per forwarder |
|---:|---:|---:|---:|---:|---:|
| 1 | 160 | 64 | 1 | 0.296 | 0.047 |
| 1 | 160 | 64 | 4 | 0.353 | 0.056 |
| 1 | 1120 | 1024 | 1 | 0.296 | 0.332 |
| 1 | 1120 | 1024 | 4 | 0.353 | 0.395 |
| 1 | 4192 | 4096 | 1 | 0.296 | 1.242 |
| 1 | 4192 | 4096 | 4 | 0.353 | 1.479 |
| 1 | 16480 | 16384 | 4 | 0.353 | 5.815 |
| 2 | 4192 | 4096 | 4 | 0.205 | 0.859 |
| 4 | 4192 | 4096 | 4 | 0.178 | 0.745 |

- Packet rate is flat across slot sizes, so the X280 forwarder's write of each header into the
  ERISC (W1, twelve strict 64-bit window stores, about 2.7 µs per header) bounds the path, not
  the link. W1 takes 81% of the sender's window at BATCH=1.
- With more forwarders the per-forwarder rate falls (harts contend on W1); P=4 aggregates to
  about 0.71 Mpkt/s.
- Single-slot ping-pong (`N_LINK=1`, 2026-09-25): 1310 ERISC ticks, about 971 ns, per round trip.

Records: `L9b-final/campaign-tip/attempts/eth_credit.throughput.*`; the 2026-09-25 full-suite
run with the ping-pong figure in `eth-credit-20260925/`.

## Results from earlier measurement paths

These were measured by bare-metal images and Python host scripts that are not part of this
tree, on 2026-04-14, at X280 PLL 1750 MHz with AICLK ramped by ARC message (frequency not
recorded), on a Blackhole PCIe card whose host was not recorded, with images built by the gcc
of the time. No current suite re-measures these paths; where one does (Tensix -> X280 inbound,
transport_bandwidth above), the earlier figure is dropped. Cycle counts are `rdcycle` at
1750 MHz.

| Path | Direction | Target | Peak GB/s | Mechanism |
|---|---|---|---:|---|
| System Port vector write | X280 -> NoC | Tensix L1 | 14.4 | 64 B vector stores through a TLB window |
| System Port vector write | X280 -> NoC | DRAM | 3.96 | same |
| System Port scalar write | X280 -> NoC | Tensix L1 | 3.60 | 8 B scalar stores |
| System Port scalar write | X280 -> NoC | DRAM | 0.95 | same |
| DMA memory-to-memory | X280 -> NoC | Tensix or DRAM | 2.34 | AHB DMA engine, 1 MB blocks |
| Memory Port cached vector read | DDR -> X280 | DDR via L3 | 7.92 | 128 KB, cache-miss path |
| System Port read | NoC -> X280 | NoC target | 0.15 | non-posted round trip |

- Scatter to 32 Tensix cores through pre-configured windows held 14.4 GB/s aggregate;
  switching between 128 KB windows cost nothing.
- Four harts writing concurrently shared one System Port and fell to 0.83 GB/s per hart.
- Concurrent DMA and System Port writes reached 4.50 GB/s combined at 128 KB; the 1 MB
  concurrent case hung.

Record: `benchmarks/COMPLETE_BANDWIDTH_RESULTS.md` (the full tables).

## Whole-suite pass of 2026-10-14

A later pass ran all 140 manifest configurations three times each (four for backpressure, six
for the Metal-peer p2p and packet_injection latency configurations), 446 attempts, all passed,
on the same source as the 2026-10-02 pass apart from the five C=8, K=16 load-sweep registrations
above. Every configuration matched that pass within 1% except the X280 sink levels noted under
transport_bandwidth, a raw core_sync multicast range diagnostic (+18%), and a 4-cycle shift of
one `x280_same_tile` RTT median. Record: `later-runs/records/full-suite-20261014/`
(its comparison against the 2026-10-08 pass).

## Not consolidated

- Multi-hart DDR scaling of an earlier migration-worker image: archived as
  `benchmarks/MULTIHART_SCALING.md`; the image is not in this tree.
- The 2026-09-23 measurement-unification campaign (40 configurations, 79 records): superseded by
  the 2026-10-02 pass; archived as `benchmarks/campaigns/20260923-measurement-unification/`
  (its raw attempt directory is gone; the values survive in `master_results.jsonl`).
- Cross-hart atomics: the micro-benchmark's output format is documented with its image; no
  result was recorded for consolidation.
