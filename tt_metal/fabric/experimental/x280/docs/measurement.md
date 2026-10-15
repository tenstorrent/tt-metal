# Characterization measurements

The maintained P2P, transport-bandwidth, pipeline-stage, packet-injection,
core-sync and eth_credit suites retain native measurements. Legacy experiments are excluded.
`characterization_manifest.json` selects 160 configurations: 28 of the 40 of the
2026-09-23 campaign (39 evidenced prior configurations and the explicitly
authorized first fixed-4×4 core-sync run, less its ten DRISC and idle-ERISC
`pipeline_stage` configurations and its two X280 `route_forward` COPY ones), the nine
`packet_injection/erisc_to_x280_header` points at C=1, K=1 (writers 1, 2, 4, 8
and all at B=1; B=4 and B=8 at writers 1 and 4), twelve points of its
consumer-settings sweep (writers 1 and all at B=1 for six further (C, K)
settings), eight more at the selected C=8, K=16 (the rest of that writer curve
and B set, and N=4096 at writers all), the Tensix `brisc.T1` throughput
point at C=8, K=16, 63 sweep points, all measured in the 2026-10-14 whole-suite campaign (the six
`packet.throughput.{brisc,paired}.T{2,4,8}` load points, 12
`packet.throughput.{brisc,paired}.T4.{c4k8,c8k16,s4,s8,s64,b4}` capacity points
at the T=4 load knee, each changing one of C/K, S or B from its T4 point, 16
`packet.throughput.{brisc,paired}.T4.s64.c<C>k<K>` C/K isolation points that
change only C and K from `T4.s64` (C in {1, 2, 4, 8, 16} at K=16 and K in
{2, 4, 8} at C=1), five `packet.throughput.paired.T1.c8k16` and
`packet.throughput.{brisc,paired}.T{2,8}.c8k16` load points at C=8, K=16, and 24
`packet.erisc_header.s<S>.c<C>k<K>.*` capacity points at S=32 and 64: writers 1
and all at B=1 and writers 1 at B=4 for (C, K) = (8, 16), (8, 32), (16, 32) and
(32, 32)), and `eth_credit.throughput`, eth_credit's slot, batch and P
sweep across the Ethernet link of cards 0 and 1 (`TT_VISIBLE_DEVICES` unset, so every card is
visible), and
the 18 `bw.tensix_to_x280` points that extend its transfer-size axis to 128, 320,
384, 512, 1024, 2048, 4096, 8192 and 16384 B under both boundaries, which no
campaign has yet measured, and the 20 `bw.x280_to_tensix` points that add 4096
and 65536 B transfers and the `sd` (8 B) and `vse32.v` (64 B) store widths
under both boundaries. An ERISC point's record count comes from
its gtest filter, or for `writers_all` from its run record. An
ERISC point the device cannot host -- more writers than inactive Ethernet cores,
none at all, or `IDLE_ETH` not enabled -- is an availability skip, and like every
skip it fails coverage.
Warmup/sample counts come from those configurations, except that one omitting
an AICLK-gated suite's warm-up leaves it to calibration (below). The 2026-09-23 campaign
collected one run each, with separately retained diagnostic attempts; the
2026-09-30 ERISC header-ingress campaigns collected three runs of each ERISC
point and one of each Tensix cross-check.

| Interval | Wall measurement | Native CPU measurement |
|---|---|---|
| X280 P2P | Per-sample fenced mtime | Per-sample fenced rdcycle |
| Metal P2P | Per-sample target wall timer | Per-sample cycle/cycleh |
| Transport and pipeline batches | Initiator's wall timer | Initiator's architectural cycle counter |
| Pipeline instrumentation | Original sample counter retained; batch is independently bracketed | X280 samples (`cpu_samples`) use rdcycle |
| Packet throughput | Producer release-to-final-credit interval on its wall counter: the Tensix live timer (`tensix_to_x280`) or the ERISC wall clock (`erisc_to_x280_header`) | Same producer's cycle CSR interval |
| Packet credit stall / stall-triggered RTT | Producer-local live timer | Same producer's cycle CSR |
| Packet X280 aggregate | X280 mtime from the measured-phase release issue (`start_event` `release_issued`: the release is posted, so delivery can follow the start read) to the final credit flush | X280 hart-0 rdcycle over the same bracket |
| Packet one-way latency | Producer-local start and remote read of the same tile timer | Unavailable: endpoints have different CPU counters |
| Core-sync release | Raw per-worker timestamp diagnostic | Unavailable: spread has no local CPU duration |
| Ethernet credit throughput | The receiving forwarder's X280 mtime from its stamp after its first batch to its last stamp; the window covers `packets` − `first_batch_packets` packets | Unavailable: the forwarder records no CPU interval |

Paired reads are ordered, not simultaneous. Every paired bracket reads wall then
CPU, except the packet X280 aggregate, which nests them: mtime then rdcycle at
the start and rdcycle then mtime at the end, so its CPU interval lies inside its
wall interval. Reader cost is included;
the records are instrumented intervals, without overhead subtraction. An
independent per-target perturbation/overhead characterization is not yet present.
Counter instances belong to their recorded participant/placement and run; no
cross-instance CPU subtraction or cross-reset continuity is assumed.

The core_sync probe reads its `tensix_live_timer` exit stamp with
`get_timestamp()`: `RISCV_DEBUG_REG_WALL_CLOCK_L`, then the high half that read
latched (`RISCV_DEBUG_REG_WALL_CLOCK_H`). The debug timestamper documents that
form for one agent reading the low half at a time
(`WormholeB0/TensixTile/DebugTimestamper.md` "The 64-bit counter", which the
Blackhole tree shares). The other members' recorded Tensix live-timer values
use the high/low/high form it documents for concurrent readers
(`kernels/shared/tensix_live_timer.h`).

`include/rt/measurement.h` supplies checked modulo-64 subtraction and the bounded
RV32 high/low/high cycle reader. Every delta the P2P, transport-bandwidth,
pipeline-stage, packet-injection and eth_credit devices record goes through that
subtraction, or, for eth_credit's per-site stall totals, its half-range rule
inline (`ec_stall_add`, which sits in the forwarder's timed window); those totals hold `EC_STALL_TICKS_INVALID`
(UINT64_MAX) once an interval is refused or the total reaches half-range, which
fails the case. Callers require one counter instance/generation
and an independently bounded duration below half-range. A zero population delta
is below resolution; a zero throughput denominator fails. P2P and packet local
sample deltas use checked 32-bit storage after native 64-bit reads. Values above
UINT32_MAX fail explicitly. This preserves the P2P result size and avoids
doubling packet sample buffers; it limits representable individual samples to
UINT32_MAX ticks. Packet start stamps remain 64-bit. The packet producer
result stays inside its 128-byte slot.

Clock validation belongs to `harness/measurement`; JSONL serialization belongs
to `harness/output`; conversion and every computed figure belong to the
offline consumer (see "Record contract"). X280 mtime has the fixed 50 MHz contract and 20 ns
resolution. Native CPU cycles never become nanoseconds. AICLK records obtain
their expected rate from CMFW `AICLK_LIMIT_MAX`; there is no environment,
command-line or compiled override.

Records carry raw samples, never statistics. The tabulator selects every
percentile by the floor-index rule, the sample at index floor(q·(n−1)) of a
record's own sorted raw samples, so percentiles stay integers and no percentile
of percentiles is reported; an empty population has none. Schema-2 producers
emit raw ticks and a self-contained clock object; the offline report derives ns
and rates at `conversion_hz` when the clock evidence validates, and at the
measured rate, marked as a warning, when well-formed evidence fails its checks
("Collection and reproduction").

## Record contract

Records carry raw values only: raw ticks or cycles, counts, byte counts, raw
endpoints and clock evidence (owner decisions 28, 34). Every converted figure
(ns, rate, GB/s, per-item time) and every computed one (mean, per-item value,
percentile, ratio, endpoint difference) comes from the tabulator,
`tools/tabulate_suites.py`, and from nowhere else. The console prints raw
values, status and failure diagnostics, and no computed value (owner decisions
30, 35).

Every suite writes its records through `record()` in
`harness/output/output.hpp`, which builds a record from typed fields:
suite and member, the wall and CPU counters (`measurement::Counter`), a raw
batch, sample populations, throughput intervals, the warm-up selection, clock
evidence and the suite's own fields. No field type holds a floating-point
number. No list of key names enforces the contract: the record functions are
the only writers, and review covers what they write (owner decision 46).
`record()` throws `RecordKeyError`, naming the key, for a key that appears
twice in one object at any depth.

Records may carry the clock evidence's host-clock fields (`host_before_ns`,
`host_after_ns`, `sample_gap_ns`, `elapsed_window_ns`, the Hz bounds,
`conversion_hz`), which are raw host timestamps and measured rates rather than
conversions, and configuration inputs (`mtime_hz`, `erisc_timeout_budget_hz`,
`aiclk_hz`, `x280_pll_mhz`).
`measurement_schema` is 2 when a record carries clock evidence and 1 when it
declares raw timestamps with no conversion offered (core_sync); a record that
names a wall counter must do one or the other. Every record ends with
`tt_metal_slow_dispatch_mode`: `TT_METAL_SLOW_DISPATCH_MODE` as the writing
process saw it, or null when unset. The host test
`tests/host/test_record_round_trip.cpp` renders one manifest configuration's
records per member, and one per packet_injection tensix_to_x280 mode and
family, through that member's own record function and has the live
tabulator validate them (`tests/host/record_round_trip.py`): every file is a
manifest configuration, the configurations
pass `campaign_report` as one campaign with every row's values reported, the
campaign validator, and the dispatch mode on every record. Table mode
and the report must render the configurations as the same result tables, each
with exactly its expected heads and holding cells derived by hand from the
test's inputs.

These are the only converted values the contract admits:

| Exception | Where | Why | Owner decision |
|---|---|---|---|
| Warm-up provenance | `warmup_target_device_ns`, `warmup_pilot_device_ns` on a calibrated record | Warm-up control values converted on the host to choose the warm-up count, not results | 29 |
| Test micro-kernels' PLL frequency | `x280_measure_pll_khz` (`include/rt/x280_pll.h`), called by `src/atomic_bench.c` | Standalone test micro-kernels outside the record path | 31 |
| Device-side control budgets | For example the P2P Metal peers' `HANG_GUARD_TICKS` (`kernels/p2p_latency/metal_peer_target.hpp`), ERISC's from the nominal 1 GHz | Timeout budgets derived from nominal clocks; never recorded as results | 32 |

## Quantities that share a label

Where one record label covers different quantities in different members, this
section names each one. The records carry no field that distinguishes them.

| Suite and label | Member | Quantity |
|---|---|---|
| transport_bandwidth `DRAINED` | `tensix_to_x280` | The stop follows `noc_async_write_barrier()`: every write has been acknowledged, so the rate is a delivered rate. INJECTION points stop before the drain. Nothing reads back the LIM destination, so its content and address are not verified; the path check writes and reads back a separate verify word. |
| transport_bandwidth `RETIRED` | `x280_to_tensix` | The stop follows the stores retiring and one fenced read through the window returning. Under the posted ordering mode the read is not ordered behind the writes, so the rate is injection-side and can end before the last posted writes land. The bias is bounded by the bytes in flight at the stop; it is judged insignificant and has not been measured. The host's L1 readback after the run confirms delivery. Older records label this boundary `DRAINED` and its ids `.drained`; they are stale and are not mapped onto `RETIRED`. |
| transport_bandwidth `transfer_size` | `tensix_to_x280` | One NoC write of `transfer_size` bytes per issue. |
| | `x280_to_tensix` | `transfer_size / store_width` separate `store_width`-byte NoC requests per issue: every store against the uncached window is its own request. |
| p2p_latency `TURN` (the responder's transmit) | `x280_same_tile` | Store issue: a LIM store and `fence ow, ow`, which waits for nothing remote. |
| | `x280_peer_tile`, and the X280 side of `tensix_peer_tile`, `erisc_peer_tile` and `drisc_peer_tile` | Store to BRESP: the window is in ordering mode 1 (Strict AXI), which holds BRESP until the NoC write completes. |
| | The Metal side of those three | `noc_async_write` to its write acknowledgement (`noc_async_write_barrier()`). |
| packet_injection bytes | per-worker `throughput` | `useful_bytes_per_request` 96 (the header) and `slot_bytes_per_request` 128 (the slot); counter writes excluded. |
| | aggregate | P·N·96, headers only, which the tabulator derives over the aggregate's interval. |
| | `erisc_to_x280_header` publication bytes | Headers plus 4 bytes per counter write, derived by the tabulator from the workers' `producer_counter_writes`; `tensix_to_x280` has no such figure. None of the three counts NoC wire bytes. |
| packet_injection header | `erisc_to_x280_header` | Bytes 0..95 are a per-writer nonce pattern, not a routable packet header. |
| eth_credit bytes | `receive_window` | `useful_bytes_per_request` 96, the header, the bytes that reach the X280; `slot_bytes_per_request` the slot that crossed the link; `payload_bytes_per_request`, a record field, the slot's remainder, which stops at E1. |
| | `tensix_to_x280` | A zero-payload unicast fabric header. |

Delivery checks differ between the two packet_injection members. `tensix_to_x280`
counts the packets X280 consumed and checks no content: the consumer loads four
header bytes and discards them. `erisc_to_x280_header` checks every resident
slot's bytes and padding after the run.

The result ABIs' primary delta field does not name its counter. TB and PS
`raw_batch_delta` holds wall ticks in every member. P2P `raw_deltas` holds CPU
cycles (`rdcycle`) in the X280 members and wall ticks in the Metal members, with
the other counter in `secondary_deltas`. The records name both counters.

## Clock evidence and limits

In the Metal these suites were written against, `syncDeviceHost` owns profiler state and a BRISC
sync kernel with 249 exchanges; realtime `run_sync` owns a reserved Tensix core,
returns a processed mapping and has AICLK fallback paths. Neither exposes a
bounded raw sampler for each actual DRISC/ERISC counter under these launch modes.
Reusing those complete mechanisms would change resource/launch ownership and
still leave cross-target rate assumptions.

The small sampler in `harness/execution/execution.hpp` reuses Metal's register
transport and existing Tensix live-timer addresses. It takes two bounded H/L/H
observations 50 ms apart immediately before the workload and two immediately
after it. The inner observations bracket the workload; their rate bounds check
the whole elapsed interval, while the outer observations establish the local
start and end rates. Every observation carries steady-clock before/after
timestamps.

CMFW samples retain `AICLK_LIMIT_MAX`, live `AICLK`, `AICLK_ARB_MAX`,
`THERM_TRIP_COUNT`, and `UPDATE_TELEM_SPEED`. Conversion uses the programmed
maximum only when start, end and elapsed rate bounds stay within 10000 ppm,
endpoint bound width is at most 2500 ppm, elapsed bound width is at most
5000 ppm, endpoint rates differ by no more than 10000 ppm, live AICLK and the
arbiter clamp remain within 10000 ppm of the programmed maximum, and the
thermal-trip count is unchanged. This establishes no throttle evidence
observed; CMFW does
not provide a complete event history for every transient clamp.

The elapsed bound's width is about the two inner host brackets (~40 us each run,
60 us worst observed) divided by the elapsed interval, so a short workload fails
it however stable the clock. `calibrate_warmup()` in `harness/measurement`
sizes an unrecorded warm-up from a pilot run's device-counter time per
iteration: the smallest `W` at or above the suite default for which `W + N`
iterations reach 25 ms of device work (`kAiclkTargetDeviceWorkNs`, under
2500 ppm at the worst observed brackets), or a typed refusal naming the bound
the suite cannot meet -- its warm-up count limit or its host wait. Records
carry `warmup_source` (`calibrated` with the pilot figures, `explicit` for a
count used as given, or `none` for core_sync and eth_credit, which have no
warm-up) and the clock's `elapsed_window_ns`, from the end of the start bracket
to the start of the end bracket. `record()` refuses a record with a clock and no
warm-up source. The campaign validator requires each member's source:
`calibrated` from a calibrating member whose configuration leaves its warm-up
variable unset, otherwise `explicit` or `none` (`WARMUP_SOURCE` in
`tools/tabulate_suites.py`).

The p2p `tensix_peer_tile`, `erisc_peer_tile` and `drisc_peer_tile` members
and transport `tensix_to_x280` calibrate this way whenever their warm-up variable is unset;
a set value is used as given and the gate judges it, as does `packet_injection`
`erisc_to_x280_header`, which is throughput-only and pilots writer 0's measured
interval. `packet_injection` `tensix_to_x280` keeps its fixed warm-up: its producer stamps the measured phase
only in throughput mode, so the latency and backpressure modes give a pilot no
device time. With a warm JIT cache their elapsed intervals measured 17-34 ms
(throughput 89 ms), inside the gate but under the calibration target.
This is deliberate: at the worst observed bracket sum (60 us) its shortest
window, ~17 ms, reads ~3500 ppm, inside the 5000 ppm gate. Calibrating it needs
the producer (`kernels/packet_injection/producer.hpp`, the reads into
`throughput_start_` and `throughput_end_`) to
take its measured-phase boundary timer and CPU reads in every mode, not only
`PI_MODE_THROUGHPUT`, with the host still deriving a rate only in throughput
mode; `throughput_start_`/`throughput_end_` would then be named for the phase.

The host applies this observed AICLK-rate relationship to Tensix
`get_timestamp()`, DRISC `get_timestamp()`, and idle-ERISC
`eth_read_wall_clock()`. Every ERISC member -- p2p `erisc_peer_tile`, packet `erisc_to_x280_header` and
`eth_credit` -- samples its ERISC's own wall clock: on an Ethernet tile the
Tensix live-timer addresses are `ETH_RISC_WALL_CLOCK_0/_1` (`tt_eth_ss_regs.h`),
so the same sampler brackets the counter the kernels read, in another read form:
the host reads the high word `_1` live, high/low/high, while the kernels'
`eth_read_wall_clock()` reads the low word and then `_1_AT`, the high word that
read latched. That `_1` is live is inferred from the timestamper layout and the
header's naming, which the Ethernet ISA docs do not document, and a host
low-word read re-latches `_1_AT`, so the host samples only while none of the
kernels runs, though base firmware's own `eth_read_wall_clock()` can still race
it (`read_live_timer_sample` in `harness/execution/execution.hpp`). The host has no
sampler for a DRISC's counter, so DRISC evidence samples logical Tensix (0,0)'s
live timer. Every AICLK evidence names what it sampled: `reference_counter`, and
`reference_core`, that core's translated (x, y), so a DRISC record shows a
Tensix counter and core beside its own wall counter. Whether the two are one
counter is open, since the ISA docs document the debug wall clock only in the
Tensix tile (`BlackholeA0/TensixTile/TileControlDebugStatus.md`). The campaign
validator requires an AICLK record's reference counter to be its wall counter,
or `tensix_live_timer` for `risc_debug_wall_clock`, and `record()` refuses to
write one that is not. The relationship does not establish shared origins, behavior across
reset, or an architectural counter contract absent from the specifications.

Specification basis: tt-isa-documentation revision `860a84ab`, Blackhole
`TensixTile/DebugTimestamper.md` imports Wormhole's **The 64-bit counter** and
**Memory map** sections. They document tile-local/NoC visibility and concurrent
high/low/high reads. They do not import that block into DRAM/Ethernet tiles.
Observed limits: DRISC CSR access and register-loop progression are observed on silicon;
DRISC counter stall, gating and reset behaviour is not established.

## Unconfirmed hardware assumptions

The results rely on the Blackhole behaviours below, which no document read
confirms; the code relying on each names it. Items keep fixed numbers, which the code
cites; numbers absent from the list are not assumptions this tree relies on. Items marked as owner assumptions or decisions are taken as given
(owner, 2026-10-06) on the evidence named, items marked closed need no assumption, and
the others are open.

- Item 1: a TT-link MMIO write lands only after an earlier TT-link L1 write on
  the same TX queue has committed (eth_credit's `pkts[p]` increment after the
  slot). Wormhole `EthernetTile/EthernetTxRx.md` "RX Queue Memory Ordering"
  orders no packets of different kinds; Blackhole's RX counters move before L1
  commit ("RX Queue Memory Map").
- Item 2, owner assumption: publication into LIM is ordered by the writer.
  Each publisher writes its data, waits for the write acks, then writes the
  flag, so the flag is issued only after the data writes were accepted at the
  L2CPU (eth_credit W6 after W5, packet_injection's produced counter,
  `PI_ASSUMED_INBOUND`, core_sync Word arrivals at an X280 director). This
  assumes the L2CPU applies accepted writes to LIM in arrival order: the
  NoC specification orders NOC2AXI delivery per VC only, and
  Front Port and L3 ordering are undocumented. No document says the ack waits
  for the LIM write; `NoC/Counters.md` defines the ack for "L1 or register
  space".
- Item 3, owner assumption: a store through a mode-1 window stalls at the store
  until BRESP, which follows its NoC write's completion
  (`PI_ASSUMED_BRIDGE_COMPLETION`, the store-to-BRESP TURN). No X280 document
  says the store, rather than a later fence, waits for BRESP. Evidence: the
  X280 TURN falls from 440 to 76 ticks on a tile whose completion blocking a
  mode-3 store disabled.
- Item 5, owner decision: `rdcycle` counts the cycles the hart executes and
  does not advance while the hart stalls on memory, a NoC transaction or a
  BRESP, so every X280 CPU-cycle column is executed cycles, not elapsed time;
  elapsed time and timeouts use `mtime`. Evidence: the SiFive X280 MC Manual
  21G3.04.00 §3.10.2.1 ("clock cycles the hart has executed"), and a
  2026-06-25 migration-worker 16 MiB READ on which rdcycle trailed mtime by
  about 50 ms. The transport_bandwidth store-bound loops show cpu/wall = 35.000 at
  1750 MHz, so those loops did not stall the hart measurably. The manual's
  §5.13 says the cycle counter advances at `rtc_clock`, which the in-tree PLL
  measurement (rdcycle about 35 times mtime at 1750 MHz) contradicts.
- Item 6, owner assumption: CLINT `mtime` counts at 50 MHz independent of the
  X280 PLL (every X280 wall interval and the PLL measurement,
  `PI_ASSUMED_MTIME_HZ`). Manual §5.13.1 defers the `rtc_toggle` rate to an
  X280 MC User Guide. Evidence: a PLL sweep fits X280 TURN as a fixed cycle
  count plus a clock-independent time in `mtime` ticks.
- Item 8, owner assumption: a write in flight through an L2CPU TLB window lands
  at its old target when the window is cleared (core_sync release).
  No L2CPU document states it. Evidence:
  `WormholeB0/PCIExpressTile/TLBs.md` states it for Wormhole PCIe TLBs, and every
  core_sync multicast release passes on silicon with no lost write, as did every
  pipeline_stage four-window clear recorded before the move into tt-metal.
- Item 9, owner assumption: an 8-byte LIM slot is one ECC granule with no
  read-modify-write into its neighbours (core_sync slots). Manual §14.2.6 gives
  L3 ECC as SECDED with no width; `L2CPUTile/Caches.md` gives none. Evidence:
  manual §14.4.8 gives a 64+8-bit granule as an example, and §14.2.1 places LIM
  in the L3 SRAM.

- Item 12: a Tensix RISC's L1 staging store is processed before the NoC command
  stores that send it (every Tensix publisher). `BabyRISCV/MemoryOrdering.md`
  "Store then store" leaves stores to different regions unordered; a stale send
  arrives as a wrong token.
- Item 13, closed with no assumption needed: whether ERISC and DRISC have the
  non-coherent L0 data cache that the Metal P2P peers invalidate while polling.
  `BabyRISCV/MemoryOrdering.md` "L0 data cache" documents it for Tensix only;
  the poll invalidates on every pass on all three targets, which is correct
  either way.
- Item 14, owner assumption: an X280 load observes an aligned NoC write of
  16 B or less into LIM whole (`PI_ASSUMED_COUNTER_ATOMICITY`, the 4-byte
  produced counter). No document covers LIM. Evidence: such a write lies in one
  atomic 16 B NoC unit (`WormholeB0/NoC/Ordering.md`, stated for L1), which
  the NOC2AXI slave writes as one AXI beat.
- Item 15: samples a hart copies from the pinned cache way into LIM's export
  region are host-visible (`PI_ASSUMED_PIN_EXPORT`). No document covers it.
- Item 20, owner assumption: a store through a mode-2 (Posted Writes) window
  gets an early BRESP and returns before its NoC write lands, and a later read
  through the window can pass a write still in flight (transport_bandwidth
  `x280_to_tensix` `RETIRED`). `L2CPUTile/TLBWindows.md`
  names the mode only. Evidence: hardware review 2026-09-03 ("returns a fake
  AXI BRESP quickly").

Item 16 (`PI_ASSUMED_LOCAL_TIMER_ORDER`, high/low/high Tensix timer reads) is
documented (`TensixTile/DebugTimestamper.md`). Item 17, successive writes from
one Metal RISC to one L2CPU's LIM on one NoC and one static VC arriving in
issue order (eth_credit E0's read-pointer shadow, E1's W5 before W6), is stated
by the NoC specification for writes on one static VC, which
Metal's unicast writes use; it is not confirmed on silicon.

## Collection and reproduction

Each benchmark prepares its own device state: the harness boots resident idle on
a chip whose L2CPUs are all held in reset, at the default 15-way LIM or the
14-way shape the pipeline and PacketInjection members request, and the
pipeline_stage suite prepares and tears down its pinned input only on a tile its
own process booted. Resets between them are explicit operator actions: a chip
left at 14 ways skips the 15-way members, and each pipeline_stage configuration
needs a reset before it runs. The runner never resets a device.

`tools/run_characterization.py` accepts explicit `--id` selections, build/firmware/
Metal paths, visible device, campaign directory and an optional `--host-label`
naming the host. A label is recorded in the campaign's `host.json` and in every
attempt record, and a campaign refuses a second label. Without one the runner
writes no `host.json` and no attempt carries `host_label`, including on a
campaign that already has a `host.json`. The manifest contains only
inputs consumed by execution or validation:

| Field | Consumer |
|---|---|
| `common_environment` | Settings shared by the selected commands |
| `configurations` | Fixed list of selectable cases; frozen with each campaign |
| Case `id` | Selection and coverage identity |
| Case `member` | Selects the suite's record validator |
| Case `argv` | Executable and exact gtest filter |
| Case `environment` | Explicit workload settings and firmware paths |

Dispatch mode is a property of the executable, not of the manifest. The
executables that launch on a core-local RISC link `x280_slow_dispatch`, which
sets `TT_METAL_SLOW_DISPATCH_MODE=1` before `main`; `common_environment` does not
set it, and the runner clears an inherited value, so every other executable runs
fast dispatch. A frozen manifest whose `common_environment` sets the variable ran
all its configurations under slow dispatch, so comparing one of these 102 of the
160 configurations with such a campaign also compares dispatch modes:
`x280_same_tile.{A,B}InitiatesFirst`, `x280_peer_tile.{A,B}InitiatesFirst`,
`tensix_peer_tile.{X280,Tensix}InitiatesFirst`,
`bw.x280_to_tensix.{,sd.,vse32.}{64,256,4096,65536}.{retired,injection}`,
`bw.tensix_to_x280.{64,128,256,320,384,512,1024,2048,4096,8192,16384}.{drained,injection}`,
`pipeline.x280.route_compute.direction_table_{plain,instrumented}`,
`pipeline.x280.route_forward.decide_plain`,
`packet.throughput.brisc.T1`, `packet.throughput.brisc.T1.c8k16`,
`packet.throughput.paired.T1`, `packet.throughput.{brisc,paired}.T{2,4,8}`,
`packet.throughput.paired.T1.c8k16`, `packet.throughput.{brisc,paired}.T{2,8}.c8k16`,
`packet.throughput.{brisc,paired}.T4.{c4k8,c8k16,s4,s8,s64,b4}`,
`packet.throughput.{brisc,paired}.T4.s64.{c1k16,c2k16,c4k16,c8k16,c16k16,c1k2,c1k4,c1k8}`,
`packet.latency-{include,exclude}.brisc.T1`,
`packet.backpressure-{include,exclude}.brisc.T1` and
`core_sync.release_skew.fixed4x4`.

Record expectations come from the suite validator and the selected sample/worker
counts; the manifest does not repeat them. Build recipes, preparation requirements
and historical eligibility evidence remain in documentation and the retained
original inventory. The runner uses prebuilt images; it does not build firmware
or establish exclusive ownership. C++ fixtures own device admission.

`--dry-run` shows the resolved command and selected environment. Each attempt has
a fresh directory containing `attempt.json`, `run.log`, `gtest.xml` and `raw/`.
The attempt record is atomically updated from `running` to `passed` or `failed`;
`running` means no terminal outcome was recorded, including interrupted runs.
It records the command/environment, host label (when given), start/end times, checkout SHA, executable
and firmware hashes, outcome, and one hash list for captured files. The existing
`producer_sha` field identifies the checkout at launch; it does not attest that
a supplied binary was built from that revision.

The gtest XML must name exactly the requested test and report completion; zero
tests and skips fail. A separate enumeration process/log is unnecessary for
that check. The shared writer uses `X280_CHARACTERIZATION_OUTPUT_DIR`, otherwise
its existing `TT_METAL_HOME/characterization` default. Failed/skipped/partial
runs cannot satisfy coverage. Retry selection is explicit; there is no resume
database, generic scheduler or implicit recovery.

Regenerate into a new report directory:

```sh
python3 tt_metal/fabric/x280/tools/tabulate_suites.py --campaign /path/to/campaign --report-dir /new/report
```

The report checks raw hashes and exact configuration populations and emits
`master_results.jsonl`, `coverage.json`, `campaign.json` and `MASTER_RESULTS.md`.
`MASTER_RESULTS.md` holds the coverage table and, under "Results", a result table
per member and record shape, each row linking its raw file. Table mode,
`python3 tt_metal/fabric/x280/tools/tabulate_suites.py FILE...`, prints the same tables for
current-format JSONL. Both render the p2p_latency, transport_bandwidth,
pipeline_stage, packet_injection, core_sync and eth_credit members; eth_credit's
kernel, attribution and link records are recognised and tabulated by no table.
Both outputs come from one render path, the only place a value is computed:
percentiles by the floor-index rule, and nanoseconds and rates at the record's
`conversion_hz` once its clock evidence validates (owner decisions 33 and 42).
Clock evidence that is well-formed but fails a measurement check -- the
producer's own `errors`, AICLK telemetry or arbiter outside tolerance, a thermal
trip, a rate outside the band, excessive uncertainty -- does not fail its
attempt: its row is converted at the measured rate, unvalidated (X280 mtime's
fixed 50 MHz, or the midpoint of the AICLK elapsed-rate bounds recomputed from
the raw samples), and its `clock` column gives that rate and the failed checks as
a warning. The row counts as covered, and `campaign.json` counts such records in
`wall_records_with_clock_warning`. Malformed evidence -- a missing or ill-typed
field, an unknown domain or reference counter, bounds that disagree with the raw
samples -- fails the attempt; table mode shows such a row raw, with the reason in
its `clock` column, as it does core_sync's rows, which offer no conversion. A
record that matches none of its member's tables fails table mode. The runner and the report read an attempt the
same way, so such a record fails its attempt; a passed attempt the report cannot
read counts as failed in its coverage, with the reason.
`campaign.json` contains only derived coverage/conversion counts, the evidence
directory and the frozen manifest hash. It does not consume `provenance.json`;
hardware context, build instructions and clock-evidence limits belong in the
verification note. The live tabulator reads only the current record format: it
reads no console capture. Earlier campaigns (including two-file attempt logs and run records at PI
schema 5 or earlier), records and console captures need the tabulator of the tree that
wrote them. Historical captures retain their original schemas and clock metadata; they are not
new campaign evidence.

### Tabulation

Tabulate current-format JSONL records of the p2p_latency, transport_bandwidth,
pipeline_stage, packet_injection, core_sync and eth_credit members, one result
table per member and record shape:

```bash
# Tabulate suite results to stdout:
python3 tt_metal/fabric/x280/tools/tabulate_suites.py <path_to_jsonl>
```

A campaign's report (`--campaign DIR --report-dir NEW`) carries the same tables in
its `MASTER_RESULTS.md`; this section describes
both outputs and how `tools/run_characterization.py` collects a campaign from
`characterization_manifest.json`. The tabulator reads only the current record
format. The `x280_characterization` CTest runs the tabulator's and the
campaign runner's tests, and `x280_record_round_trip` renders every member's
records and reads them back with the live tabulator.
