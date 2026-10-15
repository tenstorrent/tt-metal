# Packet Injection Characterization

The 2026-09-23 rerun (archived record `benchmarks/campaigns/20260923-measurement-unification/`)
includes both throughput families and all four latency/backpressure variants.
Its worker records retain native CPU measurements where the interval is local.
Current schema-2 records retain raw wall ticks and per-run AICLK evidence for
offline conversion. Historical captures below retain their original schemas
and conversion assumptions.

**Status:** Implementation and silicon execution are available. The
implementation uses core-sync for its phase boundary; measurement schema 2 owns
the clock-evidence contract.

**Scope:** feasibility benchmarks ID 10 and ID 11: characterization of
dedicated single-producer single-consumer (SPSC) packet injection between
Tensix workers (BRISC on NoC 0 and NCRISC on NoC 1) and an active X280 consumer
over Blackhole NoC and LIM. Covers three compile-time measurement modes across
five distinct image and test variants. The same producer, consumer, ABI and host
stages also implement ID 14, `packet_injection/erisc_to_x280_header`, with idle
ERISC producers; see [ERISC Header-Ingress Member](#erisc-header-ingress-member).

## Architectural Overview

The benchmark establishes dedicated SPSC ring loops between Tensix producers
and an active X280 consumer:

- **Producers (Tensix):** Up to 280 concurrent workers (BRISC on NoC 0 and
  NCRISC on NoC 1 across up to 140 physical tiles). Each worker maintains an
  independent L1 staging buffer, credit counter, and ring pointer.
- **Consumer (X280 Hart 0):** Services worker rings in a round-robin schedule
  (fixed tile prefix, BRISC then NCRISC), reading arrival snapshots from LIM,
  performing four volatile destination-ID byte loads at offsets 83..86,
  recording timing endpoints, and returning batch credits to worker L1.
- **Harts 1–3:** Quiescent parked participants that return success through the
  active lifecycle harness.

```text
[Tensix Worker]                                      [X280 Hart 0]
[one core-sync barrier after warmup credits drain]
[Mode 2: take throughput_start once, after barrier release]
Construct fresh 96 B header in staging slot
Acquire space in ring (check credit / stall)
Issue 128 B non-posted write to LIM ring
Advance reserved progress r; pending++
if pending==B || r-c_seen==S || issued==limit:
  barrier(assigned_noc)
  write stable produced word r to LIM ---------->  Poll produced[w] snapshot
                                                    Check available <= S and <= limit
                                                    Read 4 volatile ID bytes (83..86)
                                                    [Mode 1: read remote timer, store sample]
                                                    Advance consumed[w]
                                                    if consumed - credited == C || visit_end:
Poll local consumed mirror <------------------------- return_credit(consumed[w]) via mode 1
Retire requests (and record RTTs in Mode 3)
[Mode 2: take throughput_end once, after the final measured credit]
```

The consumer's arrival observation is four steps:

```text
fence inbound
load produced
if work is available: fence inbound
load the published ring slot and four destination-ID bytes
```

The second fence executes only on the work-taken path. The producer's barrier
before publication orders the producer's writes and does not order the
consumer's two loads: they are related by a control dependency, which RVWMO does
not order, and the slot address derives from the consumer's own `consumed`
counter rather than from the loaded snapshot, so no address dependency orders
them either. That second fence is what orders the counter load against every
slot load of the visit.

## Measurement Modes and Variants

The benchmark defines three compile-time modes and five build variants via
`PACKET_INJECTION_MODE` and `PACKET_INJECTION_INCLUDE_CREATION`:

| Variant | Mode | Include Creation | Measurement Semantics |
|---|---|---|---|
| `latency-include` | `PI_MODE_LATENCY` (1) | 1 | Tensix local timer before header construction (before ring-space wait) to X280 remote timer read after ID loads. |
| `latency-exclude` | `PI_MODE_LATENCY` (1) | 0 | Tensix local timer after header construction and before ring-space wait, to X280 remote timer read after ID loads. |
| `throughput` | `PI_MODE_THROUGHPUT` (2) | 0 | Each producer's own `tensix_live_timer_read` interval, from core-sync release to its final measured credit. Reported per worker. |
| `backpressure-include` | `PI_MODE_BACKPRESSURE` (3) | 1 | Tensix producer-only credit stall durations and stall-triggered request RTTs (with creation). |
| `backpressure-exclude` | `PI_MODE_BACKPRESSURE` (3) | 0 | Tensix producer-only credit stall durations and stall-triggered request RTTs (without creation). |

### Wire Slot and Header Syntax

Every ring slot is 128 bytes, aligned to 64 bytes (`pi_slot_t`):
- Bytes 0..95: Syntactically valid Fabric shadow unicast header (`fs_hybrid_mesh_packet_header_t`).
  Created afresh in volatile staging storage by `fs_seed_unicast_header()`: zero command union,
  zero payload, `NOC_UNICAST_WRITE=0` at byte 42, `src_ch_id=0` at byte 43, `routing_fields=0`,
  35 route bytes `FS_HOP_NOOP`, destination chip at 83..84, destination mesh at 85..86,
  zero multicast parameters.
- Bytes 96..103: 8-byte start timestamp (`uint64_t start`), populated only in latency mode.
- Bytes 104..127: 24 bytes of padding.

The X280 consumer loads four volatile destination-ID bytes at offsets 83, 84, 85, 86 in that
exact order to assemble destination chip and mesh IDs. Values are discarded; no payload scanning
or checksumming is performed.

### Latency Measurement

In latency mode, the producer samples its tile's local 64-bit timestamper (`tensix_live_timer_read`)
either before header construction (`include_creation=1`) or immediately after header construction
(`include_creation=0`), but in both cases strictly before `acquire_space_for_current_request()`.
Therefore, any ring-space credit wait is included in the timed request interval. The timestamp is
written into `slot.start`. Upon observing the arrival and loading the four destination ID bytes,
the X280 consumer remotely samples the *same* producing tile's timestamper through an uncached
TLB window (`X280_TLB_PACKET_INJECTION_TIMER=156`), computes the duration `end - start`,
and stores it directly into a reserved cached way (`0x0A000000`). Timing spans from producer
start through the remote NoC low-word read; the high2 read/response and post-bracket fence fall after
the recorded endpoint but delay subsequent requests. Durations are cached until run completion,
then exported to LIM outside measurement. Ring-space waiting is explicitly included in the measured interval.
The consumer's load of `slot.start` follows the four ID bytes, so both sit after the work-taken
fence of the publication ordering above.

X280 creates the samples, so `pi_consumer_slot_t::phase_done` is the authoritative exported
sample-prefix length. The producer's `latency_count` counts requests it has seen credited and is
independent retirement evidence: a run that fails between the sample store and the credit
observation leaves `latency_count` smaller than the prefix that exists, and truncating by it would
discard valid samples. The host bounds the consumer count by the worker's export share
(`latency_sample_stride / 8`, which is `N`) and by the worker's containment in the export region,
and rejects a count exceeding either rather than clamping it — a rejected prefix hands back no
length. A `latency_count` above `phase_done` is a corruption check with no reachable trigger, since
the producer retires only on the credit X280 returns after servicing.

#### Measurement dispersion

The only repeated-run figures in this document are the latency pair, and they
are therefore the only estimate of how much any single number here can be
trusted. Two runs of the identical N=1 configuration gave 2188 and 2199 ticks, a
spread of **0.50%**; two of N=3 gave 3426 and 3416, a spread of 0.29%.

That bounds a *single-request* measurement. It does not bound the throughput
rows directly, which average over 1000 requests per worker — but it does explain
them: 0.50% on one request implies roughly 0.016% once averaged over a thousand,
which is the same order as the ~0.04% agreement actually observed across builds
and between topologies. The two are consistent, so the sub-0.05% agreements in
the throughput table are real rather than an artefact of a quiet instrument.
Every figure in this document other than the latency pair is a single run.

#### Packet-creation cost

`latency-include` and `latency-exclude` differ only in whether the producer's
start timestamp is taken before or after header construction, so the difference
between them is the cost of constructing one 96-byte header:

| N | include (2 runs) | exclude (2 runs) | difference |
|---|---|---|---|
| 1 | 2188, 2199 → 2193.5 ticks | 1704, 1728 → 1716.0 ticks | 477.5 ticks = 354 ns |
| 3 | 3426, 3416 → 3421.0 ticks | 2961.7, 2958.7 → 2960.2 ticks | 460.8 ticks = 341 ns |

So header construction costs roughly **341–354 ns**. Both ends of each
difference are two runs, but two runs bound a spread only loosely: the observed
spreads are 0.50% (include) against 1.39% (exclude) at N=1, and 0.29% against
0.10% at N=3. The exclude side is the wider of the two at one N point and the
narrower at the other, so these measurements do not support any claim that one
side of the creation boundary is noisier than the other — with two runs per
cell the spreads themselves are not resolvable. The range above is a range of
two point estimates, not a confidence interval.

### Throughput Measurement

Throughput mode disables all per-request timestamping and measures each producer
independently. A producer reads its own tile's local timestamper twice: `throughput_start`
immediately after core-sync releases the warmup barrier and before its first measured header,
and `throughput_end` after its final measured request is fully credited. Both are raw reads
of a concurrent-safe local timer, so no producer's interval depends on another's progress.

Each worker record keeps that raw interval, and the host derives no rate. Its `throughput`
object carries `start_ticks`, `end_ticks`, `requests` (N),
`useful_bytes_per_request` (96) and `slot_bytes_per_request` (128); the offline tool derives
one rate per worker from it and the run's clean AICLK evidence:

```text
elapsed             = end_ticks - start_ticks
requests_per_second = requests * conversion_hz / elapsed
```

Workers are never summed or averaged into a capacity figure. The useful-header and slot byte
rates are that one worker's rate scaled by the 96 useful header bytes and the 128 slot bytes
respectively, so they describe a single producer, not aggregate bandwidth.

Raw endpoints and the delta are retained in every case, including failure. An interval of
zero or one at or beyond the timer half-range is rejected rather than reported: the worker's
`throughput` carries `interval_error` `invalid_interval`, and the run fails naming the first
offending worker and its two endpoints.

**Aggregate interval.** Separately from the per-worker intervals, and in every mode, X280
reads its own `mtime` and `rdcycle` once when the measured-phase core-sync release returns
and again after the measured phase's final credit flush, and publishes the four raw values
in `pi_run_result_t` (`aggregate_start_mtime`, `aggregate_end_mtime`,
`aggregate_start_rdcycle`, `aggregate_end_rdcycle`). This is the one common clock over all
P workers. The release is a posted write that returns once issued, not once delivered, so
the record labels the start `release_issued`. The start sample therefore does not mark when
the producers began: one released by an earlier write may already have published, which
makes the rate look higher, and one whose release is still in flight starts after it, which
makes it look lower. The error is of the order of the release's issue and delivery time.
The host records the raw `mtime` and `rdcycle` endpoints with the fixed 50 MHz `x280_mtime`
clock; the offline tool derives the interval over `P*N` headers and `P*N*96` header bytes, and
rates from it; `rdcycle` stays raw hart-0 cycles. On a successful run a zero, inverted or half-range interval in either counter fails the
run, naming both pairs.

`empty_visits` counts measured-phase round-robin visits that found the visited worker's
ring empty. Warmup visits and visits to a worker that has already completed its phase are
not counted.

Each throughput producer also counts its measured-phase credit-stall episodes in
`pi_producer_result_t.stall_count`: entries into the ring-full wait, the episode
backpressure mode samples, counted with no durations and no added timer reads. A full ring
whose credit has already returned when first checked is not an episode. The count is taken
on entry, where it overlaps the wait; taken on exit it sits on the issue path. A failed
run can therefore count one episode that never ended, which backpressure mode would not
sample. Many empty visits with few stalls point at the producers or NoC ingress;
many stalls with few empty visits point at X280 service.

**Measured scaling, reported without attribution.** On silicon at S=16, B=1,
C=1, per-worker rate falls as 1/P across an eightfold range:

| P | shape | per-worker requests/s | intra-shape spread |
|---|---|---|---|
| 1 | T=1 BRISC-only | 1418936 | — |
| 2 | T=1 paired | 719005..719211 | 0.029% |
| 2 | T=2 BRISC-only | 718322..718527 | 0.028% |
| 4 | T=2 paired | 361312..361464 | 0.042% |
| 4 | T=4 BRISC-only | 361492..361641 | 0.041% |
| 8 | T=4 paired | 181041..181130 | 0.049% |

Workers within a shape complete within 0.05% of each other, and the two
independent topologies at each of P=2 and P=4 agree, so the service order is
fair and the rate does not depend on whether two workers share a tile across
both NoCs or occupy separate tiles on one NoC.

These records describe individual worker completion and fairness only. They do
not establish aggregate X280 capacity, injection bandwidth, or NoC bandwidth,
and the per-worker rates must not be summed: every shape reconstructs the same
constant, and presenting that constant as a capacity attributes the 1/P falloff
to a mechanism these measurements do not distinguish. A single consumer hart
serving workers in turn and a saturated shared NoC path both produce 1/P with a
flat total. **A B/C sweep is required before attributing the observed scaling to
any one mechanism**; this configuration holds B and C at 1, so every request
carries a publication and a credit return.

**The rates do not depend on the consumer's code layout.** They were re-measured
after a change that made the consumer image 464 bytes larger and relocated its
polling loop, and they agree with the pre-change measurements to about 0.04% --
the same order as the spread within a single shape. That is a reproducibility
result about the measurement, not evidence for any mechanism: it rules out the
reading that the earlier figures were an artefact of one build which a
differently optimised consumer would move.

### Backpressure Measurement

Backpressure mode uses producer-local instrumentation exclusively:
- The producer retains the original start time for every issued request in a ring-sized array `starts[S]`.
- When the ring is full (`r - c_seen == S`), the producer flushes pending writes and checks for credit.
- If credit has already arrived, requests retire immediately without a stall sample.
- If still full, the producer starts a stall episode (`stall_start = timer()`) and spins until progress
  is observed. Upon progress, `end = timer()` is sampled as the first action.
- A single stall duration `end - stall_start` is recorded in L1 `stalls[stall_count++]`.
- For each earlier request released by that observation (`jump` requests), an RTT sample
  `end - starts[(c_seen + j) & (S - 1)]` is recorded in L1 `rtts[rtt_count++]`.
- Warmup and final drain phases never record stall or RTT samples.

**The axis produces no sample at P=1.** At S=16, B=1, C=1 with a single worker,
both creation variants complete successfully and report `credit_stalls` and
`stall_triggered_rtt` at count 0 with `reason="no_samples"`. **An empty
backpressure population at P=1 is a configuration artifact and must not be read as
"no stalls observed".**

At P=2 and above the consumer falls behind and both populations are fully
exercised. A T=2 BRISC-only run recorded 1939 credit-stall and 1939 RTT samples,
mean 768.8 ticks and 31256.3 ticks respectively — the stall is one episode's spin
while the RTT spans a request's original start through the credit that released
it, a full ring's queueing at S=16, which is why the two differ by three orders
of magnitude.

### Mode-specific timer code

A mode's timer endpoints are absent from the linked image of every other mode, not branched around
in it. Disassembling each variant's producer kernel and X280 firmware with
`riscv-tt-elf-objdump -d -l --inlines` and attributing every inlined `tensix_live_timer_read`
expansion to its call site gives:

- The throughput producer holds exactly two endpoint call sites, taken once each per run:
  `throughput_start` in `run()` and `throughput_end` in the measured `run_phase`. It is also the
  only image holding the retry-failure path that publishes `PI_ERROR_TIMER_RETRY_EXHAUSTED` at
  `PI_PHASE_MEASURE` for the first of them. The other four producer images reference none of the
  three.
- The latency producer holds only its per-request start read, before header construction in
  `latency-include` and after it in `latency-exclude`, instantiated in both the warmup and the
  measured `run_phase`.
- The backpressure producer holds the stall-episode start and end reads in
  `acquire_space_for_current_request` and the per-request start an RTT is measured from, all three
  only in the measured instantiation: the warmup instantiation takes no sample read.
- All five producer images retain the two safety reads, the phase-deadline origin in
  `reset_phase_deadline()` and the amortized deadline check in `amortized_safety_check()`. Those
  are the only timer reads in the throughput image besides its two endpoints.
- The remote end read belongs to the X280 image rather than the producer. The object code of
  `tensix_live_timer.h` and of `ensure_timer_window()` appears only in the two latency firmware
  images; the throughput and both backpressure images contain neither, so outside latency mode the
  X280 samples no Tensix timestamper and programs no timer window. Its `mtime` safety reads and
  the aggregate `mtime`/`rdcycle` bracket remain in all five.

### Results and clock evidence

Per-worker records preserve the initialized `raw_ticks` arrays, and the run record carries
no population: the offline tool pools the workers' raw samples. `latency`, `credit_stalls`,
and `stall_triggered_rtt` are separate populations, whose statistics the offline tool
computes from the raw samples. Empty populations contain `reason="no_samples"`. Producers retain raw ticks rather than
nanoseconds. Each run and worker record carries the programmed maximum, four
host-bracketed counter samples, start/end/elapsed rate bounds and CMFW throttle
telemetry; the offline tool converts at `conversion_hz` when that evidence
validates, and otherwise at the measured rate with a warning (`docs/measurement.md`
"Collection and reproduction"). RTTs describe
only requests released by observations that end a credit stall, not all requests.
Failed or invalid-interval throughput records preserve the raw endpoints,
counts and byte geometry with an explicit `interval_error`; rates are derived
offline. Per-worker throughput records carry `credit_stall_episodes`, that
worker's `stall_count`.

The elapsed rate bound narrows as the run lengthens, and the Tensix member's
warm-up is fixed rather than calibrated; `docs/measurement.md` gives the margin and
what calibration would need. The ERISC member calibrates its warm-up (below).

Every producer counts the produced-counter writes it issues in the measured phase in
`pi_producer_result_t.measured_counter_writes`, in every mode: one per publication of a
non-empty batch, whether a full batch, a full ring or the phase end forced it. Every
per-worker record ends with that count as `producer_counter_writes`. A successful run
requires each worker's count to lie in [ceil(N/D), N] with D = min(B, S), since each write
publishes at least one and at most D headers.

Each run also writes an aggregate record (`"record":"aggregate"`) to the run file, after
the run record. It repeats the run's identity, configuration and outcome fields, so it joins
its run by `run_value`, then carries `start_event` (`release_issued`),
`aggregate_start_mtime`, `aggregate_end_mtime`, `aggregate_start_rdcycle`,
`aggregate_end_rdcycle`, `empty_visits`, `aggregate_error`, `measurement_schema` 2,
`wall_counter` `x280_mtime`, `cpu_counter` `x280_rdcycle`, and its own `clock`, the fixed
50 MHz `x280_mtime` domain rather than the run's AICLK evidence. The interval and the P*N
headers and bytes it covers are derived offline from those. On a failed run or an invalid
interval `aggregate_error` is `run_failed` or `invalid_interval`, and the raw endpoints are
kept. Run records at PI schema 5 or earlier, such as the 2026-09-23 campaign's, predate the
aggregate record; the frozen tabulator (`docs/measurement.md` "Collection and reproduction")
reads them, and the live campaign validator requires the aggregate record.

Records include configuration, ordered tiles and selected prefix length, per-worker
placement, memory budgets, cleanup status, and source revisions/dirty state captured
at CMake configuration time. Both run records carry `released_group_dests` and
`released_endpoints`, the release X280 executed (`pi_run_result_t`): the group's receiver
count, zero for none, and the unicast endpoint count. The run record also carries
`release_guards`, the guard words armed (zero for a unicast release), and
`release_guards_overwritten`. Records written before 2026-10-01 carry
`release_multicast_safe` instead, the flag the host passed: 1 released tiles that filled their
bounding box by one multicast, 0 released one unicast per tile. A dirty source revision is not a reproducible commit.
Reconfigure before rebuilding after changing revisions.

The reported assumption mask covers inbound LIM visibility, scalar counter atomicity, mode-1 credit completion,
nominal mtime frequency, local timer access ordering, and (for latency) pinned export
visibility. The mode-1 store plus immediate fence follows existing transport code;
a CPU fence alone is not an independently established NoC completion guarantee.
`PACKET_INJECTION_BRIDGE_HISTORY` defaults to `unknown`; the accepted
`reported_no_mode3_since_reset` value is provenance, not proof of clean hardware.
`mode3_observed` refuses reuse without initiating a reset.

## Memory Contract

### ABI Records

Every args and result record carries an `abi_tag`: bits [31:16] `PI_ABI_MAGIC` (0x5049),
[15:8] zero, [7:4] the mode, and [3:0] the include-creation flag, so one comparison rejects a
mismatched firmware variant and uninitialized memory. The tag carries nothing about the layout;
the compile-time goldens in `packet_injection_abi.h` are what reject a moved span, and nothing
rejects a peer built against another layout.

```text
sizeof(pi_x280_args_t)       = 128
sizeof(pi_producer_args_t)   = 72 = 18 runtime words (PI_PRODUCER_ARG_WORDS)
sizeof(x280_noc_coord_t)     = 8, one tile-table entry
sizeof(pi_consumer_slot_t)   = 24
sizeof(pi_counter_slot_t)    = 64
sizeof(pi_ready_slot_t)      = 64
sizeof(pi_producer_result_t) = 128, filling its 128-byte L1 allocation
sizeof(pi_run_result_t)      = 128
```

These are the terms of the arena formula below: 320 is `pi_x280_args_t` at 128 bytes,
`pi_host_control_t` at 64 and `pi_run_result_t` at 128, `8*T` the coordinate table, `8*P` the core-sync arrival slots,
`24*P` the consumer slots, and `128*P` the produced and completion slots.

### Layout Rejection

`pi_compute_layout()` returns a `pi_layout_check_t` of `code`, `value`, and `bound`, and writes its
output layout only on success — a rejected call leaves the caller's `pi_layout_t` untouched, so no
address can be derived from rejected input. `code` names the class of rejection and `value` and
`bound` the offending quantity against the limit it failed, both zero where the code names neither.
Host, X280, and producer map the code through the same `pi_layout_code_to_error()`, so one cause
reaches a report under one name from three actors.

`PI_LAYOUT_INVARIANT` is the function catching its own arithmetic against its own postcondition. No
input can provoke it and it accuses no argument, which is why it maps to
`PI_ERROR_LAYOUT_INVARIANT` rather than to `PI_ERROR_BAD_ARGS`: it reports an implementation
self-check failure, and sends nobody auditing parameters that may be entirely valid.

### X280 LIM Profile (`LimLayout::PacketInjection`)

The benchmark requires the opt-in `LimLayout::PacketInjection` memory profile:
- **L3 WayEnable:** Exactly 1 (`lim_ways = 14`, 2 cache ways, 14 LIM ways).
- **Lower Arena (`0x08020000..0x080FFF80`):** 917,376 bytes dedicated to the packet injection
  harness. Houses the X280 argument block, host control, the run result, the tile coordinate
  table, per-worker core-sync arrivals, consumer slots, produced and completion slots, rings,
  and latency sample export.
- **Image Text Bound:** Image text/rodata/data is constrained to `[0x08001000, 0x08020000)`
  (max 126,976 bytes). Linker scripts enforce `_bss_end <= 0x08020000` via `--defsym=X280_PACKET_INJECTION_LAYOUT=1`.
- **Upper Reserved Spans:** Stacks (`0x080FFF80..0x0811FF80`), trap blocks (`0x0811FF80..0x08120000`),
  and boot/socket/protocol reservations (`0x08120000..0x08154000`) remain completely untouched.

Arena sizing formula:
`F(P, T, S) = align64(align64(320 + align64(8*T) + 8*P) + 24*P) + 128*P + 128*P*S`.
The tile-coordinate table occupies whole cache lines because a LIM write admits
only whole cache lines, so the host writes the reserved span and leaves the tail
beyond the `T` records zero.
The produced and completion slots are 64 bytes each at a 64-aligned base -- the inner
`align64` and the `128*P` term -- because the producer publishes each of them with a
length-carrying NoC write whose Tensix L1 source is congruent to 0 mod 64, and that
write path requires source and destination congruence. Only `value` is semantic; the
remaining 60 bytes of each slot are padding.
For `P=280, T=140, S=16, N=48`: base `F = 619,072` bytes; export `8*P*N = 107,520` bytes; total arena `726,592` bytes.
Request `PACKET_INJECTION_N=48` explicitly for that initial maximum-worker point.
The latency default remains N=4096 and is rejected, not clamped, when it exceeds capacity.
The core-sync arrival table adds eight bytes per worker before alignment; every admitted
configuration is decided by `pi_compute_layout()` against the 917,376-byte arena.
Small tile counts, not the maximum, discriminate this arithmetic. At `P=280` the consumer
array is a whole number of cache lines (`24*280 = 6,720 = 105*64`), so `produced_offset`'s
`align64` rounds nothing and the coordinate table's 16 bytes of padding are absorbed; at
`P=2` that `align64` rounds 16 bytes and the padding moves the arena by 64. A layout change
that is wrong in the padding terms can pass at 280 workers and fail at one.

### Pinned Cache Way

Way 1 (`0x0A000000..0x0A020000`, 128 KiB) is reserved for writable sample storage in latency mode:
- Initialized prior to warmup via `x280_pinned_input_prepare` with `X280_PINNED_INPUT_WRITABLE_SAMPLES`.
- Stale cached copies are flushed using L3 `CCACHE0_FLUSH64` (`0x02010200`), lines zero-populated,
  and active way masks updated to bit 0 alone across all 38 masters.
- Exported to LIM after producer quiescence, and torn down via `x280_pinned_input_teardown`
  only after host collection is acknowledged.

Throughput and backpressure reserve no part of that region. They instead call
`x280_pinned_input_check_policy(..., X280_PINNED_INPUT_BASELINE)` before warmup, so a prepared way
mask a prior run left behind fails the run at `PI_PHASE_INIT` rather than starving the measured
phase into a timeout. A wrong WayEnable, a wrong way mask, and a wrong region shape keep distinct
codes — `PI_ERROR_PIN_WAYENABLE`, `PI_ERROR_PIN_WAYMASK`, `PI_ERROR_PIN_SHAPE` — and an admission
code outside that enum names no cause and reports `PI_ERROR_PIN_POLICY`.

A failed baseline check reserved nothing, so it does not set uncertain pin ownership and does not
block the run's own teardown. Only a refused or partial preparation does, and that retains the way:
cleanup then refuses to relinquish it rather than handing back a way whose state is unknown.

### Tensix L1 Shards

One height-sharded `MeshBuffer` covers every participating tile, one page per tile at a
single device-local base `l1_tile_base`, which is the only L1 address the ABI transmits.
Admission requires the per-tile span to be no larger than the target's L1 window, which `Capacity`
carries as `[l1_window_base, l1_span_limit)`; the Tensix suite fills it with the allocator's L1 base
and `l1_size_per_core()`, and `allocate_tile_l1_buffer()` checks that the allocated span ends within
`l1_size_per_core()`:
- The first cache line holds the tile-shared core-sync release word.
- Worker 0 (BRISC) starts at offset 64; Worker 1 (NCRISC, if Paired) starts at offset `64 + L`.
- Worker block: consumed mirror (64 B), control line (64 B), stable produced source (64 B),
  stable ready source (64 B), producer result (128 B), staging (`128*D` B),
  plus start/stall/RTT arrays in backpressure mode. The layout retains paired
  64-bit wall/CPU start stamps and paired 32-bit stall/RTT deltas; intervals
  exceeding UINT32_MAX ticks fail before storage, never truncate. The producer
  result includes native CPU endpoints for throughput without enlarging its
  reserved 128-byte slot.

Page order is asserted, not assumed. The host reads the whole buffer in one transfer and
slices it by page index, and nothing on the host can prove that page `i` is ordered tile `i`
without silicon. Each slice therefore identifies itself: the host requires the
`pi_producer_result_t` in page `i` to carry the expected `abi_tag` and `worker_index`, and a
permutation is a hard failure naming both the expected and the found index. A silent
permutation would attribute one tile's counters to another, which no later check would catch.

## Outbound TLB Window Map

The X280 firmware reserves:
- **Entries 16..155:** 140 fixed uncached System Port L1 windows (one per physical tile).
  Programmed once before warmup; credit writes use Mode 1 (`NOC_ORDER_STRICT`, Strict AXI).
- **Entry 156:** Latency debug-timer window (`NOC_ORDER_STRICT`), mapped to page
  `0xFFA00000` to access Debug Timestamper at `0xFFB121F0/4`.
- **Entry 157:** Core-sync ordered release window. A selection the shared release selector
  gives a group (its tiles fill one logical rectangle) takes one multicast rectangle; every other
  selection takes one table-ordered unicast write per tile.
- **Entries 192..223:** Host-owned (untouched by firmware).

## Host Orchestration and Safety

1. Pre-run verification: checks AICLK telemetry, validates configuration bounds, computes layout,
   allocates L1 buffers and primes LIM.
2. Two-phase collection:
   - Producers signal data quiescence via the one-word `PI_READY_DONE` completion slot.
   - X280 copies cached samples to LIM export, then sets `data_quiescent = 1`, which is the
     host's authorization to collect: every consumer slot, producer result, and export byte
     is final before it, and only teardown fields change after it.
   - Host copies valid prefixes into host memory.
   - Host writes the active `run_value` to `collect_ack_run_value` in `pi_host_control_t`;
     zero means not acknowledged, so a run value is never zero.
   - X280 performs cache teardown, clears touched windows, and returns through active harness.
3. Bounded joinable collection thread:
   Host runs `ttmd::Finish` and `EnqueueReadMeshBuffer` on an isolated joinable thread bounded
   by the host deadline.
4. STOP and retained-resource policy:
   On producer failure or the host deadline, a concurrent runtime cleanup publishes host STOP.
   X280 writes packet `PI_CONTROL_STOP`, performs a validated recovery release when the normal
   barrier did not release, and allows a bounded recovery grace period for completion,
   collection, ACK, and cleanup. If release, counter quiescence, or cleanup remains unproven,
   the host invokes `retain_failed_run()` without releasing any owner. A host error raised
   while that cleanup runs first joins it, within its grace, and then retains.

The target-agnostic stages above live as free functions in
`tests/device/suites/packet_injection/common.hpp`: run-value minting, strict number parsing and
failure rendering; the LIM slot dump; priming the arena with the X280 args and tile table; the
X280 launch, quiescence poll and recovery under the host deadline; consumer-slot, run-result and
export readback; the bounded collection wait, which runs the target's readback on its own
thread under the X280 run's host deadline; the checked X280 and producer argument builders; the
collection ACK and cleanup checks; and per-worker count, success,
export-accounting and throughput-interval validation plus the shared JSONL
fragments. The aggregate-interval predicate is pure host and lives in
`tests/device/suites/packet_injection/regions.hpp`, where the host tests reach it. The Tensix
suite, `tensix_to_x280.hpp`, keeps its `run_axis` as a straight-line sequence over those calls and
owns tile placement, the sharded L1 buffer, producer kernel creation and runtime args, the
collection body (`Finish`, Tensix wall samples and the sharded L1 readback), and the
producer-result readback and identity checks.

A producer writes its identity — `abi_tag`, `worker_index`, `run_nonce` — into its L1 slice as
soon as that slice's address is trustworthy, so every later failure carries identity plus a
failure record. Four failures precede that point and can leave no L1 trace at all: an
`abi_tag` mismatch, `pi_compute_layout()` rejecting the transmitted parameters, a
worker-index/RISC parity disagreement, and an L1 span exceeding the tile's L1. None of the four
yields a slice address. The `abi_tag`, layout and L1-span rejections are identical for every
worker, so they present as uniform zeros across every page; a parity disagreement belongs to
the one image that was handed another worker's args and zeroes only its own page. The host
reports the count on that path: `N of P slices are unpublished`, with `N == P` meaning the
launch itself was rejected before any producer could derive its slice, and `N < P` meaning
that one producer failed before its result write. This is the first thing to read on a silent
bring-up, since a wholly rejected launch and a single dead producer are otherwise
indistinguishable. Of the four, the parity disagreement is the only one identifiable from the
dump alone: it is the only launch-side rejection that yields `N < P`, so a single unpublished
slice among written ones narrows to it or to a producer that died mid-run, while the other
three always yield `N == P`.

A bad `worker_index` and an out-of-range peer coordinate do carry identity but deliberately
publish no `READY_FAILED`, because the ready slot they would announce on belongs to another
worker. These surface as an X280 timeout, or as `PI_ERROR_STOPPED` if a host STOP arrives
first; the reason is in the collected L1 result, not in the readiness mirror.

Cooperative failures export each worker's measured prefix independently only after all
producers publish completion and the counter-distance checks prove quiescence. Warmup
progress is not a latency sample count.
An ACK timeout or STOP leaves the active image, export, and pinned samples intact;
a later matching ACK permits teardown but does not erase the first failure.
Failed cache restoration or window readback never authorizes host cleanup.

### Synchronization and completion

Producers run warmup immediately, drain every warmup credit, then publish one indexed
core-sync word arrival and wait on the tile-shared release word. X280 consumes warmup,
flushes credits, verifies `consumed == credited` for every worker, waits for all arrivals,
and releases either the host-selected group or every tile by an ordered unicast write.
Paired BRISC and NCRISC workers publish separate arrivals but poll the same tile release word.

`pi_x280_args_t.release_num_dests` and `release_rect` carry the host release selector's group:
a nonzero `release_num_dests` releases the virtual rectangle `release_rect` by one multicast, and
zero releases every tile by its own unicast write, as a core-sync endpoint rather than a group.
X280 does not choose; it checks that a group is the tile table's -- a valid rectangle containing
every entry, `release_num_dests` equal to `tile_count` -- and that no group leaves the rectangle
zero, and rejects anything else as `PI_ERROR_BAD_ARGS`. A singleton group would not do: the
transport releases every group, whatever its size, through a multicast-enabled window, so a
one-tile rectangle is still a broadcast. NoC broadcast is
delivered only to Tensix tiles -- firmware opts every other NIU out through the broadcast
opt-out masks (tt-isa-documentation `BlackholeA0/NoC/MemoryMap.md`, the broadcast-rectangle
description and the column and row opt-out masks; `NoC/README.md`) -- so a rectangle of any
other tile type, one tile included, would drop the release and leave every producer waiting until
timeout.
The Tensix suite passes the group the shared release selector (`select_release`) returns when its
tiles fill one logical rectangle, and none otherwise.

Whenever the executed release is a group, every run also checks that it stayed inside its
tiles: every other compute-grid core holds a guard word at the release address, written before
the AICLK start read and read back after the end read, and a run that overwrote any of them fails,
naming the cores. The helpers are in `harness/regions/release_guard.hpp`, shared with the core-sync
check (`X280CoreSyncDirector.RequiredMulticastReleasesExactlyItsSelection`).

The packet control line carries only `PI_CONTROL_RUN` and `PI_CONTROL_STOP`; it is not a
phase protocol. The completion slot carries only `PI_READY_NONE`, `PI_READY_DONE`, and
`PI_READY_FAILED`. Failure detail remains in `pi_producer_result_t`.

## Benchmark Parameters

### Build-time selectors

Each selector pair is one firmware image and one producer kernel build; a run
changes it only by switching image and gtest selector.

| Parameter | Values | Meaning |
|---|---|---|
| `PACKET_INJECTION_MODE` | 1 latency, 2 throughput, 3 backpressure | Measurement mode. A kernel define the host sets from the gtest selector, not an environment variable. |
| `PACKET_INJECTION_INCLUDE_CREATION` | 0, 1 | Whether the timed interval includes header construction. Must be 0 in throughput. |
| `X280_PACKET_INJECTION_FW_IMAGE` | path | X280 firmware image. Unset, each point uses the build's `firmware-packet-injection-<variant>.bin`. A set path must contain the variant name the gtest selector expects, or the test skips. |

### Runtime parameters

Read from the environment by `tests/device/suites/packet_injection/tensix_to_x280.hpp`,
defaulted by `Config` in `tests/device/suites/packet_injection/regions.hpp`, and
rejected rather than clamped at admission when out of range.

| Environment variable | Symbol | Default | Admitted range | Meaning |
|---|---|---|---|---|
| `PACKET_INJECTION_FAMILY` | family | `brisc` | `brisc`, `paired` | Workers per tile: BRISC on NoC 0 only, or BRISC on NoC 0 plus NCRISC on NoC 1. |
| `PACKET_INJECTION_T` | T | 1 | [1, 140] | Tile count. |
| — | P | T, or 2·T paired | [1, 280] | Worker count, derived from family and T. |
| `PACKET_INJECTION_TILES` | ordered tiles | `0,0` | at least T distinct entries | Logical tile list, `x,y;x,y;...`. The first T entries are the selected prefix, in service order. |
| `PACKET_INJECTION_S` | S | 16 | power of two, < 2^31 | Ring depth per worker, in 128-byte slots. |
| `PACKET_INJECTION_B` | B | 1 | ≥ 1 | Producer batch: requests issued between publications of `produced`. Staging depth is D = min(B, S). |
| `PACKET_INJECTION_C` | C | 1 | [1, K] | Credit batch: the consumer returns credit once C requests are consumed since the last credit, or at visit end. |
| `PACKET_INJECTION_K` | K | 1 | ≥ 1 | Per-visit cap: most requests the consumer takes from one worker per round-robin visit. |
| `PACKET_INJECTION_W` | W | 1000 | W + N and P·W fit in 64 bits | Warmup requests per worker, drained before the core-sync barrier and never sampled. |
| `PACKET_INJECTION_N` | N | 100000 throughput, 4096 otherwise | ≥ 1, plus the capacity bounds below | Measured requests per worker. |
| `PACKET_INJECTION_DST_CHIP` | dst_chip | 0 | < 256 | Destination chip ID written to header bytes 83..84. |
| `PACKET_INJECTION_DST_MESH` | dst_mesh | 0 | < 1024 | Destination mesh ID written to header bytes 85..86. |
| `PACKET_INJECTION_POLL_CADENCE` | poll_cadence | 1024 | power of two in [1, 2^20] | Poll iterations between amortized deadline and STOP checks, on producer and consumer. |
| `PACKET_INJECTION_PHASE_TIMEOUT_MS` | phase_timeout_ms | 60000 | [1, 600000] | Per-phase deadline, converted to X280 mtime ticks and Tensix timer ticks. |
| `PACKET_INJECTION_HOST_TIMEOUT_MS` | host_timeout_ms | 600000 | ≥ 9·phase_timeout_ms + 5000 | Host deadline for the whole run. |
| `PACKET_INJECTION_BRIDGE_HISTORY` | — | `unknown` | `unknown`, `reported_no_mode3_since_reset` | Recorded provenance. `mode3_observed` refuses the run. |
| `PACKET_INJECTION_TEST_RETAIN_UNANNOUNCED_FAILURE` | — | unset | `1` | Test only: forces an unannounced producer failure to exercise `retain_failed_run()`. |

N is further bounded by capacity, which `pi_compute_layout()` decides:

- Latency: the export `align64(8·P·N)` must fit the 131,072-byte pinned way, and
  `F(P, T, S)` plus the export must fit the 917,376-byte arena.
- Backpressure: each worker's L1 slice grows by `align64(16·S) + 2·align64(8·N)`
  for the start, stall and RTT arrays.

### Fixed and derived quantities

| Quantity | Value | Source |
|---|---|---|
| Slot size | 128 B, of which 96 B header | `PI_SLOT_BYTES`, `pi_slot_t` |
| Staging depth D | min(B, S) slots of 128 B per worker | `pi_compute_layout()` |
| Tensix timer rate | pre-run AICLK | read from the device before launch |
| X280 mtime rate | 50 MHz | `measurement::kX280MtimeHz` |
| LIM arena | 917,376 B | `PI_ARENA_BYTES` |
| Latency sample cache | 131,072 B (way 1) | `PI_CACHE_LIMIT_BYTES` |

## Ring Protocol Semantics

Each worker `w` owns one SPSC ring of `S` 128-byte slots at
`rings_offset + w·S·128` in the LIM arena. Rings share no state; the only
resource workers contend for is the consumer hart and the NoC path to it.
Symbols are those of [Benchmark Parameters](#benchmark-parameters).

### Counters

Every counter is 32 bits, wraps modulo 2^32, and is compared only through
`pi_counter_distance()`. None is reset at the core-sync barrier: the measured
phase continues from the warmup's final values.

| Counter | Owner | Where | Meaning |
|---|---|---|---|
| `r` | producer | private | Requests written into the ring. |
| `p` | producer | private, published as `produced[w]` in LIM | Requests made visible to the consumer. |
| `c_seen` | producer | private | Last credit the producer has observed in its consumed mirror. |
| `consumed` | consumer | `pi_consumer_slot_t` in LIM | Requests serviced. |
| `credited` | consumer | `pi_consumer_slot_t` in LIM | Last value written to the producer's L1 consumed mirror. |

By construction `c_seen ≤ credited ≤ consumed ≤ p ≤ r ≤ c_seen + S`. The
producer rejects an occupancy `r − c_seen` above `S` or a credit that retires
more than is in flight; the consumer rejects `produced − consumed` above `S`
(`PI_ERROR_COUNTER_DISTANCE`) and more available work than the phase has left
(`PI_ERROR_PHASE_OVERRUN`).

### Producer

For each request:

1. Build a fresh header in staging slot `pending`, where `pending < D = min(B, S)`.
2. Acquire space. With `r − c_seen < S` it proceeds at once. With the ring full it
   publishes what is pending and reads its consumed mirror: credit that has
   already arrived retires requests with no stall, otherwise it spins until the
   mirror moves. Backpressure mode records that spin as one stall episode.
3. Write the staging slot to ring slot `r mod S` on its assigned NoC, then
   `r++` and `pending++`.
4. Publish when `pending == B`, the ring is full, or this is the phase's last
   request: NoC barrier, then write `r` to `produced[w]`, and `pending = 0`.

The barrier in step 4 drains every header write of the batch, which is what
lets the next batch overwrite staging from slot 0; `D` slots suffice because a
publication always precedes `pending` reaching `B` or `S`.

The producer reads credit only with a full ring, and at phase end, where it
drains until `c_seen == r`. `c_seen` therefore lags `credited`, and a worker
whose ring never fills observes no credit until its phase ends.

### Consumer

X280 hart 0 visits workers in index order `0..P−1` — tile order from
`PACKET_INJECTION_TILES`, BRISC before NCRISC on a paired tile — and repeats the
round until every worker has completed the phase's count (`W`, then `N`). A
visit to `w`:

1. Skips `w` if it has finished the phase.
2. Loads `produced[w]`; `available = produced − consumed`. Zero moves to the next
   worker.
3. Fences, then services `min(available, K)` slots in ring order: the four
   destination-ID byte loads, the remote timer read in latency mode, then
   `consumed++`.
4. Returns credit whenever `consumed − credited` reaches `C`, and once more at
   visit end if any is outstanding. A credit is a mode-1 store of `consumed`
   into the worker's L1 consumed mirror.

A visit servicing `v` slots therefore issues `⌈v / C⌉` credit writes, and every
visit ends with `credited == consumed`.

### How the parameters shape the protocol

- **`S` (ring depth):** the maximum number of outstanding requests from a worker
  before it stalls for credit.
- **`B` (producer batch):** requests a worker writes before publishing them to the
  consumer. A full ring publishes early, so `B ≥ S` behaves as `B = S`.
- **`K` (visit cap):** the most requests the consumer takes from one worker before
  moving to the next; the consumer's scheduling quantum.
- **`C` (credit batch):** consumed requests the consumer accumulates before
  returning credit. A visit always ends with a credit, hence `C ≤ K`.
- **`W` (warmup):** unmeasured requests per worker, fully drained before the
  measured phase begins.
- **`N` (measured):** measured requests per worker; also sizes sample storage.
- **`T`, family, `TILES` (placement):** the number of rings and the order the
  consumer services them.
- **`POLL_CADENCE`:** how often both sides check for a deadline or STOP; no effect
  on the protocol.

Credit is visible to a producer only when its ring fills, so `C` and `K` shape
producer behaviour only where the consumer falls behind — small `S`, or enough
workers that a round takes longer than `S` requests.

## ERISC Header-Ingress Member

`packet_injection/erisc_to_x280_header` (ID 14) measures a receiving ERISC publishing
ready 96-byte packet headers into the X280 forwarder, scaled across concurrent
writers. One to all available inactive `IDLE_ETH` cores, one per Ethernet tile,
each run the shared `Producer` against their own SPSC ring into the one shared
X280 consumer image (the throughput variant). There is no second single-writer
implementation: `writers=1` is the same loop, protocol and result schema as every
larger point. § Measured operation below states what is timed, and § Credit-protocol cost
compares it with the single-writer prototype it replaced.

**Axes.** The GTest executable `x280_packet_injection_erisc_to_x280_header`
registers `Axes/PacketInjectionEriscToX280Header.Run/<writers>_b<B>`:
`writers1`, `writers2`, `writers4`, `writers8` and `writers_all` at B=1, and B=4
and B=8 at `writers1` and at the four-writer deployment fan-in. With the default
S=16, B=8 is the largest batch that leaves the ring double-buffered; B=S would
publish the whole ring and then wait for all of it. The writer count and B come
only from the axis; the suite rejects `PACKET_INJECTION_B`, `_T`, `_TILES` and
`_FAMILY`. `PACKET_INJECTION_S`, `_C`, `_K`, `_W` (calibrated when unset; see
below), `_N` (default 1024), `_POLL_CADENCE` and the two timeouts are read as for
the Tensix member.
`characterization_manifest.json` registers all nine points as
`packet.erisc_header.<writers>.b<B>` at N=1024, S=16, C=1, K=1, with W calibrated,
the per-header-credit reference. The consumer settings are a manifest axis, not a
GTest one: `packet.erisc_header.c<C>k<K>.<writers>.b1` runs the same filter at
`writers1` and `writers_all` with (C, K) in {(1, 8), (1, 16), (4, 8), (4, 16),
(8, 8), (8, 16)}, which with the two C=1/K=1 rows sweeps C in {1, 4, 8} by K in
{1, 8, 16}, C <= K, at S=16. K is the X280 visit cap (slots consumed per writer
per round-robin visit) and C the credit batch (a credit after C consumed slots,
and at the end of every visit), so K=1 is one credit write per header. The
campaign validator checks every packet run record's S, C, K and W against the
configuration, since those rows share a filter; where an ERISC configuration sets
no W, the run must record a calibrated one. At C=8, K=16, the setting with
the highest `writers_all` aggregate in that sweep, `packet.erisc_header.c8k16.*`
also registers `writers2`, `writers4` and `writers8` at B=1, B=4 and B=8 at
`writers1` and `writers4`, and `writers_all.b1.n4096` (N=4096) for the N versus
4N agreement check; `packet.throughput.brisc.T1.c8k16` runs one Tensix producer
at the same consumer settings as the one-producer cross-check.
At S=32 and S=64, `packet.erisc_header.s<S>.c<C>k<K>.<writers>.b<B>` registers
`writers1` and `writers_all` at B=1 and `writers1` at B=4 for (C, K) in
{(8, 16), (8, 32), (16, 32), (32, 32)}; the 2026-10-14 whole-suite campaign
measured them ([results.md](../results.md#packet_injection)).
At the Tensix load knee T=4, `packet.throughput.{brisc,paired}.T4.<setting>`
registers `c4k8` (C=4, K=8), `c8k16` (C=8, K=16), `s4`, `s8` and `s64` (S=4, 8
and 64) and `b4` (B=4), each changing that one setting from its `T4` point at
S=16, B=C=K=1; the 2026-10-14 whole-suite campaign
measured them ([results.md](../results.md#packet_injection)).
To isolate the (C, K) gain, `packet.throughput.{brisc,paired}.T4.s64.c<C>k<K>`
registers C in {1, 2, 4, 8, 16} at K=16 and K in {2, 4, 8} at C=1, each
changing only C and K from its `T4.s64` point at S=64, B=C=K=1; the 2026-10-14 whole-suite campaign
measured them ([results.md](../results.md#packet_injection)).
Across the load sweep, `packet.throughput.paired.T1.c8k16` and
`packet.throughput.{brisc,paired}.T{2,8}.c8k16` set C=8, K=16 on their `T<n>`
point at S=16, B=1; the 2026-10-14 whole-suite campaign
measured them ([results.md](../results.md#packet_injection)).

**Writers.** `select_ethernet_writers()` in `tests/device/suites/packet_injection/regions.hpp`
sorts the inactive cores by logical (x, y) and takes a prefix, so the one-writer
core is present in every point. `writers_all` is every available core, capped at
`kMaxEthernetWriters` (14). A point needing more cores than the device has skips
with the requested and available counts; it never runs a smaller count. A device
with no ERISC writer at all skips every point: when the HAL does not enable
`IDLE_ETH` programmable cores, and when it reports no inactive Ethernet core,
which a `writers_all` request names as such. All three are availability skips,
raised before any launch: the device cannot host the point, and nothing was
measured. The campaign tools draw no distinction: `run_characterization.py`
and `tabulate_suites.py` fail every skipped point, so an availability skip
leaves that configuration uncovered. A duplicate core fails the run. Admission uses `WorkerFamily::Ethernet`, one worker
per tile (P == T).

**Transport.** `DeviceEriscTransport` (`kernels/packet_injection/erisc_producer.hpp`)
shares its NoC writes, barrier, cache invalidation and `rdcycle` with
`DeviceTensixTransport` through `MetalProducerTransport`. It reads
`eth_read_wall_clock()` as its wall counter, uses NoC 0 (asserted against
`NOC_INDEX`), bounds the tile span by `MEM_ERISC_MAX_SIZE` (host mirror 455,264,
asserted in the device build), records `PI_ACTOR_ERISC` on failure, and its
`prepare_header()` does nothing: the host seeds the header. The image asserts
`PI_MODE_THROUGHPUT` with creation excluded. The kernel compiles at `-Os`, the
EthernetConfig default and what the prototype ran.

**Placement and seeding.** Each writer's slice sits at the aligned base of the
IDLE_ETH unreserved window (the suite requires the HAL window to end at the
producer's `kL1SpanLimit`). The host zeroes the whole tile span through the
cluster, writes a writer-specific header, word k = (`run_nonce` + k) XOR a
per-writer constant, into every one of the `D = min(B, S)` staging records, and
reads the span back before launch. The X280 tile table holds each writer's
virtual (TRANSLATED) coordinate, which the X280 window takes; the X280 peer
coordinate comes from `X280Runtime::virtual_core_from_tile`. `timeout_ticks` is
converted at the programmed AICLK (CMFW `AICLK_LIMIT_MAX`), the rate the run's
AICLK evidence checks; the run record names it `erisc_timeout_budget_hz`.

**Release.** The release selector gives Ethernet writers no group, so X280
releases every writer by its own unicast write, and the run fails unless the
run result reports `P` endpoints and no group.

**Warm-up.** With `PACKET_INJECTION_W` unset, a pilot launch at W=32 records
nothing, and `calibrate_warmup` sizes W from writer 0's measured interval at the
programmed AICLK so that W+N reaches the AICLK target device time
([docs/measurement.md](../measurement.md)), refusing a W whose device time exceeds the
host timeout. A refusal fails the case. Records carry `warmup_source`
(`calibrated` with the pilot figures, or `explicit` for a set W, used as given).

**Validation.** Counters as the Tensix member checks them, plus, before the
collection ACK, every resident ring slot of every writer -- the first
min(S, W+N) -- compared in full: bytes 0..95 against that writer's header and
96..127 against the zeroed staging tail the 128-byte write carries. The 60
padding bytes of every produced and ready slot must still be zero. Because the
ring recycles slots and each writer re-sends one record, completeness rests on
the counters, not on content.

**Records.** The run record carries `writers_requested`,
`inactive_eth_available`, and the resident and guard counts. The bytes the
writers published into LIM -- 96 per header plus 4 per produced-counter write,
P*N*96 + 4 * the workers' summed `producer_counter_writes`, which is P*N*100 at
B=1 -- are derived offline from the worker records. As for the Tensix member,
the X280 aggregate's raw endpoints and `empty_visits` are in the run's aggregate
record, with the fixed 50 MHz `x280_mtime` clock. Run and
worker records are measurement schema 2 with per-run AICLK evidence (`clock`),
taken on writer 0's own ERISC wall clock immediately before launch and after
`Finish` and applied to every writer, on the assumption that all idle ERISCs on
the chip share one AICLK rate; per [docs/measurement.md](../measurement.md) that evidence
converts the idle ERISC's `eth_read_wall_clock()`, so writer intervals convert to
time offline when it is clean. Worker records carry logical, virtual, translated
and NOC0 (physical) coordinates (the virtual one is what the X280 tile table
holds, so it is where X280 aims each writer's credits and release), raw ERISC
wall ticks (`wall_counter` `erisc_wall_clock`) and `rdcycle`,
`credit_stall_episodes` and `producer_counter_writes`. AICLK evidence with errors
fails the test after every record is written, and the offline tool then leaves
those ticks raw.

**Dispatch.** The executable sets `TT_METAL_SLOW_DISPATCH_MODE=1` through its
`x280_slow_dispatch` link: fast dispatch does not launch IDLE_ETH kernels, and an
all-zero ERISC result there is an invalid launch, not a measurement. The fixture
fails before the device opens, naming the missing link, if the variable is not `1`.

```bash
env TT_VISIBLE_DEVICES=0 TT_METAL_RUNTIME_ROOT="$PWD" \
  build/test/tt_metal/x280/x280_packet_injection_erisc_to_x280_header \
  --gtest_filter='Axes/PacketInjectionEriscToX280Header.Run/writers2_b1'
```

**Silicon (2026-09-30).** A Blackhole p150b, tile 0 at
1750 MHz with 14 LIM ways, exposes 8 inactive IDLE_ETH cores, so `writers_all`
is 8 there and `writers8` duplicates it. Every point passed at N=1024, S=16, C=1,
K=1 with W calibrated, every resident slot and guard byte intact, and clean AICLK
evidence at 1350 MHz on writer 0's ERISC wall clock. The records carry
`source_dirty` true: they were collected before the member was committed.

| Point | B | P | W | Aggregate mtime ticks | M headers/s | Per-writer wall ticks | Stall episodes per writer | Empty visits | Resident slots / mismatches |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `writers1` | 1 | 1 | 33,680 | 36,898 | 1.388 | 995,898 | 992 | 0 | 16 / 0 |
| `writers2` | 1 | 2 | 16,543 | 72,894 | 1.405 | 1,967,523..1,967,616 | 1003 | 0 | 32 / 0 |
| `writers4` | 1 | 4 | 7,809 | 144,969 | 1.413 | 3,913,017..3,913,394 | 1006 | 0 | 64 / 0 |
| `writers8` | 1 | 8 | 3,404 | 289,171 | 1.416 | 7,805,486..7,806,391 | 1007..1008 | 0 | 128 / 0 |
| `writers_all` | 1 | 8 | 3,404 | 289,143 | 1.417 | 7,804,725..7,805,631 | 1007 | 0 | 128 / 0 |
| `writers1` | 4 | 1 | 33,618 | 36,957 | 1.385 | 997,772 | 1004 | 1 | 16 / 0 |
| `writers1` | 8 | 1 | 33,607 | 36,966 | 1.385 | 997,958 | 1006 | 3 | 16 / 0 |
| `writers4` | 4 | 4 | 7,807 | 144,990 | 1.413 | 3,913,609..3,914,037 | 1008 | 0 | 64 / 0 |
| `writers4` | 8 | 4 | 7,807 | 144,990 | 1.413 | 3,913,588..3,913,989 | 1008 | 0 | 64 / 0 |

Aggregate ingress does not scale with writers at this consumer configuration:
the X280 interval grows in proportion to P and stays at about 1.4 M headers/s.
Empty visits stay at zero while almost every header enters the credit wait,
which is the X280-service signature in the plan's discriminating checks; batching
does not move it at C=1/K=1. ERISC wall and `rdcycle` deltas agree within two
ticks, and one writer's wall delta over the aggregate interval implies about
1.35 GHz, matching the AICLK evidence. The scaling campaign below re-ran these
nine points and reproduced them within 0.09%.

**Scaling campaign (2026-09-30).** Card 0 as above, three runs per point through
`run_characterization.py`; provenance and raw JSONL are the archived record
`benchmarks/campaigns/20260929-erisc-header-ingress-scaling/` ([results.md](../results.md#erisc---x280-header-ingress)).
Rates are X280 `mtime` aggregates, mean [min, max] over the runs; stall
episodes, counter writes and wall ticks are per writer. W is calibrated per point.
ERISC ticks are shown raw and convert offline under each run's clean AICLK
evidence, taken on writer 0's ERISC wall clock.
Every figure is for inactive idle ERISCs and does not show that an ERISC doing
live Ethernet receive and router work has the same cycles left. No target is
asserted. The re-run C=1, K=1 table, the derivations and the Tensix cross-check
are in [`docs/results.md`](../results.md#erisc---x280-header-ingress).

The consumer-settings sweep (S=16, N=1024, B=1):

| C | K | writers1 M headers/s | writers1 stalls | writers1 empty | writers_all M headers/s | writers_all stalls per writer | writers_all empty |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 1.386 [1.386, 1.386] | 992 | 0 | 1.417 [1.416, 1.417] | 1,007 | 0 |
| 1 | 8 | 1.965 [1.965, 1.965] | 963..964 | 0 | 1.976 [1.976, 1.976] | 999..1,008 | 0 |
| 1 | 16 | 2.017 [2.016, 2.017] | 958..959 | 0 | 2.033 [2.033, 2.033] | 994..1,008 | 0 |
| 4 | 8 | 2.550 [2.550, 2.551] | 0 | 0 | 3.810 [3.809, 3.810] | 125..126 | 0 |
| 4 | 16 | 2.549 [2.548, 2.551] | 0 | 0 | 4.029 [4.029, 4.030] | 63 | 0 |
| 8 | 8 | 2.548 [2.548, 2.548] | 0 | 0 | 4.516 [4.515, 4.516] | 126 | 0 |
| 8 | 16 | 2.548 [2.548, 2.548] | 0 | 0 | 4.824 [4.823, 4.824] | 63..64 | 0 |

At C=8, K=16, the highest `writers_all` aggregate (6.8% above C=8, K=8; spreads
at most 0.14%), speedup is against `writers1` at the same B and efficiency is
speedup/P:

| Point | P | B | M headers/s | Logical MB/s | Publication MB/s | Speedup | Efficiency | Slowest/fastest writer | Empty visits | Stall episodes per writer | Counter writes per writer | Per-writer ERISC wall ticks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `writers1` | 1 | 1 | 2.548 [2.548, 2.548] | 244.6 | 254.8 | 1.00 | 1.00 | 1.0000 | 0 | 0 | 1,024 | 542,457..542,633 |
| `writers2` | 2 | 1 | 4.532 [4.527, 4.534] | 435.0 | 453.2 | 1.78 | 0.89 | 1.0018 | 0 | 76 | 1,024 | 608,595..610,561 |
| `writers4` | 4 | 1 | 4.817 [4.816, 4.818] | 462.4 | 481.7 | 1.89 | 0.47 | 1.0073 | 0 | 63 | 1,024 | 1,140,090..1,148,696 |
| `writers8` | 8 | 1 | 4.824 [4.823, 4.824] | 463.1 | 482.4 | 1.89 | 0.24 | 1.0129 | 0 | 63..64 | 1,024 | 2,268,123..2,297,678 |
| `writers_all` | 8 | 1 | 4.824 [4.823, 4.824] | 463.1 | 482.4 | 1.89 | 0.24 | 1.0129 | 0 | 63..64 | 1,024 | 2,267,865..2,297,441 |
| `writers1` | 1 | 4 | 4.459 [4.459, 4.460] | 428.1 | 432.6 | 1.00 | 1.00 | 1.0000 | 1 | 125..126 | 256 | 309,818..310,052 |
| `writers4` | 4 | 4 | 4.813 [4.812, 4.813] | 462.0 | 466.9 | 1.08 | 0.27 | 1.0100 | 0 | 126 | 256 | 1,139,394..1,151,031 |
| `writers1` | 1 | 8 | 4.459 [4.459, 4.460] | 428.1 | 430.3 | 1.00 | 1.00 | 1.0000 | 3 | 126 | 128 | 309,863..309,869 |
| `writers4` | 4 | 8 | 4.814 [4.813, 4.814] | 462.1 | 464.5 | 1.08 | 0.27 | 1.0096 | 0 | 126 | 128 | 1,137,499..1,148,473 |

`writers_all` agrees between N=1024 (4.8238 M headers/s) and N=4096 (4.8273), a
0.07% difference.

Reading, by the empty-visit versus stall-episode check: at C=1, K=1 every point
is X280-service bound (about 1,000 stall episodes per 1,024 headers, empty
visits at most 3), and that service time is mostly credit writes -- at K=16 the
X280 cost per header falls linearly with credits per header, about 570 X280
cycles per credit. At C=8, K=16 four or more writers plateau at 4.82 M
headers/s, still X280-service bound (about one stall episode per ring, no empty
visit), and B=4 or 8 does not move it. One writer at B=1 enters neither wait;
batching it to B=4 lifts it from 2.55 to 4.46 M headers/s, so its own per-header
publication paces it, an attribution from the response to B rather than from
the counters. C=8, K=16 is the edge of the S=16 grid, so where larger visits and
credit batches stop helping is not measured.

The first `writers1` attempt, on 2026-09-29 against firmware that still released
each tile through a one-tile multicast group, failed at the release
(`PI_ERROR_STOPPED` from the producer, `PI_ERROR_PEER_FAILED` from X280): a
one-tile broadcast does not reach an Ethernet tile.

### Measured operation

On production receive, ERISC owns two ordered destinations:

```text
Ethernet RX staging
    |
    +-- full packet ----------> selected DRISC slot
    |                            wait for copy completion
    |
    +-- 96 B header ----------> X280 FIFO slot
                                 drain header write
                                 publish
```

The X280 must not observe a header before its packet is resident in DRISC SRAM.
The benchmark starts from a backlog for which that invariant is already true;
the timed ERISC code performs only FIFO publication. This makes the result a
ceiling for the header-ingress operation rather than a measurement of Ethernet
receive or the full receive-side pipeline.

The unified member publishes with the packet-injection ring protocol: the slot
write, a write barrier, then the ring-tail produced counter, which X280 polls
with a fence before the counter load and another before the slot loads (see
[Architectural Overview](#architectural-overview)).
Comparing that encoding with a per-slot or in-header valid flag, such as the
prototype's, is Fabric 2.0 ID 15.

### Credit-protocol cost

The prototype never waits on its consumer: its X280 validation pass runs after
the ERISC batch ends. The unified member's writer waits for ring credit from the
X280 consumer, so the difference between the two per-header costs is the cost of
the credit protocol at the unified member's settings. On 2026-09-30 the
prototype's `DRAINED` point took 420.17 ERISC wall ticks per header and
`writers1_b1` (S=16, C=1, K=1) 973.4, so the credit protocol costs 553.2 ticks per
header there. 992 of 1,024 measured headers entered the credit wait: the writer
is paced by X280 service and credit return, not by its own publication code.
Conditions, both intervals and the evidence are in
[results.md](../results.md#erisc---x280-header-ingress).

## Running the Benchmark

`x280_packet_injection_tensix_to_x280` injects fabric packet headers from
Tensix BRISC/NCRISC producers into an X280 consumer, and
`x280_packet_injection_erisc_to_x280_header` from 1 to 14 idle ERISCs. Both
open the 14-way `LimLayout::PacketInjection` profile. The ERISC member links
`x280_slow_dispatch` ([harness.md](../harness.md#running-benchmarks)). The header-ingress member, which
runs on the packet-injection throughput image, is in
[§ ERISC Header-Ingress Member](#erisc-header-ingress-member), with the ID 14
contract under [§ Measured operation](#measured-operation).
Each record names the tt-metal revision the executable was built from
(`tt_metal_sha`, `source_dirty`).

The host executable build is not a Tensix kernel build. BRISC/NCRISC JIT
compilation against Metal's device-generated configuration is verified by
`PacketInjectionCompile.BothRiscsCompileForEveryVariant`, which compiles both
data-movement RISCs for all five mode/creation-flag variants on a live device
without enqueue or dispatch.

The schema-2 baseline consists of seventeen successful p150b runs at UMD device 0,
all against one set of binaries, covering every mode and
every variant: throughput at P=1, 2, 4 and 8 across BRISC-only and paired shapes;
`latency-include` and `latency-exclude` at N=1 and N=3, each run twice; and
backpressure in both creation variants at P=1 and at P=2. Every run's producer and
consumer counted the same traffic independently, per worker, and every run's
`arena_bytes` matched the layout's compile-time golden for its shape.

Those measurements predate schema 3 and do not qualify the core-sync boundary.
They remain the throughput, latency, and backpressure baseline; their five binaries
were byte-identical to a clean build of the schema-2 commit that recorded them.

On 2026-09-22, schema 3 was exercised on Blackhole device 0 against the
tt-metal of the time. Both BRISC and NCRISC compiled for all five variants.
All five variants then completed on silicon. Focused synchronization runs covered
one BRISC tile, one paired BRISC/NCRISC tile, a dense four-tile row released as
one rectangle, two sparse tiles released as ordered singletons, and B=4/C=3/K=5
warmup and measured traffic. A producer rejected before arrival published its
original `PI_ERROR_BAD_ARGS`; host polling observed `PI_READY_FAILED`, issued
runtime STOP, and the X280 recovery release completed collection and cleanup.
The opt-in unannounced-failure case rejected the producer's peer coordinate,
left completion at `PI_READY_NONE`, reported `data_quiescent=0`, and entered
`retain_failed_run()` after the recovery grace period; the reserved device was
then reset and reprovisioned.
These runs used an uncommitted working tree, so they
qualify the implementation behavior but are not reproducible commit evidence.

**L1 page order is confirmed ordered rather than assumed.** The strongest case is
T=4 paired, where eight producers occupy eight pages and the host's per-page
identity check passed on all of them, with worker `w` landing on tile `w >> 1`
and RISC `w & 1` — four tiles, both NoCs, in ascending page order. Note that
`family=paired` doubles the worker count, so T=4 paired is P=8, not P=4.

The assumption mask below remains an explicit limitation of the benchmark
contract, not a statement about whether the benchmark runs.

Build in a tt-metal tree configured with `TT_METAL_BUILD_X280` ([README](../../README.md#2-build-boot-and-run-a-benchmark)), from the repository root:

```bash
# Images (all five variants), header checks, host tests and the executable
ninja -C build x280_images x280_abi_headers x280_bench_host_tests \
  x280_packet_injection_tensix_to_x280

# Host-only tests (admission, ABI, state-model, layout), image placement and
# the pinned-input and packet-injection selection guards
ctest --test-dir build -R 'x280_(bench_host_tests|image_layout|pinned_input|packet_injection)' --output-on-failure

# Execution example (requires exclusive device reservation); the point's image
# defaults to build/x280/firmware-packet-injection-latency-include.bin
env TT_METAL_RUNTIME_ROOT="$PWD" \
  PACKET_INJECTION_TILES="0,0" PACKET_INJECTION_T=1 \
  PACKET_INJECTION_FAMILY=brisc PACKET_INJECTION_S=16 \
  PACKET_INJECTION_B=1 PACKET_INJECTION_C=1 PACKET_INJECTION_K=1 \
  PACKET_INJECTION_W=1000 PACKET_INJECTION_N=4096 \
  build/test/tt_metal/x280/x280_packet_injection_tensix_to_x280 \
  --gtest_filter='Config/PacketInjectionTensixToX280.Run/LatencyInclude'

# Producer JIT compile gate: no firmware image, no prepared runtime, no dispatch
env TT_VISIBLE_DEVICES=0 TT_METAL_RUNTIME_ROOT="$PWD" \
  TT_METAL_CACHE="$PWD/build/jit-cache-bu1" TT_METAL_FORCE_JIT_COMPILE=1 \
  TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1 \
  build/test/tt_metal/x280/x280_packet_injection_tensix_to_x280 \
  --gtest_filter='PacketInjectionCompile.BothRiscsCompileForEveryVariant'
```
