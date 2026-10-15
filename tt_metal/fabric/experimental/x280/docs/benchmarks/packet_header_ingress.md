# Packet-Header Ingress Characterization

**Status:** Implemented by `packet_injection/erisc_to_x280_header` and
silicon-validated at every registered point on Blackhole with the pinned
tt-metal revision. ERISC records retain raw ticks and CPU cycles with the AICLK
evidence that brackets each run, from which nanoseconds are derived offline when
it is clean; aggregate ingress rates come from the X280's 50 MHz `mtime` interval.

**Scope:** Phase 0 feasibility benchmark ID 14, X280 packet-header ingress
bandwidth. It measures the receiving ERISC publishing packet headers from an
already-populated receive staging area into the X280 forwarder's header FIFO.
Ethernet arrival, header construction, the packet copy to DRISC, and waiting
for that packet copy to complete are preconditions outside the measured region.

**Implementation:** `packet_injection/erisc_to_x280_header` implements ID 14
for one to all available ERISC writers and publication batch sizes, on the shared
packet-injection producer, X280 consumer, credit protocol and host stages. Its
axes, ABI, validation, records, reproduction and silicon results are in
[docs/benchmarks/packet_injection.md](packet_injection.md#erisc-header-ingress-member).
Its `writers=1,B=1` point supersedes the `transport_bandwidth/erisc_to_x280_header`
prototype described under [Prototype Member](#prototype-member-superseded),
which has been removed.

## Architectural Contract

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
with a fence before the counter load and another before the slot loads (see the
[packet-injection architectural overview](packet_injection.md#architectural-overview)).
Comparing that encoding with a per-slot or in-header valid flag, such as the
prototype's, is Fabric 2.0 ID 15.

## Credit-Protocol Cost

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

## Prototype Member (Superseded)

`transport_bandwidth/erisc_to_x280_header` was the first single-writer
implementation of ID 14. Its results are prototype evidence, superseded by the
unified member's `writers=1` row. The member has been removed, and this section keeps its contract so its results
can be read. It published each header with a sideband
`pkt_valid` word rather than a ring counter, and never reused a slot:

```text
ERISC                                      X280
write 96 B header body
noc_async_write_barrier()
write sideband pkt_valid  ------------->  fence before polling pkt_valid
                                           observe pkt_valid
                                           fence before reading header
                                           validate all 96 B
                                           clear pkt_valid
```

### FIFO and ABI

`transport_bw_header_slot_t` is a 128-byte, cache-line-aligned record:

| Offset | Bytes | Owner | Meaning |
|---:|---:|---|---|
| 0 | 96 | ERISC | Packet-header bytes copied from staging |
| 96 | 4 | ERISC/X280 | Sideband `pkt_valid`: producer sets, consumer clears |
| 100 | 28 | — | Padding to the 128-byte slot stride |

The deployed 96-byte HybridMesh header has no `pkt_valid` member. The sideband
word therefore does not alias or repurpose header data. Logical ingress bytes
count only the 96-byte header; the validity write and slot padding are reported
as protocol geometry, not useful header bytes.

Warmup and measurement use distinct successive slots. The producer never wraps
or waits for consumer credit inside the timed batch. The host allocates a slot
for every warmup and measured publication, with zeroed cache-line guards before
and after the FIFO.

Warm-up calibration (`calibrate_warmup`) does not apply to this member: its
warm-up is LIM-bounded, because each header uses a distinct slot, so the W+N
slots that available LIM holds (about 14,600 at 15 ways) cap the warm-up near
4.5 ms of device work, under the 25 ms calibration target. `TRANSPORT_BW_HEADER_W`
(default 128) is used as given and the AICLK elapsed gate judges each run; on
the prototype's own ERISC reference its elapsed windows measured 10.5-17.7 ms
and were clean in every run of the 2026-09-30 re-run below.

### Measurement Contract

The member extends the existing `transport_bandwidth` suite rather than adding
a timing stack. The shared frontend owns verification, warmup, timing, drain,
publication, and terminal states.

| Boundary | Timed interval |
|---|---|
| `DRAINED` | `N` header-body/drain/valid publications plus completion of the final valid write |
| `INJECTION` | `N` header-body/drain/valid publications; completion of the final valid write occurs after the endpoint |

The body-completion barrier is part of every production publication and is
inside both boundaries. Only the final `pkt_valid` completion distinguishes the
two modes.

ERISC records two native intervals around the same bracket:

- `raw_batch_delta`: `eth_read_wall_clock()` ticks;
- `secondary_batch_delta`: architectural `rdcycle` ticks.

Both deltas use `measurement_delta()` and reject half-range or invalid samples.
The host brackets the ERISC launch through
`Finish` with the AICLK sampler on the selected ERISC's own wall clock and emits a
measurement-schema-2 batch record: the raw `eth_wall_clock` and `rdcycle`
deltas, and that evidence as `clock`. Per [docs/measurement.md](../measurement.md) the
evidence converts idle-ERISC `eth_read_wall_clock()`; the offline tool derives
nanoseconds only when it carries no errors, and no nominal ERISC clock is
substituted. AICLK evidence with errors fails the test after the record is
written.

### Host Orchestration and Validation

The host:

1. selects one inactive `IDLE_ETH` core;
2. allocates admitted ERISC L1 staging, scratch, valid-source, and result spans;
3. fills the 96-byte staging record before launch and verifies host readback;
4. allocates guarded FIFO, verification, result, and argument regions in X280
   available LIM;
5. runs the ERISC producer through Metal;
6. runs the active X280 consumer after ERISC measurement completes;
7. validates every header byte, cleared validity word, guard, count, terminal
   state, logical-byte count, and raw measurement interval.

The separate X280 validation pass keeps header inspection outside the ERISC
measurement. It also proves the same consumer-side fence and validity semantics
the forwarder uses without requiring FIFO reuse or application credits.

Argument and placement validation fails before transport. The ERISC publishes
`ARG_LAYOUT_MISMATCH` when its admitted result region is available. The X280
consumer publishes `BAD_ARGS`, `HEADER_MISMATCH`, `BAD_VALID`, `TIMEOUT`, or
`STOPPED`; only missing or unsafe result storage is unpublishable.

### Silicon Validation

Prototype evidence, superseded by the unified member's `writers=1` row.
Validated on 2026-09-23:

| Boundary | Warmup | Samples | Logical bytes | Raw wall ticks | CPU cycles |
|---|---:|---:|---:|---:|---:|
| `DRAINED` | 1 | 4 | 384 | 2,075 | 2,075 |
| `INJECTION` | 1 | 4 | 384 | 1,754 | 1,755 |
| `DRAINED` | 128 | 1024 | 98,304 | 430,219 | 430,219 |
| `INJECTION` | 128 | 1024 | 98,304 | 429,890 | 429,891 |

These runs validate the functional path and raw measurement contract. They are
not a canonical production-rate capture; they predate the AICLK evidence, so
their records carry raw counters only.

Re-run on 2026-09-30 on card 0 of a Blackhole p150b host,
with AICLK evidence on the ERISC's own wall clock, clean at 1350 MHz:

| Boundary | Warmup | Samples | Logical bytes | Raw wall ticks | CPU cycles |
|---|---:|---:|---:|---:|---:|
| `DRAINED` | 1 | 4 | 384 | 2,067 | 2,067 |
| `INJECTION` | 1 | 4 | 384 | 1,746 | 1,747 |
| `DRAINED` | 128 | 1024 | 98,304 | 430,211 | 430,211 |
| `INJECTION` | 128 | 1024 | 98,304 | 429,850 | 429,851 |
