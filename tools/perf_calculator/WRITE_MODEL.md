# Non-posted DRAM writes

`predict(topology, case, issue_cycles=(noc0_cost, noc1_cost))` accepts
`traffic="write"`, with `mode="noc0"`, `"noc1"`, or `"split"`.
The default remains reads. Payloads are one packet (1–16384 bytes); batches must
be complete. Bank placement is balanced or bank-0 hotspot. Split assigns half
each batch to each NoC, running independently. This is a transport schedule,
not an operation-specific compute pipeline.

## Sources and assumptions

The [Blackhole ISA NoC documentation](https://github.com/tenstorrent/tt-isa-documentation/blob/main/BlackholeA0/NoC/README.md)
and [command registers](https://github.com/tenstorrent/tt-isa-documentation/blob/main/BlackholeA0/NoC/MemoryMap.md)
describe 64-byte flits, one header flit, up to 256 payload flits, marked write
acknowledgements, and nominal router/NIU timings. Local inspected ISA revision:
`8a1d28fb06a3d6d4eb5aa22ec0aa6795e8ea902b`.

Ordinary `noc_async_write` defaults to non-posted in
`tt_metal/hw/inc/api/dataflow/dataflow_api.h`. Its write barrier checks received
acknowledgements through `ncrisc_noc_nonposted_writes_flushed` in
`tt_metal/hw/inc/internal/tt-1xx/blackhole/noc_nonblocking_api.h`.
`noc_async_writes_flushed` instead checks source departure. These endpoints are
not interchangeable. Here completion means remote acknowledgement, not a claim
about physical DRAM array persistence.

A write request contains `1 + ceil(payload / 64)` flits; its acknowledgement
contains one flit. Each follows the directed physical NoC route and competes
for injection, links, and ejection. Nominal delays are 5 cycles at each NIU
endpoint and 9 cycles per router hop. At 1350 MHz, 9 cycles is 6.667 ns.

After the payload tail arrives, a shared bank resource reserves
`ceil(payload / 47.0805)` cycles. The acknowledgement starts 22 cycles after
bank service begins. Payload serialization is not added again.
Both NoCs share this bank resource.

**The 22-cycle delay and 47.0805 bytes/cycle are provisional for writes.**
Their source is the measurement-derived
[Wolfpine simulator unit costs](https://github.com/hedgehope/wolfpine/blob/c0d2cb8842c520c48e3a28aef66a167e674d100f/framework/perf/unit_costs.yaml),
not an ISA guarantee or official simulator specification. The write delay is
an incremental DRAM-versus-L1 latency estimate; bandwidth comes from bulk read
measurements. The simulator explicitly reuses the read-derived rate for write **occupancy**,
arguing bus symmetry; it does not add that drain time to isolated write latency.
Its notes report an independent p150 saturation experiment: 47.159 B/cycle read
and 46.815 B/cycle write. These are the simulator author's reported results,
not measurements repeated here. This supports investigating the shared occupancy
assumption, but does not establish write admission, credit capacity, or mixed
read/write turnaround costs.

The measured read-admission profile is never applied to writes.
`traffic="mixed", mode="split"` represents independent NoC0 reads and NoC1
writes sharing banks; mixed timing is experimental and unvalidated. It does
not express a producer/consumer copy dependency.

## Validation

Generic probe: `tests/ttnn/unit_tests/operations/examples/test_noc_transport.py`.
Select cases with `NOC_TRANSPORT_CASES` and output with `NOC_TRANSPORT_REPORT`.
Run through `scripts/run_safe_pytest.sh --no-precompile`.
The probe checks the entire output exactly, including untouched hotspot pages.
Timing is the maximum local per-core/RISC span divided by four iterations;
there is no synchronized cross-core start. Five trials follow a warmup.
Source data is initialized directly in L1 outside timing for write-only probes.

On BH at 1350 MHz, all 24 single-core cases passed on 2026-10-21:
64/128/1024/4096-byte payloads, batch sizes 1/8/16, both NoCs.
Only batch-16 single-core ISSUE scopes supply issue costs; total durations
are not fitted. Median absolute prediction error was 7.79%, all underestimates.
Batch-1 errors were 14.1–15.2%; batch-16 errors were 3.6–7.1%.
After charging the first write command its issue cost, median absolute error
improved to **1.35%**, with a maximum of **3.66%**. No bank constant was changed.
The same measured data informed this correction, so this is a development-set
comparison, not independent validation of the correction.

The write issuer schedules requests at `(j + 1) * issue_cycles`; the existing
read timing origin remains unchanged. Barrier/loop instructions are not explicitly modeled;
small batches are especially sensitive. Instrumentation also has overhead.
The single-core ISSUE scope is an effective issue interval and can contain
backpressure, especially for large packets; it is not a pure instruction cost.
Do not change bank latency to absorb these costs. Finite router queues,
backpressure/credits, and write-specific bank admission remain unmodeled.

Comparison command (paths are campaign artifacts, not required runtime files):

```sh
python_env/bin/python tools/perf_calculator/validate_transport.py \
  --topology tools/perf_calculator/topology_bh.json \
  --measurements /tmp/noc-write-validation/phase1_measured.jsonl \
  --issue-measurements /tmp/noc-write-validation/phase1_measured.jsonl \
  --output /tmp/noc-write-validation/phase1_comparison.json
```

## Multicore holdout results

All 36 write-only cases passed exact correctness: 32/96/110 cores,
128/1024-byte payloads, NoC0/NoC1/split, balanced/bank-0 hotspot, batch 16.
Predictions including the first-command correction were frozen before measurement.
See `write_validation_bh.json` for every comparison and source hashes.

At 110 cores, 1024-byte payloads (microseconds per 64 writes/core):

| Banks | NoC | Estimate | Measured | Error |
|---|---|---:|---:|---:|
| Balanced | 0 | 27.742 | 42.072 | -34.1% |
| Balanced | 1 | 17.297 | 24.649 | -29.8% |
| Balanced | split | 14.442 | 21.813 | -33.8% |
| Hotspot | 0 | 114.750 | 113.814 | +0.8% |
| Hotspot | 1 | 114.750 | 113.848 | +0.8% |
| Hotspot | split | 114.750 | 113.639 | +1.0% |

The bank-limited large-payload results support the occupancy assumption in this
regime. Balanced traffic exposes substantial missing transport behavior; finite
buffer/credit backpressure, arbitration, and outstanding-request limits are
hypotheses, not identified causes. Small-payload hotspots also have errors up to
9.7%. **Do not treat this model as an accurate general multicore write predictor.**
No empirical multicore correction factor has been added. Mixed read/write
turnaround remains unvalidated; pure-write results cannot validate it.

The ISA's virtual-channel section states that static VC ordering can leave
otherwise available bandwidth unused. The ordinary write API ultimately uses
`ncrisc_noc_fast_write`, which sets `NOC_CMD_VC_STATIC`. The current model does
not represent dateline/class/buddy VC reservations or finite buffer credits.
This is a documented mechanism worth testing, not proof that it explains the
observed gap. Numeric queue/credit limits were not found in the inspected docs.
