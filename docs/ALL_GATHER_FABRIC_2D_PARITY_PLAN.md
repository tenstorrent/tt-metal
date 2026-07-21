# Fabric 2D logical-ring bandwidth parity plan

## Objective

Bring an eight-device logical ring running on `FABRIC_2D` to within 10% of
the bandwidth of the same all-gather running on `FABRIC_1D_RING`. The first
target is the isolated sparse-MLA 512K-row workload; the final result must be
generic fabric and CCL behavior, with no model-name checks, environment
variables, or workload-specific routing gates.

The matched workload is:

- mesh: `8x1` logical ring on the current eight-device Blackhole system;
- rows per device: `65,536`;
- BF16 row/page size: `1,152 B`;
- scaled-FP8 row/page size: `704 B`;
- fabric packet payload: `14,336 B` in both configurations;
- two CCL links, receiver-L1 bank fan-out, and persistent interleaved DRAM
  output;
- Galaxy-compatible Ethernet firmware;
- release firmware only; never use `--dev`.

`effective_receive_bw` is the tensor bytes received by one rank from the other
seven ranks divided by the device-program duration. It is not aggregate wire
bandwidth across every link in the system.

## Baseline and acceptance gates

Measurements below use the same tensor geometry, packet size, all-gather
policy, firmware, and profiler calculation. Changing packet size invalidates
the comparison.

| Format | Fabric | Median | Effective receive BW | Gap to matched 1D |
| --- | --- | ---: | ---: | ---: |
| BF16 | `FABRIC_1D_RING` | 5.734 ms | 92.170 GB/s | baseline |
| BF16 | `FABRIC_2D` | 27.015 ms | 19.562 GB/s | 4.71x slower |
| scaled FP8 | `FABRIC_1D_RING` | 3.529 ms | 91.524 GB/s | baseline |
| scaled FP8 | `FABRIC_2D` | 16.297 ms | 19.818 GB/s | 4.62x slower |

Progress gates:

1. Attribution gate: identify the first hop, turn, VC transition, or scheduler
   stage at which sustained bandwidth falls below 80 GB/s.
2. Intermediate gate: at least 45 GB/s on both formats.
3. Productive-path gate: at least 70 GB/s on both formats.
4. Parity gate: at least 85 GB/s and within 10% of the matched Fabric1D run.
5. Stretch gate: at least 90 GB/s with no Fabric1D or model regression.

Every gate also requires exact output correctness, ten consecutive non-hanging
runs, and a p90 no more than 5% above the median.

## What the current evidence says

- Both dtypes plateau near 20 GB/s under Fabric2D despite their different byte
  counts. This points to the Fabric2D transport path, not tensor conversion or
  DRAM output, as the primary bottleneck.
- The same CCL worker, receiver-L1 fan-out, persistent output, hardware, and
  firmware sustain more than 91 GB/s under Fabric1D. Those stages therefore
  have enough capacity when the transport keeps them supplied.
- Sender-channel trimming improved Fabric2D by only about 5%, from roughly
  19.8 to 20.8 GB/s. Channel scanning is secondary, not the 4.6x root cause.
- Receiver-channel trimming caused a transport hang with workers waiting on
  semaphores. It must not be used as a performance workaround.
- Packet-service bursts of four and two channels also hung. Preserve strict
  fairness until the protocol and credit state prove that a different
  scheduler is live.
- Removing redundant L1 invalidation, using `-O3`, enlarging VC0 buffers, and
  increasing terminal-offload depth produced no material gain.
- The current Fabric2D firmware cannot fit the otherwise attractive `2,816 B`
  FP8 packet configuration. Firmware-size changes are not part of the first
  attribution experiments.

The leading hypothesis is serialization or poor overlap in the Fabric2D
receive-to-forward path, especially at direction changes or the dateline VC
transition. It remains a hypothesis until the hop ladder below locates the
first collapse.

## Measurement discipline

Use test-driven, one-variable-at-a-time A/B measurements:

1. Compile and warm both arms before recording samples.
2. Keep payload size, page size, link count, output placement, and worker
   policy identical.
3. Collect at least seven steady-state samples and report median, minimum,
   p90, and every sample.
4. Use A/B/A when drift can explain the result.
5. Retain a change only for at least a 3% median gain, no material p90 loss,
   exact correctness, and repeated stability.
6. Run hardware tests with `scripts/run_safe_pytest.sh`.
7. On any hang, reset safely and capture `tt-triage` state before changing
   synchronization.
8. Keep instrumentation small enough for the release ERISC firmware budget.
   Prefer existing profiler events and host-selected test variants over large
   diagnostic code in the router.

Canonical full-workload command:

```bash
TT_METAL_DISABLE_PRECOMPILED_FW=1 \
scripts/run_safe_pytest.sh \
  tests/ttnn/unit_tests/operations/ccl/test_all_gather_fabric_2d.py \
  -k 'large_single_axis_correctness or sparse_mla_row_perf' -x -s
```

## Phase 0: create a matched 1D/2D benchmark fixture

Before optimizing, make the current manual comparison a permanent parameterized
test fixture. It must run exactly the same BF16 and FP8 geometries under
`FABRIC_1D_RING` and `FABRIC_2D`, explicitly selecting the same `14,336 B`
payload. Log the physical path, hop count, direction changes, escape-VC hop,
and terminal-offload policy alongside the existing bandwidth fields.

This fixture prevents accidental comparisons against Fabric1D's default
`4,352 B` payload and provides the regression control for every later phase.

## Phase 1: locate the first bandwidth collapse

Add isolated, correctness-checked routes in this order:

| Experiment | Route property | Question |
| --- | --- | --- |
| A | one direct physical hop, no turn | Is explicit-path injection/receive itself fast? |
| B | two or more straight hops | Does ordinary transit forwarding sustain line rate? |
| C | same hop count with one corner | What is the cost of receive-to-local-relay-to-egress forwarding? |
| D | straight route crossing the dateline | Does switching from VC0 to VC1 serialize traffic? |
| E | one corner plus dateline transition | Do corner relay and escape VC interact? |
| F | complete bidirectional 8x1 ring | Does contention or plane imbalance appear only at full load? |

For every row, run normal Fabric2D routing and the explicit-path form where
both are legal. Measure each routing plane separately before combining them.
The first experiment below 80 GB/s determines the next implementation phase;
do not start with full-ring scheduler tuning.

Minimal counters should distinguish:

- ingress packet available wait;
- downstream sender space/credit wait;
- local relay space wait;
- packet forwarding and completion count per direction;
- VC0 and VC1 packet count;
- terminal-offload queue full wait.

Counters must be compile-time test instrumentation or existing telemetry, not
new runtime environment-variable behavior.

## Phase 2: optimize straight transit if it is the first collapse

If experiment B fails while A passes, focus on the ERISC
receive-to-downstream-sender pipeline:

1. Establish whether the incoming packet is copied or reparsed more than once.
2. Forward directly from the receiver buffer into the egress Ethernet path
   where source-buffer lifetime permits it.
3. Overlap credit return, next-packet receive, and downstream issue without
   relaxing ownership rules.
4. Keep independent transaction and completion state per egress direction.
5. Check that both Ethernet planes carry balanced traffic.

The retained implementation must be generic for forwarded Fabric packets, not
selected by all-gather shape or model identity.

## Phase 3: optimize corner forwarding if it is the first collapse

If straight multi-hop traffic is fast and experiment C collapses:

1. Attribute the time between ingress availability, local-relay reservation,
   downstream sender reservation, and egress issue.
2. Remove avoidable NOC round trips or intermediate copies at a corner.
3. Allow a generic direct corner relay when header semantics and buffer
   lifetime make it safe.
4. Pipeline successive packets while retaining per-channel fairness and
   downstream backpressure.
5. Validate every physical turn orientation and both routing planes.

Corner handling must remain a packet/route property. Do not introduce a
`production_sparse_mla_ring`-style special case.

## Phase 4: fix escape-VC overlap if the dateline is the first collapse

If experiment D or E first loses bandwidth:

1. Verify that only packets which cross the dateline move to VC1 and that they
   stay on VC1 for the remainder of the explicit route.
2. Measure VC0/VC1 buffer occupancy and downstream-credit wait independently.
3. Prove packet ownership, ordering, and credit return before separating VC
   producer state or increasing concurrency.
4. Test one direction first, then the opposite direction, then both planes.
5. Preserve the deadlock escape invariant in every optimization.

A previous attempt to separate or burst channel service hung. Any new version
must start with a small deterministic test that exhausts both VC buffers and
demonstrates forward progress before it is benchmarked.

## Phase 5: address full-ring contention and scheduling

Enter this phase only if A through E remain fast but F collapses.

1. Compare unidirectional and bidirectional traffic at identical aggregate
   bytes.
2. Attribute traffic and backpressure per plane, direction, sender channel,
   and receiver channel.
3. Check whether explicit logical-ring paths overload a physical edge or
   corner despite balanced logical route selection.
4. Rebalance route assignment before changing scheduler fairness.
5. If scanning remains measurable, implement an always-safe active-channel
   scheduler with bounded polling of cold channels. Do not trim receiver
   channels or depend on captured workload profiles.

The sender-only trimming result sets expectations: scheduling cleanup alone is
unlikely to close more than a small fraction of the current gap.

## Phase 6: qualification and model validation

For every final candidate:

1. Run small direct and explicit-path correctness tests.
2. Run full `65,536`-row BF16 and scaled-FP8 correctness.
3. Run ten consecutive full-size Fabric2D measurements.
4. Re-run the matched Fabric1D control and require at least 90 GB/s.
5. Run sparse-MLA block correctness and warm/cold/long performance tests.
6. Validate a four-ring Galaxy configuration when that system is available.
7. Build with `./build_metal.sh --release`.

No optimization is complete if it only improves the isolated CCL while
regressing the sparse-MLA block, creates corruption, or introduces an
intermittent hang.

## Commit discipline

Keep each proven mechanism in a standalone commit containing its focused test.
Record the before/after table in the commit message or accompanying performance
document. Drop experiments below the retention threshold. Prefer always-on
generic behavior; if a genuine user-visible knob is unavoidable, expose it in
the operation API with documentation and tests rather than an environment
variable.
