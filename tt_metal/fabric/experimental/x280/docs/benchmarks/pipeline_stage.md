# Router Micro-Op Characterization

## Status and authority

This document defines the current `pipeline_stage` contract for the routing
and forwarding members on the host-booted SiFive X280 RISC-V cores. The lookup
workload is one destination-to-next-hop operation over a fixed, suite-prepared
input.

Results are in [`docs/results.md`](../results.md).
X280 micro-benchmark figures move with firmware code layout alone (a code-placement
pad probe), so a figure compares only against a run of the same image.

Host lifecycle, device ownership, shape admission, argument extents, STOP, and
window cleanup are owned by
[`docs/harness.md`](../harness.md). Exact operator
commands are owned by [`README.md`](../../README.md). This document owns the pipeline
subjects, fixed layouts, configuration matrix, validation, and result identity.
How benchmarks are named — `ID <n>` from the feasibility table, and why that number
stays out of paths and result records — is owned by
[`docs/benchmarks/design.md`](design.md)
§ Benchmark naming.

## Current suite

The members share frontend timed loops (`pipeline_stage_frontend_cxx.hpp`).
Each has one subject (`route_compute_subject.hpp`, `route_forward_subject.hpp`)
and one X280 backend:

| Member | Measured operation | Target | Configurations |
|---|---|---|---:|
| `route_compute` | Select and decode one next-hop direction from destination IDs in an existing packet header | X280 | 2: `direction_table_plain` and `direction_table_instrumented` |
| `route_forward` | Consume one seeded hop from fixed source and decide its direction | X280 | 1: `decide_plain` (`decide_instrumented` unsupported) |

Backend IDs are distinct per member
(`kernels/shared/ubench_backend_ids.h`: core family, suite, member):
- `PIPELINE_STAGE_BACKEND_ID_X280_ROUTE_COMPUTE` = `0x10200001u`
- `PIPELINE_STAGE_BACKEND_ID_X280_ROUTE_FORWARD` = `0x10200002u`

Backend IDs `0x40200003u`/`0x30200003u` and `0x40200004u`/`0x30200004u` are
reserved: no member publishes them.

There is no route-construction member. Lookup has no PACKED/ALIGNED selector, construction style, packet tier,
destination-window sweep, table-placement sweep, or header-placement sweep.
Instrumentation remains a measurement mode, not an implementation alternative.

Forwarding's source is the first 96-byte header in the prepared Zero Device image
at `0x0A000AC0`. DECIDE stops after
the load, decode, cursor update, and witness update without issuing the packet.

## ID 6: destination-to-next-hop lookup

One timed item reads `dst_start_chip_id` and `dst_start_mesh_id` from one of
sixteen resident packet headers and returns one decoded direction:

```text
if destination_mesh == routing.my_mesh_id:
    direction = intra_mesh_direction_table[destination_chip]
else:
    direction = inter_mesh_direction_table[destination_mesh]
```

This is the operation expressed by
`get_next_hop_router_direction()` in
`tt-metal/tt_metal/fabric/hw/inc/tt_fabric_api.h`. The X280 implementation
retains only the freestanding layout and packed-entry decoding needed to perform
that operation.

The result of every lookup contributes to a caller-owned witness so the compiler
cannot remove the subject. The host derives the expected witness independently
from the fixture. An untimed check verifies every fixture entry before warmup,
and the input bytes are checked again after the timed loop.

### Excluded work

ID 6 does not:

- construct or encode a route buffer;
- load or expand a compressed full route;
- extract hop counts, direction bits, or a turn point from a compressed route;
- read or advance a route cursor;
- copy or publish a packet;
- perform a remote header read; or
- vary implementation style, packet size, placement, or working-set size.

The measured value is therefore not comparable to the historical
route-construction or compressed-route lookup values.

### Fixed lookup fixture

The fixture contains sixteen 96-byte headers. Eight select the intra-mesh table
and eight select the inter-mesh table. Destination IDs include boundary values,
and table entries produce varying directions. Unused direction entries are
initialized to the invalid value rather than left as plausible zero directions.

The host builds both the routing image and an independent expected-direction
sequence. Correctness does not depend on a device calculation being compared
only with itself. The timed header index is generated from the run nonce and
masked to the fixed sixteen-header count; there is no 8/256 destination-window
axis.

## Fixed packet and routing layouts

One concrete packet layout is shared by lookup and forwarding:

| Property | Value |
|---|---:|
| Packet header bytes | 96 |
| Route-buffer capacity | 35 |
| Header-base alignment | 16 bytes |
| Routing-fields offset | 44 |
| Route-buffer offset | 48 |
| `dst_start_chip_id` offset | 83 |
| `dst_start_mesh_id` offset | 85 |
| Lookup header count | 16 |

The destination fields are packed. Their offsets are not naturally aligned
even when the header base is 16-byte aligned, so they are accessed through the
declared packed fields rather than an aligned cast.

The complete `routing_l1_info_t` layout remains present:

| Region | Offset | Bytes |
|---|---:|---:|
| Full routing image | `0x000` | 2704 |
| Intra-mesh direction table | 36 | 96 |
| Inter-mesh direction table | 132 | 384 |
| Alignment padding after routing image | `0xA90` | 48 |
| Header array | `0xAC0` | 1536 |
| End of reserved input | `0x10C0` | — |

The routing image is retained at its full 2704-byte production layout even
though the lookup reads only `my_mesh_id` and one direction table. This keeps
the production offsets and packed 3-bit direction representation. The retired
compressed routing-path and exit-node fields are opaque unused bytes, not
alternate lookup subjects.

The shadow and host descriptor jointly assert the header size, destination
offsets, routing-image size, and direction-table offsets. Packed direction
decoding handles entries that cross byte boundaries through:

```text
fs_packed_direction(pointer, entry_count, index, output)
fs_original_direction(pointer, entry_count, index, output)
```

Both return failure before reading when `index >= entry_count`; callers pass
256 entries for the intra-mesh table and 1024 for the inter-mesh table.

## Suite-prepared input (X280 only)

The lookup routing image and headers occupy one fixed cached Zero Device region
on each participating L2CPU tile:

```text
[0x0A000000, 0x0A0010C0)  4288 bytes
    +0x000  full 2704-byte routing image
    +0xAC0  sixteen 96-byte packet headers
```

The required prepared profile is:

| Property | Required value |
|---|---|
| Cache/LIM shape | `WAYENABLE=1`, 14 LIM ways |
| Case admission | `LimShape{14}` with `RequirePrepared` |
| Pinned way | way 1 |
| General-purpose cache | way 0 remains enabled |
| Baseline active mask bits | `0b11` for every master |
| Prepared active mask bits | `0b01` for every master |
| Population owner | native hart 0, cache master 34 |

Preparation is one operation before the compatible sequence, run by the
runtime (`X280Tile::prepare_pinned_input`, [runtime.md](../runtime.md#prepared-pipeline-inputs-x280-only)).
Every device case of both pipeline executables calls it on the session tile
(`tests/device/suites/pipeline_stage/pinned_input.hpp`): the first call after an
ASIC reset prepares and records the region in the tile's scratch, and every
later call, in this or a later executable, returns `AlreadyPrepared`. A case
skips on `WrongShape` or `ResetRequired` and fails on any other outcome. There
is no teardown; an ASIC reset clears the region and the record.

Preparation cannot establish exclusive ownership. The operator must own the
device from preparation until the next reset. The record shows whether
preparation ran since the reset; it cannot show that nothing else cached the
reserved addresses.

The prepare image is compiled with exactly one mandatory operation define; the
source rejects none or more than one. The suite takes the image from the build
or from `X280_PINNED_INPUT_PREPARE_FW_IMAGE`; the selected path is still trusted
and the runtime rejects a mismatched operation result. `ninja x280_images`
builds it.

Preparation places all 67 cache lines in way 1, then removes way 1 from all 38
masters' replacement masks. Compatible cases keep those masks fixed. They may
coherently replace the bytes in already pinned lines before warmup, but they do
not flush the region, change `WAYENABLE`, invalidate or repin lines, allocate a
lease, or restore masks at case close.

Both pipeline executables require this exact prepared profile; every
benchmark opens at its 14 LIM ways ([harness.md](../harness.md#running-benchmarks)). Legacy/stress
suites, cache experiments, and unreviewed executables are excluded from the
prepared sequence.

Each pipeline firmware binary calls the read-only
`x280_pinned_input_check_policy(&X280_PINNED_INPUT_ROUTER_REGION, X280_PINNED_INPUT_PREPARED)` before case work. Compute checks before
overwriting or reading the canonical region. Forwarding checks before
route seeding, warmup, or timing. A
non-`ADMITTED` result refuses the case without repairing policy or writing a
mask. The refusal publishes `verify_sentinel` as
`0xB000 | ((code & 0xFF) << 8) | (master_id & 0xFF)`, the full master ID in
`workload_size`, and the full observed register in `secondary_batch_delta`.

A failed preparation leaves the record aborted, and the region is unusable
until an ASIC reset. Normal case STOP/ACK/IDLE still runs on every
configuration.

## Arguments

Lookup retains fixture
`PIPELINE_STAGE_LOOKUP_FIXTURE_V1` (value 1). It occupies one 64-byte argument
region. Fifty-two bytes are meaningful and the remaining bytes must be zero:

| Order | Field | Width |
|---:|---|---:|
| 1 | common section cookie | 32 |
| 2 | warmup count | 32 |
| 3 | timed count | 32 |
| 4 | instrumentation | 32 |
| 5 | run nonce | 32 |
| 6 | backend section cookie | 32 |
| 7 | staged input base | 64 |
| 8 | result base | 64 |
| 9 | sample-region base, or zero when plain | 64 |
| 10 | fixture ID | 32 |

On X280, the staged input is ordinary owned LIM. At case initialization, hart 0 copies
its 4288 bytes coherently into the already prepared region and verifies the
live bytes and fixture. This update is not cache preparation and does not
change the fixed masks.

The X280 image validates the trusted published argument extent before parsing. A
cookie, extent, fixture, alignment, bounds, instrumentation, or
zero-padding mismatch fails during initialization. Nothing in the section
identifies its layout, so a host and an image built from different layouts
are not told apart.

Before binding any output, each X280 entry validates every applicable owned span:
arguments, staged lookup input, result, and optional samples. Each must be
aligned, have a non-overflowing checked end, lie wholly inside the available-LIM
window the host publishes (`[0x08131B00, 0x081C0000)` at 14 LIM ways), and be pairwise
non-overlapping. Forwarding's fixed Zero Device source is not an owned-LIM span;
it is checked separately with `x280_pinned_input_contains(&X280_PINNED_INPUT_ROUTER_REGION, addr, bytes)`.

The X280 forwarding section is exactly 64 bytes, with 44 defined bytes;
bytes 44 through 63 must be zero. The composed block appends the 24-byte
host/device layout, making `sizeof(pipeline_stage_route_forward_args_block_t)`
88 bytes; `Arena` allocates and publishes its 128-byte cache-line-rounded
extent. The section uses the same common prefix, followed by
the backend cookie, the result and sample-region addresses, and the hop count
at byte 40. Forwarding runs uninstrumented, so the sample-region address must
be zero. The source address
is not an argument: it is fixed at `0x0A000AC0`.

## Timing and result contract

Both lookup configurations use the normal untimed warmup and one timed
operation per item. Plain timing brackets the batch. Instrumented timing also
records per-item deltas up to the fixed 4096-sample capacity. In every
pipeline-stage member, an instrumented case with `PIPELINE_STAGE_N` set above
that capacity fails before launch, and with it unset times 4096 items. The plain result
is the primary cost; instrumentation reports distribution and perturbation.

The common result remains one 64-byte cache line:

| Offset | Field | Meaning |
|---:|---|---|
| 0 | `backend_id` | selected member |
| 4 | `terminal_state` | completion or explicit failure |
| 8 | `packet_header_bytes` | 96, written by the executable |
| 12 | `instrumentation` | plain or per-sample |
| 16 | `warmup_completed` | completed warmup items |
| 20 | `timed_count` | requested timed items |
| 24 | `raw_batch_delta` | primary counter interval (`mtime`) |
| 32 | `secondary_batch_delta` | secondary counter interval (`rdcycle`) |
| 40 | `items_completed` | completed timed items |
| 48 | `verify_sentinel` | match or packed mismatch |
| 52 | `reserved` | zero |
| 56 | `workload_size` | 16 headers for lookup; hop count for forwarding. On a prepared-input admission refusal ([Suite-prepared input](#suite-prepared-input-x280-only)) both X280 firmwares overwrite it with the refusing master's full ID (`admit_pinned_input`, `kernels/pipeline_stage/backends/x280_member_backend.hpp`), which the host prints as `policy_master=` in its terminal-state failure message |
| 60 | `stage_witness` | member-specific observable result |

The instrumented sample region starts with `samples_completed` and one
reserved word, followed by at most 4096 64-bit deltas.

### Counter monotonicity

Timestamps must be nondecreasing during a bounded run. Every check is
`measurement_delta(begin, end, &delta)` (`include/rt/measurement.h`): a
modulo-64 subtraction that returns 0 without writing delta when the result is
at or past half range, so a backward reading is refused and a forward one across
a 64-bit wrap is accepted. Two sites apply it:
- `pipeline_stage_publish_batch_delta`, called by `publish_delta`, latches
  `COUNTER_ERROR` (value 13) on a refused batch pair, wall or CPU, and leaves
  that delta at 0.
- The shared frontend's instrumented loop latches `COUNTER_ERROR` on a
  decreasing per-sample pair, stops the loop, and records the samples completed
  so far.

Once latched, the frontend stops before `VERIFY_AFTER`: `terminal_state` stays
`COUNTER_ERROR`, which maps to the `TRAP` outcome, and the run publishes no
measurement. No distinction between counter wrap, reset, or reader error is
claimed from two samples alone.
Hosts additionally require `delta > 0` for the batch and every published sample
delta; zero deltas are rejected, nothing is subtracted from measured deltas, and
rejections write no performance JSON. Runtime formatters validate the
instrumentation enum, sample counts, and sample bounds before accessing the
fixed sample array.

### Summary row identities and field mapping

On X280, the lookup summary row contains exactly these base fields, then the
record envelope (`docs/measurement.md` "Record contract"):

```text
suite, member, kind, backend_id, run_nonce, packet_header_bytes,
route_buffer_capacity, instrumentation, fixture, header_count,
prepared_input_base, prepared_input_bytes, wayenable, pinned_way, waymask_bits,
warmup, timed_count, items_completed, stage_witness, raw_batch_delta,
secondary_batch_delta, verify_sentinel, subject, cpu_samples
```

Its fixed identities are `member="route_compute"`,
`subject="next-hop direction-table lookup"`,
`fixture="direction_table_v1"`, `packet_header_bytes=96`,
`route_buffer_capacity=35`, `header_count=16`, and the fixed prepared base and
4288-byte extent. Successful rows report `wayenable=1`, `pinned_way=1`, and
`waymask_bits=1`; these identify the admitted policy and are distinct from the
full observed-register value published only on refusal. The instrumented row's
`cpu_samples` holds the raw rdcycle samples; the tabulator derives the per-item
figures and the sample mean and percentiles.

Shared `output::write_jsonl` propagates directory creation, open, write, and
close failures as `std::runtime_error` naming the path, with checked close; no
fsync or durability guarantees beyond checked close are claimed.

There are no live `tier`, lookup `copy`/style, lookup placement, destination
window, declared-geometry, or effective-working-set fields. Historical output
containing those fields remains interpretable only as the retired profile.

## ID 7: fixed-source forwarding contract

### X280 placement

ID 7 reads one hop command from the first header in the prepared image, decodes
it, advances the cursor, and adds the decoded direction to the witness. This
arm, DECIDE, issues no packet and opens no NoC window.

The exact registration is:

| Label | Instrumentation |
|---|---|
| `decide_plain` | off |

Forwarding retains its seeded four-hop route, hop-count bound, independent
witness, output publication, and ordinary active-session cleanup. The untimed
seed overwrites all 96 bytes, sets the cursor to zero, writes
SOUTH/EAST/NORTH/WEST, and fills the remaining route buffer with NOOP. Each
consumed command must decode to one of those four directions; an invalid
command selects no direction and fails verification. Per-sample
instrumentation is refused. There is no header-placement axis, source window,
local-DDR path, or header allocation in the case.

The prepared cache profile applies to the compatible suite even when a
forwarding case does not use the reserved lookup bytes. That changes available
cache capacity relative to the historical runs, so preserved forwarding code
does not imply preserved cycle values.

### Forwarding summary rows

Forwarding summary rows report
`source_placement="pinned_zero_device"`,
`source_base=0x0A000AC0`,
`source_bytes=96`, `packet_header_bytes`, `route_buffer_capacity`,
`instrumentation`, `hop_count`,
`stage_witness`, verification, and the common timing fields.
Rows report the fixed cache-profile fields `wayenable=1`, `pinned_way=1`,
and `waymask_bits=1`, `boundary="decision"` and
`subject="router forward decision"`.
The tabulator rejects a fixed-source forwarding row unless its source, packet,
capacity, hop count, boundary, subject, and instrumentation match this
identity. Current rows carry no `args_schema`;
rows with a schema, and historical variable-source forwarding rows, are read by
the frozen tabulator.

## Validity

A current lookup observation is valid only when:
- On X280:
  * preparation completed under continuing external exclusive ownership;
  * the exact 14-LIM-way profile and all prepared active mask bits are unchanged;
  * the per-case read-only prepared-policy check admitted the executable before
    any canonical-input access;
  * every owned LIM span passed bounds, alignment, checked-end, and non-overlap
    admission before outputs were bound;
  * the case updated and verified the complete live input;
  * the fixture ID matches;
  * terminal state is `DONE`, verification reports a match, the witness equals
    the host oracle, and the item count is complete;
  * packet bytes, header count, and prepared base and extent identify the fixed workload; and
  * the row is not combined with a historical result as though the subjects or
    execution environments were identical.

A current forwarding observation is valid only when:
- On X280:
  * preparation completed under continuing external exclusive ownership;
  * the exact 14-LIM-way profile and all prepared active mask bits are unchanged;
  * the per-case read-only prepared-policy check admitted the executable before
    route seeding;
  * every owned LIM span passed bounds, alignment, checked-end, and non-overlap
    admission before outputs were bound;
  * terminal state is `DONE`, verification reports a match, the witness equals
    the host oracle, and the item count is complete;
  * source is `0x0A000AC0` and hop count is 4.

**Acceptance boundaries**:
A host build establishes that host C++ executables compile and link. Separate
executed offline host unit tests establish that layout assertions hold, cookie and
padding checks function, fixture construction succeeds, and runtime formatters validate bounds and
reject invalid inputs.
A host build does NOT demonstrate:
1. Fresh cache placement, retention, or mask writes on X280;
2. Device silicon correctness; or
3. Measured timing performance on hardware.
Silicon correctness and timing observations are recorded in
[`docs/results.md`](../results.md).

## Stable references

- `tt-metal/tt_metal/hostdevcommon/api/hostdevcommon/fabric_common.h` owns
  `routing_l1_info_t` and packed direction-table layouts.
- `tt-metal/tt_metal/fabric/hw/inc/tt_fabric_api.h` owns the production
  destination-to-next-hop selection semantics.
- `kernels/pipeline_stage/pipeline_stage_common.h` owns the fixed packet
  constants, state values, and
  `pipeline_stage_publish_batch_delta`, which publishes a checked batch delta.
- `kernels/pipeline_stage/route_compute_args.h` owns the shared lookup argument
  section and composed block.
- `include/rt/x280_pinned_input.h` owns the prepared-region constants.
- `tests/device/suites/pipeline_stage/route_compute.hpp` and `route_forward.hpp`
  own X280 host descriptors.
- `kernels/pipeline_stage/route_compute_subject.hpp` and `route_forward_subject.hpp`
  own the subjects; `kernels/pipeline_stage/pipeline_stage_frontend_cxx.hpp`
  owns measurement sequencing.
- `tests/device/suites/pipeline_stage/route_oracle.hpp` owns fixture
  construction and the independent oracle; the two suite descriptors own host
  packing into the shared flat section types.
