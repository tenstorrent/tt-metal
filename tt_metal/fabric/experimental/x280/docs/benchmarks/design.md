# Characterization Ubench Design

**Status:** v0 implementation design

## Purpose

This document defines the microbenchmark program used to characterize
communication and per-item processing across Blackhole programmable core
classes. The immediate goal is to produce comparable measurements that can
guide how control logic and data movement are partitioned across X280, Tensix,
DRISC, and ERISC.

The benchmark program answers three kinds of questions:

1. **Peer-to-peer latency:** what does one live exchange cost between two
   participants?
2. **Transport bandwidth:** what sustained rate can one transport leg achieve?
3. **Pipeline-stage cost:** what does one per-item processing stage cost on a
   candidate core class?

The document sets experiment semantics and measurement boundaries. It is not a
production runtime design, a project tracker, or a complete implementation
specification.

## Document boundaries

Two design documents have distinct authority:

- [`docs/harness.md`](../harness.md) owns
  host lifecycle and ownership, claims where the hardware has a reversible
  reservation, regions and address views, execution, deadlines, safety,
  runtime gates and API blockers, clock
  mechanisms, the common result envelope, and the cross-source X280 NOC
  invariants used by characterization.
- This document owns performance-suite membership and participant rows,
  protocol loops and retained pseudocode, measurement boundaries, axes, suite
  payloads and validation, result interpretation, and suite-specific kernel
  structure.

Raw X280 NOC encodings and helper behavior remain defined by
[`x280/include/noc.h`](../../include/noc.h). The harness LLD summarizes the
non-obvious consequences that span that header, atomic behavior, and L3/LIM
visibility.

Implementation and verification tasks belong in issues. This document retains
the durable preconditions that determine whether a performance configuration
is admissible, but it does not prescribe the verification procedure used to
establish them. Functional verification is not a performance suite in this
document.

The pseudocode in this document records more than an example. Its suite
sequencing, dataflow, protocol ownership, and measurement brackets are
normative design decisions. Hardware ownership and lifecycle follow the
harness LLD and landed programming components. Type names, target-adapter
shapes, function names, and exact source layout are illustrative unless
explicitly stated otherwise.

## Common design principles

### Behavior-named suites

Implementation suites are named for the measurement they perform:

```text
tests/
  p2p_latency/
  transport_bandwidth/
  pipeline_stage/
kernels/
  shared/
  p2p_latency/
  transport_bandwidth/
  pipeline_stage/
```

The logical shape within a measurement suite is:

```text
tests/<measurement-suite>/
  config              typed hardware selections, roles, axes, validation
  registry            stable case names and allowed/default configurations
  case                allocate, resolve, prepare, launch, wait, collect
  handlers/<class>    adapt the suite protocol to one programmable class
  reduce              suite validity checks and derived quantities
kernels/<measurement-suite>/
  frontend            the suite's loop and roles
  backends/<class>    hardware primitives needed by that loop
```

This shape is a separation of responsibilities, not a commitment to exact
filenames.

Editorial family numbers and historical benchmark IDs are not part of APIs,
paths, selectors, or result records.

### Measured cost, not validated function

A member reports what a code sequence costs. It does not claim the sequence is
functionally correct, and it carries no pass/fail threshold on the cost it
reports: the number is the output, and its interpretation belongs to the
reader. Validity checks — token match, byte accounting, path verification —
decide whether an observation is a measurement at all and are not a verdict on
the cost.

A member may therefore stand in for logic it does not implement: a parse that
reads the bytes and discards them, a route computation whose result is never
consumed, a Tensix core in place of an Ethernet port. Cost-equivalence bounds
that simplification. Whatever determines the cycle count is faithful — the
real header size, the real number and locality of the memory regions touched,
the real ordering mode of the stores, the real instruction mix — while the
semantics may be emulated. The failure the bound exists to prevent is a
simplification cheap enough to report a cost the real logic could not achieve.

### One exact configuration per invocation

The harness invokes one case at a time and owns participant lifecycle,
per-case regions, and collision handling through its
[`ownership model`](../harness.md#ownership-and-lifecycle).
For suite design, the durable rule is that one invocation receives one exact
configuration. A case may still launch multiple participants when that
concurrency is part of the experiment.

Platform state explicitly prepared for a compatible sequence may outlive one
invocation. The fixed pinned pipeline input is such a state: external exclusive
ownership spans preparation, every sequential case, and final teardown, while
each case still performs its ordinary STOP and window cleanup. A case does not
restore the shared cache policy when it closes.

A named sweep expands to an ordered list of exact configurations before the
runner is called:

```text
named sweep + explicit baseline
    -> ordered exact run configurations
    -> one sequential case invocation per configuration
```

The suite, not the harness, owns sweep expansion:

```cpp
auto runs = p2p_latency::make_payload_sweep(
    baseline,
    payload_points);

for (const P2PLatencyRunConfig& cfg : runs) {
    runner.run("p2p_latency.x280_tensix", cfg);
}
```

Each invocation fixes placement, payload size, directional access settings,
artifact choice, and every other axis except protocol phases that must share
hardware state. By default a named sweep varies one primary axis and pins the
rest to a recorded baseline. A multi-axis experiment must name the interaction
being tested and construct a bounded set of points explicitly; the harness does
not form implicit Cartesian products.

Execution order is recorded. Forward and reverse passes are the default drift
control for ordered sweeps.

### Typed hardware and region ownership

Hardware identity, ownership, usable spans, region finalization, viewer-relative
resolution, binding, and restoration are defined by the harness
[`Concrete target model`](../harness.md#concrete-target-model)
and
[`Memory regions and address views`](../harness.md#memory-regions-and-address-views).
The host-shape pseudocode below consumes those typed services; its target and
selection names are illustrative, not a second host API specification.

### Shared kernel-side conventions

The frontend is a suite's unit of variation and is not shared: phase graphs
differ legitimately between roles and directions, a boundary with a drain,
and per-item granularity, and would differ again for the multi-producer
arrival scanning of § Client handling. One frontend per suite, each
absorbing its members as backends, is the shape — extracting a common one
would couple those phase graphs to save a driver of about thirty lines.

What every suite does share are the conventions around that frontend, and
those live in `kernels/shared/` as suite-agnostic headers:

| Artifact | Contents | Per-suite part |
|---|---|---|
| `ubench_cookies.h` | `UBENCH_SECTION_COMMON` and `UBENCH_SECTION_BACKEND`, and the cookie function that XORs a section into a suite magic | the magic, so one suite's args block cannot satisfy another's image |
| `ubench_section_load.h` | checked copy from a volatile argument span into flat POD storage, plus the C/C++ layout-assert macro | each suite's section and composed block types |
| `ubench_outcome.h` | the outcome vocabulary a host needs to decide whether an observation is a measurement at all: `OK`, `ARG_LAYOUT_MISMATCH`, `TIMEOUT`, `TRAP`, `PATH_MISMATCH`, and the non-terminal `IN_PROGRESS` | the mapping from the suite's own phase enum onto that vocabulary |

Flat POD sections are the canonical argument mechanism. The host and firmware
compile the same section types, whose sizes and field offsets are pinned by
layout assertions. The entry point copies a complete section only after
checking its address, extent, and alignment, then validates the common and
backend cookies before trusting an embedded address.

Phase enums stay per-suite and keep their own numbering. The outcome
vocabulary is a mapping over them rather than a renumbering, so a suite
already publishing a result record adopts it without changing the values its
host reads.

### Host shape for peer-to-peer latency

Small registered setup functions perform typed allocation, then call one shared
suite sequence:

```cpp
P2PLatencyResult run_x280_hart_hart_latency(
    CaseContext& ctx,
    const X280HartLatencyRunConfig& cfg) {

    auto tile = ctx.targets().x280().select(
        X280TileRequest{cfg.tile});

    return p2p_latency::run(
        ctx,
        x280_participant(tile, cfg.hart_a),
        x280_participant(tile, cfg.hart_b),
        cfg.protocol);
}

P2PLatencyResult run_x280_tensix_latency(
    CaseContext& ctx,
    const X280TensixLatencyRunConfig& cfg) {

    auto tile = ctx.targets().x280().select(
        X280TileRequest{cfg.tile});
    auto workers = ctx.allocations().allocate(
        TensixCoreSetRequest{cfg.worker_range});

    return p2p_latency::run(
        ctx,
        x280_participant(tile, cfg.hart),
        workers.core(cfg.worker),
        cfg.protocol);
}
```

The shared body composes the harness's region, addressing, execution, and
recording services:

```cpp
P2PLatencyResult p2p_latency::run(
    CaseContext& ctx,
    SelectedParticipant a,
    SelectedParticipant b,
    const P2PLatencyRunConfig& cfg) {

    validate(cfg, ctx.inventory(), a, b);

    auto layout = make_io_layout(cfg);
    auto io_a = ctx.regions().allocate(a, layout.a);
    auto io_b = ctx.regions().allocate(b, layout.b);
    ctx.regions().finalize();

    auto initial = initial_contents(ctx.run_id(), cfg);
    ctx.regions().write(io_a, initial.a);
    ctx.regions().write(io_b, initial.b);
    auto waits = make_p2p_wait_limits(cfg, a, b);

    auto b_for_a =
        ctx.addressing().resolve(io_b.inbound, a);
    auto a_for_b =
        ctx.addressing().resolve(io_a.inbound, b);

    auto b_as_seen_by_a =
        bind_peer_view(ctx, a, b_for_a, cfg);
    auto a_as_seen_by_b =
        bind_peer_view(ctx, b, a_for_b, cfg);

    auto workload_a = handler_for(a).prepare(
        ctx, LatencyParticipant::A,
        a, io_a, b_as_seen_by_a, cfg,
        initial.run_nonce, waits.a_warmup_cycles);
    auto workload_b = handler_for(b).prepare(
        ctx, LatencyParticipant::B,
        b, io_b, a_as_seen_by_b, cfg,
        initial.run_nonce, waits.b_warmup_cycles);

    auto plan = ctx.execution().prepare(
        {std::move(workload_a), std::move(workload_b)});
    auto raw = ctx.execution().run_and_collect(
        plan, waits.host_deadline);

    return validate_and_reduce(ctx, cfg, raw);
}
```

The participant parameters denote concrete values produced by typed setup; the
design does not require a common participant base class. `x280_participant` is
an illustrative target-adapter operation. Prepared workloads retain their
native execution-domain identity. Two same-tile X280 harts may be grouped into
one image launch, while an X280-to-Tensix case produces an X280 launch and a
Metal `Program`.

`bind_peer_view` selects the suite access form. Cross-tile X280 rows pass the
directional access plan into explicit binding and retain all leased descriptors
for the invocation. ID 2 selects a coherent or LIM self-view without binding a
NOC window.

### Host shape for transport bandwidth

Addressable-memory members use the same harness capabilities with a different
suite-owned sequence:

```cpp
TransportBandwidthResult run_tensix_x280_bandwidth(
    CaseContext& ctx,
    const TensixX280BandwidthRunConfig& cfg) {

    auto workers = ctx.allocations().allocate(
        TensixCoreSetRequest{cfg.worker_range});
    auto tile = ctx.targets().x280().select(
        X280TileRequest{cfg.tile});

    return transport_bandwidth::run_addressable(
        ctx,
        workers.core(cfg.worker),
        tile,
        cfg.protocol);
}
```

This is a passive X280 memory target and adds no active X280 workload. Its
lifecycle and usable destination span follow the harness target contract.

The suite supplies semantics and the harness supplies mechanisms:

```cpp
TransportBandwidthResult transport_bandwidth::run_addressable(
    CaseContext& ctx,
    SelectedInitiator initiator,
    SelectedTarget target,
    const TransportBandwidthRunConfig& cfg) {

    validate(cfg, ctx.inventory(), initiator, target);

    auto layout = make_region_layout(cfg);
    auto source = ctx.regions().allocate(initiator, layout.source);
    auto destination =
        ctx.regions().allocate(target, layout.destination);
    ctx.regions().finalize();

    ctx.regions().write(source, source_contents(cfg));
    ctx.regions().write(destination, destination_contents(cfg));
    auto host_deadline = make_host_deadline(cfg);

    auto resolved_destination =
        ctx.addressing().resolve(destination, initiator);
    auto destination_view = ctx.addressing().bind(
        initiator, resolved_destination, cfg.access);

    auto workloads = prepare_workloads(
        ctx,
        initiator,
        target,
        source,
        destination,
        destination_view,
        cfg);

    auto plan =
        ctx.execution().prepare(std::move(workloads));
    auto raw = ctx.execution().run_and_collect(
        plan, host_deadline);

    return validate_and_reduce(ctx, cfg, raw);
}
```

A passive addressable target adds no workload. A raw-Ethernet member binds a
concrete link and adds only the peer behavior required to operate the Ethernet
primitive; it does not acquire a generic remote drainer.

### Device and result conventions

The harness
[`Result composition`](../harness.md#result-composition)
defines the common envelope, reference-type vocabulary, and the
device/host/suite responsibility split. This document defines each suite
payload, selects the applicable reference type, validates observations, and
defines the meaning of derived values.

## Readiness and implementation sequence

The suites have different levels of maturity:

1. **Peer-to-peer latency** has the first suite implementation contract.
   Same-tile X280, X280-to-Tensix and cross-tile X280 are landed on the
   active-image contract; cross-tile X280 boots its tiles through
   `X280Runtime::boot_tiles`. These host mechanisms
   are tracked by the harness
   [`Current decision index`](../harness.md#current-decision-index)
   (D-021, D-023, and D-026). The `erisc_peer_tile` member is a landed
   mechanical smoke path to one inactive idle ERISC over same-chip NOC; it does
   not use an Ethernet link. It requires `TT_METAL_SLOW_DISPATCH_MODE=1` before
   `MeshDevice` construction because fast dispatch does not support `IDLE_ETH`
   kernels. Its raw-tick output carries per-run host and CMFW AICLK evidence for
   external time conversion. The
   `drisc_peer_tile` member is also a landed mechanical smoke path: it uses
   bank 0's free non-endpoint DRISC over NOC0, changes only NOC0 to stream
   mode, and publishes raw counter ticks. Active ERISC and cross-chip rows
   follow only when their target mechanisms are concrete; raw-link ERISC rows
   also require the session-level fabric transition in D-027.
2. **Transport bandwidth** has a common loop and measurement contract.
   The addressable Tensix/X280 members are concrete; receiving-ERISC
   packet-header ingress is a `packet_injection` member. DRISC and raw-link
   ERISC members still require member-specific backend decisions.
3. **Pipeline-stage cost** has a concrete lookup and forwarding contract in
   `docs/benchmarks/pipeline_stage.md`, which IDs 6 and 7 supply in full.
   The lookup is a fixed destination-to-next-hop operation; forwarding has one
   DECIDE configuration.
4. **Client handling and composed experiments** remain roadmap items.

Later suites are not prerequisites for the initial harness surface.

## Benchmark naming

A benchmark is named by its Phase 0 ID, written `ID <n>`, matching the table
in § Phase 0 correspondence verbatim. That table is the source of truth, so a
document that invents its own handle for a benchmark forces every reader to
translate. A document scoped to one benchmark says so in its opening
`**Scope:**` line; prose and headings lead with the ID and use it alone
thereafter.

IDs stay out of code. APIs, paths, selectors and result records keep the
behavior-named suite and member vocabulary, which is the rule § Behavior-named
suites already states. The reason it states without giving: an editorial number
is not a property of the thing it numbers, so a benchmark renumbered upstream
would otherwise rename files. A reader who needs to run a benchmark reads the member name; a reader
who needs to find it in the plan reads the ID.

**A member is named `<peer kind>_<locality>`.** The peer kind is what sits at
the far end, `x280` or `tensix`. The locality is the L2CPU tile boundary,
`same_tile` or `peer_tile`; that boundary is where cache coherence stops
(`BlackholeA0/L2CPUTile/Caches.md`: there is no coherence between the caches of
different L2CPU tiles), which is why two members either side of it measure
different things. The x280 side goes unnamed because every member of
`p2p_latency` has one. Where a member has a backend on each side, that backend
is `BACKEND_ID_<own side>_<member>`.

**`Row <n>` is retired.** It is repo-internal, it does not appear in the
Phase 0 table, and it collides with the IDs: "ID 3" and "ID 3" name the same
member, and nothing but the correspondence table tells a reader so. It survives
in no identifier, path, target, environment variable, record key or sentence —
prose says `ID <n>`, code says the member token.

A member answering no numbered row says so, and names the unnumbered bullet it
answers if there is one. Silence there reads as an oversight rather than a
fact.

The ID earns its place where the benchmark is the subject of a claim, and most
of all where bare IDs appear in the same sentence — "ID 3 is the only one of
the five" against "IDs 6, 7, 10 and 11" reads clearly where two numbering
schemes would not. Mechanism prose is different: a passage describing how a
backend is shaped, which fences it issues or which ordering mode it selects is
about the implementation, and there the member name is the accurate one.

Where a numbered benchmark is compared against an unnumbered sibling, use the
Phase 0 ID for the numbered benchmark and the member name for the sibling.
Cross-cluster signalling therefore compares ID 3 with `tensix_peer_tile`;
neither side needs an editorial family number.

## Phase 0 correspondence

The suites answer the Fabric 2.0 feasibility benchmark table (as read on 2026-09-02),
which numbers 19 microbenchmarks and lists seven unnumbered
X280 microbenchmarks beside them. The IDs below are cross-references to that
table and nothing more: no API, path, selector, or result record carries one.

| Phase 0 ID | Microbenchmark | Member |
|---|---|---|
| 2 | X280 pipeline signal latency/cycle count (same cluster) | `p2p_latency` `x280_same_tile`, landed and measured |
| 3 | X280 pipeline signal latency/cycle count (different cluster) | `p2p_latency` `x280_peer_tile`, landed and measured |
| — | X280 ↔ Tensix, one channel both ways | `p2p_latency` `tensix_peer_tile` for latency and both `transport_bandwidth` members for bandwidth, landed and measured |
| — | X280 ↔ inactive idle ERISC, same-chip NOC | `p2p_latency` `erisc_peer_tile`, landed as a mechanical smoke path; raw ERISC ticks only |
| — | X280 ↔ free non-endpoint DRISC, same-chip NOC0 | `p2p_latency` `drisc_peer_tile`, landed as a mechanical smoke path; raw DRISC ticks only |
| 6 | X280 Routing module | `pipeline_stage` `route_compute`, landed and measured, specified in `docs/benchmarks/pipeline_stage.md` |
| 7 | X280 Forwarding module | `pipeline_stage` `route_forward`, landed and measured, specified in `docs/benchmarks/pipeline_stage.md` |
| 10 | X280 packet injection handler module (single connection) | client handling, implemented in `docs/benchmarks/packet_injection.md` |
| 11 | X280 packet injection handler (all workers) | client handling, implemented in `docs/benchmarks/packet_injection.md` |

The three Metal-peer members share one X280 firmware source,
`src/p2p_latency_metal_peer_tile.cpp`, and one Metal entry,
`kernels/p2p_latency/metal_peer_entry.cpp`. Peer addresses are runtime arguments;
the X280 image has no peer-type branch. The Metal target is selected at compile
time, independently of the common 16-word layout cookie. Host member names and
target counter/lifecycle contracts remain distinct.

The source audit keeps the following splits:

| Sources | Contract requiring the split |
|---|---|
| P2P Tensix, ERISC and DRISC host descriptors/runners | Worker L1 allocation; inactive Ethernet admission and Ethernet launch; free DRISC arena admission and forced slow dispatch respectively. JIT tests have the same distinct admission requirements. |
| P2P same-tile and cross-tile X280 images | Different placement and transport: local mailbox versus paired NoC window. |
| Pipeline route-compute and route-forward | Different algorithms. |
| Transport-bandwidth direction sources | Different initiator, counter, issue primitive and completion boundary. |
| DRISC endpoint helper | Owns a resource with no equivalent ERISC/Tensix ownership contract. |

The audit of `include/`, `src/`, `kernels/`, and `tests/device/` found no
remaining target-specific P2P entry, argument wrapper, source-local target
definition, or one-caller Metal policy forwarding header. No member reduces
its samples: every member records its raw populations, and the tabulator's one
floor-index rule computes their percentiles (`docs/measurement.md`).

The unnumbered row is grouped that way because IDs 2 and 3 describe the whole
communication path of one two-core channel rather than a single quantity, and
X280 ↔ Tensix is such a channel: `p2p_latency` `tensix_peer_tile` times it and
`transport_bandwidth` measures its rate in each direction. That channel has no
numbered counterpart, and its bandwidth half answers the unnumbered
"Tensix -> x280 NOC BW" bullet.

ID 14, X280 packet header ingress BW, is implemented by
`packet_injection/erisc_to_x280_header`, specified in
[`docs/benchmarks/packet_injection.md`](packet_injection.md#erisc-header-ingress-member):
one to all available idle ERISCs, one per Ethernet tile, publish ready 96 B
headers through the shared packet-injection producer, credit protocol and X280
consumer, at writer counts 1, 2, 4, 8 and all available and publication batch
sizes 1, 4 and 8. Aggregate ingress comes from one X280 `mtime` interval; each
ERISC keeps raw wall ticks and `rdcycle`. Header creation, Ethernet arrival, the
packet copy to DRISC, and the wait for that copy are preconditions outside the
measured bracket. It superseded, at its `writers=1,B=1` point, the single-writer
prototype `transport_bandwidth/erisc_to_x280_header`, which has been removed:
that member published into successive X280 FIFO slots with a sideband
`pkt_valid` word and never waited for its consumer. Its per-header cost against
the unified point, the cost of the credit protocol, and its historical results
are in
[`CHARACTERIZATION_RESULTS.md`](../../CHARACTERIZATION_RESULTS.md#44-erisc---x280-packet-header-ingress-96-b);
its contract is kept in
[`docs/benchmarks/packet_header_ingress.md`](packet_header_ingress.md#prototype-member-superseded).

`p2p_latency` participant entries 3–6 have no numbered counterpart. The IDs
touching entries 4–6 — 4, 5, and 17–19 — are credit-flow and link-bandwidth
benchmarks rather than signal-latency members. IDs 1 and 13 are DRISC subjects, 8 and 9 are
the X280 credit modules, and 12, 15, and 16 are P2 scaling variants of numbers
above; none has a member here.

## Functional preconditions

Performance suites rely on functional evidence established outside their timed
loops. The implementation work for these checks is issue-owned, but the
following admissibility rules are durable:

- Every invocation first satisfies the harness
  [`Safety invariants`](../harness.md#safety-invariants) and
  current X280 runtime gates, including host access, lifecycle, memory bounds,
  window binding, and clock metadata.
- X280-to-Tensix readiness must be established in both directions before a
  timed cross-class result is accepted. The suite's own bounded warmup and
  rendezvous establish it (p2p_latency's warmup exchange), and its result
  checks reject a run where they did not.
- A Tensix-initiated X280 mailbox must use a destination and write shape whose
  visibility to the X280 local polling path the suite's own warmup has
  demonstrated.
- An unordered payload publication mechanism is inadmissible until a receiver
  has demonstrated payload-before-token visibility, including a positive
  mismatch control.
- A cached X280 Memory Port path is inadmissible until its peer-visibility
  operation is established.
- The `drisc_peer_tile` smoke member selects bank 0's free non-endpoint DRISC,
  requires NOC0 in the Stream mode DRISC firmware sets, and switches no NIU. The
  tensor prefetcher is inactive.
- A raw-link or active-ERISC member is inadmissible until a session-level transition from
  production `FabricConfig` to raw characterization and back is established.
- The idle `erisc_peer_tile` smoke member may select only a core returned by
  `get_inactive_ethernet_cores()`. It uses same-chip NOC traffic and exercises
  no Ethernet link or production-fabric state.
- Each target's measurement-counter domain and per-run frequency source must be
  established before cross-class physical-time comparisons are reported.

The strict-order System Port path is the X280 NOC baseline. Optional mechanisms
that have not met their preconditions do not appear as failed performance
points; they are invalid configurations and are not launched.

## Peer-to-peer latency

### Question and scope

`p2p_latency` measures the cost of one live exchange between exactly two
participants. The initiator sends a token or payload, the responder echoes the
complete transfer, and the initiator observes the reply.

The primary output is round-trip latency as a function of payload size and
transport configuration. The 4-byte point is the measured credit-sized exchange
latency, denoted `R_credit`. A downstream protocol may contain one or more
serialized exchanges and additional handler work; it must derive its own total
protocol latency from those facts. The benchmark does not multiply `R_credit`
by an assumed exchange count.

For a protocol-specific total latency `T_protocol` in seconds and line rate `L`
in bytes per second, the associated buffering calculation is:

```text
bytes_in_flight = L × T_protocol
```

`T_protocol` is not itself a `p2p_latency` result.

### Participant rows

Rows follow the accepted implementation sequence:

| Entry | Participants | Scope | Capability exercised |
|---|---|---|---|
| 1 | X280 hart ↔ X280 hart, same tile | one chip | session-owned tile selection, two-hart image launch, coherent/LIM self-views |
| 2 | X280 tile ↔ X280 tile | one chip | cross-instance resolution and firmware-programmed X280 window binding |
| 3 | X280 ↔ Tensix | one chip | two programmable classes, two address-view forms, two counter domains |
| 4 | X280 ↔ DRISC | one chip | firmware-owned NOC0 Stream mode |
| 5 | X280 ↔ inactive idle ERISC | one chip | same-chip NOC mechanical smoke path; no Ethernet-link traffic |
| 6 | ERISC ↔ ERISC | two chips | link-pair selection, raw ERISC session, and independent clock state |

Participant entries 1–3 define the initial implementation sequence.
Entry 4 has the separately named `drisc_peer_tile` mechanical smoke member.
It uses forced slow dispatch and the free non-endpoint DRISC, requires NOC0 in
the stream mode DRISC firmware sets, switches no NIU, and separates the peer's untagged local address from the host's tagged L1 view.
Its `get_timestamp()` deltas are raw ticks with per-run AICLK evidence and a
validated conversion rate.
Entry 5 now has the separately named `erisc_peer_tile` mechanical smoke member:
X280 maps the idle ERISC mailbox through its checked paired-range TLB, while
ERISC returns the token through Metal NOC operations. The member requires an
inactive Ethernet core and publishes ERISC wall-clock deltas as raw ticks with
the same per-run AICLK evidence. Entry 6 remains unspecified.
Entries 1 and 2 answer Phase 0 IDs 2 and 3; entries 3–6
answer no numbered ID, and § Phase 0 correspondence carries the whole mapping.

ID 2 is the only member with a medium choice: coherent cached memory or
uncached LIM. It consumes no X280 NOC window. Entry 6 is the only cross-chip
member.

### Roles and mailbox model

Every image contains both roles:

- The **initiator** performs `W` warm-up exchanges and records `N` round trips.
- The **responder** echoes `W + N` exchanges and records turnaround for the
  timed `N`.

The direction swap is a runtime phase in one invocation, not a separate build.

Two rules are mandatory:

1. A mailbox has exactly one writer: the remote peer.
2. Each participant polls only its own local memory.

These rules are required for X280 correctness and make the protocol symmetric
across component classes. Each participant owns an inbound mailbox and a
host-read result region. Backends that transmit from memory also own a
host-seeded staging region used when they initiate; the `x280_peer_tile` and
`tensix_peer_tile` X280 participants store the token directly and need no
staging:

```text
  inbound mailbox       local staging        outbound region
  (peer writes; I       (host seeds; source  (I write, host reads)
   poll and echo)        when I initiate)
  ┌───────────────────┐ ┌───────────────────┐ ┌────────────────────────────┐
  │ PAYLOAD_SIZE bytes│ │ PAYLOAD_SIZE bytes│ │ STATUS     state machine   │
  │ TOKEN = last 4 B  │ │ token slot last   │ │ HEARTBEAT  advancing count│
  └───────────────────┘ └───────────────────┘ │ RESULT[]   raw deltas      │
                                               └────────────────────────────┘
```

The token is always the final 4 bytes of the transfer. A 4-byte transfer is the
degenerate token-only case, with poll offset zero. A larger response echoes the
entire received buffer, not just its token.

Tokens are per-exchange-distinct and latching. The reader compares against an
expected value and does not clear or consume it. The host chooses a non-zero
32-bit `run_nonce` and clears mailboxes before launch:

```text
global_exchange = phase_ordinal × (W + N) + iteration
token           = run_nonce + global_exchange
total_tokens    = 2 × (W + N)
```

`phase_ordinal` is 0 or 1. Within each phase, `iteration` is the combined
exchange ordinal `0..W+N-1`: warm-up uses `0..W-1` and timed samples use
`W..W+N-1`. It is not the timed-loop index reset to zero.

The host computes the interval in 64 bits and requires:

```text
run_nonce + total_tokens - 1 ≤ UINT32_MAX
```

The kernel can then increment the token in the hot loop without a wrap branch.

### Rendezvous and warm-up

Launch does not imply simultaneous loop entry. The first warm-up exchange is
the rendezvous:

- The initiator cannot advance until it observes the first reply.
- A token that arrives before the responder polls remains latched.

The protocol then completes the remaining `W - 1` warm-up exchanges and records
`N` samples. `W` is at least one and is tuned empirically by increasing it until
the measured distribution no longer moves materially. Cold and warm latency
are separate invocations; a cold result requires a defined idle condition.

Warm-up polling has a participant-local bound supplied in that participant's
counter ticks. X280 measured polling remains unbounded on-device to avoid
contaminating the latency tail; the harness host deadline remains the safety
boundary.

The idle `erisc_peer_tile` smoke member keeps its warm-up and emergency hang
guards as conservative raw-tick constants; result conversion instead uses the
host's per-run clock evidence. If its shared Metal backend's emergency
timed-poll guard expires, the timed poll returns false and the frontend ends
the phase `TIMEOUT` without recording that sample.

The `drisc_peer_tile` smoke member likewise uses a fixed raw-tick warm-up budget
and emergency guard. Its reported raw ticks use the separately validated
per-run AICLK conversion rate.

### Axes and validity

| Axis | Values | Binding |
|---|---|---|
| first direction | A initiates first or B initiates first | runtime |
| placement | one exact ordered participant pair | host |
| directional access plans | one plan for A→B and one for B→A | host, programmed before launch |
| ID 2 medium | coherent or uncached LIM | runtime |
| sample and warm-up counts | `N > 0`, `W ≥ 1` | runtime |
| payload size | at least 4 bytes, 4-byte aligned | exact invocation |
| transmit mode | fire-and-forget, wait-for-departure, wait-for-completion | compile-time behavior |
| polling | tight or backoff | compile-time behavior |

A directional X280 access plan may contain port, posted mode, strict ordering,
and NOC selector. A Metal-initiated plan contains its NOC selection and scalar
destination view. Both directions are bound before launch and remain fixed
through both phases.

The 4-byte case uses a token-only specialization so its hot path contains no
payload-size branch. Larger sizes may use one generic runtime-sized artifact;
each size remains a separate invocation. Initial payload anchors should cover:

- 4 bytes: credit/token exchange;
- 64 bytes: one X280 vector-store width;
- the current production payload; and
- the configured maximum payload.

The exact production and maximum values come from the tested configuration and
are recorded with the run rather than treated as timeless constants.

Transmit-mode validity is backend-specific:

- Tensix supports fire-and-forget, departure wait, and completion wait through
  its asynchronous NOC write, flush, and barrier operations.
- An X280 NOC-window fire-and-forget leg requires a posted view.
- X280 completion wait uses a non-posted view or a posted view followed by a
  proven completion-forcing read-back.
- X280 has no established departure-only primitive; that mode is rejected.
- ID 2 maps fire-and-forget and completion wait to a local store without or
  with the required visibility fence. Departure wait is not applicable.

Receiver-visible payload-before-token ordering is distinct from when
`transmit()` returns:

- The X280 NOC baseline uses strict ordering for every payload size.
- A normal unordered full-buffer X280 transmission is invalid for
  `PAYLOAD_SIZE > 4`, regardless of transmit mode.
- An unordered split publication is admitted only after its exact
  prefix-confirmation or separate non-posted-token mechanism passes functional
  verification.
- A cached Memory Port configuration is admitted only after its explicit
  visibility operation is established.

The host validates these combinations before launch; the kernel does not branch
to repair an invalid artifact/configuration pair.

### Timing and clock domains

Each kernel is clock-solipsistic: it reads only its own counter and never sends
or receives timestamps.

Each participant records one local quantity:

- `RTT`: initiator time from before `transmit()` until the expected reply is
  observed.
- `TURN`: responder time from the successful poll observation until its reply
  has been issued according to the selected transmit semantics.

Counter selection, fencing, lifecycle timebases, and pre-launch versus per-run
frequency validation follow the harness
[`Measurement discipline`](../harness.md#measurement-discipline).
The host converts each participant's raw deltas independently before the suite
compares physical time. The suite uses the harness's fenced X280 bracket and
still performs any transport completion operation required by the selected
`transmit()` endpoint.

The optional X280 phase-level clock assertion is:

```text
clock_residual =
    1 - rdcycle_delta / (mtime_delta × measured_hz / mtime_hz)
```

The deltas and frequencies are positive. The residual is evaluated in real
arithmetic or represented as signed parts per million. `measured_hz` and
`mtime_hz` are the independently established values published under the
harness clock contract. Deriving the frequency from the same two deltas would
make the residual identically zero.

### Both directions in one invocation

Both directions run as two phases under one allocation and hardware state.
Each phase performs its own warm-up. The first direction alternates across
repeated invocations so a first-phase effect cannot masquerade as asymmetry.

After each participant's deltas have been converted independently to the same
physical unit, the suite may compute:

```text
net_A = RTT_A - TURN_B
net_B = RTT_B - TURN_A
```

These are useful consistency diagnostics, but they are not generally two
estimates of one identical leg sum. They retain initiator issue cost, polling
cost, and any departure work not covered by the responder's `TURN`; those costs
can differ across component classes and transmit modes. For homogeneous,
symmetrically configured rows, agreement is an expected repeatability check.
For heterogeneous rows, the difference is a directional diagnostic and must
not be reported as the measurement error.

The suite does not infer one-way latency by halving a round trip. Even when
directional NOC selections produce equal hop counts, local issue and observation
costs remain in the interval. ID 3 records directional NOC selection and
derived hop counts, and uses same-NOC versus opposite-NOC configurations as
topology controls without presenting a one-way result.

### Placement and sweep policy

Placement is resolved outside the kernel. Each invocation receives one exact
ordered pair and two exact directional access plans.

The latency baseline uses one initiator at a time. Concurrent harts measure
shared-port contention and are a separate environmental-concurrency
experiment, not another placement point.

In addition to the harness common envelope, the latency payload records port
and ordering attributes. For each on-chip NOC leg it also records the selected
NOC and the directional hop count derived by the harness
`hops(from, to, noc_id)` service. A cross-chip ERISC link records link identity
and mode instead; it has no NOC hop count.

Start with a bounded placement probe and compare its spread against
same-configuration repeatability. Expand to a full placement sweep only if the
observed effect is large enough to change placement decisions.

Keep default sweeps separate:

- `payload_sweep` varies payload at a canonical placement and access plan;
- `noc_comparison` varies directional NOC settings for one ordered pair;
- `placement_probe` varies selected targets at one payload and fixed NOC
  context; and
- `full_placement` is optional and fixes all non-placement axes.

If the two directional NOC selectors are varied independently, the sweep must
construct and name all intended directional pairs explicitly. A two-point
comparison must not be described as the full product of two independent binary
selectors.

The primary summaries are p50, p90, and p99. Per-sample records remain
available. The production Blackhole 2-ERISC/multi-TXQ reference uses TT-link
packet-resend mode; its 16-byte hardware header and Go-Back-N behavior are part
of the measured path and are not subtracted. Per-sample records are mandatory
because retransmission is not directly visible to the kernel, and a token-sized
transfer is interpreted as header-dominated.

A raw/compliance ERISC experiment is a distinct mode, not a production-fabric
toggle. It is admitted only when its backend defines an explicit supported
setup; the current production TXQ helper enables packet resend and does not
provide a disable operation. Such a result records its actual mode and does not
attribute a tail to retransmission without evidence.

### State and diagnostics

Both participants publish the same protocol phases:

| State | Meaning |
|---|---|
| `INIT` | parse arguments, consume prepared views, and initialize backend-local state |
| `RENDEZVOUS` | perform the first warm-up exchange |
| `WARMUP` | complete the remaining untimed exchanges |
| `TIMED` | record the current direction's samples |
| `DONE` | successful terminal state |
| `TIMEOUT` | bounded rendezvous or warm-up poll expired |
| `TRAP` | target-specific trap or fatal error |
| `ARG_LAYOUT_MISMATCH` | backend initialization refused the argument section |

Each phase record also carries its direction. Heartbeat or sample progress is
published outside measurement brackets and therefore appears in the harness
timeout diagnostics. Which participant is stuck helps localize the broken
direction without adding instrumentation to the transport.

### Kernel frontend and backend

One suite frontend owns rendezvous, warm-up, both roles, both direction phases,
timing, status, and result publication. One statically selected backend per
component class supplies hardware operations. No NOC, DMA, or Ethernet API
appears in the frontend.

This split is a measurement-validity decision: participant rows must share one
protocol loop so a source-code divergence is not mistaken for a hardware
difference. Exact source paths and C++ names remain implementation choices.

The current interface shape is:

```cpp
enum class Role { INITIATOR, RESPONDER };
enum class TransferShape { TOKEN_ONLY, PAYLOAD_WITH_TOKEN };
enum class PollMode { TIGHT, BACKOFF };
enum class TransmitMode {
    FIRE_AND_FORGET,
    WAIT_FOR_DEPARTURE,
    WAIT_FOR_COMPLETION,
};

struct protocol_args_t {
    Participant self;
    Direction first_direction;
    uint32_t payload_size;
    uint32_t warmup_count;
    uint32_t sample_count;
    uint32_t run_nonce;
    uint64_t warmup_max_poll_cycles;
};

// Defined by exactly one selected component backend adapter.
struct Backend {
    struct state_t;
    struct timestamp_t;
    struct counter_delta_t;

    static void bind_direction(state_t&, Direction);
    static timestamp_t measurement_now(state_t&);
    // Primary counter's t1 - t0 through measurement_delta; false when the
    // interval is refused or either read failed.
    static bool delta(state_t&, timestamp_t, timestamp_t, counter_delta_t* primary);

template <Role role, TransferShape shape>
    static void transmit(state_t&, uint32_t token);

// Measurement poll: minimal and intentionally unbounded on device. False on
// host STOP or a backend's emergency hang guard.
template <PollMode mode>
    static bool poll(state_t&, uint32_t expected_token);

// Rendezvous/warm-up only: relative limit in this participant's counter ticks.
template <PollMode mode>
    static bool poll_for(
        state_t&, uint32_t expected_token, uint64_t max_poll_cycles);

    static uint32_t observed(state_t&);
};
```

The host admits `TOKEN_ONLY` only when `payload_size == 4` and
`PAYLOAD_WITH_TOKEN` only when `payload_size > 4`. Transmit mode is build
configuration interpreted by the selected backend. Invalid transport and
access-plan combinations are rejected before launch.

The entry point copies and validates the member's flat section, initializes its
backend state, and runs both directions:

```cpp
void kernel_main() {
    Section section;
    if (!load_section(args_addr, args_bytes, section)) return;
    if (!Member::validate(section, hartid)) return;

    Backend::state_t backend;
    auto& result = result_at(Member::result_addr(section));
    if (!Member::backend_init(backend, section, hartid, run_value, result)) {
        result.terminal_state = ARG_LAYOUT_MISMATCH;
        return;
    }
    run_participant<Backend, Member::SHAPE, Member::POLL_MODE, Member::TX_MODE>(
        backend, section.common, result);
    Member::backend_shutdown(backend);
}
```

The role branch runs once per phase. The specialized phase starts with one
token calculation and then increments in the loops:

```cpp
const uint32_t W = args.warmup_count;
const uint32_t N = args.sample_count;
uint32_t token =
    args.run_nonce + phase_ordinal * (W + N);

set_status(direction, RENDEZVOUS);
for (uint32_t i = 0; i < W; ++i, ++token) {
    if (i == 1) {
        set_status(direction, WARMUP);
    }

    if constexpr (role == Role::INITIATOR) {
        transmit<Role::INITIATOR, TRANSFER_SHAPE>(backend, token);
        if (!poll_for<POLL_MODE>(
                backend, token, args.warmup_max_poll_cycles)) {
            set_status(direction, TIMEOUT);
            return false;
        }
    } else {
        if (!poll_for<POLL_MODE>(
                backend, token, args.warmup_max_poll_cycles)) {
            set_status(direction, TIMEOUT);
            return false;
        }
        transmit<Role::RESPONDER, TRANSFER_SHAPE>(backend, token);
    }
    publish_warmup_progress(backend, i);
}

set_status(direction, TIMED);
```

The measured bodies contain no timeout or role dispatch:

```cpp
// Initiator specialization.
for (uint32_t i = 0; i < N; ++i, ++token) {
    auto t0 = measurement_now();
    transmit<Role::INITIATOR, TRANSFER_SHAPE>(backend, token);
    auto observed = poll<POLL_MODE>(backend, token);
    store_rtt_sample(backend, phase_ordinal, i, delta(t0, observed));
    publish_progress_if_due(backend, i);  // outside the bracket
}

// Responder specialization.
for (uint32_t i = 0; i < N; ++i, ++token) {
    auto observed = poll<POLL_MODE>(backend, token);
    transmit<Role::RESPONDER, TRANSFER_SHAPE>(backend, token);
    auto finished = measurement_now();
    store_turn_sample(backend, phase_ordinal, i, delta(observed, finished));
    publish_progress_if_due(backend, i);  // outside the bracket
}
```

Warm-up uses the same transfer shape, transmit mode, payload, peer view, and
poll mode as measurement. `poll_for()` applies a relative budget from its own
entry, not an absolute invocation deadline; native unsigned subtraction handles
counter wrap.

For memory-sourced transfers, the initiator places the token in the final word
of its host-seeded staging region before transmitting the complete buffer. The
responder transmits the complete inbound mailbox, preserving payload echo
semantics. The `x280_peer_tile` and `tensix_peer_tile` X280 backend stores the
token directly through its peer window for both roles. A strict
ping-pong is self-draining: observing the reply proves the request traversed the
path before the next sample begins.

The backend seam remains outside the dynamic hot path:

- `transmit()` is the semantic transport boundary;
- backend, role, transfer shape, polling, and transmit mode are statically
  specialized inside a selected artifact;
- polling remains component-specific;
- `Backend::state_t` is constructed once during `INIT`; and
- the kernel records no rates or statistics.

Arguments are copied as one flat section. Section-tagged cookies make a
host/device layout disagreement fail during `INIT`:

```cpp
if (section.cookie_common != cookie_for(UBENCH_SECTION_COMMON) ||
    section.cookie_backend != cookie_for(UBENCH_SECTION_BACKEND)) {
    return false;  // result left untouched
}
```

Cookies are always enabled because they execute outside the measured loop.
Each backend publishes `BACKEND_ID`, and the host verifies it against the
selected placement.

### X280 view binding

X280 device-window allocation, programming, verification, capacity, and
aperture-offset handling follow the harness
[`Memory regions and address views`](../harness.md#memory-regions-and-address-views).
For this suite, ID 2 uses a self-view and no device window. Other X280 outbound
legs program and verify one paired-range window in `INIT` from the coordinate,
address, slot, and ordering their directional access plans select on the host.
Those mappings remain fixed through both direction phases; firmware does not
reprogram a suite-bound window after `INIT`.

X280 aperture transfers use explicit volatile stores; `memcpy()` is not used.

### Backend semantic responsibilities

| Operation | X280 | Tensix |
|---|---|---|
| initiator source | direct token for `x280_peer_tile` and `tensix_peer_tile`; staging for memory-sourced transfers | host-seeded staging region |
| responder source | direct token for `x280_peer_tile` and `tensix_peer_tile`; inbound mailbox for memory-sourced transfers | complete inbound mailbox |
| token-only transmit | one 4-byte store | one 4-byte asynchronous NOC write |
| payload transmit | explicit complete-buffer stores through the selected aperture | one complete asynchronous NOC write |
| transmit return | posted issue or selected completion plan; no departure-only mode | issue, flush, or barrier |
| measured poll | unbounded volatile local load, optionally with pause/backoff | unbounded local L1 load with required cache handling |
| bounded poll | same poll plus relative counter budget | same poll plus relative counter budget |
| measurement clock | harness-fenced `rdcycle` bracket | accepted Metal timestamp counter |
| one-time setup | select local medium and consume host-prepared windows | configure local cache behavior |

Active/raw-link ERISC members must supply the same semantic operations before
their rows are implementation-ready. `drisc_peer_tile` and idle
`erisc_peer_tile` use the same staged Metal NOC write, completion barrier, and
cache-invalidating poll as the Tensix peer; their target policies supply the raw
wall-clock read, backend identity, and, for DRISC, the selected-NIU lifecycle.

For X280, non-posted per-store acknowledgement and posted-buffer-plus-read-back
are distinct completion plans and are recorded separately in the access plan.
Under the posted ordering mode the read-back does not prove the buffered stores
completed.
The Tensix baseline disables the L1 data cache for mailbox polling, following
the existing fabric latency kernels, and uses the cache-invalidating behavior
required by `noc_semaphore_wait`. Any alternative must prove that a cached load
cannot hide remote token arrival.

### Results and validation

Each participant publishes two phase records. A phase record contains:

- phase ordinal and logical direction;
- local role and sample kind (`RTT` or `TURN`);
- raw counter delta for every sample;
- terminal state and progress;
- backend identity; and
- optional backend calibration observations.

The common envelope comes from the harness
[`Result composition`](../harness.md#result-composition).
The suite rejects mismatched backend identity, unexpected sample count,
invalid terminal state, clock disagreement, or token-interval overflow.

Derived-under-model values are labelled as such. A handler cost measured by
`pipeline_stage` may later be combined with a transport result, but that sum is
a model until a composed measurement validates it.

### Open design questions

- Which optional X280 payload-ordering mechanism is functionally valid and
  cheaper than strict NOC ordering?
- Is there a peer-visible Memory Port signalling configuration with a measured
  visibility operation?
- What exact `transmit`, local poll, and bounded-poll primitives implement the
  DRISC and active/raw-link ERISC members?
- What idle condition defines a reproducible cold-latency invocation?

These questions do not block the local ID 2 baseline or the strict-order X280
outbound leg used by `x280_peer_tile`, `tensix_peer_tile`, `erisc_peer_tile`,
and `drisc_peer_tile`.

## Transport bandwidth

### Question and maturity

`transport_bandwidth` measures the achieved rate of one transport leg. Its
timed loop repeatedly issues the same logical transfer; source, destination,
and contents may remain unchanged. It does not inspect payloads, exchange
application credits, or require an echo in the timed path.

The common loop and measurement boundaries are specified. Receiving-ERISC
packet-header ingress (ID 14) is not a member of this suite: it returns ring
credit in its measured path, and is `packet_injection/erisc_to_x280_header`.

A raw-Ethernet member uses the raw link primitive; production link and
application credit semantics remain separate paths.

### Protocol

The canonical state sequence is:

```text
INIT → VERIFY_PATH ─┬→ PATH_MISMATCH
                   └→ WARMUP → TIMED_DRAINED → PUBLISH → DONE
                                └→ TIMED_INJECTION → DRAIN → PUBLISH → DONE
```

- `INIT` parses arguments and performs backend setup.
- `VERIFY_PATH` performs one untimed, member-appropriate sentinel or readiness
  operation.
- `WARMUP` uses the measured `issue()` primitive and then drains.
- `TIMED_DRAINED` issues the batch and drains inside one uninterrupted bracket.
- `TIMED_INJECTION` measures issue only; `DRAIN` completes the same batch after
  the bracket.
- Every exit runs the backend's cleanup (the X280 TLB clear) before it
  publishes its terminal state; a cleanup that fails ends the run
  `WINDOW_CLEANUP_FAILED` instead.
- `PUBLISH` records raw counters and logical counts.

There is no generic remote drain kernel. An addressable sink is passive. A raw
link may need a minimal peer workload to operate the receive primitive, but
that workload does not add payload validation or application flow control.

### Validity and byte accounting

Functional path verification remains outside the timed loop. Repeating one
address is valid only when transport backpressure guarantees that accepted work
reaches the measured leg. A cacheable path must use a span larger than the
absorbing cache or a proven visibility operation; a sentinel alone cannot prove
that repeated timed stores reached the transport.

The addressable-memory definition is:

```text
logical_bytes = uint64_t(N) × uint64_t(SIZE)
rate          = logical_bytes / elapsed_time
```

Both `N` and `SIZE` are 32-bit values. Their maximum product,
`(2^32 - 1)^2 = 2^64 - 2^33 + 1`, fits in `uint64_t`; conversion to 64 bits
must occur before multiplication. The host requires `N > 0`, `SIZE > 0`, and
all member-specific resource limits before launch. A zero elapsed interval is
an invalid result.

For addressable memory, `SIZE` is the number of application bytes moved by one
issue. Raw Ethernet must separately define and report payload bytes and wire
bytes, including mandatory hardware headers, before its rate can be compared
with line rate. Until that accounting and its completion boundary are fixed,
the raw-Ethernet member is not implementation-ready.

### Measurement boundary

```cpp
enum class MeasurementBoundary {
    INJECTION,
    DRAINED,  // baseline
};
```

- `DRAINED` takes `t1` after `drain()`. For Tensix->X280 that follows every
  write's acknowledgement; for X280->Tensix it follows one read through the
  window, which does not prove the posted writes landed (docs/measurement.md). It is
  the registered sustained-throughput baseline.
- `INJECTION` takes `t1` after the last `issue()`, then drains outside the
  measured interval. It describes finite-batch issue rate from a warm, empty
  transport, not sustained throughput.

For one batch whose total completion interval is decomposed into positive
measured issue interval `I` and non-negative final drain interval `D`:

```text
drained_rate / injection_rate = I / (I + D)
```

This identity explains the two boundaries; it is not computed by dividing rates
from separate invocations, whose issue intervals may differ. The two registered
modes are expected to converge for sufficiently large batches, but boundary and
batch size remain recorded so that expectation is tested rather than assumed.

### Frontend and backend contract

One suite frontend owns phase order, the issue loop, measurement bracket,
status, and publication. One statically selected backend owns the component's
transport operations:

```cpp
struct protocol_args_t {
    uint32_t transfer_size;
    uint32_t warmup_count;
    uint32_t transfer_count;
    MeasurementBoundary boundary;
};

// Defined by exactly one selected component backend adapter. The member entry
// point has already copied and validated its flat section before constructing
// this state.
struct backend_state_t;
struct timestamp_t;
struct counter_delta_t;

// Untimed verification of this member's resolved source/destination.
static inline bool verify_path(backend_state_t& backend);

// Issues one logical transfer. Required component-local readiness and
// hardware backpressure are contained here.
static inline void issue(backend_state_t& backend);

// Completes all work previously issued by this component.
static inline void drain(backend_state_t& backend);

static inline timestamp_t measurement_now(backend_state_t& backend);
static inline counter_delta_t delta(
    backend_state_t& backend, timestamp_t t0, timestamp_t t1);

static constexpr uint32_t BACKEND_ID = /* per component class */;
```

`issue()` deliberately takes no iteration or payload pattern. The backend
prepares its fixed source and destination during `INIT`. Address progression is
added only when a member needs it to defeat cache absorption or to study a
specific DMA buffer partition.

For the initial addressable mappings, Tensix `issue()` uses the selected
asynchronous NOC operation and `drain()` uses its matching barrier. X280
posted-store members end with a read through the window, which does not force
completion under the posted ordering mode. DRISC DMA and
ERISC raw-link mappings are not fixed until their concrete members are
specified.

### Kernel entry point

```cpp
void kernel_main() {
    Section section;
    if (!load_section(args_addr, args_bytes, section)) return;
    if (!Member::validate(section)) return;

    backend_state_t backend;
    auto& result = result_at(Member::result_addr(section));
    if (!Member::initialize(backend, section, result)) {
        result.terminal_state = ARG_LAYOUT_MISMATCH;
        return;
    }
    run_member<Backend>(backend);
}
```

The boundary branch executes before `t0` and selects a static specialization.
There is no component, boundary, operation, or correctness branch in the
transfer loop.

### Timed loop

```cpp
template <MeasurementBoundary boundary>
void run_timed(
    backend_state_t& backend,
    const protocol_args_t& args) {

    set_status(boundary == MeasurementBoundary::DRAINED
        ? TIMED_DRAINED
        : TIMED_INJECTION);
    auto t0 = measurement_now();

    for (uint32_t i = 0; i < args.transfer_count; ++i) {
        issue(backend);
    }

    if constexpr (boundary == MeasurementBoundary::DRAINED) {
        drain(backend);
        auto t1 = measurement_now();
        store_raw_cycles(backend, delta(t0, t1));
    } else {
        auto t1 = measurement_now();
        store_raw_cycles(backend, delta(t0, t1));

        set_status(DRAIN);
        drain(backend);  // mandatory, but outside this measurement
    }

    store_transfer_count(backend, args.transfer_count);
    store_logical_bytes(
        backend,
        uint64_t(args.transfer_count) *
            uint64_t(args.transfer_size));
}
```

Device timeout checks and status writes remain outside the measured issue and
drain interval. The harness host deadline remains the safety boundary.

Every function from `run_member` down to `issue()` is `always_inline`. Once
the backend state reaches an out-of-line call, every NoC counter store in
`issue()` forces a reload of the transfer operands, about 9 AICLK cycles per
Tensix -> X280 issue. The X280 -> Tensix firmware's `main`, which receives the
inlined loops, is marked `hot`: GCC otherwise lays out loops in `main` as
run-once code, top-tested with an extra jump per issue.

X280 uses the fenced bracket defined by the harness
[`Measurement discipline`](../harness.md#measurement-discipline).
That bracket does not substitute for the backend's remote-completion `drain()`.
A backend must verify that its `INJECTION` endpoint still corresponds to the
documented `issue()` semantics on that component.

### Members and axes

| Member | Primitive | Readiness |
|---|---|---|
| Tensix write to X280 | asynchronous NOC write into X280 LIM; passive X280 target | landed (`tensix_to_x280`) |
| X280 write to Tensix | X280 hart 0 issues `SIZE/4` 4-byte uncached stores through paired TLB slot 120 in ordering mode 2 (posted) into a Tensix L1 buffer; passive Tensix target. `DRAINED` ends on one read through the window, so it is an issue rate, not a delivered rate, and is recorded as `RETIRED` ([docs/measurement.md "Quantities that share a label"](../measurement.md#quantities-that-share-a-label)) | landed (`x280_to_tensix`) |
| ERISC raw transmit | raw Ethernet issue and completion | peer operation, completion, and byte accounting unresolved |
| DRISC read from Tensix | repeated NOC read from host-seeded Tensix L1 | primitive, drain, and supported depth unresolved |
| DRISC GDDR DMA | repeated DMA read or write | direction, buffer partition, and completion mapping unresolved |
| concurrent pumps | multiple unchanged transport legs | endpoint topology and aggregation contract deferred |

Common runtime axes are transfer size, transfer count, warm-up count, and
measurement boundary. Placement and access plans are host-resolved.
Outstanding depth and buffer partition are member-specific axes; they are not
forced into a universal frontend parameter.

The first concrete member design must state:

- exact source and destination memory;
- alignment and size constraints;
- `issue()` acceptance semantics;
- `drain()` completion semantics;
- warm-up and path-verification behavior;
- logical versus physical byte accounting;
- required functional preconditions; and
- result fields and reference bound.

### Results

The device stores transfer count, transfer size, logical bytes, raw batch delta,
boundary, terminal state, and backend identity. A result is invalid if path
verification fails, the backend identity mismatches, the invocation times out,
the byte definition is ambiguous, or a cacheable path lacks a proven visibility
mechanism.

## Pipeline-stage cost

### Question and maturity

`pipeline_stage` measures the per-item X280 cycle cost of a processing stage
and whether it fits the available per-item budget.

For positive subject clock `f_subject` in cycles per second, positive payload
`P` in bytes, and positive line rate `L` in bytes per second:

```text
budget_cycles = f_subject × P / L
```

The clock must be the clock of the class running the subject. The payload and
line-rate definitions must match the traffic being budgeted.

The current X280 contract is concrete. `route_compute` performs one
destination-to-next-hop direction-table lookup. `route_forward` has one
fixed-source configuration, `decide_plain`. Exact layouts, arguments, result
identity, prepared-input requirements, and the historical boundary are defined
in
[`docs/benchmarks/pipeline_stage.md`](pipeline_stage.md).

### Stage spine

The candidate per-item spine is:

```text
pushed header
    -> load header
    -> parse
    -> table read
    -> decide
    -> issue action
```

Memory operations define useful boundaries. Action issue is local stage work;
transport after that issue is measured by a transport suite rather than silently
included in a compute-stage measurement.

Initial subjects are routing, forwarding, credit receive, and credit issue.
The current routing member does not make route construction, lookup style,
packet tier, table placement, or working-set size into axes. A future subject
must justify any new axis from its own measurement question rather than inherit
the retired lookup matrix.

### Measurement granularities

- **Module:** time one stage per item. This is the primary measurement.
- **In-situ phase:** timestamp stage boundaries inside the full chain and
  analyze intervals after the run.
- **Chain:** time the complete composition per item. This is the additivity
  ground truth.

An isolated stage can optimize differently from the same stage in a chain.
Module-versus-chain residual is recorded rather than assumed away. Compiler
barriers at declared boundaries constrain cross-stage motion without claiming
to eliminate all compiler effects.

Stages are functions in the subject headers, each member with one X280
backend. The fixed pipeline order avoids a power set of arbitrary
stage combinations. The current lookup has two configurations only: plain and
per-sample instrumented. Instrumentation is a measurement mode, not another
lookup implementation.

### Bring-up and input seeding

Bring up the timing shell before subject logic:

1. An all-empty skeleton validates argument handling, clock metadata, batched
   timing, output consumption, and result publication.
2. A real stage should have a thin input and minimal dependence on missing
   upstream stages. The current lookup reads sixteen seeded packet headers and
   the complete direction-table image.
3. Fill stages in pipeline order so upstream prefix replay becomes available.

Every real stage input is deliberate and every output is consumed. An empty
skeleton is a measurement-instrument check, not a performance estimate.

Inputs fall into two groups:

- **State inputs**, such as routing and connection tables, come from production
  generators where available.
- **Per-item streams** use thin scalar interfaces that can be seeded without
  reproducing upstream semantics.

Input preference order:

1. untimed prefix replay through implemented upstream stages;
2. direct host seeding for a thin interface; then
3. production trace capture when correlations make synthetic input suspect.

Seeded and replayed paths produce the same in-memory stage input before the
timed loop.

For the current lookup and forwarding source, the in-memory input is fixed
rather than swept. It is
one 4288-byte image containing the complete 2704-byte routing layout and
sixteen 96-byte packet headers. The fixture covers same-mesh and cross-mesh
table selection and varying expected directions. The host supplies an
independent direction oracle, and the timed loop consumes the returned
directions through a witness. Forwarding always consumes the first header at
`0x0A000AC0`.

### Prepared storage

The lookup image resides at `[0x0A000000, 0x0A0010C0)` in cached Zero Device.
It requires the exact 14-LIM-way shape with way 1 pinned and way 0 retained for
general cache use. All masters keep active replacement bits `01` while the
compatible sequence runs.

Preparation and final teardown are suite operations outside the case lifetime:
each pipeline executable prepares before its first device case and tears down
after its last. Ownership and fresh-placement provenance are prerequisites the
suite cannot establish; it prepares only on a tile its own process booted from
reset-held, so each executable runs after an ASIC reset. Between cases
there is no cache flush, `WAYENABLE` change, mask change, repinning, allocation,
or restoration. A failed setup, crash, or unconfirmed participant cleanup ends
the prepared sequence.

Both pipeline executables require `LimShape{14}` with `RequireExact`. Latency
members and both bandwidth executables run at the reset shape.
Legacy/stress suites, cache experiments, and unreviewed executables remain
outside that sequence.

### Current layouts and subject boundaries

- One concrete packet header is 96 bytes with a 35-byte route buffer.
- Packed destination chip and mesh fields begin at offsets 83 and 85. Base
  alignment does not permit aligned loads of those fields.
- The complete routing image is 2704 bytes, tt-metal's `routing_l1_info_t`. Its intra-mesh direction
  table begins at offset 36 and its inter-mesh table at offset 132.
- `route_compute` selects the intra table by destination chip when the
  destination mesh equals the local mesh; otherwise it selects the inter table
  by destination mesh. It performs no route construction, compressed-route
  expansion, or packet publication.
- `route_forward` uses one pinned Zero Device source. DECIDE has one plain
  load/decode/cursor/witness configuration and opens no NoC window.

### Counters and validation

Per-item intervals use the class's accepted high-resolution counter and the
harness timing bracket for that counter. A lower-resolution independent clock
may bracket the batch for a frequency residual. Store-producing stages close
their bracket after the declared output-visibility step.

Module, phase, and chain results record raw intervals and the exact source,
artifact, instrumentation mode, input source, clock metadata, and every
dimension the member actually varies. Pattern and placement are recorded only
for members that vary them. Instrumentation-on versus instrumentation-off
totals provide a perturbation check.

The current lookup reports fixed packet bytes, fixture identity and header
count, prepared-input base and extent, admitted `wayenable`, pinned-way and
way-mask identity, instrumentation, counts, raw counter deltas, per-item
reductions, a stage witness, and verification state. A result from the retired
construction/compressed-route profile is not a baseline for this subject.

## Deferred and reference measurements

### Client handling

An arrival-driven client-handling suite may later measure per-writer mailbox
scanning, connection-state lookup, and service scaling across clients and
harts. Under the harness X280 single-writer invariant, it uses one cell per
producer rather than one shared atomic counter. Producer topology,
live-versus-seeded input, synchronization, result shape, and reference arms
remain unresolved.

### Composed experiments

Multi-participant credit cycles, chained data moves, and a pipeline distributed
across several X280 harts require a composed-experiment model. They remain
deferred until the primitive latency, bandwidth, and stage measurements show
that composition is decision-relevant.

A distributed hart pipeline would combine measured stage costs with same-tile
handoff cost. That arithmetic is a model until the composed run measures its
overlap and contention directly.

### Production-path references

Production-fabric line rate, configured maximum-payload throughput, existing
fabric round-trip latency, and production-router phase telemetry are reference
measurements. The harness supplies recording discipline, but the production
datapath is not reclassified as a ubench backend.

## Open questions

The current design leaves the following suite decisions to the implementation
work that first needs them. Host API, argument-placement, and output-writer
questions remain in the harness LLD rather than being reopened here:

- optional X280 ordering and Memory Port mechanisms;
- DRISC and active/raw-link ERISC latency backend primitives;
- concrete transport-bandwidth member contracts;
- client-handling and composed-experiment topology.

An open question does not authorize an implementation to silently choose new
measurement semantics. Any choice that changes a timing boundary, validity
rule, participant topology, or result interpretation updates this document.

## Stable implementation references

Host-runtime and X280 window implementation references are maintained in the
harness LLD. Suite implementations additionally rely on:

- [`tt_metal/hw/inc/api/dataflow/noc.h`](../../../../hw/inc/api/dataflow/noc.h)
  — Tensix NOC issue, flush, and barrier operations.
- [`tests/tt_metal/tt_fabric/test_infra/kernels/tt_fabric_latency_sender.cpp`](../../../../../tests/tt_metal/tt_fabric/test_infra/kernels/tt_fabric_latency_sender.cpp)
  and
  [`tests/tt_metal/tt_fabric/test_infra/kernels/tt_fabric_latency_responder.cpp`](../../../../../tests/tt_metal/tt_fabric/test_infra/kernels/tt_fabric_latency_responder.cpp)
  — production-fabric RTT prior art for cache handling and polling, not the
  p2p mailbox backend.
- [`tt_metal/fabric/routing_table_generator.cpp`](../../../routing_table_generator.cpp)
  and
  [`tt_metal/fabric/control_plane.cpp`](../../../control_plane.cpp)
  and
  [`tt_metal/fabric/compressed_routing_path.cpp`](../../../compressed_routing_path.cpp)
  — host routing-direction generation, compressed/source-routed forms, and
  device routing-table publication.
- [`tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp`](../../../hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp)
  and
  [`tt_metal/fabric/hw/inc/edm_fabric/fabric_txq_setup.h`](../../../hw/inc/edm_fabric/fabric_txq_setup.h)
  — production worker/EDM buffer-slot flow control and Ethernet TXQ/RXQ plus
  packet-resend setup; these are not application-credit semantics.
