# Cross-Cluster Signalling Characterization

**Status:** Measured on silicon. The numbers, the placement sweep and the run
environment are in
[`CHARACTERIZATION_RESULTS.md`](CHARACTERIZATION_RESULTS.md) § 3.3 and § 6.

**Scope:** Phase 0 feasibility benchmark ID 3 (priority P2): hart-to-hart
signal latency where the two harts sit on different X280 complexes. This is
the [`ubench design`](CHARACTERIZATION_UBENCH_DESIGN.md)'s `p2p_latency`
ID 3 — X280 tile to X280 tile, one chip — and is implemented as that row
rather than as a new suite.

## Document boundaries

Host lifecycle, ownership, regions, deadlines, safety, clock mechanisms, and
the common result envelope are owned by
[`CHARACTERIZATION_HARNESS.md`](CHARACTERIZATION_HARNESS.md). Protocol loop,
roles, measurement boundaries, axes, and the frontend/backend contract are
owned by [`CHARACTERIZATION_UBENCH_DESIGN.md`](CHARACTERIZATION_UBENCH_DESIGN.md)
§ Peer-to-peer latency. This document adds only what ID 3 needs beyond
those two.

Citations of the form `NoC/RoutingPaths.md` and `L2CPUTile/Caches.md` are
`tt-isa-documentation`'s `BlackholeA0` tree (the public Tenstorrent ISA documentation
repository), which is not a submodule of this repository.

## Tile placement

The four L2CPU complexes sit in NOC0 column 8 — hardware indices 0 to 3 at
`(8,3)`, `(8,9)`, `(8,5)`, `(8,7)` (`noc0_from_hw_l2cpu` in
`src/device/x280_state.cpp`) — so
every inter-complex leg runs the 12-row column ring and only the y term of
`hops(from, to, noc_id)` contributes (`harness/inventory/inventory.hpp`).
Router-to-router hops on NOC0, in each direction:

| Pair | a → b | b → a |
|---|---|---|
| 0 ↔ 1 | 6 | 6 |
| 0 ↔ 2 | 2 | 10 |
| 0 ↔ 3 | 4 | 8 |
| 1 ↔ 2 | 8 | 4 |
| 1 ↔ 3 | 10 | 2 |
| 2 ↔ 3 | 2 | 10 |

The rows all sum to 12, and the ISA documentation says why: NOC0 flows
rightwards and downwards only, NOC1 leftwards and upwards, each a torus with
its edges joined (`NoC/RoutingPaths.md:11`, `:35`), which `ring_distance`
reproduces by returning the forward distance rather than the shorter one
(`hops()` in
`harness/inventory/inventory.hpp`). The invariance runs deeper than the
table shows, because acknowledgements travel the way requests do: on each NOC,
unicast requests and all types of response and acknowledgement packet take
that NOC's one direction (`NoC/RoutingPaths.md:17`, `:41`). A non-posted
write's ack returns the long way round rather than retracing the request, so a
single non-posted store crosses exactly one lap — 12 hops — on whichever pair
it runs. ID 3's exchange is two such stores, which puts 24 hops of routing in
every RTT independent of placement.

The count is over every physical node on the ring. Whether a harvested tile
still forwards, and so whether harvesting shrinks the effective ring, is
settled in neither tree and the service encodes no guess about it. A measured
latency that does not track these counts on a harvested part puts that
assumption first in line.

## Why the backend is NOC-shaped

LIM is private to its complex. ID 2 signals with plain loads and stores
because both of its harts address one LIM; a hart on another complex has no
load/store path to that memory. Cross-complex signalling therefore traverses
the NOC, and the ID 3 backend is shaped like the X280-Tensix channel's — program a
paired-range TLB window onto the peer's inbound mailbox, store the token
through the window, poll own LIM with `fence ir, ir` — with the peer's LIM
in place of Tensix L1 as the destination.

Three consequences follow from the destination being LIM rather than L1:

- An inbound NOC write into peer LIM terminates in the scratchpad and cannot
  allocate an L3 line. L3 is sixteen 128 KiB pieces, each configured as a
  regular cache way, a pinned way, or uncached scratchpad, and LIM is the
  scratchpad option (`L2CPUTile/Caches.md:68-71`); the data-cacheable physical
  ranges are enumerated and LIM is not among them (`:14-15`, `:27-30`). A NOC
  access to a data-cacheable address is performed through the L3 cache and is
  coherent with the caches (`:77`). That same sentence adds that coherence does
  not extend between L2CPU tiles, which is why a cross-complex mailbox belongs
  in uncached LIM rather than anywhere the two tiles' caches would have to
  agree.
- Inbound NOC atomics cannot target X280 LIM, and the outbound path has none
  to offer either: the System Port the window pointer addresses is uncached
  and carries no atomics under its PMA (`NOC_2M_WINDOW_BASE` in
  `include/noc.h`). Each mailbox
  therefore keeps exactly one writer, which the `p2p_latency` mailbox model
  already satisfies.
- The `fence ir, ir` the poll issues before every load orders the uncached
  path rather than a stale line. It drains the inbound store buffer,
  and without it a LIM poll wedges even through `volatile`
  (`kernels/p2p_latency/backends/x280_noc.hpp:133-147`); the wedge is not a
  cached value, because LIM is not a data-cacheable address
  (`L2CPUTile/Caches.md:27-30`; LIM is the uncached scratchpad at
  `0x0800_0000`, `L2CPUTile/MemoryMap.md:84`). The fence is required either
  way. There is no window on the receive side, which is why the mailbox must
  live in the receiving hart's own LIM.

## Running it

ID 3 needs two complexes in live resident idle, and reset release is one-way
(D-022), so every tile it uses must leave reset inside one chip-reset
lifetime. The harness's session fixture boots every enabled L2CPU of a chip
whose L2CPUs are all held in reset in one batch (README § Benchmark harness),
and there is one batch per lifetime, so the placement sweep finds every pair
live. The device cases that require a reset-held tile
(`ColdBootRefusedWithoutOptIn`,
`OversizedIdleImageIsRejectedWithoutWritingLim` and
`OversizedActiveImageIsRejectedWithoutWritingLim` in
`tests/device/test_x280_device_hw.cpp`) then skip until the next ASIC reset. Extra live tiles
do not perturb the measurement — the resident idle touches no DRAM, AXI, or
MMIO outside its own L2CPU tile (`src/lim_idle.c:20-22`) — and cost only
power. Images, environment knobs and the full sequence are in
`CHARACTERIZATION_RESULTS.md` § 6.

## Open questions

- Whether load changes any of this. Deferred, and the one qualification the
  lap invariance carries: it is a statement about hop count, not about
  queueing delay. Congestion occurs when two packets contend for the same
  link between routers (`NoC/RoutingPaths.md:59`), so a 10-hop leg offers ten
  routers' worth of contention where a 2-hop leg offers two, and under traffic
  placement can matter even though the hop count does not. Every number this
  member produces is a quiet-part number: one exchange in flight, the other
  complexes in resident idle, no concurrent fabric traffic. What the deferred
  work would establish is how the total degrades against offered load and
  whether the pairs separate once they are contending, which needs a traffic
  generator this set does not build.
