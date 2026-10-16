# p2p_latency: one live exchange between two cores

**Status:** Measured on silicon. Results are in [`docs/results.md`](../results.md); running it is
below.

**Scope:** hart-to-hart
signal latency where the two harts sit on different X280 complexes. This is
the [`ubench design`](design.md)'s `p2p_latency`
`x280_peer_tile` — X280 tile to X280 tile, one chip — and is implemented as that member
rather than as a new suite.

## Document boundaries

Host lifecycle, ownership, regions, deadlines, safety, clock mechanisms, and
the common result envelope are owned by
[`docs/harness.md`](../harness.md). Protocol loop,
roles, measurement boundaries, axes, and the frontend/backend contract are
owned by [`docs/benchmarks/design.md`](design.md)
§ Peer-to-peer latency. This document adds only what `x280_peer_tile` needs beyond
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
it runs. `x280_peer_tile`'s exchange is two such stores, which puts 24 hops of routing in
every RTT independent of placement.

The count is over every physical node on the ring. Whether a harvested tile
still forwards, and so whether harvesting shrinks the effective ring, is
settled in neither tree and the service encodes no guess about it. A measured
latency that does not track these counts on a harvested part puts that
assumption first in line.

## Why the backend is NOC-shaped

LIM is private to its complex. `x280_same_tile` signals with plain loads and stores
because both of its harts address one LIM; a hart on another complex has no
load/store path to that memory. Cross-complex signalling therefore traverses
the NOC, and the `x280_peer_tile` backend is shaped like the X280-Tensix channel's — program a
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

`x280_peer_tile` needs two complexes in live resident idle, and reset release is one-way
([harness.md](../harness.md#design-rules)), so every tile it uses must leave reset inside one chip-reset
lifetime. The harness's session fixture boots every enabled L2CPU of a chip
whose L2CPUs are all held in reset in one batch ([harness.md](../harness.md#running-benchmarks)),
and there is one batch per lifetime, so the placement sweep finds every pair
live. The device cases that require a reset-held tile
(`ColdBootRefusedWithoutOptIn`,
`OversizedIdleImageIsRejectedWithoutWritingLim` and
`OversizedActiveImageIsRejectedWithoutWritingLim` in
`tests/device/test_x280_device_hw.cpp`) then skip until the next ASIC reset. Extra live tiles
do not perturb the measurement — the resident idle touches no DRAM, AXI, or
MMIO outside its own L2CPU tile (`src/lim_idle.c:20-22`) — and cost only
power. Images and environment knobs are under § Metal peers and in
[harness.md](../harness.md#running-benchmarks).

## Metal peers

All three Metal-peer executables consume
`X280_P2P_LATENCY_METAL_PEER_TILE_FW_IMAGE`, defaulting to
`build/x280/firmware-p2p-latency-metal-peer-tile.bin`. They compile
`kernels/p2p_latency/metal_peer_entry.cpp` with exactly one of
`P2P_LATENCY_TARGET_TENSIX`, `P2P_LATENCY_TARGET_IDLE_ERISC`, or
`P2P_LATENCY_TARGET_DRISC`, and with the peer's transmit mode in
`P2P_LATENCY_METAL_PEER_TX_MODE`. The X280 backend ID is `0x10000006`;
the Metal result retains its target-specific backend ID. The shared
`metal_peer_args.h` cookie identifies the 16-word layout, independently of
target selection.
Unless `P2P_LATENCY_{TENSIX,ERISC,DRISC}_PEER_TILE_W` is set, each first runs
an unrecorded `W=8` pilot and sizes the measured launch's warm-up to 25 ms of
device work, so the AICLK elapsed bound stays well inside its limit
([measurement.md](../measurement.md)); a set value is used as given.

The environment variable `P2P_LATENCY_METAL_PEER_TX_MODE` selects the Metal
peer's transmit mode: `wait_for_completion` (also when unset) ends `transmit()`
at the write's acknowledgement, `wait_for_departure` at the NIU having sent it.
The X280 side always waits for completion. Every record names its participant's
mode in `tx_mode`. The manifest's `.departure` configurations run the second
mode, and the tabulator's `TURN split` table gives, per Metal peer and first
direction, the peer's TURN at each mode and their difference, the
acknowledgement wait. Both terms are the peer's own wall ticks.

`x280_p2p_latency_erisc_peer_tile` launches only an inactive Ethernet core in
`Eth::IDLE` mode and exchanges 4-byte tokens over same-chip NOC; it sends no
Ethernet-link traffic. The ERISC participant reports raw wall-clock ticks with
per-run AICLK evidence. The host reads CMFW's programmed maximum and validates
start, end and whole-run counter rates plus throttle telemetry before publishing
a conversion rate. It is an implementation smoke path, not a production fabric
transition. It and `x280_p2p_latency_drisc_peer_tile` link `x280_slow_dispatch`
([harness.md](../harness.md#running-benchmarks)).

`x280_p2p_latency_drisc_peer_tile` selects bank 0's free non-endpoint DRISC and
exchanges 4-byte tokens over NOC0 as a mechanical same-chip smoke target. Its
kernel is launched through forced slow dispatch with the tensor prefetcher
inactive. It requires NOC0 in the stream mode DRISC firmware sets, and switches
no NIU. X280 addresses the DRISC mailbox by its
untagged local offset while host reads and writes use the tagged DRISC-L1 view.
DRISC measurements are raw `get_timestamp()` ticks carrying the same per-run
AICLK evidence and externally usable conversion rate.

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
