# PR description for #67119: host-initiated RDMA on a DRAM core

Paste the sections below into the PR body. Verification numbers are from `80c8c9cf845`, before `main` was merged in; update the "Not yet verified" list once the merged branch has been built and run.

### PR Category

Feature

### Summary

Part of #66251. Moves the host-initiated RDMA request service from a Tensix worker core to a DRAM core (DRISC),
freeing that Tensix core for compute.

The request format, the service's packet loop, the host API and the fabric routers are unchanged. The work is in
what a Tensix core provided implicitly and a DRAM core does not.

| Area | Tensix (before) | DRAM core (this PR) |
|---|---|---|
| Launch | Mesh workload on the command queue | Program compiled and launched through slow dispatch with `force_slow_dispatch`, as the tensor prefetcher launches DRAM kernels; stopped with `WaitProgramDone` |
| Host access to the service's L1 | Plain L1 addresses | Every read and write adds the DRAM-core L1 NoC offset |
| Kernel build | Defaults | Fabric kernel defines passed explicitly, and `-O2` |
| Routing table | Already in the core's L1 | Copied once at start-up from a Tensix core on the same chip |
| Route lookup | `fabric_set_unicast_route(header, dev, mesh)` | New overload taking a pointer to the table |
| NoC | NoC 0 | The NoC the fabric routers credit workers on |

`ServiceCore::Dram` is the default. The Tensix path is still present and selectable with `ServiceCore::Tensix`; it is
unused and intended for removal in a follow-up.

### Notes for reviewers

**Router credits and the NoC rule.** A router returns credits by writing to a plain 32-bit L1 address on the
producer. A DRAM core's L1 is normally addressed with a high offset that the router's address format cannot carry.
No router change is needed, because a plain address does reach DRAM-core L1 through a NIU in stream mode, which is
also the only mode a DRAM core can initiate NoC traffic from. `DramKernelPlainAddrFixture` checks this on hardware
for a write and for an atomic increment, on both NoCs.

The consequence is a placement rule: the service must run on a DRAM core whose NIU is in stream mode on the NoC the
routers credit on (NoC 0 with two ERISCs per router, NoC 1 with one). `HostRdmaService` checks this at construction
and fails with a message naming the core and the NoC.

**Routing table.** The control plane writes each chip's routing table to Tensix and Ethernet cores only. The
service reads it once at start-up from Tensix logical core (0, 0) into its own L1 (2,704 bytes, placed between the
layout and the staging slots) and routes from that copy. Three words of `Header` padding carry the copy's address
and the source core's coordinates.

**Shared header.** `tt_fabric_api.h` is the only file changed outside host RDMA. `fabric_set_unicast_route` gained
an overload that takes the routing table as an argument; the existing three-argument form calls it with
`ROUTING_TABLE_BASE`. Router `.text` built with and without this change is byte-identical.

**Build flags.** DRAM kernels are not given the fabric kernel defines and default to `-Os`. Without the defines the
kernel builds with the 1D packet header. `-Os` cost about a third of the throughput at 16 KB and above; `-O2`
restored it and produced a slightly smaller kernel (5,844 bytes of text).

**Test behaviour.** `test_host_rdma.cpp` runs the service on the first DRAM core tt-metal holds that satisfies the
NoC rule, and skips when DRAM programmable cores are off (firmware older than 19.12.0.0, or their seats held by
another process). A CI machine in that state would pass this job without running the tests.

**Not in this PR.**
- Running alongside the DRISC service. tt-metal still takes every DRAM core it can program. A flag to take only
  named seats is a separate change.
- Inline L1 writes from a DRAM core use the Tensix scratch location, which falls in the DRAM core's reserved first
  64 bytes of L1. The service's open and close handshakes are the first DRAM-core code to use them. This works but
  should be given a DRAM-core branch in `noc_nonblocking_api.h`.
- The limits of the Tensix version are unchanged: writes only, one domain, Blackhole, 2D fabric, and callers must
  `drain()` before teardown.

### Verification

Run on eight Blackhole p150b boards in a 2x4 mesh, firmware bundle 19.16.1, at `80c8c9cf845`.

| Check | Result |
|---|---|
| `HostRdmaFabric2DFixture.*`, slow dispatch | 4 of 4 |
| `HostRdmaFabric2DFixture.*`, fast dispatch | 4 of 4 |
| `HostRdmaFabric2DFixture.*`, five repeated runs | 20 of 20 |
| `HostRdmaFabric2DFixture.*`, single-ERISC routers (`TT_METAL_DISABLE_MULTI_AERISC=1`) | 4 of 4 |
| `Fabric2D*Fixture.*` | 40 passed, 13 skipped (need more hops than a 2x4 mesh has) |
| `FabricTrafficGeneratorKernelIntegrationTest.*` | 1 of 1 |
| `Fabric1D*Fixture.*` | 27 passed, 7 skipped (other chip counts, T3000-only, slow-dispatch-only) |
| `*ChannelTrimming*`, multi-ERISC and single-ERISC | 63 passed, 3 skipped (T3000-only), each |
| Router `.text`, this branch against `main`, feature off | identical across 128 router binaries |
| `DramKernelPlainAddrFixture.*` | 2 of 2 (NoC 0 and NoC 1) |

Single-hop bandwidth through the service, batch of 1, from `DISABLED_Bandwidth`:

| Request size | Tensix | DRAM core |
|---|---|---|
| 1 KB | 1.37 GB/s | 1.85 GB/s |
| 4 KB | 5.42 GB/s | 7.33 GB/s |
| 16 KB | 8.65 GB/s | 8.56 GB/s |
| 64 KB | 9.27 GB/s | 9.21 GB/s |

At batch sizes of 8 and 32 the two are within about 1.5% of each other at 1 KB and 4 KB.

With the DRISC seat locks (#67011) merged in, the host RDMA tests passed 4 of 4 again with the seat-aware core
selection.

Not yet verified:
- The branch after merging `main` at `22e8a7c341a`.
- Galaxy (32 chips).
- CI on `bh_loudbox` and `bh_quietbox_2`, and the Wormhole and simulator builds of `tt_fabric_api.h`.

### Code owners

- [ ] Ping pending code owners
