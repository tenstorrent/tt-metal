# HostMeshSocket: a D2D socket over the host interconnect

<!-- pyml disable MD013 -->

> **Scope:** a device-to-device socket whose two endpoints live on different hosts and are
> connected by host RDMA (RoCE) rather than TT-Fabric. Ethernet D2D is `MeshSocket`; the
> host/device legs on their own are `D2HSocket` / `H2DSocket` (see
> [HDSocketsModel.md](HDSocketsModel.md)).
>
> **Hardware scope: two Blackhole Galaxies with a RoCE path between their hosts.**

## 1. Why

`MeshSocket` reaches its peer over TT-Fabric. When two meshes have no ethernet between them
but their hosts share a NIC fabric, the same stream can be carried over the host
interconnect instead. `HostMeshSocket` does that while keeping `MeshSocket`'s interface, so a
program that already drives a D2D socket changes its socket type and one kernel compile-time
argument, not its structure.

## 2. Data path

```text
Galaxy A                Host A                      Host B                 Galaxy B
sender tensix           relay poll loop             relay poll loop        recv tensix
 socket_reserve_pages    pages_available()           poll CQ for imm        socket_wait_for_pages
 chunked PCIe write -->  pinned D2H ring  --RDMA-->  pinned H2D ring  -->   chunked NOC read
 socket_push_pages       post_write x N              commit_pages()         socket_pop_pages
 socket_notify_receiver  + doorbell write            (one TLB write)        socket_notify_sender
      ^                       |                            |                     |
      +---- bytes_acked --  pop() on local completion   post_credit <-- bytes_acked
```

Legs 1 and 3 are `D2HSocket` and `H2DSocket` (in `DEVICE_PULL` mode by default;
`TransportConfig::h2d_mode = H2DMode::HOST_DMA` moves leg 3 to the PCIe DMA engine, see §7)
unchanged. Only the host-to-host leg is new. It sits behind the `HostTransport` interface
(`host_transport/host_transport.hpp`), implemented by `MpiTransport`: two-sided
point-to-point through `DistributedContext`.

**Bandwidth is the MPI build's, not this code's.** `MPI_Isend`/`Irecv` run over whatever the
local MPI has underneath — `pml_ucx` or `mtl_ofi` over RDMA where those exist, plain TCP where
they do not. That is the whole reason this leg is MPI rather than raw verbs: ~990 lines of
queue-pair, memory-region and completion-queue management disappear, and the deployment's own
MPI decides how the bytes cross. The cost is that a misconfigured MPI degrades silently to
TCP, which §4 quantifies.

The seam is *how many pages are valid in the ring*, not *post a write*, so a one-sided
transport can be added later without touching the relay: one-sided would land bytes with the
NIC and have the receiver read a counter, two-sided has the receiver complete a receive, and
both answer the former.

### The device side needs no new primitives

`tt_metal/hw/inc/api/socket_api.h` is already transport-agnostic. `socket_reserve_pages`,
`socket_push_pages`, `socket_wait_for_pages`, `socket_pop_pages`, `socket_barrier`,
`set_*_socket_page_size` and `update_socket_config` do not know what the transport is, and the
only two calls that do — `socket_notify_receiver` and `socket_notify_sender` — already branch
on the existing `is_d2h` / `is_h2d` discriminators in `hostdev/socket.h`. This socket sets
those, so neither header changes.

What does differ per transport is the payload move itself: a D2D sender writes to a downstream
NOC address while this sender writes to a PCIe address, and a D2D receiver copies L1 to L1
while this receiver issues chunked NOC reads from host memory. The test kernels
(`host_socket_{sender,receiver}.cpp`) select between the two on a `SOCKET_MODE` compile-time
argument, which is what makes swapping transports a one-argument change.

### No copy on our side of the host

Both FIFOs are page-aligned `NamedShm` mappings handed straight to `MPI_Isend`/`Irecv` as
spans, so the pinned ring *is* the staging buffer and this code never copies a payload byte.

Whether a copy happens *below* that is the MPI implementation's choice: a rendezvous protocol
over UCX registers the ring and lets the NIC read it directly, while an eager protocol copies
into MPI's own buffers first. That is part of why the TCP figure in §4 is what it is, and it is
not something this layer can control.

### Ordering and pipelining

One message per page, so there is no framing and no message ever spans the ring wrap. MPI
point-to-point between a rank pair on one tag is ordered, so the receiver's queued receives
fill in send order and each lands on the page the sender took it from — which is what lets
both sides agree on the page index without putting it on the wire.

The sender keeps a deque of `Isend` requests and retires only from the front, because
completion order is not guaranteed and a count that retired out of order would claim a page is
free early. The receiver keeps a receive posted for every page the ring can hold, topped up as
the device consumes, so a receive is always already waiting when payload arrives. A single
stream is never striped across links; parallelism comes from more streams (§ Multiple planes).

**Deliberately point-to-point rather than MPI one-sided RMA**, which looks like the closer
analogue to verbs but is not usable here: `MPI_Rput` completes on origin-buffer reuse rather
than remote visibility, separate `Rput`s to one window are unordered, and `MPI_Cancel` is
illegal on an RMA request, so a half-finished stream cannot be torn down.

### Flow control

The same `bytes_sent` / `bytes_acked` credit scheme as every other socket, with one extra hop:

| counter | lives in | written by |
| --- | --- | --- |
| D2H `bytes_sent` | pinned host RAM on A | sender device |
| D2H `bytes_acked` | pinned host RAM on A | relay A, on local send completion |
| H2D `bytes_sent` | device L1 on B | relay B, one TLB write |
| H2D `bytes_acked` | pinned host RAM on B | receiver device |
| host doorbell | registered slot on B | relay A, by RDMA (absolute page total) |
| host credit | registered slot on A | relay B, by RDMA (absolute page total) |

The last two rows are the RDMA backend's. `MpiTransport` carries the same two signals as
messages instead: the doorbell is implicit in a payload receive completing, and the credit is a
single absolute `uint64_t` on its own tag, with at most one send in flight — being absolute,
a skipped one costs latency and nothing else.

The host credit is an **absolute page count**, so a duplicated or stale update costs a lap of
latency and never corrupts. Relay A retires a batch only once the NIC has finished reading it;
that retirement is what frees D2H ring space, which is what unblocks the sender kernel.

### Non-blocking relay

One process-wide thread runs a single non-blocking sweep over every registered connection, so
owning a socket imposes no polling duty. It drains fully before parking, then spins briefly
and backs off to a 50 µs sleep. Set `TransportConfig::own_relay_thread = false` and call
`poll()` to drive it inline instead, which makes a failure deterministic under a debugger.

An endpoint is **single-consumer**: it owns the MPI requests for its stream, which exactly one
thread may test. `poll()` therefore refuses to run on a socket the relay thread
already owns, and `barrier()` does not poll in that case — two threads testing the same requests
double-consume completions and corrupt the arrival and credit accounting. An exception
from `poll()` is recorded against the endpoint and surfaced by `barrier()` rather than
escaping the relay thread, where it would terminate the process instead of failing the test.

## 3. Usage

```cpp
#include <tt-metalium/experimental/sockets/host_mesh_socket.hpp>

SocketMemoryConfig mem(BufferType::L1, /*fifo_size=*/14336 * 64);
SocketConfig config(connections, mem, /*sender_rank=*/Rank{0}, /*receiver_rank=*/Rank{1}, ctx);

HostMeshSocket::TransportConfig transport;
transport.page_size = 14336;          // must divide fifo_size
transport.max_batch_pages = 8;

HostMeshSocket socket(mesh_device, config, transport);
// ... CreateKernel(..., {socket.get_config_buffer_address(), ...}) ...
socket.barrier();
```

Both ranks construct the socket with the *same* `SocketConfig`; each picks its own role by
comparing its rank to `sender_rank` / `receiver_rank`, exactly as a rank-scoped `MeshSocket`
does. A rank that is neither logs a warning and allocates nothing.

`get_config_buffer_address()` is one address valid on every endpoint core, because the per-core
sockets share a single height-sharded config buffer.

**`get_data_buffer()` throws.** `MeshSocket`'s data buffer *is* its FIFO in device L1; here the
FIFO is a pinned host ring, so there is no socket-owned device buffer. A receiver kernel pulls
into a landing buffer the caller allocates.

## 4. Measured throughput

Read this section in two halves: what the *datapath* can carry, and what the *MPI build*
delivers over it. They are different numbers, and the gap is the main thing to know before
deploying this.

### What the datapath can carry

Between two Blackhole Galaxies, 14336 B pages, PCIe x8 chip, 64-page ring, 56 MB per core per
iteration over 16 iterations:

| Sender cores | GB/s |
|---|---|
| 1 | 6.06 – 6.21 |
| 2 | 10.93 |
| 4 | 11.48 |
| 8 | 11.84 |

**Measured with a one-sided verbs transport that this branch no longer contains**, so treat it
as a characterisation of legs 1 and 3 plus the NIC rather than a number you can reproduce from
this tree. It establishes the useful fact: the device side and the 100 GbE link are not the
bottleneck. Core count is the lever at 14 KB — one Tensix core issuing chunked PCIe writes tops
out near 6.2 GB/s, two already clear the link, and eight reach ~97% of the ~12.24 GB/s
host-to-host ceiling measured independently on these NICs.

### What the MPI build delivers

| MPI configuration | Host-to-host | Status |
|---|---|---|
| `pml_ucx` / `mtl_ofi` over RDMA | reaches the link | measured outside this branch; **not reproduced here** |
| in-tree ULFM OpenMPI 5.0.7 (`self`/`sm`/`tcp` only) | ~2.2 GB/s, flat from 2 cores | measured here |

**The in-tree ULFM build cannot hit the 7-11 GB/s target.** It is configured with only the
`self`, `sm` and `tcp` BTLs — no UCX, no `openib`, no `ofi` — so every cross-host page goes
through the kernel TCP stack. A runtime probe confirms it lands on `ens5f0np0`, which is the
*same* 100 GbE port an RDMA path would use: same wire, ~18% of it. Adding sender cores does not
move the plateau, which is how you tell the limit is the host hop and not the device.

So this socket's throughput is a deployment property. Check it before blaming the socket:

```bash
ompi_info | grep -E 'MCA (pml|mtl): (ucx|ofi)'   # non-empty => RDMA-capable path available
```

The system OpenMPI 4.1.2 on these hosts *does* carry `pml: ucx`, `btl: openib` and `mtl: ofi`,
while the ULFM build tt-metal prefers does not — so reaching the target is a matter of which
MPI tt-metal links against, not of this code.

### Ring depth and the bandwidth-delay product

Measured at 14336 B on one core, varying `fifo_pages` only:

| `fifo_pages` | ring | GB/s |
|---|---|---|
| 2 | 28 KB | 2.16 |
| 4 | 57 KB | 4.03 |
| **8** | **115 KB** | **6.24** |
| 16 | 229 KB | 6.25 |
| 32 | 459 KB | 6.15 |
| 64 | 896 KB | 6.27 |
| 128 | 1.79 MB | 6.19 |

**The knee is at 8 pages, about 115 KB**, and throughput is flat above it. That puts the
bandwidth-delay product at ~115 KB, which back-solves to a credit round trip of
115 KB / 6.2 GB/s ≈ 18.5 µs — consistent with the independently measured 20.5 µs idle round
trip in the next section.

Two consequences. Past the knee the ring buys nothing but latency: 64 pages is ~8x BDP and
holds ~148 µs of data at a single core's rate, which is exactly the queueing seen below.
**16 pages is the better default** — it keeps full throughput with ~4x less buffering. And
single-core throughput past the knee is set by how fast one Tensix core can issue chunked PCIe
writes, not by the ring; that ceiling rises with page size (11.1 GB/s at 64 KB pages), which
is why core count is the lever at 14 KB specifically.

### Latency

Measured without any cross-host clock synchronisation: each figure is timed on a single
clock, and the one-way estimate is half the round trip, assuming the two directions are
symmetric. 14336 B pages, one core.

| | samples | min | p50 | avg | p99 | max |
|---|---|---|---|---|---|---|
| Device-to-device round trip (idle) | 200 | 20.26 | 20.53 | 20.98 | 28.14 | 59.75 |
| One-way estimate (RTT/2) | 200 | 10.13 | **10.27** | 10.49 | 14.07 | 29.87 |
| Streaming ack round trip (loaded) | 3881 | 43.36 | 132.65 | 136.05 | 140.02 | 13053 |
| Forward path under load (ack − RTT/2) | 3881 | 33.09 | **122.38** | 125.78 | 129.75 | 13043 |

All values in microseconds.

**An idle page crosses in about 10 µs; under a saturating stream the forward path is about
122 µs.** The ~112 µs difference is queueing, not distance, and it is the deep ring doing its
job: 64 pages of 14336 B is 917 KB, which at a single core's ~6.2 GB/s is ~148 µs of buffered
data, so an ack round trip of ~133 µs is what a nearly-full pipeline should cost. `fifo_pages`
is the throughput-versus-latency knob — shrink it for latency, grow it for bandwidth.

Two caveats on the loaded figures. The 13 ms maximum is a single straggler (p99 is 140 µs) and
looks like a scheduling hiccup on the relay thread, not socket behaviour. And the measurement
deliberately uses one long streaming pass: across several iterations the relay also samples
over a kernel relaunch, where the far device is not consuming at all, which produced a
meaningless 34 ms average in an earlier run.

## 5. Requirements

- **vIOMMU enabled.** Gated on `GetMemoryPinningParameters(mesh).can_map_to_noc`; the tests
  skip rather than fail when it is off.
- **An MPI build** (`ENABLE_DISTRIBUTED`). The socket is not built without one
  (`TT_METAL_ENABLE_HOST_TRANSPORT`, auto-detected), and `host_transport_available()` reports
  whether it can run here so both ends agree before the handshake rather than one blocking.
  There is no longer any libibverbs dependency.
- **For target throughput, an MPI with an RDMA-capable path** (`pml_ucx` or `mtl_ofi`). The
  socket is correct over plain TCP but ~5x slower; see §4.
- **PCIe x8 endpoints.** Only 4 of a Galaxy's 32 chips have an x8 link (one per tray, ASIC
  location 6); the other 28 are x1 and cannot carry this traffic.
- `fifo_size % page_size == 0`. A partial tail page would have to be charged to the
  flow-control counters on both sides, and the page size need not be a power of two (14336 is
  not), so the arithmetic is modular throughout — `tt::align()` is wrong here.
- KMD >= 2.7, or PCIe links train at Gen 1.

## 6. Running the tests

Always through SLURM; the launcher needs two nodes because the two endpoints are two ranks.

```bash
cd tests/tt_metal/multihost/host_socket
sbatch -p <galaxy-partition> --nodes=2 run_host_socket_tests.sh smoke   # correctness
sbatch -p <galaxy-partition> --nodes=2 run_host_socket_tests.sh perf    # 14 KiB throughput
sbatch -p <galaxy-partition> --nodes=2 run_host_socket_tests.sh sweep   # page size + core count
sbatch -p <galaxy-partition> --nodes=2 --time=2:00:00 \
    run_host_socket_tests.sh soak                                       # 1 h+ correctness soak
```

Throughput rows are appended to `results/host_socket_<jobid>.csv`. `TT_HOST_SOCKET_MIN_GBPS`
turns the throughput test into a gate; without it throughput is reported but not asserted.

Tests fall into two suites.

`HostTransportTest` needs **no Tenstorrent device** and runs in well under a second, so it is
the right place to reproduce a protocol bug: it streams pages between the two ranks and
verifies every byte, across ring laps, one page at a time, and under credit back-pressure.

```bash
sbatch -p <any-partition> --nodes=1 run_host_socket_tests.sh transport   # both, no device
```

`HostSocketLatencyTest` needs two ranks and real devices, and reports the tables in §4 and §7:
`RoundTrip` runs a one-page ping-pong against an echo kernel on the peer, timed on the
initiator's Tensix clock; `StreamingAckLatency` samples the relay's forward-to-credit time
under load and, given the round trip from the first test, also reports the forward path with
the return transit removed; `BurstToLastCredit` sends a burst of pages back to back and times
how long the credit for the *last* one takes to come back, both the local D2H credit (what a
sender kernel blocks on) and the receiver's end-to-end ack over a reverse socket. The
`latency` launcher mode runs the host-only `HostTransportTest.PingPongLatency` first (the
host-to-host hop on its own), then `RoundTrip` per page size, then `StreamingAckLatency` with
the idle round trip threaded in. The `burst` mode runs `BurstToLastCredit` once per page size
(`HOST_SOCKET_BURST_PAGE_SIZES`), each run sweeping `TT_HOST_SOCKET_BURSTS` on one kernel
launch; rows go to `results/burst_<jobid>.csv`.

Every socket can also stamp its relay hops (`HostMeshSocket::set_latency_sampling` /
`take_hop_samples`): on the sender host, transport send → source slot released (the NIC is done
reading it) and send → peer credit; on the receiver host, page published to the device → device
acked it. Each is on one host's clock, so they decompose the end-to-end figure without any
cross-host clock sync. The latency and burst tests report them as `*_hop_*` rows.

`HostSocketTest` needs two ranks and real devices: `SingleCoreCorrectness`,
`MultiCoreCorrectness` (8 core pairs), `PageSizeSweepCorrectness` (including non-power-of-two
sizes), `RingWrapCorrectness` (64 laps of the ring), `Throughput`, and `Soak`. Every
correctness test prefills the destination with the **complement** of the expected payload, so
a page that never arrives cannot pass.

One caution about these hosts: devices are held by processes outside SLURM, so a node that
looks free to `squeue` may not be. A contended chip blocks inside the kernel driver
(`Waiting for KMD lock ... held by another handle`), and a `CHIP_IN_USE_<n>_PCIe` lock left by
an older UMD build can be wedged permanently — the holder is dead but the shared mutex was
created without `PTHREAD_MUTEX_ROBUST`, and `/dev/shm` is sticky so another user cannot clear
it. Both present as device open hanging rather than failing, which is why the launcher tries
each candidate chip under a timeout.

The launcher handles three further things that are easy to get wrong on these hosts: it pins MPI away
from the docker and flannel interfaces (otherwise OpenMPI tries to reach a peer at 172.17.0.1
and aborts), it sets a distinct `TT_MESH_ID` per rank against a two-mesh graph descriptor
(the control plane requires a mesh binding once the world is larger than one rank), and it
tries each x8 chip in turn because another tenant's process can hold a chip's
`CHIP_IN_USE_<n>_PCIe` lock for its lifetime, which makes device open block rather than fail.

## 7. Exabox measurements, October 2026

Re-measured after rebasing onto main (branch `snijjar/d2h2h2d-rebased`), on two Blackhole
Galaxy hosts of the `bh_rev_c_sp4_A2A9` partition (`bh-glx-120-a07u08` + `bh-glx-120-a08u02`),
x8 chip 5 on each, through the in-tree ULFM OpenMPI 5.0.7 whose only inter-host BTL is `tcp`
(`ens5f0np0`, 100 GbE). Every number below is therefore a **TCP** number; the verbs figures in §4
are not reproduced here because that transport is not in this tree. SLURM jobs 138149 and
138158; CSVs under `tests/tt_metal/multihost/host_socket/results/`.

### Host-to-host alone (`HostTransportTest.PingPongLatency`)

One page out, wait for the peer's consumed credit back, no device on either end. Microseconds,
200 samples per row; two separate jobs shown because the TCP path is bimodal from run to run.

| page | job 138149 p50 (min / p99) | job 138158 p50 (min / p99) |
|---|---|---|
| 64 B | 201.8 (132.2 / 223.7) | 255.5 (191.0 / 327.4) |
| 1 KiB | 203.1 (135.9 / 271.9) | 323.6 (249.8 / 331.2) |
| 4 KiB | 200.2 (132.6 / 269.9) | 126.5 (59.0 / 334.4) |
| 14336 B | 155.5 (87.0 / 213.5) | 72.2 (68.2 / 79.6) |
| 64 KiB | 165.2 (143.7 / 232.0) | 295.5 (179.8 / 430.3) |

The smoke run of the same test in job 138149 (first thing after `MPI_Init`, before any device
was opened) sat at **69 µs p50 for every size up to 14 KiB** (65 – 87 µs min – p99) and 318 µs
at 64 KiB. So the host hop is ~70 µs when the TCP path is behaving and 200 – 320 µs when it is
not; the two modes are stable within a run and switch between runs, which points at interrupt
coalescing / CPU placement of the MPI progress loop (`--bind-to none`) rather than at the
socket. This hop is the floor under everything below.

### Device-to-device round trip, idle (`RoundTrip`)

Ping-pong through D2H → TCP → H2D and back, timed on the initiator's Tensix clock. One core,
16-page ring, `max_batch_pages = 1`. Microseconds, 200 samples.

| page | job 138149 p50 (min / p99) | job 138158 p50 (min / p99) | one-way estimate (p50/2) |
|---|---|---|---|
| 64 B | 213.0 (99.6 / 292.4) | 277.0 (262.7 / 291.8) | 107 – 139 |
| 1 KiB | 240.7 (105.3 / 322.4) | 303.4 (212.7 / 374.1) | 120 – 152 |
| 4 KiB | 189.6 (85.1 / 364.4) | 317.3 (227.6 / 380.5) | 95 – 159 |
| 14336 B | 318.5 (183.8 / 399.7) | 178.0 (123.6 / 244.1) | 89 – 159 |
| 64 KiB | 343.6 (308.6 / 447.7) | 719.4 (393.8 / 960.9) | 172 – 360 |

Subtracting the host-to-host row from the same job leaves **10 – 40 µs for both device legs
plus the two relays**, which is consistent with §4's 20.5 µs idle round trip over verbs: the
device side has not regressed; the transport is what moved.

### Per-hop stamps under a saturating stream (`StreamingAckLatency`, job 138158)

14336 B pages, one core, 64-page ring, warm kernel cache. Microseconds.

| hop (clock) | batches | min | p50 | avg | p99 |
|---|---|---|---|---|---|
| sender: send → source slot released | 1245 | 0.47 | 0.66 | 3.1 | 22.5 |
| sender: send → peer credit (ack round trip) | 1245 | 201 | 448 | 501 | 575 |
| receiver: published → device drained | 340 | 2.6 | 28.2 | 172 | 7336 |
| forward path under load (ack − idle RTT/2 = 89 µs) | 1229 | 112 | 358 | 359 | 470 |

Throughput during this run: **1.18 GB/s** (1.70 GB/s in the dedicated `Throughput` test, 16 ×
56 MB). The ring is 917 KB, so an ack round trip of ~450 µs at ~1.5 GB/s is the full-pipeline
queueing §4 describes, now ~3.4x longer because the pipe is ~4x narrower. `send → released`
at 0.66 µs says MPI hands the page to the kernel socket buffer immediately; the NIC is never
what the sender waits on. The receiver's 7 ms p99 drain is the kernel relaunch at the start of
the run (one long iteration; see the caveat in §4), not steady state.

(Job 138149's streaming row is omitted: it was the first launch of those kernels on a fresh
build, so the timed window included the JIT compile on both ranks and reported 0.11 GB/s.)

### Burst latency to the last credit (`BurstToLastCredit`, job 138149)

For each page size, `burst` pages are sent back to back (each one reserved, written, pushed and
notified on its own) and two times are taken on the initiator's clock: **local** = the local D2H
credit covers the last page (the relay released its slot: what a sender kernel actually blocks
on), **remote** = the receiver consumed the last page and its 64 B ack arrived over the reverse
socket. Ring = 64 pages (16 at 64 KiB: the receiver mirrors D2D and lands the ring in L1, so the
ring is capped at 1 MiB). 100 iterations per cell, `max_batch_pages = 8`. Microseconds (p50).

| page | burst 1 local / remote | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| 64 B | 38 / 121 | 41 / 114 | 49 / 141 | 51 / 180 | 47 / 149 | 84 / 219 | 189 / 335 |
| 256 B | 37 / 351 | 19 / 355 | 22 / 356 | 32 / 337 | 50 / 223 | 98 / 238 | 215 / 362 |
| 1 KiB | 38 / 284 | 29 / 287 | 26 / 284 | 40 / 180 | 59 / 197 | 158 / 300 | 284 / 448 |
| 4 KiB | 45 / 218 | 40 / 135 | 45 / 201 | 44 / 194 | 70 / 227 | 138 / 312 | 222 / 439 |
| 14336 B | 46 / 138 | 53 / 183 | 57 / 165 | 54 / 181 | 101 / 256 | 182 / 393 | 353 / 686 |
| 16 KiB | 40 / 347 | 24 / 397 | 37 / 401 | 55 / 353 | 97 / 295 | 211 / 406 | 408 / 693 |
| 64 KiB | 155 / 275 | 222 / 403 | 190 / 318 | 280 / 482 | 435 / 680 | — | — |

Reading it:

- **Local credit is flat at ~40 – 55 µs up to 8 – 16 pages** regardless of page size below
  64 KiB: that is the D2H leg plus the relay noticing the page and MPI accepting it, i.e. the
  sender is never back-pressured by the ring until the burst exceeds what the relay drains per
  poll. Past 16 pages it grows ~3 µs per page at 14 KiB (the chunked PCIe write issue rate of one
  core), and at 64 KiB pages the PCIe write itself dominates (155 µs for one page).
- **Remote credit is 120 – 350 µs for a single message** and is the TCP hop's bimodality again
  (compare 64 B at 121 µs with 256 B at 351 µs: same code path, different run). On top of that
  it grows ~5 µs per page at 14 KiB.
- **Amortised cost per message at burst 64** is 5 – 11 µs across all page sizes, and the implied
  bandwidth at 14336 B × 64 is 1.34 GB/s, i.e. the burst test reproduces the streaming number
  once the burst is a ring's worth. The knee — where remote time stops being flat and starts
  growing with the burst — sits at 8 – 16 pages at every size, which is the same ~115 – 230 KB
  bandwidth-delay product §4 found for the ring, now measured from the kernel's side.

### Throughput (`Throughput`, 14336 B, 64-page ring, 16 × 56 MB)

| cores | GB/s | job |
|---|---|---|
| 1 | 1.67 | 138149 |
| 1 | 1.70 | 138158 |
| 2 | 2.23 | 138160 |
| 4 | 1.81 | 138161 (different node pair: a08u08 + a09u02) |
| 8 | 2.63 | 138162 |

Same shape as §4's TCP result (~2.2 GB/s, flat from 2 cores): the host hop is the limit and the
target of 7 – 11 GB/s needs the RDMA-capable transport, not more sender cores.

### Receiver leg over PCIe DMA (`H2DMode::HOST_DMA`, jobs 138650 + 138834 vs 138496)

`DEVICE_PULL` has the receiver core NOC-read each page out of the pinned host ring. `HOST_DMA`
keeps the ring in L1 like D2D does, stages each page in the pinned ring at the same offset as
its L1 slot, and moves it with the chip's PCIe DMA engine (`TTDevice::dma_write_zero_copy`),
then writes the doorbell over MMIO. The device side is the `HOST_PUSH` kernel path. Measured in
isolation the DMA costs ~5 µs fixed per call plus ~15 GB/s, so it is the better push mode above
~2 KiB pages; `HOST_PUSH` (MMIO, ~0.3 GB/s) is only kept for comparison.

Two things to know before reading the numbers:

- **It costs L1.** The ring is an L1 buffer that the lockstep allocator charges to every bank,
  and `HostMeshSocket` allocates a second, ring-sized receiver data buffer on top of it, so a
  64 × 14336 B ring (917 KB) does not fit twice in a 1.43 MB bank. The tests clamp the ring to
  a 1.1 MB budget (`fit_fifo_to_l1`, `TT_HOST_SOCKET_L1_RING_BUDGET`): 37 – 38 pages at 14336 B,
  33 at 16 KiB, 8 at 64 KiB. That is why the burst table has no 64-page column for this leg and
  why the throughput row ran with a 37-page ring.
- **Not every chip's DMA engine reaches L1.** On `bh-glx-120-a08u08`, x8 chips 13
  (`0000:46:00.0`) and 29 (`0000:86:00.0`) never deliver PCIe-DMA writes to L1, while chips 5 and
  21 do and NOC-initiated reads of the same pinned ring work on all four; the IOVA value is not
  the discriminator (chip 5 works from an IOVA above 2^58, 13 and 29 fail from 0x3fbc0000).
  `H2DSocket` therefore DMAs a 256 B pattern at construction, reads it back, and falls back to
  MMIO for the payload with a warning when it did not land. `dmesg` is not readable on the nodes,
  so whether the engine's reads fault at the IOMMU is still open. Every number below is from
  chip 5 with the DMA engine in use (zero fallbacks in the logs).

Same hosts and chip as job 138496 (`DEVICE_PULL`), same binary apart from the mode. The
host-to-host rows have no device in them and still differ by up to 120 µs between the two jobs,
which is the TCP hop's run-to-run bimodality again: it is larger than anything the receiver leg
adds, so the idle rows cannot rank the two legs. Microseconds, p50 (min / p99).

| label | page | DEVICE_PULL (138496) | HOST_DMA (138650) |
|---|---|---|---|
| host-to-host put → credit | 64 B | 146.5 (78.2 / 223.1) | 268.3 (91.4 / 293.1) |
| host-to-host put → credit | 14336 B | 88.9 (82.9 / 96.6) | 88.0 (82.4 / 146.8) |
| host-to-host put → credit | 64 KiB | 198.9 (192.7 / 318.7) | 311.8 (192.4 / 445.8) |
| D2D round trip, idle | 64 B | 180.5 (71.6 / 347.2) | 366.4 (229.9 / 423.8) |
| D2D round trip, idle | 1 KiB | 165.7 (106.1 / 260.6) | 302.2 (287.5 / 322.3) |
| D2D round trip, idle | 4 KiB | 267.7 (251.5 / 341.9) | 378.5 (302.5 / 429.2) |
| D2D round trip, idle | 14336 B | 91.5 (80.5 / 352.2) | 165.7 (103.9 / 237.7) |
| D2D round trip, idle | 64 KiB | 673.3 (486.1 / 845.1) | 551.0 (521.8 / 675.4) |
| stream: send → peer credit | 14336 B | 503.4 (318.2 / 1063.7) | 464.0 (210.7 / 735.4) |
| stream: published → drained | 14336 B | 23.1 (2.5 / 50782.7) | 21.4 (1.4 / 227.9) |
| stream: forward path under load | 14336 B | 455.9 (272.4 / 1006.9) | 378.5 (127.8 / 617.5) |

Burst to the last credit, p50 µs, local / remote, same layout as the table above (ring = 64
pages for `DEVICE_PULL`, L1-clamped for `HOST_DMA`; bursts larger than the ring are dropped):

| page | leg | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|---|
| 64 B | pull | 16 / 296 | 18 / 297 | 22 / 298 | 26 / 323 | 47 / 276 | 78 / 292 | 174 / 344 |
| 64 B | dma | 33 / 136 | 36 / 201 | 29 / 176 | 33 / 265 | 50 / 253 | 91 / 221 | 210 / 347 |
| 1 KiB | pull | 27 / 487 | 20 / 493 | 27 / 496 | 39 / 394 | 56 / 370 | 95 / 378 | 229 / 417 |
| 1 KiB | dma | 35 / 205 | 30 / 332 | 23 / 381 | 38 / 337 | 56 / 240 | 115 / 274 | 246 / 396 |
| 4 KiB | pull | 38 / 294 | 33 / 307 | 30 / 206 | 47 / 183 | 71 / 226 | 145 / 307 | 304 / 486 |
| 4 KiB | dma | 13 / 356 | 20 / 238 | 27 / 243 | 43 / 261 | 65 / 274 | 118 / 368 | 315 / 476 |
| 14336 B | pull | 40 / 302 | 45 / 315 | 35 / 293 | 56 / 340 | 100 / 272 | 200 / 410 | 359 / 702 |
| 14336 B | dma | 14 / 105 | 27 / 143 | 37 / 198 | 58 / 199 | 105 / 281 | 182 / 427 | — |
| 16 KiB | pull | 21 / 287 | 37 / 219 | 39 / 223 | 57 / 219 | 99 / 326 | 219 / 438 | 377 / 763 |
| 16 KiB | dma | 19 / 310 | 25 / 176 | 42 / 183 | 60 / 222 | 118 / 287 | 220 / 440 | — |
| 64 KiB | pull | 174 / 280 | 227 / 393 | 207 / 354 | 289 / 483 | 526 / 808 | — | — |
| 64 KiB | dma | 204 / 427 | 210 / 398 | 216 / 391 | 310 / 513 | — | — | — |

The local column is the sender's own D2H leg and is the same for both modes, as it should be.
The remote column tracks the pull leg within the TCP noise at every size and burst: the DMA
engine's ~5 µs per commit is not visible against a 100 – 400 µs host hop, and at 64 KiB pages
`HOST_DMA` sits ~30 – 100 µs above `DEVICE_PULL` at small bursts, which is the one place the
synchronous DMA call (UMD busy-polls the engine; there is no submit/wait pair) shows up: the
relay thread is stuck in it instead of posting the next receive.

Throughput, 14336 B, one core, 16 × 56 MB:

| leg | ring | GB/s | job |
|---|---|---|---|
| DEVICE_PULL | 64 pages (917 KB) | 1.71 | 138496 |
| HOST_DMA | 37 pages (530 KB, L1 budget) | 1.10 | 138834 |
| DEVICE_PULL | 37 pages | 0.82, 1.17 (two runs) | 138837 |
| HOST_DMA | 37 pages | 1.09, 1.08 (two runs) | 138837 |

At the same ring depth the two legs are indistinguishable: a 530 KB ring over the ~460 µs ack
round trip measured under load is 1.15 GB/s of bandwidth-delay product, and both land there.
The 1.71 GB/s `DEVICE_PULL` figure is the deeper ring, not the faster leg. So on this transport
`HOST_DMA` buys nothing and costs L1 (which caps the ring, which caps the bandwidth), and
`DEVICE_PULL` stays the default. It becomes interesting when the host hop is RDMA (§4) and
the receiver core's NOC reads of host memory are what is left on the critical path.
