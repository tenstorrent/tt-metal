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

Legs 1 and 3 are `D2HSocket` and `H2DSocket` (in `DEVICE_PULL` mode) unchanged. Only the
host-to-host leg is new. It sits behind the `HostTransport` interface
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

`HostSocketLatencyTest` needs two ranks and real devices, and reports the table in S.4:
`RoundTrip` runs a one-page ping-pong against an echo kernel on the peer, timed on the
initiator's Tensix clock; `StreamingAckLatency` samples the relay's forward-to-credit time
under load and, given the round trip from the first test, also reports the forward path with
the return transit removed. The `latency` launcher mode runs them in that order and threads
the first result into the second.

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
