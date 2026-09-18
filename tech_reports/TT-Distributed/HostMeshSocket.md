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
host-to-host leg is new.

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

### No copy on the host

Both FIFOs are page-aligned `NamedShm` mappings, so each is registered once with `ibv_reg_mr`
and the NIC reads the D2H ring and writes the peer's H2D ring directly. The pinned ring *is*
the CPU-DRAM staging buffer; there is no second copy and no per-transfer registration.

### Ordering and pipelining

One RC queue pair per connection. RC delivery is in-order, which buys both properties at once:

- Payload work requests are posted back to back and left unsignaled (one signaled per 64) so
  the send queue can be reaped. `ibv_post_send` is a doorbell write, not a syscall, so nothing
  per-message enters the kernel.
- The batch's trailing doorbell is a 4-byte RDMA write of the sender's **absolute** running
  page total into a small registered slot on the receiver. Because it sits behind the payload
  on the same queue pair, the peer cannot observe it before those bytes have landed, so the
  fast path needs no read fence. The receiver reads the slot with an ordinary load.
- Payload is never striped across queue pairs; that would break ordering, and one queue pair
  already reaches the NIC's ceiling at these sizes.

**Immediate data is deliberately not used.** `RDMA_WRITE_WITH_IMM` looks like the natural fit
for a doorbell, but whether a completion reports `IBV_WC_RECV_RDMA_WITH_IMM` and sets
`IBV_WC_WITH_IMM` varies by provider — on the RoCE NICs in these hosts it does neither, so
`imm_data` is not meaningful. A dropped or bogus immediate is also very hard to diagnose,
because it presents as a device-side hang or as a page count that drifts. Writing an absolute
counter into memory instead matches how every other counter in the socket protocol works, and
it removes the receive queue entirely: no recv work requests to keep posted, no RNR risk.

Both signals are absolute counters for the same reason: a duplicated, coalesced or stale
update is then a no-op rather than silent corruption. The receiver additionally rejects a
doorbell implying more than a ring's worth outstanding, which is the signature of a misread
slot rather than real progress.

Both rings hold the same number of pages, so a page's source and destination indices are the
same value: the wrap is a single modulo and a batch is at most two contiguous runs, which lets
`max_batch_pages` pages coalesce into one work request.

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

The host credit is an **absolute page count**, so a duplicated or stale update costs a lap of
latency and never corrupts. Relay A retires a batch only once the NIC has finished reading it;
that retirement is what frees D2H ring space, which is what unblocks the sender kernel.

### Non-blocking relay

One process-wide thread runs a single non-blocking sweep over every registered connection, so
owning a socket imposes no polling duty. It drains fully before parking, then spins briefly
and backs off to a 50 µs sleep. Set `TransportConfig::own_relay_thread = false` and call
`poll()` to drive it inline instead, which makes a failure deterministic under a debugger.

An endpoint is **single-consumer**: it owns a queue pair whose completion queue must be
drained by exactly one thread. `poll()` therefore refuses to run on a socket the relay thread
already owns, and `barrier()` does not poll in that case — two threads in the same completion
queue double-consume completions and corrupt the doorbell and credit accounting. An exception
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

Between two Blackhole Galaxies (`bh-glx-120-c06u02` and `c06u08`), 14336 B pages, PCIe x8
chip, 64-page ring, 56 MB per core per iteration over 16 iterations:

| Sender cores | GB/s |
|---|---|
| 1 | 6.06 – 6.21 |
| 2 | 10.93 |
| 4 | 11.48 |
| 8 | 11.84 |

**Core count is the lever that matters at 14 KB.** A single Tensix core issuing chunked PCIe
writes tops out near 6.2 GB/s at this page size and cannot saturate the path; two cores
already clear the link, and eight reach about 97% of the ~12.24 GB/s ceiling measured
independently for host-to-host RDMA on these NICs.

Throughput here is repeatable only to about +/-15%: the same 14336/64-page/1-core point
measured 5.25, 6.27 and 6.71 GB/s within one job and 3.15 GB/s on a different node pair. Treat
single points accordingly, and prefer comparing shapes (core scaling, ring depth) over
absolute values.

The `sweep` mode also varies page size, holding total volume fixed so the points are
comparable. Even so, a kernel launch costs tens of milliseconds on these systems, so a
small-page point still carries proportionally more launch overhead than a large-page one;
read the curve as a lower bound at the small end.

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
- **A RoCE device reachable from both hosts.** Built only when libibverbs is present
  (`TT_METAL_ENABLE_HOST_TRANSPORT`, auto-detected).
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

`HostTransportTest` needs **no Tenstorrent device** — it cross-connects two queue pairs on one
NIC port (the HCA loops back internally) and checks the transport on its own: payload/doorbell
ordering, 500 ring laps, absolute credits, counter widening across a 32-bit wrap, and
send-queue back-pressure. It runs anywhere with a RoCE port, in well under a second, and is
the right place to reproduce a protocol bug:

```bash
./build_Release/test/tt_metal/multi_host_socket_transport_tests --gtest_filter='HostTransportTest.*'
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
