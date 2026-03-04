# H2D / D2H PCIe Socket  -  Technical Report

> **Hardware scope: Blackhole Galaxy  -  Rev A and Rev B only.**
> The PCIe link topology, chip count, bandwidth figures, and ASIC location assignments described in this document have been validated on Blackhole Galaxy Rev A and Rev B systems. Results may differ on later revisions or other Blackhole form factors.

---

## Table of Contents

1. [Background  -  What Is a Socket?](#1-background--what-is-a-socket)
   - 1.1 System Topology and PCIe Link Asymmetry
   - 1.2 Blackhole Hardware Facts Relevant to This Guide
   - 1.3 Training & Inference Use Cases for PCIe Sockets
2. [Transfer Modes and Flow Control](#2-transfer-modes-and-flow-control)
   - 2.1 H2D  -  HOST\_PUSH
   - 2.2 H2D  -  DEVICE\_PULL
   - 2.3 D2H
   - 2.4 Flow control (shared)
3. [API Walkthrough](#3-api-walkthrough)
   - 3.1 Host Side  -  Setup
   - 3.2 Device Side  -  Initialization
   - 3.3 D2H Kernel: pcie\_socket\_sender.cpp
   - 3.4 H2D HOST\_PUSH Kernel: h2d\_throughput\_host\_push.cpp
   - 3.5 H2D DEVICE\_PULL Kernel: h2d\_throughput\_device\_pull.cpp
4. [Performance Results](#4-performance-results)
   - 4.1 D2H Throughput
   - 4.2 D2H Latency
   - 4.3 H2D Throughput
   - 4.4 H2D Latency
   - 4.5 Ping / Jitter
   - 4.6 Multi-Chip Throughput
5. [Interpreting Results](#5-interpreting-results)
6. [Running the Benchmarks](#6-running-the-benchmarks)
7. [Benchmark Suite Reference](#7-benchmark-suite-reference)

---

## 1. Background  -  What Is a Socket?

A **socket** is a streaming FIFO abstraction for moving data across the PCIe link between a host CPU and a Tenstorrent AI core. It hides the low-level details of PCIe TLB mapping, NOC address encoding, and flow-control signaling behind a simple `write` / `read` + `barrier` API.

> **Naming note  -  "socket" here does not mean a POSIX/network socket.**
> The term is used in the Tenstorrent-specific sense of a hardware-level streaming channel between host and device over PCIe. It has nothing to do with TCP/UDP sockets, Berkeley sockets, or any OS networking API.
>
> There are also two distinct flavours of Tenstorrent socket to be aware of:
>
> - **H2D/D2H PCIe sockets** (this document)  -  move data between the host CPU and a device core over PCIe. Implemented in `tt_metal/api/tt-metalium/experimental/sockets/`.
> - **TT-Fabric / Ethernet sockets**  -  move tensor data between AI cores across chips via Ethernet links. Documented in [`tech_reports/TT-Fabric/TT-Fabric-Architecture.md`](../../../tech_reports/TT-Fabric/TT-Fabric-Architecture.md) and [`tech_reports/Programming_Multiple_Meshes/`](../../../tech_reports/Programming_Multiple_Meshes/Programming_Multiple_Meshes.md).
>
> These are completely independent mechanisms. This document covers only the PCIe variant.

Two socket types are relevant here:

| Type | Direction | Host-side API | Device-side API |
|------|-----------|---------------|-----------------|
| `H2DSocket` | Host -> Device | `write()`, `barrier()` | `SocketReceiverInterface` |
| `D2HSocket` | Device -> Host | `read()`, `barrier()` | `SocketSenderInterface` |

Both types use a **circular FIFO** backed by **pinned host memory**  -  memory locked in physical RAM and mapped through the vIOMMU so the device can address it directly via PCIe. This is a hard system requirement: **vIOMMU must be enabled** for socket transfers to work.

The FIFO is parameterised by two quantities that directly control performance:

- **Page size**  -  the unit of transfer, in bytes. On Blackhole, the NOC word width is 512 bits (64 B), which sets the natural PCIe read alignment. The minimum page size is therefore **64 B**, which is why the benchmark sweep starts there. Larger pages amortise per-page overhead but require bigger L1 allocations on the device.
- **FIFO (socket buffer) size**  -  the total capacity of the ring buffer, in bytes. A larger FIFO allows the sender to get further ahead of the receiver before back-pressure kicks in, which is critical for high throughput over high-latency PCIe.

---

### 1.1 System Topology and PCIe Link Asymmetry

Not all chips on a Tenstorrent tray are equal from a host-connectivity standpoint. In a 32-chip Blackhole Galaxy system, **all 32 chips are directly MMIO-mapped**  -  each has its own physical PCIe connection to the host root complex. However, chips fall into two classes based on the **bandwidth** of that connection:

| Class | Count | PCIe Generation & Width | Measured D2H Peak |
|-------|-------|------------------------|-------------------|
| High-bandwidth (ASIC 6 per tray) | 4 | **Gen 4 x 8** | **~15.1 GB/s** |
| Low-bandwidth (all others) | 28 | **Gen 4 x 1** (~2 GB/s peak) | see S.4.6  -  requires KMD >= 2.7 |

> **KMD >= 2.7 required for correct PCIe link speeds on low-bandwidth chips.**
> Linux kernels 6.5 through 6.12 contain a quirk that can force all PCIe links to **Gen 1 (2.5 GT/s, ~250 MB/s)** during hot-plug enumeration on Blackhole Galaxy systems. KMD 2.7 detects this condition and retrains each link to its full speed. Linux 6.13+ does not have this quirk. Benchmark results for the 28 low-bandwidth chips in this report are only valid when running KMD >= 2.7; numbers collected on earlier KMD versions reflect Gen 1 performance (~250 MB/s) rather than the true Gen 4 x1 ceiling (~2 GB/s). See [ttkmd-2.7.0 release notes](https://github.com/tenstorrent/tt-kmd/releases/tag/ttkmd-2.7.0) for details.

The **4 high-bandwidth chips** (one per tray, ASIC Location 6) have a full Gen 4 x8 link to the host PCIe root complex. Data written by the device kernel over NOC reaches host RAM in a single PCIe hop at full link bandwidth.

The **28 low-bandwidth chips** each have their own direct (not tunnelled) PCIe link to the host  -  there is no chip-to-chip relay. Their peak host-facing bandwidth is significantly lower than the high-bandwidth chips regardless of page size, FIFO size, or transfer mode (see S.4.6 for measured values).

> **This asymmetry is the single most important architectural fact for training job placement.**
> Any workload that requires high-bandwidth streaming to or from the host  -  gradient checkpointing, activation offloading, weight streaming  -  must target one of the 4 high-bandwidth chips (ASIC 6 per tray). The 28 low-bandwidth chips are significantly bandwidth-constrained for host-facing socket I/O regardless of tuning (see S.4.6 for measured values).

Due to this asymmetry, the benchmarks in this report show two distinct performance regimes: high-bandwidth chips saturate the PCIe link with low latency, while low-bandwidth chips hit a hard throughput ceiling orders of magnitude lower and incur significantly higher round-trip latency, regardless of page size or FIFO tuning.

### Related documentation in this repo

| Document | Relevance |
|----------|-----------|
| [`tech_reports/Blackhole/BlackholeBringUpProgrammingGuide.md`](../../../tech_reports/Blackhole/BlackholeBringUpProgrammingGuide.md) | Blackhole chip specs: Tensix grid (13x10 compute), L1 (1464 KB + data cache), DRAM (~4 GB x 8 banks), NOC alignment constraints. |
| [`tech_reports/EthernetMultichip/BasicEthernetGuide.md`](../../../tech_reports/EthernetMultichip/BasicEthernetGuide.md) | Multi-chip topology and MMIO concepts (Wormhole-era). Note: in Wormhole only a subset of chips were MMIO-mapped; in Blackhole **all 32 chips** have direct PCIe connections. The relevant Blackhole asymmetry is PCIe link bandwidth (high-bandwidth vs low-bandwidth chips), not MMIO vs non-MMIO. |
| [`tech_reports/TT-Fabric/TT-Fabric-Architecture.md`](../../../tech_reports/TT-Fabric/TT-Fabric-Architecture.md) | TT-Fabric Ethernet sockets (chip-to-chip via Ethernet, **not** PCIe). Do not confuse with H2D/D2H PCIe sockets. |
| [`tech_reports/Programming_Multiple_Meshes/Programming_Multiple_Meshes.md`](../../../tech_reports/Programming_Multiple_Meshes/Programming_Multiple_Meshes.md) | Multi-mesh pipeline parallelism using Ethernet sockets (not PCIe sockets). Relevant context for where H2D/D2H fits in a broader multi-mesh system: H2D/D2H handles host<->device ingestion/egress, while inter-mesh data flow uses Ethernet sockets. |
| [`tech_reports/memory/allocator.md`](../../../tech_reports/memory/allocator.md) | L1 and DRAM allocation, alignment constraints. |

---

### 1.2 Blackhole Hardware Facts Relevant to This Guide

| Parameter | Value |
|-----------|-------|
| AI clock | **1.35 GHz** (used throughout for cycles -> us conversion) |
| L1 per core | **1464 KB** (governs maximum socket page size and buffer allocation) |
| PCIe link  -  high-bandwidth chips (ASIC 6 per tray, 4 total) | **Gen 4 x8 -> ~16 GB/s** unidirectional theoretical; ~15.1 GB/s measured D2H |
| PCIe link  -  low-bandwidth chips (all others, 28 total) | **Gen 4 x1 -> ~2 GB/s** unidirectional (requires KMD >= 2.7  -  see S.1.1 note) |

Sources: L1, DRAM, and NOC alignment from [`BlackholeBringUpProgrammingGuide.md`](../../../tech_reports/Blackhole/BlackholeBringUpProgrammingGuide.md); AI clock (1.35 GHz) from [`GEMM_FLOPS.md`](../../../tech_reports/GEMM_FLOPS/GEMM_FLOPS.md); PCIe link bandwidth from benchmark measurements in this report.

---

### 1.3 Training & Inference Use Cases for PCIe Sockets

| Scenario | Direction | Driver | Bottleneck |
|----------|-----------|--------|------------|
| **Weight loading**  -  loading tokenised batches from host dataloader into device L1/DRAM before each step | **H2D** | Host-side data pipeline writes to socket | PCIe Gen 4 x8 ceiling (~16 GB/s on high-bandwidth chips). Choose `HOST_PUSH` for lowest latency, `DEVICE_PULL` to offload CPU. |
| **Activation / gradient offloading**  -  streaming activations or gradients out to host RAM to free device DRAM (e.g., during FSDP or gradient checkpointing) | **D2H** | Device kernel writes to host pinned buffer | PCIe Gen 4 x8 (~16 GB/s, high-bandwidth chips only). Low-bandwidth chips have significantly lower host-facing bandwidth (see S.4.6)  -  **do not use low-bandwidth chips for high-throughput offloading**. |
| **Loss / logit collection**  -  reading per-step loss scalars or logit tensors from the device for host-side logging or early stopping | **D2H** | Device kernel writes small result tensors | Latency-bound (small pages). Use large page sizes even for small payloads (pad to 4 KB+) to amortise protocol overhead. |
| **Pipeline stage I/O**  -  feeding the **first stage** of a pipeline-parallel model from a CPU-resident dataset server | **H2D** | Dataset server writes to socket into Stage 0 device | Similar to data ingestion. The socket forms the CPU->device boundary of the pipeline. Downstream stage-to-stage communication should use **Ethernet sockets** (not PCIe sockets). |
| **Telemetry / profiling streams**  -  continuously streaming device-side cycle counter data or custom metrics to a host monitoring process | **D2H** | Device writer kernel sends fixed-size telemetry records | Very latency-sensitive. Use small, fixed page sizes and a large FIFO. Run on a high-bandwidth chip (ASIC 6). |

> **Rule of thumb:** Any scenario that requires moving more than a few MB per second between host and device must land on one of the **4 high-bandwidth chips** (ASIC 6 per tray, Gen 4 x8 PCIe). All 28 low-bandwidth chips are significantly bandwidth-constrained for host-facing socket I/O (see S.4.6), which is insufficient for streaming training data or large activation offloads at training speed.

---

## 2. Transfer Modes and Flow Control

### 2.1 H2D  -  HOST\_PUSH

The host writes data directly into the device's L1 FIFO through a TLB-mapped PCIe posted write. The device kernel polls for new pages and acknowledges consumption by writing `bytes_acked` back to host-pinned memory.

```text
Host CPU  --[TLB PCIe write]--->  Device L1 FIFO  --->  Device kernel
           (posted, no completion required)
```

### 2.2 H2D  -  DEVICE\_PULL

The host writes data to a pinned host buffer and updates `bytes_sent`. The device kernel detects the notification, then issues a **NOC read** (non-posted; requires PCIe completion TLPs) to pull the data from host RAM into its own L1.

```text
Host CPU  --[writes to pinned RAM, updates bytes_sent]--->  device detects
Device kernel  --[NOC read <--- PCIe <--- pinned host RAM]--->  data in L1
```

This mode frees the host CPU from driving the bulk DMA, at the cost of PCIe read completion overhead which caps throughput below the posted-write ceiling of HOST\_PUSH and D2H.

### 2.3 D2H

The device is always the initiator. It waits for FIFO space, issues chunked NOC writes into the pinned host FIFO, then notifies the host by writing `bytes_sent` to host-pinned memory. The host polls `bytes_sent`, copies data out, and writes `bytes_acked` back to release FIFO space.

```text
Device kernel  --[NOC write ---> PCIe ---> pinned host FIFO]--->  Host CPU reads
```

### 2.4 Flow control (shared)

All three modes share the same **`bytes_sent` / `bytes_acked`** credit protocol. Both fields live in host-pinned memory and are accessible to both sides via PCIe.

```text
Sender                                          Receiver
----------------------------------------------------------------
spin-wait: fifo_size - (bytes_sent - bytes_acked) >= page_size
write data into FIFO
bytes_sent += page_size
notify_receiver() ------------------------------>  detects new data, consumes page
                                                   bytes_acked += page_size
<----------------------------- notify_sender()      written to host-pinned memory
```

The sender blocks when the FIFO is full (`bytes_sent - bytes_acked == fifo_size`); the receiver spins when it is empty. Both counters are written via NOC + PCIe and require `invalidate_l1_cache()` before reading on the device side.

See **S.3** for fully annotated kernel code showing exactly how each call maps to the protocol.

---

## 3. API Walkthrough

This section walks through the benchmark kernels and their host-side counterparts, annotating each API call. Code is drawn from `tests/tt_metal/distributed/test_hd_sockets.cpp` and `tests/tt_metal/tt_metal/test_kernels/misc/socket/`.

---

### 3.1 Host Side  -  Setup

The host side is identical for all socket directions and modes. The workflow is:

1. **Construct the socket**  -  allocates pinned host memory for the FIFO and the flow-control fields, creates the L1 config buffer on the device, and (for HOST\_PUSH) opens a TLB window into the device L1 FIFO region.
2. **Set the page size**  -  aligns the internal write/read pointer to the new page granularity.
3. **Pass the config buffer address to the device kernel**  -  the device needs to know where in L1 to find the socket metadata.
4. **Dispatch the device kernel**  -  enqueue the workload before (or concurrently with) the host's write/read loop.
5. **Run the host transfer loop**  -  call `write()` or `read()` once per page; calls block when the FIFO is full/empty.
6. **Call `barrier()`**  -  wait for all data to be acknowledged by the other side.

```cpp
// -- test_hd_sockets.cpp: test_h2d_socket() ---------------------------------

// 1. Construct the socket.
//    Allocates the circular FIFO in pinned host memory and the L1 config buffer.
auto input_socket = H2DSocket(mesh_device, recv_core, BufferType::L1,
                              socket_fifo_size, h2d_mode);

// 2. Set page size. Must be called before write(). Aligns write_ptr to the
//    new page boundary inside the FIFO.
input_socket.set_page_size(page_size);

// 3. Allocate the device-side destination L1 buffer (separate from the FIFO).
auto recv_data_buffer = MeshBuffer::create(buffer_config,
                                           recv_device_local_config,
                                           mesh_device.get());

// 4. Build the device kernel, passing socket and buffer addresses as
//    compile-time args so they are resolved at kernel compile time.
CreateKernel(
    recv_program, "...h2d_throughput_device_pull.cpp", recv_core.core_coord,
    DataMovementConfig{
        .compile_args = {
            input_socket.get_config_buffer_address(),  // where to find socket metadata in L1
            recv_data_buffer->address(),               // destination L1 buffer address
            page_size,
            data_size,
            measurement_buffer->address(),
            num_iterations,
        }});

// 5. Dispatch the kernel. The workload is enqueued non-blocking; the kernel
//    begins running once the dispatch queue reaches it and will poll bytes_sent
//    waiting for the host to produce data.
EnqueueMeshWorkload(mesh_device->mesh_command_queue(), mesh_workload, false);

// 6. Host transfer loop: write one page at a time.
//    write() blocks if the FIFO is full (FIFO size - (bytes_sent - bytes_acked) < page_size).
//    After writing, it updates bytes_sent in pinned memory and notifies the device.
for (uint32_t j = 0; j < num_writes; j++) {
    input_socket.write(src_vec.data() + j * page_size_words, 1);
}

// 7. Barrier: spin until bytes_acked == bytes_sent (device has consumed everything).
input_socket.barrier();
```

For **D2H** the pattern is symmetric  -  construct `D2HSocket`, dispatch the sender kernel, then call `output_socket.read()` in a loop followed by `output_socket.barrier()`.

---

### 3.2 Device Side  -  Initialization

Every device kernel begins with the same two calls, regardless of direction or mode:

```cpp
// From any socket kernel (e.g. h2d_throughput_device_pull.cpp)

// Reads the socket metadata struct from the L1 config buffer address and
// caches the fields (read_ptr, bytes_acked, fifo_addr, PCIe address info, ...)
// onto the kernel stack. All subsequent socket calls operate on this cached
// struct; L1 is only written back at the end via update_socket_config().
SocketReceiverInterface receiver_socket =
    create_receiver_socket_interface(socket_config_addr);

// Aligns read_ptr to the new page boundary. Must match the host-side
// set_page_size() call or the FIFO pointers will diverge.
set_receiver_socket_page_size(receiver_socket, page_size);
```

For D2H sender kernels the equivalent calls are:

```cpp
SocketSenderInterface sender_socket = create_sender_socket_interface(socket_config_addr);
set_sender_socket_page_size(sender_socket, page_size);
```

---

### 3.3 D2H Kernel: `pcie_socket_sender.cpp`

Full kernel, annotated:

```cpp
// pcie_socket_sender.cpp

SocketSenderInterface sender_socket = create_sender_socket_interface(socket_config_addr);
uint32_t pcie_xy_enc  = sender_socket.d2h.pcie_xy_enc;   // NOC encoding of the PCIe endpoint
uint32_t data_addr_hi = sender_socket.d2h.data_addr_hi;  // high 32 bits of host FIFO PCIe address
set_sender_socket_page_size(sender_socket, page_size);

uint64_t start_timestamp = get_timestamp();

for (uint32_t i = 0; i < num_iterations; i++) {
    uint32_t outstanding_data_size = data_size;
    while (outstanding_data_size) {

        // Spin until the host has freed at least one page of FIFO space.
        // Reads bytes_acked from host-pinned memory via L1 cache invalidation.
        socket_reserve_pages(sender_socket, 1);

        // Compute the destination PCIe address for this page.
        // write_ptr is a relative offset; add the FIFO base and high address word
        // to form the full 64-bit host RAM address.
        uint64_t pcie_data_addr =
            ((uint64_t)data_addr_hi << 32 | sender_socket.downstream_fifo_addr)
            + sender_socket.write_ptr;

        // Issue chunked NOC writes into host RAM (PCIe posted write).
        // Each chunk is at most NOC_MAX_BURST_SIZE bytes.
        uint32_t page_bytes_remaining = page_size;
        while (page_bytes_remaining) {
            uint32_t chunk = std::min(page_bytes_remaining, max_noc_burst_bytes);
            noc_wwrite_with_state<...>(noc_index, src_addr, pcie_xy_enc,
                                      pcie_data_addr, chunk, 1);
            src_addr      += chunk;
            pcie_data_addr += chunk;
            page_bytes_remaining -= chunk;
        }

        // Advance write_ptr and increment bytes_sent in the local struct.
        socket_push_pages(sender_socket, 1);

        // NOC write of bytes_sent to host-pinned memory so the host's read()
        // call can detect that new data has arrived.
        socket_notify_receiver(sender_socket);

        outstanding_data_size -= page_size;
    }
}

// Spin until all bytes_acked == bytes_sent (host has consumed and freed all pages).
socket_barrier(sender_socket);
noc_async_write_barrier();

uint64_t end_timestamp = get_timestamp();
// Store elapsed cycles for the host to read back.
*reinterpret_cast<volatile uint64_t*>(measurement_buffer_addr)
    = end_timestamp - start_timestamp;

// Write the updated write_ptr and bytes_sent back to the L1 config buffer
// so the state is preserved across kernel invocations.
update_socket_config(sender_socket);
```

---

### 3.4 H2D HOST\_PUSH Kernel: `h2d_throughput_host_push.cpp`

The host has already placed data in the L1 FIFO via TLB write  -  the device kernel only needs to drain it.

```cpp
// h2d_throughput_host_push.cpp

SocketReceiverInterface receiver_socket =
    create_receiver_socket_interface(socket_config_addr);
set_receiver_socket_page_size(receiver_socket, page_size);

uint64_t start_timestamp = get_timestamp();

for (uint32_t i = 0; i < num_iterations; i++) {
    uint32_t outstanding_data_size = data_size;
    uint64_t dst_noc_addr = get_noc_addr(local_l1_buffer_addr);
    while (outstanding_data_size) {

        // Spin until bytes_sent - bytes_acked >= page_size.
        // bytes_sent is updated by the host's write() call via a TLB write to
        // host-pinned memory; the device reads it via cache invalidation.
        socket_wait_for_pages(receiver_socket, 1);

        // The page is already sitting in the L1 FIFO at receiver_socket.read_ptr.
        // Copy it from the FIFO to the destination L1 buffer via a local NOC write.
        noc_async_write(receiver_socket.read_ptr, dst_noc_addr, page_size);
        dst_noc_addr += page_size;
        outstanding_data_size -= page_size;

        noc_async_write_barrier();  // wait for the local L1->L1 copy to finish

        // Advance read_ptr and increment bytes_acked locally.
        socket_pop_pages(receiver_socket, 1);

        // NOC write of bytes_acked to host-pinned memory. This unblocks the
        // host's write() call when the FIFO was full.
        socket_notify_sender(receiver_socket);
    }
}

uint64_t end_timestamp = get_timestamp();
*reinterpret_cast<volatile uint64_t*>(measurement_buffer_addr)
    = end_timestamp - start_timestamp;

update_socket_config(receiver_socket);
noc_async_write_barrier();
```

---

### 3.5 H2D DEVICE\_PULL Kernel: `h2d_throughput_device_pull.cpp`

Unlike HOST\_PUSH, the device must explicitly fetch data from host RAM over PCIe  -  the NOC read and its completion barrier are the dominant cost.

```cpp
// h2d_throughput_device_pull.cpp

SocketReceiverInterface receiver_socket =
    create_receiver_socket_interface(socket_config_addr);
set_receiver_socket_page_size(receiver_socket, page_size);

// Compute the base PCIe address of the host-side pinned data buffer.
// data_addr_hi/lo are stored in the socket config and set by the host at
// socket construction time.
uint64_t pcie_data_addr =
    ((uint64_t)receiver_socket.h2d.data_addr_hi << 32)
    | receiver_socket.h2d.data_addr_lo;
uint32_t pcie_xy_enc = receiver_socket.h2d.pcie_xy_enc;

uint64_t start_timestamp = get_timestamp();

for (uint32_t i = 0; i < num_iterations; i++) {
    uint32_t outstanding_data_size = data_size;
    uint64_t dst_noc_addr = get_noc_addr(local_l1_buffer_addr);
    while (outstanding_data_size) {

        // Spin until the host has written at least one page to pinned memory
        // and updated bytes_sent.
        socket_wait_for_pages(receiver_socket, 1);

        // Compute the source PCIe address for this page: base + offset from FIFO start.
        uint64_t page_src_addr =
            pcie_data_addr + receiver_socket.read_ptr - receiver_socket.fifo_addr;
        uint32_t page_dst_addr = receiver_socket.read_ptr;  // write into the L1 FIFO slot

        // Issue chunked NOC reads from host pinned RAM into device L1 (PCIe non-posted reads).
        // Completions must come back before the barrier below.
        uint32_t page_bytes_remaining = page_size;
        while (page_bytes_remaining) {
            uint32_t chunk = std::min(page_bytes_remaining, max_noc_burst_bytes);
            noc_read_with_state<...>(NOC_INDEX, pcie_xy_enc,
                                    page_src_addr, page_dst_addr, chunk);
            page_src_addr += chunk;
            page_dst_addr += chunk;
            page_bytes_remaining -= chunk;
        }
        // Wait for all PCIe read completions to arrive and fill L1.
        noc_async_read_barrier();

        // Copy from the L1 FIFO slot to the destination L1 buffer.
        noc_async_write(receiver_socket.read_ptr, dst_noc_addr, page_size);
        dst_noc_addr += page_size;
        outstanding_data_size -= page_size;

        noc_async_write_barrier();

        // Advance read_ptr and bytes_acked, then notify the host so it can
        // reclaim FIFO space for the next write().
        socket_pop_pages(receiver_socket, 1);
        socket_notify_sender(receiver_socket);
    }
}

uint64_t end_timestamp = get_timestamp();
*reinterpret_cast<volatile uint64_t*>(measurement_buffer_addr)
    = end_timestamp - start_timestamp;

update_socket_config(receiver_socket);
noc_async_write_barrier();
```

---

## 4. Performance Results

> Charts are generated by `analyze_hd_sockets.py`  -  see [S.6 Running the Benchmarks](#6-running-the-benchmarks).

---

### 4.1 D2H Throughput

**`d2h_throughput.png`**  -  Throughput (GB/s) vs page size at the maximum FIFO size. Each line is one total-transfer-data size (16 KB -> 1 GB). Shows how throughput saturates as pages get larger and as more data is moved.

![D2H Throughput vs Page Size](charts/d2h_throughput.png)

---

**`d2h_tp_vs_fifo.png`**  -  Throughput vs socket FIFO size at the maximum total-data size. Each line is one page size. The key chart for choosing a FIFO size: throughput climbs steeply with FIFO size then plateaus once back-pressure disappears.

![D2H Throughput vs FIFO Size](charts/d2h_tp_vs_fifo.png)

---

### 4.2 D2H Latency

**`d2h_latency.png`**  -  Round-trip latency (us) vs page size, one line per FIFO size. p50 is shown as a solid line; min/max as dashed. Log-log scale makes both the protocol-overhead floor (small pages) and the DMA-time slope (large pages) visible.

![D2H Round-Trip Latency vs Page Size](charts/d2h_latency.png)

---

### 4.3 H2D Throughput

**`h2d_throughput.png`**  -  H2D throughput vs page size for both `HOST_PUSH` and `DEVICE_PULL`, at the maximum FIFO and maximum total-data size. The primary chart for comparing the two transfer modes head-to-head on throughput.

![H2D Throughput vs Page Size  -  HOST_PUSH vs DEVICE_PULL](charts/h2d_throughput.png)

---

**`h2d_tp_vs_fifo.png`**  -  H2D throughput vs FIFO size, `HOST_PUSH` (left) and `DEVICE_PULL` (right) in separate panels. Each line is one page size. Shows whether the two modes need the same FIFO depth to reach their respective plateaus.

![H2D Throughput vs FIFO Size  -  HOST_PUSH vs DEVICE_PULL](charts/h2d_tp_vs_fifo.png)

---

### 4.4 H2D Latency

**`h2d_latency.png`**  -  H2D round-trip latency vs page size, `HOST_PUSH` vs `DEVICE_PULL` overlaid on one axis (p50 solid, min/max dashed). The single most useful chart for mode selection: shows which mode has lower latency and how the gap evolves with page size.

![H2D Round-Trip Latency  -  HOST_PUSH vs DEVICE_PULL](charts/h2d_latency.png)

---

### 4.5 Ping / Jitter

**`d2h_ping_timeseries.png`**  -  D2H pure-signalling latency plotted iteration-by-iteration (no data DMA). Exposes tail-latency spikes from OS scheduler interference, PCIe power-state transitions, or NUMA effects. p50 and mean reference lines are overlaid.

![D2H Pure Ping: Per-Iteration Latency](charts/d2h_ping_timeseries.png)

---

**`h2d_ping_timeseries.png`**  -  H2D pure-signalling latency per iteration, `HOST_PUSH` and `DEVICE_PULL` overlaid. Directly compares the flow-control protocol overhead between the two modes with no DMA noise.

![H2D Pure Ping: Per-Iteration Latency  -  HOST_PUSH vs DEVICE_PULL](charts/h2d_ping_timeseries.png)

---

### 4.6 Multi-Chip Throughput

> **Charts pending re-run with KMD >= 2.7.** Initial measurements for the 28 low-bandwidth chips were collected while those links were running at **Gen 1 (2.5 GT/s)** due to a Linux kernel 6.5-6.12 quirk (see S.1.1). KMD 2.7 retrains all links to Gen 4 speed; results will be updated once benchmarks are re-run on a KMD 2.7 system.

**`mc_d2h_throughput_heatmap.png`**  -  Heatmap of D2H peak throughput (GB/s) across every chip on the system. Rows = chips (identified by Tray ID / ASIC Location), columns = FIFO sizes (1 MB -> 256 MB), fixed at 64 KB pages and 1 GB total transfer. Reveals per-chip performance variation across the tray.

<!-- FIXME: shelved  -  re-run after PCIe link config is validated
![D2H Multi-Chip Throughput Heatmap  -  All Chips x FIFO Size](charts/mc_d2h_throughput_heatmap.png)
-->

---

**`mc_d2h_throughput_vs_fifo.png`**  -  Line chart version of the multi-chip sweep: throughput vs FIFO size, one line per chip. Makes it easy to see which chips plateau earlier or higher than others.

<!-- FIXME: shelved  -  re-run after PCIe link config is validated
![D2H Multi-Chip Throughput vs FIFO Size](charts/mc_d2h_throughput_vs_fifo.png)
-->

---

**`h2d_mc_d2h_throughput_heatmap.png`**  -  Heatmap of H2D (DEVICE\_PULL) peak throughput (GB/s) across every chip on the system. Fixed at 256 KB pages across FIFO sizes 256 KB, 512 KB, 1 MB. Directly comparable to the D2H heatmap above  -  the throughput gap between high-bandwidth and low-bandwidth chips is visible in both directions.

<!-- FIXME: shelved  -  re-run after PCIe link config is validated
![H2D Multi-Chip Throughput Heatmap  -  All Chips x FIFO Size](charts/h2d_mc_d2h_throughput_heatmap.png)
-->

---

**`h2d_mc_d2h_throughput_vs_fifo.png`**  -  Line chart version of the H2D multi-chip sweep: DEVICE\_PULL throughput vs FIFO size, one line per chip.

<!-- FIXME: shelved  -  re-run after PCIe link config is validated
![H2D Multi-Chip Throughput vs FIFO Size](charts/h2d_mc_d2h_throughput_vs_fifo.png)
-->

---

**`h2d_mc_d2h_throughput_bar.png`**  -  Grouped bar chart of the H2D multi-chip sweep: one group per FIFO size, one bar per chip. An alternative view that makes magnitude differences between chips easier to compare at a glance.

<!-- FIXME: shelved  -  re-run after PCIe link config is validated
![H2D Multi-Chip Throughput Bar Chart  -  All Chips x FIFO Size](charts/h2d_mc_d2h_throughput_bar.png)
-->

---

## 5. Interpreting Results

The charts in S.4 encode several independent variables at once. This section explains which variable to read against which axis for each metric.

### Throughput vs. FIFO size

Throughput rises as FIFO size grows and then **plateaus**. The plateau begins when the FIFO is large enough that the sender is never stalled waiting for receiver acknowledgements. Before the plateau, the sender is back-pressured after every page (or small batch), and the throughput equals approximately `page_size / round_trip_latency`. After the plateau, throughput is limited by the PCIe bandwidth ceiling.

### Throughput vs. page size

Very small pages (64-256 B) have very low throughput because the per-page fixed overhead (NOC command setup, PCIe transaction framing, `bytes_sent` notification write) dominates over the data transfer time. Throughput rises roughly linearly with page size until it saturates the PCIe link bandwidth. D2H saturates around 4 KB pages; H2D DEVICE_PULL saturates later due to per-page completion overhead.

### Latency vs. page size

Latency grows with page size because more data must traverse PCIe. For small pages the dominant cost is the protocol overhead (roughly a fixed number of NOC round-trips), not the data volume. For large pages the DMA time dominates.

### HOST\_PUSH vs. DEVICE\_PULL (H2D)

- **HOST\_PUSH** generally has lower latency because the host can write directly into device L1 with a single TLB write, avoiding the device issuing a separate NOC read.
- **DEVICE\_PULL** frees the host CPU  -  the host only updates `bytes_sent`, while the device issues the bulk PCIe read. The trade-off is throughput: non-posted reads require PCIe completion TLPs, which cap DEVICE\_PULL below the posted-write ceiling of HOST\_PUSH and D2H.

### Tail latency (p99 vs. avg)

Large gaps between `avg_us` and `p99_us` indicate interference from the OS scheduler, PCIe power management, or NUMA effects. The warmup is designed to minimise this for early iterations, but OS preemption can still spike individual iterations. Training teams should budget for p99 latency, not average, when sizing timeout windows or synchronisation barriers.

The per-iteration ping plots (**S.4.5**) expose this jitter directly  -  any iteration that spikes significantly above the median is a scheduling artefact, not a hardware limit.

### Per-chip throughput variation

As described in **S.1.1**, the throughput gap between high-bandwidth and low-bandwidth chips does not improve with larger pages or larger FIFOs  -  it is a PCIe physical-layer constraint (see S.4.6 for measured values).

Within the 4 high-bandwidth chips there may also be chip-to-chip variation  -  see **S.4.6** (`mc_d2h_throughput_heatmap.png`) for the measured spread across your specific system.

---

## 6. Running the Benchmarks

All tests require a system with vIOMMU enabled. They will `GTEST_SKIP` automatically on unsupported systems via the `GetMemoryPinningParameters` check.

> **Prerequisite  -  KMD >= 2.7:** Linux kernels 6.5-6.12 force all Blackhole Galaxy PCIe links to Gen 1 during hot-plug enumeration. KMD 2.7 retrains them to full Gen 4 speed. Results collected on earlier KMD versions will show ~250 MB/s for the 28 low-bandwidth chips instead of the true ~2 GB/s Gen 4 x1 ceiling. Verify with `cat /sys/bus/pci/devices/<bdf>/current_link_speed` before benchmarking.

Single-chip benchmarks target **Tray 1, ASIC Location 6**  -  one of the 4 chips with PCIe Gen 4 x8  -  as a fixed reference for peak numbers. The multi-chip benchmark sweeps all 32 chips to capture the full spread of performance regimes.

Build the test binary:

```bash
./build_metal.sh
```

Run a specific benchmark (e.g., D2H latency):

```bash
./build/test/tt_metal/distributed/test_hd_sockets \
    --gtest_filter="HDSocketFixture.D2HSocketLatencyBenchmark" \
    2>&1 | tee d2h_latency_results.csv
```

Analyse and plot results:

```bash
# Throughput chart
python3 tests/tt_metal/distributed/analyze_hd_sockets.py \
    --throughput d2h_throughput_results.csv

# Latency chart
python3 tests/tt_metal/distributed/analyze_hd_sockets.py \
    --latency d2h_latency_results.csv

# Multi-chip throughput chart
python3 tests/tt_metal/distributed/analyze_hd_sockets.py \
    --multichip multichip_bench.log
```

---

## 7. Benchmark Suite Reference

All benchmarks live in `tests/tt_metal/distributed/test_hd_sockets.cpp` and run under the `HDSocketFixture` Google Test fixture.

**Device targeting:** All single-chip benchmarks run on a standardised target: **Tray 1, ASIC Location 6**, selected using `get_target_benchmark_worker_core()`. ASIC 6 is one of the 4 Gen 4 x8 chips (one per tray, see **S.1.1**); any of the 4 would give similar results  -  Tray 1 ASIC 6 is used as a fixed reference. The multi-chip benchmark additionally sweeps every chip on the system to give a system-wide picture.

| Test Name | Direction | What It Measures |
|-----------|-----------|-----------------|
| `D2HSocketThroughputBenchmark` | D2H | Steady-state bulk throughput |
| `D2HSocketLatencyBenchmark` | D2H | Per-iteration round-trip latency (with data DMA) |
| `D2HSocketPingBenchmark` | D2H | Pure signalling round-trip (no data DMA) |
| `D2HSocketMultiChipMaxThroughputBenchmark` | D2H | Peak throughput across all chips on system |
| `H2DSocketThroughputBenchmark` | H2D | Steady-state bulk throughput (both modes) |
| `H2DSocketMultiChipMaxThroughputBenchmark` | H2D | Peak DEVICE_PULL throughput across all chips (256 KB pages, FIFO 256 KB-1 MB) |
| `H2DSocketLatencyBenchmark` | H2D | Per-iteration round-trip latency (both modes) |
| `H2DSocketPingBenchmark` | H2D | Pure signalling round-trip (both modes) |

### D2HSocketThroughputBenchmark

The baseline D2H throughput test. The device kernel (`pcie_socket_sender.cpp`) writes `data_size` bytes per iteration using chunked NOC writes across PCIe into a host-pinned FIFO. A single pair of device-side timestamps brackets the full multi-iteration run; the host computes average per-page cycles and GB/s. Sweeps page sizes up to 256 KB and FIFO sizes up to 512 MB.

### D2HSocketLatencyBenchmark

Measures per-iteration round-trip latency on the D2H path with actual data DMA. Uses `pcie_socket_data_ping.cpp` on the device: each iteration sends one page to the host, then calls `socket_barrier` to wait for the host's acknowledgement. Five warmup iterations precede 100 timed iterations; each timed delta is stored in the L1 measurement buffer and read back for percentile reporting. Sweeps page sizes and a subset of FIFO sizes chosen to cover latency-sensitive operating points (1 KB to 512 MB FIFO).

### D2HSocketPingBenchmark

Measures **pure signalling overhead** on the D2H path  -  no data DMA occurs. Uses `pcie_socket_ping.cpp`: the device calls `socket_reserve_pages` / `socket_push_pages` / `socket_notify_receiver` / `socket_barrier` with no actual payload write. The host calls `output_socket.read()` to consume the page slot and send the ack. This isolates the flow-control protocol overhead from the data transfer cost. The first config also dumps raw per-iteration data to `tests/tt_metal/distributed/ping_iterations.csv` for jitter analysis.

### D2HSocketMultiChipMaxThroughputBenchmark

Sweeps **every chip** on the system (identified via `PhysicalSystemDescriptor` + tray/ASIC location metadata) and measures D2H throughput at 64 KB pages  -  safely above the ~4 KB throughput knee  -  across five FIFO sizes (1 MB, 4 MB, 16 MB, 64 MB, 256 MB). Produces a CSV with tray ID, ASIC location, and mesh coordinate columns so per-chip variation across the tray can be compared. Total data transferred per configuration: 1 GB.

### H2DSocketMultiChipMaxThroughputBenchmark

Sweeps **every chip** on the system (all 32 chips in a Blackhole Galaxy) and measures H2D throughput using **DEVICE\_PULL** at 256 KB pages  -  the empirically highest-throughput page size for this mode  -  across three FIFO sizes: 256 KB, 512 KB, and 1 MB. Total data transferred per configuration: 1 GB. Analogous to `D2HSocketMultiChipMaxThroughputBenchmark` on the H2D path (same CSV format, same chip enumeration).

### H2DSocketThroughputBenchmark

Measures H2D steady-state throughput for both `HOST_PUSH` and `DEVICE_PULL` modes in a single test. For HOST\_PUSH, the device kernel is `h2d_throughput_host_push.cpp`; for DEVICE\_PULL, it is `h2d_throughput_device_pull.cpp`. Same single-aggregate-timestamp methodology as `D2HSocketThroughputBenchmark`. Sweeps FIFO sizes up to 1 MB, page sizes up to 256 KB.

### H2DSocketLatencyBenchmark

Measures per-iteration round-trip latency on the H2D path for both `HOST_PUSH` and `DEVICE_PULL`. Uses `h2d_socket_data_ping_host_push.cpp` and `h2d_socket_data_ping_device_pull.cpp` respectively. Both kernels follow the same 5-warmup + 100-timed-iteration pattern with per-iteration L1 measurement buffers.

**What the device timer captures:** Each iteration records `start = get_timestamp()` *before* calling `socket_wait_for_pages`, then `end = get_timestamp()` after `socket_notify_sender` completes. The measured cycles therefore cover the full round-trip: (a) spin-wait for the host write to arrive over PCIe, (b) device-side copy to local buffer, and (c) the acknowledgement NOC write back to host-pinned memory.

### H2DSocketPingBenchmark

Measures **pure signalling overhead** on the H2D path for both modes. Uses `h2d_socket_ping.cpp` on the device (no DMA in device kernel). The host issues `write + barrier` per iteration (5 warmup + 100 timed). Each iteration's cycle delta is stored per-iteration and also dumped to `h2d_ping_iterations_HOST_PUSH.csv` / `h2d_ping_iterations_DEVICE_PULL.csv` for jitter analysis. Compares mode overhead directly since the kernel path is identical for both modes.
