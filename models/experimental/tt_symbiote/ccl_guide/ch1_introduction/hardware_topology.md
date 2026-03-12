# 1.2 Hardware Topology

## Why Topology Matters

The speed of a collective operation is bounded by two things: the compute required to reduce data, and the bandwidth available to move data across the chip boundary. On GPU clusters the network is relatively uniform — InfiniBand switches create an all-to-all bisection bandwidth fabric, so algorithms can largely ignore physical placement.

Tenstorrent systems are different. Chips are connected via point-to-point Ethernet links with **no switching fabric** between them. There is no router that can forward packets from chip 0 directly to chip 3 if they are not physically adjacent. Data must *hop* through intermediate chips. The physical wiring — which chip is bolted to which — directly determines what collective algorithms are efficient and which topology enum value you should pass to the CCL API.

This section explains the hardware from the chip outward: first the intra-chip communication network (NOC), then the inter-chip Ethernet subsystem (ERISC + EDM), and finally how collections of chips are wired into rings, linear chains, and meshes.

---

## Inside One Chip: The NOC

### Architecture Overview

Each Tenstorrent chip is a 2-D grid of tiles. Each tile can be one of several types:

```
┌─────────────────────────────────────────────────────┐
│  Tenstorrent Wormhole Chip (conceptual tile grid)    │
│                                                      │
│  T  T  T  T  T  T  T  T  T  T  T  T  T  T         │
│  T  T  T  T  T  T  T  T  T  T  T  T  T  T         │
│  T  T  T  T  T  T  T  T  T  T  T  T  T  T         │
│  T  T  T  T  T  T  T  T  T  T  T  T  T  T         │
│  T  T  T  T  T  T  T  T  T  T  T  T  T  T         │
│  T  T  T  T  T  T  T  T  T  T  T  T  T  T         │
│  T  T  T  T  T  T  T  T  T  T  T  T  T  T         │
│  T  T  T  T  T  T  T  T  T  T  T  T  T  T         │
│                                                      │
│  E  E  E  E  E  E  E  E  E  E  E  E  E  E  ← ERISC │
│                                                      │
│  Legend: T = Tensix compute tile,  E = ERISC tile   │
└─────────────────────────────────────────────────────┘
```

The relevant tile types for CCL are:

| Tile Type | Full Name | Role in CCL |
|-----------|-----------|-------------|
| **Tensix** | Tensix compute core | Runs reduction compute kernels; sends/receives data via NOC |
| **ERISC** | Ethernet RISC core | Runs EDM kernel; handles inter-chip transfers over physical Ethernet |
| **DRAM** | DRAM controller tile | Source/destination of large tensors; accessed via NOC |

### Network-on-Chip (NOC)

The NOC is the intra-chip interconnect. It is a 2-D torus mesh of wires connecting every tile to every other tile via a fast on-chip bus. Two NOC planes exist (NOC0 and NOC1), providing bidirectional data paths to reduce contention.

Key NOC primitives used by CCL kernels:

```cpp
// Asynchronous read: DMA from another tile's L1 or DRAM into this tile's L1
noc_async_read(src_noc_addr, dst_local_l1_addr, size);

// Asynchronous write: DMA from this tile's L1 to another tile's L1 or DRAM
noc_async_write(src_local_l1_addr, dst_noc_addr, size);

// Barrier: wait until all outstanding NOC transactions complete
noc_async_read_barrier();
noc_async_write_barrier();
```

These are called from within Tensix or ERISC kernel code. The NOC handles address translation, arbitration, and flow control. From a kernel programmer's perspective, the NOC appears as a flat address space across all tiles on the chip.

> **Gotcha:** NOC transactions are fire-and-forget until the barrier. If you issue `noc_async_write` and then immediately check the destination buffer on another tile, you may observe stale data. Always pair writes with `noc_async_write_barrier()` before signaling the consumer.

### L1 Memory

Each tile has a small amount of on-chip SRAM called **L1** (analogous to a GPU's shared memory but per-tile rather than per-SM). For CCL, L1 serves as the staging buffer between compute tiles and ERISC tiles:

```
  Compute tile L1                ERISC tile L1
  ┌──────────┐  noc_async_write  ┌──────────┐   Ethernet   ┌──────────────────┐
  │ tensor   │ ──────────────→  │ EDM buf  │ ──────────→  │  Remote chip     │
  │ shard    │                   │ (outbox) │              │  ERISC L1        │
  └──────────┘                   └──────────┘              └──────────────────┘
```

L1 size limits how much data can be "in flight" on the inter-chip link at any one time. This is the primary reason EDM channel configuration — buffer count, slot sizes, and channel count — must be sized to fit within ERISC L1. These values are passed to `EriscDatamoverBuilder`; see [Section 1.3](ttnn_ecosystem.md) for details.

---

## Inter-Chip Communication: ERISC and EDM

### ERISC Cores

ERISC (Ethernet RISC) cores are specialized processors on the periphery of the chip that are directly wired to the physical Ethernet PHY. Each ERISC core manages one Ethernet link. A typical Wormhole chip has multiple ERISC cores, one per physical Ethernet port.

ERISC cores are **not** general-purpose compute cores. They run a dedicated firmware called the **Ethernet Data Mover (EDM)**. When CCL launches a collective, it configures the EDM and then lets it run; the Tensix compute cores proceed with whatever work they can do in parallel (the foundation of async overlap, covered in Chapter 4).

### Ethernet Data Mover (EDM)

The EDM is a kernel that runs on ERISC cores throughout the duration of a CCL operation. Its source lives at:

```
ttnn/cpp/ttnn/operations/ccl/kernels/edm/erisc_datamover.cpp
```

The EDM implements a **credit-based flow control** protocol over the raw Ethernet link:

1. **Sender side**: pulls data from an L1 "outbox" buffer, packetizes it, and sends it over the Ethernet link. Decrements a credit counter for each packet sent.
2. **Receiver side**: receives packets, writes data into an L1 "inbox" buffer, and signals the consumer (either a Tensix tile via semaphore, or another EDM instance for multi-hop forwarding).
3. **Flow control**: the sender only sends when credits are available. The receiver grants credits by acknowledging consumed packets. This prevents the receiver's inbox buffer from overflowing.

```
Sender Chip                              Receiver Chip
─────────────────────────────────────────────────────────
Tensix L1                                                   Tensix L1
  │                                                              ▲
  │ noc_async_write                                   noc_async_write
  ▼                                                              │
ERISC L1 (outbox)                                     ERISC L1 (inbox)
  │                                                              │
  │  EDM kernel (sender)         EDM kernel (receiver)          │
  │  ┌──────────────────┐        ┌──────────────────┐           │
  └──│ pull → packetize │──ETH──▶│ receive → write  │───────────┘
     │ credit check     │◀──ACK──│ grant credit     │
     └──────────────────┘        └──────────────────┘
```

#### EDM Configuration

The `EriscDatamoverConfig` struct (`ttnn/cpp/ttnn/operations/ccl/ccl_host_datastructures.hpp`) controls how EDM channels are set up. See [Section 1.3 — Key C++ Types](ttnn_ecosystem.md) for the full definition and usage guidance.

> **Gotcha:** The total buffer space requested across all EDM channels (channel count × per-channel buffer size × slots per channel) must fit within the ERISC L1 unreserved region. Overshooting causes a silent runtime failure where the EDM cannot allocate its buffers and the collective hangs indefinitely. When in doubt, start with the defaults and only tune after profiling.

### Multi-Hop Forwarding

When device A needs to send data to device C, but A and C are not directly connected (only A–B and B–C links exist), the EDM on chip B acts as a **forwarder**:

```
  Chip A ──ETH──▶ Chip B ──ETH──▶ Chip C
         ERISC_A     ERISC_B     ERISC_C
         (sender)  (forwarder)  (receiver)
```

The EDM forwarder receives packets on its "inbox" Ethernet port and immediately re-sends them on its "outbox" port, without involving any Tensix compute cores. This is how a 4-device ring collective works even when the logical ring requires data to traverse intermediate devices.

---

## Physical Topology: How Chips Are Wired

### Single-Link Chain (Linear)

The simplest multi-chip configuration is a linear chain: chip 0 connects to chip 1, chip 1 connects to chip 2, and so on. There is no wrap-around link.

```
Chip 0 ──── Chip 1 ──── Chip 2 ──── Chip 3
  [0]         [1]         [2]         [3]
```

In a linear topology, data can only flow left-to-right or right-to-left. An AllGather must therefore proceed in two phases (one direction then the other) or use a pipeline that initiates from one end.

Use `ttnn.Topology.Linear` when:
- The physical wiring is a chain with no wrap-around
- You are broadcasting from one end (Broadcast op)
- You want minimal-latency unicast from a single sender

### Ring

A ring adds a wrap-around link from the last chip back to the first:

```
Chip 0 ──── Chip 1 ──── Chip 2 ──── Chip 3
  │                                    │
  └────────────────────────────────────┘
  (wrap-around Ethernet link)
```

With a ring, data can travel in *both* directions simultaneously. A well-implemented ring AllGather sends half the data clockwise and half counterclockwise, halving the effective latency compared to a linear topology at the same bandwidth.

Use `ttnn.Topology.Ring` when:
- There is a physical wrap-around Ethernet cable installed
- Your workload is latency-sensitive and benefits from bidirectional traversal
- You are running AllGather or ReduceScatter (both benefit significantly from ring)

> **Gotcha:** Specifying `Topology.Ring` without a physical wrap-around link will cause the EDM to wait for packets that never arrive — the collective will hang indefinitely. Always verify your hardware cabling before choosing Ring topology.

### 2-D Mesh

For larger systems (e.g., a T3000 or TG board), chips are arranged in a 2-D grid:

```
  Col:  0       1       2       3
Row 0: Chip00 ─ Chip01 ─ Chip02 ─ Chip03
        │         │         │         │
Row 1: Chip10 ─ Chip11 ─ Chip12 ─ Chip13
        │         │         │         │
Row 2: Chip20 ─ Chip21 ─ Chip22 ─ Chip23
        │         │         │         │
Row 3: Chip30 ─ Chip31 ─ Chip32 ─ Chip33
```

In the TTNN API, the `MeshDevice` encodes this geometry. The `cluster_axis` parameter in CCL ops selects whether the collective runs along mesh rows (axis=0) or mesh columns (axis=1).

```python
# AllReduce across all devices in the same mesh row (axis=0 = row direction)
result = ttnn.all_reduce(tensor, cluster_axis=0, topology=ttnn.Topology.Ring)

# AllReduce across all devices in the same mesh column (axis=1 = column direction)
result = ttnn.all_reduce(tensor, cluster_axis=1, topology=ttnn.Topology.Ring)
```

> **Gotcha:** `cluster_axis` semantics depend on how your `MeshDevice` was initialized. A `cluster_axis=0` collective runs across all devices that share the same column index (i.e., it strides along axis 0). Double-check the axis convention against your `MeshDevice` shape before deploying.

### Sub-Mesh and Multi-Axis Collectives

Some parallelism strategies (e.g., hybrid data-parallel + tensor-parallel) require collectives on *subsets* of the mesh. CCL handles this by operating on a sub-mesh view of the `MeshDevice`. Fully covered in Chapter 3; for now, note that every CCL op accepts an optional `device_range` or equivalent argument to restrict which devices participate.

---

## Link Count and Bandwidth

A single Ethernet link between two chips provides a fixed bandwidth. When multiple physical links exist between the same pair of chips (e.g., in a higher-end configuration), the `num_links` parameter tells CCL to stripe data across all available links:

```python
# Use 2 physical links between adjacent chips for higher throughput
result = ttnn.all_gather(tensor, dim=0, num_links=2, topology=ttnn.Topology.Ring)
```

Internally, CCL assigns one EDM channel per link and pipelines data across them. The benefit is roughly linear in `num_links` up to the point where the bottleneck shifts from link bandwidth to NOC bandwidth or L1 buffer capacity.

> **Gotcha:** Requesting `num_links=2` when only one physical link is available may not cause an immediate error at the Python API level, but will fail or silently misconfigure at device setup time. Always verify your hardware port count before setting `num_links` — check your board spec for the number of Ethernet ports available between adjacent chips.

---

## Bandwidth and Latency Characteristics

Understanding the rough numbers helps set performance expectations:

| Path | Approximate bandwidth |
|------|----------------------|
| NOC (intra-chip, L1 → L1) | ~500 GB/s aggregate |
| Ethernet link (inter-chip) | ~12.5 GB/s per link (e.g., Wormhole 100 Gbps Ethernet; generation-specific) |
| DRAM → L1 (via NOC) | ~256 GB/s aggregate |

The inter-chip Ethernet bandwidth is roughly 20–40x lower than the intra-chip NOC bandwidth. This asymmetry has a direct implication for collective design: **minimize the volume of data that crosses chip boundaries**. ReduceScatter (which reduces before scattering) is almost always preferable to AllGather (which moves full data first) when the result does not need to be replicated.

> **Gotcha:** These numbers are for fully-pipelined steady-state throughput. At small tensor sizes (< a few hundred KB), latency dominates: link startup overhead, EDM initialization, and synchronization costs swamp the transfer time. For very small tensors, consider batching multiple collectives or using a lower-level point-to-point primitive.

---

*Next: [1.3 CCL in the TTNN Ecosystem](ttnn_ecosystem.md)*
