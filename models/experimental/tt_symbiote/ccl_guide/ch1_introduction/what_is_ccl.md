# 1.1 What Is CCL?

## The Problem: Tensors Don't Know About Chips

A neural network layer does not care whether its input tensor lives on one chip or forty. The mathematics of a matrix multiplication is the same regardless. What changes is the *logistics*: when a tensor is sharded across N devices, every device needs to either (a) access the pieces it does not own, or (b) agree with its peers on a reduced result, before it can produce a correct output.

This logistics problem is the entire reason CCL exists. CCL is the subsystem in tt-metal that answers the question: *given a tensor that is distributed across a set of chips according to some sharding strategy, how do we move or combine its pieces efficiently over the physical Ethernet fabric?*

Without CCL, every model developer would have to hand-write the Ethernet data movement kernels, manage synchronization semaphores, handle link failure recovery, and tune buffer sizes for throughput — for every collective pattern, for every topology. CCL packages all of that complexity behind a small, composable Python API.

---

## Analogy: MPI and NCCL

If you have written distributed training code with PyTorch, you have likely used NCCL (NVIDIA Collective Communications Library) through `torch.distributed`. The conceptual model is nearly identical:

| Concept | MPI / NCCL | tt-metal CCL |
|---------|-----------|--------------|
| Participating processes | MPI ranks / NCCL ranks | Tenstorrent devices in a mesh |
| Communication backend | InfiniBand / NVLink | Ethernet fabric (ERISC cores) |
| Collective API | `dist.all_reduce(tensor)` | `ttnn.all_reduce(tensor)` |
| Topology abstraction | communicator (`MPI_Comm`) | `MeshDevice` + `Topology` enum |
| Low-level buffer management | NCCL channels | `EriscDatamoverConfig` channels |

The key difference is that NCCL sits *outside* the compute graph — it is a separate communication library bolted onto a GPU runtime. In tt-metal, CCL operations are first-class TTNN operations that participate in the same program compilation pipeline as compute kernels. This means CCL can be *fused* with adjacent compute ops (Chapter 5) and *overlapped* with independent compute (Chapter 4) at the compiler level, not just by hand.

---

## What CCL Provides: The Operation Catalogue

CCL in tt-metal is split into two layers:

- **Core ops** — stable, tested, production-ready collective operations
- **Experimental async ops** — higher-performance variants that use asynchronous pipelines and op fusion; API may change

### Core Operations

All core ops live under `ttnn/cpp/ttnn/operations/ccl/`.

#### AllGather

Collects all shards of a tensor from every device and makes the full tensor available on every device.

```
Before AllGather (4 devices, tensor sharded on dim=0):
  Device 0: [A]
  Device 1: [B]
  Device 2: [C]
  Device 3: [D]

After AllGather:
  Device 0: [A, B, C, D]
  Device 1: [A, B, C, D]
  Device 2: [A, B, C, D]
  Device 3: [A, B, C, D]
```

**Typical use:** Gather distributed activations before a layer that cannot be sharded (e.g., LayerNorm across the full hidden dimension), or replicate weights that are too small to shard profitably.

```python
# All four devices end up with the full tensor
full_tensor = ttnn.all_gather(sharded_tensor, dim=0, num_links=1, topology=ttnn.Topology.Ring)
```

#### ReduceScatter

The *inverse* of AllGather in terms of data flow. Every device contributes a shard; the shards are element-wise reduced (summed, maxed, etc.) and then the result is distributed so each device holds one slice of the reduced output.

```
Before ReduceScatter (4 devices, each holding partial sums):
  Device 0: [A0+B0+C0+D0_partial, A1+..., A2+..., A3+...]  (per-device partial)
  ...

After ReduceScatter (dim=0, op=sum):
  Device 0: reduced[0:N/4]
  Device 1: reduced[N/4:N/2]
  Device 2: reduced[N/2:3N/4]
  Device 3: reduced[3N/4:N]
```

**Typical use:** After a tensor-parallel linear layer where each device accumulated partial dot products — scatter-reduce to distribute the final output.

```python
scattered = ttnn.reduce_scatter(
    partial_output,
    dim=3,                   # which dimension to scatter across devices
    num_links=1,
    topology=ttnn.Topology.Ring,
)
```

#### AllReduce

Conceptually AllReduce = ReduceScatter + AllGather. Every device ends up with the *full* reduced tensor. In practice CCL may implement this differently for efficiency.

```
Before AllReduce (4 devices, each holding partial gradient):
  Device 0: grad_partial_0
  Device 1: grad_partial_1
  Device 2: grad_partial_2
  Device 3: grad_partial_3

After AllReduce:
  All devices: grad_partial_0 + grad_partial_1 + grad_partial_2 + grad_partial_3
```

**Typical use:** Data-parallel gradient synchronization, or combining partial attention scores in tensor-parallel attention.

```python
synced_grad = ttnn.all_reduce(
    local_grad,
    cluster_axis=0,          # which mesh axis to reduce across
    num_links=1,
    topology=ttnn.Topology.Ring,
)
```

#### Broadcast

Sends a tensor from one *sender* device to all other devices in the group. Unlike AllGather, only one device contributes data.

```
Before Broadcast (sender = Device 0):
  Device 0: [data]
  Device 1: [empty / stale]
  Device 2: [empty / stale]
  Device 3: [empty / stale]

After Broadcast:
  All devices: [data]
```

**Typical use:** Distributing a freshly computed scalar loss or a shared weight update from a parameter server chip.

```python
replicated = ttnn.broadcast(
    tensor,
    sender_coord=ttnn.MeshCoordinate((0, 0)),  # mesh coordinate of the sender
    topology=ttnn.Topology.Linear,
)
```

#### AllBroadcast

Each device sends its local tensor to all other devices, and all devices end up with a concatenation of all shards — but unlike AllGather, no specific ordering is implied and the scatter dimension semantics differ. Useful for MoE expert broadcasting patterns.

#### ReduceToRoot

Reduces across all devices but only *one* device (the "root") receives the final result. Others discard the intermediate. Useful when only one chip needs the aggregated value (e.g., a loss computation that only the host-facing chip needs to read back).

#### MeshPartition

Partitions a tensor according to the mesh geometry, distributing slices to each device. The logical inverse of AllGather in the data-placement sense.

#### AllToAllDispatch / AllToAllCombine

Specialized operations for Mixture-of-Experts (MoE) routing. Tokens are dispatched to expert devices (AllToAllDispatch) and then the expert outputs are combined back (AllToAllCombine). Chapter 6 covers these in depth.

---

### Experimental Async Operations

These live under `ttnn/cpp/ttnn/operations/experimental/ccl/` and are the frontier of CCL development. They expose the same conceptual operations as the core ops but with:

- **Asynchronous execution**: the operation returns immediately; a `GlobalSemaphore` signals completion
- **Op fusion**: communication fused directly with adjacent matmuls in a single program
- **Minimal-buffer variants**: reduced L1 usage for memory-constrained topologies

| Operation | Description |
|-----------|-------------|
| `all_gather_async` | Non-blocking AllGather |
| `all_reduce_async` | Non-blocking AllReduce |
| `reduce_scatter_minimal_async` | ReduceScatter with minimal L1 footprint |
| `all_to_all_async` | Non-blocking AllToAll for MoE |

Additional fused variants exist for specific model architectures (Llama, DeepSeek, ring-attention); see `ttnn/cpp/ttnn/operations/experimental/ccl/` for the current list.

> **Gotcha:** Experimental ops have no stability guarantee. Their signatures, configuration structs, and even existence may change between tt-metal releases. Use them when you need maximum performance and are prepared to track upstream changes. For stable production code, prefer the core ops.

---

## What CCL Is NOT

It is equally important to clarify scope:

- **CCL is not a general mesh networking library.** It is specifically designed for collective communication patterns over a fixed device topology. Point-to-point sends between arbitrary pairs of devices are handled at a lower level (the fabric layer).
- **CCL is not responsible for sharding tensors.** The `ttnn.MeshDevice` and tensor distribution utilities handle how tensors are laid out across devices before a collective is called. CCL assumes the sharding is already in place.
- **CCL does not manage device discovery or initialization.** That is handled by `tt_metal` and `MeshDevice`.
- **CCL is not a fault-tolerance layer.** There is no automatic retry or replication; if an Ethernet link fails during a collective, the operation fails.

---

## Relationship to the tt-fabric Layer

Below CCL sits the **tt-fabric** layer, which provides the raw Ethernet link abstractions: link enumeration, buffer management, and the `Topology` enum that CCL exposes at the Python level. CCL builds its collective algorithms *on top of* tt-fabric primitives, rather than directly manipulating Ethernet hardware.

The `Topology` enum (`tt::tt_fabric::Topology`) has five values in total; two are currently surfaced through the CCL Python API:

- `Topology.Ring` — packets circulate in a ring; every device is both sender and forwarder
- `Topology.Linear` — one end initiates; data flows in one direction only

Choosing the wrong topology for your physical wiring will either produce incorrect results or a runtime error. Section 1.2 explains how physical topology maps to this enum.

---

*Next: [1.2 Hardware Topology](hardware_topology.md)*
