# 2.3 Broadcast and AllBroadcast

## Concept

Broadcast and AllBroadcast answer different but related questions:

- **`ttnn.broadcast`**: *one device holds a tensor — how does every other device end up holding an identical copy?* The sending device is identified by a `MeshCoordinate`; all other devices receive the same tensor.

- **`ttnn.all_broadcast`**: *every device holds its own tensor — how does every device end up holding N copies, one from each peer?* After AllBroadcast, each device holds a list of N tensors: one from each device in the participating group.

Both operations preserve tensor shape. Neither performs any reduction.

```
ttnn.broadcast  (sender = Device 0)

  Before:
    Device 0: [T]          ← holds the source tensor
    Device 1: [?]
    Device 2: [?]
    Device 3: [?]

  After:
    Device 0: [T]
    Device 1: [T]
    Device 2: [T]
    Device 3: [T]

ttnn.all_broadcast  (4 devices, each with its own tensor)

  Before:
    Device 0: [A]
    Device 1: [B]
    Device 2: [C]
    Device 3: [D]

  After:
    Device 0: [A, B, C, D]  ← list of 4 tensors
    Device 1: [A, B, C, D]
    Device 2: [A, B, C, D]
    Device 3: [A, B, C, D]
```

### When to use Broadcast

| Situation | Why Broadcast |
|-----------|---------------|
| Distributing a weight tensor that is only loaded on the host-connected device | Broadcast from device 0 to all peers before inference starts |
| Sharing position embeddings or global bias across tensor-parallel devices | Cheaper to broadcast once from a known sender than to reload from host |
| MoE expert dispatch: router logits computed on one device, sent to all | Single-source distribution before expert selection |

### When to use AllBroadcast

| Situation | Why AllBroadcast |
|-----------|-----------------|
| MoE expert routing: each device must know every device's token assignments | Each device broadcasts its local routing decision; all devices receive all decisions |
| Collecting partial KV-caches for cross-attention in pipeline-parallel models | Each device holds one pipeline stage's KV; AllBroadcast gives all devices a view of all stages |

> **Gotcha:** `ttnn.all_broadcast` currently supports only `num_links=1` and `topology=Linear`. Ring topology and multi-link support are tracked in issue [#30798](https://github.com/tenstorrent/tt-metal/issues/30798). Do not pass `num_links > 1` or `topology=ttnn.Topology.Ring` to `ttnn.all_broadcast`; the operation will either raise an error or produce incorrect results.

---

## Python API: `ttnn.broadcast`

Source: `ttnn/cpp/ttnn/operations/ccl/broadcast/broadcast.hpp`

```python
output_tensor = ttnn.broadcast(
    input_tensor,                    # ttnn.Tensor — the source tensor (on sender and receivers)
    sender_coord,                    # ttnn.MeshCoordinate — identifies the sending device
    num_links=1,                     # int — Ethernet links to use; default 1
    memory_config=None,              # Optional[ttnn.MemoryConfig] — output layout; None = match input
    topology=ttnn.Topology.Linear,   # ttnn.Topology — Ring or Linear; default Linear
    cluster_axis=None,               # Optional[int] — mesh axis; None = all devices
    subdevice_id=None,               # Optional[ttnn.SubDeviceId] — sub-device pin; leave None normally
)
# Returns: ttnn.Tensor — identical tensor on every device
```

### Parameter notes

**`sender_coord`** (required, positional): A `ttnn.MeshCoordinate` identifying which device is the source. Constructed as `ttnn.MeshCoordinate((row, col))`. The device at that coordinate transmits its local copy of `input_tensor`; all other devices receive it.

> **Gotcha:** `sender_coord` is a `ttnn.MeshCoordinate`, not a plain Python tuple. Passing `(0, 0)` will raise a type error at the nanobind boundary. Always construct it explicitly: `ttnn.MeshCoordinate((0, 0))`.

**`topology`**: Defaults to `ttnn.Topology.Linear` — unlike AllGather and AllReduce which default to `None`. The Linear default is safe when physical ring cabling is absent. Set to `ttnn.Topology.Ring` explicitly if a wrap-around link is available and latency matters.

**`cluster_axis`**: For 2-D meshes, restricts the broadcast to one axis. `cluster_axis=0` broadcasts within each column group; `cluster_axis=1` broadcasts within each row group. Omit for 1-D meshes.

**`num_links`**: Defaults to `1`. Unlike AllGather and AllReduce, the default is explicit rather than auto-selected. Increase only if multiple physical links exist between adjacent chips.

---

## Python API: `ttnn.all_broadcast`

Source: `ttnn/cpp/ttnn/operations/ccl/all_broadcast/all_broadcast.hpp`

```python
output_tensors = ttnn.all_broadcast(
    input_tensor,                    # ttnn.Tensor — each device's local tensor
    cluster_axis=None,               # Optional[int] — mesh axis; None = all devices
    subdevice_id=None,               # Optional[ttnn.SubDeviceId] — sub-device pin; leave None normally
    memory_config=None,              # Optional[ttnn.MemoryConfig] — output layout; None = match input
    num_links=1,                     # int — MUST be 1 (see issue #30798)
    topology=ttnn.Topology.Linear,   # ttnn.Topology — MUST be Linear (see issue #30798)
)
# Returns: List[ttnn.Tensor] — N tensors, one from each participating device
```

> **Gotcha:** The return type is `List[ttnn.Tensor]`, not a single tensor. Index the result to access each device's contribution: `output_tensors[0]` is the tensor from device 0, `output_tensors[1]` from device 1, and so on. This differs from AllGather, which returns a single concatenated tensor.

### Parameter notes

**`num_links`**: Hard-coded default of `1` in the source. Do not change until issue #30798 is resolved.

**`topology`**: Hard-coded default of `ttnn.Topology.Linear`. Ring topology is not yet tested or supported. Do not pass `ttnn.Topology.Ring`.

**`cluster_axis`**: Same semantics as in AllGather and AllReduce. Restricts AllBroadcast to one axis of a 2-D mesh.

---

## Illustrative Examples

### Basic Broadcast from device 0

```python
import torch
import ttnn

# 1×4 mesh (4 devices in a row)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))

# Load tensor only on device 0 (the host-connected device)
tensor_data = torch.randn(1, 1, 32, 256, dtype=torch.bfloat16)

# Put the tensor on all devices (required by ttnn.from_torch even for broadcast)
# In practice you might load it on device 0 only via ttnn.allocate_tensor_on_device
distributed = ttnn.from_torch(
    tensor_data,
    device=mesh,
    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)

# Broadcast: device at (0, 0) is the sender
sender = ttnn.MeshCoordinate((0, 0))
broadcasted = ttnn.broadcast(
    distributed,
    sender_coord=sender,
    topology=ttnn.Topology.Linear,  # default; explicit for clarity
)
# broadcasted is the same tensor on every device
```

### Broadcast on a 2-D mesh along one axis

```python
# 2×4 mesh; broadcast within each row (cluster_axis=1 = same row, different columns)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(2, 4))

# Send from column 0 of each row (row 0 col 0, row 1 col 0)
sender = ttnn.MeshCoordinate((0, 0))
broadcasted = ttnn.broadcast(
    distributed,
    sender_coord=sender,
    cluster_axis=1,
    topology=ttnn.Topology.Linear,
)
# Within row 0: device (0,0) sends to (0,1), (0,2), (0,3)
# Within row 1: device (1,0) sends to (1,1), (1,2), (1,3)
```

### AllBroadcast: gather all devices' tensors

```python
# Each device holds a [1, 1, 32, 64] routing logit tensor
# After AllBroadcast, each device holds all 4 routing tensors

output_list = ttnn.all_broadcast(
    local_routing_logits,
    topology=ttnn.Topology.Linear,  # required; Ring not yet supported
    num_links=1,                    # required; > 1 not yet supported
)
# output_list[0] = routing logits from device 0
# output_list[1] = routing logits from device 1
# output_list[2] = routing logits from device 2
# output_list[3] = routing logits from device 3

# Concatenate if you need a single tensor view
all_logits = ttnn.concat(output_list, dim=3)
```

---

## Data Flow

### Broadcast: Linear chain

With `topology=ttnn.Topology.Linear` and a sender at device 0 in a 4-device chain:

```
Dev 0 (sender) → Dev 1 → Dev 2 → Dev 3

Step 1: Dev 0 sends T to Dev 1
Step 2: Dev 1 forwards T to Dev 2
Step 3: Dev 2 forwards T to Dev 3
```

Each step is sequential; total latency is `(N-1) * hop_latency`. For small tensors (model weight initialization, position embeddings) this is acceptable. For large tensors on many devices, consider whether AllGather from a sharded source would be faster.

### AllBroadcast: Simultaneous multi-device distribution

AllBroadcast runs its own dedicated kernel program across all devices simultaneously. Each device acts as both sender (for its own tensor) and receiver (for all other devices' tensors), coordinated through a ring-rotation-style data movement pattern.

Total data transferred per device: `(N-1) * tensor_size`. AllBroadcast is significantly more bandwidth-intensive than AllGather for the same data volume.

> **Bandwidth comparison:** AllGather on a ring transfers `(N-1) * shard_size` per device. AllBroadcast transfers `N * (N-1) * shard_size`. For 4 devices this is a 4× difference. Use AllBroadcast only when each device genuinely needs all N distinct tensors, not a concatenation of shards.

---

## Under the Hood

### Broadcast kernel structure

The Broadcast program factory (`broadcast_program_factory.hpp`) builds a program with:

1. **Sender kernel** (on the device identified by `sender_coord`): reads the full input tensor tile-by-tile from L1 or DRAM via `noc_async_read`, writes chunks to the ERISC L1 outbox buffer, signals the EDM via semaphore.

2. **Receiver kernel** (on all other devices): waits on the ERISC inbox semaphore, NOC-reads arriving data into the output tensor's storage location.

The `ring_size` field in `operation_attributes_t` records the number of participating devices, computed at program creation time from the mesh shape and `cluster_axis`. This value is baked into the Tensix kernel compile-time arguments.

### AllBroadcast kernel structure

AllBroadcast (`AllBroadcastProgramFactory`) builds its own dedicated mesh workload with sender/receiver kernels (analogous in structure to BroadcastProgramFactory) configured to collect tensors from all participating devices simultaneously. The program is cached per `(cluster_axis, topology, num_links, memory_config)`.

### GlobalSemaphore

Both Broadcast and AllBroadcast use a `GlobalSemaphore` for cross-device synchronization, created via `ttnn.create_global_semaphore(mesh_device, core_range_set, 0)`. The semaphore ensures all receiving devices have written the incoming tensor before the operation is considered complete.

---

## Memory Layout and Performance

### Input layout

Broadcast throughput is bounded by the Ethernet link rate, not the compute cores. The sender's reader kernel issues large contiguous NOC transfers when the input is L1-sharded. DRAM inputs add memory latency per tile, reducing effective link utilization. The L1-sharding setup is identical to AllGather (see [Section 2.1 — Memory Layout and Performance](all_gather.md#memory-layout-and-performance)); substitute `source_tensor` for `input_tensor` and call `ttnn.broadcast` in place of `ttnn.all_gather`.

### AllBroadcast output memory

The output `List[ttnn.Tensor]` allocates one output buffer per contributing device. If `memory_config=None`, each output tensor inherits the input's memory config. When N is large, this can consume significant L1. Specify `memory_config=ttnn.DRAM_MEMORY_CONFIG` explicitly to place outputs in DRAM when L1 is tight:

```python
output_list = ttnn.all_broadcast(
    local_tensor,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
```

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `TypeError: incompatible function arguments` on `sender_coord` | Passing a plain tuple `(0, 0)` instead of `ttnn.MeshCoordinate((0, 0))` | Wrap in `ttnn.MeshCoordinate(...)` |
| Broadcast hangs indefinitely | `Topology.Ring` on a system without a physical wrap-around Ethernet cable | Switch to `Topology.Linear` or install the wrap-around link |
| AllBroadcast produces incorrect results with `num_links > 1` | Issue #30798: multi-link not yet supported | Keep `num_links=1` until the issue is resolved |
| `IndexError` accessing `output_list[i]` | `all_broadcast` returned fewer tensors than expected | Verify `cluster_axis` matches the intended device count; 1-D meshes have fewer participants |
| Output shape unexpected | `broadcast` preserves the input shape on all devices including the sender | The sender's local tensor is unchanged; receivers receive an exact copy |
| `cluster_axis` out of range | Passing `cluster_axis=1` to a 1×N mesh | 1-D meshes have only one axis; omit `cluster_axis` |

---

*Back to [Chapter 2 Index](index.md)*

*Next: [Chapter 3 — Intermediate Operations](../ch3_intermediate_operations/index.md)*
