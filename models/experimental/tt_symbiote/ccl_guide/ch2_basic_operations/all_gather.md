# 2.1 AllGather

## Concept

AllGather answers a single question: *each device holds one shard of a tensor — how do all devices end up holding the entire tensor?*

After AllGather every participating device has an identical full copy of the tensor, assembled by concatenating the shards in device order along the specified dimension.

```
Before AllGather   (4 devices, tensor sharded on dim=3, shape [1,1,32,64] per device)

  Device 0: [1, 1, 32,  64]  ← columns   0.. 63
  Device 1: [1, 1, 32,  64]  ← columns  64..127
  Device 2: [1, 1, 32,  64]  ← columns 128..191
  Device 3: [1, 1, 32,  64]  ← columns 192..255

After AllGather   (dim=3)

  Device 0: [1, 1, 32, 256]  ← full tensor
  Device 1: [1, 1, 32, 256]
  Device 2: [1, 1, 32, 256]
  Device 3: [1, 1, 32, 256]
```

The output shape on `dim` is `input_shape[dim] * N` where N is the number of participating devices.

### When to use AllGather

| Situation | Why AllGather |
|-----------|--------------|
| Tensor-parallel linear layer (column-parallel): each device computed `Y_local = X @ W_col_shard` | Need full `Y` on all devices before the next layer that cannot be sharded |
| KV-cache sharded across heads | Gather all head slices before attention score computation that spans all heads |
| Small weight tensor replicated after initial sharding | Cheaper to gather once than to keep it sharded through non-shardable operations |
| LayerNorm / RMSNorm input that requires full hidden dim | These norms reduce over the full hidden dimension; the input must be unsharded |

AllGather is *not* the right choice when the downstream layer can directly consume the scattered output — in that case ReduceScatter is more bandwidth-efficient (covered in Chapter 3).

---

## Python API

Source: `ttnn/cpp/ttnn/operations/ccl/all_gather/all_gather.hpp`

```python
output_tensor = ttnn.all_gather(
    input_tensor,                    # ttnn.Tensor — the local shard on each device
    dim,                             # int — dimension along which to gather
    cluster_axis=None,               # Optional[int] — mesh axis (0=rows, 1=cols); None = all devices
    subdevice_id=None,               # Optional[ttnn.SubDeviceId] — sub-device pin; leave None normally
    memory_config=None,              # Optional[ttnn.MemoryConfig] — output layout; None = match input
    output_tensor=None,              # Optional[ttnn.Tensor] — preallocated output buffer
    num_links=None,                  # Optional[int] — Ethernet links to use; None = auto
    topology=None,                   # Optional[ttnn.Topology] — Ring or Linear; None = auto
    sub_core_grids=None,             # Optional[CoreRangeSet] — restrict worker core range
)
# Returns: ttnn.Tensor with output_shape[dim] = input_shape[dim] * num_devices
```

### Parameter notes

**`dim`** (required, positional): The tensor dimension along which shards are concatenated. For a 4-D tensor `[W, Z, Y, X]`:
- `dim=0` — batch gather (unusual; shards differ in batch)
- `dim=1` — channel/head gather (common in multi-head attention)
- `dim=2` — sequence gather (used in context-parallel models)
- `dim=3` — hidden-dim gather (most common in column-parallel linear)

Negative values are accepted (`dim=-1` = last dim).

**`cluster_axis`**: On a 2-D mesh, specifies which axis the AllGather runs along. `cluster_axis=1` gathers across all devices with the same row index (i.e., along columns). When unset, AllGather runs across the entire flattened device list.

> **Gotcha:** `cluster_axis=0` gathers across all devices sharing the same *column* index (it strides along axis 0, the row dimension). This is the opposite of what "axis 0 = rows" might suggest. Think of it as: "gather along the axis-0 direction" means iterating over different row values at fixed column. Verify with a small mesh before deploying.

**`num_links`**: Each link uses ERISC L1 for EDM channel buffers. With `num_links=None`, the runtime picks the maximum available links, which is usually optimal. Set it explicitly only when L1 pressure requires limiting link count.

**`output_tensor`**: When provided, AllGather writes directly into this pre-allocated buffer instead of allocating a new one. Useful in tight memory budgets or when the output buffer is pinned for a downstream fused op.

**`sub_core_grids`**: Restricts which Tensix cores run the AllGather reader/writer kernels. Rarely needed; used when other kernels have reserved specific core ranges.

---

## Illustrative Examples

### Basic 1-D mesh AllGather

```python
import torch
import ttnn

# 1×4 mesh (4 devices in a row)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))

# Full tensor: [1, 1, 32, 256]
full = torch.randn(1, 1, 32, 256, dtype=torch.bfloat16)

# Shard along last dim: each device gets [1, 1, 32, 64]
sharded = ttnn.from_torch(
    full,
    device=mesh,
    mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)

# Gather: every device gets [1, 1, 32, 256]
gathered = ttnn.all_gather(sharded, dim=3, topology=ttnn.Topology.Ring)
# gathered.shape == [1, 1, 32, 256] on every device
```

### 2-D mesh AllGather along one axis

```python
# 2×4 mesh: 2 rows, 4 columns (8 devices total)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(2, 4))

# Tensor sharded along dim=3 across columns only (4-way shard)
# Each device holds [1, 1, 32, 64]
sharded = ttnn.from_torch(
    full,
    device=mesh,
    mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=(2, 4), dims=(-1, -2)),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
)

# Gather only along column axis (cluster_axis=1 = same row, different columns)
# Each device ends up with [1, 1, 32, 256]; row axis stays independent
gathered = ttnn.all_gather(sharded, dim=3, cluster_axis=1, topology=ttnn.Topology.Ring)
```

### Pre-allocated output buffer

```python
# Pre-allocate the output on each device
output_shape = list(sharded.shape)
output_shape[3] *= 4  # 4 devices
output_buf = ttnn.allocate_tensor_on_device(
    ttnn.Shape(output_shape),
    ttnn.bfloat16,
    ttnn.TILE_LAYOUT,
    mesh,
    ttnn.DRAM_MEMORY_CONFIG,
)

gathered = ttnn.all_gather(sharded, dim=3, output_tensor=output_buf, topology=ttnn.Topology.Ring)
```

---

## Data Flow: Ring AllGather

With `topology=ttnn.Topology.Ring` and N devices, the ring AllGather uses a **pipeline rotation** algorithm: each device's shard travels around the ring, with each device forwarding what it received to the next device while simultaneously writing the arrived shard into its output buffer.

```
Initial state (each device owns one shard, labeled A/B/C/D):

  Dev 0: [A | _ | _ | _]      Dev 1: [_ | B | _ | _]
  Dev 2: [_ | _ | C | _]      Dev 3: [_ | _ | _ | D]

Round 1 — each device sends its own shard to the next:
  Dev 0 → Dev 1 sends A
  Dev 1 → Dev 2 sends B
  Dev 2 → Dev 3 sends C
  Dev 3 → Dev 0 sends D   (wrap-around link)

After round 1:
  Dev 0: [A | _ | _ | D]     Dev 1: [A | B | _ | _]
  Dev 2: [_ | B | C | _]     Dev 3: [_ | _ | C | D]

Round 2 — each device forwards what it received:
  Dev 0 → Dev 1 sends D
  Dev 1 → Dev 2 sends A
  Dev 2 → Dev 3 sends B
  Dev 3 → Dev 0 sends C

... (N-1 rounds total)

Final state:
  Dev 0: [A | B | C | D]     Dev 1: [A | B | C | D]
  Dev 2: [A | B | C | D]     Dev 3: [A | B | C | D]
```

The total data transferred per device is `(N-1) * shard_size`. With a ring each device acts as both sender and forwarder, so all N links are active simultaneously — bandwidth scales with N.

### Linear AllGather

With `topology=ttnn.Topology.Linear`, there is no wrap-around link. The algorithm runs a single sweep from device 0 to device N-1: each device accumulates shards as they arrive, then forwards the growing partial result to the next device.

```
Dev 0 → Dev 1 → Dev 2 → Dev 3

Dev 0 sends [A]          →  Dev 1 now has [A, B]
Dev 1 sends [A, B]       →  Dev 2 now has [A, B, C]
Dev 2 sends [A, B, C]    →  Dev 3 now has [A, B, C, D]
```

This is simpler but serialized — only one hop is active at a time. Data volume per hop grows linearly (hop 1 sends `S`, hop 2 sends `2S`, ...), making linear AllGather significantly slower than ring for more than 2 devices. Use it only when no wrap-around link is physically available.

> **Gotcha:** The `topology` parameter must match your physical cabling. If you specify `ttnn.Topology.Ring` on a system without a physical wrap-around Ethernet cable, the EDM on device N-1 will wait indefinitely for the message from device 0 — the collective hangs with no timeout or error.

---

## Under the Hood: Kernel Internals

### Program structure

When `ttnn.all_gather` is called, the program factory (`all_gather_program_factory.cpp`) builds a Metal program with two kernel types per device:

1. **Reader kernel** (Tensix dataflow): reads the local input shard from L1 or DRAM tile-by-tile and writes it into the ERISC L1 "outbox" buffer via `noc_async_write`. Signals the ERISC via semaphore once a chunk is ready.

2. **Writer kernel** (Tensix dataflow): waits on a semaphore from the ERISC indicating that received data has been written into an ERISC L1 "inbox" buffer, then NOC-reads it into the output tensor's L1 or DRAM location.

The EDM kernel on the ERISC core runs concurrently with both Tensix kernels, handling the actual Ethernet transfer.

### L1 buffer layout

```
ERISC L1 (per-link)
┌─────────────────────────────────────────────────┐
│  Handshake region   (16 bytes, fixed address)   │
│  Semaphore region   (32 bytes × N channels)     │
│  Buffer region                                  │
│   slot 0: [  eth_buffer_size_bytes  ]           │
│   slot 1: [  eth_buffer_size_bytes  ]           │
│   ...                                           │
│   slot k: [  eth_buffer_size_bytes  ]           │
└─────────────────────────────────────────────────┘
```

`eth_buffer_size_bytes` is computed by `EriscDatamoverConfig::compute_buffer_size(num_channels, num_buffers_per_channel, page_size)`. Multiple buffer slots enable pipelining: the EDM can be transmitting slot 0 while the reader kernel is filling slot 1.

### GlobalSemaphore in AllGather

`AllGatherDeviceOperation::AllGatherProgram` allocates a `GlobalSemaphore` (seen in `all_gather_device_operation.hpp` as `multidevice_semaphores` and `barrier_semaphore`). The barrier semaphore provides a cross-device synchronization point: the AllGather is not considered complete until all devices have confirmed receipt of all shards. This prevents a device from proceeding to the next operation before its incoming data is fully written.

### Program caching

The program factory uses a hash over `operation_attributes_t` (dim, topology, num_links, memory_config, cluster_axis, sub_core_grid) to cache compiled programs. If you call `ttnn.all_gather` repeatedly with the same parameters and same tensor shape, subsequent calls reuse the cached program, avoiding recompilation. Changing any attribute — including `memory_config` — invalidates the cache.

> **Gotcha:** Changing the tensor shape between calls (e.g., varying batch size) invalidates the program cache and triggers recompilation, which is expensive. Keep tensor shapes constant across calls within a model's inference loop.

---

## Memory Layout and Performance

### Input layout impact

AllGather performance is sensitive to where the input shard lives:

| Input location | Effect on AllGather |
|---------------|---------------------|
| L1 sharded (contiguous per-core) | Reader kernel can issue a few large NOC transfers; best throughput |
| DRAM interleaved | Reader kernel must gather pages from multiple banks; adds DRAM latency |
| L1 interleaved | Intermediate; faster than DRAM but less efficient than L1-sharded |

To maximise AllGather throughput, use L1-sharded inputs:

```python
shard_config = ttnn.create_sharded_memory_config(
    shape=(1, 1, 32, 64),          # per-device shard shape
    core_grid=ttnn.CoreGrid(r=1, c=8),
    strategy=ttnn.ShardStrategy.WIDTH,
)
input_sharded = ttnn.to_memory_config(input_tensor, shard_config)
gathered = ttnn.all_gather(input_sharded, dim=3, topology=ttnn.Topology.Ring)
```

### Output layout

By default, the gathered output uses the same memory config as the input. When the next operation requires a different layout, specify `memory_config` explicitly to avoid a separate layout conversion:

```python
gathered = ttnn.all_gather(
    input_sharded,
    dim=3,
    topology=ttnn.Topology.Ring,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,   # output goes to DRAM
)
```

> **Gotcha:** Not all input shard strategies are supported. If you see "Unsupported memory layout for all_gather", check that your `ShardStrategy` is WIDTH or HEIGHT — BLOCK sharding is not always supported depending on the gather dimension. Check `all_gather_program_factory.cpp` for the specific `TensorMemoryLayout` guards.

### Effect of `num_links`

With `num_links=2` (if 2 physical links exist between adjacent chips), AllGather uses two independent EDM channels, effectively doubling Ethernet bandwidth utilisation. The reader kernel stripes data alternately across both channels. Each added link consumes approximately `2 * eth_buffer_size_bytes * num_slots` of ERISC L1.

For a 256 KB tensor on a 4-device ring, the difference between 1 and 2 links is usually small because the tensor fits in a single EDM pipeline cycle. For larger tensors (several MB), `num_links=2` can approach 2× throughput.

---

## Tile Padding and Row-Major Fallback

When the gather dimension has tile padding, or when the input is in row-major layout, AllGather automatically falls back to a composite implementation using AllBroadcast internally. This fallback is described in the nanobind docstring:

> "When the layout is row-major or we have tile padding on the gather dim, we use the composite all-gather implementation that falls back to all-broadcast."

This fallback is transparent to the caller but has higher bandwidth overhead (AllBroadcast transfers N times the data that a ring AllGather would). For performance-critical paths, ensure the gather dimension is tile-aligned and use TILE_LAYOUT.

> **Gotcha:** A tensor with `dim=3` and `shape[-1] = 100` will trigger the AllBroadcast fallback because 100 is not a multiple of 32 (the tile width). Pad to 128 before gathering if throughput matters.

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| AllGather hangs indefinitely | `Topology.Ring` on a chain without wrap-around cable | Switch to `Topology.Linear` or install the wrap-around link |
| `RuntimeError: Unsupported memory layout` | Input uses BLOCK sharding on an unsupported gather dim | Convert to WIDTH or HEIGHT sharding first |
| Output shape mismatch | `dim` points to a non-sharded dimension | Confirm the input was sharded on `dim` before calling AllGather |
| Program recompiles every iteration | Tensor shape changes between calls | Fix tensor shape; use padding to keep shapes constant |
| `cluster_axis` out of range | Passing `cluster_axis=1` to a 1×N mesh | 1-D meshes have only one axis; omit `cluster_axis` |

---

*Back to [Chapter 2 Index](index.md)*

*Next: [2.2 AllReduce](all_reduce.md)*
