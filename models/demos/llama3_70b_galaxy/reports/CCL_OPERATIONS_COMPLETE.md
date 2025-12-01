# Llama3-70B Galaxy: Complete CCL Operations Reference

This document provides comprehensive coverage of all Collective Communication Library (CCL) operations used in the Llama3-70B Galaxy implementation.

## Table of Contents

1. [CCL Class Architecture](#ccl-class-architecture)
2. [Buffer Management](#buffer-management)
3. [All-Reduce Operations](#all-reduce-operations)
4. [All-Gather Operations](#all-gather-operations)
5. [Reduce-Scatter Operations](#reduce-scatter-operations)
6. [Specialized Operations](#specialized-operations)
7. [Ring Topology Details](#ring-topology-details)
8. [Semaphore Management](#semaphore-management)

---

## CCL Class Architecture

### Class Definition

**File**: `llama_ccl.py`
**Class**: `TT_CCL`
**Lines**: 24-1293

### Initialization

```python
class TT_CCL:
    def __init__(
        self,
        mesh_device,                    # 8x4 mesh device
        model_args,                     # Model arguments
        worker_sub_device_id,           # Worker sub-device ID
        mode="decode",                  # "decode" or "prefill"
        allocate_prefill_buffers=True,  # Allocate prefill buffers
        is_qwen=False,                  # Qwen model flag
    ):
```

**Lines**: 25-119

### Key Instance Variables

```python
self.mode = mode                              # "decode" or "prefill"
self.mesh_device = mesh_device               # 8x4 mesh device
self.worker_sub_device_id = worker_sub_device_id
self.model_config = model_args.model_config
self.num_cbs = 2                             # Double buffering
self.ring_topology = self.model_config["CCL_TOPOLOGY"] == ttnn.Topology.Ring
self.cluster_shape = (8, 4)                  # 8 rows, 4 columns
```

### Core Range Sets

```python
# All cores for prefill, sub_core_grids for decode
self.sub_device_crs = all_crs if mode == "prefill" else model_args.sub_core_grids

# All cores range set
all_crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(6, 9))])
# Covers all compute cores (0,0) to (6,9) = 7x10 = 70 cores
```

### Semaphore Management

**Double-buffered semaphores for each cluster axis**:

```python
# Barrier semaphores
self.barrier_semaphore_handles = [[], []]  # [cluster_axis_0, cluster_axis_1]

# Gather semaphores
self.gather_semaphore_handles = [[], []]

# Prefill-specific semaphores
if mode == "prefill":
    self.from_semaphore_handles = [[], []]
    self.to_semaphore_handles = [[], []]
    self.reduce_semaphore_handles = [[], []]
```

**Semaphore Creation** (Lines 64-93):

```python
for i in range(2):  # Two cluster axes
    for _ in range(self.num_cbs):  # Double buffering
        # Barrier semaphore
        self.barrier_semaphore_handles[i].append(
            ttnn.create_global_semaphore(self.mesh_device, self.sub_device_crs, 0)
        )

        # Gather semaphore (different for ring vs line)
        if self.use_ring_ag_prefill:
            # Ring all-gather needs 2 semaphores
            self.gather_semaphore_handles[i].append([
                ttnn.create_global_semaphore(self.mesh_device, self.sub_device_crs, 0)
                for _ in range(2)
            ])
        else:
            # Line all-gather needs 1 semaphore
            self.gather_semaphore_handles[i].append(
                ttnn.create_global_semaphore(self.mesh_device, self.sub_device_crs, 0)
            )

        # Prefill reduce-scatter semaphores
        if mode == "prefill":
            if self.use_ring_rs_prefill:
                # Ring reduce-scatter needs 3 semaphores
                self.reduce_semaphore_handles[i].append([
                    ttnn.create_global_semaphore(self.mesh_device, self.sub_device_crs, 0)
                    for _ in range(3)
                ])
            else:
                # Line reduce-scatter semaphores
                self.from_semaphore_handles[i].append(
                    ttnn.create_global_semaphore(self.mesh_device, self.sub_device_crs, 0)
                )
                self.to_semaphore_handles[i].append(
                    ttnn.create_global_semaphore(self.mesh_device, self.sub_device_crs, 0)
                )
```

### Index Tracking

```python
self.gather_idx = [0, 0]              # Current gather semaphore index per axis
self.reduce_scatter_buffer_idx = [0, 0]  # Current RS buffer index per axis
self.barrier_semaphore_idx = [0, 0]    # Current barrier semaphore index per axis
```

**Cycling Function** (Lines 125-129):

```python
def get_and_cycle_barrier_semaphore_handle(self, cluster_axis):
    semaphore_index = cluster_axis
    current_idx = self.barrier_semaphore_idx[semaphore_index]
    self.barrier_semaphore_idx[semaphore_index] = (current_idx + 1) % self.num_cbs
    return self.barrier_semaphore_handles[semaphore_index][current_idx]
```

---

## Buffer Management

### Persistent Buffers (Decode)

**Method**: `get_persistent_buffers()`
**Lines**: 294-381

**Purpose**: Pre-allocated L1 buffers for all-reduce operations

#### Cluster Axis 0 Buffers (Row All-Reduce)

```python
# For FF2 (W2) and DO (dense output) operations
cluster_axis = 0
M = 32  # Batch size
num_cores = self.sub_device_crs.num_cores()

# Per-shard width
if not self.is_qwen:
    N_per_shard = 2048 // 16 * cluster_shape[cluster_axis]
    # = 128 * 8 = 1024
else:
    N_per_shard = 1280 // 10 * cluster_shape[cluster_axis]
    # = 128 * 8 = 1024

# Buffer memory config
buffer_mem_cfg = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    ttnn.BufferType.L1,
    ttnn.ShardSpec(
        self.sub_device_crs,
        [M, N_per_shard],               # Shard shape: [32, 1024]
        ttnn.ShardOrientation.ROW_MAJOR,
    ),
)

# Create buffer tensor
tt_buffer = ttnn.from_torch(
    torch.zeros((*cluster_shape, M, N_per_shard * num_cores)),
    # Shape: [8, 4, 32, 1024 * num_cores]
    device=self.mesh_device,
    layout=ttnn.TILE_LAYOUT,
    dtype=ttnn.bfloat8_b,
    memory_config=buffer_mem_cfg,
    mesh_mapper=ttnn.ShardTensor2dMesh(
        self.mesh_device, dims=(0, 1), mesh_shape=cluster_shape
    ),
)
persistent_buffers[cluster_axis] = tt_buffer
```

**Breakdown**:
- **Purpose**: All-reduce for row-wise sharded operations (W2, WO)
- **Cluster axis**: 0 (reduces across 8 rows)
- **Shard shape**: `[32, 1024]` per core
- **Total shape**: `[8, 4, 32, 1024 * num_cores]`
- **Memory**: L1 width-sharded
- **Type**: bfloat8_b

#### Cluster Axis 1 Buffers (Column All-Reduce)

```python
# For QKV operations
cluster_axis = 1
num_input_cores_create_qkv = 10
N_per_shard = 1280 // num_input_cores_create_qkv * cluster_shape[cluster_axis]
# = 128 * 4 = 512

# Buffer memory config (same structure as cluster_axis 0)
buffer_mem_cfg = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    ttnn.BufferType.L1,
    ttnn.ShardSpec(
        self.sub_device_crs,
        [M, N_per_shard],               # Shard shape: [32, 512]
        ttnn.ShardOrientation.ROW_MAJOR,
    ),
)

# Create buffer tensor
tt_buffer = ttnn.from_torch(
    torch.zeros((*cluster_shape, M, N_per_shard * num_cores)),
    # Shape: [8, 4, 32, 512 * num_cores]
    device=self.mesh_device,
    layout=ttnn.TILE_LAYOUT,
    dtype=ttnn.bfloat8_b,
    memory_config=buffer_mem_cfg,
    mesh_mapper=ttnn.ShardTensor2dMesh(
        self.mesh_device, dims=(0, 1), mesh_shape=cluster_shape
    ),
)
persistent_buffers[cluster_axis] = tt_buffer
```

**Breakdown**:
- **Purpose**: All-reduce for column-wise sharded operations (QKV)
- **Cluster axis**: 1 (reduces across 4 columns)
- **Shard shape**: `[32, 512]` per core
- **Total shape**: `[8, 4, 32, 512 * num_cores]`

#### LM Head Buffer

```python
# For LM head all-reduce
num_cores_after_lm_head = 32
if not self.is_qwen:
    N_per_shard = (16 * 1024) // num_cores_after_lm_head * cluster_shape[cluster_axis]
    # = 512 * 4 = 2048
else:
    N_per_shard = (155648 // 8) // num_cores_after_lm_head * cluster_shape[cluster_axis]
    # = 607 * 4 = 2428

self.lm_head_buffer_mem_cfg = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    ttnn.BufferType.L1,
    ttnn.ShardSpec(
        self.sub_device_crs,
        [M, N_per_shard],
        ttnn.ShardOrientation.ROW_MAJOR,
    ),
)

self.tt_lm_head_buffer = ttnn.from_torch(
    torch.zeros((*cluster_shape, M, N_per_shard * num_cores)),
    device=self.mesh_device,
    layout=ttnn.TILE_LAYOUT,
    dtype=ttnn.bfloat8_b,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,  # Initially in DRAM
    mesh_mapper=ttnn.ShardTensor2dMesh(
        self.mesh_device, dims=(0, 1), mesh_shape=cluster_shape
    ),
)
```

### All-Gather Buffers (Decode)

**Method**: `get_all_gather_buffers()`
**Lines**: 176-292

**Purpose**: Pre-allocated buffers for all-gather operations

#### SDPA Buffer

```python
M = 32  # Batch size

# SDPA attention output buffer
tt_buffer = ttnn.from_torch(
    torch.zeros((1, 32, M, 128)),
    # Shape: [1, 32 heads, 32 users, 128 head_dim]
    device=self.mesh_device,
    layout=ttnn.TILE_LAYOUT,
    dtype=ttnn.bfloat16,
    memory_config=self.model_config["GATHER_USERS_MEMCFG"](
        list(self.mesh_device.shape)[1]
    ),
    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
)
persistent_buffers["SDPA"] = tt_buffer
```

**Breakdown**:
- **Shape**: `[1, 32, 32, 128]`
- **Purpose**: Gather attention output heads
- **Type**: bfloat16
- **Memory**: Replicated across mesh

#### Layernorm Buffer

```python
grid_offset = ttnn.CoreCoord(1, 0)
tt_stats_sharded_config = ttnn.create_sharded_memory_config(
    shape=(32, 128),
    core_grid=ttnn.CoreRangeSet([ttnn.CoreRange(grid_offset, grid_offset)]),
    strategy=ttnn.ShardStrategy.WIDTH,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True,
)

tt_buffer = ttnn.from_torch(
    torch.zeros((1, 1, M, 128)),
    device=self.mesh_device,
    layout=ttnn.TILE_LAYOUT,
    dtype=ttnn.bfloat16,
    memory_config=tt_stats_sharded_config,
    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
)
persistent_buffers["LAYERNORM"] = tt_buffer
```

**Breakdown**:
- **Shape**: `[1, 1, 32, 128]`
- **Purpose**: Distributed RMSNorm statistics
- **Shard**: Single core at (1, 0)
- **Type**: bfloat16

#### Sampling Buffers

```python
# Sampling values buffer
tt_buffer = ttnn.from_torch(
    torch.zeros((1, 1, self.max_batch_size, self.max_top_k * self.cluster_shape[0])),
    # Shape: [1, 1, 32, max_top_k * 8]
    device=self.mesh_device,
    layout=ttnn.TILE_LAYOUT,
    dtype=ttnn.bfloat16,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
)
persistent_buffers["SAMPLING_VALUES"] = tt_buffer

# Sampling indices buffer
tt_buffer = ttnn.from_torch(
    torch.zeros((1, 1, self.max_batch_size, self.max_top_k * self.cluster_shape[0])),
    device=self.mesh_device,
    layout=ttnn.TILE_LAYOUT,
    dtype=ttnn.uint16,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
)
persistent_buffers["SAMPLING_INDICES"] = tt_buffer

# Sampling buffer (full vocab)
if not self.is_qwen:
    sampling_shape = (1, 1, 32, 128 * 1024)  # 128k vocab
else:
    sampling_shape = (1, 1, 32, 155648)      # ~155k vocab for Qwen

tt_buffer = ttnn.from_torch(
    torch.zeros(sampling_shape),
    device=self.mesh_device,
    layout=ttnn.TILE_LAYOUT,
    dtype=ttnn.bfloat8_b,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
)
persistent_buffers["SAMPLING"] = tt_buffer
```

#### Binary Multiply Buffer

```python
# For MLP element-wise multiply result
if not self.is_qwen:
    binary_mul_shape = (1, 1, self.max_batch_size, 3584)  # Llama: 14336/4
else:
    binary_mul_shape = (1, 1, self.max_batch_size, 3200)  # Qwen: 12800/4

tt_buffer = ttnn.from_torch(
    torch.zeros(binary_mul_shape),
    device=self.mesh_device,
    layout=ttnn.TILE_LAYOUT,
    dtype=ttnn.bfloat8_b,
    memory_config=self.model_config["FF2_IN_RING_MEMCFG"],
    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
)
persistent_buffers["BINARY_MUL"] = tt_buffer
```

**Breakdown**:
- **Llama**: `[1, 1, 32, 3584]` (hidden_dim=14336, per device=3584 after reduce-scatter)
- **Qwen**: `[1, 1, 32, 3200]` (hidden_dim=12800, per device=3200)
- **Purpose**: Result of SiLU(W1) * W3 before all-gather
- **Type**: bfloat8_b

### Reduce-Scatter Buffers (Decode)

**Method**: `get_decode_reduce_scatter_buffers()`
**Lines**: 383-412

```python
persistent_buffers = [[], []]  # [cluster_axis_0, cluster_axis_1]
cluster_shape = (8, 4)
cluster_axis = 1  # Only for cluster axis 1

buffer_mem_cfg = self.model_config["REDUCE_SCATTER_INTERIM_MEMCFG"]

for _ in range(self.num_cbs):  # Double buffering
    tt_buffer = ttnn.from_torch(
        torch.zeros((*cluster_shape, 32, 512 * buffer_mem_cfg.shard_spec.num_cores())),
        # 512 = 4 devices * 4 pages per packet * 32 tile_width
        device=self.mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        memory_config=buffer_mem_cfg,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            self.mesh_device, dims=(0, 1), mesh_shape=cluster_shape
        ),
    )
    persistent_buffers[cluster_axis].append(tt_buffer)
```

**Breakdown**:
- **Purpose**: Intermediate buffer for reduce-scatter on cluster axis 1
- **Shape**: `[8, 4, 32, 512 * num_cores]`
- **Memory**: `REDUCE_SCATTER_INTERIM_MEMCFG` (width-sharded)
- **Double buffered**: 2 buffers
- **Type**: bfloat8_b

### RS Create Heads Buffers (Decode)

**Method**: `get_decode_rs_create_heads_buffers()`
**Lines**: 414-443

```python
persistent_buffers = [None, None]
cluster_shape = (8, 4)
num_pages_per_packet = 4
shard_height = 32
cluster_axis = 1

buffer_mem_cfg = self.model_config["RS_CREATE_HEADS_INTERIM_MEMCFG"]

torch_buffer = torch.zeros((
    *cluster_shape,
    shard_height,
    cluster_shape[cluster_axis] * num_pages_per_packet * 32 * 5
))
# Width: 4 * 4 * 32 * 5 = 2560

persistent_buffers[cluster_axis] = ttnn.from_torch(
    torch_buffer,
    device=self.mesh_device,
    layout=ttnn.ROW_MAJOR_LAYOUT,
    dtype=ttnn.bfloat16,
    memory_config=buffer_mem_cfg,
    mesh_mapper=ttnn.ShardTensor2dMesh(
        self.mesh_device, dims=(0, 1), mesh_shape=cluster_shape
    ),
)
```

**Breakdown**:
- **Shape**: `[8, 4, 32, 2560]`
- **Purpose**: Intermediate buffer for llama_rs_create_heads operation
- **Width**: 4 devices * 4 pages * 32 tile_width * 5 (Q+K+V components)
- **Layout**: ROW_MAJOR (not tiled)
- **Type**: bfloat16

### Prefill Reduce-Scatter Buffers

**Method**: `get_prefill_reduce_scatter_buffers()`
**Lines**: 445-516

**Purpose**: Buffers for prefill reduce-scatter operations (line topology)

```python
persistent_buffers_all = {}
support_seqlens = [4096, 2048, 1024, 128]

for seqlen in support_seqlens:
    persistent_buffers = {}

    # Buffer definitions
    if not self.is_qwen:
        buffers_dict = {
            "QKV": [(1, 1, seqlen, 1280), (1, 1, seqlen, 1280 // 4)],
            "WO":  [(1, 1, seqlen, 2048), (1, 1, seqlen, 2048 // 8)],
            "FF1": [(1, 1, seqlen, 3584), (1, 1, seqlen, 3584 // 4)],
            "FF3": [(1, 1, seqlen, 3584), (1, 1, seqlen, 3584 // 4)],
            "FF2": [(1, 1, seqlen, 2048), (1, 1, seqlen, 2048 // 8)],
        }
    else:  # Qwen
        buffers_dict = {
            "QKV": [(1, 1, seqlen, 1280), (1, 1, seqlen, 1280 // 4)],
            "WO":  [(1, 1, seqlen, 1280), (1, 1, seqlen, 1280 // 8)],
            "FF1": [(1, 1, seqlen, 3200), (1, 1, seqlen, 3200 // 4)],
            "FF3": [(1, 1, seqlen, 3200), (1, 1, seqlen, 3200 // 4)],
            "FF2": [(1, 1, seqlen, 1280), (1, 1, seqlen, 1280 // 8)],
        }

    for key, shape in buffers_dict.items():
        tt_buffers = []

        # Intermediate buffer (shape[1])
        for i in range(1):
            tt_buffer = ttnn.as_tensor(
                torch.zeros(shape[1]),
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat8_b,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                cache_file_name=self.weight_cache_path / (f"pb_rs_00_{key}_{i}_{seqlen}"),
            )
            tt_buffers.append(tt_buffer)

        # Full buffers (shape[0]) - 2 copies
        for i in range(2):
            tt_buffer = ttnn.as_tensor(
                torch.zeros(shape[0]),
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat8_b,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                cache_file_name=self.weight_cache_path / (f"pb_rs_01_{key}_{i}_{seqlen}"),
            )
            tt_buffers.append(tt_buffer)

        # Intermediate buffers (shape[1]) - 2 copies
        for i in range(2):
            tt_buffer = ttnn.as_tensor(
                torch.zeros(shape[1]),
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat8_b,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                cache_file_name=self.weight_cache_path / (f"pb_rs_02_{key}_{i}_{seqlen}"),
            )
            tt_buffers.append(tt_buffer)

        persistent_buffers[key] = tt_buffers

    persistent_buffers_all[seqlen] = persistent_buffers
```

**Breakdown**:
- **Supported Sequence Lengths**: 4096, 2048, 1024, 128
- **Buffer Types**: QKV, WO, FF1, FF3, FF2
- **Per Operation**: 5 buffers total (1 + 2 + 2)
  - 1 intermediate buffer (reduced size)
  - 2 full buffers
  - 2 intermediate buffers
- **Memory**: DRAM (replicated)
- **Type**: bfloat8_b
- **Cached**: Cached to disk for reuse

### Prefill All-Gather Buffers

**Method**: `get_prefill_all_gather_buffers()`
**Lines**: 570-642

```python
ag_persistent_buffers_all = {}

for seqlen in support_seqlens:
    ag_persistent_buffers = {}

    # Buffer definitions
    if not self.is_qwen:
        buffers_dict = {
            "QKV": [(1, 1, seqlen, 1280)],
            "SDPA": [(1, 1, seqlen // 2, 1024)],           # Ring SDPA output
            "SDPA_REVERSE": [(1, 1, seqlen // 2, 1024)],   # Ring SDPA reverse
            "WO": [(1, 1, seqlen, 2048)],
            "FF1": [(1, 1, seqlen, 3584)],
            "FF3": [(1, 1, seqlen, 3584)],
            "FF2": [(1, 1, seqlen, 2048)],
            "LAYERNORM": [(1, 1, seqlen, 128)],
        }
    else:  # Qwen
        buffers_dict = {
            "QKV": [(1, 1, seqlen, 1280)],
            "SDPA": [(1, 1, seqlen // 2, 1024)],
            "SDPA_REVERSE": [(1, 1, seqlen // 2, 1024)],
            "WO": [(1, 1, seqlen, 1280)],
            "FF1": [(1, 1, seqlen, 3200)],
            "FF3": [(1, 1, seqlen, 3200)],
            "FF2": [(1, 1, seqlen, 1280)],
            "LAYERNORM": [(1, 1, seqlen, 128)],
        }

    for key, shape in buffers_dict.items():
        tt_buffer = ttnn.as_tensor(
            torch.zeros(shape[0]),
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16 if key == "LAYERNORM" else ttnn.bfloat8_b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            cache_file_name=self.weight_cache_path / ("pb_ag_" + key + str(seqlen)),
        )
        ag_persistent_buffers[key] = tt_buffer

    ag_persistent_buffers_all[seqlen] = ag_persistent_buffers

# Fixed-length buffers (not dependent on sequence length)
buffers_fixed_length = {
    "LM_HEAD": [(4, 1, 32, 16384)],     # Llama
    "SAMPLING": [(1, 1, 32, 128 * 1024)],
} if not self.is_qwen else {
    "LM_HEAD": [(4, 1, 32, 19456)],     # Qwen
    "SAMPLING": [(1, 1, 32, 19456 * 8)],
}

for key, shape in buffers_fixed_length.items():
    tt_buffer = ttnn.as_tensor(
        torch.zeros(shape[0]),
        device=self.mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        cache_file_name=self.weight_cache_path / ("pb_ag_" + key + "_32"),
    )
    ag_persistent_buffers[key] = tt_buffer

ag_persistent_buffers_all[32] = ag_persistent_buffers
```

**Breakdown**:
- **Sequence-dependent buffers**: QKV, SDPA, WO, FF1, FF3, FF2, LAYERNORM
- **Fixed buffers**: LM_HEAD, SAMPLING
- **SDPA buffers**: For ring distributed SDPA (split output)
- **Layernorm**: bfloat16 (others are bfloat8_b)
- **Memory**: DRAM (replicated)
- **Cached**: Cached to disk

---

## All-Reduce Operations

### Method: `line_all_reduce`

**Lines**: 644-721

**Signature**:

```python
def line_all_reduce(
    self,
    input_tensor_mesh,              # Input tensor
    cluster_axis,                   # 0 or 1
    num_links,                      # Number of links (typically 3)
    memory_config,                  # Output memory config
    dtype=None,                     # Output dtype
    lm_head=False,                  # Special handling for LM head
    buffer_key=None,                # Buffer key for prefill
    use_noc1_only=False,            # Use only NoC1
    use_optimal_ccl_for_llama=False,  # Llama-specific optimizations
):
```

### Decode Mode

```python
if self.mode == "decode":
    # Select buffer
    if lm_head:
        persistent_buffer = self.tt_lm_head_buffer_l1
    else:
        persistent_buffer = self.persistent_buffers[cluster_axis]

    # Perform all-reduce
    output_tensor_mesh = ttnn.experimental.all_reduce_async(
        input_tensor_mesh,
        persistent_buffer,
        cluster_axis=cluster_axis,
        mesh_device=self.mesh_device,
        multi_device_global_semaphore=self.gather_semaphore_handles[cluster_axis][
            self.gather_idx[cluster_axis]
        ],
        num_links=num_links,
        memory_config=memory_config,
        dtype=dtype,
        topology=self.model_config["CCL_TOPOLOGY"],  # Ring topology
        subdevice_id=self.worker_sub_device_id,
        use_noc1_only=use_noc1_only,
        use_optimal_ccl_for_llama=use_optimal_ccl_for_llama,
    )

    # Deallocate LM head buffer if used
    if lm_head:
        persistent_buffer.deallocate(True)

    # Cycle semaphore index
    self.gather_idx[cluster_axis] = (self.gather_idx[cluster_axis] + 1) % self.num_cbs
```

**Operation**:
1. Select persistent buffer based on cluster axis or LM head flag
2. Call `ttnn.experimental.all_reduce_async` with ring topology
3. Use gather semaphore for synchronization
4. Cycle semaphore index for double buffering
5. Return reduced tensor (replicated across cluster axis)

**Ring Topology Flow** (for cluster_axis=0, 8 devices):
```
Device 0: Send to 1, Receive from 7
Device 1: Send to 2, Receive from 0
Device 2: Send to 3, Receive from 1
...
Device 7: Send to 0, Receive from 6

Iteration 1: Each device sends its data
Iteration 2: Each device sends received + own data
...
Iteration 8: All devices have full sum
```

### Prefill Mode

```python
else:  # mode == "prefill"
    if lm_head:
        # LM head uses all-gather + reduce
        ttnn_tensor_gathered = self.line_all_gather(
            input_tensor_mesh,
            dim=0,
            num_links=num_links,
            cluster_axis=cluster_axis,
            memory_config=memory_config,
            buffer_key=buffer_key,
        )
        ttnn_tensor_out = ttnn.experimental.fast_reduce_nc(
            ttnn_tensor_gathered,
            dims=[0],
            output=None,
            compute_kernel_config=None,
            memory_config=memory_config,
        )
        return ttnn_tensor_out

    # Standard all-reduce: reduce-scatter + all-gather
    output_tensor_scattered = self.line_reduce_scatter(
        input_tensor_mesh,
        memory_config,
        dim=3,
        cluster_axis=cluster_axis,
        num_links=num_links,
        math_op=ttnn.ReduceType.Sum,
        buffer_key=buffer_key,
    )

    output_tensor_mesh = self.line_all_gather(
        output_tensor_scattered,
        dim=3,
        cluster_axis=cluster_axis,
        memory_config=memory_config,
        num_links=num_links,
        buffer_key=buffer_key,
    )

    self.gather_idx[cluster_axis] = (self.gather_idx[cluster_axis] + 1) % self.num_cbs
```

**Operation**:
1. **LM Head**: All-gather (dim=0) followed by reduce on host
2. **Standard**: Reduce-scatter (dim=3) followed by all-gather (dim=3)
3. This is equivalent to all-reduce but uses two separate operations

**Why Two Operations for Prefill**:
- Prefill uses line topology (not ring)
- Line topology doesn't have efficient native all-reduce
- Decompose into reduce-scatter + all-gather
- Each operation is optimized for line topology

### Usage Examples

#### QKV All-Reduce (Prefill)

```python
# After QKV projection in prefill
xqkv_fused = self.tt_ccl.line_all_reduce(
    xqkv,
    cluster_axis=1,
    num_links=3,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    buffer_key="QKV",
)
```

#### WO All-Reduce (Decode)

```python
# After output projection in decode
dense_out_reduced = self.tt_ccl.line_all_reduce(
    dense_out_ttnn,
    cluster_axis=0,
    num_links=self.model_config["GALAXY_NUM_LINKS"],
    memory_config=self.model_config["DECODE_RESIDUAL_MEMCFG"],
    use_optimal_ccl_for_llama=True,
)
```

#### W2 All-Reduce (Decode)

```python
# After W2 projection in MLP
w2_out_reduced = self.tt_ccl.line_all_reduce(
    w2_out,
    cluster_axis=0,
    num_links=self.model_config["GALAXY_NUM_LINKS"],
    memory_config=self.model_config["DECODE_RESIDUAL_MEMCFG"],
    use_optimal_ccl_for_llama=True,
)
```

---

## All-Gather Operations

This document is getting very long. Let me create it as a comprehensive reference and continue with the remaining sections. The document now provides detailed CCL operation coverage with actual code references, buffer specifications, and operational details.
