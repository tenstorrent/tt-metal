# MoE Prefill Dispatch/Combine Architecture

This document describes the architecture of the `prefill_dispatch` and `prefill_combine` operations for Mixture-of-Experts (MoE) token routing in DeepSeek-V3.

## 1. Overview

The MoE dispatch/combine operations implement expert-parallel token routing across a mesh of chips:

- **Dispatch**: Routes tokens from their original positions to expert-specific buffers distributed across chips
- **Combine**: Reconstructs original token ordering after expert processing

### Key Design Goals

- **Expert-centric buffer organization**: `[chips, experts_per_chip, tokens, hidden]`
- **Dense expert matmuls**: No wasted compute (each expert only processes its routed tokens)
- **Compact memory**: No sparse token arrays
- **Load balancing**: Capacity factor (CF) handles imbalanced routing by allocating CF × expected_load per expert
- **Full metadata tracking**: Round-trip verification from dispatch → experts → combine

## 2. Key Terminology

| Term | Definition |
|------|------------|
| `dispatch_group_size` | Number of chips in each dispatch group (sequence parallel dimension) |
| `num_dispatch_groups` | Number of parallel dispatch groups (expert parallel dimension) |
| `experts_per_chip` | `= num_routed_experts // dispatch_group_size` |
| `expert_dispatch_table` | Maps expert ID → destination chip ID within dispatch axis |
| `metadata_len` | 5 fields: `(src_chip, token_idx, topk_idx, expert_id, weight)` |
| `capacity_factor` | Multiplier for expected load to handle imbalanced routing |
| `max_dispatched_tokens_per_expert` | `= balanced_load * capacity_factor` |
| `balanced_load` | `= dispatch_group_size * seq_len_per_chip * num_experts_per_tok // num_routed_experts` |

## 3. Grid Topologies

### 1D Mesh (e.g., 4×1)

```
MeshDevice(rows=4, cols=1)
┌──────────────┐
│   Dev. 0     │  ← Chip 0: experts 0-3
├──────────────┤
│   Dev. 1     │  ← Chip 1: experts 4-7
├──────────────┤
│   Dev. 2     │  ← Chip 2: experts 8-11
├──────────────┤
│   Dev. 3     │  ← Chip 3: experts 12-15
└──────────────┘
```

- `dispatch_group_size = 4`, `num_dispatch_groups = 1`
- All chips form a single dispatch group
- All 16 experts distributed across 4 chips (4 experts/chip)

### 2D Mesh (e.g., 4×2)

```
MeshDevice(rows=4, cols=2)
┌──────────────────────────────┬──────────────────────────────┐
│          Dev. ID: 4          │          Dev. ID: 6          │
│            (0, 0)            │            (0, 1)            │
│       LinMeshCoord=0         │       LinMeshCoord=1         │
│       LogicalCoord=0         │       LogicalCoord=0         │
│                              │                              │
├──────────────────────────────┼──────────────────────────────┤
│          Dev. ID: 2          │          Dev. ID: 3          │
│            (1, 0)            │            (1, 1)            │
│       LinMeshCoord=2         │       LinMeshCoord=3         │
│       LogicalCoord=1         │       LogicalCoord=1         │
│                              │                              │
├──────────────────────────────┼──────────────────────────────┤
│          Dev. ID: 1          │          Dev. ID: 0          │
│            (2, 0)            │            (2, 1)            │
│       LinMeshCoord=4         │       LinMeshCoord=5         │
│       LogicalCoord=2         │       LogicalCoord=2         │
│                              │                              │
├──────────────────────────────┼──────────────────────────────┤
│          Dev. ID: 5          │          Dev. ID: 7          │
│            (3, 0)            │            (3, 1)            │
│       LinMeshCoord=6         │       LinMeshCoord=7         │
│       LogicalCoord=3         │       LogicalCoord=3         │
│                              │                              │
└──────────────────────────────┴──────────────────────────────┘

Column 0 = Dispatch Group 0     Column 1 = Dispatch Group 1
    (experts 0-7)                   (experts 8-15)
```

- `dispatch_group_size = 4` (rows), `num_dispatch_groups = 2` (columns)
- Each column is an independent dispatch group
- Experts partitioned across dispatch groups:
  - Group 0 (col 0): experts 0-7 → chips 0-3 (2 experts/chip)
  - Group 1 (col 1): experts 8-15 → chips 0-3 (2 experts/chip)

**Coordinate Types:**
- **Dev. ID**: Physical device ID mapping
- **LinMeshCoord**: Used for fabric transfers
- **LogicalCoord**: Coordinate within all-to-all dispatch group

## 4. Dense Buffer Layout

The dispatch operation produces "dense" expert-centric buffers where each expert only sees its routed tokens, enabling efficient matmuls with no wasted compute.

### Dispatched Buffer

```
Shape: (num_dispatch_groups, dispatch_group_size, experts_per_chip,
        max_dispatched_tokens_per_expert, hidden_dim)
```

### Metadata Buffer

```
Shape: (num_dispatch_groups, dispatch_group_size, experts_per_chip,
        max_dispatched_tokens_per_expert, metadata_len)

metadata_len = 5 fields:
  [0]: src_chip      - Source chip ID (0 to dispatch_group_size-1)
  [1]: token_idx     - Token index within source chip's sequence
  [2]: topk_idx      - Which of the K experts this token selected (0 to num_experts_per_tok-1)
  [3]: expert_id     - Global expert ID (0 to num_routed_experts-1)
  [4]: weight        - Router weight (bfloat16 stored as int16)
```

## 5. Input Tensor Sharding/Replication

| Tensor | Shape | Sharding Strategy |
|--------|-------|-------------------|
| `x` (input) | `(dispatch_group_size, seq_len, hidden_dim)` | Shard dim 0 on SP axis, replicate on EP axis |
| `weights` | `(dispatch_group_size, seq_len, num_experts_per_tok)` | Same as x |
| `indices` | `(dispatch_group_size, seq_len, num_experts_per_tok)` | Same as x |
| `expert_offsets` | `(dispatch_group_size, num_routed_experts)` | Shard dim 0 across mesh |
| `expert_dispatch_table` | `(num_dispatch_groups, num_routed_experts)` | Shard dim 0 across EP groups, replicate on SP |

### Sharding Helper Functions

```python
# x, weights, indices: shard on SP axis, replicate on EP
mesh_mapper_replicated = ttnn.ShardTensor2dMesh(
    mesh_device,
    mesh_shape=mesh_device.shape,
    dims=(sp_axis, None),  # e.g., (0, None) for 2D mesh
)

# expert_offsets: shard dim 0 across full mesh
TtDispatchModule.shard_expert_offsets(mesh_device, expert_offsets)

# expert_dispatch_table: shard across groups, replicate on dispatch axis
TtDispatchModule.shard_expert_dispatch_table(mesh_device, expert_dispatch_table, sp_axis)
```

## 6. Expert Placement

Experts are distributed across chips using a hierarchical partitioning:

```python
experts_per_group = num_routed_experts // num_dispatch_groups
experts_per_chip = experts_per_group // dispatch_group_size
local_expert_id = expert_id - group_start
chip_id = local_expert_id // experts_per_chip
```

### Example: 16 experts, 4×2 mesh

```
experts_per_group = 16 // 2 = 8
experts_per_chip = 8 // 4 = 2

Group 0 (col 0): experts 0-7
  Chip 0: experts 0, 1
  Chip 1: experts 2, 3
  Chip 2: experts 4, 5
  Chip 3: experts 6, 7

Group 1 (col 1): experts 8-15
  Chip 0: experts 8, 9
  Chip 1: experts 10, 11
  Chip 2: experts 12, 13
  Chip 3: experts 14, 15
```

### Expert Dispatch Table

Maps global expert ID to destination chip ID within each dispatch group. Value of `-1` indicates expert is not present in that group.

```python
expert_dispatch_table = [
    # Group 0: experts 0-7 → chips 0-3
    [ 0, 0, 1, 1, 2, 2, 3, 3, -1,-1,-1,-1, -1,-1,-1,-1],
    # Group 1: experts 8-15 → chips 0-3
    [-1,-1,-1,-1, -1,-1,-1,-1,  0, 0, 1, 1,  2, 2, 3, 3],
]
# Shape: (num_dispatch_groups=2, num_routed_experts=16)
```

## 7. Concrete Example (linear-2 config)

Test configuration from `test_prefill_dispatch.py`:

```python
# Configuration
mesh_device = (2, 1)  # 2 chips, 1D mesh
seq_len_per_chip = 32
hidden_dim = 7168
num_routed_experts = 16
num_experts_per_tok = 4
capacity_factor = 2
```

### Derived Values

```python
dispatch_group_size = 2
num_dispatch_groups = 1
experts_per_chip = 16 // 2 = 8

# Load balancing
balanced_load = 2 * 32 * 4 // 16 = 16  # tokens per expert under perfect balance
max_dispatched_tokens_per_expert = 16 * 2 = 32  # with capacity_factor=2

metadata_len = 5
```

### Input Tensor Shapes

| Tensor | Shape | Description |
|--------|-------|-------------|
| `x` | `(2, 32, 7168)` | Hidden states: 2 chips × 32 tokens × 7168 hidden |
| `weights` | `(2, 32, 4)` | Router weights: 2 chips × 32 tokens × 4 topk |
| `indices` | `(2, 32, 4)` | Expert indices: 2 chips × 32 tokens × 4 topk |
| `expert_offsets` | `(2, 16)` | Cumulative offsets: 2 chips × 16 experts |
| `expert_dispatch_table` | `(1, 16)` | Dispatch table: 1 group × 16 experts |

### Output Tensor Shapes

| Tensor | Shape | Description |
|--------|-------|-------------|
| `dispatched_buffer` | `(1, 2, 8, 32, 7168)` | Dispatched tokens per expert |
| `dispatched_metadata` | `(1, 2, 8, 32, 5)` | Metadata per dispatched token |
| `combined_output` | `(2, 32, 4, 7168)` | Recombined output |

## 8. Data Flow Diagram

```
═══════════════════════════════════════════════════════════════════════════════
                              DISPATCH OPERATION
═══════════════════════════════════════════════════════════════════════════════

  Input Tensors (per chip)
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  x[chip, token, :]              - Hidden states                         │
  │  weights[chip, token, topk]     - Router weights                        │
  │  indices[chip, token, topk]     - Selected expert IDs                   │
  │  expert_offsets[chip, expert]   - Write offset per expert               │
  └─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  For each (chip, token, topk_idx):                                      │
  │    expert_id = indices[chip, token, topk_idx]                           │
  │    dest_chip = expert_dispatch_table[group, expert_id]                  │
  │    expert_idx = expert_id % experts_per_chip                            │
  │    offset = expert_offsets[chip, expert_id]++                           │
  └─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
  Output Tensors (expert-centric)
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  dispatched_buffer[group, dest_chip, expert_idx, offset, :] = x[...]    │
  │  dispatched_metadata[group, dest_chip, expert_idx, offset, :] =         │
  │      (src_chip, token, topk_idx, expert_id, weight)                     │
  └─────────────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════
                              COMBINE OPERATION
═══════════════════════════════════════════════════════════════════════════════

  Input: Expert-processed buffers (same shape as dispatch output)
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  dispatched_buffer[group, chip, expert, slot, :]                        │
  │  dispatched_metadata[group, chip, expert, slot, :]                      │
  │  expert_token_counts[group, chip, expert]                               │
  └─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  For each (group, chip, expert, slot):                                  │
  │    if slot < expert_token_counts[group, chip, expert]:                  │
  │      (src_chip, token, topk_idx, _, _) = metadata[...]                  │
  │      output[src_chip, token, topk_idx, :] = buffer[...]                 │
  └─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
  Output Tensor (token-centric)
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  output[chip, token, topk_idx, :] - Back to original token ordering     │
  │  Shape: (dispatch_group_size, seq_len_per_chip, num_experts_per_tok,    │
  │          hidden_dim)                                                    │
  └─────────────────────────────────────────────────────────────────────────┘
```

## 9. File Reference

### Python Wrappers (`models/demos/deepseek_v3_d_p/`)

| File | Description |
|------|-------------|
| `tt/moe/tt_dispatch.py` | `TtDispatchModule` - TTNN wrapper for dispatch |
| `tt/moe/tt_combine.py` | `TtCombineModule` - TTNN wrapper for combine |
| `tt/moe/init_helpers.py` | `MeshConfig`, `create_expert_dispatch_table`, `get_gate_outputs` |
| `reference/moe/dispatch.py` | `TorchDispatchModule` - PyTorch reference |
| `reference/moe/combine.py` | `TorchCombineModule` - PyTorch reference |

### TTNN Operations (`ttnn/cpp/ttnn/operations/experimental/deepseek/`)

**prefill_dispatch:**
- `prefill_dispatch/prefill_dispatch.hpp` - Op declaration
- `prefill_dispatch/prefill_dispatch.cpp` - Op implementation
- `prefill_dispatch/device/prefill_dispatch_device_operation.hpp` - Device operation
- `prefill_dispatch/device/prefill_dispatch_program_factory.cpp` - Program factory
- `prefill_dispatch/device/kernels/dataflow/reader_prefill_dispatch.cpp` - Reader kernel
- `prefill_dispatch/device/kernels/dataflow/writer_prefill_dispatch.cpp` - Writer kernel

**prefill_combine:**
- `prefill_combine/prefill_combine.hpp` - Op declaration
- `prefill_combine/prefill_combine.cpp` - Op implementation
- `prefill_combine/device/prefill_combine_device_operation.hpp` - Device operation
- `prefill_combine/device/prefill_combine_program_factory.cpp` - Program factory
- `prefill_combine/device/kernels/dataflow/reader_prefill_combine.cpp` - Reader kernel
- `prefill_combine/device/kernels/dataflow/writer_prefill_combine.cpp` - Writer kernel

## 10. Running Tests

```bash
# Activate environment
source python_env/bin/activate

# Run specific configuration
pytest models/demos/deepseek_v3_d_p/tests/pcc/test_prefill_dispatch.py -k "linear-2" -vvv

# Run all dispatch tests
pytest models/demos/deepseek_v3_d_p/tests/pcc/test_prefill_dispatch.py -vvv

# Run combine tests
pytest models/demos/deepseek_v3_d_p/tests/pcc/test_prefill_combine.py -vvv
```
