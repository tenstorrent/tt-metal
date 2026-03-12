# 3.2 AllToAllDispatch and AllToAllCombine

## Concept: MoE Expert Routing

Mixture-of-Experts (MoE) layers require routing each token to one or more expert networks that may reside on different devices. This is the most asymmetric communication pattern in transformer inference: different tokens go to different experts, and different experts live on different devices. No symmetric collective (AllGather, ReduceScatter, AllReduce) can express this routing efficiently because the data movement is *sparse and token-dependent*.

AllToAllDispatch and AllToAllCombine solve this together as a matched pair:

- **`ttnn.all_to_all_dispatch`**: Sends each token to the device(s) holding its assigned experts. The output is a sparse tensor on each device containing only the tokens routed to that device, with placeholder rows for unrouted positions.

- **`ttnn.all_to_all_combine`**: After the expert computation, sends each expert's output back to the device that originated the token. This is the inverse routing step.

```
MoE forward pass with AllToAllDispatch/Combine:

  Step 1: Router (on each device) computes expert assignments
          expert_indices[token_i] = [expert_2, expert_7]   (top-K experts)

  Step 2: ttnn.all_to_all_dispatch
          Token i goes to device holding expert_2, token i also goes to expert_7's device
          Each device receives its assigned tokens; non-assigned positions are placeholder rows

          Dev 0: [token_3 | token_9 | ??? | ???]    (expert 0,1 here; received tokens 3,9)
          Dev 1: [token_0 | ??? | ??? | ???]         (expert 2,3 here; received token 0)
          Dev 2: [token_1 | token_5 | ??? | ???]     (expert 4,5 here)
          Dev 3: [token_2 | token_4 | token_7 | ???] (expert 6,7 here)

  Step 3: Each device runs its local expert compute on the received tokens

  Step 4: ttnn.all_to_all_combine
          Expert outputs routed back to originating devices
          Results aggregated per token (optionally locally reduced if locally_reduced=True)

          Each device holds combined expert outputs for its original tokens
```

### Tensor dimension conventions (from nanobind docstring)

The dispatch operation uses a specific set of dimension annotations:

```
B = local batch size per device
S = local sequence length per device
H = hidden size
K = selected experts per token (top-K)
D = total number of devices
A = cluster axis to dispatch/combine along
D[A] = number of devices along cluster axis (= D if cluster_axis is None)
E = number of experts per device
T = total tokens per device = B * S
```

---

## Python API: `ttnn.all_to_all_dispatch`

Source: `ttnn/cpp/ttnn/operations/ccl/all_to_all_dispatch/all_to_all_dispatch.hpp`

```python
output_tensor, expert_metadata_tensor = ttnn.all_to_all_dispatch(
    input_tensor,                    # ttnn.Tensor — tokens to dispatch, shape [B, S, 1, H] per device
    expert_indices_tensor,           # ttnn.Tensor — expert rankings per token, shape [B, S, 1, K] per device
    expert_mapping_tensor,           # ttnn.Tensor — one-hot expert-to-device map, shape [1, 1, E, D] replicated
    output_concat_dim=1,             # int — dimension to concat output tokens; 1=batch, 2=sequence
    cluster_axis=None,               # Optional[int] — mesh axis to dispatch along; currently required
    subdevice_id=None,               # Optional[ttnn.SubDeviceId] — sub-device pin
    memory_config=None,              # Optional[ttnn.MemoryConfig] — output layout
    output_tensors=None,             # Optional[Tuple[ttnn.Tensor, ttnn.Tensor]] — preallocated outputs
    num_links=None,                  # Optional[int] — Ethernet links; None = auto
    topology=None,                   # Optional[ttnn.Topology] — Ring or Linear; None = auto
)
# Returns: Tuple[ttnn.Tensor, ttnn.Tensor]
#   [0] output_tensor:          sparse dispatched tokens, shape [1, B*D[A], S, H] (output_concat_dim=1)
#   [1] expert_metadata_tensor: all-gathered expert indices, shape [1, B*D[A], S, K]
```

> **Gotcha:** The `cluster_axis` parameter is currently required despite being typed `Optional`. The nanobind docstring states: "we assert out when it is not specified." Always pass `cluster_axis` explicitly.

### Input tensor requirements

**`input_tensor`**: Shape `[B, S, 1, H]` per device in **Row Major, Interleaved** format. Sharded along either the batch dimension or the sequence dimension so the global shape is `[B*D[A], S, 1, H]` or `[B, S*D[A], 1, H]`. Must be **duplicated** on the non-cluster axis.

**`expert_indices_tensor`**: Shape `[B, S, 1, K]` per device in **Row Major, Interleaved** format. Sharded identically to `input_tensor`. Contains the top-K expert rankings for each token (not absolute expert indices — the mapping from ranking to expert identity is in `expert_mapping_tensor`).

**`expert_mapping_tensor`**: Shape `[1, 1, E, D]` per device, **fully replicated** across all devices. One-hot encoded: row `e` has a `1` in column `d` if expert `e` resides on device `d`, `0` otherwise. Must be identical on all devices.

> **Gotcha:** All three input tensors must be in Row Major format. Tile layout is not supported for AllToAllDispatch. If your pipeline uses Tile layout for matmuls, convert to Row Major before dispatching: `ttnn.to_layout(tensor, ttnn.ROW_MAJOR_LAYOUT)`.

### Output format

**`output_tensor`**: The dispatched tokens. Each device receives all tokens that were routed to any of its experts. Rows corresponding to tokens NOT routed to this device are populated with **garbage placeholder values** — the metadata tensor tracks which rows are valid. Shape `[1, B*D[A], S, H]` when `output_concat_dim=1`.

**`expert_metadata_tensor`**: An all-gathered version of the expert indices, replicated on all devices. Shape `[1, B*D[A], S, K]`. This tensor is the input to `ttnn.all_to_all_combine`.

---

## Python API: `ttnn.all_to_all_combine`

Source: `ttnn/cpp/ttnn/operations/ccl/all_to_all_combine/all_to_all_combine.hpp`

```python
combined_tensor = ttnn.all_to_all_combine(
    input_tensor,                    # ttnn.Tensor — expert outputs after local compute
    expert_metadata_tensor,          # ttnn.Tensor — metadata from all_to_all_dispatch output[1]
    expert_mapping_tensor,           # ttnn.Tensor — same one-hot map used in dispatch
    local_reduce=False,              # bool — True if expert outputs are already locally reduced
    output_shard_dim=1,              # int — dimension to shard combined output; 1=batch, 2=sequence
    cluster_axis=None,               # Optional[int] — mesh axis; currently required
    subdevice_id=None,               # Optional[ttnn.SubDeviceId] — sub-device pin
    memory_config=None,              # Optional[ttnn.MemoryConfig] — output layout
    output_tensor=None,              # Optional[ttnn.Tensor] — preallocated output
    num_links=None,                  # Optional[int] — Ethernet links; None = auto
    topology=None,                   # Optional[ttnn.Topology] — Ring or Linear; None = auto
)
# Returns: ttnn.Tensor — combined tokens
#   Shape [K, B/D[A], S, H] per device if output_shard_dim=1
#   or [K, B, S/D[A], H] per device if output_shard_dim=2
#   Rows are sparsely populated; non-dispatched positions hold placeholder values
```

### Parameter notes

**`local_reduce`** (default `False`): Set to `True` if the expert output tokens corresponding to a dispatched token have already been reduced locally on the device before the combine step. When `False`, the combine operation performs the reduction as tokens arrive from remote devices.

> **Gotcha:** The argument name in Python is `local_reduce`, but the C++ header uses `locally_reduced`. The nanobind binding maps `local_reduce` (Python keyword) → `locally_reduced` (C++ field). Use the Python name `local_reduce` when calling from Python.

**`output_shard_dim`**: Controls how the combined output is distributed. `1` shards along the batch dimension, `2` along the sequence dimension. Must match the `output_concat_dim` used in dispatch.

**`cluster_axis`**: Currently required — same assertion as in dispatch.

---

## Complete MoE Dispatch/Combine Example

```python
import ttnn

# Setup (assume 1×4 mesh, 8 experts across 4 devices = 2 experts per device)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
cluster_axis = 1   # dispatch along column axis

# expert_mapping_tensor: [1, 1, 8, 4] — one-hot, replicated on all devices
# Row e, Col d = 1 if expert e is on device d
# Example: expert 0,1 → dev 0; expert 2,3 → dev 1; expert 4,5 → dev 2; expert 6,7 → dev 3
# (creation not shown — depends on model configuration)

# ------- Step 1: Route tokens to expert devices -------
# input_tensor: [B, S, 1, H] in Row Major Interleaved, sharded on batch
# expert_indices_tensor: [B, S, 1, K] — top-2 expert rankings per token

dispatched_tokens, expert_metadata = ttnn.all_to_all_dispatch(
    input_tensor,
    expert_indices_tensor,
    expert_mapping_tensor,
    cluster_axis=cluster_axis,
    output_concat_dim=1,
    topology=ttnn.Topology.Linear,
)
# dispatched_tokens: [1, B*4, S, H] on each device — sparse, placeholder rows for non-assigned tokens
# expert_metadata: [1, B*4, S, K] on each device — all-gathered expert indices

# ------- Step 2: Run local expert compute -------
# Each device runs its expert FFN on the received tokens
# (rows with placeholder values should be masked or ignored in the compute)
expert_output = run_local_expert_ffn(dispatched_tokens)   # → [B*4, S, 1, H]

# ------- Step 3: Route expert outputs back to originating devices -------
combined_output = ttnn.all_to_all_combine(
    expert_output,
    expert_metadata,
    expert_mapping_tensor,
    cluster_axis=cluster_axis,
    local_reduce=False,           # combine performs the reduction
    output_shard_dim=1,           # shard combined output along batch
    topology=ttnn.Topology.Linear,
)
# combined_output: [K, B/4, S, H] per device — K=top-K, sparsely populated
```

---

## Under the Hood

### AllToAllDispatch kernel structure

The `AllToAllDispatchSparse` program factory builds per-device programs with:

1. **Ternary reader kernel** (`ternary_reader_kernel_id`): reads input tokens, expert indices, and expert mapping to determine routing destinations. Packs token pages into ERISC L1 outbox buffers addressed to specific remote devices.

2. **Binary writer kernel** (`binary_writer_kernel_id`): receives pages from remote devices via the ERISC inbox, writes valid token rows to the output tensor at the correct positions, writes placeholder rows for non-dispatched positions.

The `AllToAllTransferType` enum in `operation_attributes_t` controls the transfer mode:
- **`FullPacket`**: All pages are sent to an intermediate buffer first, then written to output. Higher latency, simpler implementation.
- **`PageByPage`**: Each page is sent directly to the output buffer as it arrives, conserving L1. Lower peak L1 usage.

The program factory selects the transfer type based on available L1 via `detail::get_cb_sizes()`.

Synchronization uses two `GlobalSemaphore`s in `shared_variables_t`:
- `init_semaphore`: ensures all devices have initialized their buffers before any transfer begins
- `cross_device_semaphore`: signals completion of cross-device transfers for each round

### AllToAllCombine kernel structure

The `AllToAllCombineFromSparse` factory mirrors the dispatch structure. The writer kernel (`unary_writer_kernel_id`) performs element-wise addition as rows arrive when `locally_reduced=False`. The same two `GlobalSemaphore` types (`init_semaphore`, `cross_device_semaphore`) synchronize the operation.

### Program caching

Both operations cache on `(axis, topology, num_links, memory_config, output_concat_dim/output_shard_dim)`. The `worker_core_range_set` is computed from the `subdevice_id` at program creation and is included in the cache key. Changing tensor shapes or expert count between calls forces recompilation.

---

## Memory Layout Considerations

### Row Major requirement

AllToAllDispatch and AllToAllCombine require Row Major Interleaved inputs. This is a hard constraint — the reader kernels use page-by-page transfer semantics that assume row-contiguous layout.

```python
# Convert from Tile layout (used for matmuls) to Row Major before dispatch
input_rm = ttnn.to_layout(tile_layout_tensor, ttnn.ROW_MAJOR_LAYOUT)
expert_indices_rm = ttnn.to_layout(expert_indices, ttnn.ROW_MAJOR_LAYOUT)

dispatched, metadata = ttnn.all_to_all_dispatch(
    input_rm, expert_indices_rm, expert_mapping, cluster_axis=1
)
```

### Placeholder rows and sparse output

The output tensor of AllToAllDispatch is **sparse**: rows for tokens not routed to a device contain garbage values. Do not apply operations that aggregate over all rows (e.g., mean pooling, layer norm over the full sequence) to the raw dispatched output without masking. Use the metadata tensor to identify valid rows.

### L1 pressure

The transfer type (`FullPacket` vs `PageByPage`) is selected automatically based on available ERISC L1. If you observe OOM errors during AllToAllDispatch, L1 exhaustion in the EDM buffers is a likely cause. Reducing `num_links` decreases ERISC L1 consumption per dispatch call.

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| Assertion failure: `cluster_axis` not specified | `cluster_axis=None` (the default) is not supported; operation asserts | Always pass `cluster_axis` explicitly |
| `TypeError` on `input_tensor` | Tensor is in Tile layout instead of Row Major | Convert: `ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)` |
| Garbage values in combined output | `local_reduce=False` but expert outputs were already pre-summed | Set `local_reduce=True` |
| Shape error in `all_to_all_combine` | `output_shard_dim` does not match `output_concat_dim` from dispatch | Ensure both sides use the same dimension (1 or 2) |
| `expert_metadata_tensor` mismatch | Passing wrong tensor or wrong device count | Pass the second element of `all_to_all_dispatch`'s return tuple directly |
| OOM in ERISC L1 | `num_links` too high; EDM buffers exceed ERISC L1 | Reduce `num_links` to 1 |

---

*Back to [Chapter 3 Index](index.md)*

*Next: [3.3 ReduceToRoot and MeshPartition](reduce_to_root.md)*
