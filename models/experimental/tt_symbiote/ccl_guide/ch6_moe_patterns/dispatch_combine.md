# 6.2 Dispatch and Combine

This section covers the full dispatch→expert compute→combine pipeline in detail. For the basic API introduction see [Ch3 §3.2](../ch3_intermediate_operations/all_to_all.md); this chapter focuses on practical MoE patterns, the `local_reduce` flag, load imbalance handling, and a full worked example.

---

## `ttnn.all_to_all_dispatch`

Source: `ttnn/cpp/ttnn/operations/ccl/all_to_all_dispatch/all_to_all_dispatch.hpp`

Registered under `ttnn.all_to_all_dispatch` (not `ttnn.experimental`).

### API

```python
sparse_tokens, expert_metadata = ttnn.all_to_all_dispatch(
    input_tensor,                  # ttnn.Tensor [B, S, 1, H] — Row Major Interleaved
    expert_indices_tensor,         # ttnn.Tensor [B, S, 1, K] — Row Major Interleaved
    expert_mapping_tensor,         # ttnn.Tensor [1, 1, E, D] — Row Major Interleaved, fully replicated
    output_concat_dim=1,           # int — 1=concat along batch, 2=concat along sequence
    cluster_axis=None,             # Optional[int] — ASSERTS if not specified
    subdevice_id=None,             # Optional[ttnn.SubDeviceId]
    memory_config=None,            # Optional[ttnn.MemoryConfig]
    output_tensors=None,           # Optional[Tuple[ttnn.Tensor, ttnn.Tensor]] — pre-allocated outputs
    num_links=None,                # Optional[int] — auto if None
    topology=None,                 # Optional[ttnn.Topology] — must match mesh Fabric init
)
# Returns: Tuple[ttnn.Tensor, ttnn.Tensor]
#   [0] sparse_tokens:    [1, B×D[A], S, H]  — sparsely populated; placeholder rows for unrouted tokens
#   [1] expert_metadata:  [1, B×D[A], S, K]  — gathered expert indices; drive routing in combine
```

### Parameter notes

**`cluster_axis`**: Despite being `Optional[int]` in the C++ signature, the nanobind binding asserts if it is `None`. Always pass an explicit `cluster_axis` value. For a 1D mesh, use `cluster_axis=0`.

**`output_concat_dim`**: Default 1. Determines whether the dispatched token dimension grows along batch (dim 1) or sequence (dim 2).
- Use `output_concat_dim=1` when your model shards batch across devices.
- Use `output_concat_dim=2` when your model shards sequence across devices.
- Must match how `expert_indices_tensor` was sharded.

**`topology`**: Must match the Fabric topology the mesh was initialized with. Do not leave as `None` unless you have confirmed the default matches your init. An incorrect topology silently misconfigures the EDM.

**`output_tensors`**: Optional pre-allocated output pair `(sparse_tokens_buf, metadata_buf)`. Use this to avoid re-allocation between inference steps (analogous to `persistent_output_buffer` in async ops).

> **Gotcha:** `cluster_axis` is not optional in practice. The nanobind binding raises an assertion error at dispatch time (not Python time) if `cluster_axis=None`. Pass an integer; do not rely on `None` auto-selection.

> **Gotcha:** All three input tensors must be Row Major Interleaved. TILE layout and sharded layouts are not supported. Convert with `ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)` and `ttnn.to_memory_config(x, ttnn.DRAM_INTERLEAVED_MEMORY_CONFIG)` before dispatch.

### Output structure: sparse tokens

The `sparse_tokens` output has shape `[1, B×D[A], S, H]` per device (when `output_concat_dim=1`). Reading this shape:
- Dimension 0: always 1 (unused batch outer)
- Dimension 1: `B×D[A]` — the full batch across all devices on the cluster axis, concatenated
- Dimension 2: `S` — sequence length
- Dimension 3: `H` — hidden size

Each row `[1, i, j, :]` is either:
- A real token (if token at global position `(i, j)` selected an expert on this device), or
- A placeholder row with garbage data (if that token did not select any expert on this device)

The `expert_metadata` tensor (shape `[1, B×D[A], S, K]`) tells you which case: row `(i, j)` in metadata contains the expert indices for token `(i, j)`. If none of those K indices map (via `expert_mapping_tensor`) to the current device, the corresponding row in `sparse_tokens` is a placeholder.

---

## `ttnn.all_to_all_combine`

Source: `ttnn/cpp/ttnn/operations/ccl/all_to_all_combine/all_to_all_combine.hpp`

Registered under `ttnn.all_to_all_combine` (not `ttnn.experimental`).

### API

```python
combined_output = ttnn.all_to_all_combine(
    input_tensor,                  # ttnn.Tensor — post-expert sparse results, Row Major Interleaved
    expert_metadata_tensor,        # ttnn.Tensor — output[1] from all_to_all_dispatch, unmodified
    expert_mapping_tensor,         # ttnn.Tensor [1, 1, E, D] — same tensor used in dispatch
    local_reduce=False,            # bool — True if expert outputs already locally summed
    output_shard_dim=1,            # int — dimension to shard the combined output (1=batch, 2=seq)
    cluster_axis=None,             # Optional[int] — ASSERTS if not specified
    subdevice_id=None,             # Optional[ttnn.SubDeviceId]
    memory_config=None,            # Optional[ttnn.MemoryConfig]
    output_tensor=None,            # Optional[ttnn.Tensor] — pre-allocated output
    num_links=None,                # Optional[int]
    topology=None,                 # Optional[ttnn.Topology].noconvert()
)
# Returns: ttnn.Tensor
#   Shape: [K, B/D[A], S, H]  (if output_shard_dim=1, locally_reduced=False)
#   Sparsely populated — each row is either a real result or a placeholder
```

### Parameter notes

**`local_reduce`** (Python keyword, maps to C++ `locally_reduced`): This is the most consequential parameter choice in combine. See the dedicated section below.

**`output_shard_dim`**: Default 1. Controls how the combined output is distributed:
- `output_shard_dim=1`: output sharded along batch — each device holds `B/D[A]` rows of the full batch
- `output_shard_dim=2`: output sharded along sequence — each device holds `S/D[A]` tokens of each batch element

Use the same dimension as `output_concat_dim` in dispatch to maintain consistent tensor topology.

**`topology`**: Registered with `.noconvert()` in nanobind — must be a `ttnn.Topology` enum value, not a string or int.

> **Gotcha:** `cluster_axis` is required (asserts if None), same as dispatch. Always pass explicitly.

> **Gotcha:** The `input_tensor` to combine must be in Row Major Interleaved format — the same constraint as dispatch. Expert FFN outputs are typically Tile layout; convert before calling combine.

---

## The `local_reduce` Flag

`local_reduce` is the most frequently misunderstood parameter. It controls whether the combine operation expects its input to already contain locally-reduced expert contributions.

### `local_reduce=False` (default): K separate results per token

When `local_reduce=False`, the combine operation collects K separate expert outputs for each token (one from each of the K selected experts across the ring). The returned tensor has shape `[K, B/D[A], S, H]` — the leading K dimension holds one result per expert contribution, unnormalized.

The caller is then responsible for the weighted sum:
```python
# After combine with local_reduce=False:
# combined: [K, B/D[A], S, H]
# router_weights: [B/D[A], S, 1, K] — router softmax scores
token_output = (combined * router_weights.permute(3, 0, 1, 2)).sum(dim=0)
```

Use this mode when you need to inspect or log individual expert contributions, or when the weighted sum must be deferred to a later point in the compute graph.

### `local_reduce=True`: single pre-reduced result per token

When `local_reduce=True`, the combine operation expects that the expert computation on each device has already multiplied by the router weight and summed across the local experts on that device. The returned tensor has shape `[1, B/D[A], S, H]` — each row is the pre-weighted partial sum.

The remaining combine across devices then reduces these partial sums:
```python
# Expert computation (between dispatch and combine):
# For each token routed to this device, multiply result by router_score
# and sum across the local experts_per_device experts

# After combine with local_reduce=True:
# combined: [1, B/D[A], S, H] — partial sums already weighted
# Only need to sum across the K/D[A] contributions from different devices
```

Use this mode for maximum efficiency: it reduces the output tensor dimension from K to 1, roughly K× less data in the combine collective.

### Decision table

| Condition | Use `local_reduce` |
|-----------|-------------------|
| K=1 (only one expert per token) | `False` — only one contribution, no local reduction possible |
| K>1, all K experts on different devices | `False` — no local reduction opportunity |
| K>1, some devices host multiple selected experts for a token | `True` — pre-sum those contributions before combine |
| Debugging / need individual expert outputs | `False` |
| Production / maximize throughput | `True` when K > D[A] (each device handles K/D[A] > 1 experts per token on average) |

---

For the kernel implementation details (`AllToAllDispatchSparse`, `AllToAllTransferType`, `AllToAllCombineFromSparse`, `GlobalSemaphore` types), see [Ch3 §3.2 — Under the Hood](../ch3_intermediate_operations/all_to_all.md#under-the-hood).

---

## Full Worked Example: MoE Layer Forward Pass

This example shows a complete MoE layer with 8 experts across 4 devices (2 experts per device), top-2 routing (K=2), batch=4, seq=1, hidden=4096.

```python
import ttnn
import torch

# Configuration
num_devices = 4
num_experts = 8  # 2 per device
experts_per_device = num_experts // num_devices
K = 2            # top-K experts per token
B = 1            # batch per device
S = 1            # sequence per device
H = 4096         # hidden size
cluster_axis = 1

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, num_devices))

# ─── Setup: expert mapping tensor ─────────────────────────────────────────────
# Expert e is on device e // experts_per_device
# expert_mapping[e, d] = 1 iff expert e is on device d
expert_mapping_np = torch.zeros(1, 1, num_experts, num_devices, dtype=torch.bfloat16)
for e in range(num_experts):
    device_for_expert = e // experts_per_device
    expert_mapping_np[0, 0, e, device_for_expert] = 1.0

expert_mapping = ttnn.from_torch(
    expert_mapping_np,
    dtype=ttnn.bfloat16,
    layout=ttnn.ROW_MAJOR_LAYOUT,
    device=mesh,
    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)

# ─── Stage 1: Local router (not shown — produces expert_indices per device) ──
# expert_indices: [B, S, 1, K] per device, Row Major Interleaved
# Values are expert IDs (0..num_experts-1) selected by the router

# ─── Stage 2: Dispatch ────────────────────────────────────────────────────────
# Ensure inputs are Row Major Interleaved
tokens_rm = ttnn.to_layout(token_embeddings, ttnn.ROW_MAJOR_LAYOUT)
indices_rm = ttnn.to_layout(expert_indices, ttnn.ROW_MAJOR_LAYOUT)

sparse_tokens, expert_metadata = ttnn.all_to_all_dispatch(
    tokens_rm,
    indices_rm,
    expert_mapping,
    output_concat_dim=1,     # concatenate along batch dimension
    cluster_axis=cluster_axis,
    num_links=1,
    topology=ttnn.Topology.Linear,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
# sparse_tokens:   [1, B*num_devices, S, H] = [1, 4, 1, 4096] per device
# expert_metadata: [1, B*num_devices, S, K] = [1, 4, 1, 2]   per device

# ─── Stage 3: Expert computation ─────────────────────────────────────────────
# Each device runs its local experts on the rows routed to it.
# Rows not routed to this device are placeholders (garbage) — skip them.
# In practice, use expert_metadata + expert_mapping to build a row mask.

# Convert to Tile layout for expert FFN compute
sparse_tokens_tile = ttnn.to_layout(sparse_tokens, ttnn.TILE_LAYOUT)

# Run local expert FFNs (pseudo-code: replace with actual expert op)
# expert_output_tile: [1, B*num_devices, S, H] — results for locally-routed tokens
#                     placeholder rows for tokens not routed here
expert_outputs_tile = run_local_experts(sparse_tokens_tile, expert_metadata, expert_mapping)

# If using local_reduce=True, multiply by router scores and sum local experts here
# router_scores_for_local: [1, B*num_devices, S, 1] — scores for the locally active expert
# expert_outputs_weighted = expert_outputs_tile * router_scores_for_local

# Convert back to Row Major Interleaved for combine
expert_outputs_rm = ttnn.to_layout(expert_outputs_tile, ttnn.ROW_MAJOR_LAYOUT)
expert_outputs_rm = ttnn.to_memory_config(expert_outputs_rm, ttnn.DRAM_MEMORY_CONFIG)

# ─── Stage 4: Combine ────────────────────────────────────────────────────────
combined = ttnn.all_to_all_combine(
    expert_outputs_rm,
    expert_metadata,          # unmodified output from dispatch
    expert_mapping,           # same mapping tensor
    local_reduce=False,       # False: return K separate results (shape [K, B, S, H])
    output_shard_dim=1,       # shard combined output along batch
    cluster_axis=cluster_axis,
    num_links=1,
    topology=ttnn.Topology.Linear,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
# combined: [K, B, S, H] = [2, 1, 1, 4096] per device

# ─── Stage 5: Weighted sum ──────────────────────────────────────────────────
# Router scores: [B, S, 1, K] per device (softmax of top-K logits)
# Reshape and multiply
combined_tile = ttnn.to_layout(combined, ttnn.TILE_LAYOUT)
# [K, B, S, H] * [B, S, 1, K].permute → weighted sum → [B, S, 1, H]
token_output = weighted_sum(combined_tile, router_scores)
```

---

## Handling Variable Token Counts (Load Imbalance)

In practice, the number of tokens routed to each device varies per forward step. The dispatch operation allocates output tensor rows assuming a worst-case full batch (`B × D[A]` rows), so the sparse output is always a fixed shape regardless of actual routing density.

### Masking placeholder rows before expert compute

To avoid computing on placeholder rows, derive a boolean mask from `expert_metadata` and `expert_mapping`:

```python
def get_active_mask(expert_metadata, expert_mapping, device_id, num_experts_per_device):
    """
    Returns a [1, B*D, S, 1] boolean mask: True for rows routed to this device.
    expert_metadata: [1, B*D, S, K]
    expert_mapping:  [1, 1, E, D]
    """
    # For each token row, check if any of its K expert indices maps to device_id
    # This is a local tensor operation — no CCL needed
    # expert_metadata[:, :, :, k] in range [device_id * E_local, (device_id+1) * E_local)
    local_expert_start = device_id * num_experts_per_device
    local_expert_end   = local_expert_start + num_experts_per_device
    meta = ttnn.to_torch(expert_metadata)   # bring to CPU for mask computation if needed
    active = ((meta >= local_expert_start) & (meta < local_expert_end)).any(dim=-1, keepdim=True)
    return active  # [1, B*D, S, 1] boolean
```

For large batches, derive the mask once and apply it as a multiply to zero-out placeholder rows before passing through the expert FFN. This avoids wasted compute on garbage data.

### Capacity factor

If your training framework imposes an expert capacity factor C (each expert processes at most C × T/N tokens), you can cap the dispatch output size:

```python
max_tokens_per_expert = int(capacity_factor * T / num_experts)
# pre-allocate output tensors sized to max_tokens_per_expert rows
out_sparse = ttnn.allocate_tensor_on_device(
    ttnn.Shape([1, max_tokens_per_expert * num_devices, S, H]),
    ...
)
sparse_tokens, expert_metadata = ttnn.all_to_all_dispatch(
    ...,
    output_tensors=(out_sparse, out_meta),  # use pre-allocated buffers
)
```

Tokens beyond the capacity are discarded. The placeholder mechanism ensures the tensor shape remains valid even when fewer than `max_tokens_per_expert × num_devices` tokens are dispatched.

---

## Performance Considerations

### Batch size

AllToAllDispatch and Combine are most efficient at larger batch sizes because:
- Per-token overhead (metadata, routing logic) amortizes over more tokens
- ERISC link utilization improves when there is more data to transfer per call
- The fixed overhead of the init_semaphore synchronization is amortized

For batch=1 (single-token decode), the MoE all-to-all may dominate inference latency. Consider:
- Using fewer experts per token (K=1 instead of K=2) for decode-only passes
- Caching expert routing decisions across decode steps when the input changes slowly
- Running expert computation in a speculative/early-exit mode

### `num_links`

For small tensors (H < 2048 or B × S < 16), `num_links=1` is typically sufficient — the transfer completes in one ERISC round. For larger hidden sizes or batch sizes, `num_links=2` or `num_links=4` can improve throughput proportionally.

### Topology

Use `ttnn.Topology.Linear` for a 1D device mesh (most common MoE deployment). Use `ttnn.Topology.Ring` when devices are arranged in a physical ring and you want bidirectional transfer. The topology must match the Fabric initialization — mixing them causes silent malfunction.

### `output_concat_dim` and layout compatibility

Choose `output_concat_dim` to align with how your model shards data. If the batch dimension is sequence-parallel across a 2D mesh, `output_concat_dim=2` (sequence-based concat) avoids a costly transpose after dispatch.

---

*Back to [Chapter 6 Index](index.md)*

*Next: [6.3 DeepSeek Patterns](deepseek_patterns.md)*
