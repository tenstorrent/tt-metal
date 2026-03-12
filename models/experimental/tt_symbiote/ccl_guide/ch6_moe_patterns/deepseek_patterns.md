# 6.3 DeepSeek Patterns

This section covers `ttnn.experimental.deepseek_minimal_broadcast` and the broader MoE patterns it supports, including DeepSeek-specific architectural differences from standard MoE models.

---

## `ttnn.experimental.deepseek_minimal_broadcast`

Source: `ttnn/cpp/ttnn/operations/experimental/ccl/deepseek_minimal_broadcast/deepseek_minimal_broadcast.hpp`

A broadcast primitive that sends a tensor from one specific device (`sender_coord`) to all other devices along a cluster axis. Unlike `ttnn.broadcast` (Ch2 §2.3) which is a general one-to-all broadcast, `deepseek_minimal_broadcast` is a *minimal* implementation tuned for the DeepSeek MoE use case: small tensors broadcast at high frequency during expert selection.

### API

```python
output = ttnn.experimental.deepseek_minimal_broadcast(
    input_tensor,           # ttnn.Tensor — only the sender device's data is meaningful
    sender_coord,           # MeshCoordinate — coordinate of the sending device
    cluster_axis=None,      # Optional[int] — axis along which to broadcast
    subdevice_id=None,      # Optional[ttnn.SubDeviceId]
    memory_config=None,     # Optional[ttnn.MemoryConfig]
    num_links=1,            # int — Ethernet links (default 1)
    topology=ttnn.Topology.Linear,  # default Linear
)
# Returns: ttnn.Tensor — input_tensor broadcast to all devices along cluster_axis
```

`input_tensor` and `sender_coord` are positional; all remaining parameters are keyword.

### How it differs from `ttnn.broadcast`

| Property | `ttnn.broadcast` (§2.3) | `ttnn.experimental.deepseek_minimal_broadcast` |
|----------|------------------------|------------------------------------------------|
| Namespace | `ttnn.` | `ttnn.experimental.` |
| Topology default | Linear | Linear |
| Semaphore | Uses standard Broadcast infra | "Minimal" implementation — lighter overhead |
| Cluster axis | Optional (Linear requires it) | Optional (asserts if needed by topology) |
| Use case | General one-to-all broadcast | DeepSeek expert routing sync, small tensors at high frequency |
| Returns | Single tensor | Single tensor |

The "minimal" designation means the implementation avoids some of the general-purpose overhead of `ttnn.broadcast` — specifically, it does not build the full `BroadcastProgramFactory` mesh workload; instead it uses a streamlined program suited for the repeated small-tensor broadcast pattern in DeepSeek's routing.

### Operational semantics

Every device in the mesh receives the tensor from `sender_coord`. Non-sender devices provide an `input_tensor` that may contain garbage or zeros — only the sender's data is transmitted and replicated.

The `operation_attributes_t` for this op includes `ring_size` (derived at device operation creation time from the cluster axis length) and `sender_coord`, which the kernel uses to suppress transmission on all devices except the sender and to write incoming data on all devices except the sender.

```python
# Setup: mesh of shape (1, 4); device (0, 0) is the sender
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
sender = ttnn.MeshCoordinate((0, 0))

# The sender device holds the real data; other devices may hold zeros
# The mesh tensor is already on-device, with valid data only at sender_coord

output = ttnn.experimental.deepseek_minimal_broadcast(
    expert_routing_tensor,    # [1, 1, num_experts, num_devices] or similar small tensor
    sender_coord=sender,
    cluster_axis=1,           # broadcast along the device axis
    num_links=1,
    topology=ttnn.Topology.Linear,
)
# After return: all 4 devices hold the same expert_routing_tensor from device (0, 0)
```

> **Gotcha:** `sender_coord` must be a `ttnn.MeshCoordinate` object, not a plain tuple. Passing a tuple `(0, 0)` causes a type error. Construct with `ttnn.MeshCoordinate((row, col))`.

> **Gotcha:** The docstring example shows `ttnn.experimental.broadcast(...)` in the usage snippet — this is a documentation artifact; the actual registered Python name is `ttnn.experimental.deepseek_minimal_broadcast`.

---

## DeepSeek MoE Architecture Overview

DeepSeek-V2 and DeepSeek-V3 use a specific MoE architecture that differs from standard Mixtral-style MoE in several ways that affect CCL strategy:

### Fine-grained expert sharing

Standard MoE (Mixtral): each expert is a full FFN (gate+up+down projection). Expert parameters are non-overlapping across devices.

DeepSeek MoE: uses *fine-grained* expert decomposition — a large number of small experts (e.g., 64 or 128 experts per MoE layer) with K=6 or K=8 selected per token. The increased expert count and top-K increases the probability that multiple selected experts land on the same device — directly enabling `local_reduce=True` in combine.

### Shared experts

DeepSeek models include a set of **shared experts** — a small number of expert networks that receive tokens from every device, regardless of routing. These shared experts act as a stable "catch-all" pathway alongside the sparse routed experts.

The broadcast requirement for shared experts is where `deepseek_minimal_broadcast` fits: the shared expert's routing tensor (which is always the same across the ring) needs to be broadcast from a designated coordinator device to all other devices at the start of each MoE layer.

### DeepSeek MoE dispatch pattern

A DeepSeek MoE layer has a slightly different pipeline than standard MoE:

```
1. Router: compute routing scores → top-K expert indices (routed)
            + constant shared expert indices (always selected)

2. Broadcast shared expert descriptor (deepseek_minimal_broadcast)
   ↓ every device now has the shared expert layout

3. Dispatch (all_to_all_dispatch):
   - Routed tokens: sent to the devices holding selected experts
   - Shared expert tokens: sent to ALL devices (every device runs shared experts)

4. Expert compute:
   - Routed experts: sparse compute on received tokens
   - Shared experts: dense compute on all tokens (no routing needed)

5. Combine (all_to_all_combine):
   - Routed results: returned to originating devices (standard combine)
   - Shared results: already local (no combine needed for shared experts)

6. Weighted sum:
   - Combine routed K contributions + 1 shared contribution per token
```

The `deepseek_minimal_broadcast` in step 2 distributes the shared expert's configuration/routing metadata to all devices so that every device knows which rows in the dispatch output correspond to shared-expert tokens. This broadcast is small (routing metadata only, not the expert weights) and occurs every forward pass.

---

## Optimizing MoE Dispatch for DeepSeek-Style Architectures

### Using `local_reduce=True` in DeepSeek

With 64+ experts and K=8, each device hosts 4+ experts. For any given token, multiple of its 8 selected experts may be on the same device. This makes `local_reduce=True` effective:

```
Standard MoE (K=2, 8 experts, 4 devices):
  Each device has 2 experts. Probability both K experts land on same device = 1/4.
  Expected local reduction opportunity: 25% of tokens.
  → local_reduce=True marginal benefit.

DeepSeek MoE (K=8, 64 experts, 8 devices):
  Each device has 8 experts. Expected experts per token per device ≈ K * E_local / E = 8 * 8/64 = 1.
  Variance is high; many tokens will have 2-3 experts on one device.
  → local_reduce=True saves ~30-50% of combine bandwidth.
```

```python
# DeepSeek-optimized combine:
combined = ttnn.all_to_all_combine(
    expert_outputs_weighted_rm,   # pre-multiply by router score, sum local experts
    expert_metadata,
    expert_mapping,
    local_reduce=True,            # DeepSeek: always use True when K/D > 1
    output_shard_dim=1,
    cluster_axis=cluster_axis,
    topology=ttnn.Topology.Linear,
)
# combined: [1, B/D, S, H] — already a partial sum; only cross-device reduction remains
```

### Batching the shared expert broadcast

`deepseek_minimal_broadcast` is lightweight but still has a per-call overhead. If your model has multiple MoE layers in sequence (e.g., every-other layer is MoE), batch the shared expert descriptor broadcast at the beginning of the forward pass rather than once per layer:

```python
# At the start of the forward pass, broadcast shared expert metadata once
shared_expert_meta = ttnn.experimental.deepseek_minimal_broadcast(
    shared_expert_descriptor,
    sender_coord=ttnn.MeshCoordinate((0, 0)),
    cluster_axis=1,
    topology=ttnn.Topology.Linear,
)

# Then reuse shared_expert_meta in all subsequent MoE layers
for layer in moe_layers:
    layer.forward(tokens, shared_expert_meta, ...)
```

### Expert placement strategy for DeepSeek on multi-dimensional meshes

DeepSeek-V3 is often deployed on 2D meshes (e.g., 8×8 = 64 devices). The expert_mapping tensor and the cluster_axis parameter must be set consistently:

- For tensor-parallel axis (e.g., axis 0): all devices on this axis hold the same expert subset. Use `cluster_axis=0` in dispatch/combine.
- For expert-parallel axis (e.g., axis 1): each device along this axis holds a different expert subset. Use `cluster_axis=1` in dispatch/combine.
- `deepseek_minimal_broadcast` with `cluster_axis=0` broadcasts the shared expert descriptor within each tensor-parallel group.

```python
# 2D mesh: shape (tensor_parallel=8, expert_parallel=8)
# Expert-parallel dispatch along axis 1
sparse_tokens, metadata = ttnn.all_to_all_dispatch(
    tokens,
    expert_indices,
    expert_mapping,      # [1, 1, E, 8] — D[1] = 8 expert-parallel devices
    output_concat_dim=1,
    cluster_axis=1,      # dispatch along expert-parallel axis
    topology=ttnn.Topology.Linear,
)

# Shared expert broadcast within each TP group (axis 0)
shared_meta = ttnn.experimental.deepseek_minimal_broadcast(
    shared_expert_descriptor,
    sender_coord=ttnn.MeshCoordinate((0, col)),   # first device in TP group
    cluster_axis=0,
    topology=ttnn.Topology.Linear,
)
```

> **Gotcha:** When using `deepseek_minimal_broadcast` on a 2D mesh with `cluster_axis=0`, the `sender_coord` must specify only the coordinate along axis 0. All devices sharing the same axis-1 coordinate (i.e., the same expert-parallel rank) will receive the broadcast. Verify the `sender_coord` dimension matches the `cluster_axis`.

---

## Combining DeepSeek Broadcast with Async Overlap

For latency-critical inference, `deepseek_minimal_broadcast` can be dispatched asynchronously before the previous layer's computation completes, overlapping the broadcast latency with compute:

```python
# Dispatch shared expert broadcast early (while previous layer compute runs)
shared_meta_future = ttnn.experimental.deepseek_minimal_broadcast(
    shared_expert_descriptor,
    sender_coord=ttnn.MeshCoordinate((0, 0)),
    cluster_axis=1,
    subdevice_id=ccl_sub_id,   # use CCL sub-device for overlap
    topology=ttnn.Topology.Linear,
)

# Previous layer's matmul (running on compute sub-device simultaneously)
prev_output = ttnn.matmul(prev_input, weight, subdevice_id=compute_sub_id)

# Now both are complete (or wait explicitly)
ttnn.experimental.synchronize_devices(mesh, subdevice_id=ccl_sub_id)

# Use shared_meta_future in current layer's dispatch
sparse_tokens, metadata = ttnn.all_to_all_dispatch(
    prev_output,
    expert_indices,
    expert_mapping,
    cluster_axis=1,
    ...
)
```

This pattern requires a `SubDevice` split as described in [Ch4 §4.1](../ch4_async_overlap/why_async.md#subdevice-partitioning-tensix-cores).

---

## Summary of Ch6 Operations

| Operation | Namespace | Use case | Topology default |
|-----------|-----------|----------|-----------------|
| `ttnn.all_to_all_dispatch` | `ttnn.` | Route tokens to expert devices | `None` (must specify) |
| `ttnn.all_to_all_combine` | `ttnn.` | Collect expert results back to originating devices | `None` (must specify) |
| `ttnn.experimental.deepseek_minimal_broadcast` | `ttnn.experimental.` | Broadcast small shared-expert descriptor | `Linear` |

All three ops require `cluster_axis` to be explicitly specified. All inputs must be Row Major Interleaved. The `expert_mapping_tensor` must be fully replicated across all devices.

---

*Back to [Chapter 6 Index](index.md)*

*Back to [6.2 Dispatch and Combine](dispatch_combine.md)*
