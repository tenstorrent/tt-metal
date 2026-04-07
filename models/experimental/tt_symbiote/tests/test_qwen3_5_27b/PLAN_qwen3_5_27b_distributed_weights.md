# Plan: Fix Qwen3.5-27B Distributed Weight Tests on T3K

## Problem Description

The four tests in `test_distributed_weights.py` are failing on T3K (1x8 Wormhole mesh). These tests validate distributed weight sharding (col-sharded across 8 devices) for:

1. `test_attention_distributed_weight_sharding` -- Full attention (GQA) with col-sharded Q/K/V/O projections
2. `test_mlp_distributed_weight_sharding` -- SwiGLU MLP with col-sharded gate/up/down projections
3. `test_gdn_distributed_weight_sharding` -- Gated DeltaNet with col-sharded projections
4. `test_decoder_layer_distributed_forward` -- Full decoder layer combining all above

All tests use `TTNNLinearIReplicatedWColSharded` which shards weight along `dim=-1` (out_features after transposition), then rely on `_maybe_all_gather()` to reconstruct the full tensor before element-wise operations.

## Root Cause Analysis

There are **five distinct root causes** identified, ordered by severity:

### RC-1: GDN `in_proj_a` and `in_proj_b` tile alignment failure (CRITICAL)

**Dimensions:** `in_proj_a` and `in_proj_b` both have `out_features=48` (num_v_heads=48).
After col-sharding across 8 devices: `48 / 8 = 6` elements per device.
TTNN TILE_LAYOUT requires the last dimension to be a multiple of 32.
`6 % 32 != 0` -- this causes a hard failure at weight preprocessing or matmul time.

**Affected tests:** `test_gdn_distributed_weight_sharding`, `test_decoder_layer_distributed_forward`

**Files:**
- `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/qwen35_gated_deltanet.py` (lines 203-207)

### RC-2: Missing `mesh_mapper` for non-weight tensors on mesh device (CRITICAL)

Multiple modules move learned parameters to the mesh device using `ttnn.from_torch(..., device=self.device)` or `ttnn.to_device(tensor, self.device)` **without specifying a `mesh_mapper`**. On a mesh device with 8 chips, this either fails (tensor doesn't know how to distribute) or puts data on only one device.

**Affected modules and tensors:**
- `TTNNQwen35RMSNorm.move_weights_to_device_impl()` -- `tt_weight` (line 64)
- `TTNNQwen35GatedDeltaNet.move_weights_to_device_impl()` -- `tt_conv_taps`, `tt_dt_bias`, `tt_neg_exp_A`, `tt_norm_weight` (lines 268-307)
- `TTNNQwen35FullAttention.move_weights_to_device_impl()` -- `tt_q_norm_weight`, `tt_k_norm_weight` (lines 166-181)
- `TTNNQwen35GatedDeltaNet._reset_decode_state()` -- `conv_states`, `rec_states` via `ttnn.zeros()` (lines 318-334)

All of these need `mesh_mapper=ttnn.ReplicateTensorToMesh(self.device)` for multi-device operation.

**Affected tests:** All four tests

**Files:**
- `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/qwen35_normalization.py`
- `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/qwen35_gated_deltanet.py`
- `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/qwen35_attention.py`

### RC-3: `_maybe_all_gather` missing `cluster_axis` parameter (HIGH)

The `_maybe_all_gather()` method in `TTNNQwen35MLP`, `TTNNQwen35FullAttention`, and `TTNNQwen35GatedDeltaNet` calls `ttnn.all_gather(dim=-1, num_links=1, topology=ttnn.Topology.Linear)` without specifying `cluster_axis`.

Contrast with the working `TTNNLinearIColShardedWAllReduced.forward()` (linear.py line 222) which correctly uses `cluster_axis=1` for T3K's (1, 8) mesh shape. Without `cluster_axis`, `all_gather` may not know which mesh dimension to gather along, leading to incorrect gathering or an error.

**Affected tests:** All four tests (any path that uses col-sharded weights and then gathers)

**Files:**
- `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/qwen35_mlp.py` (line 36)
- `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/qwen35_attention.py` (line 213)
- `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/qwen35_gated_deltanet.py` (line 163)

### RC-4: `_is_distributed` always returns False without explicit CCL setup (MEDIUM)

The `_is_distributed` property checks for `self.device_state.ccl_manager`. While `set_device()` calls `_initialize_module_on_device()` which sets `device_state` when `device.get_num_devices() > 1`, the default `DeviceInit.init_state_impl()` does create a `DistributedConfig` with a `TT_CCL` manager.

However, the `_is_distributed` check is fragile. If `device_state` is not propagated to all child modules (e.g., `TTNNQwen35GatedDeltaNet` creates `conv_states` and `rec_states` dynamically which are NOT TTNNModule children), or if `set_device` doesn't recurse into dynamically-created attributes, the CCL manager may be missing.

This is a secondary issue -- if `_is_distributed` returns False, the `all_gather` is skipped entirely and each device operates on its `1/8th` shard independently, producing garbage.

**Affected tests:** All four tests

### RC-5: MLP SwiGLU architecture mismatch for col-sharded down_proj (MEDIUM)

The MLP forward does:
```
gate_out = gate_proj(x)          # [B, S, intermediate/8] (col-sharded)
gate_out = _maybe_all_gather()   # [B, S, intermediate] (replicated)
up_out = up_proj(x)              # [B, S, intermediate/8] (col-sharded)
up_out = _maybe_all_gather()     # [B, S, intermediate] (replicated)
intermediate = silu(gate_out) * up_out  # replicated
output = down_proj(intermediate)  # replicated input, col-sharded weight
output = _maybe_all_gather()     # replicated
```

The `down_proj` has input `intermediate_size=17408` and output `hidden_size=5120`. Its weight is col-sharded on dim=-1 (out_features). So the input to `down_proj` is replicated (17408) and weight is sharded on output dim. The matmul `[B,S,17408] x [17408, 5120/8]` produces `[B,S,640]` per device, and the all_gather reconstructs `[B,S,5120]`. This is actually correct for this column-parallel pattern.

However, the current architecture does full all_gather after gate_proj and up_proj, which means each device computes the FULL intermediate (17408 activations) only to then use a col-sharded down_proj. This works but is inefficient. The real concern is whether the down_proj weight sharding is correct: `down_proj` weight is `[hidden_size, intermediate_size]` = `[5120, 17408]`. After transposition: `[17408, 5120]`. Sharded on dim=-1: `[17408, 640]` per device. Matmul: `[B,S,17408] x [17408,640] = [B,S,640]`. All-gather along dim=-1: `[B,S,5120]`. This is numerically correct.

**Status:** This is actually correct. Noting for documentation purposes only.

## Step-by-Step Implementation Plan

### Step 1: Fix GDN `in_proj_a`/`in_proj_b` tile alignment (RC-1)

**File:** `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/qwen35_gated_deltanet.py`

**Change:** In `from_torch()`, do NOT use `TTNNLinearIReplicatedWColSharded` for `in_proj_a` and `in_proj_b` because their `out_features=48` is not tile-aligned when divided by 8. Use `TTNNLinear` (replicated) instead.

```python
# in from_torch():
LinearCls = TTNNLinearIReplicatedWColSharded if distributed else TTNNLinear

new_gdn.in_proj_qkv = LinearCls.from_torch(gated_deltanet.in_proj_qkv)  # 10240 / 8 = 1280 OK
new_gdn.in_proj_z = LinearCls.from_torch(gated_deltanet.in_proj_z)      # 6144 / 8 = 768 OK
new_gdn.out_proj = LinearCls.from_torch(gated_deltanet.out_proj)        # 5120 / 8 = 640 OK

# These have out_features=48, 48/8=6 which is NOT tile-aligned (need %32==0)
# Keep them replicated
new_gdn.in_proj_b = TTNNLinear.from_torch(gated_deltanet.in_proj_b)     # 48 NOT divisible into tiles
new_gdn.in_proj_a = TTNNLinear.from_torch(gated_deltanet.in_proj_a)     # 48 NOT divisible into tiles
```

### Step 2: Add `mesh_mapper` for all non-weight tensors (RC-2)

#### Step 2a: Fix `TTNNQwen35RMSNorm`

**File:** `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/qwen35_normalization.py`

In `move_weights_to_device_impl()`, add mesh_mapper for multi-device:

```python
def move_weights_to_device_impl(self):
    if self.device.get_num_devices() > 1:
        weight_torch = ttnn.to_torch(self.tt_weight)
        self.tt_weight = ttnn.from_torch(
            weight_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )
    else:
        self.tt_weight = ttnn.to_device(self.tt_weight, self.device)
```

#### Step 2b: Fix `TTNNQwen35GatedDeltaNet`

**File:** `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/qwen35_gated_deltanet.py`

In `move_weights_to_device_impl()`, add `mesh_mapper=ttnn.ReplicateTensorToMesh(self.device)` to all `ttnn.from_torch()` calls when `self.device.get_num_devices() > 1`. This applies to:
- Conv taps (lines 270-279)
- `tt_dt_bias` (lines 282-289)
- `tt_neg_exp_A` (lines 291-298)
- `tt_norm_weight` (lines 300-307)

In `_reset_decode_state()`, add `mesh_mapper` to `ttnn.zeros()` calls for:
- `conv_states` (lines 318-325)
- `rec_states` (lines 328-333)

Note: `ttnn.zeros()` may not support `mesh_mapper` directly. If not, create via `ttnn.from_torch(torch.zeros(...), ..., mesh_mapper=...)` instead.

#### Step 2c: Fix `TTNNQwen35FullAttention`

**File:** `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/qwen35_attention.py`

In `move_weights_to_device_impl()`, add `mesh_mapper=ttnn.ReplicateTensorToMesh(self.device)` to the `ttnn.from_torch()` calls for `tt_q_norm_weight` (line 168) and `tt_k_norm_weight` (line 176) when `self.device.get_num_devices() > 1`.

### Step 3: Fix `_maybe_all_gather` missing `cluster_axis` (RC-3)

**Files:** All three module files

Add `cluster_axis=1` to all `_maybe_all_gather` implementations:

```python
def _maybe_all_gather(self, tensor):
    if not self._is_distributed:
        return tensor
    gathered = ttnn.all_gather(
        tensor,
        dim=-1,
        num_links=1,
        cluster_axis=1,  # T3K mesh shape is (1, 8)
        topology=ttnn.Topology.Linear,
    )
    ttnn.synchronize_device(self.device)
    return gathered
```

Apply to:
- `qwen35_mlp.py` line 36
- `qwen35_attention.py` line 213
- `qwen35_gated_deltanet.py` line 163

### Step 4: Validate `_is_distributed` propagation (RC-4)

**File:** `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/utils/device_management.py`

Verify that `set_device()` correctly propagates `device_state` to all TTNNModule children. The current code at line 38-39 only sets device_state when `device.get_num_devices() > 1`, which should be correct for T3K.

Add debug logging (temporary) to `_is_distributed` in each module to verify:
```python
@property
def _is_distributed(self):
    has_state = self.device_state is not None
    has_ccl = has_state and hasattr(self.device_state, "ccl_manager") and self.device_state.ccl_manager is not None
    return has_ccl
```

### Step 5: Update test assertions for in_proj_a/in_proj_b

**File:** `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/tests/test_qwen3_5_27b/test_distributed_weights.py`

The `test_gdn_distributed_weight_sharding` test at line 239 asserts `in_proj_qkv` is `TTNNLinearIReplicatedWColSharded`. This is still correct. But if we add similar assertions for `in_proj_a`/`in_proj_b`, they should assert `TTNNLinear` (replicated).

### Step 6: Run tests and validate

Execute on T3K:
```bash
MESH_DEVICE=T3K pytest models/experimental/tt_symbiote/tests/test_qwen3_5_27b/test_distributed_weights.py -v --timeout=600
```

## Success Criteria

1. All four tests pass on T3K (MESH_DEVICE=T3K, 1x8 mesh)
2. Attention PCC >= 0.90 vs PyTorch reference
3. MLP PCC >= 0.90 vs PyTorch reference
4. GDN PCC >= 0.90 vs PyTorch reference
5. Decoder layer output has no NaN/Inf and correct shape
6. No regressions on single-device (N150) tests

## Key Dimensions Reference

| Projection | out_features | per-device (T3K) | Tile-aligned | Sharding Strategy |
|-----------|-------------|-------------------|--------------|-------------------|
| Attention q_proj (gated) | 12288 | 1536 | Yes | Col-sharded |
| Attention k_proj | 1024 | 128 | Yes | Col-sharded |
| Attention v_proj | 1024 | 128 | Yes | Col-sharded |
| Attention o_proj | 5120 | 640 | Yes | Col-sharded |
| MLP gate_proj | 17408 | 2176 | Yes | Col-sharded |
| MLP up_proj | 17408 | 2176 | Yes | Col-sharded |
| MLP down_proj | 5120 | 640 | Yes | Col-sharded |
| GDN in_proj_qkv | 10240 | 1280 | Yes | Col-sharded |
| GDN in_proj_z | 6144 | 768 | Yes | Col-sharded |
| GDN in_proj_b | **48** | **6** | **No** | **Replicated** |
| GDN in_proj_a | **48** | **6** | **No** | **Replicated** |
| GDN out_proj | 5120 | 640 | Yes | Col-sharded |

## Additional Root Cause: ShardTensor2dMesh Inadvertent Sharding (E2E Test)

### RC-6: Position embeddings and attention mask inadvertently sharded by framework default (CRITICAL)

**Problem:** The framework's default `ShardTensor2dMesh(mesh_device, mesh_shape, (0, -1))` config is applied to ALL input tensors passing through `module_run` transforms. This shards position embeddings (cos/sin) and attention masks on `dim=-1`, even though they must be replicated.

**Mechanism:**
1. `TorchTTNNTensor.__new__` calls `get_default_distributed_tensor_config` → returns `ShardTensor2dMesh` for tensors with `shape[-1] % 8 == 0`
2. `to_ttnn` uses the ShardTensor2dMesh `mesh_mapper` → cos/sin with `rotary_dim=64` get sharded to `64/8=8` per device
3. RoPE sees `rotary_dim=8`, pads to 32, TTNN kernel asserts `padded_shape[-1] % 64 == 0` → fails

**Fix:** `TTNNQwen35DecoderLayer.forward()` now all-gathers:
- Position embeddings: `_all_gather_position_embeddings()` with `rotary_dim` threshold
- Attention mask: `_all_gather_if_sharded()` with `hidden_size` threshold

**Files modified:**
- `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/qwen35_decoder_layer.py`

## Missing Research Topics to Report to Team Lead

**Topic: Qwen3.5-27B T3K Distributed Weight Test Failures -- Root Cause Analysis**

This topic has been investigated and resolved as part of this plan. No additional research is needed -- all root causes are code-level issues in the TTNN module implementations. The findings should be captured if the Team Lead deems it useful for future reference.
