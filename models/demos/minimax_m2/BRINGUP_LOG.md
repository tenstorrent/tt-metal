# MiniMax-M2.5 TTNN Bringup Log

## Target Platform
Galaxy (TG) — mesh device `(8, 4)` = 32 × Wormhole B0 chips

## Parallelism Strategy

| Component | Strategy | Sharding details |
|---|---|---|
| Attention QKV | TP=4, column-parallel | `[H, (NQ+NK+NK)*D]` → `[H, (NQ+NK+NK)*D/TP]` per col device |
| Attention O-proj | TP=4, row-parallel | `[NQ*D, H]` → `[NQ*D/TP, H]` per col device |
| Attention all-reduce | `mesh_config.allreduce` (reduce-scatter + all-gather) | axis=cols (axis=1) |
| QK-norm | Replicated weight, local norm per TP shard | Approximation: norm is over `NQ*D/TP` instead of `NQ*D` |
| Partial RoPE | Local per device | cos/sin replicated; no CCL needed |
| MoE router gate | Replicated `[H, E]` | Selection on CPU, weights on device |
| MoE expert gate/up | EP=8 + TP=4, `dims=(1, -1)` | `[1, E, H, FF]` → `[1, E/EP, H, FF/TP]` per device |
| MoE expert down | EP=8 + TP=4, `dims=(1, -2)` | `[1, E, FF, H]` → `[1, E/EP, FF/TP, H]` per device |
| MoE EP all-reduce | `ttnn.all_reduce` | axis=rows (axis=0) |
| MoE TP all-reduce | `ttnn.all_reduce` | axis=cols (axis=1) |
| Embeddings / norms / lm_head | Replicated | `ReplicateTensorToMesh` |

## Files Changed

| File | Change summary |
|---|---|
| `tt/model_config.py` | Added `make_mesh_config()` using `gpt_oss.MeshConfig`; mesh (8,4), TP=4, EP=8 |
| `tt/rms_norm.py` | Added `mesh_mapper` parameter; defaults to `ReplicateTensorToMesh` for `MeshDevice` |
| `tt/rope.py` | cos/sin replicated via `ReplicateTensorToMesh`; `apply_partial_rope` is local per device |
| `tt/attention.py` | Full rewrite: TP=4 col-parallel QKV + QK-norm + row-parallel O-proj + `apply_allreduce` |
| `tt/moe.py` | Full rewrite: EP=8+TP=4 on-device expert weights; dense batched matmul; EP+TP all-reduce |
| `tt/model.py` | Opens MeshDevice; `CCLManager`; replicated embeddings/norms/lm_head |
| `tests/test_minimax_m2_tt.py` | `device` fixture: `open_mesh_device(8,4)` + `set_fabric_config(FABRIC_1D_RING)`; `tt_to_torch` reads from `device[0]`; 8 test cases |

## Test Results

### Passing ✅

| Test | PCC | Notes |
|---|---|---|
| `test_rmsnorm` | 0.999983 | Replicated norm weight across mesh |
| `test_partial_rope` | Q: ~0.9999, K: ~0.9999 | Replicated cos/sin; local RoPE per device |
| `test_attention` | 0.994625 | TP=4; local QK-norm approximation causes small PCC loss |

### Failing ❌

| Test | PCC | Root cause |
|---|---|---|
| `test_moe` | 0.940 | Dense batched matmul (E_local=32 experts) PCC below 0.99 threshold — under investigation |

## Known Issues & Root Causes

### 1. Fabric must be initialized before opening MeshDevice

**Error:** `TT_FATAL: Trying to get un-initialized fabric context`

**Cause:** `mesh_config.allreduce` (used for attention TP all-reduce) uses
`reduce_scatter_minimal_async` + `all_gather_async`, which require the Ethernet
fabric. Without calling `ttnn.set_fabric_config(FABRIC_1D_RING)` before
`ttnn.open_mesh_device`, all CCL ops that use the fabric fail.

**Fix applied:** Updated the `device` fixture to call `ttnn.set_fabric_config`
with `FABRIC_1D_RING` before `open_mesh_device`, and reset to `DISABLED` on teardown:

```python
ttnn.set_fabric_config(
    ttnn.FabricConfig.FABRIC_1D_RING,
    ttnn.FabricReliabilityMode.STRICT_INIT,
    None,
    ttnn.FabricTensixConfig.DISABLED,
    ttnn.FabricUDMMode.DISABLED,
    ttnn.FabricManagerMode.DEFAULT,
)
mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(8, 4))
```

### 2. CCL all-reduce ops require 4D tensors

**Error:** `TT_THROW: ShapeBase[] index out of range. 3 not in [-4, 3)`

**Cause:** `mesh_config.allreduce` internally calls `reduce_scatter_minimal_async`
with `dim=3`, requiring a 4D tensor. Our attention O-proj output was 3D `[B, S, H]`.

**Fix applied:** Unsqueeze to 4D before `apply_allreduce`, reshape back to 3D after:

```python
out_4d = ttnn.unsqueeze_to_4D(out)
out_4d = apply_allreduce(out_4d, self.mesh_config, self.ccl_manager, H)
out = ttnn.reshape(out_4d, (B, S, H))
```

**Important:** `ttnn.unsqueeze_to_4D` returns a view sharing the same buffer.
Do NOT call `out.deallocate(True)` after unsqueeze — it frees the shared buffer,
causing `TT_FATAL: Buffer must be allocated on device!` on the next access.

### 3. `ttnn.matmul` does NOT support broadcast in batch dims

**Error:** `TT_FATAL: bmm expects input tensors of shapes BCMK*BCKN=BCMN`

**Cause:** Attempting `[1, 1, T, H] × [1, E_local, H, FF]` — TTNN requires exact
batch dim match. Broadcasting (e.g., 1 → E_local) is not supported by `ttnn.matmul`.

**Fix applied:** Use `ttnn.repeat(x_flat, ttnn.Shape([1, E_local, 1, 1]))` to
explicitly expand `x_flat` to `[1, E_local, T, H]` before the matmul.

### 4. MoE PCC = 0.94 (UNDER INVESTIGATION)

**Symptom:** `test_moe` consistently gives PCC ≈ 0.940, below the 0.99 threshold.

**Confirmed not caused by:**
- Routing tensor shape mismatch (T vs T_pad) — fixed by using `T_pad` in `_route`
- Wrong reduction op (`ttnn.sum` vs `fast_reduce_nc`) — both give same PCC
- Wrong weight convention (w1/w2/w3 transpose) — verified against reference

**Leading hypothesis:** Dense batched matmul over E_local=32 experts in bfloat16
introduces accumulated rounding error that the sparse reference avoids. The reference
computes only the 8 selected experts per token, while our dense implementation computes
all 32 local experts and multiplies non-selected ones by 0 routing weight.
Even though `0.0 × finite_value = 0.0` exactly in IEEE 754, possible sources of error:
- `ttnn.repeat` may not produce exact copies on MeshDevice (view semantics unknown)
- `fast_reduce_nc` over 32 values in bfloat16 may accumulate ~6% error

**Next steps to investigate:**
1. Test with CPU-reference routing injected into TTNN (bypass TTNN sigmoid) to isolate routing error vs expert error
2. Test with single expert loop (`ttnn.linear` per expert in Python) to bypass batched matmul
3. Try `ttnn.sparse_matmul` following `gpt_oss` patterns (requires `ttnn.moe_routing_remap`)

## Architecture Notes

### MiniMax-M2.5 Specific

- **QK-norm**: Applied per TP shard (local approximation). Norm is over `NQ*D/TP`
  instead of the full `NQ*D`, so results differ slightly from reference. This
  causes ~0.5% PCC loss in attention (0.9946 vs 1.0).

- **Partial RoPE**: Only first `rotary_dim=64` of `head_dim=128` get rotary embedding;
  remaining 64 are NoPE (no positional encoding). Each TP device applies RoPE
  locally to its head shard.

- **Sigmoid routing with bias**: Router uses sigmoid (not softmax) + additive bias
  `e_score_correction_bias` only for TOP-K selection, not for actual routing weights.
  Routing weights are normalized sigmoid values.

- **SwiGLU**: Standard `silu(gate) * up`, no gpt_oss-style SwiGLU variant (no clamp, no alpha).

### Memory Per Device (estimated)

| Component | Per device | Basis |
|---|---|---|
| Attention weights | ~22 MB × 62 = 1.4 GB | TP=4: QKV [3072, 2048], O [1536, 3072] |
| Expert weights | ~225 MB × 62 = 14 GB | EP+TP: [1,32,3072,384]×3 per layer |
| Embeddings/norms/lm_head | ~1.5 GB | Replicated |
| **Total** | **~17 GB/device** | Fits in 12 GB DRAM? See note |

> **Note:** The 14 GB estimate for expert weights exceeds the 12 GB per-chip DRAM.
> May need FP8 quantization for expert weights, or further EP/TP factoring.
> Actual measurement on hardware pending successful test run.

## Dependencies on gpt_oss

The following are directly imported from `models/demos/gpt_oss`:

| Import | Used for |
|---|---|
| `gpt_oss.config.MeshConfig` | mesh shape, TP/EP axis, `column_parallel`, `row_parallel`, `allreduce` |
| `gpt_oss.config.ModeConfig` | decode/prefill mode config |
| `gpt_oss.tt.ccl.CCLManager` | semaphore management for CCL ops |
| `gpt_oss.tt.attention.operations.apply_allreduce` | TP all-reduce for attention O-proj |
| `gpt_oss.tt.experts.operations.apply_expert_parallel_allreduce` | EP all-reduce for MoE |
| `gpt_oss.tt.experts.operations.apply_tensor_parallel_allreduce` | TP all-reduce for MoE |

## Run Commands

```bash
export ARCH_NAME=wormhole_b0
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)
source python_env/bin/activate

# Individual block tests
pytest models/demos/minimax_m2/tests/test_minimax_m2_tt.py::test_rmsnorm -xvs
pytest models/demos/minimax_m2/tests/test_minimax_m2_tt.py::test_partial_rope -xvs
pytest models/demos/minimax_m2/tests/test_minimax_m2_tt.py::test_attention -xvs
pytest models/demos/minimax_m2/tests/test_minimax_m2_tt.py::test_moe -xvs

# All tests
pytest models/demos/minimax_m2/tests/test_minimax_m2_tt.py -v
```
