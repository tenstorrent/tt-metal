# OLMo Prefill QK-Norm Optimization Plan (Approach B)

## Problem

OLMo's prefill QK-norm consumes **31% of per-layer time (620 us/layer)** due to unnecessary
data movement. The current flow creates Q/K heads first, then must transpose+reshape them
back into a flat layout for global norm, then reshape+transpose again afterward:

```
Current Q-norm path (9 steps):
  nlp_create_qkv_heads → Q[1,5,S,128]      ~14 us
  typecast Q bf8→bf16                         ~4 us
  transpose(1,2) → [1,S,5,128]              ~13 us   ← unnecessary
  reshape → [1,1,S,640]                     ~50 us   ← unnecessary
  to_layout(TILE)                            (included)
  rms_norm_pre_all_gather → stats            ~14 us
  all_gather(stats, axis=0)                  ~50 us
  rms_norm_post_all_gather(weight)           ~25 us
  reshape → [1,S,5,128]                     ~50 us   ← kept (needed)
  transpose(1,2) → [1,5,S,128]              ~13 us   ← kept (needed)
```

## Solution

Move QK-norm to operate on the fused QKV tensor BEFORE `nlp_create_qkv_heads`. After
`line_all_reduce`, `xqkv_fused` is `[1,1,S,896]` in TILE layout with Q occupying the
first 640 elements (tile-aligned: 640/32=20 tiles) and K the next 128 (4 tiles).

Slice Q and K directly from fused tensor → norm them in their natural flat shape → then
split Q into heads for RoPE.

```
Optimized Q-norm path (7 steps):
  slice Q from fused → [1,1,S,640]           ~1 us   (new, cheap)
  typecast Q bf8→bf16                         ~4 us
  rms_norm_pre_all_gather → stats            ~14 us
  all_gather(stats, axis=0)                  ~50 us
  rms_norm_post_all_gather(weight)           ~25 us
  reshape → [1,S,5,128]                     ~50 us   (same as before)
  transpose(1,2) → [1,5,S,128]              ~13 us   (same as before)
```

## Estimated Savings

Per layer: skip 1 transpose (13 us) + 1 reshape (50 us) + nlp_create_qkv_heads (14 us),
add 3 slices (~3 us) = **~74 us/layer**.

For 64 layers: **~4.7 ms** saved on prefill.

At longer sequences (1024+, 4096+) the reshape/transpose scale with seq_len so savings
grow proportionally.

## PCC Gate

Baseline (must match after changes):
- 1L prefill: hidden state PCC ≥ 0.999, logits PCC ≥ 0.999
- 64L prefill: hidden state PCC ≥ 0.977, logits PCC ≥ 0.943

```bash
pytest models/demos/llama3_70b_galaxy/tests/test_olmo_e2e_pcc.py::TestOlmoE2EPCC::test_prefill_pcc_1layer -xvs
```

## Implementation

### Files Changed

1. `models/demos/llama3_70b_galaxy/tt/llama_attention.py` — `forward_prefill` method

### Task 1: Restructure Q-norm to operate pre-split

**Before (lines 1228-1285):**
```python
# 1. Split fused QKV into heads
(q_heads, k_heads, v_heads) = ttnn.experimental.nlp_create_qkv_heads(xqkv_fused, ...)

# 2. Typecast Q
q_heads = ttnn.typecast(q_heads, dtype=ttnn.bfloat16)

# 3. OLMo Q-norm: transpose → reshape → norm → reshape → transpose
q_transposed = ttnn.transpose(q_heads, 1, 2)              # [1,S,5,128]
q_flat = ttnn.reshape(q_transposed, [1,1,S,640])           # flatten
q_flat = ttnn.to_layout(q_flat, ttnn.TILE_LAYOUT)
q_stats = ttnn.rms_norm_pre_all_gather(q_flat, ...)
q_stats_g = self._olmo_qk_norm_all_gather(q_stats, cluster_axis=0)
q_normed = ttnn.rms_norm_post_all_gather(q_flat, q_stats_g, weight=q_norm_weight, ...)
q_unflat = ttnn.reshape(q_normed, [1,S,5,128])
q_heads = ttnn.transpose(q_unflat, 1, 2)                   # [1,5,S,128]
```

**After:**
```python
# 1. Slice Q, K, V from fused tensor (tile-aligned boundaries)
q_dim = self.n_local_heads * self.head_dim       # 640
k_dim = self.n_local_kv_heads * self.head_dim    # 128
q_fused = xqkv_fused[:, :, :, :q_dim]                      # [1,1,S,640] TILE
k_fused = xqkv_fused[:, :, :, q_dim:q_dim+k_dim]           # [1,1,S,128] TILE
v_heads_1VSD = xqkv_fused[:, :, :, q_dim+k_dim:]           # [1,1,S,128] TILE
ttnn.deallocate(xqkv_fused)

# 2. Typecast Q to bf16 (already in TILE layout from matmul→all_reduce)
if q_fused.dtype != ttnn.bfloat16:
    q_fused_bf8 = q_fused
    q_fused = ttnn.typecast(q_fused, dtype=ttnn.bfloat16)
    ttnn.deallocate(q_fused_bf8)

# 3. Q-norm directly on [1,1,S,640] — no transpose/reshape needed!
q_stats = ttnn.rms_norm_pre_all_gather(q_fused, ..., dtype=ttnn.bfloat16)
q_stats_g = self._olmo_qk_norm_all_gather(q_stats, cluster_axis=0)
ttnn.deallocate(q_stats)
q_normed = ttnn.rms_norm_post_all_gather(q_fused, q_stats_g, weight=q_norm_weight, ...)
ttnn.deallocate(q_fused)
ttnn.deallocate(q_stats_g)

# 4. Split normed Q into heads: [1,1,S,640] → [1,5,S,128]
q_unflat = ttnn.reshape(q_normed, [1, seq_len, self.n_local_heads, self.head_dim])
ttnn.deallocate(q_normed)
q_heads_1QSD_pre_rot = ttnn.transpose(q_unflat, 1, 2)
ttnn.deallocate(q_unflat)
```

### Task 2: Restructure K-norm to operate pre-split

K is simpler since K shape is already `[1,1,S,128]` after slicing — same shape the
current code uses. Replace the typecast + to_layout with typecast on the slice:

```python
# K-norm on [1,1,S,128] — same as current but from slice, skips nlp_create_heads
if k_fused.dtype != ttnn.bfloat16:
    k_fused_bf8 = k_fused
    k_fused = ttnn.typecast(k_fused, dtype=ttnn.bfloat16)
    ttnn.deallocate(k_fused_bf8)

k_fused = ttnn.to_layout(k_fused, ttnn.TILE_LAYOUT)  # may already be TILE from slice
k_stats = ttnn.rms_norm_pre_all_gather(k_fused, ..., dtype=ttnn.bfloat16)
k_stats_g = self._olmo_qk_norm_all_gather(k_stats, cluster_axis=0)
ttnn.deallocate(k_stats)
k_heads_1KSD_pre_rot = ttnn.rms_norm_post_all_gather(k_fused, k_stats_g, weight=k_norm_weight, ...)
ttnn.deallocate(k_stats_g)
```

### Task 3: Handle `batch_size > 1` and `seq_len > 2048` edge cases

The existing code reshapes for batch>1 and seq>2048 before nlp_create_qkv_heads.
These reshapes must still happen before slicing. Verify slice boundaries remain
tile-aligned after reshaping.

### Task 4: Run PCC gate

```bash
pytest models/demos/llama3_70b_galaxy/tests/test_olmo_e2e_pcc.py::TestOlmoE2EPCC::test_prefill_pcc_1layer -xvs
pytest models/demos/llama3_70b_galaxy/tests/test_olmo_e2e_pcc.py::TestOlmoE2EPCC::test_prefill_pcc_64layers -xvs
```

### Task 5: Profile and verify savings

```bash
python -m tracy -p -v -r -m "pytest models/demos/llama3_70b_galaxy/demo/demo_olmo_decode.py::test_olmo_demo -v -k profiler"
```

Verify:
- No ReshapeViewDeviceOperation(128x640) before norm (should only appear after norm)
- No TransposeDeviceOperation(128x128) before norm
- No NlpCreateHeadsDeviceOperation in the prefill path (replaced by slices)

### Task 6: Commit
