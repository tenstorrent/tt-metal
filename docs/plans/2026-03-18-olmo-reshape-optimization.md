# OLMo Prefill Reshape Optimization Plan

**Date**: 2026-03-18
**Author**: ssinghal
**Status**: Draft
**Target**: Eliminate per-layer reshape bottleneck in Q-norm prefill path

---

## Problem

Two `ReshapeViewDeviceOperation` calls dominate prefill time at large ISLs,
occurring **once per transformer layer** (64 total):

| ISL | Reshape cost/layer | × 64 layers | % of total prefill |
|-----|-------------------|-------------|-------------------|
| 128 | 105 us | 6.7 ms | 6.9% |
| 4k | 2,573 us | 165 ms | 23.1% |
| 8k | 4,437 us | 284 ms | 21.5% |
| 16k | 18,082 us | 1,157 ms | 24.2% |

Additionally, each reshape is bracketed by a `TransposeDeviceOperation` (also expensive):

| ISL | Transpose cost/layer | × 64 layers |
|-----|---------------------|-------------|
| 4k | 695 us | 44 ms |
| 8k | 1,363 us | 87 ms |
| 16k | 5,528 us | 354 ms |

Total avoidable overhead at 16k: **(18,082 + 5,528) × 64 ≈ 1,511 ms** of wasted reshape+transpose.

---

## Root Cause

The reshapes appear inside the **OLMo Q-norm path** in `llama_attention.py` (lines 1247–1270):

```
nlp_create_qkv_heads(xqkv_fused)
  → Q: [1, 5, seq, 128]   # 5 local Q heads, head_dim=128

# Q-norm needs flat form:
→ transpose(1,2)           # [1, seq, 5, 128]
→ reshape                  # [1, 1, seq, 640]     ← ReshapeViewDeviceOperation #1
→ rms_norm_pre_all_gather  # [1, 1, seq, 640]
→ all_gather (axis=0)      # collect partial stats across 8 column devices
→ rms_norm_post_all_gather # [1, 1, seq, 640]
→ reshape                  # [1, seq, 5, 128]     ← ReshapeViewDeviceOperation #2
→ transpose(1,2)           # [1, 5, seq, 128]
→ rotary_embedding_llama
```

**Why K-norm has no reshape**: K has `n_local_kv_heads = 1` so its shape after
`nlp_create_qkv_heads` is already `[1, 1, seq, 128]` — directly compatible with
`rms_norm_pre_all_gather`. Q has `n_local_heads = 5`, so it needs the flatten.

**The fix**: Apply Q-norm BEFORE `nlp_create_qkv_heads`, while Q is still in its
natural flat form `[1, 1, seq, 640]` inside `xqkv_fused`. This eliminates all
4 ops (transpose, reshape, reshape, transpose).

---

## Architecture

OLMo-3.1-32B tensor shapes at prefill (per device, 8 devices):
- `xqkv_fused` after line_all_reduce: `[1, 1, seq, 896]`
  - Q slice: columns `[0:640]` = 5 heads × 128 dim
  - K slice: columns `[640:768]` = 1 head × 128 dim
  - V slice: columns `[768:896]` = 1 head × 128 dim
- `n_local_heads = 5`, `n_local_kv_heads = 1`, `head_dim = 128`

---

## Proposed Solution

### Option A — Pre-head Q-norm (Recommended)

Apply Q-norm directly on `xqkv_fused` before `nlp_create_qkv_heads`.

**Current flow (expensive)**:
```python
q, k, v = nlp_create_qkv_heads(xqkv_fused)     # Q: [1,5,seq,128]
q = transpose(q, 1, 2)                           # [1,seq,5,128]
q_flat = reshape(q, [1,1,seq,640])               # ReshapeView #1
q_normed = rms_norm_with_allgather(q_flat)       # [1,1,seq,640]
q_unflat = reshape(q_normed, [1,seq,5,128])      # ReshapeView #2
q = transpose(q_unflat, 1, 2)                    # [1,5,seq,128]
```

**Proposed flow (no reshape)**:
```python
q_size = n_local_heads * head_dim                # 640
q_flat = xqkv_fused[..., :q_size]               # [1,1,seq,640] — free slice
q_normed_flat = rms_norm_with_allgather(q_flat)  # [1,1,seq,640] — no reshape!
# rebuild xqkv_fused with normed Q
kv_tail = xqkv_fused[..., q_size:]              # [1,1,seq,256]
xqkv_normed = ttnn.concat([q_normed_flat, kv_tail], dim=-1)  # [1,1,seq,896]
q, k, v = nlp_create_qkv_heads(xqkv_normed)     # Q: [1,5,seq,128] — already normed
# K-norm stays the same (already no-reshape)
```

**Saves per layer**: 2× ReshapeViewDeviceOperation + 2× TransposeDeviceOperation

**Risk**: Medium — requires verifying `ttnn.slice` / indexing on DRAM tensor,
and that `ttnn.concat` on `xqkv_normed` is cheaper than the 4 removed ops.

---

### Option B — rms_norm on multi-head tensor

Check if `rms_norm_pre_all_gather` accepts `[1, n_heads, seq, head_dim]` input
treating each row of the last dimension independently. If yes, pass Q directly
after `nlp_create_qkv_heads` without transpose or reshape.

**Saves per layer**: 2× ReshapeViewDeviceOperation + 2× TransposeDeviceOperation
**Risk**: Low if API supports it; requires checking kernel constraints. The K-norm
path `[1, 1, seq, 128]` works, and this is equivalent with n_heads in dim-1.

---

### Option C — Change xqkv_fused layout at source

At ISL > 2048, `xqkv_fused` goes through `reshape` to `[1, 1, seq, 896]` anyway
(line 1215). The Q slice is already naturally flat at that point. Combine Option A
with this to reuse the already-flat form and avoid even the concat.

---

## Implementation Plan

### Step 1: Reference verification (no device needed)

Verify the math equivalence in PyTorch reference:

```python
# In reference/olmo.py or a standalone test
# Show that:
#   norm(q.reshape(1, 1, seq, 5*128)) == norm_per_head(q)
# for OLMo's specific norm weight layout
```

This confirms the reshape is a layout reshuffling, not a semantic change,
so moving it earlier is numerically equivalent.

### Step 2: Try Option B first (lowest code change)

In `llama_attention.py`, remove the transpose+reshape before Q-norm and
test if `rms_norm_pre_all_gather` accepts `[1, 5, seq, 128]`:

```python
elif self.is_olmo and self.qk_norm:
    q_heads_1QSD_pre_rot = ttnn.to_layout(q_heads_1QSD_pre_rot, ttnn.TILE_LAYOUT)
    q_stats = ttnn.rms_norm_pre_all_gather(
        q_heads_1QSD_pre_rot, ...   # try [1, 5, seq, 128] directly
    )
    # if this raises shape error → fall back to Option A
```

Verify PCC on 1L test:
```bash
pytest models/demos/llama3_70b_galaxy/demo/demo_olmo_decode.py -k "isl-128-b1"
```

### Step 3: Implement Option A if Option B fails

Extract Q slice before `nlp_create_qkv_heads`:

```python
# In llama_prefill_forward(), after line_all_reduce:
if seq_len > 2048:
    xqkv_fused = ttnn.reshape(xqkv_fused, [1, 1, seq_len, -1])  # already done

if self.is_olmo and self.qk_norm:
    q_size = self.n_local_heads * self.head_dim  # 640
    q_flat = ttnn.slice(xqkv_fused,
                        begins=[0, 0, 0, 0],
                        ends=[1, 1, seq_len, q_size])
    q_normed_flat = self._olmo_q_norm_flat(q_flat)   # extracted helper
    ttnn.deallocate(q_flat)
    kv_tail = ttnn.slice(xqkv_fused,
                         begins=[0, 0, 0, q_size],
                         ends=[1, 1, seq_len, xqkv_fused.shape[-1]])
    xqkv_fused = ttnn.concat([q_normed_flat, kv_tail], dim=-1)
    ttnn.deallocate(q_normed_flat)
    ttnn.deallocate(kv_tail)

q_heads, k_heads, v_heads = ttnn.experimental.nlp_create_qkv_heads(xqkv_fused, ...)
# Q-norm block in the is_olmo branch below becomes a no-op (already normed)
```

### Step 4: PCC gate

Run the full ISL sweep PCC tests before and after:

```bash
pytest models/demos/llama3_70b_galaxy/demo/demo_olmo_decode.py \
    -k "isl-128-b1 or isl-1k-b1 or isl-2k-b1 or isl-4k-b1" --no-header -q
```

Target: PCC > 0.99 on hidden states and logits for all ISLs.

### Step 5: Profile to measure savings

Re-run profiler sweep and compare:
```bash
bash scripts/run_profiler_sweep.sh --model olmo \
    --run-name olmo_no_reshape \
    --prompt-lengths 128,4k,8k,16k
python scripts/compare_ops_raw.py profiler_sweep_results/olmo_no_reshape/
```

---

## Expected Gains

| ISL | Current reshape+transpose/layer | Expected saving/layer | × 64 layers |
|-----|--------------------------------|-----------------------|-------------|
| 4k | 3,268 us | ~3,268 us | **~209 ms** |
| 8k | 5,800 us | ~5,800 us | **~371 ms** |
| 16k | 23,610 us | ~23,610 us | **~1,511 ms** |

At 16k, this alone represents **~17% of total prefill time** reduction (1,511 ms out
of ~4,772 ms extrapolated 64L total). At 8k it's **~13%** (371 ms out of ~1,319 ms 64L).

---

## Risk and Rollback

| Risk | Mitigation |
|------|-----------|
| PCC regression if norm semantics change | Step 1 reference check before any device run |
| `ttnn.slice` / `ttnn.concat` adds new overhead | Profile both; revert if net gain is negative |
| xqkv_fused layout differs at ISL <= 2048 | Condition on `seq_len > 2048`; keep current path for short ISLs |
| All-gather buffer key mismatch after concat | Ensure concat output matches persistent buffer dtype (bfloat8_b for seq >= 4096) |

Rollback: `git revert` to pre-change commit; the existing `_capture_prefill_attn`
hooks will immediately show if Q after norm drifts from reference.

---

## Files to Modify

- `models/demos/llama3_70b_galaxy/tt/llama_attention.py` — main change (lines 1247–1270)
- `models/demos/llama3_70b_galaxy/reference/olmo.py` — add reference verification test
