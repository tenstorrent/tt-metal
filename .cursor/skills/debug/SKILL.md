---
name: debug
description: >-
  Diagnose and fix PCC failures, device hangs, and other issues during TTNN
  bring-up. Use when debugging low PCC, NaN/Inf values, device hangs, bfloat8_b
  overflow, sharding mismatches, tracing errors, or when model output is
  noise/garbage.
---

# SKILL: Debugging TTNN Implementations

## Purpose
Diagnose and fix PCC failures, device hangs, and other issues during TTNN bring-up.

## CRITICAL FIRST STEP: Verify Reference Is Correct

**Before debugging TTNN, verify your reference implementation produces correct output!**

### The "Both Are Wrong" Trap

If TTNN output is wrong AND reference output is wrong:
- High PCC between TTNN and reference is MEANINGLESS
- You're measuring agreement between two broken implementations
- The bug might be in the reference, not TTNN

### Verification Steps

1. **Test reference against official package:**
```python
official_output = official_model.generate(input)
reference_output = reference_model(input)

pcc = compute_pcc(official_output, reference_output)
print(f"Reference vs Official PCC: {pcc}")  # MUST be > 0.99
```

2. **For Audio/TTS: LISTEN to the output:**
```python
sf.write("/tmp/reference.wav", reference_output.numpy(), 24000)
# LISTEN TO IT! Is it intelligible speech or noise?
```

3. **If reference produces noise/garbage:**
   - DO NOT debug TTNN yet
   - Fix the reference first
   - The reference must match official EXACTLY

## PCC Debugging (< 0.99)

### Layer-by-Layer Debugging Strategy
1. **Isolate the failing layer**: Run each layer independently
2. **Compare intermediate outputs**: Save TTNN and reference outputs at each step
3. **Binary search**: If many layers, bisect to find first failure

```python
def debug_layer_by_layer(model, reference_model, x):
    """Debug by comparing each layer's output."""
    x_ttnn = ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT)
    x_ref = x.clone()

    for i, (ttnn_layer, ref_layer) in enumerate(zip(model.layers, reference_model.layers)):
        x_ttnn = ttnn_layer(x_ttnn)
        x_ref = ref_layer(x_ref)

        pcc = compute_pcc(ttnn.to_torch(x_ttnn), x_ref)
        print(f"Layer {i}: PCC = {pcc:.6f}")

        if pcc < 0.99:
            print(f"  FAIL at layer {i}")
            torch.save({'ttnn': ttnn.to_torch(x_ttnn), 'ref': x_ref}, f'debug_layer_{i}.pt')
            break
```

### Common PCC Failure Patterns

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| PCC ~0.98-0.99 | Dtype precision | Use bfloat16 instead of bfloat8_b |
| PCC drops at layer N | Overflow accumulation | Check MLP/attention weights at layer N |
| PCC ~0.5 or random | Wrong weight mapping | Verify weight names match HuggingFace |
| PCC ~-1 | Sign flip | Check transpose/reshape operations |
| PCC NaN | Overflow/underflow | Check for Inf values, reduce precision |

### bfloat8_b Numerical Overflow Diagnosis

**Detection**:
```python
def check_for_overflow(tensor):
    """Check if tensor has overflow values."""
    t = ttnn.to_torch(tensor)
    has_inf = torch.isinf(t).any().item()
    has_nan = torch.isnan(t).any().item()
    max_val = t.abs().max().item()
    print(f"Has Inf: {has_inf}, Has NaN: {has_nan}, Max: {max_val}")
    return has_inf or has_nan
```

**Solution**:
```python
# Option 1: Use bfloat16 for all weights
weight = ttnn.from_torch(w, dtype=ttnn.bfloat16, ...)

# Option 2: Store CPU copies, convert on-demand (for decode hot path)
self.weight_cpu = state_dict["weight"].clone()

def forward_decode(self, x):
    weight = ttnn.from_torch(self.weight_cpu.to(torch.bfloat16), device=self.device, ...)
    return ttnn.linear(x, weight)
```

### TILE_LAYOUT Issues
```python
def safe_linear(x, weight):
    assert x.layout == ttnn.TILE_LAYOUT, f"Input layout: {x.layout}"
    assert weight.layout == ttnn.TILE_LAYOUT, f"Weight layout: {weight.layout}"
    return ttnn.linear(x, weight)
```

## Device Hang Recovery

### T3K Device Reset
```bash
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)
source python_env/bin/activate

tt-smi -r
```

### Galaxy Device Reset
```bash
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)
source python_env/bin/activate

tt-smi -glx_reset
```

### Common Hang Causes
1. **Memory exhaustion**: Reduce batch size, use DRAM instead of L1
2. **Infinite loop in kernel**: Check op parameters
3. **Deadlock in multi-chip**: Verify CCL synchronization

## Tensor Inspection Techniques

```python
def inspect_tensor(tensor, name="tensor"):
    """Full tensor inspection."""
    t = ttnn.to_torch(tensor)
    print(f"\n{name}:")
    print(f"  Shape: {t.shape}")
    print(f"  Dtype: {t.dtype}")
    print(f"  Min: {t.min().item():.6f}")
    print(f"  Max: {t.max().item():.6f}")
    print(f"  Mean: {t.mean().item():.6f}")
    print(f"  Std: {t.std().item():.6f}")
    print(f"  Has NaN: {torch.isnan(t).any().item()}")
    print(f"  Has Inf: {torch.isinf(t).any().item()}")
    print(f"  Sample values: {t.flatten()[:5].tolist()}")
```

## Tracing Debugging

### "Writes not supported during trace capture"
**Cause**: Using ops that write to device memory during trace
**Solution**: Move writes outside trace, use pre-allocated tensors

### "Allocating device buffers is unsafe" Warning
**Cause**: Allocating after trace capture started
**Solution**: Pre-allocate all tensors before `begin_trace_capture`

## "Output Is Garbage" Debugging

When the model produces noise/garbage instead of correct output:

### Step 1: Check Component Inventory
```markdown
Is the model COMPLETE? Check ARCHITECTURE.md for:
- [ ] ALL components listed in inventory
- [ ] ALL components have implementations
- [ ] No components marked as "Not implemented"
```

### Step 2: Verify Reference End-to-End
```python
reference_audio = reference_full_pipeline(text_input)
sf.write("/tmp/reference.wav", reference_audio.numpy(), 24000)
# LISTEN - is it correct?
```

If reference also produces garbage, the bug is in reference, not TTNN.

### Step 3: Compare at Each Stage
```python
stages = ["embedding", "talker_out", "code_predictor_out", "decoder_out"]
for stage in stages:
    pcc = compute_pcc(ttnn_outputs[stage], reference_outputs[stage])
    print(f"{stage}: PCC = {pcc:.6f}")
```

## Galaxy / TG Multi-Device Debug Patterns

### Matmul Producing 10^35x Larger Values Than Expected
**Symptom**: Output max ~3e38 from reasonable input (max ~1.0) and weights (max ~0.6).
**Cause**: Using wrong weight tensor variant. E.g., `w1_interleaved` (sharded with `dims=(-1,-2)` giving K=640/device) vs `w1` (ring-sharded with K=1280/device). Dimension mismatch makes matmul read past the weight buffer into garbage memory.
**Fix**: Verify the weight's K dimension matches the input's last dimension. Use `ttnn.to_torch(weight).shape` to confirm.

### reduce_scatter Producing Inf From Valid Input
**Symptom**: `reduce_scatter_minimal_async` output has Inf, but input tensor values are all reasonable.
**Cause**: Ring matmul output is padded (e.g., 3840 from 3456). Padding region contains uninitialized garbage. `reduce_scatter` sums ALL elements including garbage across devices → overflow to Inf.
**Fix**: Slice to unpadded width BEFORE cross-device communication. Or use `all_gather(dim=0) + fast_reduce_nc(dims=[0])` pattern which avoids summing padded regions.

### Vocab Logits PCC ~0.13 Despite Correct Computation
**Symptom**: Per-device logits look correct, but after all_gather PCC drops to ~0.13.
**Cause**: `padded_vocab_size` not aligned to `num_devices × tile_size`. E.g., vocab=100278 → nearest_32=100288 → per-device=12536 (not tile-aligned). TTNN tile-pads 12536→12544 per shard, inserting zeros that shift token indices after gather.
**Fix**: Pad vocab to nearest multiple of `num_vocab_devices × tile_size` (e.g., 8×32=256). This ensures per-device shards are already tile-aligned with no extra padding.

### `ttnn.slice` Aliasing Causes "Buffer Is Not Allocated"
**Symptom**: `TT_FATAL: Buffer is not allocated` when using a tensor obtained from `ttnn.slice`.
**Cause**: `ttnn.slice` creates a memory alias (view) into the parent tensor. Deallocating the parent before the alias is consumed frees the underlying buffer.
**Fix**: Reorder deallocations — keep parent tensor alive until all slice aliases are consumed.

### `rms_norm_pre/post_all_gather` Requires Height == TILE_HEIGHT (32)
**Symptom**: `TT_FATAL: a.padded_shape()[-2] == TILE_HEIGHT` when using sharded `rms_norm_pre_all_gather` with `program_config`.
**Cause**: The distributed norm ops (`rms_norm_pre_all_gather`, `rms_norm_post_all_gather`) with `LayerNormShardedMultiCoreProgramConfig` strictly require tensor height = 32 (one tile). Regular `ttnn.rms_norm` with `program_config` supports height > 32.
**Fix**: Reshape tensor so height dimension is ≤ 32 (tile-padded to 32). Merge extra rows into width. E.g., [1,1,64,128] → [1,1,8,1024] via permute+reshape, then WIDTH_SHARD across 8 cores.

### "Kernel Group Cores Do Not Match Sub Device Cores"
**Symptom**: `TT_FATAL: Kernel group cores do not match sub device cores for programmable core type TENSIX` during `ttnn.reshape` or other op.
**Cause**: When prefetcher is enabled, a sub-device manager restricts which cores can run kernels. Operations that dispatch to the full grid (e.g., `ttnn.reshape` on DRAM interleaved) conflict with the worker sub-device context.
**Fix**: Use L1-sharded memory on worker cores for reshapes. Or use metadata-only reshape (`ttnn.view`) which doesn't dispatch a kernel. Or move the operation outside the sub-device context.

### Device Hang After TT_FATAL
**Symptom**: `TT_FATAL: Read unexpected run_mailbox value from core (x=N,y=M)` on every subsequent run.
**Cause**: A previous crash corrupted device state. The mailbox values are stale from the failed kernel.
**Fix**: `tt-smi -r` (Wormhole) or `tt-smi -glx_reset` (Galaxy) to reset devices. Must be done before any further tests.

## Per-Op PCC Isolation for Decode (Galaxy / Multi-Device)

When multi-layer PCC is below target but single-layer appears fine, the issue may be a **false positive** — the residual connection (std≈1.0) masks a broken sublayer (std≈0.01).

### Step 1: Capture Intermediates on Device

Add capture points in the TTNN model to record tensors at each op:

```python
# In the model's forward method (e.g., llama_attention.py):
def _capture_attn(self, name, tensor):
    if hasattr(self, '_attn_captures') and self._attn_captures is not None:
        t_cpu = ttnn.to_torch(tensor, mesh_composer=ttnn.ConcatMesh2dToTensor(
            self.mesh_device, dims=(0, 1), mesh_shape=self.cluster_shape))
        self._attn_captures[name] = t_cpu
```

Enable captures in the test, run one forward pass, then compare each capture against the reference.

### Step 2: Reconstruct Full Tensors from Multi-Device Captures

After `ConcatMesh2dToTensor`, the captured tensor has shape `[row_devs, col_devs, batch, dim]` or similar. To reconstruct a single user's full hidden state, you must understand the parallelism layout:

```python
# Decoder captures (ConcatMesh2dToTensor dims=(3,1) for [8,4] mesh):
#   Shape: [1, 4, 32, 10240] — dim1=4 col groups, dim3=8× replicated
#   User 0 full state: cat([tt[0, j, 0, :1280] for j in range(4)])  → [5120]

# Attention captures (ConcatMesh2dToTensor dims=(0,1) for [8,4] mesh):
#   Shape: [8, 32, NH, D] — dim0=8 row devs, dim1=4×batch_per_dev
#   User 0: depends on which dim holds batch vs heads
```

**CRITICAL**: Verify which dimension is batch and which is heads. SDPA output is `[1, B, NH_padded, D]` (batch-major), NOT `[1, NH, B, D]`. Getting this wrong gives garbage PCC but won't crash — the most insidious bug type.

### Step 3: Use Multiple Metrics, Not Just PCC

PCC alone can mislead. Always check standard deviations to detect magnitude collapse:

```python
def safe_pcc(label, ref_flat, tt_flat):
    ref_f, tt_f = ref_flat.float(), tt_flat.float()
    pcc_val = comp_pcc(ref_f.unsqueeze(0), tt_f.unsqueeze(0), 0.0)[0]
    logger.info(f"  {label}: PCC={pcc_val:.4f}  len={ref_f.numel()}  "
                f"ref_std={ref_f.std():.4f} tt_std={tt_f.std():.4f}")
```

| Symptom | Diagnosis |
|---------|-----------|
| PCC high but tt_std << ref_std | **Magnitude collapse** — data path is broken, signal attenuated |
| PCC ~0 but both stds match | Completely wrong data (e.g., different users' data compared) |
| PCC ~0 and tt_std ≈ 0 | Output is near-zero — op is producing zeros or near-zeros |
| PCC oscillates by head | Per-head indexing error in comparison code |

### Step 4: Check the Comparison Code Itself

The most common false negative: **bugs in the PCC test's tensor extraction**, not in the model. Before concluding an op is broken:

1. Print shapes of both sides: `tt.shape` vs `ref.shape`
2. Verify the extracted slice corresponds to the SAME data (same user, same head, same device)
3. For `ConcatMesh2dToTensor` outputs, trace which dim maps to which mesh axis
4. Test extraction on a known-good tensor (e.g., embedding input) before trusting it on unknowns

### Case Study: OLMo SDPA→WO Magnitude Collapse

**Symptom**: `wo_input` PCC=0.05, `tt_std=0.017` vs `ref_std=0.276` (16× collapse). But `SDPA_out` PCC=0.9999.

**Root cause (two bugs)**:
1. **Dimension confusion**: SDPA output is `[1, batch=8, heads_padded=32, head_dim=128]`. Code read `shape[2]` (=32, tile-padded heads) as batch, sliced dim1 (batch=8→5) thinking it was slicing heads.
2. **Missing batch all-gather**: `llama_rs_create_heads` splits batch 32→8 across 4 col devices via reduce_scatter. The reverse all-gather was skipped, so WO only processed 8 of 32 users.

**Why it was hard to find**: The garbled tensor happened to be mostly zeros (24/32 positions were tile padding), producing near-zero WO output. The residual (std≈1.0) masked this in per-layer PCC. Only per-op std comparison revealed the 16× collapse.

**Fix**: Correct the slice dimension, add `line_all_gather(dim=2, cluster_axis=1)` to restore full 32-user batch before WO matmul.

**Lesson**: Always compare `tt_std` vs `ref_std` alongside PCC. A 10×+ std ratio indicates a broken data path even if overall layer PCC is 0.99+.

## Debugging Checklist

- [ ] **Reference produces correct output** (not just runs without errors)
- [ ] **All model components implemented** (check ARCHITECTURE.md inventory)
- [ ] Verify weight names match HuggingFace exactly
- [ ] Check dtypes (bfloat16 vs bfloat8_b)
- [ ] Confirm TILE_LAYOUT for all matmul inputs
- [ ] Verify shapes are padded to tile size (32x32)
- [ ] Compare against similar model in ARCHITECTURE.md
- [ ] Run unit tests for individual ops
- [ ] Check for NaN/Inf values at each layer
- [ ] Verify sharding specs match between ops
