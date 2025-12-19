# 📚 Learned Patterns from Other Model E2E Tests

## 🎯 Key Findings

After analyzing Gemma3, CLIP, Mamba, and other model E2E tests, here are the critical patterns for successful torch-to-TTNN conversion:

---

## ✅ Pattern 1: Hybrid Processing (Torch CPU → TTNN Device)

### What We Learned
**Models accept torch.Tensor inputs and do preprocessing on CPU before TTNN conversion.**

### Example from Gemma3 Conv2dPatch

```python
def forward(self, x: torch.Tensor):
    # 1. Torch preprocessing on CPU
    x = self._unfold(x)  # torch operation
    x = x.permute(0, 2, 1)  # torch operation
    
    # 2. Padding for tile alignment
    pad_len = nearest_32(x.shape[-1]) - x.shape[-1]
    padding = torch.zeros((x.shape[0], x.shape[1], pad_len), dtype=x.dtype, device=x.device)
    x = torch.cat([x, padding], dim=-1)
    
    # 3. Convert to TTNN right before device operation
    x = ttnn.as_tensor(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=self.mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
    )
    
    # 4. TTNN device operation
    out = ttnn.linear(x, self._linear_weight, bias=self.bias, ...)
    
    return out
```

### Key Takeaways
- ✅ Accept torch.Tensor as input (don't require pre-conversion)
- ✅ Do CPU preprocessing with torch (unfold, permute, pad)
- ✅ Convert to TTNN using `ttnn.as_tensor` right before device ops
- ✅ Use `ttnn.from_torch` for masks/metadata
- ❌ Don't try to use TTNN for CPU-only operations

---

## ✅ Pattern 2: Position Embedding Addition

### What We Learned
**Use ttnn.embedding + ttnn.add for position embeddings, not direct tensor addition.**

### Example from Gemma3 SiglipVisionEmbeddings

```python
def forward(self, pixel_values: torch.Tensor) -> ttnn.Tensor:
    # 1. Get patch embeddings (torch → TTNN inside)
    patch_embeddings = self.patch_embed(pixel_values)  # Returns TTNN tensor
    
    # 2. Reshape if needed
    batch_size = patch_embeddings.shape[1]
    patch_embeddings = ttnn.reshape(patch_embeddings, (batch_size, -1, self.hidden_dim))
    
    # 3. Use ttnn.embedding for position embeddings
    positional_embeddings = ttnn.embedding(
        self.position_ids,  # Pre-created TTNN tensor of indices
        self.pos_emb_weights,  # Pre-loaded TTNN tensor
        layout=ttnn.TILE_LAYOUT
    )
    
    # 4. Use ttnn.add (NOT direct +)
    embeddings = ttnn.add(patch_embeddings, positional_embeddings)
    
    return embeddings
```

### Key Takeaways
- ✅ Pre-create position_ids as TTNN tensor in __init__
- ✅ Use `ttnn.embedding` to fetch position embeddings
- ✅ Use `ttnn.add` explicitly (not `+` operator)
- ✅ Ensure both tensors have compatible layouts (TILE_LAYOUT)
- ❌ Don't try to add torch.Tensor to ttnn.Tensor

---

## ✅ Pattern 3: Attention Mask Conversion

### What We Learned
**Create masks in torch, then convert to TTNN with proper config.**

### Example from Gemma3 VisionModel

```python
def forward(self, images):
    # 1. Process embeddings first (torch → TTNN)
    x = self.embeddings(images)
    
    # 2. Create mask in torch
    attention_mask = torch.zeros(bsz, 1, x.shape[1], x.shape[1])
    
    # 3. Convert mask to TTNN
    tt_mask = ttnn.from_torch(
        attention_mask,
        device=self.mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    
    # 4. Pass TTNN mask to encoder
    x = self.encoder(x, mask=tt_mask)
    
    return x
```

### Key Takeaways
- ✅ Create masks in torch first
- ✅ Use `ttnn.from_torch` to convert masks
- ✅ Specify proper layout (TILE_LAYOUT)
- ✅ Use DRAM_MEMORY_CONFIG for masks
- ❌ Don't create masks on device directly

---

## ✅ Pattern 4: End-to-End Test Structure

### What We Learned
**E2E tests should:**
1. Use real torch inputs
2. Run torch reference model
3. Run TTNN model with same inputs
4. Convert TTNN outputs back to torch
5. Compare with PCC

### Example from CLIP E2E Test

```python
def test_clip_encoder(mesh_device, ...):
    # 1. Load HF reference model
    hf_model = CLIPTextModel.from_pretrained(model_checkpoint)
    hf_model.eval()
    
    # 2. Prepare torch inputs
    hf_inputs = tokenizer(test_text, padding=True, return_tensors="pt")
    
    # 3. Convert inputs to TTNN
    tt_prompt = ttnn.from_torch(
        hf_inputs.input_ids,
        dtype=ttnn.uint32,
        layout=ttnn.TILE_LAYOUT,
        device=encoder_submesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(encoder_submesh),
    )
    
    # 4. Run TTNN model
    tt_sequence_output, tt_pooled_output = tt_clip(tt_prompt, encoder_submesh)
    
    # 5. Run HF reference
    with torch.no_grad():
        hf_output = hf_model(hf_inputs.input_ids, output_hidden_states=True)
    
    # 6. Convert TTNN outputs to torch
    tt_sequence_output_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_sequence_output[-1])[0])
    tt_pooled_output_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_pooled_output)[0])
    
    # 7. Compare with PCC
    assert_quality(hf_sequence_output, tt_sequence_output_torch, pcc=expected_pcc)
    assert_quality(hf_pooled_output, tt_pooled_output_torch, pcc=expected_pcc)
```

### Key Takeaways
- ✅ Load real model weights
- ✅ Use HF/torch reference as ground truth
- ✅ Convert inputs properly with `ttnn.from_torch`
- ✅ Run both models
- ✅ Convert TTNN outputs back with `ttnn.to_torch`
- ✅ Use proper PCC thresholds (0.95-0.99)

---

## ✅ Pattern 5: Tensor Layout and Memory Config

### What We Learned
**Different operations need different layouts and memory configs.**

### Layouts
```python
# For linear/matmul operations
ttnn.TILE_LAYOUT  # Most common, required for matmul

# For element-wise operations
ttnn.ROW_MAJOR_LAYOUT  # Sometimes better for broadcasting
```

### Memory Configs
```python
# For weights (read-only)
ttnn.DRAM_MEMORY_CONFIG

# For activations (read-write)
ttnn.L1_MEMORY_CONFIG  # Faster but limited

# For large tensors
ttnn.DRAM_MEMORY_CONFIG  # Safer
```

### Mesh Mappers (for multi-device)
```python
# Replicate weights across all devices
ttnn.ReplicateTensorToMesh(mesh_device)

# Shard data across devices
ttnn.ShardTensorToMesh(mesh_device, dim=0)
```

---

## ✅ Pattern 6: Common TTNN Operations

### What We Learned
**Use TTNN operations, not PyTorch operations, once on device.**

```python
# ✅ CORRECT - TTNN operations
ttnn.add(a, b)
ttnn.mul(a, b)
ttnn.reshape(a, shape)
ttnn.permute(a, dims)  # or ttnn.transpose
ttnn.concat([a, b], dim=0)
ttnn.linear(x, weight, bias=bias)
ttnn.embedding(indices, weights)
ttnn.layer_norm(x, weight=w, bias=b)

# ❌ WRONG - PyTorch operations on TTNN tensors
a + b  # Use ttnn.add
a * b  # Use ttnn.mul
a.permute(...)  # Use ttnn.permute
a.reshape(...)  # Use ttnn.reshape
```

---

## 🎯 Our Current Issue: Broadcast Type Error

### The Error
```
TT_THROW: Invalid subtile broadcast type
Location: During hidden_states + position_embedding
```

### Root Cause
We were trying to:
```python
# ❌ WRONG - Direct addition
hidden_states = hidden_states + self.position_embedding
```

### The Fix (from Gemma3)
```python
# ✅ CORRECT - Use ttnn.embedding + ttnn.add
# 1. Pre-create position indices
self.position_ids = ttnn.arange(0, num_positions, 1, dtype=ttnn.uint32, device=device)
self.position_ids = ttnn.reshape(self.position_ids, (1, -1))

# 2. Pre-load position embedding weights
self.pos_emb_weights = ttnn.as_tensor(
    positional_embedding_tensor,
    dtype=dtype,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)

# 3. In forward, use ttnn.embedding
positional_embeddings = ttnn.embedding(
    self.position_ids,
    self.pos_emb_weights,
    layout=ttnn.TILE_LAYOUT
)

# 4. Use explicit ttnn.add
embeddings = ttnn.add(patch_embeddings, positional_embeddings)
```

---

## 📊 Success Metrics from Other Models

| Model | PCC Threshold | Test Type | Status |
|-------|--------------|-----------|---------|
| Gemma3 Vision | 0.95 | E2E Pipeline | ✅ Passing |
| CLIP Encoder | 0.99 (enc1), 0.984 (enc2) | E2E | ✅ Passing |
| Mamba | 0.99 | Full Model | ⚠️ Skipped (issue #7551) |
| Gemma3 Embeddings | 0.99 | Component | ✅ Passing |
| Gemma3 Encoder | 0.99 | Component | ✅ Passing |

---

## 🚀 Action Plan for PI0

### Immediate Fixes Needed

1. **Fix Position Embedding Addition**
   - Replace direct `+` with `ttnn.add`
   - Use `ttnn.embedding` pattern from Gemma3
   - Ensure proper tensor layouts

2. **Adopt Hybrid Processing**
   - Keep torch preprocessing for patch embedding
   - Convert to TTNN only for device operations
   - Follow Gemma3 Conv2dPatch pattern

3. **Fix Tensor Operations**
   - Replace all `.permute()` with `ttnn.permute()`
   - Replace all direct math with `ttnn.add`, `ttnn.mul`, etc.
   - Ensure proper memory configs

### Long-term Improvements

1. **Component-by-Component Testing**
   - Test each module individually
   - Validate PCC at each stage
   - Match Gemma3's piecewise test approach

2. **Proper Weight Conversion**
   - Use `ttnn.as_tensor` for all weights
   - Replicate weights across devices
   - Cache in DRAM

3. **Memory Optimization**
   - Use `ttnn.deallocate` for intermediate tensors
   - Profile memory usage
   - Optimize grid sizes

---

## ✅ Summary

### What Works in Production Models
- ✅ Hybrid torch/TTNN processing
- ✅ `ttnn.as_tensor` for weights
- ✅ `ttnn.from_torch` for activations
- ✅ `ttnn.embedding` for embeddings
- ✅ Explicit `ttnn.add`, `ttnn.mul`, etc.
- ✅ TILE_LAYOUT for most operations
- ✅ Component-by-component validation

### What Doesn't Work
- ❌ Pure TTNN from input to output
- ❌ Direct `+`, `*` operators on TTNN tensors
- ❌ `.permute()`, `.reshape()` methods on TTNN
- ❌ Mixing torch and TTNN operations
- ❌ Assuming tensor broadcasts work like PyTorch

---

## 🎯 Next Steps

1. **Apply Gemma3 pattern to SigLIPVisionTowerTTNN**
   - Use hybrid processing
   - Fix position embedding addition
   - Use ttnn.embedding + ttnn.add

2. **Test component-by-component**
   - Patch embedding first
   - Then position embedding
   - Then encoder blocks
   - Finally full vision tower

3. **Run E2E test**
   - Compare torch vs TTNN outputs
   - Validate PCC > 0.95
   - Measure performance gains

---

**Status:** Ready to implement fixes based on these learnings! 🚀

