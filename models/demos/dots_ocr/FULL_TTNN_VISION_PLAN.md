# Full TTNN Vision Stack Implementation Plan for Dots OCR

## Current State vs Goal

**Current (Hybrid):** HF `vision_tower` (42 ViT layers) on CPU + TTNN `PatchMerger`
**Goal:** Full TTNN implementation of `DotsVisionTransformer`

## Vision Architecture Analysis

From the model config, Dots Vision Tower has:
- **42 transformer layers** (`num_hidden_layers: 42`)
- **Hidden size: 1536**
- **12 attention heads**
- **Patch size: 14x14**
- **Spatial merge size: 2**
- **RMSNorm** (eps=1e-5, post-norm architecture)
- **~1.2B parameters** for vision encoder

## Implementation Options

### **Option 1: Full Native TTNN (Recommended - Most Complex)**

**Components to implement:**

1. **Patch Embedder** (`tt/vision_patch_embed.py`)
   - Convert image → patches (14x14)
   - Linear projection + positional embeddings
   - Handle `grid_thw` (temporal, height, width) from processor

2. **Vision Transformer Blocks** (`tt/vision_block.py`)
   - 42 layers of:
     - RMSNorm (post-norm)
     - Self-Attention (need to check if it's standard or has special windowing)
     - MLP (SwiGLU or GeLU?)
   - Similar to `qwen25_vl/tt/vision_block.py` but adapted for Dots

3. **Full Vision Transformer** (`tt/vision_transformer.py`)
   - Orchestrates patch embedding + 42 blocks + patch merger
   - Handle `pixel_values` and `grid_thw` inputs
   - Return vision tokens in format expected by fusion

4. **Enhanced Patch Merger** (update existing)
   - Current implementation exists but may need adaptation

### **Option 2: Reference-Guided TTNN (Balanced)**

Use HF reference model as a "teacher" for validation while implementing TTNN components:

```python
class DotsVisionTransformerTT(LightweightModule):
    def __init__(self, mesh_device, model_args, state_dict, ...):
        self.patch_embed = PatchEmbedTT(...)           # New
        self.blocks = nn.ModuleList([                  # 42 blocks
            VisionBlockTT(...) for _ in range(42)
        ])
        self.patch_merger = PatchMergerTT(...)         # Existing (enhanced)

    def forward(self, pixel_values, grid_thw):
        # Full TTNN implementation
        x = self.patch_embed(pixel_values, grid_thw)
        for block in self.blocks:
            x = block(x)
        x = self.patch_merger(x)
        return x
```

### **Implementation Roadmap**

**Phase 1: Foundation (2-3 days)**
1. Analyze HF `DotsVisionTransformer` implementation
2. Create `tt/vision_model_config.py` with Dots-specific vision params
3. Implement `PatchEmbedTT` with proper tensor layouts
4. Create comprehensive test infrastructure

**Phase 2: Core Components (4-5 days)**
5. Implement `VisionAttentionTT` (RoPE, attention mechanisms)
6. Implement `VisionMLP` (feed-forward network)
7. Create `VisionBlockTT` combining attention + MLP + RMSNorm
8. Update `PatchMergerTT` if needed for exact match

**Phase 3: Integration & Validation (3-4 days)**
9. Create `VisionTransformerTT` orchestrating all components
10. Implement weight loading from HF checkpoint
11. Create `test_vision_full_pcc.py` with layer-by-layer PCC validation
12. Update `tt/model.py` to use full TTNN vision

**Phase 4: Optimization (2-3 days)**
13. WHLB memory optimization (L1 vs DRAM placement)
14. Performance tuning for 42 layers
15. Benchmark vs hybrid approach
16. Long sequence handling

### **Technical Challenges**

1. **42 Layers**: Significant compute/memory. Need careful memory planning.
2. **Post-Norm Architecture**: Different from typical pre-norm transformers.
3. **grid_thw handling**: Special 3D patching for documents (temporal, height, width).
4. **High resolution**: Model handles up to 11M pixels - need efficient patching.
5. **Weight loading**: Complex key mapping from HF `DotsVisionConfig`.
6. **RoPE alignment**: Must match HF Qwen2-style RoPE exactly.

### **Recommended Starting Point**

Would you like me to:

**A. Start with Phase 1** - Create the foundation (`vision_model_config.py`, `PatchEmbedTT`, test infrastructure)

**B. First do a detailed code analysis** - Reverse engineer the exact HF `DotsVisionTransformer` structure before coding

**C. Create a hybrid-to-full migration plan** - Gradually replace HF components with TTNN ones

**D. Focus on specific component** - e.g., just the PatchEmbed first

---

The full TTNN implementation will be **significantly more complex** than the hybrid approach (roughly 3-4x the effort) but will provide better performance on Tenstorrent hardware.

**Which approach would you like to take?**
