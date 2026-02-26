```instructions
---
applyTo: "models/experimental/tt_symbiote/**"
---

# TT-Symbiote Model Bring-Up Guidelines

For bringing up new models in the TT-Symbiote PyTorch-to-TTNN acceleration framework:

## Core Workflow

**Standard 5-step pattern:**
1. Load PyTorch model (prefer `torch.bfloat16`)
2. Define module replacement mappings
3. Set device via `set_device(model, device)`
4. Preprocess and move weights to device
5. Run inference with `model.eval()` and `torch.set_grad_enabled(False)`

**Multi-stage replacement strategy:**
- Replace composite modules FIRST (e.g., `ViTEmbeddings`, `ViTLayer`)
- Then replace basic modules (e.g., `nn.Linear`, `nn.LayerNorm`)
- Merge module dictionaries: `modules = {**modules1, **modules2}`
- This prevents "module already replaced" issues and maintains proper initialization order

## Module Replacement Guidelines

**When to create rewritten `nn.Module` wrappers:**
- Composite modules with non-TTNN components (dropouts, cls_tokens, etc.)
- Modules requiring custom forward logic before/after TTNN ops
- Pattern: Wrap TTNN modules inside standard `nn.Module`, implement `from_torch()` classmethod

**When to create full `TTNNModule` implementations:**
- Basic operations that map directly to TTNN ops (Linear, LayerNorm, Conv, etc.)
- Must implement: `from_torch()`, `preprocess_weights_impl()`, `move_weights_to_device_impl()`, `deallocate_weights_impl()`, `forward()`
- Always assign `_fallback_torch_layer` in `from_torch()` for automatic fallback

**Weight lifecycle:**
- Call in strict order: `preprocess_weights()` → `move_weights_to_device()` → `deallocate_weights()`
- Use `tqdm` for large models (preprocessing can take minutes)
- For memory-constrained scenarios, use `@deallocate_weights_after` decorator on forward()

## Testing Patterns

**Pytest fixtures (from conftest.py):**
- `device` - Single device (function-scoped)
- `mesh_device` - Multi-device mesh (parametrize with device count or topology dict)
- `device_params` - Device configuration (use `indirect=True`)

**Example parametrization:**
```python
@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_model(device):
    # Single device test
```

**Mesh device parametrization:**
```python
@pytest.mark.parametrize(
    "mesh_device",
    [{
        "N150": (1, 1),
        "N300": (1, 2),
        "T3K": (1, 8),
    }.get(os.environ.get("MESH_DEVICE"), (1, len(ttnn.get_device_ids())))],
    indirect=True,
)
def test_model(mesh_device):
    # Multi-device test
```

## Debugging Strategies

**Run modes (set via `TT_SYMBIOTE_RUN_MODE`):**
- `CPU` - Verify logic without device
- Use `DPL` - Runs both TTNN and PyTorch in parallel, compares with PCC

**Dispatcher options (set via `TT_SYMBIOTE_DISPATCHER`):**
- `CPU` - Test without TTNN backend. Don't change this.

**Selective exclusion:**
```python
register_module_replacement_dict(
    model, nn_to_ttnn,
    exclude_replacement={"layer1.0", "lm_head"}  # Keep these as PyTorch
)
```

**Timing and profiling:**
```python
from models.experimental.tt_symbiote.core.run_config import DispatchManager
DispatchManager.clear_timings()
# ... run inference ...
DispatchManager.save_stats_to_file("timing_stats.csv")
```

## Common Pitfalls

**Mesh device configuration:**
- Use CCLManager for distributed tensor operations (see test_glm.py)

## Quick Reference by Complexity

**Start here (simple):**
- [test_resnet50.py](models/experimental/tt_symbiote/tests/test_resnet50.py) - Direct module replacement, Conv + Bottleneck
- [test_vit.py](models/experimental/tt_symbiote/tests/test_vit.py) - Multi-stage replacement, rewritten modules

**Moderate complexity:**
- [test_llama.py](models/experimental/tt_symbiote/tests/test_llama.py) - LLM with generation, custom MLP, bfloat8 optimizations
- [test_whisper3.py](models/experimental/tt_symbiote/tests/test_whisper3.py) - Custom encoder layers, preprocessing hooks

**Advanced (MoE, distributed):**
- [test_glm.py](models/experimental/tt_symbiote/tests/test_glm.py) - Mesh devices, MoE, distributed sharding, CCLManager
- [test_gptoss.py](models/experimental/tt_symbiote/tests/test_gptoss.py) - Large model (20B), memory optimization

**Specialized domains:**
- [test_hunyuan_video.py](models/experimental/tt_symbiote/tests/test_hunyuan_video.py) - Text-to-video generation
- [test_openvla.py](models/experimental/tt_symbiote/tests/test_openvla.py) - Vision-language-action models
- [test_yunet.py](models/experimental/tt_symbiote/tests/test_yunet.py) - Face detection with specialized preprocessing

## Available TTNN Modules

**Linear:** `TTNNLinear`, `TTNNLinearLLama` (bfloat8 + auto-deallocate), `TTNNLinearGelu`, `TTNNLinearIColShardedWRowSharded`

**Activation:** `TTNNSilu`, `TTNNReLU`, `TTNNGelu`

**Normalization:** `TTNNLayerNorm`, `TTNNRMSNorm`

**Attention:** `TTNNSelfAttention`, `TTNNSDPAAttention`, `TTNNFusedQKVSelfAttention`, `TTNNWhisperAttention`

**Conv:** `TTNNConv2dNHWC`, `TTNNConv2dBNNHWC`, `TTNNConv2dBNActivationNHWC`, `TTNNBottleneck`, `TTNNMaxPool2dNHWC`, `TTNNUpsampleNHWC`

**Embeddings:** `TTNNPatchEmbedding`, `TTNNViTEmbeddings`

**Tensor Ops:** `TTNNPermute`, `TTNNReshape`, `TTNNAdd`

**MoE:** `TTNNMoE`, `TTNNGlm4MoeMoE`

See [modules/](models/experimental/tt_symbiote/modules/) for full implementations.

## Final Notes

- Emphasize **incremental development**: Start with basic ops, gradually add complexity
- **Compare against existing tests** - patterns are consistent across model types
- Framework design prioritizes **reliability over performance** during bring-up phase
```
