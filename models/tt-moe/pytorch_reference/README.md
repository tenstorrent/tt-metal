# PyTorch Reference Implementations

This directory contains PyTorch reference implementations that serve as golden references for testing and validating the tt-moe components.

## Structure

```
pytorch_reference/
├── deepseek/
│   ├── __init__.py
│   └── moe_gate.py       # DeepSeek-V3 MoE Gate reference
└── README.md
```

## DeepSeek Reference Implementations

### MoEGate (`deepseek/moe_gate.py`)

Reference implementation of the DeepSeek-V3 MoE Gate module that performs hierarchical expert routing using grouped top-k selection with sigmoid scoring.

**Key features:**
- Hierarchical routing with expert groups
- Sigmoid activation for scoring
- Top-k selection within groups and across groups
- Score normalization and scaling
- Support for both deterministic (bitonic) and standard top-k

**Usage example:**
```python
from pytorch_reference.deepseek import ReferenceMoEGate

# Create model with HuggingFace config
model = ReferenceMoEGate(hf_config, use_bitonic_sort=True)

# Initialize bias properly
torch.nn.init.zeros_(model.e_score_correction_bias)

# Run forward pass
indices, weights = model(input_tensor)
```

## Testing

These reference implementations are used in the test suite to validate the TTNN implementations:

```bash
# Run tests using the reference implementation
python -m pytest models/tt-moe/tests/test_grouped_topk_router.py
```

## Notes

- The reference implementations are copied from `models/demos/deepseek_v3/reference/` for local access
- The `e_score_correction_bias` parameter must be manually initialized (typically to zeros) as the original `reset_parameters()` doesn't initialize it
- These implementations serve as the golden reference for correctness testing
