# How to Add Validation to Existing Code

This guide shows step-by-step how to add validation decorators to the existing Qwen model implementation.

## Step 1: Add Validation to RMSNorm

### Current Code
```python
class RMSNorm:
    def __init__(self, weight: torch.Tensor, eps: float, device):
        self.eps = eps
        self.weight = ttnn.from_torch(
            weight.unsqueeze(0).unsqueeze(0),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT
        )

    def __call__(self, x):
        x_squared = ttnn.mul(x, x)
        mean_x_squared = ttnn.mean(x_squared, dim=-1, keepdim=True)
        rms = ttnn.sqrt(ttnn.add(mean_x_squared, self.eps))
        x_normed = ttnn.mul(x, ttnn.reciprocal(rms))
        return ttnn.mul(x_normed, self.weight)
```

### Modified Code with Validation

```python
class RMSNorm:
    def __init__(self, weight: torch.Tensor, eps: float, device):
        self.eps = eps
        self.weight = ttnn.from_torch(
            weight.unsqueeze(0).unsqueeze(0),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT
        )
        # Store original weight for validation
        self.weight_torch = weight

    @validate_against(
        reference_fn=lambda x, w, eps: w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps),
        input_map=lambda args, kwargs: (
            # args[0] is self, args[1] is x
            (
                ttnn.to_torch(args[1]).squeeze(0),  # Convert TTNN x to PyTorch
                args[0].weight_torch,                # Use stored PyTorch weight
                args[0].eps                          # Use stored eps
            ),
            {}  # No kwargs
        ),
        output_map_impl=lambda x: ttnn.to_torch(x).squeeze(0),  # TTNN -> PyTorch
        tolerances={
            'max_abs_error': 1e-2,
            'mean_abs_error': 1e-3,
        }
    )
    def __call__(self, x):
        x_squared = ttnn.mul(x, x)
        mean_x_squared = ttnn.mean(x_squared, dim=-1, keepdim=True)
        rms = ttnn.sqrt(ttnn.add(mean_x_squared, self.eps))
        x_normed = ttnn.mul(x, ttnn.reciprocal(rms))
        return ttnn.mul(x_normed, self.weight)
```

**Key changes:**
1. Store `self.weight_torch = weight` in `__init__` for reference comparison
2. Add `@validate_against` decorator with PyTorch RMS norm as reference
3. `input_map` extracts `self`, converts TTNN to PyTorch, and passes stored attributes
4. `output_map_impl` converts TTNN output back to PyTorch for comparison

## Step 2: Add Validation to MLP

### Current Code
```python
class MLP:
    def __init__(self, hidden_size, intermediate_size, gate_proj, up_proj, down_proj, device):
        self.gate_proj = ttnn.from_torch(gate_proj.T.unsqueeze(0).unsqueeze(0), device=device, ...)
        self.up_proj = ttnn.from_torch(up_proj.T.unsqueeze(0).unsqueeze(0), device=device, ...)
        self.down_proj = ttnn.from_torch(down_proj.T.unsqueeze(0).unsqueeze(0), device=device, ...)

    def __call__(self, x):
        gate = ttnn.matmul(x, self.gate_proj)
        gate = ttnn.silu(gate)
        up = ttnn.matmul(x, self.up_proj)
        hidden = ttnn.mul(gate, up)
        output = ttnn.matmul(hidden, self.down_proj)
        return output
```

### Modified Code with Validation

```python
class MLP:
    def __init__(self, hidden_size, intermediate_size, gate_proj, up_proj, down_proj, device):
        self.gate_proj = ttnn.from_torch(gate_proj.T.unsqueeze(0).unsqueeze(0), device=device, ...)
        self.up_proj = ttnn.from_torch(up_proj.T.unsqueeze(0).unsqueeze(0), device=device, ...)
        self.down_proj = ttnn.from_torch(down_proj.T.unsqueeze(0).unsqueeze(0), device=device, ...)

        # Store original weights for validation
        self.gate_proj_torch = gate_proj
        self.up_proj_torch = up_proj
        self.down_proj_torch = down_proj

    @validate_against(
        reference_fn=lambda x, gate_w, up_w, down_w: (
            torch.nn.functional.silu(x @ gate_w) * (x @ up_w)
        ) @ down_w,
        input_map=lambda args, kwargs: (
            (
                ttnn.to_torch(args[1]).squeeze(0),
                args[0].gate_proj_torch,
                args[0].up_proj_torch,
                args[0].down_proj_torch
            ),
            {}
        ),
        output_map_impl=lambda x: ttnn.to_torch(x).squeeze(0),
        tolerances={
            'max_abs_error': 1e-1,  # Looser for accumulated matmuls
            'mean_abs_error': 1e-2,
        }
    )
    def __call__(self, x):
        gate = ttnn.matmul(x, self.gate_proj)
        gate = ttnn.silu(gate)
        up = ttnn.matmul(x, self.up_proj)
        hidden = ttnn.mul(gate, up)
        output = ttnn.matmul(hidden, self.down_proj)
        return output
```

## Step 3: Add Validation to Attention (More Complex)

For Attention, we might want to validate just the core attention computation, not the full forward pass with KV cache. Here's how:

```python
class Attention:
    def __init__(self, ...):
        # ... existing code ...

        # Store torch versions
        self.wq_torch = wq
        self.wk_torch = wk
        self.wv_torch = wv
        self.wo_torch = wo

    def _compute_attention_torch(self, xq, xk, xv, mask, scale):
        """Reference PyTorch attention for validation"""
        scores = torch.matmul(xq, xk.transpose(-2, -1)) / scale
        if mask is not None:
            scores = scores + mask
        scores = torch.nn.functional.softmax(scores, dim=-1)
        return torch.matmul(scores, xv)

    @validate_against(
        reference_fn=lambda xq, xk, xv, mask, scale, ref_fn: ref_fn(xq, xk, xv, mask, scale),
        input_map=lambda args, kwargs: (
            (
                ttnn.to_torch(args[1]).squeeze(0),  # xq
                ttnn.to_torch(args[2]).squeeze(0),  # xk
                ttnn.to_torch(args[3]).squeeze(0),  # xv
                args[4],  # mask
                math.sqrt(args[0].head_dim),  # scale
                args[0]._compute_attention_torch  # reference function
            ),
            {}
        ),
        output_map_impl=lambda x: ttnn.to_torch(x).squeeze(0),
        tolerances={'max_abs_error': 0.1}
    )
    def _attention_core(self, xq, xk, xv, mask):
        """Core attention computation (for validation)"""
        scale = math.sqrt(self.head_dim)
        scores = ttnn.matmul(xq, ttnn.transpose(xk, -2, -1))
        scores = ttnn.div(scores, scale)
        if mask is not None:
            scores = ttnn.add(scores, mask)
        scores = ttnn.softmax(scores, dim=-1)
        return ttnn.matmul(scores, xv)

    def __call__(self, x, start_pos, mask=None):
        # ... projection and reshaping code ...

        # Compute attention (validated)
        output = self._attention_core(xq_tt, xk_tt, xv_tt, mask)

        # ... rest of the code ...
```

## Step 4: Selective Validation (Recommended for Full Model)

For a full model run, validating every layer every time is expensive. Here's how to validate selectively:

```python
import os

# Enable validation via environment variable
VALIDATE = os.environ.get("TTNN_VALIDATE", "0") == "1"
# Or validate only specific layers
VALIDATE_LAYERS = [0, 1, 2]  # Only first 3 layers

class TransformerBlock:
    def __init__(self, layer_id, ...):
        self.layer_id = layer_id
        self.validate = VALIDATE and (layer_id in VALIDATE_LAYERS)

        # ... rest of init ...

        self.attention = Attention(..., validate=self.validate)
        self.mlp = MLP(..., validate=self.validate)

    def __call__(self, x, start_pos, mask=None):
        # Conditionally validate attention
        if self.validate:
            h = self._validated_attention(x, start_pos, mask)
        else:
            h = self.attention(x, start_pos, mask)

        x = ttnn.add(x, h)

        # ... rest of the code ...
```

Run with validation:
```bash
TTNN_VALIDATE=1 python ds_r1_qwen.py
```

## Step 5: Collect and Report Results

At the end of your main function, add:

```python
def main():
    # ... model setup and inference ...

    # Generate response
    response = generate(model, tokenizer, prompt, max_new_tokens=50)
    print("Response:", response)

    # Print validation report
    registry = get_validation_registry()
    if registry.results:
        print("\n")
        registry.print_report()

        # Optionally save to file
        summary = registry.get_summary()
        with open("validation_results.json", "w") as f:
            json.dump({
                "summary": summary,
                "results": [
                    {
                        "function": r.function_name,
                        "passed": r.passed,
                        "metrics": r.metrics,
                        "errors": r.errors,
                        "impl_time_ms": r.execution_time_impl * 1000,
                        "ref_time_ms": r.execution_time_ref * 1000,
                    }
                    for r in registry.results
                ]
            }, f, indent=2)
```

## Complete Example: Validated RMSNorm in Context

Here's a complete, working example you can copy-paste:

```python
class RMSNorm:
    """RMS Normalization with validation"""
    def __init__(self, weight: torch.Tensor, eps: float, device, validate: bool = False):
        self.eps = eps
        self.validate = validate

        # Convert to TTNN
        self.weight = ttnn.from_torch(
            weight.unsqueeze(0).unsqueeze(0),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT
        )

        # Store for validation
        if validate:
            self.weight_torch = weight

    @staticmethod
    def _reference_rms_norm(x, weight, eps):
        """Reference PyTorch implementation"""
        variance = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + eps)
        return weight * x_normed

    @validate_against(
        reference_fn=lambda x, w, eps: RMSNorm._reference_rms_norm(x, w, eps),
        input_map=lambda args, kwargs: (
            (
                ttnn.to_torch(args[1]).squeeze(0),
                args[0].weight_torch if hasattr(args[0], 'weight_torch') else None,
                args[0].eps
            ),
            {}
        ),
        output_map_impl=lambda x: ttnn.to_torch(x).squeeze(0),
        tolerances={
            'max_abs_error': 1e-2,
            'mean_abs_error': 1e-3,
        },
        enabled=lambda: hasattr(args[0], 'validate') and args[0].validate
    )
    def __call__(self, x):
        # TTNN implementation
        x_squared = ttnn.mul(x, x)
        mean_x_squared = ttnn.mean(x_squared, dim=-1, keepdim=True)
        rms = ttnn.sqrt(ttnn.add(mean_x_squared, self.eps))
        x_normed = ttnn.mul(x, ttnn.reciprocal(rms))
        return ttnn.mul(x_normed, self.weight)
```

## Tips for Success

1. **Start small**: Validate individual operations first (matmul, softmax, etc.)
2. **Check shapes**: Most errors come from shape mismatches in input/output mappings
3. **Use reasonable tolerances**: BF16 precision means errors of 1e-2 are normal
4. **Validate incrementally**: Add validation to one component at a time
5. **Use conditional validation**: Don't validate everything in production
6. **Keep reference code separate**: Makes it easier to verify correctness independently

## Debugging Tips

If validation fails:

1. **Print shapes**:
```python
input_map=lambda args, kwargs: (
    print(f"TTNN shape: {ttnn.to_torch(args[1]).shape}"),
    (ttnn.to_torch(args[1]).squeeze(0),),
    {}
)[1:]
```

2. **Validate reference independently**:
```python
# Test reference function separately
ref_out = RMSNorm._reference_rms_norm(x_torch, weight_torch, eps)
print(f"Reference output shape: {ref_out.shape}")
```

3. **Check intermediate values**:
```python
def __call__(self, x):
    x_squared = ttnn.mul(x, x)
    print(f"x_squared stats: {ttnn.to_torch(x_squared).mean()}, {ttnn.to_torch(x_squared).std()}")
    # ... rest of computation ...
```

4. **Use looser tolerances initially**:
```python
tolerances={'max_abs_error': 1.0}  # Very loose, just check it runs
```

Then tighten once it passes.
