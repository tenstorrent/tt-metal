# Real Example: Testing LLaMA-3 with Context + Builder Pattern

## Simple Example

```python
from tt_transformers_v2.testing import TestSuite

# Create and test model in one go
config = LLaMA3Config(
    hidden_dim=4096,
    num_heads=32,
    num_layers=32,
    vocab_size=32000,
)
model = LLaMA3Model(config)

# Test with auto-detected shapes!
with TestSuite(model) as suite:
    suite.test(model.attention).expect(latency_ms=5.0)
    suite.test(model.ffn).expect(latency_ms=3.0)
    suite.test(model.norm).tolerance(1e-5)

# That's it! Tests run automatically on context exit
```

## Why This Works So Well

### 1. Shape Auto-Detection

The TestSuite introspects the model instance:

```python
# Inside TestSuite, it can detect:
model.attention.hidden_dim  # → 4096
model.ffn.intermediate_dim  # → 11008
model.config.max_seq_len    # → 2048

# So you don't need to specify shapes!
```

### 2. Fluent Builder Pattern

Chain what you need, skip what you don't:

```python
suite.test("attention") \
    .tolerance(1e-3) \           # Optional: defaults to 1e-3
    .expect(latency_ms=5.0) \    # Optional: no expectations if not set
    .input_shape((2, 1024, 4096)) # Optional: override auto-detection
```

### 3. Context Manager Benefits

- **Automatic execution**: Tests run when context exits
- **Resource cleanup**: Handles GPU memory, etc.
- **Error handling**: Context manager can catch and report errors
- **Grouping**: All tests for a model in one block

## Advanced Example

```python
# Test model with different configurations
model = LLaMA3Model(config)

with TestSuite(model, batch_size=4) as suite:
    # Test attention with multiple sequence lengths
    suite.test("attention") \
        .with_seq_lengths([512, 1024, 2048, 4096]) \
        .expect_scaling("quadratic") \
        .expect(memory_mb=lambda seq_len: seq_len * 0.5)

    # Test optional components
    suite.test("sparse_attention") \
        .if_exists() \
        .tolerance(1e-4)

    # Custom validation
    suite.test("ffn") \
        .validate(lambda output: output.mean() < 1.0)

# Get detailed results
for result in suite.results:
    print(f"{result.module_name}: {result.metrics}")
```

## Comparison: Before vs After

### Before (Manual Testing)

```python
# Lots of boilerplate
def test_llama_attention():
    model = LLaMA3Model(config)
    x = torch.randn(1, 2048, 4096)  # Must specify shape
    y = model.attention(x)
    assert torch.isfinite(y).all()
    # ... performance testing code ...
    # ... memory testing code ...
```

### After (Context + Builder)

```python
# Clean and declarative
model = LLaMA3Model(config)
with TestSuite(model) as suite:
    suite.test("attention").expect(latency_ms=5.0)
```

## Integration with Model Development

```python
class LLaMA3Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attention = MultiHeadAttention(...)
        self.ffn = SwiGLU(...)

    def validate_performance(self):
        """Built-in performance validation."""
        with TestSuite(self) as suite:
            suite.test("attention").expect(
                latency_ms=5.0 if self.config.num_heads == 32 else 10.0
            )
            suite.test("ffn").expect(
                memory_mb=self.config.hidden_dim * 0.1
            )
        return suite.results

# During development
model = LLaMA3Model(config)
results = model.validate_performance()
if not all(r.passed for r in results):
    print("Performance regression detected!")
```
