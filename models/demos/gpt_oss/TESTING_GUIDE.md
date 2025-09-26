# GPT-OSS Testing Guide

**Minimal guide for testing distributed MoE systems**

---

## 🚀 Quick Start

```bash
cd tt-metal
source .venv/bin/activate

# Run all tests
pytest models/demos/gpt_oss/tests/unit/ -v

# Run specific test
pytest models/demos/gpt_oss/tests/unit/test_core_components.py::test_experts_core -v -s
```

---

## 🏗️ Test Architecture

**Ultra-minimal design:**
- **5 files, 1,054 lines** (was 14 files, 2,392 lines)
- **Zero duplication** - TestFactory handles all setup
- **Universal compatibility** - works with any model/mesh

```
tests/
├── conftest.py              # Shared fixtures
├── test_factory.py          # Universal setup
└── unit/
    ├── test_core_components.py   # MoE, attention, router
    ├── test_integration.py       # Full pipeline
    └── test_utilities.py         # RoPE, SDPA, inference
```

---

## 🧪 TestFactory Pattern

**One line replaces 50+ lines of boilerplate:**

```python
from ..test_factory import TestFactory, parametrize_mesh

@parametrize_mesh(["1x8", "4x4"])  # Test multiple mesh configs
def test_my_component(mesh_device, reset_seeds):
    # Single line setup
    setup = TestFactory.setup_test(mesh_device)

    # Everything you need:
    config = setup["config"]              # Model config
    mesh_config = setup["mesh_config"]    # MeshConfig(1x8, tp=8)
    state_dict = setup["state_dict"]      # Random weights
    ccl_manager = setup["ccl_manager"]    # Communication
    dtype = setup["dtype"]               # ttnn.bfloat8_b

    # Test your component
    component = YourComponent(
        setup["mesh_device"], config, state_dict, ccl_manager,
        mesh_config=mesh_config
    )

    # Test forward pass
    input_tensor = ttnn.from_torch(
        torch.randn(1, 32, config.hidden_size),
        device=setup["mesh_device"], layout=ttnn.TILE_LAYOUT, dtype=dtype
    )
    output = component(input_tensor)
    assert output.shape == input_tensor.shape
```

---

## 🎯 Test Templates

### Component Test
```python
@parametrize_mesh(["1x8", "4x4"])
def test_component(mesh_device, reset_seeds):
    setup = TestFactory.setup_test(mesh_device)

    component = YourComponent(
        setup["mesh_device"], setup["config"], setup["state_dict"],
        setup["ccl_manager"], mesh_config=setup["mesh_config"]
    )

    hidden_states = torch.randn(1, 32, setup["config"].hidden_size)
    tt_input = ttnn.from_torch(hidden_states, device=setup["mesh_device"],
                               layout=ttnn.TILE_LAYOUT, dtype=setup["dtype"])

    output = component(tt_input)
    assert output.shape == tt_input.shape
```

### Multi-Scale Test
```python
@pytest.mark.parametrize("model_scale", [
    {"hidden_size": 2048, "num_experts": 64},   # GPT-20B
    {"hidden_size": 4096, "num_experts": 128},  # GPT-120B
])
def test_scalability(mesh_device, model_scale, reset_seeds):
    setup = TestFactory.setup_test(mesh_device)

    # Override config
    for key, value in model_scale.items():
        setattr(setup["config"], key, value)

    # Generate scaled state dict
    state_dict = TestFactory._generate_dummy_state_dict(setup["config"])

    # Test with scaled config
    component = YourComponent(setup["mesh_device"], setup["config"], state_dict,
                             setup["ccl_manager"], mesh_config=setup["mesh_config"])
```

---

## 🔍 Debugging

### Shape Debugging
```python
def test_debug_shapes(mesh_device, reset_seeds):
    setup = TestFactory.setup_test(mesh_device)

    # Log key info
    logger.info(f"Mesh: {setup['mesh_config']}")
    logger.info(f"Hidden size: {setup['config'].hidden_size}")

    input_tensor = ttnn.from_torch(torch.randn(1, 32, setup["config"].hidden_size),
                                   device=setup["mesh_device"], layout=ttnn.TILE_LAYOUT,
                                   dtype=setup["dtype"])
    logger.info(f"Input shape: {input_tensor.shape}")

    output = component(input_tensor)
    logger.info(f"Output shape: {output.shape}")
```

### Performance Check
```python
import time

def test_performance(mesh_device, reset_seeds):
    setup = TestFactory.setup_test(mesh_device)
    component = YourComponent(setup["mesh_device"], setup["config"], setup["state_dict"],
                             setup["ccl_manager"], mesh_config=setup["mesh_config"])

    input_tensor = ttnn.from_torch(torch.randn(1, 32, setup["config"].hidden_size),
                                   device=setup["mesh_device"], layout=ttnn.TILE_LAYOUT,
                                   dtype=setup["dtype"])

    # Warmup
    for _ in range(3):
        output = component(input_tensor)

    # Benchmark
    start = time.time()
    for _ in range(10):
        output = component(input_tensor)
    avg_time = (time.time() - start) / 10

    logger.info(f"Average time: {avg_time*1000:.2f}ms")
    assert avg_time < 0.1  # Should be fast
```

---

## 🚨 Common Issues

### Import Errors
```bash
# Fix: Set PYTHONPATH
export PYTHONPATH="/home/models-team/sraizada/tt-metal:$PYTHONPATH"
cd /home/models-team/sraizada/tt-metal
```

### Memory Issues
```python
# Use smaller inputs for memory-constrained tests
small_input = torch.randn(1, 16, setup["config"].hidden_size)  # 16 vs 128
```

### Shape Mismatches
```python
# Always verify shapes
assert input_tensor.shape[-1] == setup["config"].hidden_size
assert output.shape == input_tensor.shape
```

---

## 📈 Best Practices

### ✅ DO
- **Use TestFactory.setup_test()** for all tests
- **Test multiple mesh configs** with `@parametrize_mesh`
- **Verify tensor shapes** before/after operations
- **Use descriptive test names**

### ❌ DON'T
- **Don't hardcode mesh shapes** - use parametrized tests
- **Don't hardcode dimensions** - read from config
- **Don't duplicate setup** - always use TestFactory
- **Don't write complex tests** - keep them focused

---

## ⚡ Key Commands

```bash
# Run specific test categories
pytest models/demos/gpt_oss/tests/unit/test_core_components.py -v  # Components
pytest models/demos/gpt_oss/tests/unit/test_integration.py -v      # Integration
pytest models/demos/gpt_oss/tests/unit/test_utilities.py -v        # Utilities

# Debug options
pytest -v -s                # Verbose with prints
pytest -x                   # Stop on first failure
pytest -k "1x8"            # Only 1x8 mesh tests

# Environment
export GPT_DIR="/mnt/MLPerf/tt_dnn-models/tt/GPT-OSS-20B"  # For real weights
export LOGURU_LEVEL=INFO    # Logging level
```

---

## 🎯 Test Coverage

**17 focused tests cover everything:**
- **Core MoE**: Experts, Router, MLP pipeline
- **Attention**: Multi-head, RoPE, SDPA
- **Integration**: Full layers, model construction
- **Flexibility**: Any mesh config, any model size
- **Performance**: Memory efficiency, CCL communication

**Result: Comprehensive coverage with minimal code (1,054 lines vs 2,392 original)**

---

**Ready to write efficient tests for distributed MoE systems!** 🚀
