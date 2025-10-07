# GPT-OSS: batch=1 inference

Inference for GPT-OSS models on Tenstorrent Wormhole devices LoudBox and Galaxy.
This model is under active development.
Currently we have only support prefill upto sequence length 128 and batch=1.

## Quick Start

```bash
# Set model path
export GPT_DIR="/mnt/MLPerf/tt_dnn-models/tt/GPT-OSS-20B"

# Run demo
cd tt-metal/models/demos/gpt_oss/demo
pytest simple_text_demo.py
```

## Configuration

### Model Selection
```bash
# GPT-OSS-20B (faster)
export GPT_DIR="/mnt/MLPerf/tt_dnn-models/tt/GPT-OSS-20B"

# GPT-OSS-120B (higher quality)
export GPT_DIR="/mnt/MLPerf/tt_dnn-models/tt/GPT-OSS-120B"
```

## Testing

```bash
# Run all tests
pytest models/demos/gpt_oss/tests/unit/ -v

```

### Writing Tests
```python
from models.demos.gpt_oss.tests.test_factory import TestFactory, parametrize_mesh_with_fabric

@parametrize_mesh_with_fabric()
def test_my_component(mesh_device, device_params, reset_seeds):
    setup = TestFactory.setup_test(mesh_device, use_real_weights=False)

    component = YourComponent(
        setup["mesh_device"],
        setup["config"],
        setup["state_dict"],
        setup["ccl_manager"],
        mesh_config=setup["mesh_config"]
    )

    input_tensor = ttnn.from_torch(
        torch.randn(1, 32, setup["config"].hidden_size),
        device=setup["mesh_device"],
        layout=ttnn.TILE_LAYOUT,
        dtype=setup["dtype"]
    )
    output = component(input_tensor)
    assert output.shape == input_tensor.shape
```
