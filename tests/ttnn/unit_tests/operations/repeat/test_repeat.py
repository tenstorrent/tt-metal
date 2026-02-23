import torch
import ttnn
import tests.ttnn.unit_tests.operations.repeat.utils as utils


def test_repeat():
    device = utils.DeviceGetter.get_device((1, 1))
    torch_tensor = torch.randn(1, 1, 7, 25281, 2, dtype=torch.float32)
    input_tensor = ttnn.from_torch(
        torch_tensor,
        dtype=ttnn.DataType.FLOAT32,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_result = ttnn.repeat(input_tensor, ttnn.Shape([1, 256, 1, 1, 1]))
