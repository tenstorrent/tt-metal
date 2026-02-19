import ttnn
import tests.ttnn.unit_tests.operations.sort.utils as utils

def test_sort():
    device = utils.DeviceGetter.get_device((1, 1))
    inputs = utils.load_tensor(
        "arg0.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )

    v_0, v_1 = ttnn.sort(
        inputs,
        0,
        True,
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
