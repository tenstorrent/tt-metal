import ttnn
import tests.ttnn.unit_tests.operations.reshape.utils as utils

def test_reshape():
    utils_DeviceGetter_get_device_12 = utils.DeviceGetter.get_device((1, 1))
    ttnn_ones = ttnn.ones(
        shape=ttnn.Shape([8, 32, 17821, 4]),
        dtype=ttnn.DataType.INT32,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_12,
    )

    ttnn_reshape = ttnn.reshape(
    ttnn_ones,
    [8, 32, 17821, 4, 1],
    memory_config=ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
    ))
    return ttnn_reshape

if __name__ == "__main__":
    test_reshape()
