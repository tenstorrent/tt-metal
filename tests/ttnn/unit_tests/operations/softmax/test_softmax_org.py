import ttnn
import tests.ttnn.unit_tests.operations.softmax.utils as utils

_CONST_EVAL_CACHE = {}


def _main(input):
    ttnn_to_layout_0 = ttnn.to_layout(
        input[0], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(input[0], False)
    ttnn_softmax_0 = ttnn.softmax(
        ttnn_to_layout_0,
        2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        numeric_stable=True,
    )
    ttnn.deallocate(ttnn_to_layout_0, False)
    util_create_list_0 = [ttnn_softmax_0]
    return util_create_list_0


def create_inputs_for__main():
    utils_DeviceGetter_get_device_0 = utils.DeviceGetter.get_device((1, 1))
    ttnn_ones_0 = ttnn.ones(
        shape=ttnn.Shape([1, 100, 6800]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.ROW_MAJOR,
        device=utils_DeviceGetter_get_device_0,
    )
    util_create_list_1 = [ttnn_ones_0]
    return util_create_list_1


def test_softmax_org():
    create_inputs_for__main_0 = create_inputs_for__main()
    _main_0 = _main(create_inputs_for__main_0)
   
