import torch
import ttnn
from tests.ttnn.unit_tests.operations.eltwise.backward.utility_funcs import compare_pcc, data_gen_with_range
from tests.ttnn.utils_for_testing import check_with_pcc
import math

input_shapes = [
    torch.Size([1, 1, 512, 512]),
    torch.Size([1, 1, 1024, 1024]),
    torch.Size([1, 3, 2048, 2048]),
]
# input_shape = torch.Size([1, 1, 1024, 1024])
bench_funcs = [
    (ttnn.gelu, "gelu"),
    (ttnn.silu, "silu"),
    (ttnn.tanh, "tanh"),
]

device_id = 0
dispatch_core_type = ttnn.device.DispatchCoreType.ETH
device = ttnn.open_device(
    device_id=device_id, l1_small_size=8192, dispatch_core_config=ttnn.device.DispatchCoreConfig(dispatch_core_type)
)
ttnn.enable_program_cache(device)

try:
    for func, name in bench_funcs:
        for input_shape in input_shapes:
            _, tt_input = data_gen_with_range(input_shape, -100, 100, device, True)
            # Make sure that the function is compiled and change of arguments works
            _ = func(tt_input)
    for func, name in bench_funcs:
        for input_shape in input_shapes:
            torch_input, tt_input = data_gen_with_range(input_shape, -50, 50, device, True)
            for i in range(100):
                tt_output_device = func(tt_input)
            golden_func = ttnn.get_golden_function(func)
            golden_tensor = golden_func(torch_input)

            tt_output_device_torch = ttnn.to_torch(tt_output_device)

            res = check_with_pcc(golden_tensor, tt_output_device_torch, 0.999)
            print(f"> {name} pcc: {res}, shape: {input_shape}")

finally:
    ttnn.close_device(device)
