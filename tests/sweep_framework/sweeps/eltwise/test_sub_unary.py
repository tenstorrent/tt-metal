import ttnn
import torch
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from models.utility_functions import torch_random


device_id = 0
device = ttnn.open_device(device_id=device_id)


input_shape_a = [1, 1, 32, 32]
input_shape_b = [1, 1, 32, 32]

input_a_dtype, input_b_dtype = ttnn.bfloat16, ttnn.bfloat16
input_a_layout, input_b_layout = ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT
input_a_memory_config, input_b_memory_config = ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG
output_memory_config = ttnn.DRAM_MEMORY_CONFIG

torch_input_tensor_a = torch_random(input_shape_a, -100, 100, dtype=torch.bfloat16)
torch_input_tensor_b = torch_random(1, -100, 100, dtype=torch.bfloat16).item()

torch_output_tensor = torch.sub(torch_input_tensor_a, torch_input_tensor_b)


input_tensor_a = ttnn.from_torch(
    torch_input_tensor_a,
    dtype=input_a_dtype,
    layout=input_a_layout,
    device=device,
    memory_config=input_a_memory_config,
)

input_tensor_b = torch_input_tensor_b

output_tensor = ttnn.subtract(input_tensor_a, input_tensor_b, memory_config=output_memory_config)
output_tensor = ttnn.to_torch(output_tensor)

passing, output_str = comp_pcc(torch_output_tensor, output_tensor)
print(output_str)

ttnn.close_device(device)
