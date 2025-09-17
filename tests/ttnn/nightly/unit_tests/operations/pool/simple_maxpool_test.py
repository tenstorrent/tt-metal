import torch
import ttnn
import math

device = ttnn.CreateDevice(0, l1_small_size=8192)

in_n = 1
in_h = 2
in_w = 2
in_c = 32
kernel_size = [2, 2]
stride = [1, 1]
padding = [0, 0]
dilation = [1, 1]
shard_scheme = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
ceil_mode = False
ttnn_dtype = ttnn.bfloat16
# ttnn_dtype = ttnn.bfloat8_b

tensor_shape = (in_n, in_c, in_h, in_w)  # NCHW format

# Create tensor filled with height and width coordinates
torch.manual_seed(0)
# torch_input = torch.randn(tensor_shape, dtype=torch.bfloat16)

# Create tensor where each element equals its HW coordinate (h * in_w + w)
# torch_input = torch.randn(tensor_shape, dtype=torch.bfloat16)
torch_input = torch.zeros(tensor_shape, dtype=torch.bfloat16)
for n in range(in_n):
    for c in range(in_c):
        for h in range(in_h):
            for w in range(in_w):
                coordinate_value = 1
                torch_input[n, c, h, w] = 1

ttnn_input_shape = (1, 1, in_n * in_h * in_w, in_c)
torch_input_permuted = torch.permute(torch_input, (0, 2, 3, 1))  # N, H, W, C
torch_input_reshaped = torch_input_permuted.reshape(ttnn_input_shape)  # NHW, C
ttnn_layout = ttnn.ROW_MAJOR_LAYOUT
if ttnn_dtype == ttnn.bfloat8_b:
    ttnn_layout = ttnn.TILE_LAYOUT
ttnn_input = ttnn.from_torch(torch_input_reshaped, ttnn_dtype, layout=ttnn_layout, device=device)

# print("Output without indices:")
# print(ttnn.to_torch(ttnn_output))

ttnn_output, indices = ttnn.max_pool2d(
    input_tensor=ttnn_input,
    batch_size=in_n,
    input_h=in_h,
    input_w=in_w,
    channels=in_c,
    kernel_size=kernel_size,
    stride=stride,
    padding=padding,
    dilation=dilation,
    applied_shard_scheme=shard_scheme,
    return_indices=True,
    ceil_mode=ceil_mode,
)
