import torch
import ttnn

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
tensor_shape = (in_n, in_c, in_h, in_w)  # NCHW format
torch_input = torch.ones(tensor_shape, dtype=torch.bfloat16)

ttnn_input_shape = (1, 1, in_n * in_h * in_w, in_c)
torch_input_permuted = torch.permute(torch_input, (0, 2, 3, 1))  # N, H, W, C
torch_input_reshaped = torch_input_permuted.reshape(ttnn_input_shape)  # NHW, C
ttnn_input = ttnn.from_torch(torch_input_reshaped, ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

# ttnn_output = ttnn.max_pool2d(
#     input_tensor=ttnn_input,
#     batch_size=in_n,
#     input_h=in_h,
#     input_w=in_w,
#     channels=in_c,
#     kernel_size=kernel_size,
#     stride=stride,
#     padding=padding,  # ttnn is padding in the order (top, bottom, left, right)
#     dilation=dilation,
#     applied_shard_scheme=shard_scheme,
# )

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
    padding=padding,  # ttnn is padding in the order (top, bottom, left, right)
    dilation=dilation,
    applied_shard_scheme=shard_scheme,
    return_indices=True,
)

print("\nOutput with indices:")
print(ttnn.to_torch(ttnn_output))
print("Indices:")
print(ttnn.to_torch(indices))
