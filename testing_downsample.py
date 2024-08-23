import math
import torch
import ttnn


device_id = 0
device = ttnn.open_device(device_id=device_id)


size = [8, 256, 56, 56]
batch_size, input_channels, input_width, input_height = 8, 256, 56, 56
stride_h, stride_w = 2, 2
output_height = math.ceil(input_height / stride_h)
output_width = math.ceil(input_width / stride_w)

A_pyt = torch.normal(mean=0, std=0.1, size=size).bfloat16()
A_pyt_nhwc = torch.permute(A_pyt, (0, 2, 3, 1))
A_pyt_nhwc = A_pyt_nhwc.reshape(1, 1, batch_size * input_height * input_width, input_channels)


# golden
out_golden = torch.nn.functional.max_pool2d(A_pyt, 1, stride=stride_h)
out_golden_2d_nhwc = torch.permute(out_golden, (0, 2, 3, 1)).reshape(
    1, 1, batch_size * output_height * output_width, input_channels
)

# tt
A_tt_nhwc = ttnn.from_torch(
    torch.permute(A_pyt, (0, 2, 3, 1)), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
)

out_tt = ttnn.downsample(A_tt_nhwc, [batch_size, 1, 1, stride_h, stride_w])

ttnn.close_device(device)


print(out_tt)
