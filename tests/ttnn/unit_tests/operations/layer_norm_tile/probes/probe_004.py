import torch, ttnn
from ttnn.operations.layer_norm import layer_norm

device = ttnn.open_device(device_id=0)
try:
    g = layer_norm.__globals__
    for shape in [(16, 1, 128, 768), (8, 1, 512, 768), (32, 1, 32, 768)]:
        x = ttnn.from_torch(
            torch.randn(shape).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        b = g["derive_blocking"](x, has_residual=True, fp32_dest_acc_en=True)
        print(
            "PROBE",
            shape,
            "block_height",
            b.block_height,
            "input_depth",
            b.input_depth,
            "res_on_writer",
            b.residual_on_writer,
            "split_x",
            b.split_x_nocs,
            "cores",
            len(b.assignment),
        )
finally:
    ttnn.close_device(device)
