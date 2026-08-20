import torch, ttnn
from ttnn.operations.rms_norm import rms_norm, default_compute_kernel_config
from ttnn.operations.rms_norm import rms_norm_program_descriptor as pd

device = ttnn.open_device(device_id=0)
cfg = ttnn.ComputeConfigDescriptor()
cfg.math_fidelity = ttnn.MathFidelity.HiFi2
cfg.fp32_dest_acc_en = False
cfg.math_approx_mode = False
for shape, dt in [
    ((1, 1, 8192, 1024), ttnn.bfloat8_b),
    ((1, 1, 8192, 1024), ttnn.bfloat16),
    ((1, 1, 4096, 1024), ttnn.bfloat8_b),
    ((1, 1, 32, 7168), ttnn.bfloat16),
    ((1, 1, 100, 736), ttnn.bfloat16),
]:
    x = ttnn.from_torch(torch.rand(shape), dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
    g = ttnn.from_torch(torch.rand((1, 1, 1, shape[-1])), dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
    o = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), dt, ttnn.TILE_LAYOUT, device, x.memory_config())
    p = pd.blocking_plan(x, g, o, device, cfg)
    print(
        "PLAN",
        shape,
        dt,
        "regime",
        p.regime,
        "BLOCK_HT",
        p.BLOCK_HT,
        "rowblocks",
        p.num_row_blocks,
        "G",
        p.group_size,
        "gy",
        p.combine_gy,
    )
ttnn.close_device(device)
