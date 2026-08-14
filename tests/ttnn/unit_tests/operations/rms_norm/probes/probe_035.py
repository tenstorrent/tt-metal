import ttnn, torch
from ttnn.operations.rms_norm.rms_norm import create_program_descriptor as cpd

pd = cpd.__globals__
device = ttnn.open_device(device_id=0)
try:
    bytes_ = {"in_tile": 2048, "out_tile": 2048, "gamma_tile": 2048, "stat_tile": 4096, "bf16_tile": 2048}
    for shape in [
        (1, 1, 8192, 1024),
        (1, 1, 2048, 1024),
        (1, 1, 1024, 512),
        (1, 1, 512, 256),
        (1, 1, 256, 512),
        (2, 4, 128, 256),
        (1, 1, 32, 7168),
        (1, 1, 64, 12288),
        (1, 1, 4064, 160),
        (1, 1, 3232, 96),
    ]:
        t = ttnn.from_torch(
            torch.zeros(shape, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        p = pd["_plan"](device, t, has_gamma=True, bytes_=bytes_)
        Rt, g = p["row_tiles"], p["num_row_groups"]
        base, rem = divmod(Rt, g)
        mx = base + (1 if rem else 0)
        print(
            shape,
            "g=%d s=%d S=%d B=%d in_depth=%d max_rows=%d"
            % (g, p["num_hidden_slices"], p["slice_hidden_tiles"], p["block_rows"], p["in_depth"], mx),
        )
        del t
finally:
    ttnn.close_device(device)
