import torch, ttnn
from ttnn.operations.g6_zeros_like import g6_zeros_like_program_descriptor as pd
from ttnn.operations.g6_zeros_like import g6_zeros_like

pd.ZERO_SCRATCH_PAGES = 2
device = ttnn.open_device(device_id=0)
try:
    for shape, dt, tdt in (
        ((1, 12, 128, 128), ttnn.float32, torch.float32),
        ((5, 1, 32, 256), ttnn.bfloat8_b, torch.bfloat16),
    ):
        o = ttnn.from_torch(
            torch.ones(shape, dtype=tdt),
            dtype=dt,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.generic_op([o], pd.get_program_descriptor(o))
        t = ttnn.to_torch(o)
        bad = int((t.float().view(torch.int32) != 0).sum()) if dt == ttnn.float32 else int((t.float() != 0).sum())
        print("PROBE", dt, "scratch_pages=2 nonzero", bad, "of", t.numel())
finally:
    ttnn.close_device(device)
