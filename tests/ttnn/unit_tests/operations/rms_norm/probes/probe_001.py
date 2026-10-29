import torch, ttnn
from ttnn.operations.rms_norm import rms_norm_program_descriptor as pd

dev = ttnn.open_device(device_id=0)
try:
    print(
        "OUT unreserved",
        ttnn.get_max_worker_l1_unreserved_size(),
        "l1_size",
        dev.l1_size_per_core() if hasattr(dev, "l1_size_per_core") else None,
    )
    for shape in [(128, 8192), (1, 1, 32, 8192)]:
        x = ttnn.from_torch(torch.randn(shape, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=dev)
        b = pd.select_blocking(x, None)
        print(
            "OUT",
            shape,
            b.block_row_tiles,
            b.w_chunk_tiles,
            b.in_depth,
            pd._footprint(
                b.block_row_tiles,
                b.w_chunk_tiles,
                b.in_depth,
                Wt=b.tensor_w_tiles,
                P=0,
                xt=2048,
                ot=2048,
                gamma_stick_bytes=0,
                gt=0,
                has_gamma=False,
            ),
        )
finally:
    ttnn.close_device(dev)
