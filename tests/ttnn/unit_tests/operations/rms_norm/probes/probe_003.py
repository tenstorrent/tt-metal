import torch, ttnn
from ttnn.operations.rms_norm.rms_norm_program_descriptor import blocking_plan
from ttnn.operations.rms_norm import default_compute_kernel_config

dev = ttnn.open_device(device_id=0)
try:
    for shape, dt, lay in [
        ((1, 1, 32, 4096), ttnn.float32, ttnn.TILE_LAYOUT),
        ((1, 1, 32, 4096), ttnn.bfloat16, ttnn.TILE_LAYOUT),
        ((1, 1, 32, 16384), ttnn.bfloat16, ttnn.TILE_LAYOUT),
        ((1, 1, 32, 4096), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
        ((1, 1, 32, 72), ttnn.bfloat16, ttnn.TILE_LAYOUT),
    ]:
        x = ttnn.from_torch(torch.randn(shape), dtype=dt, layout=lay, device=dev)
        g = ttnn.from_torch(torch.randn(1, 1, 1, shape[-1]), dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
        p = blocking_plan(x, g, None, dev, default_compute_kernel_config())
        print(
            shape,
            dt,
            lay,
            "->",
            p.regime,
            "Wt",
            p.Wt,
            "BLK",
            p.BLOCK_HT,
            "WR",
            p.WT_REDUCE_BLOCK,
            p.WT_REDUCE_TAIL,
            "WS",
            p.WT_SCALE_BLOCK,
            p.WT_SCALE_TAIL,
            "G",
            p.GAMMA_INGEST_BLOCK,
            "depths",
            p.IN_BUF_DEPTH,
            p.OUT_BUF_DEPTH,
            p.RM_BUF_DEPTH,
            "budget",
            p.l1_cb_budget,
        )
finally:
    ttnn.close_device(dev)
