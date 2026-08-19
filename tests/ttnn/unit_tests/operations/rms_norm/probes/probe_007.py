import torch, ttnn
from ttnn.operations.rms_norm.rms_norm_program_descriptor import blocking_plan
from ttnn.operations.rms_norm import default_compute_kernel_config

dev = ttnn.open_device(device_id=0)
try:
    for name, (shape, dt, lay) in {
        "grid_filling": ((1, 1, 8192, 1024), ttnn.bfloat16, ttnn.TILE_LAYOUT),
        "wide_prefill": ((1, 1, 8192, 7168), ttnn.bfloat16, ttnn.TILE_LAYOUT),
        "grid_starved": ((1, 1, 32, 7168), ttnn.bfloat16, ttnn.TILE_LAYOUT),
        "smallest": ((32, 17), ttnn.bfloat16, ttnn.TILE_LAYOUT),
        "row_major": ((1, 1, 8192, 1024), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
    }.items():
        x = ttnn.from_torch(torch.randn(shape), dtype=dt, layout=lay, device=dev)
        g = ttnn.from_torch(torch.randn(1, 1, 1, shape[-1]), dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
        for lv in [None, dict(double_buffer=0)]:
            p = blocking_plan(x, g, None, dev, default_compute_kernel_config(), lv)
            print(
                f"{name:<14} lev={'ON ' if lv is None else 'C16off'} regime={p.regime} Rt={p.Rt} Wt={p.Wt} "
                f"BLK={p.BLOCK_HT} WR={p.WT_REDUCE_BLOCK} G={p.GAMMA_INGEST_BLOCK} "
                f"depths(in/out/rm)={p.IN_BUF_DEPTH}/{p.OUT_BUF_DEPTH}/{p.RM_BUF_DEPTH} blocks={p.num_row_blocks}"
            )
finally:
    ttnn.close_device(dev)
