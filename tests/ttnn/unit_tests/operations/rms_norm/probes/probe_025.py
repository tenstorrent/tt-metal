import ttnn, torch
from ttnn.operations.rms_norm.rms_norm_program_descriptor import blocking_plan
from ttnn.operations.rms_norm._bench_rms_norm import _cfg_loose, _cfg_default

dev = ttnn.open_device(device_id=0)
print("budget:", ttnn.get_max_worker_l1_unreserved_size(), "grid:", dev.compute_with_storage_grid_size())
for name, shape, gl in [
    ("focus", (1, 1, 32, 7168), ttnn.TILE_LAYOUT),
    ("prefill_7168", (1, 1, 8192, 7168), ttnn.TILE_LAYOUT),
    ("prefill_1024", (1, 1, 8192, 1024), ttnn.TILE_LAYOUT),
    ("decode_1024", (1, 1, 32, 1024), ttnn.TILE_LAYOUT),
]:
    x = ttnn.from_torch(torch.randn(shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    g = ttnn.from_torch(torch.randn((1, 1, 1, shape[-1])), dtype=ttnn.bfloat16, layout=gl, device=dev)
    for cn, cfg in (("loose", _cfg_loose()),):
        p = blocking_plan(x, g, x, dev, cfg)
        print(
            f"{name}/{cn}: regime={p.regime} Rt={p.Rt} Wt={p.Wt} BLOCK_HT={p.BLOCK_HT} WT_RED={p.WT_REDUCE_BLOCK} "
            f"WT_SCALE={p.WT_SCALE_BLOCK} DEST={p.DEST_BLOCK} in_d={p.IN_BUF_DEPTH} out_d={p.OUT_BUF_DEPTH} "
            f"nblocks={p.num_row_blocks} ws={p.working_set_bytes()} budget={p.l1_cb_budget} acc={p.acc_dtype} interm={p.interm_dtype}"
        )
        print("   cbs:", [(i, n, b) for i, n, b, k in p.cb_layout])
    ttnn.deallocate(x)
    ttnn.deallocate(g)
ttnn.close_device(dev)
