import torch, ttnn
from ttnn.operations.rms_norm.perf_experiments.tree_combine import tc_descriptor as tc

device = ttnn.open_device(device_id=0)
cfg = ttnn.ComputeConfigDescriptor()
cfg.math_fidelity = ttnn.MathFidelity.HiFi2
cfg.fp32_dest_acc_en = False
cfg.math_approx_mode = False
shapes = {
    "prefill_1024": (1, 1, 8192, 1024),
    "h256_w2048": (1, 1, 256, 2048),
    "h1024_w7168": (1, 1, 1024, 7168),
    "h128_w7168": (1, 1, 128, 7168),
    "h64_w7168": (1, 1, 64, 7168),
}
for name, shape in shapes.items():
    x = ttnn.from_torch(torch.randn(shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    g = ttnn.from_torch(torch.randn((1, 1, 1, shape[-1])), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), x.dtype, x.layout, device, x.memory_config())
    for forced in (None, 32):
        lv = {"w_group": forced} if forced else None
        try:
            p = tc.blocking_plan(x, g, out, device, cfg, lv)
            print(
                f"PLAN {name:<14} force={forced} G={p.group_size} gx={p.group_x} gy={p.group_y} Wt_core={p.Wt_core} BLOCK_HT={p.BLOCK_HT} regime={p.regime} row_blocks={p.num_row_blocks} groups_used={p.groups_used}"
            )
        except Exception as e:
            print(f"PLAN {name:<14} force={forced} ERR {e}")
    x.deallocate()
    g.deallocate()
    out.deallocate()
ttnn.close_device(device)
