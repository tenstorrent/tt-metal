import torch, ttnn
from ttnn.operations.rms_norm.perf_experiments.tree_combine import tc_descriptor as tc

device = ttnn.open_device(device_id=0)
cfg = ttnn.ComputeConfigDescriptor()
cfg.math_fidelity = ttnn.MathFidelity.HiFi2
cfg.fp32_dest_acc_en = False
cfg.math_approx_mode = False

shapes = {
    "focus": (1, 1, 32, 7168),
    "prefill_7168": (1, 1, 8192, 7168),
    "decode_1024": (1, 1, 32, 1024),
    "decode_2304": (1, 1, 32, 2304),
    "decode_5120": (1, 1, 32, 5120),
    "rm_in_7168": (1, 1, 1024, 7168),
}
for name, shape in shapes.items():
    rm = name.startswith("rm_")
    lay = ttnn.ROW_MAJOR_LAYOUT if rm else ttnn.TILE_LAYOUT
    x = ttnn.from_torch(torch.randn(shape), dtype=ttnn.bfloat16, layout=lay, device=device)
    g = ttnn.from_torch(
        torch.randn((1, 1, 1, shape[-1])),
        dtype=ttnn.bfloat16,
        layout=(ttnn.ROW_MAJOR_LAYOUT if rm else ttnn.TILE_LAYOUT),
        device=device,
    )
    out = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), x.dtype, x.layout, device, x.memory_config())
    p = tc.blocking_plan(x, g, out, device, cfg, None)
    print(
        f"PLAN {name:<14} G={p.group_size} gx={p.group_x} gy={p.group_y} Wt={p.Wt} Wt_core={p.Wt_core} "
        f"BLOCK_HT={p.BLOCK_HT} regime={p.regime} row_blocks={p.num_row_blocks} groups_used={p.groups_used}"
    )
    x.deallocate()
    g.deallocate()
    out.deallocate()
ttnn.close_device(device)
