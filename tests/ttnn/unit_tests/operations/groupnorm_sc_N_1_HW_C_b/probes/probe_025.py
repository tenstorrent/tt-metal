import torch, ttnn
from eval.metrics import compute_metrics
from eval.golden_tests.groupnorm_sc_N_1_HW_C_b.helpers import pytorch_groupnorm_sc_N_1_HW_C
from ttnn.operations.groupnorm_sc_N_1_HW_C_b import groupnorm_sc_N_1_HW_C_b

dev = ttnn.open_device(device_id=0)
TD = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}
for shape, g in [((1, 1, 17, 64), 1), ((1, 1, 47, 256), 1), ((2, 1, 100, 128), 1), ((4, 1, 6360, 768), 32)]:
    for dt in (ttnn.bfloat16, ttnn.float32):
        for lay in (ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT):
            torch.manual_seed(0)
            C = shape[-1]
            x = torch.randn(shape).to(TD[dt])
            gm = torch.randn(1, 1, 1, C).to(TD[dt])
            bt = torch.randn(1, 1, 1, C).to(TD[dt])
            exp = pytorch_groupnorm_sc_N_1_HW_C(x, g, gamma=gm, beta=bt)
            mk = lambda t, l: ttnn.from_torch(t, dtype=dt, layout=l, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            out = groupnorm_sc_N_1_HW_C_b(
                mk(x, lay), g, gamma=mk(gm, ttnn.ROW_MAJOR_LAYOUT), beta=mk(bt, ttnn.ROW_MAJOR_LAYOUT)
            )
            m = compute_metrics(out, exp)
            print(f"R5 {shape} g{g} {dt} {lay}: pcc={m.pcc:.6f} rel_rms={m.rms:.5f}")
ttnn.close_device(dev)
