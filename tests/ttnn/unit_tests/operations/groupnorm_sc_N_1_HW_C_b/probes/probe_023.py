import torch, ttnn
from eval.golden_tests.groupnorm_sc_N_1_HW_C_b.helpers import pytorch_groupnorm_sc_N_1_HW_C
from ttnn.operations.groupnorm_sc_N_1_HW_C_b import groupnorm_sc_N_1_HW_C_b

dev = ttnn.open_device(device_id=0)
try:
    for shape, G in [((1, 1, 256, 256), 8), ((1, 1, 4096, 320), 32), ((1, 1, 64, 1280), 32), ((1, 1, 16384, 512), 32)]:
        for layout in (ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT):
            for shift in (0.0, 10.0):
                torch.manual_seed(0)
                x = torch.randn(shape) + shift
                C = shape[-1]
                g = torch.randn(1, 1, 1, C)
                b = torch.randn(1, 1, 1, C)
                ref = pytorch_groupnorm_sc_N_1_HW_C(x, G, gamma=g, beta=b)
                tx = ttnn.from_torch(x, dtype=ttnn.float32, layout=layout, device=dev)
                tg = ttnn.from_torch(g, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
                tb = ttnn.from_torch(b, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
                y = ttnn.to_torch(groupnorm_sc_N_1_HW_C_b(tx, G, gamma=tg, beta=tb)).float()
                d = (y - ref).float()
                rrms = (d.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
                pcc = torch.corrcoef(torch.stack([y.flatten(), ref.flatten()]))[0, 1].item()
                print(
                    f"{shape} G={G} {'TILE' if layout==ttnn.TILE_LAYOUT else 'RM'} shift={shift}: pcc={pcc:.7f} rel_rms={rrms:.3e} maxabs={d.abs().max().item():.3e} dtype={y.dtype}"
                )
finally:
    ttnn.close_device(dev)
