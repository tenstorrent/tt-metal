# Precision-limitation check (Refinement 2): the offset-heavy one-pass variance cancellation on a
# TILE-ALIGNED shape (unchanged code path) vs the hw_non_aligned shape, to separate it from the ragged path.
import torch, ttnn
from ttnn.operations.groupnorm_sc_N_1_HW_C import groupnorm_sc_N_1_HW_C


def ref(x, G):
    xf = x.float()
    N, _, HW, C = xf.shape
    out = torch.nn.functional.group_norm(xf.squeeze(1).permute(0, 2, 1), G)
    return out.permute(0, 2, 1).unsqueeze(1)


def pcc(a, b):
    a = a.double().flatten() - a.double().mean()
    b = b.double().flatten() - b.double().mean()
    return float((a @ b) / (a.norm() * b.norm()))


device = ttnn.open_device(device_id=0)
try:
    for shape, G in [((3, 1, 64, 96), 3), ((3, 1, 50, 96), 3)]:
        for off in [2.0, 20.0]:
            torch.manual_seed(11)
            x = torch.randn(shape)
            for n in range(shape[0]):
                x[n] += off * (n + 1) * (-1) ** n
            x = x.to(torch.bfloat16)
            for layout in [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT]:
                tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=layout, device=device)
                got = ttnn.to_torch(groupnorm_sc_N_1_HW_C(tx, G)).float()
                print(
                    f"shape={shape} off={off} layout={'TILE' if layout == ttnn.TILE_LAYOUT else 'RM'} PCC={pcc(got, ref(x, G)):.6f} maxabs={(got - ref(x, G)).abs().max():.4f}"
                )
finally:
    ttnn.close_device(device)
