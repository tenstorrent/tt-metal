import torch, ttnn
from ttnn.operations.matmul import matmul


def pcc(g, c):
    a = g.flatten().double()
    b = c.flatten().double()
    a = a - a.mean()
    b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


for ash, bsh in [((256, 256), (256, 256)), ((512, 512), (512, 512)), ((1, 2, 128, 256), (256, 128))]:
    torch.manual_seed(42)
    ta = torch.randn(ash)
    tb = torch.randn(bsh)
    exp = torch.matmul(ta, tb)
    A = ttnn.from_torch(ta, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    B = ttnn.from_torch(tb, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(matmul(A, B)).float()
    rms = float(((out - exp).pow(2).mean().sqrt()) / exp.std())
    print(f"{ash}@{bsh}: PCC={pcc(exp,out):.6f} relRMS={rms:.5f}")
