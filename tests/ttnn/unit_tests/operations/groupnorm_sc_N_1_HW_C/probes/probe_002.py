import torch, ttnn
from ttnn.operations.groupnorm_sc_N_1_HW_C import groupnorm_sc_N_1_HW_C


def ref(x, G, gm, bt, eps=1e-5):
    N, _, HW, C = x.shape
    xn = x.to(torch.float32).squeeze(1).permute(0, 2, 1)
    w = gm.to(torch.float32).reshape(C) if gm is not None else None
    b = bt.to(torch.float32).reshape(C) if bt is not None else None
    y = torch.nn.functional.group_norm(xn, G, weight=w, bias=b, eps=eps)
    return y.permute(0, 2, 1).unsqueeze(1)


device = ttnn.open_device(device_id=0)
try:
    for shape, G in [((1, 1, 64, 128), 4), ((2, 1, 64, 288), 9), ((1, 1, 32, 1024), 32)]:
        torch.manual_seed(0)
        N, _, HW, C = shape
        x = torch.randn(shape, dtype=torch.bfloat16)
        gm = torch.randn(1, 1, 1, C, dtype=torch.bfloat16)
        bt = torch.randn(1, 1, 1, C, dtype=torch.bfloat16)
        tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        tg = ttnn.from_torch(gm, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        tb = ttnn.from_torch(bt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        out = ttnn.to_torch(groupnorm_sc_N_1_HW_C(tx, G, gamma=tg, beta=tb)).to(torch.float32)
        exp = ref(x, G, gm, bt)
        pcc = torch.corrcoef(torch.stack([out.flatten(), exp.flatten()]))[0, 1].item()
        print(shape, "TILE-affine pcc:", round(pcc, 6))
finally:
    ttnn.close_device(device)
