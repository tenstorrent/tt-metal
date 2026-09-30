import torch, ttnn
from eval.golden_tests.mhc_pre.helpers import sinkhorn_knopp
from ttnn.operations.mhc_pre import mhc_pre

device = ttnn.open_device(device_id=0)
try:
    g = torch.Generator().manual_seed(3)
    x = torch.randn((32, 128), generator=g)
    w = torch.zeros((128, 24))
    dev = lambda t: ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    tx, tw = dev(x), dev(w)
    worst = []
    for k in range(40):
        b = torch.randn((1, 24), generator=g) * 30.0
        _, _, comb = mhc_pre(tx, tw, dev(b), scale=(1.0, 1.0, 1.0))
        c = ttnn.to_torch(comb).double()[0].reshape(4, 4)
        L = b[0, 8:].reshape(1, 4, 4)
        r32 = sinkhorn_knopp(L.float(), 20, 1e-6)[0].double()
        r64 = sinkhorn_knopp(L.double(), 20, 1e-6)[0]
        dev_row = (c.sum(-1) - 1).abs().max().item()
        ref_row = (r32.sum(-1) - 1).abs().max().item()
        worst.append((dev_row - ref_row, (c - r64).abs().max().item(), (r32 - r64).abs().max().item(), ref_row))
    worst.sort(reverse=True)
    for w_ in worst[:6]:
        print("PROBE rowerr(dev)-rowerr(ref32)=%.3e  max|dev-ref64|=%.3e  max|ref32-ref64|=%.3e  ref_row=%.4f" % w_)
    print("PROBE max |dev-ref64| over all:", max(v[1] for v in worst), " max |ref32-ref64|:", max(v[2] for v in worst))
finally:
    ttnn.close_device(device)
