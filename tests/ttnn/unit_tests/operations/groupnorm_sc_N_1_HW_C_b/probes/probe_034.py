import torch, ttnn
from eval.golden_tests.groupnorm_sc_N_1_HW_C_b.helpers import pytorch_groupnorm_sc_N_1_HW_C
from ttnn.operations.groupnorm_sc_N_1_HW_C_b import groupnorm_sc_N_1_HW_C_b

device = ttnn.open_device(device_id=0)
B8, BF16, F32 = ttnn.bfloat8_b, ttnn.bfloat16, ttnn.float32
TILE = ttnn.TILE_LAYOUT


def metrics(got, exp):
    g, e = got.float().flatten(), exp.float().flatten()
    pcc = torch.corrcoef(torch.stack([g, e]))[0, 1].item()
    rms = (g - e).pow(2).mean().sqrt().item() / e.std().item()
    return pcc, rms


def run(shape, G, offs=0.0, quant_ref=False):
    torch.manual_seed(0)
    N, _, HW, C = shape
    Cg = C // G
    x = torch.randn(shape)
    if offs:
        x = x + torch.linspace(-offs, offs, G).repeat_interleave(Cg)
    x = x.to(torch.bfloat16)
    g = torch.randn(1, 1, 1, C).to(torch.bfloat16)
    b = torch.randn(1, 1, 1, C).to(torch.bfloat16)
    tx = ttnn.from_torch(x, dtype=B8, layout=TILE, device=device)
    tg = ttnn.from_torch(g, dtype=B8, layout=TILE, device=device)
    tb = ttnn.from_torch(b, dtype=B8, layout=TILE, device=device)
    if quant_ref:
        x, g, b = ttnn.to_torch(tx).float(), ttnn.to_torch(tg).float(), ttnn.to_torch(tb).float()
    exp = pytorch_groupnorm_sc_N_1_HW_C(x.float(), G, gamma=g.float(), beta=b.float())
    out = ttnn.to_torch(groupnorm_sc_N_1_HW_C_b(tx, G, gamma=tg, beta=tb))
    p, r = metrics(out, exp)
    print(f"M {shape} G{G} offs={offs} qref={quant_ref}: PCC={p:.6f} relRMS={r:.4f}")


for shape, G in [
    ((1, 1, 256, 256), 8),
    ((2, 1, 512, 128), 4),
    ((1, 1, 64, 1280), 32),
    ((2, 1, 100, 128), 4),
    ((1, 1, 64, 100), 2),
    ((4, 1, 6360, 768), 32),
    ((1, 1, 16384, 512), 32),
]:
    run(shape, G)
for shape, G in [((2, 1, 40, 256), 32), ((1, 1, 47, 320), 32), ((2, 1, 64, 256), 16)]:
    run(shape, G, offs=12.0, quant_ref=True)
    run(shape, G, offs=12.0, quant_ref=False)
ttnn.close_device(device)
