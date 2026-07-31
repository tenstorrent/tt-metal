"""Format floor: how much PCC do bfp4 weights / bfp8 h / bfp8 out cost in TORCH alone?"""
import torch, ttnn
from tests.ttnn.utils_for_testing import comp_pcc

device = ttnn.open_device(device_id=0)
try:

    def rt(t, dt):
        tt = ttnn.from_torch(
            t.to(torch.bfloat16),
            dtype=dt,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return ttnn.to_torch(tt).to(torch.float32)

    torch.manual_seed(42)
    emb, capacity, count, HID = 7168, 1024, 255, 2048
    x = torch.randn((capacity, emb), dtype=torch.float32)
    wg = torch.randn((emb, HID))
    wu = torch.randn((emb, HID))
    wd = torch.randn((HID, emb))
    xr = x[:count].to(torch.bfloat16).to(torch.float32)

    def chain(x_, g, u, d, h_dt=None, o_dt=None):
        h = torch.nn.functional.silu(x_ @ g) * (x_ @ u)
        if h_dt is not None:
            h = rt(h, h_dt)
        o = h @ d
        if o_dt is not None:
            o = rt(o, o_dt)
        return o

    ref = chain(xr, wg, wu, wd)
    g4, u4, d4 = rt(wg, ttnn.bfloat4_b), rt(wu, ttnn.bfloat4_b), rt(wd, ttnn.bfloat4_b)
    print("bfp4 weights only          :", comp_pcc(ref, chain(xr, g4, u4, d4))[1])
    print("bfp4 w + bfp8 h            :", comp_pcc(ref, chain(xr, g4, u4, d4, h_dt=ttnn.bfloat8_b))[1])
    print(
        "bfp4 w + bfp8 h + bfp8 out :",
        comp_pcc(ref, chain(xr, g4, u4, d4, h_dt=ttnn.bfloat8_b, o_dt=ttnn.bfloat8_b))[1],
    )
    xb = rt(x[:count], ttnn.bfloat8_b)
    print(
        "+ bfp8 x                   :",
        comp_pcc(ref, chain(xb, g4, u4, d4, h_dt=ttnn.bfloat8_b, o_dt=ttnn.bfloat8_b))[1],
    )

    # bf16 partials instead of bfp8 for the gate/up split into 10 row-groups
    def chain_split(x_, g, u, d, part_dt, groups=10):
        EMB_T = emb // 32
        base, rem = EMB_T // groups, EMB_T % groups
        sizes = [base + (1 if i < rem else 0) for i in range(groups)]
        s, gate, up = 0, 0.0, 0.0
        for sz in sizes:
            sl = slice(s * 32, (s + sz) * 32)
            s += sz
            gate = gate + rt(x_[:, sl] @ g[sl], part_dt)
            up = up + rt(x_[:, sl] @ u[sl], part_dt)
        h = rt(torch.nn.functional.silu(gate) * up, ttnn.bfloat8_b)
        return rt(h @ d, ttnn.bfloat8_b)

    print("split partials bfp8        :", comp_pcc(ref, chain_split(xb, g4, u4, d4, ttnn.bfloat8_b))[1])
    print("split partials bf16        :", comp_pcc(ref, chain_split(xb, g4, u4, d4, ttnn.bfloat16))[1])
finally:
    ttnn.close_device(device)
