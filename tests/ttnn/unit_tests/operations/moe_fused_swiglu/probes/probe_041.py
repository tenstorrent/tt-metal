import sys

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "ttnn/ttnn/operations/moe_fused_swiglu/perf_experiments/gateup_reduce_overlap")
import torch
import ttnn
import gru_program_descriptor as gpd

device = ttnn.open_device(device_id=0)
try:
    TILE = 32
    emb_t, hn_pad, m_eff = 10, 6, 8  # kgroups=10 needs emb_t >= 10 so every row gets >=1 K tile
    torch.manual_seed(0)
    k = emb_t * TILE
    x_torch = torch.randn(m_eff * TILE, k) * 0.5
    wg_torch = torch.randn(k, hn_pad * TILE) * 0.5
    wu_torch = torch.randn(k, hn_pad * TILE) * 0.5
    tt_x = gpd.make_x_tensor(x_torch, device)
    tt_wg = gpd.make_weight_tensor(wg_torch, device)
    tt_wu = gpd.make_weight_tensor(wu_torch, device)
    tt_out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([m_eff * TILE, hn_pad * TILE]), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    print("building descriptor (kgroups=10, full topology)", flush=True)
    desc = gpd.create_program_descriptor(
        tt_x,
        tt_wg,
        tt_wu,
        tt_out,
        device=device,
        emb_t=emb_t,
        hn_pad=hn_pad,
        m_eff=m_eff,
        s_stages=1,
        split_axis="hn",
        pipelined=False,
        kgroups=10,
    )
    print("launching generic_op", flush=True)
    ttnn.generic_op([tt_x, tt_wg, tt_wu, tt_out], desc)
    print("generic_op RETURNED (no hang)", flush=True)

    x_q = ttnn.to_torch(tt_x).to(torch.float32)
    wg_q = ttnn.to_torch(tt_wg).to(torch.float32)
    wu_q = ttnn.to_torch(tt_wu).to(torch.float32)
    ref = torch.nn.functional.silu(x_q @ wg_q) * (x_q @ wu_q)
    out_torch = ttnn.to_torch(tt_out).to(torch.float32)
    diff = (out_torch - ref).abs()
    print("max abs diff", diff.max().item(), "ref max", ref.abs().max().item(), flush=True)
    a = (out_torch - out_torch.mean()).flatten().double()
    b = (ref - ref.mean()).flatten().double()
    pcc = (a @ b / (a.norm() * b.norm())).item()
    print("PCC", pcc, flush=True)
finally:
    ttnn.close_device(device)
