import sys

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "ttnn/ttnn/operations/moe_fused_swiglu/perf_experiments/gateup_reduce_overlap")
import torch
import ttnn
import gru_program_descriptor as gpd

print("opening device", flush=True)
device = ttnn.open_device(device_id=0)
try:
    TILE = 32
    emb_t, hn_pad, m_eff = 4, 6, 8
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
    print("building descriptor (kgroups=2)", flush=True)
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
        kgroups=2,
    )
    print("launching generic_op", flush=True)
    ttnn.generic_op([tt_x, tt_wg, tt_wu, tt_out], desc)
    print("generic_op RETURNED (no hang)", flush=True)
    out_torch = ttnn.to_torch(tt_out).to(torch.float32)
    print("out[0,:6]", out_torch[0, :6], flush=True)
finally:
    ttnn.close_device(device)
