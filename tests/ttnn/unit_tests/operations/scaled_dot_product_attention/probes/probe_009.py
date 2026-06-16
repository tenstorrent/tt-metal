import math, torch, ttnn
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention


def ref(Q, K, V, scale=None):
    Qf, Kf, Vf = Q.float(), K.float(), V.float()
    D = Qf.shape[-1]
    s = scale or 1.0 / math.sqrt(D)
    sc = torch.matmul(Qf, Kf.transpose(-2, -1)) * s
    w = torch.softmax(sc, dim=-1)
    return torch.matmul(w, Vf)


def relrms(out, exp):
    o = out.float()
    e = exp.float()
    return ((o - e).pow(2).mean().sqrt() / e.std()).item()


def pcc(out, exp):
    o = out.float().flatten()
    e = exp.float().flatten()
    return torch.corrcoef(torch.stack([o, e]))[0, 1].item()


dev = ttnn.open_device(device_id=0)
try:
    cfg_hifi2_noacc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False
    )
    for dt, tdt, cfg, tag in [
        (ttnn.bfloat16, torch.bfloat16, cfg_hifi2_noacc, "bf16 acc=False HiFi2"),
        (ttnn.bfloat8_b, torch.bfloat16, cfg_hifi2_noacc, "bf8b acc=False HiFi2"),
        (ttnn.float32, torch.float32, None, "fp32 acc=True HiFi4(default)"),
    ]:
        for S in [4096, 8192]:
            torch.manual_seed(0)
            shp = (1, 1, S, 64)
            Q = torch.randn(shp, dtype=tdt)
            K = torch.randn(shp, dtype=tdt)
            V = torch.randn(shp, dtype=tdt)
            e = ref(Q, K, V)
            tq = ttnn.from_torch(Q, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
            tk = ttnn.from_torch(K, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
            tv = ttnn.from_torch(V, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
            o = scaled_dot_product_attention(tq, tk, tv, compute_kernel_config=cfg)
            ot = ttnn.to_torch(o)
            print(f"{tag} S={S}: relrms={relrms(ot,e):.4f} pcc={pcc(ot,e):.6f}")
finally:
    ttnn.close_device(dev)
