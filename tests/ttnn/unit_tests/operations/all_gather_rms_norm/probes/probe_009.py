import torch, ttnn
from eval.golden_tests.all_gather_rms_norm.conftest import _FabricMeshCache
from eval.golden_tests.all_gather_rms_norm import helpers as H

c = _FabricMeshCache()
m = c.get((2, 2))


def run(shape, dtype, fp32, norm, gamma_dtype=ttnn.bfloat16, beta=False):
    ax = dict(
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        fp32_dest_acc_en=fp32,
        gamma_mode="gamma",
        gamma_dtype=gamma_dtype,
        gamma_layout=ttnn.TILE_LAYOUT,
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        cluster_axis=1,
        topology=ttnn.Topology.Linear,
        num_links=1,
        norm_type=norm,
        beta_mode="beta" if beta else "no_beta",
    )
    got = {}

    def check(out):
        got["t"] = out

    call, chk = H.prepare_all_gather_rms_norm((shape,), mesh_device=m, **ax)
    # rebuild expected through the closure
    exp = chk.__closure__
    out = call()
    dev = [ttnn.to_torch(t).double() for t in ttnn.get_device_tensors(out)]
    e = [cl.cell_contents for cl in exp if isinstance(cl.cell_contents, torch.Tensor)][0].double()
    a = dev[0]
    r = e[0, 0]
    pcc, rms = H._pcc_rel_rms(a, r)
    # per-row least-squares scale  s_row = <a,r>/<r,r>
    A = a.reshape(-1, a.shape[-1])
    R = r.reshape(-1, r.shape[-1])
    valid = R.abs().sum(-1) > 0
    s = (A * R).sum(-1)[valid] / (R * R).sum(-1)[valid]
    resid = A[valid] - s[:, None] * R[valid]
    rres = (resid.pow(2).mean().sqrt() / R[valid].pow(2).mean().sqrt()).item()
    msk = R.abs() > 1e-2
    ratio = A[msk] / R[msk]
    print(
        f"{shape} {dtype} fp32={fp32} {norm} g={gamma_dtype}: pcc={pcc:.6f} rel_rms={rms:.4f} | per-row scale mean={s.mean():.5f} "
        f"std={s.std() if s.numel()>1 else 0:.5f} min={s.min():.5f} max={s.max():.5f} | rel_rms after per-row scale fit={rres:.4f} "
        f"| ratio median={ratio.median():.5f} p5={ratio.quantile(0.05):.4f} p95={ratio.quantile(0.95):.4f}"
    )


for shp in [(1, 1, 1, 384), (1, 1, 32, 384), (1, 1, 32, 1792), (1, 1, 640, 384)]:
    run(shp, ttnn.bfloat16, True, "layer_norm")
    run(shp, ttnn.bfloat16, False, "layer_norm")
    run(shp, ttnn.bfloat16, False, "rms_norm")
run((1, 1, 32, 1792), ttnn.bfloat16, False, "layer_norm", gamma_dtype=ttnn.float32)
c.close()
