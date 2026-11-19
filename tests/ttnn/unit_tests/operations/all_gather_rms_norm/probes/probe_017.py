import torch, ttnn
from eval.golden_tests.all_gather_rms_norm.conftest import _FabricMeshCache
from eval.golden_tests.all_gather_rms_norm import helpers as H

c = _FabricMeshCache()
m = c.get((2, 2))


def run(tag, shape, fp32, offset=2.0):
    H.LN_ROW_OFFSET_SCALE = offset
    ax = dict(
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        fp32_dest_acc_en=fp32,
        gamma_mode="no_gamma",
        gamma_dtype="none",
        gamma_layout="none",
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        cluster_axis=1,
        topology=ttnn.Topology.Linear,
        num_links=1,
        norm_type="layer_norm",
        beta_mode="no_beta",
    )
    call, chk = H.prepare_all_gather_rms_norm((shape,), mesh_device=m, **ax)
    e = [cl.cell_contents for cl in chk.__closure__ if isinstance(cl.cell_contents, torch.Tensor)][0].double()
    a = ttnn.to_torch(ttnn.get_device_tensors(call())[0]).double()
    r = e[0, 0]
    pcc, rms = H._pcc_rel_rms(a, r)
    A = a.reshape(-1, a.shape[-1])
    R = r.reshape(-1, r.shape[-1])
    v = R.abs().sum(-1) > 0
    A, R = A[v], R[v]
    # per-row least squares a ~ s*r + o
    rm = R.mean(-1, keepdim=True)
    am = A.mean(-1, keepdim=True)
    s = ((A - am) * (R - rm)).sum(-1) / ((R - rm) ** 2).sum(-1)
    o = (am - s[:, None] * rm).squeeze(-1)
    res = A - (s[:, None] * R + o[:, None])
    rr = (res.pow(2).mean().sqrt() / R.pow(2).mean().sqrt()).item()
    print(
        f"RES {tag:22s} rel_rms={rms:.4f} | row scale mean={s.mean():.4f} std={s.std():.4f} | row offset rms={o.pow(2).mean().sqrt():.4f} | resid after scale+offset={rr:.4f}"
    )


run("ln False off2 1792", (1, 1, 32, 1792), False)
run("ln False off0 1792", (1, 1, 32, 1792), False, offset=0.0)
run("ln False off2 384", (1, 1, 640, 384), False)
run("ln True off2 1792", (1, 1, 32, 1792), True)
c.close()
