"""precision of the sampler's primitives: ttnn.exp (approx on/off) and the triangular-matmul prefix sums vs fp64 torch"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(HERE, "head_device.py")).read().split("for v in VARIANTS:")[0])
g = torch.Generator().manual_seed(0)
s = -torch.rand(1, 1, 64, 2048, generator=g) * 30.0
st = ttnn.from_torch(
    s, device=md, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(md)
)
ref = torch.exp(s.double())


def dev(t):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0))[:1].double()


for name, f in [
    ("exp_default", lambda: ttnn.exp(st)),
    ("exp_approx_true", lambda: ttnn.exp(st, fast_and_approximate_mode=True)),
    ("exp_approx_false", lambda: ttnn.exp(st, fast_and_approximate_mode=False)),
]:
    try:
        o = dev(f())
        rel = ((o - ref).abs() / ref.clamp(min=1e-30)).max()
        res(event="prec_exp", name=name, max_rel=float(rel), mean_rel=float(((o - ref).abs() / ref).mean()))
    except Exception as e:
        res(event="prec_exp", name=name, err=str(e)[:150])
# prefix inside a row of 32 by matmul with the triangular matrix, fp32 HiFi4, vs fp64 cumsum
w = torch.rand(1, 1, 472, 32, generator=g) * torch.exp(-torch.rand(1, 1, 472, 1, generator=g) * 20)
tri = torch.triu(torch.ones(32, 32))
wt = ttnn.from_torch(
    w, device=md, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(md)
)
tt_ = ttnn.from_torch(
    tri.reshape(1, 1, 32, 32),
    device=md,
    dtype=ttnn.float32,
    layout=ttnn.TILE_LAYOUT,
    mesh_mapper=ttnn.ReplicateTensorToMesh(md),
)
for fid in ["HiFi4", "HiFi3", "HiFi2"]:
    ck = ttnn.init_device_compute_kernel_config(
        md.arch(),
        math_fidelity=getattr(ttnn.MathFidelity, fid),
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    o = dev(ttnn.matmul(wt, tt_, compute_kernel_config=ck))
    r = w.double() @ tri.double()
    res(
        event="prec_prefix",
        fid=fid,
        max_rel=float(((o - r).abs() / r.clamp(min=1e-30)).max()),
        rel_of_rowmax=float(((o - r).abs().max(-1).values / r.max(-1).values).max()),
    )
ttnn.cumsum  # existence check
ck = ttnn.init_device_compute_kernel_config(
    md.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
)
x = ttnn.from_torch(
    torch.rand(1, 1, 8, 15104, generator=g),
    device=md,
    dtype=ttnn.float32,
    layout=ttnn.TILE_LAYOUT,
    mesh_mapper=ttnn.ReplicateTensorToMesh(md),
)
try:
    o = dev(ttnn.cumsum(x, dim=-1))
    r = torch.cumsum(ttnn.to_torch(x, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0))[:1].double(), -1)
    res(
        event="cumsum",
        max_rel=float(((o - r).abs() / r).max()),
        us=round(timed_trace(lambda: ttnn.cumsum(x, dim=-1)), 1),
    )
except Exception as e:
    res(event="cumsum", err=str(e)[:200])
ttnn.close_mesh_device(md)
