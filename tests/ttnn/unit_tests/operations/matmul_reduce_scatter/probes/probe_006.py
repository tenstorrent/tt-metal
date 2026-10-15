import torch, ttnn
from tests.ttnn.unit_tests.operations.matmul_reduce_scatter.test_matmul_reduce_scatter import _open_mesh, _close_mesh
from eval.golden_tests.matmul_reduce_scatter.helpers import (
    create_ttnn_input_tensor,
    pytorch_matmul_reduce_scatter,
    random_stacked,
)
from ttnn.operations.matmul_reduce_scatter import matmul_reduce_scatter

mesh = _open_mesh((1, 4), ttnn.FabricConfig.FABRIC_2D_TORUS_XY)
try:
    a_shape, w_shape = (1, 1, 640, 1024), (1024, 2048)
    a = random_stacked(mesh, a_shape, torch.bfloat16, seed=0)
    w = random_stacked(mesh, w_shape, torch.bfloat16, seed=1, scale=w_shape[0] ** -0.5)
    ref = pytorch_matmul_reduce_scatter(a.float(), w.float(), 1, -1)
    ta = create_ttnn_input_tensor(a, mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tw = create_ttnn_input_tensor(w, mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    for fid in (ttnn.MathFidelity.LoFi, ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.HiFi4):
        for topo in (ttnn.Topology.Ring, ttnn.Topology.Linear):
            out = matmul_reduce_scatter(
                ta,
                tw,
                cluster_axis=1,
                scatter_dim=-1,
                topology=topo,
                compute_kernel_config=ttnn.ComputeConfigDescriptor(math_fidelity=fid, fp32_dest_acc_en=True),
            )
            got = torch.stack([ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(out)])
            r = torch.stack([ref[0, c] for c in range(4)]).float()
            nz = r.abs() > 1e-3
            ratio = got[nz] / r[nz]
            rel = ((got - r).pow(2).mean().sqrt() / r.pow(2).mean().sqrt()).item()
            # isolate rounding: compare against the reference rounded to bf16
            rb = r.to(torch.bfloat16).float()
            print(
                "PROBE",
                fid,
                topo,
                "ratio_median",
                round(ratio.median().item(), 5),
                "rel_rms",
                round(rel, 5),
                "mean(got-ref)/mean|ref|",
                round(((got - r).mean() / r.abs().mean()).item(), 5),
                "bf16-rounded-ref rel_rms",
                round(((rb - r).pow(2).mean().sqrt() / r.pow(2).mean().sqrt()).item(), 5),
            )
finally:
    _close_mesh(mesh)
