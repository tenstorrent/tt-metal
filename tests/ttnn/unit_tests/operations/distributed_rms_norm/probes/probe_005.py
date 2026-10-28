import torch, ttnn
from eval.golden_tests.distributed_rms_norm.conftest import _FabricMeshCache
from ttnn.operations.distributed_rms_norm import distributed_rms_norm

cache = _FabricMeshCache()
mesh = cache.get((1, 4))


def run(xs, dtype, acc):
    tx = ttnn.from_torch(
        torch.cat(list(xs), dim=-1),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(0, 3), mesh_shape=(1, 4)),
    )
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=acc, math_approx_mode=False
    )
    out = distributed_rms_norm(tx, cluster_axis=1, epsilon=1e-5, topology=ttnn.Topology.Ring, compute_kernel_config=cfg)
    got = torch.cat([ttnn.to_torch(d).float() for d in ttnn.get_device_tensors(out)], dim=-1)
    full = torch.cat(list(xs.float()), dim=-1)
    ref = torch.nn.functional.rms_norm(full.double(), (full.shape[-1],), None, 1e-5).float()
    return got, ref, full


for dtype, td, acc in (
    (ttnn.float32, torch.float32, True),
    (ttnn.bfloat16, torch.bfloat16, True),
    (ttnn.bfloat16, torch.bfloat16, False),
):
    for c in (1.0, 3.0):
        xs = torch.full((4, 1, 1, 32, 256), c).to(td)
        got, ref, _ = run(xs, dtype, acc)
        print(
            "PROBE",
            dtype,
            "acc",
            acc,
            f"const={c} got/ref",
            float((got / ref).mean()),
            "got",
            float(got[0, 0, 0, 0]),
            "ref",
            float(ref[0, 0, 0, 0]),
        )
    xs = torch.randn((4, 1, 1, 64, 256)).to(td)
    got, ref, full = run(xs, dtype, acc)
    inv_true = torch.rsqrt(full.double().pow(2).mean(-1, keepdim=True) + 1e-5).expand_as(full)
    m = full.abs() > 0.5
    print(
        "PROBE",
        dtype,
        "acc",
        acc,
        "random got/ref median",
        float((got / ref)[m].median()),
        "inv_got/inv_true median",
        float(((got.double() / full.double())[m] / inv_true[m]).median()),
    )
cache.close()
