import torch, ttnn, sys

sys.path.insert(0, "tests/ttnn/unit_tests/operations/matmul_reduce_scatter")
from tests.scripts.common import get_updated_device_params
from ttnn.operations.matmul_reduce_scatter import matmul_reduce_scatter
import test_matmul_reduce_scatter as T

shape = (2, 4)
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D, ttnn.FabricReliabilityMode.STRICT_INIT)
params = get_updated_device_params({"fabric_config": ttnn.FabricConfig.FABRIC_2D})
params.pop("fabric_config")
mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*shape), **params)
try:
    import os

    CASE = os.environ.get("CASE", "1,-1,256,256,256,ones")
    ax, sd, M, K, N, mode = CASE.split(",")
    ax, sd, M, K, N = int(ax), int(sd), int(M), int(K), int(N)
    a_shape, w_shape = (1, 1, M, K), (K, N)
    rows, cols = shape
    if mode == "ones":
        # A = 1 on first 32 K columns, W[r,c] = rank+1 on first 32 rows -> partial = 32*(rank+1) everywhere
        a = torch.zeros((rows, cols, *a_shape), dtype=torch.bfloat16)
        a[..., :32] = 1
        w = torch.zeros((rows, cols, *w_shape))
        w[:, :, :32, :] = torch.arange(1, rows * cols + 1).float().reshape(rows, cols, 1, 1)
        w = w.to(torch.bfloat16)
    elif mode == "ident":
        # A = identity-like (K>=M), W = tile-position code + device offset -> partial = W rows [0, M)
        a = torch.zeros((rows, cols, *a_shape), dtype=torch.bfloat16)
        for i in range(min(M, K)):
            a[..., i, i] = 1
        kk = torch.arange(K).reshape(K, 1) // 32
        nn = torch.arange(N).reshape(1, N) // 32
        base = (kk * 8 + nn).float()
        w = torch.stack([torch.stack([base + 0.0 * (r * cols + c) for c in range(cols)]) for r in range(rows)]).to(
            torch.bfloat16
        )
    else:
        a = T._stacked_randn(mesh, a_shape, 0)
        w = T._stacked_randn(mesh, w_shape, 1, scale=K**-0.5)
    exp = T._reference(a, w, ax, sd)
    for _rep in range(int(os.environ.get("REPS", "1"))):
        out = matmul_reduce_scatter(
            T._to_mesh(a, mesh, ttnn.bfloat16),
            T._to_mesh(w, mesh, ttnn.bfloat16),
            cluster_axis=ax,
            scatter_dim=sd,
            topology=ttnn.Topology.Linear,
            num_links=int(os.environ.get("L", "1")),
            **(
                {
                    "compute_kernel_config": ttnn.ComputeConfigDescriptor(
                        math_fidelity=getattr(ttnn.MathFidelity, os.environ["FID"]),
                        fp32_dest_acc_en=os.environ.get("FP32", "1") == "1",
                    )
                }
                if "FID" in os.environ
                else {}
            ),
        )
        for (r, c), act in T._per_device(out, cols):
            ref = exp[r, c].float()
            err = (act - ref).abs()
            H, W = ref.shape[-2] // 32, ref.shape[-1] // 32
            tm = err.reshape(H, 32, W, 32).amax(dim=(1, 3))
            bad = (tm > 0.05 * ref.abs().max().clamp(min=1)).int()
            print(f"dev({r},{c}) bad tiles {int(bad.sum())}/{H*W}  pcc {T._pcc(act, ref):.4f}")
            if bad.sum():
                print(bad.tolist())
                # decompose act - ref onto the group's partials of this block (missing / doubled contributions)
                g_axis = ax
                p = (r, c)[ax]
                G = shape[ax]
                parts = []
                for g in range(G):
                    rr, cc = (g, c) if ax == 0 else (r, g)
                    part = torch.matmul(a[rr, cc].float(), w[rr, cc].float())
                    parts.append(
                        torch.chunk(part, G, dim=sd)[p][0, 0] if part.dim() == 4 else torch.chunk(part, G, dim=sd)[p]
                    )
                diff = (act - ref).reshape(ref.shape[-2], ref.shape[-1])
                for ti in range(H):
                    for tj in range(W):
                        if not bad[ti, tj]:
                            continue
                        d = diff[ti * 32 : (ti + 1) * 32, tj * 32 : (tj + 1) * 32].reshape(-1, 1)
                        X = torch.stack(
                            [pp[ti * 32 : (ti + 1) * 32, tj * 32 : (tj + 1) * 32].reshape(-1) for pp in parts], 1
                        )
                        coef = torch.linalg.lstsq(X, d).solution.reshape(-1)
                        resid = (d - X @ coef.reshape(-1, 1)).abs().max()
                        print(
                            f"  tile({ti},{tj}) coef {[round(float(x),2) for x in coef]} resid {float(resid):.3f} dmax {float(d.abs().max()):.3f}"
                        )
                if mode != "randn":
                    # show actual/ref tile means
                    am = act.reshape(H, 32, W, 32).mean(dim=(1, 3))
                    rm = ref.reshape(H, 32, W, 32).mean(dim=(1, 3))
                    print("act", am.round().int().tolist())
                    print("ref", rm.round().int().tolist())
finally:
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
