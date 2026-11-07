import os, ttnn, torch, collections

cfg = ttnn._ttnn.fabric.FabricRouterConfig()
cfg.max_packet_payload_size_bytes = 14 * 1024 + 64
ttnn.set_fabric_config(
    ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
    ttnn.FabricReliabilityMode.STRICT_INIT,
    None,
    ttnn.FabricTensixConfig.DISABLED,
    ttnn.FabricUDMMode.DISABLED,
    ttnn.FabricManagerMode.DEFAULT,
    cfg,
)
MS = tuple(int(v) for v in os.environ.get("ST_MESH", "2,2").split(","))
AX = int(os.environ.get("ST_AXIS", "0"))
SHAPE = tuple(int(v) for v in os.environ.get("ST_SHAPE", "1,1,2048,1280").split(","))
DT = getattr(ttnn, os.environ.get("ST_DTYPE", "bfloat16"))
FP32 = os.environ.get("ST_FP32", "0") == "1"
L = int(os.environ.get("ST_L", "1"))
N = int(os.environ.get("ST_N", "20"))
TOPO = ttnn.Topology.Ring if os.environ.get("ST_RING") == "1" else ttnn.Topology.Linear
mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*MS))
from ttnn.operations.all_gather_rms_norm import all_gather_rms_norm

R, C = MS
G = MS[AX]
cc = ttnn.ComputeConfigDescriptor()
cc.math_fidelity = ttnn.MathFidelity.HiFi4
cc.fp32_dest_acc_en = FP32


def run(seed):
    torch.manual_seed(seed)
    xs = torch.randn(R, C, *SHAPE) * (1 + seed % 3)
    glob = (
        torch.cat([torch.cat(list(xs[r]), dim=-1) for r in range(R)], dim=-len(SHAPE) - 0 if False else 0)
        if False
        else None
    )
    # device (r,c) holds xs[r,c]
    tens = torch.cat([torch.cat([xs[r, c] for c in range(C)], dim=-1).unsqueeze(0) for r in range(R)], dim=0)
    x = ttnn.from_torch(
        tens,
        dtype=DT,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=MS, dims=(0, -1)),
    )
    y = all_gather_rms_norm(x, cluster_axis=AX, topology=TOPO, num_links=L, compute_kernel_config=cc)
    outs = ttnn.get_device_tensors(y)
    xq = ttnn.to_torch(x, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, mesh_shape=MS, dims=(0, -1))).float()
    bad = []
    for i, o in enumerate(outs):
        r, c = divmod(i, C)
        o = ttnn.to_torch(o).float().reshape(*SHAPE[:-1], -1)
        mem = [(rr, c) for rr in range(R)] if AX == 0 else [(r, cc_) for cc_ in range(C)]
        Wl = SHAPE[-1]
        full = torch.cat([xq[rr][..., cc_ * Wl : (cc_ + 1) * Wl].reshape(SHAPE) for rr, cc_ in mem], dim=-1)
        ref = torch.nn.functional.rms_norm(full, (full.shape[-1],), eps=1e-6)
        err = o - ref
        fin = torch.isfinite(o).all().item()
        rel = (err.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item() if fin else float("inf")
        if not fin or rel > (0.002 if DT == ttnn.float32 else 0.03):
            rows = (err.abs().reshape(-1, err.shape[-1]).amax(-1) > 0.1).nonzero().flatten()
            cols = (err.abs().reshape(-1, err.shape[-1]).amax(0) > 0.1).nonzero().flatten()
            bad.append(
                (
                    i,
                    rel,
                    fin,
                    rows[:6].tolist(),
                    len(rows),
                    cols.min().item() if len(cols) else -1,
                    cols.max().item() if len(cols) else -1,
                )
            )
    ttnn.deallocate(y)
    ttnn.deallocate(x)
    return bad


nb = 0
for s in range(N):
    b = run(s)
    if b:
        nb += 1
        print("BAD", s, b)
print(
    "RESULT bad_calls",
    nb,
    "of",
    N,
    dict(shape=SHAPE, dt=str(DT), fp32=FP32, L=L, axis=AX, force_k=os.environ.get("AGRMS_FORCE_K")),
)
ttnn.close_mesh_device(mesh)
ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
