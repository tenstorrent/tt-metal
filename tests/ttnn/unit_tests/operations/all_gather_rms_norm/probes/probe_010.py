import os, ttnn, torch

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
mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(2, 2))
from eval.golden_tests.all_gather_rms_norm import helpers as H
from ttnn.operations.all_gather_rms_norm import all_gather_rms_norm
import ttnn.operations.all_gather_rms_norm.all_gather_rms_norm_program_descriptor as PD

SH = tuple(int(v) for v in os.environ.get("CP_SHAPE", "1,1,2048,1280").split(","))
DT = getattr(ttnn, os.environ.get("CP_DTYPE", "bfloat8_b"))
ML = getattr(ttnn.TensorMemoryLayout, os.environ.get("CP_ML", "HEIGHT_SHARDED"))
FP32 = os.environ.get("CP_FP32", "0") == "1"
N = int(os.environ.get("CP_N", "8"))
nbad = 0
for it in range(N):
    if os.environ.get("CP_CLEAR") == "1":
        mesh.disable_and_clear_program_cache()
        mesh.enable_program_cache()
    seed = it if os.environ.get("CP_VARY") == "1" else 0
    call, check = H.prepare_all_gather_rms_norm(
        (SH,),
        dtype=DT,
        layout=ttnn.TILE_LAYOUT,
        fp32_dest_acc_en=FP32,
        gamma_mode="no_gamma",
        gamma_dtype="none",
        gamma_layout="none",
        memory_layout=ML,
        cluster_axis=0,
        topology=ttnn.Topology.Linear,
        num_links=1,
        mesh_device=mesh,
        extras={"seed": seed},
        op=all_gather_rms_norm,
    )
    y = call()
    fv = dict(zip(check.__code__.co_freevars, [c.cell_contents for c in check.__closure__]))
    E = fv["expected"].float()
    ref_all = [E[i // 2, i % 2] for i in range(4)]
    outs = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(y)]
    try:
        check(y)
        ok = True
    except Exception as e:
        ok = False
        nbad += 1
        print("BAD", it, str(e)[:200])
    if not ok:
        # locate: non-finite / huge elements per device, in tile coordinates
        for d, o in enumerate(outs):
            o2 = o.reshape(-1, o.shape[-1])
            badm = ~torch.isfinite(o2) | (o2.abs() > 50)
            if badm.any():
                rr, cc = badm.nonzero(as_tuple=True)
                tr = sorted(set((rr // 32).tolist()))
                tc = sorted(set((cc // 32).tolist()))
                print(
                    "  dev",
                    d,
                    "n",
                    int(badm.sum()),
                    "tile_rows",
                    tr[:20],
                    "tile_cols",
                    tc[:20],
                    "rows",
                    sorted(set(rr.tolist()))[:8],
                )
                ref = ref_all[d].reshape(-1, o.shape[-1])
                for row in sorted(set(rr.tolist()))[:3]:
                    c0 = tc[0] * 32
                    c1 = (tc[-1] + 1) * 32
                    rat = o2[row, c0:c1] / ref[row, c0:c1]
                    rat = rat[torch.isfinite(rat)]
                    print(
                        "    row",
                        row,
                        "ratio median",
                        rat.median().item() if len(rat) else None,
                        "p5",
                        rat.quantile(0.05).item() if len(rat) else None,
                        "p95",
                        rat.quantile(0.95).item() if len(rat) else None,
                        "finite",
                        len(rat),
                    )
    if it == 0:
        geo = PD.derive_geometry(SH, 2)
        print("GEO", geo)
        print("CACHE", [(k[1:],) for k in list(PD._PLAN_CACHE.keys())])
    ttnn.deallocate(y)
print("RESULT", nbad, "of", N)
ttnn.close_mesh_device(mesh)
ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
