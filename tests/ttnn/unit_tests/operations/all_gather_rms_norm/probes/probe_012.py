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

T = ttnn.TensorMemoryLayout
SEQ = [
    ((1, 1, 2048, 1280), ttnn.float32, True, "no_gamma", "none", "none", T.INTERLEAVED),
    ((1, 1, 2048, 1280), ttnn.float32, True, "no_gamma", "none", "none", T.HEIGHT_SHARDED),
    ((1, 1, 2048, 1280), ttnn.bfloat16, False, "gamma", ttnn.bfloat16, ttnn.TILE_LAYOUT, T.INTERLEAVED),
    ((1, 1, 2048, 1280), ttnn.bfloat8_b, False, "no_gamma", "none", "none", T.HEIGHT_SHARDED),
    ((1, 1, 2048, 384), ttnn.float32, True, "gamma", ttnn.float32, ttnn.TILE_LAYOUT, T.INTERLEAVED),
]
REPS = int(os.environ.get("SP_REPS", "4"))
ONLY = os.environ.get("SP_ONLY")
nbad = 0
ncalls = 0
for rep in range(REPS):
    for ci, (SH, DT, FP32, gm, gd, gl, ML) in enumerate(SEQ):
        call, check = H.prepare_all_gather_rms_norm(
            (SH,),
            dtype=DT,
            layout=ttnn.TILE_LAYOUT,
            fp32_dest_acc_en=FP32,
            gamma_mode=gm,
            gamma_dtype=gd,
            gamma_layout=gl,
            memory_layout=ML,
            cluster_axis=0,
            topology=ttnn.Topology.Linear,
            num_links=1,
            mesh_device=mesh,
            extras={"seed": 0},
            op=all_gather_rms_norm,
        )
        if os.environ.get("SP_CLEAR") == "1":
            mesh.disable_and_clear_program_cache()
            mesh.enable_program_cache()
        y = call()
        ncalls += 1
        try:
            check(y)
        except Exception as e:
            nbad += 1
            print("BAD rep", rep, "cell", ci, str(e)[:160])
            fv = dict(zip(check.__code__.co_freevars, [c.cell_contents for c in check.__closure__]))
            E = fv["expected"].float()
            for d, t in enumerate(ttnn.get_device_tensors(y)):
                o = ttnn.to_torch(t).float()
                o2 = o.reshape(-1, o.shape[-1])
                ref = E[d // 2, d % 2].reshape(-1, o.shape[-1])
                err = (o2 - ref).abs()
                badm = ~torch.isfinite(o2) | (err > 0.05 * (1 + ref.abs()))
                if badm.any():
                    rr, cc = badm.nonzero(as_tuple=True)
                    tr = sorted(set((rr // 32).tolist()))
                    tcs = sorted(set((cc // 32).tolist()))
                    print("  dev", d, "n", int(badm.sum()), "tile_rows", tr[:12], "tile_cols", tcs[:24])
                    for row in sorted(set(rr.tolist()))[:2]:
                        cols = cc[rr == row]
                        c0, c1 = int(cols.min()), int(cols.max()) + 1
                        rat = o2[row, c0:c1] / ref[row, c0:c1]
                        rat = rat[torch.isfinite(rat)]
                        if len(rat) > 4:
                            print(
                                "    row",
                                row,
                                "cols",
                                c0,
                                c1,
                                "ratio med %.4g p5 %.4g p95 %.4g"
                                % (rat.median(), rat.quantile(0.05), rat.quantile(0.95)),
                            )
        ttnn.deallocate(y)
print("RESULT", nbad, "of", ncalls)
ttnn.close_mesh_device(mesh)
ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
