import torch, ttnn
from eval.golden_tests.all_gather_rms_norm.conftest import _FabricMeshCache
from eval.golden_tests.all_gather_rms_norm import helpers as H

c = _FabricMeshCache()
m = c.get((2, 2))


def run(tag, shape, dtype, fp32, norm, gamma=True, gamma_dtype=ttnn.bfloat16, beta=False, offset=2.0):
    H.LN_ROW_OFFSET_SCALE = offset
    ax = dict(
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        fp32_dest_acc_en=fp32,
        gamma_mode="gamma" if gamma else "no_gamma",
        gamma_dtype=gamma_dtype if gamma else "none",
        gamma_layout=ttnn.TILE_LAYOUT if gamma else "none",
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        cluster_axis=1,
        topology=ttnn.Topology.Linear,
        num_links=1,
        norm_type=norm,
        beta_mode="beta" if beta else "no_beta",
    )
    call, chk = H.prepare_all_gather_rms_norm((shape,), mesh_device=m, **ax)
    e = [cl.cell_contents for cl in chk.__closure__ if isinstance(cl.cell_contents, torch.Tensor)][0].double()
    out = call()
    a = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).double()
    r = e[0, 0]
    pcc, rms = H._pcc_rel_rms(a, r)
    A = a.reshape(-1, a.shape[-1])
    R = r.reshape(-1, r.shape[-1])
    v = R.abs().sum(-1) > 0
    s = (A * R).sum(-1)[v] / (R * R).sum(-1)[v]
    print(
        f"RES {tag:28s} pcc={pcc:.6f} rel_rms={rms:.4f} row-scale mean={s.mean():.5f} std={s.std() if s.numel()>1 else 0:.5f}"
    )


run("ln bf16 FPU k4 off2", (1, 1, 32, 384), ttnn.bfloat16, True, "layer_norm")
run("ln bf16 FPU k4 off0", (1, 1, 32, 384), ttnn.bfloat16, True, "layer_norm", offset=0.0)
run("ln bf16 FPU nogamma", (1, 1, 32, 384), ttnn.bfloat16, True, "layer_norm", gamma=False)
run("ln bf16 FPU k1 (32w)", (1, 1, 640, 32), ttnn.bfloat16, True, "layer_norm")
run("ln fp32 SFPU k4", (1, 1, 32, 384), ttnn.float32, True, "layer_norm", gamma_dtype=ttnn.float32)
run("ln bf16 SFPU wna 1000", (1, 1, 64, 1000), ttnn.bfloat16, True, "layer_norm")
run("ln bf8b FPU k4", (1, 1, 32, 384), ttnn.bfloat8_b, True, "layer_norm")
run("rms bf16 k4", (1, 1, 32, 384), ttnn.bfloat16, True, "rms_norm")
run("rms fp32 k4", (1, 1, 32, 384), ttnn.float32, True, "rms_norm", gamma_dtype=ttnn.float32)
c.close()
