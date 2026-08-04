import torch, ttnn
import ttnn.operations.rms_norm.rms_norm_program_descriptor as pdmod
from ttnn.operations.rms_norm import rms_norm
from eval.sharding import shard_config

captured = []
_orig = pdmod.ttnn.KernelDescriptor


def spy(**kw):
    src = str(kw.get("kernel_source", ""))
    if "compute" in src:
        captured.append(list(kw.get("compile_time_args", [])))
    return _orig(**kw)


pdmod.ttnn.KernelDescriptor = spy


def cfg():
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = ttnn.MathFidelity.HiFi2
    c.fp32_dest_acc_en = False
    c.math_approx_mode = False
    return c


device = ttnn.open_device(device_id=0)
try:
    print("grid", device.compute_with_storage_grid_size())
    print("L1 unreserved", ttnn.get_max_worker_l1_unreserved_size())
    cases = [((1, 1, 8192, W), None, None, ttnn.TensorMemoryLayout.INTERLEAVED) for W in (1024, 2304, 5120, 7168)]
    cases += [
        ((1, 1, 32, 1024), [32, 128], (8, 1), ttnn.TensorMemoryLayout.WIDTH_SHARDED),
        ((1, 1, 32, 2304), [32, 256], (9, 1), ttnn.TensorMemoryLayout.WIDTH_SHARDED),
        ((1, 1, 32, 5120), [32, 160], (8, 4), ttnn.TensorMemoryLayout.WIDTH_SHARDED),
        ((1, 1, 32, 7168), [32, 256], (7, 4), ttnn.TensorMemoryLayout.WIDTH_SHARDED),
        ((1, 1, 8192, 1024), [1024, 128], (8, 8), ttnn.TensorMemoryLayout.BLOCK_SHARDED),
    ]
    for shape, ss, cg, ml in cases:
        W = shape[-1]
        mc = None
        if ml != ttnn.TensorMemoryLayout.INTERLEAVED:
            mc = shard_config(ss, cg, ml, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
        x = ttnn.from_torch(
            torch.randn(shape, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=mc,
        )
        g = ttnn.from_torch(
            torch.randn(W, dtype=torch.bfloat16).reshape(1, 1, 1, W),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        captured.clear()
        out = rms_norm(x, gamma=g, compute_kernel_config=cfg(), memory_config=mc)
        ct = captured[-1]
        names = [
            "IS_TILE",
            "WT_CHUNK",
            "NUM_W_CHUNKS",
            "BLOCK_ROWS",
            "PARTIAL_W",
            "HAS_GAMMA",
            "GAMMA_IS_RM",
            "INV_W",
            "EPS",
            "REDUCE_BULK",
            "ACC_VIA_ADD",
            "SCALER_TILES",
            "COMBINE",
            "GROUP_SIZE",
        ]
        d = {n: v for n, v in zip(names, ct) if n not in ("INV_W", "EPS")}
        print(shape, ml.name, "Wt=", (W + 31) // 32, d)
        ttnn.deallocate(x)
        ttnn.deallocate(g)
        ttnn.deallocate(out)
finally:
    ttnn.close_device(device)
