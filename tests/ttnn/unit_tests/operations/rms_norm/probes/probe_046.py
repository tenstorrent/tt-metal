import torch, ttnn
import ttnn.operations.rms_norm.rms_norm_program_descriptor as pdmod
from ttnn.operations.rms_norm import rms_norm

captured = []
_orig = pdmod.ttnn.KernelDescriptor


def spy(**kw):
    if "compute" in str(kw.get("kernel_source", "")):
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
        "X_SQ_WT",
        "X_RES",
    ]
    for shape, glay in [
        ((1, 1, 8192, 1024), ttnn.TILE_LAYOUT),
        ((1, 1, 8192, 2304), ttnn.TILE_LAYOUT),
        ((1, 1, 8192, 5120), ttnn.TILE_LAYOUT),
        ((1, 1, 8192, 7168), ttnn.TILE_LAYOUT),
        ((1, 1, 32, 7168), ttnn.TILE_LAYOUT),
        ((1, 1, 32, 4032), ttnn.TILE_LAYOUT),
        ((1, 1, 8192, 1024), ttnn.ROW_MAJOR_LAYOUT),
    ]:
        W = shape[-1]
        x = ttnn.from_torch(
            torch.randn(shape, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        g = ttnn.from_torch(
            torch.randn(W, dtype=torch.bfloat16).reshape(1, 1, 1, W), dtype=ttnn.bfloat16, layout=glay, device=device
        )
        captured.clear()
        out = rms_norm(x, gamma=g, compute_kernel_config=cfg())
        d = {n: v for n, v in zip(names, captured[-1]) if n not in ("INV_W", "EPS")}
        regime = "RESIDENT" if d["NUM_W_CHUNKS"] == 1 else ("ROW_RESIDENT" if d["X_RES"] else "STREAM")
        print(shape, "g=", "TILE" if glay == ttnn.TILE_LAYOUT else "RM", "Wt=", (W + 31) // 32, regime, d)
        # correctness
        t = ttnn.to_torch(out).float()
        ref = torch.from_numpy(__import__("numpy").array(0))  # placeholder
        ttnn.deallocate(x)
        ttnn.deallocate(g)
        ttnn.deallocate(out)
finally:
    ttnn.close_device(device)
