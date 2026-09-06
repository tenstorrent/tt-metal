import os

os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")
import torch, ttnn
import ttnn.operations.rms_norm_ttnn.rms_norm_ttnn_program_descriptor as PD

CASES = [
    ((1, 1, 8192, 1024), "gamma"),
    ((1, 1, 8192, 7168), "gamma"),
    ((1, 1, 8192, 5120), "gamma_bias_residual"),
    ((1, 1, 8192, 1024), "gamma_bias"),
    ((1, 1, 8192, 2048), "gamma"),
    ((1, 1, 8192, 1024), "no_gamma"),
]
NAMES = [
    "IS_TILE",
    "WT_CHUNK",
    "NUM_W_CHUNKS",
    "BLOCK_ROWS",
    "PARTIAL_W",
    "HAS_G",
    "PC_RM",
    "INV_W",
    "EPS",
    "REDUCE_BULK",
    "ACC_VIA_ADD",
    "SCALER_T",
    "COMBINE",
    "GROUP",
    "X_SQ_WT",
    "X_RESIDENT",
    "NATIVE_IN",
    "F0",
    "F1",
    "HAS_B",
    "HAS_R",
    "PBBLK",
    "NARROW",
    "FIN",
]


def cfg():
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = ttnn.MathFidelity.HiFi2
    c.fp32_dest_acc_en = False
    c.math_approx_mode = False
    return c


device = ttnn.open_device(device_id=0)
try:
    for shape, mode in CASES:
        W = shape[-1]
        tx = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
        x = ttnn.from_torch(
            tx, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        w = b = r = None
        if "gamma" in mode and mode != "no_gamma":
            w = ttnn.from_torch(
                torch.randn(1, 1, 1, W).to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
            )
        if "bias" in mode:
            b = ttnn.from_torch(
                torch.randn(1, 1, 1, W).to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
            )
        if "residual" in mode:
            r = ttnn.from_torch(
                torch.randn(shape).to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        out_t = ttnn.allocate_tensor_on_device(
            ttnn.Shape(list(shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, x.memory_config()
        )
        d = PD.create_program_descriptor(
            x, out_t, weight=w, bias=b, residual=r, epsilon=1e-12, compute_kernel_config=cfg()
        )
        ct = list(d.kernels[2].compile_time_args)[:24]
        print("PLAN", shape, mode, {n: v for n, v in zip(NAMES, ct) if n not in ("INV_W", "EPS")})
        cr = d.kernels[2].core_ranges
        print("   cores:", cr)
        for t in [x, w, b, r, out_t]:
            if t is not None:
                ttnn.deallocate(t)
finally:
    ttnn.close_device(device)
