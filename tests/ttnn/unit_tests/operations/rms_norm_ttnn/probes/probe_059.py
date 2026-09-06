import os

os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")
import torch, ttnn
import ttnn.operations.rms_norm_ttnn.rms_norm_ttnn_program_descriptor as PD

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
    "SQBLK",
    "RESFUSE",
]


def cfg():
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = ttnn.MathFidelity.HiFi2
    c.fp32_dest_acc_en = False
    c.math_approx_mode = False
    return c


CASES = [
    ((1, 1, 1024, 16384), "residual"),
    ((1, 1, 1024, 32768), "gamma_bias_residual"),
    ((1, 1, 2048, 8192), "residual"),
    ((1, 1, 1024, 16384), "gamma_bias_residual"),
]
device = ttnn.open_device(device_id=0)
try:
    for shape, mode in CASES:
        W = shape[-1]
        try:
            x = ttnn.from_torch(
                torch.zeros(shape, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            w = (
                ttnn.from_torch(
                    torch.zeros(1, 1, 1, W, dtype=torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                )
                if "gamma" in mode
                else None
            )
            b = (
                ttnn.from_torch(
                    torch.zeros(1, 1, 1, W, dtype=torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                )
                if "bias" in mode
                else None
            )
            r = ttnn.from_torch(
                torch.zeros(shape, dtype=torch.bfloat16),
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
            ct = list(d.kernels[2].compile_time_args)[:26]
            print("PLAN", shape, mode, {n: v for n, v in zip(NAMES, ct) if n not in ("INV_W", "EPS")})
            for t in [x, w, b, r, out_t]:
                if t is not None:
                    ttnn.deallocate(t)
        except Exception as e:
            print("PLAN", shape, mode, "ERR", type(e).__name__, str(e)[:200])
finally:
    ttnn.close_device(device)
