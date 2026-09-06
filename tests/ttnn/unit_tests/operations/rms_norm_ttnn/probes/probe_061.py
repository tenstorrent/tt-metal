import os

os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")
import torch, ttnn
import ttnn.operations.rms_norm_ttnn.rms_norm_ttnn_program_descriptor as PD
from eval.sharding import shard_config, auto_shard_config

_ML = ttnn.TensorMemoryLayout


def cfg():
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = ttnn.MathFidelity.HiFi2
    c.fp32_dest_acc_en = False
    c.math_approx_mode = False
    return c


CASES = [
    ((1, 1, 8192, 256), None, _ML.INTERLEAVED, "gamma"),
    ((1, 1, 8192, 128), None, _ML.INTERLEAVED, "gamma_bias_residual"),
    ((1, 1, 8192, 1024), ([1024, 128], (8, 8)), _ML.BLOCK_SHARDED, "gamma"),
    ((1, 1, 7168, 1024), ([896, 128], (8, 8)), _ML.BLOCK_SHARDED, "gamma_bias_residual"),
    ((1, 1, 32, 7168), ([32, 256], (7, 4)), _ML.WIDTH_SHARDED, "gamma"),
    ((1, 1, 1024, 512), ([1024, 128], (4, 1)), _ML.WIDTH_SHARDED, "gamma"),
    ((1, 1, 8192, 1024), None, _ML.INTERLEAVED, "gamma"),
    ((1, 1, 4096, 256), None, _ML.INTERLEAVED, "gamma_bias"),
    ((1, 1, 2048, 192), None, _ML.INTERLEAVED, "gamma_bias_residual"),
]
device = ttnn.open_device(device_id=0)
try:
    for shape, shard, ml, mode in CASES:
        W = shape[-1]
        mc = (
            ttnn.DRAM_MEMORY_CONFIG
            if shard is None
            else shard_config(shard[0], shard[1], ml, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
        )
        x = ttnn.from_torch(
            torch.zeros(shape, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=mc,
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
        r = (
            ttnn.from_torch(
                torch.zeros(shape, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=mc,
            )
            if "residual" in mode
            else None
        )
        out_t = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, mc)
        res = {}
        for v in (0, 1):
            PD.CB_SQ_EXACT = v
            d = PD.create_program_descriptor(
                x, out_t, weight=w, bias=b, residual=r, epsilon=1e-12, compute_kernel_config=cfg()
            )
            ct = list(d.kernels[2].compile_time_args)
            res[v] = (ct[3] & 0xFFFF if False else ct[3], ct[1], ct[2], ct[14], sum(c.total_size for c in d.cbs))
        PD.CB_SQ_EXACT = 0
        flag = "CHANGED" if res[0] != res[1] else "same"
        print(f"SQX {str(shape):22s} {mode:22s} exact0=(blk,wtc,nwc,sqwt,L1)={res[0]} exact1={res[1]}  {flag}")
        for t in [x, w, b, r, out_t]:
            if t is not None:
                ttnn.deallocate(t)
finally:
    ttnn.close_device(device)
