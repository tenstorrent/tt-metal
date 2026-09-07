"""Perf 2 graduation A (stream_regime): verify the compact per-channel hold in the REAL op."""
import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")
os.environ["RMS_TRACE_BLOCKING"] = "1"
import torch, ttnn
from ttnn.operations.rms_norm_ttnn import rms_norm_ttnn, torch_rms_norm_ttnn
from eval.sharding import shard_config

K = "DEVICE KERNEL DURATION [ns]"
_ML = ttnn.TensorMemoryLayout


def ns(dev):
    ttnn.ReadDeviceProfiler(dev)
    t, f = 0.0, False
    for progs in (ttnn.get_latest_programs_perf_data() or {}).values():
        for p in progs:
            e = (getattr(p, "program_analyses_results", None) or {}).get(K)
            if e is not None:
                t += float(e.duration)
                f = True
    return t if f else float("nan")


def pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-30))


CASES = [
    ("#15 TARGET  8192x7168 gbr f32T", (1, 1, 8192, 7168), None, None, "gamma_bias_residual", True),
    ("FOCUS       8192x2304 gamma", (1, 1, 8192, 2304), None, None, "gamma", False),
    ("            8192x1024 gamma", (1, 1, 8192, 1024), None, None, "gamma", False),
    ("            8192x5120 gbr f32T", (1, 1, 8192, 5120), None, None, "gamma_bias_residual", True),
    ("            8192x7168 gamma", (1, 1, 8192, 7168), None, None, "gamma", False),
    ("            1024x16384 gbr(32c)", (1, 1, 1024, 16384), None, None, "gamma_bias_residual", False),
    ("WIDTH shard 32x7168", (1, 1, 32, 7168), _ML.WIDTH_SHARDED, ([32, 256], (7, 4)), "gamma", False),
    ("BLOCK shard 8192x1024", (1, 1, 8192, 1024), _ML.BLOCK_SHARDED, ([1024, 128], (8, 8)), "gamma", False),
    ("RM weight   128x4096 f32T", (1, 1, 128, 4096), None, None, "gamma_rm", True),
]
dev = ttnn.open_device(device_id=0)
try:
    for label, shape, ml, shard, mode, f32 in CASES:
        W = shape[-1]
        cfg = ttnn.ComputeConfigDescriptor()
        cfg.math_fidelity = ttnn.MathFidelity.HiFi2
        cfg.fp32_dest_acc_en = f32
        cfg.math_approx_mode = False
        torch.manual_seed(0)
        tx = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
        mc = (
            ttnn.DRAM_MEMORY_CONFIG
            if ml is None
            else shard_config(shard[0], shard[1], ml, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=dev)
        )
        x = ttnn.from_torch(tx, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)
        kw = dict(epsilon=1e-12, compute_kernel_config=cfg, memory_config=x.memory_config())
        ref = {"input_tensor": tx.float()}
        live = [x]
        glay = ttnn.ROW_MAJOR_LAYOUT if mode == "gamma_rm" else ttnn.TILE_LAYOUT

        def vec(s):
            torch.manual_seed(s)
            t = torch.randn(1, 1, 1, W, dtype=torch.float32).to(torch.bfloat16)
            v = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=glay, device=dev)
            live.append(v)
            return t, v

        if "gamma" in mode:
            t, v = vec(1)
            kw["weight"] = v
            ref["weight"] = t.float()
        if "bias" in mode:
            t, v = vec(2)
            kw["bias"] = v
            ref["bias"] = t.float()
        if "residual" in mode:
            torch.manual_seed(3)
            tr = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
            r = ttnn.from_torch(tr, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)
            kw["residual_input_tensor"] = r
            live.append(r)
            ref["residual_input_tensor"] = tr.float()
        exp = torch_rms_norm_ttnn(
            ref["input_tensor"],
            epsilon=1e-12,
            weight=ref.get("weight"),
            bias=ref.get("bias"),
            residual_input_tensor=ref.get("residual_input_tensor"),
        )
        o = rms_norm_ttnn(x, **kw)
        got = ttnn.to_torch(o)
        p = pcc(got, exp)
        del got
        ttnn.deallocate(o)
        ttnn.synchronize_device(dev)
        ns(dev)
        o = rms_norm_ttnn(x, **kw)
        ttnn.synchronize_device(dev)
        t = ns(dev)
        ttnn.deallocate(o)
        print(f"RESULT {label:32s} ns={t:10.0f}  pcc={p:.6f}")
        for v in live:
            try:
                ttnn.deallocate(v)
            except Exception:
                pass
finally:
    ttnn.close_device(dev)
