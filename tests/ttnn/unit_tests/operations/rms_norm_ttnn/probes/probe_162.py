"""Perf 1 -- whole-op re-measure on the focus shape + the guard set.

One representative per distinct kernel path x layout x placement, plus every
combine geometry the graduated changes touch.
"""
import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")

import statistics, torch, ttnn
from ttnn.operations.rms_norm_ttnn import rms_norm_ttnn, torch_rms_norm_ttnn
from eval.sharding import shard_config, auto_shard_config

_ML = ttnn.TensorMemoryLayout
K = "DEVICE KERNEL DURATION [ns]"
TRIALS = int(os.environ.get("RMS_TRIALS", "5"))

# name: (shape, shard|"auto"|None, memory_layout, mode, fp32_dest, layout, PRE_ns)
CASES = {
    # --- the FOCUS shape and the other combine geometries -------------------
    "F_w7168_g28": ((1, 1, 32, 7168), ([32, 256], (7, 4)), _ML.WIDTH_SHARDED, "gamma", False, "T", 5408),
    "w1024_g8": ((1, 1, 32, 1024), ([32, 128], (8, 1)), _ML.WIDTH_SHARDED, "gamma", False, "T", 3504),
    "w2304_g9": ((1, 1, 32, 2304), ([32, 256], (9, 1)), _ML.WIDTH_SHARDED, "gamma", False, "T", 4256),
    "w5120_g32tree": ((1, 1, 32, 5120), ([32, 160], (8, 4)), _ML.WIDTH_SHARDED, "gamma", False, "T", 4601),
    "w5120_gbr_f32": ((1, 1, 32, 5120), ([32, 160], (8, 4)), _ML.WIDTH_SHARDED, "gamma_bias_residual", True, "T", 6076),
    "blk8192": ((1, 1, 8192, 1024), ([1024, 128], (8, 8)), _ML.BLOCK_SHARDED, "gamma", False, "T", 22961),
    "blk7168_gbr": (
        (1, 1, 7168, 1024),
        ([896, 128], (8, 8)),
        _ML.BLOCK_SHARDED,
        "gamma_bias_residual",
        False,
        "T",
        32346,
    ),
    # --- HEIGHT shard: native, no combine ------------------------------------
    "hgt2048": ((1, 1, 2048, 256), "auto", _ML.HEIGHT_SHARDED, "gamma", False, "T", 0),
    # --- ROW_MAJOR BAND ------------------------------------------------------
    "band512_rm": ((1, 1, 256, 512), "auto", _ML.WIDTH_SHARDED, "gamma", False, "RM", 0),
    # --- interleaved: width-split combine, prefill RESIDENT, STREAM ----------
    "int7168_dec": ((1, 1, 32, 7168), None, _ML.INTERLEAVED, "gamma", False, "T", 8792),
    "int1024_pre": ((1, 1, 8192, 1024), None, _ML.INTERLEAVED, "gamma", False, "T", 86992),
    "int7168_pre": ((1, 1, 8192, 7168), None, _ML.INTERLEAVED, "gamma", False, "T", 578937),
    "int5120_gbr": ((1, 1, 8192, 5120), None, _ML.INTERLEAVED, "gamma_bias_residual", True, "T", 669301),
    "stream_gbr": ((1, 1, 1024, 16384), None, _ML.INTERLEAVED, "gamma_bias_residual", False, "T", 0),
    "int4096_rmw": ((1, 1, 128, 4096), None, _ML.INTERLEAVED, "gamma", True, "T", 11421),
    "ragged_w": ((1, 1, 32, 4064), None, _ML.INTERLEAVED, "gamma", False, "T", 0),
}
ONLY = os.environ.get("RMS_ONLY")
if ONLY:
    CASES = {k: v for k, v in CASES.items() if k in ONLY.split(",")}


def ns(d):
    ttnn.ReadDeviceProfiler(d)
    pc = ttnn.get_latest_programs_perf_data()
    tot = 0.0
    f = False
    for progs in (pc or {}).values():
        for p in progs:
            e = (getattr(p, "program_analyses_results", None) or {}).get(K)
            if e is not None:
                tot += float(e.duration)
                f = True
    return tot if f else None


def cfg(fp32):
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = ttnn.MathFidelity.HiFi2
    c.fp32_dest_acc_en = fp32
    c.math_approx_mode = False
    return c


def pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-30))


def relrms(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    return float(((a - b).pow(2).mean().sqrt()) / (b.pow(2).mean().sqrt() + 1e-30))


d = ttnn.open_device(device_id=0)
try:
    for name, (shape, shard, ml, mode, fp32, lay, pre) in CASES.items():
        try:
            L = ttnn.ROW_MAJOR_LAYOUT if lay == "RM" else ttnn.TILE_LAYOUT
            W = shape[-1]
            torch.manual_seed(0)
            tx = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
            if shard == "auto":
                mc = auto_shard_config(list(shape), ml, layout=L, dtype=ttnn.bfloat16, device=d)
            elif shard is not None:
                mc = shard_config(shard[0], shard[1], ml, layout=L, dtype=ttnn.bfloat16, device=d)
            else:
                mc = ttnn.DRAM_MEMORY_CONFIG
            x = ttnn.from_torch(tx, dtype=ttnn.bfloat16, layout=L, device=d, memory_config=mc)
            kw = {"epsilon": 1e-12, "compute_kernel_config": cfg(fp32), "memory_config": x.memory_config()}
            ref = {"input_tensor": tx.float()}

            def vec(seed):
                torch.manual_seed(seed)
                t = torch.randn(1, 1, 1, W, dtype=torch.float32).to(torch.bfloat16)
                return t, ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=L, device=d)

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
                kw["residual_input_tensor"] = ttnn.from_torch(
                    tr, dtype=ttnn.bfloat16, layout=L, device=d, memory_config=mc
                )
                ref["residual_input_tensor"] = tr.float()
            exp = torch_rms_norm_ttnn(
                ref["input_tensor"],
                epsilon=1e-12,
                weight=ref.get("weight"),
                bias=ref.get("bias"),
                residual_input_tensor=ref.get("residual_input_tensor"),
            )
            out = rms_norm_ttnn(x, **kw)
            got = ttnn.to_torch(out)
            p, r = pcc(got, exp), relrms(got, exp)
            del out, got
            ttnn.synchronize_device(d)
            ns(d)
            s = []
            for _ in range(TRIALS):
                rms_norm_ttnn(x, **kw)
                ttnn.synchronize_device(d)
                v = ns(d)
                if v:
                    s.append(v)
            m = statistics.median(s)
            rat = f"{pre/m:6.3f}x" if pre else "   --  "
            print(f"RESULT {name:16s} ns={m:9.0f} pre={pre:9d} {rat}  pcc={p:.6f} relrms={r:.5f}")
            for t in [x] + [v for v in kw.values() if isinstance(v, ttnn.Tensor)]:
                try:
                    ttnn.deallocate(t)
                except Exception:
                    pass
        except Exception as e:
            print(f"RESULT {name:16s} ERROR {type(e).__name__}: {e}")
finally:
    ttnn.close_device(d)
