"""Perf 2 GUARD SET -- one representative per distinct kernel path x layout x placement.

Run twice in one session (`min` and both readings printed) so a sub-1.00x cell can be
told apart from session noise before it is called a regression.
"""
import os

for k, v in (
    ("TT_METAL_DEVICE_PROFILER", "1"),
    ("TT_METAL_PROFILER_MID_RUN_DUMP", "1"),
    ("TT_METAL_PROFILER_CPP_POST_PROCESS", "1"),
    ("TT_METAL_LOGGER_LEVEL", "error"),
):
    os.environ.setdefault(k, v)
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


# label, shape, memory_layout, (shard_shape, grid), mode, fp32_dest, layout, PRE-ROUND-2 ns
CASES = [
    (
        "interleaved RESIDENT, D40 engaged  (1,1,8192,2304)",
        (1, 1, 8192, 2304),
        None,
        None,
        "gamma",
        False,
        ttnn.TILE_LAYOUT,
        191879,
    ),
    (
        "interleaved RESIDENT, D41 engaged  (1,1,8192,1024)",
        (1, 1, 8192, 1024),
        None,
        None,
        "gamma",
        False,
        ttnn.TILE_LAYOUT,
        85966,
    ),
    (
        "ROW_RESIDENT + D39 compact         (1,1,8192,7168) gbr f32T",
        (1, 1, 8192, 7168),
        None,
        None,
        "gamma_bias_residual",
        True,
        ttnn.TILE_LAYOUT,
        1578655,
    ),
    (
        "ROW_RESIDENT tiled, no compact     (1,1,8192,5120) gamma",
        (1, 1, 8192, 5120),
        None,
        None,
        "gamma",
        False,
        ttnn.TILE_LAYOUT,
        412498,
    ),
    (
        "STREAM, gated out of D39 and D40   (1,1,1024,16384) gbr",
        (1, 1, 1024, 16384),
        None,
        None,
        "gamma_bias_residual",
        False,
        ttnn.TILE_LAYOUT,
        504124,
    ),
    (
        "WIDTH shard, native + flat combine (1,1,32,7168)",
        (1, 1, 32, 7168),
        _ML.WIDTH_SHARDED,
        ([32, 256], (7, 4)),
        "gamma",
        False,
        ttnn.TILE_LAYOUT,
        3794,
    ),
    (
        "BLOCK shard, native + slot tree    (1,1,8192,1024)",
        (1, 1, 8192, 1024),
        _ML.BLOCK_SHARDED,
        ([1024, 128], (8, 8)),
        "gamma",
        False,
        ttnn.TILE_LAYOUT,
        20458,
    ),
    (
        "HEIGHT shard, native, no combine   (1,1,2048,256)",
        (1, 1, 2048, 256),
        _ML.HEIGHT_SHARDED,
        ([32, 256], (8, 8)),
        "gamma",
        False,
        ttnn.TILE_LAYOUT,
        3510,
    ),
    (
        "ROW_MAJOR BAND, WIDTH-sharded      (1,1,256,512)",
        (1, 1, 256, 512),
        _ML.WIDTH_SHARDED,
        ([256, 8], (8, 8)),
        "gamma",
        False,
        ttnn.ROW_MAJOR_LAYOUT,
        22599,
    ),
    (
        "interleaved decode + AUTO W-split  (1,1,32,7168)",
        (1, 1, 32, 7168),
        None,
        None,
        "gamma",
        False,
        ttnn.TILE_LAYOUT,
        8411,
    ),
    (
        "interleaved, TILE weight, f32dest  (1,1,128,4096)",
        (1, 1, 128, 4096),
        None,
        None,
        "gamma",
        True,
        ttnn.TILE_LAYOUT,
        11603,
    ),
    (
        "interleaved, ROW_MAJOR weight      (1,1,128,4096) f32T",
        (1, 1, 128, 4096),
        None,
        None,
        "gamma_rm",
        True,
        ttnn.TILE_LAYOUT,
        11327,
    ),
    (
        "ragged-Wt interleaved              (1,1,32,4064)",
        (1, 1, 32, 4064),
        None,
        None,
        "gamma",
        False,
        ttnn.TILE_LAYOUT,
        32339,
    ),
    (
        "W non-aligned (masked reduce)      (1,1,224,1000)",
        (1, 1, 224, 1000),
        None,
        None,
        "gamma",
        False,
        ttnn.TILE_LAYOUT,
        None,
    ),
    (
        "H non-aligned                      (1,1,333,544)",
        (1, 1, 333, 544),
        None,
        None,
        "gamma",
        False,
        ttnn.TILE_LAYOUT,
        None,
    ),
    (
        "interleaved decode, all 3 operands (1,1,32,5120) gbr f32T",
        (1, 1, 32, 5120),
        None,
        None,
        "gamma_bias_residual",
        True,
        ttnn.TILE_LAYOUT,
        9614,
    ),
    (
        "interleaved decode small           (1,1,32,5120) gamma",
        (1, 1, 32, 5120),
        None,
        None,
        "gamma",
        False,
        ttnn.TILE_LAYOUT,
        7047,
    ),
]

CASES = CASES[int(os.environ.get("GUARD_FROM", "0")) :]

dev = ttnn.open_device(device_id=0)
try:
    for label, shape, ml, shard, mode, f32, lay, base in CASES:
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
            else shard_config(shard[0], shard[1], ml, layout=lay, dtype=ttnn.bfloat16, device=dev)
        )
        x = ttnn.from_torch(tx, dtype=ttnn.bfloat16, layout=lay, device=dev, memory_config=mc)
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
            r = ttnn.from_torch(tr, dtype=ttnn.bfloat16, layout=lay, device=dev, memory_config=mc)
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
        reads = []
        for _ in range(3):
            o = rms_norm_ttnn(x, **kw)
            ttnn.synchronize_device(dev)
            reads.append(ns(dev))
            ttnn.deallocate(o)
        best = min(reads)
        rel = f"{base/best:6.3f}x  (was {base})" if base else "   n/a"
        spread = (max(reads) - min(reads)) / min(reads) * 100
        print(f"RESULT {label:52s} ns={best:9.0f}  {rel}  spread={spread:4.1f}%  pcc={p:.6f}")
        for v in live:
            try:
                ttnn.deallocate(v)
            except Exception:
                pass
finally:
    ttnn.close_device(dev)
