"""Perf 3 -- the GUARD SET: one representative per distinct kernel path x layout x
placement.  Same set as Perf 2's, so the two rounds' tables are comparable, plus the
paths this round's graduations newly reach.

Run it TWICE in the same session -- once with the pre-round op checked out
(`git checkout <perf2-tip> -- kernels rms_norm_ttnn_program_descriptor.py`) and once
with HEAD restored -- and diff.  `RMS_TAG` labels the column.
"""
import os
import statistics

for k, v in (
    ("TT_METAL_DEVICE_PROFILER", "1"),
    ("TT_METAL_PROFILER_MID_RUN_DUMP", "1"),
    ("TT_METAL_PROFILER_CPP_POST_PROCESS", "1"),
    ("TT_METAL_LOGGER_LEVEL", "error"),
):
    os.environ.setdefault(k, v)
import ttnn  # noqa: E402
from ttnn.operations.rms_norm_ttnn import rms_norm_ttnn, torch_rms_norm_ttnn  # noqa: E402
from eval.sharding import shard_config  # noqa: E402

K = "DEVICE KERNEL DURATION [ns]"
TAG = os.environ.get("RMS_TAG", "run")
READS = int(os.environ.get("RMS_READS", "3"))
_ML = ttnn.TensorMemoryLayout

# label, shape, gamma_mode, fp32_dest, act_layout, gamma_layout, memory_layout, shard
CASES = [
    ("HEIGHT native no-combine  (1,1,2048,256)", (1, 1, 2048, 256), "gamma", False, "T", "T", _ML.HEIGHT_SHARDED, ([32, 256], (8, 8))),
    ("ROW_RESIDENT+D39 compact  (1,1,8192,7168) gbr f32T", (1, 1, 8192, 7168), "gamma_bias_residual", True, "T", "T", _ML.INTERLEAVED, None),
    ("INT RESIDENT combine=F    (1,1,8192,2304)", (1, 1, 8192, 2304), "gamma", False, "T", "T", _ML.INTERLEAVED, None),
    ("ROW_MAJOR act, TILE w     (1,1,8192,1024)", (1, 1, 8192, 1024), "gamma", False, "R", "T", _ML.INTERLEAVED, None),
    ("INT RESIDENT D42  FOCUS   (1,1,8192,1024)", (1, 1, 8192, 1024), "gamma", False, "T", "T", _ML.INTERLEAVED, None),
    ("ROW_RESIDENT tiled        (1,1,8192,5120)", (1, 1, 8192, 5120), "gamma", False, "T", "T", _ML.INTERLEAVED, None),
    ("INT TILE w f32dest        (1,1,128,4096)", (1, 1, 128, 4096), "gamma", True, "T", "T", _ML.INTERLEAVED, None),
    ("STREAM                    (1,1,1024,16384) gbr", (1, 1, 1024, 16384), "gamma_bias_residual", False, "T", "T", _ML.INTERLEAVED, None),
    ("WIDTH native flat combine (1,1,32,7168)", (1, 1, 32, 7168), "gamma", False, "T", "T", _ML.WIDTH_SHARDED, ([32, 256], (7, 4))),
    ("INT ROW_MAJOR w f32dest   (1,1,128,4096)", (1, 1, 128, 4096), "gamma", True, "T", "R", _ML.INTERLEAVED, None),
    ("BLOCK native slot tree    (1,1,8192,1024)", (1, 1, 8192, 1024), "gamma", False, "T", "T", _ML.BLOCK_SHARDED, ([1024, 128], (8, 8))),
    ("ROW_MAJOR BAND WIDTH-shrd (1,1,256,512)", (1, 1, 256, 512), "gamma", False, "R", "T", _ML.WIDTH_SHARDED, ([256, 8], (64, 1))),
    ("ragged-Wt interleaved     (1,1,32,4064)", (1, 1, 32, 4064), "gamma", False, "T", "T", _ML.INTERLEAVED, None),
    ("RM act 2-core line        (1,1,64,128)", (1, 1, 64, 128), "gamma", False, "R", "T", _ML.INTERLEAVED, None),
    ("W non-aligned mask reduce (1,1,224,1000)", (1, 1, 224, 1000), "gamma", False, "T", "T", _ML.INTERLEAVED, None),
    ("H non-aligned             (1,1,333,544)", (1, 1, 333, 544), "gamma", False, "T", "T", _ML.INTERLEAVED, None),
]


def main():
    import torch

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

    lay = {"T": ttnn.TILE_LAYOUT, "R": ttnn.ROW_MAJOR_LAYOUT}
    dev = ttnn.open_device(device_id=0)
    try:
        for label, shape, mode, f32, al, gl, ml, shard in CASES:
            W = shape[-1]
            try:
                cfg = ttnn.ComputeConfigDescriptor()
                cfg.math_fidelity = ttnn.MathFidelity.HiFi2
                cfg.fp32_dest_acc_en = f32
                cfg.math_approx_mode = False
                torch.manual_seed(0)
                tx = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
                if shard is not None:
                    mc = shard_config(shard[0], shard[1], ml, layout=lay[al], dtype=ttnn.bfloat16, device=dev)
                else:
                    mc = ttnn.DRAM_MEMORY_CONFIG
                x = ttnn.from_torch(tx, dtype=ttnn.bfloat16, layout=lay[al], device=dev, memory_config=mc)
                kw = dict(epsilon=1e-12, compute_kernel_config=cfg, memory_config=x.memory_config())
                ref = dict(input_tensor=tx.float())
                torch.manual_seed(1)
                tg = torch.randn(1, 1, 1, W, dtype=torch.float32).to(torch.bfloat16)
                kw["weight"] = ttnn.from_torch(tg, dtype=ttnn.bfloat16, layout=lay[gl], device=dev)
                ref["weight"] = tg.float()
                if "bias" in mode:
                    torch.manual_seed(2)
                    tb = torch.randn(1, 1, 1, W, dtype=torch.float32).to(torch.bfloat16)
                    kw["bias"] = ttnn.from_torch(tb, dtype=ttnn.bfloat16, layout=lay[gl], device=dev)
                    ref["bias"] = tb.float()
                if "residual" in mode:
                    torch.manual_seed(3)
                    tr = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
                    kw["residual_input_tensor"] = ttnn.from_torch(
                        tr, dtype=ttnn.bfloat16, layout=lay[al], device=dev, memory_config=mc
                    )
                    ref["residual_input_tensor"] = tr.float()
                exp = torch_rms_norm_ttnn(**ref, epsilon=1e-12)
                o = rms_norm_ttnn(x, **kw)
                p = pcc(ttnn.to_torch(o), exp)
                ttnn.deallocate(o)
                ttnn.synchronize_device(dev)
                ns(dev)
                s = []
                for _ in range(READS):
                    o = rms_norm_ttnn(x, **kw)
                    ttnn.synchronize_device(dev)
                    v = ns(dev)
                    ttnn.deallocate(o)
                    if v == v:
                        s.append(v)
                print(f"RESULT {TAG} | {label} | ns={min(s):9.0f} med={statistics.median(s):9.0f} pcc={p:.6f}", flush=True)
            except Exception as e:
                print(f"RESULT {TAG} | {label} | ERROR {type(e).__name__}: {str(e)[:110]}", flush=True)
    finally:
        ttnn.close_device(dev)


main()
