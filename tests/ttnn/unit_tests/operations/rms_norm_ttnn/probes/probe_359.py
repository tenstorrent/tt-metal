"""Is the fold's SERIAL accumulation depth a real precision cost?  ADVERSARIAL inputs.

`DEST_ACC_SQUARE_MAX_WT = 8` is a PRECISION bound: the fold accumulates the chunk's x^2
tiles serially inside a DEST register that is 16-bit at fp32_dest_acc_en=False, while the
packed path's reduce accumulates PAIRWISE.  On randn the fold measures MORE accurate than
the packed path (the packed path round-trips every x^2 tile through a bf16 L1 CB), so this
probe feeds the accumulation the inputs that are supposed to break it:

  randn     the golden suite's distribution -- the control.
  ones      every term of the sum EQUAL and positive.  This is the textbook worst case for
            a serial sum: the running total grows monotonically, so the k-th add rounds at
            k*a and the relative error accumulates as ~n*eps/2 (n == the fold depth).
            Pairwise accumulation is ~log2(n)*eps/2 on the same data.
  graded    x scaled by 2^(-10..10) across the width, so the x^2 terms span 40 binades:
            once the accumulator has swallowed a big term, small ones are absorbed --
            and WHICH terms are absorbed depends on the accumulation ORDER.
  spike     one lane per row 2^10 larger than the rest.

Reference: float64, taken from the tensor the DEVICE holds, so what is reported is the
kernel's error and not the input quantization's.  The user's precision config is FIXED at
the focus case's (bf16 / HiFi2 / fp32_dest_acc_en=False / math_approx_mode=False) for
every variant; W = 1024 so WT_CHUNK = 32 and the fold depth is the variant's own.

  RMS_SVARIANTS=base,grp8,grp16,flatinf RMS_SPATTERNS=randn,ones,graded,spike
  scripts/tt-probe.sh rms_norm_ttnn < precision_stress.py
"""

import importlib.util
import os
import sys
from pathlib import Path

os.environ["RMS_NO_MAIN"] = "1"
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")

HERE = Path(
    os.environ.get("RMS_EXP_DIR")
    or "/localdev/dnijemcevic/2026_09_04/1519_dnijemcevic_agent_eval_new/clones/rms_norm_ttnn_run1/tt-metal"
    "/ttnn/ttnn/operations/rms_norm_ttnn/perf_experiments/square_fold_ceiling"
)
spec = importlib.util.spec_from_file_location("bench_fold", HERE / "bench_fold.py")
B = importlib.util.module_from_spec(spec)
sys.modules["bench_fold"] = B
spec.loader.exec_module(B)

import ttnn  # noqa: E402

ROWS = int(os.environ.get("RMS_SROWS", "1024"))
W = int(os.environ.get("RMS_SW", "1024"))


def make(pattern, torch):
    torch.manual_seed(0)
    if pattern == "randn":
        return torch.randn(1, 1, ROWS, W, dtype=torch.float32)
    if pattern == "ones":
        return torch.ones(1, 1, ROWS, W, dtype=torch.float32)
    if pattern == "graded":
        e = torch.linspace(-10, 10, W, dtype=torch.float32)
        return (2.0**e).reshape(1, 1, 1, W) * (1.0 + 0.1 * torch.randn(1, 1, ROWS, W, dtype=torch.float32))
    if pattern == "spike":
        t = torch.randn(1, 1, ROWS, W, dtype=torch.float32) * 2.0**-5
        t[..., W // 2] = 2.0**5
        return t
    raise KeyError(pattern)


def main():
    import torch

    variants = os.environ.get("RMS_SVARIANTS", "base,grp8,grp16,flatinf").split(",")
    patterns = os.environ.get("RMS_SPATTERNS", "randn,ones,graded,spike").split(",")
    device = ttnn.open_device(device_id=0)
    try:
        for pattern in patterns:
            tx = make(pattern, torch)
            x = ttnn.from_torch(
                tx,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            torch.manual_seed(1)
            tg = torch.randn(1, 1, 1, W, dtype=torch.float32)
            g = ttnn.from_torch(tg, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            xd = ttnn.to_torch(x).double()
            gd = ttnn.to_torch(g).double()
            ref = B.TORCH_REF(xd, epsilon=1e-12, weight=gd)
            cfg = ttnn.ComputeConfigDescriptor()
            cfg.math_fidelity = ttnn.MathFidelity.HiFi2
            cfg.fp32_dest_acc_en = False
            cfg.math_approx_mode = False
            for label in variants:
                B._reset()
                B.VARIANTS[label]()
                out = B.OP.rms_norm_ttnn(
                    x,
                    epsilon=1e-12,
                    weight=g,
                    compute_kernel_config=cfg,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                got = ttnn.to_torch(out).double()
                ttnn.deallocate(out)
                print(
                    f"RESULT {pattern:8s} {label:10s} pcc={B.pcc(got, ref):.7f} relrms={B.relrms(got, ref):.6f} "
                    f"maxrel={float(((got - ref).abs() / (ref.abs() + 1e-30)).max()):.5f}",
                    flush=True,
                )
            ttnn.deallocate(x)
            ttnn.deallocate(g)
    finally:
        B._reset()
        ttnn.close_device(device)


main()
