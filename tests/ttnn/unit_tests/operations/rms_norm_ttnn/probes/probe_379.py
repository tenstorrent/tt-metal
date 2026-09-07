"""passb_op_count -- the ONE numeric asymmetry the pass-B reorder introduces.

Shipped order:  (x * stat) * gamma  -- the bf16 intermediate is NORMALIZED, |.| ~ 1.
Reordered:      (x * gamma) * stat  -- the bf16 intermediate is UN-normalized, so it
                overflows to inf when |x * gamma| > bf16 max (~3.39e38) even though
                the final result is representable.

This probes that boundary directly: |x| = 1e18 (x^2 = 1e36 still fits the bf16
cb_x_squared, so the reduce itself is fine) with gamma = 1e21.  Shipped order
answers 1e21; the reorder answers inf.  Nothing in the suite's INPUTS reaches
here (they are randn-drawn), but it is a real domain statement about the reorder.
"""

import os

os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")

from pathlib import Path

import ttnn

from ttnn.operations.rms_norm_ttnn import rms_norm_ttnn
import ttnn.operations.rms_norm_ttnn.rms_norm_ttnn_program_descriptor as PD

# tt-probe.sh COPIES this script into the probes dir, so __file__ is not the
# experiment dir -- the variant path has to be absolute.
HERE = Path(
    "/localdev/dnijemcevic/2026_09_04/1519_dnijemcevic_agent_eval_new/clones/rms_norm_ttnn_run1/tt-metal"
    "/ttnn/ttnn/operations/rms_norm_ttnn/perf_experiments/passb_op_count"
)


def main():
    import torch

    dev = ttnn.open_device(device_id=0)
    saved = PD.KERNEL_DIR
    try:
        for xmag, gmag in ((1e18, 1e21), (1e18, 1e15), (1.0, 1.0)):
            tx = torch.full((1, 1, 32, 1024), xmag, dtype=torch.float32).to(torch.bfloat16)
            tg = torch.full((1, 1, 1, 1024), gmag, dtype=torch.float32).to(torch.bfloat16)
            for v in ("base", "swap"):
                PD.KERNEL_DIR = HERE / f"k_{v}"
                cfg = ttnn.ComputeConfigDescriptor()
                cfg.math_fidelity = ttnn.MathFidelity.HiFi2
                cfg.fp32_dest_acc_en = False
                cfg.math_approx_mode = False
                x = ttnn.from_torch(
                    tx, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
                g = ttnn.from_torch(tg, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
                out = rms_norm_ttnn(x, weight=g, epsilon=1e-12, compute_kernel_config=cfg)
                got = ttnn.to_torch(out).float()
                n_inf = int(torch.isinf(got).sum())
                n_nan = int(torch.isnan(got).sum())
                print(
                    f"RESULT x={xmag:.0e} gamma={gmag:.0e} {v:5s} "
                    f"out[0,0,0,0]={got[0, 0, 0, 0].item():.4e} inf={n_inf} nan={n_nan} (expect {gmag:.0e})",
                    flush=True,
                )
                ttnn.deallocate(out)
                ttnn.deallocate(x)
                ttnn.deallocate(g)
                PD.KERNEL_DIR = saved
    finally:
        PD.KERNEL_DIR = saved
        ttnn.close_device(dev)


main()
