"""Can a smaller act_block_h make L1_FULL fit, and does that cut ops enough to matter?

ttnn.conv1d's DRAM slicing loop turns one call into PaddedSlice -> Halo -> Move -> Conv2d ->
SliceWrite, so 345 Python calls become ~2070 device ops -- and at ~142 us/op that is ~290 ms. L1_FULL
skips the loop entirely.

`verify` mode found only 1 of 42 production shapes could use L1, but the failures were circular-buffer
sizing ("Statically allocated circular buffers ... clash with L1 buffers"), not the activation failing
to fit. CB size is driven by act_block_h, which Conv1dConfig exposes as act_block_h_override. So sweep
it: for each real shape, find the largest override where L1_FULL both runs and is bit-exact against
the DRAM path, and time it.

Shapes are the ones `verify` actually logged from a decode, at the production batch of 2.
"""

import os
import statistics
import time

import torch

import ttnn

from models.tt_dit.models.audio_vae.minimax_h3 import decoder_minimax_h3_audio  # noqa: F401
from models.tt_dit.layers.audio_ops import _make_kaiser_sinc_kernel_1d, depthwise_tap_filter

# (label, C, T_pad, K, stride) -- observed in the verify log of a real decode.
SHAPES = [
    ("C16  T165606 K7", 16, 165606, 7, 1),
    ("C32  T124206 K7", 32, 124206, 7, 1),
    ("C8   T331211 K12", 8, 331211, 12, 2),
]
OVERRIDES = [0, 32, 64, 128, 256, 512]
ITERS = 3


def run(device, x, taps, stride, mode, override):
    os.environ["MINIMAX_H3_AUDIO_CONV1D_L1"] = mode
    os.environ["MINIMAX_H3_AUDIO_DEPTHWISE_MAC"] = "0"
    os.environ["MINIMAX_H3_AUDIO_ACT_BLOCK_H"] = str(override)
    xd = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    return depthwise_tap_filter(xd, taps, stride, mesh_device=device, dtype=ttnn.float32, cache={})


def main():
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=65536)
    try:
        print(f"{'shape':<18} {'mode':<10} {'abh':>5} {'ms':>8}  {'exact':>6}")
        print("-" * 58)
        for label, C, T_pad, K, stride in SHAPES:
            torch.manual_seed(0)
            taps = _make_kaiser_sinc_kernel_1d(0.5 / 2, 0.6 / 2, K).tolist()
            x = torch.randn(2, T_pad, C) * 0.3
            try:
                ref_t = ttnn.to_torch(run(device, x, taps, stride, "off", 0)).float()
            except Exception as exc:  # noqa: BLE001
                print(f"{label:<18} {'dram':<10} {0:>5}  FAILED {str(exc).splitlines()[0][:40]}")
                continue

            def timed(mode, ovr):
                run(device, x, taps, stride, mode, ovr)
                ttnn.synchronize_device(device)
                ts = []
                for _ in range(ITERS):
                    s = time.perf_counter()
                    run(device, x, taps, stride, mode, ovr)
                    ttnn.synchronize_device(device)
                    ts.append((time.perf_counter() - s) * 1e3)
                return statistics.median(ts)

            print(f"{label:<18} {'dram':<10} {0:>5} {timed('off', 0):>8.2f}  {'ref':>6}")
            for ovr in OVERRIDES:
                try:
                    got = ttnn.to_torch(run(device, x, taps, stride, "aggressive", ovr)).float()
                    exact = torch.equal(got, ref_t)
                    print(f"{label:<18} {'l1':<10} {ovr:>5} {timed('aggressive', ovr):>8.2f}  {str(exact):>6}")
                except Exception as exc:  # noqa: BLE001
                    first = str(exc).splitlines()[0]
                    why = "L1 clash" if "circular" in first or "clash" in first else first[:34]
                    print(f"{label:<18} {'l1':<10} {ovr:>5} {'-':>8}  {why}")
            print()
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
