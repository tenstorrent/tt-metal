"""Per-shape FIR microbenchmark: correctness vs torch and wall time for each execution path.

Covers the five production depthwise-FIR shapes from MiniMaxH3_audio_decode_kernels.md §3, and for
each one runs the three paths `depthwise_tap_filter` can take -- L1_FULL conv1d, DRAM-sliced conv1d,
and the shift-multiply-add MAC form -- reporting rel_rmse against a float64 torch golden plus the
median of N timed calls.

Run under the worktree PYTHONPATH; see run_bench.sh for the env.
"""

import os
import statistics
import time

import torch

import ttnn
from models.tt_dit.layers.audio_ops import _make_kaiser_sinc_kernel_1d, depthwise_tap_filter

# Import for the side effect the audio tests rely on: registering the H3 conv blockings, without which
# every conv here measures a different op than production (STATE.md am. 111).
from models.tt_dit.models.audio_vae.minimax_h3 import decoder_minimax_h3_audio  # noqa: F401

# (label, C, T_pad, K, stride) -- the §3 table. T_pad is the padded input length the filter sees.
SHAPES = [
    ("s0_up   C512 K7  s1", 512, 1041, 7, 1),
    ("s0_down C512 K12 s2", 512, 2081, 12, 2),
    ("s5_up   C16  K7  s1", 16, 82806, 7, 1),
    ("s6_up   C8   K7  s1", 8, 165606, 7, 1),
    ("s6_down C8   K12 s2", 8, 331211, 12, 2),
]

ITERS = int(os.environ.get("FIR_BENCH_ITERS", "5"))


def golden(x: torch.Tensor, taps, stride: int, t_out: int) -> torch.Tensor:
    """float64 reference, computed as a strided depthwise FIR."""
    xd = x.double()
    acc = torch.zeros(xd.shape[0], t_out, xd.shape[2], dtype=torch.float64)
    for k, tap in enumerate(taps):
        acc += float(tap) * xd[:, k : k + stride * t_out : stride, :]
    return acc


def main():
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=65536)
    try:
        print(f"{'shape':<22} {'path':<5} {'rel_rmse':>11} {'ms':>9}  note")
        print("-" * 72)
        for label, C, T_pad, K, stride in SHAPES:
            t_out = (T_pad - K) // stride + 1
            torch.manual_seed(0)
            taps = _make_kaiser_sinc_kernel_1d(0.5 / 2, 0.6 / 2, K).tolist()
            x = torch.randn(1, T_pad, C) * 0.3
            ref = golden(x, taps, stride, t_out)
            denom = ref.std()

            # (label, MINIMAX_H3_AUDIO_CONV1D_L1, MINIMAX_H3_AUDIO_DEPTHWISE_SPLIT, expected path)
            for path, env, split, expect in (
                ("l1", "aggressive", "off", "l1"),
                ("dram", "off", "off", "dram"),
                ("spl-w", "aggressive", "weight", "l1"),
                ("spl-f", "aggressive", "full", "l1"),
                ("mac", "aggressive", "off", "mac"),
            ):
                os.environ["MINIMAX_H3_AUDIO_DEPTHWISE_MAC"] = "1" if path == "mac" else "0"
                os.environ["MINIMAX_H3_AUDIO_CONV1D_L1"] = env
                os.environ["MINIMAX_H3_AUDIO_DEPTHWISE_SPLIT"] = split
                cache = {}
                note = ""
                try:
                    xd = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
                    out = depthwise_tap_filter(xd, taps, stride, mesh_device=device, dtype=ttnn.float32, cache=cache)
                    ttnn.synchronize_device(device)
                    got = ttnn.to_torch(out).float()
                    if got.shape[1] != t_out:
                        note = f"T_out {got.shape[1]} != {t_out}"
                    err = float((got.double() - ref).pow(2).mean().sqrt() / denom)

                    times = []
                    for _ in range(ITERS):
                        xd2 = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
                        ttnn.synchronize_device(device)
                        start = time.perf_counter()
                        depthwise_tap_filter(xd2, taps, stride, mesh_device=device, dtype=ttnn.float32, cache=cache)
                        ttnn.synchronize_device(device)
                        times.append((time.perf_counter() - start) * 1e3)
                    ms = statistics.median(times)
                    # Report the path actually taken -- `auto`/`on` can still fall back internally.
                    taken = [v for k, v in cache.items() if isinstance(k, tuple) and k and k[0] == "path"]
                    if taken and taken[0] != expect:
                        note = (note + " " if note else "") + f"fell back to {taken[0]}"
                    print(f"{label:<22} {path:<5} {err:>11.3e} {ms:>9.2f}  {note}")
                except Exception as exc:  # noqa: BLE001 - a path that cannot run is a result here
                    first = str(exc).strip().splitlines()[0][:60]
                    print(f"{label:<22} {path:<5} {'-':>11} {'-':>9}  FAILED: {first}")
            print()
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
