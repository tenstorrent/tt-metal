"""Do per-channel taps reproduce N separate filters, and is one wide conv cheaper than N narrow ones?

`depthwise_tap_filter` broadcast a single tap vector across every channel, which is why the polyphase
upsampler issues two conv calls over the same input. Per-channel taps let both phases go through one
conv at double the channel width. That should win twice: one op instead of two, and by
`row_cost.py` a wider row costs barely more than a narrow one.

Checks the equivalence first -- stacking two inputs along C with their two tap vectors must give
exactly what the two separate calls give -- then times both.
"""

import statistics
import time

import torch

import ttnn
from models.tt_dit.layers.audio_ops import _make_kaiser_sinc_kernel_1d, depthwise_tap_filter
from models.tt_dit.models.audio_vae.minimax_h3 import decoder_minimax_h3_audio  # noqa: F401

SHAPES = [("s6 C8  T165606", 8, 165606), ("s5 C16 T82806", 16, 82806)]
K = 7
ITERS = 5


def timed(fn, device):
    fn()
    ttnn.synchronize_device(device)
    ts = []
    for _ in range(ITERS):
        s = time.perf_counter()
        fn()
        ttnn.synchronize_device(device)
        ts.append((time.perf_counter() - s) * 1e3)
    return statistics.median(ts)


def main():
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=65536)
    try:
        print(f"{'shape':<18} {'variant':<14} {'ms':>8}  {'match':>8}")
        print("-" * 56)
        for label, C, T in SHAPES:
            torch.manual_seed(0)
            base = _make_kaiser_sinc_kernel_1d(0.5 / 2, 0.6 / 2, 2 * K).tolist()
            sub0 = [base[2 * j] for j in range(K)]
            sub1 = [base[2 * j + 1] for j in range(K)]
            x = torch.randn(2, T, C) * 0.3
            xd = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

            def two_calls():
                a = depthwise_tap_filter(xd, sub0, 1, mesh_device=device, dtype=ttnn.float32, cache=cache_a)
                b = depthwise_tap_filter(xd, sub1, 1, mesh_device=device, dtype=ttnn.float32, cache=cache_b)
                return a, b

            def one_call():
                x2 = ttnn.concat([xd, xd], dim=2)
                taps2 = [sub0] * C + [sub1] * C
                out = depthwise_tap_filter(x2, taps2, 1, mesh_device=device, dtype=ttnn.float32, cache=cache_c)
                return out

            cache_a, cache_b, cache_c = {}, {}, {}
            try:
                a, b = two_calls()
                ref = torch.cat([ttnn.to_torch(a).float(), ttnn.to_torch(b).float()], dim=2)
                got = ttnn.to_torch(one_call()).float()
                same = got.shape == ref.shape and torch.equal(got, ref)
                maxd = float((got - ref).abs().max()) if got.shape == ref.shape else float("nan")
                print(f"{label:<18} {'two calls':<14} {timed(two_calls, device):>8.2f}  {'(ref)':>8}")
                print(
                    f"{label:<18} {'one wide call':<14} {timed(one_call, device):>8.2f}  {str(same):>8}  maxdiff={maxd:.2e}"
                )
            except Exception as exc:  # noqa: BLE001
                print(f"{label:<18} FAILED {str(exc).splitlines()[0][:60]}")
            print()
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
