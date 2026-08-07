"""How much of a band is convolution, and how much is scaffolding around it?

This is what sizes the fused-band kernel honestly. AUDIO_FUSION_PLAN.md prices the win as
(ops removed) x (142 us average op cost), but `branch_batch.py` showed the convolutions cost 2.8-4.2 ms
each -- ~20x that average -- so they are far past the flat-cost regime and merging them saves only
dispatch. Averaging over all ops therefore over-credits removing a conv and under-credits removing a
Slice.

The number that matters is the split: of one band's wall clock, how much is the ~8 convolutions
(irreducible work) and how much is the ~45 scaffolding ops (Halo/Move/Slice/Concat/Untilize -- what a
fused kernel actually removes)? Scaffolding time is the ceiling on the fusion win.

Measured by running a real `Activation1d` band, then the same convolutions alone, at the production
shapes. The difference is the scaffolding.
"""

import statistics
import time

import torch

import ttnn
from models.tt_dit.layers.audio_ops import SnakeBeta, _make_kaiser_sinc_kernel_1d, depthwise_tap_filter
from models.tt_dit.layers.audio_resample import Activation1d
from models.tt_dit.models.audio_vae.minimax_h3 import decoder_minimax_h3_audio  # noqa: F401

# (label, C, T) -- the tail stages, where the band count and the row counts are both highest
CASES = [("s5 C16", 16, 82806), ("s6 C8", 8, 165606)]
ITERS = 5


def timed(fn, iters=ITERS):
    fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e3


def main():
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=65536)
    try:
        print(f"{'case':<10} {'band ms':>9} {'convs alone':>12} {'scaffold ms':>12} {'scaffold %':>11}")
        print("-" * 60)
        for label, C, T in CASES:
            torch.manual_seed(0)
            x = torch.randn(2, T, C) * 0.3
            xd = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

            band = Activation1d(
                channels=C,
                activation=SnakeBeta(channels=C, mesh_device=device, dtype=ttnn.float32),
                mesh_device=device,
                dtype=ttnn.float32,
            )
            # give the snake real parameters
            band.act.load_torch_state_dict(
                {"alpha": torch.rand(1, 1, C) * 0.5 + 0.5, "beta": torch.rand(1, 1, C) * 0.5 + 0.5}, strict=False
            )

            try:
                t_band = timed(lambda: band(xd))
            except Exception as exc:  # noqa: BLE001
                print(f"{label:<10}  BAND FAILED {str(exc).splitlines()[0][:40]}")
                continue

            # the convolutions the band performs: up 2x (K=12) then down 2x (K=12), i.e. two depthwise
            # FIRs, the up one over the 2x-length signal
            up_taps = _make_kaiser_sinc_kernel_1d(0.5 / 2, 0.6 / 2, 12).tolist()
            dn_taps = _make_kaiser_sinc_kernel_1d(0.5 / 2, 0.6 / 2, 12).tolist()
            x2 = torch.randn(2, 2 * T, C) * 0.3
            x2d = ttnn.from_torch(x2, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
            cache_up, cache_dn = {}, {}

            def convs_only():
                a = depthwise_tap_filter(x2d, up_taps, 1, mesh_device=device, dtype=ttnn.float32, cache=cache_up)
                b = depthwise_tap_filter(x2d, dn_taps, 2, mesh_device=device, dtype=ttnn.float32, cache=cache_dn)
                return a, b

            try:
                t_conv = timed(convs_only)
            except Exception as exc:  # noqa: BLE001
                print(f"{label:<10} {t_band:>9.3f}  CONVS FAILED {str(exc).splitlines()[0][:34]}")
                continue

            scaffold = t_band - t_conv
            print(f"{label:<10} {t_band:>9.3f} {t_conv:>12.3f} {scaffold:>12.3f} {100 * scaffold / t_band:>10.1f}%")

        print("\nScaffolding % is the ceiling on what fusing the band can remove; the convolution")
        print("time is irreducible work that a fused kernel still has to do.")
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
