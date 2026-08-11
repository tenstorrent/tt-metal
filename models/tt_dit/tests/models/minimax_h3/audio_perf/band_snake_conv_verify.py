"""A/B the band with the snake folded into the phase conv against the band with it as its own op.

The fused band (`MINIMAX_H3_AUDIO_FUSE_BAND=1`) is already verified exact, so it is the reference
here: turning on `MINIMAX_H3_AUDIO_FUSE_SNAKE_CONV` must not move the output beyond fp32 round-off.
What it should move is the op count -- two `snake_beta` calls and their tilize/untilize disappear per
band, replaced by nothing at all, because the conv that produced the phase now applies the snake to
its own output before packing.

Run at the production tail shapes, where the bands are cheapest per element and so most sensitive to
per-op overhead. `TT_CONV1D_SNAKE_PARAMS` is not set here: `depthwise_tap_filter_snake` scopes it to
its own conv1d calls.
"""

import os
import statistics
import time

import torch

import ttnn
from models.tt_dit.layers import audio_resample
from models.tt_dit.layers.audio_ops import SnakeBeta
from models.tt_dit.layers.audio_resample import Activation1d
from models.tt_dit.models.audio_vae.minimax_h3 import decoder_minimax_h3_audio  # noqa: F401

SHAPES = [
    ("s3 C64  T20701", 64, 20701),
    ("s4 C32  T41403", 32, 41403),
    ("s5 C16  T82806", 16, 82806),
    ("s6 C8   T165606", 8, 165606),
]
ITERS = int(os.environ.get("BAND_ITERS", "3"))


def main():
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=65536)
    os.environ["MINIMAX_H3_AUDIO_FUSE_BAND"] = "1"
    try:
        print(f"{'shape':<18} {'variant':<12} {'ms':>9} {'speedup':>8}  {'rel_rmse vs base':>17}")
        print("-" * 72)
        worst = 0.0
        for label, C, T in SHAPES:
            torch.manual_seed(0)
            act = SnakeBeta(C, mesh_device=device, dtype=ttnn.float32)
            act.load_torch_state_dict({"alpha": torch.rand(C) + 0.5, "beta": torch.rand(C) + 0.5})
            band = Activation1d(channels=C, activation=act, mesh_device=device, dtype=ttnn.float32)

            x = torch.randn(2, T, C) * 0.3
            xd = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

            results, times = {}, {}
            for variant, flag in (("snake-op", "0"), ("snake-in-conv", "1")):
                os.environ["MINIMAX_H3_AUDIO_FUSE_SNAKE_CONV"] = flag
                assert audio_resample.fuse_snake_into_conv_enabled() == (flag == "1")
                # Warm the weight cache (the fused path prepares once per band) before timing.
                out = band(xd)
                ttnn.synchronize_device(device)
                results[variant] = ttnn.to_torch(out).float()
                ttnn.deallocate(out)

                samples = []
                for _ in range(ITERS):
                    t0 = time.perf_counter()
                    out = band(xd)
                    ttnn.synchronize_device(device)
                    samples.append((time.perf_counter() - t0) * 1e3)
                    ttnn.deallocate(out)
                times[variant] = statistics.median(samples)

            base = results["snake-op"].double()
            for variant in ("snake-op", "snake-in-conv"):
                d = (results[variant].double() - base).abs()
                rel = float(d.pow(2).mean().sqrt() / base.std())
                worst = max(worst, rel)
                sp = times["snake-op"] / times[variant]
                print(f"{label:<18} {variant:<12} {times[variant]:9.2f} {sp:7.2f}x  {rel:17.3e}")
            print("-" * 72)
        print(f"\nworst rel_rmse vs the snake-op band: {worst:.3e}")
        print("PASS" if worst < 1e-6 else "FAIL -- folding the snake into the conv changed the result")
    finally:
        os.environ.pop("MINIMAX_H3_AUDIO_FUSE_SNAKE_CONV", None)
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
