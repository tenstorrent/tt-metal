"""A/B the fused Activation1d band against the literal one: correctness and time.

Builds a real Activation1d with a real SnakeBeta at the production tail shapes and runs it both ways
by toggling MINIMAX_H3_AUDIO_FUSE_BAND. The unfused path is the reference; the fused path claims to
be exact, so anything above fp32 round-off is a bug in the index algebra.
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

# (label, C, T) at the tail; T is the band's input length.
SHAPES = [
    ("s3 C64  T20701", 64, 20701),
    ("s4 C32  T41403", 32, 41403),
    ("s5 C16  T82806", 16, 82806),
    ("s6 C8   T165606", 8, 165606),
]
ITERS = int(os.environ.get("BAND_ITERS", "3"))


def main():
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=65536)
    try:
        print(f"{'shape':<18} {'variant':<9} {'ms':>9}  {'rel_rmse vs unfused':>20}")
        print("-" * 64)
        for label, C, T in SHAPES:
            torch.manual_seed(0)
            act = SnakeBeta(C, mesh_device=device, dtype=ttnn.float32)
            act.load_torch_state_dict({"alpha": torch.rand(C) + 0.5, "beta": torch.rand(C) + 0.5})
            band = Activation1d(channels=C, activation=act, mesh_device=device, dtype=ttnn.float32)

            x = torch.randn(2, T, C) * 0.3
            xd = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

            results = {}
            for variant, flag in (("unfused", "0"), ("fused", "1")):
                os.environ["MINIMAX_H3_AUDIO_FUSE_BAND"] = flag
                assert audio_resample.fuse_band_enabled() == (flag == "1")
                out = band(xd)
                ttnn.synchronize_device(device)
                results[variant] = ttnn.to_torch(out).float()

                ts = []
                for _ in range(ITERS):
                    start = time.perf_counter()
                    band(xd)
                    ttnn.synchronize_device(device)
                    ts.append((time.perf_counter() - start) * 1e3)
                ms = statistics.median(ts)
                if variant == "unfused":
                    print(f"{label:<18} {variant:<9} {ms:>9.2f}  {'(reference)':>20}")
                else:
                    a, b = results["unfused"].double(), results["fused"].double()
                    if a.shape != b.shape:
                        print(f"{label:<18} {variant:<9} {ms:>9.2f}  SHAPE {tuple(b.shape)} != {tuple(a.shape)}")
                    else:
                        err = float((b - a).pow(2).mean().sqrt() / a.std())
                        print(f"{label:<18} {variant:<9} {ms:>9.2f}  {err:>20.3e}")
            print()
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
