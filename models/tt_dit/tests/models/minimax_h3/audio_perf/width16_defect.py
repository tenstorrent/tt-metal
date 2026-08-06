"""Is there a width-16 defect in the depthwise conv, and which side of the disagreement is wrong?

Two anomalies point at C=16. L1_FULL returns a wrong answer there (max abs diff 1.456) while C=8, 32
and 512 are bit-exact. And merging two C=8 filters into one C=16 call disagrees with running them
separately by 4.78e-04, which should be impossible now that conv1d is exact at both widths taken
alone. A shared width-16 defect would explain both.

Neither of those says *which* result is wrong -- they only say two results differ. This scores each
width against a float64 golden, so the wrong one is named rather than inferred.

For each width the same logical filter is run over the same data, laid out at C = 8, 16, 24, 32, 40:
if the error jumps at 16 (and only at 16) the defect is width-specific; if it grows with padding waste
generally, it is something else.
"""

import torch

import ttnn
from models.tt_dit.layers.audio_ops import _make_kaiser_sinc_kernel_1d, depthwise_tap_filter
from models.tt_dit.models.audio_vae.minimax_h3 import decoder_minimax_h3_audio  # noqa: F401

T_PAD, K = 82806, 7
WIDTHS = [8, 16, 24, 32, 40, 64]


def main():
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=65536)
    try:
        taps = _make_kaiser_sinc_kernel_1d(0.5 / 2, 0.6 / 2, K).tolist()
        t_out = T_PAD - K + 1
        print(f"{'C':>4} {'rel_rmse vs float64':>20} {'max abs':>12}   verdict")
        print("-" * 58)
        for C in WIDTHS:
            torch.manual_seed(0)
            x = torch.randn(2, T_PAD, C) * 0.3
            xd64 = x.double()
            golden = torch.zeros(2, t_out, C, dtype=torch.float64)
            for k, tap in enumerate(taps):
                golden += float(tap) * xd64[:, k : k + t_out, :]
            try:
                xd = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
                got = ttnn.to_torch(
                    depthwise_tap_filter(xd, taps, 1, mesh_device=device, dtype=ttnn.float32, cache={})
                ).float()
                err = float((got.double() - golden).pow(2).mean().sqrt() / golden.std())
                mx = float((got.double() - golden).abs().max())
                verdict = "fp32-grade" if err < 1e-6 else "DEGRADED"
                print(f"{C:>4} {err:>20.3e} {mx:>12.3e}   {verdict}")
            except Exception as exc:  # noqa: BLE001
                print(f"{C:>4}  FAILED {str(exc).splitlines()[0][:50]}")
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
