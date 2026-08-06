"""Gate 1: is the L1 routing change bit-exact against the previous DRAM path?

The FIR sweep showed the two paths agreeing to 4 significant figures of rel_rmse, which is suggestive
but not proof. This compares the output tensors directly, at fp32 and at bf16 (the dtype every
existing LTX caller uses), so a claim of "routing change, not a numerics change" is either established
or falsified.
"""

import os

import torch

import ttnn
from models.tt_dit.layers.audio_ops import _make_kaiser_sinc_kernel_1d, depthwise_tap_filter
from models.tt_dit.models.audio_vae.minimax_h3 import decoder_minimax_h3_audio  # noqa: F401  (blockings)

SHAPES = [
    ("s0_down C512 K12 s2", 512, 2081, 12, 2),
    ("s5_up   C16  K7  s1", 16, 82806, 7, 1),
    ("s6_up   C8   K7  s1", 8, 165606, 7, 1),
    ("s6_down C8   K12 s2", 8, 331211, 12, 2),
]


def run(device, x, taps, stride, dtype, mode):
    os.environ["MINIMAX_H3_AUDIO_CONV1D_L1"] = mode
    os.environ["MINIMAX_H3_AUDIO_DEPTHWISE_MAC"] = "0"
    os.environ["MINIMAX_H3_AUDIO_DEPTHWISE_SPLIT"] = "off"
    xd = ttnn.from_torch(x, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    out = depthwise_tap_filter(xd, taps, stride, mesh_device=device, dtype=dtype, cache={})
    ttnn.synchronize_device(device)
    return ttnn.to_torch(out).float()


def main():
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=65536)
    try:
        print(f"{'shape':<22} {'dtype':<9} {'bit-exact':<10} {'max abs diff':>13}")
        print("-" * 60)
        for label, C, T_pad, K, stride in SHAPES:
            torch.manual_seed(0)
            taps = _make_kaiser_sinc_kernel_1d(0.5 / 2, 0.6 / 2, K).tolist()
            x = torch.randn(2, T_pad, C) * 0.3
            for dtype, name in ((ttnn.float32, "float32"), (ttnn.bfloat16, "bfloat16")):
                try:
                    a = run(device, x, taps, stride, dtype, "off")
                    b = run(device, x, taps, stride, dtype, "aggressive")
                    same = torch.equal(a, b)
                    diff = float((a - b).abs().max())
                    print(f"{label:<22} {name:<9} {str(same):<10} {diff:>13.3e}")
                except Exception as exc:  # noqa: BLE001
                    print(f"{label:<22} {name:<9} {'ERROR':<10} {str(exc).splitlines()[0][:40]}")
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
