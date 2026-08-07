"""What does folding the activation into the conv actually save, per band?

Step 2a plans to carry snake's per-channel alpha/inv_beta on the depthwise conv so the separate
activation op disappears. That is a six-plus-file C++ change (optional input tensor or an extended
weights CB), so it is worth pricing before building. PROFILE_2026_08_06.txt attributes 140.3 ms to
127 Ternary ops and 29.0 ms to 127 TilizeWithValPadding -- one of each per band -- but whether fusing
recovers that is an assumption, and this session has watched three such assumptions evaporate.

The mechanism can be priced today with no new plumbing, because a *scalar* fused activation already
works on this kernel (relu/gelu, verified 7.6e-08). Three variants at the production tail shapes:

    conv                 baseline
    conv + fused relu    activation rides in DST on the last tap -- what Step 2a would look like
    conv + separate relu the activation as its own op, which is what the band does today

(separate - fused) is the per-band saving fusing buys. x127 bands is the Step 2a payoff. If that lands
near the 169 ms the profile attributes to Ternary + Tilize, the plumbing is worth writing; if it comes
out near zero, Step 2a is another evaporating win and the effort belongs in the full band op instead.

**KNOWN BROKEN -- do not quote the number this currently prints.** As written it reports

    case        conv     fused  separate   saving   x127
    s5 C16    13.707     9.645    12.230    2.584    328 ms
    s6 C8      9.778     7.812     8.016    0.204     26 ms

and the fused conv comes out *faster than the plain conv*, which is not physical: an activation cannot
make a convolution 30 % quicker. Setting `activation` evidently also changes the program or sharding
path selected, so the two arms are not the same convolution and the difference is not the activation.

To fix: force both arms onto an identical conv config -- same shard layout, same act_block_h, same
core grid, ideally by constructing one config and only toggling the activation field -- and confirm
plain and fused agree to within noise on the *conv* portion before trusting the saving.
"""

import statistics
import time

import torch

import ttnn
from models.tt_dit.layers.audio_ops import _make_kaiser_sinc_kernel_1d
from models.tt_dit.models.audio_vae.minimax_h3 import decoder_minimax_h3_audio  # noqa: F401

CASES = [("s5 C16", 16, 82806), ("s6 C8", 8, 165606), ("s6up C8", 8, 331212)]
K = 12
ITERS = 5
BANDS = 127


def timed(fn):
    fn()
    ts = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e3


def make_conv(device, C, T, activation):
    """A depthwise conv1d closure at this shape, optionally with a fused activation."""
    # `activation` is a UnaryWithParam, not a string -- passing a string is what the older harness got
    # wrong and why it reported the config as rejected.
    conv_config = ttnn.Conv1dConfig(
        weights_dtype=ttnn.float32,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        **({"activation": ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)} if activation else {}),
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    taps = _make_kaiser_sinc_kernel_1d(0.5 / 2, 0.6 / 2, K).tolist()
    wt = torch.tensor(taps, dtype=torch.float32).reshape(1, 1, K).expand(C, 1, K).contiguous()
    weight = ttnn.from_torch(
        wt, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.float32, mesh_mapper=ttnn.ReplicateTensorToMesh(device)
    )
    torch.manual_seed(0)
    x = torch.randn(2, T, C) * 0.3
    xd = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    xr = ttnn.reshape(xd, (2, T, 1, C))

    def run():
        out, _, _ = ttnn.conv1d(
            input_tensor=xr,
            weight_tensor=weight,
            device=device,
            in_channels=C,
            out_channels=C,
            batch_size=2,
            input_length=T,
            kernel_size=K,
            stride=1,
            padding=0,
            dilation=1,
            groups=C,
            dtype=ttnn.float32,
            conv_config=conv_config,
            compute_config=compute_config,
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        return out

    return run


def main():
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=65536)
    try:
        print(f"{'case':<10} {'conv':>8} {'fused':>8} {'separate':>9} {'saving':>8} {'x127 ms':>9}")
        print("-" * 60)
        total = 0.0
        for label, C, T in CASES:
            try:
                plain = make_conv(device, C, T, None)
                fused = make_conv(device, C, T, "relu")
            except Exception as exc:  # noqa: BLE001
                print(f"{label:<10}  SETUP FAILED {str(exc).splitlines()[0][:40]}")
                continue

            def separate():
                y = plain()
                # the band applies its activation in TILE layout, so include the round trip it pays
                yt = ttnn.to_layout(y, ttnn.TILE_LAYOUT)
                z = ttnn.relu(yt)
                return ttnn.to_layout(z, ttnn.ROW_MAJOR_LAYOUT)

            try:
                t_plain = timed(plain)
                t_fused = timed(fused)
                t_sep = timed(separate)
            except Exception as exc:  # noqa: BLE001
                print(f"{label:<10}  FAILED {str(exc).splitlines()[0][:46]}")
                continue

            saving = t_sep - t_fused
            total += saving
            print(f"{label:<10} {t_plain:>8.3f} {t_fused:>8.3f} {t_sep:>9.3f} {saving:>8.3f} {saving * BANDS:>9.1f}")

        print("-" * 60)
        print(f"mean saving x {BANDS} bands: {total / max(len(CASES), 1) * BANDS:.1f} ms")
        print("Compare against the 169 ms the profile attributes to Ternary (140.3) + Tilize (29.0).")
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
