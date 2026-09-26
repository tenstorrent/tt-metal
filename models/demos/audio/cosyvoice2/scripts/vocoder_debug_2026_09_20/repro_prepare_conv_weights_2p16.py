"""Standalone repro (pure ttnn + torch, no CosyVoice imports): `ttnn.prepare_conv_weights` silently
produces a wrong conv1d result at large input widths on Wormhole B0.

The conv is the framing step of an STFT: in_channels=1, out_channels=16, kernel=16, stride=4, a diagonal
(window-scaled) weight, so out[o, t] = w[o] * x[4t + o]. For every input width tried below the *raw*
weight (the op prepares it internally) matches torch; the weight hoisted with `ttnn.prepare_conv_weights`
matches up to some width and is then wrong -- output ~30x too small and uncorrelated with the reference --
with no error or warning. Found in CosyVoice2's TtStft (see tt/hifigan/stft.py); same defect class as
tenstorrent/tt-metal#55545.

    python repro_prepare_conv_weights_2p16.py

Prints, per input width, the relative error of each variant against torch.nn.functional.conv1d (float64).
"""

import torch

import ttnn

N_FFT, HOP = 16, 4
WIDTHS = [4096, 32768, 64016, 65520, 65535, 65536, 65537, 65552, 100000, 222736]


def rel_err(got, want):
    got, want = got.double().reshape(-1), want.double().reshape(-1)
    return float((got - want).norm() / want.norm())


def main():
    device = ttnn.open_device(device_id=0, l1_small_size=65536)
    try:
        window = torch.hann_window(N_FFT, periodic=True)
        w3d = torch.zeros(N_FFT, 1, N_FFT)
        w3d[torch.arange(N_FFT), 0, torch.arange(N_FFT)] = window
        w4d = ttnn.from_torch(w3d.unsqueeze(2), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT)  # OIHW, H=1
        conv_config = ttnn.Conv1dConfig(weights_dtype=ttnn.float32, deallocate_activation=False)

        print(f"{'input width':>12s} | {'raw weight rel err':>19s} | {'prepared weight rel err':>24s}")
        for n in WIDTHS:
            g = torch.Generator().manual_seed(n)
            x = torch.randn(1, n, 1, generator=g) * 0.05  # [N, L, C]
            want = torch.nn.functional.conv1d(x.transpose(1, 2).double(), w3d.double(), stride=HOP).transpose(1, 2)

            x_dev = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
            common = dict(
                in_channels=1,
                out_channels=N_FFT,
                batch_size=1,
                kernel_size=N_FFT,
                stride=HOP,
                padding=0,
                dilation=1,
                groups=1,
                conv_config=conv_config,
                dtype=ttnn.float32,
                return_output_dim=True,
            )
            prepared = ttnn.prepare_conv_weights(
                weight_tensor=w4d,
                weights_format="OIHW",
                has_bias=False,
                input_memory_config=x_dev.memory_config(),
                input_layout=x_dev.layout,
                in_channels=1,
                out_channels=N_FFT,
                batch_size=1,
                input_height=1,
                input_width=n,
                kernel_size=(1, N_FFT),
                stride=(1, HOP),
                padding=(0, 0),
                dilation=(1, 1),
                groups=1,
                device=device,
                input_dtype=ttnn.float32,
                conv_config=conv_config,
            )
            errs = []
            for weight in (w4d, prepared):
                out, out_len = ttnn.conv1d(
                    input_tensor=x_dev, weight_tensor=weight, device=device, input_length=n, **common
                )
                got = ttnn.to_torch(out).float().reshape(1, out_len, N_FFT)
                errs.append(rel_err(got, want[:, :out_len]))
                ttnn.deallocate(out)
            flag = "   <-- WRONG" if errs[1] > 0.05 else ""
            print(f"{n:12d} | {errs[0]:19.4g} | {errs[1]:24.4g}{flag}")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
