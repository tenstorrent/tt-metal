"""`TtStft` (as it now ships: raw framing weight) vs float64 `torch.stft` across input lengths, fp32.

Regression sweep for the >=65,536-sample corruption fixed in tt/hifigan/stft.py (the hoisted-weight version
was PCC ~0.27 / gain ~0.03 from 65,536 samples up). Self-contained: builds its own excitation-like input.
The pytest version of the boundary cases is tests/pcc/test_stft.py; this one goes out to 80 s of audio.

    python stft_length_sweep.py
"""

import numpy as np
import torch

import ttnn
from models.demos.audio.cosyvoice2.tt.hifigan.istft import periodic_hann
from models.demos.audio.cosyvoice2.tt.hifigan.stft import TtStft

LENGTHS = [480, 4800, 24000, 55680, 64000, 65536, 100000, 222720, 480000, 960000, 1920000]


def excitation_like(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(n, dtype=torch.float64)
    phase = 2 * torch.pi * torch.cumsum((120.0 + 40.0 * torch.sin(2 * torch.pi * t / 24000.0)) / 24000.0, 0)
    return (
        0.05 * torch.sin(phase) + 0.02 * torch.sin(2 * phase) + 0.003 * torch.randn(n, generator=g, dtype=torch.float64)
    ).float()


def main():
    window = torch.from_numpy(np.asarray(periodic_hann(16))).double()
    device = ttnn.open_device(device_id=0, l1_small_size=65536)
    try:
        op = TtStft(device, 16, 4, window=window.float(), dtype=ttnn.float32)
        print(f"{'samples':>9s} | {'pcc':>9s} {'gain':>7s} {'rel err':>8s}")
        for n in LENGTHS:
            x = excitation_like(n)
            sp = torch.stft(x.double().unsqueeze(0), 16, 4, 16, window=window, return_complex=True)
            want = torch.cat([sp.real, sp.imag], dim=1).reshape(-1)
            x_dev = ttnn.from_torch(x.reshape(1, n, 1), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
            spec, _ = op(x_dev, n, 1)
            got = ttnn.to_torch(spec).float().reshape(-1).double()
            m = min(got.numel(), want.numel())
            got, want = got[:m], want[:m]
            pcc = float(np.corrcoef(got.numpy(), want.numpy())[0, 1])
            print(
                f"{n:9d} | {pcc:9.5f} {float((got @ want) / (want @ want)):7.4f} {float((got - want).norm() / want.norm()):8.4f}"
            )
            ttnn.deallocate(spec)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
