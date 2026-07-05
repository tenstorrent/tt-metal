# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `mel_scale` (coqui/XTTS-v2
`hifigan_decoder.speaker_encoder.torch_spec.1.mel_scale`).

The submodule is a `torchaudio.transforms.MelScale`: it projects a linear-frequency
spectrogram `[B, n_freqs, T]` onto `n_mels` mel bands with the learned/precomputed
mel filterbank `fb` of shape `[n_freqs, n_mels]`:

    mel = matmul(spec.transpose(-1,-2), fb).transpose(-1,-2)
        = fb^T @ spec        (per batch, over the frequency axis)

Native ttnn: a single batched matmul of `fb^T` `[1, n_mels, n_freqs]` with the
spectrogram `[B, n_freqs, T]` -> `[B, n_mels, T]`. Computed in float32.
"""

from __future__ import annotations

import ttnn


HF_MODEL_ID = "coqui/XTTS-v2"


def build(device, torch_module):
    """Bind the mel filterbank and return a native ttnn forward closure."""
    m = torch_module
    fb = m.fb.detach()                       # [n_freqs, n_mels]
    n_freqs, n_mels = fb.shape
    fb_t = ttnn.from_torch(
        fb.t().reshape(1, n_mels, n_freqs).contiguous().float(),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    def forward(x, *args, **kwargs):
        if not isinstance(x, ttnn.Tensor):
            x = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        if x.get_dtype() != ttnn.float32:
            x = ttnn.typecast(x, ttnn.float32)
        # x: [B, n_freqs, T] -> [B, n_mels, T]
        return ttnn.matmul(fb_t, x, compute_kernel_config=compute_config)

    return forward


def mel_scale(*args, **kwargs):
    raise RuntimeError(
        "mel_scale requires build(device, torch_module) to bind the mel filterbank; "
        "the bare callable has no parameters."
    )
