# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Audio (WAV) output writer — for T2S task templates."""

from __future__ import annotations


def emit_helper_snippet(sample_rate: int = 16000) -> str:
    """Source for a write_wav helper.

    Templates that produce audio output (T2S) inline this in their demo.
    """
    return f'''
def write_wav(waveform, output_path, sample_rate: int = {sample_rate}):
    """Write a 1-D or 2-D float waveform as 16-bit PCM WAV."""
    import numpy as np
    from scipy.io import wavfile
    from pathlib import Path
    import torch
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.detach().cpu().numpy()
    arr = np.asarray(waveform).reshape(-1).astype(np.float32)
    arr = np.clip(arr, -1.0, 1.0)
    pcm = (arr * 32767.0).astype("int16")
    wavfile.write(str(output_path), sample_rate, pcm)
'''
