# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""`HiFTGenerator.inference`: mel (+ f0) -> waveform, composing the excitation
branch (`SineGen2`/`SourceModuleHnNSF`, Phase 3) with `TtHiFTDecoder.decode`
(Phase 2). See `tt/hifigan/generator.py`'s module docstring (the section right
above `TorchHiFTGeneratorInferenceRef`) for the verified real-source control flow
and the one still-deferred piece (`f0_predictor`, an explicit external input here).
"""

from __future__ import annotations

import pytest
import torch

from models.common.utility_functions import comp_pcc

GATE_BF16 = 0.99


def _synthetic_f0(mel_frames: int, seed: int = 0) -> torch.Tensor:
    """Mixed voiced/unvoiced f0, matching test_sine_gen2.py's own convention --
    a stand-in for a real f0_predictor's output, one Hz value per mel frame."""
    g = torch.Generator().manual_seed(seed)
    voiced = torch.rand(1, mel_frames, generator=g) > 0.3
    return torch.where(voiced, 80.0 + torch.rand(1, mel_frames, generator=g) * 320.0, torch.zeros(1, mel_frames))


# --------------------------------------------------------------------------
# host tier -- no device
# --------------------------------------------------------------------------
def test_upsample_scale_matches_shape_trace_convention():
    """`f0_upsamp`'s scale factor (`prod(upsample_rates) * hop_len`) is the same
    480 `test_sine_gen2.py`/`test_hift_decode.py` already use for CosyVoice2's
    3-stage [8, 5, 3] config -- one shared constant, not independently derived
    per file."""
    from models.demos.audio.cosyvoice2.tt.hifigan.generator import TorchHiFTDecodeRef, TorchHiFTGeneratorInferenceRef

    decode_ref = TorchHiFTDecodeRef(seed=0)
    ref = TorchHiFTGeneratorInferenceRef(decode_ref, torch.randn(1, 9) * 0.1, torch.randn(1) * 0.1)
    assert ref.upsample_scale == 480


def test_torch_ref_waveform_length_matches_mel_frames_times_upsample_scale():
    from models.demos.audio.cosyvoice2.tt.hifigan.generator import TorchHiFTDecodeRef, TorchHiFTGeneratorInferenceRef

    torch.manual_seed(0)
    decode_ref = TorchHiFTDecodeRef(seed=0)
    ref = TorchHiFTGeneratorInferenceRef(decode_ref, torch.randn(1, 9) * 0.1, torch.randn(1) * 0.1)

    mel_frames = 6
    mel = torch.randn(1, mel_frames, 80) * 0.5
    f0 = _synthetic_f0(mel_frames)
    with torch.no_grad():
        out = ref.inference(mel, f0)
    assert out.shape == (1, mel_frames * 480)


# --------------------------------------------------------------------------
# device tier -- needs silicon
# --------------------------------------------------------------------------
needs_l1_small = pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)


@needs_l1_small
@pytest.mark.parametrize("mel_frames", [8, 20])
def test_device_hift_generator_inference_matches_torch_reference(device, mel_frames):
    """`TtHiFTGenerator.inference` (mel + f0 -> waveform, composing
    `TtSourceModuleHnNSF` with `TtHiFTDecoder.decode`) vs. the torch reference.
    Random-init throughout (no CosyVoice2 checkpoint yet, matching every other
    component in this package)."""
    import ttnn
    from models.demos.audio.cosyvoice2.tt.hifigan.generator import (
        TorchHiFTDecodeRef,
        TorchHiFTGeneratorInferenceRef,
        TtHiFTDecoder,
        TtHiFTGenerator,
    )

    torch.manual_seed(mel_frames)
    decode_ref = TorchHiFTDecodeRef(seed=mel_frames)
    source_w = torch.randn(1, 9) * 0.1
    source_b = torch.randn(1) * 0.1
    ref = TorchHiFTGeneratorInferenceRef(decode_ref, source_w, source_b)

    mel = torch.randn(1, mel_frames, 80) * 0.5
    f0 = _synthetic_f0(mel_frames, seed=mel_frames)
    with torch.no_grad():
        want = ref.inference(mel, f0)

    dec = TtHiFTDecoder(device, decode_ref, dtype=ttnn.bfloat16)
    tt_gen = TtHiFTGenerator(device, ref, dec)
    mel_dev = ttnn.from_torch(mel, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    got_dev = tt_gen.inference(mel_dev, f0, mel_frames, 1)
    got = ttnn.to_torch(got_dev).reshape(1, -1).float()

    assert got.shape == want.shape
    passed, pcc = comp_pcc(want, got, GATE_BF16)
    print(f"\n  device HiFTGenerator.inference (T_mel={mel_frames}) PCC {pcc}")
    assert passed, pcc
