# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""`HiFTGenerator.inference`: mel -> waveform, composing the F0 predictor
(`tt/hifigan/f0_predictor.py`), the excitation branch (`SineGen2`/
`SourceModuleHnNSF`, Phase 3), and `TtHiFTDecoder.decode` (Phase 2). See
`tt/hifigan/generator.py`'s module docstring (the section right above
`TorchHiFTGeneratorInferenceRef`) for the verified real-source control flow.
`f0` is computed internally from `mel` by the real `ConvRNNF0Predictor` now --
no external f0 stand-in, matching real upstream `HiFTGenerator.inference`'s
own signature (`speech_feat` only).
"""

from __future__ import annotations

import pytest
import torch

from models.common.utility_functions import comp_pcc

GATE_BF16 = 0.99


# --------------------------------------------------------------------------
# host tier -- no device
# --------------------------------------------------------------------------
def test_upsample_scale_matches_shape_trace_convention():
    """`f0_upsamp`'s scale factor (`prod(upsample_rates) * hop_len`) is the same
    480 `test_sine_gen2.py`/`test_hift_decode.py` already use for CosyVoice2's
    3-stage [8, 5, 3] config -- one shared constant, not independently derived
    per file."""
    from models.demos.audio.cosyvoice2.tt.hifigan.f0_predictor import TorchConvRNNF0PredictorRef
    from models.demos.audio.cosyvoice2.tt.hifigan.generator import TorchHiFTDecodeRef, TorchHiFTGeneratorInferenceRef

    decode_ref = TorchHiFTDecodeRef(seed=0)
    f0_ref = TorchConvRNNF0PredictorRef(seed=0)
    ref = TorchHiFTGeneratorInferenceRef(decode_ref, f0_ref, torch.randn(1, 9) * 0.1, torch.randn(1) * 0.1)
    assert ref.upsample_scale == 480


def test_torch_ref_waveform_length_matches_mel_frames_times_upsample_scale():
    from models.demos.audio.cosyvoice2.tt.hifigan.f0_predictor import TorchConvRNNF0PredictorRef
    from models.demos.audio.cosyvoice2.tt.hifigan.generator import TorchHiFTDecodeRef, TorchHiFTGeneratorInferenceRef

    torch.manual_seed(0)
    decode_ref = TorchHiFTDecodeRef(seed=0)
    f0_ref = TorchConvRNNF0PredictorRef(seed=0)
    ref = TorchHiFTGeneratorInferenceRef(decode_ref, f0_ref, torch.randn(1, 9) * 0.1, torch.randn(1) * 0.1)

    mel_frames = 6
    mel = torch.randn(1, mel_frames, 80) * 0.5
    with torch.no_grad():
        out = ref.inference(mel)
    assert out.shape == (1, mel_frames * 480)


# --------------------------------------------------------------------------
# device tier -- needs silicon
# --------------------------------------------------------------------------
needs_l1_small = pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)


@needs_l1_small
@pytest.mark.parametrize("mel_frames", [8, 20])
def test_device_hift_generator_inference_matches_torch_reference(device, mel_frames):
    """`TtHiFTGenerator.inference` (mel -> waveform, composing
    `TtConvRNNF0Predictor`, `TtSourceModuleHnNSF`, and `TtHiFTDecoder.decode`)
    vs. the torch reference. Random-init throughout (no CosyVoice2 checkpoint
    yet, matching every other component in this package)."""
    import ttnn
    from models.demos.audio.cosyvoice2.tt.hifigan.f0_predictor import TorchConvRNNF0PredictorRef
    from models.demos.audio.cosyvoice2.tt.hifigan.generator import (
        TorchHiFTDecodeRef,
        TorchHiFTGeneratorInferenceRef,
        TtHiFTDecoder,
        TtHiFTGenerator,
    )

    torch.manual_seed(mel_frames)
    decode_ref = TorchHiFTDecodeRef(seed=mel_frames)
    f0_ref = TorchConvRNNF0PredictorRef(seed=mel_frames)
    source_w = torch.randn(1, 9) * 0.1
    source_b = torch.randn(1) * 0.1
    ref = TorchHiFTGeneratorInferenceRef(decode_ref, f0_ref, source_w, source_b)

    mel = torch.randn(1, mel_frames, 80) * 0.5
    # SineGen2's harmonic excitation noise is a real per-call random draw in
    # upstream inference (see TtHiFTGenerator.inference's docstring) -- shared
    # here, not left to each side's own default, so this stays a comparison
    # of the computation, not of two independent random draws.
    audio_len = mel_frames * ref.upsample_scale
    sine_noise = torch.randn(1, audio_len, ref.harmonic_num + 1)
    with torch.no_grad():
        want = ref.inference(mel, sine_noise=sine_noise)

    dec = TtHiFTDecoder(device, decode_ref, dtype=ttnn.bfloat16)
    tt_gen = TtHiFTGenerator(device, ref, dec)
    mel_dev = ttnn.from_torch(mel, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    got_dev = tt_gen.inference(mel_dev, mel_frames, 1, sine_noise=sine_noise)
    got = ttnn.to_torch(got_dev).reshape(1, -1).float()

    assert got.shape == want.shape
    passed, pcc = comp_pcc(want, got, GATE_BF16)
    print(f"\n  device HiFTGenerator.inference (T_mel={mel_frames}) PCC {pcc}")
    assert passed, pcc
