# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""The flow decoder's mel output wired directly into the HiFT vocoder: speech
tokens -> `TtCausalMaskedDiffWithXvec` (Phase 5) -> mel -> `TtHiFTGenerator`
(Phase 2 + Phase 3 composition) -> waveform.

The interface contract between the two, verified against real upstream source
(not assumed from "both modules are already validated"): `HiFTGenerator.inference`
feeds its `speech_feat` argument into `decode()` with NO transform anywhere in the
real source -- no denormalisation, no scaling, no clamping (see
`tt/hifigan/generator.py`'s module docstring, the section above
`TorchHiFTGeneratorInferenceRef`). In this port specifically, that connection also
needs no transpose: `tt/flow/`'s mel output and `tt/hifigan/`'s mel input are both
this package's channels-last `[N, L, C]` convention -- confirmed by this file's
tests actually chaining the two with no reshape/transpose in between, not by
asserting it in prose.

`f0_predictor` (mel -> f0, a separate small CNN+GRU network) is not built in this
environment -- the same deliberate scope boundary `tt/hifigan/generator.py`'s
module docstring already draws for `TtHiFTGenerator.inference` on its own; `f0` is
a synthetic, externally-supplied stand-in here too, same as
test_hift_generator_inference.py.
"""

from __future__ import annotations

import pytest
import torch

from models.common.utility_functions import comp_pcc

GATE_BF16 = 0.99


def _synthetic_f0(mel_frames: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    voiced = torch.rand(1, mel_frames, generator=g) > 0.3
    return torch.where(voiced, 80.0 + torch.rand(1, mel_frames, generator=g) * 320.0, torch.zeros(1, mel_frames))


needs_l1_small = pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)


@needs_l1_small
def test_device_speech_tokens_to_waveform_matches_torch_reference(device):
    """The full chain: speech tokens -> flow decoder -> mel -> HiFT vocoder ->
    waveform, TT vs. torch, with the TT and torch pipelines run completely
    independently (the TT mel is computed on device and re-uploaded for the TT
    vocoder call; it is never substituted into the torch pipeline, and vice
    versa) -- so this is a genuine two-independent-pipelines comparison, not two
    stages sharing an intermediate result."""
    import ttnn
    from models.demos.audio.cosyvoice2.tt.flow.flow import CausalMaskedDiffWithXvecRef, TtCausalMaskedDiffWithXvec
    from models.demos.audio.cosyvoice2.tt.hifigan.generator import (
        TorchHiFTDecodeRef,
        TorchHiFTGeneratorInferenceRef,
        TtHiFTDecoder,
        TtHiFTGenerator,
    )

    torch.manual_seed(5)
    flow_ref = CausalMaskedDiffWithXvecRef()
    flow_ref.eval()

    prompt_token_len, new_token_len = 4, 6
    prompt_token = torch.randint(0, 6561, (1, prompt_token_len))
    token = torch.randint(0, 6561, (1, new_token_len))
    prompt_feat = torch.randn(1, prompt_token_len * 2, 80) * 0.1
    embedding = torch.randn(1, 192)

    with torch.no_grad():
        mel_ref = flow_ref.inference(token, prompt_token, prompt_feat, embedding)
    mel_frames = mel_ref.shape[1]
    assert mel_ref.shape == (1, new_token_len * 2, 80)

    decode_ref = TorchHiFTDecodeRef(seed=7)
    source_w = torch.randn(1, 9) * 0.1
    source_b = torch.randn(1) * 0.1
    hift_ref = TorchHiFTGeneratorInferenceRef(decode_ref, source_w, source_b)
    f0 = _synthetic_f0(mel_frames, seed=1)

    with torch.no_grad():
        want = hift_ref.inference(mel_ref, f0)

    tt_flow = TtCausalMaskedDiffWithXvec(device, flow_ref)
    mel_got = tt_flow.inference(token, prompt_token, prompt_feat, embedding)
    assert mel_got.shape == mel_ref.shape

    mel_dev = ttnn.from_torch(mel_got, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    dec = TtHiFTDecoder(device, decode_ref, dtype=ttnn.bfloat16)
    tt_gen = TtHiFTGenerator(device, hift_ref, dec)
    got_dev = tt_gen.inference(mel_dev, f0, mel_frames, 1)
    got = ttnn.to_torch(got_dev).reshape(1, -1).float()

    assert got.shape == want.shape == (1, mel_frames * 480)
    passed, pcc = comp_pcc(want, got, GATE_BF16)
    print(f"\n  speech tokens -> mel -> waveform, end to end, PCC {pcc}")
    assert passed, pcc
