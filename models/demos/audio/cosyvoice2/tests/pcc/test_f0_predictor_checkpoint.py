# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""Real CosyVoice2-0.5B checkpoint weights (`hift.pt`'s `f0_predictor.*` keys,
from `FunAudioLLM/CosyVoice2-0.5B`) loaded into
`TorchConvRNNF0PredictorRef`/`TtConvRNNF0Predictor`, checked with the SAME
TT-vs-torch-with-shared-weights isolation `test_f0_predictor.py`'s existing
device tests already use -- just with the random init those tests use
replaced by real trained weights. See `tt/checkpoint.py` for the download
and `TorchConvRNNF0PredictorRef.from_checkpoint`'s docstring for the exact
key mapping (confirmed 1:1, zero renaming needed).

Second module in this bring-up's checkpoint-loading order (HiFT vocoder,
done -- see test_hift_checkpoint.py -- then F0 predictor, here, then flow
decoder, then LLM backbone).

**Checked bf16 against the real checkpoint before assuming either way** --
`test_hift_checkpoint.py` found the HiFT decode module needs fp32 with real
weights (bf16 PCC collapsed to ~0.49 there, from real `conv_post` values
hitting `exp()` at a magnitude bf16 can't track precisely enough). That
finding does NOT transfer here: measured directly, bf16 PCC is 0.996-0.999
at both mel lengths tested, comfortably clearing the same 0.99 gate every
other bf16 test in this package uses. This module's own final nonlinearity
is `abs()` (see `tt/hifigan/f0_predictor.py`), not `exp()` -- `abs()`
doesn't amplify a small relative error into a large one the way `exp()`
does, which is almost certainly why real weights don't stress bf16 precision
here the way they did for HiFT decode. Real weights DO push this module's
output to a much wider range than random init did (std~65-82 vs random
init's much smaller scale on synthetic mel), so this isn't "nothing
changed" -- bf16 is just adequate for it, verified rather than assumed.
"""

from __future__ import annotations

import pytest
import torch

from models.common.utility_functions import comp_pcc

GATE_BF16 = 0.99

needs_l1_small = pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)


@needs_l1_small
@pytest.mark.parametrize("mel_frames", [8, 20])
def test_device_f0_predictor_matches_torch_reference_real_checkpoint(device, mel_frames):
    """`TtConvRNNF0Predictor` vs `TorchConvRNNF0PredictorRef`, both built from
    the SAME real `hift.pt` `f0_predictor.*` weights (not random init) -- the
    real-weight counterpart of `test_f0_predictor.py`'s
    `test_device_f0_predictor_matches_torch_reference`. Random synthetic mel
    input, same as that test (a real mel input is a later, separate concern
    once the flow decoder's own real weights are loaded) -- this isolates
    "does the real checkpoint's weight distribution expose a device-op
    numerical issue random init didn't", which for THIS module (unlike HiFT
    decode) it does not, at bf16."""
    import ttnn
    from models.demos.audio.cosyvoice2.tt.checkpoint import load_checkpoint_file, sub_state_dict
    from models.demos.audio.cosyvoice2.tt.hifigan.f0_predictor import TorchConvRNNF0PredictorRef, TtConvRNNF0Predictor

    hift_sd = load_checkpoint_file("hift.pt")
    f0_sub = sub_state_dict(hift_sd, "f0_predictor.")
    ref = TorchConvRNNF0PredictorRef.from_checkpoint(f0_sub)

    torch.manual_seed(mel_frames)
    mel_cl = torch.randn(1, mel_frames, 80) * 0.5  # channels-last, this port's convention
    with torch.no_grad():
        want = ref(mel_cl.transpose(1, 2))  # -> channel-first for the real-upstream-shaped reference

    tt_f0 = TtConvRNNF0Predictor(device, ref, dtype=ttnn.bfloat16)
    mel_dev = ttnn.from_torch(mel_cl, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    got_dev = tt_f0(mel_dev, mel_frames, batch_size=1)
    got = ttnn.to_torch(got_dev).reshape(1, -1).float()

    assert got.shape == want.shape
    passed, pcc = comp_pcc(want, got, GATE_BF16)
    print(f"\n  real-checkpoint device ConvRNNF0Predictor (T_mel={mel_frames}) PCC {pcc}")
    assert passed, pcc


def test_condnet_layer_shapes_match_real_checkpoint():
    """Direct shape check against the real checkpoint's own tensors -- confirms
    `TorchConvRNNF0PredictorRef`'s architecture (five 80->512->512->512->512->512
    Conv1d(k=3) layers, `f0_predictor.py`'s module docstring) against real
    trained weights, not just the config file."""
    from models.demos.audio.cosyvoice2.tt.checkpoint import load_checkpoint_file, sub_state_dict

    hift_sd = load_checkpoint_file("hift.pt")
    f0_sub = sub_state_dict(hift_sd, "f0_predictor.")
    assert f0_sub["condnet.0.parametrizations.weight.original1"].shape == (512, 80, 3)
    for i in (2, 4, 6, 8):
        assert f0_sub[f"condnet.{i}.parametrizations.weight.original1"].shape == (512, 512, 3)
    assert f0_sub["classifier.weight"].shape == (1, 512)
    assert f0_sub["classifier.bias"].shape == (1,)
