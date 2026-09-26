# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""`ConvRNNF0Predictor`: mel -> f0. See `tt/hifigan/f0_predictor.py`'s module
docstring for the architecture (five plain Conv1d+ELU layers plus a Linear
classifier -- despite the class name, no recurrent layer at all, confirmed
directly against real upstream source) and the channel-convention split
between the torch reference (channel-first, matching real upstream) and the
device module (channels-last, matching every other Tt module in this
package).
"""

from __future__ import annotations

import pytest
import torch

from models.common.utility_functions import comp_pcc

GATE_BF16 = 0.99


# --------------------------------------------------------------------------
# host tier -- no device
# --------------------------------------------------------------------------
def test_torch_ref_has_no_recurrent_layer():
    """Despite the class name `ConvRNNF0Predictor`, real upstream source has no
    RNN/GRU/LSTM anywhere in it -- confirmed directly against
    `cosyvoice/hifigan/f0_predictor.py`, not assumed from the name. Guards
    against a future edit accidentally "fixing" this into matching the name."""
    from models.demos.audio.cosyvoice2.tt.hifigan.f0_predictor import TorchConvRNNF0PredictorRef

    ref = TorchConvRNNF0PredictorRef(seed=0)
    recurrent_types = (torch.nn.RNNBase, torch.nn.RNN, torch.nn.GRU, torch.nn.LSTM)
    for module in ref.modules():
        assert not isinstance(module, recurrent_types), module


def test_torch_ref_layer_shapes_match_cosyvoice2_yaml_defaults():
    """Five Conv1d(k=3, pad=1) layers, 80->512 then 512->512 x4, plus
    Linear(512, 1) -- exactly `cosyvoice2.yaml`'s `hift.f0_predictor` block
    (`num_class: 1, in_channels: 80, cond_channels: 512`), confirmed directly
    against that real config file, not assumed from the class defaults alone."""
    from models.demos.audio.cosyvoice2.tt.hifigan.f0_predictor import NUM_CONV_LAYERS, TorchConvRNNF0PredictorRef

    ref = TorchConvRNNF0PredictorRef(seed=0)
    convs = [m for m in ref.condnet if isinstance(m, torch.nn.Conv1d)]
    assert len(convs) == NUM_CONV_LAYERS == 5
    assert convs[0].in_channels == 80
    for c in convs:
        assert c.out_channels == 512
        assert c.kernel_size == (3,)
        assert c.padding == (1,)
    assert all(c.in_channels == 512 for c in convs[1:])
    assert ref.classifier.in_features == 512
    assert ref.classifier.out_features == 1


def test_torch_ref_output_shape_and_nonnegative():
    """`abs(classifier(...))` at the end of real upstream `forward` makes f0
    non-negative by construction -- checked here as a property of the real
    architecture, not asserted as a testing convenience."""
    from models.demos.audio.cosyvoice2.tt.hifigan.f0_predictor import TorchConvRNNF0PredictorRef

    torch.manual_seed(1)
    ref = TorchConvRNNF0PredictorRef(seed=1)
    mel_cf = torch.randn(1, 80, 17) * 0.5  # channel-first, matching real upstream's own convention
    with torch.no_grad():
        f0 = ref(mel_cf)
    assert f0.shape == (1, 17)
    assert torch.all(f0 >= 0)


def test_torch_ref_preserves_temporal_length():
    """Every conv is `kernel_size=3, padding=1, stride=1` -- "same" padding, so
    T is unchanged end to end. Checked at two different mel lengths."""
    from models.demos.audio.cosyvoice2.tt.hifigan.f0_predictor import TorchConvRNNF0PredictorRef

    ref = TorchConvRNNF0PredictorRef(seed=2)
    for mel_frames in (6, 33):
        with torch.no_grad():
            f0 = ref(torch.randn(1, 80, mel_frames) * 0.3)
        assert f0.shape == (1, mel_frames)


# --------------------------------------------------------------------------
# device tier -- needs silicon
# --------------------------------------------------------------------------
needs_l1_small = pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)


@needs_l1_small
@pytest.mark.parametrize("mel_frames", [8, 20])
def test_device_f0_predictor_matches_torch_reference(device, mel_frames):
    """`TtConvRNNF0Predictor` vs `TorchConvRNNF0PredictorRef`, both built from the
    SAME random weights (`TtConvRNNF0Predictor(device, ref)` folds `ref`'s own
    weight_norm conv weights via `TtConv1d.from_module`), so a PCC gap here is
    the device op disagreeing with real upstream math, not the weights
    differing. mel enters channels-last (this port's convention, matching
    `TtHiFTGenerator.inference`'s own mel argument) and is transposed to
    channel-first only for the torch reference call, matching real upstream's
    own convention -- see module docstring."""
    import ttnn
    from models.demos.audio.cosyvoice2.tt.hifigan.f0_predictor import TorchConvRNNF0PredictorRef, TtConvRNNF0Predictor

    torch.manual_seed(mel_frames)
    ref = TorchConvRNNF0PredictorRef(seed=mel_frames)
    mel_cl = torch.randn(1, mel_frames, 80) * 0.5  # channels-last, this port's convention
    with torch.no_grad():
        want = ref(mel_cl.transpose(1, 2))  # -> channel-first for the real-upstream-shaped reference

    tt_f0 = TtConvRNNF0Predictor(device, ref, dtype=ttnn.bfloat16)
    mel_dev = ttnn.from_torch(mel_cl, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    got_dev = tt_f0(mel_dev, mel_frames, batch_size=1)
    got = ttnn.to_torch(got_dev).reshape(1, -1).float()

    assert got.shape == want.shape
    passed, pcc = comp_pcc(want, got, GATE_BF16)
    print(f"\n  device ConvRNNF0Predictor (T_mel={mel_frames}) PCC {pcc}")
    assert passed, pcc


@needs_l1_small
def test_device_f0_predictor_output_nonnegative(device):
    """`ttnn.abs` at the end must make the device output non-negative too, the
    same real-architecture property `test_torch_ref_output_shape_and_nonnegative`
    checks on the torch side."""
    import ttnn
    from models.demos.audio.cosyvoice2.tt.hifigan.f0_predictor import TorchConvRNNF0PredictorRef, TtConvRNNF0Predictor

    torch.manual_seed(7)
    ref = TorchConvRNNF0PredictorRef(seed=7)
    mel_cl = torch.randn(1, 12, 80) * 0.5

    tt_f0 = TtConvRNNF0Predictor(device, ref, dtype=ttnn.bfloat16)
    mel_dev = ttnn.from_torch(mel_cl, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    got = ttnn.to_torch(tt_f0(mel_dev, 12, batch_size=1)).float()
    assert torch.all(got >= 0)
