# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the vision FFN torch golden and weight schema. No device needed."""

import torch
import torch.nn.functional as F

from models.experimental.kimi_k3_moonvit_v2.reference.ffn import TorchVisionFfn
from models.experimental.kimi_k3_moonvit_v2.reference.weights import VisionFfnWeights
from models.experimental.kimi_k3_moonvit_v2.reference.config import KimiK3VisionConfig as Cfg
from models.experimental.kimi_k3_moonvit_v2.tt.ffn import build_vision_ffn


def test_vision_ffn_shard_shape():
    """One SP chip's padded shard goes through with its shape unchanged and activations O(1)."""
    ffn = TorchVisionFfn(VisionFfnWeights.synthetic())
    x = torch.randn(Cfg.INPUT_SHAPE, generator=torch.Generator().manual_seed(0))
    out = ffn(x)
    assert out.shape == Cfg.INPUT_SHAPE
    assert 0.1 < out.std() < 10


def test_vision_ffn_matches_formula():
    """The golden is `fc1(gelu_tanh(fc0(x)))`; the erf GELU differs by ~1e-3 and fails this."""
    weights = VisionFfnWeights.synthetic()
    x = torch.randn(1, 1, 64, Cfg.VT_HIDDEN_SIZE, generator=torch.Generator().manual_seed(0))
    expected = F.linear(F.gelu(F.linear(x, weights.fc0), approximate="tanh"), weights.fc1)
    torch.testing.assert_close(TorchVisionFfn(weights)(x), expected)


def test_vision_ffn_ignores_padding_rows():
    """An image's rows come out the same with or without the padding rows after it."""
    ffn = TorchVisionFfn(VisionFfnWeights.synthetic())
    image = torch.randn(1, 1, 100, Cfg.VT_HIDDEN_SIZE, generator=torch.Generator().manual_seed(0))
    padded = torch.cat([image, torch.zeros(1, 1, 28, Cfg.VT_HIDDEN_SIZE)], dim=2)
    torch.testing.assert_close(ffn(padded)[:, :, :100], ffn(image))


def test_vision_ffn_bf16_inputs_compute_in_fp32():
    """bf16 weights and input, as the checkpoint and the device provide them, give the fp32 result."""
    synthetic = VisionFfnWeights.synthetic()
    bf16 = VisionFfnWeights(synthetic.fc0.bfloat16(), synthetic.fc1.bfloat16())
    x = torch.randn(1, 1, 32, Cfg.VT_HIDDEN_SIZE, generator=torch.Generator().manual_seed(0)).bfloat16()
    out = TorchVisionFfn(bf16)(x)
    assert out.dtype == torch.float32
    fp32 = TorchVisionFfn(VisionFfnWeights(bf16.fc0.float(), bf16.fc1.float()))
    torch.testing.assert_close(out, fp32(x.float()))


def test_vision_ffn_weights_from_state_dict():
    synthetic = VisionFfnWeights.synthetic()
    prefix = "vision_tower.encoder.blocks.0.mlp."
    loaded = VisionFfnWeights.from_state_dict(
        {prefix + "fc0.weight": synthetic.fc0, prefix + "fc1.weight": synthetic.fc1}, prefix
    )
    assert loaded.fc0 is synthetic.fc0 and loaded.fc1 is synthetic.fc1


def test_vision_ffn_weights_reject_wrong_shape(expect_error):
    good = VisionFfnWeights.synthetic()
    with expect_error(ValueError, "fc0.weight has shape"):
        VisionFfnWeights(good.fc0.T, good.fc1).validate()


def test_vision_ffn_ttnn_slot_not_implemented(expect_error):
    with expect_error(NotImplementedError, "TtVisionFfn"):
        build_vision_ffn(object(), VisionFfnWeights.synthetic())
