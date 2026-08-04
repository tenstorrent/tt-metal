# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Gate M8d.0: the MiniMax-H3 audio checkpoint conversion. Host only, no device.

This is the cheapest high-value gate in the audio port, because it kills the two bugs
most likely to produce a decoder that is *well-formed but subtly wrong*:

1. **The ConvTranspose1d weight-norm axis.** ``torch.nn.utils.weight_norm`` defaults to
   ``dim=0``, and for ``ConvTranspose1d`` axis 0 is ``in_channels``, not ``out_channels``,
   because the weight is stored ``(in, out, k)``. Fusing over the wrong axis still yields
   correctly-shaped weights, so nothing downstream complains -- the audio just sounds
   wrong, and it gets misattributed to precision.
2. **The ``activations`` interleave.** H3 stores six activations per AMP block flat;
   ``AMPBlock1`` wants two lists of three. Swapping them is equally invisible.

Correctness is measured against torch itself: build the reference module, call
``remove_weight_norm``, and compare. That makes the reference the oracle rather than a
second copy of the same arithmetic.
"""

import json
import os
import struct

import pytest
import torch

from ....models.audio_vae.minimax_h3.convert_minimax_h3_audio import (
    assert_weight_norm_axes_consistent,
    convert_minimax_h3_audio_state_dict,
    fuse_attention_biases,
    fuse_weight_norm,
    remap_amp_activations,
)


def _weights_dir() -> str | None:
    base = os.environ.get("MINIMAX_H3_DIFFUSERS_DIR", "/data/cglagovich/MiniMax-H3-diffusers")
    candidate = os.path.join(base, "audio_vae")
    return candidate if os.path.isfile(os.path.join(candidate, "config.json")) else None


def _checkpoint_header(path: str) -> dict:
    """safetensors header only -- shapes without reading 605 MB of tensor data."""
    with open(path, "rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        return {k: v for k, v in json.loads(handle.read(length)).items() if k != "__metadata__"}


def test_fuse_weight_norm_matches_torch_conv1d():
    """``fuse_weight_norm`` == what torch's own ``remove_weight_norm`` leaves behind."""
    torch.manual_seed(0)
    conv = torch.nn.Conv1d(16, 32, kernel_size=7)
    conv = torch.nn.utils.weight_norm(conv)
    with torch.no_grad():
        conv.weight_g.normal_(1.0, 0.2)
        conv.weight_v.normal_()

    fused = fuse_weight_norm(conv.weight_g.detach(), conv.weight_v.detach())
    torch.nn.utils.remove_weight_norm(conv)

    assert fused.shape == conv.weight.shape
    relative = (fused - conv.weight).abs().max().item() / conv.weight.abs().max().item()
    assert relative < 1e-6, f"Conv1d fusion differs from torch by {relative:.3e}"


def test_fuse_weight_norm_matches_torch_conv_transpose1d():
    """The load-bearing case: axis 0 of a ConvTranspose1d weight is ``in_channels``."""
    torch.manual_seed(1)
    conv = torch.nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2)
    assert conv.weight.shape == (32, 16, 4), "ConvTranspose1d weight is (in, out, k)"
    conv = torch.nn.utils.weight_norm(conv)
    with torch.no_grad():
        conv.weight_g.normal_(1.0, 0.2)
        conv.weight_v.normal_()
    assert conv.weight_g.shape == (32, 1, 1), "weight_g is per-in_channel, confirming dim=0"

    weight_g = conv.weight_g.detach().clone()
    weight_v = conv.weight_v.detach().clone()
    fused = fuse_weight_norm(weight_g, weight_v)
    torch.nn.utils.remove_weight_norm(conv)

    relative = (fused - conv.weight).abs().max().item() / conv.weight.abs().max().item()
    assert relative < 1e-6, f"ConvTranspose1d fusion differs from torch by {relative:.3e}"

    # Show that reducing over the wrong axis is silently type-correct: same shape, wrong
    # values. That is precisely why the axis needs a test rather than a comment.
    wrong_norm = weight_v.transpose(0, 1).flatten(1).norm(dim=1).view(1, -1, 1)
    wrong = weight_g * weight_v / wrong_norm
    assert wrong.shape == fused.shape, "the wrong axis still type-checks -- hence this test"
    assert not torch.allclose(wrong, fused, atol=1e-4), "the two axes agree, so this test proves nothing"


def test_remap_amp_activations_interleaves_correctly():
    """``activations.{0,2,4}`` -> ``acts1.{0,1,2}`` and ``{1,3,5}`` -> ``acts2.{0,1,2}``."""
    state = {f"resblocks.0.activations.{i}.act.alpha": torch.tensor([float(i)]) for i in range(6)}
    remapped = remap_amp_activations(state)

    for i in range(3):
        assert remapped[f"resblocks.0.acts1.{i}.act.alpha"].item() == 2 * i
        assert remapped[f"resblocks.0.acts2.{i}.act.alpha"].item() == 2 * i + 1
    assert not any("activations." in key for key in remapped), "an activations key survived"


def test_fuse_attention_biases_rejects_a_nonzero_k_bias():
    """A ``zero_k_bias`` that is not zero must fail loudly, not be dropped."""
    state = {
        "pre_block.attn.q_bias": torch.ones(8),
        "pre_block.attn.v_bias": torch.full((8,), 2.0),
        "pre_block.attn.zero_k_bias": torch.zeros(8),
    }
    fused = fuse_attention_biases(state)
    bias = fused["pre_block.attn.qkv.bias"]
    assert bias.shape == (24,)
    assert torch.equal(bias[:8], torch.ones(8))
    assert torch.equal(bias[8:16], torch.zeros(8))
    assert torch.equal(bias[16:], torch.full((8,), 2.0))

    state["pre_block.attn.zero_k_bias"] = torch.ones(8)
    with pytest.raises(AssertionError, match="not all zero"):  # allow-pytest.raises: guards a silent data-loss path
        fuse_attention_biases(state)


def test_real_checkpoint_axes_and_conversion():
    """The real 1087-tensor checkpoint: axis assumptions hold and every pair fuses."""
    weights_dir = _weights_dir()
    if weights_dir is None:
        pytest.skip("MiniMax-H3 audio_vae not found; set MINIMAX_H3_DIFFUSERS_DIR")
    from safetensors.torch import load_file

    state = load_file(os.path.join(weights_dir, "diffusion_pytorch_model.safetensors"))
    assert_weight_norm_axes_consistent(state)

    num_pairs = len([k for k in state if k.endswith(".weight_g")])
    assert num_pairs > 100, f"expected ~172 weight-normed convs, found {num_pairs}"

    converted = convert_minimax_h3_audio_state_dict(state)
    assert not [k for k in converted if k.endswith((".weight_g", ".weight_v"))], "a weight-norm pair survived"
    assert not [k for k in converted if k.endswith(("q_bias", "v_bias", "zero_k_bias"))], "an attn bias survived"
    assert not [k for k in converted if "activations." in k], "an activations key survived"
    # Every fused conv should have produced exactly one weight, and nothing else was lost.
    assert (
        len(converted) == len(state) - num_pairs - 2
    ), f"key count {len(converted)} does not match {len(state)} minus {num_pairs} g/v pairs and 2 folded biases"


def test_real_checkpoint_fusion_matches_reference_module():
    """Fused weights equal what the reference's own ``remove_weight_norm`` produces."""
    weights_dir = _weights_dir()
    if weights_dir is None:
        pytest.skip("MiniMax-H3 audio_vae not found; set MINIMAX_H3_DIFFUSERS_DIR")
    pytest.importorskip("diffusers", reason="pinned diffusers reference not installed")
    from diffusers import AutoencoderKLMiniMaxH3Audio
    from safetensors.torch import load_file

    config = {
        k: v
        for k, v in json.loads(open(os.path.join(weights_dir, "config.json")).read()).items()
        if not k.startswith("_")
    }
    reference = AutoencoderKLMiniMaxH3Audio(**config)
    state = load_file(os.path.join(weights_dir, "diffusion_pytorch_model.safetensors"))
    reference.load_state_dict(state)

    for module in reference.modules():
        if hasattr(module, "weight_g"):
            torch.nn.utils.remove_weight_norm(module)
    expected = dict(reference.state_dict())

    converted = convert_minimax_h3_audio_state_dict(state)

    checked = 0
    worst_key, worst = None, 0.0
    for key, value in expected.items():
        if not key.endswith(".weight") or key not in converted:
            continue
        scale = max(value.abs().max().item(), 1e-12)
        relative = (converted[key] - value).abs().max().item() / scale
        if relative > worst:
            worst_key, worst = key, relative
        checked += 1
    assert checked > 100, f"only compared {checked} fused weights"
    assert worst < 1e-6, f"worst fused weight is {worst_key} at relative {worst:.3e}"
