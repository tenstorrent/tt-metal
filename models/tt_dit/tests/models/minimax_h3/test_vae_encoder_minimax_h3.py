# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Gate for the MiniMax-H3 visual VAE keyframe encoder.

The claim this rests on is that a single frame collapses H3's causal 3D encoder to
a 2D one, because the causal front-pad is zeros and so only ``weight[:, :, -1]``
survives. The gate is therefore split in two, and the chain gives device == 3D
reference:

- host: the true 3D reference equals the 2D form (``test_causal_conv3d_*``);
- device: the tt encoder equals the 2D form.

Keeping them apart matters -- if the collapse ever stops holding (a checkpoint with
non-zero temporal padding, say) the host half fails on its own and says why, rather
than surfacing as a vague PCC drop in the encoder.
"""

import pytest
import torch
import torch.nn.functional as F

import ttnn

from ....models.vae.minimax_h3.vae_minimax_h3 import (
    MINIMAX_H3_VAE_NORM_EPS,
    MINIMAX_H3_VAE_NUM_GROUPS,
    MiniMaxH3VaeEncoder,
    MiniMaxH3VaeResnetBlock,
)
from ....utils.check import assert_quality

# A reduced encoder: the level/channel pattern is what matters, and the real
# 128..1024 stack at 544x960 is validated separately against the checkpoint.
CH = 32
CH_MULT = (1, 2, 2)
SPACE_DOWN = (2, 2, 1)
TIME_DOWN = (1, 2, 1)
NUM_RES_BLOCKS = 2
Z_CHANNELS = 8
HEIGHT, WIDTH = 32, 48


def _ref_causal_conv3d(x5, weight5, bias, *, stride=(1, 1, 1), padding=(1, 1, 1), pad_mode="reflect"):
    """The reference ``BaseConv3d`` at T=1: reflect on H/W, causal zeros on T."""
    if padding[1] or padding[2]:
        x5 = F.pad(x5, (padding[2], padding[2], padding[1], padding[1], 0, 0), mode=pad_mode)
    if padding[0]:
        # T == 1 takes the reference's single-frame branch: front-pad by k_t - 1.
        x5 = F.pad(x5, (0, 0, 0, 0, weight5.shape[2] - 1, 0), mode="constant")
    return F.conv3d(x5, weight5, bias, stride=stride, padding=0)


def _ref_conv2d(x4, weight5, bias, *, stride=1, pad=1, pad_mode="reflect"):
    """The 2D form: the kernel's last temporal tap."""
    if pad:
        x4 = F.pad(x4, (pad, pad, pad, pad), mode=pad_mode)
    return F.conv2d(x4, weight5[:, :, -1], bias, stride=stride)


@pytest.mark.parametrize(
    ("kernel_t", "stride", "padding"),
    [
        (3, (1, 1, 1), (1, 1, 1)),
        (3, (1, 2, 2), (1, 0, 0)),
        (3, (2, 2, 2), (1, 0, 0)),
        (1, (1, 1, 1), (0, 0, 0)),
    ],
)
def test_causal_conv3d_collapses_to_conv2d_at_one_frame(kernel_t, stride, padding):
    """Host-only: the collapse claim, per conv shape the encoder uses."""
    torch.manual_seed(0)
    weight = torch.randn(16, 8, kernel_t, 3, 3) if kernel_t == 3 else torch.randn(16, 8, 1, 1, 1)
    bias = torch.randn(16)
    x5 = torch.randn(1, 8, 1, 16, 20)

    reference = _ref_causal_conv3d(x5, weight, bias, stride=stride, padding=padding)
    pad = padding[1]
    collapsed = _ref_conv2d(x5[:, :, 0], weight, bias, stride=stride[1:], pad=pad)[:, :, None]

    assert reference.shape == collapsed.shape
    relative = ((reference - collapsed).norm() / reference.norm()).item()
    assert relative < 1e-5, f"collapse broke: rel err {relative:.3e}"


def test_collapse_does_not_compound():
    """Twelve chained convs -- the encoder's resnet depth -- must not drift."""
    torch.manual_seed(1)
    weights = [(torch.randn(16, 16, 3, 3, 3), torch.randn(16)) for _ in range(12)]
    x5 = torch.randn(1, 16, 1, 24, 24)

    three_d = two_d = x5
    for weight, bias in weights:
        three_d = _ref_causal_conv3d(three_d, weight, bias)
        two_d = _ref_conv2d(two_d[:, :, 0], weight, bias)[:, :, None]

    relative = ((three_d - two_d).norm() / three_d.norm()).item()
    assert relative < 1e-5, f"error compounded to {relative:.3e} over 12 convs"


def _torch_reference_encoder(state, x4, *, ch_mult=CH_MULT, num_res_blocks=NUM_RES_BLOCKS, space_down=SPACE_DOWN):
    """The 2D form of the whole encoder, straight from the checkpoint state."""

    def conv(name, h, stride=1, pad=1, pad_mode="reflect"):
        return _ref_conv2d(h, state[f"{name}.weight"], state[f"{name}.bias"], stride=stride, pad=pad, pad_mode=pad_mode)

    def group_norm(name, h):
        return F.group_norm(
            h, MINIMAX_H3_VAE_NUM_GROUPS, state[f"{name}.weight"], state[f"{name}.bias"], eps=MINIMAX_H3_VAE_NORM_EPS
        )

    h = conv("encoder.conv_in", x4)
    for level in range(len(ch_mult)):
        for index in range(num_res_blocks):
            prefix = f"encoder.down.{level}.block.{index}"
            residual = h
            h = conv(f"{prefix}.conv1", F.silu(group_norm(f"{prefix}.norm1", h)))
            h = conv(f"{prefix}.conv2", F.silu(group_norm(f"{prefix}.norm2", h)))
            if f"{prefix}.nin_shortcut.weight" in state:
                residual = conv(f"{prefix}.nin_shortcut", residual, pad=0)
            h = h + residual
        if f"encoder.down.{level}.downsample.conv.weight" in state:
            h = F.pad(h, (0, 1, 0, 1), mode="reflect")
            h = conv(f"encoder.down.{level}.downsample.conv", h, stride=2, pad=0)
    h = conv("encoder.conv_out", F.silu(group_norm("encoder.norm_out", h)))
    return conv("quant_conv", h, pad=0)


def _synthetic_encoder_state(seed=0):
    """Checkpoint-shaped fp32 weights for the reduced encoder."""
    generator = torch.Generator().manual_seed(seed)

    def randn(*shape):
        return torch.randn(shape, generator=generator, dtype=torch.float32) * 0.05

    block_mid = [CH * mult for mult in CH_MULT]
    block_in = [block_mid[0], *block_mid[:-1]]
    state = {
        "encoder.conv_in.weight": randn(block_in[0], 3, 3, 3, 3),
        "encoder.conv_in.bias": randn(block_in[0]),
        "encoder.norm_out.weight": randn(block_mid[-1]),
        "encoder.norm_out.bias": randn(block_mid[-1]),
        "encoder.conv_out.weight": randn(2 * Z_CHANNELS, block_mid[-1], 3, 3, 3),
        "encoder.conv_out.bias": randn(2 * Z_CHANNELS),
        "quant_conv.weight": randn(2 * Z_CHANNELS, 2 * Z_CHANNELS, 1, 1, 1),
        "quant_conv.bias": randn(2 * Z_CHANNELS),
    }
    for level, mult in enumerate(CH_MULT):
        for index in range(NUM_RES_BLOCKS):
            inp = block_in[level] if index == 0 else block_mid[level]
            out = block_mid[level]
            prefix = f"encoder.down.{level}.block.{index}"
            state[f"{prefix}.norm1.weight"] = randn(inp)
            state[f"{prefix}.norm1.bias"] = randn(inp)
            state[f"{prefix}.norm2.weight"] = randn(out)
            state[f"{prefix}.norm2.bias"] = randn(out)
            state[f"{prefix}.conv1.weight"] = randn(out, inp, 3, 3, 3)
            state[f"{prefix}.conv1.bias"] = randn(out)
            state[f"{prefix}.conv2.weight"] = randn(out, out, 3, 3, 3)
            state[f"{prefix}.conv2.bias"] = randn(out)
            if inp != out:
                state[f"{prefix}.nin_shortcut.weight"] = randn(out, inp, 1, 1, 1)
                state[f"{prefix}.nin_shortcut.bias"] = randn(out)
        if SPACE_DOWN[level] * TIME_DOWN[level] > 1:
            channels = block_mid[level]
            state[f"encoder.down.{level}.downsample.conv.weight"] = randn(channels, channels, 3, 3, 3)
            state[f"encoder.down.{level}.downsample.conv.bias"] = randn(channels)
    return state


def _to_device(x4, mesh_device):
    return ttnn.from_torch(
        x4.permute(0, 2, 3, 1).contiguous(),
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _from_device(tensor, mesh_device):
    out = ttnn.to_torch(tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    return out[:1].permute(0, 3, 1, 2)


@pytest.mark.parametrize(
    ("mesh_device", "device_params"),
    # No fabric: a single device has no ring partner, and requesting one makes
    # the ethernet handshake time out before any kernel runs.
    [pytest.param((1, 1), {}, id="single_device")],
    indirect=["mesh_device", "device_params"],
)
def test_resnet_block(mesh_device):
    """One ResnetBlock, including the k1 shortcut path."""
    torch.manual_seed(2)
    in_channels, out_channels = CH, 2 * CH
    state = {
        "norm1.weight": torch.randn(in_channels),
        "norm1.bias": torch.randn(in_channels),
        "norm2.weight": torch.randn(out_channels),
        "norm2.bias": torch.randn(out_channels),
        "conv1.weight": torch.randn(out_channels, in_channels, 3, 3, 3) * 0.05,
        "conv1.bias": torch.randn(out_channels) * 0.05,
        "conv2.weight": torch.randn(out_channels, out_channels, 3, 3, 3) * 0.05,
        "conv2.bias": torch.randn(out_channels) * 0.05,
        "nin_shortcut.weight": torch.randn(out_channels, in_channels, 1, 1, 1) * 0.05,
        "nin_shortcut.bias": torch.randn(out_channels) * 0.05,
    }
    x4 = torch.randn(1, in_channels, HEIGHT, WIDTH)

    def group_norm(name, h, channels):
        return F.group_norm(
            h, MINIMAX_H3_VAE_NUM_GROUPS, state[f"{name}.weight"], state[f"{name}.bias"], eps=MINIMAX_H3_VAE_NORM_EPS
        )

    expected = _ref_conv2d(F.silu(group_norm("norm1", x4, in_channels)), state["conv1.weight"], state["conv1.bias"])
    expected = _ref_conv2d(
        F.silu(group_norm("norm2", expected, out_channels)), state["conv2.weight"], state["conv2.bias"]
    )
    expected = expected + _ref_conv2d(x4, state["nin_shortcut.weight"], state["nin_shortcut.bias"], pad=0)

    block = MiniMaxH3VaeResnetBlock(in_channels, out_channels, mesh_device=mesh_device)
    block.load_torch_state_dict(state, strict=True)
    actual = _from_device(block(_to_device(x4, mesh_device)), mesh_device)

    assert_quality(expected, actual, pcc=0.999)


@pytest.mark.parametrize(
    ("mesh_device", "device_params"),
    # No fabric: a single device has no ring partner, and requesting one makes
    # the ethernet handshake time out before any kernel runs.
    [pytest.param((1, 1), {}, id="single_device")],
    indirect=["mesh_device", "device_params"],
)
def test_encoder_moments(mesh_device):
    """The whole reduced encoder: pixels to 2 * z_channels moments."""
    state = _synthetic_encoder_state()
    x4 = torch.randn(1, 3, HEIGHT, WIDTH, generator=torch.Generator().manual_seed(3))
    expected = _torch_reference_encoder(state, x4)

    encoder = MiniMaxH3VaeEncoder(
        ch=CH,
        ch_mult=CH_MULT,
        num_res_blocks=NUM_RES_BLOCKS,
        space_down=SPACE_DOWN,
        time_down=TIME_DOWN,
        z_channels=Z_CHANNELS,
        mesh_device=mesh_device,
    )
    encoder.load_torch_state_dict(state, strict=True)
    actual = _from_device(encoder(_to_device(x4, mesh_device)), mesh_device)

    # 16x spatial reduction is 2 * 2 over the two downsampling levels here.
    assert actual.shape == expected.shape
    assert_quality(expected, actual, pcc=0.999)
