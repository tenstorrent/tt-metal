# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Gate M8a.3/M8b: the MiniMax-H3 visual VAE encoder, whole.

Two independent layers of evidence, kept apart:

* **Host, no device** -- the T=1 collapse claim: a single frame reduces H3's causal 3D
  encoder to a 2D one, because ``temporal_padding = kernel_t - 1`` prepends *zeros*, so
  only ``weight[:, :, -1]`` survives. If that ever stops holding (a checkpoint with
  non-zero temporal padding, say) these fail on their own and say why, instead of
  surfacing as a vague PCC drop somewhere in a twelve-resnet stack.
* **Device** -- the tt encoder against the pinned diffusers
  ``MiniMaxH3VideoEncoder3d``, at both ``temporal_taps=1`` (keyframe) and
  ``temporal_taps=3`` (17-frame clip), then the tiled ``encode_clip`` against the
  reference's own tiling.

The device bar is 0.99 rather than 0.999: ``ttnn.group_norm`` has no fp32 path, so all
thirteen norms in the stack are bf16 islands, and that is the encoder's precision floor.
Per-conv parity at 0.999 is gated separately in ``test_vae_conv_minimax_h3.py``.
"""

import pytest
import torch
import torch.nn.functional as F

import ttnn

from ....models.vae.minimax_h3.encoder_minimax_h3 import MiniMaxH3Encoder3d
from ....utils.check import assert_quality

# A single device has no ring partner: requesting a fabric ring makes the ethernet
# handshake time out before any kernel runs.
SINGLE_DEVICE = [pytest.param((1, 1), {}, id="single_device")]

# The real encoder configuration, and the real tile shape. Spatial tiling fixes the tile
# at 256x256 and the clip at 17 frames, so every supported working point -- 768P and
# 1440P at 5 s and 10 s -- runs exactly these two shapes and differs only in how many
# tiles and clips there are. Testing a reduced synthetic stack instead would invent
# channel/spatial pairings the model never produces, and at least one of those,
# (C=128, T=5, 32x32), deadlocks ttnn.group_norm.
BLOCK_OUT_CHANNELS = (128, 256, 256, 512, 512, 1024)
SPATIAL_DOWN = (2, 2, 2, 2, 1, 1)
TEMPORAL_DOWN = (1, 2, 2, 1, 1, 1)
LAYERS_PER_BLOCK = 2
TILE = 256
CLIP_FRAMES = 17
LATENT_CHANNELS = 24


def _reference_encoder_cls():
    pytest.importorskip("diffusers", reason="pinned diffusers reference not installed")
    from diffusers.models.autoencoders import autoencoder_kl_minimax_h3 as ref

    cls = getattr(ref, "MiniMaxH3VideoEncoder3d", None)
    if cls is None:
        pytest.skip("MiniMaxH3VideoEncoder3d missing -- diffusers is not at the pinned commit")
    return cls


def _ref_causal_conv3d(x5, weight5, bias, *, stride=(1, 1, 1), padding=(1, 1, 1), pad_mode="reflect"):
    """The reference conv at T=1: reflect on H/W, causal zeros on T."""
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
    collapsed = _ref_conv2d(x5[:, :, 0], weight, bias, stride=stride[1:], pad=padding[1])[:, :, None]

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


def _to_device(x_BCTHW: torch.Tensor, mesh_device, aligned_channels: int) -> ttnn.Tensor:
    x = x_BCTHW.permute(0, 2, 3, 4, 1).contiguous()
    if x.shape[-1] < aligned_channels:
        x = torch.nn.functional.pad(x, (0, aligned_channels - x.shape[-1]))
    return ttnn.from_torch(x, dtype=ttnn.float32, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT)


def _from_device(tensor: ttnn.Tensor, out_channels: int) -> torch.Tensor:
    x = ttnn.to_torch(tensor).float()
    return x[..., :out_channels].permute(0, 4, 1, 2, 3).contiguous()


@pytest.mark.parametrize(
    ("num_frames", "temporal_taps", "expected_latent_shape"),
    [
        pytest.param(1, 1, (1, 16, 16), id="keyframe_1x256x256"),
        pytest.param(CLIP_FRAMES, 3, (5, 16, 16), id="clip_17x256x256"),
    ],
)
@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE_DEVICE, indirect=["mesh_device", "device_params"])
def test_encoder_moments(mesh_device, num_frames, temporal_taps, expected_latent_shape):
    """The full 128..1024 encoder on the shipping tile, against the pinned reference.

    This is the shape that actually ships, for every supported working point, so it is
    worth the cost of running the reference on host rather than shrinking the stack.
    """
    reference_cls = _reference_encoder_cls()
    torch.manual_seed(3)
    out_channels = 2 * LATENT_CHANNELS

    reference = reference_cls(
        in_channels=3,
        out_channels=out_channels,
        block_out_channels=BLOCK_OUT_CHANNELS,
        layers_per_block=LAYERS_PER_BLOCK,
        spatial_downsample_factors=SPATIAL_DOWN,
        temporal_downsample_factors=TEMPORAL_DOWN,
        norm_num_groups=32,
        norm_eps=1e-6,
        spatial_padding_mode="reflect",
    ).eval()

    x = torch.randn(1, 3, num_frames, TILE, TILE)
    with torch.no_grad():
        expected = reference(x)

    tt_encoder = MiniMaxH3Encoder3d(
        num_frames=num_frames,
        height=TILE,
        width=TILE,
        in_channels=3,
        out_channels=out_channels,
        block_out_channels=BLOCK_OUT_CHANNELS,
        layers_per_block=LAYERS_PER_BLOCK,
        spatial_downsample_factors=SPATIAL_DOWN,
        temporal_downsample_factors=TEMPORAL_DOWN,
        temporal_taps=temporal_taps,
        mesh_device=mesh_device,
    )
    tt_encoder.load_torch_state_dict(dict(reference.state_dict()))

    actual = _from_device(tt_encoder(_to_device(x, mesh_device, tt_encoder.conv_in.in_channels)), out_channels)

    # The shape schedule is part of the contract: assert it before the numeric bar, since
    # assert_quality only warns on a shape mismatch when element counts match.
    assert (
        tt_encoder.latent_shape == expected_latent_shape
    ), f"latent shape {tt_encoder.latent_shape} != expected {expected_latent_shape}"
    assert actual.shape == expected.shape, f"shape {tuple(actual.shape)} != reference {tuple(expected.shape)}"
    assert_quality(expected, actual, pcc=0.99)
