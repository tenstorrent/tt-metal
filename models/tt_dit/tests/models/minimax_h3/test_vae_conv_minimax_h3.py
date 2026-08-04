# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Gate M8a.1/M8a.2: the MiniMax-H3 visual VAE convolutions and resnet blocks.

Every check compares against the **pinned diffusers reference classes**
(``MiniMaxH3VideoCausalConv3d``, ``MiniMaxH3VideoDownsample3d``,
``MiniMaxH3VideoResnetBlock3d``) rather than a hand-written port, so a divergence
localises to this code rather than to a second implementation of the same maths.

What each case is actually here to catch:

* the **reflect** spatial pad -- ``neighbor_pad_async`` has no reflect mode, so it is
  done locally; reflect differs from replicate only in the outermost pixel, which is
  exactly the kind of error that survives a loose PCC bar and reads as a vignette;
* the **causal** temporal pad (``kernel_t - 1`` zero frames prepended, nothing
  appended) and its degenerate T=1 form, where only ``weight[:, :, -1]`` survives;
* the downsample's **asymmetric** bottom/right pre-pad, which is what makes the
  strided output exactly ``ceil(size / 2)``;
* ``conv_out`` at **1024 -> 48**, a non-32-multiple output channel count that reaches
  ``conv3d`` through the ``max(32, out)`` rule.

Every shape here is taken from the encoder's **real** schedule for the shipping tile, not
invented. Spatial tiling fixes the tile at 256x256 and the clip at 17 frames, so 768P and
1440P at 5 s and 10 s all reduce to the same per-level shapes -- only the tile and clip
*counts* differ. Picking channel/spatial pairings the real model never produces is not
just wasted coverage: ``(C=128, T=5, 32x32)`` deadlocks ``ttnn.group_norm``, and that
combination exists only in a synthetic stack.
"""

import pytest
import torch

import ttnn

from ....models.vae.minimax_h3.conv_minimax_h3 import MiniMaxH3CausalConv3d
from ....models.vae.minimax_h3.encoder_minimax_h3 import MiniMaxH3Downsample3d, MiniMaxH3ResnetBlock3d
from ....utils.check import assert_quality

# A single device has no ring partner: requesting a fabric ring makes the ethernet
# handshake time out before any kernel runs.
SINGLE_DEVICE = [pytest.param((1, 1), {}, id="single_device")]


def _reference_module(name):
    pytest.importorskip("diffusers", reason="pinned diffusers reference not installed")
    from diffusers.models.autoencoders import autoencoder_kl_minimax_h3 as ref

    module = getattr(ref, name, None)
    if module is None:
        pytest.skip(f"{name} missing -- diffusers is not at the pinned MiniMax-H3 commit")
    return module


def _to_device(x_BCTHW: torch.Tensor, mesh_device, aligned_channels: int) -> ttnn.Tensor:
    """``(B, C, T, H, W)`` torch -> ``(B, T, H, W, C)`` ROW_MAJOR fp32, C zero-padded."""
    x = x_BCTHW.permute(0, 2, 3, 4, 1).contiguous()
    if x.shape[-1] < aligned_channels:
        x = torch.nn.functional.pad(x, (0, aligned_channels - x.shape[-1]))
    return ttnn.from_torch(x, dtype=ttnn.float32, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT)


def _from_device(tensor: ttnn.Tensor, out_channels: int) -> torch.Tensor:
    """``(B, T, H, W, C)`` -> ``(B, C, T, H, W)`` torch, dropping padded out-channels."""
    x = ttnn.to_torch(tensor).float()
    return x[..., :out_channels].permute(0, 4, 1, 2, 3).contiguous()


def _assert_same(expected: torch.Tensor, actual: torch.Tensor, *, pcc: float) -> None:
    # assert_quality only warns on a shape mismatch when the element counts match, so a
    # transposed output would slip through silently. Pin the shape first.
    assert actual.shape == expected.shape, f"shape {tuple(actual.shape)} != reference {tuple(expected.shape)}"
    assert_quality(expected, actual, pcc=pcc)


# One case per distinct conv the encoder contains. Spatial extents are the real ones for
# that level; convs carry no norms, so these are cheap to run at true size.
@pytest.mark.parametrize(
    ("in_channels", "out_channels", "kernel_size", "spatial_padding", "height", "width"),
    [
        pytest.param(3, 128, 3, 1, 64, 64, id="conv_in_3to128"),
        pytest.param(256, 256, 3, 1, 64, 64, id="resnet_conv_256"),
        pytest.param(512, 1024, 1, 0, 16, 16, id="conv_shortcut_k1_512to1024"),
        pytest.param(1024, 48, 3, 1, 16, 16, id="conv_out_1024to48"),
    ],
)
@pytest.mark.parametrize(
    ("num_frames", "temporal_taps"),
    [pytest.param(1, 1, id="keyframe_T1"), pytest.param(5, 3, id="clip_T5")],
)
@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE_DEVICE, indirect=["mesh_device", "device_params"])
def test_causal_conv3d(
    mesh_device, in_channels, out_channels, kernel_size, spatial_padding, height, width, num_frames, temporal_taps
):
    """One conv per shape the encoder uses, reflect edges and causal pad included."""
    reference_cls = _reference_module("MiniMaxH3VideoCausalConv3d")
    torch.manual_seed(0)

    # k1 convs carry no temporal extent; k3 convs prepend kernel_t - 1 = 2 zero frames.
    temporal_padding = 0 if kernel_size == 1 else 2
    reference = reference_cls(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        spatial_padding=spatial_padding,
        temporal_padding=temporal_padding,
        spatial_padding_mode="reflect",
    ).eval()

    x = torch.randn(1, in_channels, num_frames, height, width)
    with torch.no_grad():
        expected = reference(x)

    tt_conv = MiniMaxH3CausalConv3d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        spatial_padding=spatial_padding,
        temporal_taps=temporal_taps,
        mesh_device=mesh_device,
    )
    tt_conv.load_torch_state_dict(dict(reference.state_dict()))
    actual = _from_device(tt_conv(_to_device(x, mesh_device, tt_conv.in_channels)), out_channels)

    _assert_same(expected, actual, pcc=0.999)


# Real downsample sites: level 2 is space+time at 64x64 (T=9), level 3 is space-only at
# 32x32 (T=5).
@pytest.mark.parametrize(
    ("channels", "temporal_stride", "clip_frames", "height", "width"),
    [
        pytest.param(256, 2, 9, 64, 64, id="L2_space_and_time_64x64"),
        pytest.param(512, 1, 5, 32, 32, id="L3_space_only_32x32"),
    ],
)
@pytest.mark.parametrize("keyframe", [pytest.param(True, id="keyframe_T1"), pytest.param(False, id="clip")])
@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE_DEVICE, indirect=["mesh_device", "device_params"])
def test_downsample(mesh_device, channels, temporal_stride, clip_frames, height, width, keyframe):
    """Stride-2 conv behind the asymmetric bottom/right reflect pre-pad."""
    reference_cls = _reference_module("MiniMaxH3VideoDownsample3d")
    torch.manual_seed(1)
    num_frames, temporal_taps = (1, 1) if keyframe else (clip_frames, 3)

    reference = reference_cls(
        channels, channels, temporal_stride=temporal_stride, spatial_stride=2, spatial_padding_mode="reflect"
    ).eval()
    x = torch.randn(1, channels, num_frames, height, width)
    with torch.no_grad():
        expected = reference(x)

    tt_downsample = MiniMaxH3Downsample3d(
        channels,
        channels,
        temporal_stride=temporal_stride,
        spatial_stride=2,
        temporal_taps=temporal_taps,
        mesh_device=mesh_device,
    )
    tt_downsample.load_torch_state_dict(dict(reference.state_dict()))
    actual = _from_device(tt_downsample(_to_device(x, mesh_device, tt_downsample.conv.in_channels)), channels)

    _assert_same(expected, actual, pcc=0.999)


# (in_channels, out_channels, T, H, W) drawn from the real encoder schedule for a
# 17x256x256 tile: level 2 sits at 64x64 with T=9, levels 3-5 at 32x32 and 16x16 with
# T=5. The keyframe variant is the same stack at T=1.
REAL_RESNET_SHAPES = [
    pytest.param(256, 256, 9, 64, 64, id="L2_256_64x64"),
    pytest.param(256, 512, 5, 32, 32, id="L3_shortcut_256to512_32x32"),
    pytest.param(512, 512, 5, 16, 16, id="L4_512_16x16"),
    pytest.param(512, 1024, 5, 16, 16, id="L5_shortcut_512to1024_16x16"),
]


@pytest.mark.parametrize(("in_channels", "out_channels", "clip_frames", "height", "width"), REAL_RESNET_SHAPES)
@pytest.mark.parametrize("keyframe", [pytest.param(True, id="keyframe_T1"), pytest.param(False, id="clip")])
@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE_DEVICE, indirect=["mesh_device", "device_params"])
def test_resnet_block(mesh_device, in_channels, out_channels, clip_frames, height, width, keyframe):
    """A whole resnet, so the two GroupNorms and the k1 shortcut are exercised together.

    The bar is looser than for a bare conv: ``ttnn.group_norm`` has no fp32 path, so each
    of the two norms is a bf16 island and that is the block's precision floor.
    """
    reference_cls = _reference_module("MiniMaxH3VideoResnetBlock3d")
    torch.manual_seed(2)
    num_frames, temporal_taps = (1, 1) if keyframe else (clip_frames, 3)

    reference = reference_cls(
        in_channels, out_channels, norm_num_groups=32, norm_eps=1e-6, spatial_padding_mode="reflect"
    ).eval()
    x = torch.randn(1, in_channels, num_frames, height, width)
    with torch.no_grad():
        expected = reference(x)

    tt_block = MiniMaxH3ResnetBlock3d(
        in_channels,
        out_channels,
        num_frames=num_frames,
        height=height,
        width=width,
        temporal_taps=temporal_taps,
        mesh_device=mesh_device,
    )
    tt_block.load_torch_state_dict(dict(reference.state_dict()))
    actual = _from_device(tt_block(_to_device(x, mesh_device, in_channels)), out_channels)

    _assert_same(expected, actual, pcc=0.995)
