# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""End-to-end gate for the MiniMax-H3 visual VAE encode path.

Three layers, cheapest first:

1. **Tiling geometry, host only.** For every supported working point -- 768P and 1440P
   at 5 s and 10 s -- our ``split_tiles`` must agree with the reference's ``_split_tiles``
   exactly, and the derived temporal geometry must match the reference's derived fields.
   This is where an off-by-one in the tile solver would otherwise hide until it showed up
   as a seam in the output.
2. **Tiled encode against the reference, on device.** At 512x512 the solver emits 3x3
   real 256x256 tiles with real overlaps, so this exercises the production tile shape,
   the tile loop, and the cross-fade stitch -- at a size where the host reference is
   still affordable. A full 1344x768 clip is 28 tiles x ~8 TFLOP, which the torch
   reference cannot do in reasonable time; the geometry test above is what covers the
   difference between 3x3 and 4x7.
3. **Video chunking against the reference.** ``encode`` pads to a whole number of
   17-frame clips and drops the trailing ``token_drop`` latent frames; both are easy to
   get subtly wrong and neither is visible in a single-clip test.

The real checkpoint is used when it is present -- only the ~180 M encoder tensors are
read, not the whole 10.4 GB -- and the test falls back to reference-initialised random
weights otherwise, since the wiring is what is under test either way.
"""

import os

import pytest
import torch
import torch.nn.functional as F

import ttnn

from ....models.vae.minimax_h3.conv_minimax_h3 import MiniMaxH3CausalConv3d
from ....models.vae.minimax_h3.decoder_minimax_h3 import MiniMaxH3TransformerBlock, MiniMaxH3ViTDecoder3d, unpatchify
from ....models.vae.minimax_h3.encoder_minimax_h3 import (
    MiniMaxH3Downsample3d,
    MiniMaxH3Encoder3d,
    MiniMaxH3ResnetBlock3d,
)
from ....models.vae.minimax_h3.rope_minimax_h3 import (
    head_lane_permutation,
    permuted_rotate,
    position_grid,
    reference_rotate,
    rope_tables,
)
from ....models.vae.minimax_h3.vae_minimax_h3 import MiniMaxH3Vae, MiniMaxH3VaeConfig, split_tiles
from ....utils.check import assert_quality

SINGLE_DEVICE = [pytest.param((1, 1), {}, id="single_device")]

# The supported working points. Frame counts follow the 17n+5 -> 5n+2 rule at 24 fps, so
# 5 s is 124 frames (n=7) and 10 s is 243 frames (n=14).
PRODUCTION_CONFIGS = [
    pytest.param(1344, 768, 124, id="768P_5s"),
    pytest.param(1344, 768, 243, id="768P_10s"),
    pytest.param(2560, 1440, 124, id="1440P_5s"),
    pytest.param(2560, 1440, 243, id="1440P_10s"),
]


def _weights_dir() -> str | None:
    candidates = [
        os.environ.get("MINIMAX_H3_DIFFUSERS_DIR", "") and os.path.join(os.environ["MINIMAX_H3_DIFFUSERS_DIR"], "vae"),
        "/data/cglagovich/MiniMax-H3-diffusers/vae",
    ]
    for candidate in candidates:
        if candidate and os.path.isfile(os.path.join(candidate, "config.json")):
            return candidate
    return None


def _reference_vae_cls():
    pytest.importorskip("diffusers", reason="pinned diffusers reference not installed")
    from diffusers.models.autoencoders import autoencoder_kl_minimax_h3 as ref

    cls = getattr(ref, "AutoencoderKLMiniMaxH3", None)
    if cls is None:
        pytest.skip("AutoencoderKLMiniMaxH3 missing -- diffusers is not at the pinned commit")
    return cls


def _build_reference(weights_dir: str | None):
    """The reference VAE, from real weights when available.

    Only the encoder side is needed, so the 2.4 B-parameter ViT decoder is left at its
    random initialisation rather than read off disk.
    """
    cls = _reference_vae_cls()
    if weights_dir is None:
        pytest.skip("MiniMax-H3 vae/config.json not found; set MINIMAX_H3_DIFFUSERS_DIR")
    config = MiniMaxH3VaeConfig.from_pretrained(weights_dir)
    import json

    raw = {
        k: v
        for k, v in json.loads(open(os.path.join(weights_dir, "config.json")).read()).items()
        if not k.startswith("_")
    }
    reference = cls(**raw).eval()

    index_path = os.path.join(weights_dir, "diffusion_pytorch_model.safetensors.index.json")
    if os.path.isfile(index_path):
        from safetensors.torch import load_file

        weight_map = json.loads(open(index_path).read())["weight_map"]
        wanted = {k: f for k, f in weight_map.items() if k.startswith("encoder.") or k.startswith("quant_conv.")}
        loaded: dict[str, torch.Tensor] = {}
        for shard in sorted(set(wanted.values())):
            shard_tensors = load_file(os.path.join(weights_dir, shard))
            loaded.update({k: v for k, v in shard_tensors.items() if k in wanted})
        missing = reference.load_state_dict(loaded, strict=False)
        assert not [
            k for k in missing.missing_keys if k.startswith(("encoder.", "quant_conv."))
        ], "real encoder weights did not fully load"
    return reference, config


@pytest.mark.parametrize(("width", "height", "num_frames"), PRODUCTION_CONFIGS)
def test_tiling_geometry_matches_reference(width, height, num_frames):
    """Host-only: our tile solver and derived geometry equal the reference's."""
    reference_cls = _reference_vae_cls()
    reference = reference_cls()
    config = MiniMaxH3VaeConfig()

    for key in ("frame_pre_padding", "tokens_chunk_size", "token_overlap", "frame_overlap"):
        assert getattr(config, key) == getattr(
            reference, key
        ), f"{key}: {getattr(config, key)} != reference {getattr(reference, key)}"

    ratio = config.spatial_compression_ratio
    for length in (height, width):
        expected_starts, expected_lengths, expected_overlaps = reference._split_tiles(
            length, reference.tile_sample_min_height, reference.tile_sample_min_overlap_height
        )
        starts, lengths, overlaps = split_tiles(
            length, reference.tile_sample_min_height, reference.tile_sample_min_overlap_height, ratio
        )
        assert (starts, lengths, overlaps) == (
            expected_starts,
            expected_lengths,
            expected_overlaps,
        ), f"tile solver disagrees at length {length}"
        # Every tile is exactly tile_size, which is what lets one encoder serve them all.
        assert set(lengths) == {reference.tile_sample_min_height}, f"ragged tiles at {length}: {set(lengths)}"

    # The frame count must be a clean 17n+5, or the latent count is not 5n+2.
    assert (num_frames - 5) % config.clip_length == 0, f"{num_frames} is not 17n+5"


@pytest.mark.parametrize(
    ("num_frames", "temporal_taps"),
    [pytest.param(1, 1, id="keyframe"), pytest.param(17, 3, id="clip")],
)
@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE_DEVICE, indirect=["mesh_device", "device_params"])
def test_encode_clip_tiled(mesh_device, num_frames, temporal_taps):
    """Tiled ``encode_clip`` against the reference's own tiled ``_encode_clip``.

    512x512 is the smallest frame the solver splits into a 3x3 grid of full 256x256
    tiles, so the tile shape, the tile loop and the overlap cross-fade are all the
    production ones.
    """
    weights_dir = _weights_dir()
    reference, config = _build_reference(weights_dir)
    torch.manual_seed(4)

    extent = 512
    x = torch.randn(1, 3, num_frames, extent, extent)
    with torch.no_grad():
        expected = reference._encode_clip(x)

    tt_vae = MiniMaxH3Vae(config, mesh_device=mesh_device)
    tt_vae.load_encoder_state(dict(reference.state_dict()))
    actual = tt_vae.encode_clip(x, temporal_taps=temporal_taps)

    assert actual.shape == expected.shape, f"shape {tuple(actual.shape)} != reference {tuple(expected.shape)}"
    assert_quality(expected, actual, pcc=0.99)


@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE_DEVICE, indirect=["mesh_device", "device_params"])
def test_encode_video_chunking(mesh_device):
    """``encode`` against the reference ``_encode``: clip padding and ``token_drop``.

    39 frames is not a multiple of 17, so the last-frame repeat padding runs;
    the expected latent count is ``ceil(39/17) * 5 - 3 = 12``.
    """
    weights_dir = _weights_dir()
    reference, config = _build_reference(weights_dir)
    torch.manual_seed(5)

    num_frames, extent = 39, 256
    x = torch.randn(1, 3, num_frames, extent, extent)
    with torch.no_grad():
        expected = reference._encode(x)

    tt_vae = MiniMaxH3Vae(config, mesh_device=mesh_device)
    tt_vae.load_encoder_state(dict(reference.state_dict()))
    actual = tt_vae.encode(x)

    expected_latent_frames = (num_frames + (-num_frames) % config.clip_length) // config.clip_length
    expected_latent_frames = expected_latent_frames * config.tokens_chunk_size - config.token_drop
    assert (
        actual.shape[2] == expected_latent_frames
    ), f"{actual.shape[2]} latent frames, expected {expected_latent_frames}"
    assert actual.shape == expected.shape, f"shape {tuple(actual.shape)} != reference {tuple(expected.shape)}"
    assert_quality(expected, actual, pcc=0.99)


def _shallow_decoder_reference(num_layers: int):
    """A reference VAE with a ``num_layers``-deep decoder, for the tiling/chunking gates.

    The 36-layer numerics are gated per-tile in the ViT-decoder section below. What is
    left to prove end to end is the *geometry* -- tile layout, pixel-space blend extents,
    the ``token_overlap`` chunk stride and the trailing-frame crop -- and running 36 layers
    over 9 tiles on host would be ~95 TFLOP to prove something a 2-layer decoder proves
    just as well.
    """
    cls = _reference_vae_cls()
    weights_dir = _weights_dir()
    if weights_dir is None:
        pytest.skip("MiniMax-H3 vae/config.json not found; set MINIMAX_H3_DIFFUSERS_DIR")
    import json

    raw = {
        k: v
        for k, v in json.loads(open(os.path.join(weights_dir, "config.json")).read()).items()
        if not k.startswith("_")
    }
    raw["decoder_num_layers"] = num_layers
    reference = cls(**raw).eval()
    with torch.no_grad():
        for block in reference.decoder.transformer_blocks:
            # They initialise to zero, which would make every block the identity.
            block.scale1.normal_(0, 0.1)
            block.scale2.normal_(0, 0.1)
    return reference, MiniMaxH3VaeConfig(**raw)


@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE_DEVICE, indirect=["mesh_device", "device_params"])
def test_decode_clip_tiled(mesh_device):
    """Tiled ``decode_clip`` vs the reference's, including the pixel-space cross-fade.

    Decode tiles are laid out in pixel space and mapped back onto the latent grid, so the
    blend extents are *not* divided by the compression ratio the way encode's are -- an easy
    thing to get backwards, and invisible except as a seam.
    """
    reference, config = _shallow_decoder_reference(2)
    torch.manual_seed(6)

    # 512 pixels -> a 3x3 grid of full 256x256 tiles; 7 latent frames is one decode chunk.
    latent_extent = 512 // config.spatial_compression_ratio
    z = torch.randn(1, config.latent_channels, 7, latent_extent, latent_extent)
    with torch.no_grad():
        expected = reference._decode_clip(z)

    tt_vae = MiniMaxH3Vae(config, mesh_device=mesh_device)
    tt_vae.load_decoder_state(dict(reference.state_dict()))
    actual = tt_vae.decode_clip(z)

    assert actual.shape == expected.shape, f"shape {tuple(actual.shape)} != reference {tuple(expected.shape)}"
    assert_quality(expected, actual, pcc=0.99)


@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE_DEVICE, indirect=["mesh_device", "device_params"])
def test_decode_video_chunking(mesh_device):
    """``decode`` vs the reference ``_decode``: chunk stride, cross-fade and trailing crop."""
    reference, config = _shallow_decoder_reference(2)
    torch.manual_seed(7)

    latent_extent = 256 // config.spatial_compression_ratio
    # 12 latent frames is what a 39-frame encode produces, so this pairs with the encode gate.
    z = torch.randn(1, config.latent_channels, 12, latent_extent, latent_extent)
    with torch.no_grad():
        expected = reference._decode(z)

    tt_vae = MiniMaxH3Vae(config, mesh_device=mesh_device)
    tt_vae.load_decoder_state(dict(reference.state_dict()))
    actual = tt_vae.decode(z)

    assert actual.shape == expected.shape, f"shape {tuple(actual.shape)} != reference {tuple(expected.shape)}"
    assert_quality(expected, actual, pcc=0.99)


# -------------------------------------------------------------------- the convolutions and resnet blocks
#
# Gate M8a.1/M8a.2: the MiniMax-H3 visual VAE convolutions and resnet blocks.
#
# Every check compares against the **pinned diffusers reference classes**
# (``MiniMaxH3VideoCausalConv3d``, ``MiniMaxH3VideoDownsample3d``,
# ``MiniMaxH3VideoResnetBlock3d``) rather than a hand-written port, so a divergence
# localises to this code rather than to a second implementation of the same maths.
#
# What each case is actually here to catch:
#
# * the **reflect** spatial pad -- ``neighbor_pad_async`` has no reflect mode, so it is
#   done locally; reflect differs from replicate only in the outermost pixel, which is
#   exactly the kind of error that survives a loose PCC bar and reads as a vignette;
# * the **causal** temporal pad (``kernel_t - 1`` zero frames prepended, nothing
#   appended) and its degenerate T=1 form, where only ``weight[:, :, -1]`` survives;
# * the downsample's **asymmetric** bottom/right pre-pad, which is what makes the
#   strided output exactly ``ceil(size / 2)``;
# * ``conv_out`` at **1024 -> 48**, a non-32-multiple output channel count that reaches
#   ``conv3d`` through the ``max(32, out)`` rule.
#
# Every shape here is taken from the encoder's **real** schedule for the shipping tile, not
# invented. Spatial tiling fixes the tile at 256x256 and the clip at 17 frames, so 768P and
# 1440P at 5 s and 10 s all reduce to the same per-level shapes -- only the tile and clip
# *counts* differ. Picking channel/spatial pairings the real model never produces is not
# just wasted coverage: ``(C=128, T=5, 32x32)`` deadlocks ``ttnn.group_norm``, and that
# combination exists only in a synthetic stack.


# A single device has no ring partner: requesting a fabric ring makes the ethernet
# handshake time out before any kernel runs.


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


# -------------------------------------------------------------------- the encoder, whole
#
# Gate M8a.3/M8b: the MiniMax-H3 visual VAE encoder, whole.
#
# Two independent layers of evidence, kept apart:
#
# * **Host, no device** -- the T=1 collapse claim: a single frame reduces H3's causal 3D
#   encoder to a 2D one, because ``temporal_padding = kernel_t - 1`` prepends *zeros*, so
#   only ``weight[:, :, -1]`` survives. If that ever stops holding (a checkpoint with
#   non-zero temporal padding, say) these fail on their own and say why, instead of
#   surfacing as a vague PCC drop somewhere in a twelve-resnet stack.
# * **Device** -- the tt encoder against the pinned diffusers
#   ``MiniMaxH3VideoEncoder3d``, at both ``temporal_taps=1`` (keyframe) and
#   ``temporal_taps=3`` (17-frame clip), then the tiled ``encode_clip`` against the
#   reference's own tiling.
#
# The device bar is 0.99 rather than 0.999: ``ttnn.group_norm`` has no fp32 path, so all
# thirteen norms in the stack are bf16 islands, and that is the encoder's precision floor.
# Per-conv parity at 0.999 is gated in the convolutions section above.


# A single device has no ring partner: requesting a fabric ring makes the ethernet
# handshake time out before any kernel runs.

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


# -------------------------------------------------------------------- the 36-layer ViT decoder
#
# Gate M8c: the MiniMax-H3 visual VAE's 36-layer ViT decoder.
#
# The production decode shape is fixed by tiling: one call is always a
# ``(1, 24, 7, 16, 16)`` latent tile, i.e. ``7*16*16 = 1792`` patches plus a 5-token
# suffix, so every test here runs at that shape.
#
# Ordered cheapest-first, and the first two need no device at all:
#
# * **RoPE**, host: the tables must be bit-exact against the reference module, and the
#   permuted ``alt_complex_rotate90`` form must reproduce the reference rotation exactly.
#   This is the decoder's riskiest detail -- only 48 of each head's 64 lanes rotate, and
#   the pairing is ``(i, i+24)``, so the usual full-width RoPE op is simply wrong here.
# * **swiglu half order**, host: the checkpoint's ``ff.net.0.proj`` packs ``[value; gate]``
#   and tt_dit's swiglu wants the same, so no swap is needed -- unlike the H3 *DiT*, where
#   a recorded amendment says the halves must be swapped. Applying that amendment here
#   would corrupt every FFN, so the order is asserted rather than assumed.
# * **one block** (which is where a missing ``scale1``/``scale2`` shows
#   up as PCC near zero), then the **full 36 layers**.


# The production decode unit: one 256x256 pixel tile over one 7-latent-frame chunk.
LATENT_FRAMES, LATENT_H, LATENT_W = 7, 16, 16
NUM_PATCHES = LATENT_FRAMES * LATENT_H * LATENT_W
LATENT_CHANNELS = 24
DIM, NUM_HEADS, HEAD_DIM = 2048, 32, 64
NUM_REGISTER_TOKENS = 4
NUM_SUFFIX_TOKENS = NUM_REGISTER_TOKENS + 1
EPS = 1e-5
ROPE_THETA, ROPE_DIM_RATIO = 100.0, 0.75


def _reference(name):
    pytest.importorskip("diffusers", reason="pinned diffusers reference not installed")
    from diffusers.models.autoencoders import autoencoder_kl_minimax_h3 as ref

    attribute = getattr(ref, name, None)
    if attribute is None:
        pytest.skip(f"{name} missing -- diffusers is not at the pinned MiniMax-H3 commit")
    return attribute


def _reference_rope(num_suffix_tokens: int):
    """The reference module's own ``(cos, sin)`` for the production latent shape."""
    rope_cls = _reference("MiniMaxH3VideoRotaryPosEmbed")
    module = rope_cls(int(HEAD_DIM * ROPE_DIM_RATIO), theta=ROPE_THETA)
    positions = position_grid(LATENT_FRAMES, LATENT_H, LATENT_W)
    positions = torch.cat([positions, positions.new_zeros((num_suffix_tokens, 3))], dim=0).unsqueeze(0)
    cos, sin = module(positions)
    return cos[0, :, 0, :], sin[0, :, 0, :]


def test_rope_tables_are_bit_exact():
    """Host-only: our cos/sin must equal the reference module's exactly."""
    reference_cos, reference_sin = _reference_rope(NUM_SUFFIX_TOKENS)
    cos, sin = rope_tables(
        LATENT_FRAMES,
        LATENT_H,
        LATENT_W,
        num_suffix_tokens=NUM_SUFFIX_TOKENS,
        attention_head_dim=HEAD_DIM,
        rope_dim_ratio=ROPE_DIM_RATIO,
        theta=ROPE_THETA,
        permuted=False,
    )
    assert cos.shape == reference_cos.shape, f"{tuple(cos.shape)} != {tuple(reference_cos.shape)}"
    assert torch.equal(cos, reference_cos), f"cos differs by {(cos - reference_cos).abs().max()}"
    assert torch.equal(sin, reference_sin), f"sin differs by {(sin - reference_sin).abs().max()}"


def test_permuted_rope_matches_reference_rotation():
    """Host-only: lane permute + rot90 == the reference's half-split rotation.

    Also pins the two properties that make the no-slice trick valid: the pass-through
    lanes are untouched, and the suffix rows are the identity.
    """
    reference_cos, reference_sin = _reference_rope(NUM_SUFFIX_TOKENS)
    torch.manual_seed(0)
    total = NUM_PATCHES + NUM_SUFFIX_TOKENS
    x = torch.randn(1, total, NUM_HEADS, HEAD_DIM)

    expected = reference_rotate(x, reference_cos.unsqueeze(1), reference_sin.unsqueeze(1))

    permutation = head_lane_permutation(HEAD_DIM, ROPE_DIM_RATIO)
    cos, sin = rope_tables(
        LATENT_FRAMES,
        LATENT_H,
        LATENT_W,
        num_suffix_tokens=NUM_SUFFIX_TOKENS,
        attention_head_dim=HEAD_DIM,
        rope_dim_ratio=ROPE_DIM_RATIO,
        theta=ROPE_THETA,
        permuted=True,
    )
    rotated = permuted_rotate(x.index_select(-1, permutation), cos.unsqueeze(1), sin.unsqueeze(1))
    actual = rotated.index_select(-1, torch.argsort(permutation))

    assert torch.equal(actual, expected), f"rotation differs by {(actual - expected).abs().max()}"

    rotary_dim = reference_cos.shape[-1]
    assert torch.equal(actual[..., rotary_dim:], x[..., rotary_dim:]), "pass-through lanes were modified"
    assert torch.equal(actual[:, -NUM_SUFFIX_TOKENS:], x[:, -NUM_SUFFIX_TOKENS:]), "suffix rows are not identity"


def test_swiglu_half_order_needs_no_swap():
    """Host-only: the checkpoint packs ``[value; gate]``, matching tt_dit -- no swap.

    A recorded amendment says the H3 *DiT*'s ``fc1`` halves must be swapped. That came from
    the raw MiniMax layout, not the diffusers-converted one, so applying it to the VAE
    decoder would silently corrupt every FFN. Exactly one of the two orders must match.
    """
    pytest.importorskip("diffusers", reason="pinned diffusers reference not installed")
    from diffusers.models.attention import FeedForward

    torch.manual_seed(1)
    dim, mult = 256, 4
    reference = FeedForward(dim, mult=mult, activation_fn="swiglu", bias=True).eval()
    x = torch.randn(2, 8, dim)
    with torch.no_grad():
        expected = reference(x)

    projected = reference.net[0].proj(x)
    first, second = projected.chunk(2, dim=-1)
    value_times_silu_gate = reference.net[2](first * torch.nn.functional.silu(second))
    gate_times_silu_value = reference.net[2](second * torch.nn.functional.silu(first))

    assert torch.allclose(
        value_times_silu_gate, expected, atol=1e-5
    ), "first half is not the value: tt_dit's [value; gate] convention would be wrong here"
    assert not torch.allclose(
        gate_times_silu_value, expected, atol=1e-5
    ), "both orders matched, so this test cannot detect a swap"


def _to_device_tiled(x: torch.Tensor, mesh_device) -> ttnn.Tensor:
    """bf16 TILE, for the ViT decoder's token tensors.

    Distinct from `_to_device` above, which is fp32 ROW_MAJOR with channel padding for the
    convolutional encoder path.
    """
    return ttnn.from_torch(x, dtype=ttnn.bfloat16, device=mesh_device, layout=ttnn.TILE_LAYOUT)


@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE_DEVICE, indirect=["mesh_device", "device_params"])
def test_transformer_block(mesh_device):
    """One block: LayerScale, the weight-only RMS norms, and the SwiGLU FFN together.

    A missing or mis-shaped ``scale1``/``scale2`` shows up here as PCC near zero rather
    than as a subtle drift, which is exactly why this is a separate gate from attention.
    """
    block_cls = _reference("MiniMaxH3VideoTransformerBlock")
    torch.manual_seed(3)
    total = NUM_PATCHES + NUM_SUFFIX_TOKENS

    reference = block_cls(dim=DIM, heads=NUM_HEADS, dim_head=HEAD_DIM, ffn_mult=4, eps=EPS, bias=True).eval()
    # scale1/scale2 initialise to zeros, which would make the block the identity and hide
    # any error in the attention or FFN. Give them realistic non-zero values.
    with torch.no_grad():
        reference.scale1.normal_(0, 0.1)
        reference.scale2.normal_(0, 0.1)

    x = torch.randn(1, total, DIM)
    reference_cos, reference_sin = _reference_rope(NUM_SUFFIX_TOKENS)
    with torch.no_grad():
        expected = reference(x, (reference_cos.view(1, total, 1, -1), reference_sin.view(1, total, 1, -1)))

    tt_block = MiniMaxH3TransformerBlock(
        DIM, num_heads=NUM_HEADS, head_dim=HEAD_DIM, ffn_mult=4, eps=EPS, mesh_device=mesh_device
    )
    tt_block.load_torch_state_dict(dict(reference.state_dict()))

    cos, sin = rope_tables(
        LATENT_FRAMES,
        LATENT_H,
        LATENT_W,
        num_suffix_tokens=NUM_SUFFIX_TOKENS,
        attention_head_dim=HEAD_DIM,
        rope_dim_ratio=ROPE_DIM_RATIO,
        theta=ROPE_THETA,
        permuted=True,
    )
    actual = ttnn.to_torch(
        tt_block(
            _to_device_tiled(x, mesh_device),
            _to_device_tiled(cos.view(1, 1, total, HEAD_DIM), mesh_device),
            _to_device_tiled(sin.view(1, 1, total, HEAD_DIM), mesh_device),
        )
    ).float()

    assert actual.shape == expected.shape, f"shape {tuple(actual.shape)} != {tuple(expected.shape)}"
    assert_quality(expected, actual, pcc=0.998)


@pytest.mark.parametrize("num_layers", [pytest.param(36, id="full_36_layers")])
@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE_DEVICE, indirect=["mesh_device", "device_params"])
def test_decoder(mesh_device, num_layers):
    """The decoder on the production latent tile, against the reference decoder.

    Covers ``proj_in``, the fused suffix constant, the RoPE constants, ``norm_out``,
    ``proj_out`` and the unpatchify tail, at the full 2.4 B parameters.
    """
    decoder_cls = _reference("MiniMaxH3VideoViTDecoder3d")
    torch.manual_seed(4)

    reference = decoder_cls(
        in_channels=LATENT_CHANNELS,
        out_channels=3,
        patch_size=16,
        patch_size_t=4,
        num_layers=num_layers,
        num_attention_heads=NUM_HEADS,
        attention_head_dim=HEAD_DIM,
        num_register_tokens=NUM_REGISTER_TOKENS,
        ffn_mult=4,
        rope_theta=ROPE_THETA,
        rope_dim_ratio=ROPE_DIM_RATIO,
        norm_eps=EPS,
    ).eval()
    with torch.no_grad():
        for block in reference.transformer_blocks:
            block.scale1.normal_(0, 0.1)
            block.scale2.normal_(0, 0.1)

    z = torch.randn(1, LATENT_CHANNELS, LATENT_FRAMES, LATENT_H, LATENT_W)
    with torch.no_grad():
        expected = reference(z)

    tt_decoder = MiniMaxH3ViTDecoder3d(
        num_frames=LATENT_FRAMES,
        height=LATENT_H,
        width=LATENT_W,
        in_channels=LATENT_CHANNELS,
        out_channels=3,
        num_layers=num_layers,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        num_register_tokens=NUM_REGISTER_TOKENS,
        rope_theta=ROPE_THETA,
        rope_dim_ratio=ROPE_DIM_RATIO,
        eps=EPS,
        mesh_device=mesh_device,
    )
    tt_decoder.load_torch_state_dict(dict(reference.state_dict()))

    # The caller owns the latent-to-token flatten, mirroring the reference's own permute.
    tokens = z.permute(0, 2, 3, 4, 1).reshape(1, NUM_PATCHES, LATENT_CHANNELS)
    out_tokens = ttnn.to_torch(tt_decoder(_to_device_tiled(tokens, mesh_device))).float()
    actual = unpatchify(out_tokens, num_frames=LATENT_FRAMES, height=LATENT_H, width=LATENT_W, out_channels=3)

    assert actual.shape == expected.shape, f"shape {tuple(actual.shape)} != {tuple(expected.shape)}"
    assert_quality(expected, actual, pcc=0.99)
