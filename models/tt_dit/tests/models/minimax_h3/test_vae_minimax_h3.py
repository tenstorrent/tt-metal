# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Gates for the MiniMax-H3 visual VAE: host tiling geometry, tiled encode/decode vs the
pinned diffusers reference, per-conv/resnet parity, the ViT decoder, and the chunked
roundtrip. Uses the real checkpoint's encoder tensors when present, random otherwise."""

import os

import pytest
import torch
from loguru import logger

import ttnn

from ....models.vae.minimax_h3.conv_minimax_h3 import MiniMaxH3CausalConv3d
from ....models.vae.minimax_h3.decoder_minimax_h3 import MiniMaxH3TransformerBlock, MiniMaxH3ViTDecoder3d, unpatchify
from ....models.vae.minimax_h3.encoder_minimax_h3 import MiniMaxH3ResnetBlock3d
from ....models.vae.minimax_h3.rope_minimax_h3 import (
    head_lane_permutation,
    permuted_rotate,
    position_grid,
    reference_rotate,
    rope_tables,
)
from ....models.vae.minimax_h3.vae_minimax_h3 import MiniMaxH3Vae, MiniMaxH3VaeConfig, split_tiles
from ....utils.check import assert_quality
from .common import load_config, psnr

SINGLE_DEVICE = [pytest.param((1, 1), {}, id="single_device")]

# Frame counts follow the 17n+5 -> 5n+2 rule at 24 fps: 5 s = 124 frames, 10 s = 243.
PRODUCTION_CONFIGS = [
    pytest.param(1344, 768, 124, id="768P_5s"),
    pytest.param(1344, 768, 243, id="768P_10s"),
    pytest.param(2560, 1440, 124, id="1440P_5s"),
    pytest.param(2560, 1440, 243, id="1440P_10s"),
]


def _weights_dir() -> str | None:
    base = os.environ.get("MINIMAX_H3_MODEL_PATH")
    if not base:
        return None
    candidate = os.path.join(base, "vae")
    return candidate if os.path.isfile(os.path.join(candidate, "config.json")) else None


def _reference(name: str):
    pytest.importorskip("diffusers", reason="pinned diffusers reference not installed")
    from diffusers.models.autoencoders import autoencoder_kl_minimax_h3 as ref

    attribute = getattr(ref, name, None)
    if attribute is None:
        pytest.skip(f"{name} missing -- diffusers is not at the pinned MiniMax-H3 commit")
    return attribute


def _raw_config(weights_dir: str | None) -> dict:
    if weights_dir is None:
        pytest.skip("MiniMax-H3 vae/config.json not found; set MINIMAX_H3_MODEL_PATH")
    return load_config(weights_dir)


def _build_reference(weights_dir: str | None):
    """The reference VAE, loading only the encoder-side tensors from the checkpoint."""
    cls = _reference("AutoencoderKLMiniMaxH3")
    raw = _raw_config(weights_dir)
    config = MiniMaxH3VaeConfig.from_pretrained(weights_dir)
    reference = cls(**raw).eval()

    index_path = os.path.join(weights_dir, "diffusion_pytorch_model.safetensors.index.json")
    if os.path.isfile(index_path):
        import json

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
    reference_cls = _reference("AutoencoderKLMiniMaxH3")
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
        assert set(lengths) == {reference.tile_sample_min_height}, f"ragged tiles at {length}: {set(lengths)}"

    assert (num_frames - 5) % config.clip_length == 0, f"{num_frames} is not 17n+5"


@pytest.mark.parametrize(
    ("num_frames", "temporal_taps", "expected_latent_frames"),
    [pytest.param(1, 1, 1, id="keyframe"), pytest.param(17, 3, 5, id="clip")],
)
@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE_DEVICE, indirect=["mesh_device", "device_params"])
def test_encode_clip_tiled(mesh_device, num_frames, temporal_taps, expected_latent_frames):
    """Tiled ``encode_clip`` vs the reference's tiled ``_encode_clip``; also the whole-encoder gate."""
    # Whole-encoder coverage skips without MINIMAX_H3_MODEL_PATH.
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

    assert (
        actual.shape[2] == expected_latent_frames
    ), f"{actual.shape[2]} latent frames, expected {expected_latent_frames}"
    _assert_same(expected, actual, pcc=0.99)  # ttnn.group_norm has no fp32 path: bf16 floor


@pytest.mark.parametrize(
    ("num_frames", "temporal_taps", "expected_latent_frames"),
    [pytest.param(1, 1, 1, id="keyframe"), pytest.param(17, 3, 5, id="clip")],
)
@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE_DEVICE, indirect=["mesh_device", "device_params"])
def test_encode_clip_uint8_pixel_norm(mesh_device, num_frames, temporal_taps, expected_latent_frames):
    """The `pixel_norm` fold: raw uint8 pixels into a folded encoder vs the reference on
    normalized fp32. Gates the conv_in weight/bias fold, the uint8 -> typecast -> pad upload
    chain, and -- on the clip case -- the `255 * mean` causal front-pad values, which only a
    taps=3 encode with a temporal zero-pad in the reference can catch."""
    from ....pipelines.minimax_h3.conditioning import MINIMAX_H3_PIXEL_MEAN, MINIMAX_H3_PIXEL_STD

    weights_dir = _weights_dir()
    reference, config = _build_reference(weights_dir)
    torch.manual_seed(6)

    extent = 512
    pixels_uint8 = torch.randint(0, 256, (1, 3, num_frames, extent, extent), dtype=torch.uint8)
    mean = torch.tensor(MINIMAX_H3_PIXEL_MEAN).view(1, -1, 1, 1, 1)
    std = torch.tensor(MINIMAX_H3_PIXEL_STD).view(1, -1, 1, 1, 1)
    normalized = (pixels_uint8.float().div(255.0) - mean) / std
    with torch.no_grad():
        expected = reference._encode_clip(normalized)

    tt_vae = MiniMaxH3Vae(config, mesh_device=mesh_device, pixel_norm=(MINIMAX_H3_PIXEL_MEAN, MINIMAX_H3_PIXEL_STD))
    tt_vae.load_encoder_state(dict(reference.state_dict()))
    actual = tt_vae.encode_clip(pixels_uint8, temporal_taps=temporal_taps)

    assert (
        actual.shape[2] == expected_latent_frames
    ), f"{actual.shape[2]} latent frames, expected {expected_latent_frames}"
    _assert_same(expected, actual, pcc=0.99)


def _shallow_decoder_reference(num_layers: int):
    """Reference VAE with a shallow decoder: geometry gates don't need 36-layer numerics."""
    cls = _reference("AutoencoderKLMiniMaxH3")
    raw = _raw_config(_weights_dir())
    raw["decoder_num_layers"] = num_layers
    reference = cls(**raw).eval()
    with torch.no_grad():
        for block in reference.decoder.transformer_blocks:
            # scale1/scale2 initialise to zero, which would make every block the identity.
            block.scale1.normal_(0, 0.1)
            block.scale2.normal_(0, 0.1)
    return reference, MiniMaxH3VaeConfig(**raw)


@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE_DEVICE, indirect=["mesh_device", "device_params"])
def test_decode_clip_tiled(mesh_device):
    """Tiled ``decode_clip`` vs the reference's, including the pixel-space cross-fade."""
    reference, config = _shallow_decoder_reference(2)
    torch.manual_seed(6)

    latent_extent = 512 // config.spatial_compression_ratio
    z = torch.randn(1, config.latent_channels, 7, latent_extent, latent_extent)
    with torch.no_grad():
        expected = reference._decode_clip(z)

    tt_vae = MiniMaxH3Vae(config, mesh_device=mesh_device)
    tt_vae.load_decoder_state(dict(reference.state_dict()))
    actual = tt_vae.decode_clip(z)

    _assert_same(expected, actual, pcc=0.99)


@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE_DEVICE, indirect=["mesh_device", "device_params"])
def test_visual_roundtrip_quality(mesh_device):
    """Encode -> decode vs the reference's own round trip; also the video-chunking gate (39 frames)."""
    reference, config = _shallow_decoder_reference(2)
    torch.manual_seed(3)

    num_frames = 39
    x = torch.randn(1, 3, num_frames, 256, 256) * 0.5
    with torch.no_grad():
        expected_moments = reference._encode(x)
        expected = reference.decode(expected_moments.chunk(2, dim=1)[0]).sample

    tt_vae = MiniMaxH3Vae(config, mesh_device=mesh_device)
    state = dict(reference.state_dict())
    tt_vae.load_encoder_state(state)
    tt_vae.load_decoder_state(state)

    moments = tt_vae.encode(x)
    expected_latent_frames = (num_frames + (-num_frames) % config.clip_length) // config.clip_length
    expected_latent_frames = expected_latent_frames * config.tokens_chunk_size - config.token_drop
    assert (
        moments.shape[2] == expected_latent_frames
    ), f"{moments.shape[2]} latent frames, expected {expected_latent_frames}"
    _assert_same(expected_moments, moments, pcc=0.99)

    actual = tt_vae.decode(moments.chunk(2, dim=1)[0])

    _assert_same(expected, actual, pcc=0.99)
    psnr_db = psnr(expected, actual)
    logger.info(f"ROUNDTRIP visual PSNR: {psnr_db:.2f} dB")
    assert psnr_db >= 25.0, f"visual roundtrip PSNR {psnr_db:.2f} dB < 25 dB"


# -------------------------------------------------------------------- convolutions and resnet blocks

# A single device has no ring partner: requesting a fabric ring times out the ethernet handshake.


def _to_device(x_BCTHW: torch.Tensor, mesh_device, aligned_channels: int) -> ttnn.Tensor:
    """``(B, C, T, H, W)`` torch -> ``(B, T, H, W, C)`` ROW_MAJOR fp32, C zero-padded."""
    x = x_BCTHW.permute(0, 2, 3, 4, 1).contiguous()
    if x.shape[-1] < aligned_channels:
        x = torch.nn.functional.pad(x, (0, aligned_channels - x.shape[-1]))
    return ttnn.from_torch(x, dtype=ttnn.float32, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT)


def _from_device(tensor: ttnn.Tensor, out_channels: int) -> torch.Tensor:
    x = ttnn.to_torch(tensor).float()
    return x[..., :out_channels].permute(0, 4, 1, 2, 3).contiguous()


def _assert_same(expected: torch.Tensor, actual: torch.Tensor, *, pcc: float) -> None:
    # assert_quality only warns on same-element-count shape mismatches; pin the shape first.
    assert actual.shape == expected.shape, f"shape {tuple(actual.shape)} != reference {tuple(expected.shape)}"
    assert_quality(expected, actual, pcc=pcc)


@pytest.mark.parametrize(
    (
        "in_channels",
        "out_channels",
        "kernel_size",
        "spatial_padding",
        "stride",
        "trailing_pad",
        "num_frames",
        "height",
        "width",
    ),
    [
        pytest.param(3, 128, 3, 1, (1, 1, 1), 0, 5, 64, 64, id="conv_in_3to128"),
        pytest.param(256, 256, 3, 1, (1, 1, 1), 0, 5, 64, 64, id="resnet_conv_256"),
        pytest.param(512, 1024, 1, 0, (1, 1, 1), 0, 5, 16, 16, id="conv_shortcut_k1_512to1024"),
        pytest.param(1024, 48, 3, 1, (1, 1, 1), 0, 5, 16, 16, id="conv_out_1024to48"),
        pytest.param(256, 256, 3, 0, (2, 2, 2), 1, 9, 64, 64, id="downsample_L2_space_and_time_64x64"),
        pytest.param(512, 512, 3, 0, (1, 2, 2), 1, 5, 32, 32, id="downsample_L3_space_only_32x32"),
    ],
)
@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE_DEVICE, indirect=["mesh_device", "device_params"])
def test_causal_conv3d(
    mesh_device,
    in_channels,
    out_channels,
    kernel_size,
    spatial_padding,
    stride,
    trailing_pad,
    num_frames,
    height,
    width,
):
    """One conv per shape the encoder uses, reflect edges and causal pad included."""
    torch.manual_seed(0)

    if trailing_pad:
        reference_cls = _reference("MiniMaxH3VideoDownsample3d")
        reference = reference_cls(
            in_channels,
            out_channels,
            temporal_stride=stride[0],
            spatial_stride=stride[1],
            spatial_padding_mode="reflect",
        ).eval()
        state = {k.removeprefix("conv."): v for k, v in reference.state_dict().items()}
    else:
        reference_cls = _reference("MiniMaxH3VideoCausalConv3d")
        temporal_padding = 0 if kernel_size == 1 else 2  # k3 prepends kernel_t - 1 causal zero frames
        reference = reference_cls(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            spatial_padding=spatial_padding,
            temporal_padding=temporal_padding,
            spatial_padding_mode="reflect",
        ).eval()
        state = dict(reference.state_dict())

    x = torch.randn(1, in_channels, num_frames, height, width)
    with torch.no_grad():
        expected = reference(x)

    tt_conv = MiniMaxH3CausalConv3d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        spatial_padding=spatial_padding,
        trailing_spatial_padding=trailing_pad,
        temporal_taps=3,
        mesh_device=mesh_device,
    )
    tt_conv.load_torch_state_dict(state)
    actual = _from_device(tt_conv(_to_device(x, mesh_device, tt_conv.in_channels)), out_channels)

    _assert_same(expected, actual, pcc=0.999)


# Real encoder schedule only -- synthetic (C=128, T=5, 32x32) deadlocks ttnn.group_norm.
REAL_RESNET_SHAPES = [
    pytest.param(256, 256, 9, 64, 64, id="L2_256_64x64"),
    pytest.param(256, 512, 5, 32, 32, id="L3_shortcut_256to512_32x32"),
    pytest.param(512, 512, 5, 16, 16, id="L4_512_16x16"),
    pytest.param(512, 1024, 5, 16, 16, id="L5_shortcut_512to1024_16x16"),
]


@pytest.mark.parametrize(("in_channels", "out_channels", "num_frames", "height", "width"), REAL_RESNET_SHAPES)
@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE_DEVICE, indirect=["mesh_device", "device_params"])
def test_resnet_block(mesh_device, in_channels, out_channels, num_frames, height, width):
    """A whole resnet, so the two GroupNorms and the k1 shortcut are exercised together."""
    reference_cls = _reference("MiniMaxH3VideoResnetBlock3d")
    torch.manual_seed(2)
    temporal_taps = 3

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

    _assert_same(expected, actual, pcc=0.995)  # ttnn.group_norm has no fp32 path: bf16 floor


# -------------------------------------------------------------------- the 36-layer ViT decoder

# The production decode unit: one 256x256 pixel tile over one 7-latent-frame chunk.
LATENT_FRAMES, LATENT_H, LATENT_W = 7, 16, 16
NUM_PATCHES = LATENT_FRAMES * LATENT_H * LATENT_W
LATENT_CHANNELS = 24
DIM, NUM_HEADS, HEAD_DIM = 2048, 32, 64
NUM_REGISTER_TOKENS = 4
NUM_SUFFIX_TOKENS = NUM_REGISTER_TOKENS + 1
EPS = 1e-5
ROPE_THETA, ROPE_DIM_RATIO = 100.0, 0.75


def _reference_rope(num_suffix_tokens: int):
    rope_cls = _reference("MiniMaxH3VideoRotaryPosEmbed")
    module = rope_cls(int(HEAD_DIM * ROPE_DIM_RATIO), theta=ROPE_THETA)
    positions = position_grid(LATENT_FRAMES, LATENT_H, LATENT_W)
    positions = torch.cat([positions, positions.new_zeros((num_suffix_tokens, 3))], dim=0).unsqueeze(0)
    cos, sin = module(positions)
    return cos[0, :, 0, :], sin[0, :, 0, :]


def _rope_tables(permuted: bool):
    return rope_tables(
        LATENT_FRAMES,
        LATENT_H,
        LATENT_W,
        num_suffix_tokens=NUM_SUFFIX_TOKENS,
        attention_head_dim=HEAD_DIM,
        rope_dim_ratio=ROPE_DIM_RATIO,
        theta=ROPE_THETA,
        permuted=permuted,
    )


def test_rope_tables_are_bit_exact():
    """Host-only: rope tables bit-exact vs the reference, and permuted rotation == reference rotation."""
    reference_cos, reference_sin = _reference_rope(NUM_SUFFIX_TOKENS)
    cos, sin = _rope_tables(permuted=False)
    assert cos.shape == reference_cos.shape, f"{tuple(cos.shape)} != {tuple(reference_cos.shape)}"
    assert torch.equal(cos, reference_cos), f"cos differs by {(cos - reference_cos).abs().max()}"
    assert torch.equal(sin, reference_sin), f"sin differs by {(sin - reference_sin).abs().max()}"

    torch.manual_seed(0)
    total = NUM_PATCHES + NUM_SUFFIX_TOKENS
    x = torch.randn(1, total, NUM_HEADS, HEAD_DIM)

    expected = reference_rotate(x, reference_cos.unsqueeze(1), reference_sin.unsqueeze(1))

    permutation = head_lane_permutation(HEAD_DIM, ROPE_DIM_RATIO)
    cos, sin = _rope_tables(permuted=True)
    rotated = permuted_rotate(x.index_select(-1, permutation), cos.unsqueeze(1), sin.unsqueeze(1))
    actual = rotated.index_select(-1, torch.argsort(permutation))

    assert torch.equal(actual, expected), f"rotation differs by {(actual - expected).abs().max()}"

    rotary_dim = reference_cos.shape[-1]
    assert torch.equal(actual[..., rotary_dim:], x[..., rotary_dim:]), "pass-through lanes were modified"
    assert torch.equal(actual[:, -NUM_SUFFIX_TOKENS:], x[:, -NUM_SUFFIX_TOKENS:]), "suffix rows are not identity"


def _to_device_tiled(x: torch.Tensor, mesh_device) -> ttnn.Tensor:
    return ttnn.from_torch(x, dtype=ttnn.bfloat16, device=mesh_device, layout=ttnn.TILE_LAYOUT)


@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE_DEVICE, indirect=["mesh_device", "device_params"])
def test_transformer_block(mesh_device):
    """One block: LayerScale, the weight-only RMS norms, and the SwiGLU FFN together."""
    block_cls = _reference("MiniMaxH3VideoTransformerBlock")
    torch.manual_seed(3)
    total = NUM_PATCHES + NUM_SUFFIX_TOKENS

    reference = block_cls(dim=DIM, heads=NUM_HEADS, dim_head=HEAD_DIM, ffn_mult=4, eps=EPS, bias=True).eval()
    # scale1/scale2 initialise to zeros, which would make the block the identity.
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

    cos, sin = _rope_tables(permuted=True)
    actual = ttnn.to_torch(
        tt_block(
            _to_device_tiled(x, mesh_device),
            _to_device_tiled(cos.view(1, 1, total, HEAD_DIM), mesh_device),
            _to_device_tiled(sin.view(1, 1, total, HEAD_DIM), mesh_device),
        )
    ).float()

    _assert_same(expected, actual, pcc=0.998)


@pytest.mark.parametrize("num_layers", [pytest.param(36, id="full_36_layers")])
@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE_DEVICE, indirect=["mesh_device", "device_params"])
def test_decoder(mesh_device, num_layers):
    """The full decoder on the production latent tile, against the reference decoder."""
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

    tokens = z.permute(0, 2, 3, 4, 1).reshape(1, NUM_PATCHES, LATENT_CHANNELS)
    out_tokens = ttnn.to_torch(tt_decoder(_to_device_tiled(tokens, mesh_device))).float()
    actual = unpatchify(out_tokens, num_frames=LATENT_FRAMES, height=LATENT_H, width=LATENT_W, out_channels=3)

    _assert_same(expected, actual, pcc=0.99)
