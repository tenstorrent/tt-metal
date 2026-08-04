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

    39 frames is deliberately not a multiple of 17, so the last-frame repeat padding runs;
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

    The 36-layer numerics are gated per-tile in ``test_vae_decoder_minimax_h3.py``. What is
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
