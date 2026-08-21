# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the MiniMax-H3 bringup tests."""

import json
import math
import os

import numpy as np
import pytest
import torch
from PIL import Image

import ttnn

from ....utils.tensor import from_torch
from ....utils.test import ring_params_8k_req_exact_devices, ring_params_req_exact_devices

# Fixed VAE work units: encoder (17, 256, 256) tiles, decoder (7, 16, 16) latent chunks.
TILE = 256
CLIP_FRAMES = 17
LATENT_TILE = 16
DECODE_LATENT_FRAMES = 7


def weights_subdir(subfolder: str) -> str | None:
    base = os.environ.get("MINIMAX_H3_MODEL_PATH")
    if not base:
        return None
    candidate = os.path.join(base, subfolder)
    return candidate if os.path.isfile(os.path.join(candidate, "config.json")) else None


def load_config(weights_dir: str) -> dict:
    return {
        k: v
        for k, v in json.loads(open(os.path.join(weights_dir, "config.json")).read()).items()
        if not k.startswith("_")
    }


def _reference_class(name: str):
    pytest.importorskip("diffusers", reason="pinned diffusers reference not installed")
    from diffusers.models.autoencoders import autoencoder_kl_minimax_h3 as ref

    return getattr(ref, name)


def random_encoder_state(config: dict) -> dict:
    cls = _reference_class("MiniMaxH3VideoEncoder3d")
    module = cls(
        in_channels=3,
        out_channels=2 * config["latent_channels"],
        block_out_channels=tuple(config["block_out_channels"]),
        layers_per_block=config["layers_per_block"],
        spatial_downsample_factors=tuple(config["spatial_downsample_factors"]),
        temporal_downsample_factors=tuple(config["temporal_downsample_factors"]),
        norm_num_groups=config["norm_num_groups"],
        norm_eps=config["norm_eps"],
        spatial_padding_mode=config["spatial_padding_mode"],
    )
    return dict(module.state_dict())


def random_decoder_state(config: dict, *, num_layers: int | None = None) -> dict:
    cls = _reference_class("MiniMaxH3VideoViTDecoder3d")
    module = cls(
        in_channels=config["latent_channels"],
        out_channels=config["out_channels"],
        patch_size=16,
        patch_size_t=4,
        num_layers=config["decoder_num_layers"] if num_layers is None else num_layers,
        num_attention_heads=config["decoder_num_attention_heads"],
        attention_head_dim=config["decoder_attention_head_dim"],
        num_register_tokens=config["decoder_num_register_tokens"],
        ffn_mult=config["decoder_ffn_mult"],
        rope_theta=config["decoder_rope_theta"],
        rope_dim_ratio=config["decoder_rope_dim_ratio"],
        norm_eps=config["decoder_norm_eps"],
    )
    return dict(module.state_dict())


def psnr(reference: torch.Tensor, test: torch.Tensor) -> float:
    """PSNR in dB, with the peak taken from the reference's own range."""
    mse = torch.mean((reference.float() - test.float()) ** 2).item()
    if mse == 0.0:
        return float("inf")
    peak = reference.abs().max().item()
    return float("inf") if peak == 0.0 else 20.0 * math.log10(peak) - 10.0 * math.log10(mse)


def create_fractal_image(width: int, height: int) -> Image.Image:
    """Mandelbrot I2V seed: content the model would never generate, so a pipeline that ignores the keyframe cannot pass."""
    c = np.linspace(-2.0, 1.0, width)[None, :] + 1j * np.linspace(-1.5, 1.5, height)[:, None]
    z = np.zeros_like(c)
    img = np.zeros(c.shape, dtype=np.uint8)
    for i in range(32):
        z = z * z + c
        img[(img == 0) & (np.abs(z) > 2)] = 255 - 8 * i
    return Image.fromarray(np.dstack((img, np.roll(img, width // 10, 1), np.roll(img, height // 10, 0))), "RGB")


# The mesh shapes MiniMax-H3 is tuned for, named individually and composed into a list -- the shape
# `ltx_mesh_params.py` uses. Both carry `require_exact_physical_num_devices`, so exactly one runs on
# any given cluster and the other skips.
#
# The fabric must be FABRIC_1D_RING: `CCLManager` runs ring collectives, and on a line fabric one
# cannot resolve a forwarding direction (`TT_FATAL fabric.cpp:174 forwarding_direction.has_value()`).
# 4x32 additionally takes the 8 KB router payload, matching Wan's 4x32 rows, and a trace region for
# the quad's `trace_denoise`; the region is only reserved, so 4x8 pays nothing but address space.
_L1_SMALL = 65536
_ring = {**ring_params_req_exact_devices, "l1_small_size": _L1_SMALL}
_ring_8k_trace = {**ring_params_8k_req_exact_devices, "trace_region_size": 150_000_000, "l1_small_size": _L1_SMALL}

MESH_4X8_RING = pytest.param((4, 8), _ring, id="4x8")
MESH_4X32_RING = pytest.param((4, 32), _ring_8k_trace, id="4x32")

GALAXY_MESHES = [MESH_4X8_RING, MESH_4X32_RING]


def randomize_norm_weights(module: torch.nn.Module, *, scale: float = 0.5) -> torch.nn.Module:
    """Randomize every `nn.RMSNorm` affine in place, BEFORE taking `state_dict`: all-ones norms make norm-weight loading invisible to PCC."""
    for submodule in module.modules():
        if isinstance(submodule, torch.nn.RMSNorm) and submodule.weight is not None:
            submodule.weight.data = 1.0 + scale * torch.randn_like(submodule.weight.data)
    return module


# ---- transformer test fixtures, shared by the correctness and perf tests so they cannot drift ----

# The transformer tests take the axes explicitly: TP stays on axis 0 at factor 4 and SP absorbs the
# rest, so 4x8 -> 4x32 moves only `sp_factor`, which every test body derives from `mesh_device.shape`.
# `device_params` travels inside the tuple because the router payload differs per shape; crossing them
# independently would pair a 4x8 mesh with the 4x32 router config.
# The transformer tests take the axes explicitly: TP stays on axis 0 at factor 4 and SP absorbs the
# rest, so 4x8 -> 4x32 moves only `sp_factor`, which every test body derives from `mesh_device.shape`.
# `device_params` travels inside the tuple because the router payload differs per shape; crossing them
# independently would pair a 4x8 mesh with the 4x32 router config.
GALAXY_RING = pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "num_links", "device_params", "topology", "is_fsdp"),
    [
        pytest.param((4, 8), 1, 0, 2, _ring, ttnn.Topology.Ring, False, id="4x8sp1tp0nl2_ring_is_fsdp0"),
        pytest.param((4, 32), 1, 0, 2, _ring_8k_trace, ttnn.Topology.Ring, False, id="4x32sp1tp0nl2_ring_is_fsdp0"),
    ],
    indirect=["mesh_device", "device_params"],
)

REAL_BLOCK_CONFIG = dict(
    hidden_size=5376,
    num_attention_heads=56,
    attention_head_dim=128,
    ffn_dim=14336,
    time_embed_dim=2688,
    norm_eps=1e-5,
    qk_norm_eps=1e-5,
)
ROPE_FREQ_DIM = 16
ROPE_THETA = 10000.0

TT_BLOCK_CONFIG = dict(
    hidden_size=REAL_BLOCK_CONFIG["hidden_size"],
    num_heads=REAL_BLOCK_CONFIG["num_attention_heads"],
    head_dim=REAL_BLOCK_CONFIG["attention_head_dim"],
    ffn_dim=REAL_BLOCK_CONFIG["ffn_dim"],
    time_embed_dim=REAL_BLOCK_CONFIG["time_embed_dim"],
    norm_eps=REAL_BLOCK_CONFIG["norm_eps"],
    qk_norm_eps=REAL_BLOCK_CONFIG["qk_norm_eps"],
)

# Reference token tags: 0 video, 1 text, 2 audio.
TAG_VIDEO, TAG_TEXT, TAG_AUDIO = 0, 1, 2


def packed_layout(
    num_text: int,
    num_audio: int,
    num_video: int,
    grid_hw: tuple[int, int] = (8, 8),
    padded_len: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build one packed layout `(position_ids, token_tags, timestep_indices)`: text, audio, video rows over two timesteps, zero-padded to `padded_len`."""
    grid_h, grid_w = grid_hw
    frame = grid_h * grid_w
    assert num_video % frame == 0, "num_video must fill whole (h, w) frames"
    grid_t = num_video // frame
    assert grid_t >= 2, "need at least one conditioning frame and one target frame"

    tags = torch.cat(
        [
            torch.full((num_text,), TAG_TEXT, dtype=torch.long),
            torch.full((num_audio,), TAG_AUDIO, dtype=torch.long),
            torch.full((num_video,), TAG_VIDEO, dtype=torch.long),
        ]
    )
    timestep_indices = torch.cat(
        [
            torch.zeros(num_text, dtype=torch.long),
            torch.ones(num_audio, dtype=torch.long),
            torch.zeros(frame, dtype=torch.long),
            torch.ones(num_video - frame, dtype=torch.long),
        ]
    )

    vt, vh, vw = torch.meshgrid(torch.arange(grid_t), torch.arange(grid_h), torch.arange(grid_w), indexing="ij")
    video_pos = torch.stack([vt.reshape(-1), vh.reshape(-1), vw.reshape(-1)], dim=-1)

    def clock_pos(n: int) -> torch.Tensor:
        return torch.stack([torch.arange(n), torch.zeros(n, dtype=torch.long), torch.zeros(n, dtype=torch.long)], -1)

    position_ids = torch.cat([clock_pos(num_text), clock_pos(num_audio), video_pos], dim=0)

    if padded_len is not None:
        pad = padded_len - position_ids.shape[0]
        if pad:
            position_ids = torch.cat([position_ids, torch.zeros((pad, 3), dtype=position_ids.dtype)])
            tags = torch.cat([tags, torch.zeros(pad, dtype=tags.dtype)])
            timestep_indices = torch.cat([timestep_indices, torch.zeros(pad, dtype=timestep_indices.dtype)])
    return position_ids, tags, timestep_indices


def upload_rope(rope_cos: torch.Tensor, rope_sin: torch.Tensor, *, mesh_device, sp_axis: int):
    """Upload prepared `(cos, sin)` tables: fp32, fractured on SP along the sequence, replicated on TP."""

    def _upload(table: torch.Tensor) -> ttnn.Tensor:
        return from_torch(
            table.reshape(1, 1, *table.shape),
            device=mesh_device,
            dtype=ttnn.float32,
            mesh_axes=[..., sp_axis, None],
        )

    return _upload(rope_cos), _upload(rope_sin)


def build_visual_encoder(config: dict, mesh_device, num_frames: int, **overrides):
    from ....models.vae.minimax_h3.encoder_minimax_h3 import MiniMaxH3Encoder3d

    kwargs = dict(
        num_frames=num_frames,
        height=TILE,
        width=TILE,
        in_channels=3,
        out_channels=2 * config["latent_channels"],
        block_out_channels=tuple(config["block_out_channels"]),
        layers_per_block=config["layers_per_block"],
        spatial_downsample_factors=tuple(config["spatial_downsample_factors"]),
        temporal_downsample_factors=tuple(config["temporal_downsample_factors"]),
        mesh_device=mesh_device,
    )
    kwargs.update(overrides)
    return MiniMaxH3Encoder3d(**kwargs)


def build_visual_decoder(config: dict, mesh_device, **overrides):
    from ....models.vae.minimax_h3.decoder_minimax_h3 import MiniMaxH3ViTDecoder3d

    kwargs = dict(
        num_frames=DECODE_LATENT_FRAMES,
        height=LATENT_TILE,
        width=LATENT_TILE,
        in_channels=config["latent_channels"],
        out_channels=config["out_channels"],
        patch_size=16,
        patch_size_t=4,
        num_layers=config["decoder_num_layers"],
        num_heads=config["decoder_num_attention_heads"],
        head_dim=config["decoder_attention_head_dim"],
        num_register_tokens=config["decoder_num_register_tokens"],
        ffn_mult=config["decoder_ffn_mult"],
        rope_theta=config["decoder_rope_theta"],
        rope_dim_ratio=config["decoder_rope_dim_ratio"],
        eps=config["decoder_norm_eps"],
        mesh_device=mesh_device,
    )
    kwargs.update(overrides)
    return MiniMaxH3ViTDecoder3d(**kwargs)


def build_audio_decoder(config: dict, mesh_device, **overrides):
    from ....models.audio_vae.minimax_h3.decoder_minimax_h3_audio import MiniMaxH3AudioDecoder

    kwargs = dict(
        latent_channels=config["latent_channels"],
        latent_dim=config["latent_dim"],
        decoder_dim=config["decoder_dim"],
        decoder_rates=tuple(config["decoder_rates"]),
        decoder_kernel_sizes=tuple(config["decoder_kernel_sizes"]),
        resblock_kernel_sizes=tuple(config["resblock_kernel_sizes"]),
        resblock_dilation_sizes=tuple(tuple(d) for d in config["resblock_dilation_sizes"]),
        mesh_device=mesh_device,
    )
    kwargs.update(overrides)
    return MiniMaxH3AudioDecoder(**kwargs)


CONDITIONER_SUBFOLDER = "text_encoder"


def conditioner_checkpoint_dir(patterns: list[str]) -> str:
    """Resolve the conditioner directory from $MINIMAX_H3_MODEL_PATH; a missing checkpoint skips.

    `patterns` are the shard patterns the test needs present (validation only, nothing is fetched).
    """
    import glob

    root = os.environ.get("MINIMAX_H3_MODEL_PATH", "")
    if not root or not os.path.isdir(root):
        pytest.skip("set MINIMAX_H3_MODEL_PATH to a MiniMax-H3 diffusers snapshot")
    missing = [pattern for pattern in patterns if not glob.glob(os.path.join(root, pattern))]
    if missing:
        pytest.skip(f"MiniMax-H3 conditioner checkpoint at {root} is missing {missing}")
    return os.path.join(root, CONDITIONER_SUBFOLDER)


def load_reference_conditioner(path: str):
    """Load the conditioner with load-info checked: a silently partial load would pass parity while testing the fresh init."""
    import transformers

    hf, info = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
        path, dtype=torch.bfloat16, output_loading_info=True
    )
    bad = {k: sorted(info[k])[:5] for k in ("missing_keys", "unexpected_keys", "mismatched_keys") if info[k]}
    assert not bad, f"conditioner load key mismatch: {bad}"
    return hf
