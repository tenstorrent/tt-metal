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
from ....utils.test import ring_params_req_exact_devices

# The VAE's fixed work units: the encoder always runs (17, 256, 256) tiles and the decoder
# always (7, 16, 16) latent chunks, so every gate that builds one uses these shapes.
TILE = 256
CLIP_FRAMES = 17
LATENT_TILE = 16
DECODE_LATENT_FRAMES = 7


def weights_subdir(subfolder: str) -> str | None:
    base = os.environ.get("MINIMAX_H3_DIFFUSERS_DIR", "/data/cglagovich/MiniMax-H3-diffusers")
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
    """State dict from a randomly-initialised reference encoder -- fast, and enough for timing."""
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
    """Likewise for the 36-layer decoder: 2.4 B random parameters beat a 10.4 GB read.

    ``num_layers`` overrides the config depth, for gates that only need the ops exercised
    rather than the full 2.4 B parameters materialised.
    """
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
    """Peak signal-to-noise ratio in dB, with the peak taken from the reference's own range.

    The roundtrip quality gates use this rather than PCC alone: PCC per component says the
    port matches the reference, but a faint vignette or a dull high end sails through a
    0.99 PCC and shows up as a PSNR drop.
    """
    mse = torch.mean((reference.float() - test.float()) ** 2).item()
    if mse == 0.0:
        return float("inf")
    peak = reference.abs().max().item()
    return float("inf") if peak == 0.0 else 20.0 * math.log10(peak) - 10.0 * math.log10(mse)


def create_fractal_image(width: int, height: int) -> Image.Image:
    """A Mandelbrot escape-time image, the repo's existing convention for an I2V seed.

    Copied from `tests/models/wan2_2/test_pipeline_wan_i2v.py`, and it is the right tool for the
    *discriminating* case for one reason: a fractal is content the model would never generate for this
    prompt, so "decoded frame 0 resembles the keyframe" cannot be satisfied by a pipeline that ignores
    the keyframe. See `test_fl2va_follows_the_keyframe`.
    """
    c = np.linspace(-2.0, 1.0, width)[None, :] + 1j * np.linspace(-1.5, 1.5, height)[:, None]
    z = np.zeros_like(c)
    img = np.zeros(c.shape, dtype=np.uint8)
    for i in range(32):
        z = z * z + c
        img[(img == 0) & (np.abs(z) > 2)] = 255 - 8 * i
    return Image.fromarray(np.dstack((img, np.roll(img, width // 10, 1), np.roll(img, height // 10, 0))), "RGB")


def randomize_norm_weights(module: torch.nn.Module, *, scale: float = 0.5) -> torch.nn.Module:
    """Give every `nn.RMSNorm` in `module` a non-trivial affine weight, in place.

    `nn.RMSNorm` initialises `weight` to all ones, so a reference model built with random weights
    (rather than loaded from the checkpoint) has an *identity* affine in every norm. That makes the
    norm weights invisible to a PCC comparison: a port that loaded the wrong norm weight, swapped two
    of them, or never loaded them at all would still match the reference exactly.

    MiniMax-H3 is full of RMSNorms -- `norm1`, `norm2`, the per-head `norm_q`/`norm_k`, the refiner's
    `final_norm` -- so this blind spot covers most of the model's non-matmul parameters. Measured on
    the token refiner at real dims, randomizing the norms moves "norm weights never loaded" from PCC
    1.000000 (undetectable) to 0.887, and "norm1/norm2 swapped" from 1.000000 to 0.986.

    Call this on the torch reference *before* taking its `state_dict`, so the TT module under test
    loads the same non-trivial values.
    """
    for submodule in module.modules():
        if isinstance(submodule, torch.nn.RMSNorm) and submodule.weight is not None:
            submodule.weight.data = 1.0 + scale * torch.randn_like(submodule.weight.data)
    return module


# ------------------------------------------------------------ transformer test fixtures
#
# Shared by `test_transformer_minimax_h3.py` (correctness) and `test_performance_minimax_h3.py`
# (device perf), so the mesh parametrization, the real block config and the packed-sequence
# layout cannot drift between the two files.

# The Galaxy 4x8 ring mesh every device test in this directory runs on: SP=8 on mesh axis 1,
# TP=4 on axis 0, 2 links.
GALAXY_4X8_RING = pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "num_links", "device_params", "topology", "is_fsdp"),
    [
        pytest.param(
            (4, 8), 1, 0, 2, ring_params_req_exact_devices, ttnn.Topology.Ring, False, id="4x8sp1tp0nl2_ring_is_fsdp0"
        ),
    ],
    indirect=["mesh_device", "device_params"],
)

# Real MiniMax-H3 transformer-block config, under the torch reference's kwarg names. The rope
# config sits beside it rather than in it: the TT modules consume precomputed rope tables, so
# `rope_freq_dim`/`rope_theta` are caller-owned there.
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

# The same block config under the TT module's kwarg names (`num_heads`/`head_dim` rather than
# the reference's `num_attention_heads`/`attention_head_dim`).
TT_BLOCK_CONFIG = dict(
    hidden_size=REAL_BLOCK_CONFIG["hidden_size"],
    num_heads=REAL_BLOCK_CONFIG["num_attention_heads"],
    head_dim=REAL_BLOCK_CONFIG["attention_head_dim"],
    ffn_dim=REAL_BLOCK_CONFIG["ffn_dim"],
    time_embed_dim=REAL_BLOCK_CONFIG["time_embed_dim"],
    norm_eps=REAL_BLOCK_CONFIG["norm_eps"],
    qk_norm_eps=REAL_BLOCK_CONFIG["qk_norm_eps"],
)

# Token tags, per the reference: 0 video, 1 text, 2 audio (-1 padding, unused in these tests).
TAG_VIDEO, TAG_TEXT, TAG_AUDIO = 0, 1, 2


def packed_layout(
    num_text: int,
    num_audio: int,
    num_video: int,
    grid_hw: tuple[int, int] = (8, 8),
    padded_len: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build one packed-sequence layout: `(position_ids, token_tags, timestep_indices)`.

    The block is agnostic to how the pipeline orders rows -- it only reads per-row modality tags and
    timestep indices through `adaln_indices` -- so this is a representative layout rather than the
    real t2va one: text rows, then audio rows, then video rows.

    Two distinct timesteps are used so the AdaLN table is addressed at more than one noise level, as
    the real model does when it serves conditioning rows and target rows in a single forward: text and
    the first video frame are clean (timestep 0), the remaining video and all audio are noisy
    (timestep 1). That covers four distinct `(timestep, modality)` table rows including row 0, so an
    off-by-one-modality error in the per-row gather cannot pass unnoticed.

    Video rows get a (t, h, w) patch grid over `grid_hw`; text and audio rows advance the shared `t`
    clock with h = w = 0, which is enough to exercise the 3-axis rope on every modality.

    If `padded_len` is given, all three tensors are zero-padded at the tail to that length: pad rows
    are excluded from attention via ring attention's logical_n, so their values only need to stay in
    range for the gathers.
    """
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
    # Text rows clean; audio noisy; first video frame clean (conditioning), rest noisy (target).
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
    """Upload one prepared `(cos, sin)` table pair, sharded the way every rope consumer wants it.

    Takes the 2D `(seq_len, dim)` tables `prepare_rope_tables` returns, shaped to
    `[1, 1, seq_len, dim]` on device: fp32, fractured on SP along the sequence. cos/sin are shared
    by every head, so they are replicated on TP.
    """

    def _upload(table: torch.Tensor) -> ttnn.Tensor:
        return from_torch(
            table.reshape(1, 1, *table.shape),
            device=mesh_device,
            dtype=ttnn.float32,
            mesh_axes=[..., sp_axis, None],
        )

    return _upload(rope_cos), _upload(rope_sin)


# ------------------------------------------------------------------- tt VAE builders
#
# The constructor-from-config kwarg lists below mirror `random_encoder_state` /
# `random_decoder_state` above: one place derives the tt module from the checkpoint
# config. Test-specific choices (`temporal_taps`, `num_layers`, `parallel_config`, ...)
# are deliberately not defaulted here -- pass them explicitly at the call site.


def build_visual_encoder(config: dict, mesh_device, num_frames: int, **overrides):
    """The tt visual encoder at the fixed 256x256 tile, from the checkpoint config."""
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
    """The tt ViT decoder at the fixed (7, 16, 16) latent chunk, from the checkpoint config."""
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
    """The tt audio (BigVGAN) decoder, from the checkpoint config."""
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


# ------------------------------------------------------------- Qwen3-VL conditioner
#
# Shared by the text-encoder (t2va) and vision-conditioner (fl2va) gates. The download
# scope is the caller's: the repository carries ~190 GB across three partitions, and the
# text-encoder tests deliberately fetch a narrower pattern set than the vision tests.

CONDITIONER_LOCAL_MIRROR = "/data/cglagovich/MiniMax-H3-diffusers"
CONDITIONER_HF_REPO = "MiniMaxAI/MiniMax-H3"
CONDITIONER_SUBFOLDER = "text_encoder"


def conditioner_checkpoint_dir(patterns: list[str]) -> str:
    """The directory holding the Qwen3-VL conditioner.

    `MINIMAX_H3_REPO` (a local directory or a Hub repo id), then the local mirror, then a
    Hub snapshot scoped to `patterns`. A missing checkpoint is a skip, not a failure: there
    is nothing to compare against, and that is an environment gap rather than a defect in
    the port.
    """
    from huggingface_hub import snapshot_download
    from loguru import logger

    try:
        ref = os.environ.get("MINIMAX_H3_REPO", "").strip()
        if ref and os.path.isdir(ref):
            root = ref
        elif not ref and os.path.isdir(CONDITIONER_LOCAL_MIRROR):
            root = CONDITIONER_LOCAL_MIRROR
        else:
            repo_id = ref or CONDITIONER_HF_REPO
            logger.info(f"MiniMax-H3 conditioner not local; fetching {patterns} from {repo_id}")
            root = snapshot_download(repo_id=repo_id, allow_patterns=patterns)
        return os.path.join(root, CONDITIONER_SUBFOLDER)
    except Exception as exc:  # noqa: BLE001 - transport/auth/gating failures are a skip, not a failure
        pytest.skip(
            f"MiniMax-H3 conditioner unavailable (tried $MINIMAX_H3_REPO, {CONDITIONER_LOCAL_MIRROR}, then "
            f"{CONDITIONER_HF_REPO}): {exc}"
        )


def load_reference_conditioner(path: str):
    """The released conditioner through `Qwen3VLForConditionalGeneration`, load-info checked.

    Loaded through the full class diffusers declares in its `ComponentSpec`, in the
    checkpoint's own bf16. The load-info assert proves the shipped weights actually landed,
    rather than leaving parts of the reference on its fresh init -- a silently partial load
    is the one way a parity comparison could go green without having tested the checkpoint.
    `loading_info` values are *sets*, so they are sorted before slicing; indexing a set
    raises, and this runs on the failure path where a crash would hide the mismatch it is
    meant to report.
    """
    import transformers

    hf, info = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
        path, dtype=torch.bfloat16, output_loading_info=True
    )
    bad = {k: sorted(info[k])[:5] for k in ("missing_keys", "unexpected_keys", "mismatched_keys") if info[k]}
    assert not bad, f"conditioner load key mismatch: {bad}"
    return hf
