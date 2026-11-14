# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""MiniMax-H3 (t2va) transformer for LoRA training on a still image."""

from .packing import (
    AUDIO_CHANNELS,
    AUDIO_LATENTS,
    AUDIO_TAG,
    PATCH_SIZE,
    TEXT_TAG,
    VIDEO_TAG,
    Layout,
    adaln_indices,
    build_packed_sequence,
    build_row_timesteps,
    patchify_video_latents,
    sample_timestep,
    shared_timesteps,
    tile_pad,
)
from .flops import calculate_flops_per_token, calculate_sequence_flops
from .transformer import H3Block, H3Config, H3Inputs, H3RefinerBlock, H3Transformer, upload
from .weights import load_transformer, read_time_embedder, unpermute_qk_rows

__all__ = [
    "AUDIO_CHANNELS",
    "AUDIO_LATENTS",
    "AUDIO_TAG",
    "PATCH_SIZE",
    "TEXT_TAG",
    "VIDEO_TAG",
    "H3Block",
    "H3Config",
    "H3Inputs",
    "H3RefinerBlock",
    "H3Transformer",
    "Layout",
    "adaln_indices",
    "build_packed_sequence",
    "build_row_timesteps",
    "calculate_flops_per_token",
    "calculate_sequence_flops",
    "load_transformer",
    "patchify_video_latents",
    "read_time_embedder",
    "sample_timestep",
    "shared_timesteps",
    "tile_pad",
    "unpermute_qk_rows",
    "upload",
]
