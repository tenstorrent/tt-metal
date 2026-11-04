# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
MiniMax-M3 vision encoder on one chip, for the encoder service's media module.

Usage:

    encoder = TtMiniMaxM3VisionEncoder.from_weights_path(mesh_device, weights_path, hf_config)  # once, in load()
    tokens = encoder.forward(image.pixels)  # per image, in encode()
    try:
        read_embeddings(tokens, out)  # out: [n_tokens, 6144] bf16, for example in shared memory
    finally:
        ttnn.deallocate(tokens)

This module imports ttnn, so import it only where the chip is used, as the media module's load() does.
The first image of each patch count compiles the chip's programs, which is slow. Later images of that count reuse them.
forward does not support trace capture, as each call uploads its input to new buffers. The encoder also keeps the
RoPE tables of recent grids on the chip, where a trace replay may overwrite them.
The encoder is not thread safe: make one call at a time.
"""
from __future__ import annotations

import json
import os
from typing import Mapping

import numpy as np
import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.minimax_m3.tt.vision.config import MiniMaxM3VisionConfig
from models.demos.minimax_m3.tt.vision.image_input import MERGE_SIZE, num_tokens
from models.demos.minimax_m3.tt.vision.loader import CHECKPOINT_PREFIXES, load_vision_state_dict
from models.demos.minimax_m3.tt.vision.model import M3VLVisionModel
from models.demos.minimax_m3.tt.vision.preprocess import NUM_CHANNELS, PATCH_SIZE, TEMPORAL_PATCH_SIZE, patch_rows


def vision_config(config) -> MiniMaxM3VisionConfig:
    """Read the vision config from a model config.

    Args:
        config: a MiniMaxM3VisionConfig, returned as is; a dict such as the checkpoint's config.json; or anything
            with to_dict(), such as a transformers MiniMaxM3VLConfig.

    Raises:
        TypeError: for any other config.
        ValueError: for a config without a field that the vision config needs, such as the vision part on its own.
    """
    if isinstance(config, MiniMaxM3VisionConfig):
        return config
    if isinstance(config, Mapping):
        values = dict(config)
    elif callable(getattr(config, "to_dict", None)):
        values = config.to_dict()
    else:
        raise TypeError(
            "config must be a transformers config, a config.json dict or a MiniMaxM3VisionConfig,"
            f" got {type(config).__name__}"
        )
    try:
        return MiniMaxM3VisionConfig.from_hf_config(values)
    except KeyError as error:
        raise ValueError(
            f"config has no field {error.args[0]!r}: it needs vision_config (with hidden_size, intermediate_size,"
            " num_hidden_layers, num_attention_heads and patch_size), projector_hidden_size and text_config.hidden_size,"
            f" as the checkpoint's config.json and MiniMaxM3VLConfig have them; got a {type(config).__name__}"
        ) from error


def _checked_config(mesh_device, config) -> MiniMaxM3VisionConfig:
    """Return vision_config(config) after checking it and mesh_device.

    Raises ValueError unless the config's patch geometry is preprocess.py's and mesh_device is one chip.
    """
    vision = vision_config(config)
    geometry = (vision.patch_size, vision.temporal_patch_size, vision.num_channels, vision.spatial_merge_size)
    if geometry != (PATCH_SIZE, TEMPORAL_PATCH_SIZE, NUM_CHANNELS, MERGE_SIZE):
        raise ValueError(
            f"the image code makes patches of {PATCH_SIZE} x {PATCH_SIZE} pixels, {TEMPORAL_PATCH_SIZE} frames and"
            f" {NUM_CHANNELS} channels, merged {MERGE_SIZE} x {MERGE_SIZE}; config has patch_size {geometry[0]},"
            f" temporal_patch_size {geometry[1]}, num_channels {geometry[2]} and spatial_merge_size {geometry[3]}"
        )
    num_devices = mesh_device.get_num_devices()
    if num_devices != 1:
        raise ValueError(
            f"TtMiniMaxM3VisionEncoder runs on one chip, got a mesh of {num_devices}: give it a 1x1 mesh or"
            " mesh_device.create_submesh(ttnn.MeshShape(1, 1))"
        )
    return vision


class TtMiniMaxM3VisionEncoder(LightweightModule):
    """The vision tower and projector of MiniMax-M3 on one chip: resized pixels in, vision tokens on the chip out."""

    def __init__(self, mesh_device, state_dict: Mapping[str, torch.Tensor], config):
        """Upload the vision weights to one chip.

        Args:
            mesh_device: a mesh of one chip. For a larger mesh, pass mesh_device.create_submesh(ttnn.MeshShape(1, 1)).
            state_dict: the vision weights under transformers 5.12's names (vision_tower.*, multi_modal_projector.*),
                as load_vision_state_dict returns them, in any float dtype. Only the model's weights are read. Other
                keys are ignored, such as a language model's or those of layers past the config's num_hidden_layers.
                So this may be a lazy mapping over a whole checkpoint.
            config: anything vision_config takes, such as the checkpoint's Hugging Face config.

        Raises:
            ValueError: before any upload, for a mesh of several chips, another patch geometry, weights under the
                checkpoint's own names, or a vision weight that is missing, unknown or of the wrong shape.
        """
        super().__init__()
        self.config = _checked_config(mesh_device, config)
        stored = sorted(key for key in state_dict if key.startswith(CHECKPOINT_PREFIXES))
        if stored:
            raise ValueError(
                f"state_dict has {len(stored)} weights under the checkpoint's own names, such as {stored[0]}: read"
                " them with load_vision_state_dict(weights_path), which renames them"
            )
        self.mesh_device = mesh_device
        self.out_dim = self.config.text_hidden_size
        self.model = M3VLVisionModel.from_state_dict(
            mesh_device, state_dict, self.config, with_projector=True, dtype=ttnn.bfloat16
        )

    @classmethod
    def from_weights_path(cls, mesh_device, weights_path: str, config=None) -> "TtMiniMaxM3VisionEncoder":
        """Build the encoder from the MiniMax-M3 checkpoint in weights_path.

        Reads only the shards that hold vision weights, as bf16. Checks the config and the mesh before reading any
        weight.

        Args:
            config: as for __init__. None reads the checkpoint's config.json.

        Raises:
            ValueError: for an empty weights_path, and as __init__ does.
        """
        if not weights_path:
            raise ValueError(f"weights_path must be a MiniMax-M3 checkpoint directory, got {weights_path!r}")
        if config is None:
            with open(os.path.join(weights_path, "config.json")) as f:
                config = json.load(f)
        config = _checked_config(mesh_device, config)
        return cls(mesh_device, load_vision_state_dict(weights_path, dtype=torch.bfloat16), config)

    def forward(self, pixels: np.ndarray) -> ttnn.Tensor:
        """Encode one resized image into vision tokens on the chip.

        Args:
            pixels: uint8 (height, width, 3), as the media module's prepare returns it (M3Image.pixels).

        Returns:
            The tokens as bf16 [n_tokens, out_dim], row-major in DRAM. The caller frees them. n_tokens is
            (height / 28) * (width / 28), and out_dim is 6144 for MiniMax-M3.

        Raises:
            TypeError: before any device work, for pixels that are not a uint8 numpy array.
            ValueError: before any device work, for pixels that are not a resized image (check_pixels).
        """
        pixel_values, grid_thw = patch_rows(pixels)
        # TODO: pad the patch rows to a few fixed patch counts and mask the padded keys in SDPA, so that a rare patch
        # count does not compile new programs. Pad whole tokens (2 x 2 patches each) and drop the padded tokens from
        # the output.
        tokens = self.model(pixel_values, grid_thw)  # (1, 1, n_tokens, out_dim) in tiles
        rows = ttnn.to_layout(tokens, ttnn.ROW_MAJOR_LAYOUT)  # untilized on the chip, without the tiles' padding
        ttnn.deallocate(tokens)
        return ttnn.reshape(rows, (num_tokens(grid_thw), self.out_dim))


# The numpy dtypes that hold bf16 values bit for bit: their raw 16 bits, or ml_dtypes' bfloat16.
_BF16_HOLDERS = ("uint16", "int16", "bfloat16")


def read_embeddings(tokens: ttnn.Tensor, out: np.ndarray) -> None:
    """Copy forward's output from the chip into out.

    It releases Python's GIL while it waits for the chip.

    Args:
        tokens: forward's output, a 2D bf16 tensor.
        out: a writable, C-contiguous numpy array of the same shape that holds bf16 bits: uint16, int16 or ml_dtypes'
            bfloat16. For example, format.embeddings of a stored value in shared memory.

    Raises:
        TypeError: before reading, for an out that is not a numpy array of one of those dtypes.
        ValueError: before reading, for tokens that are not 2D bf16, or for an out of another shape, not C-contiguous
            or read-only.
    """
    shape = tuple(tokens.shape)
    if len(shape) != 2 or tokens.dtype != ttnn.bfloat16:
        raise ValueError(f"tokens must be forward's 2D bf16 output, got {tokens.dtype} {shape}")
    if not isinstance(out, np.ndarray) or out.dtype.name not in _BF16_HOLDERS:
        raise TypeError(
            f"out must be a numpy array of {', '.join(_BF16_HOLDERS)} that holds bf16 values,"
            f" got {getattr(out, 'dtype', type(out).__name__)}"
        )
    if out.shape != shape or not (out.flags.c_contiguous and out.flags.writeable):
        raise ValueError(
            f"out must be a writable C-contiguous array of shape {shape}, got {out.shape},"
            f" C-contiguous {out.flags.c_contiguous}, writable {out.flags.writeable}"
        )
    (chip_tokens,) = ttnn.get_device_tensors(tokens)
    # TODO: read the tokens straight into out: copy_device_to_host_tensor into a host tensor that borrows out's memory.
    # That saves about 1.2 ms per 672x672 image on a P150, but the watcher's NOC check then aborts the run.
    rows = chip_tokens.cpu().to_torch()
    np.copyto(out.view(np.uint16), rows.view(torch.uint16).numpy())
