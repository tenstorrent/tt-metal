# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
The MiniMax-M3 vision encoder on one Tenstorrent chip, in the form the encoder service's media module calls it:

    encoder = TtMiniMaxM3VisionEncoder.from_weights_path(mesh_device, weights_path, hf_config)  # load(), once
    tokens = encoder.forward(image.pixels)  # encode(), per image: image as the media module's prepare gives it
    try:
        read_embeddings(tokens, out)  # out: [n_tokens, 6144] bf16 values, for example in shared memory
    finally:
        ttnn.deallocate(tokens)

This module imports ttnn, so import it only where the chip is used, for example inside the media module's load(), and
never in a process that must run without ttnn; the media module's prepare, which resizes images, does not import ttnn.

forward takes one resized image, uint8 (height, width, 3) with sides that are multiples of 28 pixels, as the media
module's prepare returns it (M3Image.pixels). It turns the pixels into patch rows on the host (patch_rows, which
checks them first), then runs the patch embedding, the vision blocks, the 2 x 2 merge and the projector to the
text model's hidden size on the chip: M3VLVisionModel of tt/vision/model.py, so no model code is written here. It
returns the image's vision tokens on the chip, bf16 [n_tokens, 6144] in row-major layout in DRAM, untilized on the
chip, so they read back as contiguous rows with no tile padding to strip and no untilize on the host. n_tokens is
grid_h * grid_w = height / 28 * width / 28.

Compilation: the chip's programs depend on the image's patch count (4 patches per token), not on the shape of its
grid, which the tower test checks for the tower and test_vision_encoder_load_and_encode for the whole encoder (both
need a device). The first image of each patch count builds them, which took 15-42 s with an empty kernel cache and
0.1-1.4 s with the kernels already in the kernel cache on disk, both measured on an older version of this encoder;
later images of that count reuse them. The resize gives 534 patch counts for images of up to 8192 pixels a side, so
a warm-up in load() can only cover chosen counts, for example 576 tokens (672 x 672 pixels, and 1920 x 1080 photos,
resized to 896 x 504) and 540 tokens (4032 x 3024 photos, resized to 756 x 560). An image of another count compiles
on its first call and holds that encoder meanwhile.

Traces: forward is not ready for trace capture. Each call uploads its pixel values to a new device buffer and may
stage the RoPE tables of a new grid (M3VLVisionModel keeps those of 16 grids and frees the least recently used), so
a replay would read the buffers of the captured call. A trace per patch count needs its input and RoPE tables in
buffers that stay allocated and are written before each replay; that is not done here. The RoPE tables are also device
memory that forward allocates after load() and keeps after the call. tt-metal warns that a buffer allocated after a
trace was captured on the device, and still allocated when the trace runs, may be overwritten by that run, so on a
device where any trace was captured, its own or another model's, these tables are at risk.

One encoder runs on one chip and is not thread safe: one call at a time. read_embeddings releases Python's GIL
while it waits for the chip, so a service that shares one encoder between threads needs a lock.
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
    """The vision config of config: a MiniMaxM3VisionConfig as it is; a dict, the checkpoint's config.json or a
    transformers config's to_dict(); or a transformers config (anything with to_dict()), such as MiniMaxM3VLConfig or
    the config an encoder service builds from the checkpoint. Raises TypeError for anything else, and ValueError for a
    dict or a transformers config without a field that MiniMaxM3VisionConfig.from_hf_config reads, such as the vision
    part of the config on its own."""
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
    """vision_config(config), after checking that its patches are the image code's and that mesh_device is one chip."""
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
        """Put the vision weights on mesh_device, a mesh of one chip (ValueError otherwise; take a 1x1 submesh of a
        larger mesh with mesh_device.create_submesh(ttnn.MeshShape(1, 1))).

        state_dict: the vision tower and projector weights under transformers 5.12's MiniMaxM3VLModel names
        (vision_tower.*, multi_modal_projector.*), as load_vision_state_dict(weights_path) returns them for the
        checkpoint, which stores them under older names, in any float dtype: the chip holds them in bf16. Other keys, a
        language model's for example, and those of the layers past the config's num_hidden_layers are ignored and their
        values never read, so state_dict may be a lazy mapping over a whole checkpoint. Weights under the checkpoint's
        own names raise ValueError, and so do a missing weight, one of another shape than the config's and a vision
        weight that the model has no place for (M3VLVisionModel.from_state_dict), before any upload.
        config: see vision_config; the service's Hugging Face config of the checkpoint will do. Its patch geometry must
        be the image code's (ValueError otherwise).
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
        """The encoder with the vision weights of the MiniMax-M3 checkpoint in weights_path: only the safetensors shards
        that its model.safetensors.index.json lists for the vision tower and projector are read, as bf16, the dtype
        the chip holds them in. config: as for __init__; None reads the checkpoint's config.json. The config and the
        mesh are checked before any weight is read."""
        if not weights_path:
            raise ValueError(f"weights_path must be a MiniMax-M3 checkpoint directory, got {weights_path!r}")
        if config is None:
            with open(os.path.join(weights_path, "config.json")) as f:
                config = json.load(f)
        config = _checked_config(mesh_device, config)
        return cls(mesh_device, load_vision_state_dict(weights_path, dtype=torch.bfloat16), config)

    def forward(self, pixels: np.ndarray) -> ttnn.Tensor:
        """The vision tokens of one resized image on the chip: bf16 [n_tokens, out_dim] (out_dim 6144 for MiniMax-M3),
        row-major, in DRAM. The caller frees it.

        pixels: uint8 (height, width, 3), as the media module's prepare returns it (M3Image.pixels). patch_rows raises
        TypeError or ValueError for pixels that no image resizes to (check_pixels) before any device work.
        """
        pixel_values, grid_thw = patch_rows(pixels)
        # TODO: pad the patch rows to a few patch-count buckets and mask the padded keys in SDPA, so that the first
        # image of an uncommon patch count does not compile the chip's programs while other images wait (see
        # Compilation in the module docstring). Pad whole tokens, for the 2 x 2 merge, and drop the padded ones.
        tokens = self.model(pixel_values, grid_thw)  # (1, 1, n_tokens, out_dim) in tiles
        rows = ttnn.to_layout(tokens, ttnn.ROW_MAJOR_LAYOUT)  # untilized on the chip, without the tiles' padding
        ttnn.deallocate(tokens)
        return ttnn.reshape(rows, (num_tokens(grid_thw), self.out_dim))


# The numpy dtypes that hold bf16 values bit for bit: their raw 16 bits, or ml_dtypes' bfloat16.
_BF16_HOLDERS = ("uint16", "int16", "bfloat16")


def read_embeddings(tokens: ttnn.Tensor, out: np.ndarray) -> None:
    """Copy forward's output into out, for example format's embeddings view of a value in shared memory.

    out: a writable C-contiguous numpy array of tokens' shape [n_tokens, out_dim] that holds bf16 values: dtype uint16
    or int16 (the raw bits) or ml_dtypes' bfloat16. Raises TypeError or ValueError for another out or for tokens that
    are not a 2D bf16 tensor, before reading. Reads the tokens from the chip with Tensor.cpu, which releases Python's
    GIL while it waits for the chip to finish, then copies them into out once (ttnn's to_torch makes one more host
    copy before that).
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
    rows = chip_tokens.cpu().to_torch()
    np.copyto(out.view(np.uint16), rows.view(torch.uint16).numpy())
