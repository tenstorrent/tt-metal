# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""VAE decoder weight loading from PyTorch reference checkpoints."""

from __future__ import annotations

import torch
import ttnn

from models.experimental.hunyuan_image_3_0.ref.vae.decoder import (
    load_conv_in,
    load_decoder_tail,
    load_decoder_up,
    load_mid,
)

_REF_DTYPE = torch.float32


def _load_state(module, state_dict) -> None:
    module.load_torch_state_dict(state_dict)


def _load_gn(module, state_dict) -> None:
    """Load a GroupNorm3D and stash its raw per-channel affine ([1,1,1,C] fp32,
    replicated across the mesh) so the distributed (spatially-sharded) group_norm can
    apply the affine without the packed ttnn.group_norm weights. See spatial.py."""
    module.load_torch_state_dict(state_dict)
    C = module.num_channels
    dev = module.mesh_device
    w = state_dict.get("weight")
    b = state_dict.get("bias")
    gamma = w.reshape(1, 1, 1, C).float() if w is not None else torch.ones(1, 1, 1, C)
    beta = b.reshape(1, 1, 1, C).float() if b is not None else torch.zeros(1, 1, 1, C)
    module._raw_gamma = ttnn.from_torch(
        gamma,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=dev,
        mesh_mapper=ttnn.ReplicateTensorToMesh(dev),
    )
    module._raw_beta = ttnn.from_torch(
        beta,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=dev,
        mesh_mapper=ttnn.ReplicateTensorToMesh(dev),
    )


def init_conv_in(module) -> None:
    ref = load_conv_in(dtype=_REF_DTYPE)
    _load_state(module.conv, ref.conv.state_dict())
    del ref


def load_resnet_block(module, ref_block) -> None:
    _load_gn(module.norm1, ref_block.norm1.state_dict())
    _load_gn(module.norm2, ref_block.norm2.state_dict())
    _load_state(module.conv1, ref_block.conv1.state_dict())
    _load_state(module.conv2, ref_block.conv2.state_dict())
    if module.nin_shortcut is not None:
        _load_state(module.nin_shortcut, ref_block.nin_shortcut.state_dict())


def load_attn_block(module, ref_block) -> None:
    _load_state(module.norm, ref_block.norm.state_dict())
    _load_state(module.q, ref_block.q.state_dict())
    _load_state(module.k, ref_block.k.state_dict())
    _load_state(module.v, ref_block.v.state_dict())
    _load_state(module.proj_out, ref_block.proj_out.state_dict())


def init_mid_block(module) -> None:
    ref = load_mid(dtype=_REF_DTYPE)
    load_resnet_block(module.block_1, ref.block_1)
    load_attn_block(module.attn_1, ref.attn_1)
    load_resnet_block(module.block_2, ref.block_2)
    del ref


def load_upsample(module, ref_upsample) -> None:
    _load_state(module.conv, ref_upsample.conv.state_dict())


def load_up_block(module, ref_block) -> None:
    for tt_block, pt_block in zip(module.blocks, ref_block.block):
        load_resnet_block(tt_block, pt_block)
    if module.upsample is not None:
        load_upsample(module.upsample, ref_block.upsample)


def load_norm_out(module, ref_norm_out) -> None:
    _load_gn(module.norm, ref_norm_out.norm.state_dict())


def load_conv_out(module, ref_conv_out) -> None:
    _load_state(module.conv, ref_conv_out.conv.state_dict())


def init_decoder_tail(module) -> None:
    ref = load_decoder_tail(dtype=_REF_DTYPE)
    load_norm_out(module.norm_out, ref.norm_out)
    load_conv_out(module.conv_out, ref.conv_out)
    del ref


def init_decoder_up(module) -> None:
    ref = load_decoder_up(dtype=_REF_DTYPE)
    for tt_block, pt_block in zip(module.up_blocks, ref.up):
        load_up_block(tt_block, pt_block)
    del ref
