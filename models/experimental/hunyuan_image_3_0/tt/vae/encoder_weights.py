# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""VAE encoder weight loading from PyTorch reference checkpoints."""

from __future__ import annotations

import torch

from models.experimental.hunyuan_image_3_0.ref.vae.encoder import (
    load_conv_in as load_encoder_conv_in,
    load_encoder_down,
    load_encoder_head,
    load_mid as load_encoder_mid,
)
from models.experimental.hunyuan_image_3_0.tt.vae.decoder_weights import (
    load_attn_block,
    load_resnet_block,
)

_REF_DTYPE = torch.float32


def _load_state(module, state_dict) -> None:
    module.load_torch_state_dict(state_dict)


def init_encoder_conv_in(module) -> None:
    ref = load_encoder_conv_in(dtype=_REF_DTYPE)
    _load_state(module.conv, ref.conv.state_dict())
    del ref


def load_downsample(module, ref_downsample) -> None:
    _load_state(module.conv, ref_downsample.conv.state_dict())


def load_down_block(module, ref_block) -> None:
    for tt_block, pt_block in zip(module.blocks, ref_block.block):
        load_resnet_block(tt_block, pt_block)
    if module.downsample is not None:
        load_downsample(module.downsample, ref_block.downsample)


def init_encoder_down(module) -> None:
    ref = load_encoder_down(dtype=_REF_DTYPE)
    for tt_block, pt_block in zip(module.down_blocks, ref.down):
        load_down_block(tt_block, pt_block)
    del ref


def init_encoder_mid(module) -> None:
    ref = load_encoder_mid(dtype=_REF_DTYPE)
    load_resnet_block(module.block_1, ref.block_1)
    load_attn_block(module.attn_1, ref.attn_1)
    load_resnet_block(module.block_2, ref.block_2)
    del ref


def init_encoder_head(module) -> None:
    ref = load_encoder_head(dtype=_REF_DTYPE)
    _load_state(module.norm_out, ref.norm_out.state_dict())
    _load_state(module.conv_out, ref.conv_out.state_dict())
    del ref
