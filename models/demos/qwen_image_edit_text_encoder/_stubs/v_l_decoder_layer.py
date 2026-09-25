# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native, tensor-parallel TTNN port of the Qwen2.5-VL text decoder layer (`Qwen2_5_VLDecoderLayer`).

Same module as `layer`: GQA attention split by KV group + SwiGLU MLP, column/row-parallel with
all_reduce after each row-parallel projection; fp32 residual stream. See layer.py.
"""

from __future__ import annotations

from models.demos.qwen_image_edit_text_encoder._stubs.layer import TtTextDecoderLayer


def build(device, torch_module=None, mlp=None, pair=None):
    return TtTextDecoderLayer(device, torch_module, mlp=mlp, pair=pair)


def v_l_decoder_layer(device, torch_module=None):
    return build(device, torch_module)
