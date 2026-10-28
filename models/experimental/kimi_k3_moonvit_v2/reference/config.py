# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Kimi K3 Vision Configuration (MoonViT2 vision tower only).

Single source of truth for vision-tower dimension constants. Values from HuggingFace config.json
for Kimi-K3 (``vision_config``); the text tower lives in
``models/demos/deepseek_v3_d_p/reference/kimi_k3_config.py``.

Input contract: images only, no video. Every image is padded to ``MAX_PATCHES`` patches and the
patch sequence is split across ``SP_DEVICES`` chips, so each chip holds
``[1, 1, MAX_PATCHES / SP_DEVICES, VT_HIDDEN_SIZE]``; the leading ``1, 1`` are only the 4D format.
Padding rows are computed like any other row and masked or discarded downstream.
"""


class KimiK3VisionConfig:
    """Kimi K3 vision-tower dimensions."""

    # Core dimensions (HF: vt_hidden_size / vt_intermediate_size)
    VT_HIDDEN_SIZE = 1024
    VT_INTERMEDIATE_SIZE = 4096

    # FFN (HF: mlp_type="mlp2", activation_func="gelu_pytorch_tanh", linear_bias=False)
    FFN_BIAS = False

    # Input layout
    MAX_PATCHES = 65536
    SP_DEVICES = 4
    PATCHES_PER_DEVICE = MAX_PATCHES // SP_DEVICES  # 16384
    INPUT_SHAPE = (1, 1, PATCHES_PER_DEVICE, VT_HIDDEN_SIZE)
