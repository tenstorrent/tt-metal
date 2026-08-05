# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTNN Swin-L backbone for ATSS.

Re-exports the standalone Swin-L TTNN implementation from
models/experimental/swin_l/ and provides an ATSS-specific factory
that configures the backbone with ATSS constants (out_indices, input
resolution, attention masks).
"""

import ttnn

from models.experimental.swin_l.tt.tt_backbone import TtSwinLBackbone
from models.experimental.swin_l.tt.model_preprocessing import (
    load_backbone_weights,
    compute_attn_masks,
)
from models.experimental.atss_swin_l_dyhead.common import (
    ATSS_INPUT_H,
    ATSS_INPUT_W,
    ATSS_OUT_INDICES,
)


# ATSS-specific precision promotion for the Swin-L backbone. Stage 2 (18 blocks,
# dim=768) accumulates the most LoFi+bf8 error and is the dominant contributor to the
# centerness level-0 PCC drop, so we promote its MLP linears to HiFi2/bf16 here. Attention
# promotion is available (ATSS_HIGH_PRECISION_ATTN_STAGES) but left off unless needed.
ATSS_HIGH_PRECISION_MLP_STAGES = (2,)
ATSS_HIGH_PRECISION_ATTN_STAGES = ()


def build_atss_backbone(
    checkpoint_path: str,
    device: ttnn.Device,
    input_h: int = None,
    input_w: int = None,
    high_precision_mlp_stages: tuple = ATSS_HIGH_PRECISION_MLP_STAGES,
    high_precision_attn_stages: tuple = ATSS_HIGH_PRECISION_ATTN_STAGES,
) -> TtSwinLBackbone:
    """Build a Swin-L backbone configured for ATSS.

    Args:
        checkpoint_path: path to the mmdet .pth checkpoint.
        device: TTNN device.
        input_h: padded input height (default: ATSS_INPUT_H).
        input_w: padded input width  (default: ATSS_INPUT_W).
        high_precision_mlp_stages: stages whose MLP linears run at HiFi2/bf16.
        high_precision_attn_stages: stages whose attention matmuls run at HiFi2/bf16.

    Returns:
        TtSwinLBackbone with loaded weights, attention masks, and
        out_indices=(1, 2, 3) producing 3 feature maps at
        [384, 768, 1536] channels.
    """
    h = input_h or ATSS_INPUT_H
    w = input_w or ATSS_INPUT_W

    params = load_backbone_weights(
        checkpoint_path,
        device,
        out_indices=ATSS_OUT_INDICES,
        high_precision_attn_stages=high_precision_attn_stages,
        high_precision_mlp_stages=high_precision_mlp_stages,
    )
    attn_masks = compute_attn_masks(h, w, patch_size=4, window_size=12, device=device)

    return TtSwinLBackbone(
        device,
        params,
        attn_masks=attn_masks,
        out_indices=ATSS_OUT_INDICES,
        high_precision_attn_stages=high_precision_attn_stages,
        high_precision_mlp_stages=high_precision_mlp_stages,
    )
