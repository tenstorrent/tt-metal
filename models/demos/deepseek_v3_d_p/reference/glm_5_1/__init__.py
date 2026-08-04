# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.1 CPU reference helpers (block-level composition over cpu_deepseek_v32 + torch FFN)."""

from models.demos.deepseek_v3_d_p.reference.glm_5_1.block import (
    dense_ffn,
    glm_decoder_layer_reference,
    rms_norm,
)
from models.demos.deepseek_v3_d_p.reference.glm_5_1.moe import glm_moe_reference

__all__ = ["glm_decoder_layer_reference", "rms_norm", "dense_ffn", "glm_moe_reference"]
