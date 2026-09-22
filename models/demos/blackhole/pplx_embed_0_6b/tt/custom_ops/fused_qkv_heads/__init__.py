# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from models.demos.blackhole.pplx_embed_0_6b.tt.custom_ops.fused_qkv_heads.op import (
    nlp_create_qkv_heads_headsplit,
    supported,
)

__all__ = ["nlp_create_qkv_heads_headsplit", "supported"]
