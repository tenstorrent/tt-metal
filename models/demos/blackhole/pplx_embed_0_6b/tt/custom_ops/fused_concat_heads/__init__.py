# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from models.demos.blackhole.pplx_embed_0_6b.tt.custom_ops.fused_concat_heads.op import (
    nlp_concat_heads_headsplit,
    supported,
)

__all__ = ["nlp_concat_heads_headsplit", "supported"]
