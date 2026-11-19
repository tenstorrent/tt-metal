# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Explicit shared-harness selection; no runtime fallback."""
import os


def select_decoder():
    name = os.environ.get("QWEN_DECODER", "functional")
    if name == "optimized":
        from models.autoports.qwen_qwen3_6_35b_a3b.tt.optimized_decoder import OptimizedDecoder

        return OptimizedDecoder
    if name == "fused":
        from models.autoports.qwen_qwen3_6_35b_a3b.tt.fused_decoder import FusedDecoder

        return FusedDecoder
    if name == "functional":
        from models.autoports.qwen_qwen3_6_35b_a3b.tt.functional_decoder import FunctionalDecoder

        return FunctionalDecoder
    raise ValueError("QWEN_DECODER must be functional, fused, or optimized")


Decoder = select_decoder()
