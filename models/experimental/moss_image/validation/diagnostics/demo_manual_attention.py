# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic wrapper: unmodified demo.py generation with the FP32-score attention replacement.

Same CLI as demo.py. Text-encoder causal attention keeps the native kernel.
"""
import ttnn

from models.experimental.moss_image.validation.diagnostics.dit_trajectory import install_manual_attention

install_manual_attention(ttnn, 6)
from models.experimental.moss_image import demo  # noqa: E402

if __name__ == "__main__":
    demo.main()
