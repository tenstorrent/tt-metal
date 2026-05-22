# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Phase 3 end-to-end PCC tests for dots.ocr text + vision paths.

These tests stitch the random-weights ``TTNNModule`` components into a
single forward pass (embedding -> decoder -> norm -> lm_head or
vision_tower full pipeline) and assert PCC against a PyTorch reference at
a looser threshold to account for compounded error.
"""
