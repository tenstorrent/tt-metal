# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Pearson correlation (PCC) between two tensors of the same **logical** shape.

Used across Dots OCR to compare **TT** module outputs to the **HF Dots** reference
(:class:`~models.demos.dots_ocr.reference.model.DotsOCRReference` and helpers): text prefill
logits, vision tower rows, patch merger activations, fused embeddings, etc.

This is a single implementation so all PCC gates stay consistent. Prefer
:func:`comp_pcc` over ad-hoc ``corrcoef`` / flattened comparisons.
"""

from __future__ import annotations

import torch


def comp_pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Pearson correlation coefficient between ``a`` and ``b`` (element-wise after flattening).

    Same as cosine similarity of mean-centered, flattened vectors (up to numerical details).
    """
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).clamp(min=1e-12)
    return float((a @ b) / denom)
