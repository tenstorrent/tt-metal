# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
PyTorch reference implementation for Bark PCC comparison.

Runs the HuggingFace Bark model on CPU/GPU and returns intermediate
outputs for Pearson Correlation Coefficient (PCC) validation against
the TTNN implementation.
"""

import torch
from transformers import BarkModel


def load_bark_reference(model_name="suno/bark-small"):
    """Load the HuggingFace Bark model for reference comparison.

    Returns:
        model: HuggingFace BarkModel
    """
    model = BarkModel.from_pretrained(model_name)
    model.eval()
    return model


def run_semantic_forward(model, input_ids):
    """Run a single forward pass through the semantic model.

    Args:
        model: HuggingFace BarkModel
        input_ids: [batch, seq_len] token indices

    Returns:
        logits: [batch, seq_len, vocab_size] output logits
    """
    with torch.no_grad():
        outputs = model.semantic(input_ids=input_ids, return_dict=True)
    return outputs.logits


def run_coarse_forward(model, input_ids):
    """Run a single forward pass through the coarse acoustics model.

    Args:
        model: HuggingFace BarkModel
        input_ids: [batch, seq_len] token indices

    Returns:
        logits: [batch, seq_len, vocab_size] output logits
    """
    with torch.no_grad():
        outputs = model.coarse_acoustics(input_ids=input_ids, return_dict=True)
    return outputs.logits


def run_fine_forward(model, codebook_idx, input_ids):
    """Run a single forward pass through the fine acoustics model.

    Args:
        model: HuggingFace BarkModel
        codebook_idx: Which codebook to predict (2-7)
        input_ids: [batch, seq_len, n_codes_total] all codebook tokens

    Returns:
        logits: [batch, seq_len, vocab_size] output logits
    """
    with torch.no_grad():
        outputs = model.fine_acoustics(codebook_idx=codebook_idx, input_ids=input_ids, return_dict=True)
    return outputs.logits


def compute_pcc(tensor1, tensor2):
    """Compute Pearson Correlation Coefficient between two tensors.

    Args:
        tensor1, tensor2: PyTorch tensors of same shape

    Returns:
        pcc: float, correlation coefficient (1.0 = perfect match)
    """
    t1 = tensor1.float().flatten()
    t2 = tensor2.float().flatten()

    # Remove any NaN/Inf values
    valid = torch.isfinite(t1) & torch.isfinite(t2)
    t1 = t1[valid]
    t2 = t2[valid]

    if len(t1) == 0:
        return 0.0

    t1_centered = t1 - t1.mean()
    t2_centered = t2 - t2.mean()

    numerator = (t1_centered * t2_centered).sum()
    denominator = torch.sqrt((t1_centered**2).sum() * (t2_centered**2).sum())

    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0

    pcc = (numerator / denominator).item()
    return pcc
