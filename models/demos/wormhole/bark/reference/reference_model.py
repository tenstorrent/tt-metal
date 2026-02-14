# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Reference PyTorch implementation of Bark Small.

Used for accuracy comparison against the TTNN implementation.
"""

import torch
import numpy as np
from loguru import logger


def generate_reference_audio(text: str, model_name: str = "suno/bark-small") -> np.ndarray:
    """
    Generate reference audio using the HuggingFace Bark model on CPU.

    Args:
        text: Input text string
        model_name: HuggingFace model name

    Returns:
        audio: numpy array at 24 kHz
    """
    from transformers import AutoProcessor, BarkModel

    processor = AutoProcessor.from_pretrained(model_name)
    model = BarkModel.from_pretrained(model_name)

    inputs = processor(text=[text], return_tensors="pt")

    with torch.no_grad():
        speech_values = model.generate(**inputs, do_sample=True)

    audio = speech_values.cpu().numpy().squeeze()
    return audio


def compare_logits(tt_logits: torch.Tensor, ref_logits: torch.Tensor) -> dict:
    """
    Compare TT model logits against reference PyTorch logits.

    Returns dict with PCC, max absolute error, etc.
    """
    # Ensure same shape
    assert tt_logits.shape == ref_logits.shape, (
        f"Shape mismatch: TT={tt_logits.shape}, Ref={ref_logits.shape}"
    )

    tt_flat = tt_logits.flatten().float()
    ref_flat = ref_logits.flatten().float()

    # Pearson Correlation Coefficient
    pcc = torch.corrcoef(torch.stack([tt_flat, ref_flat]))[0, 1].item()

    # Max absolute error
    max_abs_err = (tt_flat - ref_flat).abs().max().item()

    # Mean absolute error
    mean_abs_err = (tt_flat - ref_flat).abs().mean().item()

    # Cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(
        tt_flat.unsqueeze(0), ref_flat.unsqueeze(0)
    ).item()

    metrics = {
        "pcc": pcc,
        "max_abs_error": max_abs_err,
        "mean_abs_error": mean_abs_err,
        "cosine_similarity": cos_sim,
    }

    logger.info(
        f"Accuracy: PCC={pcc:.4f}, CosSim={cos_sim:.4f}, "
        f"MaxErr={max_abs_err:.4f}, MeanErr={mean_abs_err:.4f}"
    )

    return metrics
