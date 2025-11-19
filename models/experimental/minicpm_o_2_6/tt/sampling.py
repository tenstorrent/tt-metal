# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
from loguru import logger


def sample_greedy(logits: torch.Tensor) -> torch.Tensor:
    """
    Greedy sampling (argmax).

    Args:
        logits: Logits tensor [batch, vocab_size] or [vocab_size]

    Returns:
        Selected token indices [batch, 1] or [1]
    """
    next_token = torch.argmax(logits, dim=-1)

    if next_token.dim() == 1:  # if sampling a single token re-add the batch dim
        next_token = next_token.unsqueeze(0)

    return next_token


def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    """
    Nucleus (top-p) sampling.

    Adapted from sample_top_p in models/tt_transformers/tt/common.py:515-525

    Args:
        probs: Probability distribution [batch, vocab_size] or [vocab_size]
        p: Nucleus probability mass (0 < p <= 1)

    Returns:
        Selected token indices [batch, 1] or [1]
    """
    assert 0 <= p <= 1, f"Top-p value {p} must be in range [0, 1]"

    # Sort probabilities in descending order
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)

    # Compute cumulative probabilities
    probs_sum = torch.cumsum(probs_sort, dim=-1)

    # Create mask for tokens to remove (those beyond cumulative probability p)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0

    # Renormalize
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))

    # Sample from the filtered distribution
    next_token = torch.multinomial(probs_sort, num_samples=1)

    # Map back to original token indices
    result = torch.gather(probs_idx, -1, next_token)

    return result


def sample_top_k(probs: torch.Tensor, k: int) -> torch.Tensor:
    """
    Top-k sampling.

    Args:
        probs: Probability distribution [batch, vocab_size] or [vocab_size]
        k: Number of top tokens to consider (k > 0)

    Returns:
        Selected token indices [batch, 1] or [1]
    """
    assert k > 0, f"Top-k value {k} must be positive"

    # Get top-k probabilities and their indices
    top_k_probs, top_k_indices = torch.topk(probs, k, dim=-1)

    # Sample from top-k distribution
    next_token_idx = torch.multinomial(top_k_probs, num_samples=1)

    # Map back to original vocabulary indices
    result = torch.gather(top_k_indices, -1, next_token_idx)

    return result


def sample_with_temperature(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
) -> torch.Tensor:
    """
    Combined sampling with temperature, top-p, and top-k.

    Adapted from sample_host in models/tt_transformers/tt/common.py:528-540

    Args:
        logits: Logits tensor [batch, vocab_size] or [vocab_size]
        temperature: Sampling temperature (> 0)
        top_p: Nucleus sampling probability (0 < p <= 1)
        top_k: Top-k sampling count (> 0)

    Returns:
        Selected token indices [batch, 1] or [1]
    """
    if temperature == 0.0:
        # Greedy sampling
        return sample_greedy(logits)

    if temperature > 0:
        # Apply temperature scaling
        probs = torch.softmax(logits / temperature, dim=-1)

        if top_p is not None and top_k is not None:
            # Apply both top-p and top-k (take intersection)
            logger.warning("Both top_p and top_k specified. Applying top_p first, then top_k.")

            # First apply top-p
            probs = _apply_top_p_mask(probs, top_p)

            # Then apply top-k on the filtered distribution
            probs = _apply_top_k_mask(probs, top_k)

            # Renormalize
            probs.div_(probs.sum(dim=-1, keepdim=True))

            # Sample from the filtered distribution
            next_token = torch.multinomial(probs, num_samples=1)
            return next_token

        elif top_p is not None:
            # Apply top-p sampling
            return sample_top_p(probs, top_p)

        elif top_k is not None:
            # Apply top-k sampling
            return sample_top_k(probs, top_k)

        else:
            # Standard multinomial sampling
            next_token = torch.multinomial(probs, num_samples=1)
            return next_token

    else:
        raise ValueError(f"Temperature {temperature} must be >= 0")


def _apply_top_p_mask(probs: torch.Tensor, p: float) -> torch.Tensor:
    """
    Apply top-p masking to probability distribution.

    Args:
        probs: Probability distribution
        p: Nucleus probability mass

    Returns:
        Masked probability distribution
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    return probs_sort


def _apply_top_k_mask(probs: torch.Tensor, k: int) -> torch.Tensor:
    """
    Apply top-k masking to probability distribution.

    Args:
        probs: Probability distribution
        k: Number of top tokens to keep

    Returns:
        Masked probability distribution
    """
    top_k_probs, _ = torch.topk(probs, k, dim=-1)
    # Create mask for values below top-k threshold
    threshold = top_k_probs[:, -1:]  # Last element of top-k is the threshold
    mask = probs < threshold
    probs = probs.clone()
    probs[mask] = 0.0
    return probs


def sample_host(
    logits: torch.Tensor,
    temperature: float = 0.6,
    top_p: float = 0.08,
    on_host: bool = True,
) -> Tuple[None, torch.Tensor]:
    """
    Host-based sampling function.

    Adapted from sample_host in models/tt_transformers/tt/common.py:528-540

    Args:
        logits: Logits tensor
        temperature: Sampling temperature
        top_p: Top-p probability
        on_host: Whether sampling on host (always True for this implementation)

    Returns:
        Tuple of (None, sampled_tokens)
    """
    vocab_size = logits.shape[-1]
    pt_input = logits[..., :vocab_size]

    if temperature > 0:
        probs = torch.softmax(pt_input / temperature, dim=-1)
        pt_out = sample_top_p(probs.squeeze(), top_p)
    else:
        pt_out = torch.argmax(pt_input, dim=-1)

    if pt_out.dim() == 1:  # if sampling a single token re-add the batch dim
        pt_out = pt_out.unsqueeze(0)

    return None, pt_out


@dataclass
class SamplingParams:
    """
    Sampling parameters for generation.

    Adapted from SamplingParams in models/tt_transformers/tt/generator.py:31-41
    """

    temperature: Union[float, list[float]] = 0.6
    top_k: Union[int, list[int]] = -1  # -1 means no top-k
    top_p: Union[float, list[float]] = 0.9

    def get_temperature(self, batch_idx: int = 0) -> float:
        """Get temperature for batch index."""
        if isinstance(self.temperature, list):
            return self.temperature[batch_idx]
        return self.temperature

    def get_top_k(self, batch_idx: int = 0) -> int:
        """Get top_k for batch index."""
        if isinstance(self.top_k, list):
            return self.top_k[batch_idx]
        return self.top_k

    def get_top_p(self, batch_idx: int = 0) -> float:
        """Get top_p for batch index."""
        if isinstance(self.top_p, list):
            return self.top_p[batch_idx]
        return self.top_p
