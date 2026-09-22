# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared scaffolding for the per-module PCC tests.

Each module test loads the real weights for one submodule, drives the TTNN module and the
reference module with the same input, and compares. These helpers are the parts that would
otherwise be restated in nine files.

Activations are seeded randn. That is representative here rather than a convenience: the model
is post-norm, so every block input has been through a layer norm and is close to zero-mean
unit-variance. Weight operands are always the real checkpoint, because a bfloat16 matmul's error
tracks the operand distribution.
"""

from __future__ import annotations

import torch

import ttnn

from models.experimental.nomic_embed_text_v2_moe.common import slice_state_dict

# (batch, seqlen). 37 is deliberately off-tile: padding bugs only surface when S is not a
# multiple of 32, and S is the batch's longest tokenized sequence, so that is the common case.
TOKEN_SHAPES = [(1, 128), (2, 512), (2, 37)]

# A wrong layout or convention does not lose precision, it decorrelates. Anything under this is
# the wrong tensor, not a less precise one.
DECORRELATED_PCC = 0.5

# Layer 0 is dense, layer 1 is the first MoE layer. config.is_moe_layer owns the predicate; these
# are the two prefixes the module tests read weights from.
DENSE_LAYER = 0
MOE_LAYER = 1


def load_reference(factory, state_dict: dict, prefix: str):
    """Build a reference submodule and load its slice of the real checkpoint.

    strict=True is the point: the reference's parameter names mirror upstream exactly, so a
    structural mismatch raises here rather than leaving a randomly initialized tensor in place.

    Args:
        factory: Zero-argument callable returning the reference module.
        state_dict: Full checkpoint.
        prefix: Dotted prefix of this submodule, trailing dot included.

    Returns:
        torch.nn.Module: In eval mode, holding the checkpoint's weights.
    """
    module = factory()
    module.load_state_dict(slice_state_dict(state_dict, prefix), strict=True)
    return module.eval()


def hidden_states(batch: int, seqlen: int, hidden: int) -> torch.Tensor:
    """Representative (B, S, H) block input. See the module docstring on why randn."""
    return torch.randn(batch, seqlen, hidden)


def to_block_layout(x: torch.Tensor) -> torch.Tensor:
    """(B, S, H) -> (B, 1, S, H), the layout every TTNN block takes."""
    batch, seqlen, hidden = x.shape
    return x.reshape(batch, 1, seqlen, hidden)


def from_block_layout(x: ttnn.Tensor) -> torch.Tensor:
    """(B, 1, S, H) device tensor -> (B, S, H) fp32 torch, ready to compare."""
    out = ttnn.to_torch(x).float()
    batch, _, seqlen, hidden = out.shape
    return out.reshape(batch, seqlen, hidden)


def keep_mask(batch: int, seqlen: int, keep) -> torch.Tensor:
    """(B, S) keep-mask with the trailing positions padded out.

    Args:
        batch: Rows.
        seqlen: Sequence length.
        keep: Number of leading real tokens, either one count for every row or one per row. A
            ragged mask is what catches a pooling divisor that counts padding, since a uniform
            keep count makes that error a pure scale, which PCC cannot see.

    Returns:
        torch.Tensor: (B, S) int64, 1 for real tokens and 0 for padding.
    """
    counts = [keep] * batch if isinstance(keep, int) else list(keep)
    assert len(counts) == batch, f"{len(counts)} keep counts for {batch} rows"
    mask = torch.ones(batch, seqlen, dtype=torch.long)
    for row, count in enumerate(counts):
        mask[row, count:] = 0
    return mask


def dense_routing(tokens: int, experts: int, top_k: int) -> torch.Tensor:
    """A plausible (T, E) dense routing tensor, zero off the top-k and not renormalized.

    Injected rather than routed so the expert tests measure the expert arithmetic alone; a
    routing flip would otherwise show up as an expert error.
    """
    probabilities = torch.randn(tokens, experts).softmax(dim=-1)
    values, indices = torch.topk(probabilities, top_k, dim=-1)
    return torch.zeros_like(probabilities).scatter_(-1, indices, values)
