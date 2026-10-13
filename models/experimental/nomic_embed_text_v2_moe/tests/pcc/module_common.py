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

from models.common.metrics import compute_max_abs_error, compute_pcc
from models.experimental.nomic_embed_text_v2_moe.common import slice_state_dict

# (batch, seqlen). 37 is deliberately off-tile: padding bugs only surface when S is not a
# multiple of 32, and S is the batch's longest tokenized sequence, so that is the common case.
TOKEN_SHAPES = [(1, 128), (2, 512), (2, 37)]

# Past 32 tile rows of M = B * S the dense projections run through minimal_matmul, which no
# TOKEN_SHAPES entry reaches. 4x512 folds to 64: M is below N for QKV and fc1 and above it for
# out_proj and fc2, so both orientations run, and the QKV, out_proj and fc1 outputs go to L1.
DENSE_SHAPES = [*TOKEN_SHAPES, (4, 512)]

# A wrong layout or convention does not lose precision, it decorrelates. Anything under this is
# the wrong tensor, not a less precise one.
DECORRELATED_PCC = 0.5

# Layer 0 is dense, layer 1 is the first MoE layer. config.is_moe_layer owns the predicate; these
# are the two prefixes the module tests read weights from.
DENSE_LAYER = 0
MOE_LAYER = 1

# The shared-bias offset the wrong placement leaves behind. Measured well above this; PCC cannot
# see it at all.
BIAS_MISPLACEMENT_MAX_ABS = 1e-3

# How far from 1 the measured count of shared biases a token holds may be. Measured 1.000 and 1.007
# where a stacked pass adds the bias in w2's fp32 accumulator, and 0.946 and 0.950 where a transposed
# pass adds it to the bfloat16 sum in bfloat16, which drops it from the half of the elements where
# it is under half a step of the sum. A bias added inside the per-expert loop counts the mean
# routed-weight sum instead, about 0.55.
BIAS_SCALE_TOLERANCE = 0.1


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


def bias_scale(got: torch.Tensor, without_bias: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """How many times the output holds the bias: its difference from the bias-free run, projected on it.

    1 where the bias is added once to each token's sum, the mean routed-weight sum where it is added
    inside every expert's term. Projected over every token, the rounding the two runs do differently,
    up to a bfloat16 step when the bias is added inside a matmul's accumulator, averages out.
    """
    difference = (got - without_bias).double().reshape(-1, bias.shape[-1])
    direction = bias.double().reshape(1, -1)
    return (difference * direction).sum(-1).mean() / (direction * direction).sum()


def assert_bias_added_once(
    got: torch.Tensor, without_bias: torch.Tensor, bias: torch.Tensor, routed_weight_sum: torch.Tensor
) -> None:
    """The shared expert bias is in the output once per token, not once per routed expert.

    Both placements are built from the module's own bias-free output rather than the reference: at
    real weights the bfloat16 noise is 0.22 against an offset of (sum(w) - 1) * bias far below it.

    Args:
        got: The module output.
        without_bias: The same module's output with its bias zeroed.
        bias: The (H,) shared bias.
        routed_weight_sum: Each token's sum of routing weights, broadcastable against got.
    """
    assert (routed_weight_sum < 1.0).all(), "top-k weights are not renormalized, so they must sum below 1"
    correct = without_bias + bias
    inside_the_loop = without_bias + bias * routed_weight_sum
    assert (
        compute_max_abs_error(correct, inside_the_loop) > BIAS_MISPLACEMENT_MAX_ABS
    ), "the two placements agreed; the offset this test exists to catch is not present"
    assert compute_pcc(inside_the_loop, correct) > 0.9999, "if PCC caught this, the docstring is stale"
    inside_scale = float(routed_weight_sum.mean())
    assert 1 - inside_scale > BIAS_SCALE_TOLERANCE, "a bias added inside the loop would pass"
    scale = float(bias_scale(got, without_bias, bias))
    assert (
        abs(scale - 1) < BIAS_SCALE_TOLERANCE
    ), f"the output holds the bias {scale:.3f} times: once is 1, inside the loop {inside_scale:.3f}"
