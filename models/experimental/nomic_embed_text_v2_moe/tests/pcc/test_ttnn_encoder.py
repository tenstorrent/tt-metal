# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Module PCC for TtNomicBertEncoder, all 12 blocks. Bring-up gate 9.

The input is the reference's own embedding output, not randn, and that choice is load-bearing
rather than cosmetic. Every other module test uses randn because a block's input has been through
a layer norm, so randn matches its statistics. The encoder's input has not: it is the embedding
lookup followed by emb_ln, and feeding randn there is out of domain in a way that changes the
answer.

Measured, at B=2 S=128, comparing the two inputs at the same shapes:

  input               reference absmax    encoder PCC     pooled cosine
  embedding output         11 to 13       0.9937+         0.99950+
  randn                   186 to 330      0.949 to 0.998  0.911 to 0.9998

randn drives activations 25x larger, and at that magnitude the 3072-deep fc2 reduction is
dominated by cancellation: its bfloat16 error scales with the summands, not with the result. One
randn draw put layer 6's dense MLP output at absmax 57.5 against the reference's 27.4, and the
error propagated from there. No token sequence can produce that input, so the encoder is measured
on activations the model can actually see. The per-operator gates in
test_ttnn_operators.py cover fc2 itself.

Two statistics are asserted, because they fail differently. The pooled cosine is what the model
actually emits and is stable to four decimal places across draws. All-token PCC is the looser of
the two on purpose: a handful of rerouted tokens are legitimately different, and with only a few
hundred tokens that moves the aggregate more than it moves the embedding.
"""

import pytest
import torch

import ttnn

from models.common.metrics import compute_pcc
from models.common.utility_functions import run_for_blackhole
from models.experimental.nomic_embed_text_v2_moe.common import capture_hidden_states, random_input_ids
from models.experimental.nomic_embed_text_v2_moe.reference.modeling_nomic_moe import build_extended_attention_mask
from models.experimental.nomic_embed_text_v2_moe.reference.postprocessing import l2_normalize, mean_pool
from models.experimental.nomic_embed_text_v2_moe.tests.pcc.module_common import (
    from_block_layout,
    keep_mask,
    to_block_layout,
)
from models.experimental.nomic_embed_text_v2_moe.tt.common import (
    additive_attention_mask,
    flatten_tokens,
    rotary_tables,
    to_device,
)
from models.experimental.nomic_embed_text_v2_moe.tt.encoder import TtNomicBertEncoder
from tests.ttnn.utils_for_testing import assert_with_pcc

pytestmark = [run_for_blackhole(), pytest.mark.use_module_device, pytest.mark.needs_weights]

# All-token PCC over the full stack. Measured 0.9937 to 0.9983 in domain; the gate keeps the
# plan's margin because the statistic moves with how many tokens happened to reroute.
MODULE_PCC = 0.98

# The pooled, normalized embedding: what the model emits, and the tight gate. Measured 0.99950 to
# 0.99978 across draws and both sequence lengths.
POOLED_COSINE = 0.999

# Minimum per-layer PCC across the ladder. Measured 0.9937 to 0.9981.
LADDER_PCC = 0.99

# Routing disagreement on the first MoE layer, which sees activations that have not yet diverged.
# This is gate 5 applied in situ; measured 0.000 to 0.008.
FIRST_MOE_DISAGREEMENT = 0.01

# Deeper MoE layers, where the port's own divergence has had layers to accumulate and a token
# already rerouted once is effectively routed independently. Measured up to 0.051.
DEEP_MOE_DISAGREEMENT = 0.06

STACK_SHAPES = [(2, 128), (2, 512)]

PREFIX = "encoder."


@pytest.fixture
def reference(reference_model):
    """The encoder of the loaded reference model, which also supplies the embeddings."""
    return reference_model.encoder


def encoder_input(reference_model, config, batch, seqlen, seed=0):
    """(B, S, H) encoder input, built the way the model builds it: embeddings then emb_ln."""
    input_ids, _ = random_input_ids(batch, seqlen, config, seed=seed)
    with torch.no_grad():
        return reference_model.emb_ln(reference_model.embeddings(input_ids))


@pytest.fixture
def tt_encoder(device, config, tt_config, state_dict):
    return TtNomicBertEncoder(device, config, tt_config, state_dict, state_dict_prefix="encoder")


def pooled_cosine(reference_output: torch.Tensor, got: torch.Tensor, mask: torch.Tensor) -> float:
    """Worst per-row cosine between the two pooled, unit-norm embeddings."""
    ref_pooled = l2_normalize(mean_pool(reference_output, mask))
    got_pooled = l2_normalize(mean_pool(got, mask))
    return float((ref_pooled * got_pooled).sum(-1).min())


@pytest.mark.parametrize("batch, seqlen", STACK_SHAPES)
def test_encoder(device, config, reference_model, reference, tt_encoder, batch, seqlen):
    """The whole stack, no padding, gated on both the PCC and the pooled embedding."""
    x = encoder_input(reference_model, config, batch, seqlen)

    out = tt_encoder(to_device(to_block_layout(x), device), rotary_tables(device, config, seqlen))

    with torch.no_grad():
        ref = reference(x, attention_mask=None)
    got = from_block_layout(out)
    cosine = pooled_cosine(ref, got, torch.ones(batch, seqlen, dtype=torch.long))

    assert tuple(out.shape) == (batch, 1, seqlen, config.hidden_size)
    assert cosine > POOLED_COSINE, f"pooled embedding cosine {cosine:.6f}"
    assert_with_pcc(ref, got, MODULE_PCC)


@pytest.mark.parametrize("batch, seqlen", STACK_SHAPES)
def test_encoder_with_ragged_padding(device, config, reference_model, reference, tt_encoder, batch, seqlen):
    """The whole stack with 25% of each row padded, compared on the kept positions.

    The pooled embedding is taken with the same mask, so padded positions are excluded from it
    the way the model excludes them.
    """
    keep = (seqlen * 3) // 4
    mask = keep_mask(batch, seqlen, keep)
    x = encoder_input(reference_model, config, batch, seqlen)

    out = tt_encoder(
        to_device(to_block_layout(x), device),
        rotary_tables(device, config, seqlen),
        additive_attention_mask(mask, device),
    )

    with torch.no_grad():
        ref = reference(x, attention_mask=build_extended_attention_mask(mask, torch.float32))
    got = from_block_layout(out)
    cosine = pooled_cosine(ref, got, mask)

    assert torch.isfinite(got).all(), "dtype-min in the mask saturated somewhere"
    assert cosine > POOLED_COSINE, f"pooled embedding cosine {cosine:.6f}"
    assert_with_pcc(ref[:, :keep], got[:, :keep], MODULE_PCC)


def test_layer_ladder(device, config, reference_model, reference, tt_encoder):
    """Per-layer parity across the stack, which end-to-end PCC alone cannot show.

    Each TTNN block is driven by its own previous TTNN output, so this measures the port's
    accumulated error rather than comparing two per-layer results from the same input. The
    reference side is instrumented with forward hooks, the same mechanism
    test_reference_vs_hf_e2e.py uses against upstream.
    """
    batch, seqlen = 2, 128
    x = encoder_input(reference_model, config, batch, seqlen)
    rot_mats = rotary_tables(device, config, seqlen)

    captures, handles = capture_hidden_states(reference, [f"layers.{idx}" for idx in range(config.num_hidden_layers)])
    try:
        with torch.no_grad():
            reference(x, attention_mask=None)
    finally:
        for handle in handles:
            handle.remove()

    ladder = {}
    tensor = to_device(to_block_layout(x), device)
    for idx, layer in enumerate(tt_encoder.layers):
        tensor = layer(tensor, rot_mats)
        ladder[idx] = compute_pcc(from_block_layout(tensor), captures[f"layers.{idx}"])

    worst = min(ladder.values())
    assert worst > LADDER_PCC, "layer PCC ladder: " + ", ".join(f"{idx}:{pcc:.5f}" for idx, pcc in ladder.items())


@pytest.mark.parametrize("batch, seqlen", STACK_SHAPES)
def test_routing_agreement_holds_at_depth(device, config, reference_model, reference, tt_encoder, batch, seqlen):
    """Gate 5 applied layer by layer, on each MoE layer's own input.

    A routing flip is the one error in this port that a loose PCC gate genuinely cannot see, so it
    is counted rather than inferred. Each layer is measured on the TTNN activations that actually
    reach it, which is where a flip would first appear.

    The first MoE layer is held to gate 5's own bound, since its input has not diverged yet.
    Deeper layers are allowed more: a token rerouted once is routed from different activations
    from then on, so the rate compounds with depth (measured 0.000 to 0.008 at layer 1, up to
    0.051 at layer 11). That compounding is the router's documented near-tie rate accumulating,
    not a second defect: roughly 1% to 2% of tokens sit within the softmax's own 1.4e-3 error of
    a tie at every layer.
    """
    tokens = batch * seqlen
    x = encoder_input(reference_model, config, batch, seqlen)
    rot_mats = rotary_tables(device, config, seqlen)

    captures, handles = capture_hidden_states(reference, [f"layers.{idx}" for idx in range(config.num_hidden_layers)])
    try:
        with torch.no_grad():
            reference(x, attention_mask=None)
    finally:
        for handle in handles:
            handle.remove()

    disagreements = {}
    tensor = to_device(to_block_layout(x), device)
    reference_input = x
    for idx, layer in enumerate(tt_encoder.layers):
        if config.is_moe_layer(idx):
            _, _, indices = layer.mlp.router.select(flatten_tokens(tensor))
            selected = ttnn.to_torch(indices).long().reshape(tokens, config.moe_top_k)
            with torch.no_grad():
                _, _, ref_indices = reference.layers[idx].mlp.router(reference_input)
            disagreeing = sum(
                set(selected[token].tolist()) != set(ref_indices[token].tolist()) for token in range(tokens)
            )
            disagreements[idx] = disagreeing / tokens

        tensor = layer(tensor, rot_mats)
        reference_input = captures[f"layers.{idx}"]

    assert disagreements, "no MoE layer was measured; check config.is_moe_layer"
    report = "per-layer routing disagreement: " + ", ".join(
        f"{idx}:{fraction:.4f}" for idx, fraction in disagreements.items()
    )
    first = min(disagreements)
    assert disagreements[first] <= FIRST_MOE_DISAGREEMENT, report
    assert max(disagreements.values()) <= DEEP_MOE_DISAGREEMENT, report
