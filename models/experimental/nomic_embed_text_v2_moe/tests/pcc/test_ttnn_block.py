# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Module PCC for TtNomicBertBlock, one full encoder layer. Bring-up gate 8.

Both FFN variants are covered: layer 0 is dense, layer 1 is the first MoE layer. The MoE case is
compared on tokens whose routing agrees, for the reason set out in test_ttnn_moe.py.

Post-norm is what keeps this gate loose rather than tight: the residual is added before the norm,
so each sub-block output is re-centred and bfloat16 error does not compound the way it would in a
pre-norm block.
"""

import pytest
import torch

import ttnn

from models.common.metrics import compute_pcc
from models.common.utility_functions import run_for_blackhole
from models.experimental.nomic_embed_text_v2_moe.reference.modeling_nomic_moe import (
    NomicBertBlock,
    build_extended_attention_mask,
)
from models.experimental.nomic_embed_text_v2_moe.tests.pcc.module_common import (
    DENSE_LAYER,
    MOE_LAYER,
    TOKEN_SHAPES,
    from_block_layout,
    hidden_states,
    keep_mask,
    load_reference,
    to_block_layout,
)
from models.experimental.nomic_embed_text_v2_moe.tt.block import TtNomicBertBlock
from models.experimental.nomic_embed_text_v2_moe.tt.common import (
    additive_attention_mask,
    flatten_tokens,
    rotary_tables,
    to_device,
)
from tests.ttnn.utils_for_testing import assert_with_pcc

pytestmark = [run_for_blackhole(), pytest.mark.use_module_device, pytest.mark.needs_weights]

MODULE_PCC = 0.99

LAYERS = [
    pytest.param(DENSE_LAYER, False, id="dense"),
    pytest.param(MOE_LAYER, True, id="moe"),
]


def build(device, config, tt_config, state_dict, layer, moe):
    """The reference block and the TTNN block for one layer, sharing the checkpoint."""
    prefix = f"encoder.layers.{layer}."
    assert config.is_moe_layer(layer) == moe, "the layer index and the moe flag disagree"
    reference = load_reference(lambda: NomicBertBlock(config, moe=moe), state_dict, prefix)
    return reference, TtNomicBertBlock(device, config, tt_config, state_dict, prefix, moe=moe)


def routing_agreement(config, reference, tt_block, x, x_tt, moe, rot_mats, attn_mask=None, ref_mask=None):
    """(B, S) bool mask of tokens both sides routed alike, or all True on a dense layer.

    The router is probed on the tensor the MoE actually receives, which is norm1(attn(x) + x) and
    not the block input: routing on the block input describes a decision the model never makes,
    and a mask built from it would let genuine reroutes into the PCC comparison while excluding
    tokens that never moved. Both sides are stepped through their own first half to get there.
    """
    batch, seqlen, _ = x.shape
    if not moe:
        return torch.ones(batch, seqlen, dtype=torch.bool)

    tt_attn = tt_block.attn(x_tt, rot_mats, attn_mask)
    tt_hidden = tt_block._norm(tt_attn, x_tt, tt_block.norm1_weight, tt_block.norm1_bias)
    _, _, indices = tt_block.mlp.router.select(flatten_tokens(tt_hidden))
    selected = ttnn.to_torch(indices).long().reshape(batch * seqlen, config.moe_top_k)

    with torch.no_grad():
        ref_hidden = reference.norm1(reference.attn(x, attention_mask=ref_mask) + x)
        _, _, ref_indices = reference.mlp.router(ref_hidden)

    agreeing = torch.tensor(
        [set(selected[token].tolist()) == set(ref_indices[token].tolist()) for token in range(batch * seqlen)]
    )
    return agreeing.reshape(batch, seqlen)


@pytest.mark.parametrize("layer, moe", LAYERS)
@pytest.mark.parametrize("batch, seqlen", TOKEN_SHAPES)
def test_block(device, config, tt_config, state_dict, layer, moe, batch, seqlen):
    """Attention, FFN and both fused residual-add norms, with no padding."""
    reference, tt_block = build(device, config, tt_config, state_dict, layer, moe)
    x = hidden_states(batch, seqlen, config.hidden_size)
    x_tt = to_device(to_block_layout(x), device)

    rot_mats = rotary_tables(device, config, seqlen)

    out = tt_block(x_tt, rot_mats)

    with torch.no_grad():
        ref = reference(x, attention_mask=None)
    got = from_block_layout(out)
    agreeing = routing_agreement(config, reference, tt_block, x, x_tt, moe, rot_mats)

    assert tuple(out.shape) == (batch, 1, seqlen, config.hidden_size)
    assert_with_pcc(ref[agreeing], got[agreeing], MODULE_PCC)


@pytest.mark.parametrize("layer, moe", LAYERS)
@pytest.mark.parametrize("batch, seqlen", TOKEN_SHAPES)
def test_block_with_ragged_padding(device, config, tt_config, state_dict, layer, moe, batch, seqlen):
    """The same block with 25% of each row padded, compared on the kept positions."""
    reference, tt_block = build(device, config, tt_config, state_dict, layer, moe)
    keep = (seqlen * 3) // 4
    mask = keep_mask(batch, seqlen, keep)
    x = hidden_states(batch, seqlen, config.hidden_size)
    x_tt = to_device(to_block_layout(x), device)

    rot_mats = rotary_tables(device, config, seqlen)
    attn_mask = additive_attention_mask(mask, device)
    ref_mask = build_extended_attention_mask(mask, torch.float32)

    out = tt_block(x_tt, rot_mats, attn_mask)

    with torch.no_grad():
        ref = reference(x, attention_mask=ref_mask)
    got = from_block_layout(out)
    kept = routing_agreement(
        config, reference, tt_block, x, x_tt, moe, rot_mats, attn_mask=attn_mask, ref_mask=ref_mask
    )[:, :keep]

    assert torch.isfinite(got).all(), "dtype-min in the mask saturated somewhere"
    assert_with_pcc(ref[:, :keep][kept], got[:, :keep][kept], MODULE_PCC)


@pytest.mark.parametrize("layer, moe", LAYERS)
def test_the_block_is_post_norm(device, config, tt_config, state_dict, layer, moe):
    """The residual is added before each norm, not after.

    Pre-norm is the other plausible reading of the same four weights and runs identically. It is
    separated here by the output statistics rather than by PCC against the reference, so this
    holds even if both sides were wrong in the same way: a post-norm block's output is the direct
    output of a layer norm, so every row is centred and unit-scaled up to the norm's own affine
    weights.
    """
    reference, tt_block = build(device, config, tt_config, state_dict, layer, moe)
    batch, seqlen = 2, 128
    x = hidden_states(batch, seqlen, config.hidden_size)

    got = from_block_layout(tt_block(to_device(to_block_layout(x), device), rotary_tables(device, config, seqlen)))

    weight = reference.norm2.weight.detach()
    bias = reference.norm2.bias.detach()
    normalized = (got - bias) / weight
    assert normalized.mean(dim=-1).abs().max() < 5e-2, "block output is not the output of a layer norm"
    assert abs(float(normalized.std(dim=-1).mean()) - 1.0) < 5e-2


@pytest.mark.parametrize("layer, moe", LAYERS)
def test_dropping_a_residual_is_decorrelated(device, config, tt_config, state_dict, layer, moe):
    """Negative control: a missing residual still norms to plausible output.

    layer_norm(residual_input_tensor=...) fuses the add, so forgetting the argument does not
    raise, does not change the shape, and returns a normalized tensor. Only a comparison catches
    it.
    """
    reference, tt_block = build(device, config, tt_config, state_dict, layer, moe)
    batch, seqlen = 2, 128
    x = hidden_states(batch, seqlen, config.hidden_size)
    x_tt = to_device(to_block_layout(x), device)
    rot_mats = rotary_tables(device, config, seqlen)

    with torch.no_grad():
        ref = reference(x, attention_mask=None)
    agreeing = routing_agreement(config, reference, tt_block, x, x_tt, moe, rot_mats)

    correct = from_block_layout(tt_block(x_tt, rot_mats))

    attn_out = tt_block.attn(x_tt, rot_mats)
    without_residual = ttnn.layer_norm(
        attn_out,
        weight=tt_block.norm1_weight,
        bias=tt_block.norm1_bias,
        epsilon=config.layer_norm_epsilon,
        compute_kernel_config=tt_config.compute_kernel_config,
    )
    dropped = from_block_layout(
        ttnn.layer_norm(
            tt_block.mlp(without_residual),
            residual_input_tensor=without_residual,
            weight=tt_block.norm2_weight,
            bias=tt_block.norm2_bias,
            epsilon=config.layer_norm_epsilon,
            compute_kernel_config=tt_config.compute_kernel_config,
        )
    )

    assert compute_pcc(correct[agreeing], ref[agreeing]) > MODULE_PCC
    assert compute_pcc(dropped[agreeing], ref[agreeing]) < MODULE_PCC
