# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Module PCC for TtNomicBertAttention. Bring-up gates 2 (rotary) and 3 (attention).

The masked case is compared on kept positions only. Padded queries attend to nothing useful on
either side and their outputs are not consumed downstream, since pooling masks them out; scoring
them would measure the fill value rather than the attention.

Two conventions are checked by negative control instead of by PCC, because both produce finite,
plausible output: the rotary table widening, and the is_causal default the module has to override.
"""

import pytest
import torch

import ttnn

from models.common.metrics import compute_pcc
from models.common.utility_functions import run_for_blackhole
from models.experimental.nomic_embed_text_v2_moe.reference.modeling_nomic_moe import (
    NomicBertAttention,
    NomicBertRotaryEmbedding,
    build_extended_attention_mask,
)
from models.experimental.nomic_embed_text_v2_moe.tests.pcc.module_common import (
    DECORRELATED_PCC,
    DENSE_LAYER,
    TOKEN_SHAPES,
    from_block_layout,
    hidden_states,
    keep_mask,
    load_reference,
    to_block_layout,
)
from models.experimental.nomic_embed_text_v2_moe.tt.attention import TtNomicBertAttention
from models.experimental.nomic_embed_text_v2_moe.tt.common import (
    additive_attention_mask,
    rotary_tables,
    to_device,
)
from tests.ttnn.utils_for_testing import assert_with_pcc

pytestmark = [run_for_blackhole(), pytest.mark.use_module_device, pytest.mark.needs_weights]

# SDPA accumulates over the full key axis in bfloat16 and is the tightest operator in the model
# at 0.9998; the module adds two projections and the rotary on top of it.
MODULE_PCC = 0.99

PREFIX = f"encoder.layers.{DENSE_LAYER}.attn."


@pytest.fixture
def reference(config, state_dict):
    return load_reference(lambda: NomicBertAttention(config), state_dict, PREFIX)


@pytest.fixture
def tt_attention(device, config, tt_config, state_dict):
    return TtNomicBertAttention(device, config, tt_config, state_dict, PREFIX)


def interleaved_tables(device, config, seqlen, dtype=ttnn.bfloat16):
    """The GPT-J lane pairing: repeat_interleave instead of concat when widening D//2 to D.

    The other plausible way to widen the half-width tables, and wrong under the kernel's NeoX
    rotate-half: the pair is not a rotation and does not preserve the per-plane norm.
    """
    rotary = NomicBertRotaryEmbedding(dim=config.rotary_dim, base=config.rotary_emb_base)
    rotary._update_cos_sin_cache(seqlen, device=torch.device("cpu"), dtype=torch.float32)
    return tuple(
        to_device(torch.repeat_interleave(table, 2, dim=-1)[None, None], device, dtype=dtype)
        for table in (rotary._cos_cached, rotary._sin_cached)
    )


@pytest.mark.parametrize("batch, seqlen", TOKEN_SHAPES)
def test_attention_unmasked(device, config, reference, tt_attention, batch, seqlen):
    """Projection, rotary, bidirectional SDPA and the output projection, with no padding."""
    x = hidden_states(batch, seqlen, config.hidden_size)

    out = tt_attention(to_device(to_block_layout(x), device), rotary_tables(device, config, seqlen))

    with torch.no_grad():
        ref = reference(x, attention_mask=None)
    assert tuple(out.shape) == (batch, 1, seqlen, config.hidden_size)
    assert_with_pcc(ref, from_block_layout(out), MODULE_PCC)


@pytest.mark.parametrize("batch, seqlen", TOKEN_SHAPES)
def test_attention_with_ragged_padding(device, config, reference, tt_attention, batch, seqlen):
    """The same module with 25% of each row padded, compared on the kept positions.

    Finiteness is asserted too: the mask carries dtype-min, and saturation is how that would
    surface.
    """
    keep = (seqlen * 3) // 4
    mask = keep_mask(batch, seqlen, keep)
    x = hidden_states(batch, seqlen, config.hidden_size)

    out = tt_attention(
        to_device(to_block_layout(x), device),
        rotary_tables(device, config, seqlen),
        additive_attention_mask(mask, device),
    )

    with torch.no_grad():
        ref = reference(x, attention_mask=build_extended_attention_mask(mask, torch.float32))
    got = from_block_layout(out)
    assert torch.isfinite(got).all(), "dtype-min in the mask saturated somewhere"
    assert_with_pcc(ref[:, :keep], got[:, :keep], MODULE_PCC)


@pytest.mark.parametrize("seqlen", [37, 128])
def test_an_all_ones_mask_is_a_no_op(device, config, tt_attention, seqlen):
    """Masking nothing must change nothing, which is what pins the mask's tile padding.

    The mask is (B, 1, S, S) in TILE layout, so S rounds up to a multiple of 32 and the pad
    columns take whatever the conversion fills them with. 0 is additively neutral, meaning
    "attend here", so SDPA would count them in the softmax denominator: at S=37 that took the
    output norm to 0.69x. Gated on equality rather than PCC, which barely moves.
    """
    batch = 2
    x = to_device(to_block_layout(hidden_states(batch, seqlen, config.hidden_size)), device)
    rot_mats = rotary_tables(device, config, seqlen)

    unmasked = from_block_layout(tt_attention(x, rot_mats))
    masked = from_block_layout(
        tt_attention(x, rot_mats, additive_attention_mask(keep_mask(batch, seqlen, seqlen), device))
    )

    assert torch.equal(masked, unmasked), (
        f"an all-ones mask changed the result at S={seqlen}; norm ratio "
        f"{(masked.norm() / unmasked.norm()).item():.4f}"
    )


def test_rotary_at_position_zero_is_the_identity(device, config, state_dict, tt_config):
    """Analytic anchor for gate 2: position 0 has cos 1 and sin 0, so it must not rotate.

    A PCC comparison against a reference that shares the port's own convention cannot catch a
    convention that is wrong on both sides. This probe does not depend on the reference.
    """
    attention = TtNomicBertAttention(device, config, tt_config, state_dict, PREFIX)
    heads, head_dim = config.num_attention_heads, config.head_dim
    x = torch.randn(1, heads, ttnn.TILE_SIZE, head_dim)
    cos, sin = rotary_tables(device, config, ttnn.TILE_SIZE)

    rotated = ttnn.to_torch(attention._rotate(to_device(x, device), cos, sin)).float()

    assert_with_pcc(x[:, :, 0], rotated[:, :, 0], 0.9999)


def test_interleaved_rotary_tables_are_decorrelated(device, config, reference, tt_attention):
    """Negative control for gate 2: the module must be fed concat-widened tables.

    Both widenings run and return finite output, so only a comparison separates them.
    """
    batch, seqlen = 2, 128
    x = hidden_states(batch, seqlen, config.hidden_size)
    x_tt = to_device(to_block_layout(x), device)

    with torch.no_grad():
        ref = reference(x, attention_mask=None)

    correct = from_block_layout(tt_attention(x_tt, rotary_tables(device, config, seqlen)))
    interleaved = from_block_layout(tt_attention(x_tt, interleaved_tables(device, config, seqlen)))

    # Guards against a vacuous pass: compute_pcc returns 0.0 on a broken comparison.
    assert compute_pcc(correct, ref) > MODULE_PCC
    assert compute_pcc(interleaved, ref) < DECORRELATED_PCC


def test_the_module_overrides_the_causal_default(device, config, reference, tt_attention, tt_config):
    """Negative control for gate 3: ttnn defaults is_causal to True, torch to False.

    This is an encoder, so the module has to pass the flag explicitly. Left at the ttnn default
    every token still gets a finite output, just one computed from its prefix alone.
    """
    batch, seqlen = 2, 128
    x = hidden_states(batch, seqlen, config.hidden_size)

    with torch.no_grad():
        ref = reference(x, attention_mask=None)

    bidirectional = from_block_layout(
        tt_attention(to_device(to_block_layout(x), device), rotary_tables(device, config, seqlen))
    )

    heads, head_dim = config.num_attention_heads, config.head_dim
    operands = [to_device(torch.randn(batch, heads, seqlen, head_dim), device) for _ in range(3)]
    causal, acausal = (
        ttnn.to_torch(
            ttnn.transformer.scaled_dot_product_attention(
                *operands, is_causal=flag, compute_kernel_config=tt_config.compute_kernel_config
            )
        ).float()
        for flag in (True, False)
    )

    assert compute_pcc(bidirectional, ref) > MODULE_PCC
    assert compute_pcc(causal, acausal) < 0.7, "causal and bidirectional SDPA agreed; re-check the default"
