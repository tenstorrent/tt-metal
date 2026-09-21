# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-operator PCC validation for the attention sub-block.

Four device operators stand in for the reference's whole attention path. Three of them encode
a convention that the wrong choice satisfies just as well:

  nlp_create_qkv_heads   three-major [q | k | v], not head-major
  rotary_embedding_hf    NeoX half-split with concat-widened tables, not GPT-J interleaved
  SDPA                   bidirectional, not causal

Each produces finite, plausible output taken the other way, so each carries a negative control
alongside its PCC test. The causal one is the easiest to hit by accident: ttnn's is_causal
defaults to True where torch's defaults to False.

Activations only, so no checkpoint is needed. Measured results are in docs/OPERATOR_MAPPING.md.
"""

import pytest
import torch

import ttnn

from models.common.metrics import compute_pcc
from models.common.utility_functions import run_for_blackhole
from models.experimental.nomic_embed_text_v2_moe.reference.modeling_nomic_moe import (
    NomicBertRotaryEmbedding,
    apply_rotary_emb,
    build_extended_attention_mask,
)
from models.experimental.nomic_embed_text_v2_moe.tt.common import additive_attention_mask, rotary_tables, to_device
from tests.ttnn.utils_for_testing import assert_with_pcc

pytestmark = [run_for_blackhole(), pytest.mark.use_module_device]

# SDPA is the tightest operator in the model at 0.9998, accumulating over the full key axis in
# bf16; everything else here reaches 0.99999.
OPERATOR_PCC = 0.999

# A wrong layout does not lose precision, it decorrelates: the two layout controls measure 0.08
# (head-major) and 0.21 (interleaved rotary). Anything under this is the wrong tensor.
DECORRELATED_PCC = 0.5

# (batch, seqlen). 37 is deliberately off-tile: padding bugs only surface when S is not a
# multiple of 32, and S is the batch's longest tokenized sequence, so that is the common case.
TOKEN_SHAPES = [(1, 128), (2, 512), (2, 37)]


def reference_cos_sin(config, seqlen: int):
    """Half-width cos/sin tables as NomicBertRotaryEmbedding caches them, (S, D // 2)."""
    rotary = NomicBertRotaryEmbedding(dim=config.rotary_dim, base=config.rotary_emb_base)
    rotary._update_cos_sin_cache(seqlen, device=torch.device("cpu"), dtype=torch.float32)
    return rotary._cos_cached, rotary._sin_cached


def fold_batch_into_heads(x: torch.Tensor) -> torch.Tensor:
    """(B, A, S, D) -> (1, B*A, S, D), the prefill layout rotary_embedding_hf takes.

    The op's prefill mode wants a leading batch of 1. cos/sin are (1, 1, S, D) and broadcast
    over the head axis, so folding batch into heads applies the same table to every row.
    """
    batch, heads, seqlen, head_dim = x.shape
    return x.reshape(1, batch * heads, seqlen, head_dim)


# Head splitting.


@pytest.mark.parametrize("batch, seqlen", TOKEN_SHAPES)
@pytest.mark.parametrize("index", [pytest.param(0, id="query"), pytest.param(1, id="key"), pytest.param(2, id="value")])
def test_nlp_create_qkv_heads(device, config, batch, seqlen, index):
    """aten.view + aten.select + aten.permute -> ttnn.experimental.nlp_create_qkv_heads.

    Wqkv is three-major: q, k and v are contiguous blocks of hidden_size with heads contiguous
    inside each.
    """
    heads, head_dim = config.num_attention_heads, config.head_dim
    qkv = torch.randn(batch, 1, seqlen, config.qkv_dim)

    parts = ttnn.experimental.nlp_create_qkv_heads(
        to_device(qkv, device), num_heads=heads, num_kv_heads=heads, transpose_k_heads=False
    )

    ref = qkv.view(batch, seqlen, 3, heads, head_dim)[:, :, index].permute(0, 2, 1, 3)
    assert_with_pcc(ref, parts[index], OPERATOR_PCC)


def test_head_major_qkv_view_is_decorrelated(device, config):
    """Negative control: reading Wqkv head-major strides across the q/k/v boundaries.

    The element count is the same either way, so the reshape succeeds and every downstream op
    typechecks.
    """
    batch, seqlen = 2, 128
    heads, head_dim = config.num_attention_heads, config.head_dim
    qkv = torch.randn(batch, 1, seqlen, config.qkv_dim)

    query, _, _ = ttnn.experimental.nlp_create_qkv_heads(
        to_device(qkv, device), num_heads=heads, num_kv_heads=heads, transpose_k_heads=False
    )

    got = ttnn.to_torch(query).float()
    three_major = qkv.view(batch, seqlen, 3, heads, head_dim)[:, :, 0].permute(0, 2, 1, 3)
    head_major = qkv.view(batch, seqlen, heads, 3, head_dim)[:, :, :, 0].permute(0, 2, 1, 3)

    # Guards against a vacuous pass: compute_pcc returns 0.0 on a broken comparison.
    assert compute_pcc(got, three_major) > OPERATOR_PCC
    assert compute_pcc(got, head_major) < DECORRELATED_PCC


# Rotary.


@pytest.mark.parametrize("seqlen", [128, 512])
def test_rotary_tables_match_the_reference(device, config, seqlen):
    """The tables come from tt_transformers' get_rot_mats_hf; this is the guard on that reuse.

    Its unscaled path computes the same inv_freq, outer product and concat widening the
    reference does, so the two are bit-exact in fp32. If a change upstream ever breaks that,
    this fails before any rotated tensor does.
    """
    cos_tt, sin_tt = rotary_tables(device, config, seqlen, dtype=ttnn.float32)
    cos_ref, sin_ref = reference_cos_sin(config, seqlen)

    assert torch.equal(ttnn.to_torch(cos_tt), torch.cat((cos_ref, cos_ref), dim=-1)[None, None])
    assert torch.equal(ttnn.to_torch(sin_tt), torch.cat((sin_ref, sin_ref), dim=-1)[None, None])


@pytest.mark.parametrize("batch, seqlen", TOKEN_SHAPES)
def test_rotary_embedding_hf(device, config, batch, seqlen):
    """aten.cos + sin + cat + neg + split + mul -> ttnn.experimental.rotary_embedding_hf."""
    heads, head_dim = config.num_attention_heads, config.head_dim
    x = torch.randn(batch, seqlen, heads, head_dim)
    cos_ref, sin_ref = reference_cos_sin(config, seqlen)
    cos_tt, sin_tt = rotary_tables(device, config, seqlen)

    x_tt = to_device(fold_batch_into_heads(x.permute(0, 2, 1, 3)), device)
    out = ttnn.experimental.rotary_embedding_hf(x_tt, cos_tt, sin_tt, is_decode_mode=False)

    ref = apply_rotary_emb(x, cos_ref, sin_ref).permute(0, 2, 1, 3)
    assert_with_pcc(ref, ttnn.to_torch(out).reshape(batch, heads, seqlen, head_dim), OPERATOR_PCC)


def test_rotary_at_position_zero_is_the_identity(device, config):
    """Analytic anchor: position 0 has cos 1 and sin 0, so the rotation must leave x alone.

    A PCC test against a reference that shares the port's own convention cannot catch a
    convention that is wrong on both sides. This one does not depend on the reference at all.
    """
    heads, head_dim = config.num_attention_heads, config.head_dim
    x = torch.randn(1, heads, ttnn.TILE_SIZE, head_dim)
    cos_tt, sin_tt = rotary_tables(device, config, ttnn.TILE_SIZE)

    out = ttnn.to_torch(
        ttnn.experimental.rotary_embedding_hf(to_device(x, device), cos_tt, sin_tt, is_decode_mode=False)
    ).float()

    assert_with_pcc(x[:, :, 0], out[:, :, 0], 0.9999)


def test_interleaved_rotary_tables_are_decorrelated(device, config):
    """Negative control: GPT-J lane pairing under the kernel's NeoX rotate-half.

    repeat_interleave instead of concat is the other plausible way to widen the half-dim
    tables. Combined with half-split rotation it is not a rotation and does not preserve the
    per-plane norm, but it runs and produces finite output.
    """
    batch, seqlen = 2, 128
    heads, head_dim = config.num_attention_heads, config.head_dim
    x = torch.randn(batch, seqlen, heads, head_dim)
    cos_ref, sin_ref = reference_cos_sin(config, seqlen)

    x_tt = to_device(fold_batch_into_heads(x.permute(0, 2, 1, 3)), device)
    out = ttnn.experimental.rotary_embedding_hf(
        x_tt,
        to_device(torch.repeat_interleave(cos_ref, 2, dim=-1)[None, None], device),
        to_device(torch.repeat_interleave(sin_ref, 2, dim=-1)[None, None], device),
        is_decode_mode=False,
    )

    correct = ttnn.experimental.rotary_embedding_hf(x_tt, *rotary_tables(device, config, seqlen), is_decode_mode=False)
    ref = apply_rotary_emb(x, cos_ref, sin_ref).permute(0, 2, 1, 3)
    shape = (batch, heads, seqlen, head_dim)

    # Guards against a vacuous pass: compute_pcc returns 0.0 on a broken comparison.
    assert compute_pcc(ttnn.to_torch(correct).reshape(shape).float(), ref) > OPERATOR_PCC
    assert compute_pcc(ttnn.to_torch(out).reshape(shape).float(), ref) < DECORRELATED_PCC


# Attention proper.


@pytest.mark.parametrize("batch, seqlen", TOKEN_SHAPES)
def test_sdpa_unmasked(device, tt_config, config, batch, seqlen):
    """aten.bmm + aten.mul + aten._safe_softmax + aten.bmm -> one SDPA call.

    is_causal is always passed explicitly: ttnn defaults it to True, torch to False, and this
    is an encoder.
    """
    heads, head_dim = config.num_attention_heads, config.head_dim
    query, key, value = (torch.randn(batch, heads, seqlen, head_dim) for _ in range(3))

    out = ttnn.transformer.scaled_dot_product_attention(
        to_device(query, device),
        to_device(key, device),
        to_device(value, device),
        is_causal=False,
        compute_kernel_config=tt_config.compute_kernel_config,
    )

    ref = torch.nn.functional.scaled_dot_product_attention(query, key, value, is_causal=False)
    assert_with_pcc(ref, out, OPERATOR_PCC)


@pytest.mark.parametrize("batch, seqlen", TOKEN_SHAPES)
def test_sdpa_with_ragged_padding(device, tt_config, config, batch, seqlen):
    """The same call with a 25%-padded additive mask, compared on kept positions.

    The mask carries dtype-min at padded keys. Nothing here is fully masked, so no row
    degenerates, but the output is still checked for finiteness because a saturating fill value
    is the way that would show up.
    """
    heads, head_dim = config.num_attention_heads, config.head_dim
    keep = (seqlen * 3) // 4
    mask = torch.ones(batch, seqlen, dtype=torch.long)
    mask[:, keep:] = 0
    query, key, value = (torch.randn(batch, heads, seqlen, head_dim) for _ in range(3))

    out = ttnn.transformer.scaled_dot_product_attention(
        to_device(query, device),
        to_device(key, device),
        to_device(value, device),
        attn_mask=additive_attention_mask(mask, device),
        is_causal=False,
        compute_kernel_config=tt_config.compute_kernel_config,
    )

    ref = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=build_extended_attention_mask(mask, torch.float32), is_causal=False
    )
    got = ttnn.to_torch(out).float()
    assert torch.isfinite(got).all(), "dtype-min in the mask saturated somewhere"
    assert_with_pcc(ref[:, :, :keep], got[:, :, :keep], OPERATOR_PCC)


def test_causal_attention_is_decorrelated(device, tt_config, config):
    """Negative control: this is an encoder, and ttnn's is_causal defaults to True.

    Omitting the flag silently applies a decoder mask. Every token still gets a finite output,
    just one computed from its prefix alone. Causal keeps more correlation than the layout
    controls do, since it averages a subset of the same values rather than the wrong ones.
    """
    batch, seqlen = 2, 512
    heads, head_dim = config.num_attention_heads, config.head_dim
    query, key, value = (torch.randn(batch, heads, seqlen, head_dim) for _ in range(3))
    operands = [to_device(t, device) for t in (query, key, value)]
    ref = torch.nn.functional.scaled_dot_product_attention(query, key, value, is_causal=False)

    def pcc(is_causal):
        out = ttnn.transformer.scaled_dot_product_attention(
            *operands, is_causal=is_causal, compute_kernel_config=tt_config.compute_kernel_config
        )
        return compute_pcc(ttnn.to_torch(out).float(), ref)

    assert pcc(False) > OPERATOR_PCC
    causal = pcc(True)
    assert causal < 0.7, f"causal should sit far from bidirectional, measured 0.43 to 0.48, got {causal:.3f}"


@pytest.mark.parametrize("seqlen", [37, 100, 128, 513])
def test_all_ones_mask_matches_no_mask(device, tt_config, config, seqlen):
    """A mask that keeps everything must be a no-op. Tile padding is what breaks this.

    The mask is (B, 1, S, S) in TILE layout, so S rounds up to a multiple of 32 and the pad
    columns take whatever the conversion fills them with. 0 is additively neutral, meaning
    "attend here", so SDPA counts them in the softmax denominator and the output shrinks: at
    S=37 the norm was 0.69x before additive_attention_mask padded with dtype-min instead.

    Gated on the norm, not PCC. PCC is insensitive to a near-uniform scale, and at S=513 it
    did not move at all while the norm was 0.96x.
    """
    batch = 2
    heads, head_dim = config.num_attention_heads, config.head_dim
    query, key, value = (torch.randn(batch, heads, seqlen, head_dim) for _ in range(3))
    operands = [to_device(t, device) for t in (query, key, value)]

    def run(mask):
        out = ttnn.transformer.scaled_dot_product_attention(
            *operands, attn_mask=mask, is_causal=False, compute_kernel_config=tt_config.compute_kernel_config
        )
        return ttnn.to_torch(out).float()

    unmasked = run(None)
    kept = run(additive_attention_mask(torch.ones(batch, seqlen, dtype=torch.long), device))

    assert torch.equal(kept, unmasked), (
        f"an all-ones mask changed the result at S={seqlen}; "
        f"norm ratio {(kept.norm() / unmasked.norm()).item():.4f}"
    )


@pytest.mark.parametrize("batch, seqlen", TOKEN_SHAPES)
def test_nlp_concat_heads(device, config, batch, seqlen):
    """aten.permute + aten.reshape -> ttnn.experimental.nlp_concat_heads."""
    heads, head_dim = config.num_attention_heads, config.head_dim
    context = torch.randn(batch, heads, seqlen, head_dim)

    out = ttnn.experimental.nlp_concat_heads(to_device(context, device))

    ref = context.permute(0, 2, 1, 3).reshape(batch, 1, seqlen, config.hidden_size)
    assert_with_pcc(ref, out, OPERATOR_PCC)
