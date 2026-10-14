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
from models.experimental.nomic_embed_text_v2_moe.tt.attention import attention_placement, sdpa_program_config
from models.experimental.nomic_embed_text_v2_moe.tt.common import (
    RotaryTables,
    additive_attention_mask,
    rotary_tables,
    to_device,
)
from models.experimental.nomic_embed_text_v2_moe.tt.model_config import OpGroup
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


def sdpa_kwargs(tt_config, config, batch, seqlen, masked):
    """The program and compute configs the model passes SDPA for this call."""
    compute_kernel_config = tt_config.compute_kernel_config(OpGroup.SDPA)
    return {
        "program_config": sdpa_program_config(
            batch,
            seqlen,
            config.num_attention_heads,
            tt_config.core_grid,
            compute_kernel_config.fp32_dest_acc_en,
            masked,
        ),
        "compute_kernel_config": compute_kernel_config,
    }


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


def padding_rows(tensor: ttnn.Tensor, seqlen: int) -> torch.Tensor:
    """The tile-padding rows of a device tensor's sequence axis, read back with the padded shape."""
    return tensor.cpu().to_torch_with_padded_shape().float()[..., seqlen:, :]


def test_rotary_zeroes_the_padding_of_q_only_with_zero_padded_tables(device, config):
    """The rotary tables' tile padding is what keeps the padding rows of q and k finite.

    The MoE layer's output padding is unwritten (tt.common.unflatten_tokens) and reaches q and k one
    block later. An S-row table pads with zeros, and inf times those zeros comes out 0; a longer
    table puts real cos/sin there and the inf stays. Non-finite padded keys move SDPA's output
    (test_sdpa_is_moved_by_non_finite_padded_keys), which is why RotaryTables cuts its tables.
    """
    batch, seqlen = 2, 75
    heads, head_dim = config.num_attention_heads, config.head_dim
    # In place: the fill returns a tensor on the same buffer.
    q = ttnn.fill_implicit_tile_padding(
        to_device(torch.randn(1, batch * heads, seqlen, head_dim), device), float("inf")
    )

    exact = rotary_tables(device, config, seqlen)
    longer = rotary_tables(device, config, 512)

    rotated = padding_rows(ttnn.experimental.rotary_embedding_hf(q, *exact, is_decode_mode=False), seqlen)
    assert torch.equal(rotated, torch.zeros_like(rotated))
    rotated = padding_rows(ttnn.experimental.rotary_embedding_hf(q, *longer, is_decode_mode=False), seqlen)
    passes_through = not torch.isfinite(rotated).all()
    assert passes_through, "a longer table no longer passes the padding through; re-check RotaryTables"


def test_rotary_tables_cut_from_a_longer_pair_match_a_fresh_build(device, config):
    """RotaryTables keeps the longest pair and serves each call what a build for that S would give.

    Off the tile grid the pair is cut to S with the padding zeroed, bit for bit a fresh build's,
    padding included; on the grid there are no padding rows, and the kept pair is handed out as is.
    """
    cache = RotaryTables(device, config)
    kept = cache(512)
    assert cache(64) is kept

    cut = cache(75)
    for got, fresh in zip(cut, rotary_tables(device, config, 75)):
        assert tuple(got.shape) == tuple(fresh.shape)
        assert torch.equal(got.cpu().to_torch_with_padded_shape(), fresh.cpu().to_torch_with_padded_shape())
    cache.release(cut)
    assert cache(512) is kept


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
        **sdpa_kwargs(tt_config, config, batch, seqlen, masked=False),
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
        attn_mask=additive_attention_mask(mask, device, mask_dtype=tt_config.attention_mask_dtype),
        is_causal=False,
        **sdpa_kwargs(tt_config, config, batch, seqlen, masked=True),
    )

    ref = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=build_extended_attention_mask(mask, torch.float32), is_causal=False
    )
    got = ttnn.to_torch(out).float()
    assert torch.isfinite(got).all(), "dtype-min in the mask saturated somewhere"
    assert_with_pcc(ref[:, :, :keep], got[:, :, :keep], OPERATOR_PCC)


@pytest.mark.parametrize("operand", ["query", "key", "value"])
def test_sdpa_is_moved_by_non_finite_padded_keys(device, tt_config, config, operand):
    """The mask does not neutralize a non-finite key in the tile padding of S; q and v padding are inert.

    Measured at 2x75 with one row a third padded: inf, NaN or 3e38 in the padding rows of k move
    every output row, by 8e-3 here and up to 1e37 in the model, while the same values in q or v
    change nothing. The padding rows of k therefore have to stay finite, which the zero padding of
    the rotary tables ensures (test_rotary_zeroes_the_padding_of_q_only_with_zero_padded_tables).
    """
    batch, seqlen = 2, 75
    heads, head_dim = config.num_attention_heads, config.head_dim
    mask = torch.ones(batch, seqlen, dtype=torch.long)
    mask[0, 50:] = 0
    attn_mask = additive_attention_mask(mask, device, mask_dtype=tt_config.attention_mask_dtype)
    operands = {name: torch.randn(batch, heads, seqlen, head_dim) for name in ("query", "key", "value")}

    def attend(poisoned):
        tensors = {name: to_device(tensor, device) for name, tensor in operands.items()}
        if poisoned:
            # In place: the fill returns a tensor on the same buffer.
            tensors[operand] = ttnn.fill_implicit_tile_padding(tensors[operand], float("inf"))
        out = ttnn.transformer.scaled_dot_product_attention(
            tensors["query"],
            tensors["key"],
            tensors["value"],
            attn_mask=attn_mask,
            is_causal=False,
            scale=1.0,
            **sdpa_kwargs(tt_config, config, batch, seqlen, masked=True),
        )
        return ttnn.to_torch(out).float()

    clean, poisoned = attend(False), attend(True)
    if operand == "key":
        assert not torch.equal(clean, poisoned), "masked SDPA now ignores non-finite padded keys"
    else:
        assert torch.equal(clean, poisoned)


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
            *operands, is_causal=is_causal, compute_kernel_config=tt_config.compute_kernel_config(OpGroup.SDPA)
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

    The model skips an all-ones mask, but a padded batch's mask has the same tile padding, so
    this runs the mask in the model's dtype. Both calls take the masked program config, so the
    comparison sees the mask alone.
    """
    batch = 2
    heads, head_dim = config.num_attention_heads, config.head_dim
    query, key, value = (torch.randn(batch, heads, seqlen, head_dim) for _ in range(3))
    operands = [to_device(t, device) for t in (query, key, value)]

    def run(mask):
        out = ttnn.transformer.scaled_dot_product_attention(
            *operands,
            attn_mask=mask,
            is_causal=False,
            **sdpa_kwargs(tt_config, config, batch, seqlen, masked=True),
        )
        return ttnn.to_torch(out).float()

    unmasked = run(None)
    kept = run(
        additive_attention_mask(
            torch.ones(batch, seqlen, dtype=torch.long), device, mask_dtype=tt_config.attention_mask_dtype
        )
    )

    assert torch.equal(kept, unmasked), (
        f"an all-ones mask changed the result at S={seqlen}; "
        f"norm ratio {(kept.norm() / unmasked.norm()).item():.4f}"
    )


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("batch, seqlen", [(8, 528), (8, 264), (12, 352)])
def test_sdpa_at_shapes_where_its_default_chunks_are_wrong(device, tt_config, config, batch, seqlen, masked):
    """The model's program config at batch shapes where SDPA's default one returns wrong rows.

    Given no program config, SDPA returns errors up to 1e+38 on some rows at 8 or more sequences
    of an odd number of tiles, masked or not: rows 0 and 2 to 7 at 8x528, row 7 at 8x264. The
    chunks the model picks are correct there, which this pins row by row.
    """
    heads, head_dim = config.num_attention_heads, config.head_dim
    query, key, value = (torch.randn(batch, heads, seqlen, head_dim) for _ in range(3))
    mask = None
    if masked:
        mask = additive_attention_mask(
            torch.ones(batch, seqlen, dtype=torch.long), device, mask_dtype=tt_config.attention_mask_dtype
        )

    out = ttnn.transformer.scaled_dot_product_attention(
        to_device(query, device),
        to_device(key, device),
        to_device(value, device),
        attn_mask=mask,
        is_causal=False,
        **sdpa_kwargs(tt_config, config, batch, seqlen, masked=masked),
    )

    ref = torch.nn.functional.scaled_dot_product_attention(query, key, value, is_causal=False)
    got = ttnn.to_torch(out).float()
    per_row = (got - ref).abs().amax(dim=(1, 2, 3))
    wrong = per_row.gt(0.1).nonzero().flatten().tolist()
    assert not wrong, f"rows {wrong} are wrong, worst max abs {per_row.max():.3e}"
    assert_with_pcc(ref, got, OPERATOR_PCC)


@pytest.mark.parametrize("source", ["rule", "placement"])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("batch, seqlen", [(10, 384), (12, 384), (16, 384), (12, 352)])
def test_sdpa_where_three_query_chunks_a_head_are_wrong(device, tt_config, config, batch, seqlen, masked, source):
    """The model's chunks at batch shapes where a head split into three query chunks goes wrong.

    With explicit chunks SDPA also returns wrong rows, errors up to 0.9, in DRAM as in L1, when
    each head runs as three query chunks at 10 or more sequences: q128 with a whole-sequence key
    chunk at these shapes, which attention_placement once chose for its speed. At 8 and 9
    sequences the same pair is right. This pins both the rule's config and the placement's (with
    the operands where the placement puts them), which keep one query chunk a head here.
    """
    heads, head_dim = config.num_attention_heads, config.head_dim
    query, key, value = (torch.randn(batch, heads, seqlen, head_dim) for _ in range(3))
    kwargs = sdpa_kwargs(tt_config, config, batch, seqlen, masked=masked)
    memory = ttnn.DRAM_MEMORY_CONFIG
    if source == "placement":
        kwargs["program_config"], memory = attention_placement(
            batch,
            seqlen,
            heads,
            head_dim,
            tt_config.core_grid,
            kwargs["compute_kernel_config"].fp32_dest_acc_en,
            tt_config.attention_mask_dtype if masked else None,
            tt_config.l1_cb_bytes,
            tt_config.l1_banks,
        )
    mask = None
    if masked:
        mask = additive_attention_mask(
            torch.ones(batch, seqlen, dtype=torch.long), device, mask_dtype=tt_config.attention_mask_dtype
        )

    inputs = [to_device(t, device, memory_config=memory) for t in (query, key, value)]
    out = ttnn.transformer.scaled_dot_product_attention(
        *inputs, attn_mask=mask, is_causal=False, memory_config=memory, **kwargs
    )
    got = ttnn.to_torch(out).float()
    # Freed before asserting: a failed test's frame outlives it, and L1 tensors left in it would
    # clash with the next test's circular buffers.
    for tensor in (*inputs, out):
        ttnn.deallocate(tensor)

    ref = torch.nn.functional.scaled_dot_product_attention(query, key, value, is_causal=False)
    per_row = (got - ref).abs().amax(dim=(1, 2, 3))
    wrong = per_row.gt(0.1).nonzero().flatten().tolist()
    config_text = f"q{kwargs['program_config'].q_chunk_size} k{kwargs['program_config'].k_chunk_size}"
    assert not wrong, f"{config_text}: rows {wrong} are wrong, worst max abs {per_row.max():.3e}"
    assert_with_pcc(ref, got, OPERATOR_PCC)


@pytest.mark.parametrize("batch, seqlen", TOKEN_SHAPES)
def test_nlp_concat_heads(device, config, batch, seqlen):
    """aten.permute + aten.reshape -> ttnn.experimental.nlp_concat_heads."""
    heads, head_dim = config.num_attention_heads, config.head_dim
    context = torch.randn(batch, heads, seqlen, head_dim)

    out = ttnn.experimental.nlp_concat_heads(to_device(context, device))

    ref = context.permute(0, 2, 1, 3).reshape(batch, 1, seqlen, config.hidden_size)
    assert_with_pcc(ref, out, OPERATOR_PCC)
