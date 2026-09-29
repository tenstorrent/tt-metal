# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-operator PCC validation for everything outside attention and the MoE FFN.

Each test exercises one TTNN operator at this model's shapes, under the settings
tt/model_config.py will run it with, against the torch operator the reference uses. Clearing
these first means a later module failure is a composition bug, not a kernel surprise.

Weight operands come from the real checkpoint wherever dynamic range matters, since a bf16
matmul's error tracks the operand distribution and randn(0, 1) is not that distribution.
Activation-only operators use seeded random tensors and need no checkpoint.

Measured results are tabulated in docs/OPERATOR_MAPPING.md.
"""

import pytest
import torch

import ttnn

from models.common.metrics import compute_max_abs_error
from models.common.utility_functions import run_for_blackhole
from models.experimental.nomic_embed_text_v2_moe.common import TOKENIZER
from models.experimental.nomic_embed_text_v2_moe.reference.postprocessing import l2_normalize, mean_pool
from models.experimental.nomic_embed_text_v2_moe.tt.common import (
    flatten_tokens,
    to_device,
    transpose_linear_weight,
    unflatten_tokens,
)
from tests.ttnn.utils_for_testing import assert_with_pcc

pytestmark = [run_for_blackhole(), pytest.mark.use_module_device]

# Every operator in this file clears this at model shapes in bfloat16, all of them at 0.99999
# or better. SDPA, in the attention file, is the only one that comes close at 0.9998. Dropping
# the gate to accommodate a regression would hide the class of bug these tests exist to catch.
OPERATOR_PCC = 0.999

# (batch, seqlen). 37 is deliberately off-tile: padding bugs only surface when S is not a
# multiple of 32, and S is the batch's longest tokenized sequence, so that is the common case.
TOKEN_SHAPES = [(1, 128), (2, 512), (2, 37)]


def ids_tensor(batch: int, seqlen: int, device) -> tuple[torch.Tensor, ttnn.Tensor]:
    """Random token ids, as torch and as the uint32 ROW_MAJOR tensor ttnn.embedding indexes with."""
    ids = torch.randint(0, TOKENIZER.length, (batch, seqlen), dtype=torch.int64)
    return ids, ttnn.from_torch(ids.to(torch.uint32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)


# Embeddings.


@pytest.mark.needs_weights
@pytest.mark.parametrize("batch, seqlen", TOKEN_SHAPES)
def test_embedding(device, tt_config, state_dict, batch, seqlen):
    """aten.embedding -> ttnn.embedding, against the real 250048 x 768 table."""
    table = state_dict["embeddings.word_embeddings.weight"]
    ids, ids_tt = ids_tensor(batch, seqlen, device)

    out = ttnn.embedding(
        ids_tt,
        to_device(table, device, dtype=tt_config.weight_dtype, layout=ttnn.ROW_MAJOR_LAYOUT),
        layout=tt_config.layout,
        dtype=tt_config.activation_dtype,
    )

    assert_with_pcc(torch.nn.functional.embedding(ids, table), out, OPERATOR_PCC)


@pytest.mark.needs_weights
def test_embedding_reproduces_the_trained_pad_row(device, tt_config, config, state_dict):
    """The <pad> row is trained and non-zero, so it must survive the lookup.

    nn.Embedding(padding_idx=...) zeroes it at init only; loading the checkpoint overwrites it.
    Zeroing it here would silently change every padded batch.
    """
    table = state_dict["embeddings.word_embeddings.weight"]
    pad_ids = ttnn.from_torch(
        torch.full((1, ttnn.TILE_SIZE), config.pad_token_id, dtype=torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    out = ttnn.to_torch(
        ttnn.embedding(
            pad_ids,
            to_device(table, device, dtype=tt_config.weight_dtype, layout=ttnn.ROW_MAJOR_LAYOUT),
            layout=tt_config.layout,
            dtype=tt_config.activation_dtype,
        )
    ).float()

    assert table[config.pad_token_id].abs().max() > 0
    assert compute_max_abs_error(out, table[config.pad_token_id].expand_as(out)) < 1e-3


# Normalization.


@pytest.mark.parametrize("batch, seqlen", TOKEN_SHAPES)
def test_layer_norm(device, tt_config, config, batch, seqlen):
    """aten.native_layer_norm -> ttnn.layer_norm."""
    hidden = config.hidden_size
    x = torch.randn(1, 1, batch * seqlen, hidden)
    weight, bias = torch.randn(hidden), torch.randn(hidden)

    out = ttnn.layer_norm(
        to_device(x, device),
        weight=to_device(weight, device),
        bias=to_device(bias, device),
        epsilon=config.layer_norm_epsilon,
        compute_kernel_config=tt_config.compute_kernel_config,
    )

    ref = torch.nn.functional.layer_norm(x, (hidden,), weight, bias, eps=config.layer_norm_epsilon)
    assert_with_pcc(ref, out, OPERATOR_PCC)


@pytest.mark.parametrize("batch, seqlen", TOKEN_SHAPES)
def test_layer_norm_fuses_the_post_norm_residual(device, tt_config, config, batch, seqlen):
    """aten.add + aten.native_layer_norm -> one ttnn.layer_norm(residual_input_tensor=...).

    Every block is post-norm, norm(sub_block(x) + x), so this fused form is what all 24 norms
    in the encoder use.
    """
    hidden = config.hidden_size
    x = torch.randn(1, 1, batch * seqlen, hidden)
    residual = torch.randn(1, 1, batch * seqlen, hidden)
    weight, bias = torch.randn(hidden), torch.randn(hidden)

    out = ttnn.layer_norm(
        to_device(x, device),
        residual_input_tensor=to_device(residual, device),
        weight=to_device(weight, device),
        bias=to_device(bias, device),
        epsilon=config.layer_norm_epsilon,
        compute_kernel_config=tt_config.compute_kernel_config,
    )

    ref = torch.nn.functional.layer_norm(x + residual, (hidden,), weight, bias, eps=config.layer_norm_epsilon)
    assert_with_pcc(ref, out, OPERATOR_PCC)


# Projections.


@pytest.mark.needs_weights
@pytest.mark.parametrize(
    "key, in_dim_is_ffn",
    [
        ("encoder.layers.0.attn.Wqkv", False),
        ("encoder.layers.0.attn.out_proj", False),
        ("encoder.layers.0.mlp.fc1", False),
        ("encoder.layers.0.mlp.fc2", True),
    ],
)
def test_linear(device, tt_config, config, state_dict, key, in_dim_is_ffn):
    """aten.addmm -> ttnn.linear, at each of the four biased projection widths."""
    weight, bias = state_dict[key + ".weight"], state_dict[key + ".bias"]
    in_dim = config.intermediate_size if in_dim_is_ffn else config.hidden_size
    x = torch.randn(1, 1, 512, in_dim)

    out = ttnn.linear(
        to_device(x, device),
        to_device(transpose_linear_weight(weight), device, dtype=tt_config.weight_dtype),
        bias=to_device(bias, device, dtype=tt_config.weight_dtype),
        compute_kernel_config=tt_config.compute_kernel_config,
    )

    assert_with_pcc(torch.nn.functional.linear(x, weight, bias), out, OPERATOR_PCC)


# Activation.


@pytest.mark.parametrize("batch, seqlen", TOKEN_SHAPES)
def test_gelu(device, config, batch, seqlen):
    """aten.gelu -> ttnn.gelu, exact erf, at the dense FFN's intermediate width."""
    x = torch.randn(1, 1, batch * seqlen, config.intermediate_size)

    out = ttnn.gelu(to_device(x, device))

    assert_with_pcc(torch.nn.functional.gelu(x, approximate="none"), out, OPERATOR_PCC)


def test_gelu_fast_mode_is_worse_than_the_bfloat16_noise_floor(device, config):
    """Negative control: the LUT variant's error is not swamped by bf16 rounding.

    Run in fp32, where the two are separable at all. The repo's BERT idiom
    fused_activation=(ttnn.UnaryOpType.GELU, True) selects the LUT, which is why the port does
    not copy it.
    """
    x = torch.randn(1, 1, 512, config.intermediate_size)
    ref = torch.nn.functional.gelu(x, approximate="none")
    x_tt = to_device(x, device, dtype=ttnn.float32)

    accurate = compute_max_abs_error(ttnn.to_torch(ttnn.gelu(x_tt)).float(), ref)
    approximate = compute_max_abs_error(ttnn.to_torch(ttnn.gelu(x_tt, fast_and_approximate_mode=True)).float(), ref)

    assert accurate < 1e-5, f"expected the accurate variant to be exact to 1e-5 in fp32, got {accurate:.3e}"
    assert approximate > 1e-2, f"expected the LUT to be visibly worse, got {approximate:.3e}"


# Layout.


@pytest.mark.parametrize("batch, seqlen", TOKEN_SHAPES)
def test_token_axis_reshape_round_trip_is_exact(device, config, batch, seqlen):
    """aten.view -> ttnn.reshape, between the two activation layouts.

    Exactness matters because this runs twice per block; a lossy reshape would compound over 12
    blocks in a way a per-operator PCC gate would not catch.
    """
    x = to_device(torch.randn(batch, 1, seqlen, config.hidden_size), device)

    flat = flatten_tokens(x)
    assert tuple(flat.shape) == (1, 1, batch * seqlen, config.hidden_size)
    assert torch.equal(ttnn.to_torch(unflatten_tokens(flat, batch, seqlen)), ttnn.to_torch(x))


def test_typecast(device, config):
    """aten._to_copy -> ttnn.typecast, the fp32-to-bf16 step the router needs before scatter."""
    x = torch.randn(1, 1, 512, config.hidden_size)

    out = ttnn.typecast(to_device(x, device, dtype=ttnn.float32), ttnn.bfloat16)

    assert out.dtype == ttnn.bfloat16
    assert_with_pcc(x, out, OPERATOR_PCC)


# Pooling and output, the tensor half of reference/postprocessing.py.


@pytest.mark.parametrize("batch, seqlen", TOKEN_SHAPES)
def test_mean_pool_excludes_padding(device, config, batch, seqlen):
    """Mask-weighted mean over the sequence axis: ttnn.mul, ttnn.sum, ttnn.div.

    Padding has to be excluded because the <pad> embedding is non-zero, so counting it would
    make a text's embedding depend on its batch-mates.
    """
    hidden = config.hidden_size
    x = torch.randn(batch, 1, seqlen, hidden)
    mask = torch.ones(batch, seqlen, dtype=torch.long)
    mask[0, (seqlen * 3) // 4 :] = 0

    x_tt = to_device(x, device)
    mask_tt = to_device(mask.reshape(batch, 1, seqlen, 1).float(), device)
    pooled = ttnn.divide(
        ttnn.sum(ttnn.multiply(x_tt, mask_tt), dim=2, keepdim=True), ttnn.sum(mask_tt, dim=2, keepdim=True)
    )

    ref = mean_pool(x.squeeze(1), mask).reshape(batch, 1, 1, hidden)
    assert_with_pcc(ref, pooled, OPERATOR_PCC)


@pytest.mark.parametrize("dim", [768, 512, 256, 128])
def test_matryoshka_truncate(device, config, dim):
    """aten.slice -> ttnn.slice, on the feature axis.

    Upstream's matryoshka_dim slices the sequence axis instead, which is a different operation
    and not what the published embeddings use.
    """
    batch = 2
    x = torch.randn(batch, 1, 1, config.hidden_size)

    out = ttnn.slice(to_device(x, device), [0, 0, 0, 0], [batch, 1, 1, dim])

    assert tuple(out.shape) == (batch, 1, 1, dim)
    assert_with_pcc(x[..., :dim], out, OPERATOR_PCC)


@pytest.mark.parametrize("batch", [1, 2])
def test_l2_normalize(device, config, batch):
    """Unit-norm the pooled embedding: ttnn.mul, ttnn.sum, ttnn.rsqrt.

    Runs after pooling, so the sequence axis is already gone and only batch varies.
    """
    x = torch.randn(batch, 1, 1, config.hidden_size)

    x_tt = to_device(x, device)
    out = ttnn.multiply(x_tt, ttnn.rsqrt(ttnn.sum(ttnn.multiply(x_tt, x_tt), dim=-1, keepdim=True)))

    assert_with_pcc(l2_normalize(x), out, OPERATOR_PCC)


# Device configuration.


def test_core_grid_is_read_from_the_device(device, tt_config):
    """The grid must come from the device, not a constant copied out of another model.

    models/tt_transformers/tt/model_config.py implies (8, 10); this p300c reports 11x10, and a
    program config sized for the smaller grid would silently leave cores idle.
    """
    grid = device.compute_with_storage_grid_size()

    assert (tt_config.core_grid.x, tt_config.core_grid.y) == (grid.x, grid.y)
