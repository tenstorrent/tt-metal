# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ttml.ops.multi_head_utils.split_heads.

split_heads takes a single separately-projected activation and splits its feature axis into
heads: (B, 1, S, num_heads * head_dim) -> (B, num_heads, S, head_dim). It is the inverse of
heads_fusion, and unlike heads_creation / grouped_heads_creation it needs no fused QKV tensor,
so Q and K/V may have different sequence lengths.

Both directions are pure data movement, so every numeric check here is exact.
"""

import numpy as np
import pytest

import ttnn
import ttml


# (batch, seq_len, num_heads, head_dim). head_dim is a multiple of 32 and seq_len a multiple
# of 32, which is the whole domain the op accepts.
SHAPES = [
    (1, 128, 8, 64),
    (2, 64, 4, 32),
    (1, 96, 8, 32),  # seq_len tile-aligned but not a power of two
    (1, 32, 1, 64),
    (2, 32, 16, 32),
]


@pytest.fixture(autouse=True)
def reset_graph():
    """The autograd context is a process-wide singleton, so drop each test's graph after it."""
    yield
    ttml.autograd.AutoContext.get_instance().reset_graph()


def make_tensor(*shape, seed):
    """A ttml autograd Tensor of the given shape, tile layout, bfloat16 on device.

    Stored as bfloat16 rather than the float32 default so that the values read back match the
    ones the op saw, which is what lets the comparisons below be exact rather than tolerant.
    from_numpy defaults requires_grad to False, which would leave split_heads with no backward
    node at all, so set it here.
    """
    rng = np.random.default_rng(seed)
    data = rng.standard_normal(shape).astype(np.float32)
    tensor = ttml.autograd.Tensor.from_numpy(data, layout=ttnn.Layout.TILE, new_type=ttnn.DataType.BFLOAT16)
    tensor.set_requires_grad(True)
    return tensor


def as_numpy(tensor):
    """Device values as float32. Already bfloat16-rounded, so comparisons can be exact."""
    return tensor.to_numpy(ttnn.DataType.FLOAT32)


def reference_split(x, num_heads):
    """(B, 1, S, H*D) -> (B, S, H, D) -> (B, H, S, D)."""
    batch, _, seq_len, embedding_dim = x.shape
    return x.reshape(batch, seq_len, num_heads, embedding_dim // num_heads).transpose(0, 2, 1, 3)


def reference_fuse(grad):
    """(B, H, S, D) -> (B, S, H, D) -> (B, 1, S, H*D). The split run backwards."""
    batch, num_heads, seq_len, head_dim = grad.shape
    return grad.transpose(0, 2, 1, 3).reshape(batch, 1, seq_len, num_heads * head_dim)


@pytest.mark.parametrize("batch, seq_len, num_heads, head_dim", SHAPES)
def test_forward_matches_reference(batch, seq_len, num_heads, head_dim):
    x = make_tensor(batch, 1, seq_len, num_heads * head_dim, seed=1001)

    out = ttml.ops.multi_head_utils.split_heads(x, num_heads)

    assert out.shape() == [batch, num_heads, seq_len, head_dim]
    np.testing.assert_array_equal(as_numpy(out), reference_split(as_numpy(x), num_heads))


@pytest.mark.parametrize("batch, seq_len, num_heads, head_dim", SHAPES)
def test_backward_matches_reference(batch, seq_len, num_heads, head_dim):
    x = make_tensor(batch, 1, seq_len, num_heads * head_dim, seed=2002)
    out = ttml.ops.multi_head_utils.split_heads(x, num_heads)

    # An arbitrary (not uniform) gradient: a mean-reduction loss would give every element the
    # same gradient and so could not tell a correct un-split from a transposed one.
    grad = make_tensor(batch, num_heads, seq_len, head_dim, seed=3003)
    out.set_grad_from_tensor(grad)
    out.backward(False)

    x_grad = as_numpy(x.get_grad_tensor())
    assert list(x_grad.shape) == [batch, 1, seq_len, num_heads * head_dim]
    np.testing.assert_array_equal(x_grad, reference_fuse(as_numpy(grad)))


@pytest.mark.parametrize("batch, seq_len, num_heads, head_dim", SHAPES)
def test_round_trips_through_heads_fusion(batch, seq_len, num_heads, head_dim):
    x = make_tensor(batch, 1, seq_len, num_heads * head_dim, seed=4004)

    fused = ttml.ops.multi_head_utils.heads_fusion(ttml.ops.multi_head_utils.split_heads(x, num_heads))

    assert fused.shape() == x.shape()
    np.testing.assert_array_equal(as_numpy(fused), as_numpy(x))


def test_splits_query_and_key_value_of_different_sequence_lengths():
    """The cross-attention case heads_creation and grouped_heads_creation cannot express."""
    num_heads, head_dim = 8, 64
    q = make_tensor(1, 1, 128, num_heads * head_dim, seed=5005)
    k = make_tensor(1, 1, 96, num_heads * head_dim, seed=6006)

    q_heads = ttml.ops.multi_head_utils.split_heads(q, num_heads)
    k_heads = ttml.ops.multi_head_utils.split_heads(k, num_heads)

    assert q_heads.shape() == [1, num_heads, 128, head_dim]
    assert k_heads.shape() == [1, num_heads, 96, head_dim]
    np.testing.assert_array_equal(as_numpy(q_heads), reference_split(as_numpy(q), num_heads))
    np.testing.assert_array_equal(as_numpy(k_heads), reference_split(as_numpy(k), num_heads))


@pytest.mark.parametrize(
    "shape, num_heads, message",
    [
        ((1, 2, 32, 64), 2, "but dim 1 is 2"),
        ((1, 1, 32, 512), 0, "num_heads to be positive"),
        ((1, 1, 32, 512), 7, "divisible by num_heads"),
        # head_dim = 96 / 4 = 24 < 32. The kernel would compute head_dim / 32 = 0 tiles, write
        # nothing, and hand back a tile-padded (1, 4, 32, 32) tensor of uninitialised memory.
        ((1, 1, 32, 96), 4, "to be a multiple of 32"),
        ((1, 1, 32, 128), 8, "to be a multiple of 32"),
    ],
)
def test_rejects_invalid_arguments(shape, num_heads, message, expect_error):
    x = make_tensor(*shape, seed=7007)
    with expect_error(ValueError, message):
        ttml.ops.multi_head_utils.split_heads(x, num_heads)


def test_rejects_sharded_input(expect_error):
    """nlp_create_qkv_heads picks its program factory from the input alone, and the sharded
    factory divides by num_kv_heads, which split_heads passes as 0. The op must refuse the
    input rather than let that reach the kernel."""
    batch, seq_len, num_heads, head_dim = 1, 128, 8, 64
    embedding_dim = num_heads * head_dim

    x = make_tensor(batch, 1, seq_len, embedding_dim, seed=8008)
    sharded = ttnn.to_memory_config(
        x.get_value(),
        ttnn.create_sharded_memory_config(
            shape=(batch, 1, seq_len, embedding_dim),
            core_grid=ttnn.CoreGrid(y=1, x=4),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    assert sharded.is_sharded()

    with expect_error(ValueError, "does not support sharded inputs"):
        ttml.ops.multi_head_utils.split_heads(ttml.autograd.create_tensor(sharded, requires_grad=True), num_heads)
