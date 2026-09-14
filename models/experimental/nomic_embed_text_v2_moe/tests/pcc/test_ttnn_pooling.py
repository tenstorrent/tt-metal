# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Module PCC for tt/pooling.py, the tensor half of reference/postprocessing.py. Gate 10.

No weights here, so these run without the checkpoint. The whole three-step pipeline is measured
as well as each step, since that is what the end-to-end model calls.
"""

import pytest
import torch

import ttnn

from models.common.utility_functions import run_for_blackhole
from models.experimental.nomic_embed_text_v2_moe.reference import postprocessing
from models.experimental.nomic_embed_text_v2_moe.tests.pcc.module_common import (
    TOKEN_SHAPES,
    hidden_states,
    keep_mask,
    to_block_layout,
)
from models.experimental.nomic_embed_text_v2_moe.tt import pooling
from models.experimental.nomic_embed_text_v2_moe.tt.common import pooling_mask, to_device
from tests.ttnn.utils_for_testing import assert_with_pcc

pytestmark = [run_for_blackhole(), pytest.mark.use_module_device]

MODULE_PCC = 0.999


@pytest.mark.parametrize("batch, seqlen", TOKEN_SHAPES)
def test_mean_pool(device, config, batch, seqlen):
    """Mask-weighted mean over the sequence axis, against reference.mean_pool."""
    x = hidden_states(batch, seqlen, config.hidden_size)
    mask = keep_mask(batch, seqlen, (seqlen * 3) // 4)

    pooled = pooling.mean_pool(to_device(to_block_layout(x), device), pooling_mask(mask, device))

    ref = postprocessing.mean_pool(x, mask)
    assert tuple(pooled.shape) == (batch, 1, 1, config.hidden_size)
    assert_with_pcc(ref.reshape(batch, 1, 1, config.hidden_size), pooled, MODULE_PCC)


@pytest.mark.parametrize("batch, seqlen", TOKEN_SHAPES)
def test_mean_pool_ignores_what_padding_holds(device, config, batch, seqlen):
    """A text's embedding must not depend on its batch-mates.

    The <pad> embedding is trained and non-zero, so the mask is the only thing keeping padded
    positions out of the mean. Replacing the padded rows with arbitrary values must not move the
    result at all.
    """
    keep = (seqlen * 3) // 4
    mask = keep_mask(batch, seqlen, keep)
    x = hidden_states(batch, seqlen, config.hidden_size)
    perturbed = x.clone()
    perturbed[:, keep:] = 7.0

    mask_tt = pooling_mask(mask, device)
    pooled = pooling.mean_pool(to_device(to_block_layout(x), device), mask_tt)
    pooled_perturbed = pooling.mean_pool(to_device(to_block_layout(perturbed), device), mask_tt)

    assert torch.equal(ttnn.to_torch(pooled), ttnn.to_torch(pooled_perturbed))


@pytest.mark.parametrize("dim", [768, 512, 256, 128])
def test_matryoshka_truncate(device, config, dim):
    """Feature-axis truncation, against reference.matryoshka_truncate."""
    batch = 2
    x = torch.randn(batch, 1, 1, config.hidden_size)

    out = pooling.matryoshka_truncate(to_device(x, device), dim)

    assert tuple(out.shape) == (batch, 1, 1, dim)
    assert_with_pcc(postprocessing.matryoshka_truncate(x, dim), out, MODULE_PCC)


def test_matryoshka_truncate_passes_none_through(device, config):
    """None means full width, and must not cost a device op."""
    x = to_device(torch.randn(2, 1, 1, config.hidden_size), device)

    assert pooling.matryoshka_truncate(x, None) is x
    assert pooling.matryoshka_truncate(x, config.hidden_size) is x


def test_matryoshka_truncate_rejects_an_oversized_dim(device, config, expect_error):
    """Widening is not truncation; ttnn.slice would otherwise raise something less obvious."""
    x = to_device(torch.randn(2, 1, 1, config.hidden_size), device)

    with expect_error(ValueError, "exceeds embedding width"):
        pooling.matryoshka_truncate(x, config.hidden_size + 1)


@pytest.mark.parametrize("batch", [1, 2])
def test_l2_normalize(device, config, batch):
    """Unit norm along the feature axis, against reference.l2_normalize."""
    x = torch.randn(batch, 1, 1, config.hidden_size)

    out = pooling.l2_normalize(to_device(x, device))

    assert_with_pcc(postprocessing.l2_normalize(x), out, MODULE_PCC)


@pytest.mark.parametrize("dim", [None, 256])
@pytest.mark.parametrize("batch, seqlen", TOKEN_SHAPES)
def test_pool_truncate_normalize(device, config, batch, seqlen, dim):
    """The three steps in sequence, which is what the end-to-end model calls.

    The norm is asserted separately from PCC: PCC is insensitive to a uniform scale, so a
    normalization that is off by a constant factor would still correlate.
    """
    x = hidden_states(batch, seqlen, config.hidden_size)
    mask = keep_mask(batch, seqlen, (seqlen * 3) // 4)
    width = dim or config.hidden_size

    pooled = pooling.mean_pool(to_device(to_block_layout(x), device), pooling_mask(mask, device))
    embeddings = pooling.l2_normalize(pooling.matryoshka_truncate(pooled, dim))

    ref = postprocessing.l2_normalize(postprocessing.matryoshka_truncate(postprocessing.mean_pool(x, mask), dim))
    got = ttnn.to_torch(embeddings).float().reshape(batch, width)
    assert torch.allclose(got.norm(dim=-1), torch.ones(batch), atol=1e-2)
    assert_with_pcc(ref.reshape(batch, 1, 1, width), embeddings, MODULE_PCC)
