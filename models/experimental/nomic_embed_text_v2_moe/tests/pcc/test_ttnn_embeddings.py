# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Module PCC for TtNomicBertEmbeddings. Bring-up gate 1.

Two things beyond PCC: the token-type fold has to be exact rather than close, since it changes
the table every lookup reads, and the trained <pad> row has to survive the lookup.
"""

import pytest
import torch

import ttnn

from models.common.utility_functions import run_for_blackhole
from models.experimental.nomic_embed_text_v2_moe.common import random_input_ids
from models.experimental.nomic_embed_text_v2_moe.reference.modeling_nomic_moe import NomicBertEmbeddings
from models.experimental.nomic_embed_text_v2_moe.tests.pcc.module_common import (
    TOKEN_SHAPES,
    from_block_layout,
    load_reference,
)
from models.experimental.nomic_embed_text_v2_moe.tt.common import prepare_token_ids
from models.experimental.nomic_embed_text_v2_moe.tt.embeddings import TtNomicBertEmbeddings
from tests.ttnn.utils_for_testing import assert_with_pcc

pytestmark = [run_for_blackhole(), pytest.mark.use_module_device, pytest.mark.needs_weights]

MODULE_PCC = 0.999

PREFIX = "embeddings."


@pytest.fixture
def reference(config, state_dict):
    return load_reference(lambda: NomicBertEmbeddings(config), state_dict, PREFIX)


@pytest.fixture
def tt_embeddings(device, config, tt_config, state_dict):
    return TtNomicBertEmbeddings(device, config, tt_config, state_dict, PREFIX)


@pytest.mark.parametrize("batch, seqlen", TOKEN_SHAPES)
def test_embeddings(device, config, reference, tt_embeddings, batch, seqlen):
    """Token ids to (B, 1, S, H) vectors, against the reference module."""
    input_ids, _ = random_input_ids(batch, seqlen, config)

    out = tt_embeddings(prepare_token_ids(input_ids, device))

    with torch.no_grad():
        ref = reference(input_ids)
    assert tuple(out.shape) == (batch, 1, seqlen, config.hidden_size)
    assert_with_pcc(ref, from_block_layout(out), MODULE_PCC)


def test_token_type_fold_is_exact(config, reference, state_dict):
    """The fold is host arithmetic, so it must be bit-exact, not merely close.

    type_vocab_size is 1, so token_type_ids can only select row 0 and the reference adds that
    one row to every token. Folding it into the word table collapses two lookups and an add
    into one lookup. Asserted in fp32 against both the default token_type_ids and an explicit
    all-zeros tensor, which are the only two legal inputs.
    """
    input_ids, _ = random_input_ids(2, 64, config)
    folded = state_dict[PREFIX + "word_embeddings.weight"] + state_dict[PREFIX + "token_type_embeddings.weight"][0]

    with torch.no_grad():
        default_ids = reference(input_ids)
        explicit_zeros = reference(input_ids, token_type_ids=torch.zeros(input_ids.shape[1], dtype=torch.long))

    assert torch.equal(default_ids, torch.nn.functional.embedding(input_ids, folded))
    assert torch.equal(explicit_zeros, default_ids)


def test_pad_row_survives_the_lookup(device, config, tt_embeddings, state_dict):
    """The <pad> row is trained and non-zero, so the module must not zero it.

    nn.Embedding(padding_idx=...) zeroes it at init only, and loading the checkpoint overwrites
    it. Padding is excluded at the attention mask and at pooling instead; zeroing it here would
    silently change every padded batch.
    """
    pad_ids = torch.full((1, ttnn.TILE_SIZE), config.pad_token_id, dtype=torch.int64)
    expected = (
        state_dict[PREFIX + "word_embeddings.weight"][config.pad_token_id]
        + state_dict[PREFIX + "token_type_embeddings.weight"][0]
    )

    out = from_block_layout(tt_embeddings(prepare_token_ids(pad_ids, device)))

    # Exact, not approximate: a lookup copies a row, so the only error is the bfloat16 rounding
    # the table already took at load. A tolerance here would have to be scaled to the row's
    # dynamic range, and would then be loose enough to pass on a zeroed row at small widths.
    assert state_dict[PREFIX + "word_embeddings.weight"][config.pad_token_id].abs().max() > 0
    assert expected.abs().max() > 0
    assert torch.equal(out, expected.to(torch.bfloat16).float().expand_as(out))
