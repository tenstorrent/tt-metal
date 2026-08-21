# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Coverage for EmbeddingsType.PADDED / BINARY on the two Metal 2.0 embedding factories.

Scratch diagnostic for the Quasar uplift; untracked, delete before merge (the proper fix is
to parametrize the existing tests over embeddings_type). It has to live under the repo tree
rather than in a temp directory so that the root conftest.py's `device` fixture and
tests/ttnn/conftest.py's parametrize-id hook both apply.

It exists because nothing in
tests/ttnn/unit_tests/operations/data_movement/test_embedding.py passes an embeddings_type
other than GENERIC, and every PADDED/BINARY call site in the repo
(models/demos/metal_BERT_large_11/tt/embeddings.py) passes layout=TILE_LAYOUT with a
row-major index tensor, which dispatches EmbeddingsFusedProgramFactory and therefore the
*legacy* embeddings_common.hpp. So the PADDED and BINARY branches of the Metal 2.0 fork,
embeddings_common_metal2.hpp, are never compiled by any test.

The two paths covered here:
  - rm       -> EmbeddingsRMProgramFactory            (row-major indices, row-major output)
  - tilized  -> EmbeddingsTilizedIndicesProgramFactory (tiled indices, row-major output)

Reference: PADDED and BINARY change only *where* a weight row is fetched from (a local SRAM
cache instead of a per-token DRAM read), not which row. So the golden for both is a plain
embedding lookup, same as GENERIC.
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_equal


def _indices(batch, sentence, vocab, indices_layout):
    torch_idx = torch.randint(0, vocab, (batch, sentence))
    return torch_idx, ttnn.from_torch(torch_idx, dtype=ttnn.uint32, layout=indices_layout)


@pytest.mark.parametrize("path", ["rm", "tilized"])
@pytest.mark.parametrize(
    "embeddings_type",
    [
        # GENERIC is the control: it exercises the same two program factories and the same
        # index scratchpad, but takes no local weight cache. GENERIC passing while PADDED or
        # BINARY fails points at the weight cache specifically.
        ttnn.EmbeddingsType.GENERIC,
        ttnn.EmbeddingsType.PADDED,
        ttnn.EmbeddingsType.BINARY,
    ],
)
@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("sentence_size", [32, 256])
@pytest.mark.parametrize("hidden_embedding_dim", [768])
def test_embedding_padded_binary(device, path, embeddings_type, batch_size, sentence_size, hidden_embedding_dim):
    torch.manual_seed(1234)

    indices_layout = ttnn.TILE_LAYOUT if path == "tilized" else ttnn.ROW_MAJOR_LAYOUT

    if embeddings_type == ttnn.EmbeddingsType.BINARY:
        # BINARY requires exactly two embedding rows (validate_on_program_cache_miss asserts
        # weights.padded_shape()[-2] == 2) and caches both of them.
        vocab = 2
    else:
        vocab = 2048

    # A pad token is accepted only by PADDED; the other types reject a non-null one.
    pad_token = 7 if embeddings_type == ttnn.EmbeddingsType.PADDED else None

    torch_idx, tt_idx = _indices(batch_size, sentence_size, vocab, indices_layout)
    torch_weights = torch.rand((vocab, hidden_embedding_dim), dtype=torch.bfloat16) * 0.2 - 0.1

    torch_output = torch.nn.functional.embedding(torch_idx, torch_weights)

    tt_idx = ttnn.to_device(tt_idx, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tt_weights = ttnn.to_device(
        ttnn.from_torch(torch_weights, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT),
        device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_output = ttnn.embedding(
        tt_idx,
        tt_weights,
        padding_idx=pad_token,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        embeddings_type=embeddings_type,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # An embedding is a pure gather: the output rows are copies of weight rows, so the
    # comparison is exact, as in the GENERIC tests in test_embedding.py.
    assert_equal(torch_output, ttnn.to_torch(tt_output))
