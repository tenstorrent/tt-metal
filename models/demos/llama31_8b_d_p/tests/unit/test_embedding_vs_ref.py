# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""`tt/embedding.py` vs the torch reference, `(1,1)` mesh, TP=1, no CCL. Gate `G-CLEAN` item 9.

**Why this file exists.** P9's test-inventory sweep found `tt/embedding.py` and `tt/lm_head.py` were
the only two `tt/` modules with no test naming them: both were covered *only* transitively, through
`tt/model.py` in `tests/unit/test_model_vs_ref.py`. That is real coverage — `G-MODEL` scores the
whole stack at PCC 0.9997646 with top-1 agreement, which an embedding bug would break — but it is
end-to-end coverage, and a table-lookup failure there arrives as "the model is slightly off"
rather than as "the gather is wrong". `DEC-122`.

**The reference is exactness, not PCC.** `ttnn.embedding` is a *gather*: it copies rows, it does not
compute. So the honest threshold is `torch.equal` against `bfloat16(table)[tokens]` — the same rows,
in the same dtype the module stores. A PCC threshold here would be strictly weaker and would hide an
off-by-one in the row index behind 0.9999-something.

Two deliberate deviations from the module's deployment configuration, both so the test is cheap:

* **`vocab_size` is reduced to 8192** via `dataclasses.replace`. The real table is
  `128256 x 4096 x 4 B = 2.0 GiB` in fp32 on the host, which is not worth allocating to prove a
  gather. 8192 is not a multiple of 4096 either, so a `[vocab, hidden]` / `[hidden, vocab]` mix-up
  cannot pass by accident.
* Random weights, seeded by `reset_seeds` — the module reads no value from the table, only indexes
  it, so real weights would add nothing.

Run:
    pytest models/demos/llama31_8b_d_p/tests/unit/test_embedding_vs_ref.py -x -q
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.llama31_8b_d_p.tests.test_factory import TestFactory
from models.demos.llama31_8b_d_p.tt.embedding import Embedding

# Small enough to build on the host, not a multiple of hidden_size (4096), and tile-aligned.
TEST_VOCAB = 8192


def _small_vocab_config(hf_config):
    return replace(hf_config, vocab_size=TEST_VOCAB)


def _tokens_to_device(mesh_device, token_ids):
    """`[1, 1, 1, S]` uint32 ROW_MAJOR — the layout `tt/model.py` and the H2D producer deliver."""
    return ttnn.from_torch(
        token_ids.reshape(1, 1, 1, -1),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


@pytest.mark.parametrize("mesh_device", [TestFactory.SINGLE_CARD_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("seq_len", [32, 512], ids=["s32", "s512"])
@torch.no_grad()
def test_embedding_gathers_the_right_rows(mesh_device, seq_len, reset_seeds):
    """`[1,1,1,S]` uint32 -> `[1,1,S,4096]` bf16, **bit-exact** vs `bfloat16(table)[tokens]`."""
    objs = TestFactory.setup_test(mesh_device)
    hf_config = _small_vocab_config(objs["hf_config"])
    hidden = hf_config.hidden_size

    table = torch.randn(TEST_VOCAB, hidden, dtype=torch.float32)
    token_ids = torch.randint(0, TEST_VOCAB, (seq_len,), dtype=torch.int32)

    emb = Embedding(mesh_device, hf_config, {"weight": table}, mesh_config=objs["mesh_config"])
    assert emb.weight.dtype == ttnn.bfloat16, (
        f"the embedding table is stored as {emb.weight.dtype}, not bfloat16. It seeds the residual "
        f"stream; bfloat8_b's per-tile shared exponent crushes small channels (DEC-015)."
    )
    assert emb.weight.layout == ttnn.ROW_MAJOR_LAYOUT, "ttnn.embedding gathers from an untilized table"

    tt_tokens = _tokens_to_device(mesh_device, token_ids)
    out = ttnn.to_torch(emb(tt_tokens), mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:1]

    assert tuple(out.shape) == (1, 1, seq_len, hidden), f"got {tuple(out.shape)}"
    assert out.dtype == torch.bfloat16, f"output dtype is {out.dtype}, expected bfloat16"

    # The gather copies rows: the reference is the bf16 cast of exactly those rows.
    ref = table.to(torch.bfloat16)[token_ids.long()].reshape(1, 1, seq_len, hidden)
    mismatched = int((out != ref).sum())
    logger.info(f"[embedding] seq_len={seq_len}: mismatched elements {mismatched} / {ref.numel()} (must be 0)")
    assert torch.equal(out, ref), (
        f"[embedding] {mismatched}/{ref.numel()} elements differ from bfloat16(table)[tokens]. "
        f"A gather has no arithmetic error, so any difference is a wrong row index or a wrong dtype."
    )


@pytest.mark.parametrize("mesh_device", [TestFactory.SINGLE_CARD_MESH_SHAPE], indirect=True)
@torch.no_grad()
def test_embedding_negative_control_shuffled_tokens(mesh_device, reset_seeds):
    """The exactness above must be *specific* to these tokens, not true of any tokens.

    Without this, a module that returned (say) row 0 for everything, or the tokens in the wrong
    order, would still have to be caught by the test above — and it would be, but only because the
    reference happens to disagree. Making the disagreement an explicit assertion states the
    intent: the row index is load-bearing.
    """
    objs = TestFactory.setup_test(mesh_device)
    hf_config = _small_vocab_config(objs["hf_config"])
    hidden = hf_config.hidden_size

    table = torch.randn(TEST_VOCAB, hidden, dtype=torch.float32)
    token_ids = torch.randint(0, TEST_VOCAB, (64,), dtype=torch.int32)
    shuffled = token_ids[torch.randperm(64)]
    assert not torch.equal(token_ids, shuffled), "the permutation left the order unchanged; reseed"

    emb = Embedding(mesh_device, hf_config, {"weight": table}, mesh_config=objs["mesh_config"])
    out = ttnn.to_torch(
        emb(_tokens_to_device(mesh_device, token_ids)), mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    )[:1]

    wrong_ref = table.to(torch.bfloat16)[shuffled.long()].reshape(1, 1, 64, hidden)
    logger.info(f"[embedding] shuffled-token control: equal = {torch.equal(out, wrong_ref)} (must be False)")
    assert not torch.equal(out, wrong_ref), "the output matched a *shuffled* token order — the gather ignores the index"


@pytest.mark.parametrize("mesh_device", [TestFactory.SINGLE_CARD_MESH_SHAPE], indirect=True)
def test_embedding_refusals(mesh_device, expect_error, reset_seeds):
    """Every refusal in the constructor, matched on its message.

    A silently-wrong embedding table is one of the cheapest ways to get a plausible-but-wrong model,
    so each of these is a loud failure by design (`DEC-038` for the cache-only case).
    """
    objs = TestFactory.setup_test(mesh_device)
    hf_config = _small_vocab_config(objs["hf_config"])
    table = torch.randn(TEST_VOCAB, hf_config.hidden_size, dtype=torch.float32)

    # 1. the whole state dict instead of the stripped sub-dict
    with expect_error(AssertionError, "stripped sub-dict"):
        Embedding(
            mesh_device,
            hf_config,
            {"model.embed_tokens.weight": table},
            mesh_config=objs["mesh_config"],
        )

    # 2. a table whose shape disagrees with hf_config (here: the real vocab against the small config)
    with expect_error(AssertionError, "embedding table is"):
        Embedding(
            mesh_device,
            hf_config,
            {"weight": torch.randn(TEST_VOCAB // 2, hf_config.hidden_size)},
            mesh_config=objs["mesh_config"],
        )

    # 3. cache-only mode with no cache to read from
    with expect_error(AssertionError, "no tensor_cache_path"):
        Embedding(mesh_device, hf_config, {}, mesh_config=objs["mesh_config"])
