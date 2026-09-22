# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""M2 row: the parallel token embedding vs ``torch.nn.functional.embedding``, both layouts.

Full width and full vocab on the target mesh: ``[131072, 12288]``, SP=8 x TP=4. The reference is a
plain gather, so unlike every other device row in this package the bar here is **bit-exact**, not a
PCC. A lookup has no arithmetic to lose precision in: the 1D path reads the row and all-gathers the
hidden shards, and the 2D path adds one row of zeros to it (see :mod:`~...tt.embedding`), which in
bf16 is still exactly the row. Anything short of equality means a token landed on the wrong row or
a shard was assembled in the wrong order, and both of those are the failures a PCC would smear out
— a handful of wrong rows out of 2048 still PCCs above 0.99.

The two layouts are tested against the same reference *and* against each other. They were supposed
to be interchangeable (the toggle exists for memory, not for numerics). **They are not**, and every
test in this file passed while that was false, which is why :func:`test_embed_real_prompt_ids` was
added at the bottom.

The 2D path corrupts local sequence indices 1088..1099 of every SP shard (see
:mod:`~...tt.embedding`). ``SEQ`` here is 2048, so each of the 8 shards is 256 rows long and index
1088 does not exist — the window opens only above ``SEQ = 8704``. The rows above therefore keep
passing on 2D and are left alone: they still pin the gather, the shard assembly and the corner ids,
and they are honest at the length they run. What they cannot do is see a defect that is positional
and lives past the end of their sequence, so the new row runs the acceptance length on the real
prompt and xfails 2D.

Memory note: the table is 3.2 GB of host bfloat16 before it is sharded onto the mesh. That is why
this row builds it once per module rather than once per test, and why it is the one block test that
is expensive to *set up* rather than to run.
"""

import os

import pytest
import torch

from models.demos.mistral_medium_3_5_128b.reference.modeling import REF_DTYPE
from models.demos.mistral_medium_3_5_128b.tests.device_utils import from_mesh_sp
from models.demos.mistral_medium_3_5_128b.tt.embedding import ParallelEmbedding
from models.demos.mistral_medium_3_5_128b.tt.model import shard_tokens

SEQ = 2048  # multiple of TILE_SIZE * sp = 256


@pytest.fixture(scope="module")
def table(cfg):
    """``[vocab_size, hidden_size]`` bf16, small-scale like a real embedding."""
    torch.manual_seed(0)
    return (torch.randn(cfg.vocab_size, cfg.hidden_size, dtype=torch.float32) * 0.02).to(REF_DTYPE)


@pytest.fixture(scope="module")
def token_ids(cfg):
    """Ids spread over the whole vocab, plus the four corners every sharding scheme gets wrong.

    ``0`` and ``vocab_size - 1`` are the ends of the table; ``vocab_local`` and ``vocab_local - 1``
    straddle the first 2D shard boundary, which is exactly where the sentinel shift and clamp have
    to switch from "this row owns it" to "this row does not".
    """
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, cfg.vocab_size, (1, SEQ), generator=g, dtype=torch.int32)
    vocab_local = cfg.vocab_size // 8
    ids[0, :4] = torch.tensor([0, cfg.vocab_size - 1, vocab_local - 1, vocab_local], dtype=torch.int32)
    return ids


def _run(galaxy, cfg, mesh_config, ccl, table, token_ids, shard_vocab_on_sp):
    emb = ParallelEmbedding(galaxy, cfg, {"weight": table}, mesh_config, ccl, shard_vocab_on_sp=shard_vocab_on_sp)
    out = emb(shard_tokens(galaxy, mesh_config, token_ids))
    return from_mesh_sp(galaxy, out)


@pytest.mark.parametrize("shard_vocab_on_sp", [False, True], ids=["1d", "2d"])
def test_embedding_vs_ref(galaxy, mesh_config, ccl, cfg, table, token_ids, shard_vocab_on_sp):
    """Both layouts reproduce the gather exactly."""
    ref = torch.nn.functional.embedding(token_ids.long(), table).unsqueeze(0)
    out = _run(galaxy, cfg, mesh_config, ccl, table, token_ids, shard_vocab_on_sp)
    assert out.shape == ref.shape, f"{tuple(out.shape)} != {tuple(ref.shape)}"
    torch.testing.assert_close(out.to(REF_DTYPE), ref, rtol=0, atol=0)


def test_layouts_agree(galaxy, mesh_config, ccl, cfg, table, token_ids):
    """1D and 2D agree **at this length**. They do not in general — see the module docstring.

    Kept as-is rather than deleted or loosened: it is a true statement about SEQ = 2048, and it is
    the row that would catch a *second*, length-independent divergence between the two layouts.
    """
    a = _run(galaxy, cfg, mesh_config, ccl, table, token_ids, False)
    b = _run(galaxy, cfg, mesh_config, ccl, table, token_ids, True)
    torch.testing.assert_close(a, b, rtol=0, atol=0)


#: The acceptance sequence length. The 2D defect is positional and starts at local index 1088, so
#: it is invisible below SEQ = 8704 no matter which ids are used.
REAL_SEQ = 10240


@pytest.mark.parametrize(
    "shard_vocab_on_sp",
    [
        False,
        pytest.param(
            True,
            marks=[
                # Opt-in only: the same ttnn collective that corrupts rows has also hung the mesh
                # ("device is unrecoverable"), which kills every test after it in a suite run.
                pytest.mark.skipif(
                    os.getenv("MISTRAL_EMBED_SHARD_VOCAB") != "1",
                    reason="2D layout reproduction; set MISTRAL_EMBED_SHARD_VOCAB=1 to run it alone",
                ),
                pytest.mark.xfail(
                    strict=True,
                    reason="2D reduce-scatter corrupts local seq idx 1088..1099 of every SP shard; "
                    "76/10240 rows wrong. 1D is the package default. See tt/embedding.py.",
                ),
            ],
        ),
    ],
    ids=["1d", "2d"],
)
def test_embed_real_prompt_ids(galaxy, mesh_config, ccl, cfg, table, shard_vocab_on_sp):
    """The gather at the acceptance length, on the golden trace's own token ids.

    Two things this covers that the rows above cannot, and both were needed to find the defect.

    *The length.* The damage is at a fixed local index per shard, past the end of a 2048-token run.

    *The ids.* ``torch.randint`` over 131072 is uniform; a real prompt is not, and concentrates in
    the low vocab where the 2D shard boundaries are. The uniform draw is the weaker input here even
    though it looks like the more thorough one — worth remembering the next time a lookup is tested.

    Bit-exact, for the same reason as :func:`test_embedding_vs_ref`: a gather has no arithmetic in
    it, and ``assert_close`` with ``rtol=0`` names the wrong row rather than averaging it away.
    """
    from models.demos.mistral_medium_3_5_128b.reference.golden import GoldenTrace

    ids = GoldenTrace.from_env().token_ids(REAL_SEQ).to(torch.int32)
    ref = torch.nn.functional.embedding(ids.long(), table).unsqueeze(0)
    out = _run(galaxy, cfg, mesh_config, ccl, table, ids, shard_vocab_on_sp)
    torch.testing.assert_close(out.to(REF_DTYPE), ref, rtol=0, atol=0)


def test_sp_shards_hold_their_own_tokens(galaxy, mesh_config, ccl, cfg, table, token_ids):
    """Token ``i``'s embedding comes back at position ``i``, not at a permuted one.

    The reference comparison above would already catch a permutation, but only as "wrong numbers".
    This states the property the rest of the stack depends on: the sequence split the embedding
    produces is the same contiguous SP split the activations and the RoPE tables use, so a token
    and its positional encoding are on the same chip.
    """
    out = _run(galaxy, cfg, mesh_config, ccl, table, token_ids, False)
    seq_local = SEQ // mesh_config.sp
    for row in range(mesh_config.sp):
        pos = row * seq_local  # the first token this SP row owns
        torch.testing.assert_close(out[0, 0, pos], table[int(token_ids[0, pos])], rtol=0, atol=0)
