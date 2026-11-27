# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device tests for CSA's sparse attention: an index list must reproduce the additive mask it replaces.

CSA used to attend densely over ``[carry | sliding | compressed]`` and let a 0/-inf mask drop everything a
query may not see. It now hands ``sparse_sdpa`` a per-query list of row ids into one JOINT table instead,
and the mask is gone. The two describe the same key set, so they must produce the same answer -- that
equivalence is what this file pins, on the joint layout the model builds rather than on a synthetic one.

The op's own contract (sinks in the pre-divided logit domain, ``v_dim == K_DIM``, the V4 top-k widths) is
covered by ``tests/ttnn/unit_tests/operations/sdpa/test_sparse_sdpa.py::test_sparse_sdpa_attention_sink``.
What is NOT covered there, and is the whole risk of this change, is whether OUR index rows mean what we
think: hence the dense comparison, and the companion test showing the compaction is load-bearing."""

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric_1d_plain_device_params
from models.demos.deepseek_v3_d_p.tt.mla.compressed_sparse_attention import index_tables

_SENTINEL = 0xFFFFFFFF
_HEADS = 32  # sparse_sdpa requires H % 32 == 0
_HEAD_DIM = 512  # V4: K == V, so v_dim == K_DIM (no RoPE-only tail)
_SLIDING = 128
_CAPACITY = 256  # compressed entries the joint table holds
_CHUNK = 128  # query rows in one call
_TOPK = 64
_COMPRESS_RATE = 4
_WIDTH = 256  # round_up(_SLIDING + _TOPK, 128)
_DUMP = _SLIDING + _TOPK  # source column holding the spare sentinel
_RAW_ROWS = _SLIDING + _CHUNK  # carry followed by this chunk's raw keys
_T = _CAPACITY + _RAW_ROWS


def _sliding_ids(first_pos):
    """Joint-table row for the j-th sliding key of query row i, or the sentinel if that token predates the
    sequence.

    The raw region is ``[carry | chunk]`` based at ``_CAPACITY``, so chunk-local token offset ``d``
    (negative inside the carry) sits at ``_CAPACITY + _SLIDING + d``. Query row i wants tokens
    ``i - _SLIDING + 1 .. i``, which makes the row id affine in i and j and independent of where the chunk
    sits -- the reason the model can build it once at ``alloc_state``. Only WHICH slots are real depends on
    ``first_pos``, and only for the first chunk."""
    i = torch.arange(_CHUNK).view(_CHUNK, 1)
    j = torch.arange(_SLIDING).view(1, _SLIDING)
    offset = i - _SLIDING + 1 + j
    ids = _CAPACITY + _SLIDING + offset
    return torch.where(first_pos + offset >= 0, ids, torch.full_like(ids, _SENTINEL))


def _causal_picks(first_pos, generator):
    """Top-k compressed entries per query row, in the indexer's output form.

    Entry e is visible to the query at ``p`` iff ``e < (p + 1) // _COMPRESS_RATE``, the same threshold the
    score op applies. Rows with fewer than ``_TOPK`` visible entries get a sentinel TAIL, which is what
    ``topk_large_indices`` guarantees and what the joint row's own tail depends on."""
    picks = torch.full((_CHUNK, _TOPK), _SENTINEL, dtype=torch.int64)
    for i in range(_CHUNK):
        visible = min((first_pos + i + 1) // _COMPRESS_RATE, _CAPACITY)
        take = min(visible, _TOPK)
        if take:
            picks[i, :take] = torch.randperm(visible, generator=generator)[:take]
    return picks


def _source_rows(first_pos, generator):
    """``[sliding ids | top-k picks | sentinel]`` -- the persistent buffer the model gathers from.

    Compressed entries live at joint rows ``[0, _CAPACITY)``, so an entry id IS its row id and the picks
    need no arithmetic. The trailing column is the gather's dump target."""
    picks = _causal_picks(first_pos, generator)
    spare = torch.full((_CHUNK, 1), _SENTINEL, dtype=torch.int64)
    return torch.cat([_sliding_ids(first_pos), picks, spare], dim=-1), picks


def _permutation(first_pos):
    """Squeeze the sliding block's missing slots out, so sentinels end up a contiguous tail.

    Query ``p`` can only look back to token 0, so it has ``min(_SLIDING, p + 1)`` real sliding keys, and
    they are the LAST slots of that block -- slot j holds token ``p - _SLIDING + 1 + j``, so it is the low
    j that fall off the front. Left alone those empties sit in the MIDDLE of the row, and the reader
    binary-searches the first sentinel and treats it as the end of the row, dropping every compressed pick
    after it. So the real sliding slots slide to the front, the pick block follows them, and everything
    past it points at the source's spare sentinel column.

    Only positions below ``_SLIDING`` are short, so for a chunk starting at or past the window this is the
    identity (modulo the tail, which the source is too narrow to address directly)."""
    valid = (torch.arange(_CHUNK) + first_pos + 1).clamp(max=_SLIDING).view(_CHUNK, 1)
    out = torch.arange(_WIDTH).view(1, _WIDTH).expand(_CHUNK, _WIDTH)
    kept = out + (_SLIDING - valid)  # the block's real tail, moved down to start at 0
    shifted = _SLIDING + (out - valid)  # the pick block, moved left to abut it
    perm = torch.where(out < valid, kept, shifted)
    return torch.where(out < valid + _TOPK, perm, torch.full_like(perm, _DUMP))


def _identity(first_pos):
    """The permutation that does nothing, for showing what the compaction is worth."""
    out = torch.arange(_WIDTH).view(1, _WIDTH).expand(_CHUNK, _WIDTH)
    return out.clamp(max=_DUMP)


def _index_rows(first_pos, generator, permute=_permutation):
    source, picks = _source_rows(first_pos, generator)
    return torch.gather(source, -1, permute(first_pos)), picks


def _dense_mask(index_rows):
    """The additive mask the index list stands for: 0 where a row names a column, -inf everywhere else."""
    mask = torch.full((_CHUNK, _T), float("-inf"), dtype=torch.float32)
    real = index_rows != _SENTINEL
    rows = torch.arange(_CHUNK).view(_CHUNK, 1).expand_as(index_rows)[real]
    mask[rows, index_rows[real]] = 0.0
    return mask.view(1, 1, _CHUNK, _T)


def _assert_producer_contract(index_rows):
    """What ``sparse_sdpa`` assumes of its caller, checked on the rows we actually send."""
    sentinel = index_rows == _SENTINEL
    first = torch.where(sentinel.any(-1), sentinel.to(torch.uint8).argmax(-1), torch.full((_CHUNK,), _WIDTH))
    positions = torch.arange(_WIDTH).view(1, _WIDTH)
    assert torch.equal(sentinel, positions >= first.view(_CHUNK, 1)), "sentinels must form a contiguous tail"
    assert (first > 0).all(), "every row needs at least one valid key"
    for row in range(_CHUNK):
        real = index_rows[row][~sentinel[row]]
        assert real.max() < _T, "index out of the table"
        # A duplicate would be counted twice by the sparse path but once by the mask, so the two would
        # disagree for a reason that has nothing to do with what this test is checking.
        assert real.unique().numel() == real.numel(), f"row {row} names a column twice"


def _replicate(mesh_device, tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        tensor,
        device=mesh_device,
        dtype=dtype,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _first_chip(mesh_device, tensor):
    composed = ttnn.to_torch(
        tensor,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(0, 1)),
    )
    return composed[0:1, 0:_HEADS].float()


def _run_pair(mesh_device, index_rows, generator):
    """Same q, same keys, same sink: once through dense SDPA with the mask, once through sparse_sdpa."""
    q = torch.randn(1, _HEADS, _CHUNK, _HEAD_DIM, generator=generator).to(torch.bfloat16)
    kv = torch.randn(1, 1, _T, _HEAD_DIM, generator=generator).to(torch.bfloat16)
    scale = _HEAD_DIM**-0.5
    # V4 keeps sinks in the already-scaled logit domain, so both ops get model_sink / scale; they differ
    # only in the shape they want it in, which is exactly the bit the model has to get right.
    sink = (torch.linspace(-4.0, 4.0, _HEADS) / scale).to(torch.bfloat16)

    dense = ttnn.transformer.scaled_dot_product_attention(
        _replicate(mesh_device, q),
        _replicate(mesh_device, kv),
        _replicate(mesh_device, kv),
        attn_mask=_replicate(mesh_device, _dense_mask(index_rows)),
        is_causal=False,
        scale=scale,
        attention_sink=_replicate(mesh_device, sink.reshape(1, _HEADS, 1, 1)),
        program_config=ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=mesh_device.compute_with_storage_grid_size(),
            q_chunk_size=128,
            k_chunk_size=128,
            exp_approx_mode=False,
        ),
    )

    sparse = ttnn.transformer.sparse_sdpa(
        _replicate(mesh_device, q, layout=ttnn.ROW_MAJOR_LAYOUT),
        _replicate(mesh_device, kv, layout=ttnn.ROW_MAJOR_LAYOUT),
        _replicate(
            mesh_device,
            index_rows.view(1, 1, _CHUNK, _WIDTH).to(torch.int32),
            ttnn.uint32,
            ttnn.ROW_MAJOR_LAYOUT,
        ),
        _HEAD_DIM,
        kv_format=ttnn.transformer.SparseKVFormat.BF16,
        scale=scale,
        k_chunk_size=128,
        attention_sink=_replicate(mesh_device, sink.reshape(1, 1, 1, _HEADS), layout=ttnn.ROW_MAJOR_LAYOUT),
    )
    return _first_chip(mesh_device, dense), _first_chip(mesh_device, sparse)


_MESH = [
    pytest.param(
        (2, 2),
        fabric_1d_plain_device_params(),
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 2), topology="mesh-2x2"),
        id="2x2",
    )
]

# first_pos 0 is the only chunk whose queries lack a full window, so it is the only one the compaction
# does anything to; 256 stands for every later chunk, where the carry makes every row full.
_POSITIONS = [0, 256]


@pytest.mark.parametrize("first_pos", _POSITIONS, ids=[f"pos{p}" for p in _POSITIONS])
@pytest.mark.parametrize("mesh_device, device_params", _MESH, indirect=["mesh_device", "device_params"])
def test_csa_sparse_matches_dense_mask(mesh_device, device_params, first_pos):
    """The replacement is only valid if the index list and the mask it replaces agree."""
    generator = torch.Generator().manual_seed(1234 + first_pos)
    index_rows, picks = _index_rows(first_pos, generator)
    _assert_producer_contract(index_rows)
    if first_pos == 0:
        assert (picks == _SENTINEL).any(), "this case is meant to cover rows the indexer under-fills"
        assert (_sliding_ids(first_pos) == _SENTINEL).any(), "and rows whose window predates the sequence"
    else:
        assert (_sliding_ids(first_pos) != _SENTINEL).all(), "a later chunk's carry makes every window full"

    dense, sparse = _run_pair(mesh_device, index_rows, generator)
    score = torch.corrcoef(torch.stack([dense.flatten(), sparse.flatten()]))[0, 1].item()
    assert score >= 0.999, f"sparse/dense PCC {score:.5f} at first_pos {first_pos}"
    torch.testing.assert_close(sparse, dense, atol=0.035, rtol=0.08)


def test_csa_index_tables_match_the_golden():
    """The model builds the list the device tests above validated -- exactly, not merely equivalently.

    The helpers in this file are written from the layout definition; ``index_tables`` is written to be
    fast to upload. Comparing them as integers is what keeps the model honest to the thing that was
    actually measured against dense attention, so this is an exact compare, not a PCC."""
    source, compacted, later = index_tables(_CHUNK, _SLIDING, _TOPK, _CAPACITY)
    assert torch.equal(compacted, _permutation(0)), "first-chunk permutation"
    assert torch.equal(later, _identity(0)), "later-chunk permutation"

    # The model's source leaves the pre-sequence sliding slots holding real row ids rather than
    # sentinels, which is only sound because no permutation can select them. Gathering both sources
    # through the model's own permutations is what proves that.
    for first_pos, permutation in ((0, compacted), (256, later)):
        golden, picks = _source_rows(first_pos, torch.Generator().manual_seed(7))
        filled = source.clone()
        filled[:, _SLIDING : _SLIDING + _TOPK] = picks
        rows = torch.gather(filled, -1, permutation)
        assert torch.equal(rows, torch.gather(golden, -1, permutation)), f"source differs at pos {first_pos}"
        _assert_producer_contract(rows)

    # A row whose window predates the sequence is the whole reason the compaction exists, so pin what it
    # keeps: query 0 sees only its own token and no compressed entry, query 127 a full window plus every
    # entry its causal threshold allows.
    filled = source.clone()
    filled[:, _SLIDING : _SLIDING + _TOPK] = _causal_picks(0, torch.Generator().manual_seed(7))
    kept = (torch.gather(filled, -1, compacted) != _SENTINEL).sum(-1)
    assert int(kept[0]) == 1, f"query 0 should keep one key, kept {int(kept[0])}"
    assert int(kept[_SLIDING - 1]) == _SLIDING + min(_TOPK, _SLIDING // _COMPRESS_RATE)
    assert torch.equal(kept, kept.cummax(0).values), "a later query can never see fewer keys"


def test_csa_uncompacted_rows_violate_the_producer_contract():
    """What the compaction buys, stated on the host rather than by running the bad case.

    The uncompacted list is not merely less accurate, it is malformed, so this deliberately does NOT go to
    a device: the reader derives the row length from the first sentinel and floors it at one
    (``nv = lo == 0 ? 1 : lo`` in sparse_sdpa_reader.cpp), and for query row 0 the first sentinel IS slot
    0, so the op would gather page 0xFFFFFFFF -- an out-of-bounds read, not a wrong number.

    Everything below is the reader's own arithmetic replayed in torch, which is why it needs no fixture."""
    naive, _ = _index_rows(0, torch.Generator().manual_seed(99), permute=_identity)
    sentinel = naive == _SENTINEL

    short = sentinel[:, :_SLIDING].any(-1)  # rows whose sliding block still has a hole
    assert short[: _SLIDING - 1].all() and not short[_SLIDING - 1 :].any(), "only sub-window rows should be short"
    assert sentinel[short][:, -1].all(), "a truncated row still ends in sentinels, so a tail check cannot catch it"

    # The reader's binary search, and what it leaves behind.
    first = torch.where(sentinel.any(-1), sentinel.to(torch.uint8).argmax(-1), torch.full((_CHUNK,), _WIDTH))
    dropped = torch.tensor([int((naive[row, first[row] :] != _SENTINEL).sum()) for row in range(_CHUNK)])
    assert (dropped[short] > 0).all(), "every short row should lose keys the mask would have kept"
    assert dropped[~short].sum() == 0, "full-window rows are well formed and must lose nothing"
    assert first[0] == 0, "row 0's first slot is a sentinel, which the reader's floor turns into an OOB page"

    # The compacted list of the same picks keeps every key and is well formed.
    compacted, _ = _index_rows(0, torch.Generator().manual_seed(99))
    _assert_producer_contract(compacted)
    kept = (compacted != _SENTINEL).sum(-1)
    assert torch.equal(kept, (naive != _SENTINEL).sum(-1)), "compaction must move keys, never add or drop them"
