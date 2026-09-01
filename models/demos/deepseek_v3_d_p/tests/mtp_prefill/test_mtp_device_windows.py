# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""On-device tests for the MTP shift-window path (GLM-5.2, issue #53533).

The claim the whole token path rests on is a statement about **hardware**:

    sharding is a fixed row -> position permutation applied to the window's *contents*, so applying
    the trunk's permutation to a window shifted by ``k`` lands ``t_{p+k}`` on the row whose hidden
    sits at ``p``

Today that sentence lives in a docstring (``input_prep.prepare_prefill_mtp_window``) and is measured
nowhere else. It is also the highest-risk line in the whole token path: get it wrong and every row is
paired with the wrong hidden -- no shape error, no crash, no exception, just a model that is quietly
predicting from the wrong token. This file measures it on the mesh.

**Exact integers, not PCC.** The embedding table here is synthetic: token ``t``'s vector carries
``t`` as three base-256 digits at the head of *every* TP column block, plus a nonzero canary in the
fourth column. ``ttnn.embedding`` is a gather and the position-0 mask is a multiply by 1.0 or 0.0,
so those digits survive bf16 bit-exactly -- the activation read back off the mesh decodes to the
integer id that landed on each row, and every assertion below is integer equality rather than a
correlation. A PCC would blur exactly the failure this file exists to catch. Repeating the digits in
each TP block also makes the token tensor's TP *replication* something the test checks rather than
assumes.

**The row -> position map is measured, not re-derived.** The test never reimplements
``create_balanced_chunk_order`` or the block-cyclic reshape. It uploads the shift-0 window -- which
is what the trunk uploads -- reads it back, and *that* is what defines which chunk-local position
each mesh row holds. Every shifted window is then checked against that measured map. So the test
cannot drift from the production sharding, and it covers ``is_balanced`` True and False without
knowing anything about either layout.

**No fabric.** The ops here are an upload, ``ttnn.embedding`` (per-chip, CCL-free by construction),
an elementwise multiply, and a readback. The mesh opens with default ``device_params`` rather than a
torus profile, so this needs no cabling-certified descriptor and no fabric bring-up.

Deliberately not covered: the MTP module numerics (``test_mtp.py``), and generation for the last
chunk -- every window here comes out of an INTERIOR chunk, so no token is ever generated and the LM
head never runs. The last chunk's generated tail, through the real LM head and the real predictor,
is ``test_mtp_transformer_chunks.py``'s subject.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.mtp_prefill.token_windows import MTPEmbedSource, mtp_chunk_stream
from models.demos.deepseek_v3_d_p.tt.runners.input_prep import prepare_prefill_input_tensor
from models.demos.deepseek_v3_d_p.tt.tt_parallel_embedding import TtParallelEmbedding
from models.demos.deepseek_v3_d_p.tt.tt_prefill_transformer import TtPrefillTransformer

SP_AXIS, TP_AXIS = 0, 1

CHUNK = 5120
"""Production chunk length. At sp=8 this is 640 rows/chip, a multiple of the 32-row tile."""
LEVELS = 4
HIDDEN = 6144
"""GLM-5.2 hidden size, so the TP split is the production 1536 columns/chip."""
VOCAB = 16384
"""Large enough for two chunks of ids; unrelated to placement, which is vocab-independent."""
TOK_BASE = 1000

DIGIT = 256
"""Base of the id encoding. Integers 0..255 are exact in bf16 (8 significant bits), so are the digits."""
CANARY = 7.0
"""A nonzero column every id carries, so "this row was zeroed" is meaningful even for id 0."""

MESH = [
    pytest.param(
        (8, 4),
        # These are local ops (embedding/slice/multiply); no fabric is needed. State that explicitly:
        # a param with no device_params leaves fabric_config=None, which conftest skips everywhere.
        {"fabric_config": ttnn.FabricConfig.DISABLED},
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
        id="8x4",
    ),
]

BALANCED = [
    pytest.param(False, id="block-cyclic"),
    pytest.param(True, id="balanced"),
]


# --------------------------------------------------------------------------------------------------
# Synthetic table + exact readback
# --------------------------------------------------------------------------------------------------


def _digit_table(vocab: int, hidden: int, tp_factor: int) -> torch.Tensor:
    """``table[t]`` encodes ``t`` in base 256 at the head of every TP column block."""
    width = hidden // tp_factor
    assert width >= 4, f"need 4 columns per TP block for 3 digits + canary, have {width}"
    assert vocab <= DIGIT**3, f"vocab {vocab} does not fit in three base-{DIGIT} digits"
    table = torch.zeros(vocab, hidden, dtype=torch.float32)
    ids = torch.arange(vocab, dtype=torch.int64)
    for j in range(tp_factor):
        base = j * width
        table[:, base + 0] = (ids % DIGIT).float()
        table[:, base + 1] = (ids // DIGIT % DIGIT).float()
        table[:, base + 2] = (ids // DIGIT**2 % DIGIT).float()
        table[:, base + 3] = CANARY
    return table


def _shard_dims() -> tuple:
    dims = [None, None]
    dims[SP_AXIS] = -2
    dims[TP_AXIS] = -1
    return tuple(dims)


def _read(emb: ttnn.Tensor, mesh_device) -> torch.Tensor:
    """Pull one window's activation off the mesh as ``[C, H]`` float32 and free the device copy.

    SP concatenates on the sequence axis and TP on the hidden axis, so row ``i`` of the result is
    mesh row ``i // L`` local row ``i % L`` -- the mesh's own row order, NOT position order. Turning
    it into position order is :func:`_row_to_position`'s job, and it is measured, not assumed.
    """
    full = ttnn.to_torch(
        emb,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=_shard_dims(), mesh_shape=tuple(mesh_device.shape)),
    ).float()[0, 0]
    ttnn.deallocate(emb)
    return full


def _ids(full: torch.Tensor, tp_factor: int) -> torch.Tensor:
    """Decode ``[C, H]`` back to the ``[C]`` token id on each mesh row, checking TP replication."""
    width = full.shape[-1] // tp_factor
    blocks = torch.stack([full[:, j * width : j * width + 3] for j in range(tp_factor)])  # [tp, C, 3]
    d = blocks.round().to(torch.int64)
    ids = d[..., 0] + DIGIT * d[..., 1] + DIGIT**2 * d[..., 2]  # [tp, C]
    disagree = (ids != ids[0]).any(dim=-1).nonzero().flatten().tolist()
    assert not disagree, (
        f"TP chips {disagree} decoded a different token than TP chip 0 on some row: the token tensor "
        "is uploaded with dims=(sp_axis, None), i.e. REPLICATED across TP, so every TP chip must look "
        "up the same id in its own column slice"
    )
    return ids[0]


def _row_to_position(fs, stream: list[int], mesh_device) -> torch.Tensor:
    """Measure the row -> chunk-local-position map by embedding the shift-0 (trunk) window.

    This is the whole reason the test is layout-agnostic: rather than reimplementing the balanced or
    block-cyclic permutation and risking that the copy is the thing that is wrong, it uploads the
    window the trunk itself uploads and reads back where each id went.

    Callers pass chunk 0's window, so ids decode to ``TOK_BASE + position`` directly. Subtracting the
    known base rather than ``ids.min()`` keeps the assert strict -- it checks not just that the window
    came back as *a* contiguous permutation, but that it is the expected block of positions.
    """
    tp_factor = mesh_device.shape[TP_AXIS]
    ids = _ids(_read(fs._mtp_embed_window(stream[: fs.seq_len], False), mesh_device), tp_factor)
    pos = ids - TOK_BASE
    assert sorted(pos.tolist()) == list(range(fs.seq_len)), (
        "the shift-0 (trunk) window did not come back as a permutation of the chunk's positions, so "
        "the upload itself is wrong and nothing downstream of it can be trusted"
    )
    return pos


def _embedder(mesh_device, chunk_size: int, is_balanced: bool, *, vocab=VOCAB, hidden=HIDDEN):
    """A ``TtPrefillTransformer`` stand-in carrying the REAL MTP embed + mask methods.

    Only the attributes those two methods read are provided; the methods themselves are the
    production ones, bound to this object, so what runs on the mesh is production code.
    """
    tp_factor = mesh_device.shape[TP_AXIS]
    fs = SimpleNamespace(
        mesh_device=mesh_device,
        seq_len=chunk_size,
        sp_factor=mesh_device.shape[SP_AXIS],
        is_balanced=is_balanced,
        mesh_shape=tuple(mesh_device.shape),
        sp_axis=SP_AXIS,
        emb_dim_per_chip=hidden // tp_factor,
        embed=TtParallelEmbedding(
            mesh_device,
            vocab_size=vocab,
            emb_dim=hidden,
            torch_weight=_digit_table(vocab, hidden, tp_factor),
            sp_axis=SP_AXIS,
            tp_axis=TP_AXIS,
        ),
        _mtp_pos0_mask=None,
    )
    fs._mtp_position_zero_mask = TtPrefillTransformer._mtp_position_zero_mask.__get__(fs, SimpleNamespace)
    fs._mtp_embed_window = TtPrefillTransformer._mtp_embed_window.__get__(fs, SimpleNamespace)
    fs._mtp_embed_window_dev = TtPrefillTransformer._mtp_embed_window_dev.__get__(fs, SimpleNamespace)
    return fs


def _report(got: torch.Tensor, want: torch.Tensor, pos: torch.Tensor, what: str) -> str:
    bad = (got != want).nonzero().flatten()
    if bad.numel() == 0:
        return ""
    r = int(bad[0])
    return (
        f"{what}: {bad.numel()} of {got.numel()} rows hold the wrong token. First is mesh row {r} "
        f"(chunk-local position {int(pos[r])}): got id {int(got[r])}, expected {int(want[r])}, "
        f"off by {int(got[r]) - int(want[r])}."
    )


# --------------------------------------------------------------------------------------------------
# The headline: the shift/shard coupling, on hardware
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("is_balanced", BALANCED)
@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_shift_window_lands_on_the_row_its_hidden_occupies(mesh_device, is_balanced):
    """The design claim, measured: level ``k``'s row for position ``p`` holds ``t_{p+k+1}``.

    Rows are compared in MESH order against the map measured from the trunk window, so a bug that
    permutes rows and a bug that shifts them are both caught, and the failure message names the
    offending mesh row and its position rather than reporting a degraded correlation.
    """
    fs = _embedder(mesh_device, CHUNK, is_balanced)
    tp_factor = mesh_device.shape[TP_AXIS]
    all_tokens = [TOK_BASE + p for p in range(2 * CHUNK)]  # chunk 0 is interior and fully real
    stream, real_len = mtp_chunk_stream(all_tokens, 0, CHUNK, LEVELS)
    assert real_len == CHUNK

    pos = _row_to_position(fs, stream, mesh_device)
    src = MTPEmbedSource(stream, CHUNK, LEVELS, embed_fn=lambda w: fs._mtp_embed_window(w, False))

    for k in range(LEVELS):
        got = _ids(_read(src(k, None), mesh_device), tp_factor)
        want = torch.tensor([stream[p + k + 1] for p in pos.tolist()], dtype=torch.int64)
        assert not (msg := _report(got, want, pos, f"level {k} (shift {k + 1})")), msg


@pytest.mark.parametrize("is_balanced", BALANCED)
@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_a_shift_zero_window_is_byte_identical_to_the_trunk_upload(mesh_device, is_balanced):
    """``prepare_prefill_mtp_window`` must stay the trunk's own packing, not merely resemble it.

    It is a pass-through today, which is exactly why this is worth pinning on hardware: if it is
    ever specialised (a different reshape, an overlapped packing) the windows would keep their shape
    and every host test would keep passing while every row moved.
    """
    fs = _embedder(mesh_device, CHUNK, is_balanced)
    ids = [TOK_BASE + p for p in range(CHUNK)]

    through_window = _read(fs._mtp_embed_window(ids, False), mesh_device)
    tt_trunk = prepare_prefill_input_tensor(ids, mesh_device, fs.sp_factor, is_balanced, fs.mesh_shape, SP_AXIS)
    through_trunk = _read(ttnn.unsqueeze_to_4D(fs.embed(tt_trunk)), mesh_device)
    ttnn.deallocate(tt_trunk)

    assert torch.equal(through_window, through_trunk), (
        "the MTP window upload and the trunk upload put different tokens on the same rows, so every "
        "MTP row would be paired with the wrong trunk hidden"
    )


@pytest.mark.parametrize("is_balanced", BALANCED)
@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_position_zero_mask_zeroes_exactly_the_row_holding_position_0(mesh_device, is_balanced):
    """``build_position_zero_mask`` on the mesh: one row to zero, every other row untouched.

    The mask derives its row by pushing an indicator through the same sharding path rather than
    assuming position 0 lands on chip 0 row 0. That derivation is the thing under test, which is why
    this runs under both layouts -- under ``balanced`` the assumption and the derivation could
    disagree and only the derivation is right.
    """
    fs = _embedder(mesh_device, CHUNK, is_balanced)
    window = [TOK_BASE + p for p in range(CHUNK)]

    plain = _read(fs._mtp_embed_window(window, False), mesh_device)
    pos = _ids(plain, mesh_device.shape[TP_AXIS]) - TOK_BASE
    masked = _read(fs._mtp_embed_window(window, True), mesh_device)

    row0 = (pos == 0).nonzero().flatten()
    assert row0.numel() == 1, f"expected exactly one row holding position 0, found {row0.numel()}"
    row0 = int(row0)

    assert torch.count_nonzero(plain[row0]) > 0, "the unmasked row is already zero; the check is vacuous"
    nz = torch.count_nonzero(masked[row0])
    assert nz == 0, f"row {row0} holds absolute position 0 but {int(nz)} of its columns survived the mask"

    keep = torch.ones(CHUNK, dtype=torch.bool)
    keep[row0] = False
    changed = (masked[keep] != plain[keep]).any(dim=-1).nonzero().flatten().tolist()
    assert not changed, (
        f"the position-0 mask also altered rows {changed[:8]}{'...' if len(changed) > 8 else ''}; it "
        "must zero the row holding absolute position 0 and nothing else"
    )


@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_token_ids_above_16_bits_survive_the_upload(mesh_device):
    """GLM-5.2's vocab is 154880, so real ids run past 65535 and a narrowing would be silent.

    Every other case here uses small ids, where a 16-bit truncation anywhere in the ROW_MAJOR uint32
    path is invisible. Hidden size is reduced for this one because what is being probed is the id,
    not the column layout, and a 154880-row table at the full width is 1.9 GiB of weight for one
    assertion.
    """
    hidden, vocab, base = 512, 154880, 149000
    assert base + CHUNK + LEVELS < vocab
    fs = _embedder(mesh_device, CHUNK, False, vocab=vocab, hidden=hidden)
    tp_factor = mesh_device.shape[TP_AXIS]

    all_tokens = [base + p for p in range(2 * CHUNK)]
    stream, _ = mtp_chunk_stream(all_tokens, 0, CHUNK, LEVELS)
    ids = _ids(_read(fs._mtp_embed_window(stream[:CHUNK], False), mesh_device), tp_factor)

    assert int(ids.min()) > 0xFFFF, "the test did not actually exercise ids above 16 bits"
    assert sorted(ids.tolist()) == sorted(
        stream[:CHUNK]
    ), "ids came back altered; a 16-bit narrowing would show up here as ids folded modulo 65536"
