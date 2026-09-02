# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""On-device gate for the MTP union-embedding transport (GLM-5.2, issue #53533).

The runner moves MTP's K token windows from the producer to the rank that runs the levels without a
host round trip anywhere. Four hardware claims carry it:

1. **The lookahead ids ride the trunk's own H2D row and arrive as their own tensor.** The producer
   pushes chip ``c`` the single row ``stream[c*L : c*L + L + overhang]`` through ONE H2D socket, and
   ``inbound_socket_service_sync`` splits it as it copies it out: ``tt_ids`` is the leading ``L`` ids,
   ``tt_overhang`` the trailing ``overhang``. The trunk tensor must come back byte-identical to the
   row an MTP-*off* run sends, and the tail must be the ``overhang`` ids that immediately follow it --
   otherwise level ``k``'s window is off by a chip's worth of positions. This goes through a REAL
   ``H2DStreamService`` and the real op, with the real ``producer._h2d_rows`` on the far end, so the
   two ends cannot agree here and disagree in production.
2. **Two gathers, and the trunk gather IS the model's input.** The union is embedded as two blocks,
   never as one rejoined id row, so the trunk block can be handed to the transformer directly
   (``input_is_embedded``) instead of gathering the same ``L`` rows a second time. The 32-row
   overhang gather is the small, unusual one -- a second program-cache entry at a row count the op
   never otherwise sees.
3. **The union slices correctly at a non-tile-aligned row offset.** ``k`` runs 1..4 and is never a
   multiple of 32, so :meth:`MTPUnionEmbedding.window` untilizes, ``ttnn.slice``s and retilizes. A
   layout fallback that silently rounded to a tile boundary would return a window shifted by up to 31
   positions -- no shape error, no crash, just every row paired with the wrong hidden.
4. **The 3-way row-concat pack splits back cleanly.** The union's blocks ride the D2D socket stacked
   UNDER the hidden in one concat, and the receiver cuts the union back out at a fixed row offset.
5. **The last chunk's generated embeddings land where every level reads them.** Past the request's
   real length the stream has no ids, so level ``k`` writes ``embed(argmax(lm_head(H^k)))`` into the
   union at global position ``actual_end + k`` -- one one-hot matmul, patching every chip whose union
   covers that position -- and each level's own row slice then picks up exactly ``g_0..g_k``. The
   failure mode is silent: an embedding on a row no window reads, or on a row read at the wrong
   level, is a draft trained on the wrong token and nothing raises.

**Exact equality, not PCC.** ``embedding`` is a gather: the row it writes is the table row, byte for
byte, so ``slice(embed(ids)) == embed(slice(ids))`` holds bit-exactly and a correlation would blur
the one failure this file exists to catch. That identity is also what lets the trunk gather double as
the model input, and why the ids never need a codec to survive a bf16 wire -- which is why the vocab
size, once the whole problem, is now irrelevant to transport and is kept small here.

Claims 2-4 are per-chip ops (concat, embedding, slice, to_layout) and need no fabric; claim 1 drives
a real H2D socket, so the mesh opens on the same torus-xy profile ``test_h2d_socket_sync.py`` uses.
"""

from __future__ import annotations

import os
import struct

import pytest
import torch

import ttnn
from models.demos.common.prefill.runners import prefill_producer as producer
from models.demos.common.prefill.runners.runner_utils import h2d_row_len, mtp_overhang, mtp_union_rows
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tt.mtp_prefill.device_windows import MTPUnionEmbedding
from models.demos.deepseek_v3_d_p.tt.runners.input_prep import (
    build_mtp_generation_keep_mask,
    build_mtp_generation_select,
    mtp_generation_union_rows,
)
from models.demos.deepseek_v3_d_p.tt.tt_parallel_embedding import TtParallelEmbedding

SP, TP = 8, 4
CHUNK = 5120
"""Production chunk. At sp=8 this is L=640 rows/chip, 20 whole tiles."""
K = 4
"""MTP4 -- the level count #53533 asks for."""
HIDDEN = 6144
"""GLM-5.2 hidden size, so the TP split is the production 1536 columns/chip."""
VOCAB = 8192
"""Irrelevant to the transport now that the ids never cross a bf16 wire; kept small so the table is
8192 x 1536 bf16 = 24 MiB/chip instead of GLM-5.2's 454."""

L = h2d_row_len(CHUNK, SP)
OVERHANG = mtp_overhang(K)
UNION_ROWS = mtp_union_rows(CHUNK, SP, K)

METADATA_SIZE_BYTES = 12
"""3 x uint32 [slot_id, actual_start, actual_end] -- what prefill_runner sends."""

SOURCE_ROW = 5 * 32 + 11
"""The row of the gathered LM-head block that holds the generated token: ``device_id * 32 +
token_offset``. Fixed here (the real value comes from ``global_to_local_token_id``) and deliberately
not 0, so a patch that ignored it would show up."""

GENERATION_CASES = [
    pytest.param(CHUNK, id="ends-on-the-chunk-edge"),
    pytest.param(L + 10, id="ends-on-the-overhang-seam"),
]
"""Real lengths for claim 5. The first is production (56320 = 11 x 5120): the generated positions
sit in the LAST chip's overhang and exactly one chip holds each. The second ends 10 rows into chip
1's shard, where chip 0's overhang covers the same positions -- two chips must be patched."""

MESH = [
    pytest.param(
        (SP, TP),
        # Claim 1 drives a real H2D socket, so the mesh needs the same profile the plain
        # h2d-socket-sync gate runs on. Claims 2-4 are local ops and are indifferent to it.
        torus_xy_device_params(),
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(SP, TP), topology="mesh-8x4"),
        id="8x4",
    )
]


@pytest.fixture(autouse=True)
def producer_env(monkeypatch):
    """Bind the producer's transport constants to this file's shape.

    They are module globals rebound from the environment by ``_load_env_config()``, so the rows built
    below are the rows a real MTP4 producer builds -- not a re-derivation. ``monkeypatch`` restores the
    env, and the teardown re-reads it so a later test in the same process sees the defaults again.
    """
    for k, v in (("PREFILL_SP", SP), ("PREFILL_TP", TP), ("PREFILL_CHUNK_SIZE", CHUNK)):
        monkeypatch.setenv(k, str(v))
    monkeypatch.setenv("PREFILL_MTP_LEVELS", str(K))
    producer._load_env_config()
    yield
    monkeypatch.undo()
    producer._load_env_config()


def _producer_rows(tokens: list[int], levels: int):
    """``producer._h2d_rows`` at a chosen MTP level count -- ``levels=0`` gives the plain-path row."""
    os.environ["PREFILL_MTP_LEVELS"] = str(levels)
    producer._load_env_config()
    try:
        return producer._h2d_rows(tokens)
    finally:
        os.environ["PREFILL_MTP_LEVELS"] = str(K)
        producer._load_env_config()


@pytest.fixture
def stream() -> torch.Tensor:
    """The producer's token pool for one chunk: CHUNK ids plus the overhang past its right edge."""
    torch.manual_seed(0)
    return torch.randint(0, VOCAB, (CHUNK + OVERHANG,), dtype=torch.int64)


def _embed_fn(mesh_device, table: torch.Tensor):
    """The trunk's own gather, spelled exactly as ``TtPrefillTransformer.mtp_embed_ids`` spells it --
    a real ``TtParallelEmbedding`` over a random table, not a stand-in. Because it IS that call, the
    trunk block it returns is what the transformer's first-rank embed would have produced, which is
    the identity ``input_is_embedded`` rests on."""
    embed = TtParallelEmbedding(mesh_device, vocab_size=VOCAB, emb_dim=HIDDEN, torch_weight=table)
    return lambda ids: ttnn.unsqueeze_to_4D(embed(ids))


def _shards(t) -> list:
    """Per-chip readback, one torch tensor per device, in mesh row-major order."""
    return [ttnn.to_torch(s) for s in ttnn.get_device_tensors(t)]


def _assert_rows(tt, want_ids: torch.Tensor, table: torch.Tensor, shift: int, rows: int, what: str) -> None:
    """Every shard of `tt` holds ``table[stream[chip's rows]]`` for its own TP column block, exactly.

    `want_ids` is the whole [SP, UNION_ROWS] id grid; each device checks its chip's
    ``[shift, shift+rows)``.
    """
    got = _shards(tt)
    assert len(got) == SP * TP, f"{what}: expected {SP * TP} device shards, got {len(got)}"
    per_tp = HIDDEN // TP
    for d, g in enumerate(got):
        c, t = d // TP, d % TP  # mesh is (sp, tp) row-major: SP-sharded rows, TP-sharded columns
        want = table[want_ids[c, shift : shift + rows]][:, t * per_tp : (t + 1) * per_tp]
        g = g.reshape(-1, per_tp)
        assert g.shape == want.shape, f"{what} shift {shift} dev {d}: shape {tuple(g.shape)} != {tuple(want.shape)}"
        bad = (g != want).any(dim=-1).nonzero().flatten()
        assert bad.numel() == 0, (
            f"{what} shift {shift} dev {d} (chip {c}): {bad.numel()}/{rows} rows wrong, first at local "
            f"row {bad[0].item()} -- the window is reading the wrong positions"
        )


def _assert_union_rows(tt, want: torch.Tensor, shift: int, rows: int, what: str) -> None:
    """Every shard of a window equals ``want[chip, shift : shift + rows]`` in its own TP columns.

    Like :func:`_assert_rows` but against an explicitly built ``[SP, U, HIDDEN]`` union rather than
    ``table[ids]``, because generation makes some rows something no id maps to.
    """
    got = _shards(tt)
    assert len(got) == SP * TP, f"{what}: expected {SP * TP} device shards, got {len(got)}"
    per_tp = HIDDEN // TP
    for d, g in enumerate(got):
        c, t = d // TP, d % TP
        exp = want[c, shift : shift + rows, t * per_tp : (t + 1) * per_tp]
        g = g.reshape(-1, per_tp)
        assert g.shape == exp.shape, f"{what} dev {d}: shape {tuple(g.shape)} != {tuple(exp.shape)}"
        bad = (g != exp).any(dim=-1).nonzero().flatten()
        assert bad.numel() == 0, (
            f"{what}: dev {d} (chip {c}) has {bad.numel()}/{rows} wrong rows, first at window row "
            f"{bad[0].item()} = union row {shift + bad[0].item()}"
        )


def _h2d_service(mesh_device):
    """A real H2DStreamService shaped for one MTP4 chunk: ``[SP, 1, UNION_ROWS]`` uint32, one page per
    chip, spelled exactly as ``runner_utils.build_h2d_service`` spells it."""
    return ttnn.H2DStreamService(
        mesh_device=mesh_device,
        global_spec=ttnn.TensorSpec(
            shape=ttnn.Shape([SP, 1, UNION_ROWS]),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            buffer_type=ttnn.BufferType.DRAM,
        ),
        # One page per chip = the whole row, which is what makes the op's split an intra-page split.
        max_socket_page_size_bytes=UNION_ROWS * 4,
        mapper=ttnn.create_mesh_mapper(
            mesh_device, ttnn.MeshMapperConfig(placements=[ttnn.PlacementShard(0), ttnn.PlacementReplicate()])
        ),
        worker_cores=ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0)),
        metadata_size_bytes=METADATA_SIZE_BYTES,
    )


def _cut_stream(stream: torch.Tensor, mesh_device, service=None):
    """One producer push through a real H2D socket: ``(trunk ids, overhang ids, want_ids grid)``.

    The op returns the three tensors; nothing here slices anything.
    """
    own = service is None
    service = service or _h2d_service(mesh_device)
    try:
        rows = _producer_rows(stream.tolist(), K)
        service.forward_to_tensor_bytes(
            rows.astype("int32").reshape(SP, 1, UNION_ROWS).copy(), metadata=struct.pack("<III", 0, 0, CHUNK)
        )
        tt_ids, tt_overhang, tt_meta = ttnn.experimental.deepseek_prefill.inbound_socket_service_sync(
            service, metadata_size_bytes=METADATA_SIZE_BYTES, overhang_size_bytes=OVERHANG * 4
        )
    finally:
        if own:
            service.barrier()
    ttnn.deallocate(tt_meta)
    want_ids = torch.stack([stream[c * L : c * L + UNION_ROWS] for c in range(SP)])  # [SP, L+overhang]
    return tt_ids, tt_overhang, want_ids


@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_h2d_row_cuts_into_the_plain_trunk_and_its_lookahead(mesh_device, stream):
    """Claim 1, on the ids themselves, before any embedding is involved.

    The trunk half is compared against the row a **plain** (MTP-off) producer pushes, built by the
    same function with ``PREFILL_MTP_LEVELS=0``. That is the promise the whole one-socket design makes:
    turning MTP on does not perturb a single id the model prefills.

    Two pushes through ONE service, because the split is compiled into the program: the second must
    hit the program cache rather than rebuild, and must land the same two tensors.
    """
    service = _h2d_service(mesh_device)
    tt_ids, tt_overhang, want_ids = _cut_stream(stream, mesh_device, service)
    assert list(tt_ids.shape) == [1, 1, L], f"trunk is {tt_ids.shape}, expected the plain-path [1,1,{L}]"
    assert list(tt_overhang.shape) == [1, 1, OVERHANG], f"overhang row is {tt_overhang.shape}"

    plain = torch.from_numpy(_producer_rows(stream[:CHUNK].tolist(), 0).astype("int64"))
    assert plain.shape == (SP, 1, L), f"the MTP-off producer row is {tuple(plain.shape)}"

    for d, (g_trunk, g_over) in enumerate(zip(_shards(tt_ids), _shards(tt_overhang))):
        c = d // TP
        assert torch.equal(g_trunk.flatten().to(torch.int64), plain[c, 0]), (
            f"dev {d} (chip {c}): the trunk cut off the MTP row is NOT the row a plain run sends -- "
            "turning MTP on moved the ids the model prefills"
        )
        assert torch.equal(g_over.flatten().to(torch.int64), want_ids[c, L:]), (
            f"dev {d} (chip {c}): the tail is not stream[{c * L + L} : {c * L + UNION_ROWS}] -- "
            "every MTP window on this chip is misaligned"
        )

    for t in (tt_ids, tt_overhang):
        ttnn.deallocate(t)

    # Second push: the split lives in compile-time args, so a cache miss here would mean the op
    # rebuilds the program on every chunk of every request.
    pre = mesh_device.num_program_cache_entries()
    tt_ids, tt_overhang, _ = _cut_stream(stream, mesh_device, service)
    assert mesh_device.num_program_cache_entries() == pre, "the split op recompiled instead of cache-hitting"
    for d, (g_trunk, g_over) in enumerate(zip(_shards(tt_ids), _shards(tt_overhang))):
        c = d // TP
        assert torch.equal(g_trunk.flatten().to(torch.int64), want_ids[c, :L]), f"dev {d}: 2nd push trunk differs"
        assert torch.equal(g_over.flatten().to(torch.int64), want_ids[c, L:]), f"dev {d}: 2nd push overhang differs"
    for t in (tt_ids, tt_overhang):
        ttnn.deallocate(t)

    service.barrier()
    del service


@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_union_windows_are_exact_slices_of_the_producers_stream(mesh_device, stream):
    """Claims 2 and 3: two gathers -> trunk block + K windows, row-exact on all 32 chips.

    The reference is built from the same torch table the device holds, so a passing run means the
    gather identity ``slice(embed(ids)) == embed(slice(ids))`` survives the real ops at the real
    shape -- which is the whole argument for sending the embedding instead of the ids.
    """
    table = torch.randn(VOCAB, HIDDEN, dtype=torch.float32).to(torch.bfloat16)
    embed_fn = _embed_fn(mesh_device, table)
    tt_ids, tt_overhang, want_ids = _cut_stream(stream, mesh_device)

    union = MTPUnionEmbedding.from_ids(tt_ids, tt_overhang, embed_fn, num_levels=K)
    for t in (tt_ids, tt_overhang):
        ttnn.deallocate(t)  # consumed by the gathers, as in _mtp_prepare_input
    trunk, over = union.parts
    assert len(union.parts) == 2, "the first rank holds the union as two gathered blocks, not one"
    assert list(trunk.shape) == [1, 1, L, HIDDEN // TP], f"trunk block is {trunk.shape}"
    # The 32-row gather: its own program-cache entry, at a row count the op never otherwise sees.
    assert list(over.shape) == [1, 1, OVERHANG, HIDDEN // TP], f"overhang block is {over.shape}"

    # Claim 2: the trunk block IS the model input, so it must be the plain embedding of the plain ids.
    assert union.trunk is trunk, "union.trunk must hand back the first block, not a copy or the join"
    _assert_rows(union.trunk, want_ids, table, 0, L, "trunk gather (the model input)")
    _assert_rows(over, want_ids, table, L, OVERHANG, "overhang gather")

    # Claim 3: each level's window, at a row offset that is never tile-aligned, spanning the seam
    # between the two blocks.
    for shift in range(1, K + 1):
        window = union.window(shift)
        _assert_rows(window, want_ids, table, shift, L, "union window")
        ttnn.deallocate(window)

    union.deallocate()


@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_row_concat_pack_survives_the_d2d_split(mesh_device, stream):
    """Claim 4: the hidden and the union's blocks ride one D2D tensor and come back apart unchanged.

    The pack is a 3-way row concat, so the split point is a fixed row offset. Every block height is a
    multiple of 32, which is what keeps the receiver's slice on a tile boundary — the failure this
    pins is a packing that puts the union's first row inside the hidden's last tile.
    """
    table = torch.randn(VOCAB, HIDDEN, dtype=torch.float32).to(torch.bfloat16)
    embed_fn = _embed_fn(mesh_device, table)
    tt_ids, tt_overhang, want_ids = _cut_stream(stream, mesh_device)
    union = MTPUnionEmbedding.from_ids(tt_ids, tt_overhang, embed_fn, num_levels=K)
    for t in (tt_ids, tt_overhang):
        ttnn.deallocate(t)

    # Stand in for the trunk's output: a distinguishable per-chip hidden, so a split that lands on the
    # wrong row shows up as embedding rows appearing in the hidden half or vice versa.
    hidden = ttnn.from_torch(
        torch.full((1, 1, SP * L, HIDDEN), -1.0, dtype=torch.float32),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.create_mesh_mapper(
            mesh_device, ttnn.MeshMapperConfig(placements=[ttnn.PlacementShard(2), ttnn.PlacementShard(3)])
        ),
    )
    packed = ttnn.concat([hidden, *union.parts], dim=2)
    assert list(packed.shape) == [1, 1, L + UNION_ROWS, HIDDEN // TP], f"packed activation is {packed.shape}"

    s = list(packed.shape)
    got_hidden = ttnn.slice(packed, [0, 0, 0, 0], [s[0], s[1], L, s[3]])
    got_union = ttnn.slice(packed, [0, 0, L, 0], [s[0], s[1], L + UNION_ROWS, s[3]])

    for d, g in enumerate(_shards(got_hidden)):
        assert torch.equal(g, torch.full_like(g, -1.0)), f"dev {d}: the hidden half came back altered"

    received = MTPUnionEmbedding.from_embedding(got_union, K, L)
    for shift in range(1, K + 1):
        window = received.window(shift)
        _assert_rows(window, want_ids, table, shift, L, "unpacked window")
        ttnn.deallocate(window)

    received.deallocate()
    union.deallocate()
    for t in (packed, got_hidden, hidden):
        ttnn.deallocate(t)


@pytest.mark.parametrize("real_len", GENERATION_CASES)
@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_generated_embedding_reaches_every_window_that_reads_it(mesh_device, stream, real_len):
    """Claim 5: the last chunk's generation writes one row per position and every level picks it up.

    The chain's two halves are separable and only this one is geometry. The LM head, the argmax and
    the embedding gather are exercised whole by the runner e2e; what a wrong answer here looks like
    is a *silent* one -- an embedding written to a row no window reads, or to a row some window reads
    at the wrong level -- so it is measured against an explicitly constructed union, row by row, on
    all 32 chips. ``embed_fn``'s output is stood in for by a known ``[1, 1, 32*sp, H/tp]`` block per
    level, which is exactly what the SP all-gather delivers.

    Both cases matter and they are geometrically different: at ``real_len == CHUNK`` (GLM-5.2's
    56320 = 11 x 5120, the shape the demo actually runs) the generated positions live in the LAST
    chip's overhang and one chip holds each; at a real length that ends mid-chunk they land on the
    seam, where two chips' unions overlap and BOTH must be patched or the windows disagree across it.
    """
    table = torch.randn(VOCAB, HIDDEN, dtype=torch.float32).to(torch.bfloat16)
    embed_fn = _embed_fn(mesh_device, table)
    tt_ids, tt_overhang, want_ids = _cut_stream(stream, mesh_device)
    union = MTPUnionEmbedding.from_ids(tt_ids, tt_overhang, embed_fn, num_levels=K)
    for t in (tt_ids, tt_overhang):
        ttnn.deallocate(t)
    assert union.overhang == OVERHANG, f"union reports overhang {union.overhang}, expected {OVERHANG}"

    geom = dict(
        mesh_device=mesh_device,
        sp_factor=SP,
        chunk_size=CHUNK,
        mesh_shape=(SP, TP),
        sp_axis=0,
        overhang=OVERHANG,
        chunk_start=0,
        actual_end=real_len,
    )
    rows = [
        mtp_generation_union_rows(SP, CHUNK, overhang=OVERHANG, chunk_start=0, actual_end=real_len, level=k)
        for k in range(K)
    ]
    keep_mask = build_mtp_generation_keep_mask(**geom, emb_dim_per_chip=HIDDEN // TP, num_levels=K)
    selects = [build_mtp_generation_select(**geom, level=k, source_row=SOURCE_ROW) for k in range(K)]

    # What the SP all-gather hands each level: 32 rows per chip, replicated across SP, TP-sharded.
    # Only row SOURCE_ROW is ever read; the rest are the other chips' tiles and must be ignored.
    gathered_host = [torch.randn(1, 1, ttnn.TILE_SIZE * SP, HIDDEN).to(torch.bfloat16) for _ in range(K)]
    gathered = [
        ttnn.from_torch(
            g.float(),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.create_mesh_mapper(
                mesh_device,
                ttnn.MeshMapperConfig(placements=[ttnn.PlacementReplicate(), ttnn.PlacementShard(3)]),
            ),
        )
        for g in gathered_host
    ]

    # The union this must produce, built here from the ids and the known blocks -- not read back and
    # re-described. Cleared rows start at zero and level k's patch fills its own.
    want = torch.stack([table[want_ids[c]] for c in range(SP)]).clone()  # [SP, U, HIDDEN]
    for k in range(K):
        for c, u in enumerate(rows[k]):
            if u is not None:
                want[c, u] = 0.0

    union.clear_rows(keep_mask)
    for k in range(K):
        union.add_patch(selects[k], gathered[k])
        for c, u in enumerate(rows[k]):
            if u is not None:
                want[c, u] = gathered_host[k][0, 0, SOURCE_ROW]
        window = union.window(k + 1)
        _assert_union_rows(window, want, k + 1, L, f"level {k} window after generating level {k}")
        ttnn.deallocate(window)

    patched = {u for r in rows for u in r if u is not None}
    assert patched, "no union row was patched; the case is vacuous"
    union.deallocate()
    for t in (keep_mask, *selects, *gathered):
        ttnn.deallocate(t)
