# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""On-device gate for the MTP union-embedding transport (GLM-5.2, issue #53533).

The runner moves MTP's K token windows from the producer to the rank that runs the levels without a
host round trip anywhere. Seven hardware claims carry it:

1. **The producer's three tensors survive ONE H2D socket.** It builds the chunk (``[SP,1,L]``,
   untouched by MTP), the lookahead (``[SP,1,N]``, row ``c`` = ``stream[(c+1)*L : +N]``, where
   ``N = num_mtp_tokens(K)``)
   and the metadata as three separate buffers; the socket carries one global tensor per transfer, so
   the first two share chip ``c``'s page and ``inbound_socket_service_sync`` splits them apart again
   as it copies out. ``tt_ids`` must come back byte-identical to what an MTP-*off* run pushes and
   ``tt_mtp_tokens`` must be the ``N`` ids that immediately follow it -- otherwise level ``k``'s
   window is off by a chip's worth of positions. This drives a REAL ``H2DStreamService``, the real op
   and the real ``producer._push``, so the two ends cannot agree here and disagree in production.
2. **Two gathers, and the trunk gather IS the model's input.** The union is embedded as two blocks,
   never as one rejoined id row, so the trunk block can be handed to the transformer directly
   (``input_is_embedded``) instead of gathering the same ``L`` rows a second time. The 32-row
   lookahead gather is the small, unusual one -- a second program-cache entry at a row count the op
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
6. **The position-0 mask zeroes exactly the row holding absolute position 0.** vLLM zeroes the MTP
   embedding there on every level. Under SP the row index is not the absolute position, so the mask
   derives its row by pushing an indicator through the sharding path rather than assuming chip 0
   row 0 -- and that derivation is what is measured, under both layouts.
7. **Ids above 16 bits survive both upload paths.** GLM-5.2's vocab is 154880, so real ids run past
   65535 while every other case here uses ids under 8192, where a narrowing would be invisible.

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
from models.demos.common.prefill.runners.runner_utils import h2d_row_len, mtp_union_rows, num_mtp_tokens
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tt.mtp_prefill.device_windows import MTPUnionEmbedding
from models.demos.deepseek_v3_d_p.tt.runners.input_prep import (
    build_mtp_generation_keep_mask,
    build_mtp_generation_select,
    build_position_zero_mask,
    mtp_generation_union_rows,
    prepare_prefill_input_tensor,
)
from models.demos.deepseek_v3_d_p.tt.tt_parallel_embedding import TtParallelEmbedding

SP, TP = 8, 4
SP_AXIS = 0
"""The mesh is (sp, tp) row-major throughout: device ``d`` is chip ``d // TP``, TP shard ``d % TP``."""
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
NUM_MTP_TOKENS = num_mtp_tokens(K)
UNION_ROWS = mtp_union_rows(CHUNK, SP, K)

METADATA_SIZE_BYTES = 12
"""3 x uint32 [slot_id, actual_start, actual_end] -- what prefill_runner sends."""

SOURCE_ROW = 5 * 32 + 11
"""The row of the gathered LM-head block that holds the generated token: ``device_id * 32 +
token_offset``. Fixed here (the real value comes from ``global_to_local_token_id``) and deliberately
not 0, so a patch that ignored it would show up."""

GENERATION_CASES = [
    pytest.param(CHUNK, id="ends-on-the-chunk-edge"),
    pytest.param(L + 10, id="ends-on-the-lookahead-seam"),
]
"""Real lengths for claim 5. The first is production (56320 = 11 x 5120): the generated positions
sit in the LAST chip's lookahead and exactly one chip holds each. The second ends 10 rows into chip
1's shard, where chip 0's lookahead covers the same positions -- two chips must be patched."""

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


def _producer_tensors(pool: list[int], levels: int):
    """The producer's two token tensors for the chunk at position 0 of ``pool``, at a chosen MTP level
    count: ``(chunk [SP,1,L], lookahead [SP,1,NUM_MTP_TOKENS] or None)``. ``levels=0`` gives the plain path,
    whose chunk tensor is the whole push."""
    os.environ["PREFILL_MTP_LEVELS"] = str(levels)
    producer._load_env_config()
    try:
        return producer._h2d_rows(producer._chunk_slice(pool, 0)), producer._mtp_rows(pool, 0)
    finally:
        os.environ["PREFILL_MTP_LEVELS"] = str(K)
        producer._load_env_config()


@pytest.fixture
def stream() -> torch.Tensor:
    """The producer's token pool for one chunk: CHUNK ids plus the lookahead past its right edge."""
    torch.manual_seed(0)
    return torch.randint(0, VOCAB, (CHUNK + NUM_MTP_TOKENS,), dtype=torch.int64)


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
    """One producer push through a real H2D socket: ``(trunk ids, MTP ids, want_ids grid)``.

    The op returns the three tensors; nothing here slices anything.
    """
    own = service is None
    service = service or _h2d_service(mesh_device)
    try:
        chunk_rows, lookahead_rows = _producer_tensors(stream.tolist(), K)
        producer._push(service, SP * UNION_ROWS * 4, chunk_rows, lookahead_rows, struct.pack("<III", 0, 0, CHUNK))
        tt_ids, tt_mtp_tokens, tt_meta = ttnn.experimental.deepseek_prefill.inbound_socket_service_sync(
            service, metadata_size_bytes=METADATA_SIZE_BYTES, overhang_size_bytes=NUM_MTP_TOKENS * 4
        )
    finally:
        if own:
            service.barrier()
    ttnn.deallocate(tt_meta)
    want_ids = torch.stack([stream[c * L : c * L + UNION_ROWS] for c in range(SP)])  # [SP, L+NUM_MTP_TOKENS]
    return tt_ids, tt_mtp_tokens, want_ids


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
    tt_ids, tt_mtp_tokens, want_ids = _cut_stream(stream, mesh_device, service)
    assert list(tt_ids.shape) == [1, 1, L], f"trunk is {tt_ids.shape}, expected the plain-path [1,1,{L}]"
    assert list(tt_mtp_tokens.shape) == [1, 1, NUM_MTP_TOKENS], f"MTP token row is {tt_mtp_tokens.shape}"

    plain_rows, plain_lookahead = _producer_tensors(stream[:CHUNK].tolist(), 0)
    assert plain_lookahead is None, "the MTP-off producer must push no lookahead tensor at all"
    plain = torch.from_numpy(plain_rows.astype("int64"))
    assert plain.shape == (SP, 1, L), f"the MTP-off producer row is {tuple(plain.shape)}"

    for d, (g_trunk, g_over) in enumerate(zip(_shards(tt_ids), _shards(tt_mtp_tokens))):
        c = d // TP
        assert torch.equal(g_trunk.flatten().to(torch.int64), plain[c, 0]), (
            f"dev {d} (chip {c}): the trunk cut off the MTP row is NOT the row a plain run sends -- "
            "turning MTP on moved the ids the model prefills"
        )
        assert torch.equal(g_over.flatten().to(torch.int64), want_ids[c, L:]), (
            f"dev {d} (chip {c}): the tail is not stream[{c * L + L} : {c * L + UNION_ROWS}] -- "
            "every MTP window on this chip is misaligned"
        )

    for t in (tt_ids, tt_mtp_tokens):
        ttnn.deallocate(t)

    # Second push: the split lives in compile-time args, so a cache miss here would mean the op
    # rebuilds the program on every chunk of every request.
    pre = mesh_device.num_program_cache_entries()
    tt_ids, tt_mtp_tokens, _ = _cut_stream(stream, mesh_device, service)
    assert mesh_device.num_program_cache_entries() == pre, "the split op recompiled instead of cache-hitting"
    for d, (g_trunk, g_over) in enumerate(zip(_shards(tt_ids), _shards(tt_mtp_tokens))):
        c = d // TP
        assert torch.equal(g_trunk.flatten().to(torch.int64), want_ids[c, :L]), f"dev {d}: 2nd push trunk differs"
        assert torch.equal(g_over.flatten().to(torch.int64), want_ids[c, L:]), f"dev {d}: 2nd push MTP ids differ"
    for t in (tt_ids, tt_mtp_tokens):
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
    tt_ids, tt_mtp_tokens, want_ids = _cut_stream(stream, mesh_device)

    union = MTPUnionEmbedding.from_ids(tt_ids, tt_mtp_tokens, embed_fn, num_levels=K)
    for t in (tt_ids, tt_mtp_tokens):
        ttnn.deallocate(t)  # consumed by the gathers, as in _mtp_prepare_input
    trunk, over = union.parts
    assert len(union.parts) == 2, "the first rank holds the union as two gathered blocks, not one"
    assert list(trunk.shape) == [1, 1, L, HIDDEN // TP], f"trunk block is {trunk.shape}"
    # The 32-row gather: its own program-cache entry, at a row count the op never otherwise sees.
    assert list(over.shape) == [1, 1, NUM_MTP_TOKENS, HIDDEN // TP], f"lookahead block is {over.shape}"

    # Claim 2: the trunk block IS the model input, so it must be the plain embedding of the plain ids.
    assert union.trunk is trunk, "union.trunk must hand back the first block, not a copy or the join"
    _assert_rows(union.trunk, want_ids, table, 0, L, "trunk gather (the model input)")
    _assert_rows(over, want_ids, table, L, NUM_MTP_TOKENS, "lookahead gather")

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
    tt_ids, tt_mtp_tokens, want_ids = _cut_stream(stream, mesh_device)
    union = MTPUnionEmbedding.from_ids(tt_ids, tt_mtp_tokens, embed_fn, num_levels=K)
    for t in (tt_ids, tt_mtp_tokens):
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
    chip's lookahead and one chip holds each; at a real length that ends mid-chunk they land on the
    seam, where two chips' unions overlap and BOTH must be patched or the windows disagree across it.
    """
    table = torch.randn(VOCAB, HIDDEN, dtype=torch.float32).to(torch.bfloat16)
    embed_fn = _embed_fn(mesh_device, table)
    tt_ids, tt_mtp_tokens, want_ids = _cut_stream(stream, mesh_device)
    union = MTPUnionEmbedding.from_ids(tt_ids, tt_mtp_tokens, embed_fn, num_levels=K)
    for t in (tt_ids, tt_mtp_tokens):
        ttnn.deallocate(t)
    assert (
        union.num_mtp_tokens == NUM_MTP_TOKENS
    ), f"union reports {union.num_mtp_tokens} MTP rows, expected {NUM_MTP_TOKENS}"

    geom = dict(
        mesh_device=mesh_device,
        sp_factor=SP,
        chunk_size=CHUNK,
        mesh_shape=(SP, TP),
        sp_axis=0,
        num_mtp_tokens=NUM_MTP_TOKENS,
        chunk_start=0,
        actual_end=real_len,
    )
    rows = [
        mtp_generation_union_rows(SP, CHUNK, num_mtp_tokens=NUM_MTP_TOKENS, chunk_start=0, actual_end=real_len, level=k)
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


@pytest.mark.parametrize("is_balanced", [pytest.param(False, id="block-cyclic"), pytest.param(True, id="balanced")])
@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_position_zero_mask_zeroes_exactly_the_row_holding_position_0(mesh_device, is_balanced):
    """Claim 6: one row zeroed, at full width, and every other row left at 1.

    The row -> position map is MEASURED, not re-derived: this uploads ids ``0..C-1`` through the
    trunk's own uploader and reads them straight back, so *that* is what defines which chunk-local
    position each mesh row holds. The test therefore cannot drift from the production sharding and
    needs to know nothing about either layout -- which is the point of running it under both.

    Full width matters: the mask is materialized at ``H/tp`` rather than broadcast from width 1, so
    a partly-zeroed row would mean the expand went wrong in a way a PCC on the product would hide.
    """
    per_tp = HIDDEN // TP
    tt_ids = prepare_prefill_input_tensor(list(range(CHUNK)), mesh_device, SP, is_balanced, (SP, TP), SP_AXIS)
    positions = [g.flatten().to(torch.int64) for g in _shards(tt_ids)]
    ttnn.deallocate(tt_ids)

    assert sorted(torch.cat(positions[::TP]).tolist()) == list(range(CHUNK)), (
        "the trunk upload did not come back as a permutation of the chunk's positions, so the map "
        "this test measures the mask against is itself wrong"
    )
    for d, pos in enumerate(positions):
        assert torch.equal(pos, positions[(d // TP) * TP]), (
            f"dev {d}: the token tensor is uploaded with dims=(sp_axis, None), i.e. REPLICATED across "
            "TP, so every TP chip of a row must hold the same positions"
        )

    mask = build_position_zero_mask(mesh_device, SP, CHUNK, is_balanced, (SP, TP), SP_AXIS, emb_dim_per_chip=per_tp)
    shards = [g.reshape(-1, per_tp).float() for g in _shards(mask)]
    ttnn.deallocate(mask)

    zeroed = 0
    for d, (g, pos) in enumerate(zip(shards, positions)):
        assert g.shape == (L, per_tp), f"dev {d}: mask shard is {tuple(g.shape)}, expected {(L, per_tp)}"
        want = torch.ones(L, per_tp)
        row0 = (pos == 0).nonzero().flatten()
        if row0.numel():
            want[int(row0)] = 0.0
            zeroed += 1
        bad = (g != want).any(dim=-1).nonzero().flatten()
        assert bad.numel() == 0, (
            f"dev {d} (chip {d // TP}): {bad.numel()} mask rows are wrong, first at local row "
            f"{int(bad[0])} (chunk-local position {int(pos[bad[0]])}) -- the mask must zero the row "
            f"holding absolute position 0 (local row {row0.tolist()}) and nothing else"
        )
    assert zeroed == TP, f"position 0 was masked on {zeroed} devices, expected the {TP} TP shards of one chip"


@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_token_ids_above_16_bits_survive_both_upload_paths(mesh_device):
    """Claim 7: real GLM-5.2 ids (vocab 154880) reach the mesh unaltered.

    Both paths that carry ids are checked, because they share no code: the H2D socket plus the op's
    intra-page split, which is what the runner runs, and ``prepare_prefill_input_tensor``, which is
    what ``TtPrefillRuntime.make_chunk_input`` runs. A 16-bit narrowing anywhere in either would show
    up here as ids folded modulo 65536, and nowhere else in this file -- every other case uses ids
    under 8192 by design, since the vocab stopped mattering to transport once the ids stopped
    crossing a bf16 wire.
    """
    base = 149000
    assert base + CHUNK + NUM_MTP_TOKENS < 154880, "the probe must stay inside GLM-5.2's vocab"
    ids = torch.arange(base, base + CHUNK + NUM_MTP_TOKENS, dtype=torch.int64)
    assert int(ids.min()) > 0xFFFF, "the probe did not actually exercise ids above 16 bits"

    tt_ids, tt_mtp_tokens, want_ids = _cut_stream(ids, mesh_device)
    for d, (g_trunk, g_over) in enumerate(zip(_shards(tt_ids), _shards(tt_mtp_tokens))):
        c = d // TP
        assert torch.equal(g_trunk.flatten().to(torch.int64), want_ids[c, :L]), f"dev {d}: socket trunk ids altered"
        assert torch.equal(g_over.flatten().to(torch.int64), want_ids[c, L:]), f"dev {d}: socket MTP ids altered"
    for t in (tt_ids, tt_mtp_tokens):
        ttnn.deallocate(t)

    host = prepare_prefill_input_tensor(ids[:CHUNK].tolist(), mesh_device, SP, False, (SP, TP), SP_AXIS)
    for d, g in enumerate(_shards(host)):
        c = d // TP
        assert torch.equal(
            g.flatten().to(torch.int64), ids[c * L : (c + 1) * L]
        ), f"dev {d} (chip {c}): prepare_prefill_input_tensor altered the ids it uploaded"
    ttnn.deallocate(host)
