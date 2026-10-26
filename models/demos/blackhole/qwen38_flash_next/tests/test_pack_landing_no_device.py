# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""No-device tests of the pack landing (P3d-3): the host images of every landing select, the ring placement against the
importer's canonical rule, the pack knobs and the default-off options of the touched kernels."""

import inspect

import pytest
import torch

from models.demos.blackhole.qwen38_flash_next.tools import qwen38_batch_decode as batch
from models.demos.blackhole.qwen38_flash_next.ttnn import pack as pack_module
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import kv_page_scatter as scatter_module
from models.demos.blackhole.qwen38_flash_next.ttnn.lanes import canonical_ring_source

ROOT = scatter_module.__file__.rsplit("/", 1)[0]


# ------------------------------------------------------------------------------------------------------ the GDN ring


@pytest.mark.parametrize("position", list(range(0, 13)) + [31, 32, 33, 1021, 1022, 1023, 1024])
def test_ring_slot_sources_follow_the_importers_canonical_rule(position):
    """A single-path lane after ``P`` tokens holds, in canonical slot ``j``, the generic slot ``(P + j) % 4``, and generic
    slot ``s`` holds the conv-input row of the newest token ``t < P`` with ``t % 4 == s`` (the four newest rows).  The pack
    stores ``conv_rows`` rows ``[P - 3, P - 2, P - 1, P - 4]`` (tile rows 0..3); each slot's source tile row must name that
    token whatever ``P % 4`` is."""

    tile_token = [position - 3, position - 2, position - 1, position - 4]
    for slot in range(4):
        generic_slot = canonical_ring_source(slot, position) % 4
        token = max(t for t in range(position - 4, position) if t % 4 == generic_slot)
        assert token == position - 4 + slot
        assert tile_token[pack_module.conv_ring_source_row(slot)] == token


def test_conv_ring_source_rows_and_bad_slot(expect_error):
    assert [pack_module.conv_ring_source_row(j) for j in range(4)] == [3, 0, 1, 2]
    with expect_error(ValueError):
        pack_module.conv_ring_source_row(4)


def test_conv_images_place_one_row_per_landed_lane():
    pairs = [(7, 0), (0, 15), (3, 2)]
    keep = pack_module.conv_keep_image(8, pairs)
    assert keep.shape == (8, 1) and keep[:, 0].tolist() == [0, 1, 1, 0, 1, 1, 1, 0]
    for slot in range(4):
        hit = pack_module.conv_hit_image(8, pairs, slot).float()
        assert tuple(hit.shape) == (8, 512)
        for lane in range(8):
            cols = hit[lane].nonzero().flatten().tolist()
            expected = [32 * user + pack_module.conv_ring_source_row(slot) for ln, user in pairs if ln == lane]
            assert cols == expected
    # the placement is the exact selection write: slot * keep + hit @ conv_rows
    conv_rows = torch.randn(512, 16).to(torch.bfloat16)
    slot = torch.randn(8, 16).to(torch.bfloat16)
    out = slot.float() * keep.float() + pack_module.conv_hit_image(8, pairs, 1).float() @ conv_rows.float()
    assert torch.equal(out[7], conv_rows[0].float()) and torch.equal(out[0], conv_rows[32 * 15].float())
    assert torch.equal(out[1], slot[1].float()) and torch.equal(out[3], conv_rows[32 * 2].float())


def test_conv_images_reject_out_of_range(expect_error):
    with expect_error(ValueError):
        pack_module.conv_hit_image(8, [(8, 0)], 0)
    with expect_error(ValueError):
        pack_module.conv_hit_image(8, [(0, 16)], 0)


# ---------------------------------------------------------------------------------------- the selects, applied to ids


def rows_of(select: torch.Tensor, source_rows: int) -> list:
    """For each select row the source row it picks (None for a zero row), the matmul read as a gather."""

    out = []
    for row in select.float():
        cols = row.nonzero().flatten().tolist()
        assert len(cols) <= 1 and (not cols or row[cols[0]] == 1.0)
        assert not cols or cols[0] < source_rows
        out.append(cols[0] if cols else None)
    return out


@pytest.mark.parametrize("rows", pack_module.PACK_BUCKETS)
def test_compressed_select_keeps_only_complete_blocks_of_the_user(rows):
    bucket = pack_module.pack_bucket(rows)
    for user in (0, bucket.users - 1):
        for length in sorted({1, 3, 4, 5, 9, rows // 2 + 1, rows - 1, rows}):
            picked = rows_of(pack_module.compressed_select_image(bucket, user, length), 512)
            complete = length // 4
            base = user * rows // 4
            assert len(picked) == rows // 4
            assert picked[:complete] == [base + i for i in range(complete)]
            assert all(p is None for p in picked[complete:])


@pytest.mark.parametrize("rows", pack_module.PACK_BUCKETS)
def test_staging_select_is_the_tile_of_the_last_token(rows):
    bucket = pack_module.pack_bucket(rows)
    for user in (0, bucket.users - 1):
        for length in sorted({1, 2, 31, 32, 33, 63, 64, 65, rows - 1, rows}):
            if length > rows:
                continue
            picked = rows_of(pack_module.staging_select_image(bucket, user, length), pack_module.PACK_ROWS)
            tile, count = (length - 1) // 32, (length - 1) % 32 + 1
            base = user * rows + 32 * tile
            assert picked[:count] == [base + i for i in range(count)]
            assert all(p is None for p in picked[count:])
            assert base + count - 1 == user * rows + length - 1  # the last real token is the last picked row


@pytest.mark.parametrize("rows", pack_module.PACK_BUCKETS)
def test_raw_ring_select_is_the_open_block_up_to_the_last_token(rows):
    bucket = pack_module.pack_bucket(rows)
    for user in (0, bucket.users - 1):
        for length in sorted({1, 2, 3, 4, 5, 6, 7, 8, 9, rows - 1, rows}):
            picked = rows_of(pack_module.raw_ring_select_image(bucket, user, length), pack_module.PACK_ROWS)
            block, count = (length - 1) // 4, (length - 1) % 4 + 1
            assert picked[:count] == [user * rows + 4 * block + j for j in range(count)]
            assert all(p is None for p in picked[count:])
            assert user * rows + 4 * block + count - 1 == user * rows + length - 1


def test_ring_select_agrees_with_the_hand_off_ring_for_the_prefix_the_single_path_leaves():
    """After the hand-off at ``Q = P - 1`` the ring holds rows ``j < Q % 4`` (``qsa.chunk_handoff_ring_select_rows``) and the
    forced step adds row ``Q % 4``: the landing's ring is the hand-off ring plus that one row."""

    from models.demos.blackhole.qwen38_flash_next.ttnn import qsa as qsa_module

    bucket = pack_module.pack_bucket(256)
    for length in range(1, 70):
        q = length - 1
        hand_off = qsa_module.chunk_handoff_ring_select_rows(q)[0, 0].float()
        rows_hand_off = rows_of(hand_off, 32)
        landing = rows_of(pack_module.raw_ring_select_image(bucket, 0, length), pack_module.PACK_ROWS)
        # hand-off row j picks kept row q % 32 - q % 4 + j: position (q & ~3) + j; the landing picks the same positions
        for j in range(q % 4):
            assert rows_hand_off[j] is not None and landing[j] == 4 * (q // 4) + j
            assert rows_hand_off[j] == (q % 32) - (q % 4) + j
        assert landing[q % 4] == q
        assert all(p is None for p in landing[q % 4 + 1 :])


@pytest.mark.parametrize("length", [1, 2, 8, 9, 10, 127, 128, 129, 200, 1024])
def test_ple_history_select_picks_the_nine_rows_ending_at_the_last_token(length):
    select = pack_module.ple_history_select_image(length).float()
    assert tuple(select.shape) == (137, 32)
    picked = [select[:, k].nonzero().flatten().tolist() for k in range(32)]
    assert picked[9:] == [[]] * 23
    # window row r < 9 is the zero history, row r >= 9 is normalized row r - 9 of the block that holds the last token
    block0 = ((length - 1) // 128) * 128
    for k in range(9):
        (row,) = picked[k]
        position = block0 + (row - 9)  # the sequence position of that window row
        assert position == length - 9 + k
    assert pack_module.ple_history_first_row(length) == picked[0][0]


def test_selects_reject_bad_lengths(expect_error):
    bucket = pack_module.pack_bucket(128)
    with expect_error(ValueError):
        pack_module.staging_select_image(bucket, 0, 0)
    with expect_error(ValueError):
        pack_module.raw_ring_select_image(bucket, 0, 129)
    with expect_error(ValueError):
        pack_module.compressed_select_image(bucket, 0, 129)
    with expect_error(ValueError):
        pack_module.ple_history_select_image(0)


# ----------------------------------------------------------------------------------------------------- the pack knobs


def test_pack_knob_defaults_to_off_and_parses_strictly(expect_error):
    assert batch.prefill_pack_enabled({}) is False
    assert batch.prefill_pack_enabled({"QWEN38_PREFILL_PACK": "0"}) is False
    assert batch.prefill_pack_enabled({"QWEN38_PREFILL_PACK": "1"}) is True
    for bad in ("2", "true", "", "on"):
        with expect_error(ValueError):
            batch.prefill_pack_enabled({"QWEN38_PREFILL_PACK": bad})


def test_pack_buckets_knob(expect_error):
    assert batch.pack_buckets_from_env({}) == (128, 256, 512, 1024)  # lanes unknown
    assert batch.pack_buckets_from_env({}, lanes=8) == (256, 512, 1024)
    assert batch.pack_buckets_from_env({"QWEN38_PACK_BUCKETS": "512, 128,128"}) == (128, 512)
    for bad in ("64", "128,2048", "x", ""):
        with expect_error(ValueError):
            batch.pack_buckets_from_env({"QWEN38_PACK_BUCKETS": bad})


def test_pack_dram_estimate_counts_state_launches_and_traces():
    mib = 1 << 20
    full = batch.pack_dram_bytes((128, 256, 512, 1024))
    assert 905 * mib < full < 935 * mib  # measured 918.6 MiB: 640.6 state + 18.8 landing buffers + 259.2 of traces
    no_128 = batch.pack_dram_bytes((256, 512, 1024))
    tail_128 = (32768 // 4 - 128 // 4) * batch.PACK_TAIL_ROW_BYTES
    assert (
        full - no_128 == batch.PACK_LAUNCH_BYTES + batch.PACK_TRACE_BYTES[128] + tail_128
    )  # one GDN launch, one trace and one zero tail fewer
    assert batch.pack_dram_bytes((1024,)) < batch.pack_dram_bytes((512, 1024))


def test_the_decoder_is_default_off_without_the_knob(monkeypatch):
    monkeypatch.delenv("QWEN38_PREFILL_PACK", raising=False)
    decoder = batch.Qwen38PlainLaneDecoder(8)
    assert decoder.pack_enabled is False and decoder.pack_buckets == () and decoder.lander is None
    monkeypatch.setenv("QWEN38_PREFILL_PACK", "1")
    monkeypatch.setenv("QWEN38_PACK_BUCKETS", "128,1024")
    decoder = batch.Qwen38PlainLaneDecoder(8)
    assert decoder.pack_enabled and decoder.pack_buckets == (128, 1024)


def test_the_pool_cap_counts_the_pack_only_when_the_knob_is_on(monkeypatch):
    from models.demos.blackhole.qwen38_flash_next.tools import qwen38_vllm as vllm

    monkeypatch.delenv("QWEN38_PREFILL_PACK", raising=False)
    off = vllm.batch_pool_token_cap(32704, 8)
    monkeypatch.setenv("QWEN38_PREFILL_PACK", "1")
    on = vllm.batch_pool_token_cap(32704, 8)
    lost_bytes = (off - on) * vllm.KV_BYTES_PER_TOKEN
    want = batch.pack_dram_bytes(batch.pack_buckets_from_env({}, lanes=8), vllm.resident_allocated_context(32704))
    assert abs(lost_bytes - want) < 2 * vllm.KV_BYTES_PER_TOKEN


# --------------------------------------------------------------------------------------------- the scatter kernel option


def test_scatter_source_offset_is_a_default_off_runtime_argument(expect_error):
    assert inspect.signature(scatter_module.kv_page_scatter).parameters["src_row0"].default == 0
    assert inspect.signature(scatter_module.kv_page_scatter_composed).parameters["src_row0"].default == 0
    kernel = open(f"{ROOT}/kernels/scatter_rows.cpp").read()
    assert "const uint32_t src_row0 = get_arg_val<uint32_t>(10);" in kernel
    assert "src.get_noc_addr(src_row0 + t0 + k, 0)" in kernel
    init = open(f"{ROOT}/__init__.py").read()
    assert "j, SCATTER_CORES, src_row0])" in init  # appended after the existing runtime arguments


class _Shape:
    def __init__(self, shape):
        self.shape = shape
        self.dtype = scatter_module.ttnn.bfloat16
        self.layout = scatter_module.ttnn.ROW_MAJOR_LAYOUT


def test_scatter_check_validates_the_source_window(monkeypatch, expect_error):
    monkeypatch.setattr(scatter_module, "page_table_blocks", lambda table, size, lanes: 512)
    src, pool = _Shape((1, 1, 2048, 512)), _Shape((1, 1, 4096 * 64, 512))
    assert scatter_module._check(src, pool, None, 0, 100, 64) == (4, 512)
    assert scatter_module._check(src, pool, None, 0, 100, 64, 1024) == (4, 512)
    assert scatter_module._check(src, pool, None, 0, 1024, 64, 1024) == (32, 512)
    with expect_error(ValueError):
        scatter_module._check(src, pool, None, 0, 1025, 64, 1024)  # runs past the source
    with expect_error(ValueError):
        scatter_module._check(src, pool, None, 0, 100, 64, 16)  # not a multiple of 32
    with expect_error(ValueError):
        scatter_module._check(src, pool, None, 0, 100, 64, -32)
