# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""No-device tests of the multi-user pack's host side (``ttnn/pack.py``): the bucket plan, the constant ascending block-id rows
and the guard that rejects any other order, the host-written masks and selects, the landing layout, and the default-off
state of the conv reader's segment restart and of the prims' head count."""

import inspect
from pathlib import Path

import pytest
import torch

from models.demos.blackhole.qwen38_flash_next.ttnn import pack as pack_module
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gdn_pre_rows, gdn_source_chunk

FUSED = Path(gdn_pre_rows.__file__).parent


@pytest.mark.parametrize(
    "rows, users, launches",
    [
        (128, 16, ((0, 8), (8, 8))),
        (256, 8, ((0, 8),)),
        (512, 4, ((0, 4),)),
        (1024, 2, ((0, 2),)),
    ],
)
def test_bucket_plan(rows, users, launches):
    bucket = pack_module.pack_bucket(rows)
    assert bucket.users == users and bucket.launches == launches
    assert bucket.users * bucket.rows_per_user == pack_module.PACK_ROWS
    assert bucket.seg_chunks == rows // 32 and bucket.blocks_per_user == rows // 4
    # at most 8 users per launch: 12 * 9 = 108 heads is the scan's core cap, 8 keeps the one-value-block form
    assert all(count <= pack_module.GDN_LAUNCH_USERS for _first, count in bucket.launches)
    assert sum(count for _first, count in bucket.launches) == users
    for user in range(users):
        launch, head0 = pack_module.gdn_state_slot(bucket, user)
        first, count = bucket.launches[launch]
        assert first <= user < first + count and head0 == 12 * (user - first)


def test_bucket_rejects_other_rows(expect_error):
    for rows in (0, 64, 127, 129, 2048, 4096):
        with expect_error(ValueError):
            pack_module.pack_bucket(rows)


def test_bucket_for_length_and_plan_packs(expect_error):
    assert [pack_module.bucket_for_length(n) for n in (1, 128, 129, 256, 257, 512, 513, 1024, 1025)] == [
        128,
        128,
        256,
        256,
        512,
        512,
        1024,
        1024,
        None,
    ]
    with expect_error(ValueError):
        pack_module.bucket_for_length(0)
    lengths = [20, 130, 100, 5000, 1000, 128, 300] + [50] * 16
    plan = pack_module.plan_packs(lengths)
    flat = [i for _rows, indices in plan for i in indices]
    assert sorted(flat) == [i for i, n in enumerate(lengths) if n <= 1024]  # the 5000-token prompt is left out
    for rows, indices in plan:
        assert len(indices) <= pack_module.PACK_ROWS // rows
        assert all(pack_module.bucket_for_length(lengths[i]) == rows for i in indices)


@pytest.mark.parametrize("rows", pack_module.PACK_BUCKETS)
def test_block_ids_are_ascending_global_and_per_user(rows):
    bucket = pack_module.pack_bucket(rows)
    host = pack_module.block_id_rows(bucket)
    assert tuple(host.shape) == (2048, 512)
    pack_module.assert_block_ids_ascending(host)
    per = rows // 4
    for user in range(bucket.users):
        segment = host[user * rows : (user + 1) * rows]
        assert bool((segment == segment[0]).all()), "every row of a user holds the same constant list"
        assert segment[0, :per].tolist() == list(range(user * per, (user + 1) * per))
        assert bool((segment[0, per:] == pack_module.BLOCK_SENTINEL).all())


def test_block_id_guard_rejects_any_other_order(expect_error):
    good = pack_module.block_id_rows(pack_module.pack_bucket(256))
    pack_module.assert_block_ids_ascending(good)
    descending = good.clone()
    descending[:, :64] = descending[:, :64].flip(1)
    with expect_error(ValueError, match="ascending"):
        pack_module.assert_block_ids_ascending(descending)
    swapped = good.clone()
    swapped[5, 3], swapped[5, 4] = good[5, 4].item(), good[5, 3].item()
    with expect_error(ValueError, match="ascending"):
        pack_module.assert_block_ids_ascending(swapped)
    duplicate = good.clone()
    duplicate[0, 1] = duplicate[0, 0]
    with expect_error(ValueError, match="ascending"):
        pack_module.assert_block_ids_ascending(duplicate)
    hole = good.clone()
    hole[7, 10] = pack_module.BLOCK_SENTINEL  # a sentinel before real ids
    with expect_error(ValueError, match="sentinel"):
        pack_module.assert_block_ids_ascending(hole)
    with expect_error(ValueError):
        pack_module.assert_block_ids_ascending(good[0])


def test_the_kernel_rule_selects_every_block_of_the_user_for_a_complete_row():
    """The tiled kernel's reader rule on the host: read the first ``min(512, complete)`` ids of the row, skip ids >= complete.
    With the ascending constant list the visible set of every row is exactly its user's complete blocks."""

    for rows in pack_module.PACK_BUCKETS:
        bucket = pack_module.pack_bucket(rows)
        host = pack_module.block_id_rows(bucket)
        for user in range(bucket.users):
            for local in (0, 3, 4, rows // 2 + 1, rows - 1):
                position = user * rows + local  # the global position of the row
                complete = (position + 1) >> 2
                visible = [int(b) for b in host[position, : min(512, complete)] if int(b) < complete]
                expected = list(range(user * (rows // 4), user * (rows // 4) + (local + 1) // 4))
                assert visible == expected, (rows, user, local)


def test_local_positions():
    bucket = pack_module.pack_bucket(512)
    positions = pack_module.local_positions(bucket)
    assert positions.dtype == torch.int64 and tuple(positions.shape) == (3, 2048)
    assert positions[0, :512].tolist() == list(range(512)) and positions[0, 512:1024].tolist() == list(range(512))
    assert bool((positions[0] == positions[1]).all() and (positions[0] == positions[2]).all())
    assert pack_module.global_positions().tolist() == list(range(2048))


def test_token_rows_pad_and_empty_slots(expect_error):
    bucket = pack_module.pack_bucket(128)
    rows = pack_module.pack_token_rows(bucket, [[5, 6, 7], list(range(1, 129))], pad_token=0)
    assert len(rows) == 16 and all(len(r) == 128 for r in rows)
    assert rows[0][:3] == [5, 6, 7] and rows[0][3:] == [0] * 125
    assert rows[1] == list(range(1, 129)) and rows[2] == [0] * 128
    with expect_error(ValueError):
        pack_module.pack_token_rows(bucket, [list(range(129))])
    with expect_error(ValueError):
        pack_module.pack_token_rows(bucket, [[1]] * 17)


def test_input_images():
    bucket = pack_module.pack_bucket(256)
    lengths = [100, 2, 256, 0, 33]
    images = pack_module.pack_input_images(bucket, lengths)
    mask = images["row_mask_c"]
    assert tuple(mask.shape) == (12, 64, 32, 1)
    flat = mask[0].reshape(-1)
    for user, n in enumerate(lengths + [0] * 3):
        assert flat[user * 256 : user * 256 + n].eq(1).all() and flat[user * 256 + n : (user + 1) * 256].eq(0).all()
    assert bool((mask == mask[0:1]).all())
    gather = images["gather_select"].float()
    conv = images["conv_select"].float()
    assert tuple(gather.shape) == (32, 2048) and tuple(conv.shape) == (512, 2048)
    for user, n in enumerate(lengths + [0] * 3):
        if n:
            assert gather[user].nonzero().flatten().tolist() == [user * 256 + n - 1] and gather[user].sum() == 1
        else:
            assert gather[user].sum() == 0
        tile = conv[32 * user : 32 * user + 32]
        expected = [user * 256 + n - 3 + i for i in range(3) if n and n - 3 + i >= 0]
        got = [int(tile[i].nonzero().flatten()[0]) for i in range(3) if tile[i].sum() > 0]
        assert got == expected, (user, n)
        assert tile[4:].sum() == 0
        if n >= 4:
            assert int(tile[3].nonzero().flatten()[0]) == user * 256 + n - 4
        else:
            assert tile[3].sum() == 0
    assert gather[8:].sum() == 0  # the unused select rows
    take, keep = images["ple_take"], images["ple_keep"]
    assert take.sum() == 4 and bool((take + keep == 1).all())  # four users with rows
    for user, n in enumerate(lengths):
        if n:
            assert take[(user * 256 + n - 1) // 128] == 1


def test_landing_rows_layout():
    bucket = pack_module.pack_bucket(128)
    rows = pack_module.landing_rows(bucket, [5, 128, 0, 77] + [3] * 12)
    assert rows[1]["kv_row_base"] == 128 and rows[1]["compressed_row_base"] == 32 and rows[1]["last_row"] == 255
    assert rows[1]["staging_tile"] == 7 and rows[1]["ple_history_first_row"] == 128
    assert rows[2]["tokens"] == 0 and rows[2]["last_row"] is None
    assert (rows[9]["gdn_state_launch"], rows[9]["gdn_state_head0"]) == (1, 12) and rows[9]["conv_row0"] == 288
    assert pack_module.ple_history_first_row(1) == 1 and pack_module.ple_history_first_row(129) == 1


def test_check_lengths(expect_error):
    bucket = pack_module.pack_bucket(1024)
    assert pack_module.check_lengths(bucket, [5]) == [5, 0]
    for bad in ([1025], [1, 2, 3], [-1]):
        with expect_error(ValueError):
            pack_module.check_lengths(bucket, bad)


def test_pack_options_are_default_off():
    """Today's callers reproduce today's programs: 12 heads, no segment restart (no define at all)."""

    for function in (gdn_source_chunk.prepare, gdn_source_chunk.scan, gdn_source_chunk.chunk_source):
        assert inspect.signature(function).parameters["heads"].default == gdn_source_chunk.HEADS == 12
    assert inspect.signature(gdn_source_chunk.chunk_source).parameters["state_output"].default is None
    assert inspect.signature(gdn_pre_rows.run).parameters["seg_chunks"].default == 0
    source = inspect.getsource(gdn_pre_rows.run)
    assert 'segment_defines = [("SEG_CHUNKS", str(seg_chunks))] if seg_chunks else []' in source
    for kernel in ("reader_qkv", "compute_qkv"):
        text = (FUSED / "kernels" / f"{kernel}.cpp").read_text()
        assert "#ifndef SEG_CHUNKS\n#define SEG_CHUNKS 0\n#endif" in text
        assert "SEGMENT_CHUNKS == 0 ? chunk == 0 : chunk % SEGMENT_CHUNKS == 0" in text
    assert "segment_start(chunk)" in (FUSED / "kernels" / "reader_qkv.cpp").read_text()
    assert "chunk == 0 ?" not in (FUSED / "kernels" / "compute_qkv.cpp").read_text()
