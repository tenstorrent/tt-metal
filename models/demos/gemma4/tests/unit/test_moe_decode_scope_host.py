# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The decode MoE kernel wrapper's scoping of the custom moe_compute op and the TILE tail to decode frames.

Galaxy window-2 arms (2026-10-24, eager 128-token lane prefill warm median / traced 128-user step): base 371 ms /
46.2, fused_all 532 / 32.9, without the custom op 371 / 40.7, custom op without the skip 529 / 45.6, without the TILE
tail 538 / 34.4, GLUE only 373 / 44.4: the custom op (+ skip, + TILE tail) is the decode win and, on the 128-row
single-user owner prefill, the +160 ms prefill regression. No device: the predicates only.

    pytest models/demos/gemma4/tests/unit/test_moe_decode_scope_host.py --noconftest -p no:cacheprovider
"""

from types import SimpleNamespace

import pytest

from models.demos.gemma4.tt.experts.moe_decode import kernel_decode_frame, moe_compute_impl, tile_tail_enabled


def _lanes(owner, frame_rows):
    return SimpleNamespace(lane_sharded=True, lanes=4, _g4_single_lane_prefill_owner=owner, frame_rows=frame_rows)


@pytest.mark.parametrize("frame_rows", [128, 256], ids=["block_frame_128", "galaxy_frame_256"])
def test_lane_engine_frames_are_told_apart_by_the_owner_not_the_rows(frame_rows):
    # the 128-user decode step and the 128-token owner prefill both hand the wrapper 128 rows
    assert kernel_decode_frame(_lanes(None, frame_rows), 128) is True
    for owner in range(4):
        assert kernel_decode_frame(_lanes(owner, frame_rows), 128) is False
    assert kernel_decode_frame(_lanes(None, frame_rows), 32) is True


def test_tp_only_shapes_use_the_row_count():
    tp_only = SimpleNamespace(lane_sharded=False)
    assert kernel_decode_frame(tp_only, 1) and kernel_decode_frame(tp_only, 32)
    assert not kernel_decode_frame(tp_only, 64) and not kernel_decode_frame(tp_only, 128)
    assert kernel_decode_frame(None, 8) and not kernel_decode_frame(None, 128)


@pytest.mark.parametrize("custom_enabled", [False, True], ids=["custom_off", "custom_on"])
def test_custom_op_only_on_decode_frames(custom_enabled):
    assert moe_compute_impl(True, custom_enabled) == ("custom" if custom_enabled else "shared")
    assert moe_compute_impl(False, custom_enabled) == "shared"


@pytest.mark.parametrize("batch_per_device", [32, 64], ids=["galaxy_32_per_chip", "block_64_per_chip"])
def test_tile_tail_only_on_decode_frames(batch_per_device):
    assert tile_tail_enabled(True, True, True, batch_per_device) is True
    assert tile_tail_enabled(False, True, True, batch_per_device) is False  # prefill rows: ROW_MAJOR tail
    assert tile_tail_enabled(True, False, True, batch_per_device) is False
    assert tile_tail_enabled(True, True, False, batch_per_device) is False
    assert tile_tail_enabled(True, True, True, 8) is False  # a partial row tile per chip
