# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ACE LMHead narrow wrapper, incl. the TP (vocab-sharded) guard.

Host-only: the helper functions are pure, and the guard is exercised with a fake ``lm_head``
whose patched ``forward`` should defer to the stock forward whenever ``args.num_devices > 1``
(the LMHead is vocab-sharded across chips). No TT device is opened.

Context: today ACE-Step runs the 5 Hz LM on a 1×1 preprocess chip (num_devices == 1), so the
guard is a no-op and narrowing stays active. It only engages if the LM is moved onto the mesh
under TP, where ``split_sizes`` tile a single device's vocab shard while the band is global.
"""

from __future__ import annotations

import torch

from models.experimental.ace_step_v1_5.ttnn_impl.ace_step_lm_head_narrow import (
    ace_step_narrow_column_band,
    ace_step_patch_lm_head_narrow_forward,
    ace_step_split_column_ranges,
    ace_step_splits_for_band,
)


# --- pure helpers ----------------------------------------------------------


def test_split_column_ranges_are_contiguous():
    assert ace_step_split_column_ranges([10, 20, 5]) == [(0, 10), (10, 30), (30, 35)]


def test_narrow_column_band_inclusive_min_exclusive_max():
    idx = torch.tensor([50, 12, 37, 12])
    assert ace_step_narrow_column_band(idx) == (12, 51)


def test_narrow_column_band_none_for_empty():
    assert ace_step_narrow_column_band(torch.empty(0)) is None
    assert ace_step_narrow_column_band(None) is None


def test_splits_for_band_selects_overlapping_only():
    sizes = [100, 100, 100, 100]  # ranges: [0,100) [100,200) [200,300) [300,400)
    assert ace_step_splits_for_band(sizes, 150, 250) == [1, 2]
    assert ace_step_splits_for_band(sizes, 0, 50) == [0]
    assert ace_step_splits_for_band(sizes, 399, 400) == [3]


# --- TP guard --------------------------------------------------------------


class _FakeArgs:
    def __init__(self, num_devices: int) -> None:
        self.num_devices = num_devices


class _FakeLMHead:
    """Just enough surface for the guard branch of ``narrow_forward``."""

    def __init__(self, num_devices: int) -> None:
        self.args = _FakeArgs(num_devices)
        self._stock_called_with = None
        # A vocab-code band is set so narrowing WOULD trigger without the guard.
        self._ace_narrow_vocab_indices = torch.tensor([10, 11, 12])

        def _stock_forward(x, debug_input_torch=None, debug_weight_torch=None):
            self._stock_called_with = x
            return ("stock", x)

        self.forward = _stock_forward


def test_tp_guard_bypasses_narrowing_when_vocab_sharded():
    lm = _FakeLMHead(num_devices=4)
    ace_step_patch_lm_head_narrow_forward(lm)
    out = lm.forward("hidden")
    # Guard must defer to the stock forward (correct sharded matmul + all-reduce).
    assert out == ("stock", "hidden")
    assert lm._stock_called_with == "hidden"
    assert lm._ace_narrow_forward_used_band is False


def test_no_guard_single_device_attempts_band_path():
    # With num_devices == 1 and a set band, the guard does NOT fire. We stop the wrapper before
    # any device op by making split-size access fail loudly, proving it moved past the guard.
    lm = _FakeLMHead(num_devices=1)

    class _Boom(Exception):
        pass

    class _PropRaises:
        def __get__(self, obj, objtype=None):
            raise _Boom()

    # prefetcher access happens only AFTER the guard and band checks -> proves guard didn't fire.
    type(lm).prefetcher = _PropRaises()
    ace_step_patch_lm_head_narrow_forward(lm)
    try:
        lm.forward("hidden")
    except _Boom:
        pass  # expected: reached the narrow path (past the TP guard)
    else:
        raise AssertionError("expected narrow path to be attempted for num_devices == 1")
    finally:
        del type(lm).prefetcher
    assert lm._stock_called_with is None
