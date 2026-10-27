# SPDX-FileCopyrightText: Copyright 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""``Gemma4Model.dflash_capture_taps`` stores the ``keep_last`` cap.

The append site frees the oldest tap group once more than ``_dflash_tap_keep``
entries are held; the hook converts ``keep_last`` forwards to that entry count
(one entry per tap layer per forward). Needs the TT runtime to import the model
module, so it runs on a device host.
"""

import pytest


def _model():
    module = pytest.importorskip("models.demos.gemma4.tt.model", reason="needs the TT runtime")
    return module.Gemma4Model.__new__(module.Gemma4Model)


def test_keep_last_is_stored_in_tap_entries():
    model = _model()
    model.dflash_capture_taps([1, 2, 3], keep_last=4)
    assert model._dflash_tap_layers == {1, 2, 3}
    assert model._dflash_tap_keep == 12
    assert model._dflash_taps == [] and model._dflash_tap_idx == 0


def test_no_keep_last_means_no_cap():
    model = _model()
    model.dflash_capture_taps([1, 2], buffers=["a", "b"])
    assert model._dflash_tap_keep is None
    assert model._dflash_tap_buffers == ["a", "b"]


def test_disarming_clears_the_cap():
    model = _model()
    model.dflash_capture_taps([1], keep_last=2)
    model.dflash_capture_taps(None)
    assert model._dflash_tap_layers is None and model._dflash_tap_keep is None
