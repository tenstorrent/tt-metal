# SPDX-FileCopyrightText: Copyright 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``release_persistent_capture`` chain.

The plugin calls ``release_persistent_capture`` on the model before it closes
the mesh. Each override releases what it owns and then reaches the base
``Generator`` method, which releases the base traces once; the destructor runs
only that base method. The host tests use the stubbed base of the ``adapter``
fixture; the base ``Generator`` and DiffusionGemma chains are covered by
``test_dflash_release_runtime.py`` on a device host.
"""


import pytest

from models.demos.gemma4.tests.unit.dflash_contract_harness import _start_solo, make_expect_error


@pytest.fixture
def expect_error():
    return make_expect_error()


def test_contract_rail_releases_its_session_then_the_width_set_then_the_base(model):
    _start_solo(model)
    model.release_persistent_capture()
    assert model._dflash_retained is None and model._dflash_live_owner is None
    assert model._spec_pending is None
    assert ("release_decoder", True) in model.events
    assert model.base_releases == [1]


def _block_model(adapter):
    """A block-rail model with a pending session, a carry and a recording decoder release."""
    cls = adapter.Gemma4DFlashForCausalLM
    model = cls.__new__(cls)
    model.events = []
    model._spec_pending = ([], 5)
    model._spec_pending_owner = None
    model._spec_carry = [1, 2]
    model._spec_release_decoder = lambda drop_page_tables=True, teardown=False: model.events.append(
        ("release_decoder", bool(teardown))
    )
    return model


def test_block_rail_teardown_releases_the_width_set_and_reaches_the_base(adapter):
    model = _block_model(adapter)
    model.release_persistent_capture()
    assert model._spec_pending is None and model._spec_carry == []
    assert model.events[-1] == ("release_decoder", True)
    assert model.base_releases == [1]


def test_block_rail_reaches_the_base_when_its_own_release_fails(adapter, expect_error):
    model = _block_model(adapter)

    def failing(*args, **kwargs):
        raise RuntimeError("trace release failed")

    model._spec_release_decoder = failing
    with expect_error(RuntimeError, "trace release failed"):
        model.release_persistent_capture()
    assert model.base_releases == [1]


def test_mtp_rail_reaches_the_base(adapter):
    cls = adapter.Gemma4MTPForCausalLM
    model = cls.__new__(cls)
    model._spec_owner_slot = 3
    calls = []
    model._spec_release_session = lambda force=False: calls.append(force)
    model.release_persistent_capture()
    assert calls == [False]
    assert model._spec_owner_slot is None
    assert model.base_releases == [1]


def test_mtp_rail_reaches_the_base_when_its_own_release_fails(adapter, expect_error):
    cls = adapter.Gemma4MTPForCausalLM
    model = cls.__new__(cls)
    model._spec_owner_slot = None

    def failing(force=False):
        raise RuntimeError("session release failed")

    model._spec_release_session = failing
    with expect_error(RuntimeError, "session release failed"):
        model.release_persistent_capture()
    assert model.base_releases == [1]
