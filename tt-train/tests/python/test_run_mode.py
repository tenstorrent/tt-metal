# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``run_mode`` context manager in ``ttml.common.utils``."""

from ttml.common.utils import run_mode
from ttml.modules import AbstractModuleBase, RunMode


class _Child(AbstractModuleBase):
    def forward(self, x):
        return x


class _Parent(AbstractModuleBase):
    def __init__(self):
        super().__init__()
        self.child = _Child()

    def forward(self, x):
        return self.child(x)


def _modes(parent):
    return parent.get_run_mode(), parent.child.get_run_mode()


def test_run_mode_switches_and_restores_submodules():
    parent = _Parent()
    assert _modes(parent) == (RunMode.TRAIN, RunMode.TRAIN)

    with run_mode(parent, RunMode.EVAL) as m:
        assert m is parent
        assert _modes(parent) == (RunMode.EVAL, RunMode.EVAL)

    assert _modes(parent) == (RunMode.TRAIN, RunMode.TRAIN)


def test_run_mode_restores_on_exception(expect_error):
    parent = _Parent()

    with expect_error(RuntimeError, "boom"):
        with run_mode(parent, RunMode.EVAL):
            assert _modes(parent) == (RunMode.EVAL, RunMode.EVAL)
            raise RuntimeError("boom")

    assert _modes(parent) == (RunMode.TRAIN, RunMode.TRAIN)


def test_run_mode_restores_previous_eval():
    parent = _Parent()
    parent.eval()

    with run_mode(parent, RunMode.TRAIN):
        assert _modes(parent) == (RunMode.TRAIN, RunMode.TRAIN)

    assert _modes(parent) == (RunMode.EVAL, RunMode.EVAL)

    with run_mode(parent, RunMode.EVAL):
        assert _modes(parent) == (RunMode.EVAL, RunMode.EVAL)

    assert _modes(parent) == (RunMode.EVAL, RunMode.EVAL)
