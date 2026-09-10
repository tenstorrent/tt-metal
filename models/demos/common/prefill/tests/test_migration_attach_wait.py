# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger

from models.demos.common.prefill.runners.migration import _DEFAULT_MIGRATION_ATTACH_WAIT_S, _migration_attach_wait_s

ENV_VAR = "PREFILL_MIGRATION_ATTACH_WAIT_S"


@pytest.fixture
def warnings():
    """loguru does not feed pytest's caplog, so collect WARNING records through our own sink."""
    captured = []
    sink_id = logger.add(captured.append, level="WARNING", format="{message}")
    yield captured
    logger.remove(sink_id)


def test_unset_env_waits_forever(monkeypatch, warnings):
    monkeypatch.delenv(ENV_VAR, raising=False)

    # 0.0 is the "no deadline" sentinel _attach_with_retry tests with `budget > 0`.
    assert _migration_attach_wait_s() == _DEFAULT_MIGRATION_ATTACH_WAIT_S == 0.0
    assert not warnings


@pytest.mark.parametrize("raw", ["", "   ", "\t"], ids=["empty", "spaces", "tab"])
def test_blank_env_waits_forever_without_complaining(monkeypatch, warnings, raw):
    monkeypatch.setenv(ENV_VAR, raw)

    assert _migration_attach_wait_s() == _DEFAULT_MIGRATION_ATTACH_WAIT_S
    assert not warnings


@pytest.mark.parametrize(
    "raw, expected",
    [("120", 120.0), ("0.5", 0.5), ("1e2", 100.0), ("  90  ", 90.0), ("0", 0.0)],
    ids=["int", "fraction", "exponent", "padded", "explicit-zero"],
)
def test_valid_budget_is_used_verbatim(monkeypatch, warnings, raw, expected):
    monkeypatch.setenv(ENV_VAR, raw)

    assert _migration_attach_wait_s() == expected
    assert not warnings


@pytest.mark.parametrize(
    "raw",
    ["abc", "30s", "two minutes", "--30", "nan", "-30", "-0.5", "-inf"],
    ids=["word", "unit-suffix", "phrase", "double-sign", "nan", "negative", "negative-fraction", "negative-inf"],
)
def test_unusable_value_falls_back_to_forever_and_warns(monkeypatch, warnings, raw):
    """Negatives and NaN matter as much as junk strings: both slip past `budget > 0` into an
    unbounded wait, so a typo'd sign must not quietly drop the only bound the operator set."""
    monkeypatch.setenv(ENV_VAR, raw)

    assert _migration_attach_wait_s() == _DEFAULT_MIGRATION_ATTACH_WAIT_S
    assert len(warnings) == 1
    assert ENV_VAR in warnings[0] and raw in warnings[0]
