# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Host-only gate on the two VAE decode knobs the pipeline reads out of the environment.

`MINIMAX_H3_VAE_WAVES` sets how many decoder units go into one device program and
`MINIMAX_H3_VAE_PROFILE` turns on the VAE's own per-forward synchronize. Neither needs a device
to test, and both are the kind of knob where a typo that silently serves the default produces an
unexplained measurement -- so the parser raises instead of falling back."""

import pytest

from ....pipelines.minimax_h3.pipeline_minimax_h3 import _env_positive_int


def test_env_positive_int_defaults_when_unset(monkeypatch):
    monkeypatch.delenv("MINIMAX_H3_VAE_WAVES", raising=False)
    assert _env_positive_int("MINIMAX_H3_VAE_WAVES", 2) == 2


def test_env_positive_int_defaults_when_empty(monkeypatch):
    monkeypatch.setenv("MINIMAX_H3_VAE_WAVES", "")
    assert _env_positive_int("MINIMAX_H3_VAE_WAVES", 2) == 2


def test_env_positive_int_reads_the_override(monkeypatch):
    monkeypatch.setenv("MINIMAX_H3_VAE_WAVES", "4")
    assert _env_positive_int("MINIMAX_H3_VAE_WAVES", 2) == 4


@pytest.mark.parametrize("value", ["0", "-1", "two", "2.5", "  "])
def test_env_positive_int_rejects_rather_than_falling_back(monkeypatch, value, expect_error):
    """A bad value must not quietly serve the default: that is how a sweep measures the baseline
    four times and reports it as four different configurations."""
    monkeypatch.setenv("MINIMAX_H3_VAE_WAVES", value)
    with expect_error(ValueError, "MINIMAX_H3_VAE_WAVES"):
        _env_positive_int("MINIMAX_H3_VAE_WAVES", 2)
