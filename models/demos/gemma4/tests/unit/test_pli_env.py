# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from models.demos.gemma4.tt import pli_env


@pytest.mark.parametrize(
    "value,default,expected",
    [
        (None, False, False),
        (None, True, True),
        ("", False, False),
        ("", True, True),
        ("device", False, True),
        ("host", True, False),
        (" Device ", False, True),
    ],
)
def test_pli_mechanism_and_site_default(monkeypatch, value, default, expected):
    monkeypatch.delenv("GEMMA4_PLI", raising=False)
    if value is not None:
        monkeypatch.setenv("GEMMA4_PLI", value)
    assert pli_env.pli_on_device(default) is expected


def test_unknown_pli_mechanism_raises(monkeypatch, expect_error):
    monkeypatch.setenv("GEMMA4_PLI", "cpu")
    with expect_error(ValueError, "GEMMA4_PLI"):
        pli_env.pli_on_device(True)
