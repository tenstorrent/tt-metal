# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from models.demos.gemma4.tt import pli_env


@pytest.mark.parametrize("value,expected", [(None, True), ("device", True), ("host", False)])
def test_pli_default_and_canonical(monkeypatch, value, expected):
    monkeypatch.delenv("GEMMA4_DECODE_PLI_DEV", raising=False)
    monkeypatch.delenv("GEMMA4_PLI", raising=False)
    if value is not None:
        monkeypatch.setenv("GEMMA4_PLI", value)
    assert pli_env.pli_on_device("GEMMA4_DECODE_PLI_DEV") is expected


def test_legacy_site_override_and_warning(monkeypatch):
    monkeypatch.setenv("GEMMA4_PLI", "device")
    monkeypatch.setenv("GEMMA4_DECODE_PLI_DEV", "0")
    pli_env._warned.clear()
    assert not pli_env.pli_on_device("GEMMA4_DECODE_PLI_DEV")
    assert "GEMMA4_DECODE_PLI_DEV" in pli_env._warned


@pytest.mark.parametrize(
    "has_pli,device,trace,greedy,expected",
    [
        (True, True, True, True, "fused-packed"),
        (True, False, True, True, "host-loop"),
        (False, True, True, True, "fused-batch-dim"),
        (True, True, False, True, "host-loop"),
        (True, True, True, False, "host-loop"),
    ],
)
def test_auto_route(has_pli, device, trace, greedy, expected):
    assert pli_env.resolve_route("auto", has_pli, device, trace, greedy) == expected


@pytest.mark.parametrize(
    "route,has_pli,device,trace,greedy,message",
    [
        ("fused-batch-dim", True, True, True, True, "cannot use"),
        ("fused-packed", True, False, True, True, "requires device PLI"),
        ("fused-packed", True, True, False, True, "requires GEMMA4_SPEC_TRACE"),
        ("fused-packed", True, True, True, False, "requires greedy"),
    ],
)
def test_unsupported_explicit_route(route, has_pli, device, trace, greedy, message, expect_error):
    with expect_error(ValueError, message):
        pli_env.resolve_route(route, has_pli, device, trace, greedy)


def test_legacy_route_alias_conflict_and_warning(monkeypatch, expect_error):
    monkeypatch.setenv("GEMMA4_SPEC_ROUTE", "fused-batched")
    monkeypatch.setenv("GEMMA4_SPEC_FUSED_PLI_DEV", "1")
    pli_env._warned.clear()
    assert pli_env.spec_route() == "fused-packed"
    assert "GEMMA4_SPEC_FUSED_PLI_DEV" in pli_env._warned
    monkeypatch.setenv("GEMMA4_SPEC_FUSED_PLI_DEV", "0")
    with expect_error(ValueError, "contradicts"):
        pli_env.spec_route()
