# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
from agent import ttlang


def test_preinstalled_short_circuits(monkeypatch):
    monkeypatch.setattr(ttlang, "ttl_importable", lambda: True)
    calls = {"n": 0}

    def inst(v):
        calls["n"] += 1
        return True

    r = ttlang.ensure_ttl(installer=inst, version_lister=lambda: ["1.0.1"])
    assert r["available"] is True and r["action"] == "none" and calls["n"] == 0


def test_install_success_with_ttnn_intact(monkeypatch):
    state = {"ttl": False}
    monkeypatch.setattr(ttlang, "ttl_importable", lambda: state["ttl"])
    monkeypatch.setattr(ttlang, "ttnn_version", lambda: "0.65.1")

    def inst(v):
        state["ttl"] = True
        return True

    r = ttlang.ensure_ttl(installer=inst, version_lister=lambda: ["1.0.1"])
    assert r["available"] is True and r["version"] == "1.0.1" and r["action"] == "installed"


def test_install_that_clobbers_ttnn_is_rolled_back(monkeypatch):
    state = {"ttl": False, "ttnn": "0.65.1"}
    monkeypatch.setattr(ttlang, "ttl_importable", lambda: state["ttl"])
    monkeypatch.setattr(ttlang, "ttnn_version", lambda: state["ttnn"])
    uninstalls = {"n": 0}

    def inst(v):
        state["ttl"] = True
        state["ttnn"] = "9.9.9"
        return True

    def uninst():
        uninstalls["n"] += 1
        state["ttl"] = False
        return True

    r = ttlang.ensure_ttl(installer=inst, uninstaller=uninst, version_lister=lambda: ["2.0.0"])
    assert r["available"] is False and uninstalls["n"] >= 1


def test_all_candidates_fail_to_install(monkeypatch):
    monkeypatch.setattr(ttlang, "ttl_importable", lambda: False)
    monkeypatch.setattr(ttlang, "ttnn_version", lambda: "0.65.1")
    r = ttlang.ensure_ttl(installer=lambda v: False, version_lister=lambda: ["1.0.1", "0.9.0"])
    assert r["available"] is False and "1.0.1" in r["tried"]


def test_pinned_version_is_tried_first(monkeypatch):
    monkeypatch.setattr(ttlang, "ttl_importable", lambda: False)
    monkeypatch.setattr(ttlang, "ttnn_version", lambda: "0.65.1")
    order = []

    def inst(v):
        order.append(v)
        return False

    ttlang.ensure_ttl(installer=inst, version_lister=lambda: ["2.0.0", "1.5.0"])
    assert order[0] == "1.0.1"
