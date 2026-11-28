# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A model whose architecture the installed transformers does not register must step transformers
forward to the lowest release that does (and installs on this Python) BEFORE the reference-loader
fallback, which built a reference that broke causality for such a model. Offline: PyPI and the
wheel probe are injected."""
import platform

import pytest

from scripts.tt_hw_planner import cli
import scripts.tt_hw_planner.module_tree as module_tree

PY = platform.python_version()


def _wheel(url, requires_python=None, yanked=False):
    return {"packagetype": "bdist_wheel", "url": url, "requires_python": requires_python, "yanked": yanked}


@pytest.fixture
def installed(monkeypatch):
    def _set(version):
        orig = module_tree._installed_version
        monkeypatch.setattr(module_tree, "_installed_version", lambda p: version if p == "transformers" else orig(p))

    return _set


def test_picks_lowest_newer_release_that_registers(installed):
    installed("1.0.0")
    releases = {v: [_wheel(v)] for v in ["0.9.0", "1.0.0", "1.1.0", "1.2.0", "1.3.0", "1.4.0"]}
    registers = {"1.2.0", "1.3.0", "1.4.0"}
    got = cli._transformers_release_registering("arch", lister=lambda: releases, prober=lambda u, m: u in registers)
    assert got == "1.2.0"


def test_never_steps_backwards_or_onto_prereleases_or_yanked(installed):
    installed("1.1.0")
    releases = {
        "1.0.0": [_wheel("1.0.0")],
        "1.2.0rc1": [_wheel("1.2.0rc1")],
        "1.2.0": [_wheel("1.2.0", yanked=True)],
        "1.3.0": [_wheel("1.3.0")],
    }
    seen = []
    got = cli._transformers_release_registering(
        "arch", lister=lambda: releases, prober=lambda u, m: seen.append(u) or True
    )
    assert got == "1.3.0" and set(seen) == {"1.3.0"}


def test_skips_releases_this_python_cannot_install(installed):
    installed("1.0.0")
    major, minor = map(int, PY.split(".")[:2])
    too_new = f">={major}.{minor + 1}"
    releases = {"1.1.0": [_wheel("1.1.0")], "1.2.0": [_wheel("1.2.0", requires_python=too_new)]}
    got = cli._transformers_release_registering(
        "arch", lister=lambda: releases, prober=lambda u, m: u == "1.2.0" or u == "1.1.0"
    )
    assert got == "1.1.0"


def test_none_registers_or_offline_returns_empty(installed):
    installed("1.0.0")
    assert (
        cli._transformers_release_registering(
            "arch", lister=lambda: {"1.1.0": [_wheel("1.1.0")]}, prober=lambda u, m: False
        )
        == ""
    )
    assert cli._transformers_release_registering("arch", lister=lambda: {}, prober=lambda u, m: True) == ""


def _drive(monkeypatch, *, target, allow_fix=True, attempted=False):
    calls = {"install": [], "resolver": 0}
    monkeypatch.setattr(cli, "_can_load_with_transformers", lambda m: (False, "unknown model_type"))
    monkeypatch.setattr(cli, "_classify_load_error", lambda msg: "version")
    monkeypatch.setattr(cli, "_unregistered_model_type", lambda m, e: "arch")
    monkeypatch.setattr(cli, "_transformers_release_registering", lambda mt: target)
    monkeypatch.setattr(
        cli, "_install_transformers_and_reexec", lambda spec=None: calls["install"].append(spec) or False
    )
    monkeypatch.setattr(
        cli, "_auto_enable_loader_resolver", lambda: calls.__setitem__("resolver", calls["resolver"] + 1)
    )
    monkeypatch.setattr(cli, "_loader_resolver_available", lambda m: False)
    if attempted:
        monkeypatch.setenv(cli._ENV_FIX_ATTEMPTED_FLAG, "1")
    else:
        monkeypatch.delenv(cli._ENV_FIX_ATTEMPTED_FLAG, raising=False)
    cli._preflight_load_with_autofix("org/model", allow_fix=allow_fix)
    return calls


def test_unregistered_architecture_installs_the_release_before_the_loader_fallback(monkeypatch):
    calls = _drive(monkeypatch, target="9.9.9")
    assert calls["install"] == ["transformers==9.9.9"]
    # The install is stubbed to fail, so the previous fallback still runs afterwards.
    assert calls["resolver"] == 1


def test_no_release_found_keeps_the_previous_fallback(monkeypatch):
    calls = _drive(monkeypatch, target="")
    assert calls["install"] == [] and calls["resolver"] == 1


@pytest.mark.parametrize("allow_fix,attempted", [(False, False), (True, True)])
def test_no_env_fix_and_already_attempted_never_install(monkeypatch, allow_fix, attempted):
    calls = _drive(monkeypatch, target="9.9.9", allow_fix=allow_fix, attempted=attempted)
    assert calls["install"] == []


def test_upgrade_uses_the_envs_own_installer(monkeypatch):
    import subprocess

    from models.experimental.perf_automation.agent import pkgtools

    seen = {}
    monkeypatch.setattr(pkgtools, "pip_cmd", lambda args: ["INSTALLER", *args])

    class _P:
        returncode = 0
        stdout = stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: seen.setdefault("cmd", cmd) and _P())
    ok, _ = cli._upgrade_transformers_from_upstream("transformers==9.9.9")
    assert ok and seen["cmd"] == ["INSTALLER", "install", "-U", "transformers==9.9.9"]
