# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Explicit, test-only routing of an unchanged golden suite to a native entry point.

Load with -p tools.generic_op_to_factory.native_adapter --migration-route FILE.
No production run identity or operation name is embedded in this adapter.
"""

import importlib
import json
from pathlib import Path

import pytest


def resolve(spec):
    module_name, separator, name = spec.partition(":")
    if not separator or not all(part.isidentifier() for part in module_name.split(".")) or not name.isidentifier():
        raise ValueError("An entry point must be module.path:symbol")
    module = importlib.import_module(module_name)
    value = getattr(module, name)
    if not callable(value):
        raise ValueError("Entry point is not callable")
    return module, name, value


class Route:
    def __init__(self, specification):
        if (
            {"source", "native", "mode"} - set(specification)
            or set(specification) - {"source", "native", "mode", "aliases"}
            or specification["mode"] not in ("source", "native")
        ):
            raise ValueError("Route requires source, native and mode (source/native)")
        aliases = specification.get("aliases", [])
        if not isinstance(aliases, list) or any(not isinstance(alias, str) for alias in aliases):
            raise ValueError("Route aliases must be a list of entry-point strings")
        if len(set(aliases)) != len(aliases):
            raise ValueError("Route aliases must be unique")
        self.specification = {**specification, "aliases": list(aliases)}
        self.calls = 0
        self.restores = []

    def install(self):
        module, name, source = resolve(self.specification["source"])
        _, _, native = resolve(self.specification["native"])
        if native is source:
            raise ValueError("Native entry point must differ from the source")
        if not getattr(native, "is_cpp_operation", False):
            raise ValueError("Native entry point must be a registered C++ operation")
        selected = native if self.specification["mode"] == "native" else source

        def dispatch(*args, **kwargs):
            self.calls += 1
            return selected(*args, **kwargs)

        targets = [(module, name)]
        # Patch both a package re-export and its defining module before collection.
        defining = importlib.import_module(source.__module__)
        if defining is not module and getattr(defining, name, None) is source:
            targets.append((defining, name))
        # Resolve and identity-check every explicit alias before changing any symbol.
        # Never discover aliases by scanning modules or overwrite an unrelated API.
        for alias in self.specification["aliases"]:
            owner, symbol, value = resolve(alias)
            if value is not source:
                raise ValueError(f"Route alias does not identify the frozen source callable: {alias}")
            if (owner, symbol) not in targets:
                targets.append((owner, symbol))
        for owner, symbol in targets:
            self.restores.append((owner, symbol, getattr(owner, symbol)))
            setattr(owner, symbol, dispatch)
        if self.specification["mode"] == "native":
            ttnn = importlib.import_module("ttnn")

            def forbidden(*args, **kwargs):
                raise AssertionError("Native validation must not fall back to ttnn.generic_op")

            self.restores.append((ttnn, "generic_op", ttnn.generic_op))
            ttnn.generic_op = forbidden

    def restore(self):
        for owner, symbol, value in reversed(self.restores):
            setattr(owner, symbol, value)
        self.restores.clear()


def pytest_addoption(parser):
    parser.addoption("--migration-route", help="Explicit source/native golden-suite routing JSON")


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    path = config.getoption("--migration-route")
    if not path:
        raise pytest.UsageError("The migration adapter requires --migration-route")
    route = Route(json.loads(Path(path).read_text()))
    try:
        route.install()
    except Exception:
        route.restore()
        raise
    config._migration_route = route


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    route = config._migration_route
    terminalreporter.write_line(
        "MIGRATION_ROUTE=" + json.dumps({**route.specification, "calls": route.calls}, sort_keys=True)
    )


def pytest_sessionfinish(session, exitstatus):
    route = session.config._migration_route
    if not session.config.option.collectonly and route.calls == 0 and exitstatus == 0:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_unconfigure(config):
    route = getattr(config, "_migration_route", None)
    if route is not None:
        route.restore()
