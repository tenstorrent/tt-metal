# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import importlib
import json

import pytest

from tests.model_behavior.adapters import ADAPTERS
from tests.model_behavior.adapters.profiles import HARDWARE
from tests.model_behavior.driver import RequestDriver


def pytest_addoption(parser):
    group = parser.getgroup("model behavior")
    group.addoption("--model-behavior-backend", choices=sorted(ADAPTERS), default=None)
    group.addoption("--model-behavior-execution", choices=("eager", "traced", "both"), default="both")
    group.addoption("--model-behavior-sku", choices=sorted(HARDWARE), default=None)
    group.addoption(
        "--model-behavior-skip-model-load", action="store_true", help="Use an existing GPT-OSS TT weight cache"
    )


def pytest_generate_tests(metafunc):
    if "execution_mode" in metafunc.fixturenames:
        mode = metafunc.config.getoption("--model-behavior-execution")
        metafunc.parametrize("execution_mode", ("eager", "traced") if mode == "both" else (mode,), scope="session")


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(items):
    # Keep every eager case together before constructing the traced model.
    # Other per-test parameterizations can otherwise interleave session params.
    items.sort(
        key=lambda item: getattr(item, "callspec", None) is not None
        and item.callspec.params.get("execution_mode") == "traced"
    )


@pytest.fixture(scope="session")
def model_adapter(request, execution_mode):
    backend = request.config.getoption("--model-behavior-backend")
    if backend is None:
        pytest.skip("Select a model explicitly with --model-behavior-backend")
    module = importlib.import_module(ADAPTERS[backend])
    options = {}
    sku = request.config.getoption("--model-behavior-sku")
    skip_load = request.config.getoption("--model-behavior-skip-model-load")
    if backend != "galaxy-llama70b":
        options = dict(backend=backend, sku=sku, skip_model_load=skip_load)
    elif skip_load or sku not in (None, "wh_galaxy_perf"):
        raise pytest.UsageError("galaxy-llama70b requires wh_galaxy_perf and loads its own checkpoint")
    with module.open_adapter(execution_mode, **options) as adapter:
        yield adapter


@pytest.fixture
def request_driver(model_adapter, tmp_path, record_property):
    driver = RequestDriver(model_adapter)
    report_path = tmp_path / "requests.json"
    record_property("request_report", str(report_path))
    try:
        yield driver
    finally:
        report_path.write_text(json.dumps({"model": model_adapter.describe(), **driver.report()}, indent=2) + "\n")
