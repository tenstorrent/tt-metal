# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pytest configuration for the Qwen3.5/Qwen3.6 demo test suite.

Adds two CLI options:

* ``--max-prefill`` — cap on the prefill length that routine runs exercise. Any
  parametrized case whose ``seq_len`` / ``actual_len`` / ``T`` exceeds the cap is
  auto-skipped, so the long-context tests (T=4096, 73728, …) don't run unless the
  cap is raised.
* ``--test-modules`` — comma-separated module selector (kept for future use).

These layer on top of the repo-root ``conftest.py`` (which provides ``device``,
``mesh_device``, ``device_params``, ``reset_seeds``, ``ensure_gc``).
"""

import importlib
import sys
import types

import pytest

_ADAPTER = "models.demos.blackhole.qwen36.tt.qwen36_vllm"


def pytest_addoption(parser):
    parser.addoption(
        "--max-prefill",
        action="store",
        type=int,
        default=8192,
        help="Max prefill length to run; longer parametrized cases auto-skip (raise to run long-context tests).",
    )
    parser.addoption(
        "--test-modules",
        action="store",
        default="all",
        help="Comma-separated modules to run (attention,gdn,mlp,rms_norm,rope,layer,model). Default: all.",
    )


@pytest.fixture
def test_modules(request):
    return request.config.getoption("--test-modules")


@pytest.fixture(autouse=True)
def _enforce_max_prefill(request):
    """Skip parametrized cases whose sequence length exceeds --max-prefill.

    Decode (length 1) always runs. qwen tests express the prefill length under a
    few different param names, so all three are checked.
    """
    callspec = getattr(request.node, "callspec", None)
    if callspec is None:
        return
    cap = request.config.getoption("--max-prefill")
    for key in ("seq_len", "actual_len", "T"):
        val = callspec.params.get(key)
        if isinstance(val, int) and val > cap:
            pytest.skip(f"{key}={val} > --max-prefill={cap}")


@pytest.fixture
def qwen36_vllm(monkeypatch):
    """The vLLM adapter module; fake vllm modules stand in when vLLM isn't installed (CI)."""
    try:
        importlib.import_module("vllm")
        return importlib.import_module(_ADAPTER)
    except ImportError:
        pass
    stubs = {name: types.ModuleType(name) for name in ("vllm", "vllm.model_executor", "vllm.model_executor.models")}
    stubs["vllm.model_executor.models.interfaces"] = types.SimpleNamespace(
        SupportsMultiModal=type("SupportsMultiModal", (), {})
    )
    stubs["vllm.model_executor.models.qwen3_5"] = types.SimpleNamespace(
        **{
            n: type(n, (), {})
            for n in ("Qwen3_5ProcessingInfo", "Qwen3VLDummyInputsBuilder", "Qwen3VLMultiModalProcessor")
        }
    )
    stubs["vllm.multimodal"] = types.SimpleNamespace(
        MULTIMODAL_REGISTRY=types.SimpleNamespace(register_processor=lambda *a, **k: lambda cls: cls)
    )
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)
    parent = importlib.import_module(_ADAPTER.rpartition(".")[0])
    monkeypatch.setattr(parent, "qwen36_vllm", None, raising=False)
    monkeypatch.setitem(sys.modules, _ADAPTER, None)
    del sys.modules[_ADAPTER]
    return importlib.import_module(_ADAPTER)
