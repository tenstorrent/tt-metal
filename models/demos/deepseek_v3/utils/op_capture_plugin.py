# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import json
import time
import types
import inspect
import importlib
import importlib.abc
import importlib.machinery
from datetime import datetime
from typing import Any, Callable

import pytest
from loguru import logger

from .serialize_configs import to_jsonable as _to_jsonable


# Opt-in gating
_ENABLED: bool = False
_OUT_PATH: str | None = None

# Per-session and per-test state
_RUN_ID: str | None = None
_current_nodeid: str | None = None
_current_module_name: str | None = None
_per_test_index: dict[str, int] = {}
_per_test_buffer: dict[str, list[dict[str, Any]]] = {}


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--ops-jsonl",
        action="store",
        default=None,
        help=(
            "Path to JSONL file for captured TTNN op calls (opt-in). If not set, reads TT_OP_RESULTS_JSONL. "
            "If neither is set, op capture is disabled."
        ),
    )


def pytest_configure(config: pytest.Config) -> None:
    global _ENABLED, _OUT_PATH, _RUN_ID
    cli_path = config.getoption("ops_jsonl")
    env_path = os.getenv("TT_OP_RESULTS_JSONL")
    _OUT_PATH = cli_path or env_path
    if not _OUT_PATH:
        _ENABLED = False
        return

    _ENABLED = True
    _RUN_ID = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    try:
        _install_import_hook()
        _wrap_already_imported_ttnn_modules()
        logger.info(f"TTNN op capture enabled: {_OUT_PATH}")
    except Exception as e:
        logger.warning(f"TTNN op capture failed to initialize: {e}")
        _ENABLED = False


@pytest.fixture(autouse=True)
def _op_capture_test_context(request: pytest.FixtureRequest):
    if not _ENABLED:
        yield
        return

    global _current_nodeid, _current_module_name
    _current_nodeid = request.node.nodeid
    _current_module_name = _infer_module_name(request)
    _per_test_index[_current_nodeid] = 0
    _per_test_buffer.setdefault(_current_nodeid, [])
    try:
        yield
    finally:
        try:
            if _per_test_buffer.get(_current_nodeid):
                _write_jsonl(_per_test_buffer[_current_nodeid])
        finally:
            _per_test_buffer.pop(_current_nodeid, None)
            _per_test_index.pop(_current_nodeid, None)
            _current_nodeid = None
            _current_module_name = None


def _infer_module_name(request: pytest.FixtureRequest) -> str | None:
    funcargs = getattr(request.node, "funcargs", {}) or {}
    for key in ("MLPClass", "RMSNormClass", "DecoderBlockClass"):
        cls = funcargs.get(key)
        if cls is not None:
            try:
                return cls.__name__
            except Exception:
                pass
    nodeid = request.node.nodeid
    filename = nodeid.split("::", 1)[0]
    base = os.path.basename(filename)
    mapping = {
        "test_moe_gate.py": "MoEGate",
        "test_mla_1d.py": "MLA1D",
        "test_moe.py": "MoE",
        "test_model.py": "Model1D",
        "test_embedding_1d.py": "Embedding1D",
        "test_lm_head.py": "LMHead",
        "test_decoder_block.py": "DecoderBlock",
        "test_rms_norm.py": "RMSNorm",
    }
    return mapping.get(base)


# ---------------------------
# Import hook and wrappers
# ---------------------------

def _install_import_hook() -> None:
    class TTNNLoader(importlib.abc.Loader):
        def __init__(self, original_loader: importlib.abc.Loader):
            self.original_loader = original_loader

        def create_module(self, spec):  # type: ignore[override]
            if hasattr(self.original_loader, "create_module"):
                return self.original_loader.create_module(spec)  # type: ignore[attr-defined]
            return None

        def exec_module(self, module):  # type: ignore[override]
            self.original_loader.exec_module(module)  # type: ignore[attr-defined]
            try:
                _wrap_module_callables(module)
            except Exception as e:
                logger.debug(f"op_capture: wrap failed for module {module.__name__}: {e}")

    class TTNNFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):  # type: ignore[override]
            if fullname == "ttnn" or fullname.startswith("ttnn."):
                spec = importlib.machinery.PathFinder.find_spec(fullname, path)
                if spec and spec.loader:
                    spec.loader = TTNNLoader(spec.loader)
                return spec
            return None

    sys.meta_path.insert(0, TTNNFinder())


def _wrap_already_imported_tnn(module: types.ModuleType) -> None:
    try:
        _wrap_module_callables(module)
    except Exception as e:
        logger.debug(f"op_capture: wrap failed for already-imported module {module.__name__}: {e}")


def _wrap_already_imported_ttnn_modules() -> None:
    for name, mod in list(sys.modules.items()):
        if name == "ttnn" or (name and name.startswith("ttnn.")):
            if isinstance(mod, types.ModuleType):
                _wrap_already_imported_tnn(mod)


def _wrap_module_callables(module: types.ModuleType) -> None:
    for attr in dir(module):
        if attr.startswith("__"):
            continue
        try:
            obj = getattr(module, attr)
        except Exception:
            continue
        if inspect.isfunction(obj) or inspect.isbuiltin(obj):
            try:
                wrapped = _wrap_callable(module.__name__, attr, obj)
                setattr(module, attr, wrapped)
            except Exception:
                pass
        elif isinstance(obj, types.ModuleType) and obj.__name__.startswith("ttnn"):
            _wrap_module_callables(obj)


def _wrap_callable(modname: str, name: str, fn: Callable) -> Callable:
    def _ttnn_traced(*args, **kwargs):
        if not (_ENABLED and _OUT_PATH):
            return fn(*args, **kwargs)
        start = time.perf_counter()
        exc: BaseException | None = None
        result = None
        try:
            result = fn(*args, **kwargs)
            return result
        except BaseException as e:
            exc = e
            raise
        finally:
            try:
                duration_ms = (time.perf_counter() - start) * 1000.0
                record = _build_record(modname, name, args, kwargs, result, duration_ms, exc)
                _buffer_record(record)
            except Exception as log_e:
                logger.debug(f"op_capture: failed to record {modname}.{name}: {log_e}")

    try:
        _ttnn_traced.__name__ = getattr(fn, "__name__", name)
        _ttnn_traced.__doc__ = getattr(fn, "__doc__", None)
        _ttnn_traced.__module__ = getattr(fn, "__module__", modname)
    except Exception:
        pass
    return _ttnn_traced


# ---------------------------
# Serialization helpers (shared serializer)
# ---------------------------

def _buffer_record(rec: dict[str, Any]) -> None:
    nodeid = _current_nodeid or "<unknown>"
    _per_test_buffer.setdefault(nodeid, []).append(rec)


def _write_jsonl(records: list[dict[str, Any]]) -> None:
    assert _OUT_PATH is not None
    out_dir = os.path.dirname(_OUT_PATH)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    with open(_OUT_PATH, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, default=str))
            f.write("\n")


def _build_record(
    modname: str,
    name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    result: Any,
    duration_ms: float,
    exc: BaseException | None,
) -> dict[str, Any]:
    nodeid = _current_nodeid or "<unknown>"
    idx = _per_test_index.get(nodeid, 0)
    _per_test_index[nodeid] = idx + 1

    rec: dict[str, Any] = {
        "run_id": _RUN_ID,
        "test_nodeid": nodeid,
        "module": _current_module_name,
        "op_index": idx,
        "op_name": f"{modname}.{name}",
        "start_ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "duration_ms": duration_ms,
    }
    rec["inputs"] = _to_jsonable(list(args))
    rec["kwargs"] = _to_jsonable(dict(kwargs))
    rec["output"] = _to_jsonable(result)
    if exc is not None:
        rec["error"] = {"type": exc.__class__.__name__, "message": str(exc)}
    return rec

