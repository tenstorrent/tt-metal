# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import os
import csv
import json
import socket
import subprocess
from pathlib import Path
from datetime import datetime
import contextvars

import pytest
from loguru import logger



# Per-session and per-test state
_current_nodeid: contextvars.ContextVar[str | None] = contextvars.ContextVar("current_nodeid", default=None)
_TEST_INFO: dict[str, dict] = {}
_SESSION_ROWS: list[dict] = []
_SESSION_META: dict[str, str] = {}
_ENABLED: bool = False
_OUT_PATH: str | None = None


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--results-csv",
        action="store",
        default=None,
        help=(
            "Path to results CSV (opt-in). If not set, reads TT_TEST_RESULTS_CSV. "
            "If neither is set, results collection is disabled."
        ),
    )


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _get_ttnn_version() -> str:
    try:
        import importlib

        ttnn = importlib.import_module("ttnn")
        return getattr(ttnn, "__version__", "unknown") or "unknown"
    except Exception:
        return "unknown"


def pytest_sessionstart(session: pytest.Session) -> None:
    global _ENABLED, _OUT_PATH
    # Determine if enabled
    cli_path = session.config.getoption("results_csv")
    env_path = os.getenv("TT_TEST_RESULTS_CSV")
    _OUT_PATH = cli_path or env_path
    if not _OUT_PATH:
        _ENABLED = False
        return

    _ENABLED = True

    # Session metadata
    run_id = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    _SESSION_META.update(
        {
            "run_id": run_id,
            "session_timestamp": run_id,
            "hostname": socket.gethostname(),
            "git_commit": _get_git_commit(),
            "ttnn_version": _get_ttnn_version(),
        }
    )

    # Monkeypatch comp_pcc / comp_allclose / comp_ulp to record metrics
    try:
        import models.common.utility_functions as mcu
        import models.utility_functions as mu

        def _wrap_pcc(orig_func):
            def _wrapped(*args, **kwargs):
                passed, value = orig_func(*args, **kwargs)
                nodeid = _current_nodeid.get()
                if nodeid is not None:
                    _TEST_INFO.setdefault(nodeid, {}).setdefault("metrics", []).append(
                        {"type": "pcc", "value": float(value), "passed": bool(passed)}
                    )
                return passed, value

            return _wrapped

        def _wrap_allclose(orig_func):
            def _wrapped(*args, **kwargs):
                passed, message = orig_func(*args, **kwargs)
                nodeid = _current_nodeid.get()
                if nodeid is not None:
                    _TEST_INFO.setdefault(nodeid, {}).setdefault("metrics", []).append(
                        {"type": "allclose", "message": str(message), "passed": bool(passed)}
                    )
                return passed, message

            return _wrapped

        def _wrap_ulp(orig_func):
            def _wrapped(*args, **kwargs):
                passed, message = orig_func(*args, **kwargs)
                nodeid = _current_nodeid.get()
                if nodeid is not None:
                    _TEST_INFO.setdefault(nodeid, {}).setdefault("metrics", []).append(
                        {"type": "ulp", "message": str(message), "passed": bool(passed)}
                    )
                return passed, message

            return _wrapped

        if hasattr(mcu, "comp_pcc"):
            mcu.comp_pcc = _wrap_pcc(mcu.comp_pcc)  # type: ignore
        if hasattr(mcu, "comp_allclose"):
            mcu.comp_allclose = _wrap_allclose(mcu.comp_allclose)  # type: ignore
        if hasattr(mcu, "comp_ulp"):
            mcu.comp_ulp = _wrap_ulp(mcu.comp_ulp)  # type: ignore

        try:
            mu.comp_pcc = mcu.comp_pcc  # type: ignore
            if hasattr(mu, "comp_allclose"):
                mu.comp_allclose = mcu.comp_allclose  # type: ignore
            if hasattr(mu, "comp_ulp"):
                mu.comp_ulp = mcu.comp_ulp  # type: ignore
        except Exception:
            pass
    except Exception as e:
        logger.warning(f"Results CSV plugin: failed to patch metric functions: {e}")


@pytest.fixture(autouse=True)
def _results_csv_test_context(request: pytest.FixtureRequest):
    if not _ENABLED:
        yield
        return
    nodeid = request.node.nodeid
    token = _current_nodeid.set(nodeid)

    # Initialize per-test info
    info = _TEST_INFO.setdefault(nodeid, {})
    info["start_ts"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    info["metrics"] = []

    # Collect easily-available metadata from fixtures/params
    def _get(name):
        return request.node.funcargs.get(name) if hasattr(request.node, "funcargs") else None

    info["mode"] = _get("mode")
    info["seq_len"] = _get("seq_len")
    info["batch_size"] = _get("batch_size")
    info["weights_type"] = _get("weights_type")
    info["module_path"] = _get("module_path")

    hf_cfg = _get("hf_config") or _get("hf_config_short")
    if hf_cfg is not None and hasattr(hf_cfg, "max_seq_len"):
        try:
            info["hf_max_seq_len"] = int(hf_cfg.max_seq_len)
        except Exception:
            info["hf_max_seq_len"] = str(getattr(hf_cfg, "max_seq_len", None))
    else:
        info["hf_max_seq_len"] = None

    mesh = _get("mesh_device")
    if mesh is not None:
        try:
            shape = tuple(mesh.shape)
            info["mesh_shape"] = f"{shape}"
            info["mesh_num_devices"] = int(mesh.get_num_devices())
        except Exception:
            info["mesh_shape"] = None
            info["mesh_num_devices"] = None

    info["module"] = _infer_module_name(request)

    start = datetime.utcnow()
    try:
        yield
    finally:
        end = datetime.utcnow()
        info["end_ts"] = end.isoformat(timespec="seconds") + "Z"
        info["duration_sec"] = (end - start).total_seconds()
        _current_nodeid.reset(token)


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


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):
    outcome = yield
    rep: pytest.TestReport = outcome.get_result()
    if not _ENABLED:
        return

    info = _TEST_INFO.setdefault(item.nodeid, {})
    reports = info.setdefault("reports", {})
    reports[rep.when] = rep

    if rep.when == "teardown":
        outcome_str, error_text = _derive_outcome_and_error(reports)
        info["outcome"] = outcome_str
        info["error"] = error_text

        row = {
            "test_nodeid": item.nodeid,
            "module": info.get("module"),
            "mode": info.get("mode"),
            "seq_len": info.get("seq_len"),
            "batch_size": info.get("batch_size"),
            "hf_max_seq_len": info.get("hf_max_seq_len"),
            "mesh_shape": info.get("mesh_shape"),
            "mesh_num_devices": info.get("mesh_num_devices"),
            "weights_type": info.get("weights_type"),
            "module_path": info.get("module_path"),
            "metrics_json": json.dumps(info.get("metrics", [])),
            "outcome": info.get("outcome"),
            "duration_sec": info.get("duration_sec"),
            "error": info.get("error"),
            "run_id": _SESSION_META.get("run_id"),
            "session_timestamp": _SESSION_META.get("session_timestamp"),
            "hostname": _SESSION_META.get("hostname"),
            "git_commit": _SESSION_META.get("git_commit"),
            "ttnn_version": _SESSION_META.get("ttnn_version"),
            "start_timestamp": info.get("start_ts"),
            "end_timestamp": info.get("end_ts"),
        }
        _SESSION_ROWS.append(row)


def _derive_outcome_and_error(reports: dict) -> tuple[str, str | None]:
    for phase in ("call", "setup", "teardown"):
        rep = reports.get(phase)
        if rep is not None and rep.failed:
            return "failed", _safe_longrepr(rep)

    for phase in ("call", "setup", "teardown"):
        rep = reports.get(phase)
        if rep is not None and rep.skipped:
            return "skipped", _safe_longrepr(rep)

    return "passed", None


def _safe_longrepr(rep: pytest.TestReport) -> str:
    try:
        return rep.longreprtext  # type: ignore[attr-defined]
    except Exception:
        try:
            return str(rep.longrepr)
        except Exception:
            return "unknown error"


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if not _ENABLED or not _SESSION_ROWS:
        return
    out_path = _OUT_PATH
    assert out_path is not None

    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    fieldnames = [
        "test_nodeid",
        "module",
        "mode",
        "seq_len",
        "batch_size",
        "hf_max_seq_len",
        "mesh_shape",
        "mesh_num_devices",
        "weights_type",
        "module_path",
        "metrics_json",
        "outcome",
        "duration_sec",
        "error",
        "run_id",
        "session_timestamp",
        "hostname",
        "git_commit",
        "ttnn_version",
        "start_timestamp",
        "end_timestamp",
    ]

    write_header = True
    try:
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            write_header = False
    except Exception:
        pass

    try:
        with open(out_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for row in _SESSION_ROWS:
                writer.writerow({k: row.get(k) for k in fieldnames})
        logger.info(f"Module test results appended to CSV: {out_path}")
    except Exception as e:
        logger.error(f"Failed to write module test results CSV to {out_path}: {e}")
