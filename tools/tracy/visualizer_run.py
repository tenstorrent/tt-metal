# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tracy-side helpers for pairing TTNN memory reports with Tracy perf reports.

Mint/stamp live in ``ttnn.visualizer_run_id``; this module owns manifest write
and memory-report lookup, loading those helpers without importing heavy ``ttnn``
when the source tree is available.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from loguru import logger

MANIFEST_FILENAME = "manifest.json"
DEFAULT_TTNN_REPORTS_ROOT = Path("generated/ttnn/reports")
_ALLOWED_MANIFEST_BASENAMES = frozenset({MANIFEST_FILENAME})


def _load_run_id_module():
    """Load mint/stamp helpers without importing the heavy ``ttnn`` package when possible."""
    for name in ("ttnn.visualizer_run_id", "_ttnn_visualizer_run_id"):
        if name in sys.modules:
            return sys.modules[name]

    # Prefer a source-tree load so ``python -m tracy`` CSV tooling does not init ttnn C++.
    module_path = Path(__file__).resolve().parents[2] / "ttnn" / "ttnn" / "visualizer_run_id.py"
    if module_path.is_file():
        spec = importlib.util.spec_from_file_location("_ttnn_visualizer_run_id", module_path)
        if spec is not None and spec.loader is not None:
            mod = importlib.util.module_from_spec(spec)
            sys.modules["_ttnn_visualizer_run_id"] = mod
            spec.loader.exec_module(mod)
            return mod

    from ttnn import visualizer_run_id as mod

    return mod


_run_id = _load_run_id_module()
TT_METAL_RUN_ID_ENV = _run_id.TT_METAL_RUN_ID_ENV
RUN_ID_METADATA_KEY = _run_id.RUN_ID_METADATA_KEY
peek_run_id = _run_id.peek_run_id
get_or_create_run_id = _run_id.get_or_create_run_id
inject_run_id_into_env = _run_id.inject_run_id_into_env
read_db_run_id = _run_id.read_db_run_id
stamp_memory_run_id = _run_id.stamp_memory_run_id
stamp_report_dir_run_id = _run_id.stamp_report_dir_run_id


def _path_is_under_root(candidate: str, root: str) -> bool:
    if candidate == root:
        return True
    prefix = root if root.endswith(os.sep) else root + os.sep
    return candidate.startswith(prefix)


def _manifest_write_allowlist_roots() -> tuple[str, ...]:
    """Safelist of directory prefixes where performance manifests may be written."""
    raw: list[str] = [
        os.getcwd(),
        "generated",
        tempfile.gettempdir(),
    ]
    for key in ("TT_METAL_HOME", "TT_METAL_PROFILER_DIR"):
        value = os.environ.get(key)
        if value:
            raw.append(value)

    roots: list[str] = []
    seen: set[str] = set()
    for entry in raw:
        resolved = os.path.realpath(os.path.abspath(entry))
        if resolved not in seen:
            seen.add(resolved)
            roots.append(resolved)
    return tuple(roots)


def _write_manifest_json(payload: dict[str, Any], *, report_dir: Path | str) -> Path:
    """Write safelisted ``manifest.json`` under an allowlisted report directory.

    Matches the in-repo Cycode/CWE-22 pattern (see ``stack_trace_source.read_source_file`` /
    ``phase2_generate_tests``): safelist ``BASE_DIRECTORY``, user path only as a relative
    ``dynamic_input``, then ``abspath(join(...))`` + ``startswith`` before ``open``.
    """
    if MANIFEST_FILENAME not in _ALLOWED_MANIFEST_BASENAMES:
        raise ValueError(f"Refusing to write non-safelisted path: {MANIFEST_FILENAME}")

    report_abs = os.path.realpath(os.path.abspath(str(report_dir)))
    base_candidate = next(
        (root for root in _manifest_write_allowlist_roots() if _path_is_under_root(report_abs, root)),
        None,
    )
    if base_candidate is None:
        raise ValueError(f"Refusing to write manifest outside allowlisted roots: {report_abs}")

    BASE_DIRECTORY = os.path.abspath(base_candidate)
    rel_dir = os.path.relpath(report_abs, BASE_DIRECTORY)
    if rel_dir == "..":
        raise ValueError(f"Refusing to write outside base directory: {report_abs}")
    if rel_dir.startswith(".." + os.sep):
        raise ValueError(f"Refusing to write outside base directory: {report_abs}")

    dynamic_input = MANIFEST_FILENAME if rel_dir in (os.curdir, "") else os.path.join(rel_dir, MANIFEST_FILENAME)
    if dynamic_input == ".." or dynamic_input.startswith(".." + os.sep):
        raise ValueError(f"Refusing to write outside base directory: {dynamic_input}")

    my_path = os.path.abspath(os.path.join(BASE_DIRECTORY, dynamic_input))
    if my_path.startswith(BASE_DIRECTORY):
        os.makedirs(os.path.dirname(my_path), exist_ok=True)
        with open(my_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        return Path(my_path)

    raise ValueError(f"Refusing to write outside base directory: {my_path} not under {BASE_DIRECTORY}")


def _path_for_manifest(path: Path, report_dir: Path) -> str:
    """Return a path safe to embed in uploadable manifests (no host absolutes)."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(report_dir).as_posix()
    except ValueError:
        return resolved.name


def write_performance_manifest(
    report_dir: Path | str,
    *,
    ops_csv: Optional[Path | str] = None,
) -> Optional[Path]:
    """Write ``manifest.json`` beside a Tracy ops report. No-op if run_id unset.

    Path fields are report-relative (or basenames if outside the report dir) so
    uploaded manifests do not leak host absolute paths.
    """
    run_id = peek_run_id()
    if not run_id:
        return None

    report_dir = Path(report_dir).resolve()
    ops_csv_path = Path(ops_csv) if ops_csv else None
    if ops_csv_path is None:
        csv_candidates = sorted(report_dir.glob("ops_perf_results*.csv"))
        ops_csv_path = csv_candidates[0] if csv_candidates else None

    payload: dict[str, Any] = {
        RUN_ID_METADATA_KEY: run_id,
        "artifact": "performance",
        "report_dir": ".",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if ops_csv_path is not None:
        payload["ops_csv"] = _path_for_manifest(ops_csv_path, report_dir)

    tracy_files = sorted(report_dir.glob("*.tracy"))
    if tracy_files:
        payload["tracy_file"] = _path_for_manifest(tracy_files[0], report_dir)

    device_log = report_dir / "profile_log_device.csv"
    if device_log.is_file():
        payload["device_log"] = _path_for_manifest(device_log, report_dir)

    manifest_path = _write_manifest_json(payload, report_dir=report_dir)
    logger.info(f"Visualizer run_id={run_id} written to {manifest_path}")
    return manifest_path


def find_memory_report_dir(
    run_id: str,
    root: Optional[Path | str] = None,
) -> Optional[Path]:
    """Find the newest TTNN report dir whose ``db.sqlite`` has ``run_id``."""
    root_path = Path(root) if root is not None else DEFAULT_TTNN_REPORTS_ROOT
    if not root_path.is_dir():
        return None

    matches: list[tuple[float, Path]] = []
    for db_path in root_path.glob("**/db.sqlite"):
        if read_db_run_id(db_path) == run_id:
            matches.append((db_path.stat().st_mtime, db_path.parent))

    if not matches:
        return None
    matches.sort(key=lambda item: item[0], reverse=True)
    return matches[0][1]
