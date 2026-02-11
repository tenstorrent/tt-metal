# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import json
import os
import sys
from dataclasses import dataclass
from functools import reduce
from operator import mul
from pathlib import Path
from typing import Any, Iterable

import pytest
import torch

import ttnn
from models.demos.deepseek_v3.utils import config_helpers

SPECS_JSONL_ENV_VAR = "DEEPSEEK_V3_CACHE_SPECS_JSONL"
REPORT_JSON_ENV_VAR = "DEEPSEEK_V3_CACHE_SPECS_REPORT_JSON"

DEFAULT_SPECS_GLOB = "deepseek_v3_cache_specs*.jsonl"
DEFAULT_REPORT_NAME = "deepseek_v3_cache_specs_report.json"

HARNESS_PATH = Path("./models/demos/deepseek_v3/tmp/deepseek-cache-specs-harness")


def product(xs: Iterable[int]) -> int:
    return reduce(mul, xs, 1)


def parse_torch_dtype(dtype_str: str) -> torch.dtype:
    if not dtype_str.startswith("torch."):
        raise ValueError(f"Unexpected torch dtype string: {dtype_str}")
    name = dtype_str.split(".", 1)[1]
    try:
        return getattr(torch, name)
    except AttributeError as e:
        raise ValueError(f"Unknown torch dtype name '{name}' parsed from '{dtype_str}'") from e


def parse_ttnn_dtype(dtype_name: str | None) -> ttnn.DataType | None:
    if dtype_name is None:
        return None
    alias = getattr(ttnn, dtype_name.lower(), None)
    if alias is not None:
        return alias
    if hasattr(ttnn, "DataType") and hasattr(ttnn.DataType, dtype_name):
        return getattr(ttnn.DataType, dtype_name)
    raise ValueError(f"Unknown TTNN dtype name: {dtype_name}")


def parse_ttnn_layout(layout_name: str | None) -> ttnn.Layout | None:
    if layout_name is None:
        return None
    if layout_name in {"ROW_MAJOR", "ROW_MAJOR_LAYOUT"}:
        return ttnn.ROW_MAJOR_LAYOUT
    if layout_name in {"TILE", "TILE_LAYOUT"}:
        return ttnn.TILE_LAYOUT
    if hasattr(ttnn, "Layout") and hasattr(ttnn.Layout, layout_name):
        return getattr(ttnn.Layout, layout_name)
    raise ValueError(f"Unknown TTNN layout name: {layout_name}")


def parse_memory_config(memory_config_dict: dict[str, Any] | None) -> ttnn.MemoryConfig | None:
    if memory_config_dict is None:
        return None
    if "__error__" in memory_config_dict:
        raise ValueError(f"Cannot parse memory_config due to dump error: {memory_config_dict['__error__']}")
    return ttnn.MemoryConfig.from_json(json.dumps(memory_config_dict))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r") as f:
        for line_number, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                records.append(json.loads(s))
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Failed parsing JSONL at {path}:{line_number}: {e}") from e
    return records


def find_repo_root(start: Path) -> Path:
    p = start.resolve()
    for candidate in (p, *p.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    try:
        return p.parents[4]
    except Exception:
        return p.parent


def auto_find_specs_jsonl() -> Path | None:
    repo_root = find_repo_root(Path(__file__))
    candidates: list[Path] = []
    for d in (repo_root / "tmp", repo_root / "generated"):
        if d.exists():
            candidates.extend(sorted(d.glob(DEFAULT_SPECS_GLOB)))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def explicitly_selected(pytestconfig) -> bool:
    args = getattr(pytestconfig, "args", []) or []
    return any("test_cache_specs_harness.py" in str(a) for a in args) or any(
        "test_replay_cache_tensor_specs" in str(a) for a in args
    )


def signature_for_record(record: dict[str, Any]) -> str:
    signature_obj = {
        "requested_dtype": record.get("requested_dtype"),
        "requested_layout": record.get("requested_layout"),
        "requested_memory_config": record.get("requested_memory_config"),
        "shard_dims": record.get("shard_dims"),
        "remove_dims": record.get("remove_dims"),
        "mesh_shape": record.get("mesh_shape"),
        "torch_dtype": record.get("torch_dtype"),
    }
    return json.dumps(signature_obj, sort_keys=True)


@dataclass(frozen=True)
class ReplayResult:
    signature: str
    cache_file_relpath: str | None
    torch_shape: list[int] | None
    status: str


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        }
    ],
    indirect=True,
)
def test_replay_cache_tensor_specs(mesh_device, device_params, pytestconfig) -> None:
    specs_path_str = os.getenv(SPECS_JSONL_ENV_VAR)
    if specs_path_str:
        specs_path = Path(specs_path_str)
    else:
        if not explicitly_selected(pytestconfig):
            pytest.skip(
                f"Cache specs harness is opt-in. Re-run selecting this file explicitly, or set ${SPECS_JSONL_ENV_VAR}."
            )
        specs_path = auto_find_specs_jsonl()
        if specs_path is None:
            pytest.skip(
                f"No specs JSONL found (looked for '{DEFAULT_SPECS_GLOB}' under <repo>/tmp and <repo>/generated)."
            )

    if not specs_path.exists():
        pytest.skip(f"Specs JSONL does not exist: {specs_path}")

    records = load_jsonl(specs_path)
    ok_records = [r for r in records if r.get("status") == "ok" and r.get("event") == "deepseek_v3.cache_tensor_spec"]
    if not ok_records:
        pytest.skip(f"No usable cache spec records found in {specs_path}")

    cases: list[tuple[str, dict[str, Any]]] = []
    for record in ok_records:
        cases.append((signature_for_record(record), record))

    selected = list(cases)
    selected.sort(key=lambda kv: product(kv[1]["torch_shape"]))

    replay_results: list[ReplayResult] = []
    current_mesh_shape = list(mesh_device.shape)

    for sig, record in selected:
        if record.get("mesh_shape") != current_mesh_shape:
            replay_results.append(
                ReplayResult(
                    signature=sig,
                    cache_file_relpath=record.get("cache_file_relpath"),
                    torch_shape=record.get("torch_shape"),
                    status=f"skipped(mesh_shape_mismatch captured={record.get('mesh_shape')} current={current_mesh_shape})",
                )
            )
            continue

        torch_shape = tuple(record["torch_shape"])

        torch_dtype = parse_torch_dtype(record["torch_dtype"])
        requested_dtype = parse_ttnn_dtype(record.get("requested_dtype"))
        requested_layout = parse_ttnn_layout(record.get("requested_layout"))
        requested_memory_config = parse_memory_config(record.get("requested_memory_config"))

        shard_dims = tuple(record.get("shard_dims", (None, None)))
        remove_dims = tuple(record.get("remove_dims", (False, False)))

        torch_tensor = torch.empty(torch_shape, dtype=torch_dtype)

        tt_tensor = None
        try:
            if record.get("torch_impl"):
                tt_tensor = config_helpers._shard_torch_impl(
                    path=HARNESS_PATH,
                    tensor=torch_tensor,
                    shard_dims=shard_dims,
                    mesh_device=mesh_device,
                    remove_dims=remove_dims,
                    dtype=requested_dtype,
                    layout=requested_layout,
                    memory_config=requested_memory_config,
                )
            else:
                tt_tensor = config_helpers._shard_device_impl(
                    path=HARNESS_PATH,
                    tensor=torch_tensor,
                    shard_dims=shard_dims,
                    mesh_device=mesh_device,
                    remove_dims=remove_dims,
                    dtype=requested_dtype,
                    layout=requested_layout,
                    memory_config=requested_memory_config,
                )

            if record.get("result_dtype") is not None:
                assert config_helpers._enum_name_or_str(tt_tensor.dtype) == record["result_dtype"]
            if record.get("result_layout") is not None:
                assert config_helpers._enum_name_or_str(tt_tensor.layout) == record["result_layout"]
            if record.get("result_memory_config") is not None:
                got_mc = config_helpers._memory_config_to_dict(tt_tensor.memory_config())
                assert got_mc == record["result_memory_config"]

            replay_results.append(
                ReplayResult(
                    signature=sig,
                    cache_file_relpath=record.get("cache_file_relpath"),
                    torch_shape=record.get("torch_shape"),
                    status="ok",
                )
            )
        except Exception as e:
            replay_results.append(
                ReplayResult(
                    signature=sig,
                    cache_file_relpath=record.get("cache_file_relpath"),
                    torch_shape=record.get("torch_shape"),
                    status=f"error({type(e).__name__}: {e})",
                )
            )
        finally:
            if tt_tensor is not None:
                try:
                    ttnn.deallocate(tt_tensor)
                except Exception:
                    pass

            del torch_tensor
            gc.collect()

    report_path_str = os.getenv(REPORT_JSON_ENV_VAR)
    report_path = Path(report_path_str) if report_path_str else (specs_path.parent / DEFAULT_REPORT_NAME)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "specs_jsonl": str(specs_path),
        "current_mesh_shape": current_mesh_shape,
        "check_mode": "record",
        "num_records_total": len(records),
        "num_records_ok": len(ok_records),
        "num_cases_selected": len(selected),
        "results": [r.__dict__ for r in replay_results],
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"[deepseek_v3 cache specs] wrote report: {report_path}", file=sys.stderr)
