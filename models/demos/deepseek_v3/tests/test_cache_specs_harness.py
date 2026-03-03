# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import json
import os
import sys
import time
from dataclasses import dataclass
from functools import reduce
from operator import mul
from pathlib import Path
from typing import Any, Callable, Iterable

import pytest
import torch

import ttnn
from models.demos.deepseek_v3.utils import config_helpers

SPECS_JSONL_ENV_VAR = "DEEPSEEK_V3_CACHE_SPECS_JSONL"
LEGACY_SPECS_JSONL_ENV_VAR = "DEEPSEEK_V3_DUMP_CACHE_SPECS"
REPORT_JSON_ENV_VAR = "DEEPSEEK_V3_CACHE_SPECS_REPORT_JSON"
CONVERTERS_ENV_VAR = "DEEPSEEK_V3_CACHE_SPECS_CONVERTERS"

DEFAULT_REPORT_NAME = "deepseek_v3_cache_specs_report.json"


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
    try:
        return getattr(ttnn.DataType, dtype_name)
    except AttributeError:
        pass
    raise ValueError(f"Unknown TTNN dtype name: {dtype_name}")


def parse_ttnn_layout(layout_name: str | None) -> ttnn.Layout | None:
    if layout_name is None:
        return None
    if layout_name in {"ROW_MAJOR", "ROW_MAJOR_LAYOUT"}:
        return ttnn.ROW_MAJOR_LAYOUT
    if layout_name in {"TILE", "TILE_LAYOUT"}:
        return ttnn.TILE_LAYOUT
    try:
        return getattr(ttnn.Layout, layout_name)
    except AttributeError:
        pass
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


ConverterFn = Callable[..., Any]
CONVERTER_FNS: dict[str, ConverterFn] = {
    "host_shards_slow": config_helpers._shard_torch_impl,
    "device_from_torch": config_helpers._shard_device_impl,
}


def resolve_converter_fns() -> list[tuple[str, ConverterFn]]:
    raw = os.getenv(CONVERTERS_ENV_VAR)
    if not raw:
        return list(CONVERTER_FNS.items())

    requested = [name.strip() for name in raw.split(",") if name.strip()]
    if not requested:
        raise ValueError(
            f"${CONVERTERS_ENV_VAR} was set but empty after parsing; expected comma-separated converter names."
        )
    unknown = [name for name in requested if name not in CONVERTER_FNS]
    if unknown:
        raise ValueError(
            f"Unknown converter(s) in ${CONVERTERS_ENV_VAR}: {unknown}. " f"Known values: {sorted(CONVERTER_FNS)}"
        )
    return [(name, CONVERTER_FNS[name]) for name in requested]


def status_bucket(status: str) -> str:
    if status == "ok":
        return "ok"
    if status.startswith("skipped("):
        return "skipped"
    return "error"


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (p / 100.0) * (len(sorted_values) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def summarize_results(replay_results: list["ReplayResult"], converter_names: list[str]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for converter in converter_names:
        converter_rows = [r for r in replay_results if r.converter == converter]
        ok_rows = [r for r in converter_rows if r.status == "ok"]
        ok_total_ms = [
            r.total_ms if r.total_ms is not None else r.elapsed_ms for r in ok_rows if r.elapsed_ms is not None
        ]
        ok_convert_ms = [r.convert_ms for r in ok_rows if r.convert_ms is not None]
        ok_validate_ms = [r.validate_ms for r in ok_rows if r.validate_ms is not None]
        ok_deallocate_ms = [r.deallocate_ms for r in ok_rows if r.deallocate_ms is not None]
        counts = {
            "ok": sum(1 for row in converter_rows if status_bucket(row.status) == "ok"),
            "error": sum(1 for row in converter_rows if status_bucket(row.status) == "error"),
            "skipped": sum(1 for row in converter_rows if status_bucket(row.status) == "skipped"),
        }
        summary[converter] = {
            "total": len(converter_rows),
            **counts,
            # Keep avg/p50/p95 alias fields for backward compatibility; they represent total/e2e time.
            "avg_ok_ms": (sum(ok_total_ms) / len(ok_total_ms)) if ok_total_ms else None,
            "p50_ok_ms": percentile(ok_total_ms, 50) if ok_total_ms else None,
            "p95_ok_ms": percentile(ok_total_ms, 95) if ok_total_ms else None,
            "avg_ok_total_ms": (sum(ok_total_ms) / len(ok_total_ms)) if ok_total_ms else None,
            "p50_ok_total_ms": percentile(ok_total_ms, 50) if ok_total_ms else None,
            "p95_ok_total_ms": percentile(ok_total_ms, 95) if ok_total_ms else None,
            "avg_ok_convert_ms": (sum(ok_convert_ms) / len(ok_convert_ms)) if ok_convert_ms else None,
            "avg_ok_validate_ms": (sum(ok_validate_ms) / len(ok_validate_ms)) if ok_validate_ms else None,
            "avg_ok_deallocate_ms": (sum(ok_deallocate_ms) / len(ok_deallocate_ms)) if ok_deallocate_ms else None,
        }
    return summary


def build_summary_markdown_table(converter_summary: dict[str, dict[str, Any]]) -> str:
    def fmt(value: float | None) -> str:
        return f"{value:.3f}" if value is not None else "n/a"

    lines = [
        "| converter | total | ok | error | skipped | avg_ok_total_ms | p50_ok_total_ms | p95_ok_total_ms | avg_ok_convert_ms | avg_ok_validate_ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for converter_name, stats in converter_summary.items():
        avg_total = stats["avg_ok_total_ms"]
        p50_total = stats["p50_ok_total_ms"]
        p95_total = stats["p95_ok_total_ms"]
        avg_convert = stats["avg_ok_convert_ms"]
        avg_validate = stats["avg_ok_validate_ms"]
        lines.append(
            "| "
            f"{converter_name} | {stats['total']} | {stats['ok']} | {stats['error']} | {stats['skipped']} | "
            f"{fmt(avg_total)} | {fmt(p50_total)} | {fmt(p95_total)} | {fmt(avg_convert)} | {fmt(avg_validate)} |"
        )
    return "\n".join(lines)


@dataclass(frozen=True)
class ReplayResult:
    signature: str
    cache_file_relpath: str | None
    torch_shape: list[int] | None
    converter: str
    recorded_torch_impl: bool | None
    elapsed_ms: float | None
    convert_ms: float | None
    validate_ms: float | None
    total_ms: float | None
    deallocate_ms: float | None
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
@pytest.mark.timeout(1800)
def test_replay_cache_tensor_specs(mesh_device, device_params) -> None:
    specs_path_str = os.getenv(SPECS_JSONL_ENV_VAR) or os.getenv(LEGACY_SPECS_JSONL_ENV_VAR)
    if not specs_path_str:
        pytest.skip(
            f"Cache specs harness is opt-in. Set ${SPECS_JSONL_ENV_VAR} (or legacy ${LEGACY_SPECS_JSONL_ENV_VAR})."
        )
    specs_path = Path(specs_path_str)

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
    converter_fns = resolve_converter_fns()
    converter_names = [name for name, _ in converter_fns]

    harness_path = specs_path.parent / "deepseek-cache-specs-harness.tensorbin"

    for sig, record in selected:
        if record.get("mesh_shape") != current_mesh_shape:
            for converter_name in converter_names:
                replay_results.append(
                    ReplayResult(
                        signature=sig,
                        cache_file_relpath=record.get("cache_file_relpath"),
                        torch_shape=record.get("torch_shape"),
                        converter=converter_name,
                        recorded_torch_impl=record.get("torch_impl"),
                        elapsed_ms=None,
                        convert_ms=None,
                        validate_ms=None,
                        total_ms=None,
                        deallocate_ms=None,
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
        for converter_name, converter_fn in converter_fns:
            tt_tensor = None
            convert_ms = None
            validate_ms = None
            total_ms = None
            deallocate_ms = None
            status = "ok"
            total_start = time.perf_counter()
            validate_start = None
            try:
                convert_start = time.perf_counter()
                tt_tensor = converter_fn(
                    path=harness_path,
                    tensor=torch_tensor,
                    shard_dims=shard_dims,
                    mesh_device=mesh_device,
                    remove_dims=remove_dims,
                    dtype=requested_dtype,
                    layout=requested_layout,
                    memory_config=requested_memory_config,
                )
                convert_ms = (time.perf_counter() - convert_start) * 1000.0

                validate_start = time.perf_counter()
                if record.get("result_dtype") is not None:
                    assert config_helpers._enum_name_or_str(tt_tensor.dtype) == record["result_dtype"]
                if record.get("result_layout") is not None:
                    assert config_helpers._enum_name_or_str(tt_tensor.layout) == record["result_layout"]
                if record.get("result_memory_config") is not None:
                    got_mc = config_helpers._memory_config_to_dict(tt_tensor.memory_config())
                    assert got_mc == record["result_memory_config"]
                validate_ms = (time.perf_counter() - validate_start) * 1000.0
            except Exception as e:
                if validate_start is not None and validate_ms is None:
                    validate_ms = (time.perf_counter() - validate_start) * 1000.0
                status = f"error({type(e).__name__}: {e})"
            total_ms = (time.perf_counter() - total_start) * 1000.0

            if tt_tensor is not None:
                deallocate_start = time.perf_counter()
                try:
                    ttnn.deallocate(tt_tensor)
                except Exception as dealloc_err:
                    print(
                        "[deepseek_v3 cache specs] warning: failed to deallocate tt_tensor: "
                        f"{type(dealloc_err).__name__}: {dealloc_err}",
                        file=sys.stderr,
                    )
                deallocate_ms = (time.perf_counter() - deallocate_start) * 1000.0

            replay_results.append(
                ReplayResult(
                    signature=sig,
                    cache_file_relpath=record.get("cache_file_relpath"),
                    torch_shape=record.get("torch_shape"),
                    converter=converter_name,
                    recorded_torch_impl=record.get("torch_impl"),
                    elapsed_ms=total_ms,
                    convert_ms=convert_ms,
                    validate_ms=validate_ms,
                    total_ms=total_ms,
                    deallocate_ms=deallocate_ms,
                    status=status,
                )
            )
            gc.collect()

        del torch_tensor
        gc.collect()

    report_path_str = os.getenv(REPORT_JSON_ENV_VAR)
    report_path = Path(report_path_str) if report_path_str else (specs_path.parent / DEFAULT_REPORT_NAME)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    converter_summary = summarize_results(replay_results, converter_names)

    report = {
        "specs_jsonl": str(specs_path),
        "current_mesh_shape": current_mesh_shape,
        "check_mode": "record",
        "selected_converters": converter_names,
        "num_records_total": len(records),
        "num_records_ok": len(ok_records),
        "num_cases_selected": len(selected),
        "num_converter_runs": len(replay_results),
        "converter_summary": converter_summary,
        "converter_summary_markdown": build_summary_markdown_table(converter_summary),
        "results": [r.__dict__ for r in replay_results],
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"[deepseek_v3 cache specs] wrote report: {report_path}", file=sys.stderr)
