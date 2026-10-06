# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Preserve each run's inputs and counters before forming the legacy LCOV union."""

import dataclasses
import json
import os
import subprocess
from enum import Enum
from pathlib import Path

from .gcov_json import collect_variant


def json_value(value):
    if isinstance(value, Enum):
        return value.name
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: json_value(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def run_metadata(config) -> dict:
    return {
        "schema_version": 1,
        "test": os.environ.get("PYTEST_CURRENT_TEST", "").rsplit(" (", 1)[0],
        "runtime_parameters": json_value(config.runtimes),
        "formats": json_value(config.formats_config),
        "template_configuration": json_value(config.templates),
        "compile_time_formats": config.compile_time_formats,
        "speed_of_light": config.SPEED_OF_LIGHT,
        "input_scope": "test-run inputs, not values sampled at each LLK call",
    }


def merge_streams(tool: str, variant: Path, streams: list[Path], cwd: Path):
    for counters in (variant / "elf").glob("*.gcda"):
        counters.unlink()
    payload = b"".join(stream.read_bytes() for stream in streams)
    if not payload:
        raise ValueError(f"Empty coverage stream in {variant}")
    subprocess.run(
        [tool, "merge-stream"], input=payload, cwd=cwd, check=True, capture_output=True
    )


def collect_streams(
    gcov: str,
    tool: str,
    variant: Path,
    arch: str,
    cwd: Path,
    legacy_union: bool = False,
) -> list[dict]:
    records = []
    streams = sorted(variant.glob("*.stream"))
    if not streams:
        return collect_variant(gcov, variant, arch, initial=True)
    for stream in streams:
        merge_streams(tool, variant, [stream], cwd)
        metadata = stream.with_suffix(".run.json")
        run = (
            json.loads(metadata.read_text())
            if metadata.exists()
            else {"input_scope": "historical stream; runtime values were not recorded"}
        )
        for record in collect_variant(gcov, variant, arch):
            if record.get("measurement") == "build_only":
                raise ValueError(
                    f"Stream {stream} did not produce counters for {record['object']}; check embedded build paths"
                )
            record.update(run_id=stream.stem, run=run)
            records.append(record)
    if legacy_union and len(streams) > 1:
        merge_streams(tool, variant, streams, cwd)
    return records
