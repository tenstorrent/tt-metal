# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Generate pytest tests from a JSONL of captured TTNN op calls.

Usage:

  python -m models.demos.deepseek_v3.utils.generate_op_tests \
    --jsonl /path/to/op_calls.jsonl \
    --out-dir models/demos/deepseek_v3/tests/op_repros \
    --tests-per-group 1

The generator groups op records by a stable signature (op name + input tensor shapes/mem + kwarg config types)
and emits a parametric pytest that replays a small sample per group using the op replay helper.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _tensor_sig(spec: Dict[str, Any]) -> Dict[str, Any]:
    # Expecting a TensorRef spec
    if not isinstance(spec, dict) or spec.get("__type__") != "ttnn.TensorRef":
        return {"other": spec.get("__type__") if isinstance(spec, dict) else type(spec).__name__}
    mem = spec.get("memory_config") or spec.get("mem") or {}
    shard = mem.get("shard_spec") or mem.get("shard") or {}
    return {
        "shape": spec.get("shape"),
        "dtype": spec.get("dtype"),
        "layout": mem.get("memory_layout") or mem.get("layout"),
        "buffer": mem.get("buffer_type") or mem.get("buffer"),
        "shard_dims": shard.get("dims"),
        "shard_shape": shard.get("shape"),
    }


def _value_sig(v: Any) -> Any:
    # Reduce kwargs/configs to type names
    if isinstance(v, dict):
        t = v.get("__type__")
        if t == "ttnn.TensorRef":
            return _tensor_sig(v)
        if t:
            return {"__type__": t}
        # generic dict
        return {k: _value_sig(val) for k, val in v.items()}
    if isinstance(v, list):
        return [_value_sig(x) for x in v]
    return type(v).__name__


def _record_signature(rec: Dict[str, Any]) -> str:
    op = rec.get("op_name")
    ins = rec.get("inputs", [])
    kws = rec.get("kwargs", {})
    sig = {
        "op": op,
        "inputs": [_value_sig(x) for x in ins],
        "kwargs": {k: _value_sig(v) for k, v in kws.items()},
    }
    return json.dumps(sig, sort_keys=True)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def _group_records(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        key = _record_signature(rec)
        groups.setdefault(key, []).append(rec)
    return groups


def _emit_test_file(out_dir: Path, grouped: Dict[str, List[Dict[str, Any]]], tests_per_group: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "test_generated_ops.py"
    cases: List[str] = []
    for sig, recs in grouped.items():
        for rec in recs[:tests_per_group]:
            cases.append(json.dumps(rec, separators=(",", ":")))

    content = """# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.\n# SPDX-License-Identifier: Apache-2.0\n\nimport json\nimport pytest\n\nfrom models.demos.deepseek_v3.utils.op_replay import replay_op_record\n\n# Generated cases (JSON strings)\nRECORDS = [\n{records}\n]\n\n@pytest.mark.parametrize("rec_json", RECORDS)\ndef test_replay_generated_op(rec_json, mesh_device):\n    rec = json.loads(rec_json)\n    result = replay_op_record(rec, mesh_device, rng_seed=0)\n    # Basic sanity: output exists; assert shape if provided in record\n    out_spec = rec.get("output")\n    if isinstance(out_spec, dict) and "shape" in out_spec:\n        shape = getattr(result, "shape", None)\n        assert shape is not None, "Result has no shape attribute"\n        assert list(shape) == out_spec.get("shape"), f"Output shape mismatch: {shape} vs {out_spec.get('shape')}"\n    else:\n        assert result is not None\n""".format(records=",\n".join("    " + json.dumps(s) for s in cases))

    out_path.write_text(content)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Generate pytest tests from TTNN op JSONL")
    parser.add_argument("--jsonl", required=True, help="Path to op_calls.jsonl from op_capture_plugin")
    parser.add_argument("--out-dir", required=True, help="Directory to write pytest file into")
    parser.add_argument("--tests-per-group", type=int, default=1, help="Max cases per op signature group")
    args = parser.parse_args()

    records = _read_jsonl(Path(args.jsonl))
    if not records:
        raise SystemExit("No records read from JSONL")
    grouped = _group_records(records)
    path = _emit_test_file(Path(args.out_dir), grouped, args.tests_per_group)
    print(f"Wrote generated tests to {path}")


if __name__ == "__main__":
    main()

