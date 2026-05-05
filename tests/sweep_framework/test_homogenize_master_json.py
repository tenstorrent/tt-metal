#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for model_tracer.homogenize_master_json.

Covers the merge semantics the homogenize CLI must preserve:
  §1  Argument-based configuration dedup and execution union
  §2  Execution (source, machine_info) match ⇒ max(count) collapse
  §3  trace_uid rewrite across every execution
  §4  Top-level metadata recomputation
  §5  Hardware-uniformity enforcement
  §6  Input discovery (file / dir / glob) and empty-input rejection
  §7  Explicit --trace-uid override
"""

import json
import uuid

import pytest

from model_tracer.homogenize_master_json import (
    collect_input_files,
    load_master_jsons,
    main,
    merge_master_jsons,
    validate_hardware_uniformity,
)


# ── Fixtures ─────────────────────────────────────────────────────────────


N300_HW = {
    "board_type": "Wormhole",
    "device_series": "n300",
    "card_count": 1,
    "mesh_device_shape": [1, 2],
    "device_count": 2,
}

GALAXY_HW = {
    "board_type": "Wormhole",
    "device_series": "tt-galaxy-wh",
    "card_count": 32,
    "mesh_device_shape": [4, 8],
    "device_count": 32,
}


def _cfg(args, executions, config_hash="hash_placeholder", config_id=1):
    return {
        "config_id": config_id,
        "config_hash": config_hash,
        "arguments": args,
        "executions": executions,
    }


def _exec(source, machine_info=None, count=1, trace_uid="input-uid"):
    return {
        "source": source,
        "machine_info": machine_info or N300_HW,
        "count": count,
        "trace_uid": trace_uid,
    }


def _master(operations, models=None):
    return {
        "operations": operations,
        "metadata": {
            "models": models or [],
            "unique_operations": len(operations),
            "total_configurations": sum(len(op["configurations"]) for op in operations.values()),
        },
    }


def _write(tmp_path, name, data):
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return path


# ── §1 Configuration dedup ───────────────────────────────────────────────


def test_merges_distinct_configs_under_same_operation(tmp_path):
    master_a = _master(
        {
            "ttnn::add": {
                "configurations": [_cfg({"arg0": "tensor_a"}, [_exec("modelA.py")])],
            }
        },
        models=["modelA"],
    )
    master_b = _master(
        {
            "ttnn::add": {
                "configurations": [_cfg({"arg0": "tensor_b"}, [_exec("modelB.py")])],
            }
        },
        models=["modelB"],
    )

    result = merge_master_jsons([("a", master_a), ("b", master_b)], "fresh-uid")

    configs = result["operations"]["ttnn::add"]["configurations"]
    assert len(configs) == 2
    assert {json.dumps(c["arguments"], sort_keys=True) for c in configs} == {
        json.dumps({"arg0": "tensor_a"}, sort_keys=True),
        json.dumps({"arg0": "tensor_b"}, sort_keys=True),
    }


def test_dedupes_configs_with_identical_arguments(tmp_path):
    shared_args = {"arg0": "tensor", "scalar": 1.5}
    master_a = _master(
        {"ttnn::add": {"configurations": [_cfg(shared_args, [_exec("modelA.py")])]}},
        models=["modelA"],
    )
    master_b = _master(
        {"ttnn::add": {"configurations": [_cfg(shared_args, [_exec("modelB.py")])]}},
        models=["modelB"],
    )

    result = merge_master_jsons([("a", master_a), ("b", master_b)], "fresh-uid")

    configs = result["operations"]["ttnn::add"]["configurations"]
    assert len(configs) == 1
    sources = sorted(e["source"] for e in configs[0]["executions"])
    assert sources == ["modelA.py", "modelB.py"]


# ── §2 Execution collapse on matching (source, machine_info) ─────────────


def test_matching_source_and_machine_info_takes_max_count():
    shared_args = {"arg0": "t"}
    master_a = _master(
        {
            "ttnn::relu": {
                "configurations": [_cfg(shared_args, [_exec("demo.py", count=3)])],
            }
        }
    )
    master_b = _master(
        {
            "ttnn::relu": {
                "configurations": [_cfg(shared_args, [_exec("demo.py", count=10)])],
            }
        }
    )

    result = merge_master_jsons([("a", master_a), ("b", master_b)], "fresh-uid")

    executions = result["operations"]["ttnn::relu"]["configurations"][0]["executions"]
    assert len(executions) == 1
    assert executions[0]["count"] == 10


def test_same_source_different_machine_info_stays_distinct():
    shared_args = {"arg0": "t"}
    master_a = _master(
        {
            "ttnn::relu": {
                "configurations": [_cfg(shared_args, [_exec("demo.py", machine_info=N300_HW)])],
            }
        }
    )
    # Different mesh_device_shape within the same (board_type, device_series, card_count)
    alt_n300 = dict(N300_HW, mesh_device_shape=[2, 1])
    master_b = _master(
        {
            "ttnn::relu": {
                "configurations": [_cfg(shared_args, [_exec("demo.py", machine_info=alt_n300)])],
            }
        }
    )

    result = merge_master_jsons([("a", master_a), ("b", master_b)], "fresh-uid")
    executions = result["operations"]["ttnn::relu"]["configurations"][0]["executions"]
    assert len(executions) == 2


# ── §3 trace_uid rewrite ─────────────────────────────────────────────────


def test_trace_uid_rewritten_across_all_executions():
    master_a = _master(
        {
            "ttnn::add": {
                "configurations": [
                    _cfg({"arg0": "a"}, [_exec("m1.py", trace_uid="OLD-1")]),
                    _cfg({"arg0": "b"}, [_exec("m2.py", trace_uid="OLD-2")]),
                ]
            },
            "ttnn::mul": {
                "configurations": [_cfg({"arg0": "c"}, [_exec("m3.py", trace_uid="OLD-3")])],
            },
        }
    )

    result = merge_master_jsons([("a", master_a)], "HOMOGENIZED-UID")

    seen_uids = set()
    for op in result["operations"].values():
        for cfg in op["configurations"]:
            for execution in cfg["executions"]:
                seen_uids.add(execution["trace_uid"])
    assert seen_uids == {"HOMOGENIZED-UID"}


# ── §4 Metadata recomputation ────────────────────────────────────────────


def test_metadata_reflects_merged_contents():
    master_a = _master(
        {
            "ttnn::add": {
                "configurations": [
                    _cfg({"arg0": "x"}, [_exec("m1.py")]),
                    _cfg({"arg0": "y"}, [_exec("m1.py")]),
                ]
            }
        },
        models=["alpha", "shared"],
    )
    master_b = _master(
        {
            "ttnn::mul": {
                "configurations": [_cfg({"arg0": "z"}, [_exec("m2.py")])],
            }
        },
        models=["beta", "shared"],
    )

    result = merge_master_jsons([("a", master_a), ("b", master_b)], "uid")

    meta = result["metadata"]
    assert meta["trace_uid"] == "uid"
    assert meta["unique_operations"] == 2
    assert meta["total_configurations"] == 3
    assert meta["operations_summary"] == {"ttnn::add": 2, "ttnn::mul": 1}
    assert meta["models"] == ["alpha", "beta", "shared"]


def test_config_ids_are_reassigned_sequentially():
    master_a = _master(
        {
            "ttnn::add": {
                "configurations": [
                    _cfg({"arg0": "a"}, [_exec("m1.py")], config_id=5),
                    _cfg({"arg0": "b"}, [_exec("m1.py")], config_id=99),
                ]
            }
        }
    )
    master_b = _master(
        {
            "ttnn::mul": {
                "configurations": [_cfg({"arg0": "c"}, [_exec("m2.py")], config_id=42)],
            }
        }
    )

    result = merge_master_jsons([("a", master_a), ("b", master_b)], "uid")

    ids = []
    for op in sorted(result["operations"]):
        for cfg in result["operations"][op]["configurations"]:
            ids.append(cfg["config_id"])
    assert ids == [1, 2, 3]


# ── §5 Hardware uniformity ───────────────────────────────────────────────


def test_rejects_mixed_hardware():
    master_a = _master({"ttnn::add": {"configurations": [_cfg({}, [_exec("m1.py", machine_info=N300_HW)])]}})
    master_b = _master({"ttnn::add": {"configurations": [_cfg({}, [_exec("m2.py", machine_info=GALAXY_HW)])]}})

    with pytest.raises(ValueError, match="Hardware profile mismatch"):
        validate_hardware_uniformity([("a", master_a), ("b", master_b)])


def test_accepts_uniform_hardware_with_varying_mesh_shapes():
    alt_n300 = dict(N300_HW, mesh_device_shape=[2, 1])
    master_a = _master({"ttnn::add": {"configurations": [_cfg({}, [_exec("m1.py", machine_info=N300_HW)])]}})
    master_b = _master({"ttnn::add": {"configurations": [_cfg({}, [_exec("m2.py", machine_info=alt_n300)])]}})

    hw = validate_hardware_uniformity([("a", master_a), ("b", master_b)])
    assert hw == ("Wormhole", "n300", 1)


# ── §6 Input discovery ────────────────────────────────────────────────────


def test_collect_input_files_expands_directory(tmp_path):
    _write(tmp_path, "a.json", {})
    _write(tmp_path, "b.json", {})
    (tmp_path / "not_json.txt").write_text("skip me")

    files = collect_input_files([str(tmp_path)])
    assert sorted(f.name for f in files) == ["a.json", "b.json"]


def test_collect_input_files_expands_glob(tmp_path):
    _write(tmp_path, "leg1.json", {})
    _write(tmp_path, "leg2.json", {})
    _write(tmp_path, "other.json", {})

    files = collect_input_files([str(tmp_path / "leg*.json")])
    assert sorted(f.name for f in files) == ["leg1.json", "leg2.json"]


def test_collect_input_files_dedupes_overlapping_specs(tmp_path):
    p = _write(tmp_path, "one.json", {})

    files = collect_input_files([str(p), str(tmp_path)])
    assert [f.name for f in files] == ["one.json"]


def test_main_rejects_empty_input(tmp_path, capsys):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    exit_code = main(["--input", str(empty_dir), "--output", str(tmp_path / "out.json")])

    assert exit_code == 1
    assert "No input master JSON files" in capsys.readouterr().err


# ── §7 CLI end-to-end ────────────────────────────────────────────────────


def test_main_honors_explicit_trace_uid(tmp_path):
    master = _master(
        {
            "ttnn::add": {
                "configurations": [_cfg({"arg0": "a"}, [_exec("m1.py", trace_uid="OLD")])],
            }
        }
    )
    _write(tmp_path, "leg.json", master)
    out = tmp_path / "merged.json"

    explicit = str(uuid.uuid4())
    exit_code = main(
        [
            "--input",
            str(tmp_path),
            "--output",
            str(out),
            "--trace-uid",
            explicit,
        ]
    )

    assert exit_code == 0
    merged = json.loads(out.read_text())
    assert merged["metadata"]["trace_uid"] == explicit
    executions = merged["operations"]["ttnn::add"]["configurations"][0]["executions"]
    assert all(e["trace_uid"] == explicit for e in executions)


def test_main_generates_fresh_uid_when_not_provided(tmp_path):
    master = _master({"ttnn::add": {"configurations": [_cfg({"arg0": "a"}, [_exec("m1.py", trace_uid="OLD")])]}})
    _write(tmp_path, "leg.json", master)
    out = tmp_path / "merged.json"

    exit_code = main(["--input", str(tmp_path), "--output", str(out)])

    assert exit_code == 0
    merged = json.loads(out.read_text())
    # Validate UUID4 format
    parsed = uuid.UUID(merged["metadata"]["trace_uid"])
    assert str(parsed) == merged["metadata"]["trace_uid"]
    assert merged["metadata"]["trace_uid"] != "OLD"


def test_main_fails_fast_on_mixed_hardware(tmp_path, capsys):
    master_a = _master({"ttnn::add": {"configurations": [_cfg({}, [_exec("m1.py", machine_info=N300_HW)])]}})
    master_b = _master({"ttnn::add": {"configurations": [_cfg({}, [_exec("m2.py", machine_info=GALAXY_HW)])]}})
    _write(tmp_path, "a.json", master_a)
    _write(tmp_path, "b.json", master_b)

    exit_code = main(["--input", str(tmp_path), "--output", str(tmp_path / "out.json")])

    assert exit_code == 1
    assert "Hardware profile mismatch" in capsys.readouterr().err


def test_load_master_jsons_preserves_order(tmp_path):
    p1 = _write(tmp_path, "01.json", _master({"op": {"configurations": []}}, models=["one"]))
    p2 = _write(tmp_path, "02.json", _master({"op": {"configurations": []}}, models=["two"]))

    loaded = load_master_jsons([p1, p2])
    assert [data["metadata"]["models"] for _, data in loaded] == [["one"], ["two"]]
