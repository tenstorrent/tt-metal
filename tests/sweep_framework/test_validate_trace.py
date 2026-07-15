#!/usr/bin/env python
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for model_tracer.validate_trace.

Covers the fail-fast guarantees the validator provides:
  §1  Valid master JSON passes and prints resolved artifact paths
  §2  Missing required fields (config_hash, source, provenance, machine_info) fail
  §3  Invalid enum values (layout / status) fail
  §4  Missing / malformed manifest fails (no silent degraded mode)
  §5  Tensor shape consistency and inline-value element counts
  §6  --print-resolved bounds
  §7  Registry status enum and targets -> registry cross references
"""

import copy
import json

import pytest

from model_tracer.validate_trace import main


N150_MACHINE_INFO = {
    "board_type": "Wormhole",
    "device_series": "n150",
    "card_count": 1,
    "mesh_device_shape": [1, 1],
    "device_count": 1,
}


def _tensor_arg(shape=(1, 1, 32, 128)):
    return {
        "type": "ttnn.Tensor",
        "original_shape": list(shape),
        "original_dtype": "DataType.BFLOAT16",
        "layout": "Layout.TILE",
        "storage_type": "StorageType.DEVICE",
        "memory_config": {
            "buffer_type": "BufferType.DRAM",
            "memory_layout": "TensorMemoryLayout.INTERLEAVED",
            "shard_spec": None,
        },
        "tensor_placement": {
            "placement": "['PlacementReplicate']",
            "distribution_shape": "[1]",
            "mesh_device_shape": "[1, 1]",
        },
    }


def _config(config_hash="hash0", config_id=1, arguments=None, executions=None):
    if arguments is None:
        arguments = {"arg0": _tensor_arg()}
    if executions is None:
        executions = [
            {
                "source": "models/demos/example/demo.py::test_demo",
                "trace_uid": "uid-abc-123",
                "machine_info": copy.deepcopy(N150_MACHINE_INFO),
                "count": 4,
            }
        ]
    return {
        "config_hash": config_hash,
        "config_id": config_id,
        "arguments": arguments,
        "executions": executions,
    }


def _master(configs=None):
    if configs is None:
        configs = [_config()]
    return {
        "metadata": {"total_configurations": len(configs), "unique_operations": 1},
        "operations": {"ttnn.add": {"configurations": configs}},
    }


def _write(tmp_path, data, name="master.json"):
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return path


def test_valid_master_passes_and_prints_resolved(tmp_path, capsys):
    path = _write(tmp_path, _master())
    exit_code = main(["--manifest", str(path), "--print-resolved", "1"])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Decision:        pass" in out
    assert "Resolved artifact paths (first 1 configurations)" in out
    assert "models/demos/example/demo.py::test_demo" in out


def test_reconstructed_dialect_with_trace_run_ids_passes(tmp_path):
    # DB-reconstructed traces have trace_run_ids instead of trace_uid.
    execution = {
        "source": "models/demos/example/demo.py",
        "trace_run_ids": [41],
        "machine_info": copy.deepcopy(N150_MACHINE_INFO),
        "count": 8,
    }
    path = _write(tmp_path, _master([_config(executions=[execution])]))
    assert main(["--manifest", str(path)]) == 0


def test_missing_config_hash_fails(tmp_path, capsys):
    cfg = _config()
    del cfg["config_hash"]
    path = _write(tmp_path, _master([cfg]))
    assert main(["--manifest", str(path)]) == 1
    assert "config_hash" in capsys.readouterr().out


def test_missing_source_fails(tmp_path, capsys):
    cfg = _config()
    del cfg["executions"][0]["source"]
    path = _write(tmp_path, _master([cfg]))
    assert main(["--manifest", str(path)]) == 1
    assert "execution.source" in capsys.readouterr().out


def test_missing_provenance_fails(tmp_path, capsys):
    cfg = _config()
    # Remove both trace_uid and trace_run_ids -> no provenance at all.
    del cfg["executions"][0]["trace_uid"]
    path = _write(tmp_path, _master([cfg]))
    assert main(["--manifest", str(path)]) == 1
    assert "provenance" in capsys.readouterr().out


@pytest.mark.parametrize("field", sorted(N150_MACHINE_INFO.keys()))
def test_missing_machine_info_field_fails(tmp_path, capsys, field):
    cfg = _config()
    del cfg["executions"][0]["machine_info"][field]
    path = _write(tmp_path, _master([cfg]))
    assert main(["--manifest", str(path)]) == 1
    assert f"machine_info.{field}" in capsys.readouterr().out


def test_empty_executions_fails(tmp_path, capsys):
    path = _write(tmp_path, _master([_config(executions=[])]))
    assert main(["--manifest", str(path)]) == 1
    assert "executions" in capsys.readouterr().out


def test_invalid_layout_enum_fails(tmp_path, capsys):
    arg = _tensor_arg()
    arg["layout"] = "TILE"  # missing the "Layout." prefix
    path = _write(tmp_path, _master([_config(arguments={"arg0": arg})]))
    assert main(["--manifest", str(path)]) == 1
    assert "layout" in capsys.readouterr().out


def test_invalid_memory_config_enum_fails(tmp_path, capsys):
    arg = _tensor_arg()
    arg["memory_config"]["buffer_type"] = "DRAM"  # missing "BufferType." prefix
    path = _write(tmp_path, _master([_config(arguments={"arg0": arg})]))
    assert main(["--manifest", str(path)]) == 1
    assert "buffer_type" in capsys.readouterr().out


def test_missing_manifest_fails(tmp_path, capsys):
    missing = tmp_path / "does_not_exist.json"
    assert main(["--manifest", str(missing)]) == 1
    assert "not found" in capsys.readouterr().err


def test_malformed_json_fails(tmp_path, capsys):
    path = tmp_path / "bad.json"
    path.write_text("{ this is not valid json ")
    assert main(["--manifest", str(path)]) == 1
    assert "not valid JSON" in capsys.readouterr().err


def test_scalar_shape_allowed(tmp_path):
    arg = _tensor_arg(shape=())
    path = _write(tmp_path, _master([_config(arguments={"arg0": arg})]))
    assert main(["--manifest", str(path)]) == 0


def test_negative_shape_dim_fails(tmp_path, capsys):
    arg = _tensor_arg(shape=(1, -32))
    path = _write(tmp_path, _master([_config(arguments={"arg0": arg})]))
    assert main(["--manifest", str(path)]) == 1
    assert "original_shape" in capsys.readouterr().out


def test_inline_values_count_mismatch_fails(tmp_path, capsys):
    arg = _tensor_arg(shape=(2, 2))
    arg["values"] = [[1.0, 2.0], [3.0]]  # only 3 leaves, expected 4
    path = _write(tmp_path, _master([_config(arguments={"arg0": arg})]))
    assert main(["--manifest", str(path)]) == 1
    assert "values" in capsys.readouterr().out


def test_inline_values_count_match_passes(tmp_path):
    arg = _tensor_arg(shape=(2, 2))
    arg["values"] = [[1.0, 2.0], [3.0, 4.0]]
    path = _write(tmp_path, _master([_config(arguments={"arg0": arg})]))
    assert main(["--manifest", str(path)]) == 0


def test_print_resolved_limit(tmp_path, capsys):
    configs = [_config(config_hash=f"hash{i}", config_id=i) for i in range(5)]
    path = _write(tmp_path, _master(configs))
    main(["--manifest", str(path), "--print-resolved", "2"])
    out = capsys.readouterr().out
    assert "[1] ttnn.add" in out
    assert "[2] ttnn.add" in out
    assert "[3] ttnn.add" not in out


def _write_registry(tmp_path, text):
    path = tmp_path / "registry.yaml"
    path.write_text(text)
    return path


def test_registry_pinned_trace_missing_fails(tmp_path, capsys):
    registry = _write_registry(
        tmp_path,
        """
targets:
  lead_models:
    - model: [gpt_oss]
      trace: [999]
registry:
  - trace_id: 1
    status: active
    models: [gpt_oss]
""",
    )
    master = _write(tmp_path, _master())
    assert main(["--manifest", str(master), "--registry", str(registry)]) == 1
    assert "pinned trace 999" in capsys.readouterr().out


def test_registry_invalid_status_fails(tmp_path, capsys):
    registry = _write_registry(
        tmp_path,
        """
targets: {}
registry:
  - trace_id: 1
    status: bogus
    models: [gpt_oss]
""",
    )
    master = _write(tmp_path, _master())
    assert main(["--manifest", str(master), "--registry", str(registry)]) == 1
    assert "invalid 'status'" in capsys.readouterr().out


def test_registry_valid_passes(tmp_path):
    registry = _write_registry(
        tmp_path,
        """
targets:
  lead_models:
    - model: [gpt_oss]
      trace: [1]
registry:
  - trace_id: 1
    status: active
    models: [gpt_oss]
""",
    )
    master = _write(tmp_path, _master())
    assert main(["--manifest", str(master), "--registry", str(registry)]) == 0
