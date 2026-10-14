# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Library modules under ttnn/ttnn must not import models or tt_py_test_utils_common."""

import ast
import importlib.util
import sys
import types
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]
_LIBRARY = _REPO_ROOT / "ttnn" / "ttnn"
_FORBIDDEN_ROOTS = ("models", "tt_py_test_utils_common")


def _is_forbidden(name: str) -> bool:
    return name.split(".")[0] in _FORBIDDEN_ROOTS


def _forbidden_import_lines(path: Path) -> list[int]:
    tree = ast.parse(path.read_text())
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and _is_forbidden(node.module):
            lines.append(node.lineno)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if _is_forbidden(alias.name):
                    lines.append(node.lineno)
    return lines


def _load_library_module(filename: str):
    path = _LIBRARY / filename
    spec = importlib.util.spec_from_file_location(f"_cycle_check_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ttnn_library_python_does_not_import_models_or_test_utils():
    offenders = {
        str(path.relative_to(_REPO_ROOT)): _forbidden_import_lines(path) for path in sorted(_LIBRARY.rglob("*.py"))
    }
    offenders = {path: lines for path, lines in offenders.items() if lines}
    assert offenders == {}


def test_comparison_and_config_imports_are_at_module_level():
    decorators = ast.parse((_LIBRARY / "decorators.py").read_text())
    tracer = ast.parse((_LIBRARY / "operation_tracer.py").read_text())

    comparison_imports = [
        node for node in decorators.body if isinstance(node, ast.ImportFrom) and node.module == "ttnn.comparison"
    ]
    assert len(comparison_imports) == 1
    assert {alias.name for alias in comparison_imports[0].names} >= {"comp_pcc", "comp_ulp"}

    serializer_imports = [
        node for node in tracer.body if isinstance(node, ast.ImportFrom) and node.module == "ttnn.config_serialization"
    ]
    assert len(serializer_imports) == 1
    assert {alias.name for alias in serializer_imports[0].names} >= {
        "compute_kernel_config_to_dict",
        "memory_config_to_dict",
        "program_config_to_dict",
    }

    for tree in (decorators, tracer):
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                assert _forbidden_imports_in_node(node) == []


def _forbidden_imports_in_node(node) -> list[int]:
    lines = []
    for child in ast.walk(node):
        if isinstance(child, ast.ImportFrom) and child.module and _is_forbidden(child.module):
            lines.append(child.lineno)
        elif isinstance(child, ast.Import):
            for alias in child.names:
                if _is_forbidden(alias.name):
                    lines.append(child.lineno)
    return lines


def test_comp_pcc_matches_equal_tensors_without_importing_ttnn():
    pytest.importorskip("torch")
    import torch

    comparison = _load_library_module("comparison.py")
    golden = torch.tensor([1.0, 2.0, 3.0])
    matches, score = comparison.comp_pcc(golden, golden.clone())
    assert matches is True
    assert score == 1.0


def test_comp_ulp_matches_equal_torch_tensors(monkeypatch):
    pytest.importorskip("torch")
    import torch

    comparison = _load_library_module("comparison.py")
    if "ttnn" not in sys.modules:
        stub = types.ModuleType("ttnn")
        stub.Tensor = type("Tensor", (), {})
        monkeypatch.setitem(sys.modules, "ttnn", stub)
    golden = torch.zeros(2, 2)
    matches, _message = comparison.comp_ulp(golden, golden.clone(), ulp_threshold=1)
    assert bool(matches) is True


def test_config_serializers_round_trip_plain_objects():
    serializers = _load_library_module("config_serialization.py")

    class _MemoryConfig:
        memory_layout = "INTERLEAVED"
        buffer_type = "DRAM"
        shard_spec = None
        interleaved = True

        def is_sharded(self):
            return False

        def __hash__(self):
            return 7

    encoded = serializers.memory_config_to_dict(_MemoryConfig())
    assert encoded["memory_layout"] == "INTERLEAVED"
    assert encoded["is_sharded"] is False
    assert encoded["hash"] == 7

    class _ProgramConfig:
        def to_json(self):
            return '{"in0_block_w": 2}'

    encoded = serializers.program_config_to_dict(_ProgramConfig())
    assert encoded["type"] == "_ProgramConfig"
    assert encoded["in0_block_w"] == 2


def test_shared_helpers_reexport_the_ttnn_functions():
    for dep in ("torch", "loguru", "ttnn.device"):
        pytest.importorskip(dep)
    from ttnn.comparison import comp_pcc, comp_ulp, ulp
    from ttnn.config_serialization import memory_config_to_dict, program_config_to_dict

    from tt_py_test_utils_common import tensor_utils, utility_functions

    assert utility_functions.comp_pcc is comp_pcc
    assert utility_functions.comp_ulp is comp_ulp
    assert utility_functions.ulp is ulp
    assert tensor_utils.memory_config_to_dict is memory_config_to_dict
    assert tensor_utils.program_config_to_dict is program_config_to_dict
