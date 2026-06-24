# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Software-only standalone-import gate for the tt_pipeline port (plan §11.4).

Asserts:
  * ``import ...tt_pipeline as tp; tp.StageDenoise`` succeeds and exposes the public surface.
  * NO ``tt_symbiote`` module is pulled into ``sys.modules`` by importing the port.
  * Static lint over the subtree: NO ``tt_symbiote`` import statements, NO baked-flag
    ``os.environ`` reads (only the ``MESH_DEVICE`` arch-resolver is allowed), NO ``*TP4``
    symbols, NO ``torch.*`` calls inside any ``forward()`` body.
This runs without hardware (it imports the port -- which needs a ttnn build -- but exercises
no device); the static-lint portion is pure filesystem.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

_PKG_DIR = Path(__file__).resolve().parents[2] / "tt" / "tt_pipeline"
_PY_FILES = sorted(_PKG_DIR.rglob("*.py"))


def test_no_tt_symbiote_import_statements():
    """No .py under tt_pipeline/ may import tt_symbiote (comments/docstrings exempt)."""
    offenders = []
    pat = re.compile(r"^\s*(from|import)\s+\S*tt_symbiote")
    for f in _PY_FILES:
        for ln, line in enumerate(f.read_text().splitlines(), 1):
            if pat.match(line):
                offenders.append(f"{f}:{ln}: {line.strip()}")
    assert not offenders, "tt_symbiote import statements found:\n" + "\n".join(offenders)


def test_no_baked_flag_os_environ():
    """Only MESH_DEVICE (the run_on_devices / set_device arch resolver) may read os.environ."""
    offenders = []
    # Match ACTUAL reads (os.environ.get(...) / os.environ[...]) -- not docstring prose.
    pat = re.compile(r"os\.environ\s*(\.get\s*\(|\[)")
    for f in _PY_FILES:
        for ln, line in enumerate(f.read_text().splitlines(), 1):
            if pat.search(line) and not line.lstrip().startswith("#"):
                if "MESH_DEVICE" not in line:
                    offenders.append(f"{f}:{ln}: {line.strip()}")
    assert not offenders, "non-MESH_DEVICE os.environ reads found:\n" + "\n".join(offenders)


def test_no_tp4_symbols():
    offenders = []
    for f in _PY_FILES:
        for ln, line in enumerate(f.read_text().splitlines(), 1):
            if "TP4" in line and not line.lstrip().startswith("#") and "``*TP4``" not in line:
                offenders.append(f"{f}:{ln}: {line.strip()}")
    assert not offenders, "TP4 references found:\n" + "\n".join(offenders)


def test_no_torch_calls_in_forward_bodies():
    """No torch.* calls inside any forward() body (pure-TTNN forward path)."""
    offenders = []
    for f in _PY_FILES:
        tree = ast.parse(f.read_text(), filename=str(f))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "forward":
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name) and sub.value.id == "torch":
                        offenders.append(f"{f}:{sub.lineno}: torch.{sub.attr} in forward()")
    assert not offenders, "torch.* in forward() bodies:\n" + "\n".join(offenders)


def test_import_standalone_no_tt_symbiote_pulled():
    """Importing the port must not pull tt_symbiote into sys.modules."""
    pytest.importorskip("ttnn")
    import models.experimental.pi0_5.tt.tt_pipeline as tp  # noqa: F401

    assert hasattr(tp, "StageDenoise")
    for sym in (
        "build_denoise_loop_pipeline",
        "build_expert_only_pipeline",
        "build_n_stage_pipeline",
        "build_single_stage_reference",
        "TTNNPi05DenoiseStreamedPipeline",
        "TTNNPi05DenoiseExpertBlock",
        "carve_n_submeshes",
        "carve_four_submeshes",
        "euler_schedule",
        "SplitSocketTransport",
    ):
        assert hasattr(tp, sym), f"missing export: {sym}"
    assert not any("TP4" in n for n in dir(tp)), "TP4 symbol leaked into tt_pipeline"
    leaked = [m for m in sys.modules if m == "tt_symbiote" or m.startswith("tt_symbiote.")]
    assert not leaked, f"tt_symbiote modules pulled into sys.modules: {leaked}"


def test_run_expert_chain_signature_parity():
    """run_expert_chain must keep the Galaxy parameter names + order."""
    pytest.importorskip("ttnn")
    import inspect

    import models.experimental.pi0_5.tt.tt_pipeline as tp

    params = list(inspect.signature(tp.StageDenoise.run_expert_chain).parameters)
    assert params[:4] == ["self", "suffix_hidden_chip0", "adarms_conds", "prefix_kv_per_chip"], params
    ctor = list(inspect.signature(tp.StageDenoise.__init__).parameters)
    assert ctor[:4] == ["self", "config", "weights", "mesh_handles"], ctor
    assert "transport" in ctor and ctor.index("transport") == 4, ctor


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
