#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Enforce the Fabric2D-only policy for the production prefill migration."""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOTS = (
    Path("models/demos/common/prefill"),
    Path("models/demos/deepseek_v3_d_p"),
)
PIPELINE_FILES = (
    Path(".github/workflows/blaze-models-prefill-tests-impl.yaml"),
    Path("tests/pipeline_reorg/blackhole_e2e_tests.yaml"),
    Path("tests/pipeline_reorg/blaze_models_prefill_tests.yaml"),
    Path("tests/pipeline_reorg/demo_sp_release_tests.yaml"),
    Path("tests/pipeline_reorg/t3k_e2e_tests.yaml"),
)
PROFILE_CONFIG_ROOTS = (
    Path("models/demos/common/prefill/runners/topology_configuration"),
    Path("models/demos/deepseek_v3_d_p/tt/runners/manifests"),
)

# These are compatibility parsing/mapping points, not test or production profile choices.
ALLOWED_FABRIC1D_REFERENCES = {
    (Path("models/demos/common/prefill/runners/runner_utils.py"), "FABRIC_1D"),
    (Path("models/demos/common/prefill/runners/runner_utils.py"), "FABRIC_1D_RING"),
    (Path("models/demos/common/prefill/topology.py"), "FABRIC_1D_RING"),
}

FORBIDDEN_FABRIC_ENUMS = {"FABRIC_1D", "FABRIC_1D_RING"}
AMBIGUOUS_SELECTOR = re.compile(r"(?<![\w-])(mesh-8x4|linear-8|line|not\s+fabric2d|8x4)(?![\w-])", re.IGNORECASE)
PYTEST_SELECTOR = re.compile(r"\bpytest\b.*(?:\s-k(?:\s+|=))(?P<selector>.*)")
SCOPED_PYTEST_PATH = re.compile(r"models/demos/(?:deepseek_v3_d_p|common/prefill)/")
EXPECTED_COUNT = re.compile(r"\bEXPECT_NUM_TESTS=[1-9][0-9]*\b")
FORBIDDEN_PROFILE_VALUE = re.compile(
    r"\bPREFILL_FABRIC_MODE[\"']?\s*[:=]\s*[\"']?(?P<mode>1d(?:_ring)?)(?![\w-])", re.IGNORECASE
)
LEGACY_TOPOLOGY_ID = re.compile(r"^(?:line|ring|linear-[0-9]+)$", re.IGNORECASE)
TORUS_XY_PROFILE = re.compile(r"(?<![\w])(?:torus[-_]xy|2d_torus_xy)(?![\w])", re.IGNORECASE)
TORUS_XY_DESCRIPTOR = "single_bh_galaxy_torus_xy_graph_descriptor.textproto"
YAML_LITERAL_COMMAND = re.compile(r"^(?:cmd|run):\s*[|>]", re.IGNORECASE)


def _is_fabric_config_reference(node: ast.expr) -> bool:
    return (isinstance(node, ast.Name) and node.id == "FabricConfig") or (
        isinstance(node, ast.Attribute) and node.attr == "FabricConfig"
    )


def _attribute_name(node: ast.Attribute) -> str | None:
    if not _is_fabric_config_reference(node.value):
        return None
    if node.attr not in FORBIDDEN_FABRIC_ENUMS:
        return None
    return node.attr


def audit_python_references() -> list[str]:
    errors = []
    for relative_root in PYTHON_ROOTS:
        for path in sorted((REPO_ROOT / relative_root).rglob("*.py")):
            relative_path = path.relative_to(REPO_ROOT)
            try:
                tree = ast.parse(path.read_text(), filename=str(relative_path))
            except (OSError, SyntaxError) as error:
                errors.append(f"{relative_path}: could not audit Python source: {error}")
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute):
                    enum_name = _attribute_name(node)
                    if enum_name is not None and (relative_path, enum_name) not in ALLOWED_FABRIC1D_REFERENCES:
                        errors.append(
                            f"{relative_path}:{node.lineno}: forbidden FabricConfig.{enum_name}; "
                            "scoped prefill code and tests must use a Fabric2D profile"
                        )
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "getattr"
                    and node.args
                    and _is_fabric_config_reference(node.args[0])
                ):
                    errors.append(
                        f"{relative_path}:{node.lineno}: dynamic getattr on FabricConfig bypasses the scoped "
                        "Fabric2D policy; use an explicit audited enum mapping"
                    )
                if isinstance(node, ast.Call):
                    for keyword in node.keywords:
                        if keyword.arg not in {"id", "ids"}:
                            continue
                        for value in ast.walk(keyword.value):
                            if (
                                isinstance(value, ast.Constant)
                                and isinstance(value.value, str)
                                and LEGACY_TOPOLOGY_ID.fullmatch(value.value)
                            ):
                                errors.append(
                                    f"{relative_path}:{value.lineno}: legacy topology-only pytest id "
                                    f"{value.value!r}; encode the actual Fabric2D/TorusXY profile"
                                )
                if isinstance(node, ast.Assign):
                    if not any(
                        isinstance(target, ast.Name)
                        and (target.id == "SPARSE_MESH_BY_DEVICES" or target.id.endswith("BY_DEVICE_COUNT"))
                        for target in node.targets
                    ):
                        continue
                    for value in ast.walk(node.value):
                        if not isinstance(value, ast.Tuple) or len(value.elts) != 2:
                            continue
                        dims = [element.value for element in value.elts if isinstance(element, ast.Constant)]
                        if len(dims) == 2 and all(isinstance(dim, int) for dim in dims) and 1 in dims:
                            errors.append(
                                f"{relative_path}:{value.lineno}: degenerate communicating mesh {tuple(dims)}; "
                                "use the locked 2x2, 2x4, 4x2, or 8x4 Fabric2D profiles"
                            )
    return errors


def audit_profile_configs() -> list[str]:
    errors = []
    for relative_root in PROFILE_CONFIG_ROOTS:
        root = REPO_ROOT / relative_root
        for path in sorted(
            candidate for candidate in root.rglob("*") if candidate.suffix in {".json", ".yaml", ".yml"}
        ):
            relative_path = path.relative_to(REPO_ROOT)
            try:
                lines = path.read_text().splitlines()
            except OSError as error:
                errors.append(f"{relative_path}: could not audit profile config: {error}")
                continue
            for line_number, line in enumerate(lines, start=1):
                if match := FORBIDDEN_PROFILE_VALUE.search(line):
                    errors.append(
                        f"{relative_path}:{line_number}: forbidden PREFILL_FABRIC_MODE={match.group('mode')}; "
                        "scoped prefill profiles must use Fabric2D or TorusXY"
                    )
    return errors


def _logical_shell_command(lines: list[str], start: int) -> tuple[str, int]:
    parts = [lines[start].rstrip()]
    end = start
    while parts[-1].endswith("\\") and end + 1 < len(lines):
        parts[-1] = parts[-1][:-1]
        end += 1
        parts.append(lines[end].strip())
    return " ".join(parts), end


def _yaml_literal_command_context(lines: list[str], command_index: int) -> str:
    """Return the enclosing YAML ``cmd: |``/``run: |`` body for one shell command."""
    command_indent = len(lines[command_index]) - len(lines[command_index].lstrip())
    for start in range(command_index - 1, -1, -1):
        stripped = lines[start].strip()
        if not stripped:
            continue
        indent = len(lines[start]) - len(lines[start].lstrip())
        if indent >= command_indent or YAML_LITERAL_COMMAND.match(stripped) is None:
            continue
        end = start + 1
        while end < len(lines):
            candidate = lines[end]
            if candidate.strip() and len(candidate) - len(candidate.lstrip()) <= indent:
                break
            end += 1
        return "\n".join(lines[start:end])
    return lines[command_index]


def audit_pipeline_selectors() -> list[str]:
    errors = []
    for relative_path in PIPELINE_FILES:
        path = REPO_ROOT / relative_path
        try:
            lines = path.read_text().splitlines()
        except OSError as error:
            errors.append(f"{relative_path}: could not audit pipeline: {error}")
            continue
        index = 0
        while index < len(lines):
            command, end_index = _logical_shell_command(lines, index)
            line_number = index + 1
            index = end_index + 1
            if (
                command.lstrip().startswith("#")
                or "pytest" not in command
                or SCOPED_PYTEST_PATH.search(command) is None
            ):
                continue
            command_context = " ".join(lines[max(0, line_number - 5) : line_number - 1] + [command])
            if EXPECTED_COUNT.search(command_context) is None:
                errors.append(
                    f"{relative_path}:{line_number}: scoped pytest command has no positive EXPECT_NUM_TESTS; "
                    "lock its exact nonzero execution count"
                )
            if TORUS_XY_PROFILE.search(command) is not None:
                literal_context = _yaml_literal_command_context(lines, index - 1)
                if "TT_MESH_GRAPH_DESC_PATH" not in literal_context or TORUS_XY_DESCRIPTOR not in literal_context:
                    errors.append(
                        f"{relative_path}:{line_number}: TorusXY pytest command has no matching explicit "
                        f"TT_MESH_GRAPH_DESC_PATH={TORUS_XY_DESCRIPTOR}; production TorusXY jobs must fail "
                        "closed before device open"
                    )
            if "-k" not in command:
                continue
            match = PYTEST_SELECTOR.search(command)
            if match is None:
                continue
            ambiguous = AMBIGUOUS_SELECTOR.search(match.group("selector"))
            if ambiguous is not None:
                errors.append(
                    f"{relative_path}:{line_number}: ambiguous scoped pytest selector "
                    f"{ambiguous.group(0)!r}; use an exact Fabric2D or torus-xy profile id"
                )
    return errors


def main() -> int:
    errors = audit_python_references() + audit_profile_configs() + audit_pipeline_selectors()
    if errors:
        print("Prefill fabric policy violations:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print("Prefill fabric policy: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
