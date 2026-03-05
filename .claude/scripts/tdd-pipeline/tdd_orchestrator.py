#!/usr/bin/env python3
"""TDD Orchestrator — Stage-gated pipeline for TTNN kernel implementation.

Manages a dynamic sequence of test stages for kernel development. Each stage
is independently testable with a PyTorch reference. Provides automatic failure
classification, retry limits, and rollback to last passing commit.

Usage:
    python3 .claude/scripts/tdd-pipeline/tdd_orchestrator.py init <spec_path> [--op-path PATH]
    python3 .claude/scripts/tdd-pipeline/tdd_orchestrator.py add-stage '<json>' [--op-path PATH]
    python3 .claude/scripts/tdd-pipeline/tdd_orchestrator.py test [stage_name] [--op-path PATH]
    python3 .claude/scripts/tdd-pipeline/tdd_orchestrator.py advance [--op-path PATH]
    python3 .claude/scripts/tdd-pipeline/tdd_orchestrator.py rollback [--op-path PATH]
    python3 .claude/scripts/tdd-pipeline/tdd_orchestrator.py status [--op-path PATH] [--json]
    python3 .claude/scripts/tdd-pipeline/tdd_orchestrator.py parse-failure [--op-path PATH]

Called by: Orchestrator agent (via .claude/references/tdd-kernel-pipeline.md)
Calls: tt-test.sh, failure_parser.py
State: {op_path}/.tdd_state.json
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Resolve paths relative to this script
SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent.parent.parent  # .claude/scripts/tdd-pipeline -> repo root
TT_TEST_SCRIPT = REPO_ROOT / "scripts" / "tt-test.sh"
TEMPLATE_PATH = SCRIPT_DIR / "test_stage_template.py.j2"

# State file name (lives in the operation directory)
STATE_FILENAME = ".tdd_state.json"
GATE_MARKER = ".tdd_gate_passed"

# Stage statuses
STATUS_PENDING = "pending"
STATUS_IN_PROGRESS = "in_progress"
STATUS_PASSED = "passed"
STATUS_FAILED_PERMANENT = "failed_permanent"

# Default max attempts per stage (only "hard" failures consume attempts)
DEFAULT_MAX_ATTEMPTS = 6

# Max free retries (safety cap to prevent infinite loops on repeated easy failures)
MAX_FREE_RETRIES = 10

# Failure classifications that do NOT consume an attempt ("easy" fixes)
FREE_CLASSIFICATIONS = {"compilation_error", "shape_mismatch"}

# Default test layout
DEFAULT_LAYOUT = "ROW_MAJOR_LAYOUT"

# Last test output capture file
TEST_OUTPUT_FILE = "/tmp/tdd_last_test_output.txt"


# ---------------------------------------------------------------------------
# State File I/O
# ---------------------------------------------------------------------------


def _resolve_op_path(args) -> Path:
    """Resolve the operation path from args or the current state."""
    if hasattr(args, "op_path") and args.op_path:
        return Path(args.op_path)
    # Try current directory
    if (Path.cwd() / STATE_FILENAME).exists():
        return Path.cwd()
    print(f"ERROR: No {STATE_FILENAME} found. Specify --op-path or run from the operation directory.", file=sys.stderr)
    sys.exit(1)


def _state_path(op_path: Path) -> Path:
    return op_path / STATE_FILENAME


def _gate_path(op_path: Path) -> Path:
    return op_path / GATE_MARKER


def _load_state(op_path: Path) -> dict:
    sp = _state_path(op_path)
    if not sp.exists():
        print(f"ERROR: State file not found: {sp}", file=sys.stderr)
        sys.exit(1)
    with open(sp) as f:
        return json.load(f)


def _save_state(op_path: Path, state: dict) -> None:
    sp = _state_path(op_path)
    with open(sp, "w") as f:
        json.dump(state, f, indent=2)
        f.write("\n")


# ---------------------------------------------------------------------------
# Template Rendering
# ---------------------------------------------------------------------------


def _render_template(state: dict, stage: dict, op_path: Path) -> str:
    """Render the Jinja2 test template for a stage.

    Uses simple string replacement instead of requiring jinja2 as a dependency.
    Falls back to jinja2 if available.
    """
    try:
        from jinja2 import Environment, FileSystemLoader

        env = Environment(
            loader=FileSystemLoader(str(SCRIPT_DIR)),
            keep_trailing_newline=True,
        )
        template = env.get_template("test_stage_template.py.j2")
        return template.render(
            stage_name=stage["name"],
            stage_description=stage["description"],
            op_name=state["op_name"],
            extra_imports=stage.get("extra_imports", ""),
            reference_body=stage["reference_body"],
            extra_ref_params=_build_extra_ref_params(stage),
            shapes=stage["shapes"],
            extra_setup=stage.get("extra_setup", ""),
            extra_ref_args=_build_extra_ref_args(stage),
            layout=state.get("layout", DEFAULT_LAYOUT),
            extra_ttnn_setup=stage.get("extra_ttnn_setup", ""),
            extra_args=stage.get("extra_args", ""),
            output_shape_expr=stage.get("output_shape_expr", ""),
            dtype_parametrize=stage.get("dtype_parametrize", ""),
            tolerance_rtol=stage["tolerance"]["rtol"],
            tolerance_atol=stage["tolerance"]["atol"],
        )
    except ImportError:
        return _render_template_simple(state, stage)


def _render_template_simple(state: dict, stage: dict) -> str:
    """Render template using simple string replacement (no jinja2 dependency)."""
    with open(TEMPLATE_PATH) as f:
        template = f.read()

    op_name = state["op_name"]
    layout = state.get("layout", DEFAULT_LAYOUT)
    tolerance = stage["tolerance"]

    # Build parametrize entries
    shape_entries = []
    for shape in stage["shapes"]:
        shape_id = shape.replace("(", "").replace(")", "").replace(", ", "x")
        shape_entries.append(f'        pytest.param({shape}, id="{shape_id}"),')
    shapes_block = "\n".join(shape_entries)

    # Build output shape check
    output_shape_expr = stage.get("output_shape_expr", "")
    if output_shape_expr:
        shape_check = f"    expected_shape = {output_shape_expr}"
    else:
        shape_check = "    expected_shape = list(shape)"

    # Simple replacements
    result = f'''# Auto-generated by tdd_orchestrator.py — Stage: {stage["name"]}
# DO NOT EDIT — regenerate with: tdd_orchestrator.py add-stage ...
"""
TDD Stage: {stage["name"]}
{stage["description"]}
"""

import pytest
import torch
import ttnn
{stage.get("extra_imports", "")}

from .{op_name} import {op_name}


def pytorch_reference(input_tensor{_build_extra_ref_params(stage)}):
    """PyTorch reference for this stage."""
    {stage["reference_body"]}


@pytest.mark.parametrize(
    "shape",
    [
{shapes_block}
    ],
)
def test_{stage["name"]}(device, shape):
    """Verify {stage["description"]}."""
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    {stage.get("extra_setup", "")}

    expected = pytorch_reference(torch_input{_build_extra_ref_args(stage)})

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.{layout},
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    {stage.get("extra_ttnn_setup", "")}

    ttnn_output = {op_name}(ttnn_input{stage.get("extra_args", "")})

    # Shape check
{shape_check}
    assert list(ttnn_output.shape) == expected_shape, \\
        f"Shape: {{list(ttnn_output.shape)}} vs expected {{expected_shape}}"

    # Numerical comparison
    torch_output = ttnn.to_torch(ttnn_output)
    assert torch.allclose(
        torch_output.float(),
        expected.float(),
        rtol={tolerance["rtol"]},
        atol={tolerance["atol"]},
    ), f"Max diff: {{(torch_output.float() - expected.float()).abs().max()}}"
'''
    return result


def _build_extra_ref_params(stage: dict) -> str:
    """Build extra parameter names for the reference function signature."""
    extra_args = stage.get("extra_args", "")
    if not extra_args:
        return ""
    # Extract parameter names from ", gamma, beta" style args
    # Keep as-is since it goes directly into the function signature
    return extra_args


def _build_extra_ref_args(stage: dict) -> str:
    """Build extra arguments for the reference function call."""
    extra_args = stage.get("extra_args", "")
    if not extra_args:
        return ""
    return extra_args


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_init(args):
    """Initialize TDD pipeline from a spec file.

    Creates an empty .tdd_state.json with no stages. The agent must
    register stages via add-stage.
    """
    spec_path = Path(args.spec_path)
    if not spec_path.exists():
        print(f"ERROR: Spec file not found: {spec_path}", file=sys.stderr)
        sys.exit(1)

    # Determine op path
    if args.op_path:
        op_path = Path(args.op_path)
    else:
        # Infer from spec path (spec is usually in the op directory)
        op_path = spec_path.parent

    # Extract op_name from the directory name
    op_name = op_path.name

    # Parse spec for layout hint
    layout = DEFAULT_LAYOUT
    with open(spec_path) as f:
        spec_content = f.read()
    if "TILE_LAYOUT" in spec_content or "tile_layout" in spec_content.lower():
        layout = "TILE_LAYOUT"

    state = {
        "op_name": op_name,
        "op_path": str(op_path),
        "spec_path": str(spec_path),
        "layout": layout,
        "current_stage_index": 0,
        "last_passing_commit": None,
        "stages": [],
    }

    os.makedirs(op_path, exist_ok=True)
    _save_state(op_path, state)
    print(f"TDD pipeline initialized for '{op_name}'.")
    print(f"State: {_state_path(op_path)}")
    print("Use `add-stage` to register stages.")


def cmd_add_stage(args):
    """Register a new stage and render its test file."""
    op_path = _resolve_op_path(args)
    state = _load_state(op_path)

    # Parse stage JSON
    if args.from_file:
        with open(args.from_file) as f:
            stage_def = json.load(f)
    else:
        try:
            stage_def = json.loads(args.stage_json)
        except json.JSONDecodeError as e:
            print(f"ERROR: Invalid JSON: {e}", file=sys.stderr)
            sys.exit(1)

    # Validate required fields
    required = ["name", "description", "reference_body", "tolerance", "shapes"]
    missing = [f for f in required if f not in stage_def]
    if missing:
        print(f"ERROR: Missing required fields: {missing}", file=sys.stderr)
        sys.exit(1)

    # Validate name is snake_case
    if not re.match(r"^[a-z][a-z0-9_]*$", stage_def["name"]):
        print(f"ERROR: Stage name must be snake_case: '{stage_def['name']}'", file=sys.stderr)
        sys.exit(1)

    # Check for duplicate names
    existing_names = [s["name"] for s in state["stages"]]
    if stage_def["name"] in existing_names:
        print(f"ERROR: Stage '{stage_def['name']}' already exists.", file=sys.stderr)
        sys.exit(1)

    # Validate shapes — minimum 3 for meaningful coverage
    shapes = stage_def["shapes"]
    if not isinstance(shapes, list) or len(shapes) < 3:
        print(
            f"ERROR: shapes must be a list with at least 3 entries (got {len(shapes) if isinstance(shapes, list) else type(shapes).__name__}). "
            "Include at minimum: single-tile, multi-tile, and multi-batch shapes.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Validate tolerance
    tolerance = stage_def["tolerance"]
    if not isinstance(tolerance, dict) or "rtol" not in tolerance or "atol" not in tolerance:
        print('ERROR: tolerance must be {"rtol": float, "atol": float}', file=sys.stderr)
        sys.exit(1)

    # Build stage entry
    test_filename = f"test_stage_{stage_def['name']}.py"
    stage = {
        "name": stage_def["name"],
        "description": stage_def["description"],
        "kernel_files": stage_def.get("kernel_files", []),
        "test_file": test_filename,
        "reference_body": stage_def["reference_body"],
        "tolerance": tolerance,
        "shapes": stage_def["shapes"],
        "extra_imports": stage_def.get("extra_imports", ""),
        "extra_args": stage_def.get("extra_args", ""),
        "extra_setup": stage_def.get("extra_setup", ""),
        "extra_ttnn_setup": stage_def.get("extra_ttnn_setup", ""),
        "output_shape_expr": stage_def.get("output_shape_expr", ""),
        "dtype_parametrize": stage_def.get("dtype_parametrize", ""),
        "status": STATUS_PENDING,
        "commit": None,
        "attempts": 0,
        "free_retries": 0,
        "max_attempts": stage_def.get("max_attempts", DEFAULT_MAX_ATTEMPTS),
        "failure_history": [],
    }

    # Render test file — write to tests/ttnn/unit_tests/operations/{op_name}/
    test_content = _render_template(state, stage, op_path)
    test_dir = REPO_ROOT / "tests" / "ttnn" / "unit_tests" / "operations" / state["op_name"]
    os.makedirs(test_dir, exist_ok=True)
    test_path = test_dir / test_filename
    with open(test_path, "w") as f:
        f.write(test_content)

    # Append stage to state
    state["stages"].append(stage)
    _save_state(op_path, state)

    print(f"Stage '{stage_def['name']}' registered. Test: {test_path}")


def cmd_test(args):
    """Run the test for a stage via tt-test.sh --dev."""
    op_path = _resolve_op_path(args)
    state = _load_state(op_path)

    # Determine which stage to test
    if args.stage_name:
        stage_idx = None
        for i, s in enumerate(state["stages"]):
            if s["name"] == args.stage_name:
                stage_idx = i
                break
        if stage_idx is None:
            print(f"ERROR: Stage '{args.stage_name}' not found.", file=sys.stderr)
            sys.exit(1)
    else:
        stage_idx = state["current_stage_index"]
        if stage_idx >= len(state["stages"]):
            print("ERROR: All stages completed. No stage to test.", file=sys.stderr)
            sys.exit(1)

    stage = state["stages"][stage_idx]

    # Allow testing current or already-passed stages
    if stage_idx > state["current_stage_index"]:
        print(
            f"ERROR: Cannot test stage '{stage['name']}' — not yet reached (current: {state['current_stage_index']}).",
            file=sys.stderr,
        )
        sys.exit(1)

    # Set stage to in_progress if it's the current stage
    if stage_idx == state["current_stage_index"] and stage["status"] == STATUS_PENDING:
        stage["status"] = STATUS_IN_PROGRESS

    # Build test path — tests live at tests/ttnn/unit_tests/operations/{op_name}/
    test_dir = REPO_ROOT / "tests" / "ttnn" / "unit_tests" / "operations" / state["op_name"]
    test_path = test_dir / stage["test_file"]
    if not test_path.exists():
        # Fallback to legacy colocated path for backward compatibility
        test_path = op_path / stage["test_file"]
    if not test_path.exists():
        print(f"ERROR: Test file not found: {test_path}", file=sys.stderr)
        sys.exit(1)

    # Run tt-test.sh --dev
    hard_attempts = stage["attempts"]
    free_retries = stage.get("free_retries", 0)
    print(
        f"TDD_TEST: Running stage '{stage['name']}' "
        f"(hard attempts: {hard_attempts}/{stage['max_attempts']}, "
        f"free retries: {free_retries})..."
    )
    result = subprocess.run(
        [str(TT_TEST_SCRIPT), "--dev", str(test_path)],
        capture_output=True,
        text=True,
        timeout=300,  # 5 minute overall timeout
    )

    exit_code = result.returncode
    combined_output = result.stdout + "\n" + result.stderr

    # Save output for parse-failure
    with open(TEST_OUTPUT_FILE, "w") as f:
        f.write(combined_output)

    # Print the output through
    print(combined_output)

    if exit_code == 0:
        # PASS
        print(f"\nTDD_GATE: PASS — stage '{stage['name']}'")
        # Create gate marker
        _gate_path(op_path).touch()
        _save_state(op_path, state)
        sys.exit(0)
    else:
        # FAIL or HANG — classify before deciding cost
        from failure_parser import classify_failure

        failure_info = classify_failure(
            exit_code=exit_code,
            pytest_output=combined_output,
        )

        classification = failure_info.get("classification", "unknown")
        is_free = classification in FREE_CLASSIFICATIONS

        if is_free:
            # Easy failure — don't consume a hard attempt
            stage["free_retries"] = stage.get("free_retries", 0) + 1
            free_retries = stage["free_retries"]
            budget_exhausted = free_retries >= MAX_FREE_RETRIES
            cost_label = "FREE"
        else:
            # Hard failure — consume an attempt
            stage["attempts"] += 1
            budget_exhausted = stage["attempts"] >= stage["max_attempts"]
            cost_label = "HARD"

        failure_info["attempt"] = stage["attempts"]
        failure_info["free_retries"] = stage.get("free_retries", 0)
        failure_info["cost"] = cost_label
        failure_info["stage"] = stage["name"]
        failure_info["remaining_attempts"] = stage["max_attempts"] - stage["attempts"]
        failure_info["remaining_free_retries"] = MAX_FREE_RETRIES - stage.get("free_retries", 0)
        failure_info["budget_exhausted"] = budget_exhausted

        # Append to failure history
        stage["failure_history"].append(failure_info)

        # Remove gate marker
        gp = _gate_path(op_path)
        if gp.exists():
            gp.unlink()

        _save_state(op_path, state)

        kind = "HANG" if exit_code == 2 else "FAIL"
        remaining_hard = stage["max_attempts"] - stage["attempts"]
        remaining_free = MAX_FREE_RETRIES - stage.get("free_retries", 0)
        print(f"\nTDD_GATE: {kind} [{cost_label}] — stage '{stage['name']}'")
        print(f"  Classification: {classification}")
        print(f"  Summary: {failure_info['summary']}")
        print(f"  Hard attempts: {stage['attempts']}/{stage['max_attempts']} ({remaining_hard} remaining)")
        print(f"  Free retries: {stage.get('free_retries', 0)}/{MAX_FREE_RETRIES} ({remaining_free} remaining)")
        if budget_exhausted:
            print(f"  BUDGET EXHAUSTED — run `rollback` to restore kernels")

        sys.exit(exit_code)


def cmd_advance(args):
    """Advance to the next stage after the current one passes."""
    op_path = _resolve_op_path(args)
    state = _load_state(op_path)

    # Verify gate marker exists
    if not _gate_path(op_path).exists():
        print("ERROR: TDD gate not passed. Run `test` first and ensure it passes.", file=sys.stderr)
        sys.exit(1)

    idx = state["current_stage_index"]
    if idx >= len(state["stages"]):
        print("All stages already completed.", file=sys.stderr)
        sys.exit(0)

    stage = state["stages"][idx]

    # Record commit
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=str(REPO_ROOT)).strip()
    except subprocess.CalledProcessError:
        commit = "unknown"

    stage["status"] = STATUS_PASSED
    stage["commit"] = commit
    state["last_passing_commit"] = commit

    # Move to next stage
    state["current_stage_index"] = idx + 1

    # Set next stage to in_progress if it exists
    if idx + 1 < len(state["stages"]):
        state["stages"][idx + 1]["status"] = STATUS_IN_PROGRESS

    # Remove gate marker (next stage needs its own pass)
    gp = _gate_path(op_path)
    if gp.exists():
        gp.unlink()

    _save_state(op_path, state)

    if idx + 1 >= len(state["stages"]):
        print(f"TDD_PIPELINE: COMPLETE — all {len(state['stages'])} stages passed")
    else:
        next_stage = state["stages"][idx + 1]
        print(f"Stage '{stage['name']}' passed (commit: {commit[:8]})")
        print(f"Advanced to stage [{idx + 1}] '{next_stage['name']}'")


def cmd_rollback(args):
    """Rollback kernel files to last passing commit."""
    op_path = _resolve_op_path(args)
    state = _load_state(op_path)

    idx = state["current_stage_index"]
    if idx >= len(state["stages"]):
        print("ERROR: No stage to rollback — all stages completed.", file=sys.stderr)
        sys.exit(1)

    stage = state["stages"][idx]
    last_commit = state.get("last_passing_commit")

    # Rollback kernel files if we have a passing commit
    if last_commit:
        kernels_path = os.path.join(state["op_path"], "kernels")
        print(f"Rolling back {kernels_path} to commit {last_commit[:8]}...")
        try:
            subprocess.run(
                ["git", "checkout", last_commit, "--", kernels_path],
                check=True,
                cwd=str(REPO_ROOT),
            )
            print(f"Kernel files restored to {last_commit[:8]}")
        except subprocess.CalledProcessError as e:
            print(f"WARNING: git checkout failed: {e}", file=sys.stderr)
    else:
        print("No passing commit to rollback to (stage 0 failed).")

    # Mark stage as permanently failed
    stage["status"] = STATUS_FAILED_PERMANENT

    # Remove gate marker
    gp = _gate_path(op_path)
    if gp.exists():
        gp.unlink()

    # Generate failure report
    report_path = op_path / "tdd_failure_report.md"
    _generate_failure_report(state, stage, report_path)

    _save_state(op_path, state)

    print(f"\nHUMAN REVIEW REQUIRED: stage '{stage['name']}' failed after {stage['attempts']} attempts")
    print(f"Failure report: {report_path}")
    sys.exit(1)


def _generate_failure_report(state: dict, stage: dict, report_path: Path) -> None:
    """Generate a markdown failure report for human review."""
    lines = [
        f"# TDD Failure Report: {state['op_name']}",
        "",
        f"## Stage: {stage['name']}",
        f"**Description:** {stage['description']}",
        f"**Attempts:** {stage['attempts']}/{stage['max_attempts']}",
        f"**Status:** {stage['status']}",
        "",
        "## Failure History",
        "",
    ]

    for i, failure in enumerate(stage.get("failure_history", []), 1):
        lines.extend(
            [
                f"### Attempt {i}",
                f"- **Classification:** `{failure.get('classification', 'unknown')}`",
                f"- **Summary:** {failure.get('summary', 'N/A')}",
                f"- **Suggested Action:** {failure.get('suggested_action', 'N/A')}",
            ]
        )
        details = failure.get("details", {})
        if details:
            lines.append(f"- **Details:** `{json.dumps(details)}`")
        lines.append("")

    lines.extend(
        [
            "## Suggested Next Steps",
            "",
            "1. Review the failure classifications above for patterns",
            "2. Check the kernel design document for correctness",
            "3. If all attempts show the same classification, the design may need revision",
            "4. Run `tdd_orchestrator.py status --json` to see full pipeline state",
            f"5. Full triage log (if hang): `/tmp/tt-test-triage-dev0.log`",
            "",
            f"## Pipeline State",
            f"- **Last passing commit:** `{state.get('last_passing_commit', 'None')}`",
            f"- **Current stage index:** {state['current_stage_index']}",
            f"- **Total stages:** {len(state['stages'])}",
        ]
    )

    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def cmd_status(args):
    """Print pipeline status summary."""
    op_path = _resolve_op_path(args)
    state = _load_state(op_path)

    if hasattr(args, "json_output") and args.json_output:
        print(json.dumps(state, indent=2))
        return

    total = len(state["stages"])
    passed = sum(1 for s in state["stages"] if s["status"] == STATUS_PASSED)
    in_progress = sum(1 for s in state["stages"] if s["status"] == STATUS_IN_PROGRESS)
    pending = sum(1 for s in state["stages"] if s["status"] == STATUS_PENDING)
    failed = sum(1 for s in state["stages"] if s["status"] == STATUS_FAILED_PERMANENT)

    idx = state["current_stage_index"]

    print(f"TDD Pipeline: {state['op_name']}")
    print(f"Stages: {total} total, {passed} passed, {in_progress} in_progress, {pending} pending, {failed} failed")

    if idx < total:
        current = state["stages"][idx]
        free = current.get("free_retries", 0)
        print(
            f"Current: [{idx}] {current['name']} "
            f"(hard: {current['attempts']}/{current['max_attempts']}, "
            f"free: {free}/{MAX_FREE_RETRIES})"
        )
    else:
        print("Current: COMPLETE — all stages passed")

    last_commit = state.get("last_passing_commit")
    print(f"Last passing commit: {last_commit[:8] if last_commit else 'None'}")
    print(f"Retry policy: easy fails (compile/shape) are FREE, others cost 1 hard attempt")

    # Show per-stage summary
    if total > 0:
        print("\nStages:")
        for i, s in enumerate(state["stages"]):
            marker = ">" if i == idx else " "
            status_icon = {
                STATUS_PASSED: "PASS",
                STATUS_IN_PROGRESS: "WORK",
                STATUS_PENDING: "    ",
                STATUS_FAILED_PERMANENT: "FAIL",
            }.get(s["status"], "????")
            commit_str = f" [{s['commit'][:8]}]" if s.get("commit") else ""
            free = s.get("free_retries", 0)
            retry_str = ""
            if s["attempts"] > 0 or free > 0:
                retry_str = f" (hard:{s['attempts']}, free:{free})"
            print(f"  {marker} [{i}] [{status_icon}] {s['name']}{commit_str}{retry_str}")


def cmd_parse_failure(args):
    """Parse the last test failure and output structured JSON."""
    op_path = _resolve_op_path(args)
    state = _load_state(op_path)

    idx = state["current_stage_index"]
    if idx >= len(state["stages"]):
        print("ERROR: No active stage.", file=sys.stderr)
        sys.exit(1)

    stage = state["stages"][idx]

    # Use the most recent failure from history
    if stage["failure_history"]:
        failure = stage["failure_history"][-1]
        # Ensure it has the extra fields
        failure["stage"] = stage["name"]
        failure["attempt"] = stage["attempts"]
        failure["remaining_attempts"] = stage["max_attempts"] - stage["attempts"]
        print(json.dumps(failure, indent=2))
    else:
        # Try to parse from saved output
        pytest_output = ""
        if os.path.isfile(TEST_OUTPUT_FILE):
            with open(TEST_OUTPUT_FILE) as f:
                pytest_output = f.read()

        if not pytest_output:
            print("ERROR: No test output found. Run `test` first.", file=sys.stderr)
            sys.exit(1)

        from failure_parser import classify_failure

        failure = classify_failure(exit_code=1, pytest_output=pytest_output)
        failure["stage"] = stage["name"]
        failure["attempt"] = stage["attempts"]
        failure["remaining_attempts"] = stage["max_attempts"] - stage["attempts"]
        print(json.dumps(failure, indent=2))


# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="TDD Orchestrator for TTNN kernel implementation",
        prog="tdd_orchestrator.py",
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommand")

    # init
    p_init = subparsers.add_parser("init", help="Initialize TDD pipeline from spec")
    p_init.add_argument("spec_path", help="Path to the operation spec file")
    p_init.add_argument("--op-path", help="Operation directory (default: spec parent dir)")

    # add-stage
    p_add = subparsers.add_parser("add-stage", help="Register a new stage")
    p_add.add_argument("stage_json", nargs="?", help="Stage definition as JSON string")
    p_add.add_argument("--from-file", help="Read stage definition from a JSON file")
    p_add.add_argument("--op-path", help="Operation directory")

    # test
    p_test = subparsers.add_parser("test", help="Run test for a stage")
    p_test.add_argument("stage_name", nargs="?", help="Stage name (default: current)")
    p_test.add_argument("--op-path", help="Operation directory")

    # advance
    p_advance = subparsers.add_parser("advance", help="Advance to next stage")
    p_advance.add_argument("--op-path", help="Operation directory")

    # rollback
    p_rollback = subparsers.add_parser("rollback", help="Rollback to last passing commit")
    p_rollback.add_argument("--op-path", help="Operation directory")

    # status
    p_status = subparsers.add_parser("status", help="Show pipeline status")
    p_status.add_argument("--op-path", help="Operation directory")
    p_status.add_argument("--json", dest="json_output", action="store_true", help="Output raw JSON")

    # parse-failure
    p_parse = subparsers.add_parser("parse-failure", help="Parse last test failure")
    p_parse.add_argument("--op-path", help="Operation directory")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    commands = {
        "init": cmd_init,
        "add-stage": cmd_add_stage,
        "test": cmd_test,
        "advance": cmd_advance,
        "rollback": cmd_rollback,
        "status": cmd_status,
        "parse-failure": cmd_parse_failure,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
