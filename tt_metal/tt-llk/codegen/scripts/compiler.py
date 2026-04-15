# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
LLK compile checker — explicit template/runtime parameter interface.

Works like any test file (e.g. test_stream_integration.py): the caller
specifies the .cpp test source and the exact template/runtime parameters.
The script creates a TestConfig, calls build_elfs(), and reports pass/fail.

Usage (from codegen/ directory):
    source ../tests/.venv/bin/activate
    CHIP_ARCH=quasar python scripts/compiler.py \
        ../tests/sources/quasar/sfpu_square_quasar_test.cpp \
        -t "MATH_OP(mathop=MathOperation.Square)" \
        -t "APPROX_MODE()" \
        -r "TILE_COUNT(1)" \
        -r "NUM_FACES()" \
        -v

    # With no params (bare compilation, Float16_b format):
    CHIP_ARCH=quasar python scripts/compiler.py \
        ../tests/sources/quasar/sfpu_square_quasar_test.cpp
"""

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — must happen before test-infrastructure imports
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_CODEGEN_DIR = _SCRIPT_DIR.parent
_LLK_ROOT = _CODEGEN_DIR.parent
_TESTS_DIR = _LLK_ROOT / "tests"

sys.path.insert(0, str(_TESTS_DIR / "python_tests"))
sys.path.insert(0, str(_LLK_ROOT))

os.environ.setdefault("LLK_HOME", str(_LLK_ROOT))


# ---------------------------------------------------------------------------
# Build the eval namespace with all symbols agents might use
# ---------------------------------------------------------------------------
def _build_eval_namespace() -> dict:
    """Import all param classes and enums into a flat namespace for eval."""

    return {k: v for k, v in locals().items() if not k.startswith("_")}


# ---------------------------------------------------------------------------
# Test name resolution
# ---------------------------------------------------------------------------
def _resolve_test_name(cpp_path: Path) -> str:
    """Convert absolute .cpp path to the relative name TestConfig expects."""
    tests_dir = _LLK_ROOT / "tests"
    try:
        return str(cpp_path.resolve().relative_to(tests_dir.resolve()))
    except ValueError:
        return str(cpp_path)


# ---------------------------------------------------------------------------
# Error parsing
# ---------------------------------------------------------------------------
def _parse_errors(stderr: str) -> list[str]:
    errors = []
    for line in stderr.splitlines():
        line = line.strip()
        if not line:
            continue
        if re.search(r"\berror:", line):
            errors.append(line)
    return errors if errors else [stderr.split("\n")[0]]


# ---------------------------------------------------------------------------
# Compile
# ---------------------------------------------------------------------------
def _make_dummy_stimuli(data_format):
    """Create a minimal StimuliConfig so RuntimeParams includes buffer fields."""
    import torch
    from helpers.stimuli_config import StimuliConfig

    dummy = torch.zeros(1024, dtype=torch.float32)
    return StimuliConfig(
        buffer_A=dummy,
        stimuli_A_format=data_format,
        buffer_B=dummy,
        stimuli_B_format=data_format,
        stimuli_res_format=data_format,
        tile_count_A=1,
        tile_count_B=1,
        tile_count_res=1,
        num_faces=4,
    )


def compile_check(
    cpp_path: Path,
    templates: list = None,
    runtimes: list = None,
    arch: str = "quasar",
    verbose: bool = False,
) -> tuple[bool, str]:
    """
    Compile a .cpp test source with explicit template/runtime params.

    Returns (success: bool, message: str).
    """
    from helpers.format_config import DataFormat
    from helpers.param_config import input_output_formats
    from helpers.test_config import TestConfig

    cpp_path = Path(cpp_path).resolve()
    if not cpp_path.exists():
        return False, f"File not found: {cpp_path}"

    templates = templates or []
    runtimes = runtimes or []

    test_name = _resolve_test_name(cpp_path)

    if verbose:
        print(f"  Test name:  {test_name}")
        print(f"  Templates:  {[type(t).__name__ for t in templates]}")
        print(f"  Runtimes:   {[type(r).__name__ for r in runtimes]}")

    os.environ.setdefault("CHIP_ARCH", arch)

    artefacts = TestConfig.DEFAULT_ARTEFACTS_PATH
    if artefacts.exists():
        shutil.rmtree(artefacts, ignore_errors=True)

    TestConfig.setup_build(_LLK_ROOT)
    TestConfig.create_build_directories()

    formats = input_output_formats([DataFormat.Float16_b])[0]
    stimuli = _make_dummy_stimuli(DataFormat.Float16_b)

    configuration = TestConfig(
        test_name,
        formats,
        templates=templates,
        runtimes=runtimes,
        variant_stimuli=stimuli,
    )
    configuration.generate_variant_hash()

    try:
        configuration.build_elfs()
        return True, "Compilation successful"
    except RuntimeError as exc:
        stderr_text = str(exc)
        errors = _parse_errors(stderr_text)
        msg = f"{len(errors)} error(s)\n"
        if verbose:
            msg += stderr_text
        else:
            msg += "\n".join(errors[:15])
            if len(errors) > 15:
                msg += f"\n... and {len(errors) - 15} more (use -v for full output)"
        return False, msg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Compile-check an LLK test source with explicit params"
    )
    parser.add_argument(
        "file",
        type=Path,
        help="Path to .cpp test source",
    )
    parser.add_argument(
        "-t",
        "--template",
        action="append",
        default=[],
        dest="templates",
        help=(
            "Template parameter expression, e.g. "
            '"MATH_OP(mathop=MathOperation.Square)". '
            "Can be repeated."
        ),
    )
    parser.add_argument(
        "-r",
        "--runtime",
        action="append",
        default=[],
        dest="runtimes",
        help=(
            'Runtime parameter expression, e.g. "TILE_COUNT(1)". ' "Can be repeated."
        ),
    )
    parser.add_argument(
        "--arch",
        default=os.environ.get("CHIP_ARCH", "quasar"),
        choices=["quasar", "blackhole", "wormhole"],
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    filepath = args.file.resolve()
    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        sys.exit(1)

    # Eval template/runtime expressions in the helpers namespace
    ns = _build_eval_namespace()
    templates = [eval(expr, {"__builtins__": {}}, ns) for expr in args.templates]
    runtimes = [eval(expr, {"__builtins__": {}}, ns) for expr in args.runtimes]

    print(f"Compiling: {filepath.name}")
    print(f"Architecture: {args.arch}")

    success, message = compile_check(
        filepath, templates, runtimes, args.arch, args.verbose
    )

    if success:
        print(f"PASSED — {message}")
        sys.exit(0)
    else:
        print(f"FAILED — {message}")
        sys.exit(1)


if __name__ == "__main__":
    main()
