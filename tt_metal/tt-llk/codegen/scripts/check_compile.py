#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Check LLK kernel compilation.

Usage:
    # Check a specific file (from codegen/ directory)
    PYTHONPATH=.. python scripts/check_compile.py ../tt_llk_quasar/common/inc/sfpu/ckernel_sfpu_sigmoid.h

    # Check with custom function names
    PYTHONPATH=.. python scripts/check_compile.py my_kernel.h --func _calculate_foo_ --init _init_foo_
"""

import argparse
import sys
from pathlib import Path

# Add scripts directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))
# Add tt-llk root to path for codegen.config
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from compiler import CompileAgent, CompileResult


def detect_kernel_type(filepath: Path) -> str:
    """Detect kernel type from file path pattern."""
    path_str = str(filepath)
    if "/sfpu/" in path_str or "ckernel_sfpu_" in filepath.name:
        return "sfpu"
    elif "llk_math_" in filepath.name:
        return "math"
    elif "llk_pack_" in filepath.name:
        return "pack"
    elif "llk_unpack_" in filepath.name:
        return "unpack"
    return "sfpu"  # default


def create_wrapper(
    filename: str, func_name: str, init_name: str | None, kernel_type: str = "sfpu"
) -> str:
    """Create a test wrapper for the given function."""
    if kernel_type == "sfpu":
        return _create_sfpu_wrapper(filename, func_name, init_name)
    elif kernel_type == "math":
        return _create_math_wrapper(filename, func_name, init_name)
    elif kernel_type == "pack":
        return _create_pack_wrapper(filename, func_name, init_name)
    elif kernel_type == "unpack":
        return _create_unpack_wrapper(filename, func_name, init_name)
    return _create_sfpu_wrapper(filename, func_name, init_name)


def _create_sfpu_wrapper(filename: str, func_name: str, init_name: str | None) -> str:
    """Create SFPU kernel test wrapper."""
    init_calls = ""
    if init_name:
        init_calls = f"""
    void force_compile_init() {{
        {init_name}<true>();
        {init_name}<false>();
    }}"""

    return f"""// Auto-generated compile test wrapper
#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "{filename}"

using namespace ckernel;
using namespace ckernel::sfpu;

namespace {{
    void force_compile() {{
        {func_name}<true>(16);
        {func_name}<false>(16);
    }}
{init_calls}
}}
"""


def _create_math_wrapper(filename: str, func_name: str, init_name: str | None) -> str:
    """Create math kernel test wrapper."""
    init_calls = ""
    if init_name:
        init_calls = f"""
    void force_compile_init() {{
        {init_name}();
    }}"""

    return f"""// Auto-generated compile test wrapper
#include "ckernel_trisc_common.h"
#include "llk_math_common.h"
#include "{filename}"

using namespace ckernel;

namespace {{
    void force_compile() {{
        {func_name}();
    }}
{init_calls}
}}
"""


def _create_pack_wrapper(filename: str, func_name: str, init_name: str | None) -> str:
    """Create pack kernel test wrapper."""
    init_calls = ""
    if init_name:
        init_calls = f"""
    void force_compile_init() {{
        {init_name}();
    }}"""

    return f"""// Auto-generated compile test wrapper
#include "ckernel_trisc_common.h"
#include "llk_pack_common.h"
#include "{filename}"

using namespace ckernel;

namespace {{
    void force_compile() {{
        {func_name}();
    }}
{init_calls}
}}
"""


def _create_unpack_wrapper(filename: str, func_name: str, init_name: str | None) -> str:
    """Create unpack kernel test wrapper."""
    init_calls = ""
    if init_name:
        init_calls = f"""
    void force_compile_init() {{
        {init_name}();
    }}"""

    return f"""// Auto-generated compile test wrapper
#include "ckernel_trisc_common.h"
#include "llk_unpack_common.h"
#include "{filename}"

using namespace ckernel;

namespace {{
    void force_compile() {{
        {func_name}();
    }}
{init_calls}
}}
"""


def check_file(
    filepath: Path,
    arch: str = "quasar",
    func_name: str | None = None,
    init_name: str | None = None,
) -> CompileResult:
    """Check that an LLK kernel file compiles."""
    agent = CompileAgent(arch=arch)

    # Read the file
    code = filepath.read_text()

    # Detect kernel type
    kernel_type = detect_kernel_type(filepath)

    # Infer function names from filename if not provided
    if func_name is None:
        stem = filepath.stem
        if kernel_type == "sfpu":
            # ckernel_sfpu_sigmoid.h -> _calculate_sigmoid_
            op_name = stem.replace("ckernel_sfpu_", "")
            func_name = f"_calculate_{op_name}_"
            init_name = f"_init_{op_name}_"
        elif kernel_type == "math":
            # llk_math_reduce.h -> _llk_math_reduce_
            op_name = stem.replace("llk_math_", "")
            func_name = f"_llk_math_{op_name}_"
            init_name = f"_llk_math_{op_name}_init_"
        elif kernel_type == "pack":
            # llk_pack_untilize.h -> _llk_pack_untilize_
            op_name = stem.replace("llk_pack_", "")
            func_name = f"_llk_pack_{op_name}_"
            init_name = f"_llk_pack_{op_name}_init_"
        elif kernel_type == "unpack":
            # llk_unpack_tilize.h -> _llk_unpack_tilize_
            op_name = stem.replace("llk_unpack_", "")
            func_name = f"_llk_unpack_{op_name}_"
            init_name = f"_llk_unpack_{op_name}_init_"

    # Override the wrapper creation
    original_wrapper = agent._create_wrapper
    agent._create_wrapper = lambda code, filename, op_name=None: create_wrapper(
        filename, func_name, init_name, kernel_type
    )

    # Compile
    result = agent.compile(code, filepath.name)

    return result


def main():
    parser = argparse.ArgumentParser(description="Check LLK kernel compilation")
    parser.add_argument("file", type=Path, help="Path to LLK header file")
    parser.add_argument(
        "--arch", default="quasar", choices=["quasar", "blackhole", "wormhole"]
    )
    parser.add_argument("--func", help="Main function name (e.g., _calculate_sigmoid_)")
    parser.add_argument("--init", help="Init function name (e.g., _init_sigmoid_)")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show full error output"
    )
    args = parser.parse_args()

    if not args.file.exists():
        print(f"Error: File not found: {args.file}")
        sys.exit(1)

    # Warn if file path doesn't match specified architecture
    arch_dir_map = {
        "quasar": "tt_llk_quasar",
        "blackhole": "tt_llk_blackhole",
        "wormhole": "tt_llk_wormhole_b0",
    }
    expected_dir = arch_dir_map.get(args.arch, "")
    file_str = str(args.file.resolve())
    for other_arch, other_dir in arch_dir_map.items():
        if other_dir in file_str and other_arch != args.arch:
            print(
                f"Warning: File is in {other_dir}/ but compiling for --arch {args.arch}"
            )
            print(f"  Did you mean --arch {other_arch}?")
            break

    print(f"Checking: {args.file}")
    print(f"Architecture: {args.arch}")

    result = check_file(args.file, args.arch, args.func, args.init)

    if result.success:
        print("✓ Compilation successful!")
        sys.exit(0)
    else:
        print(f"✗ Compilation failed ({len(result.errors)} errors)")
        print()

        if args.verbose:
            print("Full compiler output:")
            print("-" * 60)
            print(result.stderr)
        else:
            print("Errors:")
            for err in result.errors[:10]:
                if err.line:
                    print(f"  Line {err.line}: {err.message}")
                else:
                    print(f"  {err.message}")

            if len(result.errors) > 10:
                print(f"  ... and {len(result.errors) - 10} more")
                print()
                print("Use --verbose for full output")

        sys.exit(1)


if __name__ == "__main__":
    main()
