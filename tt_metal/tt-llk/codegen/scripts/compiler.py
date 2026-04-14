# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Compile agent for SFPI compilation of generated LLK code."""

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from codegen.config.settings import settings


@dataclass
class CompileError:
    """Structured compile error information."""

    file: str
    line: Optional[int]
    column: Optional[int]
    severity: str  # "error", "warning", "note"
    message: str
    raw_line: str


@dataclass
class CompileResult:
    """Result of a compilation attempt."""

    success: bool
    object_path: Optional[Path] = None
    errors: list[CompileError] = field(default_factory=list)
    warnings: list[CompileError] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0

    @property
    def error_summary(self) -> str:
        """Get a summary of errors for feedback to LLM."""
        if not self.errors:
            return ""

        lines = ["Compilation Errors:"]
        for err in self.errors[:10]:  # Limit to first 10 errors
            if err.line:
                lines.append(f"  Line {err.line}: {err.message}")
            else:
                lines.append(f"  {err.message}")

        if len(self.errors) > 10:
            lines.append(f"  ... and {len(self.errors) - 10} more errors")

        return "\n".join(lines)


class CompileAgent:
    """Agent for compiling generated LLK code using SFPI compiler."""

    # Architecture to compiler flags mapping (from tests/Makefile and test_config.py)
    ARCH_FLAGS = {
        # Quasar: uses BH cpu for now, per Makefile comment "until there is official support"
        "quasar": ["-mcpu=tt-bh-tensix", "-DARCH_QUASAR"],
        "blackhole": ["-mcpu=tt-bh-tensix", "-DARCH_BLACKHOLE"],
        "wormhole": ["-mcpu=tt-wh-tensix", "-DARCH_WORMHOLE"],
    }

    def __init__(self, arch: str = "quasar"):
        if arch not in settings.supported_architectures:
            raise ValueError(f"Unsupported architecture: {arch}")

        self.arch = arch
        self.compiler = settings.sfpi_compiler
        self.build_dir = settings.build_dir / arch
        self.build_dir.mkdir(parents=True, exist_ok=True)

    def compile(
        self,
        code: str,
        filename: str = "generated_kernel.h",
        extra_includes: Optional[list[Path]] = None,
        op_name: Optional[str] = None,
    ) -> CompileResult:
        """
        Compile the generated code.

        Args:
            code: The C++ code to compile
            filename: Name for the temporary file
            extra_includes: Additional include paths
            op_name: Operation name for wrapper (e.g., "sigmoid"). If None, inferred from filename.

        Returns:
            CompileResult with success status and any errors
        """
        # Infer op name from filename if not provided
        if op_name is None:
            stem = Path(filename).stem  # ckernel_sfpu_sigmoid -> ckernel_sfpu_sigmoid
            op_name = stem.replace("ckernel_sfpu_", "")

        # Create a wrapper that includes the generated header
        wrapper_code = self._create_wrapper(code, filename, op_name)

        # Write files to build directory
        header_path = self.build_dir / filename
        wrapper_path = self.build_dir / "compile_test.cpp"

        header_path.write_text(code)
        wrapper_path.write_text(wrapper_code)

        # Build compiler command
        cmd = self._build_compile_command(wrapper_path, extra_includes)

        # Run compilation
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=settings.compile_timeout,
                cwd=self.build_dir,
            )

            errors, warnings = self._parse_compiler_output(result.stderr)

            return CompileResult(
                success=result.returncode == 0,
                object_path=(
                    self.build_dir / "compile_test.o"
                    if result.returncode == 0
                    else None
                ),
                errors=errors,
                warnings=warnings,
                stdout=result.stdout,
                stderr=result.stderr,
                return_code=result.returncode,
            )

        except subprocess.TimeoutExpired:
            return CompileResult(
                success=False,
                errors=[
                    CompileError(
                        file=str(wrapper_path),
                        line=None,
                        column=None,
                        severity="error",
                        message=f"Compilation timed out after {settings.compile_timeout}s",
                        raw_line="",
                    )
                ],
                return_code=-1,
            )

        except Exception as e:
            return CompileResult(
                success=False,
                errors=[
                    CompileError(
                        file=str(wrapper_path),
                        line=None,
                        column=None,
                        severity="error",
                        message=f"Compilation failed: {str(e)}",
                        raw_line="",
                    )
                ],
                return_code=-1,
            )

    def _create_wrapper(self, code: str, filename: str, op_name: str) -> str:
        """Create a wrapper .cpp file that includes the generated header.

        This wrapper mimics how the test infrastructure includes and uses SFPU functions.
        """
        return f"""// Auto-generated compile test wrapper
// Mimics the test infrastructure includes

#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "{filename}"

using namespace ckernel;
using namespace ckernel::sfpu;

// Force template instantiation to check compilation
namespace {{
    void force_compile_{op_name}() {{
        _calculate_{op_name}_<true>(16);
        _calculate_{op_name}_<false>(16);
    }}

    void force_compile_init() {{
        _init_{op_name}_<true>();
        _init_{op_name}_<false>();
    }}
}}
"""

    def _build_compile_command(
        self,
        source_path: Path,
        extra_includes: Optional[list[Path]] = None,
    ) -> list[str]:
        """Build the compiler command line."""
        cmd = [str(self.compiler)]

        # Standard flags (from test_config.py OPTIONS_ALL and INITIAL_OPTIONS_COMPILE)
        cmd.extend(
            [
                "-c",  # Compile only, don't link
                "-std=c++17",
                "-O3",
                "-g",
                "-ffast-math",
                "-fno-use-cxa-atexit",
                "-Wall",
                "-fno-exceptions",
                "-fno-rtti",
                "-nostdlib",
                "-fno-builtin",
                "-Wunused-parameter",
                "-Wfloat-equal",
                "-Wpointer-arith",
                "-Wnull-dereference",
                "-Wredundant-decls",
                "-Wuninitialized",
                "-Wmaybe-uninitialized",
                "-DTENSIX_FIRMWARE",
                "-DENV_LLK_INFRA",
                "-DENABLE_LLK_ASSERT",
            ]
        )

        # Architecture-specific flags
        cmd.extend(self.ARCH_FLAGS.get(self.arch, []))

        # Include paths
        for inc_path in settings.get_include_paths(self.arch):
            if inc_path.exists():
                cmd.extend(["-I", str(inc_path)])

        # Build directory for generated header
        cmd.extend(["-I", str(self.build_dir)])

        # Extra includes
        if extra_includes:
            for inc in extra_includes:
                cmd.extend(["-I", str(inc)])

        # Output
        cmd.extend(["-o", str(self.build_dir / "compile_test.o")])

        # Source file
        cmd.append(str(source_path))

        return cmd

    def _parse_compiler_output(
        self, stderr: str
    ) -> tuple[list[CompileError], list[CompileError]]:
        """
        Parse compiler stderr to extract structured errors.

        Returns:
            Tuple of (errors, warnings)
        """
        errors = []
        warnings = []

        for line in stderr.split("\n"):
            line = line.strip()
            if not line:
                continue

            # GCC format: file:line:col: severity: message
            # or: file:line: severity: message
            error = self._parse_gcc_error_line(line)
            if error:
                if error.severity == "error":
                    errors.append(error)
                elif error.severity == "warning":
                    warnings.append(error)

        return errors, warnings

    def _parse_gcc_error_line(self, line: str) -> Optional[CompileError]:
        """Parse a single GCC error/warning line."""
        import re

        # Pattern: file:line:col: severity: message
        pattern = r"^(.+?):(\d+):(\d+):\s*(error|warning|note):\s*(.+)$"
        match = re.match(pattern, line)
        if match:
            return CompileError(
                file=match.group(1),
                line=int(match.group(2)),
                column=int(match.group(3)),
                severity=match.group(4),
                message=match.group(5),
                raw_line=line,
            )

        # Pattern without column: file:line: severity: message
        pattern = r"^(.+?):(\d+):\s*(error|warning|note):\s*(.+)$"
        match = re.match(pattern, line)
        if match:
            return CompileError(
                file=match.group(1),
                line=int(match.group(2)),
                column=None,
                severity=match.group(3),
                message=match.group(4),
                raw_line=line,
            )

        # Generic error without location
        if "error:" in line.lower():
            return CompileError(
                file="",
                line=None,
                column=None,
                severity="error",
                message=line,
                raw_line=line,
            )

        return None
