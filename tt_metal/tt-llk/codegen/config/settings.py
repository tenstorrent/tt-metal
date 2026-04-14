# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Configuration settings for LLK CodeGen system."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Settings:
    """Global settings for LLK code generation."""

    # Paths
    tt_llk_root: Path = field(
        default_factory=lambda: Path(__file__).parent.parent.parent
    )
    sfpi_compiler: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "SFPI_COMPILER_PATH",
                str(
                    Path(__file__).parent.parent.parent
                    / "tests"
                    / "sfpi"
                    / "compiler"
                    / "bin"
                    / "riscv-tt-elf-g++"
                ),
            )
        )
    )
    build_dir: Path = field(
        default_factory=lambda: Path(f"/tmp/llk-codegen-build-{os.getpid()}")
    )

    # LLM settings
    llm_model: str = "claude-opus-4-6"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 8192
    num_candidates: int = 3  # N for beam search
    max_iterations: int = 5  # Max feedback loop iterations

    # Context settings
    max_context_tokens: int = 100000
    include_similar_ops: bool = True

    # Compilation settings
    compile_timeout: int = 60  # seconds

    # Architecture configs
    supported_architectures: list = field(
        default_factory=lambda: ["quasar", "blackhole", "wormhole"]
    )

    @property
    def anthropic_api_key(self) -> Optional[str]:
        """Get API key from environment."""
        return os.environ.get("ANTHROPIC_API_KEY")

    def get_llk_path(self, arch: str) -> Path:
        """Get LLK directory for architecture."""
        arch_dirs = {
            "quasar": "tt_llk_quasar",
            "blackhole": "tt_llk_blackhole",
            "wormhole": "tt_llk_wormhole_b0",
        }
        return self.tt_llk_root / arch_dirs.get(arch, arch)

    def get_sfpu_path(self, arch: str) -> Path:
        """Get SFPU include path for architecture."""
        return self.get_llk_path(arch) / "common" / "inc" / "sfpu"

    def validate(self) -> list[str]:
        """Validate settings and return list of issues. Empty list means all OK."""
        issues = []

        if not self.sfpi_compiler.exists():
            issues.append(
                f"SFPI compiler not found at: {self.sfpi_compiler}\n"
                f"  Fix: Set SFPI_COMPILER_PATH environment variable to the correct path,\n"
                f"  or ensure the compiler is installed at tests/sfpi/compiler/bin/riscv-tt-elf-g++"
            )

        venv_path = self.tt_llk_root / "tests" / ".venv"
        if not venv_path.exists():
            issues.append(
                f"Python venv not found at: {venv_path}\n"
                f"  Fix: Run the test environment setup script first"
            )

        return issues

    def get_include_paths(self, arch: str) -> list[Path]:
        """Get all include paths for compilation.

        Based on test_config.py TestConfig.INCLUDES - paths relative to tests/ directory.
        """
        llk_path = self.get_llk_path(arch)
        tests_dir = self.tt_llk_root / "tests"
        arch_name = arch if arch != "wormhole" else "wormhole"  # handle naming

        return [
            # SFPI compiler includes
            tests_dir / "sfpi" / "include",
            # LLK includes
            llk_path / "llk_lib",
            llk_path / "common" / "inc",
            llk_path / "common" / "inc" / "sfpu",
            # Common includes
            self.tt_llk_root / "common",
            # Hardware-specific headers (required for Quasar)
            tests_dir / "hw_specific" / arch_name / "inc",
            tests_dir / "hw_specific" / arch_name,
            tests_dir / "hw_specific" / arch_name / "metal_sfpu",
            # Firmware includes
            tests_dir / "firmware" / "riscv" / "common",
            # Test helpers
            tests_dir / "helpers" / "include",
        ]


# Global settings instance
settings = Settings()
