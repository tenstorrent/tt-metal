# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Where the LLK 2.0 tests find their drivers and headers."""

from __future__ import annotations

from pathlib import Path

from helpers.chip_architecture import ChipArchitecture
from helpers.test_config import TestConfig

LLK2_ROOT = Path(__file__).resolve().parents[2]  # tt-llk/llk2

# LLK 2.0 is ported to Blackhole only: inc/blackhole is the one arch folder.
_ARCH_DIR = {
    ChipArchitecture.BLACKHOLE: "blackhole",
}


def cpp_source(name: str) -> str:
    """Absolute driver path, so the harness never resolves it from tests/sources/."""
    path = LLK2_ROOT / "tests" / "sources" / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def include_dirs() -> list[str]:
    """inc/common plus the arch's folder, for TestConfig(include_dirs=...)."""
    if TestConfig.CHIP_ARCH not in _ARCH_DIR:
        raise NotImplementedError(f"LLK 2.0 is not ported to {TestConfig.CHIP_ARCH} yet")
    return [
        str(LLK2_ROOT / "inc" / "common"),
        str(LLK2_ROOT / "inc" / _ARCH_DIR[TestConfig.CHIP_ARCH]),
    ]


def kernel_test_include_dirs() -> list[str]:
    """include_dirs() plus the kernel-side helpers (kernels/) and the test-only helpers (tests/helpers/), for tests
    whose kernels use dataflow buffers."""
    return include_dirs() + [str(LLK2_ROOT / "kernels"), str(LLK2_ROOT / "tests" / "helpers")]
