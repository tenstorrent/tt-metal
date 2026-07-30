# SPDX-License-Identifier: Apache-2.0
"""Generates a compile_commands.json so editors can index kernel files.

Kernel .cpp files are not standalone translation units. The JIT build #includes
them into a firmware .cc and compiles that with the RISC-V/SFPI toolchain, using
per-kernel generated headers and per-core defines. An editor pointed at the bare
kernel file has none of that, which is why clangd reports
`'api/dataflow/dataflow_api.h' file not found` and then flags every API call as
undeclared.

This reconstructs enough of the real compile line — for a *host* compiler — that
includes resolve and go-to-definition/completion work.

Note this does not use tt-metal's own `--enable-fake-kernels-target`: that target
globs a fixed list of in-tree kernel directories, which does not include the
dojo's, and its arch include paths are stale at this revision.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

DOJO_ROOT = Path(__file__).resolve().parent.parent.parent
REPO_ROOT = DOJO_ROOT.parent
IDE_DIR = DOJO_ROOT / ".ide"

#: Arch alias used in tt-metal's directory names, and its short prefix.
ARCHES = {
    "wormhole_b0": ("wormhole", {"ARCH_WORMHOLE": None, "NUM_DRAM_BANKS": 12, "NUM_L1_BANKS": 64}),
    "blackhole": ("blackhole", {"ARCH_BLACKHOLE": None, "NUM_DRAM_BANKS": 8, "NUM_L1_BANKS": 140}),
}

#: Where the SFPI toolchain's headers live. Only needed by compute kernels.
SFPI_CANDIDATES = (
    Path("/opt/tenstorrent/sfpi/include"),
    REPO_ROOT / "runtime/sfpi/include",
)


def _include_dirs(arch: str) -> list[Path]:
    """Mirrors the JIT build's include path, plus the arch-specific dirs.

    The base list comes from `JitBuildEnv` in tt_metal/jit_build/build.cpp; the
    arch and low-level-kernel dirs from tt_metal/hw/CMakeLists.txt.
    """
    prefix, _ = ARCHES[arch]
    r = REPO_ROOT
    arch_inc = r / "tt_metal/hw/inc/internal/tt-1xx" / prefix
    dirs = [
        IDE_DIR,  # our stub chlkc_descriptors.h must win
        r,
        r / "ttnn",
        r / "ttnn/cpp",
        r / "tt_metal",
        r / "tt_metal/hw/inc",
        r / "tt_metal/hostdevcommon/api",
        r / "tt_metal/api",
        r / "tt_metal/api/tt-metalium",
        r / "tt_metal/hw/inc/internal/tt-1xx",
        r / "tt_metal/hw/firmware/src/tt-1xx",
        arch_inc,
        arch_inc / "noc",
        arch_inc / f"{arch}_defines",
        r / "tt_metal/tt-llk/common",
        r / f"tt_metal/tt-llk/tt_llk_{arch}/common/inc",
        r / f"tt_metal/tt-llk/tt_llk_{arch}/common/inc/sfpu",
        r / f"tt_metal/tt-llk/tt_llk_{arch}/llk_lib",
        r / f"tt_metal/hw/ckernels/{arch}/metal/common",
        r / f"tt_metal/hw/ckernels/{arch}/metal/llk_api",
        r / f"tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu",
        r / f"tt_metal/hw/ckernels/{arch}/metal/llk_io",
    ]
    for cand in SFPI_CANDIDATES:
        if cand.is_dir():
            dirs.append(cand)
            break
    return [d for d in dirs if d.is_dir()]


def _base_defines(arch: str) -> dict[str, object]:
    _, arch_defines = ARCHES[arch]
    d: dict[str, object] = {
        "TENSIX_FIRMWARE": None,
        "LOCAL_MEM_EN": 0,
        "KERNEL_BUILD": None,
        "NOC_INDEX": 0,
        "NOC_MODE": 0,
        "PROGRAMMABLE_CORE_TYPE": 0,
        # The real build derives these per DRAM/L1 bank count. 12 DRAM banks is
        # not a power of two, so the log2 form does not exist and the build
        # defines the IS_NOT_POW2 flag instead; 64 L1 banks is, so it does.
        "IS_NOT_POW2_NUM_DRAM_BANKS": 1,
        "LOG_BASE_2_OF_NUM_L1_BANKS": 6,
        # Referenced by the PCIe address helper in dataflow_api_addrgen.h.
        "PCIE_NOC_X": 0,
        "PCIE_NOC_Y": 3,
        # A kernel's real compile-time args are generated per dispatch. Any
        # plausible list keeps get_compile_time_arg_val() well-formed.
        "KERNEL_COMPILE_TIME_ARGS": "1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1",
    }
    d.update(arch_defines)
    return d


#: Per-processor defines. A data movement kernel is compiled for BRISC or NCRISC;
#: a compute kernel is compiled three times, once per TRISC. We index the math
#: variant, which is the one that sees the arithmetic.
DM_DEFINES = {"COMPILE_FOR_NCRISC": None, "PROCESSOR_INDEX": 1}
COMPUTE_DEFINES = {
    # TRISC_MATH (not UCK_CHLKC_MATH) is deliberate: it makes the compute API
    # headers pull in the math-thread implementation, while leaving chlkc_list.h
    # from #including the per-kernel generated wrapper, which does not exist
    # outside a real build.
    "TRISC_MATH": 1,
    "NAMESPACE": "chlkc_math",
    "COMPILE_FOR_TRISC": 1,
    "PROCESSOR_INDEX": 3,
}


def _is_compute(path: Path) -> bool:
    """Compute kernels are the ones including the compute API."""
    try:
        text = path.read_text()
    except OSError:
        return False
    return "api/compute/" in text


def _fmt_defines(defines: dict[str, object]) -> list[str]:
    out = []
    for k, v in defines.items():
        out.append(f"-D{k}" if v is None else f"-D{k}={v}")
    return out


def _compiler() -> str:
    for name in ("clang++-20", "clang++", "g++"):
        found = shutil.which(name)
        if found:
            return found
    return "c++"


def kernel_files() -> list[Path]:
    """Every kernel source in the exercises, including skeletons and solutions."""
    exercises = DOJO_ROOT / "exercises"
    return sorted(
        p
        for sub in ("kernels", "skeleton", "solution")
        for p in exercises.glob(f"*/{sub}/*.cpp")
    )


def generate(arch: str = "wormhole_b0") -> tuple[Path, int]:
    """Write compile_commands.json. Returns (path, number of entries)."""
    if arch not in ARCHES:
        raise SystemExit(f"unknown arch '{arch}'. Known: {', '.join(ARCHES)}")

    includes = [f"-I{d}" for d in _include_dirs(arch)]
    base = [
        _compiler(),
        "-std=c++20",
        "-fsyntax-only",
        # Kernels are written for a RISC-V target; a host parse produces a few
        # unavoidable complaints that are noise here.
        "-Wno-unknown-attributes",
        "-Wno-macro-redefined",
        f"-include{IDE_DIR / 'ide_prelude.h'}",
        *includes,
        *_fmt_defines(_base_defines(arch)),
    ]

    entries = []
    for f in kernel_files():
        extra = COMPUTE_DEFINES if _is_compute(f) else DM_DEFINES
        entries.append(
            {
                "directory": str(REPO_ROOT),
                "file": str(f),
                "arguments": base + _fmt_defines(extra) + [str(f)],
            }
        )

    out = DOJO_ROOT / "compile_commands.json"
    out.write_text(json.dumps(entries, indent=2) + "\n")
    return out, len(entries)
