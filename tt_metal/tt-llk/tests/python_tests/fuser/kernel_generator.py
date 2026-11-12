# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import subprocess
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import Dict

from filelock import FileLock
from helpers.chip_architecture import ChipArchitecture

from .fuser_config import FuserConfig

FUSED_TESTS_DIR = Path("sources/fused_tests")


@lru_cache(maxsize=512)
def format_cpp(source: str, source_dir: Path, cache_dir: Path) -> str:
    if not shutil.which("clang-format"):
        return source

    key = sha256(f"{source_dir}\0{source}".encode()).hexdigest()
    cached = cache_dir / f"{key}.cpp"
    cache_dir.mkdir(parents=True, exist_ok=True)
    with FileLock(cached.with_suffix(".lock")):
        if not cached.exists():
            formatted = subprocess.check_output(
                ["clang-format", "--assume-filename=kernel.cpp"],
                cwd=source_dir,
                input=source,
                text=True,
            )
            temporary = cached.with_suffix(".tmp")
            temporary.write_text(formatted)
            temporary.replace(cached)
        return cached.read_text()


def render_kernel(thread: str, headers, body: str) -> str:
    includes = "\n".join(f'#include "{header}"' for header in sorted(headers))
    return (
        f"\n"
        f"#ifdef LLK_TRISC_{thread}\n"
        f"\n"
        f"{includes}\n"
        f"\n"
        f"void run_kernel([[maybe_unused]] const volatile struct RuntimeParams& params)\n"
        f"{{\n"
        f"{body}"
        f"}}\n"
        f"\n"
        f"#endif\n"
    )


def _unpack_kernel(config: FuserConfig) -> str:
    headers = {
        header
        for op in config.pipeline
        for node in op.math_nodes
        if getattr(node, "unpacker", None) is not None
        for header in node.unpacker.get_headers()
    }
    calls = ""
    previous = None
    for op, blocks in zip(config.pipeline, config.pipeline_plans):
        calls += op.unpack(config.global_config, blocks, previous)
        previous = op.first_unpack or previous
    return render_kernel("UNPACK", headers, calls)


def _math_kernel(config: FuserConfig) -> str:
    headers = {
        header
        for op in config.pipeline
        for unit in op.get_math_units()
        for header in unit.get_headers()
    }
    if config.global_config.skip_math_init:
        headers.discard("sfpu_operations_quasar.h")
    calls = ""
    previous = None
    for op, blocks in zip(config.pipeline, config.pipeline_plans):
        calls += op.do_math(config.global_config, blocks, previous)
        previous = op.first_math
    return render_kernel("MATH", headers, calls)


def _pack_kernel(config: FuserConfig) -> str:
    headers = {
        header
        for op in config.pipeline
        for node in op.pack_nodes
        for header in node.get_headers()
    }
    if config.global_config.skip_math_init:
        headers.discard("sfpu_operations_quasar.h")
    calls = ""
    previous = None
    for op, blocks in zip(config.pipeline, config.pipeline_plans):
        calls += op.pack(config.global_config, blocks, previous)
        previous = op._get_pack_nodes()[0]
    return render_kernel("PACK", headers, calls)


def _sfpu_kernel(config: FuserConfig) -> str:
    if config.global_config.architecture != ChipArchitecture.QUASAR:
        return ""
    return (
        f"\n"
        f"#ifdef LLK_TRISC_ISOLATE_SFPU\n"
        f"\n"
        f"void run_kernel([[maybe_unused]] const volatile struct RuntimeParams& params)\n"
        f"{{\n"
        f"}}\n"
        f"\n"
        f"#endif\n"
    )


def generate_kernels(config: FuserConfig) -> Dict[str, str]:
    return {
        "unpack": _unpack_kernel(config),
        "math": _math_kernel(config),
        "pack": _pack_kernel(config),
        "sfpu": _sfpu_kernel(config),
    }


class FusedKernelGenerator:
    def __init__(self, config: FuserConfig):
        self.config = config

    def write_kernel(self, test_name: str):
        if not self.config.global_config.regenerate_cpp:
            return

        kernels = generate_kernels(self.config)

        profiler_include = ""
        if self.config.global_config.profiler_enabled:
            profiler_include += '#include "profiler.h"\n'
            profiler_include += '#include "perf.h"\n'

        if self.config.global_config.architecture == ChipArchitecture.QUASAR:
            operands = ""
        else:
            operands = self.config.operand_registry.generate_cpp(
                self.config.global_config.dest_acc.value
            )

        quasar_include = (
            '#include "llk_bfd_alloc.h"\n#include "llk_sync.h"\n#include "quasar_test_common.h"\n'
            if self.config.global_config.architecture == ChipArchitecture.QUASAR
            else '#include "operand.h"\n'
        )

        common = (
            f"#define FUSED_TEST\n"
            f'#include "ckernel.h"\n'
            f'#include "llk_defs.h"\n'
            f'#include "ckernel_defs.h"\n'
            f'#include "ckernel_sfpu.h"\n'
            f'#include "tensix_types.h"\n'
            f"{quasar_include}"
            f"{profiler_include}"
            f"\n"
            f"std::uint32_t unp_cfg_context          = 0;\n"
            f"std::uint32_t pack_sync_tile_dst_ptr   = 0;\n"
            f"std::uint32_t math_sync_tile_dst_index = 0;\n"
            f"\n"
            f"#define UNUSED __attribute__((unused))\n"
            f"struct RuntimeParams {{}};\n"
            f"\n"
            f"{operands}"
            f"\n"
        )
        test_cpp_dir = Path(os.environ.get("LLK_HOME")) / "tests"

        fused_test_cpp_dir = test_cpp_dir / FUSED_TESTS_DIR
        fused_test_cpp_dir.mkdir(parents=True, exist_ok=True)

        cpp_path = test_cpp_dir / f"{test_name}"
        cpp_path.parent.mkdir(parents=True, exist_ok=True)

        format_cache = self.config.ARTEFACTS_DIR / "fused-format"
        common = format_cpp(common, cpp_path.parent, format_cache)
        kernels = {
            name: format_cpp(kernel, cpp_path.parent, format_cache)
            for name, kernel in kernels.items()
        }
        cpp_path.write_text(common + "".join(kernels.values()))

        return {name: common + kernel for name, kernel in kernels.items()}
