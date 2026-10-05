# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import subprocess
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import Dict, List

from filelock import FileLock
from helpers.chip_architecture import ChipArchitecture

from .fuser_config import FuserConfig
from .pipeline_plan import PlannedBlock

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
            result = subprocess.run(
                ["clang-format", "--assume-filename=kernel.cpp"],
                cwd=source_dir,
                shell=False,
                input=source,
                text=True,
                stdout=subprocess.PIPE,
                check=True,
            )
            temporary = cached.with_suffix(".tmp")
            temporary.write_text(result.stdout)
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


class UnpackKernelGenerator:
    def __init__(self, config: FuserConfig):
        self.config = config

    def generate(self, plans: List[List[PlannedBlock]]) -> str:
        # Collect all unique headers from all operations
        all_headers = set()
        for op in self.config.pipeline:
            for fused_compute in op.math_nodes:
                if (
                    hasattr(fused_compute, "unpacker")
                    and fused_compute.unpacker is not None
                ):
                    all_headers.update(fused_compute.unpacker.get_headers())

        unpack_calls = "".join(
            op.unpack(self.config.global_config, blocks)
            for op, blocks in zip(self.config.pipeline, plans)
        )
        return render_kernel("UNPACK", all_headers, unpack_calls)


class MathKernelGenerator:
    def __init__(self, config: FuserConfig):
        self.config = config

    def generate(self, plans: List[List[PlannedBlock]]) -> str:
        # Collect all unique headers from all operations
        all_headers = set()
        for op in self.config.pipeline:
            for unit in op.get_math_units():
                all_headers.update(unit.get_headers())

        if self.config.global_config.skip_math_init:
            all_headers.discard("sfpu_operations_quasar.h")

        math_calls = "".join(
            op.do_math(self.config.global_config, blocks)
            for op, blocks in zip(self.config.pipeline, plans)
        )
        return render_kernel("MATH", all_headers, math_calls)


class SfpuKernelGenerator:
    def __init__(self, config: FuserConfig):
        self.config = config

    def generate(self) -> str:
        if self.config.global_config.architecture != ChipArchitecture.QUASAR:
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


class PackKernelGenerator:
    def __init__(self, config: FuserConfig):
        self.config = config

    def generate(self, plans: List[List[PlannedBlock]]) -> str:
        # Collect all unique headers from all operations
        all_headers = set()
        for op in self.config.pipeline:
            for pack_node in op.pack_nodes:
                all_headers.update(pack_node.get_headers())

        if self.config.global_config.skip_math_init:
            all_headers.discard("sfpu_operations_quasar.h")

        pack_calls = "".join(
            op.pack(self.config.global_config, blocks)
            for op, blocks in zip(self.config.pipeline, plans)
        )
        return render_kernel("PACK", all_headers, pack_calls)


class FusedKernelGenerator:
    def __init__(self, config: FuserConfig):
        self.config = config
        self.unpack_gen = UnpackKernelGenerator(self.config)
        self.math_gen = MathKernelGenerator(self.config)
        self.pack_gen = PackKernelGenerator(self.config)
        self.sfpu_gen = SfpuKernelGenerator(self.config)

    def generate_all(self) -> Dict[str, str]:
        plans = self.config.get_pipeline_plans()
        return {
            "unpack": self.unpack_gen.generate(plans),
            "math": self.math_gen.generate(plans),
            "pack": self.pack_gen.generate(plans),
            "sfpu": self.sfpu_gen.generate(),
        }

    def write_kernel(self, test_name: str):
        if not self.config.global_config.regenerate_cpp:
            return

        kernels = self.generate_all()

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
        combined = common + "".join(
            kernels[name] for name in ("unpack", "math", "sfpu", "pack")
        )

        with open(cpp_path, "w") as f:
            f.write(combined)

        return {name: common + kernel for name, kernel in kernels.items()}
