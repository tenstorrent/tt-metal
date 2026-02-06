# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from .fuser_config import FuserConfig

FUSED_TESTS_DIR = Path("sources/fused_tests")


@dataclass
class UnpackKernelGenerator:
    def __init__(self, config: FuserConfig):
        self.config = config

    def generate(self) -> str:
        # Collect all unique headers from all operations
        all_headers = set()
        for op in self.config.pipeline:
            for fused_compute in op.math.operations:
                if fused_compute.unpacker is not None:
                    unpacker_instance = fused_compute.unpacker()
                    all_headers.update(unpacker_instance.get_headers())

        # Generate include statements
        includes = "\n".join([f'#include "{header}"' for header in sorted(all_headers)])

        # Generate unpacker calls for all operations
        unpack_calls = "".join(
            [op.unpack(self.config.global_config) for op in self.config.pipeline]
        )

        code = (
            f"\n"
            f"#ifdef LLK_TRISC_UNPACK\n"
            f"\n"
            f"{includes}\n"
            f"\n"
            f"void run_kernel(const volatile struct RuntimeParams* params)\n"
            f"{{\n"
            f"{unpack_calls}"
            f"}}\n"
            f"\n"
            f"#endif\n"
        )

        return code


@dataclass
class MathKernelGenerator:
    def __init__(self, config: FuserConfig):
        self.config = config

    def generate(self) -> str:
        # Collect all unique headers from all operations
        all_headers = set()
        for op in self.config.pipeline:
            for unit in op.math.get_math_units():
                all_headers.update(unit.get_headers())

        # Generate include statements
        includes = "\n".join([f'#include "{header}"' for header in sorted(all_headers)])

        # Generate math calls for all operations
        math_calls = "".join(
            [op.do_math(self.config.global_config) for op in self.config.pipeline]
        )

        code = (
            f"\n"
            f"#ifdef LLK_TRISC_MATH\n"
            f"\n"
            f"{includes}\n"
            f"\n"
            f"void run_kernel(const volatile struct RuntimeParams* params)\n"
            f"{{\n"
            f"{math_calls}"
            f"}}\n"
            f"\n"
            f"#endif\n"
        )

        return code


@dataclass
class PackKernelGenerator:
    def __init__(self, config: FuserConfig):
        self.config = config

    def generate(self) -> str:
        # Collect all unique headers from all operations
        all_headers = set()
        for op in self.config.pipeline:
            packer_instance = op.packer()
            all_headers.update(packer_instance.get_headers())

        # Generate include statements
        includes = "\n".join([f'#include "{header}"' for header in sorted(all_headers)])

        # Generate packer calls for all operations
        pack_calls = "".join(
            [op.pack(self.config.global_config) for op in self.config.pipeline]
        )

        code = (
            f"\n"
            f"#ifdef LLK_TRISC_PACK\n"
            f"\n"
            f"{includes}\n"
            f"\n"
            f"void run_kernel(const volatile struct RuntimeParams* params)\n"
            f"{{\n"
            f"{pack_calls}"
            f"}}\n"
            f"\n"
            f"#endif\n"
        )

        return code


class FusedKernelGenerator:
    def __init__(self, config: FuserConfig):
        self.config = config
        self.unpack_gen = UnpackKernelGenerator(self.config)
        self.math_gen = MathKernelGenerator(self.config)
        self.pack_gen = PackKernelGenerator(self.config)

    def generate_all(self) -> Dict[str, str]:
        return {
            "unpack": self.unpack_gen.generate(),
            "math": self.math_gen.generate(),
            "pack": self.pack_gen.generate(),
        }

    def write_kernel(self, test_name: str, regenerate_cpp: bool = True):
        if not regenerate_cpp:
            return

        kernels = self.generate_all()

        profiler_include = ""
        if self.config.global_config.profiler_enabled:
            profiler_include += '#include "profiler.h"\n'
            profiler_include += '#include "perf.h"\n'

        combined = (
            f"#define FUSED_TEST\n"
            f'#include "ckernel.h"\n'
            f'#include "llk_defs.h"\n'
            f'#include "ckernel_debug.h"\n'
            f'#include "ckernel_defs.h"\n'
            f'#include "ckernel_sfpu.h"\n'
            f'#include "tensix_types.h"\n'
            f'#include "operand.h"\n'
            f"{profiler_include}"
            f"\n"
            f"std::uint32_t unp_cfg_context          = 0;\n"
            f"std::uint32_t pack_sync_tile_dst_ptr   = 0;\n"
            f"std::uint32_t math_sync_tile_dst_index = 0;\n"
            f"\n"
            f"#define UNUSED __attribute__((unused))\n"
            f"\n"
            f"{kernels['unpack']}"
            f"{kernels['math']}"
            f"{kernels['pack']}"
        )

        test_cpp_dir = Path(os.environ.get("LLK_HOME")) / "tests"

        fused_test_cpp_dir = test_cpp_dir / FUSED_TESTS_DIR
        fused_test_cpp_dir.mkdir(parents=True, exist_ok=True)

        cpp_path = test_cpp_dir / f"{test_name}"

        with open(cpp_path, "w") as f:
            f.write(combined)

        os.system(f'clang-format -i "{cpp_path}"')
