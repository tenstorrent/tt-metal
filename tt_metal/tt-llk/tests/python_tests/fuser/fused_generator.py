# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict

from helpers.chip_architecture import ChipArchitecture

from .fused_operand import OperandRegistry
from .fuser_config import FuserConfig

FUSED_TESTS_DIR = Path("sources/fused_tests")


class UnpackKernelGenerator:
    def __init__(self, config: FuserConfig):
        self.config = config

    def generate(self) -> str:
        # Collect all unique headers from all operations
        all_headers = set()
        for op in self.config.pipeline:
            for fused_compute in op.math.math_nodes:
                if (
                    hasattr(fused_compute, "unpacker")
                    and fused_compute.unpacker is not None
                ):
                    all_headers.update(fused_compute.unpacker.get_headers())

        # Generate include statements
        includes = "\n".join([f'#include "{header}"' for header in sorted(all_headers)])

        # Generate unpacker calls for all operations
        unpack_calls = "".join(
            [op.unpack(self.config.global_config) for op in self.config.pipeline]
        )

        buf_desc_init = ""
        if self.config.global_config.architecture == ChipArchitecture.QUASAR:
            reg = self.config.operand_registry
            buf_desc_init = OperandRegistry.emit_operand_init(reg.get_all_inputs())

        code = (
            f"\n"
            f"#ifdef LLK_TRISC_UNPACK\n"
            f"\n"
            f"{includes}\n"
            f"\n"
            f"void run_kernel([[maybe_unused]] const volatile struct RuntimeParams& params)\n"
            f"{{\n"
            f"{buf_desc_init}"
            f"{unpack_calls}"
            f"}}\n"
            f"\n"
            f"#endif\n"
        )

        return code


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
            f"void run_kernel([[maybe_unused]] const volatile struct RuntimeParams& params)\n"
            f"{{\n"
            f"{math_calls}"
            f"}}\n"
            f"\n"
            f"#endif\n"
        )

        return code


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

    def generate(self) -> str:
        # Collect all unique headers from all operations
        all_headers = set()
        for op in self.config.pipeline:
            for pack_node in op.math.pack_nodes:
                all_headers.update(pack_node.get_headers())

        # Generate include statements
        includes = "\n".join([f'#include "{header}"' for header in sorted(all_headers)])

        # Generate packer calls for all operations
        pack_calls = "".join(
            [op.pack(self.config.global_config) for op in self.config.pipeline]
        )

        buf_desc_init = ""
        if self.config.global_config.architecture == ChipArchitecture.QUASAR:
            reg = self.config.operand_registry
            buf_desc_init = OperandRegistry.emit_operand_init(reg.get_all_outputs())

        code = (
            f"\n"
            f"#ifdef LLK_TRISC_PACK\n"
            f"\n"
            f"{includes}\n"
            f"\n"
            f"void run_kernel([[maybe_unused]] const volatile struct RuntimeParams& params)\n"
            f"{{\n"
            f"{buf_desc_init}"
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
        self.sfpu_gen = SfpuKernelGenerator(self.config)

    def generate_all(self) -> Dict[str, str]:
        return {
            "unpack": self.unpack_gen.generate(),
            "math": self.math_gen.generate(),
            "pack": self.pack_gen.generate(),
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

        operand_include = (
            ""
            if self.config.global_config.architecture == ChipArchitecture.QUASAR
            else '#include "operand.h"\n'
        )

        dvalid_include = ""
        if self.config.global_config.architecture == ChipArchitecture.QUASAR:
            dvalid_include = (
                '#include "dvalid_helper.h"\n' "using namespace test_utils::dvalid;\n"
            )

        combined = (
            f"#define FUSED_TEST\n"
            f'#include "ckernel.h"\n'
            f'#include "llk_defs.h"\n'
            f'#include "ckernel_defs.h"\n'
            f'#include "ckernel_sfpu.h"\n'
            f'#include "tensix_types.h"\n'
            f"{operand_include}"
            f"{dvalid_include}"
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
            f"{kernels['unpack']}"
            f"{kernels['math']}"
            f"{kernels['sfpu']}"
            f"{kernels['pack']}"
        )

        test_cpp_dir = Path(os.environ.get("LLK_HOME")) / "tests"

        fused_test_cpp_dir = test_cpp_dir / FUSED_TESTS_DIR
        fused_test_cpp_dir.mkdir(parents=True, exist_ok=True)

        cpp_path = test_cpp_dir / f"{test_name}"

        with open(cpp_path, "w") as f:
            f.write(combined)

        if shutil.which("clang-format"):
            subprocess.run(["clang-format", "-i", str(cpp_path)])
