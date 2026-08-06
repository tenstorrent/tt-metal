# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict

from helpers.chip_architecture import ChipArchitecture

from .arch_common import isolate_sfpu_common
from .fpu_node import FpuNode
from .fuser_config import FuserConfig
from .operand import OperandRegistry
from .pack_node import PackNode

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
            # Only operands actually consumed by a real UNP_A/UNP_B unpacker need
            # a standard tile-shaped buf_desc_table entry from this thread. An
            # operand referenced only by an isolate_sfpu node (SrcS path) is
            # unpacked by the ISOLATE_SFPU thread itself via UNP_S, with its own
            # SrcS-shaped descriptor at the same buf_desc_id -- registering it
            # again here (tile-shaped) would race/clobber that entry.
            # Operand is a mutable dataclass (unhashable) -- dedup by id().
            unpack_operand_ids = {
                id(operand)
                for op in self.config.pipeline
                for cu in op.math.math_nodes
                if isinstance(cu, FpuNode) and cu.unpacker is not None
                for operand in (cu.src_a, cu.src_b)
                if operand is not None
            }
            reg = self.config.operand_registry
            buf_desc_init = OperandRegistry.emit_operand_init(
                [op for op in reg.get_all_inputs() if id(op) in unpack_operand_ids]
            )

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

        # Collect all unique headers from all isolate SFPU nodes.
        all_headers = set()
        for op in self.config.pipeline:
            for node in op.math.isolate_sfpu_nodes:
                all_headers.update(node.get_headers())

        # Generate include statements
        includes = "\n".join([f'#include "{header}"' for header in sorted(all_headers)])

        # Generate isolate SFPU calls for all operations
        sfpu_calls = "".join(
            [op.sfpu(self.config.global_config) for op in self.config.pipeline]
        )

        # The TRISC3 ELF is built for every Quasar fused test and trisc.cpp calls
        # run_kernel unconditionally, so this block is always emitted -- a test
        # with no isolate nodes gets an empty body, never a missing symbol.

        # This thread owns the descriptors for the operands it streams through
        # SrcS, on both the UNP_S and PACK1 sides -- the UNPACK and PACK threads
        # deliberately skip them (see UnpackKernelGenerator). They are SrcS-shaped
        # rather than tile-shaped, hence the dedicated emitter.
        srcs_operands = []
        seen = set()
        for op in self.config.pipeline:
            for node in op.math.isolate_sfpu_nodes:
                for operand in (node.src_a, node.src_b, node.output):
                    if operand is not None and id(operand) not in seen:
                        seen.add(id(operand))
                        srcs_operands.append(operand)
        buf_desc_init = isolate_sfpu_common.emit_operand_init(srcs_operands)

        code = (
            f"\n"
            f"#ifdef LLK_TRISC_ISOLATE_SFPU\n"
            f"\n"
            f"{includes}\n"
            f"\n"
            f"void run_kernel([[maybe_unused]] const volatile struct RuntimeParams& params)\n"
            f"{{\n"
            f"{buf_desc_init}"
            f"{sfpu_calls}"
            f"}}\n"
            f"\n"
            f"#endif\n"
        )

        return code


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
            # Same reasoning as UnpackKernelGenerator: an operand referenced only
            # by an isolate_sfpu node's SrcS output is packed by the ISOLATE_SFPU
            # thread itself via PACK1, with its own buf_desc_table entry -- only
            # register a standard PACK0 entry here for real PackNode outputs.
            pack_operand_ids = {
                id(pack_node.output)
                for op in self.config.pipeline
                for pack_node in op.math.pack_nodes
                if isinstance(pack_node, PackNode)
            }
            reg = self.config.operand_registry
            buf_desc_init = OperandRegistry.emit_operand_init(
                [op for op in reg.get_all_outputs() if id(op) in pack_operand_ids]
            )

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

        quasar_include = (
            '#include "llk_sync.h"\n'
            if self.config.global_config.architecture == ChipArchitecture.QUASAR
            else '#include "operand.h"\n'
        )

        combined = (
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
            f"{kernels['unpack']}"
            f"{kernels['math']}"
            f"{kernels['sfpu']}"
            f"{kernels['pack']}"
        )

        test_cpp_dir = Path(os.environ.get("LLK_HOME")) / "tests"

        fused_test_cpp_dir = test_cpp_dir / FUSED_TESTS_DIR
        fused_test_cpp_dir.mkdir(parents=True, exist_ok=True)

        cpp_path = test_cpp_dir / f"{test_name}"
        cpp_path.parent.mkdir(parents=True, exist_ok=True)

        with open(cpp_path, "w") as f:
            f.write(combined)

        if shutil.which("clang-format"):
            subprocess.run(["clang-format", "-i", str(cpp_path)])
