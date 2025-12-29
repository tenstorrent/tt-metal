# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from .fused_operation import FusedOperation


@dataclass
class UnpackKernelGenerator:
    def __init__(self, operations: List[FusedOperation]):
        self.operations = operations

    def generate(self) -> str:
        # Collect all unique headers from all operations
        all_headers = set()
        for op in self.operations:
            unpacker_instance = op.unpacker()
            all_headers.update(unpacker_instance.get_headers())

        # Generate include statements
        includes = "\n".join([f'#include "{header}"' for header in sorted(all_headers)])

        # Generate unpacker calls for all operations
        unpack_calls = "".join([op.unpack() for op in self.operations])

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
    def __init__(self, operations: List[FusedOperation]):
        self.operations = operations

    def generate(self) -> str:
        # Collect all unique headers from all operations
        all_headers = set()
        for op in self.operations:
            all_headers.update(op.math.get_headers())

        # Generate include statements
        includes = "\n".join([f'#include "{header}"' for header in sorted(all_headers)])

        # Generate math calls for all operations
        math_calls = "".join([op.do_math() for op in self.operations])

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
    def __init__(self, operations: List[FusedOperation]):
        self.operations = operations

    def generate(self) -> str:
        # Collect all unique headers from all operations
        all_headers = set()
        for op in self.operations:
            packer_instance = op.packer()
            all_headers.update(packer_instance.get_headers())

        # Generate include statements
        includes = "\n".join([f'#include "{header}"' for header in sorted(all_headers)])

        # Generate packer calls for all operations
        pack_calls = "".join([op.pack() for op in self.operations])

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

    def __init__(self, operations: List[FusedOperation]):
        self.operations = operations
        num_stages = len(self.operations)

        for i, op in enumerate(self.operations):
            op.stage_id = i
            op.num_stages = num_stages

        self.unpack_gen = UnpackKernelGenerator(self.operations)
        self.math_gen = MathKernelGenerator(self.operations)
        self.pack_gen = PackKernelGenerator(self.operations)

    def generate_all(self) -> Dict[str, str]:
        return {
            "unpack": self.unpack_gen.generate(),
            "math": self.math_gen.generate(),
            "pack": self.pack_gen.generate(),
        }

    def write_kernel(self):
        kernels = self.generate_all()

        combined = (
            f"#define FUSED_TEST\n"
            f'#include "ckernel.h"\n'
            f'#include "llk_defs.h"\n'
            f'#include "ckernel_debug.h"\n'
            f'#include "ckernel_defs.h"\n'
            f'#include "ckernel_sfpu.h"\n'
            f'#include "tensix_types.h"\n'
            f'#include "operand.h"\n'
            f"\n"
            f"uint32_t unp_cfg_context          = 0;\n"
            f"uint32_t pack_sync_tile_dst_ptr   = 0;\n"
            f"uint32_t math_sync_tile_dst_index = 0;\n"
            f"\n"
            f"inline uint32_t L1_ADDRESS(uint32_t buffer_address)\n"
            f"{{\n"
            f"#ifdef ARCH_QUASAR\n"
            f"    return buffer_address / 16;\n"
            f"#else\n"
            f"    return (buffer_address / 16) - 1;\n"
            f"#endif\n"
            f"}}\n"
            f"\n"
            f"#define UNUSED __attribute__((unused))\n"
            f"\n"
            f"{kernels['unpack']}"
            f"{kernels['math']}"
            f"{kernels['pack']}"
        )

        llk_home = Path(os.environ.get("LLK_HOME"))
        with open(llk_home / "tests/helpers/src/fused_test.cpp", "w") as f:
            f.write(combined)
