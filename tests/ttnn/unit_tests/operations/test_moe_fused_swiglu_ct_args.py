# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import re
from pathlib import Path

from ttnn.operations.moe_fused_swiglu.moe_fused_swiglu_program_descriptor import KERNEL_CT_ORDER


def test_python_descriptor_ct_order_matches_device_kernels():
    repo_root = Path(__file__).resolve().parents[4]
    header = (
        repo_root / "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_fused_swiglu/device/kernels/"
        "moe_fused_swiglu_ct_args.hpp"
    ).read_text()

    for kernel in ("reader", "writer", "compute"):
        match = re.search(
            rf"#define MOE_{kernel.upper()}_CT_ARGS\(X\)(.*?)(?=\n\n#define|\n// clang-format on)",
            header,
            flags=re.DOTALL,
        )
        assert match is not None
        cpp_order = tuple(re.findall(r"X\((\w+)\)", match.group(1)))
        assert cpp_order == KERNEL_CT_ORDER[kernel]
