# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import tt_lib as ttl
from models.utility_functions import is_wormhole_b0

compute_kernel_options = [
    False,  # for grayskull
]
compute_kernel_ids = ["fp32_dest_acc_en=False"]
if is_wormhole_b0:
    compute_kernel_options.append(True)
    compute_kernel_ids.append("fp32_dest_acc_en=True")


def get_compute_kernel_options(compute_kernel_options):
    if is_wormhole_b0():
        fp32_dest_acc_en = compute_kernel_options
        packer_l1_acc = False
        compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
            math_fidelity=ttl.tensor.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=fp32_dest_acc_en,
            packer_l1_acc=packer_l1_acc,
        )
    else:
        # Grayskull doesn't support fp32 but test passing a GS config is ok
        compute_kernel_config = ttl.tensor.GrayskullComputeKernelConfig(
            math_fidelity=ttl.tensor.MathFidelity.HiFi4,
            math_approx_mode=True,
        )
    return compute_kernel_config
