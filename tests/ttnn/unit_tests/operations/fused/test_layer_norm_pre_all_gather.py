# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn


def test_ones_tile_implicit_padding_matches_from_torch_layer_norm_pre_all_gather(device):
    shape = (1, 1, 37, 72)
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    t_torch = ttnn.from_torch(
        torch.ones(shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    print("TTNN input: ", t_torch)
    t_ones = ttnn.ones(shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    print("TTNN input: ", t_ones)

    out_torch = ttnn.layer_norm_pre_all_gather(
        t_torch,
        dtype=ttnn.float32,
        compute_kernel_config=compute_kernel_config,
    )
    out_ones = ttnn.layer_norm_pre_all_gather(
        t_ones,
        dtype=ttnn.float32,
        compute_kernel_config=compute_kernel_config,
    )
    layer_norm_pre_all_gather_0_torch = ttnn.to_torch(out_torch)
    layer_norm_pre_all_gather_1_torch = ttnn.to_torch(out_ones)
    print("Output from TTNN with torch input: ", layer_norm_pre_all_gather_0_torch)
    print("Output from TTNN with TTNN input: ", layer_norm_pre_all_gather_1_torch)
    assert torch.allclose(
        layer_norm_pre_all_gather_0_torch,
        layer_norm_pre_all_gather_1_torch,
        rtol=1e-2,
        atol=1e-2,
    )
