# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math
import pytest
import torch
import ttnn

from ....utils.tensor import bf16_tensor
from ....utils.check import assert_quality
from ....layers.linear import Linear


@pytest.mark.parametrize(
    ("M, K, N, bias, activation_fn"),
    [
        (9472, 5120, 1280, False, None),
        (9472, 5120, 3456, True, None),
        (9472, 3456, 5120, True, None),
    ],
    ids=[
        "wan_4x8sp1tp0_qkv",
        "wan_4x8sp1tp0_ff1",
        "wan_4x8sp1tp0_ff2",
    ],
)
def test_linear(
    mesh_device: ttnn.MeshDevice,
    M: int,
    K: int,
    N: int,
    bias: bool,
    activation_fn: str,
) -> None:
    torch_dtype = torch.bfloat16
    torch_model = torch.nn.Linear(K, N, bias=bias).to(dtype=torch_dtype)
    torch_model.eval()

    parent_mesh_device = mesh_device
    mesh_device = parent_mesh_device.create_submesh(ttnn.MeshShape(1, 1))
    tt_model = Linear(K, N, bias=bias, mesh_device=mesh_device, activation_fn=activation_fn)
    tt_model.load_state_dict(torch_model.state_dict())

    torch_input_tensor = torch.randn((M, K), dtype=torch_dtype)

    tt_input_tensor = bf16_tensor(torch_input_tensor, device=mesh_device)

    with torch.no_grad():
        torch_output = torch_model(torch_input_tensor)
        if activation_fn == "gelu":
            torch_output = torch.nn.functional.gelu(torch_output)
        else:
            assert activation_fn is None

    device_grid = mesh_device.compute_with_storage_grid_size()
    core_grid = ttnn.CoreGrid(x=device_grid.x, y=device_grid.y)
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    # tt_output = tt_model(tt_input_tensor, core_grid=core_grid, compute_kernel_config=compute_kernel_config)
    # tt_output = ttnn.to_torch(tt_output)
    # assert_quality(torch_output, tt_output, pcc=0.999_500)
    # return

    M_t = M // 32
    K_t = K // 32
    N_t = N // 32

    def get_smallest_divisor_greater_than(n: int, min_val: float) -> int:
        for i in range(math.ceil(min_val), n + 1):
            if n % i == 0:
                return i
        return n

    per_core_M = get_smallest_divisor_greater_than(M_t, M_t / device_grid.y)
    per_core_N = get_smallest_divisor_greater_than(N_t, N_t / device_grid.x)
    opt_in0_block_w = [1, 2]
    opt_out_subblock_h = [1]
    opt_out_subblock_w = [1]
    opt_out_block_h = [1]
    opt_out_block_w = [1]
    opt_per_core_M = [M_t]
    opt_per_core_N = [N_t]
    from itertools import product

    for in0_block_w, out_subblock_h, out_subblock_w, out_block_h, out_block_w, per_core_M, per_core_N in product(
        opt_in0_block_w,
        opt_out_subblock_h,
        opt_out_subblock_w,
        opt_out_block_h,
        opt_out_block_w,
        opt_per_core_M,
        opt_per_core_N,
    ):
        cfg = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=device_grid,
            in0_block_w=in0_block_w,
            out_subblock_h=out_subblock_h,
            out_subblock_w=out_subblock_w,
            per_core_M=per_core_M,
            per_core_N=per_core_N,
        )
        tt_output = tt_model(
            tt_input_tensor, core_grid=None, compute_kernel_config=compute_kernel_config, program_config=cfg
        )
        tt_output = ttnn.to_torch(tt_output)
        assert_quality(torch_output, tt_output, pcc=0.999_500)

    # tt_output = tt_model(tt_input_tensor, core_grid=core_grid, compute_kernel_config=compute_kernel_config)

    # for t in ttnn.get_device_tensors(tt_output):
    #     t = ttnn.to_torch(t)
    #     assert_quality(torch_output, t, pcc=0.999_500)
