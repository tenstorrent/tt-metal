# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

import ttnn

pytestmark = pytest.mark.use_module_device


@pytest.mark.parametrize("axis", [0, 1], ids=["H", "W"])
@pytest.mark.parametrize(
    "algorithm,extent",
    [("SMALL", extent) for extent in [15, 256, 257, 769]] + [("LARGE", extent) for extent in [15, 256, 257, 769, 1025]],
)
@pytest.mark.parametrize("operation", ["softmax", "softmin", "logsoftmax"])
@pytest.mark.parametrize("fp32_dest_acc_en", [False, True])
def test_moreh_softmax_reduce_boundaries(device, axis, extent, algorithm, operation, fp32_dest_acc_en):
    """Exercise partial tiles, block transitions and repeated accumulation in both directions."""
    torch.manual_seed(42)
    shape = (extent, 64) if axis == 0 else (64, extent)
    # Keep the autograd reference in the same dtype as the supplied y and dy;
    # log-softmax backward depends on the rounded forward output.
    x = (torch.rand(shape) + 0.25).to(torch.bfloat16).requires_grad_()
    dy = torch.randn(shape).to(torch.bfloat16)
    torch_op = {"softmax": F.softmax, "softmin": F.softmin, "logsoftmax": F.log_softmax}[operation]
    golden = torch_op(x, dim=axis)
    golden.backward(dy)
    strategy_name = f"{algorithm}_{'H' if axis == 0 else 'W'}"
    config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_dest_acc_en,
    )

    def to_device(value):
        tensor = ttnn.from_torch(value.detach().to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
        # Padding must not participate in MAX or either gradient reduction.
        ttnn.fill_implicit_tile_padding(tensor, 42)
        return tensor

    actual = getattr(ttnn.operations.moreh, operation)(
        to_device(x),
        axis,
        strategy=getattr(ttnn.operations.moreh.SoftmaxOpParallelizationStrategy, strategy_name),
        compute_kernel_config=config,
    )
    torch.testing.assert_close(
        ttnn.to_torch(actual).float(),
        golden.detach().float(),
        rtol=0.06,
        atol=0.04 if operation == "logsoftmax" else 0.001,
    )
    actual_grad = getattr(ttnn.operations.moreh, f"{operation}_backward")(
        to_device(golden),
        to_device(dy),
        axis,
        strategy=getattr(ttnn.operations.moreh.SoftmaxBackwardOpParallelizationStrategy, strategy_name),
        compute_kernel_config=config,
    )
    # Near-zero log-softmax gradients expose rounding of unit-scale BF16
    # intermediates; allow one BF16 ULP at that scale for cancellation.
    grad_atol = 1 / 128 if operation == "logsoftmax" else 0.003
    torch.testing.assert_close(ttnn.to_torch(actual_grad).float(), x.grad.float(), rtol=0.1, atol=grad_atol)
