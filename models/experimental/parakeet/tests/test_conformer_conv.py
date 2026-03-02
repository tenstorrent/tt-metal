"""
PCC test: PyTorch ConformerConvolution vs TTNN TtConformerConvolution

Validates:
    - pointwise conv1
    - GLU
    - depthwise conv
    - batch norm (inference)
    - silu
    - pointwise conv2

Using identical weights.
"""

import pytest
import torch
import ttnn
from types import SimpleNamespace
from loguru import logger

from models.experimental.parakeet.reference.pytorch_conf_layer import (
    ConformerConvolution as TorchConformerConvolution,
)

from models.experimental.parakeet.tt.ttnn_conf_layer import (
    TtConformerConvolution,
)

from tests.ttnn.utils_for_testing import check_with_pcc


CONV_L1_SMALL_SIZE = 32768


# ------------------------------------------------------------
# Config (match your encoder)
# ------------------------------------------------------------

d_model = 1024
kernel_size = 31
time_steps = 64


# ------------------------------------------------------------
# Weight Copy Helper
# ------------------------------------------------------------


def _copy_conv_weights_to_ttnn(torch_conv, device):
    params = SimpleNamespace()

    # Pointwise1
    params.pointwise1 = SimpleNamespace()
    params.pointwise1.weight = ttnn.from_torch(
        torch_conv.pointwise_conv1.weight.data.clone(),
        dtype=ttnn.bfloat16,
        device=device,
    )

    # Depthwise
    params.depthwise = SimpleNamespace()
    params.depthwise.weight = ttnn.from_torch(
        torch_conv.depthwise_conv.weight.data.clone(),
        dtype=ttnn.bfloat16,
        device=device,
    )

    # Pointwise2
    params.pointwise2 = SimpleNamespace()
    params.pointwise2.weight = ttnn.from_torch(
        torch_conv.pointwise_conv2.weight.data.clone(),
        dtype=ttnn.bfloat16,
        device=device,
    )

    # BatchNorm (IMPORTANT: include running stats)
    params.bn = SimpleNamespace()

    params.bn.weight = ttnn.from_torch(
        torch_conv.batch_norm.weight.data.clone(),
        dtype=ttnn.bfloat16,
        device=device,
    )

    params.bn.bias = ttnn.from_torch(
        torch_conv.batch_norm.bias.data.clone(),
        dtype=ttnn.bfloat16,
        device=device,
    )

    params.bn.running_mean = ttnn.from_torch(
        torch_conv.batch_norm.running_mean.data.clone(),
        dtype=ttnn.bfloat16,
        device=device,
    )

    params.bn.running_var = ttnn.from_torch(
        torch_conv.batch_norm.running_var.data.clone(),
        dtype=ttnn.bfloat16,
        device=device,
    )

    return params


# ------------------------------------------------------------
# PCC Test
# ------------------------------------------------------------


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": CONV_L1_SMALL_SIZE}],
    indirect=True,
)
@pytest.mark.parametrize("batch_size", [1, 2], ids=["batch1", "batch2"])
def test_ttnn_conformer_conv_pcc(device, batch_size):
    torch.manual_seed(0)

    # --------------------------------------------------------
    # PyTorch Reference
    # --------------------------------------------------------

    torch_conv = TorchConformerConvolution(
        d_model=d_model,
        kernel_size=kernel_size,
    ).to(torch.bfloat16)

    torch_conv.eval()

    # --------------------------------------------------------
    # TTNN Module
    # --------------------------------------------------------

    tt_conv = TtConformerConvolution(
        d_model=d_model,
        kernel_size=kernel_size,
        device=device,
        dtype=ttnn.bfloat16,
    )

    params = _copy_conv_weights_to_ttnn(torch_conv, device)

    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    pt_input = torch.randn(
        batch_size,
        time_steps,
        d_model,
        dtype=torch.bfloat16,
    )

    # Reference forward
    with torch.no_grad():
        ref_out = torch_conv(pt_input)

    # TTNN forward
    tt_input = ttnn.from_torch(
        pt_input,
        dtype=ttnn.bfloat16,
        device=device,
    )

    tt_out = tt_conv(tt_input, pad_mask=None, parameters=params)

    tt_out_torch = ttnn.to_torch(tt_out)

    # Ensure shape match
    if tt_out_torch.shape != ref_out.shape:
        tt_out_torch = tt_out_torch.reshape(ref_out.shape)

    # --------------------------------------------------------
    # PCC
    # --------------------------------------------------------

    passed, msg = check_with_pcc(
        ref_out.float(),
        tt_out_torch.float(),
        pcc=0.99,
    )

    logger.info(f"ConformerConv PCC: {passed}, {msg}")

    assert passed, f"ConformerConv PCC failed: {msg}"
    assert ref_out.shape == tt_out_torch.shape, f"Shape mismatch: ref {ref_out.shape} vs tt {tt_out_torch.shape}"
