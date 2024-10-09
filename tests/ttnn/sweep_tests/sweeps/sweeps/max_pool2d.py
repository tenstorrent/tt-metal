# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch
from tests.ttnn.utils_for_testing import check_with_pcc
import ttnn


parameters = {
    "act_shape": [  ## NCHW
        [1, 64, 112, 112],
        [4, 64, 112, 112],
        [8, 64, 112, 112],
        [16, 64, 112, 112],
    ],
    "kernel_size": ((3, 3),),
    "padding": ((1, 1),),
    "stride": ((2, 2),),
    "dilation": ((1, 1),),  ## default
    "nblocks": (1,),
    "dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
}


def skip(
    act_shape,
    kernel_size,
    padding,
    stride,
    dilation,
    nblocks,
    dtype,
) -> Tuple[bool, Optional[str]]:
    in_n, in_c, in_h, in_w = act_shape
    kernel_h, kernel_w = kernel_size
    pad_h, pad_w = padding

    if act_shape[0] >= 16 and dtype == ttnn.bfloat16:
        return True, "Configuration does not fit in L1"

    if 2 * pad_h > kernel_h or 2 * pad_w > kernel_w:
        return True, "Invalid case"

    if nblocks != 1:
        return True, f"Unsupported case: nblocks = {nblocks}"

    if in_c != 64:
        return True, "Current maxpool writer needs nchannels to be 64!"

    return False, None


def run(
    act_shape,
    kernel_size,
    padding,
    stride,
    dilation,
    nblocks,
    dtype,
    device,
) -> Tuple[bool, Optional[str]]:
    in_n, in_c, in_h, in_w = act_shape
    kernel_h, kernel_w = kernel_size
    pad_h, pad_w = padding
    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation

    torch.manual_seed(0)
    torch.set_printoptions(precision=3, sci_mode=False, linewidth=500, threshold=10000, edgeitems=32)

    ## construct the tensor in NCHW shape
    act = torch.randn(act_shape, dtype=torch.bfloat16)
    ## this op expects input tensor as { N, 1, H * W, C }, so rearrange and reshape tensor
    ## but before that, make sure in_c is multiple of tile width
    act_shape = (in_n, 1, in_h * in_w, in_c)
    act_permuted = torch.permute(act, (0, 2, 3, 1))
    act_reshaped = act_permuted.reshape(act_shape)

    if dtype == ttnn.bfloat8_b:
        ttact = ttnn.from_torch(act_reshaped, dtype, layout=ttnn.TILE_LAYOUT)
    else:
        ttact = ttnn.from_torch(act_reshaped, dtype)
    ttact_d = ttnn.to_device(ttact, device)
    out_d = ttnn.max_pool2d(
        input_tensor=ttact_d,
        batch_size=in_n,
        input_h=in_h,
        input_w=in_w,
        channels=in_c,
        kernel_size=[kernel_h, kernel_w],
        stride=[stride_h, stride_w],
        padding=[pad_h, pad_w],
        dilation=[dilation_h, dilation_w],
    )
    out_padded = out_d.cpu()
    out_pytorch_padded = ttnn.to_torch(out_padded)
    out_pytorch = out_pytorch_padded[:, :, :, :in_c]
    out_pytorch = torch.permute(out_pytorch, (0, 3, 1, 2))  ## N, C, 1, HW

    ## reference
    golden_pytorch = torch.nn.MaxPool2d(
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        return_indices=False,
        ceil_mode=False,
    )(act)

    ## test for equivalance
    out_pytorch = out_pytorch.reshape(golden_pytorch.shape)
    return check_with_pcc(out_pytorch, golden_pytorch)
