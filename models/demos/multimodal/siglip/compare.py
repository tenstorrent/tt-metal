# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
from argparse import ArgumentParser

import numpy as np
import torch
from loguru import logger


def comp_pcc(golden, calculated, pcc=0.99, rtol=1e-05, atol=1e-04):
    golden = torch.Tensor(golden)
    calculated = torch.Tensor(calculated)

    fallback_rtol = rtol
    sixteen_bit_eps = [
        torch.finfo(dtype).eps for dtype in (golden.dtype, calculated.dtype) if dtype in (torch.bfloat16, torch.float16)
    ]
    if sixteen_bit_eps:
        fallback_rtol = max(rtol, 4 * max(sixteen_bit_eps))

    if golden.dtype != calculated.dtype:
        calculated = calculated.type(golden.dtype)

    if torch.all(torch.isnan(golden)) and torch.all(torch.isnan(calculated)):
        logger.warning("Both tensors are 'nan'")
        return True, 1.0

    if torch.all(torch.isnan(golden)) or torch.all(torch.isnan(calculated)):
        logger.error("One tensor is all nan, the other is not.")
        return False, 0.0

    # Test if either is completely zero — but a zero tensor is also a constant tensor,
    # so fall back to allclose instead of a hard 0.0: zero-vs-small-constant may be
    # within the caller's tolerances.
    if torch.any(golden.bool()) != torch.any(calculated.bool()):
        logger.warning("One tensor is all zero. PCC undefined; falling back to allclose.")
        result = torch.allclose(golden, calculated, rtol=fallback_rtol, atol=atol)
        return result, float(result)

    # Mask all infs and nans so that we check the rest
    golden = golden.clone()
    golden[
        torch.logical_or(
            torch.isnan(golden),
            torch.logical_or(torch.isinf(golden), torch.isneginf(golden)),
        )
    ] = 0
    calculated = calculated.clone()
    calculated[
        torch.logical_or(
            torch.isnan(calculated),
            torch.logical_or(torch.isinf(calculated), torch.isneginf(calculated)),
        )
    ] = 0

    if torch.equal(golden, calculated):
        return True, 1.0

    # Single element or constant tensor: PCC is undefined. Fall back to allclose.
    if golden.numel() == 1 or torch.max(golden) == torch.min(golden) or torch.max(calculated) == torch.min(calculated):
        logger.warning("One or both tensors are constant (zero std dev). PCC undefined; falling back to allclose.")
        result = torch.allclose(golden, calculated, rtol=fallback_rtol, atol=atol)
        return result, float(result)

    if golden.dtype == torch.bfloat16:
        golden = golden.type(torch.float32)
        calculated = calculated.type(torch.float32)
    cal_pcc = np.ma.corrcoef(
        np.ma.masked_invalid(torch.squeeze(golden).detach().numpy()).flatten(),
        np.ma.masked_invalid(torch.squeeze(calculated).detach().numpy()).flatten(),
    )[0, 1]

    if isinstance(cal_pcc, np.ma.core.MaskedConstant) or np.isnan(float(cal_pcc)):
        logger.warning("PCC returned NaN/masked. Falling back to allclose.")
        result = torch.allclose(golden, calculated, rtol=fallback_rtol, atol=atol)
        return result, float(result)

    cal_pcc = float(cal_pcc)
    return cal_pcc >= pcc, cal_pcc


def compare(a, b):
    model = torch.load(a, weights_only=False)
    test = torch.load(b, weights_only=False)

    return comp_pcc(model, test)[1]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("a", type=str)
    parser.add_argument("b", type=str)
    args = parser.parse_args()
    print(compare(args.a, args.b))
