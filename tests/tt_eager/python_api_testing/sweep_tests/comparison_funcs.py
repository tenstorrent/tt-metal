# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np
from loguru import logger


def get_atol_rtol_pcc(golden, calculated):
    if golden.dtype != calculated.dtype:
        logger.warning(f"Converting calculated to golden.dtype {golden.dtype}")
        calculated = calculated.type(golden.dtype)

    is_calculate_float = calculated.is_floating_point()
    calculated_dtype = calculated.dtype

    if golden.is_complex() and calculated.is_complex():
        golden = torch.view_as_real(golden.clone())
        calculated = torch.view_as_real(calculated.clone())

    if not (golden.is_floating_point() or calculated.is_floating_point()):
        logger.warning(f"Converting input tensors to floating point")
        golden = golden.to(torch.float)
        calculated = calculated.to(torch.float)

    # Calculate atol and rtol
    cal_atol = torch.max(torch.abs(golden - calculated)).item()
    cal_rtol = torch.max(torch.abs(golden - calculated) / torch.abs(calculated)).item()

    # Calculate PCC
    def get_pcc(golden, calculated):
        # Both tensors are nan
        if torch.all(torch.isnan(golden)) and torch.all(torch.isnan(calculated)):
            logger.warning("Both tensors are 'nan'")
            return 1.0

        # One tensor is all nan, the other is not
        if torch.all(torch.isnan(golden)) or torch.all(torch.isnan(calculated)):
            logger.error("One tensor is all nan, the other is not.")
            return 0.0

        # One tensor is all zero, the other is not
        if torch.any(golden.bool()) != torch.any(calculated.bool()):
            logger.warning("One tensor is all zero")
            return 0.0

        if torch.any(torch.isnan(golden)):
            logger.warning(f"Tensor golden has NaNs")

        if torch.any(torch.isposinf(golden)):
            logger.warning(f"Tensor golden has infinity values")

        if torch.any(torch.isneginf(golden)):
            logger.warning(f"Tensor golden has negative infinity values")

        if torch.any(torch.isnan(calculated)):
            logger.warning(f"Tensor calculated has NaNs")

        if torch.any(torch.isposinf(calculated)):
            logger.warning(f"Tensor calculated has infinity values")

        if torch.any(torch.isneginf(calculated)):
            logger.warning(f"Tensor calculated has negative infinity values")

        # For now, mask all infs and nans so that we check the rest... TODO
        golden = golden.clone()
        calculated = calculated.clone()

        golden[torch.isnan(golden)] = 0
        calculated[torch.isnan(calculated)] = 0

        if is_calculate_float:
            golden[torch.isposinf(golden)] = torch.finfo(calculated_dtype).max
            calculated[torch.isposinf(calculated)] = torch.finfo(calculated_dtype).max

            golden[torch.isneginf(golden)] = torch.finfo(calculated_dtype).min
            calculated[torch.isneginf(calculated)] = torch.finfo(calculated_dtype).min

        if torch.equal(golden, calculated):
            return 1.0

        if golden.dtype == torch.bfloat16:
            golden = golden.type(torch.float32)
            calculated = calculated.type(torch.float32)

        # Single element case
        if golden.numel() == 1:
            return float(torch.equal(golden, calculated))

        # one tensor is constant
        if torch.max(golden) == torch.min(golden) or torch.max(calculated) == torch.min(calculated):
            return float(torch.equal(golden, calculated))

        cal_pcc = np.ma.corrcoef(
            np.ma.masked_invalid(torch.squeeze(golden).detach().numpy()).flatten(),
            np.ma.masked_invalid(torch.squeeze(calculated).detach().numpy()).flatten(),
        )
        # Remove correlation coefficient with self (typically always 1.0)
        mask = np.ones(cal_pcc.shape, dtype=bool)
        np.fill_diagonal(mask, 0)
        cal_pcc = np.min(cal_pcc[mask])

        if isinstance(cal_pcc, np.ma.core.MaskedConstant):
            return 1.0

        return cal_pcc

    cal_pcc = get_pcc(golden, calculated)

    return (
        cal_atol,
        cal_rtol,
        cal_pcc,
        f"Max ATOL Delta: {cal_atol}, Max RTOL Delta: {cal_rtol}, PCC: {cal_pcc}",
    )


def comp_equal(golden, calculated):
    while len(golden.shape) < len(calculated.shape):
        golden = torch.unsqueeze(golden, 0)

    _, _, _, output_str = get_atol_rtol_pcc(golden, calculated)
    equal = torch.equal(golden, calculated)

    if not equal:
        output_str += ", Equal check failed"

    return equal, output_str


def comp_shape(golden, calculated):
    output_str = "compare shape"
    equal = golden.shape == calculated.shape
    return equal, output_str


def comp_allclose(golden, calculated, rtol=1e-05, atol=1e-08):
    _, _, _, output_str = get_atol_rtol_pcc(golden, calculated)
    passing = torch.allclose(golden, calculated.type(golden.dtype), rtol, atol, True)
    if not passing:
        output_str += ", Allclose check failed"
    return passing, output_str


def comp_pcc(golden, calculated, pcc=0.99):
    _, _, cal_pcc, output_str = get_atol_rtol_pcc(golden, calculated)
    passing = cal_pcc >= pcc
    if not passing:
        output_str += ", PCC check failed"
    return passing, output_str


def comp_pcc_list(golden, calculated, pcc=0.99):
    total_str = ""
    min_pcc = 1

    for i in range(len(golden)):
        _, _, cal_pcc, output_str = get_atol_rtol_pcc(golden[i], calculated[i])

        total_str = f"{total_str}Tensor {i}: {output_str} "

        if cal_pcc < min_pcc:
            min_pcc = cal_pcc

    passing = min_pcc >= pcc
    if not passing:
        total_str += ", PCC check failed"
    return passing, total_str


def comp_allclose_and_pcc(golden, calculated, rtol=1e-05, atol=1e-08, pcc=0.99):
    _, _, cal_pcc, output_str = get_atol_rtol_pcc(golden, calculated)
    passing = True
    allclose_passing = torch.allclose(golden, calculated.type(golden.dtype), rtol, atol, True)
    passing &= allclose_passing
    if not allclose_passing:
        output_str += ", Allclose check failed"
    pcc_passing = cal_pcc >= pcc
    passing &= pcc_passing
    if not pcc_passing:
        output_str += ", PCC check failed"
    return passing, output_str


def comp_using_plot(tname, input, golden, calculated):
    import matplotlib.pyplot as plt

    if golden.dtype == torch.bfloat16:
        golden = golden.type(torch.float32)
        calculated = calculated.type(torch.float32)
        input = input.type(torch.float32)
    shape = "x".join(list(map(str, list(input.size()))))
    plot_name = "plot_" + tname + "_" + shape + ".png"
    input = input.flatten()
    golden = golden.flatten()
    calculated = calculated.flatten()
    plt.plot(input, golden, "+r", label="CPU (golden)")
    plt.plot(input, calculated, "-b", label="On device (calculated)")
    plt.legend(loc="upper center")
    plt.savefig(plot_name)
    plt.close()
