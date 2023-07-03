import torch
import numpy as np
from loguru import logger


def get_atol_rtol_pcc(golden, calculated):
    # Calculate atol and rtol
    cal_atol = torch.max(torch.abs(golden - calculated)).item()
    cal_rtol = torch.max(torch.abs(golden - calculated) / torch.abs(calculated)).item()

    # Calculate PCC
    # Both tensors are nan
    if torch.all(torch.isnan(golden)) and torch.all(torch.isnan(calculated)):
        logger.warning("Both tensors are 'nan'")
        cal_pcc = 1.0

    # One tensor is all nan, the other is not
    if torch.all(torch.isnan(golden)) or torch.all(torch.isnan(calculated)):
        logger.error("One tensor is all nan, the other is not.")
        cal_pcc = 0.0

    # One (or both) tensor is all zero
    if torch.any(golden.bool()) != torch.any(calculated.bool()):
        logger.warning("One tensor is all zero")
        cal_pcc = 0.0

    # if torch.any(torch.isinf(golden)) or torch.any(torch.isinf(calculated)):
    #    raise RuntimeError(f"Tensor overflow to infinity: \n{golden}\n{calculated}")

    # if torch.any(torch.isneginf(golden)) or torch.any(torch.isneginf(calculated)):
    #    raise RuntimeError(f"Tensor overflow to negative infinity: \n{golden}\n{calculated}")

    else:
        # For now, mask all infs and nans so that we check the rest... TODO
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
            cal_pcc = 1.0

        if golden.dtype == torch.bfloat16:
            golden = golden.type(torch.float32)
            calculated = calculated.type(torch.float32)
        cal_pcc = np.min(
            np.ma.corrcoef(
                np.ma.masked_invalid(torch.squeeze(golden).detach().numpy()).flatten(),
                np.ma.masked_invalid(
                    torch.squeeze(calculated).detach().numpy()
                ).flatten(),
            )
        )

        if isinstance(cal_pcc, np.ma.core.MaskedConstant):
            cal_pcc = 1.0

    return (
        cal_atol,
        cal_rtol,
        cal_pcc,
        f"Max ATOL Delta: {cal_atol}, Max RTOL Delta: {cal_rtol}, PCC: {cal_pcc}",
    )


def comp_equal(golden, calculated):
    if golden.dtype != calculated.dtype:
        calculated = calculated.type(golden.dtype)

    _, _, _, output_str = get_atol_rtol_pcc(golden, calculated)
    return torch.equal(golden, calculated), output_str


def comp_allclose(golden, calculated, rtol=1e-05, atol=1e-08):
    if golden.dtype != calculated.dtype:
        calculated = calculated.type(golden.dtype)

    _, _, _, output_str = get_atol_rtol_pcc(golden, calculated)
    return torch.allclose(golden, calculated, rtol, atol, True), output_str


def comp_allclose_nortol(golden, calculated, atol=1e-08):
    if golden.dtype != calculated.dtype:
        calculated = calculated.type(golden.dtype)

    return (
        torch.allclose(golden, calculated, atol=atol, equal_nan=False),
        "torch.allclose",
    )


def comp_pcc(golden, calculated, pcc=0.99):
    if golden.dtype != calculated.dtype:
        calculated = calculated.type(golden.dtype)
    _, _, cal_pcc, output_str = get_atol_rtol_pcc(golden, calculated)
    return cal_pcc >= pcc, output_str


def comp_allclose_and_pcc(golden, calculated, rtol=1e-05, atol=1e-08, pcc=0.99):
    if golden.dtype != calculated.dtype:
        calculated = calculated.type(golden.dtype)

    _, _, cal_pcc, output_str = get_atol_rtol_pcc(golden, calculated)
    passing = True
    passing &= torch.allclose(golden, calculated, rtol, atol, True)
    passing &= cal_pcc >= pcc
    return passing, output_str
