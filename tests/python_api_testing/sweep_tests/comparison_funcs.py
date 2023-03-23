import torch
import numpy as np
from loguru import logger


def comp_equal(golden, calculated):
    max_diff = torch.max(torch.abs(golden - calculated)).item()
    return torch.equal(golden, calculated), f"Max Absolute Delta: {max_diff}"


def comp_allclose(golden, calculated, rtol=1e-05, atol=1e-08):
    atol_delta = torch.max(torch.abs(golden - calculated)).item()
    rtol_delta = torch.max(torch.abs(golden - calculated) / calculated).item()
    return (
        torch.allclose(golden, calculated, rtol, atol, True),
        f"Max ATOL Delta: {atol_delta}, Max RTOL Delta: {rtol_delta}",
    )


def comp_pcc(golden, calculated, pcc=0.99):
    if torch.all(torch.isnan(golden)) and torch.all(torch.isnan(calculated)):
        logger.warning("Both tensors are 'nan'")
        return True, f"PCC: {1.0}"

    if torch.all(torch.isnan(golden)) or torch.all(torch.isnan(calculated)):
        logger.error("One tensor is all nan, the other is not.")
        return False, f"PCC: {0.0}"

    # Test if either is completely zero
    if torch.any(golden.bool()) != torch.any(calculated.bool()):
        logger.error("One tensor is all zero")
        return False, f"PCC: {0.0}"

    # if torch.any(torch.isinf(golden)) or torch.any(torch.isinf(calculated)):
    #    raise RuntimeError(f"Tensor overflow to infinity: \n{golden}\n{calculated}")

    # if torch.any(torch.isneginf(golden)) or torch.any(torch.isneginf(calculated)):
    #    raise RuntimeError(f"Tensor overflow to negative infinity: \n{golden}\n{calculated}")

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
        return True, f"PCC: {1.0}"

    if golden.dtype == torch.bfloat16:
        golden = golden.type(torch.float16)
        calculated = calculated.type(torch.float16)
    cal_pcc = np.min(
        np.ma.corrcoef(
            np.ma.masked_invalid(torch.squeeze(golden).detach().numpy()).flatten(),
            np.ma.masked_invalid(torch.squeeze(calculated).detach().numpy()).flatten(),
        )
    )

    if isinstance(pcc, np.ma.core.MaskedConstant):
        return True, f"PCC: {1.0}"

    return cal_pcc >= pcc, f"PCC: {cal_pcc}"


def comp_allclose_and_pcc(golden, calculated, rtol=1e-05, atol=1e-08, pcc=0.99):
    passing = True
    output = ""
    passing_allclose, output_allclose = comp_allclose(golden, calculated, rtol, atol)
    passing &= passing_allclose
    output += output_allclose
    if torch.numel(golden) != 1:
        passing_pcc, output_pcc = comp_pcc(golden, calculated, pcc)
        passing &= passing_pcc
        output += f", {output_pcc}"

    return passing, output
