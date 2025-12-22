# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


from loguru import logger
import torch
from tests.ttnn.utils_for_testing import assert_with_pcc

MOLE_L1_SMALL_REGION_SIZE = 24576
MOLE_FULL_MODEL_PCC = 0.9995
MOLE_RTOL_LIMIT = 5e-2
MOLE_ATOL_LIMIT = 8e-2


def verify_with_pcc(torch_tensor, ttnn_tensor, pcc):
    _, computed_pcc = assert_with_pcc(torch_tensor, ttnn_tensor, pcc)
    logger.info(f"PCC check was successful ({computed_pcc:.6f} > {pcc:.6f})")
    if (computed_pcc - pcc) / pcc > 0.0025:
        logger.warning(
            f"Computed PCC ({computed_pcc:.6f}) was higher than the expected PCC ({pcc:.6f}) - consider updating the expected PCC value"
        )


def verify_with_errlimit(output_torch_tensor, output_ttnn_tensor, rtol, atol):
    try:
        torch.testing.assert_close(output_torch_tensor, output_ttnn_tensor, rtol=rtol, atol=atol, equal_nan=False)
    except AssertionError as e:
        logger.error("Mismatch details:\n", e)
        assert False
    logger.info(f"error check was successful with atol_limit {atol}, rtol_limit {rtol}")
