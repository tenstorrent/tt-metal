# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch

from .fused_operation import FusedOperation
from .fuser_config import FuserConfig
from .llk_params import format_dict
from .logger import logger
from .utils import passed_test


class FusedGolden:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results = []

    def check_operation(self, operation: FusedOperation) -> bool:
        output = operation.output

        if self.verbose:
            logger.info("{}", operation)

        res_tensor = torch.tensor(
            output.raw_data, dtype=format_dict[output.data_format]
        )
        l1_golden = torch.tensor(
            output.l1_golden, dtype=format_dict[output.data_format]
        )
        master_golden = torch.tensor(
            output.master_golden, dtype=format_dict[output.data_format]
        )

        res_tensor = res_tensor.flatten()
        l1_golden = l1_golden.flatten()
        master_golden = master_golden.flatten()

        logger.info("L1 golden check:")
        l1_passed = passed_test(
            l1_golden, res_tensor, output.data_format, print_pcc=True
        )
        logger.info("Master golden check:")
        master_passed = passed_test(
            master_golden, res_tensor, output.data_format, print_pcc=True
        )

        passed = l1_passed and master_passed

        if self.verbose:
            if passed:
                logger.success("PASS")
            else:
                logger.error("FAIL")

        return passed

    def check_pipeline(self, config: FuserConfig) -> bool:
        result = True
        for operation in config.pipeline:
            operation.golden(config.global_config)
            passed = self.check_operation(operation)
            if not passed:
                result = False

        return result
