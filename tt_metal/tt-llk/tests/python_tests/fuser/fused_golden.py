# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
from helpers.llk_params import format_dict
from helpers.logger import logger
from helpers.utils import passed_test

from .fused_operation import FusedOperation
from .fuser_config import FuserConfig


class FusedGolden:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def _check_output(self, output) -> bool:
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

        logger.info(f"L1 golden check for {output.name}:")
        l1_passed = passed_test(
            l1_golden,
            res_tensor,
            output.data_format,
            print_pcc=True,
            custom_atol=0.1,
            custom_rtol=0.1,
        )
        logger.info(f"Master golden check for {output.name}:")
        master_passed = passed_test(
            master_golden,
            res_tensor,
            output.data_format,
            print_pcc=True,
            custom_atol=0.1,
            custom_rtol=0.1,
        )

        return l1_passed and master_passed

    def check_operation(self, operation: FusedOperation) -> bool:
        if self.verbose:
            logger.info(f"{operation}")

        passed = True
        for pack_node in operation.math.pack_nodes:
            if not self._check_output(pack_node.output):
                passed = False

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
