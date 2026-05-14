# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
from helpers.llk_params import DataFormat, format_dict
from helpers.logger import logger
from helpers.utils import passed_test, tolerances

from .fused_operation import FusedOperation
from .fuser_config import FuserConfig

DEFAULT_BASE_ATOL = 0.05
DEFAULT_BASE_RTOL = 0.05
DEFAULT_BASE_PCC = 0.99


class FusedGolden:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results = []

    _FORMAT_PCC = {DataFormat.Bfp4_b: 0.98, DataFormat.MxFp4: 0.95}

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
            l1_golden,
            res_tensor,
            output.data_format,
            print_pcc=True,
            custom_atol=0.1,
            custom_rtol=0.1,
        )

        if self.strict_golden:
            logger.info("Master golden check (strict):")
            master_passed = passed_test(
                master_golden, res_tensor, output.data_format, print_pcc=True
            )
        else:
            logger.info(
                f"Master golden check (stage {operation.stage_id}, format {output.data_format.name},"
                f"atol {output.acc_atol:.2f}, rtol {output.acc_rtol:.2f}, pcc {output.acc_pcc:.2f}):"
            )
            master_passed = passed_test(
                master_golden,
                res_tensor,
                output.data_format,
                print_pcc=True,
                custom_atol=output.acc_atol,
                custom_rtol=output.acc_rtol,
                custom_pcc_threshold=output.acc_pcc,
            )

        passed = l1_passed and master_passed

        if self.verbose:
            if passed:
                logger.success("PASS")
            else:
                logger.error("FAIL")

        return passed

    @staticmethod
    def _accumulate_tolerance(operation: FusedOperation) -> None:
        sources = []
        seen = set()
        for node in operation.math.operations:
            for src in (node.src_a, node.src_b):
                if src is not None and id(src) not in seen:
                    seen.add(id(src))
                    sources.append(src)

        max_input_rtol = max((s.acc_rtol for s in sources), default=0.0)
        max_input_atol = max((s.acc_atol for s in sources), default=0.0)

        base_tol = tolerances.get(operation.output.data_format)
        base_rtol = base_tol.rtol if base_tol else DEFAULT_BASE_RTOL
        base_atol = base_tol.atol if base_tol else DEFAULT_BASE_ATOL

        min_input_pcc = min((s.acc_pcc for s in sources), default=1.0)
        base_pcc = FusedGolden._FORMAT_PCC.get(
            operation.output.data_format, DEFAULT_BASE_PCC
        )

        operation.output.acc_rtol = (1 + max_input_rtol) * (1 + base_rtol) - 1
        operation.output.acc_atol = max_input_atol * (1 + base_rtol) + base_atol
        operation.output.acc_pcc = min_input_pcc * base_pcc

    def check_pipeline(self, config: FuserConfig) -> bool:
        self.strict_golden = config.global_config.strict_golden
        result = True
        for operation in config.pipeline:
            operation.golden(config.global_config)
            self._accumulate_tolerance(operation)
            passed = self.check_operation(operation)
            if not passed:
                result = False

        return result
