# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List

import torch

from .fused_operation import FusedOperation
from .llk_params import format_dict
from .utils import passed_test


class FusedGolden:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results = []

    def check_operation(self, operation: FusedOperation, step_number: int) -> bool:
        src_a = operation.src_a
        src_b = operation.src_b
        output = operation.output

        if self.verbose:
            print(operation)

        res_tensor = torch.tensor(
            output.raw_data, dtype=format_dict[output.data_format]
        )
        l1_golden = torch.tensor(
            output.l1_golden, dtype=format_dict[output.data_format]
        )
        master_golden = torch.tensor(
            output.master_golden, dtype=format_dict[output.data_format]
        )

        res_tensor = res_tensor.reshape(operation.output_pack_dims)
        l1_golden = l1_golden.reshape(operation.output_pack_dims)
        master_golden = master_golden.reshape(operation.output_pack_dims)

        print("L1 golden check:")
        l1_passed = passed_test(
            l1_golden, res_tensor, output.data_format, print_pcc=True
        )
        print("Master golden check:")
        master_passed = passed_test(
            master_golden, res_tensor, output.data_format, print_pcc=True
        )

        passed = l1_passed and master_passed

        result = {
            "step": step_number,
            "operation": str(operation.math.__class__.__name__),
            "src_a": src_a.name,
            "src_b": src_b.name,
            "output": output.name,
            "passed": passed,
        }
        self.results.append(result)

        if self.verbose:
            print("✓ PASS") if passed else print("✗ FAIL")

        return passed

    def check_pipeline(self, pipeline: List[FusedOperation]) -> bool:
        result = True
        for i, operation in enumerate(pipeline, start=1):
            operation.golden()
            passed = self.check_operation(operation, i)
            if not passed:
                result = False

        return result

    def get_results(self) -> List[dict]:
        return self.results.copy()
