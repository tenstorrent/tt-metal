# Scratch CI only (llk_analysis #58786): runs the SRAM expert accum tests and prints a sha256 of the SRAM output for a
# bitwise compare of banked against one-bank production calls between two CI builds.
import hashlib
import os

import torch
import ttnn

import models.demos.deepseek_v3_b1.tests.unit_tests.per_core_allocation.test_matmul_expert as t

_validate = t._validate_sram_output_accum


def _validate_and_save(sram_out_tensor, *args, **kwargs):
    node = os.environ.get("PYTEST_CURRENT_TEST", "unknown").split(" ")[0].split("::")[-1]
    outs = [ttnn.to_torch(d) for d in ttnn.get_device_tensors(sram_out_tensor)]
    h = hashlib.sha256(b"".join(o.contiguous().view(torch.int16).numpy().tobytes() for o in outs)).hexdigest()
    print(f"BKHASH {node} {h}")
    return _validate(sram_out_tensor, *args, **kwargs)


t._validate_sram_output_accum = _validate_and_save

from models.demos.deepseek_v3_b1.tests.unit_tests.per_core_allocation.test_matmul_expert import (  # noqa: E402,F401
    test_hybrid_expert_single_device_accum_experts_plain,
    test_hybrid_expert_single_device_sparse_accum_experts,
)
