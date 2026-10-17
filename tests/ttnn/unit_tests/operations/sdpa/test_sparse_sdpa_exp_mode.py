# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""sparse_sdpa must honor math_approx_mode=False for its P exponential (Blackhole only).

The op defaults to the approximate exponential and only callers that ask for math_approx_mode=False expect the accurate
one. The compute kernel used to run the approximate exponential for the main scores whatever the flag said, even though
the later online correction observed the flag. This test checks that the modes differ and that the accurate one is
closer to an FP32 reference. FP32 destination accumulation must also meet the full-tile standard SDPA exp gate.
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from tests.ttnn.unit_tests.operations.sdpa.sparse_sdpa_test_utils import golden, make_inputs, to_dev

K_DIM = 576  # head dim (q/kv width)
V_DIM = 512  # V width / output width (op arg)
ACCURATE_NL2_MAX = 0.01  # same normalized-L2 gate as test_sdpa_standard_exp.py

pytestmark = pytest.mark.use_module_device


def _normalized_l2(expected, actual):
    expected, actual = expected.double().flatten(), actual.double().flatten()
    return (torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected)).item()


# FP32 destination accumulation and BF16 destination accumulation select different accurate exponentials, and the two
# scales select the two ways the kernel applies the scale: 1/16 is exactly representable in BF16 (applied inside the
# accurate exponential), 1/sqrt(576) = 1/24 is not (multiplied in FP32 first).
@run_for_blackhole()
@pytest.mark.parametrize("fp32_dest_acc_en", [True, False], ids=["fp32-dest", "bf16-dest"])
@pytest.mark.parametrize("scale", [0.0625, K_DIM**-0.5], ids=["scale-bf16-exact", "scale-bf16-inexact"])
def test_sparse_sdpa_honors_math_approx_mode(device, fp32_dest_acc_en, scale):
    H, S, T, TOPK, k_chunk_size = 32, 64, 256, 64, 32
    q, kv, indices = make_inputs(H, S, T, TOPK, K_DIM, lambda s: TOPK, seed=7)
    q, kv = q.to(torch.bfloat16), kv.to(torch.bfloat16)  # reference and device see the same BF16 values
    reference = golden(q.float(), kv.float(), indices, scale, V_DIM)

    tt_q = to_dev(q, device, ttnn.bfloat16)
    tt_kv = to_dev(kv, device, ttnn.bfloat16)
    tt_indices = to_dev(indices.to(torch.int32), device, ttnn.uint32)
    outputs = {}
    for approx in (False, True):
        compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=approx,
            fp32_dest_acc_en=fp32_dest_acc_en,
            packer_l1_acc=False,
        )
        tt_out = ttnn.transformer.sparse_sdpa(
            tt_q,
            tt_kv,
            tt_indices,
            V_DIM,
            kv_format=ttnn.transformer.SparseKVFormat.BF16,
            scale=scale,
            k_chunk_size=k_chunk_size,
            compute_kernel_config=compute_config,
        )
        outputs[approx] = ttnn.to_torch(tt_out)
        assert torch.isfinite(outputs[approx]).all()

    assert not torch.equal(outputs[False], outputs[True]), "math_approx_mode False and True returned identical outputs"
    accurate = _normalized_l2(reference, outputs[False].float())
    approximate = _normalized_l2(reference, outputs[True].float())
    assert (
        accurate < approximate
    ), f"accurate {accurate:.5f} is not closer to the reference than approximate {approximate:.5f}"
    if fp32_dest_acc_en:
        assert accurate <= ACCURATE_NL2_MAX, f"accurate mode normalized L2 {accurate:.5f} exceeds {ACCURATE_NL2_MAX}"
