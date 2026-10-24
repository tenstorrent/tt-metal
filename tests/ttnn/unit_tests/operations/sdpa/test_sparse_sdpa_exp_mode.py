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


@run_for_blackhole()
def test_sparse_sdpa_accurate_mode_known_answer(device):
    """Two selected keys score 1 and 0 after the scale, with V = 1 on the first and 0 on the second, so every output
    element is exactly sigmoid(1). The modes must differ, and the accurate one must be within 0.005 of sigmoid(1)."""
    heads, sequence, head_dim, value_dim = 32, 32, 64, 32
    q = torch.zeros((1, heads, sequence, head_dim), dtype=torch.bfloat16)
    kv = torch.zeros((1, 1, 32, head_dim), dtype=torch.bfloat16)
    indices = torch.full((1, 1, sequence, 32), -1, dtype=torch.int32)
    q[..., value_dim] = 1
    kv[0, 0, 1, value_dim] = 2  # scale 0.5: key 1 scores 1, key 0 scores 0
    kv[0, 0, 1, :value_dim] = 1
    indices[..., 0], indices[..., 1] = 0, 1

    tt_q = to_dev(q, device, ttnn.bfloat16)
    tt_kv = to_dev(kv, device, ttnn.bfloat16)
    tt_indices = to_dev(indices, device, ttnn.uint32)
    outputs = {}
    for approx in (False, True):
        compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=approx,
            fp32_dest_acc_en=True,
        )
        tt_out = ttnn.transformer.sparse_sdpa(
            tt_q,
            tt_kv,
            tt_indices,
            value_dim,
            kv_format=ttnn.transformer.SparseKVFormat.BF16,
            scale=0.5,
            k_chunk_size=32,
            compute_kernel_config=compute_config,
        )
        outputs[approx] = ttnn.to_torch(tt_out).float()

    assert not torch.equal(outputs[False], outputs[True]), "math_approx_mode False and True returned identical outputs"
    error = (outputs[False] - torch.sigmoid(torch.tensor(1.0))).abs().max().item()
    assert error <= 0.005, f"accurate mode is {error:.5f} from sigmoid(1), above 0.005"
