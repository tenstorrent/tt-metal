# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Perf-NEUTRALITY check for topk_router_gpt across the reduce-partial-scaler branch.

Separate from bench_reduce_partial_suite.py for two reasons:

  1. It needs its own device fixture (dispatch_core_axis=ROW), which would force a
     device teardown/reopen in the middle of the suite's profiling session.
  2. It is not a win candidate. The branch's change here is a pure REFACTOR of two
     hand-rolled reduce blocks onto compute_kernel_lib::reduce with
     ReduceInputBlockShape::single() -- no partial scaler is used at all. The op also
     hardcodes B=32 / N=128 and requires K % 32 == 0, so there is no ragged reduce
     dimension to exercise in the first place.

So the question here is "did the refactor cost anything?", not "how much did it win?".
A delta inside the ~3% noise band is the PASS condition, not a null result.

    scripts/run_safe_pytest.sh --profile --run-all \
        tests/ttnn/unit_tests/operations/moreh_mean_perf/bench_topk_router_gpt.py
"""

import pytest
import torch

import ttnn

# (id, K) -- B=32 and N=128 are hardcoded requirements of the fused op.
B, N, TOP_K = 32, 128, 4
CASES = [
    ("topk_router_k64", 64),  # small hidden_dim edge case
    ("topk_router_k2880", 2880),  # production shape
    ("topk_router_k4096", 4096),  # large hidden_dim
]


@pytest.mark.parametrize(
    "device_params",
    [pytest.param({"dispatch_core_axis": ttnn.DispatchCoreAxis.ROW}, id="dispatch_row")],
    indirect=True,
)
@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_topk_router_bench(case, device):
    case_id, k = case
    torch.manual_seed(2024)

    torch_input = torch.rand(B, k, dtype=torch.bfloat16)
    torch_weight = torch.rand(k, N, dtype=torch.bfloat16)
    torch_bias = torch.arange(N, dtype=torch.float32).unsqueeze(0).to(torch.bfloat16)
    torch_bias_bcast = torch_bias.expand(B, N).contiguous()

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    tt_weight = ttnn.from_torch(torch_weight, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    tt_bias = ttnn.from_torch(torch_bias_bcast, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

    ttnn.experimental.topk_router_gpt(
        tt_input,
        weight_tensor=tt_weight,
        bias_tensor=tt_bias,
        k=TOP_K,
        num_experts=N,
    )
