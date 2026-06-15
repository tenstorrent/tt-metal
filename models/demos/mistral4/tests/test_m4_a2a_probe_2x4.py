# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""A10 step 1: does ttnn.all_to_all_dispatch run on a 2x4 mesh for the Mistral-Small-4 MoE config?
The flat 1x8 mesh shards all 128 experts across 8 devices (expert-parallel, tokens replicated) -> no
free axis to batch-shard tokens for dispatch. A 2x4 remap gives a 2-way data-parallel axis (tokens) x
4-way expert-parallel axis (32 experts/device), the deepseek DP x EP topology dispatch needs. This
probe (reusing the validated reference runner) confirms dispatch works on 2x4 + the cluster axis,
before the _forward_sparse rewrite."""
import pytest

import ttnn
from tests.nightly.t3000.ccl.test_all_to_all_dispatch import run_all_to_all_dispatch_test


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 30000000}],
    indirect=True,
)
def test_m4_a2a_probe_2x4(mesh_device):
    # experts on the 4-way EP axis (cluster_axis=1, 32 experts/device); tokens batch-sharded over the
    # 2-way DP axis. batch=16 -> 8 tokens per DP row.
    run_all_to_all_dispatch_test(
        mesh_device,
        mesh_shape=(2, 4),
        batch=16,
        experts=128,
        select_experts_k=4,
        hidden_size=4096,
        seq_len=1,
        num_iters=1,
        warmup_iters=0,
        trace_mode=False,
        num_links=1,
        scheme="random",
        cluster_axis=1,
        dtype=ttnn.bfloat16,
    )
