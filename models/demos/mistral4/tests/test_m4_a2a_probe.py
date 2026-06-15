# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""A10 feasibility probe: does ttnn.all_to_all_dispatch run + verify on the 1x8 mesh for the
Mistral-Small-4 MoE config (128 experts, 16/device, top-4)?

Reuses the validated reference runner (run_all_to_all_dispatch_test) with mistral4's expert config
on a (1,8) mesh: batch=8 (1 user/device), cluster_axis=1, experts=128, k=4. A pass means the sparse
all-to-all dispatch path is usable for the batched-serving MoE; a fail tells us the exact boundary
(cluster_axis / links / layout) before committing to the full MoE rewrite.
"""
import pytest

import ttnn
from tests.nightly.t3000.ccl.test_all_to_all_dispatch import run_all_to_all_dispatch_test


@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 30000000}],
    indirect=True,
)
def test_m4_a2a_probe(mesh_device):
    run_all_to_all_dispatch_test(
        mesh_device,
        mesh_shape=(1, 8),
        batch=8,
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
