# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""What does a read leave in L1, as a function of the sealed set's depth? (#54876)

A layer stack dies at four or more sealed snapshots, when a full-attention layer cannot place its
circular buffers — 192 bytes over, at the top of L1. Reading the code did not find the tenant:
`attn_res_gather_softmax`'s per-core L1 is a function of `Wt` and `ring_size`, `inter_block`'s
`partials` contract the candidate axis away, the sealed set is DRAM, and disabling the statistics
fold changes nothing. So measure it instead.

This is deliberately an OP-level probe rather than a model one. It needs no weights and no layer
stack, so it costs about a minute where a model run costs nine, and it isolates AttnRes: whatever it
reports is what AttnRes alone is holding when the next op goes to place its kernels.

Reported per depth:
  * bytes still allocated in L1 after the block's batches are built and its reads issued
  * the increment over depth 1, which is what a fix has to flatten

A flat curve means AttnRes's L1 tenancy is depth-independent and the clash lives elsewhere. A curve
that grows with depth names the culprit and bounds what depth 8 will need.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.kimi_k3.attn_res.attn_res import EPS
from models.demos.deepseek_v3_d_p.tests.attn_res.model.harness import place_case, random_case
from models.demos.deepseek_v3_d_p.tt.attn_res.attn_res import TtAttnRes
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology

HIDDEN_SIZE = 7168
TP_AXIS = 1
NUM_TOKENS = 5120
# 1..3 are the depths every shipped configuration has ever reached; 4 is where the stack breaks;
# 8 is the deepest the 93-layer model goes.
DEPTHS = [1, 2, 3, 4, 6, 8]
# One block's worth of read sites, so the batches are the size a real walk holds.
READ_SITES = 24

PLACEMENTS = [
    pytest.param(
        (8, 4),
        {"fabric_config": ttnn.FabricConfig.FABRIC_2D, "l1_small_size": 4096},
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
        id="fabric2d-8x4",
    )
]


def _l1_bytes(mesh_device):
    view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.L1)
    return view.total_bytes_allocated_per_bank * view.num_banks


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mesh_device, device_params", PLACEMENTS, indirect=True)
def test_l1_tenancy_by_sealed_depth(mesh_device, device_params):
    op = TtAttnRes(
        mesh_device,
        hidden_size=HIDDEN_SIZE,
        eps=EPS,
        tp_axis=TP_AXIS,
        topology=per_axis_topology(),
    )
    generator = torch.Generator().manual_seed(0)
    queries = [op.to_query(torch.randn(HIDDEN_SIZE, generator=generator)) for _ in range(READ_SITES)]

    baseline = None
    for depth in DEPTHS:
        running_sum, block_residual = random_case(generator, NUM_TOKENS, depth)
        tt_running, tt_block = place_case(op, running_sum, block_residual)

        before = _l1_bytes(mesh_device)
        partials, shifts, masses = op.inter_block(tt_block, queries)
        # Issue a read, so anything the fused op leaves resident is counted too.
        merged = op.merge(partials, shifts, masses, tt_running, queries[0], 0)
        ttnn.synchronize_device(mesh_device)
        held = _l1_bytes(mesh_device) - before

        if baseline is None:
            baseline = held
        logger.info(
            f"depth {depth}: L1 held {held / 1024:9.1f} KB   " f"delta vs depth 1 {(held - baseline) / 1024:+9.1f} KB"
        )

        for tensor in (merged, partials, shifts, masses, tt_running, tt_block):
            if tensor is not None:
                ttnn.deallocate(tensor)

    # No assertion on a bar: this is a measurement, and what it measures is whether the tenancy is
    # depth-dependent at all. The gate for a fix is test_attn_res_depth_l1.py, which runs a real
    # stack at depth 8.
