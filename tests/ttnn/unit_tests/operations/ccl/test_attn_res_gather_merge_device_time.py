# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device time for the AttnRes read, with nothing else in the profile.

Every dispatch is a row in a tracy report, so the read's own cost is only readable
where its program is the only one that runs. Tensors are tilized on host and copied,
and read back through `from_device`, so neither direction contributes a device
program; the settle arm's reference is the host sum rather than the `ttnn.add`
`test_attn_res_gather_merge.py` gates against. Selecting one `fuse_add` arm per run
keeps the report to one variant.

The call is repeated so the report carries a distribution rather than one cold
sample. Every iteration runs the same program over the same addresses, so the
one checked against torch stands for all of them.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from tests.ttnn.unit_tests.operations.ccl.test_attn_res_gather_merge import (
    EPS,
    HIDDEN_SIZE,
    INV_HIDDEN_SIZE,
    PCC,
    SP_AXIS,
    TP_AXIS,
    _oracle,
)
from tests.ttnn.utils_for_testing import assert_with_pcc

pytestmark = pytest.mark.skipif(
    not is_blackhole(), reason="attn_res_gather_merge has only been brought up on Blackhole"
)

FABRIC = {"fabric_config": ttnn.FabricConfig.FABRIC_1D}

# The production per-chip shape: 5120 tokens over an 8-long sequence axis.
PER_CHIP_TOKENS = 640

# Tracy drops device data past 1000 ops per device. This test dispatches exactly
# `ITERATIONS + 1` of them, so the margin is deliberate.
ITERATIONS = 20


@pytest.mark.parametrize("mesh_device, device_params", [pytest.param((2, 4), FABRIC, id="mesh-2x4")], indirect=True)
@pytest.mark.parametrize("fuse_add", [False, True], ids=["plain", "settle"])
def test_device_time(mesh_device, device_params, fuse_add):
    torch.manual_seed(2026)

    mesh_shape = tuple(mesh_device.shape)
    tp_factor, sp_factor = mesh_shape[TP_AXIS], mesh_shape[SP_AXIS]
    num_tokens = PER_CHIP_TOKENS * sp_factor

    stream_dims, vector_dims, scalar_dims = [None, None], [None, None], [None, None]
    stream_dims[SP_AXIS], stream_dims[TP_AXIS] = 2, 3
    vector_dims[TP_AXIS] = 3
    scalar_dims[SP_AXIS] = 2

    composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 3), mesh_shape=mesh_shape)

    def to_dev(t, dims, dtype=ttnn.bfloat16):
        host = ttnn.from_torch(
            t,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=dims, mesh_shape=mesh_shape),
        )
        return ttnn.to_device(host, mesh_device)

    shape = [1, 1, num_tokens, HIDDEN_SIZE]
    partial = torch.randn(shape, dtype=torch.bfloat16)
    prefix_sum = torch.randn(shape, dtype=torch.bfloat16)
    pending = torch.randn(shape, dtype=torch.bfloat16) if fuse_add else None
    # What the op scores and folds against: `prefix_sum` alone, or the sum it settles.
    stream = prefix_sum.float() + pending.float() if fuse_add else prefix_sum.float()
    query = torch.randn([1, 1, 1, HIDDEN_SIZE], dtype=torch.bfloat16) * 0.05
    shift = torch.randn([1, 1, num_tokens, 1]) * 2.0
    # A mass is a sum of exponentials against a running maximum, so it is at least one.
    # Drawn around zero it would put the denominator near zero and the check would
    # measure cancellation instead of the op.
    mass = torch.rand([1, 1, num_tokens, 1]) * 7.0 + 1.0

    tt_partial = to_dev(partial, stream_dims)
    tt_prefix = to_dev(prefix_sum, stream_dims)
    tt_pending = to_dev(pending, stream_dims) if fuse_add else None
    tt_query = to_dev(query, vector_dims)
    tt_shift = to_dev(shift, scalar_dims, ttnn.float32)
    tt_mass = to_dev(mass, scalar_dims, ttnn.float32)
    # Caller-allocated exchange scratch: one sum-of-squares and one dots plane per rank,
    # replicated across the tensor-parallel axis so a page has the same address on every
    # chip of it.
    tt_stats = to_dev(torch.zeros([1, 2 * tp_factor, num_tokens, 1]), scalar_dims, ttnn.float32)

    grid = mesh_device.compute_with_storage_grid_size()
    semaphore = ttnn.create_global_semaphore(
        mesh_device,
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))]),
        0,
    )

    def call():
        return ttnn.experimental.attn_res_gather_merge(
            tt_partial,
            tt_prefix,
            tt_shift,
            tt_mass,
            tt_query,
            tt_stats,
            semaphore,
            cluster_axis=TP_AXIS,
            inv_hidden_size=INV_HIDDEN_SIZE,
            eps=EPS,
            pending=tt_pending,
        )

    outputs = call()
    got = ttnn.to_torch(outputs[0], mesh_composer=composer)
    _, pcc = assert_with_pcc(_oracle(partial, stream, shift, mass, query), got.float(), PCC)
    logger.info(f"fused vs torch: {pcc}")
    if fuse_add:
        _, pcc_stream = assert_with_pcc(stream, ttnn.to_torch(outputs[1], mesh_composer=composer).float(), PCC)
        logger.info(f"settled stream vs torch: {pcc_stream}")
    for tensor in outputs:
        ttnn.deallocate(tensor)

    for _ in range(ITERATIONS):
        for tensor in call():
            ttnn.deallocate(tensor)
    ttnn.synchronize_device(mesh_device)
    logger.info(f"{ITERATIONS + 1} AttnResGatherMerge dispatches per device, and nothing else")

    for tensor in (tt_partial, tt_prefix, tt_query, tt_shift, tt_mass, tt_stats):
        ttnn.deallocate(tensor)
    if fuse_add:
        ttnn.deallocate(tt_pending)
