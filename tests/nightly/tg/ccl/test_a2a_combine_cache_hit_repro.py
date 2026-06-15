# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Standalone repro: all_to_all_combine produces WRONG output (all zeros) on a
program-cache HIT, with the SAME inputs reused across iterations.

Background
----------
PR #44408 migrated ccl/all_to_all_{combine,dispatch} to ProgramDescriptor. The
first invocation (cache miss) is correct, but on every subsequent invocation
(cache hit) the cross-device init barrier never completes and the device writes
all zeros.

PR #45332 added a tactical workaround for all_to_all_combine: it mixes the input
tensor *buffer addresses* into the program hash, so when a caller REALLOCATES its
inputs between dispatches the hash changes -> a fresh cache entry -> fresh
program. That hides the bug for the existing nightly tests (which allocate new
input tensors every iteration).

It does NOT help when the inputs are STATIC (same buffers reused) — e.g. trace
replay, a static KV/MLA cache, or simply calling the op repeatedly on the same
device tensors. Then the hash is identical, the program cache-hits, and the
output is wrong. This repro reuses the SAME input tensors for every iteration to
exhibit exactly that case, so it fails BOTH at the pre-#45332 state and with
#45332 applied.

Run (Galaxy / TG, 8x4):
  export TT_METAL_HOME=$(git rev-parse --show-toplevel)
  pytest -svv tests/nightly/tg/ccl/test_a2a_combine_cache_hit_repro.py
"""

import pytest
import torch
from loguru import logger

import ttnn

from tests.nightly.t3000.ccl.test_all_to_all_combine import gen_tensors, check_results


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        },
    ],
    ids=["fabric_1d_line"],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_shape, mesh_device",
    [pytest.param((8, 4), (8, 4), id="8x4_grid")],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("axis", [1])
@pytest.mark.parametrize("batches_per_device", [8])
@pytest.mark.parametrize("seq", [2])
@pytest.mark.parametrize("local_reduce", [False, True], ids=["dense", "sparse"])
@pytest.mark.parametrize("experts", [256])
@pytest.mark.parametrize("select_experts_k", [8])
@pytest.mark.parametrize("hidden_size", [7168])
@pytest.mark.parametrize("num_iters", [3])
@pytest.mark.parametrize("num_links", [4])
def test_a2a_combine_cache_hit_repro(
    mesh_device,
    mesh_shape,
    axis,
    batches_per_device,
    seq,
    local_reduce,
    experts,
    select_experts_k,
    hidden_size,
    num_iters,
    num_links,
):
    torch.manual_seed(2005)
    mesh_device.enable_program_cache()
    devices = mesh_shape[0] * mesh_shape[1]
    batch = batches_per_device * mesh_shape[axis]

    in_mem = ttnn.DRAM_MEMORY_CONFIG
    out_mem = ttnn.L1_MEMORY_CONFIG

    # Build the input tensors ONCE and reuse them for every iteration so the
    # program hash (incl. #45332's buffer-address mix) is identical -> iter 0 is
    # a cache miss, iters 1..N-1 are cache HITS on the same buffers.
    _, input_contrib, expert_mapping, metadata_tensor, golden_out, data_map = gen_tensors(
        batch, experts, select_experts_k, hidden_size, seq, mesh_shape, axis, devices,
        scheme="random", local_reduce=local_reduce,
    )

    tt_input = ttnn.from_torch(
        input_contrib, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16,
        memory_config=in_mem, mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    tt_map = ttnn.from_torch(
        expert_mapping, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint16,
        memory_config=in_mem,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=mesh_shape),
    )
    tt_meta = ttnn.from_torch(
        metadata_tensor, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint16,
        memory_config=in_mem, mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )

    outputs = []
    for i in range(num_iters):
        tt_out = ttnn.all_to_all_combine(
            tt_input, tt_meta, tt_map,
            num_links=num_links, topology=None, memory_config=out_mem,
            local_reduce=local_reduce, cluster_axis=axis,
        )
        ttnn.synchronize_device(mesh_device)
        out_agg = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1))
        outputs.append(out_agg)
        nz = out_agg.float().abs().sum().item()
        logger.info(f"iter {i}: output abs-sum = {nz:.3f}")

    # 1) Every iteration must match the golden (catches the all-zeros corruption).
    # 2) Cache-hit iterations must be identical to the cache-miss iteration 0
    #    (catches any non-determinism introduced by the cache-hit path).
    failures = []
    for i in range(num_iters):
        try:
            check_results(outputs[i], golden_out, data_map)
        except AssertionError as e:
            failures.append(f"iter {i} (cache {'MISS' if i == 0 else 'HIT'}) != golden: {str(e)[:120]}")
        if i > 0 and not torch.equal(outputs[i], outputs[0]):
            failures.append(f"iter {i} (cache HIT) != iter 0 (cache MISS) — non-deterministic")

    if failures:
        for f in failures:
            logger.error(f)
        pytest.fail(
            "all_to_all_combine is wrong on program-cache HIT with static (reused) input buffers — "
            "PR #44408 regression that #45332's buffer-address hash does NOT cover. Details:\n  "
            + "\n  ".join(failures)
        )
    logger.info("All iterations matched golden and iter 0 — no regression.")