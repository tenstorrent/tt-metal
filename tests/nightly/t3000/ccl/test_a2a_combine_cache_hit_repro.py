# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
#47108: all_to_all_combine is wrong on a program-cache HIT when the same input
buffers are reused (trace replay / static KV / a loop over the same tensors).

Existing nightlies reallocate tensors every iteration, so they miss this path.
This helper builds inputs once and dispatches num_iters times on those tensors.
"""

import pytest
import torch
from loguru import logger

import ttnn

from tests.nightly.t3000.ccl.test_all_to_all_combine import check_results, gen_tensors
from tests.tests_common.cache_entries_counter import CacheEntriesCounter


def run_a2a_combine_static_buffer_cache_hit_repro(
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
    topology=None,
    input_memory_config=None,
    output_memory_config=None,
    reuse_optional_output=False,
):
    torch.manual_seed(2005)
    mesh_device.enable_program_cache()
    mesh_device.cache_entries_counter = CacheEntriesCounter(mesh_device)

    devices = mesh_shape[0] * mesh_shape[1]
    batch = batches_per_device * mesh_shape[axis]
    input_memory_config = input_memory_config or ttnn.DRAM_MEMORY_CONFIG
    output_memory_config = output_memory_config or ttnn.L1_MEMORY_CONFIG

    logger.info(
        f"#47108 static-buffer repro: mesh={mesh_shape} axis={axis} batch={batch} seq={seq} "
        f"experts={experts} k={select_experts_k} hidden={hidden_size} "
        f"local_reduce={local_reduce} num_links={num_links} num_iters={num_iters} "
        f"reuse_optional_output={reuse_optional_output}"
    )

    _, input_contrib, expert_mapping, metadata_tensor, golden_out, data_map = gen_tensors(
        batch,
        experts,
        select_experts_k,
        hidden_size,
        seq,
        mesh_shape,
        axis,
        devices,
        scheme="random",
        local_reduce=local_reduce,
    )

    tt_input = ttnn.from_torch(
        input_contrib,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=input_memory_config,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    tt_map = ttnn.from_torch(
        expert_mapping,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint16,
        memory_config=input_memory_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=mesh_shape),
    )
    tt_meta = ttnn.from_torch(
        metadata_tensor,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint16,
        memory_config=input_memory_config,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )

    optional_output = None
    outputs = []
    abs_sums = []

    with mesh_device.cache_entries_counter.measure():
        for i in range(num_iters):
            kwargs = dict(
                num_links=num_links,
                topology=topology,
                memory_config=output_memory_config,
                local_reduce=local_reduce,
                cluster_axis=axis,
            )
            if reuse_optional_output and optional_output is not None:
                kwargs["output_tensor"] = optional_output

            tt_out = ttnn.all_to_all_combine(tt_input, tt_meta, tt_map, **kwargs)
            if reuse_optional_output and optional_output is None:
                optional_output = tt_out

            ttnn.synchronize_device(mesh_device)
            if axis == 0:
                device_shards = [
                    ttnn.to_torch(ittout, mesh_composer=None) for ittout in ttnn.get_device_tensors(tt_out)
                ]
                ordered_shards = []
                for ir in range(mesh_shape[1]):
                    for ic in range(mesh_shape[0]):
                        ordered_shards.append(device_shards[ic * mesh_shape[1] + ir])
                out_agg = torch.cat(ordered_shards, dim=1)
            else:
                out_agg = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1))
            outputs.append(out_agg.clone())
            nz = out_agg.float().abs().sum().item()
            abs_sums.append(nz)
            logger.info(f"iter {i}: output abs-sum = {nz:.3f}  cache_entries={mesh_device.num_program_cache_entries()}")

    cache_delta = mesh_device.cache_entries_counter.total
    logger.info(f"program cache delta across {num_iters} reuse dispatches = {cache_delta}")

    # combine + moreh_full (internal zeroed output when no output_tensor is passed).
    # With a reused output_tensor, only combine should miss once.
    expected_cache_entries = 1 if reuse_optional_output else 2
    failures = []
    if cache_delta != expected_cache_entries:
        failures.append(f"expected {expected_cache_entries} program cache entries on reuse hits, got {cache_delta}")

    for i in range(num_iters):
        all_zero = abs_sums[i] == 0.0
        try:
            check_results(outputs[i], golden_out, data_map)
        except AssertionError as e:
            tag = "MISS" if i == 0 else "HIT"
            extra = " ALL-ZEROS" if all_zero else ""
            failures.append(f"iter {i} (cache {tag}{extra}) != golden: {str(e)[:200]}")
        if i > 0 and not torch.equal(outputs[i], outputs[0]):
            failures.append(
                f"iter {i} (cache HIT, abs-sum={abs_sums[i]:.3f}) != iter 0 " f"(cache MISS, abs-sum={abs_sums[0]:.3f})"
            )

    if failures:
        for f in failures:
            logger.error(f)
        pytest.fail(
            "all_to_all_combine is wrong on program-cache HIT with static (reused) input buffers "
            f"(#47108). cache_delta={cache_delta} abs_sums={abs_sums}\n  " + "\n  ".join(failures)
        )

    logger.info(f"All iterations matched golden and iter 0. cache_delta={cache_delta} abs_sums={abs_sums}")


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        },
    ],
    ids=["fabric_1d_line"],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_shape, mesh_device",
    [pytest.param((2, 4), (2, 4), id="2x4_grid")],
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
@pytest.mark.parametrize("num_links", [1])
def test_a2a_combine_cache_hit_repro_t3k(
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
    run_a2a_combine_static_buffer_cache_hit_repro(
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
    )
