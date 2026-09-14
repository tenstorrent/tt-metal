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


def _agg_combine_output(tt_out, mesh_device, mesh_shape, axis):
    if axis == 0:
        device_shards = [ttnn.to_torch(ittout, mesh_composer=None) for ittout in ttnn.get_device_tensors(tt_out)]
        ordered_shards = []
        for ir in range(mesh_shape[1]):
            for ic in range(mesh_shape[0]):
                ordered_shards.append(device_shards[ic * mesh_shape[1] + ir])
        return torch.cat(ordered_shards, dim=1)
    return ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1))


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
            out_agg = _agg_combine_output(tt_out, mesh_device, mesh_shape, axis)
            outputs.append(out_agg.clone())
            nz = out_agg.float().abs().sum().item()
            abs_sums.append(nz)
            logger.info(f"iter {i}: output abs-sum = {nz:.3f}  cache_entries={mesh_device.num_program_cache_entries()}")

    cache_delta = mesh_device.cache_entries_counter.total
    logger.info(f"program cache delta across {num_iters} reuse dispatches = {cache_delta}")

    # combine + moreh_full on the first alloc (iter 0 still goes through moreh_full even
    # when later iters reuse that tensor). A genuine keying miss every dispatch would be >2.
    expected_cache_entries = 2
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


def run_a2a_combine_static_buffer_cache_hit_trace_repro(
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
    num_replays=2,
):
    """Trace-capture + replay of combine on reused input AND output buffers.

    This is the tt-xla / static-KV shape: host H2D semaphore reset is illegal
    during capture, so a cache-hit handshake bug shows up as zeros (or a hang)
    on execute_trace rather than on eager dispatch.
    """
    torch.manual_seed(2005)
    mesh_device.enable_program_cache()

    devices = mesh_shape[0] * mesh_shape[1]
    batch = batches_per_device * mesh_shape[axis]
    input_memory_config = input_memory_config or ttnn.DRAM_MEMORY_CONFIG
    output_memory_config = output_memory_config or ttnn.DRAM_MEMORY_CONFIG

    logger.info(
        f"#47108 TRACE static-buffer repro: mesh={mesh_shape} axis={axis} batch={batch} seq={seq} "
        f"experts={experts} k={select_experts_k} hidden={hidden_size} "
        f"local_reduce={local_reduce} num_links={num_links} num_iters={num_iters} "
        f"num_replays={num_replays}"
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

    def _combine(output_tensor=None):
        kwargs = dict(
            num_links=num_links,
            topology=topology,
            memory_config=output_memory_config,
            local_reduce=local_reduce,
            cluster_axis=axis,
        )
        if output_tensor is not None:
            kwargs["output_tensor"] = output_tensor
        return ttnn.all_to_all_combine(tt_input, tt_meta, tt_map, **kwargs)

    # Compile / cache-miss outside trace (moreh_full is a host write; illegal in capture).
    persistent = _combine()
    ttnn.synchronize_device(mesh_device)
    compile_out = _agg_combine_output(persistent, mesh_device, mesh_shape, axis)
    compile_sum = compile_out.float().abs().sum().item()
    logger.info(f"compile (eager miss): output abs-sum = {compile_sum:.3f}")
    check_results(compile_out, golden_out, data_map)

    logger.info(f"Capturing {num_iters} combine launches on persistent output")
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    for _ in range(num_iters):
        _combine(output_tensor=persistent)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    failures = []
    abs_sums = []
    for replay in range(num_replays):
        ttnn.execute_trace(mesh_device, trace_id, blocking=False)
        ttnn.synchronize_device(mesh_device)
        out_agg = _agg_combine_output(persistent, mesh_device, mesh_shape, axis)
        nz = out_agg.float().abs().sum().item()
        abs_sums.append(nz)
        logger.info(f"trace replay {replay}: output abs-sum = {nz:.3f}")
        all_zero = nz == 0.0
        try:
            check_results(out_agg, golden_out, data_map)
        except AssertionError as e:
            extra = " ALL-ZEROS" if all_zero else ""
            failures.append(f"trace replay {replay}{extra} != golden: {str(e)[:200]}")
        if not torch.equal(out_agg, compile_out):
            failures.append(f"trace replay {replay} (abs-sum={nz:.3f}) != compile miss (abs-sum={compile_sum:.3f})")

    ttnn.release_trace(mesh_device, trace_id)

    if failures:
        for f in failures:
            logger.error(f)
        pytest.fail(
            "all_to_all_combine is wrong on TRACE replay with static (reused) buffers "
            f"(#47108). abs_sums={abs_sums}\n  " + "\n  ".join(failures)
        )

    logger.info(f"All trace replays matched golden and compile miss. abs_sums={abs_sums}")


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
@pytest.mark.parametrize("reuse_optional_output", [False, True], ids=["new_output", "reuse_output"])
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
    reuse_optional_output,
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
        reuse_optional_output=reuse_optional_output,
    )


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "trace_region_size": 500000,
        },
    ],
    ids=["fabric_1d_line_trace"],
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
@pytest.mark.parametrize("local_reduce", [True], ids=["sparse"])
@pytest.mark.parametrize("experts", [256])
@pytest.mark.parametrize("select_experts_k", [8])
@pytest.mark.parametrize("hidden_size", [7168])
@pytest.mark.parametrize("num_iters", [2])
@pytest.mark.parametrize("num_links", [1])
def test_a2a_combine_cache_hit_trace_repro_t3k(
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
    run_a2a_combine_static_buffer_cache_hit_trace_repro(
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
