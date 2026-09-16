# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
#47108: all_to_all_combine is wrong on a program-cache HIT when the same input
buffers are reused (trace replay / static KV / a loop over the same tensors).

Existing nightlies reallocate tensors every iteration, so they miss this path.
This helper builds inputs once and dispatches num_iters times on those tensors.

reuse_optional_output without poisoning is not a #47108 oracle: if the writer
skips on a cache hit, leftover iter-0 data still matches golden. Poison the
persistent output (and optionally rescale the same input buffers) so a skipped
write cannot hide.
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


def _logical_shape_list(tensor):
    shape = getattr(tensor, "logical_shape", None)
    if shape is None:
        shape = tensor.shape
    return [int(d) for d in shape]


def _poison_output_tensor(tt_out, mesh_device, fill_value=1.0):
    """Overwrite a persistent combine output so a skipped writer cannot reuse iter-0 data."""
    sentinel = ttnn.moreh_full(
        _logical_shape_list(tt_out),
        fill_value,
        mesh_device,
        dtype=tt_out.dtype,
        layout=tt_out.layout,
        memory_config=tt_out.memory_config(),
    )
    ttnn.copy(sentinel, tt_out)
    ttnn.synchronize_device(mesh_device)


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
    mutate_inputs=False,
    poison_output=None,
):
    torch.manual_seed(2005)
    mesh_device.enable_program_cache()
    mesh_device.cache_entries_counter = CacheEntriesCounter(mesh_device)

    devices = mesh_shape[0] * mesh_shape[1]
    batch = batches_per_device * mesh_shape[axis]
    input_memory_config = input_memory_config or ttnn.DRAM_MEMORY_CONFIG
    output_memory_config = output_memory_config or ttnn.L1_MEMORY_CONFIG
    if poison_output is None:
        poison_output = reuse_optional_output

    logger.info(
        f"#47108 static-buffer repro: mesh={mesh_shape} axis={axis} batch={batch} seq={seq} "
        f"experts={experts} k={select_experts_k} hidden={hidden_size} "
        f"local_reduce={local_reduce} num_links={num_links} num_iters={num_iters} "
        f"reuse_optional_output={reuse_optional_output} mutate_inputs={mutate_inputs} "
        f"poison_output={poison_output}"
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
    goldens = []
    combine_cache_growth = []
    input_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)

    with mesh_device.cache_entries_counter.measure():
        for i in range(num_iters):
            scale = float(i + 1) if mutate_inputs else 1.0
            iter_golden = golden_out * scale if mutate_inputs else golden_out
            goldens.append(iter_golden)

            if mutate_inputs and i > 0:
                scaled = ttnn.from_torch(
                    input_contrib * scale,
                    device=mesh_device,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    dtype=ttnn.bfloat16,
                    memory_config=input_memory_config,
                    mesh_mapper=input_mapper,
                )
                ttnn.copy(scaled, tt_input)
                ttnn.synchronize_device(mesh_device)

            if poison_output and optional_output is not None:
                _poison_output_tensor(optional_output, mesh_device, fill_value=1.0)

            kwargs = dict(
                num_links=num_links,
                topology=topology,
                memory_config=output_memory_config,
                local_reduce=local_reduce,
                cluster_axis=axis,
            )
            if reuse_optional_output and optional_output is not None:
                kwargs["output_tensor"] = optional_output

            before_entries = mesh_device.num_program_cache_entries()
            tt_out = ttnn.all_to_all_combine(tt_input, tt_meta, tt_map, **kwargs)
            after_entries = mesh_device.num_program_cache_entries()
            combine_cache_growth.append(after_entries - before_entries)
            if reuse_optional_output and optional_output is None:
                optional_output = tt_out

            ttnn.synchronize_device(mesh_device)
            out_agg = _agg_combine_output(tt_out, mesh_device, mesh_shape, axis)
            outputs.append(out_agg.clone())
            nz = out_agg.float().abs().sum().item()
            abs_sums.append(nz)
            logger.info(
                f"iter {i}: output abs-sum = {nz:.3f}  cache_entries={after_entries} "
                f"combine_growth={combine_cache_growth[-1]} scale={scale}"
            )

    cache_delta = mesh_device.cache_entries_counter.total
    logger.info(
        f"program cache delta across {num_iters} reuse dispatches = {cache_delta} "
        f"combine_growth={combine_cache_growth}"
    )

    failures = []
    # Iter 0 may compile combine + moreh_full. Later combine() calls must cache-hit.
    for i, growth in enumerate(combine_cache_growth):
        if i > 0 and growth != 0:
            failures.append(f"iter {i} expected combine cache HIT (growth 0), got growth={growth}")

    for i in range(num_iters):
        all_zero = abs_sums[i] == 0.0
        try:
            check_results(outputs[i], goldens[i], data_map)
        except AssertionError as e:
            tag = "MISS" if i == 0 else "HIT"
            extra = " ALL-ZEROS" if all_zero else ""
            scale_tag = i + 1 if mutate_inputs else 1
            failures.append(f"iter {i} (cache {tag}{extra}, scale={scale_tag}) != golden: {str(e)[:200]}")
        if not mutate_inputs and i > 0 and not torch.equal(outputs[i], outputs[0]):
            failures.append(
                f"iter {i} (cache HIT, abs-sum={abs_sums[i]:.3f}) != iter 0 " f"(cache MISS, abs-sum={abs_sums[0]:.3f})"
            )
        if mutate_inputs and i > 0 and torch.equal(outputs[i], outputs[0]):
            failures.append(
                f"iter {i} output identical to iter 0 after in-place input scale={i + 1} — writer likely skipped"
            )

    if failures:
        for f in failures:
            logger.error(f)
        pytest.fail(
            "all_to_all_combine is wrong on program-cache HIT with static (reused) input buffers "
            f"(#47108). cache_delta={cache_delta} combine_growth={combine_cache_growth} "
            f"abs_sums={abs_sums}\n  " + "\n  ".join(failures)
        )

    logger.info(
        f"All iterations matched golden. cache_delta={cache_delta} "
        f"combine_growth={combine_cache_growth} abs_sums={abs_sums}"
    )


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

    # Capture a single combine. Multiple launches in one trace can hide a hit-path
    # skip: the first captured launch would still leave golden in persistent.
    logger.info("Capturing 1 combine launch on persistent output")
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    _combine(output_tensor=persistent)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    failures = []
    abs_sums = []
    for replay in range(num_replays):
        _poison_output_tensor(persistent, mesh_device, fill_value=1.0)
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
