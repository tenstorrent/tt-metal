# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn

# Reuse the shared test runner and helpers from the nightly test module
from tests.nightly.t3000.ccl.test_all_to_all_dispatch import (
    run_all_to_all_dispatch_test,
    get_max_links,
)


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "trace_region_size": 500000,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("trace_mode", [True])
@pytest.mark.parametrize(
    "mesh_shape, mesh_device", [pytest.param((2, 4), (2, 4), id="2x4_grid")], indirect=["mesh_device"]
)
@pytest.mark.parametrize("cluster_axis", [1])
@pytest.mark.parametrize("batches_per_device", [8])
@pytest.mark.parametrize("experts_per_device", [8])
@pytest.mark.parametrize("select_experts_k", [8])
@pytest.mark.parametrize("hidden_size", [7168])
@pytest.mark.parametrize(
    "seq_len, num_iters, warmup_iters",
    [
        (1, 40, 20),
    ],
    ids=["s1"],
)
@pytest.mark.parametrize("input_memory_config", [ttnn.DRAM_MEMORY_CONFIG], ids=["dram"])
@pytest.mark.parametrize("output_memory_config", [ttnn.DRAM_MEMORY_CONFIG], ids=["dram"])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
def test_decode_perf(
    mesh_device,
    trace_mode,
    mesh_shape,
    cluster_axis,
    batches_per_device,
    experts_per_device,
    select_experts_k,
    hidden_size,
    seq_len,
    num_iters,
    warmup_iters,
    num_links,
    dtype,
    input_memory_config,
    output_memory_config,
):
    if cluster_axis is None:
        dispatch_devices = mesh_shape[0] * mesh_shape[1]
    else:
        dispatch_devices = mesh_shape[cluster_axis]

    batch = batches_per_device * dispatch_devices
    experts = experts_per_device * dispatch_devices

    run_all_to_all_dispatch_test(
        mesh_device,
        mesh_shape,
        batch,
        experts,
        select_experts_k,
        hidden_size,
        seq_len,
        num_iters,
        warmup_iters,
        trace_mode,
        num_links=num_links,
        scheme="worst_perf",
        input_memory_config=input_memory_config,
        output_memory_config=output_memory_config,
        dtype=dtype,
        cluster_axis=cluster_axis,
        use_optional_output_tensors=True,
        skip_validation=True,
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
    indirect=True,
)
@pytest.mark.parametrize("trace_mode", [True])
@pytest.mark.parametrize(
    "mesh_shape, mesh_device", [pytest.param((2, 4), (2, 4), id="2x4_grid")], indirect=["mesh_device"]
)
@pytest.mark.parametrize("cluster_axis", [1])
@pytest.mark.parametrize("batches_per_device", [8])
@pytest.mark.parametrize("experts_per_device", [8])
@pytest.mark.parametrize("select_experts_k", [8])
@pytest.mark.parametrize("hidden_size", [7168])
@pytest.mark.parametrize(
    "seq_len, num_iters, warmup_iters",
    [
        (128, 10, 5),
    ],
    ids=["s128"],
)
@pytest.mark.parametrize("input_memory_config", [ttnn.DRAM_MEMORY_CONFIG], ids=["dram"])
@pytest.mark.parametrize("output_memory_config", [ttnn.DRAM_MEMORY_CONFIG], ids=["dram"])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
def test_prefill_perf(
    mesh_device,
    trace_mode,
    mesh_shape,
    cluster_axis,
    batches_per_device,
    experts_per_device,
    select_experts_k,
    hidden_size,
    seq_len,
    num_iters,
    warmup_iters,
    num_links,
    dtype,
    input_memory_config,
    output_memory_config,
):
    if cluster_axis is None:
        dispatch_devices = mesh_shape[0] * mesh_shape[1]
    else:
        dispatch_devices = mesh_shape[cluster_axis]

    batch = batches_per_device * dispatch_devices
    experts = experts_per_device * dispatch_devices

    run_all_to_all_dispatch_test(
        mesh_device,
        mesh_shape,
        batch,
        experts,
        select_experts_k,
        hidden_size,
        seq_len,
        num_iters,
        warmup_iters,
        trace_mode,
        num_links=num_links,
        scheme="worst_perf",
        input_memory_config=input_memory_config,
        output_memory_config=output_memory_config,
        dtype=dtype,
        cluster_axis=cluster_axis,
        use_optional_output_tensors=True,
        skip_validation=True,
    )


# Performance tests
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
            "trace_region_size": 500000,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_shape, mesh_device", [pytest.param((8, 4), (8, 4), id="8x4_grid")], indirect=["mesh_device"]
)
@pytest.mark.parametrize("cluster_axis", [1])
@pytest.mark.parametrize("batches_per_device", [8])
@pytest.mark.parametrize("experts", [256])
@pytest.mark.parametrize("select_experts_k", [8])
@pytest.mark.parametrize("hidden_size", [7168])
@pytest.mark.parametrize(
    "seq_len, num_iters, warmup_iters",
    [(1, 40, 10)],
    ids=["decode"],
)
@pytest.mark.parametrize("num_links", [4])
@pytest.mark.parametrize("topology", [None])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("input_memory_config", [ttnn.DRAM_MEMORY_CONFIG], ids=["dram"])
@pytest.mark.parametrize("output_memory_config", [ttnn.DRAM_MEMORY_CONFIG], ids=["dram"])
def test_decode_galaxy_perf(
    mesh_device,
    mesh_shape,
    cluster_axis,
    batches_per_device,
    experts,
    select_experts_k,
    hidden_size,
    seq_len,
    num_iters,
    warmup_iters,
    num_links,
    topology,
    dtype,
    input_memory_config,
    output_memory_config,
):
    if cluster_axis is None:
        dispatch_devices = mesh_shape[0] * mesh_shape[1]
    else:
        dispatch_devices = mesh_shape[cluster_axis]

    batch = batches_per_device * dispatch_devices
    trace_mode = True

    run_all_to_all_dispatch_test(
        mesh_device,
        mesh_shape,
        batch,
        experts,
        select_experts_k,
        hidden_size,
        seq_len,
        num_iters,
        warmup_iters,
        trace_mode,
        num_links=num_links,
        scheme="worst_perf",
        topology=topology,
        input_memory_config=input_memory_config,
        output_memory_config=output_memory_config,
        dtype=dtype,
        cluster_axis=cluster_axis,
        skip_validation=True,
    )


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
            "trace_region_size": 500000,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_shape, mesh_device", [pytest.param((8, 4), (8, 4), id="8x4_grid")], indirect=["mesh_device"]
)
@pytest.mark.parametrize("cluster_axis", [1])
@pytest.mark.parametrize("batches_per_device", [8])
@pytest.mark.parametrize("experts", [256])
@pytest.mark.parametrize("select_experts_k", [8])
@pytest.mark.parametrize("hidden_size", [7168])
@pytest.mark.parametrize(
    "seq_len, num_iters, warmup_iters",
    [(128, 10, 5)],
    ids=["prefill"],
)
@pytest.mark.parametrize("num_links", [4])
@pytest.mark.parametrize("topology", [None])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("input_memory_config", [ttnn.DRAM_MEMORY_CONFIG], ids=["dram"])
@pytest.mark.parametrize("output_memory_config", [ttnn.DRAM_MEMORY_CONFIG], ids=["dram"])
def test_prefill_galaxy_perf(
    mesh_device,
    mesh_shape,
    cluster_axis,
    batches_per_device,
    experts,
    select_experts_k,
    hidden_size,
    seq_len,
    num_iters,
    warmup_iters,
    num_links,
    topology,
    dtype,
    input_memory_config,
    output_memory_config,
):
    if cluster_axis is None:
        dispatch_devices = mesh_shape[0] * mesh_shape[1]
    else:
        dispatch_devices = mesh_shape[cluster_axis]

    batch = batches_per_device * dispatch_devices
    trace_mode = True

    run_all_to_all_dispatch_test(
        mesh_device,
        mesh_shape,
        batch,
        experts,
        select_experts_k,
        hidden_size,
        seq_len,
        num_iters,
        warmup_iters,
        trace_mode,
        num_links=num_links,
        scheme="worst_perf",
        topology=topology,
        input_memory_config=input_memory_config,
        output_memory_config=output_memory_config,
        dtype=dtype,
        cluster_axis=cluster_axis,
        skip_validation=True,
    )
