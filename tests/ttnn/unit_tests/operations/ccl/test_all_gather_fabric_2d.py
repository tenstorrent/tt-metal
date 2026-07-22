# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Sparse-MLA Fabric 2D all-gather regression coverage.

The sparse MLA 4x2 path gathers row-major KV-cache rows on the SP axis.  Its
BF16 cache has 1,152-byte rows; its scaled-FP8 cache packs each 656-byte row
into a 704-byte aligned DRAM page.  Keep these tests small so a Fabric routing
failure can be diagnosed without model execution.
"""

import statistics

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import SparseKVCacheFormat, init_sparse_kv_cache
from tests.ttnn.utils_for_testing import assert_with_pcc
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program


def _fabric_router_config():
    config = ttnn.FabricRouterConfig()
    config.max_packet_payload_size_bytes = 14 * 1024
    return config


def _sparse_mla_nd_sharded_dram_config(mesh_device, width):
    """Production KV-cache layout: 32-row chunks round-robin over DRAM banks."""
    dram_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(bank, 0), ttnn.CoreCoord(bank, 0))
            for bank in range(mesh_device.dram_grid_size().x)
        ]
    )
    return ttnn.MemoryConfig(
        buffer_type=ttnn.BufferType.DRAM,
        nd_shard_spec=ttnn.NdShardSpec(
            shard_shape=[1, 1, 32, width],
            grid=dram_grid,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
        ),
    )


_MATCHED_RING_DEVICE_PARAMS = [
    pytest.param(
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
            "fabric_router_config": _fabric_router_config(),
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
            "l1_small_size": 2048,
        },
        id="fabric_1d_ring",
    ),
    pytest.param(
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": _fabric_router_config(),
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
            "l1_small_size": 2048,
        },
        id="fabric_2d",
    ),
]


def _fabric_name(fabric_config):
    if fabric_config == ttnn.FabricConfig.FABRIC_1D_RING:
        return "fabric_1d_ring"
    if fabric_config == ttnn.FabricConfig.FABRIC_2D:
        return "fabric_2d"
    raise AssertionError(f"matched ring benchmark received unsupported fabric config {fabric_config}")


def _logical_ring_route_diagnostics(mesh_device, fabric_config, neighbor_unicast=False):
    """Describe physical link directions and packet lifetime for the selected ring transport."""
    direction_names = ("E", "W", "N", "S", "Z")
    ring_size = mesh_device.shape[0]
    assert mesh_device.shape[1] == 1
    forward_hops = (ring_size - 1 + 1) // 2
    backward_hops = ring_size - 1 - forward_hops
    path_variants = set()

    if neighbor_unicast:
        for source in range(ring_size):
            for step in (1, -1):
                neighbor = (source + step) % ring_size
                src_node = mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(source, 0))
                dst_node = mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(neighbor, 0))
                direction = ttnn.get_eth_forwarding_direction(src_node, dst_node)
                assert direction is not None, f"no physical Fabric route from logical rank {source} to {neighbor}"
                path_variants.add(direction_names[direction])
        return (
            "transport=neighbor_unicast packet_hops=1 relay=tensix "
            f"edge_directions={','.join(sorted(path_variants))}"
        )

    for source in range(ring_size):
        for step in (1, -1):
            for hop_count in (forward_hops, backward_hops):
                rank = source
                directions = []
                escape_hop = 0
                for hop in range(hop_count):
                    next_rank = (rank + step) % ring_size
                    src_node = mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(rank, 0))
                    dst_node = mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(next_rank, 0))
                    direction = ttnn.get_eth_forwarding_direction(src_node, dst_node)
                    assert direction is not None, f"no physical Fabric route from logical rank {rank} to {next_rank}"
                    directions.append(direction_names[direction])
                    crosses_dateline = {rank, next_rank} == {0, ring_size - 1}
                    if crosses_dateline and hop + 1 < hop_count:
                        escape_hop = hop + 1
                    rank = next_rank
                turns = sum(lhs != rhs for lhs, rhs in zip(directions, directions[1:]))
                path_variants.add(("".join(directions), hop_count, turns, escape_hop))

    explicit_path = fabric_config == ttnn.FabricConfig.FABRIC_2D
    variants = ",".join(
        f"{directions}:hops={hop_count}:turns={turns}:escape={escape_hop if explicit_path else 'native'}"
        for directions, hop_count, turns, escape_hop in sorted(path_variants)
    )
    return f"explicit_path={str(explicit_path).lower()} route_variants=[{variants}]"


def _direct_ring_connection_plan(mesh_device, num_links=2):
    """Resolve every logical ring edge to its physical direction and ERISC channels."""
    direction_names = ("E", "W", "N", "S", "Z")
    ring_size = mesh_device.shape[0]
    assert tuple(mesh_device.shape) == (ring_size, 1)
    plan = []
    for source in range(ring_size):
        src_node = mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(source, 0))
        for step, logical_direction in ((1, "forward"), (-1, "backward")):
            neighbor = (source + step) % ring_size
            dst_node = mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(neighbor, 0))
            direction = ttnn.get_eth_forwarding_direction(src_node, dst_node)
            assert direction is not None
            channels = []
            for link in range(num_links):
                # A fresh descriptor keeps the diagnostic independent of the
                # per-core semaphore limit. The first worker RT arg is the
                # resolved physical Ethernet channel.
                descriptor = ttnn.ProgramDescriptor()
                connection_args = ttnn.setup_fabric_connection(
                    src_node, dst_node, link, descriptor, ttnn.CoreCoord(0, 0)
                )
                channels.append(connection_args[0])
            plan.append((source, logical_direction, neighbor, direction_names[direction], tuple(channels)))
    return plan


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": _fabric_router_config(),
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
            "l1_small_size": 512,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(8, 1)], indirect=True)
def test_all_gather_fabric_2d_direct_ring_connection_plan(mesh_device):
    plan = _direct_ring_connection_plan(mesh_device)
    assert len(plan) == 2 * mesh_device.shape[0]
    assert all(len(set(channels)) == 2 for _, _, _, _, channels in plan)
    print("AG_DIRECT_RING_CONNECTIONS " + " ".join(str(edge) for edge in plan))


def _all_worker_cores(mesh_device):
    grid = mesh_device.compute_with_storage_grid_size()
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})


def _make_row_major_mesh_tensor(mesh_device, torch_input, dtype, mesh_mapper):
    """Create a row-major mesh tensor, preserving FP8's device-side conversion requirement."""
    tensor = ttnn.from_torch(
        torch_input,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16 if dtype == ttnn.fp8_e4m3 else dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
        device=mesh_device,
    )
    return ttnn.typecast(tensor, dtype) if dtype == ttnn.fp8_e4m3 else tensor


def _fp8_device_payloads(tensor, mesh_device):
    """Read each logical device tensor as opaque FP8 payload bytes."""
    assert tensor.dtype == ttnn.fp8_e4m3
    device_tensors = ttnn.get_device_tensors(tensor)
    local_shape = tuple(device_tensors[0].shape)
    host_bytes = (
        ttnn.to_torch(tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).contiguous().view(torch.uint8)
    )
    return host_bytes.reshape(len(device_tensors), *local_shape)


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": _fabric_router_config(),
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
            "l1_small_size": 512,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(8, 1)], indirect=True)
def test_all_gather_fabric_2d_single_axis_ring_small_direct(mesh_device):
    # Batch two keeps this diagnostic on the direct path; the persistent
    # output removes the fresh-allocation readiness multicast.
    global_shape = (2, 1, 128, 576)
    torch.manual_seed(0)
    torch_input = torch.rand(global_shape, dtype=torch.bfloat16)
    tt_input = _make_row_major_mesh_tensor(
        mesh_device,
        torch_input,
        ttnn.bfloat16,
        ttnn.ShardTensor2dMesh(mesh_device, dims=(2, None), mesh_shape=tuple(mesh_device.shape)),
    )
    persistent_output = _make_row_major_mesh_tensor(
        mesh_device,
        torch.zeros(global_shape, dtype=torch.bfloat16),
        ttnn.bfloat16,
        ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_output = ttnn.all_gather(
        tt_input,
        dim=2,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cluster_axis=0,
        output_tensor=persistent_output,
    )
    ttnn.synchronize_device(mesh_device)
    for device_tensor in ttnn.get_device_tensors(tt_output):
        assert torch.equal(ttnn.to_torch(device_tensor), torch_input)


def _reference_bank_owned_schedule(total_pages, num_banks, num_links, rows_per_run):
    """Reference the device schedule: link -> bank -> physical run -> page."""
    assert total_pages > 0 and num_banks > 0 and num_links > 0 and rows_per_run > 0
    assert total_pages % num_banks == 0
    assert num_banks % num_links == 0
    pages_per_bank = total_pages // num_banks
    schedules = []
    for link in range(num_links):
        pages = []
        for owned_bank_slot in range(num_banks // num_links):
            bank = link + owned_bank_slot * num_links
            for run_start in range(0, pages_per_bank, rows_per_run):
                pages_in_run = min(rows_per_run, pages_per_bank - run_start)
                pages.extend(bank + (run_start + page) * num_banks for page in range(pages_in_run))
        schedules.append(pages)
    return schedules


def _reference_interleaved_bank_owned_runs(total_pages, num_banks, num_links, rows_per_run):
    """Round-robin one physical run from every link-owned bank."""
    assert total_pages > 0 and total_pages % num_banks == 0
    assert num_banks > 0 and num_links > 0 and num_banks % num_links == 0
    assert rows_per_run > 0
    pages_per_bank = total_pages // num_banks
    runs = []
    for link in range(num_links):
        link_runs = []
        for run_start in range(0, pages_per_bank, rows_per_run):
            pages_in_run = min(rows_per_run, pages_per_bank - run_start)
            for owned_bank_slot in range(num_banks // num_links):
                bank = link + owned_bank_slot * num_links
                link_runs.append(tuple(bank + (run_start + page) * num_banks for page in range(pages_in_run)))
        runs.append(link_runs)
    return runs


def _reference_bank_owned_rows_per_run(max_rows, pages_per_bank, run_policy):
    """Mirror the bounded host policy used to isolate exact divisors from tail runs."""
    assert max_rows > 0 and pages_per_bank > 0
    if run_policy == "max_tail":
        return max_rows
    assert run_policy == "divisor"
    return next(rows for rows in range(min(max_rows, pages_per_bank), 0, -1) if pages_per_bank % rows == 0)


def _reference_grouped_credit_schedule(num_batches, slot_count, group_batches):
    """Return receiver publications and the credit sequence required before each slot reuse."""
    assert num_batches > 0 and slot_count > 0 and 0 < group_batches <= slot_count
    published = [
        (min(group_start + group_batches, num_batches), group_start // group_batches + 1)
        for group_start in range(0, num_batches, group_batches)
    ]
    reclaim = [0 if batch < slot_count else (batch - slot_count) // group_batches + 1 for batch in range(num_batches)]
    return published, reclaim


@pytest.mark.parametrize(
    "num_batches,slot_count,group_batches",
    [(17, 6, 1), (17, 6, 2), (17, 6, 3), (24, 12, 4), (25, 12, 6)],
)
def test_all_gather_grouped_credit_schedule_reference(num_batches, slot_count, group_batches):
    published, reclaim = _reference_grouped_credit_schedule(num_batches, slot_count, group_batches)
    assert published[-1][0] == num_batches
    assert published[-1][1] == (num_batches + group_batches - 1) // group_batches
    for batch, required_sequence in enumerate(reclaim):
        if required_sequence == 0:
            continue
        old_batch = batch - slot_count
        assert (old_batch // group_batches) + 1 == required_sequence
        assert published[required_sequence - 1][0] > old_batch


@pytest.mark.parametrize(
    "max_rows,pages_per_bank,run_policy,expected_rows",
    [
        (12, 4096, "divisor", 8),
        (20, 4096, "divisor", 16),
        (12, 4096, "max_tail", 12),
        (20, 4096, "max_tail", 20),
        (12, 4, "divisor", 4),
        (12, 4, "max_tail", 12),
    ],
)
def test_all_gather_bank_owned_run_policy_reference(max_rows, pages_per_bank, run_policy, expected_rows):
    assert _reference_bank_owned_rows_per_run(max_rows, pages_per_bank, run_policy) == expected_rows


@pytest.mark.parametrize("total_pages,rows_per_run", [(32, 4), (32768, 12), (32768, 20)])
def test_all_gather_bank_owned_schedule_reference(total_pages, rows_per_run):
    """The two link schedules must form a bijection of the source page interval."""
    num_banks = 8
    num_links = 2
    schedules = _reference_bank_owned_schedule(total_pages, num_banks, num_links, rows_per_run)
    assert len(schedules) == num_links
    assert all(len(pages) == total_pages // num_links for pages in schedules)
    assert sorted(page for pages in schedules for page in pages) == list(range(total_pages))
    assert set(schedules[0]).isdisjoint(schedules[1])
    for link, pages in enumerate(schedules):
        schedule_offset = 0
        pages_per_bank = total_pages // num_banks
        for owned_bank_slot in range(num_banks // num_links):
            expected_bank = link + owned_bank_slot * num_links
            for run_start in range(0, pages_per_bank, rows_per_run):
                pages_in_run = min(rows_per_run, pages_per_bank - run_start)
                run = pages[schedule_offset : schedule_offset + pages_in_run]
                schedule_offset += pages_in_run
                assert all(page % num_links == link for page in run)
                assert {page % num_banks for page in run} == {expected_bank}
                assert all(next_page - page == num_banks for page, next_page in zip(run, run[1:]))
        assert schedule_offset == len(pages)


@pytest.mark.parametrize("total_pages,rows_per_run", [(128, 4), (32768, 20), (65536, 20)])
def test_all_gather_interleaved_bank_receiver_schedule_reference(total_pages, rows_per_run):
    num_banks = 8
    num_links = 2
    receiver_cores_per_link = num_banks // num_links
    slot_count = 12
    schedules = _reference_interleaved_bank_owned_runs(total_pages, num_banks, num_links, rows_per_run)

    assert sorted(page for link_runs in schedules for run in link_runs for page in run) == list(range(total_pages))
    for link, link_runs in enumerate(schedules):
        assert all(run and len(run) <= rows_per_run for run in link_runs)
        for batch, run in enumerate(link_runs):
            receiver_idx = batch % receiver_cores_per_link
            assert {page % num_banks for page in run} == {link + receiver_idx * num_links}
            assert all(next_page - page == num_banks for page, next_page in zip(run, run[1:]))
        first_reuse = receiver_cores_per_link * slot_count
        assert [batch % receiver_cores_per_link for batch in range(first_reuse)] == list(
            range(receiver_cores_per_link)
        ) * slot_count


def _all_gather_profile_ns(mesh_device, run_fn, expected_receiver_l1=None, expected_unicast=None):
    """Return one all-gather's device critical path/runtime IDs and verify its implementation path."""
    _, records = profile_realtime_program(mesh_device, run_fn, collect_all=True, record_timeout_seconds=5.0)
    programs = {}
    receiver_l1_observed = False
    unicast_observed = False
    for record in records:
        normalized_sources = [source.replace("\\", "/") for source in record["kernel_sources"]]
        if not any("/ccl/all_gather/" in source for source in normalized_sources):
            continue
        receiver_l1_observed |= any(source.endswith("/multicast_receiver_writer.cpp") for source in normalized_sources)
        unicast_observed |= any(source.endswith("/unicast_writer.cpp") for source in normalized_sources)
        runtime_id = record["runtime_id"]
        programs[runtime_id] = max(programs.get(runtime_id, 0.0), record["duration_ns"])
    assert programs, "realtime profiler returned no native all-gather program"
    if expected_receiver_l1 is not None:
        assert (
            receiver_l1_observed == expected_receiver_l1
        ), f"expected receiver_l1={expected_receiver_l1}, observed receiver_l1={receiver_l1_observed}"
    if expected_unicast is not None:
        assert unicast_observed == expected_unicast, f"expected unicast={expected_unicast}, observed={unicast_observed}"
    return sum(programs.values()), frozenset(programs)


def _all_gather_duration_ns(mesh_device, run_fn, expected_receiver_l1=None, expected_unicast=None):
    duration_ns, _ = _all_gather_profile_ns(mesh_device, run_fn, expected_receiver_l1, expected_unicast)
    return duration_ns


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": _fabric_router_config(),
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
            "l1_small_size": 512,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(4, 2)], indirect=True)
@pytest.mark.parametrize(
    "cluster_axis,gather_dim,global_shape,dtype,pcc,use_persistent_output",
    [
        pytest.param(1, 3, (1, 1, 1, 512), ttnn.bfloat16, 1.0, False, id="tp_axis_2x_half_page"),
        pytest.param(1, 3, (1, 1, 1, 2048), ttnn.bfloat16, 1.0, False, id="tp_axis_2x_single_page"),
        pytest.param(1, 3, (1, 1, 1, 2048), ttnn.bfloat16, 1.0, True, id="tp_axis_2x_single_page_persistent"),
        pytest.param(1, 3, (1, 1, 32, 2048), ttnn.bfloat16, 1.0, False, id="tp_axis_2x_32_pages"),
        pytest.param(0, 2, (1, 1, 4, 1024), ttnn.bfloat16, 1.0, False, id="sp_axis_4x_single_page"),
        pytest.param(0, 2, (1, 1, 128, 1024), ttnn.bfloat16, 1.0, False, id="sp_axis_4x_32_pages"),
        # Sparse MLA KV cache: 576 BF16 values, one 1,152-byte physical row page.
        pytest.param(0, 2, (1, 1, 128, 576), ttnn.bfloat16, 1.0, True, id="sp_axis_4x_mla_kv_bf16"),
        # Twenty-five rows/device split over two links gives worker batches of
        # 12+1 and 12 rows, explicitly exercising a one-page final batch.
        pytest.param(
            0,
            2,
            (1, 1, 100, 576),
            ttnn.bfloat16,
            1.0,
            True,
            id="sp_axis_4x_mla_kv_bf16_one_page_tail",
        ),
        # Four BF16 cache rows coalesced into one 4,608-byte transport page. This is the largest
        # useful BF16 grouping below the fabric payload limit and is the layout experiment used by
        # the CCL performance harness.
        pytest.param(0, 2, (1, 1, 32, 2304), ttnn.bfloat16, 1.0, True, id="sp_axis_4x_mla_kv_bf16_4rows_page"),
        # Packed scaled-FP8 cache: 656 logical bytes, rounded to a 704-byte DRAM row page.
        pytest.param(0, 2, (1, 1, 128, 656), ttnn.fp8_e4m3, 0.99, True, id="sp_axis_4x_mla_kv_scaled_fp8"),
        # Eight packed FP8 cache rows in one 5,248-byte transport page.
        pytest.param(0, 2, (1, 1, 16, 5248), ttnn.fp8_e4m3, 0.99, True, id="sp_axis_4x_mla_kv_scaled_fp8_8rows_page"),
    ],
)
def test_all_gather_fabric_2d_row_major_2k_pages(
    mesh_device, cluster_axis, gather_dim, global_shape, dtype, pcc, use_persistent_output
):
    """Gather row-major pages using the same mesh, output lifetime, and dtype as sparse MLA."""
    torch.manual_seed(0)
    host_dtype = torch.float32 if dtype == ttnn.fp8_e4m3 else torch.bfloat16
    torch_input = torch.rand(global_shape, dtype=host_dtype)
    shard_dims = (None, gather_dim) if cluster_axis == 1 else (gather_dim, None)
    # Mesh-mapped host construction currently forces FP8_E4M3 through TILE. Build
    # the row-major BF16 transport tensor first, then typecast on device; this is
    # the same sequence used for the sparse MLA packed-cache setup.
    tt_input = ttnn.from_torch(
        torch_input,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16 if dtype == ttnn.fp8_e4m3 else dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=shard_dims, mesh_shape=tuple(mesh_device.shape)),
        device=mesh_device,
    )
    if dtype == ttnn.fp8_e4m3:
        tt_input = ttnn.typecast(tt_input, dtype)

    persistent_output = None
    if use_persistent_output:
        persistent_output = ttnn.from_torch(
            torch.zeros(global_shape, dtype=host_dtype),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16 if dtype == ttnn.fp8_e4m3 else dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            device=mesh_device,
        )
        if dtype == ttnn.fp8_e4m3:
            persistent_output = ttnn.typecast(persistent_output, dtype)

    def run_ag():
        return ttnn.all_gather(
            tt_input,
            dim=gather_dim,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cluster_axis=cluster_axis,
            output_tensor=persistent_output,
        )

    expected_receiver_l1 = use_persistent_output and cluster_axis == 0 and gather_dim == 2
    if ttnn.device.IsProgramRealtimeProfilerActive():
        tt_output, records = profile_realtime_program(mesh_device, run_ag, collect_all=True, record_timeout_seconds=5.0)
        all_sources = [source.replace("\\", "/") for record in records for source in record["kernel_sources"]]
        assert any("/ccl/all_gather/" in source for source in all_sources), "no native all-gather program observed"
        receiver_l1_observed = any(source.endswith("/multicast_receiver_writer.cpp") for source in all_sources)
        assert (
            receiver_l1_observed == expected_receiver_l1
        ), f"expected receiver_l1={expected_receiver_l1}, observed receiver_l1={receiver_l1_observed}"
    else:
        tt_output = run_ag()

    # FP8_E4M3 cannot be read back one device at a time. Convert after the
    # collective so we still validate every gathered replica numerically.
    check_output = ttnn.typecast(tt_output, ttnn.bfloat16) if dtype == ttnn.fp8_e4m3 else tt_output
    if dtype == ttnn.fp8_e4m3:
        # Compare against the values actually carried by the gather.  Comparing
        # FP8 output directly with the pre-quantized random input confounds CCL
        # correctness with FP8 conversion error.
        check_input = ttnn.typecast(tt_input, ttnn.bfloat16)
        input_shards = ttnn.get_device_tensors(check_input)
        mesh_rows, mesh_cols = tuple(mesh_device.shape)
        gather_group = (
            [input_shards[row * mesh_cols] for row in range(mesh_rows)]
            if cluster_axis == 0
            else input_shards[:mesh_cols]
        )
        expected = torch.cat([ttnn.to_torch(device_tensor) for device_tensor in gather_group], dim=gather_dim)
    else:
        expected = torch_input
    for device_index, device_tensor in enumerate(ttnn.get_device_tensors(check_output)):
        actual = ttnn.to_torch(device_tensor)
        mismatch = actual != expected
        mismatch_rows = mismatch.nonzero()[:, 2].unique().tolist() if mismatch.any() else []
        first_mismatch = mismatch.nonzero()[0].tolist() if mismatch.any() else None
        first_actual = actual[tuple(first_mismatch)].item() if first_mismatch is not None else None
        first_expected = expected[tuple(first_mismatch)].item() if first_mismatch is not None else None
        assert torch.equal(actual, expected), (
            f"device {device_index} gather mismatch: "
            f"max_abs={(actual.float() - expected.float()).abs().max().item()} "
            f"mismatched_elements={mismatch.sum().item()} first={first_mismatch} "
            f"actual={first_actual} expected={first_expected} rows={mismatch_rows}"
        )


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": _fabric_router_config(),
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
            "l1_small_size": 512,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(4, 2)], indirect=True)
def test_all_gather_fabric_2d_receiver_nd_sharded_sparse_kv_cache(mesh_device):
    """The model cache is ND-sharded DRAM, while the persistent gathered output is interleaved DRAM."""
    global_shape = (1, 1, 512, 576)
    cache = init_sparse_kv_cache(
        cache_format=SparseKVCacheFormat.BF16,
        mesh_device=mesh_device,
        seq_len=global_shape[2],
        mesh_shape=list(mesh_device.shape),
        sp_axis=0,
        num_kvpe_cache_layers=1,
    ).tensor
    persistent_output = ttnn.from_torch(
        torch.ones(global_shape, dtype=torch.bfloat16),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        device=mesh_device,
    )

    def run_ag():
        return ttnn.all_gather(
            cache,
            dim=2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cluster_axis=0,
            output_tensor=persistent_output,
        )

    if ttnn.device.IsProgramRealtimeProfilerActive():
        output, records = profile_realtime_program(mesh_device, run_ag, collect_all=True, record_timeout_seconds=5.0)
        sources = [source.replace("\\", "/") for record in records for source in record["kernel_sources"]]
        assert any(source.endswith("/multicast_receiver_writer.cpp") for source in sources)
    else:
        output = run_ag()
        ttnn.synchronize_device(mesh_device)

    expected = torch.zeros(global_shape, dtype=torch.bfloat16)
    for device_tensor in ttnn.get_device_tensors(output):
        assert_with_pcc(ttnn.to_torch(device_tensor), expected, pcc=1.0)


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": _fabric_router_config(),
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
            "l1_small_size": 512,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(4, 2)], indirect=True)
@pytest.mark.parametrize("fallback_case", ["tile_dram", "row_major_l1_sharded"])
def test_all_gather_fabric_2d_receiver_layout_fallbacks(mesh_device, fallback_case):
    """Unsupported layout/memory plans must use a correct non-receiver path."""
    torch.manual_seed(0)
    global_shape = (1, 1, 128, 576)
    torch_input = torch.rand(global_shape, dtype=torch.bfloat16)
    layout = ttnn.TILE_LAYOUT if fallback_case == "tile_dram" else ttnn.ROW_MAJOR_LAYOUT
    input_memory_config = ttnn.DRAM_MEMORY_CONFIG
    if fallback_case == "row_major_l1_sharded":
        input_memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
                [32, 576],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )

    tt_input = ttnn.from_torch(
        torch_input,
        layout=layout,
        dtype=ttnn.bfloat16,
        memory_config=input_memory_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(2, None), mesh_shape=tuple(mesh_device.shape)),
        device=mesh_device,
    )
    persistent_output = ttnn.from_torch(
        torch.zeros(global_shape, dtype=torch.bfloat16),
        layout=layout,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        device=mesh_device,
    )

    def run_ag():
        return ttnn.all_gather(
            tt_input,
            dim=2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cluster_axis=0,
            output_tensor=persistent_output,
        )

    if ttnn.device.IsProgramRealtimeProfilerActive():
        tt_output, records = profile_realtime_program(mesh_device, run_ag, collect_all=True, record_timeout_seconds=5.0)
        all_sources = [source.replace("\\", "/") for record in records for source in record["kernel_sources"]]
        assert any("/ccl/all_gather/" in source for source in all_sources), "no native all-gather program observed"
        assert not any(source.endswith("/multicast_receiver_writer.cpp") for source in all_sources)
    else:
        tt_output = run_ag()
        ttnn.synchronize_device(mesh_device)

    for device_tensor in ttnn.get_device_tensors(tt_output):
        assert_with_pcc(ttnn.to_torch(device_tensor), torch_input, pcc=1.0)


@pytest.mark.parametrize(
    "device_params",
    _MATCHED_RING_DEVICE_PARAMS,
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(8, 1)], indirect=True)
@pytest.mark.parametrize("use_persistent_output", [False, True], ids=["fresh_output", "persistent_output"])
@pytest.mark.parametrize(
    "dtype,width",
    [
        pytest.param(ttnn.bfloat16, 576, id="bf16_1152b_rows"),
        pytest.param(ttnn.fp8_e4m3, 656, id="scaled_fp8_704b_rows"),
    ],
)
def test_all_gather_matched_large_single_axis_correctness(mesh_device, dtype, width, use_persistent_output):
    """Check the same large eight-rank gather under native 1D and physical 2D Fabric."""
    assert ttnn.get_tt_fabric_max_payload_size_bytes() == 14 * 1024
    rows_per_device = 65536
    global_shape = (1, 1, rows_per_device * mesh_device.shape[0], width)
    torch.manual_seed(0)
    host_dtype = torch.float32 if dtype == ttnn.fp8_e4m3 else torch.bfloat16
    torch_input = torch.rand(global_shape, dtype=host_dtype)
    tt_input = _make_row_major_mesh_tensor(
        mesh_device,
        torch_input,
        dtype,
        ttnn.ShardTensor2dMesh(mesh_device, dims=(2, None), mesh_shape=tuple(mesh_device.shape)),
    )
    persistent_output = None
    if use_persistent_output:
        persistent_output = _make_row_major_mesh_tensor(
            mesh_device,
            torch.zeros(global_shape, dtype=host_dtype),
            dtype,
            ttnn.ReplicateTensorToMesh(mesh_device),
        )

    def run_ag():
        return ttnn.all_gather(
            tt_input,
            dim=2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cluster_axis=0,
            output_tensor=persistent_output,
        )

    if ttnn.device.IsProgramRealtimeProfilerActive():
        tt_output, records = profile_realtime_program(mesh_device, run_ag, collect_all=True, record_timeout_seconds=5.0)
        all_sources = [source.replace("\\", "/") for record in records for source in record["kernel_sources"]]
        assert any(source.endswith("/unicast_writer.cpp") for source in all_sources)
        assert not any(source.endswith("/multicast_receiver_writer.cpp") for source in all_sources)
    else:
        tt_output = run_ag()
        ttnn.synchronize_device(mesh_device)

    if dtype == ttnn.fp8_e4m3:
        input_shards = _fp8_device_payloads(tt_input, mesh_device)
        output_shards = _fp8_device_payloads(tt_output, mesh_device)
        expected = torch.cat(list(input_shards), dim=2)
    else:
        output_shards = [ttnn.to_torch(device_tensor) for device_tensor in ttnn.get_device_tensors(tt_output)]
        expected = torch_input
    for actual in output_shards:
        assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": _fabric_router_config(),
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
            "l1_small_size": 2048,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(8, 1)], indirect=True)
@pytest.mark.parametrize(
    "dtype,width,expected_page_size",
    [
        pytest.param(ttnn.bfloat16, 576, 1152, id="bf16_1152b_rows"),
        pytest.param(ttnn.fp8_e4m3, 656, 704, id="scaled_fp8_704b_rows"),
    ],
)
def test_all_gather_fabric_2d_matched_large_cached_stability(mesh_device, dtype, width, expected_page_size):
    """Exercise ten consecutive cached 512K gathers and bound their latency spread."""
    if not ttnn.device.IsProgramRealtimeProfilerActive():
        pytest.skip("native all-gather stability test requires an active realtime device profiler")

    rows_per_device = 65536
    global_shape = (1, 1, rows_per_device * mesh_device.shape[0], width)
    torch.manual_seed(0)
    host_dtype = torch.float32 if dtype == ttnn.fp8_e4m3 else torch.bfloat16
    torch_input = torch.rand(global_shape, dtype=host_dtype)
    tt_input = _make_row_major_mesh_tensor(
        mesh_device,
        torch_input,
        dtype,
        ttnn.ShardTensor2dMesh(mesh_device, dims=(2, None), mesh_shape=tuple(mesh_device.shape)),
    )
    persistent_output = _make_row_major_mesh_tensor(
        mesh_device,
        torch.zeros(global_shape, dtype=host_dtype),
        dtype,
        ttnn.ReplicateTensorToMesh(mesh_device),
    )
    page_size = ttnn.get_device_tensors(tt_input)[0].buffer_aligned_page_size()
    assert page_size == expected_page_size

    def run_ag():
        return ttnn.all_gather(
            tt_input,
            dim=2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cluster_axis=0,
            output_tensor=persistent_output,
        )

    # Compile once, then drain one profiler delivery so the ten samples below
    # each correspond to exactly one cached all-gather dispatch.
    run_ag()
    ttnn.synchronize_device(mesh_device)
    _all_gather_profile_ns(mesh_device, run_ag, expected_receiver_l1=False, expected_unicast=True)
    cache_entries = mesh_device.num_program_cache_entries()
    durations_ns = [
        _all_gather_profile_ns(mesh_device, run_ag, expected_receiver_l1=False, expected_unicast=True)[0]
        for _ in range(10)
    ]
    assert mesh_device.num_program_cache_entries() == cache_entries

    median_ns = statistics.median(durations_ns)
    p90_ns = sorted(durations_ns)[8]
    assert p90_ns <= 1.05 * median_ns, (
        f"unstable cached Fabric2D all-gather: median={median_ns / 1e6:.3f} ms "
        f"p90={p90_ns / 1e6:.3f} ms samples_ms={[round(value / 1e6, 3) for value in durations_ns]}"
    )
    print(
        f"ISOLATED_AG_STABILITY fabric=fabric_2d dtype={dtype} runs=10 "
        f"median={median_ns / 1e6:.3f}ms p90={p90_ns / 1e6:.3f}ms "
        f"effective_receive_bw={rows_per_device * page_size * (mesh_device.shape[0] - 1) / median_ns:.3f}GB/s "
        f"samples_ms={[round(value / 1e6, 3) for value in durations_ns]}"
    )

    if dtype == ttnn.fp8_e4m3:
        input_shards = _fp8_device_payloads(tt_input, mesh_device)
        output_shards = _fp8_device_payloads(persistent_output, mesh_device)
        expected = torch.cat(list(input_shards), dim=2)
    else:
        output_shards = [ttnn.to_torch(device_tensor) for device_tensor in ttnn.get_device_tensors(persistent_output)]
        expected = torch_input
    for actual in output_shards:
        assert torch.equal(actual, expected)


def _make_matched_two_rank_line_tensors(mesh_device, dtype, width, cluster_axis=1, rows_per_device=65536):
    assert mesh_device.shape[cluster_axis] == 2
    global_shape = (1, 1, 2 * rows_per_device, width)
    torch.manual_seed(0)
    host_dtype = torch.float32 if dtype == ttnn.fp8_e4m3 else torch.bfloat16
    torch_input = torch.rand(global_shape, dtype=host_dtype)
    shard_dims = (2, None) if cluster_axis == 0 else (None, 2)
    tt_input = _make_row_major_mesh_tensor(
        mesh_device,
        torch_input,
        dtype,
        ttnn.ShardTensor2dMesh(mesh_device, dims=shard_dims, mesh_shape=tuple(mesh_device.shape)),
    )
    persistent_output = _make_row_major_mesh_tensor(
        mesh_device,
        torch.zeros(global_shape, dtype=host_dtype),
        dtype,
        ttnn.ReplicateTensorToMesh(mesh_device),
    )
    return rows_per_device, torch_input, tt_input, persistent_output


@pytest.mark.parametrize(
    "device_params",
    _MATCHED_RING_DEVICE_PARAMS,
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(4, 2)], indirect=True)
@pytest.mark.parametrize("use_persistent_output", [False, True], ids=["fresh_output", "persistent_output"])
@pytest.mark.parametrize(
    "dtype,width",
    [
        pytest.param(ttnn.bfloat16, 576, id="bf16_1152b_rows"),
        pytest.param(ttnn.fp8_e4m3, 656, id="scaled_fp8_704b_rows"),
    ],
)
def test_all_gather_matched_two_rank_line_correctness(mesh_device, dtype, width, use_persistent_output):
    """Exercise terminal one-hop traffic without a store-and-forward relay iteration."""
    _, torch_input, tt_input, persistent_output = _make_matched_two_rank_line_tensors(mesh_device, dtype, width)

    def run_ag():
        return ttnn.all_gather(
            tt_input,
            dim=2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cluster_axis=1,
            output_tensor=persistent_output if use_persistent_output else None,
        )

    if ttnn.device.IsProgramRealtimeProfilerActive():
        tt_output, records = profile_realtime_program(mesh_device, run_ag, collect_all=True, record_timeout_seconds=5.0)
        sources = [source.replace("\\", "/") for record in records for source in record["kernel_sources"]]
        assert any(source.endswith("/unicast_writer.cpp") for source in sources)
        assert not any(source.endswith("/multicast_receiver_writer.cpp") for source in sources)
    else:
        tt_output = run_ag()
        ttnn.synchronize_device(mesh_device)

    if dtype == ttnn.fp8_e4m3:
        input_shards = _fp8_device_payloads(tt_input, mesh_device)
        output_shards = _fp8_device_payloads(tt_output, mesh_device)
        for group_start in range(0, input_shards.shape[0], 2):
            expected = torch.cat(list(input_shards[group_start : group_start + 2]), dim=2)
            for actual in output_shards[group_start : group_start + 2]:
                assert torch.equal(actual, expected)
    else:
        expected = torch_input
        check_output = tt_output

        for device_tensor in ttnn.get_device_tensors(check_output):
            actual = ttnn.to_torch(device_tensor)
            assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": _fabric_router_config(),
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
            "l1_small_size": 512,
            # Fabric2D router initialization requires the peer at every live
            # physical link. Run this topology qualification only on a real
            # four-device QuietBox rather than opening half of a larger box.
            "require_exact_physical_num_devices": True,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
@pytest.mark.requires_mesh_topology(mesh_shape=(2, 2), topology="mesh-2x2")
@pytest.mark.parametrize("use_persistent_output", [False, True], ids=["fresh_output", "persistent_output"])
@pytest.mark.parametrize(
    "dtype,width",
    [
        pytest.param(ttnn.bfloat16, 576, id="bf16_1152b_rows"),
        pytest.param(ttnn.fp8_e4m3, 656, id="scaled_fp8_704b_rows"),
    ],
)
def test_all_gather_fabric_2d_quietbox_sized_concurrent_sp_lines(mesh_device, dtype, width, use_persistent_output):
    """Qualify a physical QuietBox SP=2 x TP=2 as two concurrent native SP lines."""
    cluster_axis = 0
    _, torch_input, tt_input, persistent_output = _make_matched_two_rank_line_tensors(
        mesh_device, dtype, width, cluster_axis=cluster_axis, rows_per_device=128
    )

    def run_ag():
        return ttnn.all_gather(
            tt_input,
            dim=2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cluster_axis=cluster_axis,
            output_tensor=persistent_output if use_persistent_output else None,
        )

    if ttnn.device.IsProgramRealtimeProfilerActive():
        tt_output, records = profile_realtime_program(mesh_device, run_ag, collect_all=True, record_timeout_seconds=5.0)
        sources = [source.replace("\\", "/") for record in records for source in record["kernel_sources"]]
        assert any(source.endswith("/unicast_writer.cpp") for source in sources)
        assert not any(source.endswith("/multicast_receiver_writer.cpp") for source in sources)
    else:
        tt_output = run_ag()
        ttnn.synchronize_device(mesh_device)

    if dtype == ttnn.fp8_e4m3:
        input_shards = _fp8_device_payloads(tt_input, mesh_device)
        output_shards = _fp8_device_payloads(tt_output, mesh_device)
        mesh_rows, mesh_cols = tuple(mesh_device.shape)
        for device_index, actual in enumerate(output_shards):
            column = device_index % mesh_cols
            expected = torch.cat([input_shards[row * mesh_cols + column] for row in range(mesh_rows)], dim=2)
            assert torch.equal(actual, expected)
    else:
        for device_tensor in ttnn.get_device_tensors(tt_output):
            assert torch.equal(ttnn.to_torch(device_tensor), torch_input)


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": _fabric_router_config(),
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
            "l1_small_size": 512,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device,cluster_axis",
    [pytest.param((4, 2), 0, id="axis0_4rank"), pytest.param((2, 4), 1, id="axis1_4rank")],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("use_persistent_output", [False, True], ids=["fresh_output", "persistent_output"])
@pytest.mark.parametrize("input_storage", ["interleaved", "nd_sharded"])
@pytest.mark.parametrize(
    "dtype,width",
    [
        pytest.param(ttnn.bfloat16, 576, id="bf16_1152b_rows"),
        pytest.param(ttnn.fp8_e4m3, 656, id="scaled_fp8_704b_rows"),
    ],
)
def test_all_gather_fabric_2d_matched_four_rank_line_correctness(
    mesh_device, cluster_axis, use_persistent_output, input_storage, dtype, width
):
    """Exercise the automatic one-hop backend on four-rank lines in both mesh orientations."""
    ranks = mesh_device.shape[cluster_axis]
    assert ranks == 4
    rows_per_device = 128
    global_shape = (1, 1, rows_per_device * ranks, width)
    torch.manual_seed(0)
    host_dtype = torch.float32 if dtype == ttnn.fp8_e4m3 else torch.bfloat16
    torch_input = torch.rand(global_shape, dtype=host_dtype)
    shard_dims = (2, None) if cluster_axis == 0 else (None, 2)
    tt_input = _make_row_major_mesh_tensor(
        mesh_device,
        torch_input,
        dtype,
        ttnn.ShardTensor2dMesh(mesh_device, dims=shard_dims, mesh_shape=tuple(mesh_device.shape)),
    )
    if input_storage == "nd_sharded":
        tt_input = ttnn.to_memory_config(tt_input, _sparse_mla_nd_sharded_dram_config(mesh_device, width))
    else:
        assert input_storage == "interleaved"
    persistent_output = None
    if use_persistent_output:
        persistent_output = _make_row_major_mesh_tensor(
            mesh_device,
            torch.zeros(global_shape, dtype=host_dtype),
            dtype,
            ttnn.ReplicateTensorToMesh(mesh_device),
        )

    def run_ag():
        return ttnn.all_gather(
            tt_input,
            dim=2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cluster_axis=cluster_axis,
            output_tensor=persistent_output,
        )

    if ttnn.device.IsProgramRealtimeProfilerActive():
        tt_output, records = profile_realtime_program(mesh_device, run_ag, collect_all=True, record_timeout_seconds=5.0)
        sources = [source.replace("\\", "/") for record in records for source in record["kernel_sources"]]
        assert any(source.endswith("/unicast_writer.cpp") for source in sources)
        assert not any(source.endswith("/multicast_receiver_writer.cpp") for source in sources)
    else:
        tt_output = run_ag()
        ttnn.synchronize_device(mesh_device)

    if dtype == ttnn.fp8_e4m3:
        input_shards = _fp8_device_payloads(tt_input, mesh_device)
        output_shards = _fp8_device_payloads(tt_output, mesh_device)
        mesh_rows, mesh_cols = tuple(mesh_device.shape)
        expected_by_device = []
        for device_index in range(input_shards.shape[0]):
            if cluster_axis == 0:
                column = device_index % mesh_cols
                gather_group = [input_shards[row * mesh_cols + column] for row in range(mesh_rows)]
            else:
                row_start = (device_index // mesh_cols) * mesh_cols
                gather_group = list(input_shards[row_start : row_start + mesh_cols])
            expected_by_device.append(torch.cat(gather_group, dim=2))
    else:
        output_shards = [ttnn.to_torch(device_tensor) for device_tensor in ttnn.get_device_tensors(tt_output)]
        expected_by_device = [torch_input] * len(output_shards)
    for actual, expected in zip(output_shards, expected_by_device):
        assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    "device_params",
    _MATCHED_RING_DEVICE_PARAMS,
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(4, 2)], indirect=True)
@pytest.mark.parametrize(
    "dtype,width,expected_page_size",
    [
        pytest.param(ttnn.bfloat16, 576, 1152, id="bf16_1152b_rows"),
        pytest.param(ttnn.fp8_e4m3, 656, 704, id="scaled_fp8_704b_rows"),
    ],
)
def test_all_gather_matched_two_rank_line_perf(mesh_device, dtype, width, expected_page_size):
    """Measure the terminal one-hop ceiling separately from relay iterations."""
    if not ttnn.device.IsProgramRealtimeProfilerActive():
        pytest.skip("native all-gather perf benchmark requires an active realtime device profiler")

    rows_per_device, _, tt_input, persistent_output = _make_matched_two_rank_line_tensors(mesh_device, dtype, width)
    page_size = ttnn.get_device_tensors(tt_input)[0].buffer_aligned_page_size()
    assert page_size == expected_page_size

    def run_ag():
        return ttnn.all_gather(
            tt_input,
            dim=2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cluster_axis=1,
            output_tensor=persistent_output,
        )

    run_ag()
    ttnn.synchronize_device(mesh_device)
    run_ag()
    ttnn.synchronize_device(mesh_device)
    _all_gather_profile_ns(mesh_device, run_ag, expected_receiver_l1=False, expected_unicast=True)
    durations_ns = [
        _all_gather_profile_ns(mesh_device, run_ag, expected_receiver_l1=False, expected_unicast=True)[0]
        for _ in range(7)
    ]
    median_ns = statistics.median(durations_ns)
    min_ns = min(durations_ns)
    p90_ns = sorted(durations_ns)[6]
    bandwidth_gbps = rows_per_device * page_size / median_ns
    print(
        f"ISOLATED_AG_TERMINAL fabric={_fabric_name(ttnn.get_fabric_config())} "
        f"dtype={dtype} ranks=2 links=2 rows_per_device={rows_per_device} page_size={page_size}B "
        f"median={median_ns / 1e6:.3f}ms min={min_ns / 1e6:.3f}ms p90={p90_ns / 1e6:.3f}ms "
        f"effective_receive_bw={bandwidth_gbps:.3f}GB/s "
        f"samples_ms={[round(value / 1e6, 3) for value in durations_ns]}"
    )


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": _fabric_router_config(),
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
            "l1_small_size": 1024,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(4, 2)], indirect=True)
@pytest.mark.parametrize(
    "dtype,width,expected_page_size",
    [
        pytest.param(ttnn.bfloat16, 576, 1152, id="bf16_1152b_rows"),
        pytest.param(ttnn.fp8_e4m3, 656, 704, id="scaled_fp8_704b_rows"),
    ],
)
@pytest.mark.parametrize("input_storage", ["interleaved", "nd_sharded"])
def test_all_gather_sparse_mla_long_four_rank_line_perf(mesh_device, dtype, width, expected_page_size, input_storage):
    """Exact large gather behind the LoudBox 4x2 sparse-MLA long proxy.

    The two mesh columns execute independent four-rank SP lines concurrently.
    Each rank owns the same 64,640-row cache depth as one Galaxy SP rank in the
    512K-token case, and writes into the same persistent interleaved-DRAM shape.
    """
    if not ttnn.device.IsProgramRealtimeProfilerActive():
        pytest.skip("native all-gather perf benchmark requires an active realtime device profiler")

    ranks = mesh_device.shape[0]
    assert tuple(mesh_device.shape) == (4, 2)
    rows_per_device = 64640
    global_shape = (1, 1, rows_per_device * ranks, width)
    if input_storage == "interleaved":
        torch.manual_seed(0)
        torch_input = torch.rand(global_shape, dtype=torch.bfloat16)
        tt_input = _make_row_major_mesh_tensor(
            mesh_device,
            torch_input,
            dtype,
            ttnn.ShardTensor2dMesh(mesh_device, dims=(2, None), mesh_shape=tuple(mesh_device.shape)),
        )
    else:
        assert input_storage == "nd_sharded"
        tt_input = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, rows_per_device, width]),
            dtype,
            ttnn.ROW_MAJOR_LAYOUT,
            mesh_device,
            _sparse_mla_nd_sharded_dram_config(mesh_device, width),
        )
        # Match sparse-MLA's cache construction: every chip owns a local
        # block-cyclic shard even though its distributed topology is replicated.
        mesh_coords = list(ttnn.MeshCoordinateRange(ttnn.MeshShape(*tuple(mesh_device.shape))))
        tt_input.update_tensor_topology(
            ttnn.TensorTopology(
                ttnn.MeshShape([mesh_device.shape[0] * mesh_device.shape[1]]),
                [ttnn.PlacementReplicate()],
                mesh_coords,
            )
        )
    persistent_output = ttnn.empty(
        global_shape,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    page_size = ttnn.get_device_tensors(tt_input)[0].buffer_aligned_page_size()
    assert page_size == expected_page_size

    def run_ag():
        return ttnn.all_gather(
            tt_input,
            dim=2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cluster_axis=0,
            output_tensor=persistent_output,
        )

    run_ag()
    ttnn.synchronize_device(mesh_device)
    run_ag()
    ttnn.synchronize_device(mesh_device)
    _all_gather_profile_ns(mesh_device, run_ag, expected_receiver_l1=False, expected_unicast=True)
    durations_ns = [
        _all_gather_profile_ns(mesh_device, run_ag, expected_receiver_l1=False, expected_unicast=True)[0]
        for _ in range(7)
    ]
    median_ns = statistics.median(durations_ns)
    min_ns = min(durations_ns)
    p90_ns = sorted(durations_ns)[6]
    bytes_received_per_rank = rows_per_device * page_size * (ranks - 1)
    bandwidth_gbps = bytes_received_per_rank / median_ns
    assert bandwidth_gbps >= 40.0, (
        f"four-rank sparse-MLA line bandwidth regressed: {bandwidth_gbps:.3f} GB/s; "
        "small-row payloads should use destination-bank packet coalescing"
    )
    print(
        f"ISOLATED_AG_FOUR_RANK_LINE fabric={_fabric_name(ttnn.get_fabric_config())} "
        f"dtype={dtype} input_storage={input_storage} ranks={ranks} "
        f"concurrent_lines={mesh_device.shape[1]} links=2 "
        f"rows_per_device={rows_per_device} page_size={page_size}B "
        f"median={median_ns / 1e6:.3f}ms min={min_ns / 1e6:.3f}ms p90={p90_ns / 1e6:.3f}ms "
        f"effective_receive_bw={bandwidth_gbps:.3f}GB/s "
        f"samples_ms={[round(value / 1e6, 3) for value in durations_ns]}"
    )


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": _fabric_router_config(),
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
            "l1_small_size": 512,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(4, 2)], indirect=True)
@pytest.mark.parametrize(
    "batch_size,batch_slice_idx,valid_gather_extent,dtype,pcc,use_persistent_output,core_plan,expected_receiver_l1",
    [
        pytest.param(2, 1, None, ttnn.bfloat16, 1.0, True, "default", True, id="bf16_batch_slice_receiver"),
        pytest.param(1, None, 8, ttnn.bfloat16, 1.0, True, "default", True, id="bf16_partial_extent_receiver"),
        pytest.param(2, 1, 8, ttnn.bfloat16, 1.0, True, "default", True, id="bf16_batch_slice_partial_extent_receiver"),
        pytest.param(2, 1, None, ttnn.fp8_e4m3, 0.99, True, "default", True, id="fp8_batch_slice_receiver"),
        pytest.param(1, None, 8, ttnn.fp8_e4m3, 0.99, True, "default", True, id="fp8_partial_extent_receiver"),
        pytest.param(2, 1, 8, ttnn.fp8_e4m3, 0.99, True, "default", True, id="fp8_batch_slice_partial_extent_receiver"),
        pytest.param(1, None, None, ttnn.bfloat16, 1.0, False, "default", False, id="fresh_output_fallback"),
        pytest.param(2, None, None, ttnn.bfloat16, 1.0, True, "default", False, id="multi_batch_fallback"),
        pytest.param(1, None, None, ttnn.bfloat16, 1.0, True, "direct_only", False, id="insufficient_receiver_cores"),
    ],
)
def test_all_gather_fabric_2d_sparse_mla_selection_paths(
    mesh_device,
    batch_size,
    batch_slice_idx,
    valid_gather_extent,
    dtype,
    pcc,
    use_persistent_output,
    core_plan,
    expected_receiver_l1,
):
    """Select receiver-L1 for MLA cache selection and retain a proved fallback."""
    width = 656 if dtype == ttnn.fp8_e4m3 else 576
    global_shape = (batch_size, 1, 128, width)
    torch.manual_seed(0)
    host_dtype = torch.float32 if dtype == ttnn.fp8_e4m3 else torch.bfloat16
    torch_input = torch.rand(global_shape, dtype=host_dtype)
    tt_input = _make_row_major_mesh_tensor(
        mesh_device,
        torch_input,
        dtype,
        ttnn.ShardTensor2dMesh(mesh_device, dims=(2, None), mesh_shape=tuple(mesh_device.shape)),
    )

    output_batch = 1 if batch_slice_idx is not None else batch_size
    output_shape = (output_batch, 1, global_shape[2], global_shape[3])
    persistent_output = None
    if use_persistent_output:
        persistent_output = _make_row_major_mesh_tensor(
            mesh_device,
            torch.zeros(output_shape, dtype=host_dtype),
            dtype,
            ttnn.ReplicateTensorToMesh(mesh_device),
        )

    sub_core_grid = None
    if core_plan == "direct_only":
        # Two links need two sender cores for direct multicast, while the
        # receiver path needs one additional mirrored drain core per link.
        sub_core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))})
    else:
        assert core_plan == "default"

    def run_ag():
        return ttnn.all_gather(
            tt_input,
            dim=2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cluster_axis=0,
            output_tensor=persistent_output,
            sub_core_grids=sub_core_grid,
            batch_slice_idx=batch_slice_idx,
            valid_gather_extent=valid_gather_extent,
        )

    if ttnn.device.IsProgramRealtimeProfilerActive():
        tt_output, records = profile_realtime_program(mesh_device, run_ag, collect_all=True, record_timeout_seconds=5.0)
        all_sources = [source.replace("\\", "/") for record in records for source in record["kernel_sources"]]
        assert any("/ccl/all_gather/" in source for source in all_sources), "no native all-gather program observed"
        receiver_l1_observed = any(source.endswith("/multicast_receiver_writer.cpp") for source in all_sources)
        assert (
            receiver_l1_observed == expected_receiver_l1
        ), f"expected receiver_l1={expected_receiver_l1}, observed receiver_l1={receiver_l1_observed}"
    else:
        tt_output = run_ag()
        ttnn.synchronize_device(mesh_device)

    selected_input = torch_input[batch_slice_idx : batch_slice_idx + 1] if batch_slice_idx is not None else torch_input
    expected = torch.zeros_like(selected_input)
    local_extent = global_shape[2] // mesh_device.shape[0]
    selected_extent = valid_gather_extent if valid_gather_extent is not None else local_extent
    for source in range(mesh_device.shape[0]):
        start = source * local_extent
        expected[:, :, start : start + selected_extent, :] = selected_input[:, :, start : start + selected_extent, :]

    check_output = ttnn.typecast(tt_output, ttnn.bfloat16) if dtype == ttnn.fp8_e4m3 else tt_output
    for device_tensor in ttnn.get_device_tensors(check_output):
        assert_with_pcc(ttnn.to_torch(device_tensor), expected, pcc=pcc)


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": _fabric_router_config(),
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
            # A four-device BF16 receiver needs seven aligned semaphore slots
            # per participating core (112 B with the current 16-B alignment).
            # Keep enough room for the direct path's barrier but not the
            # receiver protocol's complete control allocation.
            "l1_small_size": 64,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(4, 2)], indirect=True)
def test_all_gather_fabric_2d_receiver_l1_small_capacity_policy(mesh_device):
    """Automatic selection falls back before dispatch when L1-small is insufficient."""
    torch.manual_seed(0)
    global_shape = (1, 1, 128, 576)
    torch_input = torch.rand(global_shape, dtype=torch.bfloat16)
    tt_input = _make_row_major_mesh_tensor(
        mesh_device,
        torch_input,
        ttnn.bfloat16,
        ttnn.ShardTensor2dMesh(mesh_device, dims=(2, None), mesh_shape=tuple(mesh_device.shape)),
    )
    persistent_output = _make_row_major_mesh_tensor(
        mesh_device,
        torch.zeros(global_shape, dtype=torch.bfloat16),
        ttnn.bfloat16,
        ttnn.ReplicateTensorToMesh(mesh_device),
    )

    def run_ag():
        return ttnn.all_gather(
            tt_input,
            dim=2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cluster_axis=0,
            output_tensor=persistent_output,
        )

    if ttnn.device.IsProgramRealtimeProfilerActive():
        tt_output, records = profile_realtime_program(mesh_device, run_ag, collect_all=True, record_timeout_seconds=5.0)
        all_sources = [source.replace("\\", "/") for record in records for source in record["kernel_sources"]]
        assert any("/ccl/all_gather/" in source for source in all_sources), "no native all-gather program observed"
        assert not any(source.endswith("/multicast_receiver_writer.cpp") for source in all_sources)
    else:
        tt_output = run_ag()
        ttnn.synchronize_device(mesh_device)

    for device_tensor in ttnn.get_device_tensors(tt_output):
        assert_with_pcc(ttnn.to_torch(device_tensor), torch_input, pcc=1.0)


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": _fabric_router_config(),
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
            "l1_small_size": 512,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(4, 2)], indirect=True)
def test_all_gather_async_fabric_2d_control_row_major_2k_page(mesh_device):
    """Control: the pre-multicast MLA all-gather with the same 4x2 TP geometry."""
    torch.manual_seed(0)
    global_shape = (1, 1, 1, 2048)
    torch_input = torch.rand(global_shape, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(
        torch_input,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 3), mesh_shape=tuple(mesh_device.shape)),
        device=mesh_device,
    )

    worker_cores = _all_worker_cores(mesh_device)
    semaphores = [ttnn.create_global_semaphore(mesh_device, worker_cores, 0) for _ in range(2)]
    barrier_semaphore = ttnn.create_global_semaphore(mesh_device, worker_cores, 0)
    tt_output = ttnn.experimental.all_gather_async(
        tt_input,
        dim=3,
        cluster_axis=1,
        multi_device_global_semaphore=semaphores,
        barrier_semaphore=barrier_semaphore,
        num_links=2,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ttnn.Topology.Ring,
    )
    ttnn.synchronize_device(mesh_device)

    for device_tensor in ttnn.get_device_tensors(tt_output):
        assert_with_pcc(ttnn.to_torch(device_tensor), torch_input, pcc=1.0)


@pytest.mark.parametrize(
    "device_params",
    _MATCHED_RING_DEVICE_PARAMS,
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(8, 1)], indirect=True)
@pytest.mark.parametrize(
    "dtype,width,expected_page_size",
    [
        pytest.param(ttnn.bfloat16, 576, 1152, id="bf16_1152b_rows"),
        pytest.param(ttnn.fp8_e4m3, 656, 704, id="scaled_fp8_704b_rows"),
    ],
)
def test_all_gather_matched_sparse_mla_row_perf(mesh_device, dtype, width, expected_page_size):
    """Matched native AG bandwidth for the sparse-MLA SP row geometry.

    The benchmark intentionally times only the compiled all-gather program: input/output creation,
    FP8 conversion, and correctness readback happen outside the realtime-profiler region.
    """
    if not ttnn.device.IsProgramRealtimeProfilerActive():
        pytest.skip("native all-gather perf benchmark requires an active realtime device profiler")

    rows_per_device = 65536
    rows_per_page = 1
    valid_gather_extent = None
    samples = 7
    perf_core_rect = None
    active_fabric_config = ttnn.get_fabric_config()
    perf_fabric_config = _fabric_name(active_fabric_config)
    perf_l1_small_size = 2048
    perf_sub_core_grid = None
    sp = mesh_device.shape[0]
    selected_pages_per_device = valid_gather_extent or rows_per_device // rows_per_page
    expected_receiver_l1 = False
    expected_unicast = True
    packet_payload_size = ttnn.get_tt_fabric_max_payload_size_bytes()
    assert packet_payload_size == 14 * 1024, f"matched benchmark requires 14336-B payload, got {packet_payload_size}"
    route_diagnostics = _logical_ring_route_diagnostics(
        mesh_device, active_fabric_config, neighbor_unicast=expected_unicast
    )
    element_size = 1 if dtype == ttnn.fp8_e4m3 else 2
    transport_page_size = ((width * rows_per_page * element_size + 63) // 64) * 64
    if transport_page_size > 14 * 1024:
        pytest.skip(f"{transport_page_size}-byte page exceeds this benchmark's configured 14-KiB fabric payload")

    # Row coalescing is an isolated layout experiment: it holds the total logical KV bytes constant
    # while replacing N token-row pages with N / rows_per_page larger transport pages. A production
    # version would need sparse-SDPA's index mapping to address a row inside such a page.
    global_shape = (1, 1, rows_per_device * sp // rows_per_page, width * rows_per_page)
    torch.manual_seed(0)
    torch_input = torch.rand(global_shape, dtype=torch.bfloat16)
    tt_input = _make_row_major_mesh_tensor(
        mesh_device,
        torch_input,
        dtype,
        ttnn.ShardTensor2dMesh(mesh_device, dims=(2, None), mesh_shape=tuple(mesh_device.shape)),
    )
    persistent_output = _make_row_major_mesh_tensor(
        mesh_device,
        torch.zeros(global_shape, dtype=torch.bfloat16),
        dtype,
        ttnn.ReplicateTensorToMesh(mesh_device),
    )
    page_size = ttnn.get_device_tensors(tt_input)[0].buffer_aligned_page_size()
    assert page_size == transport_page_size, f"expected {transport_page_size}-byte transport page, got {page_size}"
    if rows_per_page == 1:
        assert page_size == expected_page_size, f"expected {expected_page_size}-byte rows, got {page_size}"
    receiver_batch_rows = max(1, packet_payload_size // page_size)
    receiver_drain_riscs = 2

    def run_ag():
        return ttnn.all_gather(
            tt_input,
            dim=2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cluster_axis=0,
            output_tensor=persistent_output,
            sub_core_grids=perf_sub_core_grid,
            valid_gather_extent=valid_gather_extent,
        )

    # Compile and warm the program before taking device measurements.
    run_ag()
    ttnn.synchronize_device(mesh_device)
    run_ag()
    ttnn.synchronize_device(mesh_device)

    # The realtime-profiler delivery thread can still publish records from the
    # untimed warmups after the callback for the first measured sample is
    # registered. Drain one profiled invocation so every reported sample owns
    # exactly one native all-gather dispatch.
    _all_gather_profile_ns(
        mesh_device, run_ag, expected_receiver_l1=expected_receiver_l1, expected_unicast=expected_unicast
    )
    profile_samples = [
        _all_gather_profile_ns(
            mesh_device, run_ag, expected_receiver_l1=expected_receiver_l1, expected_unicast=expected_unicast
        )
        for _ in range(samples)
    ]
    durations_ns = [duration_ns for duration_ns, _ in profile_samples]
    median_ns = statistics.median(durations_ns)
    min_ns = min(durations_ns)
    p90_ns = sorted(durations_ns)[max(0, (9 * len(durations_ns) + 9) // 10 - 1)]
    # Each SP rank receives data from the other SP-1 ranks. This is effective receive bandwidth,
    # kept deliberately independent of the fabric algorithm's internal forwarding traffic.
    bytes_received_per_rank = selected_pages_per_device * page_size * (sp - 1)
    bandwidth_gbps = bytes_received_per_rank / median_ns
    print(
        f"ISOLATED_AG policy=automatic path={'neighbor_unicast' if expected_unicast else 'multicast'} "
        f"schedule=store_and_forward batch_rows={receiver_batch_rows} "
        f"credit=data_valid drain_riscs={receiver_drain_riscs} terminal_offload=disabled "
        f"core_rect={perf_core_rect or 'auto'} fabric={perf_fabric_config} l1_small={perf_l1_small_size}B "
        f"packet_payload={packet_payload_size}B {route_diagnostics} "
        f"dtype={dtype} rows_per_device={rows_per_device} rows_per_page={rows_per_page} "
        f"valid_gather_extent={valid_gather_extent or 'full'} "
        f"page_size={page_size}B "
        f"median={median_ns / 1e6:.3f}ms min={min_ns / 1e6:.3f}ms p90={p90_ns / 1e6:.3f}ms "
        f"effective_receive_bw={bandwidth_gbps:.3f}GB/s "
        f"samples_ms={[round(duration / 1e6, 3) for duration in durations_ns]}"
    )
