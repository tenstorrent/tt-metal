# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Sparse-MLA Fabric 2D all-gather regression coverage.

The sparse MLA 4x2 path gathers row-major KV-cache rows on the SP axis.  Its
BF16 cache has 1,152-byte rows; its scaled-FP8 cache packs each 656-byte row
into a 704-byte aligned DRAM page.  Keep these tests small so a Fabric routing
failure can be diagnosed without model execution.
"""

import os
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


def _all_worker_cores(mesh_device):
    grid = mesh_device.compute_with_storage_grid_size()
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})


def _parse_perf_core_rect(value):
    """Parse an inclusive logical-core rectangle used only by the isolated placement sweep."""
    if value in {None, "auto"}:
        return None
    parts = value.split(",")
    if len(parts) != 4:
        raise ValueError("TTNN_AG_PERF_CORE_RECT must be auto or x0,y0,x1,y1")
    try:
        x0, y0, x1, y1 = (int(part) for part in parts)
    except ValueError as error:
        raise ValueError("TTNN_AG_PERF_CORE_RECT coordinates must be integers") from error
    if min(x0, y0, x1, y1) < 0 or x1 < x0 or y1 < y0:
        raise ValueError("TTNN_AG_PERF_CORE_RECT must be a nonnegative inclusive rectangle")
    return x0, y0, x1, y1


def _parse_perf_fabric_config(value):
    """Select the isolated benchmark's routed mesh without changing its tensor geometry."""
    if value in {None, "mesh"}:
        return ttnn.FabricConfig.FABRIC_2D
    if value == "ring":
        return ttnn.FabricConfig.FABRIC_1D_RING
    if value == "torus_y":
        return ttnn.FabricConfig.FABRIC_2D_TORUS_Y
    raise ValueError("TTNN_AG_PERF_FABRIC_CONFIG must be mesh, ring, or torus_y")


def _parse_perf_mesh_shape(value):
    """Parse the isolated benchmark mesh, for example ``4x2`` or ``8x1``."""
    if value is None:
        return (4, 2)
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise ValueError("TTNN_AG_PERF_MESH_SHAPE must have the form rowsxcolumns")
    try:
        rows, columns = (int(part) for part in parts)
    except ValueError as error:
        raise ValueError("TTNN_AG_PERF_MESH_SHAPE dimensions must be integers") from error
    if rows <= 0 or columns <= 0:
        raise ValueError("TTNN_AG_PERF_MESH_SHAPE dimensions must be positive")
    return rows, columns


def _parse_perf_l1_small_size(value):
    """Parse the isolated benchmark's per-core L1-small reservation in bytes."""
    if value is None:
        # The perf harness holds warmup and realtime-profiler programs concurrently.
        # Two KiB covers ten 8-device receiver invocations while remaining close to
        # the 1-KiB Sparse MLA product configuration.
        return 2048
    try:
        size = int(value)
    except ValueError as error:
        raise ValueError("TTNN_AG_PERF_L1_SMALL_SIZE must be an integer byte count") from error
    if size <= 0 or size % 16 != 0:
        raise ValueError("TTNN_AG_PERF_L1_SMALL_SIZE must be a positive multiple of 16 bytes")
    return size


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, None),
        ("auto", None),
        ("0,0,3,0", (0, 0, 3, 0)),
        ("2,4,2,7", (2, 4, 2, 7)),
    ],
)
def test_parse_perf_core_rect(value, expected):
    assert _parse_perf_core_rect(value) == expected


@pytest.mark.parametrize("value", ["", "0,0,1", "x,0,1,1", "-1,0,2,0", "3,0,2,0"])
def test_parse_perf_core_rect_rejects_invalid_values(value, expect_error):
    with expect_error(ValueError, "."):
        _parse_perf_core_rect(value)


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, ttnn.FabricConfig.FABRIC_2D),
        ("mesh", ttnn.FabricConfig.FABRIC_2D),
        ("ring", ttnn.FabricConfig.FABRIC_1D_RING),
        ("torus_y", ttnn.FabricConfig.FABRIC_2D_TORUS_Y),
    ],
)
def test_parse_perf_fabric_config(value, expected):
    assert _parse_perf_fabric_config(value) == expected


def test_parse_perf_fabric_config_rejects_invalid_value(expect_error):
    with expect_error(ValueError, "."):
        _parse_perf_fabric_config("torus_x")


@pytest.mark.parametrize("value,expected", [(None, (4, 2)), ("4x2", (4, 2)), ("8x1", (8, 1))])
def test_parse_perf_mesh_shape(value, expected):
    assert _parse_perf_mesh_shape(value) == expected


@pytest.mark.parametrize("value", ["", "8", "8x1x1", "axb", "0x8", "8x-1"])
def test_parse_perf_mesh_shape_rejects_invalid_value(value, expect_error):
    with expect_error(ValueError, "."):
        _parse_perf_mesh_shape(value)


@pytest.mark.parametrize("value,expected", [(None, 2048), ("512", 512), ("2048", 2048)])
def test_parse_perf_l1_small_size(value, expected):
    assert _parse_perf_l1_small_size(value) == expected


@pytest.mark.parametrize("value", ["", "0", "-16", "17", "not-a-size"])
def test_parse_perf_l1_small_size_rejects_invalid_value(value, expect_error):
    with expect_error(ValueError, "."):
        _parse_perf_l1_small_size(value)


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


def _all_gather_profile_ns(mesh_device, run_fn, expected_receiver_l1=None):
    """Return one all-gather's device critical path/runtime IDs and verify its implementation path."""
    _, records = profile_realtime_program(mesh_device, run_fn, collect_all=True, record_timeout_seconds=5.0)
    programs = {}
    receiver_l1_observed = False
    for record in records:
        normalized_sources = [source.replace("\\", "/") for source in record["kernel_sources"]]
        if not any("/ccl/all_gather/" in source for source in normalized_sources):
            continue
        receiver_l1_observed |= any(source.endswith("/multicast_receiver_writer.cpp") for source in normalized_sources)
        runtime_id = record["runtime_id"]
        programs[runtime_id] = max(programs.get(runtime_id, 0.0), record["duration_ns"])
    assert programs, "realtime profiler returned no native all-gather program"
    if expected_receiver_l1 is not None:
        assert (
            receiver_l1_observed == expected_receiver_l1
        ), f"expected receiver_l1={expected_receiver_l1}, observed receiver_l1={receiver_l1_observed}"
    return sum(programs.values()), frozenset(programs)


def _all_gather_duration_ns(mesh_device, run_fn, expected_receiver_l1=None):
    duration_ns, _ = _all_gather_profile_ns(mesh_device, run_fn, expected_receiver_l1)
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
        assert torch.equal(actual, expected), (
            f"device {device_index} gather mismatch: "
            f"max_abs={(actual.float() - expected.float()).abs().max().item()} "
            f"mismatched_elements={mismatch.sum().item()} rows={mismatch_rows}"
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


@pytest.mark.skipif(
    os.environ.get("TTNN_ALL_GATHER_BANK_OWNED_LINKS") != "1",
    reason="set TTNN_ALL_GATHER_BANK_OWNED_LINKS=1 to exercise the bounded bank-owned schedule",
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": _parse_perf_fabric_config(os.environ.get("TTNN_AG_BANK_OWNED_FABRIC_CONFIG")),
            "fabric_router_config": _fabric_router_config(),
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
            "l1_small_size": _parse_perf_l1_small_size(os.environ.get("TTNN_AG_BANK_OWNED_L1_SMALL_SIZE")),
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device", [_parse_perf_mesh_shape(os.environ.get("TTNN_AG_BANK_OWNED_MESH_SHAPE"))], indirect=True
)
@pytest.mark.parametrize(
    "dtype,width,pcc",
    [
        pytest.param(ttnn.bfloat16, 576, 1.0, id="bf16_1152b_rows"),
        pytest.param(ttnn.fp8_e4m3, 656, 0.99, id="scaled_fp8_704b_rows"),
    ],
)
def test_all_gather_fabric_2d_bank_owned_schedule(mesh_device, dtype, width, pcc):
    """Exercise the exact-divisor two-link/eight-bank schedule with persistent output."""
    global_shape = (1, 1, 128, width)
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
        assert any(source.endswith("/multicast_receiver_writer.cpp") for source in all_sources)
    else:
        tt_output = run_ag()
        ttnn.synchronize_device(mesh_device)

    check_output = ttnn.typecast(tt_output, ttnn.bfloat16) if dtype == ttnn.fp8_e4m3 else tt_output
    if dtype == ttnn.fp8_e4m3:
        # Compare against the values actually carried by the gather. Comparing
        # FP8 output directly with the pre-quantized random input confounds CCL
        # correctness with FP8 conversion error.
        check_input = ttnn.typecast(tt_input, ttnn.bfloat16)
        expected = torch.cat(
            [ttnn.to_torch(device_tensor) for device_tensor in ttnn.get_device_tensors(check_input)], dim=2
        )
    else:
        expected = torch_input
    for device_index, device_tensor in enumerate(ttnn.get_device_tensors(check_output)):
        actual = ttnn.to_torch(device_tensor)
        mismatch = actual != expected
        mismatch_rows = mismatch.nonzero()[:, 2].unique().tolist() if mismatch.any() else []
        assert torch.equal(actual, expected), (
            f"device {device_index} gather mismatch: "
            f"max_abs={(actual.float() - expected.float()).abs().max().item()} "
            f"mismatched_elements={mismatch.sum().item()} rows={mismatch_rows}"
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
    expected_receiver_override = os.environ.get("TTNN_EXPECT_RECEIVER_L1")
    if expected_receiver_override is not None:
        assert expected_receiver_override in {"0", "1"}
        expected_receiver_l1 = expected_receiver_override == "1"
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
def test_all_gather_fabric_2d_receiver_l1_small_capacity_policy(mesh_device, monkeypatch, expect_error):
    """Auto falls back before dispatch and forced receiver reports the failed resource proof."""
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

    monkeypatch.setenv("TTNN_ALL_GATHER_RECEIVER_L1_MODE", "auto")
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

    monkeypatch.setenv("TTNN_ALL_GATHER_RECEIVER_L1_MODE", "force_receiver")
    with expect_error(RuntimeError, "control semaphores exceed the configured L1-small bank"):
        run_ag()


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
def test_all_gather_fabric_2d_receiver_policy_program_identity(mesh_device, monkeypatch):
    """Changing a test policy must compile/select a distinct cached program."""
    if not ttnn.device.IsProgramRealtimeProfilerActive():
        pytest.skip("receiver policy identity requires realtime-profiler path assertions")

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

    def run_and_assert(mode, expected_receiver_l1):
        monkeypatch.setenv("TTNN_ALL_GATHER_RECEIVER_L1_MODE", mode)

        def run_ag():
            return ttnn.all_gather(
                tt_input,
                dim=2,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cluster_axis=0,
                output_tensor=persistent_output,
            )

        tt_output, records = profile_realtime_program(mesh_device, run_ag, collect_all=True, record_timeout_seconds=5.0)
        all_sources = [source.replace("\\", "/") for record in records for source in record["kernel_sources"]]
        receiver_l1_observed = any(source.endswith("/multicast_receiver_writer.cpp") for source in all_sources)
        assert receiver_l1_observed == expected_receiver_l1
        for device_tensor in ttnn.get_device_tensors(tt_output):
            assert_with_pcc(ttnn.to_torch(device_tensor), torch_input, pcc=1.0)

    run_and_assert("auto", True)
    run_and_assert("force_direct", False)
    run_and_assert("auto", True)


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


@pytest.mark.skipif(
    os.environ.get("TTNN_RUN_AG_ISOLATED_PERF") != "1",
    reason="set TTNN_RUN_AG_ISOLATED_PERF=1 to run the isolated native all-gather benchmark",
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": _parse_perf_fabric_config(os.environ.get("TTNN_AG_PERF_FABRIC_CONFIG")),
            "fabric_router_config": _fabric_router_config(),
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
            "l1_small_size": _parse_perf_l1_small_size(os.environ.get("TTNN_AG_PERF_L1_SMALL_SIZE")),
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device", [_parse_perf_mesh_shape(os.environ.get("TTNN_AG_PERF_MESH_SHAPE"))], indirect=True
)
@pytest.mark.parametrize(
    "dtype,width,expected_page_size",
    [
        pytest.param(ttnn.bfloat16, 576, 1152, id="bf16_1152b_rows"),
        pytest.param(ttnn.fp8_e4m3, 656, 704, id="scaled_fp8_704b_rows"),
    ],
)
def test_all_gather_fabric_2d_sparse_mla_row_perf(mesh_device, dtype, width, expected_page_size):
    """Steady-state native AG bandwidth for the sparse-MLA SP row geometry.

    The benchmark intentionally times only the compiled all-gather program: input/output creation,
    FP8 conversion, and correctness readback happen outside the realtime-profiler region. Override
    ``TTNN_AG_PERF_ROWS_PER_DEVICE`` to sweep transfer size without changing the CCL page geometry.
    """
    if not ttnn.device.IsProgramRealtimeProfilerActive():
        pytest.fail("native all-gather perf benchmark requires an active realtime device profiler")

    rows_per_device = int(os.environ.get("TTNN_AG_PERF_ROWS_PER_DEVICE", "16384"))
    rows_per_page = int(os.environ.get("TTNN_AG_PERF_ROWS_PER_PAGE", "1"))
    valid_gather_extent_env = os.environ.get("TTNN_AG_PERF_VALID_GATHER_EXTENT")
    valid_gather_extent = int(valid_gather_extent_env) if valid_gather_extent_env is not None else None
    samples = int(os.environ.get("TTNN_AG_PERF_SAMPLES", "5"))
    receiver_mode = os.environ.get("TTNN_ALL_GATHER_RECEIVER_L1_MODE", "auto")
    receiver_stage = os.environ.get("TTNN_ALL_GATHER_RECEIVER_STAGE_MODE", "combined")
    receiver_slots_env = os.environ.get("TTNN_ALL_GATHER_RECEIVER_SLOTS", "auto")
    receiver_batch_rows_env = os.environ.get("TTNN_ALL_GATHER_RECEIVER_BATCH_ROWS", "max")
    receiver_notify = os.environ.get("TTNN_ALL_GATHER_RECEIVER_NOTIFY_MODE", "fused")
    receiver_credit = os.environ.get("TTNN_ALL_GATHER_RECEIVER_CREDIT_MODE", "window")
    receiver_credit_group_batches = os.environ.get("TTNN_ALL_GATHER_RECEIVER_CREDIT_GROUP_BATCHES", "auto")
    receiver_attribution = os.environ.get("TTNN_ALL_GATHER_RECEIVER_ATTRIBUTION", "0")
    address_attribution = os.environ.get("TTNN_ALL_GATHER_ADDRESS_ATTRIBUTION", "0")
    receiver_drain_riscs_env = os.environ.get("TTNN_ALL_GATHER_RECEIVER_DRAIN_RISCS", "auto")
    bank_owned_links = os.environ.get("TTNN_ALL_GATHER_BANK_OWNED_LINKS", "0")
    bank_owned_coalesce = os.environ.get("TTNN_ALL_GATHER_BANK_OWNED_COALESCE", "none")
    bank_owned_run_policy = os.environ.get("TTNN_ALL_GATHER_BANK_OWNED_RUN_POLICY", "max_tail")
    perf_core_rect = _parse_perf_core_rect(os.environ.get("TTNN_AG_PERF_CORE_RECT", "auto"))
    perf_fabric_config = os.environ.get("TTNN_AG_PERF_FABRIC_CONFIG", "mesh")
    perf_l1_small_size = _parse_perf_l1_small_size(os.environ.get("TTNN_AG_PERF_L1_SMALL_SIZE"))
    assert receiver_mode in {"auto", "force_direct", "force_receiver"}
    assert receiver_stage in {"combined", "l1_sink", "l1_overwrite", "drain_only"}
    assert receiver_slots_env == "auto" or 1 <= int(receiver_slots_env) <= 256
    assert receiver_batch_rows_env in {"max", "1", "2", "4", "8"}
    assert receiver_notify in {"fused", "split"}
    assert receiver_credit in {"per_slot", "window", "pipelined"}
    assert receiver_credit_group_batches == "auto" or 1 <= int(receiver_credit_group_batches) <= 256
    assert receiver_attribution in {"0", "1"}
    assert address_attribution in {"0", "1"}
    assert receiver_drain_riscs_env in {"auto", "1", "2"}
    assert bank_owned_links in {"0", "1"}
    assert bank_owned_coalesce in {"none", "source", "source_receiver", "source_local", "all"}
    assert bank_owned_run_policy in {"divisor", "max_tail"}
    assert receiver_stage == "combined" or receiver_mode != "force_direct", "receiver stage mode requires receiver path"
    assert (
        receiver_slots_env in {"auto", "1"} or receiver_mode != "force_direct"
    ), "multiple receiver slots require receiver mode"
    expected_receiver_l1 = receiver_mode != "force_direct"
    expected_receiver_override = os.environ.get("TTNN_EXPECT_RECEIVER_L1")
    if expected_receiver_override is not None:
        assert expected_receiver_override in {"0", "1"}
        expected_receiver_l1 = expected_receiver_override == "1"
    assert rows_per_device > 0 and rows_per_device % 4 == 0, "rows per device must be a positive multiple of 4"
    assert rows_per_page > 0 and rows_per_device % rows_per_page == 0, "rows per page must divide rows per device"
    assert (
        valid_gather_extent is None or 0 < valid_gather_extent <= rows_per_device // rows_per_page
    ), "valid gather extent must select a non-empty leading range of the local height"
    assert samples > 0, "at least one timed sample is required"
    perf_sub_core_grid = None
    if perf_core_rect is not None:
        x0, y0, x1, y1 = perf_core_rect
        compute_grid = mesh_device.compute_with_storage_grid_size()
        assert (
            x1 < compute_grid.x and y1 < compute_grid.y
        ), f"core rectangle {perf_core_rect} exceeds logical grid {compute_grid.x}x{compute_grid.y}"
        assert (x1 - x0 + 1) * (
            y1 - y0 + 1
        ) == 4, "isolated receiver placement requires exactly four logical cores: two sender/receiver pairs"
        perf_sub_core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))})
    element_size = 1 if dtype == ttnn.fp8_e4m3 else 2
    transport_page_size = ((width * rows_per_page * element_size + 63) // 64) * 64
    if transport_page_size > 14 * 1024:
        pytest.skip(f"{transport_page_size}-byte page exceeds this benchmark's configured 14-KiB fabric payload")

    sp = mesh_device.shape[0]
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
    receiver_batch_rows = (
        max(1, (14 * 1024) // page_size) if receiver_batch_rows_env == "max" else int(receiver_batch_rows_env)
    )
    bank_owned_batch_rows = (
        _reference_bank_owned_rows_per_run(receiver_batch_rows, rows_per_device // 8, bank_owned_run_policy)
        if bank_owned_links == "1"
        else receiver_batch_rows
    )
    if receiver_drain_riscs_env == "auto":
        receiver_drain_riscs = 2 if dtype == ttnn.fp8_e4m3 else 1
    else:
        receiver_drain_riscs = int(receiver_drain_riscs_env)

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
    _all_gather_profile_ns(mesh_device, run_ag, expected_receiver_l1=expected_receiver_l1)
    profile_samples = [
        _all_gather_profile_ns(mesh_device, run_ag, expected_receiver_l1=expected_receiver_l1) for _ in range(samples)
    ]
    durations_ns = [duration_ns for duration_ns, _ in profile_samples]
    median_ns = statistics.median(durations_ns)
    min_ns = min(durations_ns)
    p90_ns = sorted(durations_ns)[max(0, (9 * len(durations_ns) + 9) // 10 - 1)]
    # Each SP rank receives data from the other SP-1 ranks. This is effective receive bandwidth,
    # kept deliberately independent of the fabric algorithm's internal forwarding traffic.
    selected_pages_per_device = valid_gather_extent or rows_per_device // rows_per_page
    bytes_received_per_rank = selected_pages_per_device * page_size * (sp - 1)
    bandwidth_gbps = bytes_received_per_rank / median_ns
    print(
        f"ISOLATED_AG path={'receiver_l1' if expected_receiver_l1 else 'direct'} stage={receiver_stage} "
        f"slots={receiver_slots_env} batch_rows={bank_owned_batch_rows} notify={receiver_notify} credit={receiver_credit} "
        f"credit_group_batches={receiver_credit_group_batches} "
        f"drain_riscs={receiver_drain_riscs} bank_owned_links={bank_owned_links} "
        f"bank_owned_coalesce={bank_owned_coalesce} bank_owned_run_policy={bank_owned_run_policy} "
        f"core_rect={perf_core_rect or 'auto'} fabric={perf_fabric_config} l1_small={perf_l1_small_size}B "
        f"dtype={dtype} rows_per_device={rows_per_device} rows_per_page={rows_per_page} "
        f"valid_gather_extent={valid_gather_extent or 'full'} "
        f"page_size={page_size}B "
        f"median={median_ns / 1e6:.3f}ms min={min_ns / 1e6:.3f}ms p90={p90_ns / 1e6:.3f}ms "
        f"effective_receive_bw={bandwidth_gbps:.3f}GB/s "
        f"samples_ms={[round(duration / 1e6, 3) for duration in durations_ns]}"
    )

    if receiver_attribution == "1" or address_attribution == "1":
        runtime_ids = set().union(*(ids for _, ids in profile_samples))
        print(
            f"AG_ATTRIBUTION_READY normalized_runtime_ids={sorted(runtime_ids)}; "
            f"after safe pytest exits run scripts/analyze_all_gather_attribution.py --samples {samples}"
        )
