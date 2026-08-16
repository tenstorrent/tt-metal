# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Focused real-time-profiler benchmarks for the multicast-helper migration.

These are comparative benchmarks, not absolute performance gates.  Run the
same case on the pre-migration and post-migration snapshots and compare the
device-program durations written to MCAST_RT_OUTPUT_DIR.
"""

import json
import os
import statistics
from pathlib import Path

import pytest
import torch

import ttnn
from tests.ttnn.perf_tests.operations.conv.test_conv2d_device_perf import CONV_PERF_CONFIGS
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program

WARMUP_ITERATIONS = int(os.environ.get("MCAST_RT_WARMUP_ITERATIONS", "3"))
MEASURED_ITERATIONS = int(os.environ.get("MCAST_RT_MEASURED_ITERATIONS", "20"))


def _require_realtime_profiler():
    if not ttnn.device.IsProgramRealtimeProfilerActive():
        pytest.fail("The real-time profiler is not active on this device/dispatch configuration")


def _write_results(case_name, records):
    valid_records = [record for record in records if record["runtime_id"] != 0]
    assert valid_records, f"No valid real-time-profiler records were collected for {case_name}"

    durations_ns = [float(record["duration_ns"]) for record in valid_records]
    ordered_durations_ns = sorted(durations_ns)
    p95_index = min(len(ordered_durations_ns) - 1, int(0.95 * len(ordered_durations_ns)))
    payload = {
        "case": case_name,
        "git_commit": os.environ.get("MCAST_RT_GIT_COMMIT", "unknown"),
        "warmup_iterations": WARMUP_ITERATIONS,
        "measured_iterations": MEASURED_ITERATIONS,
        "record_count": len(valid_records),
        "median_ns": statistics.median(durations_ns),
        "mean_ns": statistics.fmean(durations_ns),
        "min_ns": min(durations_ns),
        "max_ns": max(durations_ns),
        "p95_ns": ordered_durations_ns[p95_index],
        "pstdev_ns": statistics.pstdev(durations_ns),
        "records": valid_records,
    }

    output_dir = Path(os.environ.get("MCAST_RT_OUTPUT_DIR", "generated/mcast_migration_rt"))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{payload['git_commit']}_{case_name}.json"
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"MCAST_RT_RESULT={json.dumps({key: value for key, value in payload.items() if key != 'records'})}")
    print(f"MCAST_RT_OUTPUT={output_path}")


def _run_conv2d(device, config):
    tt_input_tensor = ttnn.empty(
        (1, 1, config["input_height"] * config["input_width"] * config["batch_size"], config["input_channels"]),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    tt_weight_tensor = ttnn.from_torch(
        torch.randn(
            (
                config["output_channels"],
                config["input_channels"] // config["groups"],
                config["kernel_h"],
                config["kernel_w"],
            ),
            dtype=torch.bfloat16,
        ),
        config["weights_dtype"] if config["weights_dtype"] != ttnn.bfloat8_b else ttnn.float32,
    )
    tt_bias_tensor = ttnn.from_torch(
        torch.randn((1, 1, 1, config["output_channels"]), dtype=torch.bfloat16),
        config["weights_dtype"] if config["weights_dtype"] != ttnn.bfloat8_b else ttnn.float32,
    )
    conv_config = ttnn.Conv2dConfig(
        weights_dtype=config["weights_dtype"],
        shard_layout=config["shard_layout"],
        deallocate_activation=False,
        transpose_shards=False,
        reshard_if_not_optimal=False,
        override_sharding_config=False,
        enable_act_double_buffer=config["enable_act_double_buffer"],
        enable_weights_double_buffer=config["enable_weights_double_buffer"],
        act_block_h_override=config["act_block_h_override"],
        act_block_w_div=config["act_block_w_div"],
        enable_activation_reuse=config["enable_activation_reuse"],
        config_tensors_in_dram=True,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=config["math_fidelity"],
        math_approx_mode=True,
        fp32_dest_acc_en=config["fp32_accum"],
        packer_l1_acc=False,
    )

    def run_once():
        output = ttnn.conv2d(
            input_tensor=tt_input_tensor,
            weight_tensor=tt_weight_tensor,
            device=device,
            in_channels=config["input_channels"],
            out_channels=config["output_channels"],
            bias_tensor=tt_bias_tensor,
            kernel_size=(config["kernel_h"], config["kernel_w"]),
            stride=(config["stride_h"], config["stride_w"]),
            padding=(config["pad_h"], config["pad_w"]),
            batch_size=config["batch_size"],
            input_height=config["input_height"],
            input_width=config["input_width"],
            conv_config=conv_config,
            compute_config=compute_config,
            groups=config["groups"],
            dtype=config["output_dtype"],
            slice_config=ttnn.Conv2dL1FullSliceConfig,
        )
        ttnn.deallocate(output)

    for _ in range(WARMUP_ITERATIONS):
        run_once()
    ttnn.synchronize_device(device)

    def run_measured():
        for _ in range(MEASURED_ITERATIONS):
            run_once()

    _, records = profile_realtime_program(device, run_measured, collect_all=True)
    conv_records = [
        record for record in records if any("conv" in source.lower() for source in record["kernel_sources"])
    ]
    assert len(conv_records) == MEASURED_ITERATIONS, (
        f"Expected {MEASURED_ITERATIONS} Conv2D records, found {len(conv_records)} "
        f"out of {len(records)} total records"
    )
    _write_results(config["test_name"], conv_records)


@pytest.mark.parametrize(
    "config",
    [pytest.param(config, id=config["test_name"]) for config in CONV_PERF_CONFIGS],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
def test_conv2d_mcast_realtime_perf(device, config):
    torch.manual_seed(0)
    _require_realtime_profiler()
    _run_conv2d(device, config)


def _profile_repeated_program(device, case_name, run_once, kernel_source_fragment, required_kernel_source_fragments=()):
    for _ in range(WARMUP_ITERATIONS):
        run_once()
    ttnn.synchronize_device(device)

    def run_measured():
        for _ in range(MEASURED_ITERATIONS):
            run_once()

    _, records = profile_realtime_program(device, run_measured, collect_all=True)
    matching_records = [
        record
        for record in records
        if any(kernel_source_fragment in source.lower() for source in record["kernel_sources"])
    ]
    assert len(matching_records) == MEASURED_ITERATIONS, (
        f"Expected {MEASURED_ITERATIONS} {case_name} records, found {len(matching_records)} "
        f"out of {len(records)} total records"
    )
    kernel_sources = {source for record in matching_records for source in record["kernel_sources"]}
    for fragment in required_kernel_source_fragments:
        assert any(fragment in source.lower() for source in kernel_sources), (
            f"{case_name} did not compile the required kernel fragment {fragment!r}; "
            f"observed sources: {sorted(kernel_sources)}"
        )
    _write_results(case_name, matching_records)


def _run_sdxl_matmul_2d(device):
    m, k, n = 1024, 1280, 5120
    core_grid = ttnn.CoreGrid(y=8, x=5)
    tt_act = ttnn.from_torch(
        torch.randn((1, 1, m, k), dtype=torch.bfloat16),
        layout=ttnn.TILE_LAYOUT,
        device=device,
        dtype=ttnn.bfloat16,
    )
    tt_weights = ttnn.from_torch(
        torch.randn((1, 1, k, n), dtype=torch.bfloat16),
        layout=ttnn.TILE_LAYOUT,
        device=device,
        dtype=ttnn.bfloat8_b,
    )
    tt_bias = ttnn.from_torch(
        torch.randn((1, 1, 1, n), dtype=torch.bfloat16),
        layout=ttnn.TILE_LAYOUT,
        device=device,
        dtype=ttnn.bfloat8_b,
    )
    tt_act = ttnn.to_memory_config(
        tt_act,
        ttnn.create_sharded_memory_config(
            (1, 1, m, k),
            core_grid=core_grid,
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(core_grid.x, core_grid.y),
        in0_block_w=4,
        out_subblock_h=1,
        out_subblock_w=8,
        per_core_M=4,
        per_core_N=32,
        transpose_mcast=False,
        fuse_batch=True,
        fused_activation=[ttnn.UnaryOpType.GELU, True],
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    def run_once():
        output = ttnn.linear(
            tt_act,
            tt_weights,
            bias=tt_bias,
            program_config=program_config,
            memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
            compute_kernel_config=compute_config,
        )
        ttnn.deallocate(output)

    _profile_repeated_program(device, "matmul_2d_sdxl_ff_gelu", run_once, "matmul")


def _run_sdxl_matmul_1d(device):
    # SDXL 1024x1024 up_blocks.2.resnets.0.conv_shortcut: [1, 960, 128, 128] -> 320 channels.
    m, k, n = 128 * 128, 960, 320
    tt_act = ttnn.from_torch(
        torch.randn((1, 1, m, k), dtype=torch.bfloat16),
        layout=ttnn.TILE_LAYOUT,
        device=device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    tt_weights = ttnn.from_torch(
        torch.randn((1, 1, k, n), dtype=torch.bfloat16),
        layout=ttnn.TILE_LAYOUT,
        device=device,
        dtype=ttnn.bfloat8_b,
    )
    tt_bias = ttnn.from_torch(
        torch.randn((1, 1, 1, n), dtype=torch.bfloat16),
        layout=ttnn.TILE_LAYOUT,
        device=device,
        dtype=ttnn.bfloat8_b,
    )
    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(11, 10),
        in0_block_w=1,
        out_subblock_h=5,
        out_subblock_w=1,
        per_core_M=5,
        per_core_N=10,
        mcast_in0=False,
        gather_in0=False,
        fuse_batch=False,
        fused_activation=None,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    def run_once():
        output = ttnn.linear(
            tt_act,
            tt_weights,
            bias=tt_bias,
            program_config=program_config,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=compute_config,
        )
        ttnn.deallocate(output)

    _profile_repeated_program(device, "matmul_1d_sdxl_resnet_960_320", run_once, "matmul")


def _run_matmul_2d_transposed(device):
    m, k, n = 512, 512, 1024
    core_grid = ttnn.CoreGrid(y=8, x=8)
    tt_act = ttnn.from_torch(
        torch.randn((1, 1, m, k), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.create_sharded_memory_config(
            (1, 1, m, k),
            core_grid=core_grid,
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.COL_MAJOR,
        ),
    )
    tt_weights = ttnn.from_torch(
        torch.randn((1, 1, k, n), dtype=torch.bfloat16),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=2,
        out_subblock_h=1,
        out_subblock_w=4,
        per_core_M=2,
        per_core_N=4,
        transpose_mcast=True,
        fuse_batch=True,
        fused_activation=None,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    def run_once():
        output = ttnn.matmul(
            tt_act,
            tt_weights,
            program_config=program_config,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            compute_kernel_config=compute_config,
        )
        ttnn.deallocate(output)

    _profile_repeated_program(device, "matmul_2d_transpose_mcast", run_once, "matmul")


def _run_matmul_degenerate_1x1(device):
    m = k = n = 64
    tt_act = ttnn.from_torch(
        torch.randn((1, 1, m, k), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    tt_weights = ttnn.from_torch(
        torch.randn((1, 1, k, n), dtype=torch.bfloat16),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(1, 1),
        in0_block_w=1,
        out_subblock_h=2,
        out_subblock_w=2,
        per_core_M=2,
        per_core_N=2,
        mcast_in0=True,
        gather_in0=False,
        fuse_batch=True,
        fused_activation=None,
    )

    def run_once():
        output = ttnn.matmul(
            tt_act,
            tt_weights,
            program_config=program_config,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(output)

    _profile_repeated_program(
        device,
        "matmul_1d_degenerate_1x1",
        run_once,
        "matmul",
        required_kernel_source_fragments=("reader_bmm_tile_layout_in0_sender_padding.cpp",),
    )


def _run_matmul_2d_sender_span_exceeds_receiver_span(device):
    batch_size, m, k, n = 2, 128, 256, 512
    shape = (batch_size, 1, m, k)
    tt_act = ttnn.from_torch(
        torch.randn(shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    tt_weights = ttnn.from_torch(
        torch.randn((k, n), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    tt_act = ttnn.to_memory_config(
        tt_act,
        ttnn.create_sharded_memory_config(
            shape,
            core_grid=ttnn.CoreGrid(y=2, x=2),
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )

    def run_once():
        output = ttnn.matmul(tt_act, tt_weights, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(output)

    _profile_repeated_program(
        device,
        "matmul_2d_sender_span_exceeds_receiver_span",
        run_once,
        "matmul",
        required_kernel_source_fragments=("reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp",),
    )


@pytest.mark.parametrize(
    "case_name,run_case",
    [
        pytest.param("matmul_2d_sdxl_ff_gelu", _run_sdxl_matmul_2d, id="matmul_2d_sdxl_ff_gelu"),
        pytest.param("matmul_1d_sdxl_resnet_960_320", _run_sdxl_matmul_1d, id="matmul_1d_sdxl_resnet_960_320"),
        pytest.param("matmul_2d_transpose_mcast", _run_matmul_2d_transposed, id="matmul_2d_transpose_mcast"),
        pytest.param("matmul_1d_degenerate_1x1", _run_matmul_degenerate_1x1, id="matmul_1d_degenerate_1x1"),
        pytest.param(
            "matmul_2d_sender_span_exceeds_receiver_span",
            _run_matmul_2d_sender_span_exceeds_receiver_span,
            id="matmul_2d_sender_span_exceeds_receiver_span",
        ),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
def test_matmul_mcast_realtime_perf(device, case_name, run_case):
    del case_name
    torch.manual_seed(0)
    _require_realtime_profiler()
    run_case(device)


def _run_groupnorm(device, use_welford):
    shape = (1, 1920, 32, 32)
    num_groups = 32
    core_grid = ttnn.CoreGrid(y=8, x=8)
    torch_input = torch.rand(shape, dtype=torch.bfloat16)
    flattened_input = torch_input.permute(0, 2, 3, 1).reshape(1, 1, shape[2] * shape[3], shape[1])
    tt_input = ttnn.from_torch(
        flattened_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.create_sharded_memory_config(
            shape=flattened_input.shape,
            core_grid=core_grid,
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        ),
        device=device,
    )
    input_mask = ttnn.to_device(
        ttnn.create_group_norm_input_mask(shape[1], num_groups, core_grid.x, ttnn.DataType.BFLOAT8_B),
        device,
    )

    def run_once():
        ttnn.group_norm(
            tt_input,
            num_groups=num_groups,
            input_mask=input_mask,
            memory_config=tt_input.memory_config(),
            core_grid=core_grid,
            inplace=True,
            use_welford=use_welford,
        )

    algorithm = "welford" if use_welford else "legacy"
    _profile_repeated_program(device, f"groupnorm_sdxl_1920_{algorithm}", run_once, "groupnorm")


def _run_groupnorm_interleaved(device, use_welford):
    shape = (1, 768, 1, 512)
    num_groups = 32
    core_grid = ttnn.CoreGrid(y=8, x=8)
    torch_input = torch.rand(shape, dtype=torch.float32)
    tt_input = ttnn.from_torch(
        torch_input.permute(0, 2, 3, 1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        device=device,
    )
    [weight, bias], input_mask = ttnn.dram_group_norm_params_from_torch(
        [torch.rand(shape[1]), torch.rand(shape[1])],
        shape[1],
        num_groups,
        device,
        core_grid=core_grid,
        return_mask=True,
        dtype=ttnn.bfloat16,
    )

    def run_once():
        output = ttnn.group_norm(
            tt_input,
            num_groups=num_groups,
            input_mask=input_mask,
            weight=weight,
            bias=bias,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            core_grid=core_grid,
            use_welford=use_welford,
            num_out_blocks=2,
            inplace=False,
        )
        ttnn.deallocate(output)

    algorithm = "welford" if use_welford else "legacy"
    kernel_prefix = "welford_" if use_welford else ""
    _profile_repeated_program(
        device,
        f"groupnorm_interleaved_768_{algorithm}",
        run_once,
        f"{kernel_prefix}reader_mcast_sender_unary_gn.cpp",
        (f"{kernel_prefix}reader_mcast_receiver_unary_gn.cpp",),
    )


@pytest.mark.parametrize("use_welford", [False, True], ids=["groupnorm_legacy", "groupnorm_welford"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
def test_groupnorm_mcast_realtime_perf(device, use_welford):
    torch.manual_seed(0)
    _require_realtime_profiler()
    _run_groupnorm(device, use_welford)


@pytest.mark.parametrize("use_welford", [False, True], ids=["groupnorm_legacy", "groupnorm_welford"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
def test_groupnorm_interleaved_mcast_realtime_perf(device, use_welford):
    torch.manual_seed(0)
    _require_realtime_profiler()
    _run_groupnorm_interleaved(device, use_welford)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
def test_sort_mcast_realtime_perf(device):
    torch.manual_seed(0)
    _require_realtime_profiler()
    tt_input = ttnn.from_torch(
        torch.randn((1, 524288), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    def run_once():
        values, indices = ttnn.sort(tt_input, dim=-1, descending=False)
        ttnn.deallocate(values)
        ttnn.deallocate(indices)

    _profile_repeated_program(device, "sort_single_row_524288", run_once, "sort")
