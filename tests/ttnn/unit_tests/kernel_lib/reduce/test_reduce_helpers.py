# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Planner-driven device tests for the reduce compute and dataflow helpers."""

from dataclasses import dataclass

import pytest
import torch

import ttnn

pytestmark = pytest.mark.use_module_device

TILE = 32
CB_INPUT = 0
CB_SCALER = 1
CB_ACCUMULATOR = 2
CB_OUTPUT = 16

PLAN_SEQUENCE_KERNEL = "tests/ttnn/unit_tests/kernel_lib/reduce/kernels/reduce_plan_sequence.cpp"
PLAN_SEQUENCE_AUX_KERNEL = "tests/ttnn/unit_tests/kernel_lib/reduce/kernels/reduce_plan_sequence_aux.cpp"

DIMS = ("REDUCE_ROW", "REDUCE_COL", "REDUCE_SCALAR")
INPUT_MODES = ("bulk", "chunked", "alias")

_PLANNER = ttnn.reduce_planner
_REDUCE_MATH = {
    "SUM": _PLANNER.ReduceMath.SUM,
    "AVG": _PLANNER.ReduceMath.AVG,
    "MAX": _PLANNER.ReduceMath.MAX,
    "MIN": _PLANNER.ReduceMath.MIN,
}
_REDUCE_DIM = {
    "REDUCE_ROW": _PLANNER.ReduceDimension.ROW,
    "REDUCE_COL": _PLANNER.ReduceDimension.COLUMN,
    "REDUCE_SCALAR": _PLANNER.ReduceDimension.SCALAR,
}
_FP32_MODE = {
    "Fast": _PLANNER.ReduceFp32Mode.FAST,
    "Accurate": _PLANNER.ReduceFp32Mode.ACCURATE,
}
_INPUT_POLICY = {
    "bulk": _PLANNER.ReduceInputPolicy.BULK_WAIT_BULK_POP,
    "chunked": _PLANNER.ReduceInputPolicy.CHUNKED_WAIT_CHUNKED_POP,
    "alias": _PLANNER.ReduceInputPolicy.NO_WAIT_NO_POP,
}
_ALGORITHM = {
    "REDUCE_TILE": _PLANNER.ReduceAlgorithm.REDUCE_TILE,
    "ACCUMULATE_VIA_ADD": _PLANNER.ReduceAlgorithm.ACCUMULATE_VIA_ADD,
}


@dataclass(frozen=True)
class ReduceCase:
    name: str
    family: str
    dim: str
    rows: int
    cols: int
    batches: int = 1
    pool: str = "SUM"
    input_mode: str = "bulk"
    calls: int = 1
    input_dtype: str = "bf16"
    output_dtype: str = "bf16"
    fp32_mode: str = "Fast"
    partial_elements: int = 0
    scalar: float = 1.0

    @property
    def logical_height(self) -> int:
        if self.dim == "REDUCE_COL" and self.partial_elements:
            return (self.rows - 1) * TILE + self.partial_elements
        return self.rows * TILE

    @property
    def logical_width(self) -> int:
        if self.dim == "REDUCE_ROW" and self.partial_elements:
            return (self.cols - 1) * TILE + self.partial_elements
        return self.cols * TILE

    @property
    def output_tiles(self) -> int:
        if self.dim == "REDUCE_ROW":
            return self.rows * self.batches
        if self.dim == "REDUCE_COL":
            return self.cols * self.batches
        return self.batches

    @property
    def reduced_elements(self) -> int:
        if self.dim == "REDUCE_ROW":
            return self.logical_width
        if self.dim == "REDUCE_COL":
            return self.logical_height
        return self.logical_height * self.logical_width

    @property
    def planner_scalar(self) -> float:
        return self.scalar / self.reduced_elements if self.pool == "AVG" else self.scalar

    @property
    def expected_algorithm(self) -> str:
        reduced_tiles = self.cols if self.dim == "REDUCE_ROW" else self.rows
        if self.dim == "REDUCE_SCALAR":
            reduced_tiles = self.rows * self.cols
        additive = (
            self.pool in ("SUM", "AVG")
            and self.input_dtype in ("bf16", "fp32")
            and self.fp32_mode != "Accurate"
            and reduced_tiles >= (4 if self.dim == "REDUCE_ROW" else 8)
        )
        return "ACCUMULATE_VIA_ADD" if additive else "REDUCE_TILE"


def _shape_for_dim(dim: str) -> tuple[int, int, int]:
    if dim == "REDUCE_ROW":
        return 3, 5, 2
    if dim == "REDUCE_COL":
        return 9, 5, 2
    return 2, 4, 2


def _input_space_cases() -> list[ReduceCase]:
    cases = []
    for dim in DIMS:
        rows, cols, batches = _shape_for_dim(dim)
        modes = ("bulk", "chunked", "alias") if dim != "REDUCE_SCALAR" else ("bulk", "chunked")
        for input_mode in modes:
            for accumulated in (False, True):
                cases.append(
                    ReduceCase(
                        name=f"input-{dim}-{input_mode}-acc{int(accumulated)}",
                        family="input-space",
                        dim=dim,
                        rows=rows,
                        cols=cols,
                        batches=batches,
                        input_mode=input_mode,
                        calls=2 if accumulated else 1,
                        output_dtype="fp32",
                    )
                )
    return cases


def _regression_cases() -> list[ReduceCase]:
    return [
        # Regression for https://github.com/tenstorrent/tt-metal/issues/54177.
        ReduceCase(
            name="regression-row-9x32-four-calls",
            family="regression",
            dim="REDUCE_ROW",
            rows=9,
            cols=8,
            pool="SUM",
            input_mode="alias",
            calls=4,
            input_dtype="bf16",
            output_dtype="fp32",
        )
    ]


def _numerical_space_cases() -> list[ReduceCase]:
    cases = []
    for dtype in ("bf16", "fp32"):
        for pool in ("SUM", "AVG", "MAX"):
            for dim in DIMS:
                rows, cols, _ = _shape_for_dim(dim)
                cases.append(
                    ReduceCase(
                        name=f"numeric-{dtype}-{pool}-{dim}-fast",
                        family="numerical-space",
                        dim=dim,
                        rows=rows,
                        cols=cols,
                        pool=pool,
                        input_dtype=dtype,
                        output_dtype=dtype,
                    )
                )

    for pool in ("SUM", "MAX"):
        for dim in ("REDUCE_ROW", "REDUCE_COL"):
            rows, cols, _ = _shape_for_dim(dim)
            cases.append(
                ReduceCase(
                    name=f"numeric-fp32-{pool}-{dim}-accurate",
                    family="numerical-space",
                    dim=dim,
                    rows=rows,
                    cols=cols,
                    pool=pool,
                    input_dtype="fp32",
                    output_dtype="fp32",
                    fp32_mode="Accurate",
                )
            )

    for pool in ("SUM", "MAX", "MIN"):
        for dim in ("REDUCE_ROW", "REDUCE_COL"):
            rows, cols, _ = _shape_for_dim(dim)
            cases.append(
                ReduceCase(
                    name=f"numeric-int32-{pool}-{dim}",
                    family="numerical-space",
                    dim=dim,
                    rows=rows,
                    cols=cols,
                    pool=pool,
                    input_dtype="int32",
                    output_dtype="int32",
                )
            )
    return cases


def _boundary_cases() -> list[ReduceCase]:
    cases = [
        ReduceCase(
            name=f"singleton-{dim}",
            family="shape-boundaries",
            dim=dim,
            rows=1,
            cols=1,
            output_dtype="fp32",
        )
        for dim in DIMS
    ]

    for dim in DIMS:
        rows, cols, _ = _shape_for_dim(dim)
        cases.append(
            ReduceCase(
                name=f"scalar-add-path-{dim}",
                family="scalar",
                dim=dim,
                rows=rows,
                cols=cols,
                output_dtype="fp32",
                scalar=0.5,
            )
        )
        cases.append(
            ReduceCase(
                name=f"scalar-reduce-tile-path-{dim}",
                family="scalar",
                dim=dim,
                rows=1,
                cols=1,
                output_dtype="fp32",
                scalar=0.5,
            )
        )

    for dim in ("REDUCE_ROW", "REDUCE_COL"):
        for partial_elements in (1, 15, 16, 17, 31):
            cases.append(
                ReduceCase(
                    name=f"partial-scaler-bf16-{dim}-valid{partial_elements}",
                    family="partial",
                    dim=dim,
                    rows=1,
                    cols=1,
                    output_dtype="fp32",
                    partial_elements=partial_elements,
                )
            )
        for partial_elements in (15, 17):
            cases.append(
                ReduceCase(
                    name=f"partial-scaler-fp32-{dim}-valid{partial_elements}",
                    family="partial",
                    dim=dim,
                    rows=1,
                    cols=1,
                    input_dtype="fp32",
                    output_dtype="fp32",
                    partial_elements=partial_elements,
                )
            )
        rows, cols = (1, 5) if dim == "REDUCE_ROW" else (9, 2)
        for pool in ("SUM", "AVG"):
            cases.append(
                ReduceCase(
                    name=f"partial-mask-{pool}-{dim}-acc2",
                    family="partial",
                    dim=dim,
                    rows=rows,
                    cols=cols,
                    pool=pool,
                    calls=2,
                    output_dtype="fp32",
                    partial_elements=7,
                )
            )
    return cases


ALL_CASES = tuple(_input_space_cases() + _regression_cases() + _numerical_space_cases() + _boundary_cases())


def _assert_complete_case_matrix() -> None:
    """Keep additions/removals from silently punching holes in the advertised space."""
    input_cases = [case for case in ALL_CASES if case.family == "input-space"]
    actual = {(case.dim, case.input_mode, case.calls > 1) for case in input_cases}
    expected = {
        (dim, input_mode, accumulated)
        for dim in DIMS
        for input_mode in (("bulk", "chunked", "alias") if dim != "REDUCE_SCALAR" else ("bulk", "chunked"))
        for accumulated in (False, True)
    }
    assert actual == expected

    numerical_cases = [case for case in ALL_CASES if case.family == "numerical-space"]
    actual_numerical = {(case.input_dtype, case.pool, case.dim, case.fp32_mode) for case in numerical_cases}
    expected_numerical = {
        (dtype, pool, dim, "Fast") for dtype in ("bf16", "fp32") for pool in ("SUM", "AVG", "MAX") for dim in DIMS
    }
    expected_numerical |= {
        ("fp32", pool, dim, "Accurate") for pool in ("SUM", "MAX") for dim in ("REDUCE_ROW", "REDUCE_COL")
    }
    expected_numerical |= {
        ("int32", pool, dim, "Fast") for pool in ("SUM", "MAX", "MIN") for dim in ("REDUCE_ROW", "REDUCE_COL")
    }
    assert actual_numerical == expected_numerical


_assert_complete_case_matrix()


def _single_core() -> ttnn.CoreRangeSet:
    core = ttnn.CoreCoord(0, 0)
    return ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])


def _sharded_memory_config(
    shape: tuple[int, int], strategy: ttnn.ShardStrategy = ttnn.ShardStrategy.HEIGHT
) -> ttnn.MemoryConfig:
    return ttnn.create_sharded_memory_config(
        shape=shape,
        core_grid=_single_core(),
        strategy=strategy,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _scratch_cb(cb_id: int, dtype: ttnn.DataType, num_tiles: int) -> ttnn.CBDescriptor:
    page_size = ttnn.tile_size(dtype)
    return ttnn.CBDescriptor(
        total_size=page_size * num_tiles,
        core_ranges=_single_core(),
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=cb_id, data_format=dtype, page_size=page_size)],
    )


def _ttnn_dtype(name: str) -> ttnn.DataType:
    return {"bf16": ttnn.bfloat16, "fp32": ttnn.float32, "int32": ttnn.int32}[name]


def _scaler_dtype(case: ReduceCase) -> ttnn.DataType:
    return ttnn.float32 if case.input_dtype == "fp32" else ttnn.bfloat16


def _input_cb_ids(call_count: int) -> list[int]:
    available = [cb_id for cb_id in range(32) if cb_id not in (CB_SCALER, CB_ACCUMULATOR, CB_OUTPUT)]
    assert call_count <= len(available)
    return available[:call_count]


def _memory_strategy(case: ReduceCase) -> ttnn.ShardStrategy:
    return ttnn.ShardStrategy.WIDTH if case.dim == "REDUCE_COL" else ttnn.ShardStrategy.HEIGHT


def _tensor_spec(
    logical_shape: tuple[int, ...], dtype: ttnn.DataType, memory_config: ttnn.MemoryConfig
) -> ttnn.TensorSpec:
    return ttnn.TensorSpec(
        ttnn.Shape(logical_shape),
        dtype,
        ttnn.TILE_LAYOUT,
        memory_config.memory_layout,
        memory_config.shard_spec,
        memory_config.buffer_type,
    )


def _logical_output_shape(case: ReduceCase) -> tuple[int, int, int]:
    if case.dim == "REDUCE_ROW":
        return case.batches, case.rows * TILE, TILE
    if case.dim == "REDUCE_COL":
        return case.batches, TILE, case.cols * TILE
    return case.batches, TILE, TILE


def _max_input_cb_bytes(case: ReduceCase, input_dtype: ttnn.DataType) -> int | None:
    assert case.input_mode in INPUT_MODES
    if case.input_mode == "bulk":
        return None
    if case.input_mode == "alias":
        return 0
    assert case.input_mode == "chunked"
    # Keep two buffers of two reduction-axis tiles. H reductions retain four
    # output columns in DEST, so each input buffer needs eight tiles.
    cap_tiles = 16 if case.dim == "REDUCE_COL" else 4
    return cap_tiles * ttnn.tile_size(input_dtype)


def _serialize_plan(plan) -> tuple[list[int], list[int]]:
    compute_compile_time_args = [17]
    plan.append_to(compute_compile_time_args)
    assert compute_compile_time_args[0] == 17
    assert compute_compile_time_args[1] == plan.call_count
    assert compute_compile_time_args[1:] == plan.compile_time_args

    auxiliary_compile_time_args = [23]
    plan.auxiliary.append_to(auxiliary_compile_time_args)
    assert auxiliary_compile_time_args[0] == 23
    assert auxiliary_compile_time_args[1:] == plan.auxiliary_compile_time_args
    return compute_compile_time_args, auxiliary_compile_time_args


def _make_plan(
    device,
    case: ReduceCase,
    input_dtype: ttnn.DataType,
    output_dtype: ttnn.DataType,
    input_memory_config: ttnn.MemoryConfig,
    output_memory_config: ttnn.MemoryConfig,
    input_cb_ids: list[int],
):
    input_spec = _tensor_spec((case.batches, case.logical_height, case.logical_width), input_dtype, input_memory_config)
    output_spec = _tensor_spec(_logical_output_shape(case), output_dtype, output_memory_config)
    reductions = [
        (
            cb_id,
            _PLANNER.ReduceCallConfig(
                input_spec=input_spec,
                output_spec=output_spec,
                reduce_math=_REDUCE_MATH[case.pool],
                reduce_dim=_REDUCE_DIM[case.dim],
                scalar=case.planner_scalar,
                fp32_mode=_FP32_MODE[case.fp32_mode],
                max_input_cb_bytes=_max_input_cb_bytes(case, input_dtype),
            ),
        )
        for cb_id in input_cb_ids
    ]
    plan = _PLANNER.make_reduce_sequence_plan(
        reductions=reductions,
        cb_ids=_PLANNER.ReduceSequenceCbIds(
            auxiliary_cb_id=CB_SCALER,
            accumulator_cb_id=CB_ACCUMULATOR,
            output_cb_id=CB_OUTPUT,
        ),
        hardware=_PLANNER.ReduceHardwareConfig(
            arch=device.arch(),
            fp32_dest_acc_en=True,
            dst_full_sync_en=False,
            available_l1_bytes=ttnn.get_max_worker_l1_unreserved_size(),
        ),
    )
    _assert_plan(case, plan, input_cb_ids)
    return plan


def _assert_plan(case: ReduceCase, plan, input_cb_ids: list[int]) -> None:
    assert len(plan) == plan.call_count == len(plan.calls) == case.calls
    assert plan.auxiliary.cb_id == CB_SCALER
    assert len(plan.auxiliary.tiles) > 0

    expected_partial = _PLANNER.ReducePartialMode.NONE
    if case.partial_elements and case.pool in ("SUM", "AVG"):
        expected_partial = (
            _PLANNER.ReducePartialMode.MASK
            if case.expected_algorithm == "ACCUMULATE_VIA_ADD"
            else _PLANNER.ReducePartialMode.SCALER
        )

    for index, (call, input_cb_id) in enumerate(zip(plan.calls, input_cb_ids)):
        assert call.input_cb_id == input_cb_id
        assert call.auxiliary_cb_id == CB_SCALER
        assert call.plan.input_policy == _INPUT_POLICY[case.input_mode]
        assert call.plan.algorithm == _ALGORITHM[case.expected_algorithm]
        assert call.plan.partial_mode == expected_partial
        assert call.plan.Ht == case.rows
        assert call.plan.Wt == case.cols
        assert call.plan.batches == case.batches
        assert call.auxiliary_tile_offset + len(call.plan.auxiliary_tiles) <= len(plan.auxiliary.tiles)

        if case.calls == 1:
            assert call.accumulation_mode == _PLANNER.ReduceAccumulationMode.NONE
            assert call.accumulator_cb_id is None
            assert call.output_cb_id == CB_OUTPUT
        else:
            expected_mode = (
                _PLANNER.ReduceAccumulationMode.FINAL
                if index + 1 == case.calls
                else _PLANNER.ReduceAccumulationMode.INTERMEDIATE
            )
            assert call.accumulation_mode == expected_mode
            assert call.accumulation_index == index
            assert call.accumulator_cb_id == CB_ACCUMULATOR
            assert call.output_cb_id == (CB_OUTPUT if index + 1 == case.calls else CB_ACCUMULATOR)


def _repeated_input_cb_plan_compile_args(input_tensor, output) -> tuple[list[int], list[int]]:
    """Build two compute calls sharing one input CB and one aggregate auxiliary payload."""
    planner = ttnn.reduce_planner
    hardware = planner.ReduceHardwareConfig(
        arch=input_tensor.device().arch(),
        fp32_dest_acc_en=False,
        dst_full_sync_en=False,
        available_l1_bytes=ttnn.get_max_worker_l1_unreserved_size(),
    )
    configs = [
        (
            CB_INPUT,
            planner.ReduceCallConfig(
                input_spec=input_tensor.spec,
                output_spec=output.spec,
                reduce_math=planner.ReduceMath.SUM,
                reduce_dim=planner.ReduceDimension.ROW,
                scalar=1.0,
                fp32_mode=planner.ReduceFp32Mode.FAST,
                max_input_cb_bytes=0,
            ),
        )
        for _ in range(2)
    ]
    plan = planner.make_reduce_sequence_plan(
        reductions=configs,
        cb_ids=planner.ReduceSequenceCbIds(
            auxiliary_cb_id=CB_SCALER,
            accumulator_cb_id=CB_ACCUMULATOR,
            output_cb_id=CB_OUTPUT,
        ),
        hardware=hardware,
    )
    assert len(plan) == plan.call_count == len(plan.calls) == 2
    assert plan.calls[0].input_cb_id == plan.calls[1].input_cb_id == CB_INPUT
    assert plan.calls[0].plan.algorithm == planner.ReduceAlgorithm.REDUCE_TILE
    assert plan.calls[0].accumulation_mode == planner.ReduceAccumulationMode.INTERMEDIATE
    assert plan.calls[1].accumulation_mode == planner.ReduceAccumulationMode.FINAL
    assert plan.calls[0].auxiliary_tile_offset == plan.calls[1].auxiliary_tile_offset == 0
    assert plan.auxiliary.cb_id == CB_SCALER
    assert len(plan.auxiliary.tiles) == 1

    return _serialize_plan(plan)


def _compute_config(case: ReduceCase, input_cb_ids: list[int]) -> ttnn.ComputeConfigDescriptor:
    config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi3,
        fp32_dest_acc_en=True,
        dst_full_sync_en=False,
    )
    if case.input_dtype == "fp32" and case.fp32_mode == "Accurate":
        # Host descriptors use the maximum CB count so this vector covers both Wormhole (32) and Blackhole (64).
        unpack_modes = [ttnn.UnpackToDestMode.Default] * 64
        for input_cb_id in input_cb_ids:
            unpack_modes[input_cb_id] = ttnn.UnpackToDestMode.UnpackToDestFp32
        if case.calls > 1:
            unpack_modes[CB_ACCUMULATOR] = ttnn.UnpackToDestMode.UnpackToDestFp32
        config.unpack_to_dest_mode = unpack_modes
    return config


def _make_logical_chunks(case: ReduceCase) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(20260824)
    shape = (case.batches, case.rows * TILE, case.cols * TILE)
    chunks = []
    for call in range(case.calls):
        if case.input_dtype == "int32":
            chunk = torch.randint(-16, 17, shape, generator=generator, dtype=torch.int32) + call
        else:
            chunk = torch.rand(shape, generator=generator, dtype=torch.float32) + 0.125 * call
            if case.input_dtype == "bf16":
                chunk = chunk.to(torch.bfloat16)
        chunks.append(chunk)
    return chunks


def _logical_tiles(case: ReduceCase, logical: torch.Tensor) -> torch.Tensor:
    return logical.reshape(case.batches, case.rows, TILE, case.cols, TILE).permute(0, 1, 3, 2, 4).contiguous()


def _physical_input(case: ReduceCase, logical: torch.Tensor, call) -> torch.Tensor:
    logical_tiles = _logical_tiles(case, logical)
    physical_tiles = torch.empty_like(logical_tiles)

    streams_by_col_chunk = case.dim == "REDUCE_COL" and call.plan.input_policy in (
        _PLANNER.ReduceInputPolicy.BULK_WAIT_BULK_POP,
        _PLANNER.ReduceInputPolicy.CHUNKED_WAIT_CHUNKED_POP,
    )
    if streams_by_col_chunk:
        ordered_tiles = []
        for batch in range(case.batches):
            for col_start in range(0, case.cols, call.plan.chunk.output_tiles):
                col_end = min(col_start + call.plan.chunk.output_tiles, case.cols)
                for row in range(case.rows):
                    for col in range(col_start, col_end):
                        ordered_tiles.append(logical_tiles[batch, row, col])
        physical_tiles.view(-1, TILE, TILE).copy_(torch.stack(ordered_tiles))
    else:
        physical_tiles.copy_(logical_tiles)

    return physical_tiles.permute(0, 1, 3, 2, 4).contiguous().reshape(case.batches * case.rows * TILE, case.cols * TILE)


def _reduce_logical_chunk(case: ReduceCase, chunk: torch.Tensor) -> torch.Tensor:
    values = chunk[:, : case.logical_height, : case.logical_width]
    values = values.to(torch.float64) if case.input_dtype != "int32" else values.to(torch.int64)
    if case.dim == "REDUCE_ROW":
        reduce_axis = -1
    elif case.dim == "REDUCE_COL":
        reduce_axis = -2
    else:
        reduce_axis = (-2, -1)

    if case.pool in ("SUM", "AVG"):
        reduced = values.sum(dim=reduce_axis)
    elif case.pool == "MAX":
        reduced = values.amax(dim=reduce_axis)
    else:
        reduced = values.amin(dim=reduce_axis)
    return reduced


def _golden(case: ReduceCase, chunks: list[torch.Tensor]) -> torch.Tensor:
    partials = torch.stack([_reduce_logical_chunk(case, chunk) for chunk in chunks])
    if case.pool in ("SUM", "AVG"):
        golden = partials.sum(dim=0)
        if case.pool == "AVG":
            golden = golden / (case.calls * case.reduced_elements)
        if case.scalar != 1.0:
            golden = golden * case.scalar
    elif case.pool == "MAX":
        golden = partials.amax(dim=0)
    else:
        golden = partials.amin(dim=0)
    return golden


def _output_shape(case: ReduceCase) -> tuple[int, int]:
    if case.dim == "REDUCE_ROW":
        return case.batches * case.rows * TILE, TILE
    if case.dim == "REDUCE_COL":
        return case.batches * TILE, case.cols * TILE
    return case.batches * TILE, TILE


def _meaningful_output(case: ReduceCase, output: torch.Tensor) -> torch.Tensor:
    if case.dim == "REDUCE_ROW":
        return output.reshape(case.batches, case.rows * TILE, TILE)[:, :, 0]
    if case.dim == "REDUCE_COL":
        return output.reshape(case.batches, TILE, case.cols * TILE)[:, 0, :]
    return output.reshape(case.batches, TILE, TILE)[:, 0, 0]


def _run_case(device, case: ReduceCase) -> tuple[torch.Tensor, torch.Tensor]:
    input_dtype = _ttnn_dtype(case.input_dtype)
    output_dtype = _ttnn_dtype(case.output_dtype)
    input_shape = (case.batches * case.rows * TILE, case.cols * TILE)
    output_shape = _output_shape(case)
    strategy = _memory_strategy(case)
    input_memory_config = _sharded_memory_config(input_shape, strategy)
    output_memory_config = _sharded_memory_config(output_shape, strategy)
    input_cb_ids = _input_cb_ids(case.calls)
    plan = _make_plan(
        device,
        case,
        input_dtype,
        output_dtype,
        input_memory_config,
        output_memory_config,
        input_cb_ids,
    )
    compute_compile_time_args, auxiliary_compile_time_args = _serialize_plan(plan)

    logical_chunks = _make_logical_chunks(case)
    device_inputs = [
        ttnn.from_torch(
            _physical_input(case, logical, call),
            dtype=input_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=input_memory_config,
        )
        for logical, call in zip(logical_chunks, plan.calls)
    ]

    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        output_dtype,
        ttnn.TILE_LAYOUT,
        device,
        output_memory_config,
    )

    cbs = [ttnn.cb_descriptor_from_sharded_tensor(cb_id, tensor) for cb_id, tensor in zip(input_cb_ids, device_inputs)]
    cbs.extend(
        [
            ttnn.cb_descriptor_from_sharded_tensor(CB_OUTPUT, output),
            _scratch_cb(CB_SCALER, _scaler_dtype(case), len(plan.auxiliary.tiles)),
        ]
    )
    if case.calls > 1:
        cbs.append(_scratch_cb(CB_ACCUMULATOR, output_dtype, case.output_tiles))

    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=PLAN_SEQUENCE_AUX_KERNEL,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=_single_core(),
            compile_time_args=auxiliary_compile_time_args,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=PLAN_SEQUENCE_KERNEL,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=_single_core(),
            compile_time_args=compute_compile_time_args,
            config=_compute_config(case, input_cb_ids),
        ),
    ]

    result = ttnn.generic_op(
        [*device_inputs, output],
        ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs),
    )
    actual = _meaningful_output(case, ttnn.to_torch(result))
    return actual, _golden(case, logical_chunks)


def test_reduce_plan_sequence_repeated_input_cb(device):
    """Two independently scheduled calls reduce the same reusable input CB into one accumulator."""
    input_shape = (TILE, 3 * TILE)
    output_shape = (TILE, TILE)
    memory_config = _sharded_memory_config(input_shape)
    input_tensor = ttnn.from_torch(
        torch.ones(input_shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
    )
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        _sharded_memory_config(output_shape),
    )

    compute_compile_time_args, auxiliary_compile_time_args = _repeated_input_cb_plan_compile_args(input_tensor, output)
    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(CB_INPUT, input_tensor),
        ttnn.cb_descriptor_from_sharded_tensor(CB_OUTPUT, output),
        _scratch_cb(CB_SCALER, ttnn.bfloat16, 1),
        _scratch_cb(CB_ACCUMULATOR, ttnn.bfloat16, 1),
    ]
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=PLAN_SEQUENCE_AUX_KERNEL,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=_single_core(),
            compile_time_args=auxiliary_compile_time_args,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=PLAN_SEQUENCE_KERNEL,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=_single_core(),
            compile_time_args=compute_compile_time_args,
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi3,
                fp32_dest_acc_en=False,
                dst_full_sync_en=False,
            ),
        ),
    ]

    result = ttnn.generic_op(
        [input_tensor, output],
        ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs),
    )
    actual = ttnn.to_torch(result)[:, 0].to(torch.float32)
    torch.testing.assert_close(actual, torch.full_like(actual, 2.0 * input_shape[1]), rtol=0, atol=0)


@pytest.mark.parametrize("case", ALL_CASES, ids=lambda case: case.name)
def test_reduce_helpers_complete_input_space(device, case: ReduceCase):
    """Exercise every valid helper branch and its numerical/layout boundaries."""
    if "QUASAR" in str(device.arch()).upper() and (case.input_dtype == "int32" or case.fp32_mode == "Accurate"):
        pytest.skip("The reduce helper rejects SFPU reduce paths on Quasar")

    actual, expected = _run_case(device, case)
    if case.input_dtype == "int32":
        torch.testing.assert_close(actual.to(torch.int64), expected, rtol=0, atol=0)
    else:
        rtol = 0.05 if case.calls > 1 or case.input_dtype == "bf16" else 0.02
        torch.testing.assert_close(
            actual.to(torch.float64),
            expected.to(torch.float64),
            rtol=rtol,
            atol=0.1,
            msg=case.name,
        )
