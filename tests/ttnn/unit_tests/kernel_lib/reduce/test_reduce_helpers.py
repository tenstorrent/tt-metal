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
    fp32_dest_acc_en: bool = True
    partial_elements: int = 0
    max_identity_only: bool = False
    scalar: float = 1.0
    allow_empty_auxiliary: bool = False

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
            and self.input_dtype in ("bf16", "fp32", "bf8", "bf4")
            and self.fp32_mode != "Accurate"
            and reduced_tiles * self.calls >= (4 if self.dim == "REDUCE_ROW" else 8)
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


def _partial_max_cases() -> list[ReduceCase]:
    cases = []
    for dim in ("REDUCE_ROW", "REDUCE_COL"):
        for dtype in ("bf16", "fp32"):
            cases.append(
                ReduceCase(
                    name=f"partial-max-{dtype}-dest16-{dim}",
                    family="partial-max",
                    dim=dim,
                    rows=3,
                    cols=9,
                    pool="MAX",
                    input_dtype=dtype,
                    output_dtype=dtype,
                    calls=2,
                    fp32_dest_acc_en=False,
                    partial_elements=17,
                )
            )
            for scalar in (0.5, 2.0):
                cases.append(
                    ReduceCase(
                        name=f"partial-max-{dtype}-{dim}-scale{scalar}",
                        family="partial-max",
                        dim=dim,
                        rows=1,
                        cols=1,
                        pool="MAX",
                        input_dtype=dtype,
                        output_dtype=dtype,
                        partial_elements=17,
                        scalar=scalar,
                    )
                )
            cases.append(
                ReduceCase(
                    name=f"partial-max-{dtype}-all-identity-{dim}",
                    family="partial-max",
                    dim=dim,
                    rows=3,
                    cols=3,
                    pool="MAX",
                    input_dtype=dtype,
                    output_dtype=dtype,
                    calls=2,
                    partial_elements=17,
                    max_identity_only=True,
                )
            )
        for valid in (1, 15, 16, 17, 31):
            cases.append(
                ReduceCase(
                    name=f"partial-max-bf16-{dim}-valid{valid}",
                    family="partial-max",
                    dim=dim,
                    rows=1,
                    cols=1,
                    pool="MAX",
                    partial_elements=valid,
                )
            )
        for dtype in ("bf16", "fp32"):
            for input_mode in INPUT_MODES:
                cases.append(
                    ReduceCase(
                        name=f"partial-max-{dtype}-Fast-{dim}-{input_mode}-acc2",
                        family="partial-max",
                        dim=dim,
                        rows=3 if dim == "REDUCE_ROW" else 9,
                        cols=5,
                        batches=2,
                        pool="MAX",
                        input_dtype=dtype,
                        output_dtype=dtype,
                        input_mode=input_mode,
                        calls=2,
                        partial_elements=17,
                    )
                )
    return cases


ALL_CASES = tuple(
    _input_space_cases() + _regression_cases() + _numerical_space_cases() + _boundary_cases() + _partial_max_cases()
)


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


def _scratch_cb(
    cb_id: int, dtype: ttnn.DataType, num_tiles: int, *, core_ranges: ttnn.CoreRangeSet | None = None
) -> ttnn.CBDescriptor:
    page_size = ttnn.tile_size(dtype)
    return ttnn.CBDescriptor(
        total_size=page_size * num_tiles,
        core_ranges=_single_core() if core_ranges is None else core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=cb_id, data_format=dtype, page_size=page_size)],
    )


def _ttnn_dtype(name: str) -> ttnn.DataType:
    return {
        "bf16": ttnn.bfloat16,
        "fp32": ttnn.float32,
        "int32": ttnn.int32,
        "bf8": ttnn.bfloat8_b,
        "bf4": ttnn.bfloat4_b,
    }[name]


def _scaler_dtype(case: ReduceCase) -> ttnn.DataType:
    return ttnn.float32 if case.input_dtype == "fp32" else ttnn.bfloat16


def _input_cb_ids(call_count: int) -> list[int]:
    available = [cb_id for cb_id in range(32) if cb_id not in (CB_SCALER, CB_ACCUMULATOR, CB_OUTPUT)]
    assert call_count <= len(available)
    return available[:call_count]


def _memory_strategy(case: ReduceCase) -> ttnn.ShardStrategy:
    return ttnn.ShardStrategy.WIDTH if case.dim == "REDUCE_COL" else ttnn.ShardStrategy.HEIGHT


def _max_input_cb_bytes(case: ReduceCase, input_dtype: ttnn.DataType) -> int | None:
    assert case.input_mode in INPUT_MODES
    if case.input_mode == "bulk":
        return None
    if case.input_mode == "alias":
        return None
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
    input_cb_ids: list[int],
):
    block = _PLANNER.ReduceBlockSpec(
        case.logical_height,
        case.logical_width,
        input_dtype,
        output_dtype,
        batches=case.batches,
        padded_h=case.rows * TILE,
        padded_w=case.cols * TILE,
        resident_input_tiles=case.rows * case.cols * case.batches if case.input_mode == "alias" else None,
        resident_output_tiles=case.output_tiles if case.dim != "REDUCE_SCALAR" else None,
        allow_empty_auxiliary=case.allow_empty_auxiliary,
    )
    reductions = [
        (
            cb_id,
            _PLANNER.ReduceCallConfig(
                block=block,
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
            fp32_dest_acc_en=case.fp32_dest_acc_en,
            dst_full_sync_en=False,
            available_l1_bytes=ttnn.get_max_worker_l1_unreserved_size(),
        ),
    )
    _assert_plan(case, plan, input_cb_ids)
    return plan


def _assert_plan(case: ReduceCase, plan, input_cb_ids: list[int]) -> None:
    assert len(plan) == plan.call_count == len(plan.calls) == case.calls
    assert plan.auxiliary.cb_id == (CB_SCALER if plan.auxiliary.tiles else _PLANNER.NO_CB_ID)
    assert plan.auxiliary.tiles or case.allow_empty_auxiliary
    if case.family == "empty-auxiliary":
        assert not plan.auxiliary.tiles
        assert all(
            requirement.role != _PLANNER.ReduceCbRole.AUXILIARY
            for call in plan.calls
            for requirement in call.plan.cb_requirements
        )
    elif case.family == "empty-then-zero":
        assert not plan.calls[0].plan.auxiliary_tiles
        assert len(plan.calls[1].plan.auxiliary_tiles) == 1
        assert plan.calls[1].plan.reload_mode == _PLANNER.AccumulateReloadMode.COPY_SEED_ZERO_PAIR
        assert plan.auxiliary.tiles[0].type == _PLANNER.ReduceAuxiliaryTileType.ZERO
    if case.family == "mixed-auxiliary-format" and case.calls > 1:
        assert any(call.plan.reload_mode == _PLANNER.AccumulateReloadMode.COPY_SEED_ZERO_PAIR for call in plan.calls)
        assert any(tile.type == _PLANNER.ReduceAuxiliaryTileType.ZERO for tile in plan.auxiliary.tiles)

    expected_partial = _PLANNER.ReducePartialMode.NONE
    if case.partial_elements and case.pool in ("SUM", "AVG"):
        expected_partial = (
            _PLANNER.ReducePartialMode.MASK
            if case.expected_algorithm == "ACCUMULATE_VIA_ADD"
            else _PLANNER.ReducePartialMode.SCALER
        )
    elif case.partial_elements and case.pool == "MAX":
        expected_partial = _PLANNER.ReducePartialMode.SCALER

    for index, (call, input_cb_id) in enumerate(zip(plan.calls, input_cb_ids)):
        assert call.input_cb_id == input_cb_id
        assert call.auxiliary_cb_id == (CB_SCALER if call.plan.auxiliary_tiles else _PLANNER.NO_CB_ID)
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


def _repeated_input_cb_plan(input_tensor, output):
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
                block=planner.ReduceBlockSpec(
                    input_tensor.shape[-2],
                    input_tensor.shape[-1],
                    input_tensor.dtype,
                    output.dtype,
                    resident_input_tiles=input_tensor.buffer_num_pages(),
                    resident_output_tiles=output.buffer_num_pages(),
                ),
                reduce_math=planner.ReduceMath.SUM,
                reduce_dim=planner.ReduceDimension.ROW,
                scalar=1.0,
                fp32_mode=planner.ReduceFp32Mode.FAST,
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
    # The two three-tile calls jointly clear the sequence's four-tile threshold.
    assert plan.calls[0].plan.algorithm == planner.ReduceAlgorithm.ACCUMULATE_VIA_ADD
    assert plan.calls[0].accumulation_mode == planner.ReduceAccumulationMode.INTERMEDIATE
    assert plan.calls[1].accumulation_mode == planner.ReduceAccumulationMode.FINAL
    assert plan.calls[0].auxiliary_tile_offset == 0
    # The odd second call reloads the accumulated tile with a zero-pair recipe.
    assert plan.calls[1].auxiliary_tile_offset == 1
    assert plan.auxiliary.cb_id == CB_SCALER
    assert len(plan.auxiliary.tiles) == 2
    assert plan.auxiliary.tiles[1].type == planner.ReduceAuxiliaryTileType.ZERO

    return plan


def _compute_config(case: ReduceCase, input_cb_ids: list[int]) -> ttnn.ComputeConfigDescriptor:
    config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi3,
        fp32_dest_acc_en=case.fp32_dest_acc_en,
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
        elif case.input_dtype in ("bf8", "bf4"):
            # Exactly representable in both compressed formats, including the
            # nonzero padding which a partial reduction must exclude.
            chunk = torch.randint(1, 4, shape, generator=generator).float()
        elif case.family == "empty-auxiliary":
            # Small integer-valued floats keep these folds exact, so removing
            # a CB is checked independently of Fast-mode rounding differences.
            chunk = torch.randint(-16, 17, shape, generator=generator).float()
            if case.input_dtype == "bf16":
                chunk = chunk.to(torch.bfloat16)
        else:
            chunk = torch.rand(shape, generator=generator, dtype=torch.float32) + 0.125 * call
            if case.input_dtype == "bf16":
                chunk = chunk.to(torch.bfloat16)
        if case.partial_elements and case.pool == "MAX":
            # Zero is not an identity for MAX. Keep every valid value negative
            # and poison the padding so neither zero masking nor no masking passes.
            # Exactly representable values make MAX comparisons exact even in fast mode.
            chunk = -torch.randint(2, 128, shape, generator=generator).to(chunk.dtype)
            if case.max_identity_only:
                chunk.fill_(-torch.inf)
            padding = 99.0
            if not case.max_identity_only:
                # Make the final valid lane win, with a distinct maximum per output.
                # Alternate which accumulated call wins to exercise both fold operands.
                output_elements = case.logical_height if case.dim == "REDUCE_ROW" else case.logical_width
                index = torch.arange(output_elements)
                delta = call * (2 * (index % 2) - 1)
                edge = (-(32 + 2 * (index % 64) + delta) / 256).to(chunk.dtype)
                if case.dim == "REDUCE_ROW":
                    chunk[:, :, case.logical_width - 1] = edge
                else:
                    chunk[:, case.logical_height - 1, :] = edge
            chunk[:, case.logical_height :, :] = padding
            chunk[:, :, case.logical_width :] = padding
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
        if case.scalar != 1.0:
            golden = golden * case.scalar
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


def _run_case(device, case: ReduceCase, *, keep_empty_auxiliary_cb=False) -> tuple[torch.Tensor, torch.Tensor]:
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
    cbs.append(ttnn.cb_descriptor_from_sharded_tensor(CB_OUTPUT, output))
    if plan.auxiliary.tiles or keep_empty_auxiliary_cb:
        cbs.append(_scratch_cb(CB_SCALER, _scaler_dtype(case), max(1, len(plan.auxiliary.tiles))))
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
    physical_output = ttnn.to_torch(result)
    # Fused consumers can reduce these statistics again, so every datum
    # outside the reduced row/column/scalar must be exactly zero.
    padding = physical_output.clone()
    _meaningful_output(case, padding).zero_()
    assert torch.count_nonzero(padding).item() == 0, f"{case.name}: nonzero reduction output padding"
    actual = _meaningful_output(case, physical_output)
    return actual, _golden(case, logical_chunks)


@pytest.mark.parametrize("dim", ("REDUCE_ROW", "REDUCE_COL"))
@pytest.mark.parametrize("pool", ("SUM", "MAX"))
def test_reduce_local_blocks_on_multiple_cores(device, dim, pool):
    """Three cores share allocation sizes but reduce different local extents, including partial tiles."""
    core_count, batches = 3, 2
    allocation_rows, row_stride = 20, 8
    output_capacity = 16
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(core_count - 1, 0))])
    local_shapes = ((96, 256), (64, 135), (32, 17)) if dim == "REDUCE_ROW" else ((288, 160), (135, 96), (17, 32))
    physical = torch.full((core_count, allocation_rows * TILE, row_stride * TILE), 128, dtype=torch.bfloat16)
    expected = []
    kernels, plans = [], []
    resident_bytes = allocation_rows * row_stride * ttnn.tile_size(ttnn.bfloat16) + output_capacity * ttnn.tile_size(
        ttnn.float32
    )
    hardware = _PLANNER.ReduceHardwareConfig(
        arch=device.arch(),
        fp32_dest_acc_en=True,
        dst_full_sync_en=False,
        available_l1_bytes=ttnn.get_max_worker_l1_unreserved_size() - resident_bytes,
    )
    scalar = 1.0 / TILE if pool == "SUM" else 1.0
    for core_index, (height, width) in enumerate(local_shapes):
        block = _PLANNER.ReduceBlockSpec(
            height,
            width,
            ttnn.bfloat16,
            ttnn.float32,
            batches=batches,
            input_row_stride_tiles=row_stride,
            resident_input_tiles=allocation_rows * row_stride,
            resident_output_tiles=output_capacity,
        )
        # Values differ between cores and batches; all unvisited tiles and edge
        # padding retain a large sentinel which must never enter the reduction.
        core_expected = []
        for batch in range(batches):
            values = (torch.arange(height * width).reshape(height, width) % 7 - 3 + core_index + batch).to(
                torch.bfloat16
            )
            row_start = batch * block.padded_h
            physical[core_index, row_start : row_start + height, :width] = values
            axis = -1 if dim == "REDUCE_ROW" else -2
            core_expected.append((values.float().sum(axis) * scalar) if pool == "SUM" else values.float().amax(axis))
        expected.append(torch.stack(core_expected))
        sequence = _PLANNER.make_reduce_sequence_plan(
            reductions=[
                (
                    CB_INPUT,
                    _PLANNER.ReduceCallConfig(
                        block,
                        _REDUCE_MATH[pool],
                        _REDUCE_DIM[dim],
                        scalar,
                        _PLANNER.ReduceFp32Mode.FAST,
                    ),
                )
            ],
            cb_ids=_PLANNER.ReduceSequenceCbIds(CB_SCALER, CB_ACCUMULATOR, CB_OUTPUT),
            hardware=hardware,
        )
        assert sequence.calls[0].plan.batches == batches
        assert sequence.calls[0].plan.input_row_stride_tiles == row_stride
        plans.append(sequence)
        compute_args, auxiliary_args = _serialize_plan(sequence)
        core = ttnn.CoreCoord(core_index, 0)
        core_range = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
        kernels.extend(
            [
                ttnn.KernelDescriptor(
                    kernel_source=PLAN_SEQUENCE_AUX_KERNEL,
                    core_ranges=core_range,
                    compile_time_args=auxiliary_args,
                    config=ttnn.ReaderConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=PLAN_SEQUENCE_KERNEL,
                    core_ranges=core_range,
                    compile_time_args=compute_args,
                    config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True),
                ),
            ]
        )

    # Reserve one common capacity even when the per-core recipes use fewer pages.
    auxiliary_tiles = max(len(plan.auxiliary.tiles) for plan in plans)
    assert auxiliary_tiles * ttnn.tile_size(ttnn.bfloat16) <= hardware.available_l1_bytes

    def memory(shard_shape):
        return ttnn.create_sharded_memory_config(
            shape=shard_shape,
            core_grid=grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    input_tensor = ttnn.from_torch(
        physical.reshape(core_count * allocation_rows * TILE, row_stride * TILE),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory((allocation_rows * TILE, row_stride * TILE)),
    )
    output = ttnn.from_torch(
        torch.full((core_count * output_capacity * TILE, TILE), -999.0),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory((output_capacity * TILE, TILE)),
    )
    result = ttnn.generic_op(
        [input_tensor, output],
        ttnn.ProgramDescriptor(
            kernels=kernels,
            semaphores=[],
            cbs=[
                ttnn.cb_descriptor_from_sharded_tensor(CB_INPUT, input_tensor),
                ttnn.cb_descriptor_from_sharded_tensor(CB_OUTPUT, output),
                _scratch_cb(CB_SCALER, ttnn.bfloat16, auxiliary_tiles, core_ranges=grid),
            ],
        ),
    )
    output_tiles = ttnn.to_torch(result).reshape(core_count, output_capacity, TILE, TILE)
    for index, golden in enumerate(expected):
        count = golden.numel() // TILE
        tiles = output_tiles[index, :count]
        actual = tiles[:, :, 0] if dim == "REDUCE_ROW" else tiles[:, 0, :]
        torch.testing.assert_close(actual.reshape_as(golden), golden, rtol=1e-3, atol=1e-3)
        assert torch.all(output_tiles[index, count:] == -999), f"core {index}: wrote beyond local output"


@pytest.mark.parametrize("dim", ("REDUCE_ROW", "REDUCE_COL"))
@pytest.mark.parametrize(
    "pool,algorithm,calls",
    (
        ("SUM", "REDUCE_TILE", 1),
        ("MAX", "REDUCE_TILE", 1),
        ("AVG", "REDUCE_TILE", 1),
        ("SUM", "ACCUMULATE_VIA_ADD", 1),
        ("AVG", "ACCUMULATE_VIA_ADD", 1),
        ("SUM", "REDUCE_TILE", 2),
        ("MAX", "REDUCE_TILE", 2),
        ("SUM", "ACCUMULATE_VIA_ADD", 2),
    ),
)
@pytest.mark.parametrize("fp32_dest", (False, True))
def test_reduce_runtime_tail_cores(device, dim, pool, algorithm, calls, fp32_dest):
    """Common CB capacities across one static core and two tails; only tails receive shape RTAs."""
    core_count, max_batches = 3, 2
    max_h, max_w, row_stride = 288, 256, 10
    allocation_rows = max_batches * (max_h // TILE)
    output_capacity = max_batches * (max_h // TILE if dim == "REDUCE_ROW" else max_w // TILE)
    output_dtype = ttnn.float32 if fp32_dest else ttnn.bfloat16
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0))])
    groups = ((0,), (1, 2))
    input_bytes = allocation_rows * row_stride * ttnn.tile_size(ttnn.bfloat16)
    output_bytes = output_capacity * ttnn.tile_size(output_dtype)
    accumulator_bytes = output_bytes if calls > 1 else 0
    hardware = _PLANNER.ReduceHardwareConfig(
        arch=device.arch(),
        fp32_dest_acc_en=fp32_dest,
        dst_full_sync_en=False,
        available_l1_bytes=ttnn.get_max_worker_l1_unreserved_size() - input_bytes - output_bytes - accumulator_bytes,
    )
    reduced_extent = max_w if dim == "REDUCE_ROW" else max_h
    scalar = 1.0 / reduced_extent if pool == "AVG" else (1.0 / TILE if pool == "SUM" else 1.0)
    plans = []
    for indices in groups:
        tail = _PLANNER.ReduceTailConfig(2, 4) if indices[0] else None
        block = _PLANNER.ReduceBlockSpec(
            max_h,
            max_w,
            ttnn.bfloat16,
            output_dtype,
            batches=max_batches,
            input_row_stride_tiles=row_stride,
            resident_input_tiles=allocation_rows * row_stride,
            resident_output_tiles=output_capacity,
            tail=tail,
        )
        plans.append(
            _PLANNER.make_reduce_sequence_plan(
                reductions=[
                    (
                        CB_INPUT,
                        _PLANNER.ReduceCallConfig(
                            block, _REDUCE_MATH[pool], _REDUCE_DIM[dim], scalar, _PLANNER.ReduceFp32Mode.FAST
                        ),
                    )
                    for _ in range(calls)
                ],
                cb_ids=_PLANNER.ReduceSequenceCbIds(CB_SCALER, CB_ACCUMULATOR, CB_OUTPUT),
                hardware=hardware,
                algorithm=(
                    _PLANNER.ReduceAlgorithm.REDUCE_TILE
                    if algorithm == "REDUCE_TILE"
                    else _PLANNER.ReduceAlgorithm.ACCUMULATE_VIA_ADD
                ),
            )
        )
    assert plans[0].calls[0].plan.tail is None
    assert plans[1].calls[0].plan.tail is not None
    assert len(plans[1].calls[0].plan.auxiliary_tiles) > len(plans[0].calls[0].plan.auxiliary_tiles)
    # A tail changes the recipe used, not the physical allocation on that core.
    # Budget the largest aggregate recipe on every core, including full cores.
    auxiliary_tiles = max(len(plan.auxiliary.tiles) for plan in plans)
    assert auxiliary_tiles * ttnn.tile_size(ttnn.bfloat16) <= hardware.available_l1_bytes

    def memory(shard_shape):
        return ttnn.create_sharded_memory_config(
            shape=shard_shape,
            core_grid=grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    # Reuse identical compile-time plans for partial, aligned and sub-face tails.
    tail_shapes = (((65, 135, 2), (17, 33, 1)), ((64, 128, 1), (32, 64, 2)), ((31, 15, 2), (47, 17, 2)))
    if dim == "REDUCE_COL":
        tail_shapes = tuple(tuple((w, h, b) for h, w, b in shapes) for shapes in tail_shapes)
    for shape_pair in tail_shapes:
        shapes = ((max_h, max_w, max_batches), *shape_pair)
        physical = torch.full((core_count, allocation_rows * TILE, row_stride * TILE), 128, dtype=torch.bfloat16)
        expected = []
        for index, (height, width, batches) in enumerate(shapes):
            core_expected = []
            for batch in range(batches):
                values = ((torch.arange(height * width).reshape(height, width) + 2 * index + batch) % 7 - 3).to(
                    torch.bfloat16
                )
                start = batch * max_h  # Physical batch spacing must survive a smaller runtime height.
                physical[index, start : start + height, :width] = values
                if dim == "REDUCE_ROW":
                    physical[index, start + height : start + ((height + 31) // TILE) * TILE, :width] = float("nan")
                else:
                    physical[index, start : start + height, width : ((width + 31) // TILE) * TILE] = float("nan")
                axis = -1 if dim == "REDUCE_ROW" else -2
                golden = values.float().amax(axis) if pool == "MAX" else values.float().sum(axis)
                if pool == "AVG":
                    golden /= width if dim == "REDUCE_ROW" else height
                elif pool == "SUM":
                    golden *= scalar * calls
                core_expected.append(golden)
            expected.append(torch.stack(core_expected))

        kernels = []
        for indices, sequence in zip(groups, plans):
            cores = [ttnn.CoreCoord(i, 0) for i in indices]
            core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(cores[0], cores[-1])])
            compute_args, auxiliary_args = _serialize_plan(sequence)
            compute_runtime, auxiliary_runtime = [], []
            if sequence.calls[0].plan.tail is not None:
                for index, core in zip(indices, cores):
                    shape_args = sequence.calls[0].plan.get_runtime_shape_args(
                        _PLANNER.ReduceValidShape(*shapes[index])
                    )
                    compute_runtime.append((core, [111, 222] + shape_args))
                    auxiliary_runtime.append((core, [111, 222, 333, 444] + shape_args))
            kernels.extend(
                [
                    ttnn.KernelDescriptor(
                        kernel_source=PLAN_SEQUENCE_AUX_KERNEL,
                        core_ranges=core_ranges,
                        compile_time_args=auxiliary_args,
                        runtime_args=auxiliary_runtime,
                        config=ttnn.ReaderConfigDescriptor(),
                    ),
                    ttnn.KernelDescriptor(
                        kernel_source=PLAN_SEQUENCE_KERNEL,
                        core_ranges=core_ranges,
                        compile_time_args=compute_args,
                        runtime_args=compute_runtime,
                        config=ttnn.ComputeConfigDescriptor(
                            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=fp32_dest
                        ),
                    ),
                ]
            )
        input_tensor = ttnn.from_torch(
            physical.reshape(core_count * allocation_rows * TILE, row_stride * TILE),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=memory((allocation_rows * TILE, row_stride * TILE)),
        )
        output = ttnn.from_torch(
            torch.full((core_count * output_capacity * TILE, TILE), -992.0),
            dtype=output_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=memory((output_capacity * TILE, TILE)),
        )
        result = ttnn.generic_op(
            [input_tensor, output],
            ttnn.ProgramDescriptor(
                kernels=kernels,
                semaphores=[],
                cbs=[
                    ttnn.cb_descriptor_from_sharded_tensor(CB_INPUT, input_tensor),
                    ttnn.cb_descriptor_from_sharded_tensor(CB_OUTPUT, output),
                    _scratch_cb(CB_SCALER, ttnn.bfloat16, auxiliary_tiles, core_ranges=grid),
                    *(
                        [_scratch_cb(CB_ACCUMULATOR, output_dtype, output_capacity, core_ranges=grid)]
                        if calls > 1
                        else []
                    ),
                ],
            ),
        )
        output_tiles = ttnn.to_torch(result).reshape(core_count, output_capacity, TILE, TILE)
        for index, golden in enumerate(expected):
            batches, valid_outputs = golden.shape
            tiles_per_batch = (valid_outputs + TILE - 1) // TILE
            count = batches * tiles_per_batch
            tiles = output_tiles[index, :count]
            lanes = tiles[:, :, 0] if dim == "REDUCE_ROW" else tiles[:, 0, :]
            lanes = lanes.reshape(batches, -1)
            torch.testing.assert_close(
                lanes[:, :valid_outputs].float(),
                golden,
                rtol=0.02,
                atol=0.02,
                msg=lambda error: f"core={index}, shape={shapes[index]}\n{error}\nactual={lanes[:, :8]}\nexpected={golden[:, :8]}",
            )
            assert torch.all(lanes[:, valid_outputs:] == 0), f"core {index}: invalid output lanes were not cleared"
            assert torch.all(output_tiles[index, count:] == -992), f"core {index}: wrote beyond runtime output shape"


@pytest.mark.parametrize("dim", ("REDUCE_ROW", "REDUCE_COL"))
@pytest.mark.parametrize(
    "pool,algorithm",
    (
        ("SUM", "REDUCE_TILE"),
        ("MAX", "REDUCE_TILE"),
        ("AVG", "REDUCE_TILE"),
        ("SUM", "ACCUMULATE_VIA_ADD"),
        ("AVG", "ACCUMULATE_VIA_ADD"),
    ),
)
@pytest.mark.parametrize("fp32_input", (False, True))
def test_reduce_runtime_tail_stream_wraps(device, dim, pool, algorithm, fp32_input):
    """Tail packets retain fixed CB boundaries when either tile dimension shrinks."""
    input_dtype = ttnn.float32 if fp32_input else ttnn.bfloat16
    block = _PLANNER.ReduceBlockSpec(
        256,
        256,
        input_dtype,
        ttnn.float32,
        batches=2,
        resident_output_tiles=16,
        tail=_PLANNER.ReduceTailConfig(),
    )
    scalar = 1 / 256 if pool == "AVG" else 1.0
    cap_tiles = 4 if dim == "REDUCE_ROW" else 12
    sequence = _PLANNER.make_reduce_sequence_plan(
        reductions=[
            (
                CB_INPUT,
                _PLANNER.ReduceCallConfig(
                    block,
                    _REDUCE_MATH[pool],
                    _REDUCE_DIM[dim],
                    scalar,
                    _PLANNER.ReduceFp32Mode.FAST,
                    max_input_cb_bytes=cap_tiles * ttnn.tile_size(input_dtype),
                ),
            )
        ],
        cb_ids=_PLANNER.ReduceSequenceCbIds(CB_SCALER, CB_ACCUMULATOR, CB_OUTPUT),
        hardware=_PLANNER.ReduceHardwareConfig(
            arch=device.arch(),
            fp32_dest_acc_en=True,
            dst_full_sync_en=False,
            available_l1_bytes=ttnn.get_max_worker_l1_unreserved_size(),
        ),
        algorithm=_ALGORITHM[algorithm],
    )
    plan = sequence.calls[0].plan
    assert plan.input_policy == _PLANNER.ReduceInputPolicy.CHUNKED_WAIT_CHUNKED_POP
    axis, group = plan.chunk.reduce_axis_tiles, plan.chunk.output_tiles
    assert axis == 2
    compute_args, auxiliary_args = _serialize_plan(sequence)
    # The test reader uses the same call metadata to count runtime packets.
    auxiliary_args += sequence.calls[0].compile_time_args
    for height, width, batches in ((135, 135, 2), (64, 64, 1), (1, 17, 2)):
        ht, wt = (height + 31) // TILE, (width + 31) // TILE
        values = (torch.arange(batches * height * width).reshape(batches, height, width) % 7 - 3).to(torch.bfloat16)
        padded = torch.full((batches, ht * TILE, wt * TILE), 128, dtype=torch.bfloat16)
        padded[:, :height, :width] = values
        if dim == "REDUCE_ROW":
            padded[:, height:, :] = float("nan")
        else:
            padded[:, :, width:] = float("nan")
        tiles = []
        for batch in range(batches):
            for out in range(0, ht if dim == "REDUCE_ROW" else wt, group):
                for base in range(0, wt if dim == "REDUCE_ROW" else ht, axis):
                    for a in range(axis):
                        for o in range(group):
                            h, w = (out + o, base + a) if dim == "REDUCE_ROW" else (base + a, out + o)
                            tiles.append(
                                padded[batch, h * TILE : (h + 1) * TILE, w * TILE : (w + 1) * TILE]
                                if h < ht and w < wt
                                else torch.full((TILE, TILE), 999, dtype=torch.bfloat16)
                            )
        source_values = torch.cat(tiles, dim=0)
        source = ttnn.from_torch(
            source_values,
            dtype=input_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=_sharded_memory_config(source_values.shape),
        )
        output = ttnn.from_torch(
            torch.full((16 * TILE, TILE), -999.0),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=_sharded_memory_config((16 * TILE, TILE)),
        )
        runtime = [
            (ttnn.CoreCoord(0, 0), plan.get_runtime_shape_args(_PLANNER.ReduceValidShape(height, width, batches)))
        ]
        result = ttnn.generic_op(
            [source, output],
            ttnn.ProgramDescriptor(
                kernels=[
                    ttnn.KernelDescriptor(
                        kernel_source="tests/ttnn/unit_tests/kernel_lib/reduce/kernels/reduce_tail_stream_reader.cpp",
                        core_ranges=_single_core(),
                        compile_time_args=auxiliary_args,
                        runtime_args=runtime,
                        config=ttnn.ReaderConfigDescriptor(),
                    ),
                    ttnn.KernelDescriptor(
                        kernel_source=PLAN_SEQUENCE_KERNEL,
                        core_ranges=_single_core(),
                        compile_time_args=compute_args,
                        runtime_args=runtime,
                        defines=[("EXTERNAL_READER", "1")],
                        config=ttnn.ComputeConfigDescriptor(
                            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
                        ),
                    ),
                ],
                semaphores=[],
                cbs=[
                    ttnn.cb_descriptor_from_sharded_tensor(3, source),
                    ttnn.cb_descriptor_from_sharded_tensor(CB_OUTPUT, output),
                    _scratch_cb(CB_INPUT, input_dtype, cap_tiles),
                    _scratch_cb(CB_SCALER, input_dtype, len(sequence.auxiliary.tiles)),
                ],
            ),
        )
        count = batches * (ht if dim == "REDUCE_ROW" else wt)
        result_tiles = ttnn.to_torch(result).reshape(16, TILE, TILE)
        lanes = result_tiles[:count, :, 0] if dim == "REDUCE_ROW" else result_tiles[:count, 0, :]
        lanes = lanes.reshape(batches, -1)
        valid_outputs = height if dim == "REDUCE_ROW" else width
        reduce_dim = -1 if dim == "REDUCE_ROW" else -2
        golden = values.float().amax(reduce_dim) if pool == "MAX" else values.float().sum(reduce_dim)
        if pool == "AVG":
            golden /= width if dim == "REDUCE_ROW" else height
        torch.testing.assert_close(lanes[:, :valid_outputs], golden, rtol=0.01, atol=0.01)
        assert torch.all(lanes[:, valid_outputs:] == 0)
        assert torch.all(result_tiles[count:] == -999)


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

    plan = _repeated_input_cb_plan(input_tensor, output)
    compute_compile_time_args, auxiliary_compile_time_args = _serialize_plan(plan)
    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(CB_INPUT, input_tensor),
        ttnn.cb_descriptor_from_sharded_tensor(CB_OUTPUT, output),
        _scratch_cb(CB_SCALER, ttnn.bfloat16, len(plan.auxiliary.tiles)),
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


@pytest.mark.parametrize(
    "dtype,fp32_mode,pool", (("bf16", "Fast", "SUM"), ("int32", "Fast", "MAX"), ("fp32", "Accurate", "SUM"))
)
@pytest.mark.parametrize("dim", ("REDUCE_ROW", "REDUCE_COL"))
@pytest.mark.parametrize("calls", (1, 2))
def test_reduce_helpers_empty_auxiliary(device, dtype, fp32_mode, pool, dim, calls):
    if "QUASAR" in str(device.arch()).upper():
        pytest.skip("The planner's auxiliary-free additive and SFPU backends are not enabled on Quasar")
    case = ReduceCase(
        name=f"empty-aux-{dtype}-{dim}-{calls}",
        family="empty-auxiliary",
        dim=dim,
        rows=8 if dim == "REDUCE_COL" else 2,
        cols=8 if dim == "REDUCE_ROW" else 3,
        batches=2,
        pool=pool,
        input_mode="alias" if calls == 1 else "chunked",
        calls=calls,
        input_dtype=dtype,
        output_dtype="int32" if dtype == "int32" else "fp32",
        fp32_mode=fp32_mode,
        allow_empty_auxiliary=True,
    )
    actual, expected = _run_case(device, case)
    torch.testing.assert_close(actual.to(torch.float64), expected.to(torch.float64), rtol=0, atol=0)


def test_reduce_helpers_empty_auxiliary_with_existing_cb(device):
    if "QUASAR" in str(device.arch()).upper():
        pytest.skip("AccumulateViaAdd is not enabled on Quasar")
    case = ReduceCase(
        name="empty-aux-kept-cb",
        family="empty-auxiliary",
        dim="REDUCE_ROW",
        rows=2,
        cols=8,
        input_mode="alias",
        output_dtype="fp32",
        allow_empty_auxiliary=True,
    )
    actual, expected = _run_case(device, case, keep_empty_auxiliary_cb=True)
    torch.testing.assert_close(actual.to(torch.float64), expected.to(torch.float64), rtol=0, atol=0)


@pytest.mark.parametrize("dim", ("REDUCE_ROW", "REDUCE_COL"))
def test_reduce_helpers_empty_auxiliary_then_zero_pair(device, dim):
    if "QUASAR" in str(device.arch()).upper():
        pytest.skip("AccumulateViaAdd is not enabled on Quasar")
    case = ReduceCase(
        name=f"empty-then-zero-{dim}",
        family="empty-then-zero",
        dim=dim,
        rows=9 if dim == "REDUCE_COL" else 2,
        cols=9 if dim == "REDUCE_ROW" else 3,
        input_mode="alias",
        calls=2,
        input_dtype="bf8",
        output_dtype="fp32",
        allow_empty_auxiliary=True,
    )
    actual, expected = _run_case(device, case)
    torch.testing.assert_close(actual.to(torch.float64), expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", ("bf8", "bf4"))
@pytest.mark.parametrize("dim", ("REDUCE_ROW", "REDUCE_COL"))
@pytest.mark.parametrize(
    "input_mode,partial,calls",
    (("alias", 0, 2), ("alias", 15, 2), ("chunked", 0, 2), ("chunked", 15, 2), ("alias", 15, 1)),
)
def test_reduce_helpers_mixed_auxiliary_format(device, dtype, dim, input_mode, partial, calls):
    """Compressed input alternates with BF16 zero/mask tiles after input-only startup."""
    if "QUASAR" in str(device.arch()).upper() and calls > 1:
        pytest.skip("The planner does not select AccumulateViaAdd on Quasar")
    axis_tiles = 9 + bool(partial) if calls > 1 else 3
    case = ReduceCase(
        name=f"mixed-aux-{dtype}-{dim}-{input_mode}-partial{partial}-calls{calls}",
        family="mixed-auxiliary-format",
        dim=dim,
        rows=axis_tiles if dim == "REDUCE_COL" else 2,
        cols=axis_tiles if dim == "REDUCE_ROW" else 3,
        batches=2,
        input_mode=input_mode,
        calls=calls,
        input_dtype=dtype,
        output_dtype="fp32",
        partial_elements=partial,
    )
    actual, expected = _run_case(device, case)
    torch.testing.assert_close(actual.to(torch.float64), expected, rtol=0, atol=0)


@pytest.mark.parametrize("case", ALL_CASES, ids=lambda case: case.name)
def test_reduce_helpers_complete_input_space(device, case: ReduceCase):
    """Exercise every valid helper branch and its numerical/layout boundaries."""
    if "QUASAR" in str(device.arch()).upper() and (case.input_dtype == "int32" or case.fp32_mode == "Accurate"):
        pytest.skip("The reduce helper rejects SFPU reduce paths on Quasar")
    if "QUASAR" in str(device.arch()).upper() and case.pool == "MAX" and case.dim == "REDUCE_ROW" and case.calls > 1:
        pytest.skip("The MAX row accumulator reload is not supported on Quasar")

    actual, expected = _run_case(device, case)
    if case.input_dtype == "int32":
        torch.testing.assert_close(actual.to(torch.int64), expected, rtol=0, atol=0)
    elif case.pool == "MAX" and case.partial_elements:
        torch.testing.assert_close(actual.to(torch.float64), expected.to(torch.float64), rtol=0, atol=0)
    else:
        rtol = 0.05 if case.calls > 1 or case.input_dtype == "bf16" else 0.02
        torch.testing.assert_close(
            actual.to(torch.float64),
            expected.to(torch.float64),
            rtol=rtol,
            atol=0.1,
            msg=case.name,
        )


@pytest.mark.parametrize("dtype", ["int32", "fp32"])
@pytest.mark.parametrize("pool", ["SUM", "MAX", "MIN"])
@pytest.mark.parametrize("cols", [1, 3])
def test_reduce_helpers_narrow_sfpu(device, dtype, pool, cols):
    """A narrow H reduction must still have a separate SFPU work register."""
    if "QUASAR" in str(device.arch()).upper():
        pytest.skip("The reduce helper rejects SFPU reduce paths on Quasar")
    case = ReduceCase(
        name=f"narrow-sfpu-{dtype}-{pool}-cols{cols}",
        family="shape-boundaries",
        dim="REDUCE_COL",
        rows=3,
        cols=cols,
        batches=2,
        pool=pool,
        input_dtype=dtype,
        output_dtype=dtype,
        fp32_mode="Accurate" if dtype == "fp32" else "Fast",
    )
    actual, expected = _run_case(device, case)
    if dtype == "int32":
        torch.testing.assert_close(actual.to(torch.int64), expected, rtol=0, atol=0)
    else:
        torch.testing.assert_close(actual.to(torch.float64), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("dtype", ["int32", "fp32"])
@pytest.mark.parametrize("dim", ["REDUCE_ROW", "REDUCE_COL"])
@pytest.mark.parametrize("calls", [1, 2])
@pytest.mark.parametrize("scalar", [0.5, -0.5, 0.0])
def test_reduce_helpers_sfpu_scaled_sum(device, dtype, dim, calls, scalar):
    """Apply scaling once after the final SFPU sum, including integer truncation."""
    if "QUASAR" in str(device.arch()).upper():
        pytest.skip("The reduce helper rejects SFPU reduce paths on Quasar")
    case = ReduceCase(
        name=f"scaled-sfpu-{dtype}-{dim}-calls{calls}-scalar{scalar}",
        family="scalar",
        dim=dim,
        rows=3,
        cols=3,
        calls=calls,
        scalar=scalar,
        input_dtype=dtype,
        output_dtype=dtype,
        fp32_mode="Accurate" if dtype == "fp32" else "Fast",
    )
    actual, expected = _run_case(device, case)
    if dtype == "int32":
        # These small sums are exactly representable in FLOAT32. The helper's
        # post-multiply converts back to INT32 with truncation toward zero.
        torch.testing.assert_close(actual.to(torch.int64), expected.trunc().to(torch.int64), rtol=0, atol=0)
    else:
        torch.testing.assert_close(actual.to(torch.float64), expected, rtol=1e-5, atol=1e-5)
