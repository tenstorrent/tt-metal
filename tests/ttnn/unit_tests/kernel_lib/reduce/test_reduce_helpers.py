# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Direct device tests for reduce dispatch and compute-owned scaler lifecycle."""

import os
import struct
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
DEST_LIMIT = 4  # fp32 DEST + half synchronization, fixed by _compute_config().

COMPUTE_KERNEL = "tests/ttnn/unit_tests/kernel_lib/reduce/kernels/reduce.cpp"

POLICIES = ("WaitAndPopPerTile", "BulkWaitBulkPop", "WaitUpfrontNoPop", "NoWaitNoPop")
INDEXED_POLICIES = ("WaitUpfrontNoPop", "NoWaitNoPop")
DIMS = ("REDUCE_ROW", "REDUCE_COL", "REDUCE_SCALAR")


@dataclass(frozen=True)
class ReduceCase:
    name: str
    family: str
    dim: str
    rows: int
    cols: int
    batches: int = 1
    pool: str = "SUM"
    policy: str = "WaitAndPopPerTile"
    layout: str = "contiguous"
    calls: int = 1
    input_dtype: str = "bf16"
    output_dtype: str = "bf16"
    fp32_mode: str = "Fast"
    reconfig_mode: str = "INPUT_AND_OUTPUT"
    valid_elements: int = TILE
    later_valid_elements: int = 0
    post_multiplier: float | None = None
    algorithm: str = "Auto"

    @property
    def row_stride(self) -> int:
        return self.cols if self.layout == "contiguous" else self.cols + 2

    @property
    def output_tiles(self) -> int:
        if self.dim == "REDUCE_ROW":
            return self.rows * self.batches
        if self.dim == "REDUCE_COL":
            return self.cols * self.batches
        return self.batches

    @property
    def uses_sfpu(self) -> bool:
        return self.dim != "REDUCE_SCALAR" and (
            self.input_dtype == "int32" or (self.input_dtype == "fp32" and self.fp32_mode == "Accurate")
        )

    @property
    def col_chunk(self) -> int:
        return DEST_LIMIT - 1 if self.uses_sfpu else DEST_LIMIT


def _shape_for_dim(dim: str) -> tuple[int, int, int]:
    if dim == "REDUCE_ROW":
        return 3, 2, 2
    if dim == "REDUCE_COL":
        return 3, 5, 2  # Five columns cross the four-tile FPU DEST chunk boundary.
    return 2, 3, 2


def _input_space_cases() -> list[ReduceCase]:
    cases = []
    for dim in DIMS:
        rows, cols, batches = _shape_for_dim(dim)
        for policy in POLICIES:
            # The helper's debug contract requires a BulkWaitBulkPop REDUCE_COL
            # CB capacity to be a multiple of Ht * DEST chunk. Eight columns are
            # two exact four-tile chunks; the other policies retain a 4+1 tail.
            policy_cols = 8 if dim == "REDUCE_COL" and policy == "BulkWaitBulkPop" else cols
            layouts = ("contiguous", "strided") if policy in INDEXED_POLICIES else ("contiguous",)
            for layout in layouts:
                for accumulated in (False, True):
                    calls = 2 if accumulated else 1
                    cases.append(
                        ReduceCase(
                            name=f"input-{dim}-{policy}-{layout}-acc{int(accumulated)}",
                            family="input-space",
                            dim=dim,
                            rows=rows,
                            cols=policy_cols,
                            batches=batches,
                            policy=policy,
                            layout=layout,
                            calls=calls,
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
            policy="NoWaitNoPop",
            layout="contiguous",
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

    for mode in ("NONE", "INPUT", "OUTPUT", "INPUT_AND_OUTPUT"):
        cases.append(
            ReduceCase(
                name=f"reconfig-{mode}",
                family="reconfiguration",
                dim="REDUCE_ROW",
                rows=2,
                cols=2,
                output_dtype="fp32",
                reconfig_mode=mode,
            )
        )

    for dim in DIMS:
        rows, cols, _ = _shape_for_dim(dim)
        cases.append(
            ReduceCase(
                name=f"post-multiply-{dim}",
                family="post-operation",
                dim=dim,
                rows=rows,
                cols=cols,
                output_dtype="fp32",
                post_multiplier=0.5,
            )
        )

    for dim in ("REDUCE_ROW", "REDUCE_COL"):
        for valid_elements in (1, 15, 16, 17, 31, 32):
            cases.append(
                ReduceCase(
                    name=f"scaler-face-bf16-{dim}-valid{valid_elements}",
                    family="scaler-boundaries",
                    dim=dim,
                    rows=1,
                    cols=1,
                    output_dtype="fp32",
                    valid_elements=valid_elements,
                )
            )
        for valid_elements in (15, 17):
            cases.append(
                ReduceCase(
                    name=f"scaler-face-fp32-{dim}-valid{valid_elements}",
                    family="scaler-boundaries",
                    dim=dim,
                    rows=1,
                    cols=1,
                    input_dtype="fp32",
                    output_dtype="fp32",
                    valid_elements=valid_elements,
                )
            )

        rows, cols = (1, 2) if dim == "REDUCE_ROW" else (2, 1)
        cases.append(
            ReduceCase(
                name=f"scaler-ragged-avg-{dim}",
                family="scaler-boundaries",
                dim=dim,
                rows=rows,
                cols=cols,
                pool="AVG",
                output_dtype="fp32",
                valid_elements=17,
            )
        )

    return cases


def _managed_scaler_lifecycle_cases() -> list[ReduceCase]:
    return [
        # A full scaler becomes a [full, partial] pair in the same DFB. The old tile must be popped
        # before the pair can be reserved in the two-tile DFB.
        ReduceCase(
            name="managed-scaler-row-full-to-partial",
            family="managed-scaler-lifecycle",
            dim="REDUCE_ROW",
            rows=1,
            cols=2,
            calls=2,
            valid_elements=32,
            later_valid_elements=17,
            output_dtype="fp32",
        ),
        # The inverse transition verifies that reduce() remembers and removes both resident tiles.
        ReduceCase(
            name="managed-scaler-col-partial-to-full",
            family="managed-scaler-lifecycle",
            dim="REDUCE_COL",
            rows=2,
            cols=1,
            calls=2,
            valid_elements=17,
            later_valid_elements=32,
            output_dtype="fp32",
        ),
        # Reusing the same pair must not try to reserve a duplicate pair in the full DFB.
        ReduceCase(
            name="managed-scaler-row-reuse-partial",
            family="managed-scaler-lifecycle",
            dim="REDUCE_ROW",
            rows=1,
            cols=2,
            calls=2,
            valid_elements=17,
            output_dtype="fp32",
        ),
    ]


def _accumulate_via_add_cases() -> list[ReduceCase]:
    cases = []
    for dim in DIMS:
        rows, cols, batches = _shape_for_dim(dim)
        cases.append(
            ReduceCase(
                name=f"accumulate-via-add-{dim}",
                family="algorithm-dispatch",
                dim=dim,
                rows=rows,
                cols=cols,
                batches=batches,
                policy="WaitUpfrontNoPop",
                output_dtype="fp32",
                algorithm="AccumulateViaAdd",
            )
        )
    return cases


def _performance_cases() -> list[ReduceCase]:
    """Larger single-core shapes for stable device-profile comparisons."""
    cases = []
    shapes = {
        "REDUCE_ROW": ((1, 16), (2, 32), (4, 32), (8, 32)),
        "REDUCE_COL": ((16, 1), (32, 2), (32, 4), (32, 8)),
        "REDUCE_SCALAR": ((4, 4), (8, 8), (8, 16), (16, 16)),
    }

    # The four tiers contain 16, 64, 128, and 256 input tiles. SUM and AVG
    # exercise the same scaler setup with very different steady-state work.
    for dim, dim_shapes in shapes.items():
        for rows, cols in dim_shapes:
            tiles = rows * cols
            for pool in ("SUM", "AVG"):
                cases.append(
                    ReduceCase(
                        name=f"perf-bf16-{pool}-{dim}-t{tiles}",
                        family="performance",
                        dim=dim,
                        rows=rows,
                        cols=cols,
                        pool=pool,
                        output_dtype="fp32",
                    )
                )

    # A large indexed-policy sample separates scaler work from input ownership.
    for dim, (rows, cols) in {key: value[-1] for key, value in shapes.items()}.items():
        cases.append(
            ReduceCase(
                name=f"perf-bf16-SUM-{dim}-t256-indexed",
                family="performance",
                dim=dim,
                rows=rows,
                cols=cols,
                policy="NoWaitNoPop",
                output_dtype="fp32",
            )
        )

    # FP32 FPU reductions use fewer tiers because each input tile is twice as
    # large in L1. Accurate FP32 and Int32 select the scaler-free SFPU path.
    medium_shapes = {key: value[1:3] for key, value in shapes.items()}
    for dim, dim_shapes in medium_shapes.items():
        for rows, cols in dim_shapes:
            tiles = rows * cols
            for pool in ("SUM", "AVG"):
                cases.append(
                    ReduceCase(
                        name=f"perf-fp32-{pool}-{dim}-t{tiles}-fast",
                        family="performance",
                        dim=dim,
                        rows=rows,
                        cols=cols,
                        pool=pool,
                        input_dtype="fp32",
                        output_dtype="fp32",
                    )
                )

    for dtype, fp32_mode in (("fp32", "Accurate"), ("int32", "Fast")):
        for dim in ("REDUCE_ROW", "REDUCE_COL"):
            for rows, cols in medium_shapes[dim]:
                tiles = rows * cols
                for pool in ("SUM", "MAX"):
                    cases.append(
                        ReduceCase(
                            name=f"perf-{dtype}-{pool}-{dim}-t{tiles}-{fp32_mode.lower()}",
                            family="performance",
                            dim=dim,
                            rows=rows,
                            cols=cols,
                            pool=pool,
                            input_dtype=dtype,
                            output_dtype=dtype,
                            fp32_mode=fp32_mode,
                        )
                    )

    # Partial-edge and repeated-call cases make scaler construction/reuse
    # visible without letting a one-tile workload dominate the measurement.
    for pool in ("SUM", "AVG"):
        for dim in ("REDUCE_ROW", "REDUCE_COL"):
            rows, cols = shapes[dim][2]
            cases.append(
                ReduceCase(
                    name=f"perf-bf16-{pool}-{dim}-t128-partial17",
                    family="performance",
                    dim=dim,
                    rows=rows,
                    cols=cols,
                    pool=pool,
                    output_dtype="fp32",
                    valid_elements=17,
                )
            )

    for dim in DIMS:
        rows, cols = shapes[dim][1]
        cases.append(
            ReduceCase(
                name=f"perf-bf16-SUM-{dim}-t64-two-calls",
                family="performance",
                dim=dim,
                rows=rows,
                cols=cols,
                calls=2,
                output_dtype="fp32",
            )
        )

    for dim in DIMS:
        for rows, cols in medium_shapes[dim]:
            tiles = rows * cols
            cases.append(
                ReduceCase(
                    name=f"perf-bf16-SUM-{dim}-t{tiles}-accumulate-via-add",
                    family="performance",
                    dim=dim,
                    rows=rows,
                    cols=cols,
                    policy="WaitUpfrontNoPop",
                    output_dtype="fp32",
                    algorithm="AccumulateViaAdd",
                )
            )

    return cases


ALL_CASES = tuple(
    _input_space_cases()
    + _regression_cases()
    + _numerical_space_cases()
    + _boundary_cases()
    + _managed_scaler_lifecycle_cases()
    + _accumulate_via_add_cases()
    + _performance_cases()
)


def _assert_complete_case_matrix() -> None:
    """Keep additions/removals from silently punching holes in the advertised space."""
    input_cases = [case for case in ALL_CASES if case.family == "input-space"]
    actual = {(case.dim, case.policy, case.layout, case.calls > 1) for case in input_cases}
    expected = {
        (dim, policy, layout, accumulated)
        for dim in DIMS
        for policy in POLICIES
        for layout in (("contiguous", "strided") if policy in INDEXED_POLICIES else ("contiguous",))
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

    performance_cases = [case for case in ALL_CASES if case.family == "performance"]
    assert len(performance_cases) == 68
    assert {case.rows * case.cols for case in performance_cases} == {16, 64, 128, 256}


_assert_complete_case_matrix()


def _single_core() -> ttnn.CoreRangeSet:
    core = ttnn.CoreCoord(0, 0)
    return ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])


def _sharded_memory_config(shape: tuple[int, int]) -> ttnn.MemoryConfig:
    return ttnn.create_sharded_memory_config(
        shape=shape,
        core_grid=_single_core(),
        strategy=ttnn.ShardStrategy.HEIGHT,
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


def _float_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _defines(case: ReduceCase) -> list[tuple[str, str]]:
    defines = [
        ("REDUCE_OP", f"ckernel::PoolType::{case.pool}"),
        ("REDUCE_DIM", f"ckernel::ReduceDim::{case.dim}"),
        ("REDUCE_INPUT_POLICY", f"compute_kernel_lib::ReduceInputPolicy::{case.policy}"),
        ("REDUCE_RECONFIG_MODE", f"compute_kernel_lib::ReduceDataFormatReconfigMode::{case.reconfig_mode}"),
        ("REDUCE_FP32_MODE", f"ReduceFp32Mode::{case.fp32_mode}"),
        ("REDUCE_ALGORITHM", f"compute_kernel_lib::ReduceAlgorithm::{case.algorithm}"),
        ("REDUCE_EXPECTED_COL_CHUNK", str(case.col_chunk)),
    ]
    if case.post_multiplier is not None:
        defines.append(("REDUCE_POST_MULTIPLIER_BITS", hex(_float_bits(case.post_multiplier))))
    if os.environ.get("REDUCE_HELPERS_PROFILE"):
        defines.append(("REDUCE_HELPERS_PROFILE", "1"))
    return defines


def _compute_config(case: ReduceCase) -> ttnn.ComputeConfigDescriptor:
    config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi3,
        fp32_dest_acc_en=True,
        dst_full_sync_en=False,
    )
    if case.input_dtype == "fp32" and case.fp32_mode == "Accurate":
        # Host descriptors use the maximum CB count so this vector covers both Wormhole (32) and Blackhole (64).
        unpack_modes = [ttnn.UnpackToDestMode.Default] * 64
        unpack_modes[CB_INPUT] = ttnn.UnpackToDestMode.UnpackToDestFp32
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


def _physical_input(case: ReduceCase, logical: torch.Tensor) -> torch.Tensor:
    logical_tiles = _logical_tiles(case, logical)
    pad_value = -100_000 if case.pool == "MIN" else 100_000
    physical_tiles = torch.full(
        (case.batches, case.rows, case.row_stride, TILE, TILE),
        pad_value,
        dtype=logical.dtype,
    )

    streams_by_col_chunk = case.dim == "REDUCE_COL" and case.policy not in INDEXED_POLICIES
    if streams_by_col_chunk:
        ordered_tiles = []
        for batch in range(case.batches):
            for col_start in range(0, case.cols, case.col_chunk):
                col_end = min(col_start + case.col_chunk, case.cols)
                for row in range(case.rows):
                    for col in range(col_start, col_end):
                        ordered_tiles.append(logical_tiles[batch, row, col])
        physical_tiles.view(-1, TILE, TILE).copy_(torch.stack(ordered_tiles))
    else:
        physical_tiles[:, :, : case.cols] = logical_tiles

    return (
        physical_tiles.permute(0, 1, 3, 2, 4)
        .contiguous()
        .reshape(case.batches * case.rows * TILE, case.row_stride * TILE)
    )


def _reduce_logical_chunk(case: ReduceCase, chunk: torch.Tensor, valid_elements: int) -> torch.Tensor:
    chunk = chunk.to(torch.float64) if case.input_dtype != "int32" else chunk.to(torch.int64)
    if case.dim == "REDUCE_ROW":
        logical_width = (case.cols - 1) * TILE + valid_elements
        values = chunk[..., :logical_width]
        reduce_axis = -1
    elif case.dim == "REDUCE_COL":
        logical_height = (case.rows - 1) * TILE + valid_elements
        values = chunk[:, :logical_height, :]
        reduce_axis = 1
    else:
        values = chunk
        reduce_axis = (-2, -1)

    if case.pool == "SUM":
        reduced = values.sum(dim=reduce_axis)
    elif case.pool == "AVG":
        reduced = values.mean(dim=reduce_axis)
    elif case.pool == "MAX":
        reduced = values.amax(dim=reduce_axis)
    else:
        reduced = values.amin(dim=reduce_axis)

    return reduced


def _golden(case: ReduceCase, chunks: list[torch.Tensor]) -> torch.Tensor:
    partials = torch.stack(
        [
            _reduce_logical_chunk(
                case,
                chunk,
                case.valid_elements if call == 0 or case.later_valid_elements == 0 else case.later_valid_elements,
            )
            for call, chunk in enumerate(chunks)
        ]
    )
    if case.pool in ("SUM", "AVG"):
        golden = partials.sum(dim=0)
    elif case.pool == "MAX":
        golden = partials.amax(dim=0)
    else:
        golden = partials.amin(dim=0)
    if case.post_multiplier is not None:
        golden = golden * case.post_multiplier
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
    logical_chunks = _make_logical_chunks(case)
    input_dtype = _ttnn_dtype(case.input_dtype)
    output_dtype = _ttnn_dtype(case.output_dtype)

    physical_input = torch.cat([_physical_input(case, logical) for logical in logical_chunks], dim=0)
    device_input = ttnn.from_torch(
        physical_input,
        dtype=input_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=_sharded_memory_config(tuple(physical_input.shape)),
    )

    output_shape = _output_shape(case)
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        output_dtype,
        ttnn.TILE_LAYOUT,
        device,
        _sharded_memory_config(output_shape),
    )

    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(CB_INPUT, device_input),
        ttnn.cb_descriptor_from_sharded_tensor(CB_OUTPUT, output),
        _scratch_cb(CB_SCALER, _scaler_dtype(case), 2),
    ]
    if case.calls > 1:
        # Exact output cardinality is intentional: it covers the wraparound case
        # that exposed bulk output reservation in a non-popping policy.
        cbs.append(_scratch_cb(CB_ACCUMULATOR, output_dtype, case.output_tiles))

    defines = _defines(case)
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=COMPUTE_KERNEL,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=_single_core(),
            compile_time_args=[
                case.calls,
                case.rows,
                case.cols,
                case.batches,
                case.row_stride,
                case.valid_elements,
                case.later_valid_elements,
            ],
            defines=defines,
            config=_compute_config(case),
        ),
    ]

    result = ttnn.generic_op(
        [device_input, output],
        ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs),
    )
    actual = _meaningful_output(case, ttnn.to_torch(result))
    return actual, _golden(case, logical_chunks)


@pytest.mark.parametrize("case", ALL_CASES, ids=lambda case: case.name)
def test_reduce_helpers_complete_input_space(device, case: ReduceCase):
    """Exercise every valid helper branch and its numerical/layout boundaries."""
    if "QUASAR" in str(device.arch()).upper() and (case.input_dtype == "int32" or case.fp32_mode == "Accurate"):
        pytest.skip("The reduce helper rejects SFPU reduce paths on Quasar")

    actual, expected = _run_case(device, case)
    if case.input_dtype == "int32":
        torch.testing.assert_close(
            actual.to(torch.int64), expected, rtol=0, atol=0, msg=lambda msg: f"{case.name}\n{msg}"
        )
    else:
        rtol = 0.05 if case.calls > 1 or case.input_dtype == "bf16" else 0.02
        torch.testing.assert_close(
            actual.to(torch.float64),
            expected.to(torch.float64),
            rtol=rtol,
            atol=0.1,
            msg=lambda msg: f"{case.name}\n{msg}",
        )
