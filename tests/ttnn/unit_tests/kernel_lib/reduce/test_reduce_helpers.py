# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Direct device tests for the reduce compute and dataflow helpers."""

from dataclasses import dataclass
import struct

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
SCALER_KERNEL = "tests/ttnn/unit_tests/kernel_lib/reduce/kernels/reduce_scaler.cpp"

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
    post_multiplier: float | None = None
    custom_scaler: float | None = None

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

    @property
    def reduce_factor(self) -> int:
        if self.pool != "AVG":
            return 1
        if self.dim == "REDUCE_ROW":
            return self.cols * self.valid_elements
        if self.dim == "REDUCE_COL":
            return self.rows * self.valid_elements
        return self.rows * self.cols * TILE * TILE


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
        ),
        # Regression for https://github.com/tenstorrent/tt-metal/issues/54178.
        # calls=2 makes the second reduce() use Accumulate::at(cb_accumulator, 1),
        # which must reload all four column outputs for each batch into DEST.
        ReduceCase(
            name="regression-col-3x4-two-batches-two-calls",
            family="regression",
            dim="REDUCE_COL",
            rows=3,
            cols=4,
            batches=2,
            pool="SUM",
            policy="WaitAndPopPerTile",
            layout="contiguous",
            calls=2,
            input_dtype="bf16",
            output_dtype="fp32",
        ),
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

    for dtype in ("bf16", "fp32"):
        for dim in DIMS:
            rows, cols, _ = _shape_for_dim(dim)
            cases.append(
                ReduceCase(
                    name=f"custom-scaler-{dtype}-{dim}",
                    family="custom-scaler",
                    dim=dim,
                    rows=rows,
                    cols=cols,
                    input_dtype=dtype,
                    output_dtype="fp32",
                    custom_scaler=0.5,
                )
            )
    return cases


ALL_CASES = tuple(_input_space_cases() + _regression_cases() + _numerical_space_cases() + _boundary_cases())


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


_assert_complete_case_matrix()


def _single_core() -> ttnn.CoreRangeSet:
    core = ttnn.CoreCoord(0, 0)
    return ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])


def _runtime_args(values: list[int]) -> ttnn.RuntimeArgs:
    args = ttnn.RuntimeArgs()
    args[0][0] = values
    return args


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
        ("REDUCE_FACTOR", str(case.reduce_factor)),
        ("REDUCE_EXPECTED_COL_CHUNK", str(case.col_chunk)),
    ]
    if case.post_multiplier is not None:
        defines.append(("REDUCE_POST_MULTIPLIER_BITS", hex(_float_bits(case.post_multiplier))))
    if case.custom_scaler is not None:
        defines.append(("REDUCE_CUSTOM_SCALER_BITS", hex(_float_bits(case.custom_scaler))))
    return defines


def _compute_config(case: ReduceCase) -> ttnn.ComputeConfigDescriptor:
    config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
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


def _reduce_logical_chunk(case: ReduceCase, chunk: torch.Tensor) -> torch.Tensor:
    chunk = chunk.to(torch.float64) if case.input_dtype != "int32" else chunk.to(torch.int64)
    if case.dim == "REDUCE_ROW":
        values = chunk.reshape(case.batches, case.rows * TILE, case.cols, TILE)[..., : case.valid_elements]
        values = values.flatten(-2)
        reduce_axis = -1
    elif case.dim == "REDUCE_COL":
        values = chunk.reshape(case.batches, case.rows, TILE, case.cols * TILE)[:, :, : case.valid_elements, :]
        values = values.transpose(1, 2).flatten(1, 2)
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

    if case.custom_scaler is not None:
        scaler_applications = 2 if case.dim == "REDUCE_SCALAR" else 1
        reduced = reduced * (case.custom_scaler**scaler_applications)
    return reduced


def _golden(case: ReduceCase, chunks: list[torch.Tensor]) -> torch.Tensor:
    partials = torch.stack([_reduce_logical_chunk(case, chunk) for chunk in chunks])
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
        _scratch_cb(CB_SCALER, _scaler_dtype(case), 1),
    ]
    if case.calls > 1:
        # Exact output cardinality is intentional: it covers the wraparound case
        # that exposed bulk output reservation in a non-popping policy.
        cbs.append(_scratch_cb(CB_ACCUMULATOR, output_dtype, case.output_tiles))

    defines = _defines(case)
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=SCALER_KERNEL,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=_single_core(),
            runtime_args=_runtime_args([case.valid_elements]),
            defines=defines,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=COMPUTE_KERNEL,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=_single_core(),
            compile_time_args=[case.calls],
            runtime_args=_runtime_args([case.rows, case.cols, case.batches, case.row_stride]),
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
        torch.testing.assert_close(actual.to(torch.int64), expected, rtol=0, atol=0, msg=case.name)
    else:
        torch.testing.assert_close(
            actual.to(torch.float64),
            expected.to(torch.float64),
            rtol=0.01,
            atol=0.01,
            msg=case.name,
        )
