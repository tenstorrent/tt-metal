# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Device tests for the explicit reduce() parameters: algorithm, within-tile, reduce factor, partial mode,
reload mode, output group, auxiliary offset, batch stride and the auxiliary tile patterns."""

from dataclasses import dataclass
import math
import struct

import pytest
import torch

import ttnn

pytestmark = pytest.mark.use_module_device

TILE = 32
CB_INPUT = 0
CB_AUXILIARY = 1
CB_ACCUMULATOR = 2
CB_OUTPUT = 16
CB_STREAM_DESTINATION = 17
DEST_LIMIT = 4  # fp32 DEST + half synchronization, fixed by _compute_config().

COMPUTE_KERNEL = "tests/ttnn/unit_tests/kernel_lib/reduce/kernels/reduce_explicit.cpp"
AUXILIARY_KERNEL = "tests/ttnn/unit_tests/kernel_lib/reduce/kernels/reduce_explicit_auxiliary.cpp"

POLICIES = ("WaitAndPopPerTile", "BulkWaitBulkPop", "WaitUpfrontNoPop", "NoWaitNoPop")
INDEXED_POLICIES = ("WaitUpfrontNoPop", "NoWaitNoPop")
DIMS = ("REDUCE_ROW", "REDUCE_COL", "REDUCE_SCALAR")
RELOAD_MODES = ("FoldViaAdd", "CopySeedPairs", "CopySeedUniform", "CopySeedSfpuAdd", "CopySeedZeroPair")

TILE_TYPE = {"FirstRow": 0, "FirstColumn": 1, "FirstRowPerFaceRow": 2, "Zero": 3}


def _float_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


@dataclass(frozen=True)
class ReduceCase:
    name: str
    dim: str
    rows: int
    cols: int
    batches: int = 1
    pool: str = "SUM"
    algorithm: str = "ReduceTile"
    policy: str = "BulkWaitBulkPop"
    calls: int = 1
    input_dtype: str = "bf16"
    output_dtype: str = "fp32"
    fp32_mode: str = "Fast"
    partial: int = 0
    within_tile: str = "Collapse"
    reload: str = "CopySeedPairs"
    output_group: int = 0
    auxiliary_offset: int = 0
    row_padding: int = 0
    batch_padding: int = 0
    no_auxiliary: bool = False
    stream_output: bool = False
    post_exp: bool = False
    accumulator_to_dest: bool = False

    @property
    def additive(self) -> bool:
        return self.algorithm == "AccumulateViaAdd"

    @property
    def uses_sfpu(self) -> bool:
        return self.dim != "REDUCE_SCALAR" and (
            self.input_dtype == "int32" or (self.input_dtype == "fp32" and self.fp32_mode == "Accurate")
        )

    @property
    def partial_mode(self) -> str:
        if not self.partial:
            return "None"
        return "Mask" if self.additive else "Scaler"

    @property
    def row_stride(self) -> int:
        return self.cols + self.row_padding

    @property
    def batch_stride(self) -> int:
        return self.rows * self.row_stride + self.batch_padding

    @property
    def valid_height(self) -> int:
        if self.dim == "REDUCE_COL" and self.partial:
            return (self.rows - 1) * TILE + self.partial
        return self.rows * TILE

    @property
    def valid_width(self) -> int:
        if self.dim == "REDUCE_ROW" and self.partial:
            return (self.cols - 1) * TILE + self.partial
        return self.cols * TILE

    @property
    def reduce_axis_tiles(self) -> int:
        if self.dim == "REDUCE_ROW":
            return self.cols
        if self.dim == "REDUCE_COL":
            return self.rows
        return self.rows * self.cols

    @property
    def reduced_elements(self) -> int:
        if self.dim == "REDUCE_ROW":
            return self.valid_width
        if self.dim == "REDUCE_COL":
            return self.valid_height
        return self.valid_height * self.valid_width

    @property
    def average_divisor(self) -> int:
        per_call = self.reduce_axis_tiles if self.within_tile == "Skip" else self.reduced_elements
        return per_call * self.calls

    @property
    def reduce_factor(self) -> int:
        return self.average_divisor if self.pool == "AVG" and self.additive else 1

    @property
    def output_tiles(self) -> int:
        if self.dim == "REDUCE_ROW":
            return self.rows * self.batches
        if self.dim == "REDUCE_COL":
            return self.cols * self.batches
        return self.batches

    @property
    def col_chunk(self) -> int:
        """Column group in which a streamed REDUCE_COL input arrives."""
        if self.additive:
            return 1 if self.policy == "WaitAndPopPerTile" else (self.output_group or DEST_LIMIT)
        default = DEST_LIMIT - 1 if self.uses_sfpu else DEST_LIMIT
        return min(self.output_group, default) if self.output_group else default

    @property
    def auxiliary_tiles(self) -> list[tuple[str, int, float]]:
        if self.no_auxiliary:
            return []
        # Leading tiles that a correct auxiliary_tile_offset skips; reading any of them corrupts the result.
        tiles = [("FirstRow", TILE, 7.0)] * self.auxiliary_offset
        if self.additive:
            if self.partial:
                tiles.append(("FirstRow" if self.dim == "REDUCE_ROW" else "FirstColumn", self.partial, 1.0))
            tiles.append(("Zero", 0, 0.0))
            return tiles
        scaler = 1.0
        if self.pool == "AVG":
            scaler = 1.0 / self.average_divisor
            if self.dim == "REDUCE_SCALAR":
                scaler = 1.0 / math.sqrt(self.average_divisor)
        tiles.append(("FirstRow", TILE, scaler))
        if self.partial:
            tiles.append(("FirstRow" if self.dim == "REDUCE_ROW" else "FirstRowPerFaceRow", self.partial, scaler))
        return tiles


def _case(name: str, **kwargs) -> ReduceCase:
    return ReduceCase(name=name, **kwargs)


def _shape_for_dim(dim: str) -> tuple[int, int, int]:
    if dim == "REDUCE_ROW":
        return 2, 5, 2
    if dim == "REDUCE_COL":
        return 5, 5, 2  # Five columns cross the four-tile DEST group.
    return 2, 3, 2


def _algorithm_cases() -> list[ReduceCase]:
    cases = []
    for dim in DIMS:
        rows, cols, batches = _shape_for_dim(dim)
        for pool in ("SUM", "AVG"):
            reduce_tile_cols = 8 if dim == "REDUCE_COL" else cols
            cases.append(
                _case(
                    f"algorithm-ReduceTile-{pool}-{dim}",
                    dim=dim,
                    rows=rows,
                    cols=reduce_tile_cols,
                    batches=batches,
                    pool=pool,
                )
            )
            for policy in POLICIES:
                for calls in (1, 2):
                    cases.append(
                        _case(
                            f"algorithm-AccumulateViaAdd-{pool}-{dim}-{policy}-calls{calls}",
                            dim=dim,
                            rows=rows,
                            cols=cols,
                            batches=batches,
                            pool=pool,
                            algorithm="AccumulateViaAdd",
                            policy=policy,
                            calls=calls,
                        )
                    )
    for dtype in ("fp32", "bf8"):
        for dim in ("REDUCE_ROW", "REDUCE_COL"):
            rows, cols, batches = _shape_for_dim(dim)
            cases.append(
                _case(
                    f"algorithm-AccumulateViaAdd-{dtype}-{dim}",
                    dim=dim,
                    rows=rows,
                    cols=cols,
                    batches=batches,
                    algorithm="AccumulateViaAdd",
                    input_dtype=dtype,
                )
            )
    return cases


def _partial_cases() -> list[ReduceCase]:
    cases = []
    for dim in ("REDUCE_ROW", "REDUCE_COL"):
        rows, cols = (2, 3) if dim == "REDUCE_ROW" else (3, 2)
        for pool in ("SUM", "AVG", "MAX"):
            for calls in (1, 2):
                cases.append(
                    _case(
                        f"partial-scaler-{pool}-{dim}-calls{calls}",
                        dim=dim,
                        rows=rows,
                        cols=cols,
                        batches=2,
                        pool=pool,
                        calls=calls,
                        partial=17,
                    )
                )
        for valid in (1, 15, 16, 31):
            cases.append(_case(f"partial-scaler-bf16-{dim}-valid{valid}", dim=dim, rows=1, cols=1, partial=valid))
        for valid in (15, 17):
            cases.append(
                _case(
                    f"partial-scaler-fp32-{dim}-valid{valid}",
                    dim=dim,
                    rows=1,
                    cols=1,
                    input_dtype="fp32",
                    partial=valid,
                )
            )

        rows, cols, batches = _shape_for_dim(dim)
        for pool in ("SUM", "AVG"):
            for policy in POLICIES:
                cases.append(
                    _case(
                        f"partial-mask-{pool}-{dim}-{policy}",
                        dim=dim,
                        rows=rows,
                        cols=cols,
                        batches=batches,
                        pool=pool,
                        algorithm="AccumulateViaAdd",
                        policy=policy,
                        partial=7,
                    )
                )
            for policy in ("BulkWaitBulkPop", "NoWaitNoPop"):
                cases.append(
                    _case(
                        f"partial-mask-{pool}-{dim}-{policy}-calls2",
                        dim=dim,
                        rows=rows,
                        cols=cols,
                        batches=batches,
                        pool=pool,
                        algorithm="AccumulateViaAdd",
                        policy=policy,
                        calls=2,
                        partial=7,
                    )
                )
        for valid in (1, 16, 31):
            cases.append(
                _case(
                    f"partial-mask-{dim}-valid{valid}",
                    dim=dim,
                    rows=rows,
                    cols=cols,
                    algorithm="AccumulateViaAdd",
                    partial=valid,
                )
            )
    return cases


def _within_tile_cases() -> list[ReduceCase]:
    cases = []
    for dim in DIMS:
        rows, cols, batches = _shape_for_dim(dim)
        for pool in ("SUM", "AVG"):
            for policy in ("WaitAndPopPerTile", "NoWaitNoPop"):
                cases.append(
                    _case(
                        f"skip-{pool}-{dim}-{policy}",
                        dim=dim,
                        rows=rows,
                        cols=cols,
                        batches=batches,
                        pool=pool,
                        algorithm="AccumulateViaAdd",
                        policy=policy,
                        within_tile="Skip",
                    )
                )
        cases.append(
            _case(
                f"skip-SUM-{dim}-NoWaitNoPop-calls2",
                dim=dim,
                rows=rows,
                cols=cols,
                batches=batches,
                algorithm="AccumulateViaAdd",
                policy="NoWaitNoPop",
                calls=2,
                within_tile="Skip",
            )
        )
    return cases


def _reload_cases() -> list[ReduceCase]:
    cases = []
    for dim in ("REDUCE_ROW", "REDUCE_COL"):
        rows, cols, batches = _shape_for_dim(dim)
        for reload in RELOAD_MODES:
            for policy in INDEXED_POLICIES:
                cases.append(
                    _case(
                        f"reload-{reload}-{dim}-{policy}-calls3",
                        dim=dim,
                        rows=rows,
                        cols=cols,
                        batches=batches,
                        algorithm="AccumulateViaAdd",
                        policy=policy,
                        calls=3,
                        reload=reload,
                    )
                )
        cases.append(
            _case(
                f"reload-CopySeedZeroPair-partial-{dim}",
                dim=dim,
                rows=rows,
                cols=cols,
                batches=batches,
                algorithm="AccumulateViaAdd",
                policy="NoWaitNoPop",
                calls=2,
                reload="CopySeedZeroPair",
                partial=9,
            )
        )
    # Odd, even and single-tile reduce axes take different FoldViaAdd branches.
    for cols in (1, 2, 4):
        cases.append(
            _case(
                f"reload-FoldViaAdd-REDUCE_ROW-cols{cols}",
                dim="REDUCE_ROW",
                rows=2,
                cols=cols,
                algorithm="AccumulateViaAdd",
                policy="NoWaitNoPop",
                calls=2,
                reload="FoldViaAdd",
            )
        )
    return cases


def _chunk_cases() -> list[ReduceCase]:
    cases = []
    for chunk in (1, 2, 3):
        for policy in ("WaitAndPopPerTile", "BulkWaitBulkPop", "NoWaitNoPop"):
            cases.append(
                _case(
                    f"chunk-ReduceTile-{policy}-chunk{chunk}",
                    dim="REDUCE_COL",
                    rows=3,
                    cols=6,
                    batches=2,
                    policy=policy,
                    output_group=chunk,
                )
            )
        cases.append(
            _case(
                f"chunk-AccumulateViaAdd-BulkWaitBulkPop-chunk{chunk}",
                dim="REDUCE_COL",
                rows=3,
                cols=6,
                batches=2,
                algorithm="AccumulateViaAdd",
                output_group=chunk,
            )
        )
    cases.append(
        _case(
            "chunk-ReduceTile-int32-chunk2",
            dim="REDUCE_COL",
            rows=3,
            cols=6,
            policy="WaitAndPopPerTile",
            input_dtype="int32",
            output_dtype="int32",
            output_group=2,
        )
    )
    cases.append(
        _case(
            "chunk-ReduceTile-accumulate-chunk2",
            dim="REDUCE_COL",
            rows=3,
            cols=6,
            policy="WaitAndPopPerTile",
            calls=2,
            output_group=2,
        )
    )
    return cases


def _auxiliary_offset_cases() -> list[ReduceCase]:
    return [
        _case("offset-ReduceTile-SUM-REDUCE_ROW", dim="REDUCE_ROW", rows=2, cols=3, auxiliary_offset=2),
        _case(
            "offset-ReduceTile-AVG-REDUCE_SCALAR", dim="REDUCE_SCALAR", rows=2, cols=3, pool="AVG", auxiliary_offset=1
        ),
        _case("offset-ReduceTile-partial-REDUCE_COL", dim="REDUCE_COL", rows=3, cols=2, partial=5, auxiliary_offset=1),
        _case(
            "offset-AccumulateViaAdd-REDUCE_ROW",
            dim="REDUCE_ROW",
            rows=2,
            cols=5,
            algorithm="AccumulateViaAdd",
            auxiliary_offset=1,
        ),
        _case(
            "offset-AccumulateViaAdd-partial-REDUCE_COL",
            dim="REDUCE_COL",
            rows=5,
            cols=2,
            algorithm="AccumulateViaAdd",
            partial=3,
            auxiliary_offset=2,
        ),
    ]


def _no_auxiliary_cases() -> list[ReduceCase]:
    cases = []
    for dtype, pools in (("int32", ("SUM", "MAX", "MIN")), ("fp32", ("SUM", "MAX"))):
        for pool in pools:
            for dim in ("REDUCE_ROW", "REDUCE_COL"):
                for calls in (1, 2):
                    cases.append(
                        _case(
                            f"no-auxiliary-{dtype}-{pool}-{dim}-calls{calls}",
                            dim=dim,
                            rows=2,
                            cols=3,
                            batches=2,
                            pool=pool,
                            policy="NoWaitNoPop",
                            calls=calls,
                            input_dtype=dtype,
                            output_dtype=dtype,
                            fp32_mode="Accurate" if dtype == "fp32" else "Fast",
                            no_auxiliary=True,
                        )
                    )
    return cases


def _stride_cases() -> list[ReduceCase]:
    cases = []
    for dim in ("REDUCE_ROW", "REDUCE_COL"):
        for algorithm in ("ReduceTile", "AccumulateViaAdd"):
            for policy in INDEXED_POLICIES:
                cases.append(
                    _case(
                        f"stride-{algorithm}-{dim}-{policy}",
                        dim=dim,
                        rows=3,
                        cols=3,
                        batches=2,
                        algorithm=algorithm,
                        policy=policy,
                        row_padding=1,
                        batch_padding=2,
                    )
                )
    return cases


def _regression_cases() -> list[ReduceCase]:
    cases = []
    for policy in INDEXED_POLICIES:
        for dim in ("REDUCE_ROW", "REDUCE_COL"):
            rows, cols = (2, 1) if dim == "REDUCE_ROW" else (1, 2)
            for to_dest in (False, True):
                cases.append(
                    _case(
                        f"regression-partial-only-SfpuAdd-{dim}-{policy}-to_dest{to_dest}",
                        dim=dim,
                        rows=rows,
                        cols=cols,
                        calls=3,
                        algorithm="AccumulateViaAdd",
                        policy=policy,
                        partial=7,
                        reload="CopySeedSfpuAdd",
                        accumulator_to_dest=to_dest,
                    )
                )
        for rows, cols, row_padding in ((1, 1, 0), (2, 3, 1)):
            cases.append(
                _case(
                    f"regression-scalar-batch-stride-{policy}-{rows}x{cols}",
                    dim="REDUCE_SCALAR",
                    rows=rows,
                    cols=cols,
                    batches=2,
                    policy=policy,
                    row_padding=row_padding,
                    batch_padding=1,
                )
            )
        for dim in DIMS:
            rows, cols, batches = _shape_for_dim(dim)
            cases.append(
                _case(
                    f"regression-one-page-output-{dim}-{policy}",
                    dim=dim,
                    rows=rows,
                    cols=cols,
                    batches=batches,
                    algorithm="AccumulateViaAdd",
                    policy=policy,
                    stream_output=True,
                )
            )
    for policy in POLICIES:
        cases.append(
            _case(
                f"regression-post-exp-REDUCE_COL-{policy}",
                dim="REDUCE_COL",
                rows=2,
                cols=5,
                batches=2,
                calls=2 if policy == "NoWaitNoPop" else 1,
                algorithm="AccumulateViaAdd",
                policy=policy,
                post_exp=True,
            )
        )
    return cases


ALL_CASES = tuple(
    _algorithm_cases()
    + _partial_cases()
    + _within_tile_cases()
    + _reload_cases()
    + _chunk_cases()
    + _auxiliary_offset_cases()
    + _no_auxiliary_cases()
    + _stride_cases()
    + _regression_cases()
)
assert len({case.name for case in ALL_CASES}) == len(ALL_CASES)


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
    return {"bf16": ttnn.bfloat16, "fp32": ttnn.float32, "int32": ttnn.int32, "bf8": ttnn.bfloat8_b}[name]


def _auxiliary_dtype(case: ReduceCase) -> ttnn.DataType:
    return ttnn.float32 if case.input_dtype == "fp32" else ttnn.bfloat16


def _auxiliary_compile_args(tiles: list[tuple[str, int, float]]) -> list[int]:
    args = [len(tiles)]
    for tile_type, valid, value in tiles:
        args += [TILE_TYPE[tile_type], valid, _float_bits(value)]
    return args


def _defines(case: ReduceCase) -> list[tuple[str, str]]:
    auxiliary_cb = "compute_kernel_lib::REDUCE_NO_AUXILIARY_CB" if case.no_auxiliary else str(CB_AUXILIARY)
    defines = [
        ("REDUCE_OP", f"ckernel::PoolType::{case.pool}"),
        ("REDUCE_DIM", f"ckernel::ReduceDim::{case.dim}"),
        ("REDUCE_INPUT_POLICY", f"compute_kernel_lib::ReduceInputPolicy::{case.policy}"),
        ("REDUCE_FP32_MODE", f"ReduceFp32Mode::{case.fp32_mode}"),
        ("REDUCE_ALGORITHM", f"compute_kernel_lib::ReduceAlgorithm::{case.algorithm}"),
        ("REDUCE_WITHIN_TILE", f"compute_kernel_lib::ReduceWithinTile::{case.within_tile}"),
        ("REDUCE_FACTOR", str(case.reduce_factor)),
        ("REDUCE_PARTIAL_MODE", f"compute_kernel_lib::ReducePartialMode::{case.partial_mode}"),
        ("REDUCE_RELOAD_MODE", f"compute_kernel_lib::AccumulateReloadMode::{case.reload}"),
        ("REDUCE_OUTPUT_GROUP", str(case.output_group)),
        ("REDUCE_AUXILIARY_OFFSET", str(case.auxiliary_offset)),
        ("REDUCE_AUXILIARY_CB", auxiliary_cb),
    ]
    if case.post_exp:
        defines.append(("REDUCE_POST_EXP", "1"))
    return defines


def _compute_config(case: ReduceCase) -> ttnn.ComputeConfigDescriptor:
    config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        dst_full_sync_en=False,
    )
    accurate_input = case.input_dtype == "fp32" and case.fp32_mode == "Accurate"
    if accurate_input or case.accumulator_to_dest:
        # Host descriptors use the maximum CB count so this vector covers both Wormhole (32) and Blackhole (64).
        unpack_modes = [ttnn.UnpackToDestMode.Default] * 64
        if accurate_input:
            unpack_modes[CB_INPUT] = ttnn.UnpackToDestMode.UnpackToDestFp32
        if case.calls > 1:
            unpack_modes[CB_ACCUMULATOR] = ttnn.UnpackToDestMode.UnpackToDestFp32
        config.unpack_to_dest_mode = unpack_modes
    return config


def _padding_value(case: ReduceCase) -> float:
    if case.pool == "MAX":
        return 99.0
    if case.pool == "MIN":
        return -99.0
    return 100.0


def _make_logical_chunks(case: ReduceCase) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(20260924)
    shape = (case.batches, case.rows * TILE, case.cols * TILE)
    chunks = []
    for call in range(case.calls):
        if case.input_dtype == "int32":
            chunk = torch.randint(-16, 17, shape, generator=generator, dtype=torch.int32) + call
        elif case.input_dtype == "bf8":
            chunk = torch.randint(1, 4, shape, generator=generator).float()
        elif case.pool == "MAX" and case.partial:
            # Zero is not a MAX identity: keep valid data negative so neither a zero nor an unmasked pad passes.
            chunk = -torch.randint(2, 128, shape, generator=generator).float() / 4
        else:
            chunk = torch.rand(shape, generator=generator, dtype=torch.float32) + 0.125 * call
        if case.input_dtype == "bf16":
            chunk = chunk.to(torch.bfloat16)
        if case.partial:
            chunk[:, case.valid_height :, :] = _padding_value(case)
            chunk[:, :, case.valid_width :] = _padding_value(case)
        chunks.append(chunk)
    return chunks


def _logical_tiles(case: ReduceCase, logical: torch.Tensor) -> torch.Tensor:
    return logical.reshape(case.batches, case.rows, TILE, case.cols, TILE).permute(0, 1, 3, 2, 4).contiguous()


def _physical_tiles(case: ReduceCase, logical: torch.Tensor) -> list[torch.Tensor]:
    """Input tiles in the order the helper consumes them."""
    tiles = _logical_tiles(case, logical)
    poison = torch.full((TILE, TILE), _padding_value(case), dtype=logical.dtype)
    ordered = []
    streams_by_col_group = case.dim == "REDUCE_COL" and case.policy not in INDEXED_POLICIES
    for batch in range(case.batches):
        if streams_by_col_group:
            for col_start in range(0, case.cols, case.col_chunk):
                col_end = min(col_start + case.col_chunk, case.cols)
                for row in range(case.rows):
                    for col in range(col_start, col_end):
                        ordered.append(tiles[batch, row, col])
        else:
            for row in range(case.rows):
                for col in range(case.row_stride):
                    ordered.append(tiles[batch, row, col] if col < case.cols else poison)
        ordered += [poison] * (case.batch_stride - case.rows * case.row_stride)
    return ordered


def _reduce_logical_chunk(case: ReduceCase, chunk: torch.Tensor) -> torch.Tensor:
    wide = torch.int64 if case.input_dtype == "int32" else torch.float64
    if case.within_tile == "Skip":
        # Skip keeps the elementwise cross-tile sum; only the already-reduced lane is meaningful.
        tile_sum = _logical_tiles(case, chunk.to(wide))
        if case.dim == "REDUCE_ROW":
            return tile_sum.sum(dim=2)[..., 0].flatten(1)
        if case.dim == "REDUCE_COL":
            return tile_sum.sum(dim=1)[..., 0, :].flatten(1)
        return tile_sum.sum(dim=(1, 2))[..., 0, 0]

    values = chunk[:, : case.valid_height, : case.valid_width].to(wide)
    axis = {"REDUCE_ROW": -1, "REDUCE_COL": -2, "REDUCE_SCALAR": (-2, -1)}[case.dim]
    if case.pool in ("SUM", "AVG"):
        return values.sum(dim=axis)
    if case.pool == "MAX":
        return values.amax(dim=axis)
    return values.amin(dim=axis)


def _golden(case: ReduceCase, chunks: list[torch.Tensor]) -> torch.Tensor:
    partials = torch.stack([_reduce_logical_chunk(case, chunk) for chunk in chunks])
    if case.pool == "MAX":
        return partials.amax(dim=0)
    if case.pool == "MIN":
        return partials.amin(dim=0)
    golden = partials.sum(dim=0)
    if case.pool == "AVG":
        golden = golden / case.average_divisor
    if case.post_exp:
        golden = torch.exp(-0.01 * golden)
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
    tiles = [tile for chunk in logical_chunks for tile in _physical_tiles(case, chunk)]
    physical_input = torch.cat(tiles, dim=0)
    device_input = ttnn.from_torch(
        physical_input,
        dtype=_ttnn_dtype(case.input_dtype),
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=_sharded_memory_config(tuple(physical_input.shape)),
    )

    output_dtype = _ttnn_dtype(case.output_dtype)
    output_shape = _output_shape(case)
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        output_dtype,
        ttnn.TILE_LAYOUT,
        device,
        _sharded_memory_config(output_shape),
    )

    auxiliary_tiles = case.auxiliary_tiles
    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(CB_INPUT, device_input),
        ttnn.cb_descriptor_from_sharded_tensor(CB_STREAM_DESTINATION if case.stream_output else CB_OUTPUT, output),
    ]
    if case.stream_output:
        cbs.append(_scratch_cb(CB_OUTPUT, output_dtype, 1))
    if case.calls > 1:
        cbs.append(_scratch_cb(CB_ACCUMULATOR, output_dtype, case.output_tiles))

    defines = _defines(case)
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=COMPUTE_KERNEL,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=_single_core(),
            compile_time_args=[case.calls, len(auxiliary_tiles)],
            runtime_args=_runtime_args([case.rows, case.cols, case.batches, case.row_stride, case.batch_stride]),
            defines=defines,
            config=_compute_config(case),
        ),
    ]
    if auxiliary_tiles:
        cbs.append(_scratch_cb(CB_AUXILIARY, _auxiliary_dtype(case), len(auxiliary_tiles)))
    if auxiliary_tiles or case.stream_output:
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=AUXILIARY_KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=_single_core(),
                compile_time_args=_auxiliary_compile_args(auxiliary_tiles),
                runtime_args=_runtime_args([case.output_tiles]),
                defines=[("REDUCE_STREAM_OUTPUT", "1")] if case.stream_output else [],
                config=ttnn.WriterConfigDescriptor(),
            )
        )

    result = ttnn.generic_op(
        [device_input, output],
        ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs),
    )
    actual = _meaningful_output(case, ttnn.to_torch(result))
    return actual, _golden(case, logical_chunks)


def _skip_reason(device, case: ReduceCase) -> str | None:
    if "QUASAR" not in str(device.arch()).upper():
        return None
    if case.uses_sfpu:
        return "The reduce helper rejects SFPU reduce paths on Quasar"
    if case.additive:
        return "AccumulateViaAdd is not validated on Quasar"
    if case.pool == "MAX" and case.dim == "REDUCE_ROW" and case.calls > 1:
        return "The MAX row accumulator reload is not supported on Quasar"
    return None


@pytest.mark.parametrize("case", ALL_CASES, ids=lambda case: case.name)
def test_reduce_explicit_modes(device, case: ReduceCase):
    reason = _skip_reason(device, case)
    if reason:
        pytest.skip(reason)
    if case.uses_sfpu and case.pool == "SUM" and case.dim == "REDUCE_ROW" and case.calls > 1:
        # The LLK reduce pack mask leaves partial sums in the right faces of an SFPU row output, and the
        # accumulator reload folds them in again.
        pytest.xfail("SFPU REDUCE_ROW accumulator carries unmasked right faces until the LLK pack-mask fix lands")

    actual, expected = _run_case(device, case)
    if case.input_dtype == "int32":
        torch.testing.assert_close(actual.to(torch.int64), expected, rtol=0, atol=0, msg=case.name)
    elif (case.pool == "MAX" and case.partial) or case.input_dtype == "bf8":
        torch.testing.assert_close(actual.to(torch.float64), expected.to(torch.float64), rtol=0, atol=0, msg=case.name)
    elif case.post_exp:
        # The unclamped approximate exponential has about 3% relative error in this input range.
        torch.testing.assert_close(actual.to(torch.float64), expected, rtol=0.04, atol=0.001, msg=case.name)
    else:
        torch.testing.assert_close(
            actual.to(torch.float64), expected.to(torch.float64), rtol=0.01, atol=0.01, msg=case.name
        )


@pytest.mark.parametrize("dtype", (ttnn.bfloat16, ttnn.float32))
def test_reduce_auxiliary_tile_patterns_over_dirty_memory(device, dtype):
    """Zero-valued scalers overwrite consumed lanes; masks clear poisoned padding."""
    recipes = [
        ("FirstRow", TILE, 0.25),
        ("FirstRow", TILE, 0.0),
        ("FirstRow", 7, 1.0),
        ("FirstColumn", 7, 1.0),
        ("FirstRowPerFaceRow", 7, 0.25),
        ("FirstRowPerFaceRow", 23, 0.5),
        ("Zero", 0, 0.0),
    ]
    shape = (len(recipes) * TILE, TILE)
    auxiliary = ttnn.from_torch(
        torch.full(shape, 7.0),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=_sharded_memory_config(shape),
    )
    result = ttnn.generic_op(
        [auxiliary, auxiliary],
        ttnn.ProgramDescriptor(
            kernels=[
                ttnn.KernelDescriptor(
                    kernel_source=AUXILIARY_KERNEL,
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=_single_core(),
                    compile_time_args=_auxiliary_compile_args(recipes),
                    config=ttnn.WriterConfigDescriptor(),
                )
            ],
            semaphores=[],
            cbs=[ttnn.cb_descriptor_from_sharded_tensor(CB_AUXILIARY, auxiliary)],
        ),
    )
    tiles = ttnn.to_torch(result).reshape(len(recipes), TILE, TILE).to(torch.float32)
    for tile, value in ((tiles[0], 0.25), (tiles[1], 0.0)):
        torch.testing.assert_close(tile[[0, 16], :], torch.full((2, TILE), value), rtol=0, atol=0)
    expected = torch.zeros((len(recipes) - 2, TILE, TILE))
    expected[0, [0, 16], :7] = 1.0
    expected[1, :7, 0] = 1.0
    expected[2, 0, :7] = 0.25
    expected[2, 0, 16:23] = 0.25
    expected[3, 0, :16] = 0.5
    expected[3, 0, 16:32] = 0.5
    expected[3, 16, :7] = 0.5
    expected[3, 16, 16:23] = 0.5
    torch.testing.assert_close(tiles[2:], expected, rtol=0, atol=0)
