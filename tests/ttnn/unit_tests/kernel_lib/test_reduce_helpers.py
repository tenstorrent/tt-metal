# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Direct device-kernel tests for compute_kernel_lib::reduce.

The suite intentionally drives the helper itself through a tiny ProgramDescriptor instead of
testing it indirectly through a migrated operation.  Cases marked xfail are positive specifications
for combinations AccumulateViaAdd does not support yet; an implementation that starts working turns
those cases into strict XPASS failures so the marker must be removed. Auto-dispatch cases exercise
the absence of a size cutoff and every compile-time gate, including the complete AccumulateViaAdd
input-policy support matrix.
"""

from dataclasses import dataclass

import pytest
import torch
import ttnn

TILE = 32
CB_IN = 0
CB_SCALER = 1
CB_ACC = 2
CB_OUT = 16

DIMS = ("row", "col", "scalar")
POOLS = ("sum", "avg", "max", "min")
POLICIES = ("stream", "bulk", "wait_upfront", "no_wait")
RECONFIG_MODES = ("none", "input", "output", "input_and_output")
RELOAD_MODES = ("fold", "copy_pairs", "copy_uniform", "copy_sfpu", "copy_zero")

_DIM_ID = {name: idx for idx, name in enumerate(DIMS)}
_POOL_ID = {name: idx for idx, name in enumerate(POOLS)}
_POLICY_ID = {name: idx for idx, name in enumerate(POLICIES)}
_RECONFIG_ID = {name: idx for idx, name in enumerate(RECONFIG_MODES)}
_ALGORITHM_ID = {"auto": 0, "reduce_tile": 1, "accumulate_via_add": 2}
_RELOAD_ID = {name: idx for idx, name in enumerate(RELOAD_MODES)}


_REDUCE_KERNEL = r"""
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

namespace {
constexpr uint32_t cb_in = 0, cb_scaler = 1, cb_acc = 2, cb_out = 16;

template <
    uint32_t dim_id,
    uint32_t pool_id,
    uint32_t policy_id,
    uint32_t algorithm_id,
    uint32_t mean_id,
    uint32_t reconfig_id,
    uint32_t fp32_mode_id,
    uint32_t accumulate_id,
    uint32_t reload_id>
ALWI void run_reduce(
    compute_kernel_lib::ReduceInputBlockShape shape,
    compute_kernel_lib::ReduceInputMemoryLayout layout,
    compute_kernel_lib::ReducePartialScaler partial,
    uint32_t n_reduced) {
    using namespace compute_kernel_lib;
    using ckernel::PoolType;
    using ckernel::ReduceDim;

    constexpr ReduceDim dim = dim_id == 0u   ? ReduceDim::REDUCE_ROW
                              : dim_id == 1u ? ReduceDim::REDUCE_COL
                                             : ReduceDim::REDUCE_SCALAR;
    constexpr PoolType pool = pool_id == 0u   ? PoolType::SUM
                              : pool_id == 1u ? PoolType::AVG
                              : pool_id == 2u ? PoolType::MAX
                                             : PoolType::MIN;
    constexpr ReduceInputPolicy policy =
        policy_id == 0u   ? ReduceInputPolicy::WaitAndPopPerTile
        : policy_id == 1u ? ReduceInputPolicy::BulkWaitBulkPop
        : policy_id == 2u ? ReduceInputPolicy::WaitUpfrontNoPop
                          : ReduceInputPolicy::NoWaitNoPop;
    constexpr ReduceAlgorithm algorithm =
        algorithm_id == 0u   ? ReduceAlgorithm::Auto
        : algorithm_id == 1u ? ReduceAlgorithm::ReduceTile
                             : ReduceAlgorithm::AccumulateViaAdd;
    constexpr ReduceDataFormatReconfigMode reconfig =
        reconfig_id == 0u   ? ReduceDataFormatReconfigMode::NONE
        : reconfig_id == 1u ? ReduceDataFormatReconfigMode::INPUT
        : reconfig_id == 2u ? ReduceDataFormatReconfigMode::OUTPUT
                            : ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT;
    constexpr ReduceFp32Mode fp32_mode =
        fp32_mode_id == 0u ? ReduceFp32Mode::Fast : ReduceFp32Mode::Accurate;
    constexpr AccumulateReloadMode reload =
        reload_id == 0u   ? AccumulateReloadMode::FoldViaAdd
        : reload_id == 2u ? AccumulateReloadMode::CopySeedUniform
        : reload_id == 3u ? AccumulateReloadMode::CopySeedSfpuAdd
        : reload_id == 4u ? AccumulateReloadMode::CopySeedZeroPair
                          : AccumulateReloadMode::CopySeedPairs;

    const uint32_t row_pitch = layout.row_stride == 0u ? shape.cols : layout.row_stride;
    const uint32_t in_tiles = shape.rows * row_pitch * shape.batches;

    // Streaming/Bulk consume the chunk, so their caller publishes it again before the next call. The no-pop
    // policies intentionally reuse the same resident pages: WaitUpfrontNoPop repeats its own wait, while
    // NoWaitNoPop relies on the single caller-owned wait below and performs no CB synchronization itself.
    auto prepare_input_for_next_call = [&]() {
        if constexpr (
            policy == ReduceInputPolicy::WaitAndPopPerTile || policy == ReduceInputPolicy::BulkWaitBulkPop) {
            cb_reserve_back(cb_in, in_tiles);
            cb_push_back(cb_in, in_tiles);
        }
    };

    if constexpr (policy == ReduceInputPolicy::NoWaitNoPop) {
        cb_wait_front(cb_in, in_tiles);
    }

    if constexpr (mean_id != 0u) {
        if constexpr (accumulate_id != 0u) {
            const uint32_t total_n_reduced = 3u * n_reduced;
            reduce_mean<
                dim,
                cb_in,
                cb_scaler,
                cb_acc,
                policy,
                reconfig,
                fp32_mode,
                algorithm,
                Accumulate>(
                shape,
                total_n_reduced,
                layout,
                Accumulate::at(cb_acc, 0).with_reload(reload),
                partial);
            prepare_input_for_next_call();
            reduce_mean<
                dim,
                cb_in,
                cb_scaler,
                cb_acc,
                policy,
                reconfig,
                fp32_mode,
                algorithm,
                Accumulate>(
                shape,
                total_n_reduced,
                layout,
                Accumulate::at(cb_acc, 1).with_reload(reload),
                partial);
            prepare_input_for_next_call();
            reduce_mean<
                dim,
                cb_in,
                cb_scaler,
                cb_out,
                policy,
                reconfig,
                fp32_mode,
                algorithm,
                Accumulate>(
                shape,
                total_n_reduced,
                layout,
                Accumulate::at_last(cb_acc, 2).with_reload(reload),
                partial);
        } else {
            reduce_mean<
                dim,
                cb_in,
                cb_scaler,
                cb_out,
                policy,
                reconfig,
                fp32_mode,
                algorithm>(shape, n_reduced, layout, partial);
        }
    } else if constexpr (accumulate_id != 0u) {
        reduce<
            pool,
            dim,
            cb_in,
            cb_scaler,
            cb_acc,
            policy,
            reconfig,
            fp32_mode,
            algorithm,
            Accumulate,
            NoOp>(shape, layout, Accumulate::at(cb_acc, 0).with_reload(reload), NoOp{}, partial);
        prepare_input_for_next_call();
        reduce<
            pool,
            dim,
            cb_in,
            cb_scaler,
            cb_acc,
            policy,
            reconfig,
            fp32_mode,
            algorithm,
            Accumulate,
            NoOp>(shape, layout, Accumulate::at(cb_acc, 1).with_reload(reload), NoOp{}, partial);
        prepare_input_for_next_call();
        reduce<
            pool,
            dim,
            cb_in,
            cb_scaler,
            cb_out,
            policy,
            reconfig,
            fp32_mode,
            algorithm,
            Accumulate,
            NoOp>(shape, layout, Accumulate::at_last(cb_acc, 2).with_reload(reload), NoOp{}, partial);
    } else {
        reduce<
            pool,
            dim,
            cb_in,
            cb_scaler,
            cb_out,
            policy,
            reconfig,
            fp32_mode,
            algorithm,
            NoAccumulation,
            NoOp>(shape, layout, NoAccumulation{}, NoOp{}, partial);
    }

    if constexpr (
        policy == ReduceInputPolicy::WaitUpfrontNoPop || policy == ReduceInputPolicy::NoWaitNoPop) {
        // Prove the helper left every persistent page resident, as required by both no-pop policies.
        cb_wait_front(cb_in, in_tiles);
    }
}
}  // namespace

void kernel_main() {
    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t NC = get_compile_time_arg_val(2);
    constexpr uint32_t dim_id = get_compile_time_arg_val(3);
    constexpr uint32_t pool_id = get_compile_time_arg_val(4);
    constexpr uint32_t policy_id = get_compile_time_arg_val(5);
    constexpr uint32_t algorithm_id = get_compile_time_arg_val(6);
    constexpr uint32_t partial_elems = get_compile_time_arg_val(7);
    constexpr uint32_t row_stride = get_compile_time_arg_val(8);
    constexpr uint32_t mean_id = get_compile_time_arg_val(9);
    constexpr uint32_t n_reduced = get_compile_time_arg_val(10);
    constexpr uint32_t reconfig_id = get_compile_time_arg_val(11);
    constexpr uint32_t fp32_mode_id = get_compile_time_arg_val(12);
    constexpr uint32_t partial_mode_id = get_compile_time_arg_val(13);
    constexpr uint32_t mask_tile_idx = get_compile_time_arg_val(14);
    constexpr uint32_t accumulate_id = get_compile_time_arg_val(15);
    constexpr uint32_t reload_id = get_compile_time_arg_val(16);
    constexpr uint32_t row_pitch = row_stride == 0u ? Wt : row_stride;
    constexpr uint32_t in_tiles = Ht * row_pitch * NC;
    using namespace compute_kernel_lib;
    const auto shape = ReduceInputBlockShape::of(Ht, Wt, NC);
    const auto layout = row_stride == 0u ? ReduceInputMemoryLayout::contiguous()
                                         : ReduceInputMemoryLayout::with_row_stride(row_stride);
    const auto partial = partial_mode_id == 1u   ? ReducePartialScaler::last_tile()
                         : partial_mode_id == 2u ? ReducePartialScaler::only_tile()
                         : partial_elems == 0u   ? ReducePartialScaler::none()
                                                 : ReducePartialScaler::partial_mask(
                                                       partial_elems, mask_tile_idx);

    using ckernel::PoolType;
    using ckernel::ReduceDim;
    constexpr ReduceDim dim = dim_id == 0u   ? ReduceDim::REDUCE_ROW
                              : dim_id == 1u ? ReduceDim::REDUCE_COL
                                             : ReduceDim::REDUCE_SCALAR;
    constexpr PoolType pool = pool_id == 0u   ? PoolType::SUM
                              : pool_id == 1u ? PoolType::AVG
                              : pool_id == 2u ? PoolType::MAX
                                             : PoolType::MIN;
    constexpr ReduceFp32Mode fp32_mode =
        fp32_mode_id == 0u ? ReduceFp32Mode::Fast : ReduceFp32Mode::Accurate;
    constexpr DataFormat reduce_format = static_cast<DataFormat>(unpack_src_format[cb_in]);
    constexpr bool is_sfpu = is_sfpu_reduce_path<pool, dim, reduce_format, fp32_mode>();
    constexpr bool swap_reduce_operands = reduce_swaps_operands<pool, dim, is_sfpu>();
    constexpr bool reconfigures_input = reconfig_id == 1u || reconfig_id == 3u;

    if constexpr (algorithm_id == 2u) {
        compute_kernel_hw_startup(cb_in, cb_in, cb_out);
    } else if constexpr (!reconfigures_input && swap_reduce_operands) {
        // NONE / OUTPUT mean a prior operation already left SrcA/SrcB in reduce order.
        compute_kernel_hw_startup(cb_scaler, cb_in, cb_out);
    } else {
        compute_kernel_hw_startup(cb_in, cb_scaler, cb_out);
    }

    cb_reserve_back(cb_in, in_tiles);
    cb_push_back(cb_in, in_tiles);
    run_reduce<dim_id, pool_id, policy_id, algorithm_id, mean_id, reconfig_id, fp32_mode_id, accumulate_id, reload_id>(
        shape, layout, partial, n_reduced);
}
"""


_SCALER_KERNEL = r"""
#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

template <uint32_t dim_id, uint32_t pool_id>
FORCE_INLINE void fill(uint32_t n_reduced) {
    using ckernel::PoolType;
    using ckernel::ReduceDim;
    constexpr uint32_t cb_scaler = 1;
    constexpr ReduceDim dim = dim_id == 0u   ? ReduceDim::REDUCE_ROW
                              : dim_id == 1u ? ReduceDim::REDUCE_COL
                                             : ReduceDim::REDUCE_SCALAR;
    constexpr PoolType pool = pool_id == 0u   ? PoolType::SUM
                              : pool_id == 1u ? PoolType::AVG
                              : pool_id == 2u ? PoolType::MAX
                                             : PoolType::MIN;
    float scaler = 1.0f;
    if constexpr (pool == PoolType::AVG) {
        scaler = dim == ReduceDim::REDUCE_SCALAR
                     ? 1.0f / sqrtf(static_cast<float>(n_reduced))
                     : 1.0f / static_cast<float>(n_reduced);
    }
    dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler, pool, dim>(scaler);
}

void kernel_main() {
    constexpr uint32_t dim_id = get_compile_time_arg_val(0);
    constexpr uint32_t pool_id = get_compile_time_arg_val(1);
    constexpr uint32_t n_reduced = get_compile_time_arg_val(2);
    fill<dim_id, pool_id>(n_reduced);
}
"""


_MASK_KERNEL = r"""
#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    constexpr uint32_t cb_scaler = 1;
    constexpr uint32_t dim_id = get_compile_time_arg_val(0);
    constexpr uint32_t partial_elems = get_compile_time_arg_val(1);
    constexpr uint32_t mask_tile_idx = get_compile_time_arg_val(2);
    using ckernel::ReduceDim;
    for (uint32_t i = 0; i < mask_tile_idx; ++i) {
        dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler, ckernel::PoolType::SUM, ReduceDim::REDUCE_ROW>(
            0.0f, 32);
    }
    if constexpr (dim_id == 0u) {
        dataflow_kernel_lib::prepare_reduce_mask<cb_scaler, ReduceDim::REDUCE_ROW>(partial_elems);
    } else if constexpr (dim_id == 1u) {
        dataflow_kernel_lib::prepare_reduce_mask<cb_scaler, ReduceDim::REDUCE_COL>(partial_elems);
    } else {
        // REDUCE_SCALAR partials are deliberately unsupported. Still provide a tile so builds where
        // runtime ASSERTs are disabled exercise the invalid numeric path instead of deadlocking.
        dataflow_kernel_lib::prepare_reduce_mask<cb_scaler, ReduceDim::REDUCE_ROW>(partial_elems);
    }
}
"""


_ZERO_KERNEL = r"""
#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    constexpr uint32_t cb_scaler = 1;
    dataflow_kernel_lib::prepare_reduce_scaler<
        cb_scaler,
        ckernel::PoolType::SUM,
        ckernel::ReduceDim::REDUCE_ROW>(0.0f);
}
"""


def _single_core():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])


def _sharded_memory_config(shape):
    return ttnn.create_sharded_memory_config(
        shape=shape,
        core_grid=_single_core(),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _scratch_cb(cb_id, dtype, pages=1):
    page_size = ttnn.tile_size(dtype)
    fmt = ttnn.CBFormatDescriptor(buffer_index=cb_id, data_format=dtype, page_size=page_size)
    return ttnn.CBDescriptor(
        total_size=page_size * pages,
        core_ranges=_single_core(),
        format_descriptors=[fmt],
    )


def _input_shape(Ht, Wt, NC, row_stride=0):
    return NC * Ht * TILE, (row_stride or Wt) * TILE


def _output_shape(dim, Ht, Wt, NC):
    if dim == "row":
        return NC * Ht * TILE, TILE
    if dim == "col":
        return TILE, NC * Wt * TILE
    return NC * TILE, TILE


def _output_tiles(dim, Ht, Wt, NC):
    return Ht * NC if dim == "row" else (Wt * NC if dim == "col" else NC)


def _read_values(output, dim):
    tensor = ttnn.to_torch(output).to(torch.float64)
    if dim == "row":
        return tensor[:, 0]
    if dim == "col":
        return tensor[0, :]
    return tensor[::TILE, 0]


def _n_reduced(dim, Ht, Wt, partial_elems=0):
    if partial_elems:
        tiles = Wt if dim == "row" else Ht
        return (tiles - 1) * TILE + partial_elems
    if dim == "row":
        return Wt * TILE
    if dim == "col":
        return Ht * TILE
    return Ht * Wt * TILE * TILE


def _make_input(
    device,
    *,
    dim,
    pool,
    Ht,
    Wt,
    NC,
    dtype,
    partial_elems=0,
    row_stride=0,
):
    torch.manual_seed(2026)
    height, width = _input_shape(Ht, Wt, NC, row_stride)
    data = torch.rand(height, width, dtype=torch.float32)
    logical_width = Wt * TILE

    if row_stride:
        data[:, logical_width:] = 997.0

    if partial_elems and dim in ("row", "col"):
        valid = _n_reduced(dim, Ht, Wt, partial_elems)
        batched = data.view(NC, Ht * TILE, width)
        if dim == "row":
            batched[:, :, valid:logical_width] = 997.0
        else:
            batched[:, valid : Ht * TILE, :logical_width] = 997.0

    if dtype == ttnn.bfloat16:
        host = data.to(torch.bfloat16)
    elif dtype == ttnn.int32:
        host = (data * 8).to(torch.int32)
    else:
        host = data

    tensor = ttnn.from_torch(
        host,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=_sharded_memory_config((height, width)),
    )
    # BFP formats have no exact PyTorch storage analogue. Read back the device-quantized input so every
    # format is compared against the values the kernel actually receives rather than the pre-quantized host data.
    quantized = ttnn.to_torch(tensor).to(torch.float64)
    logical = quantized[:, :logical_width].view(NC, Ht * TILE, logical_width)
    if partial_elems and dim in ("row", "col"):
        valid = _n_reduced(dim, Ht, Wt, partial_elems)
        values = logical[:, :, :valid] if dim == "row" else logical[:, :valid, :]
    else:
        values = logical
    axis = 2 if dim == "row" else (1 if dim == "col" else (1, 2))
    if pool == "max":
        golden = values.amax(dim=axis).reshape(-1)
    elif pool == "min":
        golden = values.amin(dim=axis).reshape(-1)
    else:
        golden = values.sum(dim=axis).reshape(-1)
    return tensor, golden


def _run_reduce(
    device,
    *,
    dim,
    pool="sum",
    policy="bulk",
    algorithm="accumulate_via_add",
    Ht=2,
    Wt=3,
    NC=1,
    input_dtype=ttnn.bfloat16,
    fp32_dest=True,
    partial_elems=0,
    row_stride=0,
    use_mean=False,
    reconfig="input_and_output",
    fp32_mode="fast",
    math_fidelity=ttnn.MathFidelity.HiFi4,
    partial_mode="mask",
    mask_tile_idx=0,
    accumulate=False,
    reload="copy_pairs",
):
    input_tensor, golden = _make_input(
        device,
        dim=dim,
        pool=pool,
        Ht=Ht,
        Wt=Wt,
        NC=NC,
        dtype=input_dtype,
        partial_elems=partial_elems,
        row_stride=row_stride,
    )
    out_shape = _output_shape(dim, Ht, Wt, NC)
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(out_shape)),
        ttnn.int32 if input_dtype == ttnn.int32 else ttnn.float32,
        ttnn.TILE_LAYOUT,
        device,
        _sharded_memory_config(out_shape),
    )

    n_reduced = _n_reduced(dim, Ht, Wt, partial_elems)
    scaler_dtype = (
        ttnn.float32
        if partial_elems and input_dtype == ttnn.float32
        else (ttnn.bfloat16 if partial_elems else (ttnn.float32 if fp32_dest else ttnn.bfloat16))
    )
    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(CB_IN, input_tensor),
        ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, output),
        _scratch_cb(CB_SCALER, scaler_dtype, mask_tile_idx + 1),
    ]
    if accumulate:
        cbs.append(_scratch_cb(CB_ACC, ttnn.float32 if fp32_dest else ttnn.bfloat16, _output_tiles(dim, Ht, Wt, NC)))
    compute = ttnn.KernelDescriptor(
        kernel_source=_REDUCE_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=_single_core(),
        compile_time_args=[
            Ht,
            Wt,
            NC,
            _DIM_ID[dim],
            _POOL_ID[pool],
            _POLICY_ID[policy],
            _ALGORITHM_ID[algorithm],
            partial_elems,
            row_stride,
            int(use_mean),
            n_reduced,
            _RECONFIG_ID[reconfig],
            int(fp32_mode == "accurate"),
            {"mask": 0, "last_tile": 1, "only_tile": 2}[partial_mode],
            mask_tile_idx,
            int(accumulate),
            _RELOAD_ID[reload],
        ],
        config=ttnn.ComputeConfigDescriptor(math_fidelity=math_fidelity, fp32_dest_acc_en=fp32_dest),
    )

    kernels = [compute]
    if accumulate and reload == "copy_zero":
        kernels.insert(
            0,
            ttnn.KernelDescriptor(
                kernel_source=_ZERO_KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=_single_core(),
                compile_time_args=[],
                config=ttnn.ReaderConfigDescriptor(),
            ),
        )
    elif partial_elems:
        kernels.insert(
            0,
            ttnn.KernelDescriptor(
                kernel_source=_MASK_KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=_single_core(),
                compile_time_args=[_DIM_ID[dim], partial_elems, mask_tile_idx],
                config=ttnn.ReaderConfigDescriptor(),
            ),
        )
    elif algorithm != "accumulate_via_add":
        kernels.insert(
            0,
            ttnn.KernelDescriptor(
                kernel_source=_SCALER_KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=_single_core(),
                compile_time_args=[_DIM_ID[dim], _POOL_ID[pool], n_reduced],
                config=ttnn.ReaderConfigDescriptor(),
            ),
        )

    result = ttnn.generic_op([input_tensor, output], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    if accumulate:
        golden = golden * 3
    if pool == "avg" or use_mean:
        golden = golden / (3 * n_reduced if accumulate and use_mean else n_reduced)
    return _read_values(result, dim), golden


def _assert_result(got, expected, *, fp32_dest=True):
    if fp32_dest:
        torch.testing.assert_close(got, expected, rtol=1e-2, atol=5e-2)
    else:
        torch.testing.assert_close(got, expected, rtol=3e-2, atol=2e-1)


def _assert_cross_call_result(got, expected, reload, *, fp32_dest=True):
    if reload == "copy_uniform" and fp32_dest:
        # CopySeedUniform deliberately sends the running FP32 DEST value back through the unpack path once per
        # tile. Its error therefore grows with both the tile count and the number of accumulated chunks; keep
        # the relaxed tolerance local to this explicitly lower-precision reload strategy.
        torch.testing.assert_close(got, expected, rtol=2e-2, atol=5e-2)
    else:
        _assert_result(got, expected, fp32_dest=fp32_dest)


@dataclass(frozen=True)
class PolicyCase:
    dim: str
    pool: str
    policy: str

    @property
    def supported(self):
        return not (self.dim == "col" and self.policy == "stream")

    @property
    def id(self):
        return f"{self.dim}-{self.pool}-{self.policy}"


_POLICY_CASES = [PolicyCase(dim, pool, policy) for dim in DIMS for pool in ("sum", "avg") for policy in POLICIES]


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            case,
            id=case.id,
            marks=(
                pytest.mark.xfail(strict=True, reason="REDUCE_COL cannot consume a row-major per-tile stream")
                if not case.supported
                else ()
            ),
        )
        for case in _POLICY_CASES
    ],
)
def test_accumulate_via_add_all_dims_pools_and_input_policies(device, case):
    """Cartesian coverage of every dimension, supported pool, and ReduceInputPolicy."""
    shape = {"row": (2, 3, 2), "col": (3, 2, 2), "scalar": (2, 3, 2)}[case.dim]
    got, expected = _run_reduce(
        device,
        dim=case.dim,
        pool=case.pool,
        policy=case.policy,
        Ht=shape[0],
        Wt=shape[1],
        NC=shape[2],
    )
    _assert_result(got, expected)


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("pool", ["max", "min"])
@pytest.mark.xfail(strict=True, reason="AccumulateViaAdd does not yet implement MAX/MIN")
def test_accumulate_via_add_all_pool_types_including_unsupported(device, dim, pool):
    got, expected = _run_reduce(device, dim=dim, pool=pool)
    _assert_result(got, expected)


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("fp32_dest", [False, True], ids=["bf16-dest", "fp32-dest"])
@pytest.mark.parametrize(
    "input_dtype",
    [ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat16, ttnn.float32],
    ids=["bfp4-input", "bfp8-input", "bf16-input", "fp32-input"],
)
def test_accumulate_via_add_float_input_and_dest_formats(device, dim, input_dtype, fp32_dest):
    got, expected = _run_reduce(
        device,
        dim=dim,
        Ht=2,
        Wt=3,
        NC=1,
        input_dtype=input_dtype,
        fp32_dest=fp32_dest,
    )
    _assert_result(got, expected, fp32_dest=fp32_dest)


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("reconfig", RECONFIG_MODES)
def test_accumulate_via_add_all_reconfig_modes(device, dim, reconfig):
    got, expected = _run_reduce(device, dim=dim, reconfig=reconfig)
    _assert_result(got, expected)


@pytest.mark.parametrize(
    "dim,Ht,Wt,NC",
    [
        ("row", 1, 1, 1),
        ("row", 2, 2, 1),
        ("row", 2, 3, 2),
        ("row", 1, 17, 1),
        ("col", 1, 2, 1),
        ("col", 2, 2, 1),
        ("col", 3, 2, 2),
        ("col", 4, 17, 1),
        ("scalar", 1, 1, 1),
        ("scalar", 1, 2, 1),
        ("scalar", 1, 3, 2),
        ("scalar", 2, 2, 1),
    ],
)
def test_accumulate_via_add_single_even_odd_wide_multioutput_and_batched_shapes(device, dim, Ht, Wt, NC):
    got, expected = _run_reduce(device, dim=dim, Ht=Ht, Wt=Wt, NC=NC)
    _assert_result(got, expected)


@pytest.mark.parametrize("dim", DIMS)
def test_accumulate_via_add_reduce_mean_matches_direct_avg(device, dim):
    direct, expected = _run_reduce(device, dim=dim, pool="avg")
    explicit, explicit_expected = _run_reduce(device, dim=dim, pool="sum", use_mean=True)
    _assert_result(direct, expected)
    _assert_result(explicit, explicit_expected)
    torch.testing.assert_close(direct, explicit, rtol=0, atol=0)


def test_reduce_algorithm_auto_uses_add_for_small_reduction(device):
    auto, expected = _run_reduce(device, dim="row", pool="sum", algorithm="auto", Ht=2, Wt=3)
    explicit, explicit_expected = _run_reduce(device, dim="row", pool="sum", algorithm="accumulate_via_add", Ht=2, Wt=3)
    _assert_result(auto, expected)
    _assert_result(explicit, explicit_expected)
    torch.testing.assert_close(auto, explicit, rtol=0, atol=0)


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize(
    "reduced_tiles",
    [1, 3, 4],
    ids=["single-tile", "three-tiles", "four-tiles"],
)
def test_reduce_algorithm_auto_dispatch_has_no_size_cutoff_for_every_dimension(device, dim, reduced_tiles):
    Ht, Wt = (1, reduced_tiles) if dim != "col" else (reduced_tiles, 1)
    auto, expected = _run_reduce(
        device,
        dim=dim,
        pool="sum",
        policy="bulk",
        algorithm="auto",
        Ht=Ht,
        Wt=Wt,
    )
    explicit, explicit_expected = _run_reduce(
        device,
        dim=dim,
        pool="sum",
        policy="bulk",
        algorithm="accumulate_via_add",
        Ht=Ht,
        Wt=Wt,
    )
    _assert_result(auto, expected)
    _assert_result(explicit, explicit_expected)
    torch.testing.assert_close(auto, explicit, rtol=0, atol=0)


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("policy", POLICIES)
def test_reduce_algorithm_auto_input_policy_support_matrix(device, dim, policy):
    expected_algorithm = "reduce_tile" if dim == "col" and policy == "stream" else "accumulate_via_add"
    Ht, Wt = (4, 1) if dim == "col" else (1, 4)
    auto, expected = _run_reduce(
        device,
        dim=dim,
        pool="sum",
        policy=policy,
        algorithm="auto",
        Ht=Ht,
        Wt=Wt,
    )
    explicit, explicit_expected = _run_reduce(
        device,
        dim=dim,
        pool="sum",
        policy=policy,
        algorithm=expected_algorithm,
        Ht=Ht,
        Wt=Wt,
    )
    _assert_result(auto, expected)
    _assert_result(explicit, explicit_expected)
    torch.testing.assert_close(auto, explicit, rtol=0, atol=0)


@pytest.mark.parametrize(
    "pool,policy,reconfig,input_dtype,fp32_mode,row_stride,expected_algorithm",
    [
        pytest.param("sum", "bulk", "input_and_output", ttnn.bfloat16, "fast", 0, "accumulate_via_add", id="eligible"),
        pytest.param("avg", "bulk", "input_and_output", ttnn.bfloat16, "fast", 0, "reduce_tile", id="avg"),
        pytest.param("sum", "stream", "input_and_output", ttnn.bfloat16, "fast", 0, "accumulate_via_add", id="stream"),
        pytest.param(
            "sum",
            "wait_upfront",
            "input_and_output",
            ttnn.bfloat16,
            "fast",
            0,
            "accumulate_via_add",
            id="wait-upfront",
        ),
        pytest.param(
            "sum", "no_wait", "input_and_output", ttnn.bfloat16, "fast", 0, "accumulate_via_add", id="no-wait"
        ),
        pytest.param("sum", "bulk", "none", ttnn.bfloat16, "fast", 0, "reduce_tile", id="no-reconfig"),
        pytest.param("sum", "bulk", "output", ttnn.bfloat16, "fast", 0, "reduce_tile", id="output-reconfig-only"),
        pytest.param("sum", "bulk", "input", ttnn.bfloat16, "fast", 0, "accumulate_via_add", id="input-reconfig"),
        pytest.param("sum", "bulk", "input_and_output", ttnn.float32, "accurate", 0, "reduce_tile", id="accurate-fp32"),
        pytest.param("sum", "bulk", "input_and_output", ttnn.int32, "fast", 0, "reduce_tile", id="int32"),
        pytest.param("sum", "bulk", "input_and_output", ttnn.bfloat16, "fast", 9, "reduce_tile", id="padded-layout"),
    ],
)
def test_reduce_algorithm_auto_happy_path_support_matrix(
    device,
    pool,
    policy,
    reconfig,
    input_dtype,
    fp32_mode,
    row_stride,
    expected_algorithm,
):
    auto, expected = _run_reduce(
        device,
        dim="row",
        pool=pool,
        policy=policy,
        algorithm="auto",
        Ht=1,
        Wt=4,
        input_dtype=input_dtype,
        fp32_dest=True,
        fp32_mode=fp32_mode,
        reconfig=reconfig,
        row_stride=row_stride,
    )
    explicit, explicit_expected = _run_reduce(
        device,
        dim="row",
        pool=pool,
        policy=policy,
        algorithm=expected_algorithm,
        Ht=1,
        Wt=4,
        input_dtype=input_dtype,
        fp32_dest=True,
        fp32_mode=fp32_mode,
        reconfig=reconfig,
        row_stride=row_stride,
    )
    _assert_result(auto, expected)
    _assert_result(explicit, explicit_expected)
    torch.testing.assert_close(auto, explicit, rtol=0, atol=0)


@pytest.mark.parametrize(
    "dim,policy",
    [
        pytest.param(
            dim,
            policy,
            id=f"{dim}-{policy}",
            marks=(
                pytest.mark.xfail(
                    strict=True,
                    reason="REDUCE_COL cannot consume a row-major per-tile stream",
                )
                if dim == "col" and policy == "stream"
                else ()
            ),
        )
        for dim in ("row", "col")
        for policy in POLICIES
    ],
)
def test_accumulate_via_add_partial_mask_with_every_input_policy(device, dim, policy):
    Ht, Wt = (2, 3) if dim == "row" else (3, 2)
    got, expected = _run_reduce(
        device,
        dim=dim,
        pool="avg",
        policy=policy,
        Ht=Ht,
        Wt=Wt,
        NC=2,
        partial_elems=17,
    )
    _assert_result(got, expected)


@pytest.mark.parametrize(
    "dim,Ht,Wt,NC,partial_elems",
    [
        ("row", 2, 1, 1, 1),
        ("row", 1, 2, 2, 16),
        ("row", 2, 3, 1, 31),
        ("col", 1, 2, 1, 1),
        ("col", 2, 1, 2, 16),
        ("col", 3, 2, 1, 31),
    ],
)
def test_accumulate_via_add_partial_mask_boundaries_single_odd_even_and_batched(device, dim, Ht, Wt, NC, partial_elems):
    got, expected = _run_reduce(
        device,
        dim=dim,
        pool="sum",
        Ht=Ht,
        Wt=Wt,
        NC=NC,
        partial_elems=partial_elems,
    )
    _assert_result(got, expected)


@pytest.mark.parametrize("dim", ["row", "col"])
@pytest.mark.parametrize("partial_elems", [1, 16, 31])
def test_accumulate_via_add_partial_direct_avg_and_reduce_mean_are_identical(device, dim, partial_elems):
    Ht, Wt = (2, 3) if dim == "row" else (3, 2)
    direct, expected = _run_reduce(
        device,
        dim=dim,
        pool="avg",
        Ht=Ht,
        Wt=Wt,
        partial_elems=partial_elems,
    )
    explicit, explicit_expected = _run_reduce(
        device,
        dim=dim,
        pool="sum",
        Ht=Ht,
        Wt=Wt,
        partial_elems=partial_elems,
        use_mean=True,
    )
    _assert_result(direct, expected)
    _assert_result(explicit, explicit_expected)
    torch.testing.assert_close(direct, explicit, rtol=0, atol=0)


@pytest.mark.parametrize("dim", ["row", "col"])
@pytest.mark.parametrize("mask_tile_idx", [0, 1, 3])
def test_accumulate_via_add_partial_mask_honors_scaler_cb_tile_index(device, dim, mask_tile_idx):
    Ht, Wt = (2, 3) if dim == "row" else (3, 2)
    got, expected = _run_reduce(
        device,
        dim=dim,
        Ht=Ht,
        Wt=Wt,
        partial_elems=17,
        mask_tile_idx=mask_tile_idx,
    )
    _assert_result(got, expected)


@pytest.mark.parametrize("dim,Ht,Wt,NC,row_stride", [("row", 2, 3, 2, 5), ("col", 3, 2, 2, 5)])
@pytest.mark.parametrize("policy", ["bulk", "wait_upfront", "no_wait"])
def test_accumulate_via_add_padded_row_stride_for_every_indexed_policy(device, dim, Ht, Wt, NC, row_stride, policy):
    got, expected = _run_reduce(
        device,
        dim=dim,
        policy=policy,
        Ht=Ht,
        Wt=Wt,
        NC=NC,
        row_stride=row_stride,
    )
    _assert_result(got, expected)


@pytest.mark.parametrize("dim,Ht,Wt,row_stride", [("row", 2, 3, 5), ("col", 3, 2, 5)])
def test_accumulate_via_add_partial_mask_composes_with_row_stride(device, dim, Ht, Wt, row_stride):
    got, expected = _run_reduce(
        device,
        dim=dim,
        pool="avg",
        Ht=Ht,
        Wt=Wt,
        NC=2,
        partial_elems=17,
        row_stride=row_stride,
    )
    _assert_result(got, expected)


@pytest.mark.parametrize("Wt", [8, 16, 17, 24, 32])
@pytest.mark.parametrize("fp32_dest", [False, True], ids=["bf16-dest", "fp32-dest"])
def test_accumulate_via_add_col_has_no_dest_register_chunk_limit(device, Wt, fp32_dest):
    got, expected = _run_reduce(device, dim="col", Ht=3, Wt=Wt, NC=2, fp32_dest=fp32_dest)
    _assert_result(got, expected, fp32_dest=fp32_dest)


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize(
    "fp32_mode",
    [
        pytest.param("fast"),
        pytest.param(
            "accurate",
            marks=pytest.mark.xfail(
                strict=True,
                reason="AccumulateViaAdd does not support ReduceFp32Mode::Accurate",
            ),
        ),
    ],
)
def test_accumulate_via_add_reduce_fp32_mode_support(device, dim, fp32_mode):
    got, expected = _run_reduce(
        device,
        dim=dim,
        input_dtype=ttnn.float32,
        fp32_dest=True,
        fp32_mode=fp32_mode,
    )
    _assert_result(got, expected)


@pytest.mark.parametrize("dim", ["row", "col"])
@pytest.mark.parametrize(
    "math_fidelity",
    [ttnn.MathFidelity.LoFi, ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.HiFi3, ttnn.MathFidelity.HiFi4],
    ids=["lofi", "hifi2", "hifi3", "hifi4"],
)
def test_accumulate_via_add_partial_mask_at_every_math_fidelity(device, dim, math_fidelity):
    Ht, Wt = (2, 3) if dim == "row" else (3, 2)
    got, expected = _run_reduce(
        device,
        dim=dim,
        pool="avg",
        Ht=Ht,
        Wt=Wt,
        partial_elems=17,
        math_fidelity=math_fidelity,
    )
    _assert_result(got, expected)


# Unsupported combinations stay executable and xfailed. Algorithmic restrictions are strict so support
# landing turns them into XPASS failures. Runtime-ASSERT-only contracts are non-strict because release
# kernels may continue into a numerically coincident path after the assertion is compiled out.
@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.xfail(strict=True, reason="AccumulateViaAdd is a floating-point datapath")
def test_accumulate_via_add_int32_inputs_are_specified_but_unsupported(device, dim):
    got, expected = _run_reduce(device, dim=dim, input_dtype=ttnn.int32)
    _assert_result(got, expected)


@pytest.mark.parametrize("reload", RELOAD_MODES)
@pytest.mark.parametrize(
    "dim,Ht,Wt,NC",
    [
        ("row", 2, 3, 2),
        ("col", 4, 2, 1),
        ("scalar", 2, 2, 1),
    ],
)
def test_accumulate_via_add_cross_call_accumulation(device, dim, Ht, Wt, NC, reload):
    got, expected = _run_reduce(
        device,
        dim=dim,
        Ht=Ht,
        Wt=Wt,
        NC=NC,
        accumulate=True,
        reload=reload,
    )
    _assert_cross_call_result(got, expected, reload)


@pytest.mark.parametrize(
    "dim,policy",
    [
        pytest.param(
            dim,
            policy,
            id=f"{dim}-{policy}",
            marks=(
                pytest.mark.xfail(strict=True, reason="REDUCE_COL cannot consume a row-major per-tile stream")
                if dim == "col" and policy == "stream"
                else ()
            ),
        )
        for dim in DIMS
        for policy in POLICIES
    ],
)
@pytest.mark.parametrize("reload", RELOAD_MODES)
@pytest.mark.parametrize("fp32_dest", [False, True], ids=["bf16-dest", "fp32-dest"])
def test_accumulate_via_add_cross_call_every_input_policy(device, dim, policy, reload, fp32_dest):
    Ht, Wt = (4, 2) if dim == "col" else ((2, 4) if dim == "row" else (2, 2))
    got, expected = _run_reduce(
        device,
        dim=dim,
        policy=policy,
        Ht=Ht,
        Wt=Wt,
        NC=2,
        fp32_dest=fp32_dest,
        accumulate=True,
        reload=reload,
    )
    _assert_cross_call_result(got, expected, reload, fp32_dest=fp32_dest)


def test_accumulate_via_add_cross_call_large_streaming_bf16_dest(device):
    got, expected = _run_reduce(
        device,
        dim="row",
        policy="stream",
        Ht=1,
        Wt=81,
        NC=1,
        fp32_dest=False,
        accumulate=True,
    )
    _assert_cross_call_result(got, expected, "copy_pairs", fp32_dest=False)


@pytest.mark.parametrize(
    "dim,Ht,Wt,NC",
    [
        ("row", 2, 3, 1),
        ("col", 3, 2, 1),
        ("scalar", 2, 2, 1),
    ],
)
def test_accumulate_via_add_cross_call_mean_finalizes_once(device, dim, Ht, Wt, NC):
    got, expected = _run_reduce(
        device,
        dim=dim,
        pool="sum",
        Ht=Ht,
        Wt=Wt,
        NC=NC,
        use_mean=True,
        accumulate=True,
    )
    _assert_result(got, expected)


@pytest.mark.parametrize(
    "dim,Ht,Wt,policy",
    [
        pytest.param(
            dim,
            Ht,
            Wt,
            policy,
            id=f"{dim}-{policy}",
            marks=(
                pytest.mark.xfail(strict=True, reason="REDUCE_COL cannot consume a row-major per-tile stream")
                if dim == "col" and policy == "stream"
                else ()
            ),
        )
        for dim, Ht, Wt in [("row", 2, 3), ("col", 3, 2)]
        for policy in POLICIES
    ],
)
def test_accumulate_via_add_cross_call_partial_mask(device, dim, Ht, Wt, policy):
    got, expected = _run_reduce(
        device,
        dim=dim,
        policy=policy,
        Ht=Ht,
        Wt=Wt,
        partial_elems=17,
        accumulate=True,
    )
    _assert_result(got, expected)


@pytest.mark.parametrize(
    "dim,Ht,Wt,row_stride,policy",
    [
        pytest.param(
            dim,
            Ht,
            Wt,
            row_stride,
            policy,
            id=f"{dim}-{policy}",
        )
        for dim, Ht, Wt, row_stride in [("row", 2, 3, 5), ("col", 3, 2, 4)]
        for policy in POLICIES
        # Padded streaming is already covered by its dedicated unsupported-contract test. Executing its
        # cross-call form would violate the CB protocol before pytest could record an expected failure.
        if policy != "stream"
    ],
)
def test_accumulate_via_add_cross_call_padded_row_stride(device, dim, Ht, Wt, row_stride, policy):
    got, expected = _run_reduce(
        device,
        dim=dim,
        policy=policy,
        Ht=Ht,
        Wt=Wt,
        row_stride=row_stride,
        accumulate=True,
    )
    _assert_result(got, expected)


@pytest.mark.parametrize(
    "dim,Ht,Wt,NC",
    [
        ("row", 2, 4, 1),
        # Keep the REDUCE_COL streaming fallback single-output: ReduceTile's separate multi-output
        # cross-call accumulator-page limitation is outside this AccumulateViaAdd dispatch test.
        ("col", 4, 1, 1),
        ("scalar", 2, 2, 1),
    ],
)
@pytest.mark.parametrize("policy", POLICIES)
def test_reduce_algorithm_auto_cross_call_accumulation_uses_add_path(device, dim, Ht, Wt, NC, policy):
    expected_algorithm = "reduce_tile" if dim == "col" and policy == "stream" else "accumulate_via_add"
    auto, expected = _run_reduce(
        device,
        dim=dim,
        policy=policy,
        algorithm="auto",
        Ht=Ht,
        Wt=Wt,
        NC=NC,
        accumulate=True,
    )
    explicit, explicit_expected = _run_reduce(
        device,
        dim=dim,
        policy=policy,
        algorithm=expected_algorithm,
        Ht=Ht,
        Wt=Wt,
        NC=NC,
        accumulate=True,
    )
    _assert_result(auto, expected)
    _assert_result(explicit, explicit_expected)
    torch.testing.assert_close(auto, explicit, rtol=0, atol=0)


@pytest.mark.parametrize("partial_mode", ["last_tile", "only_tile"])
@pytest.mark.xfail(strict=True, reason="AccumulateViaAdd needs a pre-add partial mask, not a ReduceTile scaler")
def test_accumulate_via_add_reduce_tile_partial_descriptors_are_specified_but_unsupported(device, partial_mode):
    got, expected = _run_reduce(
        device,
        dim="row",
        Ht=2,
        Wt=3,
        partial_elems=17,
        partial_mode=partial_mode,
    )
    _assert_result(got, expected)


@pytest.mark.parametrize("partial_elems", [1, 16, 31])
@pytest.mark.xfail(
    strict=False,
    reason="a single mask tile cannot encode a partial scalar corner; validation is a runtime ASSERT",
)
def test_accumulate_via_add_partial_scalar_is_specified_but_unsupported(device, partial_elems):
    got, expected = _run_reduce(
        device,
        dim="scalar",
        Ht=2,
        Wt=3,
        partial_elems=partial_elems,
    )
    _assert_result(got, expected)


@pytest.mark.parametrize("dim", ["row", "col"])
@pytest.mark.xfail(
    strict=False,
    reason="partial_elems must be in [1, 32); this runtime contract is observable only in ASSERT-enabled builds",
)
def test_accumulate_via_add_full_tile_partial_count_is_specified_but_invalid(device, dim):
    Ht, Wt = (2, 3) if dim == "row" else (3, 2)
    got, expected = _run_reduce(
        device,
        dim=dim,
        Ht=Ht,
        Wt=Wt,
        partial_elems=32,
    )
    _assert_result(got, expected)


@pytest.mark.xfail(strict=True, reason="a padded resident layout is incompatible with per-tile streaming")
def test_accumulate_via_add_streaming_padded_row_stride_is_specified_but_unsupported(device):
    got, expected = _run_reduce(
        device,
        dim="row",
        policy="stream",
        Ht=2,
        Wt=3,
        row_stride=5,
    )
    _assert_result(got, expected)


@pytest.mark.xfail(strict=True, reason="REDUCE_SCALAR currently requires a contiguous logical block")
def test_accumulate_via_add_scalar_padded_row_stride_is_specified_but_unsupported(device):
    got, expected = _run_reduce(
        device,
        dim="scalar",
        Ht=2,
        Wt=3,
        row_stride=5,
    )
    _assert_result(got, expected)
