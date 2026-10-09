# Round 3 reduce, issue verdicts: device time of the ops whose kernels reach the kernel_lib reduce helper's block path
# (ROW SUM/AVG and SCALAR with a policy that keeps the tiles resident). Run on farm fm_main (main) and fm_v4 (the branch)
# with variant "dflt" (each op's default compute config), and on fm_main with "lofi" (the same calls at LoFi: the bound of
# any fidelity change of the reduce alone, #58706). Each case runs REPS times after a tracy signpost "<case>#<k>"; k = 0 is
# the warm-up. usage: python -m tracy -r -p --no-web-server -o OUT prof_users.py <variant> [case prefix]
import os as _os, sys as _sys

# 05 (12:35 UTC): the device only under scripts/hwlock.sh (it exports HWLOCK_HELD); the mock cluster (an existing descriptor) needs no lock
if not _os.environ.get("HWLOCK_HELD") and not _os.path.isfile(_os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH", "")):
    _sys.exit("not under hwlock")

import sys

import torch
import ttnn
from tracy import signpost

from tests.ttnn.unit_tests.operations.fused.sharded_test_utils import create_sharded_mem_config

REPS = 5


def cfg(device, variant, approx, fp32):
    # "lofi": the op's own default config (approx mode, fp32 DEST) with only the fidelity lowered to LoFi
    if variant != "lofi":
        return None
    return ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=approx, fp32_dest_acc_en=fp32
    )


def softmax_case(shape, dtype, stable=False):
    def setup(device, variant):
        x = ttnn.from_torch(torch.randn(shape), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
        kw = {"dim": -1, "numeric_stable": stable}
        c = cfg(device, variant, True, False)
        if c is not None:
            kw["compute_kernel_config"] = c
        return lambda: ttnn.softmax(x, **kw), [x]

    return setup


def softmax_sharded_case(b, heads, h, w, dtype, subblock_w=6, op_default=False):
    def setup(device, variant):
        t = torch.randn((b, heads, h, w))
        mem = ttnn.create_sharded_memory_config(
            t.shape,
            core_grid=ttnn.CoreGrid(y=heads, x=b),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )
        pc = ttnn.SoftmaxShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(b, heads), subblock_w=subblock_w, block_h=h // 32, block_w=w // 32
        )
        if op_default:
            c = cfg(device, variant, True, False)
        else:
            c = cfg(device, variant, False, False) or ttnn.init_device_compute_kernel_config(
                device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=False
            )
        x = ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)

        def run():
            ttnn.scale_mask_softmax_in_place(x, program_config=pc, compute_kernel_config=c, numeric_stable=True)

        return run, [x]

    return setup


def norm_sharded_case(h, w, cores_h, cores_w, rms):
    def setup(device, variant):
        mem = create_sharded_mem_config(h, w, cores_h, cores_w, False)
        x = ttnn.from_torch(
            torch.randn((1, 1, h, w)), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem
        )
        pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
            block_h=h // cores_h // 32,
            block_w=w // cores_w // 32,
            subblock_w=1,
            use_welford=False,
            inplace=False,
        )
        kw = {"memory_config": mem, "program_config": pc}
        c = cfg(device, variant, False, True)
        if c is not None:
            kw["compute_kernel_config"] = c
        fn = ttnn.rms_norm if rms else ttnn.layer_norm
        return lambda: fn(x, **kw), [x]

    return setup


def pre_allgather_case(shape, rms):
    def setup(device, variant):
        x = ttnn.from_torch(torch.randn(shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        kw = {"dtype": ttnn.bfloat16}
        c = cfg(device, variant, True, False)
        if c is not None:
            kw["compute_kernel_config"] = c
        fn = ttnn.rms_norm_pre_all_gather if rms else ttnn.layer_norm_pre_all_gather
        return lambda: fn(x, **kw), [x]

    return setup


def moreh_softmax_case(shape, large=False, dim=3):
    def setup(device, variant):
        x = ttnn.from_torch(torch.randn(shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        c = cfg(device, variant, True, False)
        kw = {} if c is None else {"compute_kernel_config": c}
        if large:
            kw["strategy"] = ttnn.operations.moreh.SoftmaxOpParallelizationStrategy.LARGE_W
        return lambda: ttnn.operations.moreh.softmax(x, dim, **kw), [x]

    return setup


def moreh_softmax_backward_case(shape, dim=3):
    def setup(device, variant):
        y = ttnn.from_torch(torch.rand(shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        dy = ttnn.from_torch(torch.randn(shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        c = cfg(device, variant, True, False)
        kw = {} if c is None else {"compute_kernel_config": c}
        return lambda: ttnn.operations.moreh.softmax_backward(y, dy, dim, **kw), [y, dy]

    return setup


def sdpa_decode_case(b, nh, nkv, s, d, k_chunk, fidelity, fp32):
    def setup(device, variant):
        q = ttnn.from_torch(torch.randn(1, b, nh, d), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        k = ttnn.from_torch(torch.randn(b, nkv, s, d), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
        v = ttnn.from_torch(torch.randn(b, nkv, s, d), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
        grid = device.compute_with_storage_grid_size()
        pc = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid, q_chunk_size=0, k_chunk_size=k_chunk, exp_approx_mode=False
        )
        lofi = cfg(device, variant, False, fp32)
        c = lofi or ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=fidelity, math_approx_mode=False, fp32_dest_acc_en=fp32
        )
        pos = [s - 1 - 7 * i for i in range(b)]

        def run():
            return ttnn.transformer.scaled_dot_product_attention_decode(
                q, k, v, cur_pos=pos, is_causal=True, program_config=pc, compute_kernel_config=c
            )

        return run, [q, k, v]

    return setup


def moreh_bias_backward_case(batch, m, n_out, k_in, scalar_bias=True):
    def setup(device, variant):
        def tt(x):
            return ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        dy = tt(torch.randn(batch, m, n_out))
        x = tt(torch.randn(batch, m, k_in))
        w = tt(torch.randn(n_out, k_in))
        bshape = (1, 1) if scalar_bias else (1, n_out)  # a bias row selects the multi-core H kernel
        b = tt(torch.randn(bshape))
        db = tt(torch.zeros(bshape))
        c = cfg(device, variant, True, False)

        def run():
            return ttnn.operations.moreh.linear_backward(
                dy, x, w, are_required_outputs=(False, False, True), bias=b, bias_grad=db, compute_kernel_config=c
            )[2]

        return run, [dy, x, w, b, db]

    return setup


def moreh_sum_case(shape, dim):
    def setup(device, variant):
        x = ttnn.from_torch(torch.randn(shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        c = cfg(device, variant, True, False)
        kw = {} if c is None else {"compute_kernel_config": c}
        return lambda: ttnn.operations.moreh.sum(x, dim=dim, keepdim=True, **kw), [x]

    return setup


def moreh_dot_case(n):
    def setup(device, variant):
        a = ttnn.from_torch(torch.randn(1, 1, 1, n), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        b = ttnn.from_torch(torch.randn(1, 1, 1, n), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        c = cfg(device, variant, True, False)
        kw = {} if c is None else {"compute_kernel_config": c}
        return lambda: ttnn.operations.moreh.dot(a, b, **kw), [a, b]

    return setup


def moreh_softmax_large_h_case(shape):
    def setup(device, variant):
        x = ttnn.from_torch(torch.randn(shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        kw = {"strategy": ttnn.operations.moreh.SoftmaxOpParallelizationStrategy.LARGE_H}
        return lambda: ttnn.operations.moreh.softmax(x, 2, **kw), [x]

    return setup


def pre_allgather_2d_case(shape, rms):
    def setup(device, variant):
        x = ttnn.from_torch(torch.randn(shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        fn = ttnn.rms_norm_pre_all_gather if rms else ttnn.layer_norm_pre_all_gather
        return lambda: fn(x, dtype=ttnn.bfloat16, use_2d_core_grid=True), [x]

    return setup


def post_allgather_case(shape, devices, rms):
    # one device's slice of a norm over `devices` devices, with gathered statistics of that many devices
    def setup(device, variant):
        x = ttnn.from_torch(torch.randn(shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        per = 1 if rms else 2
        st = torch.rand(shape[:-1] + (32 * per * devices,)) + 0.5
        s = ttnn.from_torch(st, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        fn = ttnn.rms_norm_post_all_gather if rms else ttnn.layer_norm_post_all_gather
        return lambda: fn(x, s, epsilon=1e-5, dtype=ttnn.bfloat16), [x, s]

    return setup


def softmax_vit_case():
    def setup(device, variant):
        mem = ttnn.create_sharded_memory_config(
            (224, 224),
            core_grid=ttnn.CoreGrid(y=10, x=11),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        x = ttnn.from_torch(
            torch.randn(10, 11, 224, 224), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem
        )
        pc = ttnn.SoftmaxShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(11, 10), subblock_w=7, block_h=7, block_w=7
        )
        kw = {"program_config": pc}
        c = cfg(device, variant, True, False)
        if c is not None:
            kw["compute_kernel_config"] = c

        def run():
            return ttnn.softmax_in_place(x, **kw)

        return run, [x]

    return setup

# Single-device twin of attn_res_gather_softmax's pass one (compute kernel lines 93-115 and 271-301 at 53d6213d077): per
# row, x*x and x*q (row broadcast) into a Wt-tile buffer, each reduced with the helper (SUM, ROW, BulkWaitBulkPop, scaler
# 1.0) into two fp32 statistics; one row per core, Kimi K3's hidden 7168 over TP 4 (Wt 56) and 20 rows (640 tokens).
TWIN_READER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr auto xa = TensorAccessorArgs<0>();
    constexpr auto qa = TensorAccessorArgs<xa.next_compile_time_args_offset()>();
    constexpr auto sa = TensorAccessorArgs<qa.next_compile_time_args_offset()>();
    const auto x = TensorAccessor(xa, get_arg_val<uint32_t>(0));
    const auto q = TensorAccessor(qa, get_arg_val<uint32_t>(1));
    const auto s = TensorAccessor(sa, get_arg_val<uint32_t>(2));
    const uint32_t wt = get_arg_val<uint32_t>(3);
    const uint32_t row = get_arg_val<uint32_t>(4);
    Noc noc;
    DataflowBuffer d_x(0), d_s(1), d_q(2);
    d_s.reserve_back(1);
    noc.async_read(s, d_s, 4096, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    d_s.push_back(1);
    d_q.reserve_back(wt);
    for (uint32_t i = 0; i < wt; ++i) {
        noc.async_read(q, d_q, 2048, {.page_id = i}, {.offset_bytes = i * 2048});
    }
    noc.async_read_barrier();
    d_q.push_back(wt);
    d_x.reserve_back(wt);
    for (uint32_t i = 0; i < wt; ++i) {
        noc.async_read(x, d_x, 2048, {.page_id = row * wt + i}, {.offset_bytes = i * 2048});
    }
    noc.async_read_barrier();
    d_x.push_back(wt);
}
"""

TWIN_WRITER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const auto out = TensorAccessor(TensorAccessorArgs<0>(), get_arg_val<uint32_t>(0));
    const uint32_t row = get_arg_val<uint32_t>(1);
    Noc noc;
    DataflowBuffer d_st(7);
    for (uint32_t k = 0; k < 2; ++k) {
        d_st.wait_front(1);
        noc.async_write(d_st, out, 4096, {}, {.page_id = 2 * row + k});
        noc.async_write_barrier();
        d_st.pop_front(1);
    }
}
"""

TWIN_COMPUTE = r"""
#ifdef TWIN_BLOCK
#define REDUCE_ROW_BLOCK
#endif
#ifdef TWIN_POW2
#define REDUCE_POW2_SCALER
#endif
#include "api/compute/bcast.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

constexpr uint32_t Wt = get_compile_time_arg_val(0);
constexpr uint32_t cb_x = 0, cb_scaler = 1, cb_q = 2, cb_tmp = 6, cb_stats = 7;

template <typename Init, typename TransformOne>
ALWI void reduce_transformed_row(DataflowBuffer& tmp_buf, Init init, TransformOne transform_one) {
    init();
    tmp_buf.reserve_back(Wt);
    for (uint32_t wt = 0; wt < Wt; ++wt) {
        tile_regs_acquire();
        transform_one(wt);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_tmp, wt);
        tile_regs_release();
    }
    tmp_buf.push_back(Wt);
    compute_kernel_lib::reduce<
        PoolType::SUM,
        ReduceDim::REDUCE_ROW,
        cb_tmp,
        cb_scaler,
        cb_stats,
        compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop>(compute_kernel_lib::ReduceInputBlockShape::row(Wt));
}

void kernel_main() {
    DataflowBuffer x_buf(cb_x), q_buf(cb_q), tmp_buf(cb_tmp), scaler_buf(cb_scaler);
    compute_kernel_hw_startup(cb_x, cb_x, cb_tmp);
    q_buf.wait_front(Wt);
    x_buf.wait_front(Wt);
    reduce_transformed_row(
        tmp_buf,
        [] {
            reconfig_data_format(cb_x, cb_x);
            pack_reconfig_data_format(cb_tmp);
            mul_tiles_init(cb_x, cb_x);
        },
        [](uint32_t wt) { mul_tiles(cb_x, cb_x, wt, wt, 0); });
    reduce_transformed_row(
        tmp_buf,
        [] {
            reconfig_data_format(cb_x, cb_q);
            pack_reconfig_data_format(cb_tmp);
            mul_bcast_rows_init(cb_x, cb_q);
        },
        [](uint32_t wt) { mul_tiles_bcast_rows(cb_x, cb_q, wt, wt, 0); });
    x_buf.pop_front(Wt);
    q_buf.pop_front(Wt);
    scaler_buf.pop_front(1);
}
"""


def attn_res_twin_case(block, wt=56, rows=20, pow2=False):
    def setup(device, variant):
        bf, f32 = ttnn.bfloat16, ttnn.float32
        x = ttnn.from_torch(torch.randn(1, 1, rows * 32, wt * 32), dtype=bf, layout=ttnn.TILE_LAYOUT, device=device)
        q = ttnn.from_torch(torch.randn(1, 1, 32, wt * 32) * 0.05, dtype=bf, layout=ttnn.TILE_LAYOUT, device=device)
        s = ttnn.from_torch(torch.ones(1, 1, 32, 32), dtype=f32, layout=ttnn.TILE_LAYOUT, device=device)
        y = ttnn.from_torch(torch.zeros(1, 1, 32, 2 * rows * 32), dtype=f32, layout=ttnn.TILE_LAYOUT, device=device)
        gx = device.compute_with_storage_grid_size().x
        coords = [ttnn.CoreCoord(i % gx, i // gx) for i in range(rows)]
        cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
        cb = lambda i, pages, dt, size: ttnn.CBDescriptor(
            total_size=pages * size,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=size)],
        )
        cbs = [cb(0, wt, bf, 2048), cb(1, 1, f32, 4096), cb(2, wt, bf, 2048), cb(6, wt, bf, 2048), cb(7, 2, f32, 4096)]
        rd, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for r, c in enumerate(coords):
            rd[c.x][c.y] = [x.buffer_address(), q.buffer_address(), s.buffer_address(), wt, r]
            wr[c.x][c.y] = [y.buffer_address(), r]
        program = ttnn.ProgramDescriptor(
            kernels=[
                ttnn.KernelDescriptor(
                    kernel_source=TWIN_READER,
                    source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                    core_ranges=cores,
                    compile_time_args=ttnn.TensorAccessorArgs(x).get_compile_time_args()
                    + ttnn.TensorAccessorArgs(q).get_compile_time_args()
                    + ttnn.TensorAccessorArgs(s).get_compile_time_args(),
                    runtime_args=rd,
                    config=ttnn.ReaderConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=TWIN_WRITER,
                    source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                    core_ranges=cores,
                    compile_time_args=ttnn.TensorAccessorArgs(y).get_compile_time_args(),
                    runtime_args=wr,
                    config=ttnn.WriterConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=TWIN_COMPUTE,
                    source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                    core_ranges=cores,
                    compile_time_args=[wt],
                    defines=([("TWIN_BLOCK", "1")] if block else []) + ([("TWIN_POW2", "1")] if pow2 else []),
                    runtime_args=[],
                    config=ttnn.ComputeConfigDescriptor(
                        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
                    ),
                ),
            ],
            semaphores=[],
            cbs=cbs,
        )

        def run():
            ttnn.generic_op([x, q, s, y], program)
            return y

        return run, [x, q, s, y]

    return setup


def norm_interleaved_case(shape, rms):
    def setup(device, variant):
        x = ttnn.from_torch(torch.randn(shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        c = cfg(device, variant, True, not rms)
        kw = {} if c is None else {"compute_kernel_config": c}
        fn = ttnn.rms_norm if rms else ttnn.layer_norm
        return lambda: fn(x, **kw), [x]

    return setup


def pre_allgather_sharded_case(width, grid, rms):
    def setup(device, variant):
        x = ttnn.from_torch(torch.randn((1, 1, 32, width)), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        mem = ttnn.create_sharded_memory_config(
            shape=(32, width // (grid[0] * grid[1])),
            core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid[0] - 1, grid[1] - 1))}),
            strategy=ttnn.ShardStrategy.WIDTH,
            use_height_and_width_as_shard_shape=True,
        )
        x = ttnn.to_memory_config(x, memory_config=mem)
        bw = width // (grid[0] * grid[1]) // 32
        pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[grid[0], grid[1]],
            subblock_w=min(bw, 4) if variant == "fp32" else bw,
            block_h=1,
            block_w=bw,
            inplace=False,
        )
        kw = {"program_config": pc}
        c = cfg(device, variant, True, False)
        if variant == "fp32":
            c = ttnn.init_device_compute_kernel_config(
                device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=True, fp32_dest_acc_en=True
            )
        if c is not None:
            kw["compute_kernel_config"] = c
        fn = ttnn.rms_norm_pre_all_gather if rms else ttnn.layer_norm_pre_all_gather
        return lambda: fn(x, **kw), [x]

    return setup


def groupnorm_case(N, C, H, W, groups, out_blocks, cy, cx, dtype=ttnn.bfloat16):
    def setup(device, variant):
        grid = ttnn.CoreGrid(y=cy, x=cx)
        t = torch.rand((N, C, H, W), dtype=torch.bfloat16)
        wgt = torch.rand((C,), dtype=torch.bfloat16)
        bias = torch.rand((C,), dtype=torch.bfloat16)
        rm = ttnn.from_torch(
            t.permute(0, 2, 3, 1).reshape(N, 1, W * H, C),
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        x = ttnn.tilize_with_zero_padding(rm, use_multicore=True)
        [g, b], mask = ttnn.dram_group_norm_params_from_torch([wgt, bias], C, groups, device, core_grid=grid, return_mask=True)
        kw = dict(
            num_groups=groups,
            input_mask=mask,
            weight=g,
            bias=b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            core_grid=grid,
            inplace=False,
            num_out_blocks=out_blocks,
            use_welford=False,
        )
        c = cfg(device, variant, True, dtype == ttnn.float32)
        if c is not None:
            kw["compute_kernel_config"] = c
        return lambda: ttnn.group_norm(x, **kw), [x, rm, t, wgt, bias, groups]

    return setup


def avg_pool_case(n, h, w, c, fidelity, k=2):
    # RT-DETR's ResNet-vd shortcut: avg_pool2d 2x2 stride 2, no padding (scaler 1/4), bf16, TILE in and out; the input
    # depends on the shape only, so the fidelities of one shape see the same data
    def setup(device, variant):
        g = torch.Generator().manual_seed(59143 + h * 7 + c)
        x = torch.randn((n, c, h, w), generator=g).permute(0, 2, 3, 1).reshape(1, 1, n * h * w, c)
        t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        cc = ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=fidelity, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False
        )

        def run():
            return ttnn.avg_pool2d(
                input_tensor=t, batch_size=n, input_h=h, input_w=w, channels=c, kernel_size=[k, k], stride=[k, k],
                padding=[0, 0], ceil_mode=False, count_include_pad=True, divisor_override=None,
                deallocate_input=False, dtype=ttnn.bfloat16, output_layout=ttnn.TILE_LAYOUT, compute_kernel_config=cc,
            )

        return run, [t]

    return setup


def pre_allgather_sharded_fp32_case(width, grid, rms):
    # the sharded pre-allgather with an fp32 DEST: 462b98b45da keeps it out of the exact fidelity (control, main's code)
    inner = pre_allgather_sharded_case(width, grid, rms)

    def setup(device, variant):
        run, keep = inner(device, "fp32")
        return run, keep

    return setup


def avg_pool_div_case(n, h, w, c, div, k=2):
    # avg_pool2d with a divisor_override (0 gives an infinite scaler, which keeps the requested fidelity since 736e4629a73)
    def setup(device, variant):
        g = torch.Generator().manual_seed(59143 + h * 7 + c)
        x = torch.randn((n, c, h, w), generator=g).permute(0, 2, 3, 1).reshape(1, 1, n * h * w, c)
        t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        def run():
            return ttnn.avg_pool2d(
                input_tensor=t, batch_size=n, input_h=h, input_w=w, channels=c, kernel_size=[k, k], stride=[k, k],
                padding=[0, 0], ceil_mode=False, count_include_pad=True, divisor_override=div,
                deallocate_input=False, dtype=ttnn.bfloat16, output_layout=ttnn.TILE_LAYOUT,
            )

        return run, [t]

    return setup


def ctl_add_case(shape):
    # control: an op whose kernels the PR does not change (eltwise add), for the card's drift between the two trees
    def setup(device, variant):
        a = ttnn.from_torch(torch.randn(shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        b = ttnn.from_torch(torch.randn(shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        return lambda: ttnn.add(a, b), [a, b]

    return setup


def pr_softmax_case(shape, dtype, stable):
    # production interleaved softmax (tt_dit T5 / CLIP encoders on Blackhole): HiFi4, fp32 DEST
    def setup(device, variant):
        x = ttnn.from_torch(torch.randn(shape), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
        c = ttnn.init_device_compute_kernel_config(device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True)
        return lambda: ttnn.softmax(x, dim=-1, numeric_stable=stable, compute_kernel_config=c), [x]

    return setup


def pr_vit_softmax_case():
    # ViT-base on Blackhole (ttnn_optimized_sharded_vit_bh.py:88-93, :300): [10, 12, 224, 224] bfp8, height-sharded on 12x10
    def setup(device, variant):
        mem = ttnn.create_sharded_memory_config(
            (224, 224), core_grid=ttnn.CoreGrid(y=10, x=12), strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True,
        )
        x = ttnn.from_torch(torch.randn(10, 12, 224, 224), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
        pc = ttnn.SoftmaxShardedMultiCoreProgramConfig(compute_with_storage_grid_size=(12, 10), subblock_w=7, block_h=7, block_w=7)
        return lambda: ttnn.softmax_in_place(x, program_config=pc), [x]

    return setup


def pr_moreh_softmax_backward_case(shape, precise):
    # tt-train: the composite attention backward (precise: HiFi4, fp32 DEST) and the DeepSeek router backward (op default)
    def setup(device, variant):
        y = ttnn.from_torch(torch.rand(shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        dy = ttnn.from_torch(torch.randn(shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        kw = {}
        if precise:
            kw["compute_kernel_config"] = ttnn.init_device_compute_kernel_config(device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True)
        return lambda: ttnn.operations.moreh.softmax_backward(y, dy, 3, **kw), [y, dy]

    return setup


def pr_norm_case(shape, fn_name, fp32):
    # production interleaved rms_norm / rms_norm_pre_all_gather: fp32 None = the op default, else HiFi4 with that DEST
    def setup(device, variant):
        x = ttnn.from_torch(torch.randn(shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        kw = {"dtype": ttnn.bfloat16} if fn_name == "rms_norm_pre_all_gather" else {}
        if fp32 is not None:
            kw["compute_kernel_config"] = ttnn.init_device_compute_kernel_config(device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=fp32)
        fn = getattr(ttnn, fn_name)
        return lambda: fn(x, **kw), [x]

    return setup


CASES = [
    ("sm_1x8x1024x1024_bf16", softmax_case((1, 8, 1024, 1024), ttnn.bfloat16)),
    ("sm_1x1x2048x2048_bf16", softmax_case((1, 1, 2048, 2048), ttnn.bfloat16)),
    ("sm_1x16x256x256_bfp8", softmax_case((1, 16, 256, 256), ttnn.bfloat8_b)),
    ("sm_8x4x384x384_bf16", softmax_case((8, 4, 384, 384), ttnn.bfloat16)),
    ("sm_1x32x128x128_bf16", softmax_case((1, 32, 128, 128), ttnn.bfloat16)),
    ("smst_1x8x1024x1024_bf16", softmax_case((1, 8, 1024, 1024), ttnn.bfloat16, stable=True)),
    ("smsh_8x4x384x384_bf16", softmax_sharded_case(8, 4, 384, 384, ttnn.bfloat16)),
    ("smsh_8x4x384x384_bfp8", softmax_sharded_case(8, 4, 384, 384, ttnn.bfloat8_b)),
    ("smsh_8x8x128x512_bf16", softmax_sharded_case(8, 8, 128, 512, ttnn.bfloat16, 8, True)),
    ("smsh_8x4x384x1024_bf16", softmax_sharded_case(8, 4, 384, 1024, ttnn.bfloat16, 8, True)),
    ("smsh_8x8x64x256_bf16", softmax_sharded_case(8, 8, 64, 256, ttnn.bfloat16, 8, True)),
    ("ln_sh_1024x4096_8x8", norm_sharded_case(1024, 4096, 8, 8, False)),
    ("rms_sh_1024x4096_8x8", norm_sharded_case(1024, 4096, 8, 8, True)),
    ("ln_sh_256x512_4x4", norm_sharded_case(256, 512, 4, 4, False)),
    ("rms_pre_1x1x1024x4096", pre_allgather_case((1, 1, 1024, 4096), True)),
    ("ln_pre_1x1x1024x4096", pre_allgather_case((1, 1, 1024, 4096), False)),
    ("lnsp_1x1x32x8192_8x4", pre_allgather_sharded_case(8192, (8, 4), False)),
    ("rmssp_1x1x32x8192_8x4", pre_allgather_sharded_case(8192, (8, 4), True)),
    ("msm_1x1x256x1024", moreh_softmax_case((1, 1, 256, 1024))),
    ("msm_1x8x512x512", moreh_softmax_case((1, 8, 512, 512))),
    ("msm_1x1x1024x256", moreh_softmax_case((1, 1, 1024, 256))),
    ("msm_2x4x64x2048", moreh_softmax_case((2, 4, 64, 2048))),
    ("msml_1x1x64x32768", moreh_softmax_case((1, 1, 64, 32768), large=True)),
    ("msmb_1x1x256x1024", moreh_softmax_backward_case((1, 1, 256, 1024))),
    ("msmb_1x8x512x512", moreh_softmax_backward_case((1, 8, 512, 512))),
    ("gn_1x2560x1x1024_g32_ob4", groupnorm_case(1, 2560, 1, 1024, 32, 4, 8, 8)),
    ("gn_8x768x1x512_g32_ob2", groupnorm_case(8, 768, 1, 512, 32, 2, 8, 8)),
    ("gn_1x1920x16x16_g32_ob1", groupnorm_case(1, 1920, 16, 16, 32, 1, 4, 4)),
    ("gn_1x64x192x640_g16_ob10", groupnorm_case(1, 64, 192, 640, 16, 10, 2, 4)),
    ("gnf_1x2560x1x1024_g32_ob4", groupnorm_case(1, 2560, 1, 1024, 32, 4, 8, 8, ttnn.float32)),
    ("gnf_8x768x1x512_g32_ob2", groupnorm_case(8, 768, 1, 512, 32, 2, 8, 8, ttnn.float32)),
    # follow-up of the review (2026-10-07): appended, so the inputs of the cases above stay the same
    ("lnint_1x1x1024x4096", norm_interleaved_case((1, 1, 1024, 4096), False)),
    ("rmsint_1x1x1024x4096", norm_interleaved_case((1, 1, 1024, 4096), True)),
    ("rmsint_1x1x256x8192", norm_interleaved_case((1, 1, 256, 8192), True)),
    ("msmh_1x1x1024x256", moreh_softmax_case((1, 1, 1024, 256), dim=2)),
    ("msmh_1x8x512x512", moreh_softmax_case((1, 8, 512, 512), dim=2)),
    ("msmbh_1x1x1024x256", moreh_softmax_backward_case((1, 1, 1024, 256), dim=2)),
    ("msmbh_1x8x512x512", moreh_softmax_backward_case((1, 8, 512, 512), dim=2)),
    ("gnp4_1x64x64x64_g16_ob2", groupnorm_case(1, 64, 64, 64, 16, 2, 2, 4)),
    ("sdpad_b32_s2048_k512_hifi2fp32", sdpa_decode_case(32, 32, 8, 2048, 128, 512, ttnn.MathFidelity.HiFi2, True)),
    ("sdpad_b32_s2048_k512_hifi2", sdpa_decode_case(32, 32, 8, 2048, 128, 512, ttnn.MathFidelity.HiFi2, False)),
    ("sdpad_b32_s2048_k512_hifi4fp32", sdpa_decode_case(32, 32, 8, 2048, 128, 512, ttnn.MathFidelity.HiFi4, True)),
    ("sdpad_b32_s2048_k256_hifi2fp32", sdpa_decode_case(32, 32, 8, 2048, 128, 256, ttnn.MathFidelity.HiFi2, True)),
    ("smvit_10x11x224x224_bfp8", softmax_vit_case()),
    ("smlt_1x8x32x32768_bf16", softmax_case((1, 8, 32, 32768), ttnn.bfloat16)),
    ("mlbb_8x512x3072", moreh_bias_backward_case(8, 512, 3072, 768)),
    ("mlbh_8x512x3072", moreh_bias_backward_case(8, 512, 3072, 768, scalar_bias=False)),
    ("msumh_1x1x2048x1024", moreh_sum_case((1, 1, 2048, 1024), 2)),
    ("msmhl_1x1x16384x64", moreh_softmax_large_h_case((1, 1, 16384, 64))),
    ("lnlt_1x1x32x65536", norm_interleaved_case((1, 1, 32, 65536), False)),
    ("rmslt_1x1x32x65536", norm_interleaved_case((1, 1, 32, 65536), True)),
    ("mdot_262144", moreh_dot_case(262144)),
    ("rmsp2d_1x1x1024x4096", pre_allgather_2d_case((1, 1, 1024, 4096), True)),
    ("lnpost_1x1x1024x512_d8", post_allgather_case((1, 1, 1024, 512), 8, False)),
    ("rmspost_1x1x1024x512_d8", post_allgather_case((1, 1, 1024, 512), 8, True)),
    ("artwin_tile_w56", attn_res_twin_case(False)),
    ("artwin_block_w56", attn_res_twin_case(True)),
    ("artwp2_tile_w56", attn_res_twin_case(False, pow2=True)),
    ("artwp2_block_w56", attn_res_twin_case(True, pow2=True)),
    ("pool_rtd160_hifi4", avg_pool_case(1, 160, 160, 256, ttnn.MathFidelity.HiFi4)),
    ("pool_rtd160_hifi2", avg_pool_case(1, 160, 160, 256, ttnn.MathFidelity.HiFi2)),
    ("pool_rtd80_hifi4", avg_pool_case(1, 80, 80, 512, ttnn.MathFidelity.HiFi4)),
    ("pool_rtd80_hifi2", avg_pool_case(1, 80, 80, 512, ttnn.MathFidelity.HiFi2)),
    ("pool_rtd40_hifi4", avg_pool_case(1, 40, 40, 1024, ttnn.MathFidelity.HiFi4)),
    ("pool_rtd40_hifi2", avg_pool_case(1, 40, 40, 1024, ttnn.MathFidelity.HiFi2)),
    # Gemma 3's multimodal projector: 64x64 SigLIP tokens of 1152 channels, avg_pool2d 4x4 stride 4 (scaler 1/16)
    ("pool_gem64_hifi4", avg_pool_case(1, 64, 64, 1152, ttnn.MathFidelity.HiFi4, k=4)),
    ("pool_gem64_hifi2", avg_pool_case(1, 64, 64, 1152, ttnn.MathFidelity.HiFi2, k=4)),
    # recheck controls (2026-10-08): no exact fidelity on these, so main's code
    ("ln_sh_1024x3072_8x8", norm_sharded_case(1024, 3072, 8, 8, False)),
    ("rms_sh_1024x3072_8x8", norm_sharded_case(1024, 3072, 8, 8, True)),
    ("lnspf_1x1x32x8192_8x4", pre_allgather_sharded_fp32_case(8192, (8, 4), False)),
    ("rmsspf_1x1x32x8192_8x4", pre_allgather_sharded_fp32_case(8192, (8, 4), True)),
    ("pooldiv0_rtd40", avg_pool_div_case(1, 40, 40, 1024, 0)),
    ("pooldiv3_rtd40", avg_pool_div_case(1, 40, 40, 1024, 3)),
    ("ctl_add_1x1x1024x4096", ctl_add_case((1, 1, 1024, 4096))),
    # production shapes on Blackhole (recheck, 2026-10-08)
    ("pr_t5sm_1x16x512x512", pr_softmax_case((1, 16, 512, 512), ttnn.float32, False)),
    ("pr_clipsm_1x12x77x77", pr_softmax_case((1, 12, 77, 77), ttnn.bfloat16, True)),
    ("pr_msmbw_8x32x256x256", pr_moreh_softmax_backward_case((8, 32, 256, 256), True)),
    ("pr_msmbw_32x1x256x8", pr_moreh_softmax_backward_case((32, 1, 256, 8), False)),
    ("pr_rms_1x1x2048x5376", pr_norm_case((1, 1, 2048, 5376), "rms_norm", True)),
    ("pr_rms_1x1x2048x2880", pr_norm_case((1, 1, 2048, 2880), "rms_norm", None)),
    ("pr_rmspre_1x1x640x1792", pr_norm_case((1, 1, 640, 1792), "rms_norm_pre_all_gather", None)),
    ("pr_rmspre_1x1x4096x1280", pr_norm_case((1, 1, 4096, 1280), "rms_norm_pre_all_gather", True)),
    ("pr_gn_1x128x1024x1024_g32_ob32", groupnorm_case(1, 128, 1024, 1024, 32, 32, 8, 8)),
    ("pr_gn_1x512x256x256_g32_ob4", groupnorm_case(1, 512, 256, 256, 32, 4, 8, 8)),
    ("pr_sdpad_b32_s1024_hifi4fp32", sdpa_decode_case(32, 32, 8, 1024, 128, 0, ttnn.MathFidelity.HiFi4, True)),
    ("pr_sdpad_b32_s1024_hifi2fp32", sdpa_decode_case(32, 32, 8, 1024, 128, 0, ttnn.MathFidelity.HiFi2, True)),
]


def main():
    variant = sys.argv[1]
    prefixes = tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 else ("",)
    device = ttnn.open_device(device_id=0, l1_small_size=32768)
    torch.manual_seed(58707)
    for name, setup in CASES:
        if not name.startswith(prefixes):
            continue
        try:
            run, keep = setup(device, variant)
            for k in range(REPS):
                ttnn.synchronize_device(device)
                signpost(header=f"{name}#{k}")
                y = run()
            ttnn.synchronize_device(device)
            ttnn.ReadDeviceProfiler(device)
            del y, keep
            print(f"CASE {name} ok", flush=True)
        except Exception as e:
            print(f"CASE {name} FAIL {str(e).splitlines()[0][:300]}", flush=True)
    ttnn.close_device(device)


if __name__ == "__main__":
    main()
