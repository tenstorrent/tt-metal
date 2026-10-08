# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""LOCAL EXPERIMENT: gelu_tanh(gate) * up computed in place on a width-sharded fused gate|up matmul output.

Each core's shard of the fused output holds [gate j, gate j+1, up j, up j+1] tile columns for every tile row (the
weights are interleaved per 2-tile block by interleave_gate_up). The kernel writes [gelu(gate) * up] for its two
columns into a width-sharded output on the same cores: no slices, no separate multiply.
"""

import ttnn

_READER = """
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    constexpr uint32_t in_cb = get_compile_time_arg_val(0);
    constexpr uint32_t n_in = get_compile_time_arg_val(1);
    cb_reserve_back(in_cb, n_in);
    cb_push_back(in_cb, n_in);
}
"""

_WRITER = """
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    constexpr uint32_t out_cb = get_compile_time_arg_val(0);
    constexpr uint32_t n_out = get_compile_time_arg_val(1);
    cb_wait_front(out_cb, n_out);
    cb_pop_front(out_cb, n_out);
}
"""

_COMPUTE = """
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/gelu.h"
void kernel_main() {
    constexpr uint32_t in_cb = get_compile_time_arg_val(0);
    constexpr uint32_t out_cb = get_compile_time_arg_val(1);
    constexpr uint32_t rows = get_compile_time_arg_val(2);
    constexpr uint32_t in_w = get_compile_time_arg_val(3);   // 2 * half
    constexpr uint32_t half = in_w / 2;
    init_sfpu(in_cb, out_cb);
    cb_wait_front(in_cb, rows * in_w);
    cb_reserve_back(out_cb, rows * half);
    // Batches of up to 4 tiles (fp32 DST, half sync): copy the gate tiles, GELU them, multiply by the up tiles.
    constexpr uint32_t total = rows * half;
    constexpr uint32_t batch = 4;
    for (uint32_t b0 = 0; b0 < total; b0 += batch) {
        const uint32_t nb = (total - b0) < batch ? (total - b0) : batch;
        tile_regs_acquire();
        copy_tile_init(in_cb);
        for (uint32_t i = 0; i < nb; ++i) {
            const uint32_t t = b0 + i;
            copy_tile(in_cb, (t / half) * in_w + (t % half), i);
        }
        gelu_tanh_tile_init<true>();
        for (uint32_t i = 0; i < nb; ++i) {
            gelu_tanh_tile<DST_ACCUM_MODE, true>(i);
        }
        binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(in_cb);
        for (uint32_t i = 0; i < nb; ++i) {
            const uint32_t t = b0 + i;
            binary_dest_reuse_tiles<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                in_cb, (t / half) * in_w + half + (t % half), i);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < nb; ++i) {
            pack_tile(i, out_cb);
        }
        tile_regs_release();
    }
    cb_push_back(out_cb, rows * half);
    cb_pop_front(in_cb, rows * in_w);
}
"""


def interleave_gate_up(gate, up, cols_per_core_each=64):
    """[K, N] gate and up -> [K, 2N] with per-core blocks [gate cols | up cols] of cols_per_core_each each."""
    k, n = gate.padded_shape[-2], gate.padded_shape[-1]
    nch = n // cols_per_core_each
    parts = []
    for w in (gate, up):
        rm = ttnn.to_layout(ttnn.typecast(w, ttnn.bfloat16), ttnn.ROW_MAJOR_LAYOUT)
        parts.append(ttnn.reshape(rm, (1, k, nch, cols_per_core_each)))
        rm.deallocate(True)
    cat = ttnn.concat(parts, dim=3)
    for p in parts:
        p.deallocate(True)
    flat = ttnn.reshape(cat, (1, 1, k, 2 * n))
    tiled = ttnn.to_layout(flat, ttnn.TILE_LAYOUT)
    out = ttnn.typecast(tiled, gate.dtype)
    for t in (cat, tiled):
        t.deallocate(True)
    return out


def geglu_shard(fused, mesh_device):
    """fused: width-sharded [.., M, 2N] with [gate | up] halves per shard. Returns width-sharded [.., M, N]."""
    mc = fused.memory_config()
    spec = mc.shard_spec
    sh, sw = spec.shape
    tile = ttnn.TILE_SIZE
    rows, in_w = sh // tile, sw // tile
    out_shape = list(fused.padded_shape)
    out_shape[-1] //= 2
    out_mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(spec.grid, (sh, sw // 2), spec.orientation),
    )
    out = ttnn.empty(out_shape, fused.dtype, ttnn.TILE_LAYOUT, mesh_device, out_mc)
    cores = spec.grid
    in_cb = ttnn.cb_descriptor_from_sharded_tensor(0, fused, core_ranges=cores)
    out_cb = ttnn.cb_descriptor_from_sharded_tensor(16, out, core_ranges=cores)
    reader = ttnn.KernelDescriptor(
        kernel_source=_READER,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=cores,
        compile_time_args=[0, rows * in_w],
        config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.RISCV_1_default),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=_WRITER,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=cores,
        compile_time_args=[16, rows * in_w // 2],
        config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default),
    )
    cc = ttnn.ComputeConfigDescriptor()
    cc.fp32_dest_acc_en = True
    cc.math_fidelity = ttnn.MathFidelity.HiFi4
    compute = ttnn.KernelDescriptor(
        kernel_source=_COMPUTE,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=cores,
        compile_time_args=[0, 16, rows, in_w],
        config=cc,
    )
    pd = ttnn.ProgramDescriptor(cbs=[in_cb, out_cb], kernels=[reader, writer, compute])
    mesh_pd = ttnn.MeshProgramDescriptor()
    rows_m, cols_m = tuple(mesh_device.shape)
    mesh_pd[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(rows_m - 1, cols_m - 1))] = pd
    ttnn.generic_op([fused, out], mesh_pd)
    return out


_WRITER_REMOTE = """
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    constexpr uint32_t out_cb = get_compile_time_arg_val(0);
    constexpr uint32_t rows = get_compile_time_arg_val(1);
    constexpr uint32_t half = get_compile_time_arg_val(2);
    constexpr uint32_t target_w = get_compile_time_arg_val(3);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(4);
    const uint32_t tx = get_arg_val<uint32_t>(0);
    const uint32_t ty = get_arg_val<uint32_t>(1);
    const uint32_t base = get_arg_val<uint32_t>(2);
    const uint32_t col_off = get_arg_val<uint32_t>(3);
    cb_wait_front(out_cb, rows * half);
    uint32_t src = get_read_ptr(out_cb);
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t j = 0; j < half; ++j) {
            const uint32_t dst = base + (r * target_w + col_off + j) * tile_bytes;
            noc_async_write(src, get_noc_addr(tx, ty, dst), tile_bytes);
            src += tile_bytes;
        }
    }
    noc_async_write_barrier();
    cb_pop_front(out_cb, rows * half);
}
"""


def _cores_row_major(grid):
    cores = []
    for r in grid.ranges():
        for y in range(r.start.y, r.end.y + 1):
            for x in range(r.start.x, r.end.x + 1):
                cores.append((x, y))
    return sorted(cores, key=lambda c: (c[1], c[0]))


def geglu_shard_to(fused, mesh_device, target_memcfg):
    """Like geglu_shard, but each core NoC-writes its result straight into a width-sharded target layout (e.g. the
    down projection's input), skipping the reshard."""
    spec = fused.memory_config().shard_spec
    sh, sw = spec.shape
    tile = ttnn.TILE_SIZE
    rows, in_w = sh // tile, sw // tile
    half = in_w // 2
    out_shape = list(fused.padded_shape)
    out_shape[-1] //= 2
    out = ttnn.empty(out_shape, fused.dtype, ttnn.TILE_LAYOUT, mesh_device, target_memcfg)
    tspec = target_memcfg.shard_spec
    target_w = tspec.shape[1] // tile
    tile_bytes = tile * tile * 2  # bf16
    src_cores = _cores_row_major(spec.grid)
    dst_cores = _cores_row_major(tspec.grid)
    base = out.buffer_address()
    cores = spec.grid
    in_cb = ttnn.cb_descriptor_from_sharded_tensor(0, fused, core_ranges=cores)
    out_cb = ttnn.CBDescriptor(
        total_size=rows * half * tile_bytes,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(16, fused.dtype, tile_bytes)],
    )
    reader = ttnn.KernelDescriptor(
        kernel_source=_READER,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=cores,
        compile_time_args=[0, rows * in_w],
        config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.RISCV_1_default),
    )
    wargs = ttnn.RuntimeArgs()
    for c, (x, y) in enumerate(src_cores):
        first = c * half  # first hidden tile column this core produces
        d = first // target_w
        tx, ty = dst_cores[d]
        phys = mesh_device.worker_core_from_logical_core(ttnn.CoreCoord(tx, ty))
        wargs[x][y] = [phys.x, phys.y, base, first % target_w]
    writer = ttnn.KernelDescriptor(
        kernel_source=_WRITER_REMOTE,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=cores,
        compile_time_args=[16, rows, half, target_w, tile_bytes],
        runtime_args=wargs,
        config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default),
    )
    cc = ttnn.ComputeConfigDescriptor()
    cc.fp32_dest_acc_en = True
    cc.math_fidelity = ttnn.MathFidelity.HiFi4
    compute = ttnn.KernelDescriptor(
        kernel_source=_COMPUTE,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=cores,
        compile_time_args=[0, 16, rows, in_w],
        config=cc,
    )
    pd = ttnn.ProgramDescriptor(cbs=[in_cb, out_cb], kernels=[reader, writer, compute])
    mesh_pd = ttnn.MeshProgramDescriptor()
    rows_m, cols_m = tuple(mesh_device.shape)
    mesh_pd[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(rows_m - 1, cols_m - 1))] = pd
    ttnn.generic_op([fused, out], mesh_pd)
    return out
