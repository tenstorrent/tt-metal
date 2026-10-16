# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A @ W on one device as a ProgramDescriptor, on a rectangle of cores another program leaves idle.

The output gate's projection is read only by the gated norm, so it can run beside the SP prefix relay instead of
inside the input projection. Each core of the rectangle owns an [m_block, n_block] tile block of the output and streams
K in k_block-tile blocks: the first core of each row reads that row's A blocks and multicasts them along the row on
NoC 1, the first core of each column reads that column's W blocks and multicasts them down the column on NoC 0
(ttnn.Mcast line pipes, the 2D multicast matmul's NoC split; the other assignment takes 251 us instead of 155 us at the
K3 gate's shape); each reads the next block into a two-slot staging buffer while the current one goes out.
Compute accumulates the K blocks in an FP32 buffer with packer L1 accumulation and packs the result once.
"""

from __future__ import annotations

import ttnn

_CB_A, _CB_W, _CB_ACC, _CB_OUT, _CB_A_STAGE, _CB_W_STAGE = 16, 17, 18, 19, 20, 21

_A_READER = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"

void kernel_main() {
    constexpr uint32_t cb_a = get_compile_time_arg_val(0);
    constexpr uint32_t m_block = get_compile_time_arg_val(1);
    constexpr uint32_t k_block = get_compile_time_arg_val(2);
    constexpr uint32_t k_blocks = get_compile_time_arg_val(3);
    constexpr uint32_t a_row_tiles = get_compile_time_arg_val(4);  // Kt
    constexpr bool sends = get_compile_time_arg_val(5) != 0;
    constexpr uint32_t cb_stage = get_compile_time_arg_val(6);
    constexpr auto a_args = TensorAccessorArgs<7>();
    constexpr auto mc =
        dataflow_kernel_lib::McastArgs<get_named_compile_time_arg_val("a_ct_offset"), get_named_compile_time_arg_val("a_rt_offset")>();
    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t m0 = get_arg_val<uint32_t>(1);
    const uint32_t m_valid = get_arg_val<uint32_t>(2);
    constexpr uint32_t block_tiles = m_block * k_block;
    const uint32_t tile_bytes = get_tile_size(cb_a);
    const auto a = TensorAccessor(a_args, a_addr, tile_bytes);
    Noc noc;
    if constexpr (sends) {
        // The row's first core reads block kb + 1 into the other staging slot while block kb goes out.
        auto pipe = mc.sender(noc);
        cb_reserve_back(cb_stage, 2 * block_tiles);
        const uint32_t stage = get_write_ptr(cb_stage);
        const auto issue = [&](uint32_t kb) {
            const uint32_t slot = stage + (kb & 1) * block_tiles * tile_bytes;
            for (uint32_t r = 0; r < m_valid; ++r) {
                for (uint32_t k = 0; k < k_block; ++k) {
                    noc_async_read_page((m0 + r) * a_row_tiles + kb * k_block + k, a, slot + (r * k_block + k) * tile_bytes);
                }
            }
        };
        issue(0);
        for (uint32_t kb = 0; kb < k_blocks; ++kb) {
            noc_async_read_barrier();
            if (kb + 1 < k_blocks) {
                issue(kb + 1);
            }
            cb_reserve_back(cb_a, block_tiles);
            pipe.send(stage + (kb & 1) * block_tiles * tile_bytes, get_write_ptr(cb_a), block_tiles * tile_bytes);
            cb_push_back(cb_a, block_tiles);
        }
    } else {
        auto pipe = mc.receiver(noc);
        for (uint32_t kb = 0; kb < k_blocks; ++kb) {
            cb_reserve_back(cb_a, block_tiles);
            pipe.receive(kb);
            cb_push_back(cb_a, block_tiles);
        }
    }
}
"""

_W_READER_OUT_WRITER = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"

void kernel_main() {
    constexpr uint32_t cb_w = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t m_block = get_compile_time_arg_val(2);
    constexpr uint32_t n_block = get_compile_time_arg_val(3);
    constexpr uint32_t k_block = get_compile_time_arg_val(4);
    constexpr uint32_t k_blocks = get_compile_time_arg_val(5);
    constexpr uint32_t w_row_tiles = get_compile_time_arg_val(6);    // Nt of W
    constexpr uint32_t out_row_tiles = get_compile_time_arg_val(7);  // Nt of the output
    constexpr bool sends = get_compile_time_arg_val(8) != 0;
    constexpr uint32_t cb_stage = get_compile_time_arg_val(9);
    constexpr auto w_args = TensorAccessorArgs<10>();
    constexpr auto out_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr auto mc =
        dataflow_kernel_lib::McastArgs<get_named_compile_time_arg_val("w_ct_offset"), get_named_compile_time_arg_val("w_rt_offset")>();
    const uint32_t w_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t m0 = get_arg_val<uint32_t>(2);
    const uint32_t m_valid = get_arg_val<uint32_t>(3);
    const uint32_t n0 = get_arg_val<uint32_t>(4);
    const uint32_t n_valid = get_arg_val<uint32_t>(5);
    const uint32_t out_col0 = get_arg_val<uint32_t>(6);
    constexpr uint32_t block_tiles = k_block * n_block;
    const uint32_t w_tile_bytes = get_tile_size(cb_w);
    const auto w = TensorAccessor(w_args, w_addr, w_tile_bytes);
    Noc noc;
    if constexpr (sends) {
        // The column's first core reads block kb + 1 into the other staging slot while block kb goes out.
        auto pipe = mc.sender(noc);
        cb_reserve_back(cb_stage, 2 * block_tiles);
        const uint32_t stage = get_write_ptr(cb_stage);
        const auto issue = [&](uint32_t kb) {
            const uint32_t slot = stage + (kb & 1) * block_tiles * w_tile_bytes;
            for (uint32_t k = 0; k < k_block; ++k) {
                for (uint32_t n = 0; n < n_valid; ++n) {
                    noc_async_read_page((kb * k_block + k) * w_row_tiles + n0 + n, w, slot + (k * n_block + n) * w_tile_bytes);
                }
            }
        };
        issue(0);
        for (uint32_t kb = 0; kb < k_blocks; ++kb) {
            noc_async_read_barrier();
            if (kb + 1 < k_blocks) {
                issue(kb + 1);
            }
            cb_reserve_back(cb_w, block_tiles);
            pipe.send(stage + (kb & 1) * block_tiles * w_tile_bytes, get_write_ptr(cb_w), block_tiles * w_tile_bytes);
            cb_push_back(cb_w, block_tiles);
        }
    } else {
        auto pipe = mc.receiver(noc);
        for (uint32_t kb = 0; kb < k_blocks; ++kb) {
            cb_reserve_back(cb_w, block_tiles);
            pipe.receive(kb);
            cb_push_back(cb_w, block_tiles);
        }
    }
    // The finished block: tile (r, c) at page r * n_block + c.
    const uint32_t out_tile_bytes = get_tile_size(cb_out);
    const auto out = TensorAccessor(out_args, out_addr, out_tile_bytes);
    cb_wait_front(cb_out, m_block * n_block);
    const uint32_t base = get_read_ptr(cb_out);
    for (uint32_t r = 0; r < m_valid; ++r) {
        for (uint32_t c = 0; c < n_valid; ++c) {
            noc_async_write_page((m0 + r) * out_row_tiles + out_col0 + n0 + c, out, base + (r * n_block + c) * out_tile_bytes);
        }
    }
    noc_async_write_barrier();
    cb_pop_front(cb_out, m_block * n_block);
}
"""

_COMPUTE = r"""
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t cb_a = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_acc = get_compile_time_arg_val(2);
    constexpr uint32_t cb_out = get_compile_time_arg_val(3);
    constexpr uint32_t m_block = get_compile_time_arg_val(4);
    constexpr uint32_t n_block = get_compile_time_arg_val(5);
    constexpr uint32_t k_block = get_compile_time_arg_val(6);
    constexpr uint32_t k_blocks = get_compile_time_arg_val(7);
    constexpr uint32_t sub_w = get_compile_time_arg_val(8);  // output subblock: one tile row of sub_w tiles
    constexpr uint32_t out_tiles = m_block * n_block;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_a, cb_w, cb_acc);
    cb_reserve_back(cb_acc, out_tiles);
    for (uint32_t kb = 0; kb < k_blocks; ++kb) {
        cb_wait_front(cb_a, m_block * k_block);
        cb_wait_front(cb_w, k_block * n_block);
        reconfig_data_format<SrcOrder::Reverse>(cb_a, cb_w);
        matmul_block_init(cb_a, cb_w, false, sub_w, 1, k_block);
        // The first K block overwrites the accumulator; later ones add to it in L1.
        pack_reconfig_l1_acc(kb == 0 ? 0 : 1);
        for (uint32_t r = 0; r < m_block; ++r) {
            for (uint32_t c0 = 0; c0 < n_block; c0 += sub_w) {
                tile_regs_acquire();
                for (uint32_t k = 0; k < k_block; ++k) {
                    matmul_block(cb_a, cb_w, r * k_block + k, k * n_block + c0, 0, false, sub_w, 1, k_block);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t t = 0; t < sub_w; ++t) {
                    pack_tile<true>(t, cb_acc, r * n_block + c0 + t);
                }
                tile_regs_release();
            }
        }
        cb_pop_front(cb_a, m_block * k_block);
        cb_pop_front(cb_w, k_block * n_block);
    }
    pack_reconfig_l1_acc(0);
    cb_push_back(cb_acc, out_tiles);

    // FP32 accumulator to the BF16 output.
    cb_wait_front(cb_acc, out_tiles);
    cb_reserve_back(cb_out, out_tiles);
    reconfig_data_format_srca(cb_acc);
    pack_reconfig_data_format(cb_out);
    copy_init(cb_acc);
    for (uint32_t t = 0; t < out_tiles; ++t) {
        tile_regs_acquire();
        copy_tile(cb_acc, t, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out, t);
        tile_regs_release();
    }
    cb_push_back(cb_out, out_tiles);
    cb_pop_front(cb_acc, out_tiles);
}
"""


def _line(device, rect_set, line, noc):
    size = rect_set.bounding_box().grid_size()
    return ttnn.Mcast(
        device,
        ttnn.McastConfig(noc=noc, handshake=True),
        rect_set,
        size.x if line == "row" else size.y,
        ttnn.McastFixedSenderConfig(sender_index=0, placement=ttnn.McastSenderPlacement.Uniform),
        ttnn.McastCoreOrder.RowMajor if line == "row" else ttnn.McastCoreOrder.ColumnMajor,
    )


def _cset(cores):
    return ttnn.CoreRangeSet([]).merge(ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores]))


class GateMatmulProgram:
    """A @ W on the rectangle [x0, x1] x [y0, y1] of every device of a mesh (no communication)."""

    def __init__(
        self, mesh_device, *, rows: tuple[int, int], cols: tuple[int, int], k_block: int, math_fidelity, depth: int = 2
    ):
        self.mesh_device = mesh_device
        self.y0, self.y1 = rows
        self.x0, self.x1 = cols
        self.k_block = k_block
        # Operand blocks each core's A and W circular buffers hold.
        self.depth = depth
        self.compute_config = ttnn.ComputeConfigDescriptor(math_fidelity=math_fidelity, fp32_dest_acc_en=True)

    @property
    def cores(self) -> list[ttnn.CoreCoord]:
        return [ttnn.CoreCoord(x, y) for y in range(self.y0, self.y1 + 1) for x in range(self.x0, self.x1 + 1)]

    def __call__(self, a: ttnn.Tensor, w: ttnn.Tensor, output: ttnn.Tensor, *, output_column: int = 0):
        """Return (io_tensors, MeshProgramDescriptor) writing a @ w into output's columns from output_column."""
        Mt, Kt = a.padded_shape[-2] // 32, a.padded_shape[-1] // 32
        Nt = w.padded_shape[-1] // 32
        rows, cols = self.y1 - self.y0 + 1, self.x1 - self.x0 + 1
        m_block = -(-Mt // rows)
        n_block = -(-Nt // cols)
        k_block = self.k_block
        assert Kt % k_block == 0, f"K={Kt} tiles must split into {k_block}-tile blocks"
        k_blocks = Kt // k_block
        sub_w = max(s for s in (4, 2, 1) if n_block % s == 0)
        out_row_tiles = output.padded_shape[-1] // 32
        rect = ttnn.CoreRange(ttnn.CoreCoord(self.x0, self.y0), ttnn.CoreCoord(self.x1, self.y1))
        rect_set = ttnn.CoreRangeSet([rect])
        a_tile = 2048 if a.dtype == ttnn.bfloat16 else 4096
        w_tile = 1088 if w.dtype == ttnn.bfloat8_b else (2048 if w.dtype == ttnn.bfloat16 else 4096)
        desc = ttnn.MeshProgramDescriptor()
        mesh_rows, mesh_cols = tuple(self.mesh_device.shape)
        for mr in range(mesh_rows):
            for mc in range(mesh_cols):
                program = ttnn.ProgramDescriptor()
                program.cbs = [
                    ttnn.CBDescriptor(
                        total_size=self.depth * m_block * k_block * a_tile,
                        core_ranges=rect_set,
                        format_descriptors=[
                            ttnn.CBFormatDescriptor(buffer_index=_CB_A, data_format=a.dtype, page_size=a_tile)
                        ],
                    ),
                    ttnn.CBDescriptor(
                        total_size=self.depth * k_block * n_block * w_tile,
                        core_ranges=rect_set,
                        format_descriptors=[
                            ttnn.CBFormatDescriptor(buffer_index=_CB_W, data_format=w.dtype, page_size=w_tile)
                        ],
                    ),
                    ttnn.CBDescriptor(
                        total_size=m_block * n_block * 4096,
                        core_ranges=rect_set,
                        format_descriptors=[
                            ttnn.CBFormatDescriptor(buffer_index=_CB_ACC, data_format=ttnn.float32, page_size=4096)
                        ],
                    ),
                    ttnn.CBDescriptor(
                        total_size=m_block * n_block * 2048,
                        core_ranges=rect_set,
                        format_descriptors=[
                            ttnn.CBFormatDescriptor(buffer_index=_CB_OUT, data_format=ttnn.bfloat16, page_size=2048)
                        ],
                    ),
                ]
                a_ct = list(ttnn.TensorAccessorArgs(a).get_compile_time_args())
                w_ct = list(ttnn.TensorAccessorArgs(w).get_compile_time_args())
                out_ct = list(ttnn.TensorAccessorArgs(output).get_compile_time_args())
                a_rt, w_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
                for c in self.cores:
                    m0 = (c.y - self.y0) * m_block
                    n0 = (c.x - self.x0) * n_block
                    m_valid = max(0, min(m_block, Mt - m0))
                    a_rt[c.x][c.y] = [a.buffer_address(), m0, m_valid]
                    w_rt[c.x][c.y] = [
                        w.buffer_address(),
                        output.buffer_address(),
                        m0,
                        m_valid,
                        n0,
                        max(0, min(n_block, Nt - n0)),
                        output_column // 32,
                    ]
                a_senders = [c for c in self.cores if c.x == self.x0]
                w_senders = [c for c in self.cores if c.y == self.y0]
                a_kernels, w_kernels = [], []
                for sends, cores in ((1, a_senders), (0, [c for c in self.cores if c.x != self.x0])):
                    rt = ttnn.RuntimeArgs()
                    for c in cores:
                        rt[c.x][c.y] = a_rt[c.x][c.y]
                    a_kernels.append(
                        ttnn.KernelDescriptor(
                            kernel_source=_A_READER,
                            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                            core_ranges=_cset(cores),
                            compile_time_args=[_CB_A, m_block, k_block, k_blocks, Kt, sends, _CB_A_STAGE] + a_ct,
                            runtime_args=rt,
                            config=ttnn.DataMovementConfigDescriptor(
                                processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_1
                            ),
                        )
                    )
                for sends, cores in ((1, w_senders), (0, [c for c in self.cores if c.y != self.y0])):
                    rt = ttnn.RuntimeArgs()
                    for c in cores:
                        rt[c.x][c.y] = w_rt[c.x][c.y]
                    w_kernels.append(
                        ttnn.KernelDescriptor(
                            kernel_source=_W_READER_OUT_WRITER,
                            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                            core_ranges=_cset(cores),
                            compile_time_args=[_CB_W, _CB_OUT, m_block, n_block, k_block, k_blocks, Nt, out_row_tiles]
                            + [sends, _CB_W_STAGE]
                            + w_ct
                            + out_ct,
                            runtime_args=rt,
                            config=ttnn.DataMovementConfigDescriptor(
                                processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0
                            ),
                        )
                    )
                program.cbs = list(program.cbs) + [
                    ttnn.CBDescriptor(
                        total_size=2 * m_block * k_block * a_tile,
                        core_ranges=_cset(a_senders),
                        format_descriptors=[
                            ttnn.CBFormatDescriptor(buffer_index=_CB_A_STAGE, data_format=a.dtype, page_size=a_tile)
                        ],
                    ),
                    ttnn.CBDescriptor(
                        total_size=2 * k_block * n_block * w_tile,
                        core_ranges=_cset(w_senders),
                        format_descriptors=[
                            ttnn.CBFormatDescriptor(buffer_index=_CB_W_STAGE, data_format=w.dtype, page_size=w_tile)
                        ],
                    ),
                ]
                compute = ttnn.KernelDescriptor(
                    kernel_source=_COMPUTE,
                    source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                    core_ranges=rect_set,
                    compile_time_args=[_CB_A, _CB_W, _CB_ACC, _CB_OUT, m_block, n_block, k_block, k_blocks, sub_w],
                    runtime_args=ttnn.RuntimeArgs(),
                    config=self.compute_config,
                )
                # The multicast lines append their arguments and semaphores to the kernels before the program takes
                # (copies of) them.
                mc_a = _line(self.mesh_device, rect_set, "row", ttnn.NOC.NOC_1)
                mc_w = _line(self.mesh_device, rect_set, "col", ttnn.NOC.NOC_0)
                mc_a.attach(program, "a", a_kernels, len(program.semaphores))
                mc_w.attach(program, "w", w_kernels, mc_a.next_semaphore_id())
                program.kernels = a_kernels + w_kernels + [compute]
                desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(mr, mc), ttnn.MeshCoordinate(mr, mc))] = program
        return [a, w, output], desc
