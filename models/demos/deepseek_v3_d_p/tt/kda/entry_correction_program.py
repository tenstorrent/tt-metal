# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The chunk outputs and the rank's final state from the summary's emitted terms, once the SP prefix is known.

The recurrence is affine in its seed, so the summary (summarize_chunk_recurrence with emit_outputs) leaves, per chunk,
its output for a zero seed F(0) and for an identity seed F(I), and the last segment's FP32 transform (A, B). With the
entry states from the SP prefix every chunk is independent:

    output_c = F0_c + (FI_c - F0_c) @ entry          final = A @ entry + B

where entry is the group's entry state before the chronological reset chunk and the tail's entry state from it on (the
seeds recurrent_chunk_scan restarts from). This replaces the second sequential pass over the chunks.

Each head's chunks are spread over `cores_per_head` cores, which also form one tile row of the final state each. The
emitted terms are sharded so each core's chunk rows are in its own L1; the head's first core reads the entry states
and writes them to the head's other cores. The products accumulate in the FP32 DST on top of the copied F(0) (or B),
at the scan's fidelity.
"""

from __future__ import annotations

import ttnn

_CHRONOLOGY = '#include "ttnn/cpp/ttnn/operations/experimental/kda/device/kernels/chronology.hpp"\n'

_CB_SEED, _CB_TAIL, _CB_M, _CB_BASE, _CB_OUT, _CB_A, _CB_B, _CB_FINAL, _CB_CTL, _CB_CTL_WRITER, _CB_DIFF = range(11)

_READER = (
    _CHRONOLOGY
    + r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_seed = get_compile_time_arg_val(0);
    constexpr uint32_t cb_tail = get_compile_time_arg_val(1);
    constexpr uint32_t cb_m = get_compile_time_arg_val(2);
    constexpr uint32_t cb_base = get_compile_time_arg_val(3);
    constexpr uint32_t cb_a = get_compile_time_arg_val(4);
    constexpr uint32_t cb_b = get_compile_time_arg_val(5);
    constexpr uint32_t cb_ctl = get_compile_time_arg_val(6);
    constexpr uint32_t sp_rank = get_compile_time_arg_val(7);
    constexpr uint32_t sp_size = get_compile_time_arg_val(8);
    constexpr uint32_t local_rows = get_compile_time_arg_val(9);
    constexpr uint32_t num_chunks = get_compile_time_arg_val(10);
    constexpr uint32_t kt = get_compile_time_arg_val(11);
    constexpr uint32_t has_end = get_compile_time_arg_val(12);
    constexpr uint32_t cb_ctl_writer = get_compile_time_arg_val(13);
    constexpr uint32_t final_stride = get_compile_time_arg_val(15);  // the head's cores: rows of the final state
    constexpr auto base_args = TensorAccessorArgs<16>();
    constexpr auto m_args = TensorAccessorArgs<base_args.next_compile_time_args_offset()>();
    constexpr auto a_args = TensorAccessorArgs<m_args.next_compile_time_args_offset()>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    constexpr auto entry_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
    constexpr auto tail_args = TensorAccessorArgs<entry_args.next_compile_time_args_offset()>();
    constexpr auto start_args = TensorAccessorArgs<tail_args.next_compile_time_args_offset()>();
    constexpr auto end_args = TensorAccessorArgs<start_args.next_compile_time_args_offset()>();
    constexpr uint32_t state_tiles = kt * kt;

    const uint32_t base_addr = get_arg_val<uint32_t>(0);
    const uint32_t m_addr = get_arg_val<uint32_t>(1);
    const uint32_t a_addr = get_arg_val<uint32_t>(2);
    const uint32_t b_addr = get_arg_val<uint32_t>(3);
    const uint32_t entry_addr = get_arg_val<uint32_t>(4);
    const uint32_t tail_addr = get_arg_val<uint32_t>(5);
    const uint32_t start_addr = get_arg_val<uint32_t>(6);
    const uint32_t end_addr = get_arg_val<uint32_t>(7);
    const uint32_t head = get_arg_val<uint32_t>(8);
    const uint32_t first_chunk = get_arg_val<uint32_t>(9);
    const uint32_t last_chunk = get_arg_val<uint32_t>(10);
    // The first tile row of the final state this core forms (kt: none); it forms every final_stride-th row on.
    const uint32_t final_column = get_arg_val<uint32_t>(11);
    // The head's first core reads its seeds and writes them to the head's other cores (NoC x | y << 16 each).
    const uint32_t seed_sender = get_arg_val<uint32_t>(12);
    const uint32_t mcast_dests = get_arg_val<uint32_t>(13);
    const uint32_t seed_sem = get_semaphore(get_compile_time_arg_val(14));
    const uint32_t owns_final = final_column < kt;

    // The chronology: valid chunks and the reset chunk (where the tail's seed takes over).
    cb_reserve_back(cb_ctl, 1);
    volatile tt_l1_ptr uint32_t* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_ctl));
    const auto start = TensorAccessor(start_args, start_addr);
    noc_async_read(start.get_noc_addr(0), get_write_ptr(cb_ctl), sizeof(uint32_t));
    noc_async_read_barrier();
    const uint32_t actual_start = words[0];
    kda_chronology::Topology t;
    if constexpr (has_end) {
        const auto end = TensorAccessor(end_args, end_addr);
        noc_async_read(end.get_noc_addr(0), get_write_ptr(cb_ctl), sizeof(uint32_t));
        noc_async_read_barrier();
        t = kda_chronology::derive_interval(actual_start, words[0], sp_rank, sp_size, local_rows);
    } else {
        t = kda_chronology::derive(actual_start, sp_rank, sp_size, local_rows);
    }
    const uint32_t valid = t.valid_chunks(0, 1);
    const uint32_t reset = t.reset_chunk(0, 1);
    const uint32_t end_chunk = last_chunk < valid ? last_chunk : valid;
    const uint32_t chunks = end_chunk > first_chunk ? end_chunk - first_chunk : 0;
    // Chunks before the reset take the group's seed, the rest the tail's.
    uint32_t group_chunks = chunks;
    if (reset != 0) {
        group_chunks = reset <= first_chunk ? 0 : (reset - first_chunk < chunks ? reset - first_chunk : chunks);
    }
    const uint32_t final = owns_final && valid != 0;
    const uint32_t final_tail = final && reset != 0;
    // Every core of the head takes the group's seed, and the tail's after a reset: the head's seeds are multicast.
    const uint32_t need_group = valid != 0;
    const uint32_t need_tail = valid != 0 && reset != 0;
    words[0] = chunks;
    words[1] = group_chunks;
    words[2] = final;
    words[3] = final_tail;
    words[4] = need_group;
    words[5] = need_tail;
    words[6] = final_column;
    words[7] = final_stride;
    cb_reserve_back(cb_ctl_writer, 1);
    volatile tt_l1_ptr uint32_t* writer_words =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_ctl_writer));
    writer_words[0] = chunks;
    writer_words[1] = final;
    writer_words[2] = final_column;
    writer_words[3] = final_stride;
    cb_push_back(cb_ctl_writer, 1);
    cb_push_back(cb_ctl, 1);

    const uint32_t seeds = need_group + need_tail;
    if (seed_sender) {
        const auto send = [&](uint32_t cb, uint32_t addr, const auto& args) {
            cb_reserve_back(cb, state_tiles);
            const uint32_t local = get_write_ptr(cb);
            const auto states = TensorAccessor(args, addr);
            const uint32_t tile = get_tile_size(cb);
            for (uint32_t i = 0; i < state_tiles; ++i) {
                noc_async_read_page(head * state_tiles + i, states, local + i * tile);
            }
            noc_async_read_barrier();
            for (uint32_t d = 0; d < mcast_dests; ++d) {
                const uint32_t xy = get_arg_val<uint32_t>(14 + d);
                noc_async_write(local, get_noc_addr(xy & 0xFFFF, xy >> 16, local), state_tiles * tile);
            }
            cb_push_back(cb, state_tiles);
        };
        if (need_group) {
            send(cb_seed, entry_addr, entry_args);
        }
        if (need_tail) {
            send(cb_tail, tail_addr, tail_args);
        }
        if (seeds != 0) {
            noc_async_write_barrier();
            for (uint32_t d = 0; d < mcast_dests; ++d) {
                const uint32_t xy = get_arg_val<uint32_t>(14 + d);
                noc_semaphore_inc(get_noc_addr(xy & 0xFFFF, xy >> 16, seed_sem), seeds);
            }
            noc_async_atomic_barrier();
        }
    } else if (seeds != 0) {
        // The seeds land at the same buffer addresses on every core of the program.
        volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(seed_sem);
        noc_semaphore_wait_min(sem, seeds);
        if (need_group) {
            cb_reserve_back(cb_seed, state_tiles);
            cb_push_back(cb_seed, state_tiles);
        }
        if (need_tail) {
            cb_reserve_back(cb_tail, state_tiles);
            cb_push_back(cb_tail, state_tiles);
        }
    }
    const auto base = TensorAccessor(base_args, base_addr);
    const auto m = TensorAccessor(m_args, m_addr);
    const uint32_t base_tile = get_tile_size(cb_base);
    const uint32_t m_tile = get_tile_size(cb_m);
    // This core's chunk rows: in place in its shard of the emitted terms, else read under one barrier.
    if (chunks != 0) {
        cb_reserve_back(cb_m, chunks * kt);
        cb_reserve_back(cb_base, chunks * kt);
#ifndef LOCAL_TERMS
        for (uint32_t i = 0; i < chunks * kt; ++i) {
            const uint32_t page = (head * num_chunks + first_chunk) * kt + i;
            noc_async_read_page(page, m, get_write_ptr(cb_m) + i * m_tile);
            noc_async_read_page(page, base, get_write_ptr(cb_base) + i * base_tile);
        }
#endif
        noc_async_read_barrier();
        cb_push_back(cb_m, chunks * kt);
        cb_push_back(cb_base, chunks * kt);
    }
    if (final) {
        // Tile rows of A and of B: each row of the final state is A[row] @ seed + B[row].
        const auto a = TensorAccessor(a_args, a_addr);
        const auto b = TensorAccessor(b_args, b_addr);
        const uint32_t tile = get_tile_size(cb_b);
        for (uint32_t row = final_column; row < kt; row += final_stride) {
            cb_reserve_back(cb_a, kt);
            cb_reserve_back(cb_b, kt);
            for (uint32_t c = 0; c < kt; ++c) {
                const uint32_t page = head * state_tiles + row * kt + c;
                noc_async_read_page(page, a, get_write_ptr(cb_a) + c * tile);
                noc_async_read_page(page, b, get_write_ptr(cb_b) + c * tile);
            }
            noc_async_read_barrier();
            cb_push_back(cb_a, kt);
            cb_push_back(cb_b, kt);
        }
    }
}
"""
)

_COMPUTE = r"""
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t cb_seed = get_compile_time_arg_val(0);
    constexpr uint32_t cb_tail = get_compile_time_arg_val(1);
    constexpr uint32_t cb_m = get_compile_time_arg_val(2);
    constexpr uint32_t cb_base = get_compile_time_arg_val(3);
    constexpr uint32_t cb_out = get_compile_time_arg_val(4);
    constexpr uint32_t cb_a = get_compile_time_arg_val(5);
    constexpr uint32_t cb_b = get_compile_time_arg_val(6);
    constexpr uint32_t cb_final = get_compile_time_arg_val(7);
    constexpr uint32_t cb_ctl = get_compile_time_arg_val(8);
    constexpr uint32_t kt = get_compile_time_arg_val(9);
    constexpr uint32_t cb_diff = get_compile_time_arg_val(10);
    constexpr uint32_t state_tiles = kt * kt;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_m, cb_seed, cb_out);
    cb_wait_front(cb_ctl, 1);
    const uint32_t chunks = read_tile_value(cb_ctl, 0, 0);
    const uint32_t group_chunks = read_tile_value(cb_ctl, 0, 1);
    const uint32_t final = read_tile_value(cb_ctl, 0, 2);
    const uint32_t final_tail = read_tile_value(cb_ctl, 0, 3);
    const uint32_t need_group = read_tile_value(cb_ctl, 0, 4);
    const uint32_t need_tail = read_tile_value(cb_ctl, 0, 5);
    const uint32_t first_row = read_tile_value(cb_ctl, 0, 6);
    const uint32_t row_stride = read_tile_value(cb_ctl, 0, 7);
    cb_pop_front(cb_ctl, 1);
    // output_c = F0_c + (FI_c - F0_c) @ seed. The differences first, for all of this core's chunks (they need no
    // seed, so they run while the seeds arrive); then per chunk F0's row is copied into DST and the K-sum accumulates
    // on top of it.
    if (chunks != 0) {
        cb_wait_front(cb_m, chunks * kt);
        cb_wait_front(cb_base, chunks * kt);
        cb_reserve_back(cb_diff, chunks * kt);
        pack_reconfig_data_format(cb_diff);
        reconfig_data_format(cb_m, cb_base);
        sub_init(cb_m, cb_base);
        for (uint32_t t = 0; t < chunks * kt; t += kt) {
            tile_regs_acquire();
            for (uint32_t i = 0; i < kt; ++i) {
                sub_tiles(cb_m, cb_base, t + i, t + i, i);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < kt; ++i) {
                pack_tile(i, cb_diff);
            }
            tile_regs_release();
        }
        cb_push_back(cb_diff, chunks * kt);
        cb_wait_front(cb_diff, chunks * kt);
        pack_reconfig_data_format(cb_out);
    }
    if (need_group) {
        cb_wait_front(cb_seed, state_tiles);
    }
    if (need_tail) {
        cb_wait_front(cb_tail, state_tiles);
    }
    for (uint32_t c = 0; c < chunks; ++c) {
        const uint32_t seed = c < group_chunks ? cb_seed : cb_tail;
        cb_reserve_back(cb_out, kt);
        tile_regs_acquire();
        reconfig_data_format_srca(cb_base);
        copy_init(cb_base);
        for (uint32_t j = 0; j < kt; ++j) {
            copy_tile(cb_base, c * kt + j, j);
        }
        reconfig_data_format<SrcOrder::Reverse>(cb_diff, seed);
        matmul_block_init(cb_diff, seed, false, kt, 1, kt);
        for (uint32_t k = 0; k < kt; ++k) {
            matmul_block(cb_diff, seed, c * kt + k, k * kt, 0, false, kt, 1, kt);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < kt; ++j) {
            pack_tile(j, cb_out);
        }
        tile_regs_release();
        cb_push_back(cb_out, kt);
    }
    if (chunks != 0) {
        cb_pop_front(cb_m, chunks * kt);
        cb_pop_front(cb_base, chunks * kt);
        cb_pop_front(cb_diff, chunks * kt);
    }
    // This core's tile rows of final = A @ seed + B.
    if (final) {
        const uint32_t seed = final_tail ? cb_tail : cb_seed;
        pack_reconfig_data_format(cb_final);
        for (uint32_t row = first_row; row < kt; row += row_stride) {
            cb_wait_front(cb_a, kt);
            cb_wait_front(cb_b, kt);
            cb_reserve_back(cb_final, kt);
            tile_regs_acquire();
            reconfig_data_format_srca(cb_b);
            copy_init(cb_b);
            for (uint32_t c = 0; c < kt; ++c) {
                copy_tile(cb_b, c, c);
            }
            reconfig_data_format<SrcOrder::Reverse>(cb_a, seed);
            matmul_block_init(cb_a, seed, false, kt, 1, kt);
            for (uint32_t k = 0; k < kt; ++k) {
                matmul_block(cb_a, seed, k, k * kt, 0, false, kt, 1, kt);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t r = 0; r < kt; ++r) {
                pack_tile(r, cb_final);
            }
            tile_regs_release();
            cb_push_back(cb_final, kt);
            cb_pop_front(cb_a, kt);
            cb_pop_front(cb_b, kt);
        }
    }
    if (need_group) {
        cb_pop_front(cb_seed, state_tiles);
    }
    if (need_tail) {
        cb_pop_front(cb_tail, state_tiles);
    }
}
"""

_WRITER = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t cb_final = get_compile_time_arg_val(1);
    constexpr uint32_t cb_ctl = get_compile_time_arg_val(2);
    constexpr uint32_t num_chunks = get_compile_time_arg_val(3);
    constexpr uint32_t kt = get_compile_time_arg_val(4);
    constexpr auto out_args = TensorAccessorArgs<5>();
    constexpr auto final_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t final_addr = get_arg_val<uint32_t>(1);
    const uint32_t head = get_arg_val<uint32_t>(2);
    const uint32_t first_chunk = get_arg_val<uint32_t>(3);

    // The reader's control words: this core's chunk count and whether it forms the final state.
    cb_wait_front(cb_ctl, 1);
    volatile tt_l1_ptr uint32_t* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_ctl));
    const uint32_t chunks = words[0];
    const uint32_t final = words[1];
    const uint32_t first_row = words[2];
    const uint32_t row_stride = words[3];
    cb_pop_front(cb_ctl, 1);

    const auto out = TensorAccessor(out_args, out_addr);
    const uint32_t out_tile = get_tile_size(cb_out);
    for (uint32_t c = first_chunk; c < first_chunk + chunks; ++c) {
        cb_wait_front(cb_out, kt);
        const uint32_t row = (head * num_chunks + c) * kt;
        for (uint32_t j = 0; j < kt; ++j) {
            noc_async_write_page(row + j, out, get_read_ptr(cb_out) + j * out_tile);
        }
        noc_async_writes_flushed();
        cb_pop_front(cb_out, kt);
    }
    if (final) {
        const auto states = TensorAccessor(final_args, final_addr);
        const uint32_t tile = get_tile_size(cb_final);
        // Row j of the state: tiles (j, c).
        for (uint32_t j = first_row; j < kt; j += row_stride) {
            cb_wait_front(cb_final, kt);
            for (uint32_t c = 0; c < kt; ++c) {
                noc_async_write_page(head * kt * kt + j * kt + c, states, get_read_ptr(cb_final) + c * tile);
            }
            noc_async_writes_flushed();
            cb_pop_front(cb_final, kt);
        }
    }
    noc_async_write_barrier();
}
"""


class EntryCorrection:
    """Builds the correction program for one mesh and geometry (one group per head)."""

    def __init__(
        self,
        mesh_device,
        *,
        sequence_parallel_axis: int,
        batch_heads: int,
        num_chunks: int,
        key_dim: int,
        local_rows: int,
        cores: list,
        math_fidelity=ttnn.MathFidelity.HiFi2,
    ):
        self.mesh_device = mesh_device
        self.axis = sequence_parallel_axis
        self.batch_heads = batch_heads
        self.num_chunks = num_chunks
        self.kt = key_dim // ttnn.TILE_SIZE
        self.local_rows = local_rows
        # Cores per head: the most that divide the head's chunks evenly (so each core's chunk rows are one shard).
        self.cores_per_head = max(
            d for d in range(1, max(1, min(num_chunks, len(cores) // batch_heads)) + 1) if num_chunks % d == 0
        )
        self.cores = cores[: batch_heads * self.cores_per_head]
        self.core_set = ttnn.CoreRangeSet([]).merge(ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in self.cores]))
        self.per_core = -(-num_chunks // self.cores_per_head)
        # The emitted terms land in the L1 of the core that consumes them, when each core takes whole chunks: a core's
        # shard is its chunk rows, so the correction reads nothing over the NoC for them.
        self.local_terms = self.per_core * self.cores_per_head == num_chunks
        self.terms_memory_config = (
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(self.core_set, (self.per_core * 32, key_dim), ttnn.ShardOrientation.ROW_MAJOR),
            )
            if self.local_terms
            else ttnn.L1_MEMORY_CONFIG
        )
        self.compute_config = ttnn.ComputeConfigDescriptor(math_fidelity=math_fidelity, fp32_dest_acc_en=True)
        # Where the summary writes the emitted terms, and where the chunk outputs go.
        self.memory_config = ttnn.L1_MEMORY_CONFIG
        self.output_memory_config = ttnn.L1_MEMORY_CONFIG

    def __call__(self, zero_output, identity_output, final_a, final_b, entry, tail_entry, actual_start, actual_end):
        """Return (output [BH, NC, 32, V] BF16, final state [BH, K, V] FP32)."""
        mesh = self.mesh_device
        output = ttnn.allocate_tensor_on_device(
            ttnn.Shape(list(zero_output.shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh, self.output_memory_config
        )
        final = ttnn.allocate_tensor_on_device(
            ttnn.Shape(list(final_b.shape)), ttnn.float32, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
        )
        rows, cols = tuple(mesh.shape)
        sp_size = tuple(mesh.shape)[self.axis]
        kt = self.kt
        state_tiles = kt * kt
        per_core = -(-self.num_chunks // self.cores_per_head)
        end = actual_end if actual_end is not None else actual_start
        cb = lambda index, tiles, dtype, page: ttnn.CBDescriptor(
            total_size=tiles * page,
            core_ranges=self.core_set,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)],
        )
        reader_args, writer_args = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        # Per head, the count and NoC coordinates (x | y << 16) of its cores after the first.
        mcast = {}
        for head in range(self.batch_heads):
            group = self.cores[head * self.cores_per_head : (head + 1) * self.cores_per_head]
            rest = [mesh.worker_core_from_logical_core(c) for c in group[1:]]
            mcast[head] = (len(rest), *[c.x | (c.y << 16) for c in rest])
        for index, core in enumerate(self.cores):
            head, part = divmod(index, self.cores_per_head)
            first = part * per_core
            last = min(self.num_chunks, first + per_core)
            reader_args[core.x][core.y] = [
                zero_output.buffer_address(),
                identity_output.buffer_address(),
                final_a.buffer_address(),
                final_b.buffer_address(),
                entry.buffer_address(),
                tail_entry.buffer_address(),
                actual_start.buffer_address(),
                end.buffer_address(),
                head,
                first,
                last,
                part if part < kt else kt,
                int(part == 0),
                *mcast[head],
            ]
            writer_args[core.x][core.y] = [output.buffer_address(), final.buffer_address(), head, first]
        desc = ttnn.MeshProgramDescriptor()
        src = ttnn.KernelDescriptor.SourceType.SOURCE_CODE
        for r in range(rows):
            for c in range(cols):
                sp_rank = (r, c)[self.axis]
                program = ttnn.ProgramDescriptor()
                program.semaphores = [ttnn.SemaphoreDescriptor(0, ttnn.CoreType.WORKER, self.core_set, 0)]
                program.cbs = [
                    cb(_CB_SEED, state_tiles, ttnn.float32, 4096),
                    cb(_CB_TAIL, state_tiles, ttnn.float32, 4096),
                    (
                        ttnn.cb_descriptor_from_sharded_tensor(_CB_M, identity_output)
                        if self.local_terms
                        else cb(_CB_M, per_core * kt, ttnn.float32, 4096)
                    ),
                    (
                        ttnn.cb_descriptor_from_sharded_tensor(_CB_BASE, zero_output)
                        if self.local_terms
                        else cb(_CB_BASE, per_core * kt, ttnn.float32, 4096)
                    ),
                    cb(_CB_DIFF, per_core * kt, ttnn.float32, 4096),
                    cb(_CB_OUT, 2 * kt, ttnn.bfloat16, 2048),
                    cb(_CB_A, kt, ttnn.float32, 4096),
                    cb(_CB_B, kt, ttnn.float32, 4096),
                    cb(_CB_FINAL, 2 * kt, ttnn.float32, 4096),
                    cb(_CB_CTL, 1, ttnn.uint32, 64),
                    cb(_CB_CTL_WRITER, 1, ttnn.uint32, 64),
                ]
                reader_ct = [
                    _CB_SEED,
                    _CB_TAIL,
                    _CB_M,
                    _CB_BASE,
                    _CB_A,
                    _CB_B,
                    _CB_CTL,
                    sp_rank,
                    sp_size,
                    self.local_rows,
                    self.num_chunks,
                    kt,
                    int(actual_end is not None),
                    _CB_CTL_WRITER,
                    0,
                    self.cores_per_head,
                ]
                for t in (zero_output, identity_output, final_a, final_b, entry, tail_entry, actual_start, end):
                    reader_ct += list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
                writer_ct = [_CB_OUT, _CB_FINAL, _CB_CTL_WRITER, self.num_chunks, kt]
                for t in (output, final):
                    writer_ct += list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
                program.kernels = [
                    ttnn.KernelDescriptor(
                        kernel_source=("#define LOCAL_TERMS\n" if self.local_terms else "") + _READER,
                        source_type=src,
                        core_ranges=self.core_set,
                        compile_time_args=reader_ct,
                        runtime_args=reader_args,
                        config=ttnn.DataMovementConfigDescriptor(
                            processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0
                        ),
                    ),
                    ttnn.KernelDescriptor(
                        kernel_source=_WRITER,
                        source_type=src,
                        core_ranges=self.core_set,
                        compile_time_args=writer_ct,
                        runtime_args=writer_args,
                        config=ttnn.DataMovementConfigDescriptor(
                            processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1
                        ),
                    ),
                    ttnn.KernelDescriptor(
                        kernel_source=_COMPUTE,
                        source_type=src,
                        core_ranges=self.core_set,
                        compile_time_args=[
                            _CB_SEED,
                            _CB_TAIL,
                            _CB_M,
                            _CB_BASE,
                            _CB_OUT,
                            _CB_A,
                            _CB_B,
                            _CB_FINAL,
                            _CB_CTL,
                            kt,
                            _CB_DIFF,
                        ],
                        runtime_args=ttnn.RuntimeArgs(),
                        config=self.compute_config,
                    ),
                ]
                key = ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, c), ttnn.MeshCoordinate(r, c))
                desc[key] = program
        io = [zero_output, identity_output, final_a, final_b, entry, tail_entry, actual_start]
        if actual_end is not None:
            io.append(actual_end)
        ttnn.generic_op(io + [output, final], desc)
        return output, final
