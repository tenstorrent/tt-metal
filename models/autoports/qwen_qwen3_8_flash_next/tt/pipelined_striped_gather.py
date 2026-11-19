# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Batched selected-cell reads, compatible with StripedGather's entire ABI.

Only kernel_source changes. Compile args, six runtime args, tensor accessors,
core work assignment, and the existing16KiB circular buffer remain unchanged.
The last4096 bytes hold up to64 distinct64-byte NoC destinations. A batch's
reads complete before those slots are consumed or reused. Output padding is
explicitly zeroed and no out-of-logical-range index is read from the table.
"""

from .striped_gather import _ADDRESS_SOURCE


def pipelined_striped_gather(table, columns, stripes, *, output=None):
    """Use32 independent aligned reads per barrier for the mesh PLE table."""
    from .striped_gather import striped_gather

    return striped_gather(table, columns, stripes, output=output, kernel_source_fn=lambda: kernel_source(32))


def kernel_source(batch_size=16):
    if type(batch_size) is not int or not 1 <= batch_size <= 64:
        raise ValueError("batch_size must be an integer in1..64 for the existing4096-byte read arena")
    return (
        '#include <stdint.h>\n#include "api/dataflow/dataflow_api.h"\n'
        + _ADDRESS_SOURCE
        + f"\nconstexpr uint32_t READ_BATCH = {batch_size}u;\n"
        + r"""
void kernel_main() {
    constexpr uint32_t H = get_compile_time_arg_val(0);
    constexpr uint32_t W = get_compile_time_arg_val(1);
    constexpr uint32_t C = get_compile_time_arg_val(2);
    constexpr uint32_t VALUE_BYTES = get_compile_time_arg_val(3);
    constexpr uint32_t INPUT_WT = (W + 31u) / 32u;
    constexpr uint32_t OUTPUT_BYTES = 1024u * VALUE_BYTES;
    static_assert(READ_BATCH * 64u <= 4096u);
    static_assert(VALUE_BYTES == 2 || VALUE_BYTES == 4);
    constexpr auto ta = TensorAccessorArgs<4>();
    constexpr auto ca = TensorAccessorArgs<ta.next_compile_time_args_offset()>();
    constexpr auto sa = TensorAccessorArgs<ca.next_compile_time_args_offset()>();
    constexpr auto oa = TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    const auto table = TensorAccessor(ta, get_arg_val<uint32_t>(0), OUTPUT_BYTES);
    const auto columns = TensorAccessor(ca, get_arg_val<uint32_t>(1), 4096);
    const auto stripes = TensorAccessor(sa, get_arg_val<uint32_t>(2), 4096);
    const auto output = TensorAccessor(oa, get_arg_val<uint32_t>(3), OUTPUT_BYTES);
    const uint32_t begin = get_arg_val<uint32_t>(4);
    const uint32_t end = get_arg_val<uint32_t>(5);
    const uint32_t scratch = get_write_ptr(tt::CBIndex::c_0);
    const uint32_t out_ptr = scratch + 8192;
    const uint32_t lines_ptr = scratch + 12288;
    volatile tt_l1_ptr uint32_t* c = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + 4096);
    volatile tt_l1_ptr uint32_t* out32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ptr);
    volatile tt_l1_ptr uint16_t* out16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(out_ptr);
    for (uint32_t tile = begin; tile < end; ++tile) {
        noc_async_read(columns.get_noc_addr(tile), scratch, 4096);
        noc_async_read(stripes.get_noc_addr(tile), scratch + 4096, 4096);
        noc_async_read_barrier();
        for (uint32_t i = 0; i < OUTPUT_BYTES / 4; ++i) out32[i] = 0;
        const uint32_t tile_row = tile / INPUT_WT;
        const uint32_t tile_col = tile % INPUT_WT;
        for (uint32_t base = 0; base < 1024; base += READ_BATCH) {
            // Each outstanding read has its own64-byte aligned slot. Invalid
            // tail elements issue no read; their output remains initialized zero.
            bool pending = false;
            for (uint32_t slot = 0; slot < READ_BATCH && base + slot < 1024; ++slot) {
                const uint32_t element = base + slot;
                const uint32_t face = element >> 8;
                const uint32_t row = ((face >> 1) << 4) + ((element & 255u) >> 4);
                const uint32_t col = ((face & 1u) << 4) + (element & 15u);
                if (tile_row * 32 + row >= H || tile_col * 32 + col >= W) continue;
                const uint32_t column = c[element];
                const uint32_t stripe = s[element];
                ASSERT(column < C && stripe < 32);
                const uint32_t byte_offset = striped_tile_element(stripe, column) * VALUE_BYTES;
                noc_async_read(
                    table.get_noc_addr(column >> 5) + (byte_offset & ~63u),
                    lines_ptr + slot * 64u,
                    64);
                pending = true;
            }
            if (!pending) continue;
            noc_async_read_barrier();
            for (uint32_t slot = 0; slot < READ_BATCH && base + slot < 1024; ++slot) {
                const uint32_t element = base + slot;
                const uint32_t face = element >> 8;
                const uint32_t row = ((face >> 1) << 4) + ((element & 255u) >> 4);
                const uint32_t col = ((face & 1u) << 4) + (element & 15u);
                if (tile_row * 32 + row >= H || tile_col * 32 + col >= W) continue;
                const uint32_t byte_offset = striped_tile_element(s[element], c[element]) * VALUE_BYTES;
                const uint32_t selected_ptr = lines_ptr + slot * 64u + (byte_offset & 63u);
                if constexpr (VALUE_BYTES == 4) {
                    out32[element] = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(selected_ptr);
                } else {
                    out16[element] = *reinterpret_cast<volatile tt_l1_ptr uint16_t*>(selected_ptr);
                }
            }
        }
        noc_async_write(out_ptr, output.get_noc_addr(tile), OUTPUT_BYTES);
        noc_async_write_barrier();
    }
}
"""
    )
