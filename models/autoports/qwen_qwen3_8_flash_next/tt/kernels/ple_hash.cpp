// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t seq = get_compile_time_arg_val(0);
    constexpr uint32_t token_bytes = ((seq * 4 + 63) / 64) * 64;
    constexpr uint32_t eos = get_compile_time_arg_val(1);
    constexpr uint64_t m0 = uint64_t(get_compile_time_arg_val(2)) | (uint64_t(get_compile_time_arg_val(3)) << 32);
    constexpr uint64_t m1 = uint64_t(get_compile_time_arg_val(4)) | (uint64_t(get_compile_time_arg_val(5)) << 32);
    constexpr uint64_t m2 = uint64_t(get_compile_time_arg_val(6)) | (uint64_t(get_compile_time_arg_val(7)) << 32);
    constexpr auto ta = TensorAccessorArgs<40>();
    constexpr auto pa = TensorAccessorArgs<ta.next_compile_time_args_offset()>();
    constexpr auto sa = TensorAccessorArgs<pa.next_compile_time_args_offset()>();
    constexpr auto ha = TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    constexpr auto va = TensorAccessorArgs<ha.next_compile_time_args_offset()>();
    constexpr auto oa = TensorAccessorArgs<va.next_compile_time_args_offset()>();
    const auto tok = TensorAccessor(ta, get_arg_val<uint32_t>(0));
    const auto pos = TensorAccessor(pa, get_arg_val<uint32_t>(1));
    const auto slot = TensorAccessor(sa, get_arg_val<uint32_t>(2));
    const auto history = TensorAccessor(ha, get_arg_val<uint32_t>(3));
    const auto valid = TensorAccessor(va, get_arg_val<uint32_t>(4));
    const auto out = TensorAccessor(oa, get_arg_val<uint32_t>(5));
    const uint32_t batch = get_arg_val<uint32_t>(6);
    CircularBuffer cb(0);
    cb.reserve_back(1);
    const uint32_t base = cb.get_write_ptr();
    auto input = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
    auto buf = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + token_bytes);
    Noc noc;
    noc.async_read(tok, CoreLocalMem<uint32_t>(base), seq * 4, {.page_id = batch}, {});
    noc.async_read(pos, CoreLocalMem<uint32_t>(base + token_bytes), 128, {.page_id = batch}, {});
    noc.async_read(slot, CoreLocalMem<uint32_t>(base + token_bytes + 128), 128, {.page_id = batch}, {});
    noc.async_read(valid, CoreLocalMem<uint32_t>(base + token_bytes + 256), 128, {.page_id = batch}, {});
    noc.async_read_barrier();
    const uint32_t position = buf[0], request = buf[32], length = buf[64];
    ASSERT(length <= seq);
    noc.async_read(history, CoreLocalMem<uint32_t>(base + token_bytes + 384), 128, {.page_id = request}, {});
    noc.async_read_barrier();
    uint32_t older = position ? buf[96] : eos;
    uint32_t newer = position ? buf[97] : eos;
    ASSERT(!position || position == buf[98]);
    auto ids = buf + 128;
    for (uint32_t t = 0; t < seq; ++t) {
        if (t < length) {
            const uint32_t current = input[t];
            const uint64_t bigram = uint64_t(current) * m0 ^ uint64_t(newer) * m1;
            const uint64_t trigram = bigram ^ uint64_t(older) * m2;
            // Runtime indexing of compile-time arguments is unsupported: materialize constants.
            constexpr uint32_t primes[16] = {
                get_compile_time_arg_val(8),
                get_compile_time_arg_val(9),
                get_compile_time_arg_val(10),
                get_compile_time_arg_val(11),
                get_compile_time_arg_val(12),
                get_compile_time_arg_val(13),
                get_compile_time_arg_val(14),
                get_compile_time_arg_val(15),
                get_compile_time_arg_val(16),
                get_compile_time_arg_val(17),
                get_compile_time_arg_val(18),
                get_compile_time_arg_val(19),
                get_compile_time_arg_val(20),
                get_compile_time_arg_val(21),
                get_compile_time_arg_val(22),
                get_compile_time_arg_val(23)};
            constexpr uint32_t offsets[16] = {
                get_compile_time_arg_val(24),
                get_compile_time_arg_val(25),
                get_compile_time_arg_val(26),
                get_compile_time_arg_val(27),
                get_compile_time_arg_val(28),
                get_compile_time_arg_val(29),
                get_compile_time_arg_val(30),
                get_compile_time_arg_val(31),
                get_compile_time_arg_val(32),
                get_compile_time_arg_val(33),
                get_compile_time_arg_val(34),
                get_compile_time_arg_val(35),
                get_compile_time_arg_val(36),
                get_compile_time_arg_val(37),
                get_compile_time_arg_val(38),
                get_compile_time_arg_val(39)};
            for (uint32_t h = 0; h < 16; ++h) {
                ids[h] = (h < 8 ? bigram : trigram) % primes[h] + offsets[h];
            }
            older = current == eos ? eos : newer;
            newer = current;
        } else {
            for (uint32_t h = 0; h < 16; ++h) {
                ids[h] = 0;
            }
        }
        noc.async_write(CoreLocalMem<uint32_t>(base + token_bytes + 512), out, 64, {}, {.page_id = batch * seq + t});
        noc.async_write_barrier();
    }
    buf[96] = older;
    buf[97] = newer;
    buf[98] = position + length;
    noc.async_write(CoreLocalMem<uint32_t>(base + token_bytes + 384), history, 128, {}, {.page_id = request});
    noc.async_write_barrier();
}
