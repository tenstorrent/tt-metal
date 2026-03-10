// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#ifdef ARCH_QUASAR
#include "experimental/dataflow_buffer.h"
#include "experimental/noc.h"
#endif

#include "api/debug/dprint.h"

void generate_bcast_scaler() {
#ifdef ARCH_QUASAR
    experimental::DataflowBuffer dfb1(1);
#else
    constexpr uint32_t cb_in_2 = 2;
#endif
    uint32_t scaler = get_arg_val<uint32_t>(8);
    union {
        float f;
        uint32_t u;
    } u;
    u.u = scaler;
    DPRINT << "RD: scaler raw=" << HEX() << scaler << DEC() << " float=" << F32(u.f) << ENDL();
#ifdef ARCH_QUASAR
    dfb1.reserve_back(1);
    auto ptr = reinterpret_cast<uint16_t*>(dfb1.get_write_ptr() + MEMORY_PORT_NONCACHEABLE_MEM_PORT_MEM_BASE_ADDR);
    DPRINT << "RD: scaler DFB1 write_ptr=" << HEX() << dfb1.get_write_ptr() << DEC() << ENDL();
#else
    cb_reserve_back(cb_in_2, 1);
    auto ptr = reinterpret_cast<uint16_t*>(get_write_ptr(cb_in_2));
#endif
    for (int j = 0; j < 1024; j++) {
        ptr[j] = uint16_t(0);
    }

    for (int k = 0; k < 4; k++) {
        for (int j = 0; j < 16; j++) {
            ptr[k * 256 + j] = uint16_t(u.u >> 16);
        }
    }
#ifdef ARCH_QUASAR
    DPRINT << "RD: scaler verify[0..7]=" << HEX() << (uint32_t)ptr[0] << " " << (uint32_t)ptr[1] << " "
           << (uint32_t)ptr[2] << " " << (uint32_t)ptr[3] << " " << (uint32_t)ptr[4] << " " << (uint32_t)ptr[5] << " "
           << (uint32_t)ptr[6] << " " << (uint32_t)ptr[7] << DEC() << ENDL();
    dfb1.push_back(1);
#else
    cb_push_back(cb_in_2, 1);
#endif
}

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles =
        get_arg_val<uint32_t>(3);  // same arg index as in reader_unary and in reader_unary_transpose_wh_8bank

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;

#ifdef ARCH_QUASAR
    experimental::Noc noc;
    experimental::DataflowBuffer dfb0(0);
    const uint32_t tile_bytes = dfb0.get_entry_size();
#else
    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t tile_bytes = get_tile_size(cb_id_in0);
#endif

    constexpr auto src_args = TensorAccessorArgs<0>();
    const auto src_a = TensorAccessor(src_args, src_addr, tile_bytes);

#if GENERATE_BCAST_SCALER
    // TODO(AP): cleanup, probably with named args/param pack/reflection.
    generate_bcast_scaler();
    constexpr uint32_t blk = BLOCK_SIZE;
#else
    constexpr uint32_t blk = 1;  // 1 for correctness for unfused kernels
#endif

#ifdef TILE_OFFSET
    uint32_t tile_offset = TILE_OFFSET;
#else
    constexpr uint32_t tile_offset = 0;
#endif
    // DPRINT << "Reader Tile offset=" << tile_offset << ENDL();
    DPRINT << "RD: start num_tiles=" << num_tiles << " tile_bytes=" << tile_bytes << " src_addr=" << HEX() << src_addr
           << DEC() << ENDL();
    DPRINT << "RD: input DFB0 base info: entry_size=" << dfb0.get_entry_size() << ENDL();
    // read a ublock of tiles from src to CB, and then push the ublock to unpacker
    uint32_t i_tile = 0;
    for (uint32_t i = 0; i < num_tiles; i += blk) {
        uint32_t rem = blk;  // (i + blk > num_tiles) ? num_tiles - i : blk;
#ifdef ARCH_QUASAR
        dfb0.reserve_back(rem);
        auto ptr = reinterpret_cast<uint16_t*>(dfb0.get_write_ptr());
        // DPRINT << "RD: DFB0 write_ptr=" << HEX() << dfb0.get_write_ptr() << DEC() << ENDL();
        uint32_t l1_write_addr = dfb0.get_write_ptr();

        for (uint32_t r = 0; r < rem; r++) {
            uint64_t src_noc_addr =
                get_noc_addr(i + r + tile_offset, src_a);  // not contiguous for sequential r, can be banked
            auto addr = l1_write_addr + (r * tile_bytes);
            noc_async_read(src_noc_addr, addr, tile_bytes);  // TODO(AP): data type size
        }
        noc.async_read_barrier();
        dfb0.push_back(rem);
#else
        cb_reserve_back(cb_id_in0, rem);
        uint32_t l1_write_addr = get_write_ptr(cb_id_in0);

        for (uint32_t r = 0; r < rem; r++) {
            uint64_t src_noc_addr =
                get_noc_addr(i + r + tile_offset, src_a);  // not contiguous for sequential r, can be banked
            auto addr = l1_write_addr + (r * tile_bytes);
            noc_async_read(src_noc_addr, addr, tile_bytes);  // TODO(AP): data type size
        }
        noc_async_read_barrier();
        cb_push_back(cb_id_in0, rem);
#endif
        // DPRINT << "RD: iteration i=" << i << ENDL();
    }
    DPRINT << "RD: done" << ENDL();
}
