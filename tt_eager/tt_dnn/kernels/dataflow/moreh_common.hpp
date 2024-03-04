// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

union Scalar {
    float f;
    uint32_t u;
};

class ArgFetcher {
   private:
    int arg_idx = 0;

   public:
    template <typename T>
    T get_next_arg_val() {
        return get_arg_val<T>(arg_idx++);
    }
};

#define InterleavedAddrGenFastHelper(addr, cb, idx) \
    ({ \
        constexpr bool is_dram = (get_compile_time_arg_val(idx) == 1); \
        const InterleavedAddrGenFast<is_dram> ret = InterleavedAddrGenFastHelper_<is_dram>(addr, cb, idx); \
        ret; \
    })

template <bool DRAM>
FORCE_INLINE InterleavedAddrGenFast<DRAM> InterleavedAddrGenFastHelper_(uint32_t addr, tt::CB cb, uint32_t idx) {
    uint32_t tile_bytes = get_tile_size(cb);
    auto data_format = get_dataformat(cb);

    const InterleavedAddrGenFast<DRAM> x = {
        .bank_base_address = addr,
        .page_size = tile_bytes,
        .data_format = data_format
    };

    return x;
}

template<typename AddrGen>
FORCE_INLINE void noc_async_read_tile_helper(tt::CB cb, uint32_t num_tiles, uint32_t tile_idx, AddrGen addr_gen) {
    cb_reserve_back(cb, num_tiles);
    uint32_t addr = get_write_ptr(cb);
    noc_async_read_tile(tile_idx, addr_gen, addr);
    noc_async_read_barrier();
    cb_push_back(cb, num_tiles);
}

template<typename AddrGen>
FORCE_INLINE void noc_async_write_tile_helper(tt::CB cb, uint32_t num_tiles, uint32_t tile_idx, AddrGen addr_gen) {
    cb_wait_front(cb, num_tiles);
    uint32_t l1_read_addr = get_read_ptr(cb);
    noc_async_write_tile(tile_idx, addr_gen, l1_read_addr);
    noc_async_write_barrier();
    cb_pop_front(cb, num_tiles);
}

FORCE_INLINE void generate_bcast_scaler(
    uint32_t cb_scaler,
    uint32_t scaler) {
    union { float f; uint32_t u; } u; u.u = scaler;
    cb_reserve_back(cb_scaler, 1);
    auto ptr = reinterpret_cast<uint16_t*>(get_write_ptr(cb_scaler));

    for (int j = 0; j < 1024; j++)
        ptr[j] = uint16_t(0);

    for (int k = 0; k < 4; k++)
    for (int j = 0; j < 16; j++)
        ptr[k*256 + j] = uint16_t(u.u>>16);
    cb_push_back(cb_scaler, 1);
}

FORCE_INLINE void fill_cb_with_value(uint32_t cb_id, uint32_t value, int32_t num_of_elems = 1024) {
    cb_reserve_back(cb_id, 1);
    auto ptr = reinterpret_cast<uint16_t *>(get_write_ptr(cb_id));
    for (int j = 0; j < 1024; j++) {
        ptr[j] = uint16_t(value >> 16);
    }
    cb_push_back(cb_id, 1);
}

FORCE_INLINE uint32_t get_tilized_idx(uint32_t h_idx, uint32_t w_idx, uint32_t tile_height, uint32_t tile_width) {
    const auto half_tile_height = tile_height / 2;
    const auto half_tile_width = tile_width / 2;

    if (h_idx < half_tile_height && w_idx < half_tile_width) {
        return h_idx * half_tile_width + w_idx;
    }

    if (h_idx < half_tile_height && w_idx >= half_tile_width) {
        return h_idx * half_tile_width + (w_idx % half_tile_width) + half_tile_height * half_tile_width;
    }

    if (h_idx >= half_tile_height && w_idx < half_tile_width) {
        return (h_idx % half_tile_width) * half_tile_width + w_idx + half_tile_height * tile_width;
    }

    return (h_idx % half_tile_height) * half_tile_width + (w_idx % half_tile_width) +
           half_tile_height * (tile_width + half_tile_width);
}

FORCE_INLINE uint32_t get_gamma_beta_tile_idx(uint32_t input_tile_idx, uint32_t HtWt, uint32_t C, uint32_t tile_width) {
    const auto c_idx = (input_tile_idx / HtWt) % C;
    const auto tile_idx = c_idx / tile_width;
    return tile_idx;
}

FORCE_INLINE uint32_t get_tilized_gamma_beta_idx_in_tile(
    uint32_t input_tile_idx, uint32_t HtWt, uint32_t C, uint32_t tile_height, uint32_t tile_width) {
    const auto c_idx = (input_tile_idx / HtWt) % C;
    const auto w_idx_in_tile = c_idx % tile_width;
    const auto tilized_idx_in_tile = get_tilized_idx(0, w_idx_in_tile, tile_height, tile_width);
    return tilized_idx_in_tile;
}

FORCE_INLINE void generate_mask_h(uint32_t cb_mask, uint32_t mask_h) {
    Scalar one;
    Scalar zero;

    one.f = 1.0f;
    zero.f = 0.0f;

    cb_reserve_back(cb_mask, 1);
    auto ptr = reinterpret_cast<uint16_t *>(get_write_ptr(cb_mask));

    for (uint32_t w = 0; w < 16; w++) {
        // sub tile 0
        {
            uint32_t mask_h_0 = mask_h;
            if (mask_h_0 >= 16)
                mask_h_0 = 16;
            uint32_t h = 0;
            for (; h < mask_h_0; h++) {
                ptr[h * 16 + w] = uint16_t(one.u >> 16);
            }
            for (; h < 16; h++) {
                ptr[h * 16 + w] = uint16_t(zero.u >> 16);
            }
        }

        // sub tile 1
        {
            uint32_t mask_h_0 = mask_h;
            if (mask_h_0 >= 16)
                mask_h_0 = 16;
            uint32_t h = 0;
            for (; h < mask_h_0; h++) {
                ptr[h * 16 + w + 256] = uint16_t(one.u >> 16);
            }
            for (; h < 16; h++) {
                ptr[h * 16 + w + 256] = uint16_t(zero.u >> 16);
            }
        }

        // sub tile 2
        {
            uint32_t mask_h_1 = (mask_h < 16) ? 0 : mask_h - 16;
            uint32_t h = 0;
            for (; h < mask_h_1; h++) {
                ptr[h * 16 + w + 512] = uint16_t(one.u >> 16);
            }
            for (; h < 16; h++) {
                ptr[h * 16 + w + 512] = uint16_t(zero.u >> 16);
            }
        }

        // sub tile 3
        {
            uint32_t mask_h_1 = (mask_h < 16) ? 0 : mask_h - 16;
            uint32_t h = 0;
            for (; h < mask_h_1; h++) {
                ptr[h * 16 + w + 768] = uint16_t(one.u >> 16);
            }
            for (; h < 16; h++) {
                ptr[h * 16 + w + 768] = uint16_t(zero.u >> 16);
            }
        }
    }

    cb_push_back(cb_mask, 1);
}

FORCE_INLINE void generate_mask_w(uint32_t cb_mask, uint32_t mask_w) {
    Scalar one;
    Scalar zero;

    one.f = 1.0f;
    zero.f = 0.0f;

    cb_reserve_back(cb_mask, 1);
    auto ptr = reinterpret_cast<uint16_t *>(get_write_ptr(cb_mask));

    for (uint32_t h = 0; h < 16; h++) {
        // sub tile 0
        {
            uint32_t mask_w_0 = mask_w;
            if (mask_w_0 >= 16)
                mask_w_0 = 16;
            uint32_t w = 0;
            for (; w < mask_w_0; w++) {
                ptr[h * 16 + w] = uint16_t(one.u >> 16);
            }
            for (; w < 16; w++) {
                ptr[h * 16 + w] = uint16_t(zero.u >> 16);
            }
        }

        // sub tile 1
        {
            uint32_t mask_w_1 = (mask_w < 16) ? 0 : mask_w - 16;
            uint32_t w = 0;
            for (; w < mask_w_1; w++) {
                ptr[h * 16 + w + 256] = uint16_t(one.u >> 16);
            }
            for (; w < 16; w++) {
                ptr[h * 16 + w + 256] = uint16_t(zero.u >> 16);
            }
        }

        // sub tile 2
        {
            uint32_t mask_w_0 = mask_w;
            if (mask_w_0 >= 16)
                mask_w_0 = 16;
            uint32_t w = 0;
            for (; w < mask_w_0; w++) {
                ptr[h * 16 + w + 512] = uint16_t(one.u >> 16);
            }
            for (; w < 16; w++) {
                ptr[h * 16 + w + 512] = uint16_t(zero.u >> 16);
            }
        }

        // sub tile 3
        {
            uint32_t mask_w_1 = (mask_w < 16) ? 0 : mask_w - 16;
            uint32_t w = 0;
            for (; w < mask_w_1; w++) {
                ptr[h * 16 + w + 768] = uint16_t(one.u >> 16);
            }
            for (; w < 16; w++) {
                ptr[h * 16 + w + 768] = uint16_t(zero.u >> 16);
            }
        }
    }

    cb_push_back(cb_mask, 1);
}

FORCE_INLINE void generate_mask_h_w(
    uint32_t cb_mask_h_w, uint32_t mask_h, uint32_t mask_w, uint32_t single_tile_size = 2048) {
    Scalar one;
    Scalar zero;

    one.f = 1.0f;
    zero.f = 0.0f;

    const auto u16_one = uint16_t(one.u >> 16);
    const auto u16_zero = uint16_t(zero.u >> 16);

    cb_reserve_back(cb_mask_h_w, 2);

    // mask_h
    // first tile ptr
    auto mask_h_ptr = reinterpret_cast<uint16_t *>(get_write_ptr(cb_mask_h_w));
    for (uint32_t w = 0; w < 16; w++) {
        // sub tile 0
        {
            uint32_t mask_h_0 = mask_h;
            if (mask_h_0 >= 16) {
                mask_h_0 = 16;
            }
            uint32_t h = 0;
            for (; h < mask_h_0; h++) {
                mask_h_ptr[h * 16 + w] = u16_one;
            }
            for (; h < 16; h++) {
                mask_h_ptr[h * 16 + w] = u16_zero;
            }
        }

        // sub tile 1
        {
            uint32_t mask_h_0 = mask_h;
            if (mask_h_0 >= 16) {
                mask_h_0 = 16;
            }
            uint32_t h = 0;
            for (; h < mask_h_0; h++) {
                mask_h_ptr[h * 16 + w + 256] = u16_one;
            }
            for (; h < 16; h++) {
                mask_h_ptr[h * 16 + w + 256] = u16_zero;
            }
        }

        // sub tile 2
        {
            uint32_t mask_h_1 = (mask_h < 16) ? 0 : mask_h - 16;
            uint32_t h = 0;
            for (; h < mask_h_1; h++) {
                mask_h_ptr[h * 16 + w + 512] = u16_one;
            }
            for (; h < 16; h++) {
                mask_h_ptr[h * 16 + w + 512] = u16_zero;
            }
        }

        // sub tile 3
        {
            uint32_t mask_h_1 = (mask_h < 16) ? 0 : mask_h - 16;
            uint32_t h = 0;
            for (; h < mask_h_1; h++) {
                mask_h_ptr[h * 16 + w + 768] = u16_one;
            }
            for (; h < 16; h++) {
                mask_h_ptr[h * 16 + w + 768] = u16_zero;
            }
        }
    }

    // mask_w
    // second tile ptr
    auto mask_w_ptr = reinterpret_cast<uint16_t *>(get_write_ptr(cb_mask_h_w) + single_tile_size);
    for (uint32_t h = 0; h < 16; h++) {
        // sub tile 0
        {
            uint32_t mask_w_0 = mask_w;
            if (mask_w_0 >= 16) {
                mask_w_0 = 16;
            }
            uint32_t w = 0;
            for (; w < mask_w_0; w++) {
                mask_w_ptr[h * 16 + w] = u16_one;
            }
            for (; w < 16; w++) {
                mask_w_ptr[h * 16 + w] = u16_zero;
            }
        }

        // sub tile 1
        {
            uint32_t mask_w_1 = (mask_w < 16) ? 0 : mask_w - 16;
            uint32_t w = 0;
            for (; w < mask_w_1; w++) {
                mask_w_ptr[h * 16 + w + 256] = u16_one;
            }
            for (; w < 16; w++) {
                mask_w_ptr[h * 16 + w + 256] = u16_zero;
            }
        }

        // sub tile 2
        {
            uint32_t mask_w_0 = mask_w;
            if (mask_w_0 >= 16) {
                mask_w_0 = 16;
            }
            uint32_t w = 0;
            for (; w < mask_w_0; w++) {
                mask_w_ptr[h * 16 + w + 512] = u16_one;
            }
            for (; w < 16; w++) {
                mask_w_ptr[h * 16 + w + 512] = u16_zero;
            }
        }

        // sub tile 3
        {
            uint32_t mask_w_1 = (mask_w < 16) ? 0 : mask_w - 16;
            uint32_t w = 0;
            for (; w < mask_w_1; w++) {
                mask_w_ptr[h * 16 + w + 768] = u16_one;
            }
            for (; w < 16; w++) {
                mask_w_ptr[h * 16 + w + 768] = u16_zero;
            }
        }
    }

    cb_push_back(cb_mask_h_w, 2);
}

FORCE_INLINE void generate_mask_h_w_if_needed(uint32_t cb_mask_h_w, uint32_t origin_h, uint32_t origin_w) {
    constexpr uint32_t TILE_H = 32;
    constexpr uint32_t TILE_W = 32;

    const bool do_mask_h = (origin_h % TILE_H) != 0;
    const uint32_t mask_h = do_mask_h ? (origin_h % TILE_H) : TILE_H;

    const bool do_mask_w = (origin_w % TILE_W) != 0;
    const uint32_t mask_w = do_mask_w ? (origin_w % TILE_W) : TILE_W;

    if (do_mask_h || do_mask_w) {
        const uint32_t mask_tile_bytes = get_tile_size(cb_mask_h_w);
        generate_mask_h_w(cb_mask_h_w, mask_h, mask_w, mask_tile_bytes);
    }
}

FORCE_INLINE void mask_tile_hw(uint32_t l1_addr, uint32_t mask_h = 32, uint32_t mask_w = 32) {
    Scalar zero;
    zero.f = 0.0f;
    const auto u16_zero = uint16_t(zero.u >> 16);

    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_addr);
    for (uint32_t h = 0; h < 16; h++) {
        // sub tile 0
        {
            uint32_t mask_w_0 = (mask_w >= 16) ? 16 : mask_w;
            uint32_t mask_h_0 = (mask_h >= 16) ? 16 : mask_h;
            uint32_t w = (h >= mask_h_0) ? 0 : mask_w_0;
            for (; w < 16; w++) {
                ptr[h * 16 + w] = u16_zero;
            }
        }
        // sub tile 1
        {
            uint32_t mask_w_1 = (mask_w < 16) ? 0 : mask_w - 16;
            uint32_t mask_h_0 = (mask_h >= 16) ? 16 : mask_h;
            uint32_t w = (h >= mask_h_0) ? 0 : mask_w_1;
            for (; w < 16; w++) {
                ptr[h * 16 + w + 256] = u16_zero;
            }
        }
        // sub tile 2
        {
            uint32_t mask_w_0 = (mask_w >= 16) ? 16 : mask_w;
            uint32_t mask_h_1 = (mask_h < 16) ? 0 : mask_h - 16;
            uint32_t w = (h >= mask_h_1) ? 0 : mask_w_0;
            for (; w < 16; w++) {
                ptr[h * 16 + w + 512] = u16_zero;
            }
        }
        // sub tile 3
        {
            uint32_t mask_w_1 = (mask_w < 16) ? 0 : mask_w - 16;
            uint32_t mask_h_1 = (mask_h < 16) ? 0 : mask_h - 16;
            uint32_t w = (h >= mask_h_1) ? 0 : mask_w_1;
            for (; w < 16; w++) {
                ptr[h * 16 + w + 768] = u16_zero;
            }
        }
    }
}

FORCE_INLINE void mask_tile_if_need(uint32_t l1_addr, uint32_t origin_h, uint32_t origin_w) {
    constexpr uint32_t TILE_H = 32;
    constexpr uint32_t TILE_W = 32;

    const bool do_mask_h = (origin_h % TILE_H) != 0;
    const uint32_t mask_h = do_mask_h ? (origin_h % TILE_H) : TILE_H;

    const bool do_mask_w = (origin_w % TILE_W) != 0;
    const uint32_t mask_w = do_mask_w ? (origin_w % TILE_W) : TILE_W;

    if (do_mask_h || do_mask_w) {
        mask_tile_hw(l1_addr, mask_h, mask_w);
    }
}
