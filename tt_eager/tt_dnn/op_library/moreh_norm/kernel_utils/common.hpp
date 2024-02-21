// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

void fill_cb_with_value(uint32_t cb_id, uint32_t value) {
    cb_reserve_back(cb_id, 1);
    auto ptr = reinterpret_cast<uint16_t *>(get_write_ptr(cb_id));
    for (int j = 0; j < 1024; j++) {
        ptr[j] = uint16_t(value >> 16);
    }
    cb_push_back(cb_id, 1);
}

union Scalar {
    float f;
    uint32_t u;
};

void generate_mask_w(uint32_t cb_mask, uint32_t mask_w) {
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

void generate_mask_h(uint32_t cb_mask, uint32_t mask_h) {
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
