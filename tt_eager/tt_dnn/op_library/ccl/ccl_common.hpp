// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "debug/assert.h"

namespace ccl {

// TODO(snijjar): Clean this up (mainly dependence on device ASSERT)
// so I can include this from host side as well
// template <typename ... Args>
// class ArgListConstructible {
//    public:
//    void build_with_placement_new(Args &&... args) = 0;
//    // do placement new construction with static build methods
//     ArgListConstructible(uint16_t num_args_consumed) : num_args_consumed(num_args_consumed) {
//         ASSERT(num_args_consumed >= 0);//, "num_args_consumed not set")
//     }

//     int16_t get_num_args_consumed() const {
//         ASSERT(num_args_consumed >= 0);//, "num_args_consumed not set")
//         return num_args_consumed;
//     }

//    protected:
//     int16_t set_num_args_consumed(uint16_t num_args) {
//         num_args_consumed = num_args;
//     }

//     int16_t num_args_consumed;
// };

} // namespace ccl
