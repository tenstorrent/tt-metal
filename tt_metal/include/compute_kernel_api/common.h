// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"
#include "compute_kernel_api/reg_api.h"
#include "compute_kernel_api/pack.h"
#include "compute_kernel_api/reconfig_data_format.h"
#include "compute_kernel_api/cb_api.h"
#include "compute_kernel_api/compute_kernel_hw_startup.h"

#ifdef ARCH_QUASAR
extern thread_local uint32_t tt_l1_ptr* rta_l1_base;
extern thread_local uint32_t tt_l1_ptr* crta_l1_base;
#else
extern uint32_t tt_l1_ptr* rta_l1_base;
extern uint32_t tt_l1_ptr* crta_l1_base;
#endif

// clang-format off
/**
 * Returns the address in L1 for a given runtime argument index for unique (per core) runtime arguments set via
 * SetRuntimeArgs() API.
 *
 * Return value: Associated L1 address of given unique runtime argument index
 *
 * | Argument       | Description                                                             | Type     | Valid Range | Required |
 * |----------------|-------------------------------------------------------------------------|----------|-------------|----------|
 * | arg_idx        | Unique Runtime argument index                                           | uint32_t | 0 to 341    | True     |
 */
// clang-format on
static FORCE_INLINE uint32_t get_arg_addr(int arg_idx) { return (uint32_t)&rta_l1_base[arg_idx]; }

// clang-format off
/**
 * Returns the address in L1 for a given runtime argument index for common (all cores) runtime arguments set via
 * SetCommonRuntimeArgs() API.
 *
 * Return value: Associated L1 address of given common runtime argument index
 *
 * | Argument       | Description                                                             | Type     | Valid Range | Required |
 * |----------------|-------------------------------------------------------------------------|----------|-------------|----------|
 * | arg_idx        | Common Runtime argument index                                           | uint32_t | 0 to 341    | True     |
 */
// clang-format on
static FORCE_INLINE uint32_t get_common_arg_addr(int arg_idx) { return (uint32_t)&crta_l1_base[arg_idx]; }

// clang-format off
/**
 * Returns the value at a given runtime argument index for unique (per-core) runtime arguments set via SetRuntimeArgs()
 * API.
 *
 * Return value: The value associated with the unique runtime argument index
 *
 * | Argument              | Description                                    | Type                  | Valid Range | Required |
 * |-----------------------|------------------------------------------------|-----------------------|-------------|----------|
 * | arg_idx               | Unique Runtime argument index                  | uint32_t              | 0 to 341    | True     |
 * | T (template argument) | Data type of the returned argument             | Any 4-byte sized type | N/A         | True     |
 */
// clang-format on
template <typename T>
FORCE_INLINE T get_arg_val(int arg_idx) {
    // only 4B args are supported (eg int32, uint32)
    static_assert("Error: only 4B args are supported" && sizeof(T) == 4);
    return *((tt_l1_ptr T*)(get_arg_addr(arg_idx)));
}

// clang-format off
/**
 * Returns the value at a given runtime argument index for common (all cores) runtime arguments set via
 * SetCommonRuntimeArgs() API.
 *
 * Return value: The value associated with the common runtime argument index
 *
 * | Argument              | Description                                    | Type                  | Valid Range | Required |
 * |-----------------------|------------------------------------------------|-----------------------|-------------|----------|
 * | arg_idx               | Common Runtime argument index                  | uint32_t              | 0 to 341    | True     |
 * | T (template argument) | Data type of the returned argument             | Any 4-byte sized type | N/A         | True     |
 */
// clang-format on
template <typename T>
FORCE_INLINE T get_common_arg_val(int arg_idx) {
    // only 4B args are supported (eg int32, uint32)
    static_assert("Error: only 4B args are supported" && sizeof(T) == 4);
    return *((tt_l1_ptr T*)(get_common_arg_addr(arg_idx)));
}

// clang-format off
/**
 * Returns the absolute logical X coordinate value that this kernel is running on. The absolute coordinate
 * is the one relative to the origin of the physical grid.
 *
 * Return value: X coordinate value.
 */
// clang-format on
inline uint8_t get_absolute_logical_x() {
    extern uint8_t my_logical_x_;  // Set in FW
    return my_logical_x_;
}

// clang-format off
/**
 * Returns the absolute logical Y coordinate value that this kernel is running on. The absolute coordinate
 * is the one relative to the origin of the physical grid.
 *
 * Return value: Y coordinate value.
 */
// clang-format on
inline uint8_t get_absolute_logical_y() {
    extern uint8_t my_logical_y_;  // Set in FW
    return my_logical_y_;
}

// clang-format off
/**
 * Returns the relative logical X coordinate value that this kernel is running on. The relative coordinate
 * is with respect to the origin of the sub device for this core type.
 *
 * Return value: X coordinate value.
 */
// clang-format on
inline uint8_t get_relative_logical_x() {
    extern uint8_t my_relative_x_;  // Set in FW
    return my_relative_x_;
}

// clang-format off
/**
 * Returns the relative logical Y coordinate value that this kernel is running on. The relative coordinate
 * is with respect to the origin of the sub device for this core type.
 *
 * Return value: Y coordinate value.
 */
// clang-format on
inline uint8_t get_relative_logical_y() {
    extern uint8_t my_relative_y_;  // Set in FW
    return my_relative_y_;
}
