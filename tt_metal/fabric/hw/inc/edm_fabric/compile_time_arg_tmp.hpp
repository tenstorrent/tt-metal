// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/hw/inc/compile_time_args.h"

////////////////////////////////////////
//  Conditional Arg Getters
////////////////////////////////////////
// clang-format off
/**
 * Initializes the transaction ID starts array where each element in the array is the start transaction ID for a
 * receiver channel
 *
 * Return value: constexpr std::array<uint8_t, NUM_RECEIVER_CHANNELS>
 *
 * | Template Argument                | Description                              | Type                  | Valid Range                        |
 * |----------------------------------|------------------------------------------|-----------------------|------------------------------------|
 * | NUM_RECEIVER_CHANNELS            | number of receiver channels              | size_t                | 1 to MAX_NUM_RECEIVER_CHANNELS - 1 |
 * | NUM_TRANSACTION_IDS_PER_CHANNEL  | number of transaction IDs per channel    | size_t                | 1 to 16                            |
 */
// clang-format on
template <size_t NUM_RECEIVER_CHANNELS, size_t NUM_TRANSACTION_IDS_PER_CHANNEL>
constexpr auto initialize_receiver_channel_trid_starts() -> std::array<uint8_t, NUM_RECEIVER_CHANNELS> {
    std::array<uint8_t, NUM_RECEIVER_CHANNELS> arr{};
    size_t trid_start = 0;
    for (size_t i = 0; i < NUM_RECEIVER_CHANNELS; i++) {
        arr[i] = trid_start;
        trid_start += NUM_TRANSACTION_IDS_PER_CHANNEL;
    }
    return arr;
}

// clang-format off
/**
 * constexpr function that will take the first n elements from an input array and assign those values
 * to a constexpr output array
 *
 * Return value: constexpr std::array<T, NUM_TO_TAKE>
 *
 * | Template Argument     | Description                              | Type                  | Valid Range      |
 * |-----------------------|------------------------------------------|-----------------------|------------------|
 * | NUM_TO_TAKE           | output array size in elem, also equal to | size_t                | 1 to max<size_t> |
 * |                       | the number of elements to take from the  |                       |                  |
 * |                       | input array                              |                       |                  |
 * | IN_ARRAY_SIZE         | size of input array, in elements         | size_t                | 1 to max<size_t> |
 * | T                     | element type                             | <typename>            | -----            |
 */
// clang-format on
template <size_t NUM_TO_TAKE, size_t IN_ARRAY_SIZE, typename T>
constexpr auto take_first_n_elements(const std::array<T, IN_ARRAY_SIZE>& arr_in) -> std::array<T, NUM_TO_TAKE> {
    std::array<T, NUM_TO_TAKE> arr{};
    for (size_t i = 0; i < NUM_TO_TAKE; i++) {
        arr[i] = arr_in[i];
    }
    return arr;
}

// clang-format off
/**
 * Fills an array at compile time with a given value
 *
 * Return value: constexpr std::array<T, N>
 *
 * | Template Argument     | Description              | Type                  | Valid Range      |
 * |-----------------------|--------------------------|-----------------------|------------------|
 * | N                     | array size in elements   | size_t                | 1 to max<size_t> |
 * | T                     | element type             | <typename>            | -----            |
 * | init                  | value to fill array with | T                     | -----            |
 */
// clang-format on
template <size_t N, typename T, T init>
constexpr auto initialize_array() -> std::array<T, N> {
    std::array<T, N> arr{};
    for (size_t i = 0; i < N; i++) {
        arr[i] = init;
    }
    return arr;
}

// clang-format off
/**
 * Helper type to copy a subset of a compile time args array into another array
 *
 * Return value: constexpr std::array<T, N>
 *
 * | Template Argument     | Description                         | Type                  | Valid Range      |
 * |-----------------------|-------------------------------------|-----------------------|------------------|
 * | T                     | element type                        | <typename>            | -----            |
 * | INPUT_ARR_START_IDX   | index of arg element to copy from   | size_t                | 0 to max<size_t> |
 * | NUM_ELEMS_TO_TAKE     | number of args to copy              | size_t                | 1 to max<size_t> |
 * | TAKEN_SO_FAR          | number of elements already copied   | size_t                | 0 to max<size_t> |
 * |                       | (internal use only)                 | size_t                | 0 to max<size_t> |
 */
// clang-format on
template <typename T, size_t INPUT_ARR_START_IDX, size_t NUM_ELEMS_TO_TAKE, size_t TAKEN_SO_FAR = 0>
struct ArraySliceCopier {
    static constexpr void copy(std::array<T, NUM_ELEMS_TO_TAKE>& arr) {
        // Fill the current element
        arr[TAKEN_SO_FAR] = static_cast<T>(get_compile_time_arg_val(INPUT_ARR_START_IDX + TAKEN_SO_FAR));

        // Recurse to fill the next element (if any)
        if constexpr (TAKEN_SO_FAR + 1 < NUM_ELEMS_TO_TAKE) {
            ArraySliceCopier<T, INPUT_ARR_START_IDX, NUM_ELEMS_TO_TAKE, TAKEN_SO_FAR + 1>::copy(arr);
        }
    }
};

// clang-format off
/**
 * Fills a compile time array with the next n compile time arguments
 *
 * Return value: constexpr std::array<T, NUM_ELEMS_TO_TAKE>
 *
 * | Template Argument     | Description              | Type                  | Valid Range      |
 * |-----------------------|--------------------------|-----------------------|------------------|
 * | ELEM_START_IDX        | index of first element to copy from | size_t                | 0 to max<size_t> |
 * | NUM_ELEMS_TO_TAKE     | number of elements to copy          | size_t                | 1 to max<size_t> |
 */
// clang-format on
template <typename T, size_t ELEM_START_IDX, size_t NUM_ELEMS_TO_TAKE>
constexpr auto fill_array_with_next_n_args() -> std::array<T, NUM_ELEMS_TO_TAKE> {
    std::array<T, NUM_ELEMS_TO_TAKE> arr{};
    if constexpr (NUM_ELEMS_TO_TAKE > 0) {
        ArraySliceCopier<T, ELEM_START_IDX, NUM_ELEMS_TO_TAKE>::copy(arr);
    }
    return arr;
}

// clang-format off
/**
 * Conditional compile time arg getter that is compile-time-safe in that it will not return an OoB access on a compile
 * time array if the arg is not supposed to be retrieved (indicated by GET_THE_ARG being false)
 * This helps with setup of dependent compile time args where some later compile time args are only conditionally consumed
 * if some earlier compile time arg is true
 *
 * This first function is the generic "true" path (we want to grab the arg)
 *
 * Return value: uint32_t
 *
 * | Template Argument     | Description              | Type                  | Valid Range      |
 * |-----------------------|--------------------------|-----------------------|------------------|
 * | GET_THE_ARG           | whether to get the arg   | bool                  | true or false    |
 * | ARG_IDX               | index of arg to get      | size_t                | 0 to max<size_t> |
 */
// clang-format on
template <bool GET_THE_ARG, size_t ARG_IDX>
constexpr auto conditional_get_compile_time_arg() -> typename std::enable_if<GET_THE_ARG, uint32_t>::type {
    return get_compile_time_arg_val(ARG_IDX);
}

// clang-format off
/**
 * SFINAE variant when GET_THE_ARG is false to avoid the lookup
 */
template <bool GET_THE_ARG, size_t ARG_IDX>
constexpr auto conditional_get_compile_time_arg() -> typename std::enable_if<!GET_THE_ARG, uint32_t>::type {
    return 0;
}
