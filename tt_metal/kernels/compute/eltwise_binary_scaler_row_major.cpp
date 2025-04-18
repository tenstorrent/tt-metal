// tt_metal/kernels/compute/eltwise_binary_scalar_row_major.cpp
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/common.h"  // Includes basic types like bfloat16 and L1 pointers
#include "debug/dprint.h"               // Optional: Include for debugging prints if needed

// This kernel performs element-wise binary operation: DST = SRC + scalar
// It assumes input and output data are in RowMajor format in L1 memory.
// It expects data to be bfloat16.

namespace NAMESPACE {
void MAIN {
    // Argument IDs
    constexpr uint32_t SRC_ADDR_ARG_ID = 0;
    constexpr uint32_t DST_ADDR_ARG_ID = 1;
    constexpr uint32_t NUM_ELEMENTS_ARG_ID = 2;
    constexpr uint32_t SCALAR_ARG_ID = 3;

    // Get kernel arguments using tt_metal API
    const uint32_t src_addr = get_arg_val<uint32_t>(SRC_ADDR_ARG_ID);          // Input tensor L1 address (RowMajor)
    const uint32_t dst_addr = get_arg_val<uint32_t>(DST_ADDR_ARG_ID);          // Output tensor L1 address (RowMajor)
    const uint32_t num_elements = get_arg_val<uint32_t>(NUM_ELEMENTS_ARG_ID);  // Total number of elements in the tensor
    const uint32_t scalar_value_uint32 =
        get_arg_val<uint32_t>(SCALAR_ARG_ID);  // Scalar value (packed as float bits in uint32_t by host)

    // Get pointers to L1 memory
    // Using `uint16_t` for pointer arithmetic as bfloat16 size is 2 bytes.
    // The reinterpret_cast will handle the type conversion for access.
    volatile tt_l1_ptr uint16_t* src_ptr_raw = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(src_addr);
    volatile tt_l1_ptr uint16_t* dst_ptr_raw = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(dst_addr);

    // Unpack the scalar value
    // Assumes host packed the float bits directly into the uint32_t argument.
    float scalar_f32 = *reinterpret_cast<float*>(&scalar_value_uint32);
    bfloat16 scalar_bf16 = bfloat16(scalar_f32);  // Convert float to bfloat16

    // Perform element-wise addition
    // This loop iterates through the RowMajor data directly using element count.
    for (uint32_t i = 0; i < num_elements; ++i) {
        // Read element from source using the raw uint16_t pointer and cast
        uint16_t src_val_raw = src_ptr_raw[i];
        bfloat16 src_val = *reinterpret_cast<bfloat16*>(&src_val_raw);  // Cast the raw bits to bfloat16

        // Perform addition (using bfloat16's overloaded operator+)
        bfloat16 result_bf16 = src_val + scalar_bf16;

        // Write result to destination using the raw uint16_t pointer
        // Cast the result bfloat16 back to uint16_t for writing
        dst_ptr_raw[i] = *reinterpret_cast<uint16_t*>(&result_bf16);
    }

}  // MAIN
}  // namespace NAMESPACE
