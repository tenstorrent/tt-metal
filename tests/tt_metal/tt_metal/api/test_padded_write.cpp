// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// To run:
// $ROOT/tt-metal/build_emule/test/tt_metal/unit_tests_api --gtest_filter="MeshDeviceFixture.Tensor_Padding_Violation_*"

#include <gtest/gtest.h>
#include <cstdint>
#include <string>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/core_coord.hpp>
#include "device_fixture.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace tt::tt_metal {

namespace {

// nfaces byte offset of element (row, col) within a 32x32 tile stored as 4 16x16
// faces (face 0 TL, 1 TR, 2 BL, 3 BR), row-major within a face. Mirrors
// include/jit_hw/nfaces.h's rowmajor_to_nfaces so the test addresses the exact
// byte the kernel-side check decodes.
uint32_t nfaces_byte(uint32_t row, uint32_t col, uint32_t elem_size) {
    uint32_t face = (row >= 16 ? 2u : 0u) + (col >= 16 ? 1u : 0u);
    uint32_t ni = face * 256 + (row % 16) * 16 + (col % 16);
    return ni * elem_size;
}

// Builds a one-DM-kernel program that resolves a pointer to `byte_offset` inside
// `buffer` (running the kernel-side padding check on that address) and writes one
// byte. The translate-then-write mirrors how reader/writer kernels touch L1.
Program make_write_program(const std::shared_ptr<Buffer>& buffer, CoreCoord logical_core, uint32_t byte_offset) {
    Program program = CreateProgram();
    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {
            uint32_t base_addr = get_arg_val<uint32_t>(0);
            uint32_t byte_offset = get_arg_val<uint32_t>(1);
            volatile uint8_t* p = (volatile uint8_t*)__emule_local_l1_to_ptr(base_addr + byte_offset);
            *p = 0xAA;
        }
    )";
    auto kernel = CreateKernelFromString(
        program,
        kernel_src,
        logical_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    SetRuntimeArgs(program, kernel, logical_core, {buffer->address(), byte_offset});
    return program;
}

}  // namespace

// 1-D back-compat: a single trailing pad band declared via set_logical_size. The
// kernel writes just past the logical extent → must abort.
TEST_F(MeshDeviceFixture, Tensor_Padding_Violation_Linear_SanityCheck) {
    ::setenv("TT_METAL_EMULE_ASAN", "1", 1);
    auto* device = this->devices_.at(0)->get_devices()[0];
    CoreCoord logical_core = {0, 0};

    uint32_t logical_size = 1024;
    uint32_t physical_size = 2048;
    auto buffer = Buffer::create(device, physical_size, physical_size, BufferType::L1);
    buffer->set_logical_size(logical_size);

    // byte 1028 is inside the allocation (passes OOB) but past the 1024-byte
    // logical extent (in the trailing pad band).
    Program program = make_write_program(buffer, logical_core, logical_size + 4);
    EXPECT_DEATH(
        detail::LaunchProgram(device, program),
        ".*Tensor Padding Violation: Attempted to write to a padded memory region at address 0x.*");
}

// 2-D / tiled: a 20x20 logical tensor padded up to a single 32x32 tile (fp32).
// The padding is an "L": cols 20..31 of every data row PLUS rows 20..31. This is
// exactly the interspersed padding the old single-trailing-band model could not
// represent. Write into the right-edge strip of a *data* row (row 5, col 25) —
// the old check missed this; the 2-D check must abort.
TEST_F(MeshDeviceFixture, Tensor_Padding_Violation_Tiled_RightEdge) {
    ::setenv("TT_METAL_EMULE_ASAN", "1", 1);
    auto* device = this->devices_.at(0)->get_devices()[0];
    CoreCoord logical_core = {0, 0};

    uint32_t elem_size = 4;                     // fp32
    uint32_t tile_bytes = 32 * 32 * elem_size;  // 4096
    auto buffer = Buffer::create(device, tile_bytes, tile_bytes, BufferType::L1);
    buffer->set_padded_layout(
        Buffer::PaddingLayout::Tile,
        elem_size,
        /*logical_rows=*/20,
        /*logical_cols=*/20,
        /*padded_cols=*/32,
        /*padded_page_rows=*/32);

    // row 5 (< 20, a data row), col 25 (>= 20, the right-edge pad strip).
    uint32_t off = nfaces_byte(5, 25, elem_size);
    Program program = make_write_program(buffer, logical_core, off);
    EXPECT_DEATH(
        detail::LaunchProgram(device, program),
        ".*Tensor Padding Violation: Attempted to write to a padded memory region at address 0x.*");
}

// Same tiled layout, but write into a genuine data element (row 5, col 10, inside
// the 20x20 logical region) — must NOT abort. Positive control: guards against the
// 2-D math over-firing on legitimate writes.
TEST_F(MeshDeviceFixture, Tensor_Padding_Violation_Tiled_NoViolation) {
    ::setenv("TT_METAL_EMULE_ASAN", "1", 1);
    auto* device = this->devices_.at(0)->get_devices()[0];
    CoreCoord logical_core = {0, 0};

    uint32_t elem_size = 4;
    uint32_t tile_bytes = 32 * 32 * elem_size;
    auto buffer = Buffer::create(device, tile_bytes, tile_bytes, BufferType::L1);
    buffer->set_padded_layout(
        Buffer::PaddingLayout::Tile,
        elem_size,
        /*logical_rows=*/20,
        /*logical_cols=*/20,
        /*padded_cols=*/32,
        /*padded_page_rows=*/32);

    uint32_t off = nfaces_byte(5, 10, elem_size);  // row 5, col 10 — data
    Program program = make_write_program(buffer, logical_core, off);
    detail::LaunchProgram(device, program);
    SUCCEED();
}

// 3-D / batched: a [3, 30, 30] tensor padded to [3, 32, 32] — three stacked tile
// pages, each a 30x30 logical region in a 32x32 tile (so each page has the same
// "L" of right-edge + trailing pad). padded_page_rows = 32 makes the row test reset
// per page. THE regression: page 1's first data row sits at global row 32, which
// the old single-global-row model (logical_rows=30) wrongly flagged as padding.
// Writing real data on page 1 (row 5, col 10 within the page) must NOT abort.
TEST_F(MeshDeviceFixture, Tensor_Padding_Violation_Tiled_3D_Page1Data_NoViolation) {
    ::setenv("TT_METAL_EMULE_ASAN", "1", 1);
    auto* device = this->devices_.at(0)->get_devices()[0];
    CoreCoord logical_core = {0, 0};

    uint32_t elem_size = 4;
    uint32_t tile_bytes = 32 * 32 * elem_size;  // one page = one 32x32 tile
    uint32_t num_pages = 3;
    auto buffer = Buffer::create(device, num_pages * tile_bytes, tile_bytes, BufferType::L1);
    buffer->set_padded_layout(
        Buffer::PaddingLayout::Tile,
        elem_size,
        /*logical_rows=*/30,
        /*logical_cols=*/30,
        /*padded_cols=*/32,
        /*padded_page_rows=*/32);

    // page 1, within-page (row 5, col 10) — genuine data.
    uint32_t off = 1 * tile_bytes + nfaces_byte(5, 10, elem_size);
    Program program = make_write_program(buffer, logical_core, off);
    detail::LaunchProgram(device, program);
    SUCCEED();
}

// Same [3, 30, 30]->[3, 32, 32] layout: write into page 1's right-edge pad strip
// (within-page row 5, col 30) — must abort. Confirms the per-page reset still flags
// real padding on a non-zero page.
TEST_F(MeshDeviceFixture, Tensor_Padding_Violation_Tiled_3D_Page1Pad) {
    ::setenv("TT_METAL_EMULE_ASAN", "1", 1);
    auto* device = this->devices_.at(0)->get_devices()[0];
    CoreCoord logical_core = {0, 0};

    uint32_t elem_size = 4;
    uint32_t tile_bytes = 32 * 32 * elem_size;
    uint32_t num_pages = 3;
    auto buffer = Buffer::create(device, num_pages * tile_bytes, tile_bytes, BufferType::L1);
    buffer->set_padded_layout(
        Buffer::PaddingLayout::Tile,
        elem_size,
        /*logical_rows=*/30,
        /*logical_cols=*/30,
        /*padded_cols=*/32,
        /*padded_page_rows=*/32);

    // page 1, within-page (row 5, col 30) — col 30 >= logical_cols(30) → padding.
    uint32_t off = 1 * tile_bytes + nfaces_byte(5, 30, elem_size);
    Program program = make_write_program(buffer, logical_core, off);
    EXPECT_DEATH(
        detail::LaunchProgram(device, program),
        ".*Tensor Padding Violation: Attempted to write to a padded memory region at address 0x.*");
}

}  // namespace tt::tt_metal
