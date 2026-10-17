// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compile coverage for the single-operand reduce wrappers llk_unpack_reduce_init / llk_unpack_reduce
// (hw/ckernels/<arch>/metal/llk_api/llk_unpack_reduce_api.h). No kernel in tree instantiates them and the
// tt-llk test drives the _llk_unpack_reduce_* functions directly, so the wrapper's scaler CB address math,
// scaler tile index, format selection and asserts (CB bounds, 32x32-only, SrcA/SrcB unpacker state against the
// data and scaler CBs) are only ever compiled here. The program is compiled, not run, so the asserts are not
// exercised at runtime.

#include <cstdint>

#include <gtest/gtest.h>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "impl/program/program_impl.hpp"
#include "llk_device_fixture.hpp"

namespace tt::tt_metal {

TEST_F(LLKMeshDeviceSingleCardFixture, TensixUnpackReduceScalerCbWrappersCompile) {
    if (this->device().arch() == tt::ARCH::QUASAR) {
        GTEST_SKIP() << "Quasar has no single-operand unpack reduce wrapper";
    }

    Program program = CreateProgram();
    const CoreCoord core{0, 0};
    constexpr uint32_t tile_bytes = 2 * 32 * 32;
    constexpr uint32_t num_tiles = 2;
    for (uint32_t cb : {0u, 1u, 16u}) {
        CreateCircularBuffer(
            program,
            core,
            CircularBufferConfig(num_tiles * tile_bytes, {{cb, tt::DataFormat::Float16_b}})
                .set_page_size(cb, tile_bytes));
    }
    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/unpack_reduce_scaler_cb.cpp",
        core,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .compile_args = {num_tiles}});

    EXPECT_NO_THROW(program.impl().compile(&this->device()));
}

}  // namespace tt::tt_metal
