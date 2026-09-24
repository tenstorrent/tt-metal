// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// To run (from the tt-metal repo root, after an emule build):
//   build_emule/test/tt_metal/unit_tests_api --gtest_filter="UnitMeshFixture.EmuleLlkDependency*"

// Low-level compute dependencies that fused kernels (e.g. a hand-written RMSNorm) name directly:
// the bare metal-layer SFPU header include and a raw math MOP template.

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "device_fixture.hpp"
#include "impl/program/program_impl.hpp"
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace tt::tt_metal {
namespace {

constexpr uint32_t kTileBytes = 2048;
constexpr uint32_t kBf16Ones = 0x3F803F80;
constexpr uint32_t kBf16Fours = 0x40804080;
constexpr uint32_t kBf16Halves = 0x3F003F00;
constexpr uint32_t kSentinel = 0x7FFF7FFF;

// Fills CB0 with one tile of a repeated word, then writes CB16 to DRAM.
constexpr std::string_view kDataflowSrc = R"(
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(0);
    const auto output = TensorAccessor(TensorAccessorArgs<1>(), get_arg_val<uint32_t>(0));
    cb_reserve_back(0, 1);
    auto* words = reinterpret_cast<volatile uint32_t tt_l1_ptr*>(get_write_ptr(0));
    for (uint32_t i = 0; i < tile_bytes / sizeof(uint32_t); ++i) {
        words[i] = get_arg_val<uint32_t>(1);
    }
    cb_push_back(0, 1);

    cb_wait_front(16, 1);
    noc_async_write_page(0, output, get_read_ptr(16), tile_bytes);
    noc_async_write_barrier();
    cb_pop_front(16, 1);
}
)";

// Copies CB0 to CB16 through DST[0]. `math_includes` land under TRISC_MATH; `before_copy_init`
// runs after hw startup, `on_dst` between copy_tile and pack_tile.
std::string copy_kernel(std::string_view math_includes, std::string_view before_copy_init, std::string_view on_dst) {
    std::string src = R"(
#include "api/compute/common.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#ifdef TRISC_MATH
)";
    src += math_includes;
    src += R"(
#endif

void kernel_main() {
    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);
#ifdef TRISC_MATH
)";
    src += before_copy_init;
    src += R"(
#endif
    copy_init(tt::CBIndex::c_0);
    cb_wait_front(tt::CBIndex::c_0, 1);
    cb_reserve_back(tt::CBIndex::c_16, 1);
    tile_regs_acquire();
    copy_tile(tt::CBIndex::c_0, 0, 0);
#ifdef TRISC_MATH
)";
    src += on_dst;
    src += R"(
#endif
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, tt::CBIndex::c_16);
    tile_regs_release();
    cb_pop_front(tt::CBIndex::c_0, 1);
    cb_push_back(tt::CBIndex::c_16, 1);
}
)";
    return src;
}

// Reduced ELWMUL schedule from a fused RMSNorm's math MOP setup.
constexpr std::string_view kMopIncludes = R"(
#include "ckernel_include.h"
#include "ckernel_ops.h"
)";
constexpr std::string_view kProgramElwmulMop = R"(
    ckernel::ckernel_template elwmul(
        1, 1, TT_OP_ELWMUL(0, 0, ckernel::p_elwise::SRCB_BCAST_ALL, 0, 0),
        TT_OP_ELWMUL(0, 0, ckernel::p_elwise::SRCB_BCAST_ALL, 2, 0));
    elwmul.program();
)";

std::vector<uint32_t> run_copy(distributed::MeshDevice& device, const std::string& compute_src, uint32_t input_word) {
    const CoreCoord core{0, 0};
    auto output = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = kTileBytes},
        {.page_size = kTileBytes, .buffer_type = BufferType::DRAM},
        &device);
    std::vector<uint32_t> result(kTileBytes / sizeof(uint32_t), kSentinel);
    slow_dispatch::WriteToBuffer(*output, result);

    Program program = CreateProgram();
    for (uint32_t cb : {0U, 16U}) {
        CreateCircularBuffer(
            program,
            core,
            CircularBufferConfig(kTileBytes, {{cb, tt::DataFormat::Float16_b}}).set_page_size(cb, kTileBytes));
    }
    std::vector<uint32_t> compile_args{kTileBytes};
    TensorAccessorArgs(*output).append_to(compile_args);
    auto dataflow = CreateKernelFromString(
        program,
        std::string(kDataflowSrc),
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = compile_args});
    CreateKernelFromString(program, compute_src, core, ComputeConfig{.math_fidelity = MathFidelity::HiFi4});
    SetRuntimeArgs(program, dataflow, core, {static_cast<uint32_t>(output->address()), input_word});

    LaunchProgram(device, std::move(program));
    slow_dispatch::ReadFromBuffer(*output, result);
    return result;
}

void expect_every_word(const std::vector<uint32_t>& result, uint32_t expected) {
    ASSERT_EQ(result.size(), kTileBytes / sizeof(uint32_t));
    for (size_t i = 0; i < result.size(); ++i) {
        ASSERT_EQ(result[i], expected) << "word " << i;
    }
}

}  // namespace

// The bare include resolves as on silicon's -I set, and its calculate_rsqrt runs on DST.
TEST_F(UnitMeshFixture, EmuleLlkDependencyBareRsqrtHeaderRunsRsqrt) {
    const std::string src = copy_kernel(
        R"(
#include "ckernel_sfpu_rsqrt.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
)",
        "",
        R"(
    ckernel::sfpu::rsqrt_init<false, false>();
    _llk_math_eltwise_unary_sfpu_params_<false>(
        ckernel::sfpu::calculate_rsqrt<false, 8, false, false, false>, 0,
        static_cast<int>(ckernel::VectorMode::RC));
)");
    expect_every_word(run_copy(this->device(), src, kBf16Fours), kBf16Halves);
}

// Programming a raw math MOP is harmless when the next op's init reprograms the MOP.
TEST_F(UnitMeshFixture, EmuleLlkDependencyRawMopIsReprogrammedByCopyInit) {
    const std::string src = copy_kernel(kMopIncludes, kProgramElwmulMop, "");
    expect_every_word(run_copy(this->device(), src, kBf16Ones), kBf16Ones);
}

// Running a raw MOP schedule is not modeled, so it must abort rather than write DST.
TEST_F(UnitMeshFixture, EmuleLlkDependencyRawMopRunAborts) {
    const std::string src =
        copy_kernel(kMopIncludes, std::string(kProgramElwmulMop) + "    ckernel::ckernel_template::run();\n", "");
    EXPECT_DEATH(
        run_copy(this->device(), src, kBf16Ones),
        "ckernel_template::run\\(\\): executing a raw MOP schedule is not modeled");
}

}  // namespace tt::tt_metal
