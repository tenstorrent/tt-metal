// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// To run (from the tt-metal repo root, after an emule build):
//   build_emule/test/tt_metal/unit_tests_api --gtest_filter="UnitMeshFixture.TensixGpr*"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "device_fixture.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace tt::tt_metal {
namespace {

constexpr uint32_t kTileBytes = 2048;
constexpr uint32_t kBf16OnesWord = 0x3F803F80;

// Fills CB0 with one tile of BF16 ones, then drains CB16 to DRAM.
const char* kDataflowSrc = R"(
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

// Copies the tile only if every GPR read-back matches, so a wrong GPR model shows up as
// a non-ones output. drainComputeEngine is TT-Lang's completion handshake, unchanged.
const char* kComputeSrc = R"(
    #include "api/compute/common.h"
    #include "api/compute/pack.h"
    #include "api/compute/tile_move_copy.h"
    #include "api/compute/eltwise_unary/eltwise_unary.h"

    #if defined(UCK_CHLKC_UNPACK) || defined(TRISC_UNPACK)
    #define TTL_DFB_RECONFIGURATION_UNPACK
    #endif
    #if defined(UCK_CHLKC_PACK) || defined(TRISC_PACK)
    #define TTL_DFB_RECONFIGURATION_PACK
    #endif
    constexpr uint32_t completionMarker = 0xD1FB;

    FORCE_INLINE void drainComputeEngine() {
    #if defined(TTL_DFB_RECONFIGURATION_UNPACK)
      constexpr uint32_t waitResources = p_stall::UNPACK;
      constexpr uint32_t completionGpr = p_gpr_unpack::TMP0;
    #elif defined(TTL_DFB_RECONFIGURATION_PACK)
      constexpr uint32_t waitResources = p_stall::PACK;
      constexpr uint32_t completionGpr = p_gpr_pack::TMP0;
    #endif
    #if defined(TTL_DFB_RECONFIGURATION_UNPACK) || defined(TTL_DFB_RECONFIGURATION_PACK)
      TTI_STALLWAIT(p_stall::STALL_TDMA, waitResources);
      TTI_SETDMAREG(0, completionMarker, 0, LO_16(completionGpr));
      sync_regfile_write(completionGpr);
    #endif
    }

    bool gpr_writes_land() {
        // Two half writes compose one word; a marker whose top bits spill into the
        // Payload_SigSelSize field still lands as 16 bits and leaves the high half alone.
        constexpr uint32_t word = p_gpr_pack::TMP1;
        constexpr uint32_t half = p_gpr_pack::TMP_HI;
        TT_SETDMAREG(0, 0x3F80, 0, HI_16(word));
        TTI_SETDMAREG(0, 0x3F80, 0, LO_16(word));
        TTI_SETDMAREG(0, 0xBEEF, 0, HI_16(half));
        TTI_SETDMAREG(0, completionMarker, 0, LO_16(half));
        sync_regfile_write(half);
        return regfile[word] == 0x3F803F80u && regfile[half] == ((0xBEEFu << 16) | completionMarker);
    }

    void kernel_main() {
        const bool landed = gpr_writes_land();
        compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);
        copy_init(tt::CBIndex::c_0);
        tile_regs_acquire();
        cb_wait_front(tt::CBIndex::c_0, 1);
        cb_reserve_back(tt::CBIndex::c_16, 1);
        if (landed) {
            copy_tile(tt::CBIndex::c_0, 0, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, tt::CBIndex::c_16);
        drainComputeEngine();
        cb_pop_front(tt::CBIndex::c_0, 1);
        cb_push_back(tt::CBIndex::c_16, 1);
        tile_regs_release();
    }
)";

}  // namespace

// A compute kernel writes Tensix GPRs with SETDMAREG and reads them back through
// regfile / sync_regfile_write, as TT-Lang's completion handshake does on silicon.
TEST_F(UnitMeshFixture, TensixGprSetDmaRegReadBack) {
    auto mesh_device = this->devices_.front();
    const CoreCoord core(0, 0);

    auto output = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = kTileBytes},
        {.page_size = kTileBytes, .buffer_type = BufferType::DRAM},
        mesh_device.get());
    std::vector<uint32_t> result(kTileBytes / sizeof(uint32_t), 0x7FFF7FFF);
    this->WriteBuffer(mesh_device, output, result);

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
        kDataflowSrc,
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = compile_args});
    CreateKernelFromString(program, kComputeSrc, core, ComputeConfig{});
    SetRuntimeArgs(program, dataflow, core, {static_cast<uint32_t>(output->address()), kBf16OnesWord});

    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));
    this->RunProgram(mesh_device, workload);
    this->ReadBuffer(mesh_device, output, result);

    EXPECT_EQ(result, std::vector<uint32_t>(kTileBytes / sizeof(uint32_t), kBf16OnesWord));
}

}  // namespace tt::tt_metal
