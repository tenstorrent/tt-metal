#include "../basic_harness.hpp"
#include "command_queue_test_utils.hpp"
#include "gtest/gtest.h"
#include "tt_metal/common/bfloat16.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt::tt_metal;

struct CBConfig {
    u32 num_pages;
    u32 page_size;
    tt::DataFormat data_format;
};

struct DummyProgramConfig {
    CoreRangeSet cr_set;
    CBConfig cb_config;
    u32 num_cbs;
    u32 num_sems;
    u32 first_cb_start;
};

namespace local_test_functions {

void initialize_dummy_kernels(Program& program, const CoreRangeSet& cr_set) {
    auto dummy_reader_kernel = CreateDataMovementKernel(
        program, "tt_metal/kernels/dataflow/blank.cpp", cr_set, DataMovementProcessor::RISCV_1, NOC::RISCV_1_default);

    auto dummy_writer_kernel = CreateDataMovementKernel(
        program, "tt_metal/kernels/dataflow/blank.cpp", cr_set, DataMovementProcessor::RISCV_0, NOC::RISCV_0_default);

    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    auto dummy_compute_kernel = CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/blank.cpp",
        cr_set,
        {},
        MathFidelity::HiFi4,
        fp32_dest_acc_en,
        math_approx_mode);

    dummy_writer_kernel->add_define("TT_METAL_DEVICE_DISPATCH_MODE", "1");
}

bool test_dummy_EnqueueProgram_with_cbs(Device* device, CommandQueue& cq, const DummyProgramConfig& program_config) {
    bool pass = true;

    Program program;

    u32 cb_num_pages = program_config.cb_config.num_pages;
    u32 cb_size = program_config.cb_config.num_pages * program_config.cb_config.page_size;
    u32 cb_addr = program_config.first_cb_start;

    for (u32 cb_id = 0; cb_id < program_config.num_cbs; cb_id++) {
        auto cb = CreateCircularBuffers(
            program,
            cb_id,
            program_config.cr_set,
            cb_num_pages,
            cb_size,
            program_config.cb_config.data_format,
            cb_addr);
        cb_addr += cb_size;
    }

    initialize_dummy_kernels(program, program_config.cr_set);
    CompileProgram(device, program);
    EnqueueProgram(cq, program, false);
    Finish(cq);

    // Need to use old APIs to read since we cannot allocate a buffer in the reserved space we're trying
    // to read from
    vector<u32> cb_config_vector;
    u32 cb_config_buffer_size = program_config.num_cbs * UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG * sizeof(u32);

    cb_addr = program_config.first_cb_start;
    for (const CoreRange& core_range : program_config.cr_set.ranges()) {
        CoresInCoreRangeGenerator core_range_generator(core_range, device->cluster()->get_soc_desc(0).worker_grid_size);

        bool terminate;
        do {
            auto [core_coord, terminate_] = core_range_generator();

            terminate = terminate_;
            tt::tt_metal::ReadFromDeviceL1(
                device, core_coord, CIRCULAR_BUFFER_CONFIG_BASE, cb_config_buffer_size, cb_config_vector);

            u32 cb_id = 0;
            for (u32 i = 0; i < cb_config_vector.size(); i += sizeof(u32)) {
                bool addr_match = cb_config_vector.at(i) == ((cb_addr + cb_size * cb_id) >> 4);
                cb_id++;
                bool size_match = cb_config_vector.at(i + 1) == (cb_size >> 4);
                bool num_pages_match = cb_config_vector.at(i + 2) == cb_num_pages;

                pass &= (addr_match and size_match and num_pages_match);
            }

        } while (not terminate);
    }

    return pass;
}

bool test_dummy_EnqueueProgram_with_sems(Device* device, CommandQueue& cq, const DummyProgramConfig& program_config) {
    bool pass = true;

    Program program;

    for (u32 sem_id = 0; sem_id < program_config.num_sems; sem_id++) {
        auto sem = CreateSemaphore(program, program_config.cr_set, sem_id);
    }

    CompileProgram(device, program);
    EnqueueProgram(cq, program, false);
    Finish(cq);

    vector<u32> sem_vector;
    u32 sem_buffer_size = program_config.num_sems * SEMAPHORE_ALIGNMENT;

    for (const CoreRange& core_range : program_config.cr_set.ranges()) {
        CoresInCoreRangeGenerator core_range_generator(core_range, device->cluster()->get_soc_desc(0).worker_grid_size);

        bool terminate;
        do {
            auto [core_coord, terminate_] = core_range_generator();

            terminate = terminate_;
            tt::tt_metal::ReadFromDeviceL1(device, core_coord, SEMAPHORE_BASE, sem_buffer_size, sem_vector);

            u32 sem_id = 0;
            for (u32 i = 0; i < sem_vector.size(); i += sizeof(u32)) {
                bool sem_match = sem_vector.at(i) == sem_id;
                sem_id++;

                pass &= sem_match;
            }
        } while (not terminate);
    }

    return pass;
}

bool test_EnqueueWrap_on_EnqueueWriteBuffer(Device* device, CommandQueue& cq, const BufferConfig& config) {
    EnqueueWriteBuffer_prior_to_wrap(device, cq, config);

    /*
    This just ensures we don't hang on the subsequent EnqueueWriteBuffer
    */
    size_t buf_size = config.num_pages * config.page_size;
    Buffer buffer(device, buf_size, config.page_size, config.buftype);

    vector<u32> src(buf_size / sizeof(u32), 0);

    for (u32 i = 0; i < src.size(); i++) {
        src.at(i) = i;
    }
    EnqueueWriteBuffer(cq, buffer, src, false);
    Finish(cq);

    return true;
}

bool test_EnqueueWrap_on_Finish(Device* device, CommandQueue& cq, const BufferConfig& config) {
    bool pass = true;
    EnqueueWriteBuffer_prior_to_wrap(device, cq, config);

    return pass;
}

bool test_EnqueueWrap_on_EnqueueProgram(Device* device, CommandQueue& cq, const BufferConfig& config) {
    bool pass = true;
    EnqueueWriteBuffer_prior_to_wrap(device, cq, config);

    return pass;
}

}  // namespace local_test_functions

namespace basic_tests {
namespace single_core_tests {

TEST_F(CommandQueueHarness, TestSingleCbConfigCorrectlySentSingleCore) {
    CoreRange cr = {.start = {0, 0}, .end = {0, 0}};
    CoreRangeSet cr_set({cr});

    CBConfig cb_config = {.num_pages = 1, .page_size = 2048, .data_format = tt::DataFormat::Float16_b};

    DummyProgramConfig config = {.cr_set = cr_set, .cb_config = cb_config, .num_cbs = 1, .first_cb_start = 500 * 1024};

    EXPECT_TRUE(local_test_functions::test_dummy_EnqueueProgram_with_cbs(this->device, *this->cq, config));
}

TEST_F(CommandQueueHarness, TestSingleSemaphoreConfigCorrectlySentSingleCore) {
    CoreRange cr = {.start = {0, 0}, .end = {0, 0}};
    CoreRangeSet cr_set({cr});

    DummyProgramConfig config = {.cr_set = cr_set, .num_sems = 1};

    EXPECT_TRUE(local_test_functions::test_dummy_EnqueueProgram_with_sems(this->device, *this->cq, config));
}

TEST_F(CommandQueueHarness, TestAutoInsertedBlankBriscKernelInDeviceDispatchMode) {
    char env[] = "TT_METAL_DEVICE_DISPATCH_MODE=1";
    putenv(env);
    Program program;

    CoreRange cr = {.start = {0, 0}, .end = {0, 0}};
    CoreRangeSet cr_set({cr});
    // Add an NCRISC blank manually, but in compile program, the BRISC blank will be
    // added separately
    auto dummy_reader_kernel = CreateDataMovementKernel(
        program, "tt_metal/kernels/dataflow/blank.cpp", cr_set, DataMovementProcessor::RISCV_1, NOC::RISCV_1_default);

    CompileProgram(this->device, program, false);

    EnqueueProgram(*this->cq, program, false);
    Finish(*this->cq);
}

}  // end namespace single_core_tests

namespace multicore_tests {
TEST_F(CommandQueueHarness, TestAllCbConfigsCorrectlySentMultiCore) {
    CoreCoord worker_grid_size = this->device->cluster()->get_soc_desc(0).worker_grid_size;

    CoreRange cr = {.start = {0, 0}, .end = {worker_grid_size.x - 1, worker_grid_size.y - 2}};
    CoreRangeSet cr_set({cr});

    CBConfig cb_config = {.num_pages = 1, .page_size = 2048, .data_format = tt::DataFormat::Float16_b};

    DummyProgramConfig config = {
        .cr_set = cr_set, .cb_config = cb_config, .num_cbs = NUM_CIRCULAR_BUFFERS, .first_cb_start = 500 * 1024};

    EXPECT_TRUE(local_test_functions::test_dummy_EnqueueProgram_with_cbs(this->device, *this->cq, config));
}

TEST_F(CommandQueueHarness, TestAllSemConfigsCorrectlySentMultiCore) {
    CoreCoord worker_grid_size = this->device->cluster()->get_soc_desc(0).worker_grid_size;

    CoreRange cr = {.start = {0, 0}, .end = {worker_grid_size.x - 1, worker_grid_size.y - 2}};
    CoreRangeSet cr_set({cr});

    DummyProgramConfig config = {.cr_set = cr_set, .num_sems = NUM_SEMAPHORES};

    EXPECT_TRUE(local_test_functions::test_dummy_EnqueueProgram_with_sems(this->device, *this->cq, config));
}

}  // end namespace multicore_tests

namespace dram_cache_tests {
TEST_F(CommandQueueHarness, DISABLED_TestDramCacheHit) {}

TEST_F(CommandQueueHarness, DISABLED_TestDramCacheMatch) {}

TEST_F(CommandQueueHarness, DISABLED_TestProgramVectorSizeMatch) {}

}  // end namespace dram_cache_tests
}  // end namespace basic_tests

namespace stress_tests {
TEST_F(CommandQueueHarness, DISABLED_TestSendMaxNumberOfRuntimeArgs) {}

}  // namespace stress_tests
