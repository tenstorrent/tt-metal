#include <algorithm>
#include <functional>
#include <random>
#include <cmath>
#include <vector>

#include "ll_buda/host_api.hpp"
#include "common/bfloat16.hpp"
#include "test_gold_impls.hpp"

#include "sfpu_helper/sfpu_helper.hpp"
#include "tools/profiler/profiler.hpp"

using namespace tt;


const uint32_t SINGLE_TILE_SIZE = 2 * 1024;
static Profiler ll_buda_profiler = Profiler();

struct tt_binary_program_t
{
    ll_buda::Program* program;
    ll_buda::DataMovementKernel* reader_kernel;
    ll_buda::DataMovementKernel* writer_kernel;
};


float add(float x, float y) {
    return x + y;
}
float subtract(float x, float y) {
    return x - y;
}
float multiply(float x, float y) {
    return x * y;
}

vector<uint32_t> eltwise_binary(const vector<uint32_t> &srcA, const vector<uint32_t> srcB, std::function<float(float, float)> binary_func) {
    vector<uint32_t> dst;
    TT_ASSERT(srcA.size() == srcB.size(), "Vectors being added need to have the same size");

    for (int i = 0; i < srcA.size(); i++) {
        uint32_t elA = srcA.at(i);
        uint32_t topA = elA & 0xffff0000;
        uint32_t bottomA = elA << 16;
        float topA_ = *reinterpret_cast<float*>(&topA);
        float bottomA_ = *reinterpret_cast<float*>(&bottomA);

        uint32_t elB = srcB.at(i);
        uint32_t topB = elB & 0xffff0000;
        uint32_t bottomB = elB << 16;
        float topB_ = *reinterpret_cast<float*>(&topB);
        float bottomB_ = *reinterpret_cast<float*>(&bottomB);

        float top = binary_func(topA_, topB_);
        float bottom = binary_func(bottomA_, bottomB_);

        bfloat16 bfp16_top = bfloat16(top);
        bfloat16 bfp16_bottom = bfloat16(bottom);

        uint32_t new_val = pack_two_bfloat16_into_uint32(std::pair<bfloat16, bfloat16>(bfp16_bottom, bfp16_top));
        dst.push_back(new_val);
    }
    return dst;
}

const map<string, std::function<float(float, float)>> binary_op_to_function = {
    {"add", add},
    {"subtract", subtract},
    {"multiply", multiply}
};

namespace eltwise_binary_ns {

    struct hlk_args_t {
        std::int32_t per_core_block_cnt;
        std::int32_t per_core_block_size;
    };
}

namespace unary_datacopy {
    //#include "hlks/eltwise_copy.cpp"
    // FIXME:copy pasted the args here from the kernel file,  we could refactor the HLK file
    struct hlk_args_t {
        std::uint32_t per_core_block_cnt;
        std::uint32_t per_core_block_dim;

    };
}

ll_buda::Device* setup_grayskull_device()
{
    int pci_express_slot = 0;
    ll_buda::Device *device = ll_buda::CreateDevice(tt::ARCH::GRAYSKULL, pci_express_slot);

    bool pass = ll_buda::InitializeDevice(device);

    if (not pass)
        return 0;

    return device;
}

void create_cb_in_L1(ll_buda::Program *program, ll_buda::Device* device, CoreCoord core, int single_tile_size)
{
    uint32_t src0_cb_index = 0;
    uint32_t src0_cb_addr = 200 * 1024;
    uint32_t num_input_tiles = 2;

    auto cb_src0 = ll_buda::CreateCircularBuffer(
        program,
        device,
        src0_cb_index,
        core,
        num_input_tiles,
        num_input_tiles * single_tile_size,
        src0_cb_addr,
        tt::DataFormat::Float16_b
    );

    uint32_t src1_cb_index = 1;
    uint32_t src1_cb_addr = 300 * 1024;

    auto cb_src1 = ll_buda::CreateCircularBuffer(
        program,
        device,
        src1_cb_index,
        core,
        num_input_tiles,
        num_input_tiles * single_tile_size,
        src1_cb_addr,
        tt::DataFormat::Float16_b
    );

    uint32_t ouput_cb_index = 16; // output operands start at index 16
    uint32_t output_cb_addr = 400 * 1024;
    uint32_t num_output_tiles = 2;

    auto cb_output = ll_buda::CreateCircularBuffer(
        program,
        device,
        ouput_cb_index,
        core,
        num_output_tiles,
        num_output_tiles * single_tile_size,
        output_cb_addr,
        tt::DataFormat::Float16_b
    );
}

auto core_count(const ll_buda::CoreRange& cores)
{
    return (cores.second.x - cores.first.x + 1) * (cores.second.y - cores.first.y + 1);
}

tt_binary_program_t create_binary_op_program(
    const char* op_define,
    ll_buda::Device* device,
    uint32_t dram_buffer_size,
    const ll_buda::CoreRange& cores)
{
    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////

        ll_buda::Program *program = new ll_buda::Program();

        for (size_t x = cores.first.x; x <= cores.second.x; x++)
        {
            for (size_t y = cores.first.y; y <= cores.second.y; y++)
                create_cb_in_L1(program, device, {x, y}, SINGLE_TILE_SIZE);
        }

        ////////////////////////////////////////////////////////////////////////////
        //                      Kernel Setup
        ////////////////////////////////////////////////////////////////////////////

        auto reader_kernel = ll_buda::CreateDataMovementKernel(
            program,
            "kernels/dataflow/reader_binary_diff_lengths.cpp",
            cores,
            ll_buda::DataMovementProcessor::RISCV_1,
            ll_buda::NOC::RISCV_1_default);

        auto writer_kernel = ll_buda::CreateDataMovementKernel(
            program,
            "kernels/dataflow/writer_unary.cpp",
            cores,
            ll_buda::DataMovementProcessor::RISCV_0,
            ll_buda::NOC::RISCV_0_default);

        void *hlk_args = new eltwise_binary_ns::hlk_args_t{
            .per_core_block_cnt = std::int32_t((dram_buffer_size / SINGLE_TILE_SIZE) / core_count(cores)), //2048,
            .per_core_block_size = 1
        };

        ll_buda::ComputeKernelArgs *eltwise_binary_args = ll_buda::InitializeCompileTimeComputeKernelArgs(cores, hlk_args, sizeof(eltwise_binary_ns::hlk_args_t));

        bool fp32_dest_acc_en = false;
        bool math_approx_mode = false;

        auto eltwise_binary_kernel = ll_buda::CreateComputeKernel(
            program,
            "kernels/compute/eltwise_binary.cpp",
            cores,
            eltwise_binary_args,
            MathFidelity::HiFi4,
            fp32_dest_acc_en,
            math_approx_mode
        );

        eltwise_binary_kernel->add_define("ELTWISE_OP", op_define);

        ////////////////////////////////////////////////////////////////////////////
        //                      Compile Application
        ////////////////////////////////////////////////////////////////////////////

        bool pass = ll_buda::CompileProgram(device, program);

        if (not pass)
            program = 0;

        return tt_binary_program_t {
            .program = program,
            .reader_kernel = reader_kernel,
            .writer_kernel = writer_kernel
        };
    }
    catch (const std::exception &e)
    {
        // Capture the exception error message
        log_error(LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        log_error(LogTest, "System error message: {}", std::strerror(errno));

        return tt_binary_program_t {
            .program = 0
        };
    }
}

// Each core takes part of input/output buffers
bool execute_binary_op(
    tt_binary_program_t tt_binary_program,
    uint32_t dram_buffer_src0_addr, CoreCoord dram_src0_noc_xy,
    uint32_t dram_buffer_src1_addr, CoreCoord dram_src1_noc_xy,
    uint32_t dram_buffer_dst_addr, CoreCoord dram_dst_noc_xy,
    uint32_t dram_buffer_size,
    ll_buda::Device* device,
    const ll_buda::CoreRange& cores)
{
    bool pass = true;
    uint32_t dram_buffer_size_per_core = dram_buffer_size / core_count(cores);

    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Application
        ////////////////////////////////////////////////////////////////////////////

        ll_buda_profiler.markStart("ConfigureDeviceWithProgram");
        pass &= ll_buda::ConfigureDeviceWithProgram(device, tt_binary_program.program);
        ll_buda_profiler.markStop("ConfigureDeviceWithProgram");

        ll_buda_profiler.markStart("WriteRuntimeArgsToDevice");

        uint32_t num_tiles = dram_buffer_size_per_core / SINGLE_TILE_SIZE;
        uint32_t i = 0;

        for (size_t x = cores.first.x; x <= cores.second.x; x++)
        {
            for (size_t y = cores.first.y; y <= cores.second.y; y++)
            {
                CoreCoord core {x, y};

                ll_buda::WriteRuntimeArgsToDevice(
                    device,
                    tt_binary_program.reader_kernel,
                    core,
                    {dram_buffer_src0_addr + i * dram_buffer_size_per_core,
                    (std::uint32_t)dram_src0_noc_xy.x,
                    (std::uint32_t)dram_src0_noc_xy.y,
                    num_tiles,
                    dram_buffer_src1_addr + i * dram_buffer_size_per_core,
                    (std::uint32_t)dram_src1_noc_xy.x,
                    (std::uint32_t)dram_src1_noc_xy.y,
                    num_tiles});

                ll_buda::WriteRuntimeArgsToDevice(
                    device,
                    tt_binary_program.writer_kernel,
                    core,
                    {dram_buffer_dst_addr + i * dram_buffer_size_per_core,
                    (std::uint32_t)dram_dst_noc_xy.x,
                    (std::uint32_t)dram_dst_noc_xy.y,
                    num_tiles});

                i++;
            }
        }

        ll_buda_profiler.markStop("WriteRuntimeArgsToDevice");

        ll_buda_profiler.markStart("LaunchKernels");
        pass &= ll_buda::LaunchKernels(device, tt_binary_program.program);
        ll_buda_profiler.markStop("LaunchKernels");

        return pass;

    }
    catch (const std::exception &e)
    {
        // Capture the exception error message
        log_error(LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        log_error(LogTest, "System error message: {}", std::strerror(errno));

        return false;
    }
}

std::vector<uint32_t> calculate_golden(
    std::vector<uint32_t>& src0_vec,
    std::vector<uint32_t>& src1_vec,
    std::vector<uint32_t>& src2_vec,
    std::vector<uint32_t>& src3_vec)
{
    // Compute golden and compare results
    auto golden_add = eltwise_binary(src0_vec, src1_vec, binary_op_to_function.at("add"));
    auto golden_sub = eltwise_binary(src2_vec, src3_vec, binary_op_to_function.at("subtract"));

    auto golden_mul = eltwise_binary(golden_add, golden_sub, binary_op_to_function.at("multiply"));
    return golden_mul;
}

bool is_close_0p1(float a, float b) {
    return is_close(a, b, 0.1f);
}

int main(int argc, char **argv) {
    //////////////////////////////////////////////////////////////////////////////////////////
    //
    // Exaple created to test execution of operations, specified by the graph bellow.
    // The INTERMEDIATE input can be either on Cpu/Host or L1 or Dram.
    // Where INTERMEDIATE result is stored should be selectable
    // Test forms a graph of operations like this:
    //
    //      DRAM1 DRAM2 DRAM3 DRAM4
    //           \  /     \  /
    //             add     sub
    //                \  /
    //                 mul
    //                  |
    //                DRAM5
    //
    //////////////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////
    //                      Grayskull Device Setup
    ////////////////////////////////////////////////////////////////////////////


    auto *device = setup_grayskull_device();

    ////////////////////////////////////////////////////////////////////////////
    //                      DRAM Setup
    ////////////////////////////////////////////////////////////////////////////
    //uint32_t dram_buffer_size = 2048 * SINGLE_TILE_SIZE;
    uint32_t dram_buffer_size = 2048 * SINGLE_TILE_SIZE;

    std::vector<uint32_t> src0_vec = create_random_vector_of_bfloat16(
        dram_buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());

    std::vector<uint32_t> src1_vec = create_random_vector_of_bfloat16(
        dram_buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());

    std::vector<uint32_t> src2_vec = create_random_vector_of_bfloat16(
        dram_buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());

    std::vector<uint32_t> src3_vec = create_random_vector_of_bfloat16(
        dram_buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());

    int dram_channel_id = 0;

    uint32_t dram_buffer_src0_addr = 0 * 8 * 1024 * 1024;
    uint32_t dram_buffer_src1_addr = 1 * 8 * 1024 * 1024;
    uint32_t dram_buffer_src2_addr = 2 * 8 * 1024 * 1024;
    uint32_t dram_buffer_src3_addr = 3 * 8 * 1024 * 1024;
    uint32_t dram_buffer_int_add_addr = 4 * 8 * 1024 * 1024;
    uint32_t dram_buffer_int_sub_addr = 5 * 8 * 1024 * 1024;
    uint32_t dram_buffer_int_mul_addr = 6 * 8 * 1024 * 1024;

    auto src0_dram_buffer = ll_buda::CreateDramBuffer(device, dram_channel_id, dram_buffer_size, dram_buffer_src0_addr);
    auto src1_dram_buffer = ll_buda::CreateDramBuffer(device, dram_channel_id, dram_buffer_size, dram_buffer_src1_addr);
    auto src2_dram_buffer = ll_buda::CreateDramBuffer(device, dram_channel_id, dram_buffer_size, dram_buffer_src2_addr);
    auto src3_dram_buffer = ll_buda::CreateDramBuffer(device, dram_channel_id, dram_buffer_size, dram_buffer_src3_addr);
    auto intrm_add_dram_buffer = ll_buda::CreateDramBuffer(device, dram_channel_id, dram_buffer_size, dram_buffer_int_add_addr);
    auto intrm_sub_dram_buffer = ll_buda::CreateDramBuffer(device, dram_channel_id, dram_buffer_size, dram_buffer_int_sub_addr);
    auto intrm_mul_dram_buffer = ll_buda::CreateDramBuffer(device, dram_channel_id, dram_buffer_size, dram_buffer_int_mul_addr);

    log_info(LogTest, "Created dram buffers");

    bool pass = true;

    // Copy to device
    pass &= ll_buda::WriteToDeviceDRAM(src0_dram_buffer, src0_vec);
    pass &= ll_buda::WriteToDeviceDRAM(src1_dram_buffer, src1_vec);
    pass &= ll_buda::WriteToDeviceDRAM(src2_dram_buffer, src2_vec);
    pass &= ll_buda::WriteToDeviceDRAM(src3_dram_buffer, src3_vec);


    ////////////////////////////////////////////////////////////////////////////
    //                      Compile Ops
    ////////////////////////////////////////////////////////////////////////////
    log_info(LogTest, "Compiling programs");
    ll_buda::CoreRange cores({0, 0}, {7, 7});

    log_info(LogTest, "Using {} {}", core_count(cores), " cores");

    auto add_program = create_binary_op_program("add_tiles", device, dram_buffer_size, cores);
    auto sub_program = create_binary_op_program("sub_tiles", device, dram_buffer_size, cores);
    auto mul_program = create_binary_op_program("mul_tiles", device, dram_buffer_size, cores);

    ////////////////////////////////////////////////////////////////////////////
    //                      Execute ops
    ////////////////////////////////////////////////////////////////////////////
    log_info(LogTest, "Executing ops");

    ll_buda_profiler.markStart("Execute Graph");

    // Add
    pass &= execute_binary_op(add_program,
        dram_buffer_src0_addr, src0_dram_buffer->noc_coordinates(),
        dram_buffer_src1_addr, src1_dram_buffer->noc_coordinates(),
        dram_buffer_int_add_addr, intrm_add_dram_buffer->noc_coordinates(),
        dram_buffer_size, device, cores);

    // sub
    pass &= execute_binary_op(sub_program,
        dram_buffer_src2_addr, src2_dram_buffer->noc_coordinates(),
        dram_buffer_src3_addr, src3_dram_buffer->noc_coordinates(),
        dram_buffer_int_sub_addr, intrm_sub_dram_buffer->noc_coordinates(),
        dram_buffer_size, device, cores);

    // mul
    pass &= execute_binary_op(mul_program,
        dram_buffer_int_add_addr, intrm_add_dram_buffer->noc_coordinates(),
        dram_buffer_int_sub_addr, intrm_sub_dram_buffer->noc_coordinates(),
        dram_buffer_int_mul_addr, intrm_mul_dram_buffer->noc_coordinates(),
        dram_buffer_size, device, cores);

    ll_buda_profiler.markStop("Execute Graph");

    log_info(LogTest, "Reading result from device");

    // This reads result vector from Device to CPU
    std::vector<uint32_t> result_vec;
    ll_buda::ReadFromDeviceDRAM(intrm_mul_dram_buffer, result_vec);

    log_info(LogTest, "Calculating golden output");

    // Calculate golden
    auto golden = calculate_golden(src0_vec, src1_vec, src2_vec, src3_vec);

    log_info(LogTest, "Comparing the result");

    // Compare with golden output
    pass &= packed_uint32_t_vector_comparison(result_vec, golden, is_close_0p1); //is_close_0p1 is our new function

    // Close Device
    pass &= ll_buda::CloseDevice(device);;
    delete device;

    if (pass) {
        log_info(LogTest, "Dumping profiler results");
        ll_buda_profiler.dumpResults("multicore");

        log_info(LogTest, "\nTest case Passed\n");
    } else {
        log_fatal(LogTest, "\nTest case Failed\n");
    }

    return 0;
}
