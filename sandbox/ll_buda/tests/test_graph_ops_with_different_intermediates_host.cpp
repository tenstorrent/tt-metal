#include <algorithm>
#include <functional>
#include <random>
#include <cmath>

#include "ll_buda/host_api.hpp"
#include "common/bfloat16.hpp"
#include "test_gold_impls.hpp"
#include "sfpu_helper/sfpu_helper.hpp"

using namespace tt;

float add(float x, float y) {
    return x + y;
}
float subtract(float x, float y) {
    return x - y;
}
float multiply(float x, float y) {
    return x * y;
}

bool is_close_0p1(float a, float b) {
    return is_close(a, b, 0.1f);
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

ll_buda::Device* setup_grayskull_device(){
    bool pass = true;
    int pci_express_slot = 0;
    ll_buda::Device *device =
        ll_buda::CreateDevice(tt::ARCH::GRAYSKULL, pci_express_slot);

    pass &= ll_buda::InitializeDevice(device);;
    TT_ASSERT(pass);
    return device;

}

void create_cb_in_L1(ll_buda::Program *program, ll_buda::Device* device, CoreCoord* core, int single_tile_size){
    uint32_t src0_cb_index = 0;
    uint32_t src0_cb_addr = 200 * 1024;
    uint32_t num_input_tiles = 2;
    auto cb_src0 = ll_buda::CreateCircularBuffer(
        program,
        device,
        src0_cb_index,
        *core,
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
        *core,
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
        *core,
        num_output_tiles,
        num_output_tiles * single_tile_size,
        output_cb_addr,
        tt::DataFormat::Float16_b
    );
}

std::vector<uint32_t> calculate_unary_op(string sfpu_name, std::vector<uint32_t> &operand_1, uint32_t single_tile_size, uint32_t num_tiles, uint32_t dram_buffer_size, ll_buda::Device* device)
{
    bool pass = true;
    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Grayskull Device Setup
        ////////////////////////////////////////////////////////////////////////////

        // Done

        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        ll_buda::Program *program = new ll_buda::Program();

        CoreCoord core = {0, 0};

        uint32_t dram_buffer_src_addr = 0;
        int dram_src_channel_id = 0;
        uint32_t dram_buffer_dst_addr = 512 * 1024 * 1024; // 512 MB (upper half)
        int dram_dst_channel_id = 0;

        auto src_dram_buffer = ll_buda::CreateDramBuffer(device, dram_src_channel_id, dram_buffer_size, dram_buffer_src_addr);
        auto dst_dram_buffer = ll_buda::CreateDramBuffer(device, dram_dst_channel_id, dram_buffer_size, dram_buffer_dst_addr);

        auto dram_src_noc_xy = src_dram_buffer->noc_coordinates();
        auto dram_dst_noc_xy = dst_dram_buffer->noc_coordinates();

        // input CB is larger than the output CB, to test the backpressure from the output CB all the way into the input CB
        // CB_out size = 1 forces the serialization of packer and writer kernel, generating backpressure to math kernel, input CB and reader
        uint32_t src0_cb_index = 0;
        uint32_t src0_cb_addr = 200 * 1024;
        uint32_t num_input_tiles = 8;
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

        uint32_t ouput_cb_index = 16; // output operands start at index 16
        uint32_t output_cb_addr = 300 * 1024;
        uint32_t num_output_tiles = 1;
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

        auto unary_reader_kernel = ll_buda::CreateDataMovementKernel(
            program,
            "kernels/dataflow/reader_unary_push_4.cpp",
            core,
            ll_buda::DataMovementProcessor::RISCV_1,
            ll_buda::NOC::RISCV_1_default);

        auto unary_writer_kernel = ll_buda::CreateDataMovementKernel(
            program,
            "kernels/dataflow/writer_unary.cpp",
            core,
            ll_buda::DataMovementProcessor::RISCV_0,
            ll_buda::NOC::RISCV_0_default);

        void *hlk_args = new unary_datacopy::hlk_args_t{
            .per_core_block_cnt = num_tiles,
            .per_core_block_dim = 1
        };

        ll_buda::ComputeKernelArgs *eltwise_unary_args = ll_buda::InitializeCompileTimeComputeKernelArgs(core, hlk_args, sizeof(unary_datacopy::hlk_args_t));

        bool fp32_dest_acc_en = false;
        bool math_approx_mode = false;

        string hlk_kernel_name = "kernels/compute/eltwise_sfpu.cpp";
        auto eltwise_unary_kernel = ll_buda::CreateComputeKernel(
            program,
            hlk_kernel_name,
            core,
            eltwise_unary_args,
            MathFidelity::HiFi4,
            fp32_dest_acc_en,
            math_approx_mode
        );
        const string hlk_op_name = sfpu_op_to_hlk_op_name.at(sfpu_name);
        eltwise_unary_kernel->add_define("SFPU_OP_AND_PACK", hlk_op_name);
        bool is_relu = (sfpu_name == "relu");
        eltwise_unary_kernel->add_define("INIT_RELU", is_relu ? "hlk_relu_config(nullptr, 1);" : "");
        eltwise_unary_kernel->add_define("DEINIT_RELU", is_relu ? "hlk_relu_config(nullptr, 0);" : "");

        ////////////////////////////////////////////////////////////////////////////
        //                      Compile Application
        ////////////////////////////////////////////////////////////////////////////

        pass &= ll_buda::CompileProgram(device, program);

        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Application
        ////////////////////////////////////////////////////////////////////////////

        pass &= ll_buda::WriteToDeviceDRAM(src_dram_buffer, operand_1);

        pass &= ll_buda::ConfigureDeviceWithProgram(device, program);

        ll_buda::WriteRuntimeArgsToDevice(
            device,
            unary_reader_kernel,
            core,
            {
                dram_buffer_src_addr,
                (std::uint32_t)dram_src_noc_xy.x,
                (std::uint32_t)dram_src_noc_xy.y,
                num_tiles
            }
        );

        ll_buda::WriteRuntimeArgsToDevice(
            device,
            unary_writer_kernel,
            core,
            {
                dram_buffer_dst_addr,
                (std::uint32_t)dram_dst_noc_xy.x,
                (std::uint32_t)dram_dst_noc_xy.y,
                num_tiles
            }
        );

        pass &= ll_buda::LaunchKernels(device, program);

        std::vector<uint32_t> result_vec;

        ll_buda::ReadFromDeviceDRAM(dst_dram_buffer, result_vec);

        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////

        std::vector<uint32_t> golden = sfpu(operand_1, sfpu_op_to_function.at(sfpu_name));

        pass &= packed_uint32_t_vector_comparison(result_vec, golden, sfpu_op_to_comparison_function.at(sfpu_name));

        return result_vec;

    } catch (const std::exception &e) {
        pass = false;
        // Capture the exception error message
        log_error(LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        log_fatal(LogTest, "System error message: {}", std::strerror(errno));
        throw;
    }

}

std::vector<uint32_t> calculate_binary_op(const int op_id, const char* op_name, std::vector<uint32_t> &operand_1, std::vector<uint32_t> &operand_2, uint32_t single_tile_size, uint32_t num_tiles, uint32_t dram_buffer_size, ll_buda::Device* device)
{
    //////////////////////////////////////////////////////////////////////////////////////////
    // Test should execute specified op on data copied from host to device and comapre to golden.
    //////////////////////////////////////////////////////////////////////////////////////////

    bool pass = true;

    const int current_op_id = op_id;
    const char* op_id_to_op_define[] = {"add_tiles", "sub_tiles", "mul_tiles"};
    const char* op_id_to_op_name[] = {"ADD", "SUB", "MUL"};
    auto ops = EltwiseOp::all();

    try {

        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////

        ll_buda::Program *program = new ll_buda::Program();

        CoreCoord core = {0, 0};

        ////////////////////////////////////////////////////////////////////////////
        //                      DRAM Setup
        ////////////////////////////////////////////////////////////////////////////

        uint32_t dram_buffer_src0_addr = 0;
        int dram_src0_channel_id = 0;
        uint32_t dram_buffer_src1_addr = 8 * 1024 * 1024;
        int dram_src1_channel_id = 0;
        uint32_t dram_buffer_dst_addr = 16 * 1024 * 1024;
        int dram_dst_channel_id = 0;

        auto src0_dram_buffer = ll_buda::CreateDramBuffer(device, dram_src0_channel_id, dram_buffer_size, dram_buffer_src0_addr);
        auto src1_dram_buffer = ll_buda::CreateDramBuffer(device, dram_src1_channel_id, dram_buffer_size, dram_buffer_src1_addr);
        auto dst_dram_buffer = ll_buda::CreateDramBuffer(device, dram_dst_channel_id, dram_buffer_size, dram_buffer_dst_addr);

        auto dram_src0_noc_xy = src0_dram_buffer->noc_coordinates();
        auto dram_src1_noc_xy = src1_dram_buffer->noc_coordinates();
        auto dram_dst_noc_xy = dst_dram_buffer->noc_coordinates();

        ////////////////////////////////////////////////////////////////////////////
        //                      L1 Setup
        ////////////////////////////////////////////////////////////////////////////

        create_cb_in_L1(program, device, &core, single_tile_size);

        ////////////////////////////////////////////////////////////////////////////
        //                      Kernel Setup
        ////////////////////////////////////////////////////////////////////////////

        auto binary_reader_kernel = ll_buda::CreateDataMovementKernel(
            program,
            "kernels/dataflow/reader_binary_diff_lengths.cpp",
            core,
            ll_buda::DataMovementProcessor::RISCV_1,
            ll_buda::NOC::RISCV_1_default);

        auto unary_writer_kernel = ll_buda::CreateDataMovementKernel(
            program,
            "kernels/dataflow/writer_unary.cpp",
            core,
            ll_buda::DataMovementProcessor::RISCV_0,
            ll_buda::NOC::RISCV_0_default);

        void *hlk_args = new eltwise_binary_ns::hlk_args_t{
            .per_core_block_cnt = 1024,
            .per_core_block_size = 2
        };
        ll_buda::ComputeKernelArgs *eltwise_binary_args = ll_buda::InitializeCompileTimeComputeKernelArgs(core, hlk_args, sizeof(eltwise_binary_ns::hlk_args_t));

        bool fp32_dest_acc_en = false;
        bool math_approx_mode = false;
        auto eltwise_binary_kernel = ll_buda::CreateComputeKernel(
            program,
            "kernels/compute/eltwise_binary.cpp",
            core,
            eltwise_binary_args,
            MathFidelity::HiFi4,
            fp32_dest_acc_en,
            math_approx_mode
        );
        eltwise_binary_kernel->add_define("ELTWISE_OP", op_id_to_op_define[current_op_id]);

        ////////////////////////////////////////////////////////////////////////////
        //                      Compile Application
        ////////////////////////////////////////////////////////////////////////////

        pass &= ll_buda::CompileProgram(device, program);

        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Application
        ////////////////////////////////////////////////////////////////////////////

        pass &= ll_buda::WriteToDeviceDRAM(src0_dram_buffer, operand_1);

        pass &= ll_buda::WriteToDeviceDRAM(src1_dram_buffer, operand_2);

        pass &= ll_buda::ConfigureDeviceWithProgram(device, program);

        ll_buda::WriteRuntimeArgsToDevice(
            device,
            binary_reader_kernel,
            core,
            {dram_buffer_src0_addr,
            (std::uint32_t)dram_src0_noc_xy.x,
            (std::uint32_t)dram_src0_noc_xy.y,
            num_tiles,
            dram_buffer_src1_addr,
            (std::uint32_t)dram_src1_noc_xy.x,
            (std::uint32_t)dram_src1_noc_xy.y,
            num_tiles});

        ll_buda::WriteRuntimeArgsToDevice(
            device,
            unary_writer_kernel,
            core,
            {dram_buffer_dst_addr,
            (std::uint32_t)dram_dst_noc_xy.x,
            (std::uint32_t)dram_dst_noc_xy.y,
            num_tiles});

        pass &= ll_buda::LaunchKernels(device, program);

        std::vector<uint32_t> result_vec;
        ll_buda::ReadFromDeviceDRAM(dst_dram_buffer, result_vec);

        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////

        // Compute golden and compare results
        vector<uint32_t> golden = eltwise_binary(operand_1, operand_2, binary_op_to_function.at(op_name));

        pass &= packed_uint32_t_vector_comparison(result_vec, golden, is_close_0p1);

        if (pass) {
            log_info(LogTest, "Test Binary Passed");
        } else {
            log_fatal(LogTest, "Test Binary Failed");
        }

        TT_ASSERT(pass);

        return result_vec;

    } catch (const std::exception &e) {
        pass = false;
        // Capture the exception error message
        log_error(LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        log_error(LogTest, "System error message: {}", std::strerror(errno));

        throw;
    }

}

int main(int argc, char **argv) {
    //////////////////////////////////////////////////////////////////////////////////////////
    //
    // Example created to test execution of operations, specified by the graph bellow.
    // The INTERMEDIATE input is on Cpu/Host.
    // Test forms a graph of operations like this:
    //
    //      DRAM1 DRAM2 DRAM3 DRAM4
    //           \  /     \  /
    //             add     sub
    //                \  /
    //                 mul
    //                  |
    //                recip
    //                  |
    //                DRAM5
    //
    //////////////////////////////////////////////////////////////////////////////////////////

    uint32_t single_tile_size = 2 * 1024;
    uint32_t num_tiles = 2048;
    uint32_t dram_buffer_size = single_tile_size * num_tiles;

    ////////////////////////////////////////////////////////////////////////////
    //                      Grayskull Device Setup
    ////////////////////////////////////////////////////////////////////////////

    ll_buda::Device *device = setup_grayskull_device();

    const char op_name_1[] = "add";
    int op_id = 0;
    std::vector<uint32_t> src0_vec = create_random_vector_of_bfloat16(
        dram_buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());

    std::vector<uint32_t> src1_vec = create_random_vector_of_bfloat16(
        dram_buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());

    std::vector<uint32_t> interm_result_1 = calculate_binary_op(op_id, op_name_1, src0_vec, src1_vec, single_tile_size, num_tiles, dram_buffer_size, device);

    const char op_name_2[] = "subtract";
    op_id = 1;
    std::vector<uint32_t> src2_vec = create_random_vector_of_bfloat16(
        dram_buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());

    std::vector<uint32_t> src3_vec = create_random_vector_of_bfloat16(
        dram_buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());

    std::vector<uint32_t> interm_result_2 = calculate_binary_op(op_id, op_name_2, src2_vec, src3_vec, single_tile_size, num_tiles, dram_buffer_size, device);

    const char op_name_3[] = "multiply";
    op_id = 2;
    std::vector<uint32_t> interm_result_3 = calculate_binary_op(op_id, op_name_3, interm_result_1, interm_result_2, single_tile_size, num_tiles, dram_buffer_size, device);

    std::vector<uint32_t> final_result = calculate_unary_op("reciprocal", interm_result_3, single_tile_size, num_tiles, dram_buffer_size, device);
    std::cout<<"Final vector's first element"<<std::endl;
    std::cout<<final_result.front();

    bool pass = ll_buda::CloseDevice(device);;
    delete device;

    if (pass) {
        log_info(LogTest, "\nTest case Passed\n");
    } else {
        log_fatal(LogTest, "\nTest case Failed\n");
    }

    return 0;
}
