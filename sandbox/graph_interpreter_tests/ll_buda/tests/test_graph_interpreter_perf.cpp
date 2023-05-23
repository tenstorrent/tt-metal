#include <algorithm>
#include <functional>
#include <random>

#include "ll_buda/host_api.hpp"
#include "common/bfloat16.hpp"
#include "sfpu_helper/sfpu_helper.hpp"
#include "llrt/tt_debug_print_server.hpp"
#include "llrt/tests/test_libs/debug_mailbox.hpp"
#include "build_kernels_for_riscv/build_kernels_for_riscv.hpp"
#include "common/utils.hpp"

//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;

// Used for compiling blank kernel before setting environment variable
namespace blank {
struct hlk_args_t {
    std::int32_t dummy;
};
}
void run_compile_blank() {
    std::string root_dir = tt::utils::get_root_dir();

    // Create and config an OP
    tt::build_kernel_for_riscv_options_t build_kernel_for_riscv_options("dummy_type","blank_op");
    std::string out_dir_path = root_dir + "/built_kernels/" + build_kernel_for_riscv_options.name;

    log_info(tt::LogBuildKernels, "Compiling OP: {} to {}", build_kernel_for_riscv_options.name, out_dir_path);

    void *hlk_args = new blank::hlk_args_t{
        .dummy = 0,
    };
    build_kernel_for_riscv_options.set_hlk_args_all_cores(hlk_args, sizeof(blank::hlk_args_t));
    build_kernel_for_riscv_options.set_hlk_file_name_all_cores("kernels/compute/blank.cpp");
    build_kernel_for_riscv_options.ncrisc_kernel_file_name = "kernels/dataflow/blank.cpp";
    build_kernel_for_riscv_options.brisc_kernel_file_name = "kernels/dataflow/blank.cpp";

    generate_binaries_params_t params;
    generate_binaries_all_riscs(&build_kernel_for_riscv_options, out_dir_path, "grayskull", params);
}

namespace graph_interpreter {
struct hlk_args_t {
    std::int32_t per_core_tile_cnt;
    std::int32_t num_ops;
    std::int32_t num_repetitions;

};
}

const map<string, tt::OpCode> sfpu_name_to_op_code = {
    {"exponential", tt::OpCode::Exponential},
    {"reciprocal",  tt::OpCode::Reciprocal},
    {"gelu",        tt::OpCode::Gelu}
};

const map<string, tt::OpCode> binary_name_to_op_code = {
    {"add", tt::OpCode::Add},
    {"subtract", tt::OpCode::Subtract},
    {"multiply", tt::OpCode::Multiply}
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

vector<uint32_t> binary(const vector<uint32_t> &src0, const vector<uint32_t> &src1, std::function<float(float, float)> binary_func) {
    vector<uint32_t> dst;

    for (uint32_t i = 0; i < src0.size(); i++) {
        uint32_t el0 = src0[i];
        uint32_t top0 = el0 & 0xffff0000;
        uint32_t bottom0 = el0 << 16;

        float top0_ = *reinterpret_cast<float*>(&top0);
        float bottom0_ = *reinterpret_cast<float*>(&bottom0);

        uint32_t el1 = src1[i];
        uint32_t top1 = el1 & 0xffff0000;
        uint32_t bottom1 = el1 << 16;

        float top1_ = *reinterpret_cast<float*>(&top1);
        float bottom1_ = *reinterpret_cast<float*>(&bottom1);

        float exp_top = binary_func(top0_, top1_);
        float exp_bottom = binary_func(bottom0_, bottom1_);

        bfloat16 bfp16_top = bfloat16(exp_top);
        bfloat16 bfp16_bottom = bfloat16(exp_bottom);

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

// Helpers for tests
ll_buda::Program *create_program(
    Device *device,
    uint32_t single_tile_size,
    uint32_t cb_num_tiles,
    uint32_t num_cores,
    std::vector<ll_buda::ComputeKernelArgs *> graph_interpreter_kernel_args_per_core
) {

    ll_buda::Program *program = new ll_buda::Program();

    for (uint32_t core_idx = 0; core_idx < num_cores; core_idx++) {
        CoreCoord core = {core_idx % 12, core_idx / 12};

        uint32_t src0_cb_index = 0;
        uint32_t src0_cb_addr = 200 * 1024;

        auto cb_src0 = ll_buda::CreateCircularBuffer(
            program,
            device,
            src0_cb_index,
            core,
            cb_num_tiles,
            cb_num_tiles * single_tile_size,
            src0_cb_addr,
            tt::DataFormat::Float16_b
        );

        uint32_t ouput_cb_index = 16; // output operands start at index 16
        uint32_t output_cb_addr = 300 * 1024;
        auto cb_output = ll_buda::CreateCircularBuffer(
            program,
            device,
            ouput_cb_index,
            core,
            cb_num_tiles,
            cb_num_tiles * single_tile_size,
            output_cb_addr,
            tt::DataFormat::Float16_b
        );

        uint32_t interm0_cb_index = 24;
        uint32_t interm0_cb_addr = 400 * 1024;
        auto cb_interm0 = ll_buda::CreateCircularBuffer(
            program,
            device,
            interm0_cb_index,
            core,
            cb_num_tiles,
            cb_num_tiles * single_tile_size,
            interm0_cb_addr,
            tt::DataFormat::Float16_b
        );

        auto unary_reader_kernel = ll_buda::CreateDataMovementKernel(
            program,
            "kernels/dataflow/reader_unary_loop.cpp",
            core,
            ll_buda::DataMovementProcessor::RISCV_1,
            ll_buda::NOC::RISCV_1_default);

        auto unary_writer_kernel = ll_buda::CreateDataMovementKernel(
            program,
            "kernels/dataflow/writer_unary_loop.cpp",
            core,
            ll_buda::DataMovementProcessor::RISCV_0,
            ll_buda::NOC::RISCV_0_default);

        bool fp32_dest_acc_en = false;
        bool math_approx_mode = false;
        auto graph_interpreter_kernel = ll_buda::CreateComputeKernel(
            program,
            "kernels/compute/graph_interpreter_loop.cpp",
            core,
            graph_interpreter_kernel_args_per_core[core_idx],
            MathFidelity::HiFi4,
            fp32_dest_acc_en,
            math_approx_mode
        );
    }

    return program;
}

void write_runtime_args_to_device(
    ll_buda::Device *device,
    ll_buda::Program *program,
    uint32_t num_cores,
    uint32_t num_tiles,
    uint32_t num_repetitions,
    std::vector<ll_buda::DramBuffer *> src_dram_buffers,
    vector<std::pair<uint32_t, uint32_t>> input_locs_per_core,
    vector<ll_buda::DramBuffer *> dst_dram_buffer_per_core
) {

    auto dm_kernels = program->data_movement_kernels();
    for (uint32_t core_idx = 0; core_idx < num_cores; core_idx++){
        CoreCoord core = {core_idx % 12, core_idx / 12};

        uint32_t src_dram_channel = input_locs_per_core[core_idx].first;
        uint32_t src_dram_addr = input_locs_per_core[core_idx].second;
        auto src_dram_noc_xy = src_dram_buffers[src_dram_channel]->noc_coordinates(device);

        std::vector<uint32_t> unary_reader_args{
            src_dram_addr,
            (std::uint32_t)src_dram_noc_xy.x,
            (std::uint32_t)src_dram_noc_xy.y,
            num_tiles,
            num_repetitions,
        };

        ll_buda::WriteRuntimeArgsToDevice(
            device,
            dm_kernels[core_idx * 2],
            core,
            unary_reader_args
        );

        auto dst_dram_noc_xy = dst_dram_buffer_per_core[core_idx]->noc_coordinates(device);
        std::vector<uint32_t> unary_writer_args{
            dst_dram_buffer_per_core[core_idx]->address(),
            (std::uint32_t)dst_dram_noc_xy.x,
            (std::uint32_t)dst_dram_noc_xy.y,
            num_tiles,
            num_repetitions,
        };
        ll_buda::WriteRuntimeArgsToDevice(
            device,
            dm_kernels[core_idx * 2 + 1],
            core,
            unary_writer_args
        );
    }
}

void write_op_info_to_l1(
    ll_buda::Device *device,
    uint32_t num_cores,
    vector<vector<string>> sfpu_chain_per_core
) {

    for (uint32_t core_idx = 0; core_idx < num_cores; core_idx++){
        CoreCoord core = {core_idx % 12, core_idx / 12};
        std::vector<string> sfpu_chain = sfpu_chain_per_core[core_idx];
        uint32_t chain_length = sfpu_chain.size();
        for (int idx = 0; idx < chain_length; idx++) {

            OpCode op_code = sfpu_name_to_op_code.at(sfpu_chain.at(idx));

            uint32_t in0;
            uint32_t in1;
            uint32_t out;

            if (idx == 0) {
                in0 = 0;
                in1 = 1;
            } else {
                in0 = 24;
                in1 = 25;
            }

            if (idx < chain_length - 1) {
                out = 24;
            } else {
                out = 16;
            }

            uint32_t pop_input = 1;

            op_info_t op_info = {
                .op_code = (uint32_t) op_code,
                .cb_in0_id = in0,
                .cb_in1_id = in1,
                .cb_out_id = out,
                .pop0 = pop_input,
                .pop1 = pop_input,
                .unary = 1
            };

            ll_buda::WriteToDeviceL1(device, core, op_info, idx);
        }
    }
}


// This test just runs a chain of eltwise unary sfpu ops, and it's a good baseline test for the
// graph interpreter since there is no branching
bool run_chained_sfpu_test(uint32_t chain_length, uint32_t num_cores, uint32_t num_graphs, uint32_t num_repetitions) {


    TT_ASSERT(chain_length > 0 && chain_length <= 10, "Cannot have a graph of more than 10 ops in L1");

    bool pass = true;

    vector<string> sfpu_names = {"exponential", "reciprocal", "gelu"};


    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Grayskull Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int pci_express_slot = 0;

        ll_buda::Device *device =
            ll_buda::CreateDevice(tt::ARCH::GRAYSKULL, pci_express_slot);

        pass &= ll_buda::InitializeDevice(device);

        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        uint32_t single_tile_size = 2 * 1024;
        uint32_t num_tiles = 1;

        uint32_t num_tiles_input = 120;
        uint32_t src_dram_buffer_size = single_tile_size * num_tiles_input;
        uint32_t dst_dram_buffer_size = single_tile_size * num_tiles;

        uint32_t src_dram_buffer_base_addr = 0;
        uint32_t dst_dram_buffer_base_addr = 512 * 1024 * 1024; // 512 MB (upper half)

        uint32_t cb_num_tiles = 1;

        uint32_t num_dram_channels = 8;

        std::vector<ll_buda::DramBuffer *> src_dram_buffers;
        for (uint32_t i = 0; i < num_dram_channels; i++){
            auto src_dram_buffer = ll_buda::CreateDramBuffer(device, i, src_dram_buffer_size, src_dram_buffer_base_addr);
            src_dram_buffers.push_back(src_dram_buffer);
        }

        vector<ll_buda::DramBuffer *> dst_dram_buffer_per_core;
        vector<ll_buda::ComputeKernelArgs *> graph_interpreter_kernel_args_per_core;
        for (uint32_t core_idx = 0; core_idx < num_cores; core_idx++){
            CoreCoord core = {core_idx % 12, core_idx / 12};

            //Output DRAM Loc is fixed per core
            uint32_t dst_dram_channel_id = core_idx % num_dram_channels;
            auto dst_dram_buffer = ll_buda::CreateDramBuffer(device, dst_dram_channel_id, dst_dram_buffer_size, dst_dram_buffer_base_addr + dst_dram_buffer_size * (core_idx / num_dram_channels));

            dst_dram_buffer_per_core.push_back(dst_dram_buffer);

            ////////////////////////////////////////////////////////////////////////////
            //                  Compile Time Args Setup
            ////////////////////////////////////////////////////////////////////////////
            void *hlk_args = new graph_interpreter::hlk_args_t{
                .per_core_tile_cnt = (int) num_tiles,
                .num_ops = (int) chain_length,
                .num_repetitions = (int) num_repetitions,
            };
            ll_buda::ComputeKernelArgs *graph_interpreter_kernel_args = ll_buda::InitializeCompileTimeComputeKernelArgs(core, hlk_args, sizeof(graph_interpreter::hlk_args_t));
            graph_interpreter_kernel_args_per_core.push_back(graph_interpreter_kernel_args);
        }

        ////////////////////////////////////////////////////////////////////////////
        //                      Compile Application
        ////////////////////////////////////////////////////////////////////////////
        ll_buda::Program *program = create_program(device, single_tile_size, cb_num_tiles, num_cores, graph_interpreter_kernel_args_per_core);

        pass &= ll_buda::CompileProgram(device, program);

        ////////////////////////////////////////////////////////////////////////////
        //                      Create input
        ////////////////////////////////////////////////////////////////////////////
        std::vector<std::vector<uint32_t> > src_vecs;
        for(const auto& src_dram_buffer: src_dram_buffers){
            vector<uint32_t> src_vec = create_random_ones_and_twos_vector_of_bfloat16(
                src_dram_buffer->size(),  std::chrono::system_clock::now().time_since_epoch().count());
            pass &= ll_buda::WriteToDeviceDRAM(src_dram_buffer, src_vec);
            src_vecs.push_back(src_vec);
        }

        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Application
        ////////////////////////////////////////////////////////////////////////////
        pass &= ll_buda::ConfigureDeviceWithProgram(device, program);

        for (uint32_t graph_idx = 0; graph_idx < num_graphs; graph_idx++){
            vector<vector<string> > sfpu_chain_per_core;
            vector<std::pair<uint32_t, uint32_t> > input_locs_per_core;

            for (uint32_t core_idx = 0; core_idx < num_cores; core_idx++){
                CoreCoord core = {core_idx % 12, core_idx / 12};
                vector<string> sfpu_chain;

                // Create random chain of sfpu ops
                for (int _ = 0; _ < chain_length; _++) {
                    int idx = rand() % sfpu_names.size();
                    string sfpu_name = sfpu_names.at(idx);
                    sfpu_chain.push_back(sfpu_name);
                }
                sfpu_chain_per_core.push_back(sfpu_chain);

                uint32_t src_dram_channel = rand() % num_dram_channels;
                uint32_t src_dram_addr = (rand() % num_tiles_input) * single_tile_size;
                input_locs_per_core.push_back({src_dram_channel, src_dram_addr});
            }

            write_runtime_args_to_device(device, program, num_cores, num_tiles, num_repetitions, src_dram_buffers, input_locs_per_core, dst_dram_buffer_per_core);
            write_op_info_to_l1(device, num_cores, sfpu_chain_per_core);

            pass &= ll_buda::LaunchKernels(device, program);

            std::vector<std::vector<uint32_t> > result_vecs;
            for (uint32_t core_idx = 0; core_idx < num_cores; core_idx++){
                std::vector<uint32_t> result_vec;
                ll_buda::ReadFromDeviceDRAM(dst_dram_buffer_per_core[core_idx], result_vec);
                result_vecs.push_back(result_vec);
            }

            ////////////////////////////////////////////////////////////////////////////
            //                      Validation & Teardown
            ////////////////////////////////////////////////////////////////////////////

            std::vector<std::vector<uint32_t> > golden;
            for (uint32_t core_idx = 0; core_idx < num_cores; core_idx++){
                auto input_loc = input_locs_per_core[core_idx];
                uint32_t src_dram_addr_idx = input_loc.second / sizeof(uint32_t);
                vector<uint32_t> g(src_vecs[input_loc.first].begin() + src_dram_addr_idx, src_vecs[input_loc.first].begin() + src_dram_addr_idx + result_vecs[core_idx].size());
                for (uint32_t i = 0; i < chain_length; i++) {

                    g = sfpu(g, sfpu_op_to_function.at(sfpu_chain_per_core[core_idx][i]));
                }
                golden.push_back(g);
            }

            // Disable validation check due to over/underflow of random generated graphs
            // for (uint32_t core_idx = 0; core_idx < num_cores; core_idx++){

            //     bool p = packed_uint32_t_vector_comparison(result_vecs[core_idx], golden[core_idx], is_close_0p05);
            //     pass &= p;

            //     if (not p) {
            //         std::cout << "GOLDEN" << std::endl;
            //         print_vec_of_uint32_as_packed_bfloat16(golden[core_idx], num_tiles);

            //         std::cout << "RESULT" << std::endl;
            //         print_vec_of_uint32_as_packed_bfloat16(result_vecs[core_idx], num_tiles);
            //     }
            // }
        }
        pass &= ll_buda::CloseDevice(device);
        delete device;


    } catch (const std::exception &e) {
        pass = false;
        // Capture the exception error message
        log_error(LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        log_fatal(LogTest, "System error message: {}", std::strerror(errno));
    }

    return pass;
}

// This test just runs a chain of eltwise binary ops, and it's a good baseline test for the
// graph interpreter since there is no branching
bool run_chained_binary_test(uint32_t chain_length, uint32_t num_cores, uint32_t num_graphs, uint32_t num_repetitions) {

    TT_ASSERT(chain_length > 0 && chain_length <= 10, "Cannot have a graph of more than 10 ops in L1");
    TT_ASSERT(num_cores > 0 && num_cores <= 120, "Cannot have more than 120 cores on GS");

    bool pass = true;

    vector<string> binary_names = {"add", "subtract", "multiply"};


    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Grayskull Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int pci_express_slot = 0;
        ll_buda::Device *device =
            ll_buda::CreateDevice(tt::ARCH::GRAYSKULL, pci_express_slot);

        pass &= ll_buda::InitializeDevice(device);

        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        ll_buda::Program *program = new ll_buda::Program();



        uint32_t single_tile_size = 2 * 1024;
        uint32_t num_tiles = 1;

        uint32_t num_tiles_input = 120;
        uint32_t src_dram_buffer_size = single_tile_size * num_tiles_input;
        uint32_t dst_dram_buffer_size = single_tile_size * num_tiles;

        uint32_t src_dram_buffer_base_addr = 0;
        uint32_t dst_dram_buffer_base_addr = 512 * 1024 * 1024; // 512 MB (upper half)

        uint32_t cb_num_tiles = 1;
        uint32_t num_dram_channels = 8;

        std::vector<ll_buda::DramBuffer *> src_dram_buffers;
        for (uint32_t i = 0; i < num_dram_channels; i++){
            auto src_dram_buffer = ll_buda::CreateDramBuffer(device, i, src_dram_buffer_size, src_dram_buffer_base_addr);
            src_dram_buffers.push_back(src_dram_buffer);
        }

        vector<ll_buda::DramBuffer *> dst_dram_buffer_per_core;
        vector<ll_buda::DataMovementKernel *> reader_kernel_per_core;
        vector<ll_buda::DataMovementKernel *> writer_kernel_per_core;

        for (uint32_t core_idx = 0; core_idx < num_cores; core_idx++){
            CoreCoord core = {core_idx % 12, core_idx / 12};

            uint32_t dst_dram_channel_id = core_idx % num_dram_channels;
            auto dst_dram_buffer = ll_buda::CreateDramBuffer(device, dst_dram_channel_id, dst_dram_buffer_size, dst_dram_buffer_base_addr + dst_dram_buffer_size * (core_idx / num_dram_channels));

            dst_dram_buffer_per_core.push_back(dst_dram_buffer);

            std::vector<ll_buda::CircularBuffer *> src_cb_buffers;
            for (uint32_t i = 0, src_cb_idx = 0, src_cb_addr = 200 * 1024; i < num_dram_channels; i++, src_cb_idx++, src_cb_addr += cb_num_tiles * single_tile_size){
                auto src_cb = ll_buda::CreateCircularBuffer(
                    program,
                    device,
                    src_cb_idx,
                    core,
                    cb_num_tiles,
                    cb_num_tiles * single_tile_size,
                    src_cb_addr,
                    tt::DataFormat::Float16_b
                );
                src_cb_buffers.push_back(src_cb);
            }


            uint32_t ouput_cb_index = 16; // output operands start at index 16
            uint32_t output_cb_addr = 600 * 1024;
            auto output_cb_buffer = ll_buda::CreateCircularBuffer(
                program,
                device,
                ouput_cb_index,
                core,
                cb_num_tiles,
                cb_num_tiles * single_tile_size,
                output_cb_addr,
                tt::DataFormat::Float16_b
            );

            std::vector<ll_buda::CircularBuffer *> interm_cb_buffers;
            for (uint32_t i = 0, interm_cb_idx = 24, interm_cb_addr = 700 * 1024; i < 1; i++, interm_cb_idx++, interm_cb_addr += cb_num_tiles * single_tile_size){
                auto interm_cb = ll_buda::CreateCircularBuffer(
                    program,
                    device,
                    interm_cb_idx,
                    core,
                    cb_num_tiles,
                    cb_num_tiles * single_tile_size,
                    interm_cb_addr,
                    tt::DataFormat::Float16_b
                );
                interm_cb_buffers.push_back(interm_cb);
            }

            auto nary_reader_kernel = ll_buda::CreateDataMovementKernel(
                program,
                "kernels/dataflow/reader_nary_loop.cpp",
                core,
                ll_buda::DataMovementProcessor::RISCV_1,
                ll_buda::NOC::RISCV_1_default);
            reader_kernel_per_core.push_back(nary_reader_kernel);

            auto unary_writer_kernel = ll_buda::CreateDataMovementKernel(
                program,
                "kernels/dataflow/writer_unary_loop.cpp",
                core,
                ll_buda::DataMovementProcessor::RISCV_0,
                ll_buda::NOC::RISCV_0_default);
            writer_kernel_per_core.push_back(unary_writer_kernel);

            void *hlk_args = new graph_interpreter::hlk_args_t{
                .per_core_tile_cnt = (int) num_tiles,
                .num_ops = (int) chain_length,
                .num_repetitions = (int) num_repetitions,
            };
            ll_buda::ComputeKernelArgs *graph_interpreter_kernel_args = ll_buda::InitializeCompileTimeComputeKernelArgs(core, hlk_args, sizeof(graph_interpreter::hlk_args_t));
            bool fp32_dest_acc_en = false;
            bool math_approx_mode = false;
            auto graph_interpreter_kernel = ll_buda::CreateComputeKernel(
                program,
                "kernels/compute/graph_interpreter_loop.cpp",
                core,
                graph_interpreter_kernel_args,
                MathFidelity::HiFi4,
                fp32_dest_acc_en,
                math_approx_mode
            );
        }

        ////////////////////////////////////////////////////////////////////////////
        //                      Compile Application
        ////////////////////////////////////////////////////////////////////////////
        pass &= ll_buda::CompileProgram(device, program);

        ////////////////////////////////////////////////////////////////////////////
        //                      Create input
        ////////////////////////////////////////////////////////////////////////////
        std::vector<std::vector<uint32_t> > src_vecs;
        for(const auto& src_dram_buffer: src_dram_buffers){
            vector<uint32_t> src_vec = create_random_ones_and_twos_vector_of_bfloat16(
                src_dram_buffer->size(),  std::chrono::system_clock::now().time_since_epoch().count());
            pass &= ll_buda::WriteToDeviceDRAM(src_dram_buffer, src_vec);
            src_vecs.push_back(src_vec);
        }

        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Application
        ////////////////////////////////////////////////////////////////////////////

        pass &= ll_buda::ConfigureDeviceWithProgram(device, program);

        for (uint32_t graph_idx = 0; graph_idx < num_graphs; graph_idx++){
            vector<vector<string> > binary_chain_per_core;
            vector<std::vector<std::pair<uint32_t, uint32_t> > > input_locs_per_core;
            for (uint32_t core_idx = 0; core_idx < num_cores; core_idx++){
                CoreCoord core = {core_idx % 12, core_idx / 12};
                vector<string> binary_chain;

                // Create random chain of binary ops
                for (int _ = 0; _ < chain_length; _++) {
                    int idx = rand() % binary_names.size();
                    string binary_name = binary_names.at(idx);
                    binary_chain.push_back(binary_name);
                }
                binary_chain_per_core.push_back(binary_chain);

                uint32_t src_dram_channel = rand() % num_dram_channels;
                uint32_t src_dram_addr = (rand() % num_tiles_input) * single_tile_size;
                auto src_dram_noc_xy = src_dram_buffers[src_dram_channel]->noc_coordinates(device);

                std::vector<uint32_t> nary_reader_args{
                    chain_length + 1,
                    num_tiles,
                    num_repetitions,
                };
                std::vector<std::pair<uint32_t, uint32_t> > input_locs;
                for (uint32_t i = 0; i < chain_length + 1; i++){
                    uint32_t src_dram_channel = rand() % num_dram_channels;
                    uint32_t src_dram_addr = (rand() % num_tiles_input) * single_tile_size;
                    auto src_dram_noc_xy = src_dram_buffers[src_dram_channel]->noc_coordinates(device);
                    nary_reader_args.insert(nary_reader_args.end(), {
                        src_dram_addr,
                        (std::uint32_t)src_dram_noc_xy.x,
                        (std::uint32_t)src_dram_noc_xy.y,
                        i % num_dram_channels,
                    });
                    input_locs.push_back({src_dram_channel, src_dram_addr / 4});
                }
                input_locs_per_core.push_back(input_locs);

                ll_buda::WriteRuntimeArgsToDevice(
                    device,
                    reader_kernel_per_core[core_idx],
                    core,
                    nary_reader_args
                );

                auto dst_dram_noc_xy = dst_dram_buffer_per_core[core_idx]->noc_coordinates(device);
                std::vector<uint32_t> unary_writer_args{
                    dst_dram_buffer_per_core[core_idx]->address(),
                    (std::uint32_t)dst_dram_noc_xy.x,
                    (std::uint32_t)dst_dram_noc_xy.y,
                    num_tiles,
                    num_repetitions,
                };
                ll_buda::WriteRuntimeArgsToDevice(
                    device,
                    writer_kernel_per_core[core_idx],
                    core,
                    unary_writer_args
                );

                ////////////////////////////////////////////////////////////////////////////
                //                      Write op info to L1
                ////////////////////////////////////////////////////////////////////////////
                for (uint32_t idx = 0; idx < chain_length; idx++) {
                    OpCode op_code = binary_name_to_op_code.at(binary_chain.at(idx));

                    uint32_t in0;
                    uint32_t in1;
                    uint32_t out;

                    if (idx == 0) {
                        in0 = 0;
                        in1 = 1;
                    } else {
                        in0 = 24;
                        in1 = (idx + 1) % num_dram_channels;
                    }

                    if (idx < chain_length - 1) {
                        out = 24;
                    } else {
                        out = 16;
                    }

                    uint32_t pop_input = 1;

                    op_info_t op_info = {
                        .op_code = (uint32_t) op_code,
                        .cb_in0_id = in0,
                        .cb_in1_id = in1,
                        .cb_out_id = out,
                        .pop0 = pop_input,
                        .pop1 = pop_input,
                        .unary = 0
                    };

                    ll_buda::WriteToDeviceL1(device, core, op_info, idx);
                }
            }

            pass &= ll_buda::LaunchKernels(device, program);

            std::vector<std::vector<uint32_t> > result_vecs;
            for (uint32_t core_idx = 0; core_idx < num_cores; core_idx++){
                std::vector<uint32_t> result_vec;
                ll_buda::ReadFromDeviceDRAM(dst_dram_buffer_per_core[core_idx], result_vec);
                result_vecs.push_back(result_vec);
            }

            ////////////////////////////////////////////////////////////////////////////
            //                      Validation & Teardown
            ////////////////////////////////////////////////////////////////////////////
            std::vector<std::vector<uint32_t> > golden;
            for (uint32_t core_idx = 0; core_idx < num_cores; core_idx++){
                auto input_locs = input_locs_per_core[core_idx];
                vector<uint32_t> g(src_vecs[input_locs[0].first].begin() + input_locs[0].second, src_vecs[input_locs[0].first].begin() + input_locs[0].second + result_vecs[core_idx].size());
                for (uint32_t i = 0; i < chain_length; i++) {
                    auto input_loc = input_locs[i+1];

                    vector<uint32_t> input (src_vecs[input_loc.first].begin() + input_loc.second, src_vecs[input_loc.first].begin() + input_loc.second + result_vecs[core_idx].size());
                    g = binary(g, input, binary_op_to_function.at(binary_chain_per_core[core_idx][i]));
                }
                golden.push_back(g);
            }

            for (uint32_t core_idx = 0; core_idx < num_cores; core_idx++){

                bool p = packed_uint32_t_vector_comparison(result_vecs[core_idx], golden[core_idx], is_close_0p05);
                pass &= p;

                if (not p) {
                    std::cout << "GOLDEN" << std::endl;
                    print_vec_of_uint32_as_packed_bfloat16(golden[core_idx], num_tiles);

                    std::cout << "RESULT" << std::endl;
                    print_vec_of_uint32_as_packed_bfloat16(result_vecs[core_idx], num_tiles);
                }
            }
        }


        pass &= ll_buda::CloseDevice(device);
        delete device;


    } catch (const std::exception &e) {
        pass = false;
        // Capture the exception error message
        log_error(LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        log_fatal(LogTest, "System error message: {}", std::strerror(errno));
    }

    return pass;
}


int main(int argc, char **argv) {

    srand(std::chrono::system_clock::now().time_since_epoch().count());

    // Run compile blank kernel here so that HACK_FOR_GRAPH_INTERPRETER doesn't
    // meddle with the compilation
    run_compile_blank();

    char env[] = "HACK_FOR_GRAPH_INTERPRETER=1";
    putenv(env);

    bool pass = true;
    // Trivial chain of sfpu ops
    pass &= run_chained_sfpu_test(10, 120, 1000, 1000);
    ll_buda::dumpProfilerResults("sfpu");

    // Trivial chain of binary ops
    pass &= run_chained_binary_test(10, 120, 1000, 1000);
    ll_buda::dumpProfilerResults("binary");

    TT_ASSERT(pass, "Graph interpreter perf test failed");

    return 0;
}
