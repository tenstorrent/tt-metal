#include "tt_metal/op_library/bmm/bmm_op.hpp"

#include "tt_metal/host_api.hpp"
#include "common/constants.hpp"

namespace bmm_kernel {
// FIXME:copy pasted the args here from the kernel file,  we could refactor the HLK file
struct hlk_args_t {
    uint32_t batch; // batch
    uint32_t Mt; // number of tiles in M
    uint32_t Kt; // number of tiles in K
    uint32_t Nt; // number of tiles in N
};
}

using namespace tt::constants;

namespace tt {

namespace tt_metal {


Tensor matmul_(const Tensor &a, const Tensor &b, bool bcast_batch) {

    tt_metal::Program *program = new tt_metal::Program();
    tt_xy_pair core = {0, 0};

    const auto& ashape = a.shape(), bshape = b.shape();

    // TODO: Build some sort of dispatcher based on location of op operands
    TT_ASSERT(not a.on_host() and not b.on_host(), "Operands to eltwise binary need to be on device!");
    TT_ASSERT(a.device() == b.device(), "Operands to eltwise binary need to be on the same device!");
    TT_ASSERT(a.buffer() != nullptr and b.buffer() != nullptr, "Operands to eltwise binary need to be allocated in buffers on device!");

    uint32_t single_tile_size = 2 * 1024;
    tt_metal::InterleavedDramBuffer *src0_dram_buffer = a.buffer();
    tt_metal::InterleavedDramBuffer *src1_dram_buffer = b.buffer();
    if (bcast_batch)
        TT_ASSERT(bshape[0]*bshape[1] == 1 && "matmul (batch bcast variant) expects input tensors of shapes BCMK*11KN=BCMN");
    else {
        // same condition as above, different message
        TT_ASSERT(ashape[1] == bshape[1] && ashape[0] == bshape[0]
            && "bmm (non-bcast matmul) expects input tensors of shapes BCMK*BCKN=BCMN");
    }
    TT_ASSERT(src0_dram_buffer->size() % single_tile_size == 0);
    TT_ASSERT(src1_dram_buffer->size() % single_tile_size == 0);

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();
    std::array<uint32_t, 4> cshape{ashape[0], ashape[1], ashape[2], bshape[3]}; // C=A*B, N1MK*11KN->N1MN
    tt_metal::Tensor output = tt_metal::Tensor(cshape, a.dtype(), tt::tt_metal::Layout::TILE, device);

    tt_metal::InterleavedDramBuffer *dst_dram_buffer = output.buffer();
    TT_ASSERT(dst_dram_buffer != nullptr, "Output buffer should be allocated on device!");

    bool pass = true;
    {
        // C = A*B
        // MN = MK*KN
        if (bcast_batch)
            TT_ASSERT(ashape[0] > 0 && bshape[0] == 1);
        else {
            TT_ASSERT(ashape[1] == bshape[1] && ashape[0] == bshape[0] && "Channel and batch dimensions must match in bmm op (non-bcast)");
        }
        TT_ASSERT(ashape[3] == bshape[2] && "Dimension K (A.shape[2] and B.shape[3]) must match for A and B in bmm_op"); // A.K == B.K
        TT_ASSERT(ashape[2] % TILE_HEIGHT == 0);
        TT_ASSERT(ashape[3] % TILE_WIDTH == 0);
        TT_ASSERT(bshape[2] % TILE_HEIGHT == 0);
        TT_ASSERT(bshape[3] % TILE_WIDTH == 0);
        uint32_t B = ashape[0]*ashape[1];
        uint32_t Mt = ashape[2]/TILE_HEIGHT;
        uint32_t Kt = ashape[3]/TILE_WIDTH;
        uint32_t Nt = bshape[3]/TILE_WIDTH;

        uint32_t in0_dram_addr = src0_dram_buffer->address();
        uint32_t in1_dram_addr = src1_dram_buffer->address();
        uint32_t out_dram_addr = dst_dram_buffer->address();

        {
            uint32_t src0_cb_index = 0;
            uint32_t src0_cb_addr = 200 * 1024;
            uint32_t num_input_tiles = 2;
            auto cb_src0 = tt_metal::CreateCircularBuffer(
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
            auto cb_src1 = tt_metal::CreateCircularBuffer(
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
            auto cb_output = tt_metal::CreateCircularBuffer(
                program,
                device,
                ouput_cb_index,
                core,
                num_output_tiles,
                num_output_tiles * single_tile_size,
                output_cb_addr,
                tt::DataFormat::Float16_b
            );

            auto reader = tt_metal::CreateDataMovementKernel(
                program,
                "kernels/dataflow/reader_bmm_8bank.cpp",
                core, DataMovementProcessor::RISCV_1, NOC::RISCV_1_default);

            auto writer = tt_metal::CreateDataMovementKernel(
                program,
                "kernels/dataflow/writer_bmm_8bank.cpp",
                core, DataMovementProcessor::RISCV_0, NOC::RISCV_0_default);

            void *hlk_args = new bmm_kernel::hlk_args_t{ .batch = B, .Mt = Mt, .Kt = Kt, .Nt = Nt };
            vector<uint32_t> compute_args = {
                B, // B
                Mt, // Mt
                Kt, // Kt
                Nt // Nt
            };
            tt_metal::ComputeKernelArgs *eltwise_binary_args = tt_metal::InitializeCompileTimeComputeKernelArgs(core, compute_args);

            bool fp32_dest_acc_en = false;
            bool math_approx_mode = false;
            auto eltwise_binary_kernel = tt_metal::CreateComputeKernel(
                program,
                "kernels/compute/bmm.cpp",
                core,
                eltwise_binary_args,
                MathFidelity::HiFi4,
                fp32_dest_acc_en,
                math_approx_mode
            );

            tt_metal::WriteRuntimeArgsToDevice(
                device, reader, core,
                {in0_dram_addr, in1_dram_addr, Mt, Kt, Nt, Mt*Kt, Kt*Nt, B, uint32_t(bcast_batch ? 1 : 0)}
            );
            tt_metal::WriteRuntimeArgsToDevice(
                device, writer, core,
                {out_dram_addr, 0, Mt, Kt, Nt, Mt*Kt, Kt*Nt, B}
            );

            bool skip_hlkc = false;
            pass &= tt_metal::CompileProgram(device, program, skip_hlkc);
            pass &= tt_metal::ConfigureDeviceWithProgram(device, program);
        }
        pass &= tt_metal::LaunchKernels(device, program);
    }

    TT_ASSERT(pass);

    // output does not hold any data, contains pointer to buffer on device with the data
    return output;
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    return matmul_(a, b, true);
}

Tensor bmm(const Tensor& a, const Tensor& b) {
    return matmul_(a, b, false);
}

}  // namespace tt_metal

}  // namespace tt
