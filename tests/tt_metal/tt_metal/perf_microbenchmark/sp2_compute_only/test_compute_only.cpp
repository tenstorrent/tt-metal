// SPDX-License-Identifier: Apache-2.0
//
// SP2 compute-only microbenchmark. Runs minimal_matmul's REAL compute kernel (compute.cpp) on a
// SINGLE core with DRAM/NoC stubbed out (a feeder pushes garbage in0/in1 blocks, a drain pops the
// output). This isolates the absolute best-case matmul FLOP/s (unpack + FPU + pack), independent of
// DRAM bandwidth and NoC. Sweep block sizes / subblock / K-depth. Reports KERNEL-TIME via profiler.
//
// Usage: TT_METAL_DEVICE_PROFILER=1 ./test_compute_only --mb 4 --kb 8 --nb 4 --sbh 2 --sbw 2 --knum 200

#include <cstdint>
#include <exception>
#include <string>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/tt_metal_profiler.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-logger/tt-logger.hpp>
#include "test_common.hpp"

using namespace tt;

int main(int argc, char** argv) {
    bool pass = true;
    uint32_t mb = 4, kb = 8, nb = 4, sbh = 2, sbw = 2, knum = 200, ntests = 4;
    bool full_sync = true;
    std::vector<std::string> a(argv, argv + argc);
    std::tie(mb, a) = test_args::get_command_option_uint32_and_remaining_args(a, "--mb", mb);
    std::tie(kb, a) = test_args::get_command_option_uint32_and_remaining_args(a, "--kb", kb);
    std::tie(nb, a) = test_args::get_command_option_uint32_and_remaining_args(a, "--nb", nb);
    std::tie(sbh, a) = test_args::get_command_option_uint32_and_remaining_args(a, "--sbh", sbh);
    std::tie(sbw, a) = test_args::get_command_option_uint32_and_remaining_args(a, "--sbw", sbw);
    std::tie(knum, a) = test_args::get_command_option_uint32_and_remaining_args(a, "--knum", knum);
    std::tie(ntests, a) = test_args::get_command_option_uint32_and_remaining_args(a, "--num-tests", ntests);

    try {
        auto device = tt_metal::distributed::MeshDevice::create_unit_mesh(0);
        auto bf16 = tt::DataFormat::Float16_b;
        auto fp32 = tt::DataFormat::Float32;
        uint32_t bf16_t = tt::tile_size(bf16), fp32_t = tt::tile_size(fp32);
        CoreCoord core(0, 0);
        CoreRange cr(core, core);
        CoreRangeSet cores(cr);

        uint32_t in0_blk = mb * kb, in1_blk = kb * nb, out_blk = mb * nb;
        log_info(
            LogTest,
            "block mb{} kb{} nb{} sb{}x{} knum{} : in0 {}t in1 {}t out {}t",
            mb,
            kb,
            nb,
            sbh,
            sbw,
            knum,
            in0_blk,
            in1_blk,
            out_blk);

        tt_metal::Program program = tt_metal::Program();
        auto mkcb = [&](uint32_t idx, uint32_t ntiles, tt::DataFormat df, uint32_t tsz) {
            tt_metal::CircularBufferConfig c(ntiles * tsz, {{idx, df}});
            c.set_page_size(idx, tsz);
            tt_metal::CreateCircularBuffer(program, cores, c);
        };
        mkcb(0, 2 * in0_blk, bf16, bf16_t);  // in0
        mkcb(1, 2 * in1_blk, bf16, bf16_t);  // in1
        mkcb(2, 2 * out_blk, bf16, bf16_t);  // out
        mkcb(3, 2 * out_blk, fp32, fp32_t);  // intermediate (fp32 accumulator)

        auto feeder = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/perf_microbenchmark/sp2_compute_only/kernels/feeder.cpp",
            cores,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt_metal::NOC::RISCV_0_default,
                .compile_args = {in0_blk, in1_blk, knum}});
        auto drain = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/perf_microbenchmark/sp2_compute_only/kernels/drain.cpp",
            cores,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_1,
                .noc = tt_metal::NOC::RISCV_1_default,
                .compile_args = {out_blk}});
        // compute CT args: K_num_blocks, M_block, K_block, N_block, M_blocks_per_core, N_blocks_per_core, sbh, sbw
        std::vector<uint32_t> comp_ct = {knum, mb, kb, nb, 1u, 1u, sbh, sbw};
        auto compute = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/compute.cpp",
            cores,
            tt_metal::ComputeConfig{
                .math_fidelity = MathFidelity::HiFi2,
                .fp32_dest_acc_en = true,
                .dst_full_sync_en = full_sync,
                .math_approx_mode = false,
                .compile_args = comp_ct});
        // compute rt args: M_start, M_end, N_start, N_end, is_reduce_bottom
        tt_metal::SetRuntimeArgs(program, compute, core, {0u, mb, 0u, nb, 1u});
        (void)feeder;
        (void)drain;

        auto wl = tt_metal::distributed::MeshWorkload();
        wl.add_program(tt::tt_metal::distributed::MeshCoordinateRange{{0, 0}, {0, 0}}, std::move(program));
        for (uint32_t i = 0; i < ntests; ++i) {
            tt_metal::distributed::EnqueueMeshWorkload(device->mesh_command_queue(), wl, false);
            tt_metal::distributed::Finish(device->mesh_command_queue());
        }
        tt_metal::ReadMeshDeviceProfilerResults(*device);
        double flop = 2.0 * mb * nb * kb * knum * 32.0 * 32.0 * 32.0;
        log_info(LogTest, "DONE flop_per_run {:.0f}", flop);
        pass &= device->close();
    } catch (const std::exception& e) {
        pass = false;
        log_error(LogTest, "{}", e.what());
    }
    log_info(LogTest, "{}", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
