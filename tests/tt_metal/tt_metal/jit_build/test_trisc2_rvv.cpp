// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Device-free (mock Blackhole) coverage for ComputeConfig::enable_trisc2_rvv — the opt-in
// that compiles a kernel's TRISC2 (pack) translation unit with the RISC-V Vector (Zve32f)
// extension.
//
//  - Recipe invariance: with the knob off, the exported compile recipes of all three TRISC
//    build states carry no vector flags (byte-identical to a build without the knob, since
//    the flags are appended only for opted-in kernels at recipe-export time). With the knob
//    on, only the TRISC2 recipe gains the flags; TRISC0/1 recipes are unchanged.
//  - End-to-end compile: a trivial explicit-intrinsic vadd kernel JIT-compiles with the knob
//    on, and its TRISC2 ELF disassembly contains vector instructions; the knob-off compile
//    of the same source contains none. Gated on the bundled sfpi toolchain accepting the
//    Zve32f march string (older sfpi releases skip rather than fail).
//
// Everything here runs without silicon: the fixture configures a mock Blackhole device, and
// compilation is pure host-side JIT.

#include <gtest/gtest.h>

#include <array>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>

#include <enchantum/enchantum.hpp>
#include <tt-metalium/experimental/mock_device/mock_device.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "common/mesh_dispatch_fixture.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include "jit_build/build_env_manager.hpp"
#include "llrt/rtoptions.hpp"

namespace tt::tt_metal {

namespace {

constexpr const char* kRvvKernelPath = "tests/tt_metal/tt_metal/test_kernels/compute/trisc2_rvv_vadd.cpp";
// Matches the flags HalJitBuildQueryBlackHole::rvv_compile_flags emits.
constexpr const char* kBhRvvMarch = "-march=rv32im_zmmul_zaamo_zba_zbb_xtttensixbh_zve32f";

// Resolve the sfpi tool root the JIT build uses (local runtime/sfpi, then system sfpi).
std::string sfpi_bin_dir() {
    const std::string root = MetalContext::instance().rtoptions().get_root_dir();
    for (const std::string& sfpi_root : {root + "runtime/sfpi", std::string("/opt/tenstorrent/sfpi")}) {
        if (std::filesystem::exists(sfpi_root + "/compiler/bin/riscv-tt-elf-g++")) {
            return sfpi_root + "/compiler/bin/";
        }
    }
    return {};
}

// Capture stdout of a shell command; returns nullopt on nonzero exit.
std::optional<std::string> run_command(const std::string& cmd) {
    FILE* pipe = popen((cmd + " 2>/dev/null").c_str(), "r");
    if (pipe == nullptr) {
        return std::nullopt;
    }
    std::string out;
    std::array<char, 4096> buf;
    size_t n = 0;
    while ((n = fread(buf.data(), 1, buf.size(), pipe)) > 0) {
        out.append(buf.data(), n);
    }
    return pclose(pipe) == 0 ? std::optional<std::string>(std::move(out)) : std::nullopt;
}

// True when the bundled sfpi compiler accepts the Blackhole-tensix + Zve32f march string.
bool sfpi_supports_bh_zve32f() {
    const std::string bin = sfpi_bin_dir();
    if (bin.empty()) {
        return false;
    }
    return run_command(bin + "riscv-tt-elf-g++ " + kBhRvvMarch + " -E -x c++ /dev/null -o /dev/null").has_value();
}

class Trisc2RvvMockBlackholeFixture : public MeshDispatchFixture {
protected:
    // Mock mode must be registered BEFORE the base fixture opens its shared devices — and that
    // happens at suite scope (MeshDispatchFixture::SetUpTestSuite), not in SetUp. Overriding the
    // suite hooks keeps this whole suite off silicon: the shared devices are created as mock
    // Blackhole and every compile is pure host-side JIT.
    static void SetUpTestSuite() {
        experimental::configure_mock_mode(tt::ARCH::BLACKHOLE, 1);
        MeshDispatchFixture::SetUpTestSuite();
    }
    static void TearDownTestSuite() {
        MeshDispatchFixture::TearDownTestSuite();
        experimental::disable_mock_mode();
    }

    // Create + JIT-compile the RVV vadd kernel; returns the kernel (full name is set by compile).
    std::shared_ptr<Kernel> compile_rvv_kernel(bool enable_trisc2_rvv) {
        distributed::MeshDevice* device = devices_.at(0).get();
        Program program = CreateProgram();
        KernelHandle handle = CreateKernel(
            program,
            kRvvKernelPath,
            CoreCoord{0, 0},
            ComputeConfig{
                .enable_trisc2_rvv = enable_trisc2_rvv,
                // L1 scratch base + element count for the vadd; compile-only, never dispatched.
                .compile_args = {512 * 1024, 64},
            });
        program.impl().compile(device);
        return program.impl().get_kernel(handle);
    }

    // Disassembly of the kernel's compiled ELF for one compute processor (0=unpack, 1=math, 2=pack).
    std::string disassemble(const std::shared_ptr<Kernel>& kernel, int processor_id) {
        distributed::MeshDevice* device = devices_.at(0).get();
        auto& build_env_manager = BuildEnvManager::get_instance();
        const auto& hal = MetalContext::instance().hal();
        const uint32_t core_idx = hal.get_programmable_core_type_index(HalProgrammableCoreType::TENSIX);
        const uint32_t class_idx = enchantum::to_underlying(HalProcessorClassType::COMPUTE);
        const std::string elf_path = build_env_manager.get_kernel_binary_path(
            device->build_id(),
            core_idx,
            class_idx,
            processor_id,
            build_env_manager.get_device_build_env(device->build_id()).build_env.get_out_kernel_root_path(),
            kernel->get_full_kernel_name());
        EXPECT_TRUE(std::filesystem::exists(elf_path)) << "missing ELF: " << elf_path;
        auto dis = run_command(sfpi_bin_dir() + "riscv-tt-elf-objdump -d " + elf_path);
        EXPECT_TRUE(dis.has_value()) << "objdump failed for " << elf_path;
        return dis.value_or(std::string{});
    }

    // The kernel's exported compile recipe cflags for one compute processor.
    std::string recipe_cflags(const std::shared_ptr<Kernel>& kernel, int processor_id) {
        distributed::MeshDevice* device = devices_.at(0).get();
        const auto& hal = MetalContext::instance().hal();
        const uint32_t core_idx = hal.get_programmable_core_type_index(HalProgrammableCoreType::TENSIX);
        const uint32_t class_idx = enchantum::to_underlying(HalProcessorClassType::COMPUTE);
        const JitBuildState& state = BuildEnvManager::get_instance().get_kernel_build_state(
            device->build_id(), core_idx, class_idx, processor_id);
        return state.export_target_recipe(kernel.get()).cflags;
    }
};

bool contains_vector_instructions(const std::string& disassembly) {
    // vsetvli is the unambiguous RVV fingerprint; the vadd kernel also emits loads/adds.
    return disassembly.find("vsetvli") != std::string::npos || disassembly.find("vsetivli") != std::string::npos;
}

}  // namespace

TEST_F(Trisc2RvvMockBlackholeFixture, KnobOffRecipesCarryNoVectorFlags) {
    auto kernel = compile_rvv_kernel(/*enable_trisc2_rvv=*/false);
    for (int processor_id = 0; processor_id < 3; processor_id++) {
        const std::string cflags = recipe_cflags(kernel, processor_id);
        EXPECT_EQ(cflags.find("zve32f"), std::string::npos) << "trisc" << processor_id;
        EXPECT_EQ(cflags.find("-fno-lto"), std::string::npos) << "trisc" << processor_id;
    }
}

TEST_F(Trisc2RvvMockBlackholeFixture, KnobOnFlagsReachOnlyThePackRecipe) {
    if (!sfpi_supports_bh_zve32f()) {
        GTEST_SKIP() << "bundled sfpi toolchain does not accept " << kBhRvvMarch;
    }
    auto kernel_off = compile_rvv_kernel(/*enable_trisc2_rvv=*/false);
    auto kernel_on = compile_rvv_kernel(/*enable_trisc2_rvv=*/true);

    // Opting in must re-key the JIT cache.
    EXPECT_NE(kernel_on->get_full_kernel_name(), kernel_off->get_full_kernel_name());

    // TRISC0/1 recipes are byte-identical with and without the knob; only TRISC2 changes.
    EXPECT_EQ(recipe_cflags(kernel_on, 0), recipe_cflags(kernel_off, 0));
    EXPECT_EQ(recipe_cflags(kernel_on, 1), recipe_cflags(kernel_off, 1));
    const std::string pack_cflags = recipe_cflags(kernel_on, 2);
    EXPECT_NE(pack_cflags.find(kBhRvvMarch), std::string::npos);
    EXPECT_NE(pack_cflags.find("-fno-lto"), std::string::npos);
}

TEST_F(Trisc2RvvMockBlackholeFixture, VaddKernelCompilesToVectorCode) {
    if (!sfpi_supports_bh_zve32f()) {
        GTEST_SKIP() << "bundled sfpi toolchain does not accept " << kBhRvvMarch;
    }
    auto kernel_on = compile_rvv_kernel(/*enable_trisc2_rvv=*/true);
    EXPECT_TRUE(contains_vector_instructions(disassemble(kernel_on, 2)))
        << "TRISC2 ELF of the opted-in kernel contains no vector instructions";
    // The vector unit is private to the pack compile: unpack/math stay scalar.
    EXPECT_FALSE(contains_vector_instructions(disassemble(kernel_on, 0)));
    EXPECT_FALSE(contains_vector_instructions(disassemble(kernel_on, 1)));

    auto kernel_off = compile_rvv_kernel(/*enable_trisc2_rvv=*/false);
    EXPECT_FALSE(contains_vector_instructions(disassemble(kernel_off, 2)))
        << "knob-off TRISC2 ELF unexpectedly contains vector instructions";
}

}  // namespace tt::tt_metal
