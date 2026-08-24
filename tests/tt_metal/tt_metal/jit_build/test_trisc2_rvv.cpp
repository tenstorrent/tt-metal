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
//    on, and its TRISC2 ELF contains vector instructions (checked in-process by scanning the
//    ELF's executable sections for RVV instruction encodings — no subprocess, no shell); the
//    knob-off compile of the same source contains none. Gated on the bundled sfpi toolchain
//    accepting the Zve32f march string (older sfpi releases skip rather than fail).
//
// Everything here runs without silicon: the fixture configures a mock Blackhole device, and
// compilation is pure host-side JIT.

#include <gtest/gtest.h>

#include <elf.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

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
#include "jit_build/jit_build_utils.hpp"
#include "llrt/rtoptions.hpp"

namespace tt::tt_metal {

namespace {

constexpr const char* kRvvKernelPath = "tests/tt_metal/tt_metal/test_kernels/compute/trisc2_rvv_vadd.cpp";
// Matches the flags HalJitBuildQueryBlackHole::rvv_compile_flags emits.
constexpr const char* kBhRvvMarch = "-march=rv32im_zmmul_zaamo_zba_zbb_xtttensixbh_zve32f";

// Resolve the sfpi g++ the JIT build uses (local runtime/sfpi, then system sfpi).
std::string sfpi_gxx_path() {
    const std::string root = MetalContext::instance().rtoptions().get_root_dir();
    for (const std::string& sfpi_root : {root + "runtime/sfpi", std::string("/opt/tenstorrent/sfpi")}) {
        auto gxx = sfpi_root + "/compiler/bin/riscv-tt-elf-g++";
        if (std::filesystem::exists(gxx)) {
            return gxx;
        }
    }
    return {};
}

// True when the bundled sfpi compiler accepts the Blackhole-tensix + Zve32f march string.
// Probed via the JIT build system's own shell-free process launcher (posix_spawn with a
// fixed argv — nothing is interpreted by a shell).
bool sfpi_supports_bh_zve32f() {
    const std::string gxx = sfpi_gxx_path();
    if (gxx.empty()) {
        return false;
    }
    const auto scratch =
        std::filesystem::temp_directory_path() /
        ("tt_metal_rvv_march_probe_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(scratch);
    const std::vector<std::string> args = {gxx, kBhRvvMarch, "-E", "-x", "c++", "/dev/null", "-o", "/dev/null"};
    const bool ok = tt::jit_build::utils::exec_command(args, scratch.string(), (scratch / "probe.log").string());
    std::error_code ec;
    std::filesystem::remove_all(scratch, ec);
    return ok;
}

// In-process check for RISC-V Vector instructions: parse the ELF32 section headers and scan
// every executable PROGBITS section for RVV encodings. The march in play has no compressed
// extension, so all instructions are 4-byte, 4-byte-aligned little-endian words. Fingerprints
// (RISC-V "V" spec, 32-bit encodings):
//  - OP-V major opcode 0b1010111 (0x57): every vector arithmetic instruction and the
//    vsetvli/vsetivli/vsetvl family. This opcode is reserved exclusively for OP-V — scalar
//    (including scalar-FP) code never emits it.
//  - Vector unit-stride 32-bit loads/stores: LOAD-FP/STORE-FP major opcode with the vector
//    width encoding funct3=0b110 → (insn & 0x707F) == 0x6007 (vle32.v) / 0x6027 (vse32.v).
//    Scalar flw/fsw use funct3=0b010, so there is no collision.
bool is_rvv_instruction(uint32_t insn) {
    return (insn & 0x7Fu) == 0x57u || (insn & 0x707Fu) == 0x6007u || (insn & 0x707Fu) == 0x6027u;
}

bool elf_contains_vector_instructions(const std::string& elf_path) {
    std::ifstream file(elf_path, std::ios::binary);
    EXPECT_TRUE(file.is_open()) << "cannot open ELF: " << elf_path;
    std::vector<char> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    EXPECT_GE(bytes.size(), sizeof(Elf32_Ehdr)) << elf_path;
    Elf32_Ehdr ehdr{};
    std::memcpy(&ehdr, bytes.data(), sizeof(ehdr));
    EXPECT_EQ(std::memcmp(ehdr.e_ident, ELFMAG, SELFMAG), 0) << "not an ELF: " << elf_path;
    EXPECT_EQ(ehdr.e_ident[EI_CLASS], ELFCLASS32) << elf_path;
    EXPECT_EQ(ehdr.e_ident[EI_DATA], ELFDATA2LSB) << elf_path;
    EXPECT_EQ(ehdr.e_machine, EM_RISCV) << elf_path;

    for (uint32_t i = 0; i < ehdr.e_shnum; i++) {
        const size_t shoff = static_cast<size_t>(ehdr.e_shoff) + static_cast<size_t>(i) * ehdr.e_shentsize;
        if (shoff + sizeof(Elf32_Shdr) > bytes.size()) {
            break;
        }
        Elf32_Shdr shdr{};
        std::memcpy(&shdr, bytes.data() + shoff, sizeof(shdr));
        if (shdr.sh_type != SHT_PROGBITS || (shdr.sh_flags & SHF_EXECINSTR) == 0) {
            continue;
        }
        if (static_cast<size_t>(shdr.sh_offset) + shdr.sh_size > bytes.size()) {
            continue;
        }
        for (uint32_t off = 0; off + 4 <= shdr.sh_size; off += 4) {
            uint32_t insn = 0;
            std::memcpy(&insn, bytes.data() + shdr.sh_offset + off, 4);
            if (is_rvv_instruction(insn)) {
                return true;
            }
        }
    }
    return false;
}

}  // namespace

// Fixture lives in the named namespace: gtest TEST_F classes derive from it
// with external linkage, and an anonymous-namespace base trips
// -Werror=subobject-linkage under gcc Unity builds (merge-queue build-sweeps).
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

    // Path of the kernel's compiled ELF for one compute processor (0=unpack, 1=math, 2=pack).
    std::string kernel_elf_path(const std::shared_ptr<Kernel>& kernel, int processor_id) {
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
        return elf_path;
    }

    bool kernel_elf_has_vector_code(const std::shared_ptr<Kernel>& kernel, int processor_id) {
        return elf_contains_vector_instructions(kernel_elf_path(kernel, processor_id));
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
    EXPECT_TRUE(kernel_elf_has_vector_code(kernel_on, 2))
        << "TRISC2 ELF of the opted-in kernel contains no vector instructions";
    // The vector unit is private to the pack compile: unpack/math stay scalar.
    EXPECT_FALSE(kernel_elf_has_vector_code(kernel_on, 0));
    EXPECT_FALSE(kernel_elf_has_vector_code(kernel_on, 1));

    auto kernel_off = compile_rvv_kernel(/*enable_trisc2_rvv=*/false);
    EXPECT_FALSE(kernel_elf_has_vector_code(kernel_off, 2))
        << "knob-off TRISC2 ELF unexpectedly contains vector instructions";
}

}  // namespace tt::tt_metal
