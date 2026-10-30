// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

#include "impl/context/metal_context.hpp"
#include "llrt/hal.hpp"
#include <tt-metalium/device.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "tt_metal/tt_metal/dram/mimir_access.hpp"
#include "tt_metal/tt_metal/dram/mimir_patterns.hpp"
#include "tt_metal/tt_metal/dram/mimir_test_common.hpp"

namespace tt::tt_metal::mimir_test::kernel_load {

constexpr const char* kNullKernelPath = "tests/tt_metal/tt_metal/test_kernels/cce/cce_null.cpp";
constexpr const char* kEchoKernelPath = "tests/tt_metal/tt_metal/test_kernels/cce/cce_args_echo.cpp";

constexpr uint32_t kEchoCtMarkerBase = 0xC0DE0000;
constexpr uint32_t kEchoSlotPoisonWord = 0xBADEC0DE;
constexpr size_t kEchoSlotWords = kTl1EchoSlotBytes / sizeof(uint32_t);
constexpr uint32_t kEchoRuntimeArgCount = 4;
constexpr uint32_t kEchoCommonArgCount = 4;
static_assert(3 + kEchoRuntimeArgCount + kEchoCommonArgCount <= kEchoSlotWords);

struct EchoArguments {
    uint32_t ct_marker = 0;
    std::array<uint32_t, kEchoRuntimeArgCount> runtime{};
    std::array<uint32_t, kEchoCommonArgCount> common{};
};

EchoArguments make_echo_arguments(uint32_t seed, uint32_t ct_marker) {
    EchoArguments arguments;
    arguments.ct_marker = ct_marker;
    for (uint32_t index = 0; index < kEchoRuntimeArgCount; index++) {
        arguments.runtime[index] = 0x52000000u + seed * 16 + index;
    }
    for (uint32_t index = 0; index < kEchoCommonArgCount; index++) {
        arguments.common[index] = 0xC4000000u + seed * 16 + index;
    }
    return arguments;
}

// Compile-time args are fixed per kernel, so an update keeps the ct_marker of the original.
EchoArguments make_updated_arguments(const EchoArguments& original, uint32_t seed) {
    return make_echo_arguments(seed + 0x100, original.ct_marker);
}

std::vector<uint32_t> expected_echo_words(uint32_t hart, const EchoArguments& arguments) {
    std::vector<uint32_t> words = {kArgsEchoMagic, hart, arguments.ct_marker};
    words.insert(words.end(), arguments.runtime.begin(), arguments.runtime.end());
    words.insert(words.end(), arguments.common.begin(), arguments.common.end());
    return words;
}

// The runtime places a kernel by logical core; this proves that core is CCE `cce_index`'s location.
::testing::AssertionResult check_logical_core_is_cce(
    IDevice* device, const CoreCoord& logical_core, uint32_t cce_index) {
    const CoreCoord virtual_core = device->virtual_core_from_logical_core(logical_core, CoreType::DRAM);
    const CoreCoord expected_core = translated_dram_core(soc_desc_of(device), cce_index);
    if (virtual_core == expected_core) {
        return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure() << fmt::format(
               "CCE {} logical core ({}, {}) maps to ({}, {}), expected ({}, {})",
               cce_index,
               logical_core.x,
               logical_core.y,
               virtual_core.x,
               virtual_core.y,
               expected_core.x,
               expected_core.y);
}

void poison_echo_slots(IDevice* device, uint32_t cce_index) {
    const std::vector<uint32_t> poison(cce_hart_count() * kEchoSlotWords, kEchoSlotPoisonWord);
    write_cce_tl1(device, cce_index, tl1_test_base(), poison);
}

std::vector<uint32_t> read_echo_slots(IDevice* device, uint32_t cce_index) {
    const uint32_t echo_block_bytes = static_cast<uint32_t>(cce_hart_count() * kEchoSlotWords * sizeof(uint32_t));
    return read_cce_tl1(device, cce_index, tl1_test_base(), echo_block_bytes);
}

std::vector<uint32_t> echo_slot_words(const std::vector<uint32_t>& echo_block, uint32_t hart) {
    const size_t first_index = static_cast<size_t>(hart) * kEchoSlotWords;
    return std::vector<uint32_t>(
        echo_block.begin() + static_cast<std::ptrdiff_t>(first_index),
        echo_block.begin() + static_cast<std::ptrdiff_t>(first_index + kEchoSlotWords));
}

// Compares the whole 64 B slot: the echo words, then poison in the words the kernel does not write.
::testing::AssertionResult check_echo_slot(
    uint32_t hart, const std::vector<uint32_t>& echo_block, const EchoArguments& arguments) {
    std::vector<uint32_t> expected = expected_echo_words(hart, arguments);
    expected.resize(kEchoSlotWords, kEchoSlotPoisonWord);
    return compare_words(fmt::format("echo slot of hart {}", hart), expected, echo_slot_words(echo_block, hart));
}

::testing::AssertionResult check_echo_slot_untouched(uint32_t hart, const std::vector<uint32_t>& echo_block) {
    const std::vector<uint32_t> poison(kEchoSlotWords, kEchoSlotPoisonWord);
    return compare_words(
        fmt::format("echo slot of hart {} (not launched)", hart), poison, echo_slot_words(echo_block, hart));
}

// The kernel echoes kEchoRuntimeArgCount unique args from `first_runtime_arg` and kEchoCommonArgCount common args from
// `first_common_arg`.
KernelHandle create_echo_kernel_from(
    Program& program,
    const CoreCoord& logical_core,
    uint32_t hart,
    uint32_t ct_marker,
    uint32_t first_runtime_arg,
    uint32_t first_common_arg) {
    return CreateKernel(
        program,
        kEchoKernelPath,
        logical_core,
        DramConfig{
            .processor = static_cast<DataMovementProcessor>(hart),
            .noc = NOC::NOC_0,
            .compile_args = {
                tl1_test_base(),
                ct_marker,
                kArgsEchoMagic,
                static_cast<uint32_t>(kTl1EchoSlotBytes),
                kEchoRuntimeArgCount,
                kEchoCommonArgCount,
                first_runtime_arg,
                first_common_arg}});
}

KernelHandle create_echo_kernel(Program& program, const CoreCoord& logical_core, uint32_t hart, uint32_t ct_marker) {
    return create_echo_kernel_from(program, logical_core, hart, ct_marker, 0, 0);
}

// First-time set. Common runtime args may be set only once per kernel.
void set_echo_arguments(
    Program& program, KernelHandle kernel, const CoreCoord& logical_core, const EchoArguments& arguments) {
    const std::vector<uint32_t> runtime_args(arguments.runtime.begin(), arguments.runtime.end());
    const std::vector<uint32_t> common_args(arguments.common.begin(), arguments.common.end());
    SetRuntimeArgs(program, kernel, logical_core, runtime_args);
    SetCommonRuntimeArgs(program, kernel, common_args);
}

// Update after launch. Unique args are re-set with the same count; common args are written in place.
void update_echo_arguments(
    Program& program, KernelHandle kernel, const CoreCoord& logical_core, const EchoArguments& arguments) {
    const std::vector<uint32_t> runtime_args(arguments.runtime.begin(), arguments.runtime.end());
    SetRuntimeArgs(program, kernel, logical_core, runtime_args);
    RuntimeArgsData& common_args = GetCommonRuntimeArgs(program, kernel);
    for (size_t index = 0; index < arguments.common.size(); index++) {
        common_args[index] = arguments.common[index];
    }
}

::testing::AssertionResult launch_single_hart_echo(
    IDevice* device, uint32_t cce_index, const CoreCoord& logical_core, uint32_t hart, const EchoArguments& arguments) {
    poison_echo_slots(device, cce_index);
    Program program = CreateProgram();
    const KernelHandle kernel = create_echo_kernel(program, logical_core, hart, arguments.ct_marker);
    set_echo_arguments(program, kernel, logical_core, arguments);
    launch_program_and_wait(device, program);
    return check_echo_slot(hart, read_echo_slots(device, cce_index), arguments);
}

class MimirKernelLoad : public ::testing::TestWithParam<uint32_t> {
protected:
    void SetUp() override { skip_unless_programmable_dram_package(); }
};

TEST_P(MimirKernelLoad, MinimalKernelExecutes) {
    const uint32_t cce_index = GetParam();

    ScopedDevice device_guard;
    IDevice* device = device_guard.device();
    ASSERT_EQ(device->arch(), tt::ARCH::QUASAR);

    const CoreCoord logical_core = logical_cce_core(soc_desc_of(device), cce_index);
    ASSERT_TRUE(check_logical_core_is_cce(device, logical_core, cce_index));
    constexpr uint32_t hart = kCceKernelHart;

    Program null_program = CreateProgram();
    CreateKernel(
        null_program,
        kNullKernelPath,
        logical_core,
        DramConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0});
    launch_program_and_wait(device, null_program);

    const EchoArguments arguments = make_echo_arguments(cce_index, kEchoCtMarkerBase + cce_index);
    Program echo_program = CreateProgram();
    const KernelHandle echo_kernel = create_echo_kernel(echo_program, logical_core, hart, arguments.ct_marker);
    set_echo_arguments(echo_program, echo_kernel, logical_core, arguments);
    poison_echo_slots(device, cce_index);
    launch_program_and_wait(device, echo_program);

    EXPECT_TRUE(check_echo_slot(hart, read_echo_slots(device, cce_index), arguments));
}

TEST_P(MimirKernelLoad, EachHartAlone) {
    const uint32_t cce_index = GetParam();
    const MimirTopology topology = topology_from(configured_soc_desc());

    ScopedDevice device_guard;
    IDevice* device = device_guard.device();
    ASSERT_EQ(device->arch(), tt::ARCH::QUASAR);

    const CoreCoord logical_core = logical_cce_core(soc_desc_of(device), cce_index);
    ASSERT_TRUE(check_logical_core_is_cce(device, logical_core, cce_index));
    for (uint32_t hart = 0; hart < cce_hart_count(); hart++) {
        const EchoArguments arguments = make_echo_arguments(hart, kEchoCtMarkerBase + hart);
        Program program = CreateProgram();
        const KernelHandle kernel = create_echo_kernel(program, logical_core, hart, arguments.ct_marker);
        set_echo_arguments(program, kernel, logical_core, arguments);
        for (uint32_t poisoned_cce = 0; poisoned_cce < topology.num_cces; poisoned_cce++) {
            poison_echo_slots(device, poisoned_cce);
        }
        launch_program_and_wait(device, program);
        for (uint32_t read_cce = 0; read_cce < topology.num_cces; read_cce++) {
            const std::vector<uint32_t> echo_block = read_echo_slots(device, read_cce);
            for (uint32_t slot_hart = 0; slot_hart < cce_hart_count(); slot_hart++) {
                if (read_cce == cce_index && slot_hart == hart) {
                    EXPECT_TRUE(check_echo_slot(hart, echo_block, arguments)) << "CCE " << read_cce;
                } else {
                    EXPECT_TRUE(check_echo_slot_untouched(slot_hart, echo_block)) << "CCE " << read_cce;
                }
            }
        }
    }
}

TEST_P(MimirKernelLoad, ArgsReplaceStale) {
    const uint32_t cce_index = GetParam();

    ScopedDevice device_guard;
    IDevice* device = device_guard.device();
    ASSERT_EQ(device->arch(), tt::ARCH::QUASAR);

    const CoreCoord logical_core = logical_cce_core(soc_desc_of(device), cce_index);
    ASSERT_TRUE(check_logical_core_is_cce(device, logical_core, cce_index));
    std::vector<EchoArguments> first_arguments;
    std::vector<KernelHandle> kernels;
    Program program = CreateProgram();
    for (uint32_t hart = 0; hart < cce_hart_count(); hart++) {
        first_arguments.push_back(make_echo_arguments(hart, kEchoCtMarkerBase + hart));
        kernels.push_back(create_echo_kernel(program, logical_core, hart, first_arguments[hart].ct_marker));
        set_echo_arguments(program, kernels[hart], logical_core, first_arguments[hart]);
    }

    poison_echo_slots(device, cce_index);
    launch_program_and_wait(device, program);
    const std::vector<uint32_t> first_block = read_echo_slots(device, cce_index);
    for (uint32_t hart = 0; hart < cce_hart_count(); hart++) {
        EXPECT_TRUE(check_echo_slot(hart, first_block, first_arguments[hart]));
    }

    std::vector<EchoArguments> updated_arguments;
    for (uint32_t hart = 0; hart < cce_hart_count(); hart++) {
        updated_arguments.push_back(make_updated_arguments(first_arguments[hart], hart));
        update_echo_arguments(program, kernels[hart], logical_core, updated_arguments[hart]);
    }

    poison_echo_slots(device, cce_index);
    launch_program_and_wait(device, program);
    const std::vector<uint32_t> updated_block = read_echo_slots(device, cce_index);
    for (uint32_t hart = 0; hart < cce_hart_count(); hart++) {
        EXPECT_TRUE(check_echo_slot(hart, updated_block, updated_arguments[hart]));
    }
}

// Closing the device and opening it again in one process reloads and resets nothing: MetalContext::initialize returns
// early on unchanged parameters (metal_context.cpp:203-205) and CCE harts are reset only at MetalContext teardown. So
// the launch after the reopen runs on the firmware loaded by the first open. Reset and reload on every open is
// MimirFwReinit (TT_METAL_FORCE_REINIT).
TEST_P(MimirKernelLoad, FirmwareKeepsRunningAcrossReopen) {
    const uint32_t cce_index = GetParam();

    constexpr uint32_t hart = kCceKernelHart;
    constexpr uint32_t launches_per_session = 4;
    ScopedDevice device_guard;
    IDevice* device = device_guard.device();
    ASSERT_EQ(device->arch(), tt::ARCH::QUASAR);

    const CoreCoord logical_core = logical_cce_core(soc_desc_of(device), cce_index);
    ASSERT_TRUE(check_logical_core_is_cce(device, logical_core, cce_index));
    EchoArguments arguments = make_echo_arguments(cce_index, kEchoCtMarkerBase + cce_index);
    uint32_t sequence = 0;
    for (; sequence < launches_per_session; sequence++) {
        arguments.runtime[0] = sequence;
        EXPECT_TRUE(launch_single_hart_echo(device, cce_index, logical_core, hart, arguments));
    }

    device_guard.close();
    ScopedDevice reopened_guard;
    device = reopened_guard.device();
    ASSERT_EQ(device->arch(), tt::ARCH::QUASAR);

    arguments.runtime[0] = sequence;
    EXPECT_TRUE(launch_single_hart_echo(device, cce_index, logical_core, hart, arguments));
}

INSTANTIATE_TEST_SUITE_P(MM3, MimirKernelLoad, ::testing::ValuesIn(mimir_contract::cce_indices()), cce_param_name);

TEST(MimirKernelLoad, SameBinaryAllCcesUnicast) {
    skip_unless_programmable_dram_package();
    const MimirTopology topology = topology_from(configured_soc_desc());
    if (topology.num_cces == 0) {
        GTEST_SKIP() << "Descriptor exposes no CCEs.";
    }

    ScopedDevice device_guard;
    IDevice* device = device_guard.device();
    ASSERT_EQ(device->arch(), tt::ARCH::QUASAR);

    const metal_SocDescriptor& soc = soc_desc_of(device);
    const uint32_t num_cces = num_cces_from_soc(soc);
    std::vector<std::vector<EchoArguments>> arguments_per_cce;
    std::vector<CoreCoord> logical_cores;
    Program program = CreateProgram();
    for (uint32_t cce_index = 0; cce_index < num_cces; cce_index++) {
        logical_cores.push_back(logical_cce_core(soc, cce_index));
        ASSERT_TRUE(check_logical_core_is_cce(device, logical_cores[cce_index], cce_index));
        arguments_per_cce.emplace_back();
        for (uint32_t hart = 0; hart < cce_hart_count(); hart++) {
            // Same compile-time args on every hart of every CCE; runtime and common args differ per hart.
            arguments_per_cce[cce_index].push_back(
                make_echo_arguments(cce_index * cce_hart_count() + hart, kEchoCtMarkerBase));
            const KernelHandle kernel = create_echo_kernel(program, logical_cores[cce_index], hart, kEchoCtMarkerBase);
            set_echo_arguments(program, kernel, logical_cores[cce_index], arguments_per_cce[cce_index][hart]);
        }
    }

    for (uint32_t cce_index = 0; cce_index < num_cces; cce_index++) {
        poison_echo_slots(device, cce_index);
    }
    launch_program_and_wait(device, program);

    for (uint32_t cce_index = 0; cce_index < num_cces; cce_index++) {
        const std::vector<uint32_t> echo_block = read_echo_slots(device, cce_index);
        for (uint32_t hart = 0; hart < cce_hart_count(); hart++) {
            EXPECT_TRUE(check_echo_slot(hart, echo_block, arguments_per_cce[cce_index][hart])) << "CCE " << cce_index;
        }
    }
}

// Combined unique + common runtime-arg ceiling on a DRAM core, as Kernel::validate_runtime_args_size computes it: the
// kernel-config area less the two watcher count words it reserves.
uint32_t dram_runtime_args_ceiling() {
    constexpr uint32_t watcher_reserved_count_words = 2;
    return MetalContext::instance().hal().get_dev_size(HalProgrammableCoreType::DRAM, HalL1MemAddrType::KERNEL_CONFIG) /
               sizeof(uint32_t) -
           watcher_reserved_count_words;
}

// MM-8 runtime-arg host validation on a CCE kernel (MM3 analogues of api/test_runtime_args.cpp
// TensixIllegalTooManyRuntimeArgs and TensixIllegallyModifyRTArgs).
class MimirDmRuntimeArgs : public ::testing::TestWithParam<uint32_t> {
protected:
    void SetUp() override { skip_unless_programmable_dram_package(); }
};

// Unique and common runtime args up to the DRAM-core kernel-config area are accepted; one word more throws from both
// the unique and the common API. A kernel with args at the ceiling launches and echoes its last unique and common args.
TEST_P(MimirDmRuntimeArgs, IllegalTooManyRuntimeArgs) {
    const uint32_t cce_index = GetParam();
    ScopedDevice device_closer;
    IDevice* device = device_closer.device();
    ASSERT_EQ(device->arch(), tt::ARCH::QUASAR);

    constexpr uint32_t hart = kCceKernelHart;
    constexpr uint32_t first_set_count = 100;
    const CoreCoord logical_core = logical_cce_core(soc_desc_of(device), cce_index);
    const uint32_t ceiling = dram_runtime_args_ceiling();
    ASSERT_GT(ceiling, first_set_count);
    const auto unique_then_common = [&logical_core](uint32_t common_count) {
        Program program = CreateProgram();
        const KernelHandle kernel = create_echo_kernel(program, logical_core, hart, kEchoCtMarkerBase);
        SetRuntimeArgs(program, kernel, logical_core, std::vector<uint32_t>(first_set_count));
        SetCommonRuntimeArgs(program, kernel, std::vector<uint32_t>(common_count));
    };
    const auto common_then_unique = [&logical_core](uint32_t unique_count) {
        Program program = CreateProgram();
        const KernelHandle kernel = create_echo_kernel(program, logical_core, hart, kEchoCtMarkerBase);
        SetCommonRuntimeArgs(program, kernel, std::vector<uint32_t>(first_set_count));
        SetRuntimeArgs(program, kernel, logical_core, std::vector<uint32_t>(unique_count));
    };

    EXPECT_NO_THROW(unique_then_common(ceiling - first_set_count));
    EXPECT_ANY_THROW(unique_then_common(ceiling - first_set_count + 1));
    EXPECT_NO_THROW(common_then_unique(ceiling - first_set_count));
    EXPECT_ANY_THROW(common_then_unique(ceiling - first_set_count + 1));

    // 99 unique args leave room for the 16 B alignment of the common args, with or without the watcher's count words.
    constexpr uint32_t ceiling_unique_count = 99;
    const uint32_t ceiling_common_count = ceiling - ceiling_unique_count;
    std::vector<uint32_t> runtime_args(ceiling_unique_count);
    std::iota(runtime_args.begin(), runtime_args.end(), 0x52000000u);
    std::vector<uint32_t> common_args(ceiling_common_count);
    std::iota(common_args.begin(), common_args.end(), 0xC4000000u);
    EchoArguments arguments{.ct_marker = kEchoCtMarkerBase + cce_index};
    std::copy(runtime_args.end() - kEchoRuntimeArgCount, runtime_args.end(), arguments.runtime.begin());
    std::copy(common_args.end() - kEchoCommonArgCount, common_args.end(), arguments.common.begin());
    Program program = CreateProgram();
    const KernelHandle kernel = create_echo_kernel_from(
        program,
        logical_core,
        hart,
        arguments.ct_marker,
        ceiling_unique_count - kEchoRuntimeArgCount,
        ceiling_common_count - kEchoCommonArgCount);
    SetRuntimeArgs(program, kernel, logical_core, runtime_args);
    SetCommonRuntimeArgs(program, kernel, common_args);
    poison_echo_slots(device, cce_index);
    launch_program_and_wait(device, program);
    EXPECT_TRUE(check_echo_slot(hart, read_echo_slots(device, cce_index), arguments))
        << ceiling_unique_count << " unique and " << ceiling_common_count << " common args";
}

std::vector<uint32_t> words_of(const RuntimeArgsData& args) {
    return std::vector<uint32_t>(args.data(), args.data() + args.size());
}

// After a launch, changing the count of unique runtime args throws, and so does setting common runtime args a second
// time with the same count. Neither rejected call changes the stored args.
TEST_P(MimirDmRuntimeArgs, IllegallyModifyRuntimeArgs) {
    const uint32_t cce_index = GetParam();
    ScopedDevice device_closer;
    IDevice* device = device_closer.device();
    ASSERT_EQ(device->arch(), tt::ARCH::QUASAR);

    constexpr uint32_t hart = kCceKernelHart;
    const CoreCoord logical_core = logical_cce_core(soc_desc_of(device), cce_index);
    const EchoArguments arguments = make_echo_arguments(cce_index, kEchoCtMarkerBase + cce_index);
    Program program = CreateProgram();
    const KernelHandle kernel = create_echo_kernel(program, logical_core, hart, arguments.ct_marker);
    set_echo_arguments(program, kernel, logical_core, arguments);
    poison_echo_slots(device, cce_index);
    launch_program_and_wait(device, program);
    EXPECT_TRUE(check_echo_slot(hart, read_echo_slots(device, cce_index), arguments));

    EXPECT_ANY_THROW(
        SetRuntimeArgs(program, kernel, logical_core, std::vector<uint32_t>(kEchoRuntimeArgCount + 1, 0x5050)))
        << "unique runtime arg count changed";
    EXPECT_ANY_THROW(SetCommonRuntimeArgs(program, kernel, std::vector<uint32_t>(kEchoCommonArgCount, 0x6060)))
        << "common runtime args set twice";
    EXPECT_EQ(
        words_of(GetRuntimeArgs(program, kernel, logical_core)),
        std::vector<uint32_t>(arguments.runtime.begin(), arguments.runtime.end()))
        << "a rejected SetRuntimeArgs changed the stored unique args";
    EXPECT_EQ(
        words_of(GetCommonRuntimeArgs(program, kernel)),
        std::vector<uint32_t>(arguments.common.begin(), arguments.common.end()))
        << "a rejected SetCommonRuntimeArgs changed the stored common args";
}

INSTANTIATE_TEST_SUITE_P(MM8, MimirDmRuntimeArgs, ::testing::Values(0u), cce_param_name);

}  // namespace tt::tt_metal::mimir_test::kernel_load
