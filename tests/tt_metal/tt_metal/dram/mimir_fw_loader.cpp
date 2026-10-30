// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/tt_metal/dram/mimir_fw_loader.hpp"

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <tt-logger/tt-logger.hpp>

#include "dev_mem_map.h"
#include "hostdevcommon/common_values.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/device/device_manager.hpp"
#include "jit_build/build_env_manager.hpp"
#include "llrt/hal.hpp"
#include "llrt/llrt.hpp"
#include "llrt/tt_cluster.hpp"
#include "tt_metal/tt_metal/dram/mimir_access.hpp"
#include "tt_metal/tt_metal/dram/mimir_hw_state.hpp"
#include "tt_metal/tt_metal/dram/mimir_test_common.hpp"

// dev_mem_map.h defines CCE_RESET_VECTOR_BASE as a macro, which collides with UMD's constant of the same name.
#pragma push_macro("CCE_RESET_VECTOR_BASE")
#undef CCE_RESET_VECTOR_BASE
#include "umd/device/arch/grendel_implementation.hpp"
#pragma pop_macro("CCE_RESET_VECTOR_BASE")

namespace tt::tt_metal::mimir_test {

void poison_cce_reset_vectors(const Cluster& cluster, ChipId chip, uint32_t cce_index) {
    const Hal& hal = MetalContext::instance().hal();
    const uint64_t fw_launch_addr =
        hal.get_jit_build_config(hal.get_programmable_core_type_index(HalProgrammableCoreType::DRAM), 0, 0)
            .fw_launch_addr;
    const tt_cxy_pair core(chip, cce_virtual_core(cluster, chip, cce_index));
    std::vector<uint32_t> poisons(cce_hart_count());
    for (uint32_t hart = 0; hart < poisons.size(); ++hart) {
        poisons[hart] = 0xDEAD0000u | (cce_index << 4) | hart;
        cluster.write_reg(&poisons[hart], core, fw_launch_addr + tt::umd::grendel::CCE_HART_RESET_VECTOR_STRIDE * hart);
    }
    const std::vector<uint64_t> reset_vectors =
        read_cce_reset_vectors(cluster, cluster.get_soc_desc(chip), chip, cce_index);
    for (uint32_t hart = 0; hart < poisons.size(); ++hart) {
        ASSERT_EQ(reset_vectors[hart], poisons[hart])
            << "CCE " << cce_index << " hart " << hart << " reset vector did not take the poison";
    }
}

std::vector<CceFwSpan> cce_fw_spans(const std::string& binary_path) {
    const Hal& hal = MetalContext::instance().hal();
    const uint32_t dram_core_type_index = hal.get_programmable_core_type_index(HalProgrammableCoreType::DRAM);
    const ll_api::memory& binary = tt::llrt::get_risc_binary(binary_path);
    const HalJitBuildConfig& jit_build_config = hal.get_jit_build_config(dram_core_type_index, 0, 0);

    std::vector<CceFwSpan> spans;
    // Same relocation as llrt::test_load_write_read_risc_binary, minus the l1_noc_offset.
    binary.process_spans([&](std::vector<uint32_t>::const_iterator words, uint64_t addr, uint32_t length_words) {
        CceFwSpan span;
        span.tl1_offset = hal.relocate_dev_addr(addr, jit_build_config.local_init_addr);
        span.words.assign(words, words + length_words);
        spans.push_back(std::move(span));
    });
    return spans;
}

const CceFwManifest& build_cce_fw_manifest(MetalContext& context, ChipId chip) {
    static std::map<ChipId, CceFwManifest> manifests_by_chip;
    const auto cached = manifests_by_chip.find(chip);
    if (cached != manifests_by_chip.end()) {
        return cached->second;
    }

    // MetalContext has no public initialized flag. The device manager is marked initialized only when devices are
    // opened, so a context with open devices is left alone. Otherwise the minimal initialize runs. Repeating it with
    // the same arguments returns early (MetalContext::initialize).
    if (not context.device_manager()->is_initialized()) {
        context.initialize(DispatchCoreConfig{}, 1, {}, DEFAULT_L1_SMALL_SIZE, true);
    }

    const Hal& hal = context.hal();
    const ContextId context_id = context.get_context_id();
    const uint8_t num_hw_cqs = 1;
    BuildEnvManager::get_instance(context_id).add_build_env(chip, num_hw_cqs, context_id);
    BuildEnvManager::get_instance(context_id).build_firmware(chip);

    const uint32_t dram_core_type_index = hal.get_programmable_core_type_index(HalProgrammableCoreType::DRAM);
    const std::string binary_path =
        BuildEnvManager::get_instance(context_id).get_firmware_binary_path(chip, dram_core_type_index, 0, 0);
    const HalJitBuildConfig& jit_build_config = hal.get_jit_build_config(dram_core_type_index, 0, 0);

    CceFwManifest manifest;
    manifest.binary_path = binary_path;
    manifest.spans = cce_fw_spans(manifest.binary_path);
    manifest.fw_launch_addr = jit_build_config.fw_launch_addr;
    manifest.l1_noc_offset = jit_build_config.l1_noc_offset;

    manifest.mailbox = hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::MAILBOX);
    manifest.launch = hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::LAUNCH);
    manifest.go_msg = hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::GO_MSG);
    manifest.go_msg_index = hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::GO_MSG_INDEX);
    manifest.launch_msg_rd_ptr =
        hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::LAUNCH_MSG_BUFFER_RD_PTR);
    manifest.core_info = hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::CORE_INFO);
    manifest.bank_to_noc_scratch =
        hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::BANK_TO_NOC_SCRATCH);
    manifest.bank_to_noc_size = hal.get_dev_size(HalProgrammableCoreType::DRAM, HalL1MemAddrType::BANK_TO_NOC_SCRATCH);

    // Same offsets as RiscFirmwareInitializer::initialize_firmware, stored without the NOC offset.
    const dev_msgs::Factory& factory = hal.get_dev_msgs_factory(HalProgrammableCoreType::DRAM);
    manifest.subordinate_sync =
        manifest.mailbox + factory.offset_of<dev_msgs::mailboxes_t>(dev_msgs::mailboxes_t::Field::subordinate_sync);
    manifest.fw_globals_ready = manifest.mailbox + factory.offset_of<dev_msgs::mailboxes_t>(
                                                       dev_msgs::mailboxes_t::Field::fw_shared_globals_ready);

    return manifests_by_chip.emplace(chip, std::move(manifest)).first->second;
}

}  // namespace tt::tt_metal::mimir_test
