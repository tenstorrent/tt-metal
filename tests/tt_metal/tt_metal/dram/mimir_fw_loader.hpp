// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "impl/context/metal_context.hpp"
#include "llrt/tt_cluster.hpp"

namespace tt::tt_metal::mimir_test {

// tl1_offset is the relocated dev address, without l1_noc_offset.
struct CceFwSpan {
    uint64_t tl1_offset = 0;
    std::vector<uint32_t> words;
};

struct CceFwManifest {
    std::string binary_path;
    std::vector<CceFwSpan> spans;
    uint64_t fw_launch_addr = 0;  // reset-vector register (0x02000000)
    uint64_t l1_noc_offset = 0;
    // 0-based TL1 offsets from the HAL for HalProgrammableCoreType::DRAM.
    uint64_t mailbox = 0, launch = 0, go_msg = 0, go_msg_index = 0, launch_msg_rd_ptr = 0, core_info = 0;
    uint64_t bank_to_noc_scratch = 0, bank_to_noc_size = 0;
    uint64_t subordinate_sync = 0;  // mailbox + offsetof(mailboxes_t, subordinate_sync)
    uint64_t fw_globals_ready = 0;  // mailbox + offsetof(mailboxes_t, fw_shared_globals_ready)
};

// Runs a minimal MetalContext initialize when no devices are open, then builds the DRAM-core firmware and collects its
// load image and HAL offsets. Cached per chip.
const CceFwManifest& build_cce_fw_manifest(MetalContext& context, ChipId chip);

// Relocated load spans of the CCE firmware ELF at binary_path, as the runtime loads them.
std::vector<CceFwSpan> cce_fw_spans(const std::string& binary_path);

// Writes a distinct poison to each hart's reset vector on one CCE and asserts every one reads back. Hart 0 goes first,
// because a write to its register reaches all eight. Every device open writes the same vector, so without the poison a
// dropped write passes after any earlier process. Call with the harts held in reset.
void poison_cce_reset_vectors(const Cluster& cluster, ChipId chip, uint32_t cce_index);

}  // namespace tt::tt_metal::mimir_test
