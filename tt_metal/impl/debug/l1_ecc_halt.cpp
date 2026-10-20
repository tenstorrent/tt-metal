// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "debug/l1_ecc_halt.hpp"

#include <array>
#include <bit>
#include <cstdint>
#include <string>
#include <vector>

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt_stl/assert.hpp>

#include "context/metal_context.hpp"
#include "distributed/mesh_device_impl.hpp"
#include "hostdev/l1_ecc_halt_constants.h"
#include "llrt/hal.hpp"
#include "llrt/tt_cluster.hpp"

namespace tt::tt_metal::internal {

namespace {

constexpr auto l1_ecc_check_interval = std::chrono::seconds(10);
constexpr uint32_t l1_ecc_polls_per_clock_read = 1024;
static_assert(std::has_single_bit(l1_ecc_polls_per_clock_read));

struct RiscMemoryField {
    EccView view;
    uint32_t mask;
    const char* name;
};

constexpr auto risc_memory_fields = std::to_array<RiscMemoryField>({
    {EccView::L1_AND_LOCAL_MEM_COUNTS, ECC_BRISC_LOCAL_MEM_MASK, "BRISC local memory DBE"},
    {EccView::L1_AND_LOCAL_MEM_COUNTS, ECC_TRISC0_LOCAL_MEM_MASK, "TRISC0 local memory DBE"},
    {EccView::L1_AND_LOCAL_MEM_COUNTS, ECC_TRISC1_LOCAL_MEM_MASK, "TRISC1 local memory DBE"},
    {EccView::L1_AND_LOCAL_MEM_COUNTS, ECC_TRISC2_LOCAL_MEM_MASK, "TRISC2 local memory DBE"},
    {EccView::ICACHE_AND_NCRISC_LOCAL_MEM_COUNTS, ECC_NCRISC_LOCAL_MEM_MASK, "NCRISC local memory DBE"},
    {EccView::ICACHE_AND_NCRISC_LOCAL_MEM_COUNTS, ECC_BRISC_ICACHE_MASK, "BRISC instruction cache parity error"},
    {EccView::ICACHE_AND_NCRISC_LOCAL_MEM_COUNTS, ECC_TRISC0_ICACHE_MASK, "TRISC0 instruction cache parity error"},
    {EccView::ICACHE_AND_NCRISC_LOCAL_MEM_COUNTS, ECC_TRISC1_ICACHE_MASK, "TRISC1 instruction cache parity error"},
    {EccView::ICACHE_AND_NCRISC_LOCAL_MEM_COUNTS, ECC_TRISC2_ICACHE_MASK, "TRISC2 instruction cache parity error"},
    {EccView::ICACHE_AND_NCRISC_LOCAL_MEM_COUNTS, ECC_NCRISC_ICACHE_MASK, "NCRISC instruction cache parity error"},
});

// Mock devices have no registers to read, and ttsim exits on any ECC register access.
bool is_blackhole_silicon(const MetalContext& ctx) {
    return ctx.hal().get_arch() == tt::ARCH::BLACKHOLE &&
           ctx.get_cluster().get_target_device_type() == tt::TargetDevice::Silicon;
}

bool l1_ecc_halt_enabled(const MetalContext& ctx) {
    return ctx.rtoptions().get_l1_ecc_halt_enabled() && is_blackhole_silicon(ctx);
}

bool is_halt_reason(uint32_t reason) {
    return reason == L1_ECC_HALT_REASON_DBE || reason == L1_ECC_HALT_REASON_UNEXPECTED ||
           reason == L1_ECC_HALT_REASON_RISC_MEMORY;
}

uint64_t halt_record_addr(const MetalContext& ctx) {
    return ctx.hal().get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::MAILBOX) +
           ctx.hal()
               .get_dev_msgs_factory(HalProgrammableCoreType::TENSIX)
               .offset_of<dev_msgs::mailboxes_t>(dev_msgs::mailboxes_t::Field::l1_ecc_halt);
}

// "bank B row R (L1 0x...)" from ECC_STATUS views 3 (banks 0-15) and 6 (banks 16-31). Assumes the default L1 bank
// interleave.
std::string last_error_location(uint32_t last_l1_error, uint32_t last_l1_error_banks_16_31) {
    uint32_t banks_0_15 = (last_l1_error & ECC_LAST_ERR_BANKS_MASK) >> ECC_LAST_ERR_BANKS_SHIFT;
    uint32_t banks_16_31 = (last_l1_error_banks_16_31 & ECC_LAST_ERR_BANKS_MASK) >> ECC_LAST_ERR_BANKS_SHIFT;
    if ((last_l1_error & ECC_LAST_ERR_VALID_MASK) == 0 || (banks_0_15 | banks_16_31) == 0) {
        return "unknown location";
    }
    uint32_t bank = banks_0_15 != 0 ? std::countr_zero(banks_0_15) : l1_ecc_banks / 2 + std::countr_zero(banks_16_31);
    uint32_t row = last_l1_error & ECC_LAST_ERR_ROW_MASK;
    return fmt::format("bank {} row {} (L1 0x{:x})", bank, row, (row * l1_ecc_banks + bank) * l1_ecc_word_bytes);
}

// Names the RISC memories with a non-zero count in ECC_STATUS views 1 and 5.
std::string risc_memory_errors(uint32_t l1_and_local_mem_counts, uint32_t icache_and_ncrisc_local_mem_counts) {
    std::vector<std::string> errors;
    for (const RiscMemoryField& field : risc_memory_fields) {
        uint32_t counts = field.view == EccView::L1_AND_LOCAL_MEM_COUNTS ? l1_and_local_mem_counts
                                                                         : icache_and_ncrisc_local_mem_counts;
        if (counts & field.mask) {
            errors.push_back(field.name);
        }
    }
    return fmt::format("{}", fmt::join(errors, ", "));
}

std::string halt_cause(const auto& view) {
    switch (view.reason()) {
        case L1_ECC_HALT_REASON_DBE:
            return "an L1 double-bit error at " +
                   last_error_location(view.last_l1_error(), view.last_l1_error_banks_16_31());
        case L1_ECC_HALT_REASON_RISC_MEMORY:
            return "a RISC memory error: " +
                   risc_memory_errors(view.l1_and_local_mem_counts(), view.icache_and_ncrisc_local_mem_counts());
        default: return "an unexpected IRQ 3";
    }
}

void check_chips(MetalContext& ctx, const std::vector<ChipId>& device_ids) {
    if (ctx.l1_ecc_halt_detected()) {
        TT_THROW("A Tensix core halted on an ECC error (reported above). Reset the chip with tt-smi -r.");
    }
    const auto& factory = ctx.hal().get_dev_msgs_factory(HalProgrammableCoreType::TENSIX);
    auto record = factory.create<dev_msgs::l1_ecc_halt_msg_t>();
    const uint64_t record_addr = halt_record_addr(ctx);
    for (ChipId device_id : device_ids) {
        for (const CoreCoord& core : ctx.get_cluster().get_virtual_worker_cores(device_id)) {
            ctx.get_cluster().read_core(record.data(), record.size(), tt_cxy_pair(device_id, core), record_addr);
            auto view = record.view();
            if (!is_halt_reason(view.reason())) {
                continue;
            }
            ctx.set_l1_ecc_halt_detected(true);
            TT_THROW(
                "Chip {} core {}: BRISC halted on {} (program {}, L1 DBE count {}, L1 SBE count {}, ECC_CTRL 0x{:x}). "
                "Reset the chip with tt-smi -r.",
                device_id,
                core.str(),
                halt_cause(view),
                view.program_id(),
                view.l1_and_local_mem_counts() & ECC_DBE_COUNT_MASK,
                (view.l1_and_local_mem_counts() & ECC_SBE_COUNT_MASK) >> ECC_SBE_COUNT_SHIFT,
                view.ecc_ctrl());
        }
    }
}

}  // namespace

void check_no_halted_brisc_at_init(MetalContext& ctx, ChipId device_id) {
    // A BRISC halted by an earlier process blocks firmware init whether or not the feature is on.
    if (!is_blackhole_silicon(ctx)) {
        return;
    }
    const uint32_t brisc_int_no_addr = ctx.hal().get_tensix_debug_reg_addr(TensixDebugReg::BRISC_INT_NO);
    for (const CoreCoord& core : ctx.get_cluster().get_virtual_worker_cores(device_id)) {
        uint32_t int_no = 0;
        ctx.get_cluster().read_reg(&int_no, tt_cxy_pair(device_id, core), brisc_int_no_addr);
        if (int_no != 0) {
            TT_THROW(
                "Chip {} core {}: BRISC is still halted in an interrupt handler (BRISC_INT_NO {}), most likely from "
                "an L1 ECC halt in an earlier process. Reset the chip with tt-smi -r.",
                device_id,
                core.str(),
                int_no);
        }
    }
}

void clear_l1_ecc_halt_records(MetalContext& ctx, ChipId device_id) {
    if (!l1_ecc_halt_enabled(ctx)) {
        return;
    }
    auto zeros = ctx.hal().get_dev_msgs_factory(HalProgrammableCoreType::TENSIX).create<dev_msgs::l1_ecc_halt_msg_t>();
    const uint64_t record_addr = halt_record_addr(ctx);
    for (const CoreCoord& core : ctx.get_cluster().get_virtual_worker_cores(device_id)) {
        ctx.get_cluster().write_core(zeros.data(), zeros.size(), tt_cxy_pair(device_id, core), record_addr);
    }
}

void report_l1_ecc_at_close(MetalContext& ctx, ChipId device_id) {
    if (!l1_ecc_halt_enabled(ctx)) {
        return;
    }
    const auto& factory = ctx.hal().get_dev_msgs_factory(HalProgrammableCoreType::TENSIX);
    auto record = factory.create<dev_msgs::l1_ecc_halt_msg_t>();
    const uint64_t record_addr = halt_record_addr(ctx);
    for (const CoreCoord& core : ctx.get_cluster().get_virtual_worker_cores(device_id)) {
        ctx.get_cluster().read_core(record.data(), record.size(), tt_cxy_pair(device_id, core), record_addr);
        if (is_halt_reason(record.view().reason())) {
            log_error(
                tt::LogMetal,
                "Chip {} core {}: BRISC is halted on {}. Reset the chip with tt-smi -r.",
                device_id,
                core.str(),
                halt_cause(record.view()));
        }
    }
}

L1EccHaltWatch::L1EccHaltWatch(MetalContext& ctx, ChipId device_id) :
    ctx_(l1_ecc_halt_enabled(ctx) ? &ctx : nullptr), device_id_(device_id) {}

L1EccHaltWatch::L1EccHaltWatch(const distributed::MeshDevice* mesh_device) : mesh_device_(mesh_device) {
    if (mesh_device != nullptr) {
        MetalContext& ctx = mesh_device->impl().metal_context();
        ctx_ = l1_ecc_halt_enabled(ctx) ? &ctx : nullptr;
    }
}

void L1EccHaltWatch::check_now() {
    if (ctx_ == nullptr) {
        return;
    }
    if (mesh_device_ != nullptr) {
        std::vector<ChipId> device_ids;
        for (IDevice* device : mesh_device_->get_devices()) {
            device_ids.push_back(device->id());
        }
        check_chips(*ctx_, device_ids);
        return;
    }
    check_chips(*ctx_, {device_id_});
}

void L1EccHaltWatch::stalled() {
    // The wait loops poll back to back, so the clock is read only every few polls.
    if (ctx_ == nullptr || (++polls_ & (l1_ecc_polls_per_clock_read - 1)) != 0) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    // The window starts at this watch's first clock read, so a wait that finishes early never reads the clock.
    if (next_check_ == std::chrono::steady_clock::time_point{}) {
        next_check_ = now + l1_ecc_check_interval;
    }
    // Once one wait has reported a halt, every other wait stops at once instead of at its own next window.
    if (now < next_check_ && !ctx_->l1_ecc_halt_detected()) {
        return;
    }
    next_check_ = now + l1_ecc_check_interval;
    check_now();
}

}  // namespace tt::tt_metal::internal
