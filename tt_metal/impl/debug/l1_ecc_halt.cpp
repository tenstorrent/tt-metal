// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "debug/l1_ecc_halt.hpp"

#include <bit>
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

#include <fmt/format.h>
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
// A released BRISC returns from the handler within its first few instructions.
constexpr auto brisc_release_timeout = std::chrono::milliseconds(100);

// Mock devices have no registers to read, and ttsim exits on any ECC register access.
bool is_blackhole_silicon(const MetalContext& ctx) {
    return ctx.hal().get_arch() == tt::ARCH::BLACKHOLE &&
           ctx.get_cluster().get_target_device_type() == tt::TargetDevice::Silicon;
}

bool l1_ecc_halt_enabled(const MetalContext& ctx) {
    return ctx.rtoptions().get_l1_ecc_halt_enabled() && is_blackhole_silicon(ctx);
}

bool is_halt_reason(uint32_t reason) {
    return reason == L1_ECC_HALT_REASON_DBE || reason == L1_ECC_HALT_REASON_UNEXPECTED;
}

uint64_t halt_slot_field_addr(const MetalContext& ctx, dev_msgs::l1_ecc_halt_slot_t::Field field) {
    return ctx.hal().get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::L1_ECC_HALT) +
           ctx.hal()
               .get_dev_msgs_factory(HalProgrammableCoreType::TENSIX)
               .offset_of<dev_msgs::l1_ecc_halt_slot_t>(field);
}

// "bank B row R (L1 0x...)" from ECC_STATUS views 3 (banks 0-15) and 6 (banks 16-31). Assumes the default L1 bank
// interleave.
std::string last_error_location(uint32_t last_l1_error_banks_0_15, uint32_t last_l1_error_banks_16_31) {
    uint32_t banks_0_15 = (last_l1_error_banks_0_15 & ECC_LAST_ERR_BANKS_MASK) >> ECC_LAST_ERR_BANKS_SHIFT;
    uint32_t banks_16_31 = (last_l1_error_banks_16_31 & ECC_LAST_ERR_BANKS_MASK) >> ECC_LAST_ERR_BANKS_SHIFT;
    if ((last_l1_error_banks_0_15 & ECC_LAST_ERR_VALID_MASK) == 0 || (banks_0_15 | banks_16_31) == 0) {
        return "unknown location";
    }
    uint32_t bank = banks_0_15 != 0 ? std::countr_zero(banks_0_15) : l1_ecc_banks / 2 + std::countr_zero(banks_16_31);
    uint32_t row = last_l1_error_banks_0_15 & ECC_LAST_ERR_ROW_MASK;
    return fmt::format("bank {} row {} (L1 0x{:x})", bank, row, (row * l1_ecc_banks + bank) * l1_ecc_word_bytes);
}

std::string halt_cause(const auto& view) {
    switch (view.reason()) {
        case L1_ECC_HALT_REASON_DBE:
            return "an L1 double-bit error at " +
                   last_error_location(view.last_l1_error_banks_0_15(), view.last_l1_error_banks_16_31());
        default: return "an unexpected IRQ 3";
    }
}

void check_chips(MetalContext& ctx, const std::vector<ChipId>& device_ids) {
    if (ctx.l1_ecc_halt_detected()) {
        TT_THROW("A Tensix core halted on an ECC error (reported above). Reopening the device releases it.");
    }
    const auto& factory = ctx.hal().get_dev_msgs_factory(HalProgrammableCoreType::TENSIX);
    auto record = factory.create<dev_msgs::l1_ecc_halt_msg_t>();
    const uint64_t record_addr = halt_slot_field_addr(ctx, dev_msgs::l1_ecc_halt_slot_t::Field::record);
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
                "Reopening the device releases it.",
                device_id,
                core.str(),
                halt_cause(view),
                view.program_id(),
                view.l1_counts() & ECC_DBE_COUNT_MASK,
                (view.l1_counts() & ECC_SBE_COUNT_MASK) >> ECC_SBE_COUNT_SHIFT,
                view.ecc_ctrl());
        }
    }
}

}  // namespace

std::vector<LatchedBrisc> find_latched_briscs(MetalContext& ctx, ChipId device_id) {
    std::vector<LatchedBrisc> latched;
    if (!is_blackhole_silicon(ctx)) {
        return latched;
    }
    const uint32_t brisc_int_no_addr = ctx.hal().get_tensix_debug_reg_addr(TensixDebugReg::BRISC_INT_NO);
    auto record = ctx.hal().get_dev_msgs_factory(HalProgrammableCoreType::TENSIX).create<dev_msgs::l1_ecc_halt_msg_t>();
    const uint64_t record_addr = halt_slot_field_addr(ctx, dev_msgs::l1_ecc_halt_slot_t::Field::record);
    for (const CoreCoord& core : ctx.get_cluster().get_virtual_worker_cores(device_id)) {
        const tt_cxy_pair target(device_id, core);
        uint32_t int_no = 0;
        ctx.get_cluster().read_reg(&int_no, target, brisc_int_no_addr);
        // The release writes PIC_INT_PCS[int_no]; an unreachable core reads all ones.
        if (int_no == 0 || int_no >= pic_num_sw_ints + pic_num_hw_ints) {
            continue;
        }
        ctx.get_cluster().read_core(record.data(), record.size(), target, record_addr);
        if (is_halt_reason(record.view().reason())) {
            log_warning(
                tt::LogMetal,
                "Chip {} core {}: BRISC halted on {} in an earlier process. Releasing it.",
                device_id,
                core.str(),
                halt_cause(record.view()));
        }
        latched.push_back({core, int_no});
    }
    return latched;
}

void release_latched_briscs(MetalContext& ctx, ChipId device_id, const std::vector<LatchedBrisc>& latched) {
    // The PIC keeps a BRISC's interrupt latched across the RISC reset, and the BRISC takes it again right after it is
    // released, before firmware has a stack. Pointing that interrupt at code that only returns frees the PIC; firmware
    // init then installs its own handler.
    if (latched.empty()) {  // also the only case off Blackhole silicon, which has no Tensix debug registers
        return;
    }
    const uint32_t int_pcs_addr = ctx.hal().get_tensix_debug_reg_addr(TensixDebugReg::PIC_INT_PCS);
    const uint32_t wdt_cntl_addr = ctx.hal().get_tensix_debug_reg_addr(TensixDebugReg::WDT_CNTL);
    constexpr uint32_t wdt_off = 0;
    const auto release_addr =
        static_cast<uint32_t>(halt_slot_field_addr(ctx, dev_msgs::l1_ecc_halt_slot_t::Field::brisc_int_release));
    for (const auto& [core, int_no] : latched) {
        const tt_cxy_pair target(device_id, core);
        ctx.get_cluster().write_core(brisc_int_release_code, sizeof(brisc_int_release_code), target, release_addr);
        ctx.get_cluster().write_reg(&release_addr, target, int_pcs_addr + (int_no * sizeof(uint32_t)));
        // The watchdog interrupt survives the RISC reset; left raised, the core halts again once firmware arms IRQ 3.
        ctx.get_cluster().write_reg(&wdt_off, target, wdt_cntl_addr);
    }
}

void check_briscs_released(MetalContext& ctx, ChipId device_id, const std::vector<LatchedBrisc>& latched) {
    if (latched.empty()) {
        return;
    }
    const uint32_t brisc_int_no_addr = ctx.hal().get_tensix_debug_reg_addr(TensixDebugReg::BRISC_INT_NO);
    const auto deadline = std::chrono::steady_clock::now() + brisc_release_timeout;
    for (const LatchedBrisc& brisc : latched) {
        uint32_t int_no = 0;
        do {
            ctx.get_cluster().read_reg(&int_no, tt_cxy_pair(device_id, brisc.core), brisc_int_no_addr);
        } while (int_no != 0 && std::chrono::steady_clock::now() < deadline);
        if (int_no != 0) {
            TT_THROW(
                "Chip {} core {}: BRISC is still inside an interrupt handler (BRISC_INT_NO {}) after it was released "
                "at init. Reset the chip with tt-smi -r.",
                device_id,
                brisc.core.str(),
                int_no);
        }
    }
}

void clear_l1_ecc_halt_records(MetalContext& ctx, ChipId device_id) {
    if (!l1_ecc_halt_enabled(ctx)) {
        return;
    }
    auto zeros = ctx.hal().get_dev_msgs_factory(HalProgrammableCoreType::TENSIX).create<dev_msgs::l1_ecc_halt_msg_t>();
    const uint64_t record_addr = halt_slot_field_addr(ctx, dev_msgs::l1_ecc_halt_slot_t::Field::record);
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
    const uint64_t record_addr = halt_slot_field_addr(ctx, dev_msgs::l1_ecc_halt_slot_t::Field::record);
    for (const CoreCoord& core : ctx.get_cluster().get_virtual_worker_cores(device_id)) {
        ctx.get_cluster().read_core(record.data(), record.size(), tt_cxy_pair(device_id, core), record_addr);
        if (is_halt_reason(record.view().reason())) {
            log_error(
                tt::LogMetal,
                "Chip {} core {}: BRISC is halted on {}; the next device open releases it.",
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

void L1EccHaltWatch::poll() {
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
