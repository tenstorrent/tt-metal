// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "x280/device/x280_tile.hpp"
#include "x280/device/x280_active.hpp"
#include "x280/device/x280_runtime.hpp"

#include "x280_format.hpp"
#include "x280_hw_addrs.hpp"
#include "x280/device/x280_deadline.hpp"
#include "x280_tile_impl.hpp"

#include "x280_address_map.h"
#include "x280_hart_protocol.h"

#include <tt-umd/types/telemetry.hpp>

#include <algorithm>
#include <array>
#include <exception>

namespace x280 {

namespace {

constexpr std::chrono::milliseconds kPollInterval{5};

constexpr std::array<const char*, 8> kBootOutcomeNames = {
    "booted",
    "already_running",
    "prime_required",
    "reset_required_no_lim",
    "reset_required_no_resident_idle",
    "prepared_reset_not_released",
    "invalid_observation",
    "failed",
};

bool all_resident_harts_idle(const std::array<HartState, kHartsPerTile>& harts) {
    return std::all_of(harts.begin(), harts.end(), hart_resident_idle);
}

}  // namespace

const char* to_string(BootOutcome outcome) { return enum_name(outcome, kBootOutcomeNames); }

bool reset_required(BootOutcome outcome) {
    switch (outcome) {
        case BootOutcome::ResetRequiredNoLim:
        case BootOutcome::ResetRequiredNoResidentIdle: return true;
        default: return false;
    }
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

namespace detail {

TileImpl::TileImpl(RuntimeImpl& runtime, ChipImpl& chip, TileIdentity identity) :
    runtime_(&runtime), chip_(&chip), identity_(identity) {
    if (identity_.chip_id != chip_->chip_id()) {
        fail(
            "TileImpl identity chip " + std::to_string(identity_.chip_id) + " does not match chip context " +
            std::to_string(chip_->chip_id()));
    }
}

// ---------------------------------------------------------------------------
// Observation
// ---------------------------------------------------------------------------

TileState TileImpl::inspect() const {
    runtime_->require_mesh_live("inspect", &id());
    const std::lock_guard<std::mutex> serialise(chip_->lock());
    return inspect_state();
}

TileState TileImpl::inspect_state() const {
    TileState state;
    state.id = id();

    try {
        const uint32_t reset = chip_->read_l2cpu_reset_raw();
        if (is_all_ones(reset)) {
            state.invalid_reason = "L2CPU_RESET read all ones";
            return state;
        }
        state.reset_released = ChipImpl::tile_reset_released(reset, id().hw_index);

        const auto ecc = read_lim_ecc_unlocked();
        state.raw_wayenable = ecc.raw_wayenable;
        state.raw_ecc_marker = ecc.raw_marker;
        state.raw_ecc_extent = ecc.raw_extent;
        state.ecc = ecc.state;
        state.ecc_primed_bytes = ecc.primed_bytes;
        if (ecc.state == LimEccState::Invalid) {
            state.invalid_reason = "L3 WayEnable or the ECC marker read all ones";
            return state;
        }
        state.lim_backed_bytes = x280::lim_backed_bytes(ecc.raw_wayenable);
        state.pll_mhz = transport().telemetry(id().chip_id, static_cast<uint8_t>(tt::umd::L2CPUCLK0 + id().hw_index));

        // No LIM address is meaningful when all ways are cache.
        if (ecc.state == LimEccState::LimUnavailable) {
            state.observation_valid = true;
            return state;
        }

        state.harts = read_hart_states_unlocked();
        state.resident_idle_live = state.reset_released && all_resident_harts_idle(state.harts);
        state.observation_valid = true;
    } catch (const std::exception& error) {
        state.observation_valid = false;
        state.invalid_reason = std::string("transport error while reading the tile: ") + error.what();
    }
    return state;
}

LimRange TileImpl::reserved_lim() const {
    LimRange range;
    if (runtime_->lim_layout() == LimLayout::PacketInjection) {
        // Low image/idle prefix [0x08000000, 0x08020000). Upper stacks and protocol
        // are also excluded from the lower arena.
        range.begin = X280_IDLE_FW_LOAD_ADDR;
        range.end = X280_LIM_PACKET_INJECTION_ARENA_BASE;
        return range;
    }
    // Everything from the resident idle through the boot protocol. A caller
    // placing a socket config buffer or FIFO anywhere in here corrupts a running
    // image, and nothing detects it afterwards.
    range.begin = X280_IDLE_FW_LOAD_ADDR;
    range.end = X280_HART_PROTOCOL_END;
    return range;
}

LimRange TileImpl::available_lim() const {
    const auto state = inspect();
    return available_lim_unlocked(state);
}

LimRange TileImpl::available_lim_unlocked(const TileState& state) const {
    if (!state.observation_valid || state.lim_backed_bytes == 0) {
        return LimRange{};
    }
    return available_lim_for_layout(runtime_->lim_layout(), state.lim_backed_bytes);
}

std::vector<uint8_t> TileImpl::read_available_lim(uint64_t address, size_t size) const {
    runtime_->require_mesh_live("read_available_lim", &id());
    const std::lock_guard<std::mutex> serialise(chip_->lock());
    const auto state = inspect_state();
    if (!state.observation_valid) {
        fail(label_chip_l2cpu(id().chip_id, id().hw_index) + ": cannot read LIM: " + state.invalid_reason);
    }
    if (state.ecc != LimEccState::Primed) {
        fail(
            label_chip_l2cpu(id().chip_id, id().hw_index) + ": cannot read available LIM while ECC state is " +
            to_string(state.ecc));
    }
    const auto check = validate_available_lim_read(available_lim_unlocked(state), address, size);
    if (!check.ok) {
        fail(label_chip_l2cpu(id().chip_id, id().hw_index) + ": " + check.reason);
    }

    std::vector<uint8_t> data(size, 0);
    transport().read_block(id().chip_id, core(), address, data.data(), data.size());
    return data;
}

void TileImpl::write_available_lim(uint64_t address, std::span<const uint8_t> data) {
    runtime_->require_mesh_live("write_available_lim", &id());
    const std::lock_guard<std::mutex> serialise(chip_->lock());
    const auto state = inspect_state();
    if (!state.observation_valid) {
        fail(label_chip_l2cpu(id().chip_id, id().hw_index) + ": cannot write LIM: " + state.invalid_reason);
    }
    if (state.ecc != LimEccState::Primed) {
        fail(
            label_chip_l2cpu(id().chip_id, id().hw_index) + ": cannot write available LIM while ECC state is " +
            to_string(state.ecc));
    }
    const auto check = validate_available_lim_write(available_lim_unlocked(state), address, data.size());
    if (!check.ok) {
        fail(label_chip_l2cpu(id().chip_id, id().hw_index) + ": " + check.reason);
    }

    transport().write_block(id().chip_id, core(), address, data.data(), data.size());
}

// ---------------------------------------------------------------------------
// Resident-idle boot
// ---------------------------------------------------------------------------

bool TileImpl::classify_cold_boot_unlocked(BootResult& result) const {
    result.last_state = inspect_state();
    const TileState state = result.last_state;

    if (!state.observation_valid) {
        result.outcome = BootOutcome::InvalidObservation;
        result.diagnostic = label_chip_l2cpu(id().chip_id, id().hw_index) + ": " + state.invalid_reason;
        return false;
    }
    if (state.ecc == LimEccState::LimUnavailable) {
        result.outcome = BootOutcome::ResetRequiredNoLim;
        result.diagnostic = label_chip_l2cpu(id().chip_id, id().hw_index) +
                            ": WayEnable leaves no LIM. An ASIC reset is required before resident idle can boot.";
        return false;
    }
    if (state.ecc == LimEccState::Unprimed) {
        result.outcome = BootOutcome::PrimeRequired;
        result.diagnostic = label_chip_l2cpu(id().chip_id, id().hw_index) +
                            " has unprimed LIM ECC. Close Metal, reset the device, and call X280Runtime::open().";
        return false;
    }
    if (state.resident_idle_live) {
        result.outcome = BootOutcome::AlreadyRunning;
        result.diagnostic = label_chip_l2cpu(id().chip_id, id().hw_index) +
                            " already has a live resident idle; reset and the resident image were untouched";
        return false;
    }
    if (state.reset_released) {
        result.outcome = BootOutcome::ResetRequiredNoResidentIdle;
        result.diagnostic = label_chip_l2cpu(id().chip_id, id().hw_index) +
                            " is out of reset without a live versioned per-hart resident idle: " + describe(state) +
                            ". Reset cannot be reasserted in this ASIC-reset lifetime.";
        return false;
    }
    return true;
}

size_t TileImpl::prepare_cold_boot_unlocked(const std::string& idle_image_path) {
    const auto image = read_image_file(idle_image_path, "resident-idle image", "ninja x280_images");
    load_idle_image_unlocked(image);
    set_reset_vectors_unlocked(X280_IDLE_FW_LOAD_ADDR);
    zero_hart_protocol_unlocked();
    return image.size();
}

bool TileImpl::confirm_resident_idle_unlocked(std::chrono::milliseconds timeout, BootResult& result) {
    const Deadline deadline{timeout};
    const auto poll = poll_until(deadline, kPollInterval, [&] {
        const auto phases = read_hart_phases();
        if (!std::all_of(phases.begin(), phases.end(), [](uint64_t phase) { return phase == X280_HART_PHASE_IDLE; })) {
            return false;
        }
        return all_resident_harts_idle(read_hart_states_unlocked());
    });

    if (!poll.satisfied) {
        result.outcome = BootOutcome::ResetRequiredNoResidentIdle;
        result.last_state = inspect_state();
        result.diagnostic = "resident idle did not come up within " + std::to_string(timeout.count()) + " ms on " +
                            label_chip_l2cpu(id().chip_id, id().hw_index) + " (" + std::to_string(poll.polls) +
                            " polls). " + describe(result.last_state) + ". Reset remains released.";
        return false;
    }

    result.last_state = inspect_state();
    if (!result.last_state.observation_valid || !result.last_state.resident_idle_live) {
        result.outcome = BootOutcome::ResetRequiredNoResidentIdle;
        result.diagnostic = label_chip_l2cpu(id().chip_id, id().hw_index) +
                            ": resident-idle mailboxes appeared after reset release, but the authoritative "
                            "inspection did not confirm a live resident idle";
        return false;
    }

    result.outcome = BootOutcome::Booted;
    result.diagnostic =
        label_chip_l2cpu(id().chip_id, id().hw_index) + " cold-booted into resident idle at the safe clock";
    return true;
}

BootResult TileImpl::boot(const BootRequest& request) {
    BootResult result;

    if (request.boot_timeout.count() <= 0) {
        result.diagnostic = "boot_timeout is " + std::to_string(request.boot_timeout.count()) +
                            " ms; it bounds the wait after one-way reset release, so it must be positive";
        return result;
    }
    if (request.clock_confirm_timeout.count() <= 0) {
        result.diagnostic = "clock_confirm_timeout is " + std::to_string(request.clock_confirm_timeout.count()) +
                            " ms; it bounds PLL telemetry confirmation, so it must be positive";
        return result;
    }

    bool reset_release_attempted = false;
    try {
        runtime_->require_mesh_live("boot", &id());
        const std::lock_guard<std::mutex> serialise(chip_->lock());

        const bool quiescent = chip_->all_tiles_held_in_reset();
        result.last_state = inspect_state();
        const TileState state = result.last_state;

        if (!state.observation_valid) {
            result.outcome = BootOutcome::InvalidObservation;
            result.diagnostic = label_chip_l2cpu(id().chip_id, id().hw_index) + ": " + state.invalid_reason +
                                ". Runtime initialization established a readable chip; this is a transport or "
                                "hardware failure.";
            return result;
        }

        if (state.ecc == LimEccState::LimUnavailable) {
            result.outcome = BootOutcome::ResetRequiredNoLim;
            result.diagnostic = label_chip_l2cpu(id().chip_id, id().hw_index) +
                                ": WayEnable leaves no LIM. An ASIC reset is required before resident idle can boot.";
            return result;
        }

        if (state.ecc == LimEccState::Unprimed) {
            result.outcome = BootOutcome::PrimeRequired;
            result.diagnostic = label_chip_l2cpu(id().chip_id, id().hw_index) +
                                " has unprimed LIM ECC. Close Metal, reset the device, and call X280Runtime::open() "
                                "so the runtime initialization boundary can bootstrap the whole chip.";
            return result;
        }

        if (state.resident_idle_live) {
            result.outcome = BootOutcome::AlreadyRunning;
            result.diagnostic = label_chip_l2cpu(id().chip_id, id().hw_index) +
                                " already has a live resident idle; reset and the resident image were untouched";
            return result;
        }

        if (state.reset_released) {
            result.outcome = BootOutcome::ResetRequiredNoResidentIdle;
            result.diagnostic = label_chip_l2cpu(id().chip_id, id().hw_index) +
                                " is out of reset without a live versioned per-hart resident idle: " + describe(state) +
                                ". Reset cannot be reasserted in this ASIC-reset lifetime.";
            return result;
        }

        if (!request.allow_reset_release && !request.dry_run) {
            result.diagnostic = label_chip_l2cpu(id().chip_id, id().hw_index) +
                                " is held in reset and boot requires allow_reset_release";
            return result;
        }

        if (!request.dry_run) {
            if (!quiescent) {
                result.diagnostic = label_chip_l2cpu(id().chip_id, id().hw_index) +
                                    " needs a cold boot, but another L2CPU is out of reset. The runtime owner must "
                                    "establish the safe reset-release clock before any L2CPU leaves reset.";
                return result;
            }
            // Every L2CPU on the chip is held in reset, proven immediately
            // above. That is the only window in which the chip-global PLL can
            // move without a running hart seeing its frequency change, and it
            // closes the moment reset is released below, so the safe clock is
            // established here rather than left to the caller.
            if (!state.pll_mhz.has_value() || *state.pll_mhz != kSafePllMhz) {
                const auto established = chip_->establish_pll(kSafePllMhz, request.clock_confirm_timeout);
                if (!established.confirmed) {
                    result.diagnostic = label_chip_l2cpu(id().chip_id, id().hw_index) +
                                        ": could not establish the safe reset-release clock: " + established.diagnostic;
                    return result;
                }
                result.last_state = inspect_state();
                if (!result.last_state.observation_valid) {
                    result.outcome = BootOutcome::InvalidObservation;
                    result.diagnostic =
                        "inspection after the safe reset-release clock failed: " + result.last_state.invalid_reason;
                    return result;
                }
            }
        }

        const auto image = read_image_file(request.idle_image_path, "resident-idle image", "ninja x280_images");
        load_idle_image_unlocked(image);
        set_reset_vectors_unlocked(X280_IDLE_FW_LOAD_ADDR);
        zero_hart_protocol_unlocked();

        if (request.dry_run) {
            result.last_state = inspect_state();
            if (!result.last_state.observation_valid) {
                result.outcome = BootOutcome::InvalidObservation;
                result.diagnostic = "dry-run verification inspection failed: " + result.last_state.invalid_reason;
                return result;
            }
            result.outcome = BootOutcome::PreparedResetNotReleased;
            result.diagnostic = "dry run prepared and verified " + std::to_string(image.size()) +
                                " resident-idle bytes on " + label_chip_l2cpu(id().chip_id, id().hw_index) +
                                "; reset remains asserted";
            return result;
        }

        // A throwing write may still have landed, so every later failure is
        // reset-required once the release attempt begins.
        reset_release_attempted = true;
        chip_->release_tile_reset(id().hw_index);

        const Deadline deadline{request.boot_timeout};
        const auto poll = poll_until(deadline, kPollInterval, [&] {
            const auto phases = read_hart_phases();
            if (!std::all_of(
                    phases.begin(), phases.end(), [](uint64_t phase) { return phase == X280_HART_PHASE_IDLE; })) {
                return false;
            }
            return all_resident_harts_idle(read_hart_states_unlocked());
        });

        if (!poll.satisfied) {
            result.outcome = BootOutcome::ResetRequiredNoResidentIdle;
            result.last_state = inspect_state();
            result.diagnostic = "resident idle did not come up within " + std::to_string(request.boot_timeout.count()) +
                                " ms on " + label_chip_l2cpu(id().chip_id, id().hw_index) + " (" +
                                std::to_string(poll.polls) +
                                " polls; the bound may include one final wrapper operation). " +
                                describe(result.last_state) + ". Reset remains released.";
            return result;
        }

        result.last_state = inspect_state();
        if (!result.last_state.observation_valid || !result.last_state.resident_idle_live) {
            result.outcome = BootOutcome::ResetRequiredNoResidentIdle;
            result.diagnostic =
                "resident-idle mailboxes appeared after reset release, but the authoritative "
                "inspection did not confirm a live resident idle";
            return result;
        }

        result.outcome = BootOutcome::Booted;
        result.diagnostic =
            label_chip_l2cpu(id().chip_id, id().hw_index) + " cold-booted into resident idle at the safe clock";
        return result;
    } catch (const std::exception& error) {
        result.outcome = reset_release_attempted ? BootOutcome::ResetRequiredNoResidentIdle : BootOutcome::Failed;
        result.diagnostic = reset_release_attempted
                                ? std::string("failure after reset release; an ASIC reset is required: ") + error.what()
                                : error.what();
        return result;
    } catch (...) {
        result.outcome = reset_release_attempted ? BootOutcome::ResetRequiredNoResidentIdle : BootOutcome::Failed;
        result.diagnostic = reset_release_attempted ? "unknown failure after reset release; an ASIC reset is required"
                                                    : "unknown failure before reset release";
        return result;
    }
}

// Outbound NOC windows
// ---------------------------------------------------------------------------

NocWindowResult TileImpl::program_noc_window(uint32_t index, const TranslatedCoord& target, const NocWindow2M& window) {
    NocWindowResult result;

    const auto allowed = validate_host_window_index(index);
    if (!allowed.ok) {
        result.diagnostic = allowed.reason;
        return result;
    }
    const auto encoded = encode(target, window);
    if (!encoded.ok()) {
        result.diagnostic = describe(encoded);
        return result;
    }
    runtime_->require_mesh_live("program_noc_window", &id());
    const std::lock_guard<std::mutex> serialise(chip_->lock());

    try {
        const auto tile = core();
        const uint64_t addr = kNocTlb2mConfigBase + noc_window_config_offset(index);

        // The config block is a register file on the peripheral port, so it
        // takes the register path: 32-bit chunked, strictly ordered, and
        // readback-verified per word.
        for (uint32_t i = 0; i < 4; i++) {
            transport().write_reg32_verified(id().chip_id, tile, addr + i * sizeof(uint32_t), encoded.words.data[i]);
        }

        NocWindowWords seen;
        for (uint32_t i = 0; i < 4; i++) {
            seen.data[i] = transport().read_reg32(id().chip_id, tile, addr + i * sizeof(uint32_t));
        }
        result.words = seen;
        result.ok = seen == encoded.words;
        if (!result.ok) {
            result.diagnostic = "NOC window " + std::to_string(index) + " on L2CPU " + std::to_string(id().hw_index) +
                                " did not latch what was written";
            return result;
        }
        const auto decoded = decode(seen);
        if (!decoded.ok()) {
            result.ok = false;
            result.diagnostic = "NOC window readback is not a valid unicast entry: " + describe(decoded);
            return result;
        }
        result.observed_coord = decoded.coord;
        result.observed = decoded.window;
        result.diagnostic = "NOC window " + std::to_string(index) + " on L2CPU " + std::to_string(id().hw_index) +
                            " points at translated (" + std::to_string(target.x()) + "," + std::to_string(target.y()) +
                            ") on NOC" + (window.noc1 ? "1 " : "0 ") + hex(window.target_addr);
        return result;
    } catch (const std::exception& error) {
        result.diagnostic = error.what();
        return result;
    }
}

NocWindowResult TileImpl::read_noc_window(uint32_t index) const {
    NocWindowResult result;

    if (index >= kNoc2mWindowCount) {
        result.diagnostic = "NOC window index " + std::to_string(index) + " is outside the hardware range";
        return result;
    }

    runtime_->require_mesh_live("read_noc_window", &id());
    const std::lock_guard<std::mutex> serialise(chip_->lock());

    try {
        const auto tile = core();
        const uint64_t addr = kNocTlb2mConfigBase + noc_window_config_offset(index);
        NocWindowWords seen;
        for (uint32_t i = 0; i < 4; i++) {
            seen.data[i] = transport().read_reg32(id().chip_id, tile, addr + i * sizeof(uint32_t));
        }
        result.words = seen;
        if (is_unusable(seen)) {
            result.diagnostic = "NOC window " + std::to_string(index) + " on L2CPU " + std::to_string(id().hw_index) +
                                " read all ones: the tile is unclocked or the transport is broken. Decoding it "
                                "would report coordinate (63,63) and a plausible address.";
            return result;
        }
        const auto decoded = decode(seen);
        if (!decoded.ok()) {
            result.diagnostic = describe(decoded);
            return result;
        }
        result.observed_coord = decoded.coord;
        result.observed = decoded.window;
        result.ok = true;
        return result;
    } catch (const std::exception& error) {
        result.diagnostic = error.what();
        return result;
    }
}

TileTestAccess::TileTestAccess(X280Tile& tile) {
    if (!tile.impl_) {
        fail("TileTestAccess cannot bind a moved-from tile");
    }
    tile_ = tile.impl_.get();
}

uint64_t TileTestAccess::read_u64(uint64_t addr) const {
    tile_->runtime_->require_mesh_live("TileTestAccess::read_u64", &tile_->id());
    const std::lock_guard<std::mutex> serialise(tile_->chip_->lock());
    return tile_->transport().read_u64(tile_->id().chip_id, tile_->core(), addr);
}

void TileTestAccess::write_u64(uint64_t addr, uint64_t value) {
    tile_->runtime_->require_mesh_live("TileTestAccess::write_u64", &tile_->id());
    const std::lock_guard<std::mutex> serialise(tile_->chip_->lock());
    tile_->transport().write_u64(tile_->id().chip_id, tile_->core(), addr, value);
}

std::vector<uint8_t> TileTestAccess::read_block(uint64_t addr, size_t size) const {
    tile_->runtime_->require_mesh_live("TileTestAccess::read_block", &tile_->id());
    const std::lock_guard<std::mutex> serialise(tile_->chip_->lock());
    std::vector<uint8_t> data(size, 0);
    tile_->transport().read_block(tile_->id().chip_id, tile_->core(), addr, data.data(), data.size());
    return data;
}

void TileTestAccess::write_block(uint64_t addr, const std::vector<uint8_t>& data) {
    tile_->runtime_->require_mesh_live("TileTestAccess::write_block", &tile_->id());
    const std::lock_guard<std::mutex> serialise(tile_->chip_->lock());
    tile_->transport().write_block(tile_->id().chip_id, tile_->core(), addr, data.data(), data.size());
}

NocWindowWords TileTestAccess::read_noc_window_words(uint32_t index) const {
    if (index >= kNoc2mWindowCount) {
        fail("TileTestAccess NOC window index " + std::to_string(index) + " is outside the hardware range");
    }
    tile_->runtime_->require_mesh_live("TileTestAccess::read_noc_window_words", &tile_->id());
    const std::lock_guard<std::mutex> serialise(tile_->chip_->lock());
    const uint64_t addr = kNocTlb2mConfigBase + noc_window_config_offset(index);
    NocWindowWords words;
    for (uint32_t i = 0; i < 4; ++i) {
        words.data[i] = tile_->transport().read_reg32(tile_->id().chip_id, tile_->core(), addr + i * sizeof(uint32_t));
    }
    return words;
}

void TileTestAccess::write_noc_window_words(uint32_t index, const NocWindowWords& words) {
    if (index >= kNoc2mWindowCount) {
        fail("TileTestAccess NOC window index " + std::to_string(index) + " is outside the hardware range");
    }
    tile_->runtime_->require_mesh_live("TileTestAccess::write_noc_window_words", &tile_->id());
    const std::lock_guard<std::mutex> serialise(tile_->chip_->lock());
    const uint64_t addr = kNocTlb2mConfigBase + noc_window_config_offset(index);
    for (uint32_t i = 0; i < 4; ++i) {
        tile_->transport().write_reg32_verified(
            tile_->id().chip_id, tile_->core(), addr + i * sizeof(uint32_t), words.data[i]);
    }
}

void TileTestAccess::load_idle_image(const std::vector<uint8_t>& image) {
    tile_->runtime_->require_mesh_live("TileTestAccess::load_idle_image", &tile_->id());
    const std::lock_guard<std::mutex> serialise(tile_->chip_->lock());
    tile_->load_idle_image_unlocked(image);
}

void TileTestAccess::load_active_image(const std::vector<uint8_t>& image) {
    tile_->runtime_->require_mesh_live("TileTestAccess::load_active_image", &tile_->id());
    const std::lock_guard<std::mutex> serialise(tile_->chip_->lock());
    tile_->load_active_image(image, tile_->read_lim_ecc_unlocked().primed_bytes);
}

uint32_t TileTestAccess::read_l2cpu_reset_raw() const {
    tile_->runtime_->require_mesh_live("TileTestAccess::read_l2cpu_reset_raw", &tile_->id());
    const std::lock_guard<std::mutex> serialise(tile_->chip_->lock());
    return tile_->chip_->read_l2cpu_reset_raw();
}

}  // namespace detail

X280Tile::X280Tile(detail::RuntimeImpl& runtime, detail::ChipImpl& chip, TileIdentity identity) :
    impl_(std::make_unique<detail::TileImpl>(runtime, chip, identity)) {}

X280Tile::~X280Tile() = default;

const TileIdentity& X280Tile::id() const { return impl_->id(); }

TileState X280Tile::inspect() const { return impl_->inspect(); }

BootResult X280Tile::boot(const BootRequest& request) { return impl_->boot(request); }

LimRange X280Tile::reserved_lim() const { return impl_->reserved_lim(); }

LimRange X280Tile::available_lim() const { return impl_->available_lim(); }

std::vector<uint8_t> X280Tile::read_available_lim(uint64_t address, size_t size) const {
    return impl_->read_available_lim(address, size);
}

void X280Tile::write_available_lim(uint64_t address, std::span<const uint8_t> data) {
    impl_->write_available_lim(address, data);
}

NocWindowResult X280Tile::read_noc_window(uint32_t index) const { return impl_->read_noc_window(index); }

bool X280Tile::healthy() const { return impl_->healthy(); }

std::optional<std::array<HartState, kHartsPerTile>> X280Tile::last_failure() const {
    return impl_->failure_hartstates();
}

PinnedInputState X280Tile::pinned_input_state() const { return impl_->pinned_input_state(); }

PinnedInputPrepareResult X280Tile::prepare_pinned_input(const PinnedInputPrepareRequest& request) {
    return impl_->prepare_pinned_input(request);
}

}  // namespace x280
