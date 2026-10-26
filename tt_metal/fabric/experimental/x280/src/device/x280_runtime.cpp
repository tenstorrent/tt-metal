// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "x280/device/x280_runtime.hpp"

#include "x280_format.hpp"
#include "x280_metal_transport.hpp"
#include "x280_runtime_impl.hpp"
#include "x280_tile_impl.hpp"

#include "x280_hart_protocol.h"

#include "impl/context/metal_context.hpp"

#include <llrt/tt_cluster.hpp>
#include <mesh_device.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/kernel_types.hpp>

#include <algorithm>
#include <array>
#include <exception>
#include <mutex>
#include <sstream>

namespace x280 {

namespace {

std::mutex g_runtime_mutex;
std::weak_ptr<detail::RuntimeImpl> g_runtime;

constexpr uint64_t kRequiredLimBytes = X280_HART_PROTOCOL_END - X280_LIM_BASE;

constexpr std::array<const char*, 4> kRuntimeOpenOutcomeNames = {
    "ready",
    "preparation_required",
    "reset_required",
    "failed",
};

enum class ChipAction {
    None,
    Prime,
    ApplyShape,
};

struct ChipPlan {
    std::shared_ptr<detail::ChipImpl> chip;
    ChipOpenResult result;
    ChipAction action = ChipAction::None;
};

std::vector<TileState> inspect_tiles(
    const std::shared_ptr<detail::RuntimeImpl>& runtime, const std::shared_ptr<detail::ChipImpl>& chip) {
    std::vector<TileState> states;
    states.reserve(chip->tile_identities().size());
    for (const auto& identity : chip->tile_identities()) {
        states.push_back(runtime->tile_impl(identity).inspect_state());
    }
    return states;
}

std::optional<std::string> invalid_observation(const std::vector<TileState>& states) {
    for (const auto& state : states) {
        if (!state.observation_valid) {
            return label_chip_l2cpu(state.id.chip_id, state.id.hw_index) + ": " + state.invalid_reason;
        }
    }
    return std::nullopt;
}

// Return the first released tile without live resident idle; such a tile
// requires an ASIC reset before another cold boot.
std::optional<std::string> reset_without_resident_idle(int chip_id, const std::vector<TileState>& states) {
    for (const auto& state : states) {
        if (state.reset_released && !state.resident_idle_live) {
            return label_chip_l2cpu(chip_id, state.id.hw_index) + " is out of reset without a live resident idle";
        }
    }
    return std::nullopt;
}

bool establish_safe_clock(
    const std::shared_ptr<detail::ChipImpl>& chip,
    ChipOpenResult& result,
    std::chrono::milliseconds timeout = std::chrono::milliseconds{4000}) {
    const auto established = chip->establish_pll(kSafePllMhz, timeout);
    if (!established.confirmed) {
        result.outcome = RuntimeOpenOutcome::Failed;
        result.diagnostic = "safe PLL bootstrap failed: " + established.diagnostic;
        return false;
    }
    return true;
}

ChipPlan preflight_chip(
    const std::shared_ptr<detail::RuntimeImpl>& runtime,
    const std::shared_ptr<detail::ChipImpl>& chip,
    const RuntimeOpenConfig& config,
    uint32_t target_wayenable) {
    ChipPlan plan;
    plan.chip = chip;
    plan.result.chip_id = chip->chip_id();

    runtime->require_mesh_live("open");
    const std::lock_guard<std::mutex> serialise(chip->lock());

    if (chip->tile_identities().empty()) {
        plan.result.outcome = RuntimeOpenOutcome::Failed;
        plan.result.diagnostic = "chip reports no enabled, unharvested L2CPU tiles";
        return plan;
    }

    auto states = inspect_tiles(runtime, chip);
    if (const auto invalid = invalid_observation(states); invalid.has_value()) {
        if (!chip->all_tiles_held_in_reset()) {
            plan.result.outcome = RuntimeOpenOutcome::Failed;
            plan.result.diagnostic =
                "an unreadable tile cannot be probed by moving PLL4 because another tile is out of reset";
            return plan;
        }
        if (!establish_safe_clock(chip, plan.result)) {
            return plan;
        }
        states = inspect_tiles(runtime, chip);
    }

    if (const auto invalid = invalid_observation(states); invalid.has_value()) {
        plan.result.outcome = RuntimeOpenOutcome::Failed;
        plan.result.diagnostic = *invalid;
        return plan;
    }

    if (const auto stale = reset_without_resident_idle(plan.result.chip_id, states); stale.has_value()) {
        plan.result.outcome = RuntimeOpenOutcome::ResetRequired;
        plan.result.diagnostic = *stale;
        return plan;
    }

    for (const auto& state : states) {
        if (!state.pll_mhz.has_value()) {
            plan.result.outcome = RuntimeOpenOutcome::Failed;
            plan.result.diagnostic = label_chip_l2cpu(state.id.chip_id, state.id.hw_index) +
                                     ": ARC telemetry has no L2CPU clock entry; frequency cannot be established safely";
            return plan;
        }
    }

    bool needs_shape = false;
    const TileState* reset_shape_state = nullptr;
    for (const auto& state : states) {
        const auto transition =
            classify_lim_shape_transition(state.raw_wayenable, target_wayenable, chip->all_tiles_held_in_reset());
        switch (transition) {
            case LimShapeTransition::AlreadyConfigured: break;
            case LimShapeTransition::ApplyWhileReset: needs_shape = true; break;
            case LimShapeTransition::ResetRequired:
                if (reset_shape_state == nullptr) {
                    reset_shape_state = &state;
                }
                break;
        }
    }

    const bool needs_prime = std::any_of(
        states.begin(), states.end(), [](const TileState& state) { return state.ecc == LimEccState::Unprimed; });
    if (needs_prime) {
        if (!chip->all_tiles_held_in_reset()) {
            plan.result.outcome = RuntimeOpenOutcome::ResetRequired;
            plan.result.diagnostic = "LIM ECC is unprimed while an L2CPU is out of reset; priming would remove its LIM";
            return plan;
        }
        if (config.priming_policy == LimEccPrimingPolicy::RequirePrepared) {
            plan.result.outcome = RuntimeOpenOutcome::PreparationRequired;
            plan.result.diagnostic = "LIM ECC is unprimed; the open policy requires prepared hardware";
            return plan;
        }
        plan.result.outcome = RuntimeOpenOutcome::ResetRequired;
        plan.result.diagnostic = "LIM ECC priming is required before the configured shape can be applied";
        plan.action = ChipAction::Prime;
        return plan;
    }

    if (reset_shape_state != nullptr) {
        plan.result.outcome = RuntimeOpenOutcome::ResetRequired;
        plan.result.diagnostic = label_chip_l2cpu(reset_shape_state->id.chip_id, reset_shape_state->id.hw_index) +
                                 " has WayEnable " + std::to_string(reset_shape_state->raw_wayenable & 0xFu) +
                                 ", which cannot reach configured target " + std::to_string(target_wayenable) +
                                 " without an ASIC reset";
        return plan;
    }

    plan.result.outcome = RuntimeOpenOutcome::Ready;
    plan.result.diagnostic =
        needs_shape ? "ready to apply configured LIM shape" : "every enabled L2CPU is freshly inspected and ready";
    plan.action = needs_shape ? ChipAction::ApplyShape : ChipAction::None;
    return plan;
}

void prime_chip(const std::shared_ptr<detail::RuntimeImpl>& runtime, ChipPlan& plan) {
    const std::lock_guard<std::mutex> serialise(plan.chip->lock());
    if (!plan.chip->all_tiles_held_in_reset()) {
        plan.result.outcome = RuntimeOpenOutcome::ResetRequired;
        plan.result.diagnostic = "an L2CPU left reset after preflight; refusing to prime LIM ECC";
        return;
    }
    if (!establish_safe_clock(plan.chip, plan.result)) {
        return;
    }

    bool commit_attempted = false;
    std::vector<int> primed;
    try {
        for (const auto& identity : plan.chip->tile_identities()) {
            runtime->tile_impl(identity).prime_lim_ecc_unlocked(commit_attempted);
            primed.push_back(identity.hw_index);
        }
    } catch (const std::exception& error) {
        plan.result.outcome = commit_attempted ? RuntimeOpenOutcome::ResetRequired : RuntimeOpenOutcome::Failed;
        plan.result.diagnostic = std::string("LIM ECC bootstrap failed") +
                                 (commit_attempted ? " after a one-way WayEnable commit: " : " before any commit: ") +
                                 error.what();
        return;
    } catch (...) {
        plan.result.outcome = commit_attempted ? RuntimeOpenOutcome::ResetRequired : RuntimeOpenOutcome::Failed;
        plan.result.diagnostic = commit_attempted ? "unknown LIM ECC bootstrap failure after a one-way WayEnable commit"
                                                  : "unknown LIM ECC bootstrap failure before any commit";
        return;
    }

    std::ostringstream diagnostic;
    diagnostic << "LIM ECC primed on chip " << plan.result.chip_id << " L2CPUs";
    for (int hw_index : primed) {
        diagnostic << ' ' << hw_index;
    }
    diagnostic << ". WayEnable is committed, so close Metal and ASIC-reset the device before reopening X280.";
    plan.result.outcome = RuntimeOpenOutcome::ResetRequired;
    plan.result.diagnostic = diagnostic.str();
}

void apply_shape(const std::shared_ptr<detail::RuntimeImpl>& runtime, ChipPlan& plan, uint32_t target_wayenable) {
    const std::lock_guard<std::mutex> serialise(plan.chip->lock());
    if (!plan.chip->all_tiles_held_in_reset()) {
        plan.result.outcome = RuntimeOpenOutcome::ResetRequired;
        plan.result.diagnostic = "an L2CPU left reset after preflight; refusing to shrink live LIM";
        return;
    }

    bool commit_attempted = false;
    try {
        for (const auto& identity : plan.chip->tile_identities()) {
            runtime->tile_impl(identity).set_lim_shape_unlocked(target_wayenable, commit_attempted);
        }
        const auto states = inspect_tiles(runtime, plan.chip);
        if (const auto invalid = invalid_observation(states); invalid.has_value()) {
            fail(*invalid);
        }
        for (const auto& state : states) {
            if ((state.raw_wayenable & 0xFu) != target_wayenable || state.ecc != LimEccState::Primed) {
                fail(
                    label_chip_l2cpu(state.id.chip_id, state.id.hw_index) +
                    " did not retain primed LIM at configured WayEnable " + std::to_string(target_wayenable));
            }
        }
    } catch (const std::exception& error) {
        plan.result.outcome = commit_attempted ? RuntimeOpenOutcome::ResetRequired : RuntimeOpenOutcome::Failed;
        plan.result.diagnostic = std::string("LIM shape application failed") +
                                 (commit_attempted ? " after a one-way WayEnable increase: " : " before any commit: ") +
                                 error.what();
        return;
    } catch (...) {
        plan.result.outcome = commit_attempted ? RuntimeOpenOutcome::ResetRequired : RuntimeOpenOutcome::Failed;
        plan.result.diagnostic = commit_attempted ? "unknown LIM shape failure after a one-way WayEnable increase"
                                                  : "unknown LIM shape failure before any commit";
        return;
    }

    plan.result.outcome = RuntimeOpenOutcome::Ready;
    plan.result.diagnostic = "configured WayEnable " + std::to_string(target_wayenable) + " on every enabled L2CPU";
}

detail::RuntimeImpl& require_impl(const std::shared_ptr<detail::RuntimeImpl>& impl, const char* operation) {
    if (!impl) {
        fail(std::string("X280Runtime::") + operation + " on a moved-from runtime");
    }
    return *impl;
}

}  // namespace

namespace detail {

bool RuntimeImpl::mesh_live() const {
    const auto mesh = mesh_.lock();
    return mesh && mesh->is_initialized();
}

std::shared_ptr<tt::tt_metal::distributed::MeshDevice> RuntimeImpl::mesh() const {
    auto mesh = mesh_.lock();
    if (!mesh || !mesh->is_initialized()) {
        fail("coordinate query: the caller-owned MeshDevice has been closed or dropped");
    }
    return mesh;
}

bool RuntimeImpl::owns_mesh(const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh) const {
    const auto owned = mesh_.lock();
    return owned && owned->is_initialized() && owned.get() == mesh.get();
}

void RuntimeImpl::require_mesh_live(const char* operation, const TileIdentity* tile) const {
    if (mesh_live()) {
        return;
    }
    std::ostringstream diagnostic;
    diagnostic << operation;
    if (tile != nullptr) {
        diagnostic << " on " << label_chip_l2cpu(tile->chip_id, tile->hw_index);
    }
    diagnostic << ": the caller-owned MeshDevice has been closed or dropped";
    fail(diagnostic.str());
}

RuntimeImpl::RuntimeImpl(
    std::weak_ptr<tt::tt_metal::distributed::MeshDevice> mesh,
    std::vector<std::shared_ptr<ChipImpl>> chips,
    LimShape lim_shape,
    LimLayout lim_layout) :
    mesh_(std::move(mesh)), chips_(std::move(chips)), lim_shape_(lim_shape), lim_layout_(lim_layout) {
    for (const auto& chip : chips_) {
        for (const auto& identity : chip->tile_identities()) {
            tiles_.push_back(std::unique_ptr<X280Tile>(new X280Tile(*this, *chip, identity)));
        }
    }
}

std::vector<TileId> RuntimeImpl::tile_ids() const {
    require_mesh_live("tile_ids");
    std::vector<TileId> ids;
    ids.reserve(tiles_.size());
    for (const auto& tile : tiles_) {
        ids.push_back(tile->id());
    }
    std::sort(ids.begin(), ids.end());
    return ids;
}

std::shared_ptr<ChipImpl> RuntimeImpl::chip(int chip_id) const {
    for (const auto& chip : chips_) {
        if (chip->chip_id() == chip_id) {
            return chip;
        }
    }
    fail("chip " + std::to_string(chip_id) + " is not in this runtime's mesh view");
}

X280Tile& RuntimeImpl::tile(const TileId& id) {
    for (const auto& tile : tiles_) {
        if (tile->id() == id) {
            return *tile;
        }
    }
    fail("tile identity is not an exact member of this runtime's mesh view");
}

TileImpl& RuntimeImpl::tile_impl(const TileId& id) { return *tile(id).impl_; }

}  // namespace detail

const char* to_string(RuntimeOpenOutcome outcome) { return enum_name(outcome, kRuntimeOpenOutcomeNames); }

bool preparation_required(RuntimeOpenOutcome outcome) { return outcome == RuntimeOpenOutcome::PreparationRequired; }

bool reset_required(RuntimeOpenOutcome outcome) { return outcome == RuntimeOpenOutcome::ResetRequired; }

bool RuntimeOpenResult::requires_preparation() const {
    return preparation_required(outcome) || std::any_of(chips.begin(), chips.end(), [](const ChipOpenResult& chip) {
               return preparation_required(chip.outcome);
           });
}

bool RuntimeOpenResult::requires_reset() const {
    return reset_required(outcome) || std::any_of(chips.begin(), chips.end(), [](const ChipOpenResult& chip) {
               return reset_required(chip.outcome);
           });
}

RuntimeOpenResult X280Runtime::open(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh, RuntimeOpenConfig config) {
    return open_impl(mesh, config);
}

RuntimeOpenResult X280Runtime::open_impl(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh, RuntimeOpenConfig config) {
    RuntimeOpenResult result;
    if (!mesh) {
        result.diagnostic = "x280: X280Runtime::open was given a null MeshDevice";
        return result;
    }
    const auto shape = validate_lim_shape(config.lim_shape, kRequiredLimBytes);
    if (!shape.ok) {
        result.diagnostic = "x280: invalid runtime LIM shape: " + shape.reason;
        return result;
    }
    if (config.lim_layout == LimLayout::PacketInjection) {
        if (config.lim_shape.lim_ways != kPreparedLimWays ||
            config.priming_policy != LimEccPrimingPolicy::RequirePrepared) {
            result.diagnostic = "x280: PacketInjection layout requires lim_ways=" + std::to_string(kPreparedLimWays) +
                                " and RequirePrepared priming policy";
            return result;
        }
    }

    const std::lock_guard<std::mutex> owner_lock(g_runtime_mutex);
    if (const auto current = g_runtime.lock(); current && current->mesh_live()) {
        if (!current->owns_mesh(mesh)) {
            result.diagnostic =
                "x280: a different live MeshDevice already owns the X280 runtime. Close it before opening another "
                "mesh or submesh.";
            return result;
        }
        if (current->lim_shape() != config.lim_shape) {
            result.diagnostic =
                "x280: the live MeshDevice runtime already owns a different LIM shape; close it and ASIC-reset "
                "before requesting another shape";
            return result;
        }
        if (current->lim_layout() != config.lim_layout) {
            result.diagnostic =
                "x280: the live MeshDevice runtime already owns a different LIM layout; close it before requesting "
                "another layout";
            return result;
        }

        // Recheck liveness when reusing a runtime; tile state may have changed
        // since it was opened.
        for (const auto& chip : current->chips()) {
            const std::lock_guard<std::mutex> serialise(chip->lock());
            const auto states = inspect_tiles(current, chip);
            if (const auto invalid = invalid_observation(states); invalid.has_value()) {
                result.outcome = RuntimeOpenOutcome::Failed;
                result.diagnostic = *invalid;
                return result;
            }
            if (const auto stale = reset_without_resident_idle(chip->chip_id(), states); stale.has_value()) {
                result.outcome = RuntimeOpenOutcome::ResetRequired;
                result.diagnostic = *stale;
                return result;
            }
        }

        for (const auto& chip : current->chips()) {
            ChipOpenResult chip_result;
            chip_result.chip_id = chip->chip_id();
            chip_result.outcome = RuntimeOpenOutcome::Ready;
            chip_result.diagnostic = "reused the runtime already initialized for this live MeshDevice";
            result.chips.push_back(std::move(chip_result));
        }
        result.outcome = RuntimeOpenOutcome::Ready;
        result.runtime.emplace(X280Runtime(current));
        result.diagnostic = "reused the initialized X280 runtime for this live MeshDevice";
        return result;
    }

    try {
        auto transport = X280MetalTransport::bind(*mesh);
        const auto chip_ids = sorted_usable_chip_ids(*mesh, *transport);

        std::vector<std::shared_ptr<detail::ChipImpl>> chips;
        chips.reserve(chip_ids.size());
        result.chips.resize(chip_ids.size());
        for (size_t i = 0; i < chip_ids.size(); i++) {
            const int chip_id = chip_ids[i];
            result.chips[i].chip_id = chip_id;
            try {
                chips.push_back(std::make_shared<detail::ChipImpl>(chip_id, transport));
            } catch (const std::exception& error) {
                result.chips[i].diagnostic = error.what();
            } catch (...) {
                result.chips[i].diagnostic = "unknown chip-context construction failure";
            }
        }

        // Runtime construction creates the stable tile inventory after every
        // constructible chip context exists.
        auto candidate = std::make_shared<detail::RuntimeImpl>(
            std::weak_ptr<tt::tt_metal::distributed::MeshDevice>{mesh}, chips, config.lim_shape, config.lim_layout);

        std::vector<ChipPlan> plans;
        plans.reserve(chips.size());
        for (const auto& chip : chips) {
            const auto entry = std::lower_bound(
                result.chips.begin(), result.chips.end(), chip->chip_id(), [](const ChipOpenResult& item, int chip_id) {
                    return item.chip_id < chip_id;
                });
            if (entry == result.chips.end() || entry->chip_id != chip->chip_id()) {
                fail("internal error: chip context has no runtime-open result slot");
            }
            try {
                plans.push_back(preflight_chip(candidate, chip, config, shape.target_wayenable));
                *entry = plans.back().result;
            } catch (const std::exception& error) {
                entry->outcome = RuntimeOpenOutcome::Failed;
                entry->diagnostic = error.what();
            } catch (...) {
                entry->outcome = RuntimeOpenOutcome::Failed;
                entry->diagnostic = "unknown chip initialization failure";
            }
        }

        const bool actions_allowed =
            plans.size() == chip_ids.size() && std::all_of(plans.begin(), plans.end(), [](const ChipPlan& plan) {
                return plan.result.outcome == RuntimeOpenOutcome::Ready ||
                       (plan.result.outcome == RuntimeOpenOutcome::ResetRequired && plan.action == ChipAction::Prime);
            });
        const bool any_prime = std::any_of(
            plans.begin(), plans.end(), [](const ChipPlan& plan) { return plan.action == ChipAction::Prime; });

        if (actions_allowed && any_prime) {
            for (auto& plan : plans) {
                if (plan.action == ChipAction::Prime) {
                    prime_chip(candidate, plan);
                    if (plan.result.outcome == RuntimeOpenOutcome::Failed) {
                        break;
                    }
                }
            }
        } else if (actions_allowed) {
            for (auto& plan : plans) {
                if (plan.action == ChipAction::ApplyShape) {
                    apply_shape(candidate, plan, shape.target_wayenable);
                    if (plan.result.outcome != RuntimeOpenOutcome::Ready) {
                        break;
                    }
                }
            }
        }

        for (const auto& plan : plans) {
            const auto entry = std::lower_bound(
                result.chips.begin(),
                result.chips.end(),
                plan.result.chip_id,
                [](const ChipOpenResult& item, int chip_id) { return item.chip_id < chip_id; });
            if (entry == result.chips.end() || entry->chip_id != plan.result.chip_id) {
                fail("internal error: chip plan has no runtime-open result slot");
            }
            *entry = plan.result;
        }

        const bool all_ready = result.chips.size() == chip_ids.size() &&
                               std::all_of(result.chips.begin(), result.chips.end(), [](const ChipOpenResult& chip) {
                                   return chip.outcome == RuntimeOpenOutcome::Ready;
                               });
        const auto has_outcome = [&](RuntimeOpenOutcome outcome) {
            return std::any_of(result.chips.begin(), result.chips.end(), [&](const ChipOpenResult& chip) {
                return chip.outcome == outcome;
            });
        };
        if (all_ready) {
            g_runtime = candidate;
            result.outcome = RuntimeOpenOutcome::Ready;
            result.runtime.emplace(X280Runtime(std::move(candidate)));
            result.diagnostic = "all selected X280 chips are ready";
        } else if (has_outcome(RuntimeOpenOutcome::Failed)) {
            result.outcome = RuntimeOpenOutcome::Failed;
            result.diagnostic = "one or more X280 chips failed initialization; no runtime is usable";
        } else if (has_outcome(RuntimeOpenOutcome::PreparationRequired)) {
            result.outcome = RuntimeOpenOutcome::PreparationRequired;
            result.diagnostic = "one or more X280 chips need operator preparation; no runtime is usable";
        } else if (has_outcome(RuntimeOpenOutcome::ResetRequired)) {
            result.outcome = RuntimeOpenOutcome::ResetRequired;
            result.diagnostic = "X280 initialization made or observed reset-required state; no runtime is usable";
        } else {
            result.outcome = RuntimeOpenOutcome::Failed;
            result.diagnostic = "X280 initialization produced no usable chip state";
        }
    } catch (const std::exception& error) {
        result.diagnostic = error.what();
    } catch (...) {
        result.diagnostic = "x280: unknown runtime initialization failure";
    }
    return result;
}

X280Runtime::X280Runtime(std::shared_ptr<detail::RuntimeImpl> impl) : impl_(std::move(impl)) {}
X280Runtime::X280Runtime(X280Runtime&&) noexcept = default;
X280Runtime& X280Runtime::operator=(X280Runtime&&) noexcept = default;
X280Runtime::~X280Runtime() = default;

std::vector<TileId> X280Runtime::tile_ids() const { return require_impl(impl_, "tile_ids").tile_ids(); }

namespace {

// The virtual coordinate common to `devices`. Metal has no L2CPU logical space,
// so L2CPU resolves through each chip's SoC descriptor on the default context,
// which is the one silicon has; every other type through Metal's per-device
// resolver, as MeshDevice's own resolver does.
tt::tt_metal::CoreCoord resolve_common(
    const std::vector<tt::tt_metal::IDevice*>& devices,
    const tt::tt_metal::CoreCoord& logical_coord,
    const tt::CoreType& core_type) {
    if (devices.empty()) {
        fail("cannot resolve coordinates: the MeshDevice scope holds no devices");
    }
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    std::optional<tt::tt_metal::CoreCoord> common;
    for (const auto* device : devices) {
        tt::tt_metal::CoreCoord current;
        if (core_type == tt::CoreType::L2CPU) {
            const auto translated =
                cluster.get_soc_desc(device->id())
                    .translate_coord_to(
                        tt::umd::CoreCoord(logical_coord, tt::CoreType::L2CPU, tt::CoordSystem::LOGICAL),
                        tt::CoordSystem::TRANSLATED);
            current = {translated.x, translated.y};
        } else {
            current = device->virtual_core_from_logical_core(logical_coord, core_type);
        }
        if (common && *common != current) {
            fail(
                std::string(tt::to_str(core_type)) + " logical coordinate (" + std::to_string(logical_coord.x) + ", " +
                std::to_string(logical_coord.y) + ") resolves differently on chip " +
                std::to_string(devices.front()->id()) + " and chip " + std::to_string(device->id()));
        }
        common = current;
    }
    return *common;
}

// Refuses a resolved coordinate no device consumer on `noc` can address.
void require_device_bound(const tt::tt_metal::CoreCoord& c, tt::tt_metal::NOC noc, const char* what) {
    const std::string where =
        std::string(what) + " at translated (" + std::to_string(c.x) + ", " + std::to_string(c.y) + ")";
    if (c.x > X280_NOC_COORD_MAX || c.y > X280_NOC_COORD_MAX) {
        fail(where + " exceeds the six-bit NoC coordinate range");
    }
    const x280_noc_coord_t coord{static_cast<uint32_t>(c.x), static_cast<uint32_t>(c.y)};
    if (!x280_noc_translated_addressable(coord, static_cast<uint32_t>(noc))) {
        fail(
            where + " names a different tile on NoC " + std::to_string(static_cast<uint32_t>(noc)) +
            ", where firmware leaves X untranslated for Y < 2 and X != 8");
    }
}

}  // namespace

TranslatedCoord X280Runtime::virtual_core_from_logical_core(
    const tt::tt_metal::distributed::MeshDevice& scope,
    const tt::tt_metal::CoreCoord& logical_coord,
    const tt::CoreType& core_type,
    tt::tt_metal::NOC noc) {
    // is_initialized() is the only MeshDevice query that is safe after close.
    if (!scope.is_initialized()) {
        fail("cannot resolve coordinates: the MeshDevice scope is closed");
    }
    const auto c = resolve_common(scope.get_devices(), logical_coord, core_type);
    require_device_bound(c, noc, "logical coordinate");
    return TranslatedCoord(static_cast<uint32_t>(c.x), static_cast<uint32_t>(c.y));
}

TranslatedCoord X280Runtime::virtual_core_from_logical_core(
    const tt::tt_metal::distributed::MeshDevice& scope,
    const tt::tt_metal::distributed::MeshCoordinate& chip,
    const tt::tt_metal::CoreCoord& logical_coord,
    const tt::CoreType& core_type,
    tt::tt_metal::NOC noc) {
    if (!scope.is_initialized()) {
        fail("cannot resolve coordinates: the MeshDevice scope is closed");
    }
    auto* device = scope.get_device(chip);
    if (device == nullptr) {
        fail("cannot resolve coordinates: the scope has no local chip at the requested mesh coordinate");
    }
    const auto c = resolve_common({device}, logical_coord, core_type);
    require_device_bound(c, noc, "logical coordinate");
    return TranslatedCoord(static_cast<uint32_t>(c.x), static_cast<uint32_t>(c.y));
}

TranslatedCoord X280Runtime::virtual_core_from_tile(const TileId& id, tt::tt_metal::NOC noc) const {
    auto& runtime = require_impl(impl_, "virtual_core_from_tile");
    const auto mesh = runtime.mesh();
    const auto& identity = runtime.tile(id).id();
    const auto c = runtime.chip(identity.chip_id)
                       ->transport()
                       .virtual_core_from_noc0(identity.chip_id, {identity.noc_x, identity.noc_y});
    require_device_bound(c, noc, "L2CPU tile");
    return TranslatedCoord(static_cast<uint32_t>(c.x), static_cast<uint32_t>(c.y));
}

LimShape X280Runtime::lim_shape() const { return require_impl(impl_, "lim_shape").lim_shape(); }

X280Tile& X280Runtime::tile(const TileId& id) {
    auto& runtime = require_impl(impl_, "tile");
    runtime.require_mesh_live("tile");
    return runtime.tile(id);
}

void X280Runtime::set_pll(int chip_id, uint32_t mhz, std::chrono::milliseconds confirm_timeout) {
    auto& runtime = require_impl(impl_, "set_pll");
    runtime.require_mesh_live("set_pll");
    auto chip = runtime.chip(chip_id);
    const std::lock_guard<std::mutex> serialise(chip->lock());
    const auto established = chip->establish_pll(mhz, confirm_timeout);
    if (!established.confirmed) {
        fail("could not establish PLL4 on chip " + std::to_string(chip_id) + ": " + established.diagnostic);
    }
}

MultiTileBootResult X280Runtime::boot_tiles(const MultiTileBootRequest& request) {
    MultiTileBootResult result;

    if (request.tiles.empty()) {
        result.diagnostic = "no tiles selected";
        return result;
    }
    if (request.boot_timeout.count() <= 0 || request.clock_confirm_timeout.count() <= 0) {
        result.diagnostic =
            "boot_timeout and clock_confirm_timeout bound waits after a one-way reset release, so "
            "both must be positive";
        return result;
    }
    if (!request.allow_reset_release) {
        result.diagnostic = "a multi-tile cold boot releases reset and requires allow_reset_release";
        return result;
    }

    auto& runtime = require_impl(impl_, "boot_tiles");
    runtime.require_mesh_live("boot_tiles");

    const auto enabled = runtime.tile_ids();
    for (size_t i = 0; i < request.tiles.size(); ++i) {
        const auto& id = request.tiles[i];
        if (std::find(enabled.begin(), enabled.end(), id) == enabled.end()) {
            result.diagnostic = label_chip_l2cpu(id.chip_id, id.hw_index) + " is not an enabled tile of this runtime";
            return result;
        }
        if (std::find(request.tiles.begin(), request.tiles.begin() + static_cast<ptrdiff_t>(i), id) !=
            request.tiles.begin() + static_cast<ptrdiff_t>(i)) {
            result.diagnostic = label_chip_l2cpu(id.chip_id, id.hw_index) + " is selected more than once";
            return result;
        }
        if (id.chip_id != request.tiles.front().chip_id) {
            result.diagnostic = "a batch spans one chip; reset is a per-chip register";
            return result;
        }
    }

    const int chip_id = request.tiles.front().chip_id;
    auto chip = runtime.chip(chip_id);
    const std::lock_guard<std::mutex> serialise(chip->lock());

    // Recheck under the lock before moving PLL4 or releasing any reset.
    if (!chip->all_tiles_held_in_reset()) {
        result.diagnostic = "an L2CPU on chip " + std::to_string(chip_id) +
                            " is out of reset; a multi-tile cold boot needs every enabled tile held in reset and "
                            "admits one batch per chip-reset lifetime";
        return result;
    }

    result.tiles.reserve(request.tiles.size());
    std::vector<int> to_release;
    for (const auto& id : request.tiles) {
        TileBootReport report;
        report.id = id;
        if (runtime.tile_impl(id).classify_cold_boot_unlocked(report.result)) {
            to_release.push_back(id.hw_index);
        }
        result.tiles.push_back(std::move(report));
    }

    for (const auto& report : result.tiles) {
        const bool ready_to_prepare =
            std::find(to_release.begin(), to_release.end(), report.id.hw_index) != to_release.end();
        if (!ready_to_prepare && report.result.outcome != BootOutcome::AlreadyRunning) {
            result.diagnostic = "refusing to release any reset: " + report.result.diagnostic;
            return result;
        }
    }

    if (to_release.empty()) {
        result.diagnostic = "every selected tile already has a live resident idle; nothing was released";
        return result;
    }

    try {
        const auto established = chip->establish_pll(kSafePllMhz, request.clock_confirm_timeout);
        if (!established.confirmed) {
            result.diagnostic = "could not establish the safe reset-release clock: " + established.diagnostic;
            return result;
        }

        // Prepare every selected tile before releasing any reset.
        for (auto& report : result.tiles) {
            if (std::find(to_release.begin(), to_release.end(), report.id.hw_index) == to_release.end()) {
                continue;
            }
            runtime.tile_impl(report.id).prepare_cold_boot_unlocked(request.idle_image_path);
        }
    } catch (const std::exception& error) {
        result.diagnostic = std::string("preparation failed before any reset was released: ") + error.what();
        return result;
    } catch (...) {
        result.diagnostic = "unknown preparation failure before any reset was released";
        return result;
    }

    try {
        chip->release_tile_resets(to_release);
        result.reset_released = true;
    } catch (const std::exception& error) {
        // A throwing write may still have landed, so the chip is treated as
        // released either way.
        result.reset_released = true;
        result.diagnostic = std::string("the reset-release write failed; an ASIC reset is required: ") + error.what();
        return result;
    }

    for (auto& report : result.tiles) {
        if (std::find(to_release.begin(), to_release.end(), report.id.hw_index) == to_release.end()) {
            continue;
        }
        try {
            (void)runtime.tile_impl(report.id).confirm_resident_idle_unlocked(request.boot_timeout, report.result);
        } catch (const std::exception& error) {
            report.result.outcome = BootOutcome::ResetRequiredNoResidentIdle;
            report.result.diagnostic =
                std::string("failure after reset release; an ASIC reset is required: ") + error.what();
        } catch (...) {
            report.result.outcome = BootOutcome::ResetRequiredNoResidentIdle;
            report.result.diagnostic = "unknown failure after reset release; an ASIC reset is required";
        }
    }

    std::ostringstream diagnostic;
    diagnostic << "released L2CPUs";
    for (int hw_index : to_release) {
        diagnostic << ' ' << hw_index;
    }
    diagnostic << " on chip " << chip_id << " in one write at the safe clock; "
               << (result.ready() ? "every selected tile reached resident idle"
                                  : "at least one selected tile did not reach resident idle, and no further cold "
                                    "boot is possible in this chip-reset lifetime")
               << ". The owner raises the operating clock with set_pll.";
    result.diagnostic = diagnostic.str();
    return result;
}

EnqueueResult X280Runtime::enqueue(X280Tile& tile, const ActiveRunRequest& request) {
    return require_impl(impl_, "enqueue").enqueue(tile, request);
}

std::vector<SessionId> X280Runtime::sessions() const { return require_impl(impl_, "sessions").sessions(); }

X280ActiveSession& X280Runtime::session(SessionId id) { return require_impl(impl_, "session").session(id); }

bool X280Runtime::session_live(SessionId id) const { return require_impl(impl_, "session_live").session_live(id); }

CleanupResult X280Runtime::cleanup(SessionId id, std::chrono::milliseconds timeout) {
    return require_impl(impl_, "cleanup").cleanup(id, timeout);
}

}  // namespace x280
