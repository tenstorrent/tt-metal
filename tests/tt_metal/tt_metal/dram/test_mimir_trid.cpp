// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MM43: Mimir test plan section 4.3 (transaction IDs). Plain hart loads and stores carry no software-visible ID; the
// only ID point is the per-hart RoCC command buffer driving iDMA, with a 4-bit TR_ID. Every case runs cce_idma_copy
// with an explicit MIM-407 backend VC, sets the trid per command and waits on each trid's pending count, then on the
// data. DuplicateId and BarrierThenDependentWriteAfterRead's read instead check the data once when their trids read 0
// (kArgCheckAtAck), so an early ack fails. Remote targets are host-verified.

#include <gtest/gtest.h>

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <tt-logger/tt-logger.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <exception>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "distributed/mesh_device_impl.hpp"
#include "impl/buffers/dram_receiver_config_page.hpp"
#include "impl/buffers/drisc_l1_arena.hpp"
#include "impl/buffers/global_circular_buffer_dram_sender_internal.hpp"
#include "impl/buffers/tensor_prefetcher_manager.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/program/slow_dispatch.hpp"
#include "llrt/hal.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/distributed_host_buffer.hpp>
#include <tt-metalium/experimental/distributed_tensor/distributed_tensor_apis.hpp>
#include <tt-metalium/experimental/distributed_tensor/topology/tensor_topology.hpp>
#include <tt-metalium/experimental/global_circular_buffer.hpp>
#include <tt-metalium/experimental/tensor_prefetcher.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tensor/host_tensor.hpp>
#include <tt-metalium/tensor/mesh_tensor.hpp>
#include <tt-metalium/tensor/spec/layout/tensor_layout.hpp>
#include <tt-metalium/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "tt_metal/tt_metal/dram/mimir_access.hpp"
#include "tt_metal/tt_metal/dram/mimir_dm_common.hpp"
#include "tt_metal/tt_metal/dram/mimir_gddr_map.hpp"
#include "tt_metal/tt_metal/dram/mimir_hw_state.hpp"
#include "tt_metal/tt_metal/dram/mimir_patterns.hpp"
#include "tt_metal/tt_metal/dram/mimir_prefetcher_common.hpp"
#include "tt_metal/tt_metal/dram/mimir_test_common.hpp"
#include "tt_metal/tt_metal/test_kernels/cce/cce_gcb_receiver_abi.h"
#include "tt_metal/tt_metal/test_kernels/cce/cce_idma_copy_abi.h"

namespace tt::tt_metal::mimir_test::trid {

namespace abi = cce_idma_copy_abi;

inline constexpr const char* kIdmaCopyKernel = "tests/tt_metal/tt_metal/test_kernels/cce/cce_idma_copy.cpp";
// The MM4x modes share one poll budget across every wait of a launch, so one budget plus launch.
inline constexpr std::chrono::seconds kHostDeadline{300};
// Offsets of a CCE's GDDR target in its local DRAM view, above DRAM UNRESERVED, clear of MM-8's 0x10000.
inline constexpr uint64_t kGddrTargetOffset = 0x100000;
// Observer handshake lines, in the hart's data area after its source block.
inline constexpr uint64_t kPeerFlagOffsetFromData = 0x8000;

// Probe-gated values, set from the probes' results on mimir_2x_package (2026-10-22, every CCE).
// MimirCacheProbe.EightHartLaunch: true once all 8 harts of one CCE posted their status (2x, every CCE, 2026-10-22).
inline constexpr bool kEightHartLaunchPasses = true;

// Hart data area (tl1_data) layout of the MM43 cases, clear of the observer flag lines.
inline constexpr uint64_t kOwnDestinationOffsetFromData = 0x10000;
inline constexpr uint64_t kTableOffsetFromData = 0x22000;
inline constexpr uint64_t kTraceOffsetFromData = 0x23000;
inline constexpr uint64_t kStopOffsetFromData = 0x23100;
inline constexpr uint64_t kProgressOffsetFromData = 0x23140;
inline constexpr uint64_t kOwnSourceOffsetFromData = 0x24000;
inline constexpr uint32_t kOwnRegionCapacity = 0x10000 - 2 * kGuardBytes;
static_assert(
    kOwnSourceOffsetFromData + kOwnRegionCapacity + 2 * kGuardBytes <= kTl1HartDataBytes,
    "the own source region and its guards must fit the hart data area");
// Sibling and remote TL1 regions sit in the issuing CCE's and hart's remote landing slot of the target CCE.
inline constexpr uint32_t kLandingRegionCapacity = static_cast<uint32_t>(kTl1RemoteLandingSlotBytes) - 2 * kGuardBytes;
// GDDR regions above DRAM UNRESERVED, one per issuing CCE and hart.
inline constexpr uint64_t kGddrRegionBase = 0x400000;
inline constexpr uint64_t kGddrRegionCceStride = 0x100000;
inline constexpr uint64_t kGddrRegionHartStride = 0x20000;
inline constexpr uint32_t kGddrRegionCapacity = static_cast<uint32_t>(kGddrRegionHartStride) - 2 * kGuardBytes;
inline constexpr uint32_t kRegionLeadingGuardTag = 0x800;
// The hart of every single-hart case.
inline constexpr uint32_t kTridHart = kCceKernelHart;

std::string code_name(uint32_t code) {
    switch (code) {
        case abi::kCodeLanded: return "landed";
        case abi::kCodeAcked: return "acked";
        case abi::kCodeVcSpaceTimeout: return "vc_space_timeout";
        case abi::kCodeAckTimeout: return "ack_timeout";
        case abi::kCodeDataTimeout: return "data_timeout";
        case abi::kCodeBadArgs: return "bad_args";
        case abi::kCodePeerTimeout: return "peer_timeout";
        case abi::kCodeStreamBudget: return "stream_budget";
        case abi::kCodeNotLandedAtAck: return "not_landed_at_ack";
    }
    return fmt::format("unknown_code_0x{:x}", code);
}

std::string poll_name(uint32_t poll) {
    switch (poll) {
        case abi::kPollNone: return "none";
        case abi::kPollVcSpace: return "vc_space";
        case abi::kPollAck: return "ack";
        case abi::kPollData: return "data";
        case abi::kPollPeer: return "peer";
    }
    return fmt::format("unknown_poll_{}", poll);
}

std::string status_detail(const std::vector<uint32_t>& status) {
    const uint32_t flags = status[abi::kStatusFlags];
    return fmt::format(
        "kernel code {} flags [vc_space {} acked {} landed {}], {} commands issued, last poll {} {}, cycles vc_space "
        "{} ack {} data {}, pending high water {}, probe {} / {}, kernel mismatches {} (first word {} observed "
        "0x{:08x} expected 0x{:08x})",
        code_name(status[abi::kStatusCode]),
        (flags & abi::kFlagVcSpace) != 0,
        (flags & abi::kFlagAcked) != 0,
        (flags & abi::kFlagLanded) != 0,
        status[abi::kStatusElementsIssued],
        poll_name(status[abi::kStatusLastPoll]),
        status[abi::kStatusLastPollReady] != 0 ? "done" : "timed out",
        status[abi::kStatusVcSpaceCycles],
        status[abi::kStatusAckCycles],
        status[abi::kStatusDataCycles],
        status[abi::kStatusPendingHighWater],
        status[abi::kStatusProbeFirst],
        status[abi::kStatusProbeSecond],
        status[abi::kStatusMismatchedWords],
        status[abi::kStatusFirstMismatchWord],
        status[abi::kStatusObserved],
        status[abi::kStatusExpected]);
}

// A timeout code may leave a command in flight, so the next run needs a fresh session (MIM-601).
std::string code_note(uint32_t code) {
    return code == abi::kCodeVcSpaceTimeout || code == abi::kCodeAckTimeout || code == abi::kCodeDataTimeout
               ? kFreshSessionNote
               : "";
}

std::string cce_skip_reason(uint32_t cce_index) {
    if (const std::string watcher_reason = mimir_watcher_skip_reason(); !watcher_reason.empty()) {
        return watcher_reason;
    }
    const MimirTopology topology = topology_from(configured_soc_desc());
    if (cce_index >= topology.num_cces) {
        return fmt::format("Descriptor exposes {} CCEs; CCE {} is not present.", topology.num_cces, cce_index);
    }
    return "";
}

uint64_t gddr_view_offset() { return dram_unreserved_base() + kGddrTargetOffset; }

// MIM-407: GDDR0 is the low 4 GiB tile of a Mimir, GDDR1 the high one.
uint32_t gddr_backend_for(const metal_SocDescriptor& soc, uint32_t view, uint64_t view_offset) {
    return mimir_channel_location(chiplet_offset_of(soc, view, view_offset)).tile == 0 ? abi::kBackendGddr0
                                                                                       : abi::kBackendGddr1;
}

std::vector<uint32_t> observer_args(uint32_t hart, uint32_t trid, uint64_t peer_flag_offset) {
    std::vector<uint32_t> args(abi::kNumArgs, 0);
    args[abi::kArgMode] = abi::kModeTridScopeObserver;
    args[abi::kArgStatusOffset] = static_cast<uint32_t>(tl1_done_block(hart));
    args[abi::kArgPollBudgetLo] = static_cast<uint32_t>(kPollBudgetCycles);
    args[abi::kArgPollBudgetHi] = static_cast<uint32_t>(kPollBudgetCycles >> 32);
    args[abi::kArgTrid] = trid;
    args[abi::kArgPeerFlagOffset] = static_cast<uint32_t>(peer_flag_offset);
    return args;
}

// Poisons every launch's status line, runs them all in one Program and returns their status lines in launch order;
// nullopt on the host deadline.
std::optional<std::vector<std::vector<uint32_t>>> run_kernels(
    IDevice* device, const std::vector<CceKernelLaunch>& launches) {
    for (const CceKernelLaunch& launch : launches) {
        write_cce_tl1(
            device,
            launch.cce_index,
            tl1_done_block(launch.hart),
            std::vector<uint32_t>(abi::kStatusWords, abi::kStatusPoison));
    }
    Program program = CreateProgram();
    return run_cce_kernels_with_deadline(
        device,
        program,
        launches,
        std::vector<uint32_t>(launches.size(), abi::kDoneMagic),
        abi::kStatusWords,
        kHostDeadline);
}

std::optional<std::vector<std::vector<uint32_t>>> run_on_cce(
    IDevice* device, uint32_t cce_index, const std::vector<std::pair<uint32_t, std::vector<uint32_t>>>& hart_args) {
    std::vector<CceKernelLaunch> launches;
    for (const auto& [hart, args] : hart_args) {
        launches.push_back(
            {.cce_index = cce_index, .kernel_path = kIdmaCopyKernel, .runtime_args = args, .hart = hart});
    }
    return run_kernels(device, launches);
}

// Where an MM43 command reads or writes, relative to the issuing CCE and hart. Remote targets are on the other Mimir
// at the issuing CCE's location; Gddr0 and Gddr1 are the issuing Mimir's two GDDR tiles.
enum class Target : uint32_t { OwnSource, OwnDestination, SiblingTl1, Gddr0, Gddr1, RemoteTl1, RemoteGddr };

std::string target_name(Target target) {
    switch (target) {
        case Target::OwnSource: return "own_source";
        case Target::OwnDestination: return "own_destination";
        case Target::SiblingTl1: return "sibling_tl1";
        case Target::Gddr0: return "gddr0";
        case Target::Gddr1: return "gddr1";
        case Target::RemoteTl1: return "remote_tl1";
        case Target::RemoteGddr: return "remote_gddr";
    }
    return "unknown";
}

bool is_remote_target(Target target) { return target == Target::RemoteTl1 || target == Target::RemoteGddr; }

// MIM-407: a copy from the hart's own TL1 uses the destination's backend, any other copy kBackendFromRemote.
uint32_t backend_for(Target source, Target destination) {
    if (source != Target::OwnSource) {
        return abi::kBackendFromRemote;
    }
    switch (destination) {
        case Target::OwnSource:
        case Target::OwnDestination: return abi::kBackendLocalSram;
        case Target::SiblingTl1: return abi::kBackendRemoteSram;
        case Target::Gddr0: return abi::kBackendGddr0;
        case Target::Gddr1: return abi::kBackendGddr1;
        case Target::RemoteTl1:
        case Target::RemoteGddr: return abi::kBackendD2d;
    }
    return abi::kBackendFromRemote;
}

// The view of `mimir_index`'s GDDR tile that MIM-407 serves on `backend` (kBackendGddr0 or kBackendGddr1).
uint32_t gddr_view_for_backend(const metal_SocDescriptor& soc, uint32_t mimir_index, uint32_t backend) {
    for (const uint32_t view : dram_views_for_channel(soc, mimir_index)) {
        if (gddr_backend_for(soc, view, gddr_view_offset()) == backend) {
            return view;
        }
    }
    TT_THROW("Mimir {} has no GDDR view on backend {}", mimir_index, backend);
}

// Where a region lives. Tl1 kinds name the owning CCE and a TL1 offset; Gddr names a DRAM view and an offset in it.
enum class EndpointKind { OwnTl1, SramSpa, Gddr };

struct Endpoint {
    EndpointKind kind = EndpointKind::OwnTl1;
    uint32_t cce_index = 0;
    uint32_t view = 0;
    uint64_t offset = 0;
};

Endpoint region_base(const metal_SocDescriptor& soc, uint32_t cce_index, uint32_t hart, Target target) {
    const uint64_t gddr_offset = dram_unreserved_base() + kGddrRegionBase + cce_index * kGddrRegionCceStride +
                                 hart * kGddrRegionHartStride + kGuardBytes;
    const uint32_t mimir_index = mimir_index_for_cce(cce_index);
    switch (target) {
        case Target::OwnSource:
            return {
                .kind = EndpointKind::OwnTl1,
                .cce_index = cce_index,
                .offset = tl1_data(hart) + kOwnSourceOffsetFromData};
        case Target::OwnDestination:
            return {
                .kind = EndpointKind::OwnTl1,
                .cce_index = cce_index,
                .offset = tl1_data(hart) + kOwnDestinationOffsetFromData};
        case Target::SiblingTl1:
            return {
                .kind = EndpointKind::SramSpa,
                .cce_index = sibling_cce(cce_index),
                .offset = tl1_remote_landing(cce_index, hart) + kGuardBytes};
        case Target::Gddr0:
            return {
                .kind = EndpointKind::Gddr,
                .view = gddr_view_for_backend(soc, mimir_index, abi::kBackendGddr0),
                .offset = gddr_offset};
        case Target::Gddr1:
            return {
                .kind = EndpointKind::Gddr,
                .view = gddr_view_for_backend(soc, mimir_index, abi::kBackendGddr1),
                .offset = gddr_offset};
        case Target::RemoteTl1:
            return {
                .kind = EndpointKind::SramSpa,
                .cce_index = next_mimir_cce(cce_index),
                .offset = tl1_remote_landing(cce_index, hart) + kGuardBytes};
        case Target::RemoteGddr:
            return {
                .kind = EndpointKind::Gddr,
                .view = base_dram_view_for_mimir(soc, next_mimir(mimir_index)),
                .offset = gddr_offset};
    }
    return {};
}

uint32_t region_capacity(Target target) {
    switch (target) {
        case Target::OwnSource:
        case Target::OwnDestination: return kOwnRegionCapacity;
        case Target::SiblingTl1:
        case Target::RemoteTl1: return kLandingRegionCapacity;
        case Target::Gddr0:
        case Target::Gddr1:
        case Target::RemoteGddr: return kGddrRegionCapacity;
    }
    return 0;
}

Endpoint offset_endpoint(const Endpoint& endpoint, uint64_t byte_offset) {
    Endpoint moved = endpoint;
    moved.offset += byte_offset;
    return moved;
}

Endpoint guard_before(const Endpoint& endpoint) {
    Endpoint guard = endpoint;
    guard.offset -= kGuardBytes;
    return guard;
}

void write_endpoint(
    distributed::MeshDevice& unit_mesh, IDevice* device, const Endpoint& endpoint, const std::vector<uint32_t>& words) {
    if (endpoint.kind == EndpointKind::Gddr) {
        mimir_contract::write_gddr_view(unit_mesh, endpoint.view, endpoint.offset, words);
    } else {
        write_cce_tl1(device, endpoint.cce_index, endpoint.offset, words);
    }
}

std::vector<uint32_t> read_endpoint(
    distributed::MeshDevice& unit_mesh,
    IDevice* device,
    const Endpoint& endpoint,
    uint64_t byte_offset,
    uint32_t bytes) {
    std::vector<uint32_t> words;
    if (endpoint.kind == EndpointKind::Gddr) {
        mimir_contract::read_gddr_view(unit_mesh, endpoint.view, endpoint.offset + byte_offset, bytes, words);
    } else {
        words = read_cce_tl1(device, endpoint.cce_index, endpoint.offset + byte_offset, bytes);
    }
    return words;
}

struct KernelOperand {
    uint32_t kind = 0;
    uint32_t selector = 0;
    uint64_t offset = 0;
};

KernelOperand kernel_operand(const metal_SocDescriptor& soc, uint32_t issuing_cce, const Endpoint& endpoint) {
    switch (endpoint.kind) {
        case EndpointKind::OwnTl1:
            TT_FATAL(endpoint.cce_index == issuing_cce, "own-TL1 operand names CCE {}", endpoint.cce_index);
            return {abi::kKindOwnTl1, issuing_cce, endpoint.offset};
        case EndpointKind::SramSpa: return {abi::kKindSramSpa, endpoint.cce_index, endpoint.offset};
        case EndpointKind::Gddr:
            return {
                abi::kKindGddr,
                mimir_of_view(soc, endpoint.view),
                chiplet_offset_of(soc, endpoint.view, endpoint.offset)};
    }
    return {};
}

std::string endpoint_label(const Endpoint& endpoint) {
    switch (endpoint.kind) {
        case EndpointKind::OwnTl1: return fmt::format("own TL1 0x{:x}", endpoint.offset);
        case EndpointKind::SramSpa: return fmt::format("CCE{} TL1 0x{:x}", endpoint.cce_index, endpoint.offset);
        case EndpointKind::Gddr: return fmt::format("GDDR view {} +0x{:x}", endpoint.view, endpoint.offset);
    }
    return "unknown";
}

// One kModeTridBatch launch (num_commands commands of len_bytes, trid first_trid + i * trid_stride) or one
// kModeTridTable entry (num_commands 1). Slots are byte offsets into the source and destination regions. Commands are
// packed one after another in their slots, or with shared_slots all read and write the same len_bytes.
struct Transfer {
    Target source = Target::OwnSource;
    Target destination = Target::OwnDestination;
    uint32_t source_slot = 0;
    uint32_t destination_slot = 0;
    uint32_t len_bytes = 0;
    uint32_t num_commands = 1;
    uint32_t first_trid = 0;
    uint32_t trid_stride = 0;
    bool host_verified = false;
    bool shared_slots = false;
    bool check_at_ack = false;

    uint32_t bytes() const { return shared_slots ? len_bytes : num_commands * len_bytes; }
    uint32_t trid(uint32_t command) const { return (first_trid + command * trid_stride) % abi::kNumTrids; }
    uint32_t used_trids() const {
        uint32_t mask = 0;
        for (uint32_t command = 0; command < num_commands; ++command) {
            mask |= 1u << trid(command);
        }
        return mask;
    }
};

// The regions one launch touches on one CCE and hart, with the host's model of their words. Every source slot holds
// seeded data and every destination slot poison; a guard sits before each region and right after its last slot.
// Transfers apply in the order they were added, so a later transfer may read what an earlier one wrote.
class TridScenario {
public:
    TridScenario(const metal_SocDescriptor& soc, std::string name, uint32_t cce_index, uint32_t hart, uint32_t tag) :
        soc_(soc),
        name_(std::move(name)),
        cce_index_(cce_index),
        hart_(hart),
        tag_(tag),
        seed_(mimir_test_seed() ^ (tag * 0x9E3779B9u)),
        generator_(seed_) {}

    const std::string& name() const { return name_; }
    uint32_t cce_index() const { return cce_index_; }
    uint32_t hart() const { return hart_; }
    uint32_t seed() const { return seed_; }
    const Transfer& transfer(size_t index) const { return transfers_.at(index); }
    size_t num_transfers() const { return transfers_.size(); }

    bool fits(Target target, uint32_t bytes) const {
        const auto region = regions_.find(target);
        const uint32_t used = region == regions_.end() ? 0 : static_cast<uint32_t>(region->second.words.size() * 4);
        return used + bytes <= region_capacity(target);
    }

    size_t add(
        Target source,
        Target destination,
        uint32_t len_bytes,
        uint32_t num_commands = 1,
        uint32_t first_trid = 0,
        uint32_t trid_stride = 0) {
        Transfer transfer{
            .source = source,
            .destination = destination,
            .len_bytes = len_bytes,
            .num_commands = num_commands,
            .first_trid = first_trid,
            .trid_stride = trid_stride,
            .host_verified = is_remote_target(source) || is_remote_target(destination)};
        transfer.source_slot = allocate(source, transfer.bytes(), true);
        transfer.destination_slot = allocate(destination, transfer.bytes(), false);
        transfers_.push_back(transfer);
        return transfers_.size() - 1;
    }

    // Like add, but every command reads the same source slot and writes the same destination slot, so a batch of large
    // commands fits a small region.
    size_t add_repeated(
        Target source, Target destination, uint32_t len_bytes, uint32_t num_commands, uint32_t first_trid) {
        Transfer transfer{
            .source = source,
            .destination = destination,
            .len_bytes = len_bytes,
            .num_commands = num_commands,
            .first_trid = first_trid,
            .host_verified = is_remote_target(source) || is_remote_target(destination),
            .shared_slots = true};
        transfer.source_slot = allocate(source, transfer.bytes(), true);
        transfer.destination_slot = allocate(destination, transfer.bytes(), false);
        transfers_.push_back(transfer);
        return transfers_.size() - 1;
    }

    // Transfer `previous` again, on its trid, through the same slots.
    size_t add_sharing_slots(size_t previous) {
        const Transfer transfer = transfers_.at(previous);
        transfers_.push_back(transfer);
        return transfers_.size() - 1;
    }

    // A transfer that reads what transfer `previous` wrote, into a new slot of `destination`.
    size_t add_reading_destination_of(size_t previous, Target destination) {
        Transfer transfer = transfers_.at(previous);
        transfer.source = transfer.destination;
        transfer.source_slot = transfer.destination_slot;
        transfer.destination = destination;
        transfer.destination_slot = allocate(destination, transfer.bytes(), false);
        transfer.host_verified = is_remote_target(transfer.source) || is_remote_target(destination);
        transfers_.push_back(transfer);
        return transfers_.size() - 1;
    }

    // A transfer from a new own-TL1 source slot that overwrites what transfer `previous` read.
    size_t add_overwriting_source_of(size_t previous) {
        Transfer transfer = transfers_.at(previous);
        transfer.destination = transfer.source;
        transfer.destination_slot = transfer.source_slot;
        transfer.source = Target::OwnSource;
        transfer.source_slot = allocate(Target::OwnSource, transfer.bytes(), true);
        transfer.host_verified = is_remote_target(transfer.destination);
        transfers_.push_back(transfer);
        return transfers_.size() - 1;
    }

    void set_host_verified(size_t index) { transfers_.at(index).host_verified = true; }
    void set_check_at_ack(size_t index) { transfers_.at(index).check_at_ack = true; }

    // The words transfer `index` copies, as the model has them when it runs.
    std::vector<uint32_t> copied_words(size_t index) const {
        std::map<Target, std::vector<uint32_t>> model = initial_model();
        for (size_t applied = 0; applied < index; ++applied) {
            apply(transfers_[applied], model);
        }
        const Transfer& copied = transfers_.at(index);
        const std::vector<uint32_t>& source = model.at(copied.source);
        return std::vector<uint32_t>(
            source.begin() + copied.source_slot / 4, source.begin() + (copied.source_slot + copied.bytes()) / 4);
    }

    void write(distributed::MeshDevice& unit_mesh, IDevice* device) const {
        for (const auto& [target, region] : regions_) {
            write_endpoint(unit_mesh, device, region.base, region.words);
            write_endpoint(unit_mesh, device, guard_before(region.base), leading_guard(target));
            write_endpoint(
                unit_mesh, device, offset_endpoint(region.base, region.words.size() * 4), trailing_guard(target));
        }
        std::string config;
        for (size_t index = 0; index < transfers_.size(); ++index) {
            config += describe(index) + "; ";
        }
        record_leg(name_, "config", fmt::format("CCE{} hart {}: {}", cce_index_, hart_, config));
        record_leg(name_, "seed", fmt::format("0x{:08x}", seed_));
    }

    std::string describe(size_t index) const {
        const Transfer& described = transfers_.at(index);
        return fmt::format(
            "{} x {} B {} -> {} ({} -> {}) backend VC {} trid {} stride {}{}",
            described.num_commands,
            described.len_bytes,
            target_name(described.source),
            target_name(described.destination),
            endpoint_label(slot_endpoint(described.source, described.source_slot)),
            endpoint_label(slot_endpoint(described.destination, described.destination_slot)),
            backend_for(described.source, described.destination),
            described.first_trid,
            described.trid_stride,
            described.host_verified ? " host-verified" : "");
    }

    std::vector<uint32_t> batch_args(size_t index) const {
        const Transfer& batch = transfers_.at(index);
        std::vector<uint32_t> args = common_args(abi::kModeTridBatch);
        const KernelOperand source = kernel_operand(soc_, cce_index_, slot_endpoint(batch.source, batch.source_slot));
        const KernelOperand destination =
            kernel_operand(soc_, cce_index_, slot_endpoint(batch.destination, batch.destination_slot));
        args[abi::kArgBackendVc] = backend_for(batch.source, batch.destination);
        args[abi::kArgLenBytes] = batch.len_bytes;
        args[abi::kArgSrcKind] = source.kind;
        args[abi::kArgSrcSelector] = source.selector;
        args[abi::kArgSrcOffsetLo] = static_cast<uint32_t>(source.offset);
        args[abi::kArgSrcOffsetHi] = static_cast<uint32_t>(source.offset >> 32);
        args[abi::kArgDstKind] = destination.kind;
        args[abi::kArgDstSelector] = destination.selector;
        args[abi::kArgDstOffsetLo] = static_cast<uint32_t>(destination.offset);
        args[abi::kArgDstOffsetHi] = static_cast<uint32_t>(destination.offset >> 32);
        args[abi::kArgExpectedXor] = xor_of(copied_words(index));
        TT_FATAL(
            !(batch.shared_slots && batch.host_verified && batch.destination == Target::OwnDestination),
            "the kernel's XOR poll covers num_commands * len_bytes, not one shared slot");
        args[abi::kArgNumElements] = batch.num_commands;
        args[abi::kArgSourceStrideBytes] = batch.shared_slots ? 0 : batch.len_bytes;
        args[abi::kArgDestinationStrideBytes] = batch.shared_slots ? 0 : batch.len_bytes;
        args[abi::kArgTrid] = batch.first_trid;
        args[abi::kArgTridStride] = batch.trid_stride;
        args[abi::kArgHostVerified] = batch.host_verified ? 1 : 0;
        args[abi::kArgCheckAtAck] = batch.check_at_ack ? 1 : 0;
        return args;
    }

    std::vector<uint32_t> table_args() const {
        std::vector<uint32_t> args = common_args(abi::kModeTridTable);
        args[abi::kArgNumElements] = static_cast<uint32_t>(transfers_.size());
        args[abi::kArgCmdTableOffset] = static_cast<uint32_t>(tl1_data(hart_) + kTableOffsetFromData);
        return args;
    }

    void write_table(IDevice* device) const {
        std::vector<uint32_t> table(transfers_.size() * abi::kTableWords, 0);
        for (size_t index = 0; index < transfers_.size(); ++index) {
            const Transfer& entry = transfers_[index];
            TT_FATAL(entry.num_commands == 1, "table entry {} has {} commands", index, entry.num_commands);
            const KernelOperand source =
                kernel_operand(soc_, cce_index_, slot_endpoint(entry.source, entry.source_slot));
            const KernelOperand destination =
                kernel_operand(soc_, cce_index_, slot_endpoint(entry.destination, entry.destination_slot));
            uint32_t* words = table.data() + index * abi::kTableWords;
            words[abi::kTableSrcKind] = source.kind;
            words[abi::kTableSrcSelector] = source.selector;
            words[abi::kTableSrcOffsetLo] = static_cast<uint32_t>(source.offset);
            words[abi::kTableSrcOffsetHi] = static_cast<uint32_t>(source.offset >> 32);
            words[abi::kTableDstKind] = destination.kind;
            words[abi::kTableDstSelector] = destination.selector;
            words[abi::kTableDstOffsetLo] = static_cast<uint32_t>(destination.offset);
            words[abi::kTableDstOffsetHi] = static_cast<uint32_t>(destination.offset >> 32);
            words[abi::kTableBackendVc] = backend_for(entry.source, entry.destination);
            words[abi::kTableTrid] = entry.first_trid;
            words[abi::kTableLenBytes] = entry.len_bytes;
            words[abi::kTableHostVerified] = entry.host_verified ? 1 : 0;
        }
        write_cce_tl1(device, cce_index_, tl1_data(hart_) + kTableOffsetFromData, table);
    }

    uint32_t table_success_code() const {
        const bool every_entry_host_verified = std::all_of(
            transfers_.begin(), transfers_.end(), [](const Transfer& entry) { return entry.host_verified; });
        return every_entry_host_verified ? abi::kCodeAcked : abi::kCodeLanded;
    }

    uint32_t table_used_trids() const {
        uint32_t mask = 0;
        for (const Transfer& entry : transfers_) {
            mask |= entry.used_trids();
        }
        return mask;
    }

    // Every region a transfer writes holds the model's words, and every guard is intact.
    void expect_landed(distributed::MeshDevice& unit_mesh, IDevice* device, const std::string& label) const {
        for (const auto& [target, region] : regions_) {
            const bool written = std::any_of(transfers_.begin(), transfers_.end(), [target](const Transfer& entry) {
                return entry.destination == target;
            });
            if (written) {
                expect_region(unit_mesh, device, target, label);
            }
        }
    }

    void expect_region(
        distributed::MeshDevice& unit_mesh, IDevice* device, Target target, const std::string& label) const {
        const Region& region = regions_.at(target);
        EXPECT_TRUE(compare_with_recheck(
            fmt::format("MM43 {} {}", label, target_name(target)),
            name_ + "_" + target_name(target),
            final_model().at(target),
            [&unit_mesh, device, &region](uint64_t byte_offset, uint32_t bytes) {
                return read_endpoint(unit_mesh, device, region.base, byte_offset, bytes);
            }))
            << label << " seed 0x" << std::hex << seed_;
        EXPECT_EQ(
            read_endpoint(unit_mesh, device, region.base, region.words.size() * 4, kGuardBytes), trailing_guard(target))
            << label << ": guard after " << target_name(target) << " overwritten";
        EXPECT_EQ(read_endpoint(unit_mesh, device, guard_before(region.base), 0, kGuardBytes), leading_guard(target))
            << label << ": guard before " << target_name(target) << " overwritten";
    }

private:
    struct Region {
        Endpoint base;
        std::vector<uint32_t> words;
    };

    uint32_t allocate(Target target, uint32_t bytes, bool source) {
        TT_FATAL(fits(target, bytes), "{} has no room for {} more bytes", target_name(target), bytes);
        auto [region, inserted] = regions_.try_emplace(target);
        if (inserted) {
            region->second.base = region_base(soc_, cce_index_, hart_, target);
        }
        const uint32_t slot = static_cast<uint32_t>(region->second.words.size() * 4);
        const uint32_t poison_tag = tag_ * 16 + static_cast<uint32_t>(transfers_.size());
        for (uint32_t word = 0; word < bytes / 4; ++word) {
            region->second.words.push_back(
                source ? static_cast<uint32_t>(generator_()) : 0x5EED0000u ^ (poison_tag << 16) ^ word);
        }
        return slot;
    }

    Endpoint slot_endpoint(Target target, uint32_t slot) const {
        return offset_endpoint(regions_.at(target).base, slot);
    }

    uint32_t trailing_tag(Target target) const { return (tag_ * 8 + static_cast<uint32_t>(target)) & 0x7FF; }
    std::vector<uint32_t> trailing_guard(Target target) const { return make_guard(trailing_tag(target)); }
    std::vector<uint32_t> leading_guard(Target target) const {
        return make_guard(kRegionLeadingGuardTag | trailing_tag(target));
    }

    std::map<Target, std::vector<uint32_t>> initial_model() const {
        std::map<Target, std::vector<uint32_t>> model;
        for (const auto& [target, region] : regions_) {
            model[target] = region.words;
        }
        return model;
    }

    static void apply(const Transfer& transfer, std::map<Target, std::vector<uint32_t>>& model) {
        const std::vector<uint32_t> source = model.at(transfer.source);
        std::copy(
            source.begin() + transfer.source_slot / 4,
            source.begin() + (transfer.source_slot + transfer.bytes()) / 4,
            model.at(transfer.destination).begin() + transfer.destination_slot / 4);
    }

    std::map<Target, std::vector<uint32_t>> final_model() const {
        std::map<Target, std::vector<uint32_t>> model = initial_model();
        for (const Transfer& transfer : transfers_) {
            apply(transfer, model);
        }
        return model;
    }

    std::vector<uint32_t> common_args(uint32_t mode) const {
        std::vector<uint32_t> args(abi::kNumArgs, 0);
        args[abi::kArgMode] = mode;
        args[abi::kArgStatusOffset] = static_cast<uint32_t>(tl1_done_block(hart_));
        args[abi::kArgPollBudgetLo] = static_cast<uint32_t>(kPollBudgetCycles);
        args[abi::kArgPollBudgetHi] = static_cast<uint32_t>(kPollBudgetCycles >> 32);
        args[abi::kArgTraceOffset] = static_cast<uint32_t>(tl1_data(hart_) + kTraceOffsetFromData);
        return args;
    }

    const metal_SocDescriptor& soc_;
    std::string name_;
    uint32_t cce_index_;
    uint32_t hart_;
    uint32_t tag_;
    uint32_t seed_;
    std::mt19937 generator_;
    std::map<Target, Region> regions_;
    std::vector<Transfer> transfers_;
};

// Every trid in `used_trids` reached 0 pending exactly once, at a rank no other trid holds, and no other trid was
// used. Returns the trace as "trid:rank@cycles" entries.
std::string expect_trace(
    IDevice* device, uint32_t cce_index, uint32_t hart, uint32_t used_trids, const std::string& label) {
    const std::vector<uint32_t> trace =
        read_cce_tl1(device, cce_index, tl1_data(hart) + kTraceOffsetFromData, abi::kTraceBytes);
    std::vector<bool> rank_seen(abi::kNumTrids, false);
    std::string summary;
    for (uint32_t trid = 0; trid < abi::kNumTrids; ++trid) {
        const uint32_t rank = trace[trid * abi::kTraceWordsPerTrid];
        const uint32_t cycles = trace[trid * abi::kTraceWordsPerTrid + 1];
        if ((used_trids & (1u << trid)) == 0) {
            EXPECT_EQ(rank, abi::kTraceNotUsed) << label << ": trid " << trid << " was not used but has a trace entry";
            continue;
        }
        summary += fmt::format("{}:{}@{} ", trid, rank, cycles);
        EXPECT_NE(rank, abi::kTraceNotAcked) << label << ": trid " << trid << " never read 0 pending (ID leak)";
        EXPECT_NE(rank, abi::kTraceNotUsed) << label << ": trid " << trid << " is missing from the trace";
        if (rank < abi::kNumTrids) {
            EXPECT_FALSE(rank_seen[rank]) << label << ": two trids completed at rank " << rank;
            rank_seen[rank] = true;
        }
    }
    return summary;
}

CceKernelLaunch scenario_launch(const TridScenario& scenario, std::vector<uint32_t> args) {
    return {
        .cce_index = scenario.cce_index(),
        .kernel_path = kIdmaCopyKernel,
        .runtime_args = std::move(args),
        .hart = scenario.hart()};
}

// The code, command count and trace of a finished launch of `scenario`.
void expect_status(
    IDevice* device,
    const TridScenario& scenario,
    const std::vector<uint32_t>& status,
    uint32_t success_code,
    uint32_t expected_commands,
    uint32_t used_trids,
    const std::string& label) {
    const std::string detail = status_detail(status);
    log_info(tt::LogTest, "MM43 {}: {}", label, detail);
    record_leg(scenario.name(), "status", detail);
    EXPECT_EQ(status[abi::kStatusCode], success_code)
        << label << ": " << detail << code_note(status[abi::kStatusCode]) << " seed 0x" << std::hex << scenario.seed();
    EXPECT_EQ(status[abi::kStatusElementsIssued], expected_commands) << label << ": " << detail;
    record_leg(
        scenario.name(), "trace", expect_trace(device, scenario.cce_index(), scenario.hart(), used_trids, label));
}

// Runs one launch of `scenario` that was already written. Returns its status, or nullopt when the host deadline passed;
// nothing else may then run in this session (MIM-601).
std::optional<std::vector<uint32_t>> run_launch(
    IDevice* device, const TridScenario& scenario, std::vector<uint32_t> args, const std::string& label) {
    log_info(tt::LogTest, "MM43 {}", label);
    const std::optional<std::vector<std::vector<uint32_t>>> statuses =
        run_kernels(device, {scenario_launch(scenario, std::move(args))});
    if (!statuses.has_value()) {
        record_leg(scenario.name(), "status", fmt::format("no status after {} s", kHostDeadline.count()));
        ADD_FAILURE() << label << ": no status after " << kHostDeadline.count() << " s" << kFreshSessionNote;
        return std::nullopt;
    }
    return statuses->front();
}

// Writes `scenario` and runs its transfer `index` as one kModeTridBatch launch, then checks status, trace and data.
std::optional<std::vector<uint32_t>> run_scenario_batch(
    distributed::MeshDevice& unit_mesh,
    IDevice* device,
    const TridScenario& scenario,
    size_t index,
    const std::string& label) {
    scenario.write(unit_mesh, device);
    const Transfer& batch = scenario.transfer(index);
    const std::string full_label = fmt::format("{}: {}", label, scenario.describe(index));
    const std::optional<std::vector<uint32_t>> status =
        run_launch(device, scenario, scenario.batch_args(index), full_label);
    if (status.has_value()) {
        expect_status(
            device,
            scenario,
            *status,
            batch.host_verified ? abi::kCodeAcked : abi::kCodeLanded,
            batch.num_commands,
            batch.used_trids(),
            full_label);
        scenario.expect_landed(unit_mesh, device, full_label);
    }
    return status;
}

// Writes `scenario` and its table and runs every transfer as one kModeTridTable launch, then checks it.
std::optional<std::vector<uint32_t>> run_scenario_table(
    distributed::MeshDevice& unit_mesh, IDevice* device, const TridScenario& scenario, const std::string& label) {
    scenario.write(unit_mesh, device);
    scenario.write_table(device);
    const std::string full_label =
        fmt::format("{} on CCE{}: table of {} entries", label, scenario.cce_index(), scenario.num_transfers());
    const std::optional<std::vector<uint32_t>> status = run_launch(device, scenario, scenario.table_args(), full_label);
    if (status.has_value()) {
        expect_status(
            device,
            scenario,
            *status,
            scenario.table_success_code(),
            static_cast<uint32_t>(scenario.num_transfers()),
            scenario.table_used_trids(),
            full_label);
        scenario.expect_landed(unit_mesh, device, full_label);
    }
    return status;
}

bool has_remote_mimir() { return topology_from(configured_soc_desc()).num_mimirs >= 2; }

// Opens device 0 for one test and closes it at TearDown, or earlier with close_device().
class MimirTridTest {
protected:
    IDevice* open_device() {
        device_closer_ = std::make_unique<ScopedDevice>();
        IDevice* device = device_closer_->device();
        check_tl1_layout();
        record_descriptor_properties(soc_desc_of(device));
        return device;
    }

    void close_device() { device_closer_.reset(); }

    std::unique_ptr<ScopedDevice> device_closer_;
};

// One case per CCE of the descriptor; hart kTridHart issues unless a case says otherwise.
class MimirTridIdma : public ::testing::TestWithParam<uint32_t>, protected MimirTridTest {
protected:
    void SetUp() override {
        skip_unless_programmable_dram_package();
        if (const std::string skip_reason = cce_skip_reason(GetParam()); !skip_reason.empty()) {
            GTEST_SKIP() << skip_reason;
        }
    }

    void TearDown() override { close_device(); }
};

inline constexpr uint32_t kUniqueIdsCommands = abi::kNumTrids;
inline constexpr uint32_t kUniqueIdsCommandBytes = 512;

// One kModeTridBatch launch per leg, each from its own scenario: kUniqueIdsCommands commands on trids 0-15. False once
// a launch passed the host deadline.
bool run_unique_id_legs(
    distributed::MeshDevice& unit_mesh,
    IDevice* device,
    uint32_t cce_index,
    const std::vector<std::pair<Target, Target>>& legs,
    const std::string& name,
    uint32_t tag) {
    const metal_SocDescriptor& soc = soc_desc_of(device);
    for (uint32_t leg = 0; leg < legs.size(); ++leg) {
        TridScenario scenario(soc, fmt::format("{}_leg{}", name, leg), cce_index, kTridHart, tag + leg);
        const size_t batch = scenario.add(
            legs[leg].first,
            legs[leg].second,
            kUniqueIdsCommandBytes,
            kUniqueIdsCommands,
            /*first_trid=*/0,
            /*trid_stride=*/1);
        if (!run_scenario_batch(unit_mesh, device, scenario, batch, fmt::format("{} on CCE{}", name, cce_index))
                 .has_value()) {
            return false;
        }
    }
    return true;
}

// Per-command trids 0-15 from the hart's own TL1 into its own TL1 (MIM-407 backend 5).
TEST_P(MimirTridIdma, UniqueIdsOwnTl1) {
    IDevice* device = open_device();
    run_unique_id_legs(
        device_closer_->mesh(),
        device,
        GetParam(),
        {{Target::OwnSource, Target::OwnDestination}},
        "UniqueIdsOwnTl1",
        0x10);
}

// Per-command trids 0-15 to the sibling CCE's TL1 (backend 2) and back (backend 5).
TEST_P(MimirTridIdma, UniqueIdsSiblingTl1) {
    IDevice* device = open_device();
    run_unique_id_legs(
        device_closer_->mesh(),
        device,
        GetParam(),
        {{Target::OwnSource, Target::SiblingTl1}, {Target::SiblingTl1, Target::OwnDestination}},
        "UniqueIdsSiblingTl1",
        0x20);
}

// The dram_sharded Trid intent: per-command trids 0-15 writing GDDR0 (backend 0) and GDDR1 (backend 1), and reading
// each back on backend 5.
TEST_P(MimirTridIdma, UniqueIdsLocalGddr) {
    IDevice* device = open_device();
    run_unique_id_legs(
        device_closer_->mesh(),
        device,
        GetParam(),
        {{Target::OwnSource, Target::Gddr0},
         {Target::OwnSource, Target::Gddr1},
         {Target::Gddr0, Target::OwnDestination},
         {Target::Gddr1, Target::OwnDestination}},
        "UniqueIdsLocalGddr",
        0x30);
}

inline constexpr uint32_t kTableCommandBytes = 512;

// One kModeTridTable batch across every local target, both directions, and on 2x writes to the remote TL1 and GDDR
// (host-verified). Each entry has its own trid. Remote reads stay in MimirTridRemote: a table entry into the hart's own
// TL1 that is host-verified gets no data poll, so the status store could overlap its landing (MIM-560).
TEST_P(MimirTridIdma, MixedTargets) {
    skip_unless_mimir_local_package();
    if (has_remote_mimir()) {
        MIMIR_REQUIRE_REMOTE_PREFLIGHT();
    }
    IDevice* device = open_device();
    const bool remote = has_remote_mimir();
    if (remote) {
        ASSERT_TRUE(mimir_remote_remap_result().empty()) << mimir_remote_remap_result();
    }
    std::vector<std::pair<Target, Target>> entries = {
        {Target::OwnSource, Target::OwnDestination},
        {Target::OwnSource, Target::SiblingTl1},
        {Target::OwnSource, Target::Gddr0},
        {Target::OwnSource, Target::Gddr1},
        {Target::Gddr0, Target::OwnDestination},
        {Target::Gddr1, Target::OwnDestination},
        {Target::SiblingTl1, Target::OwnDestination}};
    if (remote) {
        entries.push_back({Target::OwnSource, Target::RemoteTl1});
        entries.push_back({Target::OwnSource, Target::RemoteGddr});
    }
    TridScenario scenario(soc_desc_of(device), "MixedTargets", GetParam(), kTridHart, 0x40);
    for (uint32_t entry = 0; entry < entries.size(); ++entry) {
        scenario.add(entries[entry].first, entries[entry].second, kTableCommandBytes, 1, entry);
    }
    run_scenario_table(device_closer_->mesh(), device, scenario, "MixedTargets");
}

inline constexpr uint32_t kReuseWrapCommands = abi::kMaxElements;
inline constexpr uint32_t kReuseWrapCommandBytes = 256;

// 64 commands over 16 trids, so every trid is reused four times while earlier commands on it may be pending: in order
// from trid 0 writing GDDR0, then from trid 11 with stride 5 reading GDDR1.
TEST_P(MimirTridIdma, ReuseWrap) {
    IDevice* device = open_device();
    const metal_SocDescriptor& soc = soc_desc_of(device);
    TridScenario write_scenario(soc, "ReuseWrap_write", GetParam(), kTridHart, 0x50);
    const size_t write_batch = write_scenario.add(
        Target::OwnSource,
        Target::Gddr0,
        kReuseWrapCommandBytes,
        kReuseWrapCommands,
        /*first_trid=*/0,
        /*trid_stride=*/1);
    if (!run_scenario_batch(
             device_closer_->mesh(), device, write_scenario, write_batch, fmt::format("ReuseWrap on CCE{}", GetParam()))
             .has_value()) {
        return;
    }
    TridScenario read_scenario(soc, "ReuseWrap_read", GetParam(), kTridHart, 0x51);
    const size_t read_batch = read_scenario.add(
        Target::Gddr1,
        Target::OwnDestination,
        kReuseWrapCommandBytes,
        kReuseWrapCommands,
        /*first_trid=*/11,
        /*trid_stride=*/5);
    run_scenario_batch(
        device_closer_->mesh(), device, read_scenario, read_batch, fmt::format("ReuseWrap on CCE{}", GetParam()));
}

// 16 commands outstanding on one trid: trid 9 writing GDDR0, then trid 15 writing the sibling CCE's TL1. The trid's
// pending count must cover more than one command, and every command's data must be in place when it reads 0 (the
// kernel checks once at that point, kArgCheckAtAck, instead of waiting for the data).
TEST_P(MimirTridIdma, DuplicateId) {
    IDevice* device = open_device();
    const metal_SocDescriptor& soc = soc_desc_of(device);
    const std::string label = fmt::format("DuplicateId on CCE{}", GetParam());
    TridScenario gddr_scenario(soc, "DuplicateId_gddr0", GetParam(), kTridHart, 0x58);
    const size_t gddr_batch = gddr_scenario.add(
        Target::OwnSource, Target::Gddr0, kUniqueIdsCommandBytes, kUniqueIdsCommands, /*first_trid=*/9);
    gddr_scenario.set_check_at_ack(gddr_batch);
    const std::optional<std::vector<uint32_t>> gddr_status =
        run_scenario_batch(device_closer_->mesh(), device, gddr_scenario, gddr_batch, label);
    if (!gddr_status.has_value()) {
        return;
    }
    EXPECT_GT((*gddr_status)[abi::kStatusPendingHighWater], 1u)
        << label << ": trid 9 never counted more than one outstanding command; " << status_detail(*gddr_status);
    TridScenario sibling_scenario(soc, "DuplicateId_sibling", GetParam(), kTridHart, 0x59);
    const size_t sibling_batch = sibling_scenario.add(
        Target::OwnSource, Target::SiblingTl1, kUniqueIdsCommandBytes, kUniqueIdsCommands, /*first_trid=*/15);
    sibling_scenario.set_check_at_ack(sibling_batch);
    const std::optional<std::vector<uint32_t>> sibling_status =
        run_scenario_batch(device_closer_->mesh(), device, sibling_scenario, sibling_batch, label);
    if (sibling_status.has_value()) {
        EXPECT_GT((*sibling_status)[abi::kStatusPendingHighWater], 1u)
            << label << ": trid 15 never counted more than one outstanding command; " << status_detail(*sibling_status);
    }
}

inline constexpr uint32_t kNominalDepthCommandBytes = 1024;

// 16 trids outstanding back to back in one table: writes and reads alternating over GDDR0 and GDDR1 and, on 2x, writes
// over D2D to the remote GDDR. Asserts a trid read pending right after its issue.
TEST_P(MimirTridIdma, NominalDepth) {
    skip_unless_mimir_local_package();
    if (has_remote_mimir()) {
        MIMIR_REQUIRE_REMOTE_PREFLIGHT();
    }
    IDevice* device = open_device();
    const bool remote = has_remote_mimir();
    if (remote) {
        ASSERT_TRUE(mimir_remote_remap_result().empty()) << mimir_remote_remap_result();
    }
    std::vector<std::pair<Target, Target>> kinds = {
        {Target::OwnSource, Target::Gddr0},
        {Target::Gddr0, Target::OwnDestination},
        {Target::OwnSource, Target::Gddr1},
        {Target::Gddr1, Target::OwnDestination}};
    if (remote) {
        kinds.push_back({Target::OwnSource, Target::RemoteGddr});
    }
    TridScenario scenario(soc_desc_of(device), "NominalDepth", GetParam(), kTridHart, 0x60);
    for (uint32_t trid = 0; trid < abi::kNumTrids; ++trid) {
        const std::pair<Target, Target>& kind = kinds[trid % kinds.size()];
        scenario.add(kind.first, kind.second, kNominalDepthCommandBytes, 1, trid);
    }
    const std::optional<std::vector<uint32_t>> status =
        run_scenario_table(device_closer_->mesh(), device, scenario, "NominalDepth");
    if (!status.has_value()) {
        return;
    }
    const uint32_t high_water = (*status)[abi::kStatusPendingHighWater];
    RecordProperty("pending_high_water", static_cast<int>(high_water));
    // Each entry has its own trid, so a trid's count right after its issue is at most 1.
    EXPECT_GE(high_water, 1u) << "no trid read pending right after its issue";
}

inline constexpr uint32_t kSaturateCommands = 32;
inline constexpr uint32_t kSaturateCombinedRounds = 8;

// The write backends of the descriptor (MIM-407): 0 GDDR0, 1 GDDR1, 2 sibling TL1, 5 own TL1 and, on 2x, 3 D2D.
std::vector<Target> saturate_destinations(bool remote) {
    std::vector<Target> destinations = {Target::Gddr0, Target::Gddr1, Target::SiblingTl1, Target::OwnDestination};
    if (remote) {
        destinations.push_back(Target::RemoteGddr);
    }
    return destinations;
}

// Each write backend with kSaturateCommands 8 KiB commands on one trid, one backend at a time, then every backend in
// one table of 8 KiB entries, kSaturateCombinedRounds per backend, each backend on its own trid. Commands of a leg
// share one source and one destination slot, so 8 KiB commands fit the TL1 regions. Small commands to TL1 finish
// before the next issue (high water 1 with 256 B), so every command is kMaxBytesInPacket. The per-backend legs assert
// more than one command pending on their trid and the combined table more than one pending in total across its trids;
// every leg records its high water and VC space stall cycles. Reads only the hart's own TL1.
TEST_P(MimirTridIdma, Saturate) {
    skip_unless_mimir_local_package();
    if (has_remote_mimir()) {
        MIMIR_REQUIRE_REMOTE_PREFLIGHT();
    }
    IDevice* device = open_device();
    const bool remote = has_remote_mimir();
    if (remote) {
        ASSERT_TRUE(mimir_remote_remap_result().empty()) << mimir_remote_remap_result();
    }
    const metal_SocDescriptor& soc = soc_desc_of(device);
    const std::vector<Target> destinations = saturate_destinations(remote);
    const auto record_depth = [](const std::string& leg, const std::vector<uint32_t>& status, bool assert_depth) {
        const uint32_t high_water = status[abi::kStatusPendingHighWater];
        RecordProperty(leg + "_pending_high_water", static_cast<int>(high_water));
        RecordProperty(leg + "_vc_space_cycles", static_cast<int>(status[abi::kStatusVcSpaceCycles]));
        if (assert_depth) {
            EXPECT_GE(high_water, 2u) << leg << ": never more than one 8 KiB command pending on a trid";
        }
    };
    for (uint32_t pool = 0; pool < destinations.size(); ++pool) {
        const uint32_t backend = backend_for(Target::OwnSource, destinations[pool]);
        TridScenario scenario(soc, fmt::format("Saturate_vc{}", backend), GetParam(), kTridHart, 0x68 + pool);
        const size_t batch = scenario.add_repeated(
            Target::OwnSource, destinations[pool], abi::kMaxBytesInPacket, kSaturateCommands, backend);
        const std::optional<std::vector<uint32_t>> status = run_scenario_batch(
            device_closer_->mesh(),
            device,
            scenario,
            batch,
            fmt::format("Saturate VC {} on CCE{}", backend, GetParam()));
        if (!status.has_value()) {
            return;
        }
        record_depth(fmt::format("vc{}", backend), *status, true);
    }

    TridScenario combined(soc, "Saturate_combined", GetParam(), kTridHart, 0x6F);
    std::vector<size_t> first_entries;
    for (const Target destination : destinations) {
        first_entries.push_back(combined.add(
            Target::OwnSource, destination, abi::kMaxBytesInPacket, 1, backend_for(Target::OwnSource, destination)));
    }
    for (uint32_t round = 1; round < kSaturateCombinedRounds; ++round) {
        for (const size_t first_entry : first_entries) {
            combined.add_sharing_slots(first_entry);
        }
    }
    const std::optional<std::vector<uint32_t>> status =
        run_scenario_table(device_closer_->mesh(), device, combined, "Saturate combined");
    if (status.has_value()) {
        // A trid's previous entry is a whole round back, so the per-trid high water was 1 (rv-zebu-08, 2026-10-22);
        // combined saturation is the total pending over every backend's trid.
        record_depth("combined", *status, false);
        const uint32_t total_high_water = (*status)[abi::kStatusProbeSecond];
        const uint32_t largest_issue_gap = (*status)[abi::kStatusProbeFirst];
        RecordProperty("combined_total_pending_high_water", static_cast<int>(total_high_water));
        RecordProperty("combined_largest_issue_gap_cycles", static_cast<int>(largest_issue_gap));
        EXPECT_GE(total_high_water, 2u)
            << "combined: never more than one 8 KiB command pending across the backends; largest gap between "
            << "issues " << largest_issue_gap << " cycles";
    }
}

// BarrierThenDependent: the upstream test_transaction_id intent in two launches, since no kernel mode waits on one trid
// and then issues within a launch. The second launch issues the dependent transfer on the same trids. A host-verified
// launch to GDDR gets only the per-trid ack wait; one into the hart's own TL1 also the XOR poll.

// Write own TL1 -> GDDR0, host-verified, then read that GDDR0 slot back into own TL1, host-verified. Gates on GDDR0 and
// on the read holding what the write wrote.
TEST_P(MimirTridIdma, BarrierThenDependentReadAfterWrite) {
    IDevice* device = open_device();
    TridScenario scenario(soc_desc_of(device), "ReadAfterWrite", GetParam(), kTridHart, 0x70);
    const size_t write_batch = scenario.add(
        Target::OwnSource,
        Target::Gddr0,
        kUniqueIdsCommandBytes,
        kUniqueIdsCommands,
        /*first_trid=*/0,
        /*trid_stride=*/1);
    scenario.set_host_verified(write_batch);
    const size_t read_batch = scenario.add_reading_destination_of(write_batch, Target::OwnDestination);
    scenario.set_host_verified(read_batch);
    scenario.write(device_closer_->mesh(), device);

    const std::string label = fmt::format("BarrierThenDependent ReadAfterWrite on CCE{}", GetParam());
    const std::optional<std::vector<uint32_t>> write_status =
        run_launch(device, scenario, scenario.batch_args(write_batch), label + " write");
    if (!write_status.has_value()) {
        return;
    }
    expect_status(
        device,
        scenario,
        *write_status,
        abi::kCodeAcked,
        kUniqueIdsCommands,
        scenario.transfer(write_batch).used_trids(),
        label + " write");
    const std::optional<std::vector<uint32_t>> read_status =
        run_launch(device, scenario, scenario.batch_args(read_batch), label + " dependent read");
    if (!read_status.has_value()) {
        return;
    }
    const uint32_t read_code = (*read_status)[abi::kStatusCode];
    const std::string read_detail = status_detail(*read_status);
    log_info(tt::LogTest, "MM43 {} dependent read: {}{}", label, read_detail, code_note(read_code));
    record_leg("dependent_read", "status", read_detail);
    EXPECT_EQ(read_code, abi::kCodeAcked) << label << ": " << read_detail;
    EXPECT_EQ((*read_status)[abi::kStatusFlags] & abi::kFlagAcked, abi::kFlagAcked) << label << ": " << read_detail;

    scenario.expect_region(device_closer_->mesh(), device, Target::Gddr0, label);
    scenario.expect_region(device_closer_->mesh(), device, Target::OwnDestination, label);
    RecordProperty("dependent_data_cycles", static_cast<int>((*read_status)[abi::kStatusDataCycles]));
}

// Read GDDR0 -> own TL1, with the data checked once when its trids read 0 (kArgCheckAtAck), then overwrite that GDDR0
// slot, host-verified. Gates on both: the read holds the old data and GDDR0 the new.
TEST_P(MimirTridIdma, BarrierThenDependentWriteAfterRead) {
    IDevice* device = open_device();
    TridScenario scenario(soc_desc_of(device), "WriteAfterRead", GetParam(), kTridHart, 0x78);
    const size_t read_batch = scenario.add(
        Target::Gddr0,
        Target::OwnDestination,
        kUniqueIdsCommandBytes,
        kUniqueIdsCommands,
        /*first_trid=*/0,
        /*trid_stride=*/1);
    const size_t write_batch = scenario.add_overwriting_source_of(read_batch);
    scenario.set_host_verified(write_batch);
    scenario.set_check_at_ack(read_batch);
    scenario.write(device_closer_->mesh(), device);

    const std::string label = fmt::format("BarrierThenDependent WriteAfterRead on CCE{}", GetParam());
    const std::optional<std::vector<uint32_t>> read_status =
        run_launch(device, scenario, scenario.batch_args(read_batch), label + " read");
    if (!read_status.has_value()) {
        return;
    }
    expect_status(
        device,
        scenario,
        *read_status,
        abi::kCodeLanded,
        kUniqueIdsCommands,
        scenario.transfer(read_batch).used_trids(),
        label + " read");
    const std::optional<std::vector<uint32_t>> write_status =
        run_launch(device, scenario, scenario.batch_args(write_batch), label + " dependent write");
    if (!write_status.has_value()) {
        return;
    }
    expect_status(
        device,
        scenario,
        *write_status,
        abi::kCodeAcked,
        kUniqueIdsCommands,
        scenario.transfer(write_batch).used_trids(),
        label + " dependent write");
    scenario.expect_landed(device_closer_->mesh(), device, label);
}

inline constexpr uint32_t kMultiHartCommands = 16;

// All harts of one CCE issue at once, each its own GDDR region (even harts GDDR0, odd GDDR1), so no TL1 is written
// while any hart may store its status (MIM-560). Each hart gets its own trid, because the pending counters may be
// shared by every hart of the CCE.
TEST_P(MimirTridIdma, MultiHart) {
    if (!kEightHartLaunchPasses) {
        GTEST_SKIP() << "Gated on MimirCacheProbe.EightHartLaunch (kEightHartLaunchPasses).";
    }
    IDevice* device = open_device();
    const metal_SocDescriptor& soc = soc_desc_of(device);
    std::vector<std::unique_ptr<TridScenario>> scenarios;
    std::vector<CceKernelLaunch> launches;
    for (uint32_t hart = 0; hart < cce_hart_count(); ++hart) {
        scenarios.push_back(
            std::make_unique<TridScenario>(soc, fmt::format("MultiHart_hart{}", hart), GetParam(), hart, 0x80 + hart));
        TridScenario& scenario = *scenarios.back();
        scenario.add(
            Target::OwnSource,
            hart % 2 == 0 ? Target::Gddr0 : Target::Gddr1,
            kUniqueIdsCommandBytes,
            kMultiHartCommands,
            2 * hart,
            0);
        scenario.write(device_closer_->mesh(), device);
        launches.push_back(scenario_launch(scenario, scenario.batch_args(0)));
    }
    const std::string label = fmt::format("MultiHart on CCE{}", GetParam());
    log_info(tt::LogTest, "MM43 {}", label);
    const std::optional<std::vector<std::vector<uint32_t>>> statuses = run_kernels(device, launches);
    ASSERT_TRUE(statuses.has_value()) << label << ": no status after " << kHostDeadline.count() << " s"
                                      << kFreshSessionNote;
    for (uint32_t hart = 0; hart < cce_hart_count(); ++hart) {
        const TridScenario& scenario = *scenarios[hart];
        const std::string hart_label = fmt::format("{} hart {}: {}", label, hart, scenario.describe(0));
        expect_status(
            device,
            scenario,
            (*statuses)[hart],
            abi::kCodeLanded,
            kMultiHartCommands,
            scenario.transfer(0).used_trids(),
            hart_label);
        scenario.expect_landed(device_closer_->mesh(), device, hart_label);
    }
}

inline constexpr uint32_t kFairnessHeavyCommandBytes = 512;
inline constexpr uint32_t kFairnessTableEntries = abi::kNumTrids;

// The parameter CCE keeps 16 trids outstanding with 64 GDDR0 writes while every other CCE runs a table of GDDR reads
// and writes and, on 2x, D2D writes to the remote GDDR, all in one Program. Every source must finish; each one's VC
// space and ack stalls are recorded. Only GDDR is written, so no CCE's TL1 takes a write while another stores status.
TEST_P(MimirTridIdma, Fairness) {
    skip_unless_mimir_local_package();
    if (has_remote_mimir()) {
        MIMIR_REQUIRE_REMOTE_PREFLIGHT();
    }
    IDevice* device = open_device();
    const bool remote = has_remote_mimir();
    if (remote) {
        ASSERT_TRUE(mimir_remote_remap_result().empty()) << mimir_remote_remap_result();
    }
    const metal_SocDescriptor& soc = soc_desc_of(device);
    const uint32_t heavy_cce = GetParam();
    const uint32_t num_cces = topology_from(soc).num_cces;

    std::vector<std::unique_ptr<TridScenario>> scenarios;
    std::vector<CceKernelLaunch> launches;
    std::vector<uint32_t> expected_codes;
    std::vector<uint32_t> expected_commands;
    std::vector<uint32_t> used_trids;
    scenarios.push_back(std::make_unique<TridScenario>(
        soc, fmt::format("Fairness_heavy_cce{}", heavy_cce), heavy_cce, kTridHart, 0x90 + heavy_cce));
    scenarios.back()->add(
        Target::OwnSource,
        Target::Gddr0,
        kFairnessHeavyCommandBytes,
        abi::kMaxElements,
        /*first_trid=*/0,
        /*trid_stride=*/1);
    scenarios.back()->write(device_closer_->mesh(), device);
    launches.push_back(scenario_launch(*scenarios.back(), scenarios.back()->batch_args(0)));
    expected_codes.push_back(abi::kCodeLanded);
    expected_commands.push_back(abi::kMaxElements);
    used_trids.push_back(scenarios.back()->transfer(0).used_trids());

    std::vector<std::pair<Target, Target>> kinds = {
        {Target::OwnSource, Target::Gddr0},
        {Target::Gddr1, Target::OwnDestination},
        {Target::OwnSource, Target::Gddr1},
        {Target::Gddr0, Target::OwnDestination}};
    if (remote) {
        kinds.push_back({Target::OwnSource, Target::RemoteGddr});
    }
    for (uint32_t cce_index = 0; cce_index < num_cces; ++cce_index) {
        if (cce_index == heavy_cce) {
            continue;
        }
        scenarios.push_back(std::make_unique<TridScenario>(
            soc, fmt::format("Fairness_cce{}", cce_index), cce_index, kTridHart, 0x98 + cce_index));
        TridScenario& scenario = *scenarios.back();
        for (uint32_t entry = 0; entry < kFairnessTableEntries; ++entry) {
            const std::pair<Target, Target>& kind = kinds[entry % kinds.size()];
            scenario.add(kind.first, kind.second, kTableCommandBytes, 1, entry);
        }
        scenario.write(device_closer_->mesh(), device);
        scenario.write_table(device);
        launches.push_back(scenario_launch(scenario, scenario.table_args()));
        expected_codes.push_back(scenario.table_success_code());
        expected_commands.push_back(kFairnessTableEntries);
        used_trids.push_back(scenario.table_used_trids());
    }

    const std::string label = fmt::format("Fairness with CCE{} at depth 16", heavy_cce);
    log_info(tt::LogTest, "MM43 {}", label);
    const std::optional<std::vector<std::vector<uint32_t>>> statuses = run_kernels(device, launches);
    ASSERT_TRUE(statuses.has_value()) << label << ": no status after " << kHostDeadline.count() << " s"
                                      << kFreshSessionNote;
    for (size_t source = 0; source < scenarios.size(); ++source) {
        const TridScenario& scenario = *scenarios[source];
        const std::vector<uint32_t>& status = (*statuses)[source];
        const std::string source_label = fmt::format("{}: {} on CCE{}", label, scenario.name(), scenario.cce_index());
        expect_status(
            device,
            scenario,
            status,
            expected_codes[source],
            expected_commands[source],
            used_trids[source],
            source_label);
        scenario.expect_landed(device_closer_->mesh(), device, source_label);
        RecordProperty(
            scenario.name() + "_stalls",
            fmt::format(
                "vc_space {} ack {} data {} cycles",
                status[abi::kStatusVcSpaceCycles],
                status[abi::kStatusAckCycles],
                status[abi::kStatusDataCycles]));
    }
}

inline constexpr uint32_t kSoakRounds = 4;
inline constexpr uint32_t kSoakMaxLines = 8;

// kSoakRounds tables of 64 seeded random entries: random target and direction, 64 B to 512 B, random trid, so trids
// wrap and repeat. Every round checks completion count, unique completion ranks, no trid left pending and the data,
// and records its seed, per-trid use and stalls. Remote targets (2x) are written only, host-verified.
TEST_P(MimirTridIdma, Soak) {
    skip_unless_mimir_local_package();
    if (has_remote_mimir()) {
        MIMIR_REQUIRE_REMOTE_PREFLIGHT();
    }
    IDevice* device = open_device();
    const bool remote = has_remote_mimir();
    if (remote) {
        ASSERT_TRUE(mimir_remote_remap_result().empty()) << mimir_remote_remap_result();
    }
    std::vector<std::pair<Target, Target>> kinds = {
        {Target::OwnSource, Target::OwnDestination},
        {Target::OwnSource, Target::SiblingTl1},
        {Target::SiblingTl1, Target::OwnDestination},
        {Target::OwnSource, Target::Gddr0},
        {Target::Gddr0, Target::OwnDestination},
        {Target::OwnSource, Target::Gddr1},
        {Target::Gddr1, Target::OwnDestination}};
    if (remote) {
        kinds.push_back({Target::OwnSource, Target::RemoteTl1});
        kinds.push_back({Target::OwnSource, Target::RemoteGddr});
    }
    const metal_SocDescriptor& soc = soc_desc_of(device);
    for (uint32_t round = 0; round < kSoakRounds; ++round) {
        TridScenario scenario(soc, fmt::format("Soak_round{}", round), GetParam(), kTridHart, 0xA0 + round);
        std::mt19937 choices(scenario.seed());
        std::vector<uint32_t> trid_uses(abi::kNumTrids, 0);
        for (uint32_t entry = 0; entry < abi::kMaxElements; ++entry) {
            std::pair<Target, Target> kind = kinds[choices() % kinds.size()];
            const uint32_t len_bytes = abi::kLineBytes * (1 + choices() % kSoakMaxLines);
            const uint32_t trid = choices() % abi::kNumTrids;
            if (!scenario.fits(kind.first, len_bytes) || !scenario.fits(kind.second, len_bytes)) {
                kind = {Target::OwnSource, Target::Gddr0};
            }
            scenario.add(kind.first, kind.second, len_bytes, 1, trid);
            trid_uses[trid]++;
        }
        const std::optional<std::vector<uint32_t>> status =
            run_scenario_table(device_closer_->mesh(), device, scenario, fmt::format("Soak round {}", round));
        if (!status.has_value()) {
            return;
        }
        record_leg(scenario.name(), "trid_uses", fmt::format("{}", fmt::join(trid_uses, ",")));
        record_leg(
            scenario.name(),
            "stalls",
            fmt::format(
                "vc_space {} ack {} data {} cycles, pending high water {}",
                (*status)[abi::kStatusVcSpaceCycles],
                (*status)[abi::kStatusAckCycles],
                (*status)[abi::kStatusDataCycles],
                (*status)[abi::kStatusPendingHighWater]));
    }
}

INSTANTIATE_TEST_SUITE_P(MM43, MimirTridIdma, ::testing::ValuesIn(mimir_contract::cce_indices()), cce_param_name);

// One remote route from a source CCE to the CCE at its location on the other Mimir.
std::string remote_route_name(const ::testing::TestParamInfo<uint32_t>& info) {
    if (mimir_contract::selected_soc_desc_path() == nullptr) {
        return cce_param_name(info);
    }
    return fmt::format("Cce{}ToCce{}", info.param, next_mimir_cce(info.param));
}

// 2x only, host-verified: 16 commands on trids 0-15 from the source CCE to the remote target over D2D (backend 3), and
// back into its own TL1 (backend 5), where the kernel checks the XOR of what landed and never loads a remote address.
class MimirTridRemote : public ::testing::TestWithParam<uint32_t>, protected MimirTridTest {
protected:
    void SetUp() override {
        skip_unless_programmable_dram_package();
        skip_unless_mimir_local_package();
        if (const std::string watcher_reason = mimir_watcher_skip_reason(); !watcher_reason.empty()) {
            GTEST_SKIP() << watcher_reason;
        }
        if (topology_from(configured_soc_desc()).num_mimirs < 2) {
            GTEST_SKIP() << "Remote trid cases need a 2x descriptor (two Mimirs).";
        }
        MIMIR_REQUIRE_REMOTE_PREFLIGHT();
    }

    void TearDown() override { close_device(); }

    void run_remote(Target target, const std::string& name, uint32_t tag) {
        IDevice* device = open_device();
        ASSERT_TRUE(mimir_remote_remap_result().empty()) << mimir_remote_remap_result();
        run_unique_id_legs(
            device_closer_->mesh(),
            device,
            GetParam(),
            {{Target::OwnSource, target}, {target, Target::OwnDestination}},
            name,
            tag);
    }
};

TEST_P(MimirTridRemote, RemoteTl1) { run_remote(Target::RemoteTl1, "RemoteTl1", 0xB0 + GetParam() * 2); }

TEST_P(MimirTridRemote, RemoteGddr) { run_remote(Target::RemoteGddr, "RemoteGddr", 0xC0 + GetParam() * 2); }

INSTANTIATE_TEST_SUITE_P(MM43, MimirTridRemote, ::testing::ValuesIn(mimir_contract::cce_indices()), remote_route_name);

// Hart reset and reload with TT_METAL_FORCE_REINIT=1 (as MM2 MimirFwReinit), quiesced first: no trid counter reads
// pending after the reopen, and every trid 0-15 completes a command again.
class MimirTridReset : public ::testing::TestWithParam<uint32_t>, protected MimirTridTest {
protected:
    void SetUp() override {
        skip_unless_programmable_dram_package();
        if (!MetalContext::instance().rtoptions().get_force_context_reinit()) {
            GTEST_SKIP() << "Set TT_METAL_FORCE_REINIT=1.";
        }
        if (const std::string skip_reason = cce_skip_reason(GetParam()); !skip_reason.empty()) {
            GTEST_SKIP() << skip_reason;
        }
    }

    void TearDown() override { close_device(); }
};

// The first session ends with every trid acked. After the reopen, the observer mode with its issuer flag already
// set reads one trid's pending count on command buffer 0 without issuing; one launch per trid on the same hart.
TEST_P(MimirTridReset, NoStaleIdAfterReinit) {
    const uint32_t cce_index = GetParam();
    {
        IDevice* device = open_device();
        TridScenario scenario(soc_desc_of(device), "Reinit_before", cce_index, kTridHart, 0xD0);
        const size_t batch = scenario.add(
            Target::OwnSource,
            Target::Gddr0,
            kUniqueIdsCommandBytes,
            kUniqueIdsCommands,
            /*first_trid=*/0,
            /*trid_stride=*/1);
        if (!run_scenario_batch(
                 device_closer_->mesh(),
                 device,
                 scenario,
                 batch,
                 fmt::format("NoStaleIdAfterReinit before reopen on CCE{}", cce_index))
                 .has_value()) {
            return;
        }
        close_device();
    }

    IDevice* device = open_device();
    const uint64_t peer_flag_offset = tl1_data(kTridHart) + kPeerFlagOffsetFromData;
    for (uint32_t trid = 0; trid < abi::kNumTrids; ++trid) {
        std::vector<uint32_t> flags(2 * abi::kLineBytes / sizeof(uint32_t), 0);
        flags[0] = 1;
        write_cce_tl1(device, cce_index, peer_flag_offset, flags);
        const std::string label =
            fmt::format("NoStaleIdAfterReinit on CCE{}: pending count of trid {}", cce_index, trid);
        log_info(tt::LogTest, "MM43 {}", label);
        const std::optional<std::vector<std::vector<uint32_t>>> statuses =
            run_on_cce(device, cce_index, {{kTridHart, observer_args(kTridHart, trid, peer_flag_offset)}});
        ASSERT_TRUE(statuses.has_value())
            << label << ": no status after " << kHostDeadline.count() << " s" << kFreshSessionNote;
        const std::vector<uint32_t>& status = statuses->front();
        EXPECT_EQ(status[abi::kStatusCode], abi::kCodeAcked) << label << ": " << status_detail(status);
        EXPECT_EQ(status[abi::kStatusProbeFirst], 0u) << label << ": stale pending count after the reopen";
    }

    TridScenario scenario(soc_desc_of(device), "Reinit_after", cce_index, kTridHart, 0xD1);
    const size_t batch = scenario.add(
        Target::OwnSource,
        Target::Gddr0,
        kUniqueIdsCommandBytes,
        kUniqueIdsCommands,
        /*first_trid=*/0,
        /*trid_stride=*/1);
    run_scenario_batch(
        device_closer_->mesh(),
        device,
        scenario,
        batch,
        fmt::format("NoStaleIdAfterReinit after reopen on CCE{}", cce_index));
}

INSTANTIATE_TEST_SUITE_P(MM43, MimirTridReset, ::testing::ValuesIn(mimir_contract::cce_indices()), cce_param_name);

namespace gcb_abi = cce_gcb_receiver_abi;

inline constexpr const char* kGcbDiscardReceiverKernel =
    "tests/tt_metal/tt_metal/test_kernels/cce/cce_gcb_discard_receiver.cpp";
inline constexpr uint32_t kUint32TileWords = tt::constants::TILE_HW;
inline constexpr uint32_t kUint32TilePageBytes = kUint32TileWords * sizeof(uint32_t);
// The prefetched tensor: one-tile pages through a one-tile FIFO, so every page waits for the receiver's credit.
inline constexpr uint32_t kPrefetchTiles = 16;
// rdcycle budgets of the prefetcher, the receiver and the trid stream; ~400 s of host polling each (MM-7 S5). The
// stream's covers its first round, the queue and the whole prefetch.
inline constexpr uint64_t kConcurrentBudgetCycles = 1ull << 25;
// The trid stream's host deadline: one shared budget, plus launch.
inline constexpr std::chrono::seconds kStreamHostDeadline{500};
// How long the host waits for the stream's first round, and then for the whole prefetch.
inline constexpr std::chrono::seconds kOverlapWindow{120};
inline constexpr uint32_t kStreamCommands = 8;
inline constexpr uint32_t kStreamCommandBytes = 1024;

struct ConcurrentRoute {
    uint32_t sender_cce = 0;
    uint32_t receiver_cce = 0;
    uint32_t stream_cce = 0;
};

std::string concurrent_route_name(const ::testing::TestParamInfo<ConcurrentRoute>& info) {
    return fmt::format(
        "Sender{}Receiver{}Stream{}", info.param.sender_cce, info.param.receiver_cce, info.param.stream_cce);
}

// The DRAM bank whose prefetcher sender is CCE `cce_index`, if any.
std::optional<uint32_t> bank_with_sender_cce(distributed::MeshDevice& mesh, uint32_t cce_index) {
    IDevice* device = mesh.get_devices().front();
    const metal_SocDescriptor& soc = soc_desc_of(device);
    for (uint32_t bank = 0; bank < soc.get_num_dram_views(); ++bank) {
        const CoreCoord sender_logical = mesh.impl().dram_sender_logical_cores(device, bank).front();
        if (soc.get_physical_dram_core_from_logical(sender_logical) == translated_dram_core(soc, cce_index)) {
            return bank;
        }
    }
    return std::nullopt;
}

// A UINT32 TILE tensor of kPrefetchTiles tiles on DRAM bank `bank` alone, seeded.
MeshTensor write_prefetch_tensor(distributed::MeshDevice& mesh, uint32_t bank) {
    const CoreCoord bank_core = soc_desc_of(mesh.get_devices().front()).get_logical_core_for_dram_view(bank);
    const CoreRangeSet bank_cores(CoreRange(bank_core, bank_core));
    const Shape shape{32 * kPrefetchTiles, 32};
    const MemoryConfig memory_config(BufferType::DRAM, NdShardSpec{shape, bank_cores, ShardOrientation::ROW_MAJOR});
    const TensorSpec spec(shape, TensorLayout(DataType::UINT32, Layout::TILE, memory_config));
    std::vector<uint32_t> words =
        make_pattern(MimirPattern::Prng, kPrefetchTiles * kUint32TileWords, bank, mimir_test_seed());
    DistributedHostBuffer host_buffer = DistributedHostBuffer::create(mesh.shape());
    const distributed::MeshCoordinate first_coord = *distributed::MeshCoordinateRange(mesh.shape()).begin();
    host_buffer.emplace_shard(first_coord, [&words]() { return HostBuffer(std::vector<uint32_t>(words)); });
    const HostTensor host_tensor = host_tensor_from_buffer_with_topology(
        std::move(host_buffer), spec, TensorTopology::create_sharded_tensor_topology(mesh.shape()));
    distributed::MeshCommandQueue& command_queue = mesh.mesh_command_queue();
    MeshTensor tensor = command_queue.enqueue_write_tensor(host_tensor);
    command_queue.finish();
    return tensor;
}

// A Tensor prefetcher session on one sender bank. The destructor stops through the no-throw path; call stop() to see a
// sender that ended Failed.
class ConcurrentPrefetcherSession {
public:
    ConcurrentPrefetcherSession(distributed::MeshDevice& mesh, uint32_t sender_bank) : mesh_(mesh) {
        experimental::StartTensorPrefetcher(
            mesh_,
            experimental::TensorPrefetcherConfig{
                .quasar = experimental::QuasarTensorPrefetcherConfig{
                    .dram_sender_banks = {sender_bank}, .wait_budget_cycles = kConcurrentBudgetCycles}});
        active_ = true;
    }
    ConcurrentPrefetcherSession(const ConcurrentPrefetcherSession&) = delete;
    ConcurrentPrefetcherSession& operator=(const ConcurrentPrefetcherSession&) = delete;
    ~ConcurrentPrefetcherSession() {
        if (active_) {
            mesh_.impl().tensor_prefetcher(&mesh_).stop_without_throwing();
        }
    }

    void stop() {
        active_ = false;
        experimental::StopTensorPrefetcher(mesh_);
    }

private:
    distributed::MeshDevice& mesh_;
    bool active_ = false;
};

// 2x only, no AMOs and no accumulate: the stream CCE runs kModeTridStream rounds into its local GDDR while the
// prefetcher streams a tensor from the sender CCE's bank to a discard receiver on the other Mimir. The stream CCE is
// outside the prefetcher's sender and receiver set, so no ITN write lands in its TL1 while it stores progress
// (MIM-560). The prefetch is queued once the stream completed a round, and the stream must complete another round
// between the sender's first page and its last.
class MimirTridConcurrent : public ::testing::TestWithParam<ConcurrentRoute> {
protected:
    void SetUp() override {
        skip_unless_programmable_dram_package();
        skip_unless_mimir_local_package();
        if (const std::string watcher_reason = mimir_watcher_skip_reason(); !watcher_reason.empty()) {
            GTEST_SKIP() << watcher_reason;
        }
        const ConcurrentRoute& route = GetParam();
        const uint32_t num_cces = topology_from(configured_soc_desc()).num_cces;
        if (num_cces <= std::max({route.sender_cce, route.receiver_cce, route.stream_cce})) {
            GTEST_SKIP() << "WithPrefetcher needs a 2x descriptor; this one exposes " << num_cces << " CCEs.";
        }
        MIMIR_REQUIRE_REMOTE_PREFLIGHT();
    }
};

TEST_P(MimirTridConcurrent, WithPrefetcher) {
    const ConcurrentRoute& route = GetParam();
    std::shared_ptr<distributed::MeshDevice> mesh = distributed::MeshDevice::create_unit_mesh(0);
    ScopedMesh mesh_closer(mesh);
    IDevice* device = mesh->get_devices().front();
    check_tl1_layout();
    const metal_SocDescriptor& soc = soc_desc_of(device);
    record_descriptor_properties(soc);
    ASSERT_TRUE(mimir_remote_remap_result().empty()) << mimir_remote_remap_result();
    ASSERT_TRUE(experimental::IsTensorPrefetcherSupported(*mesh));
    const std::optional<uint32_t> sender_bank = bank_with_sender_cce(*mesh, route.sender_cce);
    ASSERT_TRUE(sender_bank.has_value()) << "no DRAM view's prefetcher sender is CCE " << route.sender_cce;
    const std::string label = fmt::format(
        "WithPrefetcher: sender CCE{} (bank {}) -> receiver CCE{}, trid stream on CCE{}",
        route.sender_cce,
        *sender_bank,
        route.receiver_cce,
        route.stream_cce);
    log_info(tt::LogTest, "MM43 {}", label);

    const MeshTensor tensor = write_prefetch_tensor(*mesh, *sender_bank);
    const experimental::GlobalCircularBuffer gcb = experimental::CreateGlobalCircularBufferForTensorPrefetcher(
        *mesh,
        std::vector<std::pair<uint32_t, std::vector<CoreCoord>>>{
            {*sender_bank, {logical_cce_core(soc, route.receiver_cce)}}},
        kUint32TilePageBytes);
    const std::shared_ptr<DriscL1Allocation> receiver_block =
        mesh->impl().drisc_l1_arena().allocate_receiver(gcb_abi::kBlockBytes, gcb_abi::kLineBytes);
    std::vector<uint32_t> block_words(gcb_abi::kBlockBytes / sizeof(uint32_t), 0);
    std::fill(
        block_words.begin() + gcb_abi::kStatusLineOffset / sizeof(uint32_t),
        block_words.begin() + gcb_abi::kHoldingLineOffset / sizeof(uint32_t),
        gcb_abi::kStatusPoison);
    write_cce_tl1(device, route.receiver_cce, receiver_block->addr(), block_words);
    std::vector<uint32_t> receiver_args(gcb_abi::schedule_arg(1, 0), 0);
    receiver_args[gcb_abi::kArgGcbAddr] = static_cast<uint32_t>(gcb.config_address());
    receiver_args[gcb_abi::kArgBlockAddr] = static_cast<uint32_t>(receiver_block->addr());
    receiver_args[gcb_abi::kArgBudgetLo] = static_cast<uint32_t>(kConcurrentBudgetCycles);
    receiver_args[gcb_abi::kArgBudgetHi] = static_cast<uint32_t>(kConcurrentBudgetCycles >> 32);
    receiver_args[gcb_abi::kArgMode] = gcb_abi::kModeDrainUntilStopped;
    receiver_args[gcb_abi::kArgNumRequests] = 1;
    receiver_args[gcb_abi::schedule_arg(0, gcb_abi::kRequestPageBytes)] = kUint32TilePageBytes;
    receiver_args[gcb_abi::schedule_arg(0, gcb_abi::kRequestNumPages)] = kPrefetchTiles;
    prefetcher::CceReceiverProgram receivers(
        device,
        {CceKernelLaunch{
            .cce_index = route.receiver_cce, .kernel_path = kGcbDiscardReceiverKernel, .runtime_args = receiver_args}});

    TridScenario scenario(soc, "WithPrefetcher_stream", route.stream_cce, kTridHart, 0xE0);
    scenario.add(
        Target::OwnSource,
        location_for_cce(route.stream_cce) == 0 ? Target::Gddr0 : Target::Gddr1,
        kStreamCommandBytes,
        kStreamCommands,
        /*first_trid=*/0,
        /*trid_stride=*/1);
    scenario.write(*mesh, device);
    const uint64_t stop_offset = tl1_data(kTridHart) + kStopOffsetFromData;
    const uint64_t progress_offset = tl1_data(kTridHart) + kProgressOffsetFromData;
    write_cce_tl1(
        device, route.stream_cce, stop_offset, std::vector<uint32_t>(2 * abi::kLineBytes / sizeof(uint32_t), 0));
    std::vector<uint32_t> stream_args = scenario.batch_args(0);
    stream_args[abi::kArgMode] = abi::kModeTridStream;
    stream_args[abi::kArgPollBudgetLo] = static_cast<uint32_t>(kConcurrentBudgetCycles);
    stream_args[abi::kArgPollBudgetHi] = static_cast<uint32_t>(kConcurrentBudgetCycles >> 32);
    stream_args[abi::kArgStopOffset] = static_cast<uint32_t>(stop_offset);
    stream_args[abi::kArgProgressOffset] = static_cast<uint32_t>(progress_offset);
    std::optional<ConcurrentPrefetcherSession> session;
    session.emplace(*mesh, *sender_bank);
    // Launched after the prefetcher started, so the stream's budget is not spent while it starts.
    const std::vector<CceKernelLaunch> stream_launch = {scenario_launch(scenario, stream_args)};
    write_cce_tl1(
        device,
        route.stream_cce,
        tl1_done_block(kTridHart),
        std::vector<uint32_t>(abi::kStatusWords, abi::kStatusPoison));
    Program stream_program = CreateProgram();
    add_cce_kernels(device, stream_program, stream_launch);
    launch_on_idle_cces(device, stream_program, stream_launch);

    const uint64_t pages_sent_address = experimental::pages_sent_drisc_l1_base(gcb);
    const auto pages_sent = [device, &route, pages_sent_address] {
        return read_cce_tl1(device, route.sender_cce, pages_sent_address, sizeof(uint32_t)).front();
    };
    const auto stream_rounds = [device, &route, progress_offset] {
        return read_cce_tl1(device, route.stream_cce, progress_offset, sizeof(uint32_t)).front();
    };

    // Reads with no sleep, so the emulated clock keeps running.
    const std::chrono::steady_clock::time_point first_round_end = std::chrono::steady_clock::now() + kOverlapWindow;
    uint32_t rounds_at_queue = stream_rounds();
    while (rounds_at_queue == 0 && std::chrono::steady_clock::now() < first_round_end) {
        rounds_at_queue = stream_rounds();
    }
    experimental::QueueTensorPrefetcherRequest(
        *mesh,
        gcb,
        std::nullopt,
        {experimental::TensorPrefetcherInput{.tensor = tensor, .block_count = kPrefetchTiles}});
    // pages_sent counts kDramReceiverCreditUnitBytes units.
    const uint32_t prefetch_units = kPrefetchTiles * kUint32TilePageBytes / kDramReceiverCreditUnitBytes;
    const std::chrono::steady_clock::time_point prefetch_end = std::chrono::steady_clock::now() + kOverlapWindow;
    uint32_t units_sent = pages_sent();
    while (units_sent == 0 && std::chrono::steady_clock::now() < prefetch_end) {
        units_sent = pages_sent();
    }
    const uint32_t rounds_at_first_page = stream_rounds();
    while (units_sent < prefetch_units && std::chrono::steady_clock::now() < prefetch_end) {
        units_sent = pages_sent();
    }
    const uint32_t rounds_at_last_page = units_sent >= prefetch_units ? stream_rounds() : rounds_at_first_page;
    const std::string overlap = fmt::format(
        "stream rounds {} at the queue, {} at the first page, {} at the last; pages_sent {} of {} units",
        rounds_at_queue,
        rounds_at_first_page,
        rounds_at_last_page,
        units_sent,
        prefetch_units);
    log_info(tt::LogTest, "MM43 {}: {}", label, overlap);
    RecordProperty("overlap", overlap);
    EXPECT_GT(rounds_at_last_page, rounds_at_first_page)
        << label << ": no stream round completed while the prefetch ran inside " << kOverlapWindow.count()
        << " s: " << overlap;

    write_cce_tl1(device, route.stream_cce, stop_offset, {abi::kStreamStop});
    const bool stream_done = wait_for_status_magics(device, stream_launch, {abi::kDoneMagic}, kStreamHostDeadline);
    ASSERT_TRUE(stream_done) << label << ": trid stream: no status after " << kStreamHostDeadline.count() << " s"
                             << kFreshSessionNote;
    slow_dispatch::WaitProgramDone(*device, stream_program);
    const std::vector<uint32_t> stream_status =
        read_cce_tl1(device, route.stream_cce, tl1_done_block(kTridHart), abi::kStatusWords * sizeof(uint32_t));
    expect_status(
        device,
        scenario,
        stream_status,
        abi::kCodeLanded,
        stream_status[abi::kStatusProbeFirst] * kStreamCommands,
        scenario.transfer(0).used_trids(),
        label + " trid stream");
    EXPECT_GE(stream_status[abi::kStatusProbeFirst], 1u) << label << ": the trid stream completed no round";
    RecordProperty("stream_rounds", static_cast<int>(stream_status[abi::kStatusProbeFirst]));
    scenario.expect_landed(*mesh, device, label + " trid stream");

    try {
        session->stop();
    } catch (const std::exception& error) {
        ADD_FAILURE() << label << ": StopTensorPrefetcher threw: " << error.what();
    }
    session.reset();
    receivers.stop_and_wait();
    const std::vector<uint32_t> receiver_status = read_cce_tl1(
        device,
        route.receiver_cce,
        receiver_block->addr() + gcb_abi::kStatusLineOffset,
        gcb_abi::kStatusWords * sizeof(uint32_t));
    EXPECT_EQ(receiver_status[gcb_abi::kStatusCode], gcb_abi::kCodeStopped) << label << ": receiver";
    EXPECT_EQ(receiver_status[gcb_abi::kStatusPagesDone], kPrefetchTiles) << label << ": receiver pages";
}

INSTANTIATE_TEST_SUITE_P(
    MM43,
    MimirTridConcurrent,
    ::testing::Values(ConcurrentRoute{.sender_cce = 0, .receiver_cce = 2, .stream_cce = 1}),
    concurrent_route_name);

}  // namespace tt::tt_metal::mimir_test::trid
