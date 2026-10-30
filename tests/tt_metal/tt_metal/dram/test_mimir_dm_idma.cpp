// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MM-8 port of quasar_idma: iDMA copies from a CCE through command buffer 0, one cce_idma_copy launch per leg, each
// with a host wall-clock deadline. Every leg must ack and land the right data, with the guards before and after the
// destination intact. Remote operands use the host-verified mode, so the kernel never loads a remote address. The
// Strided1D cases port IDMA_1D_Strided with 64 B elements; the Strided1DSubLine cases use upstream's 8 B elements.
// RemoteTl1KernelVerified is RemoteTl1 in the kernel-verified mode; its first run hung with no status (MM-8 probe).

#include <gtest/gtest.h>

#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "impl/context/metal_context.hpp"
#include "llrt/hal.hpp"
#include "llrt/tt_cluster.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "tt_metal/tt_metal/dram/mimir_access.hpp"
#include "tt_metal/tt_metal/dram/mimir_dm_common.hpp"
#include "tt_metal/tt_metal/dram/mimir_gddr_map.hpp"
#include "tt_metal/tt_metal/dram/mimir_hw_state.hpp"
#include "tt_metal/tt_metal/dram/mimir_patterns.hpp"
#include "tt_metal/tt_metal/dram/mimir_test_common.hpp"
#include "tt_metal/tt_metal/test_kernels/cce/cce_idma_copy_abi.h"

namespace tt::tt_metal::mimir_test::dm_idma {

namespace abi = cce_idma_copy_abi;

inline constexpr const char* kIdmaCopyKernel = "tests/tt_metal/tt_metal/test_kernels/cce/cce_idma_copy.cpp";
inline constexpr uint32_t kCopyHart = 0;
inline constexpr uint32_t kCopyBytes = 1024;
// Three polls at most, plus launch.
inline constexpr std::chrono::seconds kHostDeadline{400};
// Offsets into the hart's data area (tl1_data) for the destination of an own-TL1 copy and of a read leg.
inline constexpr uint64_t kDestinationOffset = 0x8000;
inline constexpr uint64_t kReadbackOffset = 0x10000;
// Above DRAM UNRESERVED in the target view.
inline constexpr uint64_t kGddrDataOffset = 0x10000;
inline constexpr uint32_t kLeadingGuardTag = 0x800;
// Strided cases: line-sized elements, every other line of the source.
inline constexpr uint32_t kStridedElementBytes = 64;
inline constexpr uint32_t kStridedElements = 10;
inline constexpr uint32_t kStridedSourceStrideBytes = 128;
// Sub-line strided cases: upstream IDMA_1D_Strided's every other 8 B element.
inline constexpr uint32_t kSubLineElementBytes = 8;
inline constexpr uint32_t kSubLineSourceStrideBytes = 16;

enum class IdmaCase : uint32_t {
    OwnTl1,
    MaxPacket,
    SiblingTl1,
    LocalGddr,
    LocalGddr1,
    RemoteTl1,
    RemoteGddr,
    RemoteTl1KernelVerified,
    Strided1D,
    Strided1DLocalGddr,
    Strided1DSubLine,
    Strided1DSubLineLocalGddr,
};

std::string idma_case_label(IdmaCase idma_case) {
    switch (idma_case) {
        case IdmaCase::OwnTl1: return "OwnTl1";
        case IdmaCase::MaxPacket: return "MaxPacket";
        case IdmaCase::SiblingTl1: return "SiblingTl1";
        case IdmaCase::LocalGddr: return "LocalGddr";
        case IdmaCase::LocalGddr1: return "LocalGddr1";
        case IdmaCase::RemoteTl1: return "RemoteTl1";
        case IdmaCase::RemoteGddr: return "RemoteGddr";
        case IdmaCase::RemoteTl1KernelVerified: return "RemoteTl1KernelVerified";
        case IdmaCase::Strided1D: return "Strided1D";
        case IdmaCase::Strided1DLocalGddr: return "Strided1DLocalGddr";
        case IdmaCase::Strided1DSubLine: return "Strided1DSubLine";
        case IdmaCase::Strided1DSubLineLocalGddr: return "Strided1DSubLineLocalGddr";
    }
    return "Unknown";
}

std::string idma_case_name(const ::testing::TestParamInfo<IdmaCase>& info) { return idma_case_label(info.param); }

std::vector<IdmaCase> all_idma_cases() {
    return {
        IdmaCase::OwnTl1,
        IdmaCase::MaxPacket,
        IdmaCase::SiblingTl1,
        IdmaCase::LocalGddr,
        IdmaCase::LocalGddr1,
        IdmaCase::RemoteTl1,
        IdmaCase::RemoteGddr,
        IdmaCase::Strided1D,
        IdmaCase::Strided1DLocalGddr,
        IdmaCase::Strided1DSubLine,
        IdmaCase::Strided1DSubLineLocalGddr,
        IdmaCase::RemoteTl1KernelVerified};
}

bool is_remote_case(IdmaCase idma_case) {
    return idma_case == IdmaCase::RemoteTl1 || idma_case == IdmaCase::RemoteGddr ||
           idma_case == IdmaCase::RemoteTl1KernelVerified;
}

// Remote cases run from CCE 0's sibling to the first CCE on the next Mimir (CCE 1 to CCE 2 with two CCEs per Mimir).
uint32_t remote_source_cce() { return sibling_cce(0); }
uint32_t remote_destination_cce() { return next_mimir_cce(0); }

std::string idma_skip_reason(IdmaCase idma_case) {
    if (const std::string watcher_reason = mimir_watcher_skip_reason(); !watcher_reason.empty()) {
        return watcher_reason;
    }
    const MimirTopology topology = topology_from(configured_soc_desc());
    const uint32_t cces_per_mimir = locations_per_channel(configured_soc_desc());
    if (idma_case == IdmaCase::SiblingTl1 && cces_per_mimir < 2) {
        return fmt::format("Descriptor has {} CCE per Mimir; the sibling case needs two.", cces_per_mimir);
    }
    if (idma_case == IdmaCase::LocalGddr1 && dram_views_for_channel(configured_soc_desc(), 0).size() < 2) {
        return "Mimir 0 has one DRAM view; LocalGddr1 needs a second.";
    }
    if (is_remote_case(idma_case) && topology.num_mimirs < 2) {
        return "Remote iDMA cases need a 2x descriptor (two Mimirs).";
    }
    return "";
}

// Where a leg's source or destination lives. Tl1 kinds name the owning CCE and a TL1 offset; Gddr names a DRAM view
// and an offset in it.
enum class EndpointKind { OwnTl1, SramSpa, Gddr };

struct Endpoint {
    EndpointKind kind = EndpointKind::OwnTl1;
    uint32_t cce_index = 0;
    uint32_t view = 0;
    uint64_t offset = 0;
};

Endpoint own_tl1_endpoint(uint32_t cce_index, uint64_t tl1_offset) {
    return {.kind = EndpointKind::OwnTl1, .cce_index = cce_index, .offset = tl1_offset};
}

Endpoint sram_spa_endpoint(uint32_t cce_index, uint64_t tl1_offset) {
    return {.kind = EndpointKind::SramSpa, .cce_index = cce_index, .offset = tl1_offset};
}

Endpoint gddr_endpoint(uint32_t view, uint64_t view_offset) {
    return {.kind = EndpointKind::Gddr, .view = view, .offset = view_offset};
}

// The package address the iDMA command carries for this endpoint, from the issuing CCE.
uint64_t command_address(const metal_SocDescriptor& soc, const CceSpaRoutes& issuer_routes, const Endpoint& endpoint) {
    if (endpoint.kind == EndpointKind::Gddr) {
        return mimir_contract::cce_gddr_address(
            issuer_routes, mimir_of_view(soc, endpoint.view), chiplet_offset_of(soc, endpoint.view, endpoint.offset));
    }
    return mimir_contract::cce_sram_address(issuer_routes, endpoint.cce_index, endpoint.offset);
}

std::string endpoint_label(
    const metal_SocDescriptor& soc, const CceSpaRoutes& issuer_routes, const Endpoint& endpoint) {
    switch (endpoint.kind) {
        case EndpointKind::OwnTl1:
            return fmt::format(
                "own TL1 0x{:x} (CCE{}, SPA 0x{:x})",
                endpoint.offset,
                endpoint.cce_index,
                command_address(soc, issuer_routes, endpoint));
        case EndpointKind::SramSpa:
            return fmt::format(
                "CCE{} TL1 0x{:x} (SPA 0x{:x})",
                endpoint.cce_index,
                endpoint.offset,
                command_address(soc, issuer_routes, endpoint));
        case EndpointKind::Gddr:
            return fmt::format(
                "GDDR view {} (M{}) +0x{:x} (chiplet 0x{:x}, SPA 0x{:x})",
                endpoint.view,
                mimir_of_view(soc, endpoint.view),
                endpoint.offset,
                chiplet_offset_of(soc, endpoint.view, endpoint.offset),
                command_address(soc, issuer_routes, endpoint));
    }
    return "unknown";
}

struct IdmaLeg {
    std::string name;
    uint32_t cce_index = 0;
    uint32_t mode = abi::kModeIdmaCopy;
    uint32_t backend_vc = abi::kBackendLocalSram;
    Endpoint source;
    Endpoint destination;
    uint32_t len_bytes = kCopyBytes;
    uint32_t num_elements = 1;
    uint32_t source_stride_bytes = 0;
};

// Element i of a strided leg is len_bytes at source + i * source_stride_bytes; a plain leg has one element.
uint32_t source_bytes_of(const IdmaLeg& leg) {
    return (leg.num_elements - 1) * leg.source_stride_bytes + leg.len_bytes;
}

std::vector<uint32_t> gathered_elements(const IdmaLeg& leg, const std::vector<uint32_t>& source_words) {
    const uint32_t element_words = leg.len_bytes / sizeof(uint32_t);
    std::vector<uint32_t> words;
    for (uint32_t element = 0; element < leg.num_elements; ++element) {
        const uint32_t first_word = element * leg.source_stride_bytes / sizeof(uint32_t);
        words.insert(words.end(), source_words.begin() + first_word, source_words.begin() + first_word + element_words);
    }
    return words;
}

Endpoint leading_guard_of(const Endpoint& destination) {
    Endpoint guard = destination;
    guard.offset -= kGuardBytes;
    return guard;
}

// MIM-407: GDDR0 is the low 4 GiB tile of a Mimir, GDDR1 the high one.
uint32_t gddr_backend_for(const metal_SocDescriptor& soc, uint32_t view, uint64_t view_offset) {
    return mimir_channel_location(chiplet_offset_of(soc, view, view_offset)).tile == 0 ? abi::kBackendGddr0
                                                                                       : abi::kBackendGddr1;
}

// A write leg (own TL1 -> target, MIM-407 backend for the target) and a read leg (target -> own TL1, the remote-source
// backend). Each leg seeds its own source, so the read leg does not depend on the write leg.
std::vector<IdmaLeg> write_and_read_legs(
    uint32_t cce_index, const Endpoint& target, uint32_t write_backend, uint32_t mode) {
    return {
        IdmaLeg{
            .name = "write",
            .cce_index = cce_index,
            .mode = mode,
            .backend_vc = write_backend,
            .source = own_tl1_endpoint(cce_index, tl1_data(kCopyHart)),
            .destination = target},
        IdmaLeg{
            .name = "read",
            .cce_index = cce_index,
            .mode = mode,
            .backend_vc = abi::kBackendFromRemote,
            .source = target,
            .destination = own_tl1_endpoint(cce_index, tl1_data(kCopyHart) + kReadbackOffset)},
    };
}

std::vector<IdmaLeg> legs_for(IdmaCase idma_case, const metal_SocDescriptor& soc) {
    const uint64_t dram_unreserved = MetalContext::instance().hal().get_dev_addr(HalDramMemAddrType::UNRESERVED);
    const Endpoint own_source = own_tl1_endpoint(0, tl1_data(kCopyHart));
    const Endpoint own_destination = own_tl1_endpoint(0, tl1_data(kCopyHart) + kDestinationOffset);
    switch (idma_case) {
        case IdmaCase::OwnTl1: return {IdmaLeg{.name = "copy", .source = own_source, .destination = own_destination}};
        case IdmaCase::MaxPacket:
            return {IdmaLeg{
                .name = "copy",
                .source = own_source,
                .destination = own_destination,
                .len_bytes = abi::kMaxBytesInPacket}};
        case IdmaCase::SiblingTl1:
            return write_and_read_legs(
                0,
                sram_spa_endpoint(sibling_cce(0), tl1_remote_landing(0, kCopyHart)),
                abi::kBackendRemoteSram,
                abi::kModeIdmaCopy);
        case IdmaCase::LocalGddr:
        case IdmaCase::LocalGddr1: {
            const uint32_t view = local_dram_view_for_cce(soc, idma_case == IdmaCase::LocalGddr ? 0 : sibling_cce(0));
            const uint64_t view_offset = dram_unreserved + kGddrDataOffset;
            const uint32_t backend = gddr_backend_for(soc, view, view_offset);
            const uint32_t expected_backend =
                idma_case == IdmaCase::LocalGddr ? abi::kBackendGddr0 : abi::kBackendGddr1;
            if (backend != expected_backend) {
                TT_THROW("GDDR view {} +0x{:x} uses backend {}, not {}", view, view_offset, backend, expected_backend);
            }
            return write_and_read_legs(0, gddr_endpoint(view, view_offset), backend, abi::kModeIdmaCopy);
        }
        case IdmaCase::RemoteTl1:
        case IdmaCase::RemoteTl1KernelVerified:
            return write_and_read_legs(
                remote_source_cce(),
                sram_spa_endpoint(remote_destination_cce(), tl1_remote_landing(remote_source_cce(), kCopyHart)),
                abi::kBackendD2d,
                idma_case == IdmaCase::RemoteTl1 ? abi::kModeIdmaCopyHostVerified : abi::kModeIdmaCopy);
        case IdmaCase::RemoteGddr:
            return write_and_read_legs(
                remote_source_cce(),
                gddr_endpoint(
                    base_dram_view_for_mimir(soc, mimir_index_for_cce(remote_destination_cce())),
                    dram_unreserved + kGddrDataOffset),
                abi::kBackendD2d,
                abi::kModeIdmaCopyHostVerified);
        case IdmaCase::Strided1D:
        case IdmaCase::Strided1DLocalGddr:
        case IdmaCase::Strided1DSubLine:
        case IdmaCase::Strided1DSubLineLocalGddr: {
            const bool sub_line =
                idma_case == IdmaCase::Strided1DSubLine || idma_case == IdmaCase::Strided1DSubLineLocalGddr;
            const bool from_gddr =
                idma_case == IdmaCase::Strided1DLocalGddr || idma_case == IdmaCase::Strided1DSubLineLocalGddr;
            return {IdmaLeg{
                .name = "gather",
                .mode = abi::kModeIdmaStrided,
                .backend_vc = abi::kBackendFromRemote,
                .source = from_gddr ? gddr_endpoint(local_dram_view_for_cce(soc, 0), dram_unreserved + kGddrDataOffset)
                                    : own_source,
                .destination = own_destination,
                .len_bytes = sub_line ? kSubLineElementBytes : kStridedElementBytes,
                .num_elements = kStridedElements,
                .source_stride_bytes = sub_line ? kSubLineSourceStrideBytes : kStridedSourceStrideBytes}};
        }
    }
    return {};
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

std::vector<uint32_t> kernel_args(
    const metal_SocDescriptor& soc, const IdmaLeg& leg, uint64_t status_offset, uint32_t expected_xor) {
    const KernelOperand source = kernel_operand(soc, leg.cce_index, leg.source);
    const KernelOperand destination = kernel_operand(soc, leg.cce_index, leg.destination);
    std::vector<uint32_t> args(abi::kNumArgs, 0);
    args[abi::kArgMode] = leg.mode;
    args[abi::kArgBackendVc] = leg.backend_vc;
    args[abi::kArgStatusOffset] = static_cast<uint32_t>(status_offset);
    args[abi::kArgLenBytes] = leg.len_bytes;
    args[abi::kArgPollBudgetLo] = static_cast<uint32_t>(kPollBudgetCycles);
    args[abi::kArgPollBudgetHi] = static_cast<uint32_t>(kPollBudgetCycles >> 32);
    args[abi::kArgSrcKind] = source.kind;
    args[abi::kArgSrcSelector] = source.selector;
    args[abi::kArgSrcOffsetLo] = static_cast<uint32_t>(source.offset);
    args[abi::kArgSrcOffsetHi] = static_cast<uint32_t>(source.offset >> 32);
    args[abi::kArgDstKind] = destination.kind;
    args[abi::kArgDstSelector] = destination.selector;
    args[abi::kArgDstOffsetLo] = static_cast<uint32_t>(destination.offset);
    args[abi::kArgDstOffsetHi] = static_cast<uint32_t>(destination.offset >> 32);
    args[abi::kArgExpectedXor] = expected_xor;
    args[abi::kArgNumElements] = leg.num_elements;
    args[abi::kArgSourceStrideBytes] = leg.source_stride_bytes;
    return args;
}

std::string code_name(uint32_t code) {
    switch (code) {
        case abi::kCodeLanded: return "landed";
        case abi::kCodeAcked: return "acked";
        case abi::kCodeVcSpaceTimeout: return "vc_space_timeout";
        case abi::kCodeAckTimeout: return "ack_timeout";
        case abi::kCodeDataTimeout: return "data_timeout";
        case abi::kCodeBadArgs: return "bad_args";
    }
    return fmt::format("unknown_code_0x{:x}", code);
}

std::string poll_name(uint32_t poll) {
    switch (poll) {
        case abi::kPollNone: return "none";
        case abi::kPollVcSpace: return "vc_space";
        case abi::kPollAck: return "ack";
        case abi::kPollData: return "data";
    }
    return fmt::format("unknown_poll_{}", poll);
}

std::string mode_name(uint32_t mode) {
    switch (mode) {
        case abi::kModeIdmaCopy: return "IdmaCopy";
        case abi::kModeIdmaCopyHostVerified: return "IdmaCopyHostVerified";
        case abi::kModeIdmaStrided: return "IdmaStrided";
    }
    return fmt::format("unknown_mode_{}", mode);
}

std::string status_detail(const std::vector<uint32_t>& status) {
    const uint32_t flags = status[abi::kStatusFlags];
    return fmt::format(
        "kernel code {} flags [vc_space {} acked {} landed {}], {} commands issued, last poll {} {}, cycles vc_space "
        "{} "
        "ack {} data {}, kernel mismatches {} (first word {} observed 0x{:08x} expected 0x{:08x})",
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
        status[abi::kStatusMismatchedWords],
        status[abi::kStatusFirstMismatchWord],
        status[abi::kStatusObserved],
        status[abi::kStatusExpected]);
}

// Runs one leg and checks it: the kernel reports success, the host reads the expected data (re-reading the first
// mismatch after kRecheckDelay before calling it wrong), and the guards before and after it are intact. Returns false
// when the host deadline passed, after which nothing else may run in this session (MIM-601).
bool run_leg(distributed::MeshDevice& unit_mesh, IDevice* device, const IdmaLeg& leg, uint32_t tag) {
    const metal_SocDescriptor& soc = soc_desc_of(device);
    const uint32_t num_words = leg.num_elements * leg.len_bytes / sizeof(uint32_t);
    const uint64_t status_offset = tl1_done_block(kCopyHart);
    TT_FATAL(status_offset % abi::kLineBytes == 0, "status line 0x{:x} is not 64 B aligned", status_offset);

    const uint32_t seed = mimir_test_seed() ^ (tag * 0x9E3779B9u);
    const std::vector<uint32_t> source_words =
        make_pattern(MimirPattern::Prng, source_bytes_of(leg) / sizeof(uint32_t), leg.source.offset, seed);
    const std::vector<uint32_t> expected = gathered_elements(leg, source_words);
    const std::vector<uint32_t> guard = make_guard(tag);
    const std::vector<uint32_t> leading_guard = make_guard(kLeadingGuardTag + tag);
    std::vector<uint32_t> poison_and_guard = make_seed_pattern(tag, num_words);
    poison_and_guard.insert(poison_and_guard.end(), guard.begin(), guard.end());
    write_endpoint(unit_mesh, device, leg.source, source_words);
    write_endpoint(unit_mesh, device, leg.destination, poison_and_guard);
    write_endpoint(unit_mesh, device, leading_guard_of(leg.destination), leading_guard);
    write_cce_tl1(device, leg.cce_index, status_offset, std::vector<uint32_t>(abi::kStatusWords, abi::kStatusPoison));

    const CoreCoord translated_core = translated_dram_core(soc, leg.cce_index);
    const CceSpaRoutes issuer_routes = mimir_contract::read_cce_spa_routes(device, leg.cce_index);
    const std::string label = fmt::format(
        "{} on CCE{} hart {}: {} backend VC {} {} x {} B (source stride {} B), {} -> {}",
        leg.name,
        leg.cce_index,
        kCopyHart,
        mode_name(leg.mode),
        leg.backend_vc,
        leg.num_elements,
        leg.len_bytes,
        leg.source_stride_bytes,
        endpoint_label(soc, issuer_routes, leg.source),
        endpoint_label(soc, issuer_routes, leg.destination));
    record_leg(leg.name, "source", endpoint_label(soc, issuer_routes, leg.source));
    record_leg(leg.name, "destination", endpoint_label(soc, issuer_routes, leg.destination));
    record_leg(
        leg.name,
        "core",
        fmt::format(
            "CCE{} hart {} translated ({}, {})", leg.cce_index, kCopyHart, translated_core.x, translated_core.y));
    record_leg(
        leg.name,
        "length",
        leg.num_elements == 1
            ? std::to_string(leg.len_bytes)
            : fmt::format(
                  "{} elements of {} B, source stride {} B", leg.num_elements, leg.len_bytes, leg.source_stride_bytes));
    record_leg(
        leg.name,
        "alignment",
        fmt::format(
            "source {} destination {}",
            command_address(soc, issuer_routes, leg.source) % abi::kLineBytes,
            command_address(soc, issuer_routes, leg.destination) % abi::kLineBytes));
    record_leg(
        leg.name, "mode", fmt::format("{} backend VC {} trid {}", mode_name(leg.mode), leg.backend_vc, abi::kTrid));
    record_leg(leg.name, "seed", fmt::format("0x{:08x}", seed));
    record_leg(leg.name, "kernel", kIdmaCopyKernel);

    const CceKernelLaunch launch{
        .cce_index = leg.cce_index,
        .kernel_path = kIdmaCopyKernel,
        .runtime_args = kernel_args(soc, leg, status_offset, xor_of(expected)),
        .hart = kCopyHart};
    Program program = CreateProgram();
    const std::optional<std::vector<std::vector<uint32_t>>> statuses =
        run_cce_kernels_with_deadline(device, program, {launch}, {abi::kDoneMagic}, abi::kStatusWords, kHostDeadline);
    if (!statuses.has_value()) {
        const std::string timeout = fmt::format("no status after {} s (host deadline)", kHostDeadline.count());
        record_leg(leg.name, "status", timeout);
        ADD_FAILURE() << label << ": " << timeout << kFreshSessionNote;
        return false;
    }

    const std::vector<uint32_t>& status = statuses->front();
    const std::string detail = status_detail(status);
    record_leg(leg.name, "status", detail);
    log_info(tt::LogTest, "MM8 iDMA {}: {}", label, detail);
    const uint32_t success_code = leg.mode == abi::kModeIdmaCopyHostVerified ? abi::kCodeAcked : abi::kCodeLanded;
    EXPECT_EQ(status[abi::kStatusCode], success_code) << label << ": " << detail;
    EXPECT_EQ(status[abi::kStatusElementsIssued], leg.num_elements) << label << ": " << detail;
    EXPECT_TRUE(compare_with_recheck(
        "iDMA " + label,
        leg.name,
        expected,
        [&unit_mesh, device, &leg](uint64_t byte_offset, uint32_t bytes) {
            return read_endpoint(unit_mesh, device, leg.destination, byte_offset, bytes);
        }))
        << detail;
    EXPECT_EQ(read_endpoint(unit_mesh, device, leg.destination, num_words * sizeof(uint32_t), abi::kLineBytes), guard)
        << label << ": guard after the destination was overwritten";
    EXPECT_EQ(read_endpoint(unit_mesh, device, leading_guard_of(leg.destination), 0, kGuardBytes), leading_guard)
        << label << ": guard before the destination was overwritten";
    return true;
}

class MimirDmIdma : public ::testing::TestWithParam<IdmaCase> {};

TEST_P(MimirDmIdma, Copy) {
    skip_unless_programmable_dram_package();
    skip_unless_mimir_local_package();
    const IdmaCase idma_case = GetParam();
    if (const std::string skip_reason = idma_skip_reason(idma_case); !skip_reason.empty()) {
        GTEST_SKIP() << skip_reason;
    }
    if (is_remote_case(idma_case)) {
        MIMIR_REQUIRE_REMOTE_PREFLIGHT();
    }
    ScopedDevice device_closer;
    IDevice* device = device_closer.device();
    check_tl1_layout();
    if (is_remote_case(idma_case)) {
        ASSERT_TRUE(mimir_remote_remap_result().empty()) << mimir_remote_remap_result();
    }

    const metal_SocDescriptor& soc = soc_desc_of(device);
    record_descriptor_properties(soc);

    const std::vector<IdmaLeg> legs = legs_for(idma_case, soc);
    for (uint32_t leg_index = 0; leg_index < legs.size(); ++leg_index) {
        if (!run_leg(
                device_closer.mesh(), device, legs[leg_index], static_cast<uint32_t>(idma_case) * 8 + leg_index + 1)) {
            return;
        }
    }
}

INSTANTIATE_TEST_SUITE_P(MM8, MimirDmIdma, ::testing::ValuesIn(all_idma_cases()), idma_case_name);

}  // namespace tt::tt_metal::mimir_test::dm_idma
