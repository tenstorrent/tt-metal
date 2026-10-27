// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/buffers/tensor_prefetcher_manager.hpp"

#include "distributed/mesh_device_impl.hpp"
#include "distributed/mesh_device_view_impl.hpp"
#include "distributed/mesh_command_queue_base.hpp"
#include "impl/allocator/allocator.hpp"
#include "impl/buffers/dram_sender_topology.hpp"
#include "impl/buffers/drisc_l1_arena.hpp"
#include "impl/buffers/h2d_socket_internal.hpp"
#include "impl/buffers/prefetcher_pipe_dram_sender_internal.hpp"
#include "impl/buffers/tensor_prefetcher_cce_abi.hpp"
#include "impl/buffers/tensor_prefetcher_transport.hpp"
#include "impl/dataflow_buffer/prefetcher_pipe.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <string_view>
#include <optional>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_distribution_spec.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/experimental/sockets/h2d_socket.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tensor/mesh_tensor.hpp>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>
#include <tt_stl/reflection.hpp>

#include "impl/context/metal_context.hpp"
#include "impl/context/metal_env_impl.hpp"
#include "impl/kernels/kernel.hpp"  // DramConfig + CreateKernel(DramConfig)
#include "impl/program/program_impl.hpp"
#include "impl/program/slow_dispatch.hpp"
#include "llrt/hal.hpp"
#include "llrt/metal_soc_descriptor.hpp"
#include "llrt/tt_cluster.hpp"
#include "tt_metal/hw/inc/hostdev/socket.h"  // receiver_socket_md (for L1 layout sizing)

namespace tt::tt_metal::distributed {

namespace {

static_assert(
    kNumOpSignalSlots == experimental::kTensorPrefetcherNumSignals,
    "Every op signal needs its slot in the signal slot table");

// Blackhole MPFE round-robin weights occupy the 3-bit field defined by
// GDDR_MC_MPFE_CFG_ROUNDROBIN_WEIGHT_MASK in the DRISC-only register map.
constexpr uint32_t kMaxMpfeWeight = 7;
constexpr std::string_view kBenchmarkMpfeEnvPrefix = "TT_METAL_BENCHMARK_TENSOR_PREFETCHER_";
constexpr std::string_view kBenchmarkTimerName = "TENSOR_PREFETCHER_MPFE_ACTIVE_LIFETIME";
// How long the host worker pauses when no TL1 request ring has room.
constexpr std::chrono::milliseconds kRequestRingRetryInterval{1};
// How long the host worker waits on a TL1 request ring that takes no page when TT_METAL_OPERATION_TIMEOUT_SECONDS is
// unset.
constexpr std::chrono::minutes kDefaultRequestRingNoProgressLimit{5};
// Alignment of a CCE sender's stage ring size: its thirds then start on 64 B lines.
constexpr uint32_t kCceStageRingAlignBytes = 6 * kCceSenderLineBytes;

// How request pages reach a sender.
enum class RequestTransportKind : uint8_t {
    H2DSocket,
    Tl1Ring,
};

// What differs between the Blackhole DRISC prefetcher and the Quasar CCE prefetcher.
struct PrefetcherArchTraits {
    const char* kernel_path;
    uint32_t senders_per_bank;
    RequestTransportKind transport;
    // GDDR subchannels per bank the sender roles assume; unset skips the check.
    std::optional<uint32_t> gddr_subchannels_per_bank;
    // Whether QuasarTensorPrefetcherConfig::stage_ring_bytes caps the stage ring.
    bool stage_ring_capped;
    // What the stage ring's size is aligned down to; unset is 6 x the L1 alignment.
    std::optional<uint32_t> stage_ring_alignment_bytes;
    // Receiver-contiguous batched delivery only: K-row-major tensors and streaming are rejected. The CCE sender
    // implements only the batched receiver-contiguous round; the Blackhole K-row-major and streaming paths are not
    // ported to it.
    bool receiver_contiguous_batched_only;
    bool supports_prefetcher_pipes;
};

constexpr PrefetcherArchTraits kBlackholeTraits{
    .kernel_path = "tt_metal/impl/buffers/kernels/tensor_prefetcher.cpp",
    .senders_per_bank = 2,
    .transport = RequestTransportKind::H2DSocket,
    .gddr_subchannels_per_bank = 3,
    .stage_ring_capped = false,
    .stage_ring_alignment_bytes = std::nullopt,
    .receiver_contiguous_batched_only = false,
    .supports_prefetcher_pipes = true,
};

constexpr PrefetcherArchTraits kQuasarTraits{
    .kernel_path = "tt_metal/impl/buffers/kernels/tensor_prefetcher_cce.cpp",
    .senders_per_bank = 1,
    .transport = RequestTransportKind::Tl1Ring,
    .gddr_subchannels_per_bank = std::nullopt,
    .stage_ring_capped = true,
    .stage_ring_alignment_bytes = kCceStageRingAlignBytes,
    .receiver_contiguous_batched_only = true,
    .supports_prefetcher_pipes = false,
};

const PrefetcherArchTraits& prefetcher_traits_for(tt::ARCH arch) {
    if (arch == tt::ARCH::BLACKHOLE) {
        return kBlackholeTraits;
    }
    if (arch == tt::ARCH::QUASAR) {
        return kQuasarTraits;
    }
    TT_THROW("Tensor prefetcher is not available on architecture {}", arch);
}

std::optional<uint32_t> stage_ring_cap(
    const PrefetcherArchTraits& traits, const experimental::QuasarTensorPrefetcherConfig& quasar_config) {
    if (!traits.stage_ring_capped) {
        return std::nullopt;
    }
    return quasar_config.stage_ring_bytes;
}

inline uint32_t align_up(uint32_t a, uint32_t align) { return (a + align - 1) & ~(align - 1); }

// Per-core carve of the DRISC L1 kernel working region (above the arena's sender-state zone):
//   Blackhole: [signal slots | socket_config | socket_data FIFO | selector scratch | stage ring].
//   Quasar:    [CCE sender control block (tensor_prefetcher_cce_abi.hpp) | stage ring].
// Each pipe's NOC XY table, credit counters and write cursors live in its sender config page in that
// zone, owned by the pipe rather than the prefetcher kernel. Each DRAM core hosts exactly one
// H2DSocket recv (for its own sender), so the layout is uniform across all DRAM cores. The
// MeshBuffer L1 allocator can't reach DRAM-core L1, so the addresses go to the H2DSocket bypass
// ctor directly (see h2d_socket_internal.hpp).
struct KernelRegionCarve {
    uint32_t signal_slots_l1_addr = 0;
    // One L1 alignment per slot rather than packed -- see signal_slot_stride_.
    uint32_t signal_slot_stride = 0;
    uint32_t socket_config_l1_addr = 0;
    uint32_t socket_data_l1_addr = 0;
    uint32_t selector_scratch_l1_addr = 0;
    uint32_t cce_control_base = 0;
    uint32_t stage_ring_base = 0;
    uint32_t stage_ring_size = 0;
    // Empty when the carve fits the region; otherwise why it does not.
    std::string failure_reason;
};

KernelRegionCarve carve_kernel_region(
    const Hal& hal,
    const DriscL1Arena& arena,
    const PrefetcherArchTraits& traits,
    const experimental::QuasarTensorPrefetcherConfig& quasar_config) {
    KernelRegionCarve carve;
    const std::optional<uint32_t> stage_ring_cap_bytes = stage_ring_cap(traits, quasar_config);
    const bool request_ring = traits.transport == RequestTransportKind::Tl1Ring;
    const char* const request_buffers = request_ring ? "request ring" : "socket buffers";
    const uint32_t kernel_region_size = arena.kernel_working_region_size();
    const uint32_t l1_alignment = hal.get_alignment(HalMemType::L1);
    if (kernel_region_size < 2 * l1_alignment) {
        carve.failure_reason = fmt::format(
            "DRISC L1 kernel region ({} B) too small for the prefetcher ping-pong stage", kernel_region_size);
        return carve;
    }
    const uint32_t kernel_region_base = static_cast<uint32_t>(arena.kernel_working_region_base());
    if (request_ring) {
        // The CQ fence slots are the control block's first lines, so a CQ fence addresses them the same way
        // on both architectures. A CCE sender serves no op signal and no group selector, so neither is carved.
        carve.cce_control_base = align_up(kernel_region_base, kCceSenderLineBytes);
        carve.signal_slots_l1_addr = carve.cce_control_base + kCceCqSignalSlotsOffset;
        carve.signal_slot_stride = kCceCqSignalSlotStrideBytes;
        carve.stage_ring_base = carve.cce_control_base + kCceSenderControlBytes;
    } else {
        const uint32_t pcie_alignment = hal.get_alignment(HalMemType::HOST);
        const uint32_t page_size = align_up(kRequestPageBytes, pcie_alignment);
        const uint32_t socket_fifo_size =
            align_up(page_size * TensorPrefetcherManager::kSocketFifoPages, pcie_alignment);
        const uint32_t socket_config_bytes = align_up(sizeof(receiver_socket_md), pcie_alignment);
        const uint32_t socket_data_bytes = socket_fifo_size + pcie_alignment;
        // Signal slots at the front of the region: a uint32 counter per command queue, written by the
        // dispatcher for WaitForCqOnTensorPrefetcher, then one per op signal, incremented by device
        // kernels; the kernel's WAIT_SIGNAL handler polls them.
        carve.signal_slot_stride = l1_alignment;
        const uint32_t signal_slots_bytes = kNumSignalSlots * carve.signal_slot_stride;
        carve.signal_slots_l1_addr = align_up(kernel_region_base, l1_alignment);
        carve.socket_config_l1_addr = align_up(carve.signal_slots_l1_addr + signal_slots_bytes, pcie_alignment);
        carve.socket_data_l1_addr = align_up(carve.socket_config_l1_addr + socket_config_bytes, pcie_alignment);
        // The selector scratch takes a NoC read from DRAM, which lands at the same offset within a
        // DRAM-alignment word as its source, and selector pages start DRAM-aligned.
        const uint32_t dram_alignment = hal.get_alignment(HalMemType::DRAM);
        carve.selector_scratch_l1_addr = align_up(carve.socket_data_l1_addr + socket_data_bytes, dram_alignment);
        carve.stage_ring_base =
            align_up(carve.selector_scratch_l1_addr + kTensorPrefetcherSelectorScratchBytes, l1_alignment);
    }
    const uint32_t kernel_region_end = kernel_region_base + kernel_region_size;
    if (carve.stage_ring_base >= kernel_region_end) {
        carve.failure_reason = fmt::format(
            "DRISC L1 kernel region ({} B) too small for {} + stage ring", kernel_region_size, request_buffers);
        return carve;
    }
    carve.stage_ring_size = kernel_region_end - carve.stage_ring_base;
    // The cap applies before the align-down, so the capped ring is still aligned.
    if (stage_ring_cap_bytes.has_value()) {
        carve.stage_ring_size = std::min(carve.stage_ring_size, *stage_ring_cap_bytes);
    }
    // Same align-down as TensorPrefetcherManager::set_stage_ring_size, so the size checked here is the
    // size the manager ends up with. A CCE sender aligns further, to a multiple of that, so its thirds
    // start on 64 B lines.
    const uint32_t kRingSizeAlign = traits.stage_ring_alignment_bytes.value_or(6u * l1_alignment);
    carve.stage_ring_size = carve.stage_ring_size / kRingSizeAlign * kRingSizeAlign;
    // After aligning down, make sure the ring is still big enough for at least
    // one minimal sub-chunk per slot. Catches accidental shrink-to-zero if the
    // socket carve-out ever grows past the L1 region.
    if (carve.stage_ring_size < kRingSizeAlign) {
        if (stage_ring_cap_bytes.has_value()) {
            carve.failure_reason = fmt::format(
                "DRISC L1 stage ring shrank to {} B after alignment — the {} B stage ring cap or the {} left too "
                "little of the {} B kernel region",
                carve.stage_ring_size,
                *stage_ring_cap_bytes,
                request_buffers,
                kernel_region_size);
        } else {
            carve.failure_reason = fmt::format(
                "DRISC L1 stage ring shrank to {} B after alignment — {} consumed too much of the {} B kernel region",
                carve.stage_ring_size,
                request_buffers,
                kernel_region_size);
        }
    }
    return carve;
}

// The GO_MSG signal of the DRAM core `sender_logical` on `device`.
uint8_t read_go_message_signal(const MetalContext& metal, const IDevice* device, const CoreCoord& sender_logical) {
    const Hal& hal = metal.hal();
    dev_msgs::go_msg_t go_message =
        hal.get_dev_msgs_factory(HalProgrammableCoreType::DRAM).create<dev_msgs::go_msg_t>();
    const uint64_t go_message_address =
        hal.get_dev_noc_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::GO_MSG) & ~uint64_t{0x3};
    const CoreCoord sender_virtual = device->virtual_core_from_logical_core(sender_logical, CoreType::DRAM);
    metal.get_cluster().read_core(
        go_message.data(), go_message.size(), tt_cxy_pair(device->id(), sender_virtual), go_message_address);
    return go_message.view().signal();
}

// Whether the CCE sender's status says it is still in its request loop, so a page may yet reach it. A status that
// cannot be read is unknown and counts as running. Called on the host worker thread, so a failed read is not thrown;
// Tl1RingTransport::target_exited, which reads the same status earlier in the pass, logs it.
bool cce_sender_running(
    const MeshDevice& mesh_device, IDevice* device, const CoreCoord& sender_logical, uint32_t control_base) {
    try {
        const uint32_t state = read_cce_sender_status(mesh_device, device, sender_logical, control_base).state;
        return state == static_cast<uint32_t>(CceSenderState::Running) ||
               state == static_cast<uint32_t>(CceSenderState::WaitingCq) ||
               state == static_cast<uint32_t>(CceSenderState::Draining);
    } catch (const std::exception&) {
        return true;
    }
}

uint32_t get_mpfe_port(const metal_SocDescriptor& soc_desc, const CoreCoord& sender_logical_core) {
    const CoreCoord sender_physical = soc_desc.get_physical_dram_core_from_logical(sender_logical_core);
    const tt::umd::CoreCoord sender_subchannel = soc_desc.translate_coord_to(
        tt::umd::CoreCoord(sender_physical.x, sender_physical.y, tt::CoreType::DRAM, tt::CoordSystem::TRANSLATED),
        tt::CoordSystem::LOGICAL);
    // MPFE P0 is tied off. Blackhole DRAM tiles D0..D2 (logical subchannels 0..2)
    // enter through the register-map ports P1..P3.
    return sender_subchannel.y + 1;
}

// Look up a Tensor Prefetcher benchmark environment variable by suffix.
const char* benchmark_mpfe_env(std::string_view suffix, std::string& name) {
    name = std::string(kBenchmarkMpfeEnvPrefix) + std::string(suffix);
    return std::getenv(name.c_str());
}

// Benchmark overrides and their telemetry are opt-in so stale individual weight
// variables cannot silently alter a normal caller's explicit configuration.
bool benchmark_mpfe_enabled() {
    std::string name;
    const char* value = benchmark_mpfe_env("ENABLE", name);
    if (value == nullptr) {
        return false;
    }
    TT_FATAL((value[0] == '0' || value[0] == '1') && value[1] == '\0', "{} must be 0 or 1, got '{}'", name, value);
    return value[0] == '1';
}

// Use the benchmark environment override when set, validating it as an MPFE weight.
uint32_t benchmark_mpfe_weight(std::string_view suffix, uint32_t fallback, bool benchmark_enabled) {
    if (!benchmark_enabled) {
        return fallback;
    }
    std::string name;
    const char* value = benchmark_mpfe_env(suffix, name);
    if (value == nullptr) {
        return fallback;
    }
    const uint32_t parsed = static_cast<uint32_t>(value[0] - '0');
    TT_FATAL(
        value[0] >= '0' && parsed <= kMaxMpfeWeight && value[1] == '\0',
        "{} must be one digit in [0, {}], got '{}'",
        name,
        kMaxMpfeWeight,
        value);
    if (parsed != fallback) {
        log_info(tt::LogMetal, "{}={} overrides configured value {}", name, parsed, fallback);
    }
    return parsed;
}

// Use the benchmark environment override when set, validating it as a boolean.
bool benchmark_mpfe_bool(std::string_view suffix, bool fallback, bool benchmark_enabled) {
    if (!benchmark_enabled) {
        return fallback;
    }
    std::string name;
    const char* value = benchmark_mpfe_env(suffix, name);
    if (value == nullptr) {
        return fallback;
    }
    TT_FATAL((value[0] == '0' || value[0] == '1') && value[1] == '\0', "{} must be 0 or 1, got '{}'", name, value);
    const bool parsed = value[0] == '1';
    if (parsed != fallback) {
        log_info(tt::LogMetal, "{}={} overrides configured value {}", name, parsed ? 1 : 0, fallback ? 1 : 0);
    }
    return parsed;
}

// Largest `page` (multiple of tile_size, <= max_page_size) such that num_tiles*tile_size
// is divisible by page. Returns (page_size, num_pages). Identical to the existing
// implementation; carried over verbatim from the pre-queueable manager.
std::pair<uint32_t, uint32_t> pick_page_size(uint32_t max_page_size, uint32_t num_tiles, uint32_t tile_size) {
    const uint64_t total = static_cast<uint64_t>(num_tiles) * tile_size;
    uint32_t page = (max_page_size / tile_size) * tile_size;
    while (page >= tile_size && total % page != 0) {
        page -= tile_size;
    }
    TT_FATAL(
        page >= tile_size,
        "pick_page_size could not find a page that divides num_tiles*tile_size={} (tile={})",
        static_cast<uint64_t>(total),
        tile_size);
    return {page, static_cast<uint32_t>(total / page)};
}

// Detection is implicit per the design doc — see detect_layout_mode().
using LayoutMode = TensorPrefetcherLayoutMode;

static_assert(static_cast<uint32_t>(LayoutMode::ReceiverContiguous) == kCceReceiverContiguousLayoutMode);

// Detection keys on how the weight was allocated, NOT the shard count: receiver-contiguous
// weights are created with an NdShardSpec (num_shards == ring_size), K-row-major weights use a
// legacy (WIDTH_SHARDED) shard spec. Counting shards is ambiguous when total_receivers ==
// num_banks (one receiver per bank, num_shards == num_banks == total_receivers): the count tie
// would route a recv-contig tensor down the K-row path, where compute_tensor_layout_krow_major
// calls shard_spec() and TT_FATALs because the buffer only has an NdShardSpec.
//
// That tie -- one shard per bank, feeding that bank's single receiver -- is read as
// receiver-contiguous whichever spec the weight carries. There the two layouts read identically
// (n_per_recv == n_per_bank, one slab at offset 0), so the bytes cannot settle it, and
// receiver-contiguous is the loop with dynamic multi-block batching. What it gives up is K-row-major's
// tolerance of a K the block count does not divide: K-row-major *ceils* K into block_count blocks
// (the short last block over-reads inside the bank, which is that receiver's own data), where
// receiver-contiguous requires exact division, because in general the over-read would cross into the
// next receiver's slab. (K-row-major's third fit rung, chunking a too-wide K-row across receivers, is
// not among the differences: its M is capped at num_receivers, so at one receiver per bank neither
// layout can chunk N.)
LayoutMode detect_layout_mode(const MeshTensor& t, const Buffer& buf, uint32_t total_receivers) {
    const auto& bds_opt = buf.buffer_distribution_spec();
    const bool one_shard_per_receiver = t.nd_shard_spec().has_value() && bds_opt.has_value() &&
                                        static_cast<uint32_t>(bds_opt->num_shards()) == total_receivers;

    // Legacy ShardSpec path: one wide shard per bank by construction. Some WIDTH_SHARDED
    // tensors also carry an NdShardSpec-like descriptor through BDS, so prefer the explicit
    // legacy shard spec before classifying a tensor as receiver-contiguous.
    //
    // The exception is the tie above: with one receiver per bank, a bank's whole shard IS that
    // receiver's slab, and ttnn hands back a legacy ShardSpec for such an NdShardSpec anyway
    // (num_shards == num_dram_banks canonicalizes to WIDTH_SHARDED), so keying on the spec kind
    // cannot separate the two layouts here -- and does not need to, since they name the same bytes.
    if (buf.has_shard_spec()) {
        return one_shard_per_receiver ? LayoutMode::ReceiverContiguous : LayoutMode::KRowMajor;
    }

    if (t.nd_shard_spec().has_value()) {
        TT_FATAL(
            one_shard_per_receiver,
            "Receiver-contiguous Tensor prefetcher weight must have num_shards == total_receivers "
            "(ring_size = {}); got {} shards.",
            total_receivers,
            bds_opt.has_value() ? static_cast<uint32_t>(bds_opt->num_shards()) : 0u);
        return LayoutMode::ReceiverContiguous;
    }
    return LayoutMode::KRowMajor;
}

// Validate a streaming (receiver-contiguous) weight and return the shard distribution strategy that
// governs how the host maps a receiver's (bank, bank-local slab index) to a global receiver position
// when slicing the rotation table. This is the consumer's concept (a ring matmul calls it a "ring
// position") — the transport stays order-agnostic and only numbers slabs bank-locally (see
// recv_index_bases_per_sender). TT_FATALs on a non-recv-contig (no BDS) tensor or an unsupported
// distribution strategy; only the two strategies below reach the packing loop:
//   ROUND_ROBIN_1D (strided):    global = bank + slab_idx * num_banks
//   CONTIGUOUS_1D  (contiguous): global = bank * receivers_per_bank + slab_idx
ShardDistributionStrategy shard_strategy_for_streaming_tensor(const MeshTensor& t, uint32_t tensor_idx) {
    const auto* ref_buffer = t.mesh_buffer().get_reference_buffer();
    const auto& bds_opt = ref_buffer->buffer_distribution_spec();
    TT_FATAL(
        bds_opt.has_value(),
        "Streaming Tensor prefetcher tensor {} must be a receiver-contiguous (nd-sharded) weight, but it has no "
        "buffer distribution spec (it looks K-row-major / legacy-sharded).",
        tensor_idx);
    const auto strategy = bds_opt->shard_distribution_strategy();
    TT_FATAL(
        strategy == ShardDistributionStrategy::ROUND_ROBIN_1D || strategy == ShardDistributionStrategy::CONTIGUOUS_1D,
        "Streaming Tensor prefetcher tensor {} uses an unsupported shard distribution strategy ({}); only "
        "ROUND_ROBIN_1D (strided) and CONTIGUOUS_1D (contiguous) receiver-contiguous weights are supported.",
        tensor_idx,
        static_cast<int>(strategy));
    return strategy;
}

// Address-independent per-tensor geometry for the K-row-major DRAM layout — see
// TensorPrefetcherTensorLayout in impl/buffers/tensor_prefetcher_request.hpp for
// the field-by-field documentation, and tt_metal/impl/buffers/prefetcher_matmul_design.md
// §6 for the fit ladder. The tensor's bank-local address is carried separately in the
// per-tensor TensorPrefetcherEntry, so identical-geometry tensors share one layout.
TensorPrefetcherTensorLayout compute_tensor_layout_krow_major(
    const MeshTensor& t, uint32_t block_count, uint32_t num_receivers, uint32_t ring_half, ContextId context_id) {
    const auto* ref_buffer = t.mesh_buffer().get_reference_buffer();
    const auto shard_shape = ref_buffer->shard_spec().shape();
    const uint32_t tile_bytes = tt::tile_size(datatype_to_dataformat_converter(t.dtype()));
    TT_FATAL(
        shard_shape[0] % tt::constants::TILE_HEIGHT == 0 && shard_shape[1] % tt::constants::TILE_WIDTH == 0,
        "Tensor prefetcher requires tile-aligned shards; got shard_shape=({}, {}), tile=({}, {})",
        shard_shape[0],
        shard_shape[1],
        tt::constants::TILE_HEIGHT,
        tt::constants::TILE_WIDTH);
    const uint32_t k_tiles_raw = shard_shape[0] / tt::constants::TILE_HEIGHT;
    const uint32_t n_per_bank = shard_shape[1] / tt::constants::TILE_WIDTH;
    TT_FATAL(block_count > 0, "Tensor prefetcher block_count must be > 0");
    TT_FATAL(
        n_per_bank % num_receivers == 0,
        "n_per_bank ({}) must divide num_receivers ({}); reduce N_per_bank or grow num_receivers",
        n_per_bank,
        num_receivers);
    // block_count K-blocks per tensor; the consuming matmul does wait_front(block_count) per
    // layer.
    const uint32_t k_block_w_tiles = (k_tiles_raw + block_count - 1) / block_count;
    const uint32_t n_per_recv = n_per_bank / num_receivers;
    const uint32_t row_bytes = n_per_bank * tile_bytes;
    const uint32_t block_bytes = k_block_w_tiles * row_bytes;
    const uint32_t noc_max_burst = MetalContext::instance(context_id).hal().get_noc_max_burst_size_bytes();
    const auto [coalesced_page_size, coalesced_num_pages] = pick_page_size(noc_max_burst, n_per_recv, tile_bytes);
    TT_FATAL(
        coalesced_page_size <= noc_max_burst,
        "Tensor prefetcher coalesced page size ({} B) exceeds the one-packet NoC write limit ({} B).",
        coalesced_page_size,
        noc_max_burst);

    uint32_t rows_per_sub = 0;
    uint32_t M = 1;
    if (block_bytes <= ring_half) {
        rows_per_sub = k_block_w_tiles;
        M = 1;
    } else if (row_bytes <= ring_half) {
        rows_per_sub = 1;
        for (uint32_t d = k_block_w_tiles; d >= 1; --d) {
            if (k_block_w_tiles % d == 0 && static_cast<uint64_t>(d) * row_bytes <= ring_half) {
                rows_per_sub = d;
                break;
            }
        }
        M = 1;
    } else {
        rows_per_sub = 1;
        bool picked = false;
        for (uint32_t m = 1; m <= num_receivers; ++m) {
            if (num_receivers % m == 0 && (row_bytes / m) <= ring_half) {
                M = m;
                picked = true;
                break;
            }
        }
        TT_FATAL(
            picked,
            "Tensor prefetcher cannot fit one K-row of tensor (k_tiles={}, n_per_bank={}, tile_bytes={}, "
            "row_bytes={} B) into DRISC L1 stage half ({} B) even with M=num_receivers ({}).",
            k_tiles_raw,
            n_per_bank,
            tile_bytes,
            row_bytes,
            ring_half,
            num_receivers);
    }
    const uint32_t num_sub = k_block_w_tiles / rows_per_sub;
    const uint32_t sub_chunk_bytes = rows_per_sub * (n_per_bank / M) * tile_bytes;
    TT_FATAL(
        sub_chunk_bytes <= ring_half, "Internal: chunk size {} B exceeds ring_half {} B", sub_chunk_bytes, ring_half);

    TensorPrefetcherTensorLayout g;
    g.num_sub = num_sub;
    g.M = M;
    g.rows_per_sub = rows_per_sub;
    g.coalesced_page_size = coalesced_page_size;
    g.coalesced_num_pages = coalesced_num_pages;
    g.sub_chunk_bytes = sub_chunk_bytes;
    g.sub_stride_bytes = rows_per_sub * row_bytes;
    g.block_stride_bytes = k_block_w_tiles * row_bytes;
    g.page_bytes_per_recv = k_block_w_tiles * coalesced_num_pages * coalesced_page_size;
    g.block_count = block_count;
    return g;
}

// Two layouts are interchangeable iff every geometry field matches. The struct is a
// packed POD of uint32_t fields, so a byte compare is exact (no padding).
bool layout_equal(const TensorPrefetcherTensorLayout& a, const TensorPrefetcherTensorLayout& b) {
    return std::memcmp(&a, &b, sizeof(TensorPrefetcherTensorLayout)) == 0;
}

// Receiver-contiguous layout: BDS holds `ring_size` shards round-robin across
// `num_senders` banks, so each bank stacks `num_receivers` shards each of shape
// (K_tiles, n_per_recv_tiles). Per (receiver, block) the source bytes are a
// single contiguous DRAM region. The kernel dynamically batches B blocks per
// per-receiver visit (B clamped by free space and fifo wrap at runtime); the
// manager just computes the static ceiling target_per_visit_pages. The tensor's
// bank-local address is carried separately in the per-tensor
// TensorPrefetcherEntry, so identical-geometry tensors share one layout.
TensorPrefetcherTensorLayout compute_tensor_layout_recv_contig(
    const MeshTensor& t, uint32_t block_count, uint32_t stage_third, ContextId context_id) {
    // Read the original (non-squeezed) NdShardSpec from the MemoryConfig — the
    // BDS internally collapses adjacent matching dims, so its
    // shard_shape_in_pages() can come back rank-1 even though the caller
    // passed a 2D shape.
    const auto& nd_opt = t.nd_shard_spec();
    TT_FATAL(
        nd_opt.has_value(),
        "Receiver-contiguous Tensor prefetcher tensor must be allocated with an "
        "NdShardSpec (e.g. ttnn.MemoryConfig(BufferType.DRAM, NdShardSpec(...))).");
    // A rank-3 shard (G, K_elems, n_per_recv_elems) is a grouped weight -- an expert stack -- whose
    // every receiver slab holds all G groups. Within a shard pages run (group, k, n) row-major, so each
    // group's (K, n_per_recv) block is contiguous and the slab is G of them back to back.
    const auto& shard_shape = nd_opt->shard_shape;
    TT_FATAL(
        shard_shape.rank() == 2 || shard_shape.rank() == 3,
        "Receiver-contiguous NdShardSpec shard shape must be (K_elems, n_per_recv_elems) or, for a grouped "
        "weight, (num_groups, K_elems, n_per_recv_elems); got {}",
        shard_shape);
    const bool grouped = shard_shape.rank() == 3;
    const uint32_t num_groups = grouped ? shard_shape[0] : 1;
    if (grouped) {
        const auto& logical_shape = t.logical_shape();
        TT_FATAL(
            logical_shape.rank() >= 3 && logical_shape[-3] == num_groups,
            "Grouped receiver-contiguous weight: the shard must hold the whole group axis, so its leading dim ({}) "
            "must equal the tensor's dim -3, but the tensor shape is {}",
            num_groups,
            logical_shape);
    }
    const uint32_t k_elems = shard_shape[-2];
    const uint32_t n_per_recv_elems = shard_shape[-1];
    TT_FATAL(
        k_elems % tt::constants::TILE_HEIGHT == 0 && n_per_recv_elems % tt::constants::TILE_WIDTH == 0,
        "Receiver-contiguous shard shape ({}, {}) must be tile-aligned (TILE={}, {})",
        k_elems,
        n_per_recv_elems,
        tt::constants::TILE_HEIGHT,
        tt::constants::TILE_WIDTH);
    const uint32_t k_tiles_raw = k_elems / tt::constants::TILE_HEIGHT;
    const uint32_t n_per_recv = n_per_recv_elems / tt::constants::TILE_WIDTH;
    TT_FATAL(
        k_tiles_raw > 0 && n_per_recv > 0,
        "Receiver-contiguous shard shape has zero dim: ({}, {})",
        k_tiles_raw,
        n_per_recv);

    const uint32_t tile_bytes = tt::tile_size(datatype_to_dataformat_converter(t.dtype()));
    TT_FATAL(block_count > 0, "Tensor prefetcher block_count must be > 0");
    // K must divide evenly into block_count K-blocks. A ceil here would make the kernel push
    // block_count * ceil(K_tiles/block_count) K-rows per receiver — more than the slab
    // (recv_stride = K_tiles * n_per_recv * tile_bytes) holds — so the last block over-reads
    // into the next receiver's slab (or past the buffer for the last slab). For a matmul-fed
    // weight, compute block_count via tensor_prefetcher_block_count_for_matmul_1d(), which
    // also pins block_count == ring_size.
    TT_FATAL(
        k_tiles_raw % block_count == 0,
        "Receiver-contiguous: weight K ({} tiles) must be divisible by block_count ({}); remainder {}. "
        "block_count must equal the matmul ring_size.",
        k_tiles_raw,
        block_count,
        k_tiles_raw % block_count);
    const uint32_t k_block_w_tiles = k_tiles_raw / block_count;

    const uint32_t noc_max_burst = MetalContext::instance(context_id).hal().get_noc_max_burst_size_bytes();
    const auto [coalesced_page_size, coalesced_num_pages] = pick_page_size(noc_max_burst, n_per_recv, tile_bytes);
    TT_FATAL(
        coalesced_page_size <= noc_max_burst,
        "Tensor prefetcher coalesced page size ({} B) exceeds the one-packet NoC write limit ({} B).",
        coalesced_page_size,
        noc_max_burst);

    const uint32_t bytes_per_recv_per_block = k_block_w_tiles * n_per_recv * tile_bytes;
    const uint32_t group_bytes = k_tiles_raw * n_per_recv * tile_bytes;
    const uint32_t recv_stride_bytes = num_groups * group_bytes;

    // Fit ladder. Rung 1: full block fits in one stage-third (the kernel uses
    // 3 rotating slots, not 2 halves, so the constraint is tighter than the
    // legacy K-row path). Rung 2: K-sub split for shapes where one block
    // exceeds the stage_third; the kernel walks sub-bands across slots and
    // multi-block batching still works because B blocks of a receiver are
    // contiguous in DRAM.
    uint32_t rows_per_sub = 0;
    uint32_t num_sub = 1;
    if (bytes_per_recv_per_block <= stage_third) {
        rows_per_sub = k_block_w_tiles;
        num_sub = 1;
    } else {
        rows_per_sub = 0;
        for (uint32_t d = k_block_w_tiles; d >= 1; --d) {
            if (k_block_w_tiles % d == 0 && static_cast<uint64_t>(d) * n_per_recv * tile_bytes <= stage_third) {
                rows_per_sub = d;
                break;
            }
        }
        TT_FATAL(
            rows_per_sub >= 1,
            "Receiver-contiguous mode cannot fit a single K-row slice "
            "(n_per_recv={}, tile_bytes={}, stage_third={} B). Spread the weight's N over more receivers to reduce "
            "n_per_recv.",
            n_per_recv,
            tile_bytes,
            stage_third);
        num_sub = k_block_w_tiles / rows_per_sub;
    }

    const uint32_t sub_chunk_bytes = rows_per_sub * n_per_recv * tile_bytes;
    TT_FATAL(
        sub_chunk_bytes <= stage_third,
        "Internal: receiver-contiguous chunk size {} B exceeds stage_third {} B",
        sub_chunk_bytes,
        stage_third);

    // target_per_visit_pages: ~6 stage thirds' worth of blocks per visit. The
    // kernel amortizes one noc_async_write_one_packet_set_state per receiver
    // visit, so making each visit cover many blocks reduces set_state cost.
    // The kernel further clamps by free downstream space, remaining blocks,
    // and fifo-wrap distance, so the static ceiling is a hint, not a contract.
    const uint32_t page_bytes_per_recv = k_block_w_tiles * coalesced_num_pages * coalesced_page_size;
    const uint32_t stage_slot_pages = page_bytes_per_recv > 0 ? stage_third / page_bytes_per_recv : 0;
    constexpr uint32_t kVisitStageSlotMultiplier = 6;
    uint32_t target_per_visit_pages = stage_slot_pages * kVisitStageSlotMultiplier;
    if (target_per_visit_pages == 0) {
        target_per_visit_pages = 1;
    }

    TensorPrefetcherTensorLayout g;
    g.num_sub = num_sub;
    g.M = 1;  // unused in recv-contig (no N-chunking)
    g.rows_per_sub = rows_per_sub;
    g.coalesced_page_size = coalesced_page_size;
    g.coalesced_num_pages = coalesced_num_pages;
    g.sub_chunk_bytes = sub_chunk_bytes;
    g.sub_stride_bytes = rows_per_sub * n_per_recv * tile_bytes;
    g.block_stride_bytes = k_block_w_tiles * n_per_recv * tile_bytes;  // within-slab K-block stride
    g.page_bytes_per_recv = page_bytes_per_recv;
    g.layout_mode = static_cast<uint32_t>(LayoutMode::ReceiverContiguous);
    g.target_per_visit_pages = target_per_visit_pages;
    g.recv_stride_bytes = recv_stride_bytes;
    g.block_count = block_count;
    g.group_stride_bytes = grouped ? group_bytes : 0;
    return g;
}

TensorPrefetcherTensorLayout compute_tensor_layout(
    const MeshTensor& t,
    uint32_t block_count,
    uint32_t num_banks,
    uint32_t receivers_per_bank,
    uint32_t total_receivers,
    uint32_t ring_half,
    uint32_t stage_third,
    ContextId context_id) {
    const auto* ref_buffer = t.mesh_buffer().get_reference_buffer();
    const LayoutMode mode = detect_layout_mode(t, *ref_buffer, total_receivers);
    if (mode == LayoutMode::KRowMajor) {
        // KRowMajor is single-sender-per-bank only, so receivers_per_bank is the bank's
        // full receiver count and the bank receiver counts must be uniform. (Receiver-
        // contiguous derives its geometry from the shard shape and ignores the receiver
        // count entirely, and tolerates non-uniform per-bank receivers.)
        TT_FATAL(
            num_banks > 0 && total_receivers % num_banks == 0,
            "Tensor prefetcher (K-row-major): total receivers ({}) must divide evenly across {} banks. "
            "Non-uniform per-bank receiver counts are only supported by the receiver-contiguous layout.",
            total_receivers,
            num_banks);
        return compute_tensor_layout_krow_major(t, block_count, receivers_per_bank, ring_half, context_id);
    }
    return compute_tensor_layout_recv_contig(t, block_count, stage_third, context_id);
}

// Virtual coords `xy` packed the way the DRISC kernel decodes a NoC endpoint from a request page or
// runtime arg: x in the low 16 bits, y in the high 16 (TensorPrefetcherEntry::selector_noc_xy).
uint32_t pack_virtual_xy(const CoreCoord& xy) {
    return static_cast<uint32_t>(xy.x) | (static_cast<uint32_t>(xy.y) << 16);
}

// Where the sender reads a group selector's mask page: a NoC endpoint packed by pack_virtual_xy, or a DRAM
// bank, and the page's address there.
struct SelectorLocation {
    TensorPrefetcherSelectorLocation location = TENSOR_PREFETCHER_SELECTOR_AT_NOC_XY;
    uint32_t noc_xy = 0;
    uint32_t addr = 0;
};

// A DRISC sender reads the mask on NOC 0 with virtual coordinates, the convention of the pipes' receiver
// table, so a DRAM page names its bank's NOC 0 endpoint -- the one that forwards DRAM reads (see
// drisc_mode.h). A dispatch-engine sender reads DRAM by bank, so a DRAM page names bank 0 instead. An L1
// page names its core's virtual coordinates. The address is mesh-uniform. One page serves every device, so
// the devices must agree on the endpoint, which is resolved per device because DRAM harvesting may move it;
// bank 0 is the same on every device.
SelectorLocation resolve_group_selector(
    const MeshDevice& mesh_device,
    const std::vector<IDevice*>& devices,
    const MeshTensor& mask,
    uint32_t num_groups,
    uint32_t tensor_idx) {
    TT_FATAL(
        num_groups <= kTensorPrefetcherSelectorMaxEntries,
        "Tensor prefetcher input {}: a group selector reads one mask word per group into a {}-entry scratch, but the "
        "weight has {} groups",
        tensor_idx,
        kTensorPrefetcherSelectorMaxEntries,
        num_groups);
    TT_FATAL(
        &mask.device() == &mesh_device,
        "Tensor prefetcher input {}: the group selector mask must live on the prefetcher's mesh device",
        tensor_idx);
    TT_FATAL(
        mask.layout() == Layout::ROW_MAJOR,
        "Tensor prefetcher input {}: the group selector mask must be ROW_MAJOR, got {}",
        tensor_idx,
        mask.layout());
    TT_FATAL(
        mask.dtype() == DataType::BFLOAT16 || mask.dtype() == DataType::UINT16,
        "Tensor prefetcher input {}: the group selector mask holds one 16-bit word per group, so it must be BFLOAT16 "
        "or UINT16, got {}",
        tensor_idx,
        mask.dtype());
    const auto& mask_shape = mask.logical_shape();
    TT_FATAL(
        mask.logical_volume() == mask_shape[-1] && mask_shape[-1] == num_groups,
        "Tensor prefetcher input {}: the group selector mask must be a single row of one entry per group ({}), got "
        "shape {}",
        tensor_idx,
        num_groups,
        mask_shape);

    const MemoryConfig& memory_config = mask.memory_config();
    auto& cluster = mesh_device.impl().metal_env().get_cluster();
    std::optional<SelectorLocation> location;
    for (IDevice* device : devices) {
        SelectorLocation here;
        if (memory_config.buffer_type() == BufferType::DRAM && !memory_config.is_sharded() &&
            dram_senders_on_dispatch_engine(mesh_device)) {
            // A dispatch engine reads DRAM by bank. Page 0 of an interleaved buffer is at its base in bank 0.
            here.location = TENSOR_PREFETCHER_SELECTOR_IN_DRAM_BANK;
            here.noc_xy = 0;
            here.addr = static_cast<uint32_t>(mask.mesh_buffer().address());
        } else if (memory_config.buffer_type() == BufferType::DRAM && !memory_config.is_sharded()) {
            // Page 0 of an interleaved buffer sits in bank 0.
            const auto channel = mesh_device.allocator()->impl().get_dram_channel_from_bank_id(0);
            const CoreCoord xy = cluster.get_soc_desc(device->id())
                                     .get_preferred_worker_core_for_dram_view(static_cast<int>(channel), /*noc=*/0);
            here.noc_xy = pack_virtual_xy(xy);
            here.addr = static_cast<uint32_t>(mask.mesh_buffer().get_reference_buffer()->page_address(0, 0));
        } else {
            TT_FATAL(
                memory_config.buffer_type() == BufferType::L1 && mask.shard_spec().has_value() &&
                    mask.shard_spec()->grid.num_cores() == 1,
                "Tensor prefetcher input {}: the group selector mask must be DRAM interleaved or sharded onto a single "
                "L1 core, got {}",
                tensor_idx,
                memory_config);
            const CoreCoord xy =
                device->worker_core_from_logical_core(mask.shard_spec()->grid.ranges().front().start_coord);
            here.noc_xy = pack_virtual_xy(xy);
            here.addr = static_cast<uint32_t>(mask.address());
        }
        if (location.has_value()) {
            TT_FATAL(
                location->noc_xy == here.noc_xy,
                "Tensor prefetcher input {}: the group selector mask page resolves to NoC xy {:#x} on device {} but "
                "{:#x} on the first device; one request page serves every device, so they must agree",
                tensor_idx,
                here.noc_xy,
                device->id(),
                location->noc_xy);
        } else {
            location = here;
        }
    }
    return location.value_or(SelectorLocation{});
}

// A CCE sender reads its own DRAM view and checks no address itself: local receiver r of mapping entry s reads slab
// recv_index_base_per_sender[s] + r of the tensor's shards on that sender's bank. So every slab one request reads must
// hold this tensor's data on that bank and end inside the view.
void require_receiver_slabs_in_tensor_and_view(
    const MeshTensor& tensor,
    const TensorPrefetcherTensorLayout& layout,
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& mapping,
    const std::vector<uint32_t>& recv_index_base_per_sender,
    uint64_t dram_view_bytes,
    uint32_t tensor_idx) {
    const Buffer* buffer = tensor.mesh_buffer().get_reference_buffer();
    const std::optional<BufferDistributionSpec>& distribution = buffer->buffer_distribution_spec();
    TT_FATAL(
        distribution.has_value(),
        "Tensor prefetcher: input tensor {} has no buffer distribution spec, so its receiver slabs cannot be placed",
        tensor_idx);
    const std::vector<CoreCoord>& shard_cores = distribution->cores();
    const uint64_t tensor_base = tensor.mesh_buffer().address();
    for (size_t sender_slot = 0; sender_slot < mapping.size(); ++sender_slot) {
        const uint32_t bank = static_cast<uint32_t>(mapping[sender_slot].first.x);
        const std::vector<CoreCoord>::const_iterator bank_core =
            std::find(shard_cores.begin(), shard_cores.end(), CoreCoord{bank, 0});
        const uint64_t shards_on_bank =
            bank_core == shard_cores.end()
                ? 0
                : distribution->num_shards_per_core(static_cast<size_t>(std::distance(shard_cores.begin(), bank_core)));
        const uint64_t slabs_read =
            static_cast<uint64_t>(recv_index_base_per_sender[sender_slot]) + mapping[sender_slot].second.num_cores();
        const uint64_t bytes_read = slabs_read * layout.recv_stride_bytes;
        TT_FATAL(
            slabs_read <= shards_on_bank && bytes_read <= buffer->aligned_size_per_bank(),
            "Tensor prefetcher: input tensor {} holds {} receiver slabs ({} B) on DRAM bank {}, but that bank's sender "
            "reads slabs [0, {}) of {} B each, so the request runs past the tensor",
            tensor_idx,
            shards_on_bank,
            buffer->aligned_size_per_bank(),
            bank,
            slabs_read,
            layout.recv_stride_bytes);
        TT_FATAL(
            tensor_base + bytes_read <= dram_view_bytes,
            "Tensor prefetcher: input tensor {} on DRAM bank {} reads [0x{:x}, 0x{:x}), past the {} B DRAM view",
            tensor_idx,
            bank,
            tensor_base,
            tensor_base + bytes_read,
            dram_view_bytes);
    }
}

}  // namespace

TensorPrefetcherManager::TensorPrefetcherManager(
    MeshDevice* mesh_device, std::function<std::lock_guard<std::mutex>()> lock_api_function) :
    mesh_device_(mesh_device), lock_api_function_(std::move(lock_api_function)) {
    TT_FATAL(mesh_device_ != nullptr, "TensorPrefetcherManager requires a non-null MeshDevice");
    TT_FATAL(static_cast<bool>(lock_api_function_), "TensorPrefetcherManager requires a valid lock_api_function");
}

TensorPrefetcherManager::~TensorPrefetcherManager() { stop_without_throwing(); }

void TensorPrefetcherManager::enumerate_dram_senders(const std::vector<uint32_t>& sender_banks) {
    TT_FATAL(!devices_.empty(), "Tensor prefetcher requires at least one device");
    // dram_grid_size() already TT_FATALs unless every device in the mesh reports the same bank count.
    num_banks_ = mesh_device_->dram_grid_size().x;
    const PrefetcherArchTraits& traits = prefetcher_traits_for(mesh_device_->arch());

    std::vector<uint32_t> provisioned_banks = sender_banks;
    if (provisioned_banks.empty()) {
        provisioned_banks.resize(num_banks_);
        std::iota(provisioned_banks.begin(), provisioned_banks.end(), 0u);
    }
    std::vector<bool> bank_listed(num_banks_, false);
    for (const uint32_t bank : provisioned_banks) {
        TT_FATAL(
            bank < num_banks_, "Tensor prefetcher sender bank {} is out of range for {} DRAM banks", bank, num_banks_);
        TT_FATAL(!bank_listed[bank], "Tensor prefetcher sender bank {} is listed more than once", bank);
        bank_listed[bank] = true;
    }

    // Logical DRAM coords name an endpoint role, so a bank's two senders have the same logical
    // coords on every device even when their DRAM harvest masks differ; only the physical
    // subchannel each one resolves to changes. Build the list from the reference device and check
    // the rest agree, because everything downstream (socket placement, kernel placement, pipe sender
    // indices) indexes senders by slot and would otherwise silently drive another device's wrong
    // DRISC core.
    // Roles per bank on DRAM cores come from the arch (two on Blackhole: the free subchannel, then the
    // NOC1-endpoint subchannel); a dispatch engine holds every endpoint, so it has one.
    const size_t roles_per_bank = on_dispatch_engine() ? 1 : traits.senders_per_bank;
    const auto senders_on = [this, roles_per_bank, &provisioned_banks](const IDevice* device) {
        std::vector<CoreCoord> senders;
        senders.reserve(roles_per_bank * provisioned_banks.size());
        for (const uint32_t b : provisioned_banks) {
            const std::vector<CoreCoord> bank_senders = mesh_device_->impl().dram_sender_logical_cores(device, b);
            TT_FATAL(
                bank_senders.size() == roles_per_bank,
                "Tensor prefetcher expected {} DRAM sender roles for bank {} on device {}, found {}",
                roles_per_bank,
                b,
                device->id(),
                bank_senders.size());
            senders.insert(senders.end(), bank_senders.begin(), bank_senders.end());
        }
        return senders;
    };

    const IDevice* reference_device = devices_.front();
    sender_logical_cores_ = senders_on(reference_device);
    for (size_t d = 1; d < devices_.size(); ++d) {
        TT_FATAL(
            senders_on(devices_[d]) == sender_logical_cores_,
            "Tensor prefetcher: DRAM sender slots on device {} name different logical cores than on reference "
            "device {}; every device must resolve slot s to the same (bank, role)",
            devices_[d]->id(),
            reference_device->id());
    }
    num_senders_ = static_cast<uint32_t>(sender_logical_cores_.size());
}

TensorPrefetcherManager::RequestTarget TensorPrefetcherManager::target_for(
    const std::vector<std::reference_wrapper<const experimental::PrefetcherPipe>>& prefetcher_pipes) const {
    TT_FATAL(!prefetcher_pipes.empty(), "QueueTensorPrefetcherRequest requires at least one PrefetcherPipe");

    RequestTarget target;
    // Bank-major mapping: a bank's pipes stay adjacent and in their own order. Taken from the
    // shared helper so this order is the one a consumer op's cache key sees, not a re-derivation of
    // it. It fixes only which page goes to which sender; each pipe carries its own slab base, so
    // the caller may pass one factory call's pipes in any order. It must pass all of them: a tensor
    // laid out for a subset has fewer slabs per bank than the pipes' slab bases assume.
    target.mapping = experimental::GetPrefetcherPipeSenderReceiverMapping(prefetcher_pipes);
    target.config_page_addr_per_sender.reserve(target.mapping.size());
    target.config_page_alloc_per_sender.reserve(target.mapping.size());
    target.recv_index_base_per_sender.reserve(target.mapping.size());

    std::unordered_set<CoreCoord> distinct_receivers;
    uint32_t total_receivers = 0;
    std::optional<uint64_t> factory_id;
    uint32_t factory_num_pipes = 0;
    const bool cce_senders = uses_cce_senders();

    for (size_t p = 0; p < prefetcher_pipes.size(); ++p) {
        const experimental::PrefetcherPipe& pipe = prefetcher_pipes[p];
        const auto bank_id = static_cast<uint32_t>(target.mapping[p].first.x);

        // config_page_addr_per_sender below is a DRISC L1 offset reserved on this mesh, so a pipe from
        // another one would aim the sender at unrelated state instead of failing here.
        TT_FATAL(
            pipe.get_device() == mesh_device_,
            "QueueTensorPrefetcherRequest requires PrefetcherPipes created on the prefetcher's own mesh device, "
            "but pipe {} (bank {}) belongs to another",
            p,
            bank_id);
        TT_FATAL(
            pipe.sender_core_type() == experimental::SenderCoreType::Dram,
            "QueueTensorPrefetcherRequest requires DRAM-sender PrefetcherPipes, but pipe {} (bank {}) has a worker "
            "sender. Build them with CreatePrefetcherPipesForTensorPrefetcher.",
            p,
            bank_id);
        // A CCE sender reads a DRAM-receiver pipe's sender state and every other sender a worker-receiver
        // pipe's; handed the other kind, it would read the wrong layout.
        TT_FATAL(
            pipe.impl().space().has_dram_receivers() == cce_senders,
            "QueueTensorPrefetcherRequest: pipe {} (bank {}) has {} receivers, but this prefetcher's senders {}. Build "
            "the pipes from a PrefetcherPipeSpace whose l1_source is {}.",
            p,
            bank_id,
            pipe.impl().space().has_dram_receivers() ? "DRAM-core" : "worker",
            cce_senders ? "are CCEs, which deliver only to DRAM receivers" : "deliver only to worker receivers",
            cce_senders ? "DramReceiverZone" : "not DramReceiverZone");

        const uint32_t ring_size = pipe.ring_size();
        if (!factory_id.has_value()) {
            factory_id = pipe.impl().tensor_prefetcher_factory_id();
            factory_num_pipes = pipe.impl().tensor_prefetcher_factory_num_pipes();
            target.per_recv_capacity_bytes = ring_size;
        }
        TT_FATAL(
            pipe.impl().tensor_prefetcher_factory_id() != 0 &&
                pipe.impl().tensor_prefetcher_factory_id() == *factory_id,
            "QueueTensorPrefetcherRequest requires every pipe to come from one "
            "CreatePrefetcherPipesForTensorPrefetcher call, but pipe {} (bank {}) has factory identity {} and the "
            "first has {}.",
            p,
            bank_id,
            pipe.impl().tensor_prefetcher_factory_id(),
            *factory_id);
        TT_FATAL(
            ring_size == target.per_recv_capacity_bytes,
            "QueueTensorPrefetcherRequest requires one ring size across every pipe: pipe {} (bank {}) has {} B, "
            "but the first pipe has {} B.",
            p,
            bank_id,
            ring_size,
            target.per_recv_capacity_bytes);

        const CoreRangeSet& receivers = target.mapping[p].second;
        for (const CoreCoord& receiver : corerange_to_cores(receivers)) {
            distinct_receivers.insert(receiver);
        }
        total_receivers += receivers.num_cores();
        // The factory handed each pipe the base its sender owns in its bank, so the queue path
        // reads it off the pipe instead of re-deriving it from where the pipe sits in this list.
        target.recv_index_base_per_sender.push_back(pipe.impl().recv_index_base());
        target.config_page_addr_per_sender.push_back(
            static_cast<uint32_t>(experimental::sender_state_drisc_l1_base(pipe)));
        target.config_page_alloc_per_sender.emplace_back(experimental::sender_state_drisc_l1_allocation(pipe));
    }
    TT_FATAL(
        distinct_receivers.size() == total_receivers,
        "QueueTensorPrefetcherRequest requires disjoint receiver sets across every PrefetcherPipe: {} receivers were "
        "listed but only {} are distinct.",
        total_receivers,
        distinct_receivers.size());

    // Same factory call and no pipe listed twice (the disjointness check above), so matching the
    // call's pipe count means every one of its pipes is here.
    TT_FATAL(
        prefetcher_pipes.size() == factory_num_pipes,
        "QueueTensorPrefetcherRequest requires every PrefetcherPipe from one "
        "CreatePrefetcherPipesForTensorPrefetcher call, in any order, but {} of that call's {} pipes were passed.",
        prefetcher_pipes.size(),
        factory_num_pipes);
    target.total_receivers = total_receivers;
    return target;
}

std::vector<uint32_t> TensorPrefetcherManager::sender_indices_for_target(const RequestTarget& target) const {
    const auto& mapping = target.mapping;
    TT_FATAL(!mapping.empty(), "Tensor prefetcher: target sender mapping must not be empty");

    std::vector<uint32_t> sender_indices;
    sender_indices.reserve(mapping.size());
    for (const auto& [sender, _receivers] : mapping) {
        const auto it = std::find(sender_logical_cores_.begin(), sender_logical_cores_.end(), sender);
        TT_FATAL(
            it != sender_logical_cores_.end(),
            "Tensor prefetcher: target sender core ({}, {}) is not one of the {} provisioned DRAM sender cores",
            sender.x,
            sender.y,
            num_senders_);
        sender_indices.push_back(static_cast<uint32_t>(std::distance(sender_logical_cores_.begin(), it)));
    }
    return sender_indices;
}

void TensorPrefetcherManager::build_and_launch_programs(
    const H2DSocketTransport& sockets,
    uint32_t stage_ring_base,
    uint32_t stage_ring_size,
    const std::optional<MpfePolicy>& mpfe_policy) {
    const uint32_t socket_page_size = sockets.page_bytes();

    const PrefetcherArchTraits& traits = prefetcher_traits_for(mesh_device_->arch());
    programs_.clear();
    for (uint32_t d = 0; d < devices_.size(); ++d) {
        auto program = std::make_unique<Program>(CreateProgram(*mesh_device_));
        const auto& soc_desc = mesh_device_->impl().metal_env().get_cluster().get_soc_desc(devices_[d]->id());
        if (traits.gddr_subchannels_per_bank.has_value()) {
            TT_FATAL(
                soc_desc.get_grid_size(tt::CoreType::DRAM).y == *traits.gddr_subchannels_per_bank,
                "Tensor prefetcher expected {} GDDR subchannels per bank, found {}",
                *traits.gddr_subchannels_per_bank,
                soc_desc.get_grid_size(tt::CoreType::DRAM).y);
        }

        for (uint32_t s = 0; s < num_senders_; ++s) {
            const CoreCoord sender_logical = sender_logical_cores_[s];
            const uint32_t bank_id = static_cast<uint32_t>(sender_logical.x);
            const uint32_t bank_sender_base = traits.senders_per_bank * bank_id;
            const bool controls_ordinary_mpfe = mpfe_policy.has_value() && s == bank_sender_base;
            uint32_t own_mpfe_port = 0;
            uint32_t ordinary_mpfe_port = 0;
            uint32_t own_active_mpfe_weight = 0;
            uint32_t ordinary_mpfe_weight = 0;
            bool dynamic_mpfe_weighting = false;
            if (mpfe_policy.has_value()) {
                const uint32_t free_sender_port = get_mpfe_port(soc_desc, sender_logical_cores_[bank_sender_base]);
                const uint32_t noc1_sender_port = get_mpfe_port(soc_desc, sender_logical_cores_[bank_sender_base + 1]);
                own_mpfe_port = controls_ordinary_mpfe ? free_sender_port : noc1_sender_port;
                own_active_mpfe_weight =
                    controls_ordinary_mpfe ? mpfe_policy->active.free_sender : mpfe_policy->active.noc1_sender;
                ordinary_mpfe_weight = mpfe_policy->active.ordinary;
                dynamic_mpfe_weighting = mpfe_policy->dynamic;
                // Logical DRAM y=0 is the harvest-stable NOC0 worker-endpoint role, not raw
                // hardware subchannel 0. get_mpfe_port resolves that role through this device's
                // physical subchannel and maps hardware subchannels 0..2 to MPFE P1..P3.
                ordinary_mpfe_port = get_mpfe_port(soc_desc, CoreCoord{bank_id, 0});
            }

            std::vector<uint32_t> compile_args = {
                stage_ring_base,
                stage_ring_size,
                socket_page_size,
                signal_slots_l1_addr_,
                signal_slot_stride_,
                static_cast<uint32_t>(controls_ordinary_mpfe),
                own_active_mpfe_weight,
                ordinary_mpfe_weight,
                static_cast<uint32_t>(dynamic_mpfe_weighting),
                static_cast<uint32_t>(mpfe_policy.has_value()),
                selector_scratch_l1_addr_,
            };

            // WATCHER_NOINLINE only takes effect in watcher builds: it lets the compiler outline the
            // FORCE_INLINE helpers there, without which the watcher's NoC checks inlined at every
            // call push this kernel past the DRISC firmware window. Other builds are unaffected.
            KernelHandle kernel_id = CreateKernel(
                *program,
                traits.kernel_path,
                sender_logical,
                DramConfig{.noc = NOC::NOC_0, .compile_args = compile_args, .defines = {{"WATCHER_NOINLINE", "1"}}});

            const uint32_t socket_addr = sockets.config_buffer_address(d * num_senders_ + s);
            // The free sender passes the op signal counts it waits for on to the bank's NOC1-endpoint sender,
            // which no signal reaches (see the kernel's WAIT_SIGNAL handler), at coords DRAM harvesting may
            // move per device.
            uint32_t op_signal_forward_xy = 0;
            if (s == bank_sender_base) {
                op_signal_forward_xy = pack_virtual_xy(devices_[d]->virtual_core_from_logical_core(
                    sender_logical_cores_[bank_sender_base + 1], CoreType::DRAM));
            }
            std::vector<uint32_t> rt_args = {
                bank_id, socket_addr, own_mpfe_port, ordinary_mpfe_port, op_signal_forward_xy};
            SetRuntimeArgs(*program, kernel_id, sender_logical, rt_args);
        }

        programs_.push_back(std::move(program));
    }
}

void TensorPrefetcherManager::require_idle_sender_cces() const {
    const MetalContext& metal = mesh_device_->impl().metal_context();
    for (const IDevice* device : devices_) {
        const metal_SocDescriptor& soc_desc = metal.get_cluster().get_soc_desc(device->id());
        for (const CoreCoord& sender_logical : sender_logical_cores_) {
            const uint8_t signal = read_go_message_signal(metal, device, sender_logical);
            TT_FATAL(
                signal == dev_msgs::RUN_MSG_DONE,
                "Tensor prefetcher: sender CCE {} (device {}, logical {}) holds GO_MSG signal 0x{:02x}, not "
                "RUN_MSG_DONE (0x{:02x}). It is still running a Program, and a launch onto it would be lost; leave its "
                "bank out of QuasarTensorPrefetcherConfig::dram_sender_banks or wait for that Program first.",
                package_dram_core_index(soc_desc, sender_logical),
                device->id(),
                sender_logical.str(),
                signal,
                dev_msgs::RUN_MSG_DONE);
        }
    }
}

void TensorPrefetcherManager::zero_cce_sender_control_blocks() const {
    const std::vector<std::byte> zeros(kCceSenderZeroedBytes, std::byte{0});
    for (IDevice* device : devices_) {
        for (const CoreCoord& sender_logical : sender_logical_cores_) {
            write_dram_core_l1(*mesh_device_, device, sender_logical, cce_control_base_, zeros);
        }
        mesh_device_->impl().metal_env().get_cluster().l1_barrier(device->id());
    }
}

void TensorPrefetcherManager::build_cce_sender_programs(uint64_t wait_budget_cycles) {
    std::vector<uint32_t> runtime_args(kCceSenderArgs, 0);
    runtime_args[kCceSenderArgControlBase] = cce_control_base_;
    runtime_args[kCceSenderArgBudgetLo] = static_cast<uint32_t>(wait_budget_cycles);
    runtime_args[kCceSenderArgBudgetHi] = static_cast<uint32_t>(wait_budget_cycles >> 32);
    runtime_args[kCceSenderArgStageRingBytes] = stage_ring_size_;
    const PrefetcherArchTraits& traits = prefetcher_traits_for(mesh_device_->arch());
    const Cluster& cluster = mesh_device_->impl().metal_env().get_cluster();
    programs_.clear();
    for (uint32_t d = 0; d < devices_.size(); ++d) {
        const metal_SocDescriptor& soc_desc = cluster.get_soc_desc(devices_[d]->id());
        const uint32_t cces_per_mimir = static_cast<uint32_t>(soc_desc.get_grid_size(tt::CoreType::DRAM).y);
        std::unique_ptr<Program> program = std::make_unique<Program>(CreateProgram(*mesh_device_));
        for (const CoreCoord& sender_logical : sender_logical_cores_) {
            // A sender reads only its own view, which must sit on its own Mimir: the GDDR route it reads through is
            // the local one.
            const uint32_t bank = static_cast<uint32_t>(sender_logical.x);
            const uint32_t view_mimir =
                static_cast<uint32_t>(soc_desc.get_channel_for_dram_view(static_cast<int>(bank)));
            const uint32_t sender_cce = package_dram_core_index(soc_desc, sender_logical);
            TT_FATAL(
                sender_cce / cces_per_mimir == view_mimir,
                "Tensor prefetcher: sender CCE {} of DRAM view {} is on Mimir {}, but the view is on Mimir {}",
                sender_cce,
                bank,
                sender_cce / cces_per_mimir,
                view_mimir);
            const uint64_t view_offset = soc_desc.get_address_offset(static_cast<int>(bank));
            runtime_args[kCceSenderArgMimir] = view_mimir;
            runtime_args[kCceSenderArgViewOffsetLo] = static_cast<uint32_t>(view_offset);
            runtime_args[kCceSenderArgViewOffsetHi] = static_cast<uint32_t>(view_offset >> 32);
            const KernelHandle kernel_id = CreateKernel(*program, traits.kernel_path, sender_logical, DramConfig{});
            SetRuntimeArgs(*program, kernel_id, sender_logical, runtime_args);
        }
        programs_.push_back(std::move(program));
    }
}

void TensorPrefetcherManager::start(const experimental::TensorPrefetcherConfig& config) {
    auto lock = lock_api_function_();
    TT_FATAL(!active_, "A Tensor prefetcher is already active on this mesh device. Call StopTensorPrefetcher first.");
    bool benchmark_enabled = false;
    std::optional<MpfePolicy> mpfe_policy = std::nullopt;
    if (mesh_device_->arch() == ARCH::BLACKHOLE) {
        benchmark_enabled = benchmark_mpfe_enabled();
        const experimental::BlackholeTensorPrefetcherConfig requested_config =
            config.blackhole.value_or(experimental::BlackholeTensorPrefetcherConfig{});
        const experimental::BlackholeTensorPrefetcherConfig effective_config{
            .free_sender_mpfe_weight = benchmark_mpfe_weight(
                "FREE_SENDER_WEIGHT", requested_config.free_sender_mpfe_weight, benchmark_enabled),
            .noc1_sender_mpfe_weight = benchmark_mpfe_weight(
                "NOC1_SENDER_WEIGHT", requested_config.noc1_sender_mpfe_weight, benchmark_enabled),
            .ordinary_mpfe_weight =
                benchmark_mpfe_weight("ORDINARY_WEIGHT", requested_config.ordinary_mpfe_weight, benchmark_enabled),
            .dynamic_mpfe_weighting = benchmark_mpfe_bool(
                "DYNAMIC_MPFE_WEIGHTING", requested_config.dynamic_mpfe_weighting, benchmark_enabled),
        };
        TT_FATAL(
            effective_config.free_sender_mpfe_weight <= kMaxMpfeWeight &&
                effective_config.noc1_sender_mpfe_weight <= kMaxMpfeWeight &&
                effective_config.ordinary_mpfe_weight <= kMaxMpfeWeight,
            "Tensor prefetcher MPFE weights must be in [0, {}], got {}/{}/{}",
            kMaxMpfeWeight,
            effective_config.free_sender_mpfe_weight,
            effective_config.noc1_sender_mpfe_weight,
            effective_config.ordinary_mpfe_weight);

        mpfe_policy = MpfePolicy{
            .active =
                {
                    .free_sender = effective_config.free_sender_mpfe_weight,
                    .noc1_sender = effective_config.noc1_sender_mpfe_weight,
                    .ordinary = effective_config.ordinary_mpfe_weight,
                },
            .dynamic = effective_config.dynamic_mpfe_weighting,
        };
        if (benchmark_enabled) {
            log_info(
                tt::LogMetal,
                "TENSOR_PREFETCHER_MPFE_POLICY active={}/{}/{} dynamic={} source=benchmark_env",
                mpfe_policy->active.free_sender,
                mpfe_policy->active.noc1_sender,
                mpfe_policy->active.ordinary,
                mpfe_policy->dynamic ? 1 : 0);
        }
    } else {
        TT_FATAL(
            !config.blackhole.has_value(),
            "Blackhole tensor prefetcher configuration cannot be used on architecture {}",
            mesh_device_->arch());
    }
    if (mesh_device_->arch() != ARCH::QUASAR) {
        TT_FATAL(
            !config.quasar.has_value(),
            "Quasar tensor prefetcher configuration cannot be used on architecture {}",
            mesh_device_->arch());
    }

    const auto& hal = mesh_device_->impl().metal_env().get_hal();
    TT_FATAL(
        mesh_device_->impl().dram_sender_placement().has_value(),
        "Tensor prefetcher requires programmable DRAM cores, which auto-enable on Blackhole with firmware "
        ">= 19.12.0.0, or fast dispatch on a dispatch engine serving one command queue");
    const PrefetcherArchTraits& traits = prefetcher_traits_for(mesh_device_->arch());
    const experimental::QuasarTensorPrefetcherConfig quasar_config =
        config.quasar.value_or(experimental::QuasarTensorPrefetcherConfig{});

    // Populate devices_ before resolving sender slots: enumerate_dram_senders checks the slots
    // against every device's DRAM topology. Build the coord->index map at the same time so
    // worker_loop fan-out is O(targets).
    devices_.clear();
    device_index_by_coord_.clear();
    for (auto* device : mesh_device_->get_view().impl().get_devices()) {
        const uint32_t d = static_cast<uint32_t>(devices_.size());
        devices_.push_back(device);
        device_index_by_coord_.emplace(mesh_device_->get_view().find_device(device->id()), d);
    }
    enumerate_dram_senders(quasar_config.dram_sender_banks);
    drain_targets_per_sender_.assign(num_senders_, {});
    if (on_dispatch_engine()) {
        start_on_dispatch_engine();
        return;
    }

    const KernelRegionCarve carve =
        carve_kernel_region(hal, mesh_device_->impl().drisc_l1_arena(), traits, quasar_config);
    TT_FATAL(carve.failure_reason.empty(), "{}", carve.failure_reason);
    signal_slot_stride_ = carve.signal_slot_stride;
    const uint32_t signal_slots_bytes = kNumSignalSlots * signal_slot_stride_;
    signal_slots_l1_addr_ = carve.signal_slots_l1_addr;
    socket_config_l1_addr_ = carve.socket_config_l1_addr;
    socket_data_l1_addr_ = carve.socket_data_l1_addr;
    selector_scratch_l1_addr_ = carve.selector_scratch_l1_addr;
    cce_control_base_ = carve.cce_control_base;
    stage_ring_base_ = carve.stage_ring_base;
    set_stage_ring_size(carve.stage_ring_size, hal.get_alignment(HalMemType::L1));

    // A retry after a failed Start must not hold the previous attempt's sockets alongside new ones.
    transport_.reset();
    // The socket transport, which its worker drives directly; null for CCE senders.
    H2DSocketTransport* sockets = nullptr;
    if (uses_cce_senders()) {
        const size_t num_cces = mesh_device_->impl()
                                    .metal_env()
                                    .get_cluster()
                                    .get_soc_desc(devices_.front()->id())
                                    .get_metal_dram_cores(tt::CoordSystem::TRANSLATED)
                                    .size();
        // A warning, not a failure: a session with no receivers, such as one that only fences, is still valid.
        if (num_senders_ >= num_cces) {
            log_warning(
                tt::LogMetal,
                "Tensor prefetcher: all {} CCEs are senders, so none is left for receiver Programs. Pass "
                "QuasarTensorPrefetcherConfig::dram_sender_banks to keep CCEs free for receivers.",
                num_cces);
        }
        // Before anything is written: a busy sender CCE must fail Start with its TL1 untouched.
        require_idle_sender_cces();
        const float timeout_seconds =
            mesh_device_->impl().metal_env().get_rtoptions().get_timeout_duration_for_operations().count();
        if (quasar_config.wait_budget_cycles != 0 && timeout_seconds == 0.0f) {
            log_warning(
                tt::LogMetal,
                "Tensor prefetcher: senders give up a wait after {} rdcycle cycles, but "
                "TT_METAL_OPERATION_TIMEOUT_SECONDS is unset, so StopTensorPrefetcher waits for them without a limit. "
                "The budget is CCE cycles, not wall time: on the emulator the chip clock advances only with host "
                "traffic (about 85,000 cycles/s while the host polls, 90 cycles/s while it is idle).",
                quasar_config.wait_budget_cycles);
        }
        request_ring_no_progress_limit_ =
            timeout_seconds > 0.0f
                ? std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                      std::chrono::duration<float>(timeout_seconds))
                : std::chrono::duration_cast<std::chrono::steady_clock::duration>(kDefaultRequestRingNoProgressLimit);
        give_up_on_running_senders_ = timeout_seconds > 0.0f;
        build_cce_sender_programs(quasar_config.wait_budget_cycles);
        // Its CQ fence slots are the control block's first lines, so this zeroes them too.
        zero_cce_sender_control_blocks();
        transport_ =
            std::make_unique<Tl1RingTransport>(*mesh_device_, devices_, sender_logical_cores_, cce_control_base_);
    } else {
        std::unique_ptr<H2DSocketTransport> socket_transport = std::make_unique<H2DSocketTransport>(
            *mesh_device_, devices_, sender_logical_cores_, socket_config_l1_addr_, socket_data_l1_addr_);
        build_and_launch_programs(*socket_transport, stage_ring_base_, stage_ring_size_, mpfe_policy);
        sockets = socket_transport.get();
        transport_ = std::move(socket_transport);
    }

    // Zero every sender's signal slots before its kernel starts. An op may signal as soon as Start
    // returns, possibly before the kernel runs its first instruction, and an increment that lands on
    // a zeroed slot is never lost; zeroing from the kernel could erase it. The host counts of waits
    // restart with the slots, before the worker that owns them starts. A CCE sender's slots were zeroed
    // with its control block above; its table is only the CQ fence slots, which this would overrun.
    if (!uses_cce_senders()) {
        const std::vector<std::byte> zero_slots(signal_slots_bytes, std::byte{0});
        for (IDevice* device : devices_) {
            for (const CoreCoord& sender : sender_logical_cores_) {
                write_dram_sender_l1(*mesh_device_, device, sender, signal_slots_l1_addr_, zero_slots);
            }
            mesh_device_->impl().metal_env().get_cluster().l1_barrier(device->id());
        }
    }
    op_signal_waits_.fill(0);

    // Launch programs (non-blocking — kernels park on the socket immediately).
    for (uint32_t d = 0; d < devices_.size(); ++d) {
        programs_[d]->impl().compile(devices_[d], /*force_slow_dispatch=*/true);
        ::tt::tt_metal::slow_dispatch::WriteRuntimeArgsToDevice(
            *devices_[d], *programs_[d], /*force_slow_dispatch=*/true);
        ::tt::tt_metal::slow_dispatch::LaunchProgramAsync(*devices_[d], *programs_[d], /*force_slow_dispatch=*/true);
    }

    stop_requested_.store(false);
    worker_gave_up_on_sender_.store(false);
    dropped_request_pages_.store(0);
    if (sockets != nullptr) {
        host_worker_ = std::thread(&TensorPrefetcherManager::worker_loop, this, std::ref(*sockets));
    } else {
        host_worker_ = std::thread(&TensorPrefetcherManager::worker_loop_request_ring, this);
    }
    ++run_id_;
    active_ = true;
    if (benchmark_enabled) {
        active_lifetime_timer_.emplace(std::string(kBenchmarkTimerName));
    }
}

MeshCoordinateRangeSet TensorPrefetcherManager::full_mesh_subset() const {
    MeshCoordinateRangeSet out;
    out.merge(MeshCoordinateRange(mesh_device_->shape()));
    return out;
}

std::vector<std::vector<std::vector<uint8_t>>> TensorPrefetcherManager::serialize_request_pages(
    const RequestTarget& target, const std::vector<experimental::TensorPrefetcherInput>& data_tensors) const {
    TT_FATAL(!data_tensors.empty(), "QueueTensorPrefetcherRequest requires at least one tensor");

    const ContextId context_id = mesh_device_->impl().get_context_id();

    // Derive the receiver counts from the target itself so each Queue call can target an object
    // with a different receiver count. total_receivers (== ring_size) and receivers_per_bank are
    // independent of how many DRISC senders drive a bank.
    const auto& mapping = target.mapping;
    const uint32_t total_receivers = target.total_receivers;
    TT_FATAL(num_banks_ > 0, "Tensor prefetcher: num_banks must be > 0");
    // receivers_per_bank is only consumed by the K-row-major layout (single sender per
    // bank). Recv-contig derives its geometry from the shard shape and ignores it, and
    // its receivers need not be uniform per bank — so the even-divisibility requirement
    // is enforced per-tensor in compute_tensor_layout(), only for K-row-major tensors.
    const uint32_t receivers_per_bank = total_receivers / num_banks_;
    // K-row-major stores one complete per-bank shard, so all receivers for a bank must
    // remain on its primary sender, and the kernel slices every bank's shard by the one
    // receivers_per_bank count, so every bank must drive exactly that many. Receiver-contiguous
    // tensors may use either this topology or a dual-sender split.
    const PrefetcherArchTraits& traits = prefetcher_traits_for(mesh_device_->arch());
    // The limit is the CCE sender's; a dispatch engine on the same arch does not have it.
    const bool receiver_contiguous_batched_only = !on_dispatch_engine() && traits.receiver_contiguous_batched_only;
    bool krow_compatible_mapping = !receiver_contiguous_batched_only && mapping.size() == num_banks_;
    std::vector<bool> primary_bank_seen(num_banks_, false);
    if (krow_compatible_mapping) {
        for (const auto& [sender, receivers] : mapping) {
            const uint32_t bank = static_cast<uint32_t>(sender.x);
            if (bank >= num_banks_ || primary_bank_seen[bank] || sender != sender_logical_cores_[2 * bank] ||
                receivers.num_cores() != receivers_per_bank) {
                krow_compatible_mapping = false;
                break;
            }
            primary_bank_seen[bank] = true;
        }
    }

    const uint32_t pcie_alignment = mesh_device_->impl().metal_env().get_hal().get_alignment(HalMemType::HOST);
    const uint32_t aligned_page_bytes = align_up(kRequestPageBytes, pcie_alignment);

    constexpr uint32_t kHeaderBytes = sizeof(TensorPrefetcherRequestHeader);
    constexpr uint32_t kEntryBytes = sizeof(TensorPrefetcherEntry);
    constexpr uint32_t kLayoutBytes = sizeof(TensorPrefetcherTensorLayout);

    // max_receivers sizes the uniform slot so one template page serves every sender (the dedup/fit
    // decisions below are sender-independent). It is host-only packing bookkeeping — the kernel
    // never sees it, because entries carry their slot's byte offset. It is just the largest
    // receiver count over the target's senders.
    uint32_t max_receivers = 0;
    for (const auto& [_sender, receivers] : mapping) {
        max_receivers = std::max(max_receivers, receivers.num_cores());
    }
    // Ahead of the streaming topology guard, so a streaming tensor on CCE senders gets this message.
    if (receiver_contiguous_batched_only) {
        for (size_t tensor_idx = 0; tensor_idx < data_tensors.size(); ++tensor_idx) {
            TT_FATAL(
                data_tensors[tensor_idx].rotation.empty(),
                "Tensor prefetcher on {} delivers batched tensors only, but input tensor {} has a streaming rotation",
                mesh_device_->arch(),
                tensor_idx);
        }
    }
    // A streaming tensor needs each receiver's bank-local slab index, which is that sender's
    // recv_index_base plus its position within the sender. The topology guard below is what makes
    // that index usable as a global receiver position, so it runs only when some tensor streams.
    bool any_streaming = false;
    for (const auto& input : data_tensors) {
        if (!input.rotation.empty()) {
            any_streaming = true;
            break;
        }
    }
    // Only a streaming tensor's slot carries a rotation table (the kernel reads it only when the
    // layout says streaming), so a request with none packs bare geometry and is not bounded by the
    // receivers per sender.
    const uint32_t layout_stride =
        kLayoutBytes + (any_streaming ? max_receivers * static_cast<uint32_t>(sizeof(uint32_t)) : 0u);
    // Byte offset of layout slot i: the slots grow backward from the end of the payload, so slot 0
    // is flush against it. Both the entry that names a slot and the write that fills it go through
    // here, so the two cannot drift apart.
    const auto slot_offset = [layout_stride](uint32_t i) { return kRequestPageBytes - (i + 1) * layout_stride; };
    if (any_streaming) {
        // Both the strided and contiguous (bank, slab index) -> global position formulas are
        // bijections onto [0, total_receivers) only when the DRAM banks are dense 0..num_banks-1 and
        // every bank has exactly receivers_per_bank receivers. Guard that topology invariant once
        // here so the per-sender rotation fill below is a plain gather with no inner-loop range check.
        std::vector<uint32_t> bank_receiver_count(num_banks_, 0);
        for (const auto& [sender_logical, receivers] : mapping) {
            const uint32_t bank = static_cast<uint32_t>(sender_logical.x);
            TT_FATAL(
                bank < num_banks_,
                "Tensor prefetcher: streaming requires dense DRAM bank ids 0..{}, but a sender occupies bank {}.",
                num_banks_ - 1,
                bank);
            bank_receiver_count[bank] += receivers.num_cores();
        }
        for (uint32_t b = 0; b < num_banks_; ++b) {
            TT_FATAL(
                bank_receiver_count[b] == receivers_per_bank,
                "Tensor prefetcher: streaming requires a uniform receiver-contiguous topology — bank {} has {} "
                "receivers but expected receivers_per_bank ({} = total_receivers {} / num_banks {}).",
                b,
                bank_receiver_count[b],
                receivers_per_bank,
                total_receivers,
                num_banks_);
        }
    }
    TT_FATAL(
        kHeaderBytes + layout_stride + kEntryBytes <= kRequestPageBytes,
        "Tensor prefetcher: request page ({} B) too small for one tensor: header({}) + layout slot ({} = "
        "geometry {} + {} rotation entries) + entry({}). Reduce receivers per sender or grow kRequestPageBytes.",
        kRequestPageBytes,
        kHeaderBytes,
        layout_stride,
        kLayoutBytes,
        max_receivers,
        kEntryBytes);

    // ---- Abstract page plan (sender-independent): entries + dedup'd geometry+rotation slots ----
    struct Slot {
        TensorPrefetcherTensorLayout geom;
        std::vector<uint32_t> rotation;  // caller's global rotation (total_receivers entries), or empty == batched
        // Shard distribution used to slice `rotation` per receiver; only meaningful when streaming. Part
        // of slot identity: two tensors with the same geometry+rotation but different strategies pack
        // different per-sender rotation bytes, so they must not dedup together.
        ShardDistributionStrategy strategy = ShardDistributionStrategy::ROUND_ROBIN_1D;
    };
    struct PlanEntry {
        uint32_t bank_local_base = 0;
        uint32_t layout_index = 0;
        uint16_t num_groups = 1;
        TensorPrefetcherSelectorMode selector_mode = TENSOR_PREFETCHER_SELECTOR_NONE;
        SelectorLocation selector;
    };
    struct PagePlan {
        std::vector<PlanEntry> entries;
        std::vector<Slot> slots;
    };
    auto slot_equal = [](const Slot& a, const Slot& b) {
        return layout_equal(a.geom, b.geom) && a.rotation == b.rotation && a.strategy == b.strategy;
    };
    std::vector<PagePlan> plans(1);

    for (size_t tensor_idx = 0; tensor_idx < data_tensors.size(); ++tensor_idx) {
        const auto& input = data_tensors[tensor_idx];
        const bool streaming = !input.rotation.empty();
        if (receiver_contiguous_batched_only) {
            // Ahead of compute_tensor_layout, whose K-row-major sizer would reject such a tensor for its own reasons.
            TT_FATAL(
                detect_layout_mode(
                    input.tensor.get(), *input.tensor.get().mesh_buffer().get_reference_buffer(), total_receivers) ==
                    LayoutMode::ReceiverContiguous,
                "Tensor prefetcher on {} supports receiver-contiguous tensors only, but input tensor {} resolves to "
                "the K-row-major layout. Shard it so each receiver owns a disjoint contiguous shard.",
                mesh_device_->arch(),
                tensor_idx);
        }
        // Streaming delivers block (rotation[r] + p) mod block_count, only a valid permutation when
        // block_count == ring_size (== total_receivers). The consuming ring matmul always uses
        // num_blocks = ring_size, so this holds for the intended use.
        if (streaming) {
            TT_FATAL(
                input.block_count == total_receivers,
                "Streaming Tensor prefetcher requires block_count ({}) == ring_size ({}) for tensor {}",
                input.block_count,
                total_receivers,
                tensor_idx);
            TT_FATAL(
                input.rotation.size() == total_receivers,
                "Streaming rotation for tensor {} has {} entries but must have total_receivers ({}); it is indexed "
                "by global receiver position.",
                tensor_idx,
                input.rotation.size(),
                total_receivers);
            for (uint32_t v : input.rotation) {
                TT_FATAL(
                    v < input.block_count,
                    "Streaming rotation entry {} for tensor {} is out of range [0, block_count={}).",
                    v,
                    tensor_idx,
                    input.block_count);
            }
        }
        // block_count is per-tensor: it sets how many K-blocks the kernel pushes
        // (and how K is divided in compute_tensor_layout).
        TensorPrefetcherTensorLayout layout = compute_tensor_layout(
            input.tensor.get(),
            input.block_count,
            num_banks_,
            receivers_per_bank,
            total_receivers,
            ring_half_,
            stage_third_,
            context_id);
        // Streaming is a per-tensor delivery attribute carried in the layout flag; the appended
        // rotation participates in dedup (slot_equal), so tensors that differ only in rotation get
        // distinct slots.
        layout.streaming = streaming ? 1u : 0u;
        if (receiver_contiguous_batched_only) {
            require_receiver_slabs_in_tensor_and_view(
                input.tensor.get(),
                layout,
                mapping,
                target.recv_index_base_per_sender,
                mesh_device_->impl().metal_env().get_cluster().get_soc_desc(devices_.front()->id()).dram_view_size,
                static_cast<uint32_t>(tensor_idx));
        }
        // One read of a K-row-major bank shard feeds every receiver of that bank, so its sender has to
        // own them all, and every bank's shard is sliced by the same receiver count.
        TT_FATAL(
            layout.layout_mode != static_cast<uint32_t>(LayoutMode::KRowMajor) || krow_compatible_mapping,
            "Tensor prefetcher: K-row-major input tensor {} requires exactly one primary sender per DRAM bank, each "
            "driving {} receivers (total receivers / banks); the supplied target uses a split, uneven or incompatible "
            "sender topology. Create the target with support_multi_receiver_shards=true over the same number of "
            "receivers on every bank.",
            tensor_idx,
            receivers_per_bank);

        // The sender's free-space poll counts whole per-receiver pages; if the target's per-receiver
        // ring can't hold even one full page the poll never reaches a usable block and the DRISC
        // kernel hangs. Guard it here (applies to both layouts).
        TT_FATAL(
            target.per_recv_capacity_bytes >= layout.page_bytes_per_recv,
            "Tensor prefetcher: target per-receiver capacity ({} B) must be at least one full per-receiver "
            "page ({} B) for input tensor {}; a smaller ring makes the sender's free-space poll spin forever.",
            target.per_recv_capacity_bytes,
            layout.page_bytes_per_recv,
            tensor_idx);

        Slot slot;
        slot.geom = layout;
        if (streaming) {
            TT_FATAL(
                layout.layout_mode == static_cast<uint32_t>(LayoutMode::ReceiverContiguous),
                "Streaming Tensor prefetcher requires a receiver-contiguous weight, but input tensor {} is "
                "K-row-major.",
                tensor_idx);
            slot.rotation = input.rotation;
            slot.strategy = shard_strategy_for_streaming_tensor(input.tensor.get(), static_cast<uint32_t>(tensor_idx));
        }

        // Find this slot in the current page (dedup), or decide it needs adding.
        PagePlan* plan = &plans.back();
        int32_t slot_idx = -1;
        for (uint32_t i = 0; i < plan->slots.size(); ++i) {
            if (slot_equal(plan->slots[i], slot)) {
                slot_idx = static_cast<int32_t>(i);
                break;
            }
        }
        const uint32_t need = kEntryBytes + (slot_idx < 0 ? layout_stride : 0);
        const uint32_t entry_high = kHeaderBytes + static_cast<uint32_t>(plan->entries.size()) * kEntryBytes;
        const uint32_t layout_low = kRequestPageBytes - static_cast<uint32_t>(plan->slots.size()) * layout_stride;
        if (need > layout_low - entry_high) {
            // No room in the current page — start a fresh one. The slot is page-local, so it
            // becomes a new slot in the next page.
            plans.emplace_back();
            plan = &plans.back();
            slot_idx = -1;
        }
        if (slot_idx < 0) {
            slot_idx = static_cast<int32_t>(plan->slots.size());
            plan->slots.push_back(std::move(slot));
        }
        PlanEntry entry{
            .bank_local_base = static_cast<uint32_t>(input.tensor.get().mesh_buffer().address()),
            .layout_index = static_cast<uint32_t>(slot_idx)};
        if (layout.group_stride_bytes != 0) {
            const uint32_t num_groups = layout.recv_stride_bytes / layout.group_stride_bytes;
            TT_FATAL(
                num_groups <= std::numeric_limits<uint16_t>::max(),
                "Tensor prefetcher input {} stacks {} groups per receiver slab; at most {} are supported",
                tensor_idx,
                num_groups,
                std::numeric_limits<uint16_t>::max());
            entry.num_groups = static_cast<uint16_t>(num_groups);
        }
        TT_FATAL(
            !receiver_contiguous_batched_only || (entry.num_groups == 1 && !input.group_selector.has_value()),
            "Tensor prefetcher on {} CCE senders delivers ungrouped tensors without a group selector, but input tensor "
            "{} stacks {} groups{}",
            mesh_device_->arch(),
            tensor_idx,
            entry.num_groups,
            input.group_selector.has_value() ? " and has a group selector" : "");
        if (input.group_selector.has_value()) {
            TT_FATAL(
                layout.group_stride_bytes != 0,
                "Tensor prefetcher input {} has a group selector but is not a grouped weight: give it a "
                "receiver-contiguous NdShardSpec shard of (num_groups, K, N / ring_size)",
                tensor_idx);
            entry.selector_mode = TENSOR_PREFETCHER_SELECTOR_NONZERO_MASK;
            entry.selector = resolve_group_selector(
                *mesh_device_, devices_, input.group_selector->mask.get(), entry.num_groups, tensor_idx);
        }
        plan->entries.push_back(entry);
    }

    // ---- Materialize each logical page into one byte buffer per sender ----
    // Entry and geometry bytes are identical across senders. The header is not -- it names this
    // sender's target state -- and neither is a slot's recv_index_base (this sender's bank-local
    // slab base) or its rotation region (this sender's slice of the caller's global rotation).
    std::vector<std::vector<std::vector<uint8_t>>> pages;
    pages.reserve(plans.size());
    for (const auto& plan : plans) {
        bool page_has_rotation = false;
        for (const auto& slot : plan.slots) {
            if (!slot.rotation.empty()) {
                page_has_rotation = true;
                break;
            }
        }
        // Build the sender-independent template once (entries + each slot's geometry, header's
        // shared fields, slab bases and rotation regions left zero); each sender's page is a copy
        // with only its own target address, slab bases and rotation slices overwritten.
        std::vector<uint8_t> templ(aligned_page_bytes, 0);
        auto* header = reinterpret_cast<TensorPrefetcherRequestHeader*>(templ.data());
        header->base.cmd_id = DRAM_PREFETCHER_CMD_PREFETCH;
        header->prefetch.num_entries = static_cast<uint16_t>(plan.entries.size());
        header->prefetch.num_layouts = static_cast<uint16_t>(plan.slots.size());
        for (uint32_t k = 0; k < plan.entries.size(); ++k) {
            TensorPrefetcherEntry entry;
            entry.bank_local_base = plan.entries[k].bank_local_base;
            entry.layout_offset = slot_offset(plan.entries[k].layout_index);
            entry.selector_noc_xy = plan.entries[k].selector.noc_xy;
            entry.selector_addr = plan.entries[k].selector.addr;
            entry.selector_location = plan.entries[k].selector.location;
            entry.num_groups = plan.entries[k].num_groups;
            entry.selector_mode = plan.entries[k].selector_mode;
            std::memcpy(templ.data() + (kHeaderBytes + k * kEntryBytes), &entry, kEntryBytes);
        }
        for (uint32_t i = 0; i < plan.slots.size(); ++i) {
            std::memcpy(templ.data() + slot_offset(i), &plan.slots[i].geom, kLayoutBytes);
        }

        std::vector<std::vector<uint8_t>> per_sender(mapping.size());
        for (uint32_t s = 0; s < mapping.size(); ++s) {
            std::vector<uint8_t> page = templ;
            auto* sender_header = reinterpret_cast<TensorPrefetcherRequestHeader*>(page.data());
            sender_header->prefetch.target_config_page_addr = target.config_page_addr_per_sender[s];
            // This sender's bank-local slab base is per-sender but not per-tensor, so it goes into
            // every slot of this sender's page: the kernel reads it beside the geometry the entry
            // points at and needs no second lookup.
            const uint32_t slab_base = target.recv_index_base_per_sender[s];
            for (uint32_t i = 0; i < plan.slots.size(); ++i) {
                std::memcpy(
                    page.data() + slot_offset(i) + offsetof(TensorPrefetcherTensorLayout, recv_index_base),
                    &slab_base,
                    sizeof(slab_base));
            }
            if (page_has_rotation) {
                const uint32_t num_local_receivers = mapping[s].second.num_cores();
                const uint32_t bank = static_cast<uint32_t>(mapping[s].first.x);
                for (uint32_t i = 0; i < plan.slots.size(); ++i) {
                    if (plan.slots[i].rotation.empty()) {
                        continue;
                    }
                    // Map each receiver's (bank, bank-local slab index) to its global receiver
                    // position, then gather this sender's slice of the caller's global rotation. The
                    // topology guard above makes both formulas a bijection onto [0, total_receivers),
                    // so no inner-loop range check is needed. Only ROUND_ROBIN_1D (strided) and
                    // CONTIGUOUS_1D reach here (see shard_strategy_for_streaming_tensor).
                    const bool strided = plan.slots[i].strategy == ShardDistributionStrategy::ROUND_ROBIN_1D;
                    auto* rot = reinterpret_cast<uint32_t*>(page.data() + slot_offset(i) + kLayoutBytes);
                    for (uint32_t r = 0; r < num_local_receivers; ++r) {
                        const uint32_t slab = slab_base + r;
                        const uint32_t g = strided ? (bank + slab * num_banks_) : (bank * receivers_per_bank + slab);
                        rot[r] = plan.slots[i].rotation[g];
                    }
                }
            }
            per_sender[s] = std::move(page);
        }
        pages.push_back(std::move(per_sender));
    }

    return pages;
}

std::vector<std::shared_ptr<const TensorPrefetcherManager::Request>> TensorPrefetcherManager::build_requests(
    const RequestTarget& target,
    ttsl::optional_reference<const MeshCoordinateRangeSet> device_subset,
    const std::vector<experimental::TensorPrefetcherInput>& tensors) {
    const std::vector<uint32_t> target_sender_indices = sender_indices_for_target(target);
    if (uses_cce_senders()) {
        require_receivers_off_sender_cces(target);
    }
    TT_FATAL(!tensors.empty(), "QueueTensorPrefetcherRequest requires at least one tensor");
    // A Queue call may span more tensors than fit in one socket page; serialize into one
    // or more pages, each an independent request. The per-target write cursor persists across
    // requests, so the split is invisible to the receiver. Each logical page is materialized once
    // per mapped sender, since its header names that sender's target state and slab base.
    std::vector<std::vector<std::vector<uint8_t>>> pages = serialize_request_pages(target, tensors);
    for (const auto& sender_pages : pages) {
        validate_request_pages(sender_pages);
        if (on_dispatch_engine()) {
            validate_dispatch_engine_pages(sender_pages, target);
        }
    }

    // Target devices: subset if given, else full mesh. Caller is responsible
    // for keeping tensors and the target alive until stop() — see the public API doc.
    std::vector<MeshCoordinate> target_devices;
    MeshCoordinateRangeSet effective_subset = device_subset.has_value() ? *device_subset : full_mesh_subset();
    for (const auto& range : effective_subset.ranges()) {
        for (const auto& coord : range) {
            TT_FATAL(
                device_index_by_coord_.contains(coord),
                "QueueTensorPrefetcherRequest target MeshCoordinate {} is not in the mesh this prefetcher was "
                "started "
                "on",
                coord);
            target_devices.push_back(coord);
        }
    }

    // Remember this target on every sender it maps to, so stop can drain it. The request has passed
    // every other check by now, and a request only captured into a trace or a command list counts too:
    // a replay sends it.
    record_drain_targets(target, target_sender_indices);

    std::vector<std::shared_ptr<const Request>> requests;
    requests.reserve(pages.size());
    for (auto& sender_pages : pages) {
        Request req;
        req.sender_pages = std::move(sender_pages);
        req.target_devices = target_devices;
        req.target_sender_indices = target_sender_indices;
        requests.push_back(std::make_shared<const Request>(std::move(req)));
    }
    return requests;
}

void TensorPrefetcherManager::queue(
    const std::vector<std::reference_wrapper<const experimental::PrefetcherPipe>>& prefetcher_pipes,
    ttsl::optional_reference<const MeshCoordinateRangeSet> device_subset,
    const std::vector<experimental::TensorPrefetcherInput>& tensors,
    MeshCommandQueue* trace_capture_cq) {
    // target_for validates the list: banks in contiguous runs appearing once each, one to two DRAM
    // senders per bank and in sender order, one shared geometry, and disjoint receivers.
    const RequestTarget target = target_for(prefetcher_pipes);
    auto lock = lock_api_function_();
    TT_FATAL(active_, "QueueTensorPrefetcherRequest called before StartTensorPrefetcher");
    validate_own_cq(trace_capture_cq, "QueueTensorPrefetcherRequest");
    send_or_capture(build_requests(target, device_subset, tensors), trace_capture_cq);
}

bool TensorPrefetcherManager::on_dispatch_engine() const { return dram_senders_on_dispatch_engine(*mesh_device_); }

void TensorPrefetcherManager::set_stage_ring_size(uint32_t raw_bytes, uint32_t l1_alignment) {
    // Align down to LCM(2, 3, l1_alignment) so both halves (K-row path) and thirds (recv-contig path)
    // are individually l1-aligned: 6 * l1_alignment when l1_alignment is even (48 B at 16 B). Not a
    // power of two, so integer division rather than a mask.
    const uint32_t ring_size_align = 6u * l1_alignment;
    stage_ring_size_ = raw_bytes / ring_size_align * ring_size_align;
    ring_half_ = stage_ring_size_ / 2;
    stage_third_ = stage_ring_size_ / 3;
}

void TensorPrefetcherManager::validate_own_cq(const MeshCommandQueue* cq, std::string_view api_name) const {
    // Only reached with a non-null queue: the null case short-circuits, so the message may
    // dereference it (TT_FATAL evaluates its arguments only when the condition fails).
    TT_FATAL(
        cq == nullptr || cq->device() == mesh_device_,
        "{} was given command queue {} of mesh device {}, but this prefetcher was started on mesh device {}. Pass a "
        "command queue of the prefetcher's own mesh device.",
        api_name,
        cq->id(),
        cq->device()->id(),
        mesh_device_->id());
}

void TensorPrefetcherManager::send_or_capture(
    const std::vector<std::shared_ptr<const Request>>& requests, MeshCommandQueue* trace_capture_cq) {
    // Engaged only when the caller named a queue that is mid trace-capture; that is what
    // routes the requests into the trace instead of out via the host worker.
    const std::optional<MeshTraceId> recording_trace_id =
        trace_capture_cq != nullptr ? trace_capture_cq->trace_id() : std::nullopt;

    {
        // Push all requests of one call under one lock so they stay contiguous and ordered
        // (required for fifo_wr_ptr continuity across a split request), whether captured or sent.
        std::lock_guard<std::mutex> lk(queue_mu_);
        if (recording_trace_id.has_value()) {
            auto& captured = trace_requests_[*recording_trace_id];
            captured.insert(captured.end(), requests.begin(), requests.end());
        } else {
            pending_.insert(pending_.end(), requests.begin(), requests.end());
            has_pending_.store(true);
        }
    }
    // Only the immediate path needs to wake the worker; captured requests wait for replay.
    if (!recording_trace_id.has_value()) {
        queue_cv_.notify_one();
    }
}

void TensorPrefetcherManager::require_receivers_off_sender_cces(const RequestTarget& target) const {
    const Cluster& cluster = mesh_device_->impl().metal_env().get_cluster();
    // Logical-to-CCE resolution follows each device's DRAM harvesting, so every device is checked.
    for (const IDevice* device : devices_) {
        const metal_SocDescriptor& soc_desc = cluster.get_soc_desc(device->id());
        std::unordered_set<uint32_t> sender_cces;
        for (const CoreCoord& sender_logical : sender_logical_cores_) {
            sender_cces.insert(package_dram_core_index(soc_desc, sender_logical));
        }
        for (const std::pair<CoreCoord, CoreRangeSet>& sender_and_receivers : target.mapping) {
            for (const CoreCoord& receiver : corerange_to_cores(sender_and_receivers.second)) {
                const uint32_t receiver_cce = package_dram_core_index(soc_desc, receiver);
                TT_FATAL(
                    !sender_cces.contains(receiver_cce),
                    "QueueTensorPrefetcherRequest: receiver CCE {} (logical {}) on device {} is one of this "
                    "prefetcher's provisioned sender CCEs, so its receiver Program cannot run while the prefetcher "
                    "does. Leave its bank out of QuasarTensorPrefetcherConfig::dram_sender_banks.",
                    receiver_cce,
                    receiver.str(),
                    device->id());
            }
        }
    }
}

void TensorPrefetcherManager::register_drain_target(
    const std::vector<std::reference_wrapper<const experimental::PrefetcherPipe>>& prefetcher_pipes) {
    // Validated as queue() validates them, and outside the API lock for the same reason.
    const RequestTarget target = target_for(prefetcher_pipes);
    const std::lock_guard<std::mutex> lock = lock_api_function_();
    TT_FATAL(active_, "register_drain_target called before StartTensorPrefetcher");
    const std::vector<uint32_t> target_sender_indices = sender_indices_for_target(target);
    if (uses_cce_senders()) {
        require_receivers_off_sender_cces(target);
    }
    record_drain_targets(target, target_sender_indices);
}

void TensorPrefetcherManager::record_drain_targets(
    const RequestTarget& target, const std::vector<uint32_t>& target_sender_indices) {
    for (size_t m = 0; m < target_sender_indices.size(); ++m) {
        record_drain_target(
            target_sender_indices[m], target.config_page_addr_per_sender[m], target.config_page_alloc_per_sender[m]);
    }
}

void TensorPrefetcherManager::record_drain_target(
    uint32_t sender, uint32_t config_page_addr, const std::weak_ptr<DriscL1Allocation>& allocation) {
    auto& targets = drain_targets_per_sender_[sender];
    std::erase_if(targets, [](const DrainTarget& t) { return t.allocation.expired(); });
    const auto it = std::find_if(
        targets.begin(), targets.end(), [&](const DrainTarget& t) { return t.config_page_addr == config_page_addr; });
    if (it == targets.end()) {
        targets.push_back(DrainTarget{config_page_addr, allocation});
    }
}

std::vector<TensorPrefetcherManager::Request> TensorPrefetcherManager::build_drain_requests() {
    std::vector<std::vector<uint32_t>> pages_per_sender(num_senders_);
    size_t max_pages = 0;
    for (uint32_t s = 0; s < num_senders_; ++s) {
        for (const DrainTarget& t : drain_targets_per_sender_[s]) {
            // Held for the rest of stop, so the range cannot be released and reused between this check
            // and the kernel reading it.
            if (auto alive = t.allocation.lock()) {
                pages_per_sender[s].push_back(t.config_page_addr);
                drain_holds_.push_back(std::move(alive));
            }
        }
        max_pages = std::max(max_pages, pages_per_sender[s].size());
    }

    const uint32_t pcie_alignment = mesh_device_->impl().metal_env().get_hal().get_alignment(HalMemType::HOST);
    const uint32_t page_bytes = align_up(kRequestPageBytes, pcie_alignment);
    const MeshCoordinateRangeSet full_subset = full_mesh_subset();
    std::vector<MeshCoordinate> all_devices;
    for (const auto& range : full_subset.ranges()) {
        for (const auto& coord : range) {
            all_devices.push_back(coord);
        }
    }

    // Page p of every sender that has more than p pages' worth of targets, so a sender with many
    // targets gets several DRAIN pages and one with none gets none.
    std::vector<Request> requests;
    for (size_t first = 0; first < max_pages; first += kDrainTargetsPerPage) {
        Request req;
        req.target_devices = all_devices;
        for (uint32_t s = 0; s < num_senders_; ++s) {
            const auto& config_pages = pages_per_sender[s];
            if (config_pages.size() <= first) {
                continue;
            }
            const size_t count = std::min<size_t>(kDrainTargetsPerPage, config_pages.size() - first);
            std::vector<uint8_t> page(page_bytes, 0);
            auto* header = reinterpret_cast<TensorPrefetcherRequestHeader*>(page.data());
            header->base.cmd_id = DRAM_PREFETCHER_CMD_DRAIN;
            header->drain.num_targets = static_cast<uint16_t>(count);
            std::memcpy(
                page.data() + sizeof(TensorPrefetcherRequestHeader),
                config_pages.data() + first,
                count * sizeof(uint32_t));
            req.sender_pages.push_back(std::move(page));
            req.target_sender_indices.push_back(s);
        }
        requests.push_back(std::move(req));
    }
    return requests;
}

void TensorPrefetcherManager::validate_request_pages(const std::vector<std::vector<uint8_t>>& sender_pages) const {
    // The transport gives every sender the same page size; a dispatch-engine FIFO lane holds one page.
    const uint32_t transport_page_bytes = on_dispatch_engine() ? kRequestPageBytes : transport_->page_bytes();
    for (const auto& page : sender_pages) {
        TT_FATAL(
            page.size() == transport_page_bytes,
            "Tensor prefetcher request page is {} B, but its socket's pages are {} B",
            page.size(),
            transport_page_bytes);
    }
}

void TensorPrefetcherManager::capture(
    CapturedRequests& captured,
    const std::vector<std::reference_wrapper<const experimental::PrefetcherPipe>>& prefetcher_pipes,
    ttsl::optional_reference<const MeshCoordinateRangeSet> device_subset,
    const std::vector<experimental::TensorPrefetcherInput>& tensors) {
    const RequestTarget target = target_for(prefetcher_pipes);
    auto lock = lock_api_function_();
    TT_FATAL(active_, "CommandListBuilder::add_tensor_prefetcher_request called before StartTensorPrefetcher");
    TT_FATAL(
        captured.requests.empty() || captured.run_id == run_id_,
        "CommandListBuilder::add_tensor_prefetcher_request: the builder holds requests from a Tensor prefetcher that "
        "has since been stopped. Call clear() on the builder after StopTensorPrefetcher.");
    std::vector<std::shared_ptr<const Request>> requests = build_requests(target, device_subset, tensors);
    captured.run_id = run_id_;
    captured.requests.insert(captured.requests.end(), requests.begin(), requests.end());
}

void TensorPrefetcherManager::replay_captured_nolock(const CapturedRequests& captured) {
    if (captured.requests.empty()) {
        return;
    }
    TT_FATAL(
        active_ && captured.run_id == run_id_,
        "This command list records Tensor prefetcher requests, but the Tensor prefetcher they were recorded for "
        "is no longer running. Rebuild the command list after StartTensorPrefetcher.");
    {
        std::lock_guard<std::mutex> lk(queue_mu_);
        // Copy the handles, not the requests, so they survive for the next replay. Pushed in capture order so
        // fifo_wr_ptr continuity matches the original calls.
        pending_.insert(pending_.end(), captured.requests.begin(), captured.requests.end());
        has_pending_.store(true);
    }
    queue_cv_.notify_one();
}

void TensorPrefetcherManager::replay_trace(const MeshTraceId& trace_id) {
    {
        std::lock_guard<std::mutex> lk(queue_mu_);
        auto it = trace_requests_.find(trace_id);
        if (it == trace_requests_.end()) {
            // No prefetcher requests were captured during this trace's capture.
            return;
        }
        // Copy the handles (not the requests) so the captured requests survive for the next replay.
        // Pushed in capture order so fifo_wr_ptr continuity matches the original Queue calls.
        pending_.insert(pending_.end(), it->second.begin(), it->second.end());
        has_pending_.store(true);
    }
    queue_cv_.notify_one();
}

std::vector<uint8_t> TensorPrefetcherManager::wait_signal_page(uint32_t slot_index, uint32_t wait_value) const {
    const uint32_t pcie_alignment = mesh_device_->impl().metal_env().get_hal().get_alignment(HalMemType::HOST);
    std::vector<uint8_t> page(align_up(kRequestPageBytes, pcie_alignment), 0);
    auto* header = reinterpret_cast<TensorPrefetcherRequestHeader*>(page.data());
    header->base.cmd_id = DRAM_PREFETCHER_CMD_WAIT_SIGNAL;
    header->wait_signal.slot_index = static_cast<uint8_t>(slot_index);
    header->wait_signal.wait_value = wait_value;
    return page;
}

void TensorPrefetcherManager::add_op_signal_wait_page(std::shared_ptr<const Request>& request) {
    if (!request->op_signal.has_value()) {
        return;
    }
    // The n-th wait on a signal handed to the worker since start waits for the count the device reaches with the
    // signal that pairs with it, n times the increments each signal makes; the count wraps with the slot, and
    // the kernel's compare is wrap-safe. Assigned here rather than at queue time, so a wait captured into a trace
    // waits for a new count on every replay. The page goes on a copy because a captured request is shared with
    // every replay; a wait carries no other page, so the copy is cheap.
    const uint32_t signal_id = *request->op_signal;
    auto with_page = std::make_shared<Request>(*request);
    with_page->sender_pages.push_back(
        wait_signal_page(op_signal_slot_index(signal_id), ++op_signal_waits_[signal_id] * op_signal_increments_));
    request = std::move(with_page);
}

void TensorPrefetcherManager::validate_signal_id(uint32_t signal_id, std::string_view api_name) {
    TT_FATAL(
        signal_id < experimental::kTensorPrefetcherNumSignals,
        "{}: signal id {} is out of range [0, {})",
        api_name,
        signal_id,
        experimental::kTensorPrefetcherNumSignals);
}

uint32_t TensorPrefetcherManager::signal_slot_addr(uint32_t slot_index) const {
    return signal_slots_l1_addr_ + slot_index * signal_slot_stride_;
}

uint32_t TensorPrefetcherManager::signal_address(uint32_t signal_id) {
    auto lock = lock_api_function_();
    TT_FATAL(active_, "GetTensorPrefetcherSignalAddress called before StartTensorPrefetcher");
    validate_signal_id(signal_id, "GetTensorPrefetcherSignalAddress");
    // The control block holds the CQ fence slots only: op signal slot 0's address is its ring meta.
    TT_FATAL(
        !uses_cce_senders(),
        "GetTensorPrefetcherSignalAddress: the {} CCE senders keep no op signal slots, so op signal {} has no address",
        mesh_device_->arch(),
        signal_id);
    return signal_slot_addr(op_signal_slot_index(signal_id));
}

void TensorPrefetcherManager::queue_wait_for_signal(uint32_t signal_id, MeshCommandQueue* trace_capture_cq) {
    auto lock = lock_api_function_();
    TT_FATAL(active_, "QueueTensorPrefetcherWaitForSignal called before StartTensorPrefetcher");
    validate_signal_id(signal_id, "QueueTensorPrefetcherWaitForSignal");
    validate_own_cq(trace_capture_cq, "QueueTensorPrefetcherWaitForSignal");
    TT_FATAL(
        !uses_cce_senders(),
        "QueueTensorPrefetcherWaitForSignal: the {} CCE senders wait on CQ fence slots only, not on op signal {}",
        mesh_device_->arch(),
        signal_id);

    // Every sender of every device waits: a signal increments the slot on all of them, and the per-signal
    // count of waits is only meaningful if every device sees every wait. worker_loop builds the page.
    Request req;
    req.target_devices = full_mesh_subset().coords();
    req.op_signal = signal_id;
    send_or_capture({std::make_shared<const Request>(std::move(req))}, trace_capture_cq);
}

void TensorPrefetcherManager::enqueue_cq_signal_and_wait(
    MeshCommandQueue& cq, ttsl::optional_reference<const MeshCoordinateRangeSet> device_subset) {
    // Hold the API lock across this whole call. Three things must be atomic together:
    //   1. the counter bump (++cq_signal_counter_[cq_id]),
    //   2. the dispatcher write that pushes that value to the device, and
    //   3. the WAIT_SIGNAL enqueue into pending_.
    // If the lock were dropped between them, two concurrent callers could interleave their
    // dispatcher writes out of counter order, and stop() could slip its STOP sentinel into
    // pending_ ahead of this WAIT_SIGNAL request — worker_loop would then send the
    // WAIT_SIGNAL to a kernel that has already exited and spin forever (its sends have no
    // stop_requested_ check). queue() and stop() take the lock the same way, so all three
    // serialize. enqueue_write_dram_core_counter is documented to run under the caller's
    // api lock and does NOT re-lock, so holding it here does not self-deadlock.
    auto lock = lock_api_function_();
    TT_FATAL(active_, "WaitForCqOnTensorPrefetcher called before StartTensorPrefetcher");
    enqueue_cq_signal_and_wait_locked(cq, device_subset);
}

bool TensorPrefetcherManager::enqueue_cq_signal_and_wait_if_active(MeshCommandQueue& cq) {
    auto lock = lock_api_function_();
    if (!active_) {
        return false;
    }
    enqueue_cq_signal_and_wait_locked(cq, std::nullopt);
    return true;
}

void TensorPrefetcherManager::enqueue_cq_signal_and_wait_locked(
    MeshCommandQueue& cq, ttsl::optional_reference<const MeshCoordinateRangeSet> device_subset) {
    validate_own_cq(&cq, "WaitForCqOnTensorPrefetcher");
    const uint32_t cq_id = cq.id();
    TT_FATAL(
        cq_id < cq_signal_counter_.size(),
        "WaitForCqOnTensorPrefetcher command queue id ({}) out of range [0, {})",
        cq_id,
        cq_signal_counter_.size());

    // Monotonic value for this signal; wrap is handled by the kernel's signed compare.
    const uint32_t signal_value = ++cq_signal_counter_[cq_id];

    // Resolve target device coords (subset if given, else full mesh).
    std::vector<MeshCoordinate> target_devices;
    const MeshCoordinateRangeSet effective_subset = device_subset.has_value() ? *device_subset : full_mesh_subset();
    for (const auto& range : effective_subset.ranges()) {
        for (const auto& coord : range) {
            TT_FATAL(
                device_index_by_coord_.contains(coord),
                "WaitForCqOnTensorPrefetcher target MeshCoordinate {} is not in the mesh this prefetcher was "
                "started on",
                coord);
            target_devices.push_back(coord);
        }
    }

    // Destination of the dispatcher write: the full device address of the CQ's fence slot, so the dispatcher
    // must NOT apply a bank offset. Every core that keeps a signal slot table gets the write: each sender's
    // DRAM core, or the one dispatch engine that keeps the table for all of them.
    const uint64_t slot_addr = dram_sender_l1_noc_addr(*mesh_device_, signal_slot_addr(cq_id));

    std::vector<DeviceMemoryAddress> targets;
    targets.reserve(target_devices.size() * num_senders_);
    for (const auto& coord : target_devices) {
        IDevice* device = devices_[device_index_by_coord_.at(coord)];
        const size_t device_targets_begin = targets.size();
        for (uint32_t s = 0; s < num_senders_; ++s) {
            // Per-device translation; see metal_SocDescriptor::dram_bank_endpoint_coords.
            const CoreCoord virtual_core = dram_sender_virtual_core(*mesh_device_, device, sender_logical_cores_[s]);
            const bool already_targeted = std::any_of(
                targets.begin() + device_targets_begin, targets.end(), [&](const DeviceMemoryAddress& target) {
                    return target.virtual_core_coord == virtual_core;
                });
            if (!already_targeted) {
                targets.push_back(DeviceMemoryAddress{coord, virtual_core, slot_addr});
            }
        }
    }

    // (a) Dispatcher write: bump every target DRAM core's signal slot for this CQ. Runs
    // under the api lock we already hold (the method does not re-lock). Re-resolving by id is
    // what gets the queue as a MeshCommandQueueBase, which is what exposes
    // enqueue_write_dram_core_counter; the identity check is what makes the round trip safe.
    // It holds for every queue a mesh device owns, so a lookup that lands anywhere else fails
    // here instead of fencing a queue the caller never named.
    auto& cq_base = mesh_device_->impl().mesh_command_queue_base(static_cast<uint8_t>(cq_id));
    TT_FATAL(
        &cq_base == &cq,
        "WaitForCqOnTensorPrefetcher was given a command queue reporting id {}, but that is not mesh device {}'s "
        "command queue for that id. Fence against a command queue obtained from the prefetcher's own mesh device.",
        cq_id,
        mesh_device_->id());
    cq_base.enqueue_write_dram_core_counter(
        ttsl::Span<const DeviceMemoryAddress>(targets), signal_value, /*blocking=*/false);

    // (b) Queue a WAIT_SIGNAL request on the CQ's fence slot. It rides the same async worker path as
    // prefetch requests, so it reaches each sender ahead of the next prefetch request; the kernel
    // blocks on it until it observes signal_value. Never captured, as the dispatcher write above
    // cannot be.
    queue_wait_cq_request(cq_id, signal_value, std::move(target_devices));
}

void TensorPrefetcherManager::queue_wait_cq_request(
    uint32_t cq_id, uint32_t signal_value, std::vector<MeshCoordinate> target_devices) {
    Request req;
    req.sender_pages.push_back(wait_signal_page(cq_id, signal_value));
    req.target_devices = std::move(target_devices);
    validate_request_pages(req.sender_pages);
    send_or_capture({std::make_shared<const Request>(std::move(req))}, /*trace_capture_cq=*/nullptr);
}

TensorPrefetcherManager::UnsignaledCqWait TensorPrefetcherManager::queue_cq_wait_without_signal() {
    // Under the api lock for the reason enqueue_cq_signal_and_wait gives: stop() cannot slip its STOP ahead of this
    // wait.
    const std::lock_guard<std::mutex> lock = lock_api_function_();
    TT_FATAL(active_, "queue_cq_wait_without_signal called before StartTensorPrefetcher");
    constexpr uint32_t kCqId = 0;
    const uint32_t signal_value = ++cq_signal_counter_[kCqId];
    std::vector<MeshCoordinate> target_devices;
    const MeshCoordinateRangeSet full_subset = full_mesh_subset();
    for (const MeshCoordinateRange& range : full_subset.ranges()) {
        for (const MeshCoordinate& coord : range) {
            target_devices.push_back(coord);
        }
    }
    queue_wait_cq_request(kCqId, signal_value, std::move(target_devices));
    return UnsignaledCqWait{.slot_address = signal_slot_addr(kCqId), .value = signal_value};
}

void TensorPrefetcherManager::release_trace(const MeshTraceId& trace_id) {
    std::lock_guard<std::mutex> lk(queue_mu_);
    trace_requests_.erase(trace_id);
}

void TensorPrefetcherManager::worker_loop(H2DSocketTransport& sockets) {
    // Pages per socket per write. One write per page costs ~0.47 us of driver overhead per 128 B page;
    // 8 consecutive pages in one write bring that to ~0.12 us a page, and longer runs gain little while
    // holding back the publish that lets the senders start.
    constexpr uint32_t kMaxPagesPerSocketWrite = 8;
    // Requests spread over the per-socket queues per pass. A trace replay hands the worker every request
    // it captured at once; spreading only the next few lets the first pages go out without waiting for
    // the rest of the burst to be spread, while still giving each socket a full write per pass. The bound
    // is per pass rather than on requests not yet fully written: a socket with no room must not hold back
    // other sockets' requests, as its sender's consumers may be waiting on theirs.
    constexpr size_t kMaxRequestsTakenPerPass = 2 * kMaxPagesPerSocketWrite;

    // A request taken off pending_, with the number of its socket pages not yet written.
    struct TakenRequest {
        std::shared_ptr<const Request> request;
        size_t unsent_pages = 0;
    };
    struct UnsentPage {
        const uint8_t* data = nullptr;
        TakenRequest* owner = nullptr;
    };
    // Each socket's unsent pages, in request order. They point into the requests in `taken`, which are
    // released in the order they were taken once all their pages are written. A socket's pages reach its
    // sender in request order; sockets advance independently, as nothing on the device orders one sender's
    // requests against another's.
    std::vector<std::deque<UnsentPage>> unsent(sockets.num_endpoints());
    std::deque<TakenRequest> taken;

    const auto take_request = [&](std::shared_ptr<const Request>&& request) {
        add_op_signal_wait_page(request);
        TakenRequest& taken_request = taken.emplace_back(TakenRequest{.request = std::move(request)});
        const Request& req = *taken_request.request;
        // PREFETCH targets only the sender cores of its pipes. STOP / WAIT_SIGNAL / DRAIN leave
        // target_sender_indices empty and broadcast to every provisioned sender.
        const bool fanout_all_senders = req.target_sender_indices.empty();
        TT_FATAL(!req.sender_pages.empty(), "Tensor prefetcher worker received a request with no socket page");
        TT_FATAL(
            fanout_all_senders || req.sender_pages.size() == 1 ||
                req.sender_pages.size() == req.target_sender_indices.size(),
            "Tensor prefetcher request has {} sender pages for {} target senders",
            req.sender_pages.size(),
            req.target_sender_indices.size());
        const auto send_page = [&](uint32_t socket_index, const std::vector<uint8_t>& page) {
            unsent[socket_index].push_back(UnsentPage{.data = page.data(), .owner = &taken_request});
            ++taken_request.unsent_pages;
        };
        for (const auto& dev_coord : req.target_devices) {
            auto it = device_index_by_coord_.find(dev_coord);
            TT_FATAL(
                it != device_index_by_coord_.end(),
                "QueueTensorPrefetcherRequest target MeshCoordinate {} is not in the mesh this prefetcher was "
                "started "
                "on; would silently drop the request for that device",
                dev_coord);
            const uint32_t d = it->second;
            if (fanout_all_senders) {
                for (uint32_t s = 0; s < num_senders_; ++s) {
                    send_page(d * num_senders_ + s, req.sender_pages.front());
                }
            } else {
                for (uint32_t page_index = 0; page_index < req.target_sender_indices.size(); ++page_index) {
                    const uint32_t sender_index = req.target_sender_indices[page_index];
                    TT_FATAL(
                        sender_index < num_senders_,
                        "Tensor prefetcher request targets sender index {} but only {} senders are provisioned",
                        sender_index,
                        num_senders_);
                    send_page(
                        d * num_senders_ + sender_index,
                        req.sender_pages[req.sender_pages.size() == 1 ? 0u : page_index]);
                }
            }
        }
    };

    std::deque<std::shared_ptr<const Request>> incoming;
    std::array<const uint8_t*, kMaxPagesPerSocketWrite> run{};
    std::vector<distributed::H2DSocket*> staged;
    staged.reserve(sockets.num_endpoints());
    while (true) {
        // A pass that starts with nothing left to send writes one page per socket, so that a request on its
        // own, or the first of a burst, is published without waiting for the pages behind it.
        const bool starting_idle =
            incoming.empty() && std::ranges::all_of(unsent, [](const auto& pages) { return pages.empty(); });
        if (starting_idle || has_pending_.load()) {
            std::unique_lock<std::mutex> lk(queue_mu_);
            if (starting_idle) {
                queue_cv_.wait(lk, [&] { return !pending_.empty() || stop_requested_.load(); });
                if (pending_.empty()) {
                    // Stop with empty queue → break out, the main thread handles
                    // sentinel broadcast outside the lock.
                    return;
                }
            }
            if (incoming.empty()) {
                incoming.swap(pending_);
            } else {
                std::ranges::move(pending_, std::back_inserter(incoming));
                pending_.clear();
            }
            has_pending_.store(false);
        }
        for (size_t taken_this_pass = 0; taken_this_pass < kMaxRequestsTakenPerPass && !incoming.empty();
             ++taken_this_pass) {
            take_request(std::move(incoming.front()));
            incoming.pop_front();
        }

        // Stage each socket's next pages in one write, then publish them all with one pair of fences.
        const uint32_t max_pages = starting_idle ? 1u : kMaxPagesPerSocketWrite;
        for (size_t socket_index = 0; socket_index < unsent.size(); ++socket_index) {
            auto& pages = unsent[socket_index];
            if (pages.empty()) {
                continue;
            }
            const auto num_pages = static_cast<uint32_t>(std::min<size_t>(max_pages, pages.size()));
            for (uint32_t i = 0; i < num_pages; ++i) {
                run[i] = pages[i].data;
            }
            distributed::H2DSocket& socket = sockets.socket(static_cast<uint32_t>(socket_index));
            const uint32_t num_staged =
                experimental::detail::try_stage(socket, ttsl::Span<const uint8_t* const>(run.data(), num_pages));
            if (num_staged == 0) {
                continue;
            }
            for (uint32_t i = 0; i < num_staged; ++i) {
                --pages[i].owner->unsent_pages;
            }
            pages.erase(pages.begin(), pages.begin() + num_staged);
            staged.push_back(&socket);
        }
        // try_stage has copied every page it wrote, so a request with none left unsent is no longer read.
        while (!taken.empty() && taken.front().unsent_pages == 0) {
            taken.pop_front();
        }
        if (staged.empty()) {
            // No socket had room this pass, so every socket with pages left is back-pressured;
            // yield rather than busy-spinning at 100% CPU until a receiver frees a page.
            std::this_thread::yield();
            continue;
        }
        experimental::detail::publish_staged(staged);
        staged.clear();
    }
}

void TensorPrefetcherManager::worker_loop_request_ring() {
    // Endpoints whose request ring took no page for request_ring_no_progress_limit_: every later page to one is dropped
    // but STOP.
    std::vector<bool> given_up_endpoints(devices_.size() * num_senders_, false);
    // Endpoints whose pages the worker has logged as retained past that limit.
    std::vector<bool> retention_logged(devices_.size() * num_senders_, false);
    while (true) {
        std::shared_ptr<const Request> request;
        {
            std::unique_lock<std::mutex> lk(queue_mu_);
            queue_cv_.wait(lk, [&] { return !pending_.empty() || stop_requested_.load(); });
            if (pending_.empty()) {
                // Stop with empty queue → break out, the main thread handles
                // sentinel broadcast outside the lock.
                return;
            }
            request = std::move(pending_.front());
            pending_.pop_front();
            has_pending_.store(!pending_.empty());
        }
        const Request& req = *request;

        // PREFETCH targets only the sender cores of its pipes. STOP / WAIT_SIGNAL / DRAIN leave
        // target_sender_indices empty and broadcast to every provisioned sender.
        const bool fanout_all_senders = req.target_sender_indices.empty();
        TT_FATAL(!req.sender_pages.empty(), "Tensor prefetcher worker received a request with no request page");
        TT_FATAL(
            fanout_all_senders || req.sender_pages.size() == 1 ||
                req.sender_pages.size() == req.target_sender_indices.size(),
            "Tensor prefetcher request has {} sender pages for {} target senders",
            req.sender_pages.size(),
            req.target_sender_indices.size());
        struct TargetEndpoint {
            uint32_t endpoint_index = 0;
            uint32_t page_index = 0;
        };
        std::vector<TargetEndpoint> remaining_targets;
        const uint32_t senders_per_device =
            fanout_all_senders ? num_senders_ : static_cast<uint32_t>(req.target_sender_indices.size());
        remaining_targets.reserve(req.target_devices.size() * senders_per_device);
        for (const auto& dev_coord : req.target_devices) {
            auto it = device_index_by_coord_.find(dev_coord);
            TT_FATAL(
                it != device_index_by_coord_.end(),
                "QueueTensorPrefetcherRequest target MeshCoordinate {} is not in the mesh this prefetcher was "
                "started on; would silently drop the request for that device",
                dev_coord);
            const uint32_t d = it->second;
            if (fanout_all_senders) {
                for (uint32_t s = 0; s < num_senders_; ++s) {
                    remaining_targets.push_back(TargetEndpoint{(d * num_senders_) + s, 0});
                }
            } else {
                for (uint32_t page_index = 0; page_index < req.target_sender_indices.size(); ++page_index) {
                    const uint32_t sender_index = req.target_sender_indices[page_index];
                    TT_FATAL(
                        sender_index < num_senders_,
                        "Tensor prefetcher request targets sender index {} but only {} senders are provisioned",
                        sender_index,
                        num_senders_);
                    remaining_targets.push_back(TargetEndpoint{
                        (d * num_senders_) + sender_index, req.sender_pages.size() == 1 ? 0u : page_index});
                }
            }
        }

        const std::function<std::string(const TargetEndpoint&)> sender_name = [this](const TargetEndpoint& target) {
            return fmt::format(
                "sender {} on device {}",
                sender_logical_cores_[target.endpoint_index % num_senders_].str(),
                devices_[target.endpoint_index / num_senders_]->id());
        };
        std::erase_if(remaining_targets, [&](const TargetEndpoint& target) {
            const uint8_t command = req.sender_pages[target.page_index].front();
            if (!given_up_endpoints[target.endpoint_index] || command == DRAM_PREFETCHER_CMD_STOP) {
                return false;
            }
            log_warning(
                tt::LogMetal,
                "Tensor prefetcher: dropping a command {} request page for {}; the worker gave up on that sender",
                command,
                sender_name(target));
            return true;
        });
        // A ring that takes none of this request's pages has made no progress since the request was taken.
        const std::chrono::steady_clock::time_point request_taken = std::chrono::steady_clock::now();

        // Round-robin try_write with non-blocking attempts.
        std::vector<TargetEndpoint> still_pending = std::move(remaining_targets);
        while (!still_pending.empty()) {
            std::vector<TargetEndpoint> next_pending;
            next_pending.reserve(still_pending.size());
            for (const TargetEndpoint& target : still_pending) {
                if (!transport_->try_write(target.endpoint_index, req.sender_pages[target.page_index])) {
                    next_pending.push_back(target);
                }
            }
            // If no endpoint drained this pass, every remaining target is back-pressured or gone. A sender
            // that has exited takes no more pages, so drop its page rather than wait for room.
            if (next_pending.size() == still_pending.size()) {
                std::erase_if(next_pending, [this, &req, &sender_name](const TargetEndpoint& target) {
                    if (!transport_->target_exited(target.endpoint_index)) {
                        return false;
                    }
                    log_warning(
                        tt::LogMetal,
                        "Tensor prefetcher: dropping a command {} request page for {}; that sender has exited",
                        req.sender_pages[target.page_index].front(),
                        sender_name(target));
                    dropped_request_pages_.fetch_add(1);
                    return true;
                });
                // A sender frees a ring slot only once it has handled the command in it, so a ring also takes no page
                // while its sender sits in a legitimate credit, CQ or drain wait, which wait_budget_cycles = 0 never
                // bounds. Without an operation timeout, every page stays while the sender may still run: dropped, a
                // page loses its data, and a dropped STOP leaves the sender running for good. With one, the page is
                // dropped, and stop()'s WaitProgramDone times out on that sender.
                const std::chrono::steady_clock::duration without_progress =
                    std::chrono::steady_clock::now() - request_taken;
                if (without_progress >= request_ring_no_progress_limit_) {
                    std::erase_if(next_pending, [&](const TargetEndpoint& target) {
                        const uint8_t command = req.sender_pages[target.page_index].front();
                        if (!give_up_on_running_senders_ &&
                            cce_sender_running(
                                *mesh_device_,
                                devices_[target.endpoint_index / num_senders_],
                                sender_logical_cores_[target.endpoint_index % num_senders_],
                                cce_control_base_)) {
                            if (!retention_logged[target.endpoint_index]) {
                                retention_logged[target.endpoint_index] = true;
                                log_warning(
                                    tt::LogMetal,
                                    "Tensor prefetcher: the request ring of {} took no page for {:.0f} s; keeping its "
                                    "command {} page and later pages while it runs, because "
                                    "TT_METAL_OPERATION_TIMEOUT_SECONDS is unset",
                                    sender_name(target),
                                    std::chrono::duration<float>(without_progress).count(),
                                    command);
                            }
                            return false;
                        }
                        log_error(
                            tt::LogMetal,
                            "Tensor prefetcher: the request ring of {} took no page for {:.0f} s; dropping its command "
                            "{} page and every later page to it but STOP",
                            sender_name(target),
                            std::chrono::duration<float>(without_progress).count(),
                            command);
                        given_up_endpoints[target.endpoint_index] = true;
                        worker_gave_up_on_sender_.store(true);
                        return true;
                    });
                }
                // Pause rather than busy-spin at 100% CPU until a receiver frees a page. Each pass reads sender
                // TL1 over the device link, so the pause is longer than a yield.
                std::this_thread::sleep_for(kRequestRingRetryInterval);
            }
            still_pending = std::move(next_pending);
        }
    }
}

bool TensorPrefetcherManager::uses_cce_senders() const {
    return mesh_device_->arch() == tt::ARCH::QUASAR && !on_dispatch_engine() &&
           prefetcher_traits_for(mesh_device_->arch()).transport == RequestTransportKind::Tl1Ring;
}

void TensorPrefetcherManager::stop() {
    // Note: the MeshDevice close path (close_impl) calls stop() without holding
    // api_mutex_, and the destructor only reaches here after active_ is already
    // false, so taking the lock here never self-deadlocks.
    auto lock = lock_api_function_();
    if (!active_) {
        return;
    }
    // Stop = a zero-filled page broadcast to every device in the mesh. The leading
    // command id byte is 0 == DRAM_PREFETCHER_CMD_STOP, so the kernel exits its
    // request loop on it. The worker_loop returns once pending is drained and
    // stop_requested_ is set.
    Request sentinel;
    const uint32_t pcie_alignment = mesh_device_->impl().metal_env().get_hal().get_alignment(HalMemType::HOST);
    const uint32_t page_bytes = align_up(kRequestPageBytes, pcie_alignment);
    // STOP is all-zero (cmd_id 0) and rotation-free, so one page broadcast to every sender.
    sentinel.sender_pages.assign(1, std::vector<uint8_t>(page_bytes, 0));
    const MeshCoordinateRangeSet full_subset = full_mesh_subset();
    for (const auto& range : full_subset.ranges()) {
        for (const auto& coord : range) {
            sentinel.target_devices.push_back(coord);
        }
    }

    // Drain every target still in memory before STOP: the kernel exits on STOP, and receivers that
    // have not acked yet would otherwise land their acks in DRISC L1 the next program may reuse.
    std::vector<Request> drains = build_drain_requests();
    for (const auto& drain : drains) {
        validate_request_pages(drain.sender_pages);
    }
    validate_request_pages(sentinel.sender_pages);

    {
        std::lock_guard<std::mutex> lk(queue_mu_);
        for (auto& drain : drains) {
            pending_.push_back(std::make_shared<const Request>(std::move(drain)));
        }
        // The dispatch-engine kernels outlive the run (fast dispatch owns them), so they get no STOP.
        if (!on_dispatch_engine()) {
            pending_.push_back(std::make_shared<const Request>(std::move(sentinel)));
        }
        has_pending_.store(true);
        stop_requested_.store(true);
    }
    queue_cv_.notify_all();

    if (host_worker_.joinable()) {
        host_worker_.join();
    }

    // Wait for kernels to drain their request loop (they exit on the sentinel).
    // read_device_profiler_results must be false: we hold the (non-recursive) MeshDevice api
    // lock, and the profiler read reaches enqueue_read_shard_from_core, which takes that same
    // lock. With the device profiler off the read is a no-op, so this only deadlocks under
    // tracy. Nothing is lost by skipping it — the read covers worker/eth cores, not the DRAM
    // cores this program runs on, and the profiler still drains on Finish and at device close.
    std::exception_ptr wait_failure;
    if (on_dispatch_engine()) {
        // The kernels outlive this run (fast dispatch owns them); wait until they have served every
        // request, the drains included, so no receiver is still acking into this run's pipes.
        wait_for_fifo_idle();
    } else {
        // WaitProgramDone throws only when TT_METAL_OPERATION_TIMEOUT_SECONDS is set. A sender that has not
        // exited may still be reading the ranges its DRAIN names, and their receivers may still be acking into
        // them, so those stay held for the manager's lifetime; everything else is released.
        try {
            for (uint32_t d = 0; d < devices_.size(); ++d) {
                ::tt::tt_metal::slow_dispatch::WaitProgramDone(*devices_[d], *programs_[d]);
            }
        } catch (const std::exception& error) {
            wait_failure = std::current_exception();
            log_error(tt::LogMetal, "Tensor prefetcher: waiting for the senders to exit failed: {}", error.what());
            log_cce_sender_statuses();
            drain_holds_after_failed_stop_.insert(
                drain_holds_after_failed_stop_.end(),
                std::make_move_iterator(drain_holds_.begin()),
                std::make_move_iterator(drain_holds_.end()));
        }
    }
    // A sender the worker gave up on may still be running, so its DRAIN ranges stay held as above.
    const bool worker_gave_up_on_sender = worker_gave_up_on_sender_.load();
    if (worker_gave_up_on_sender && !wait_failure) {
        log_cce_sender_statuses();
        drain_holds_after_failed_stop_.insert(
            drain_holds_after_failed_stop_.end(),
            std::make_move_iterator(drain_holds_.begin()),
            std::make_move_iterator(drain_holds_.end()));
    }
    std::vector<std::string> failed_senders;
    if (!worker_gave_up_on_sender && !wait_failure) {
        failed_senders = failed_cce_senders();
    }
    const uint32_t dropped_request_pages = dropped_request_pages_.load();
    active_lifetime_timer_.reset();

    transport_.reset();
    programs_.clear();
    devices_.clear();
    device_index_by_coord_.clear();
    sender_logical_cores_.clear();
    trace_requests_.clear();
    drain_targets_per_sender_.clear();
    drain_holds_.clear();
    num_senders_ = 0;
    num_banks_ = 0;
    fifos_.clear();
    active_ = false;
    if (wait_failure) {
        std::rethrow_exception(wait_failure);
    }
    if (worker_gave_up_on_sender) {
        TT_THROW(
            "Tensor prefetcher: a sender's request ring took no page for {:.0f} s, so the host worker dropped that "
            "sender's later requests; the log names the sender and its status",
            std::chrono::duration<float>(request_ring_no_progress_limit_).count());
    }
    if (!failed_senders.empty()) {
        const std::string failed_sender_list = fmt::format("{}", fmt::join(failed_senders, "; "));
        TT_THROW(
            "Tensor prefetcher: {} sender(s) ended Failed, so their requests were not all delivered ({} request pages "
            "dropped after they exited): {}",
            failed_senders.size(),
            dropped_request_pages,
            failed_sender_list);
    }
}

void TensorPrefetcherManager::stop_without_throwing() {
    // Destroying a joinable std::thread terminates, so the worker is joined however stop() ends.
    struct WorkerJoiner {
        TensorPrefetcherManager& manager;
        ~WorkerJoiner() { manager.join_worker(); }
    };
    const WorkerJoiner worker_joiner{*this};
    try {
        stop();
    } catch (const std::exception& error) {
        log_error(tt::LogMetal, "Tensor prefetcher: stop failed: {}", error.what());
    }
}

void TensorPrefetcherManager::join_worker() {
    if (!host_worker_.joinable()) {
        return;
    }
    {
        const std::lock_guard<std::mutex> queue_lock(queue_mu_);
        stop_requested_.store(true);
    }
    queue_cv_.notify_all();
    host_worker_.join();
}

std::vector<std::string> TensorPrefetcherManager::failed_cce_senders() const {
    std::vector<std::string> failed_senders;
    if (!uses_cce_senders()) {
        return failed_senders;
    }
    for (IDevice* device : devices_) {
        for (const CoreCoord& sender_logical : sender_logical_cores_) {
            try {
                const CceSenderStatus status =
                    read_cce_sender_status(*mesh_device_, device, sender_logical, cce_control_base_);
                if (status.state == static_cast<uint32_t>(CceSenderState::Failed)) {
                    failed_senders.push_back(fmt::format(
                        "sender {} on device {}: {}",
                        sender_logical.str(),
                        device->id(),
                        format_cce_sender_status(status)));
                }
            } catch (const std::exception& error) {
                failed_senders.push_back(fmt::format(
                    "sender {} on device {}: status unreadable: {}", sender_logical.str(), device->id(), error.what()));
            }
        }
    }
    if (!failed_senders.empty()) {
        log_cce_sender_statuses();
    }
    return failed_senders;
}

void TensorPrefetcherManager::log_cce_sender_statuses() const {
    if (!uses_cce_senders()) {
        return;
    }
    for (IDevice* device : devices_) {
        for (const CoreCoord& sender_logical : sender_logical_cores_) {
            try {
                const CceSenderStatus status =
                    read_cce_sender_status(*mesh_device_, device, sender_logical, cce_control_base_);
                log_error(
                    tt::LogMetal,
                    "Tensor prefetcher: sender {} on device {}: {}",
                    sender_logical.str(),
                    device->id(),
                    format_cce_sender_status(status));
            } catch (const std::exception& error) {
                log_error(
                    tt::LogMetal,
                    "Tensor prefetcher: reading the status of sender {} on device {} failed: {}",
                    sender_logical.str(),
                    device->id(),
                    error.what());
            }
        }
    }
}

uint32_t cce_sender_control_base(const MeshDevice& mesh_device) {
    TT_FATAL(
        mesh_device.arch() == tt::ARCH::QUASAR && !dram_senders_on_dispatch_engine(mesh_device),
        "CCE prefetcher senders exist only on Quasar DRAM cores, not on {} {}",
        mesh_device.arch(),
        dram_senders_on_dispatch_engine(mesh_device) ? "dispatch engines" : "devices");
    const KernelRegionCarve carve = carve_kernel_region(
        mesh_device.impl().metal_env().get_hal(),
        mesh_device.impl().drisc_l1_arena(),
        kQuasarTraits,
        experimental::QuasarTensorPrefetcherConfig{});
    TT_FATAL(carve.failure_reason.empty(), "{}", carve.failure_reason);
    return carve.cce_control_base;
}

TensorPrefetcherPreconditions tensor_prefetcher_preconditions(const MeshDevice& mesh_device) {
    const MetalContext& metal = MetalContext::instance(mesh_device.impl().get_context_id());
    // The streaming profiler owns the same DRISCs. A bank's senders are its free (non-endpoint)
    // subchannel and its NOC1 worker endpoint (dram_sender_logical_cores), and a streaming-profiler
    // relay takes that same free subchannel with BOTH of its NIUs in stream mode
    // (set_drisc_niu_stream_mode). Two resident kernels cannot share one DRISC L1, so the profiler
    // wins and the prefetcher reports unsupported rather than racing it for the core.
    if (metal.rtoptions().get_streaming_profiler_enabled()) {
        return {
            .ok = false,
            .reason =
                "the streaming profiler (TT_METAL_STREAMING_PROFILER=1) holds the DRISCs the prefetcher "
                "needs; unset it to use the prefetcher"};
    }
    const std::optional<DramSenderPlacement> placement = mesh_device.impl().dram_sender_placement();
    if (!placement.has_value()) {
        return {
            .ok = false,
            .reason =
                "the device cannot run DRAM-sender endpoints (that takes programmable DRAM cores, which auto-enable "
                "on Blackhole with firmware >= 19.12.0.0 and on Quasar with programmable DRAM views in the SoC "
                "descriptor, or Quasar fast dispatch on a dispatch engine serving one command queue)"};
    }
    // The checks below are the CCE sender's; a dispatch engine's senders need none of them.
    if (*placement == DramSenderPlacement::DispatchEngine || mesh_device.arch() != tt::ARCH::QUASAR) {
        return {.ok = true, .reason = ""};
    }
    const Hal& hal = metal.hal();

    const Cluster& cluster = metal.get_cluster();
    const std::vector<IDevice*> devices = mesh_device.get_devices();
    for (const IDevice* device : devices) {
        const metal_SocDescriptor& soc_desc = cluster.get_soc_desc(device->id());
        const uint32_t num_views = static_cast<uint32_t>(soc_desc.get_num_dram_views());
        std::unordered_map<CoreCoord, uint32_t> view_by_sender_core;
        for (uint32_t view = 0; view < num_views; ++view) {
            if (!soc_desc.is_dram_view_programmable(static_cast<int>(view))) {
                return {
                    .ok = false,
                    .reason = fmt::format("DRAM view {} on device {} has no programmable CCE", view, device->id())};
            }
            for (const CoreCoord& sender_logical : mesh_device.impl().dram_sender_logical_cores(device, view)) {
                const CoreCoord sender_translated = soc_desc.get_physical_dram_core_from_logical(sender_logical);
                if (view_by_sender_core.contains(sender_translated)) {
                    return {
                        .ok = false,
                        .reason = fmt::format(
                            "DRAM views {} and {} on device {} share the sender CCE at translated {}",
                            view_by_sender_core.at(sender_translated),
                            view,
                            device->id(),
                            sender_translated.str())};
                }
                view_by_sender_core.emplace(sender_translated, view);
            }
        }
    }

    if (metal.rtoptions().get_fast_dispatch()) {
        return {.ok = false, .reason = "the CCE prefetcher requires slow dispatch (set TT_METAL_SLOW_DISPATCH_MODE=1)"};
    }

    const KernelRegionCarve carve = carve_kernel_region(
        hal,
        mesh_device.impl().drisc_l1_arena(),
        prefetcher_traits_for(mesh_device.arch()),
        experimental::QuasarTensorPrefetcherConfig{});
    if (!carve.failure_reason.empty()) {
        return {.ok = false, .reason = carve.failure_reason};
    }

    // Device open writes RUN_MSG_INIT and waits for firmware to flip it to RUN_MSG_DONE, so INIT here
    // means the CCE was reset or re-inited since. RUN_MSG_GO is a running Program, which is fine: this
    // checks that firmware is present, not that the CCE is idle.
    for (const IDevice* device : devices) {
        const metal_SocDescriptor& soc_desc = cluster.get_soc_desc(device->id());
        const uint32_t num_views = static_cast<uint32_t>(soc_desc.get_num_dram_views());
        for (uint32_t view = 0; view < num_views; ++view) {
            for (const CoreCoord& sender_logical : mesh_device.impl().dram_sender_logical_cores(device, view)) {
                const uint8_t signal = read_go_message_signal(metal, device, sender_logical);
                if (signal == dev_msgs::RUN_MSG_DONE || signal == dev_msgs::RUN_MSG_GO) {
                    continue;
                }
                const CoreCoord sender_translated = soc_desc.get_physical_dram_core_from_logical(sender_logical);
                const uint32_t cce_index = package_dram_core_index(soc_desc, sender_logical);
                return {
                    .ok = false,
                    .reason = fmt::format(
                        "CCE {} (device {}, DRAM view {}, translated {}) holds GO_MSG signal 0x{:02x}: {}",
                        cce_index,
                        device->id(),
                        view,
                        sender_translated.str(),
                        signal,
                        signal == dev_msgs::RUN_MSG_INIT ? "its firmware has not finished init"
                                                         : "its mailbox is corrupt")};
            }
        }
    }
    return {.ok = true, .reason = ""};
}

}  // namespace tt::tt_metal::distributed

// -----------------------------------------------------------------------------
// experimental::StartTensorPrefetcher / Queue / Stop
// -----------------------------------------------------------------------------
namespace tt::tt_metal::experimental {

bool IsTensorPrefetcherSupported(const distributed::MeshDevice& mesh_device) {
    return distributed::tensor_prefetcher_preconditions(mesh_device).ok;
}

void StartTensorPrefetcher(distributed::MeshDevice& mesh_device, const TensorPrefetcherConfig& config) {
    const distributed::TensorPrefetcherPreconditions preconditions =
        distributed::tensor_prefetcher_preconditions(mesh_device);
    TT_FATAL(
        preconditions.ok,
        "Tensor prefetcher is not supported on this mesh device: {}. Call IsTensorPrefetcherSupported() first to "
        "skip the prefetcher path.",
        preconditions.reason);
    auto& manager = mesh_device.impl().tensor_prefetcher(&mesh_device);
    manager.start(config);
}

void QueueTensorPrefetcherRequest(
    distributed::MeshDevice& mesh_device,
    const std::vector<std::reference_wrapper<const PrefetcherPipe>>& prefetcher_pipes,
    ttsl::optional_reference<const distributed::MeshCoordinateRangeSet> device_subset,
    const std::vector<TensorPrefetcherInput>& input_tensors,
    distributed::MeshCommandQueue* trace_capture_cq) {
    auto& manager = mesh_device.impl().tensor_prefetcher(&mesh_device);
    manager.queue(prefetcher_pipes, device_subset, input_tensors, trace_capture_cq);
}

void WaitForCqOnTensorPrefetcher(
    distributed::MeshCommandQueue& cq,
    ttsl::optional_reference<const distributed::MeshCoordinateRangeSet> device_subset) {
    auto* mesh_device = cq.device();
    auto& manager = mesh_device->impl().tensor_prefetcher(mesh_device);
    manager.enqueue_cq_signal_and_wait(cq, device_subset);
}

uint32_t GetTensorPrefetcherSignalAddress(distributed::MeshDevice& mesh_device, uint32_t signal_id) {
    auto& manager = mesh_device.impl().tensor_prefetcher(&mesh_device);
    return manager.signal_address(signal_id);
}

void QueueTensorPrefetcherWaitForSignal(
    distributed::MeshDevice& mesh_device, uint32_t signal_id, distributed::MeshCommandQueue* trace_capture_cq) {
    auto& manager = mesh_device.impl().tensor_prefetcher(&mesh_device);
    manager.queue_wait_for_signal(signal_id, trace_capture_cq);
}

void StopTensorPrefetcher(distributed::MeshDevice& mesh_device) {
    auto& manager = mesh_device.impl().tensor_prefetcher(&mesh_device);
    manager.stop();
}

}  // namespace tt::tt_metal::experimental
