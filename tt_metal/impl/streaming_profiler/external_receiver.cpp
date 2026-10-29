// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/external_receiver.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <list>
#include <mutex>
#include <thread>
#include <utility>

#include <tt-logger/tt-logger.hpp>
#include <tt-umd/chip_helpers/silicon_sysmem_manager.hpp>
#include <tt-umd/chip_helpers/sysmem_buffer.hpp>
#include <tt-umd/pcie/pci_device.hpp>
#include <tt-umd/tt_device/tt_device.hpp>
#include <umd/device/io_window/io_window.hpp>
#include <umd/device/types/io_window_config.hpp>
#include <tt_stl/assert.hpp>
#include <tt_stl/indestructible.hpp>

#include "hostdev/streaming_profiler_common.h"
#include "impl/streaming_profiler/service.hpp"
#include "impl/streaming_profiler/spsc_marker_decode.hpp"
#include "llrt/call_sites.hpp"
#include "protobuf/external_process.pb.h"

namespace tt::tt_metal::streaming_profiler {

namespace {

namespace kp = kernel_profiler;
namespace proto = external::proto;
namespace api = experimental::device_observability;
using profiler::kSpscRingCap;

// The frames this process writes for its own decoder, in this build's format.
constexpr uint32_t kLanes = kp::PROFILER_SPSC_TENSIX_RISC;
// Each endpoint's ring, which the device overwrites and never waits on, so it only needs to hold what lands between two
// reads. Its buffer holds the device's bytes_sent in its last page, and is a whole number of 2 MiB hugepages, the only
// size UMD backs with them.
constexpr uint32_t kRingBufferBytes = 4u << 20;
// Each device's stream of repacked frames, which the consumers read like a device's FIFO.
constexpr uint64_t kStreamBytes = 16u << 20;
constexpr auto kSteadySamplePeriod = std::chrono::milliseconds(100);
// How often this process reads the refclk. Fitted over its last three reads, its host line stays within about 150 ns of
// the host's clock (10 minutes measured on a p150), and durations need no more: they come from the chips' clock sync
// frames.
constexpr auto kRefclkReadPeriod = std::chrono::seconds(1);
// The reach of the window onto an endpoint's control block. A DRISC's L1 sits high in its NoC address space, so the
// window starts at the aligned block holding the control block.
constexpr uint64_t kControlBlockWindowBytes = uint64_t{1} << 20;
// A packet range's blank words make its frame position equal its first word's slot modulo this, the NoC's PCIe write
// alignment, which the protocol fixes.
constexpr uint32_t kNocWriteAlignmentWords = 4;

// The value of `field` in the words at `words`.
uint32_t field_of(const uint32_t* words, const proto::BitField& field) {
    const uint32_t word = field.bit_offset() / 32, shift = field.bit_offset() % 32;
    uint64_t value = words[word] >> shift;
    if (shift + field.bit_size() > 32) {
        value |= uint64_t{words[word + 1]} << (32 - shift);
    }
    return static_cast<uint32_t>(field.bit_size() >= 32 ? value : value & ((uint64_t{1} << field.bit_size()) - 1));
}

uint32_t round_up(uint32_t words, uint32_t multiple) { return (words + multiple - 1) / multiple * multiple; }

// The Processor of RISC slot `slot` on a core of type `type`. An eth core has two RISCs and fills its other slots with
// ERISC1.
api::Processor processor_of(proto::CoreType type, uint32_t slot) {
    constexpr std::array<api::Processor, kLanes> kTensix = {
        api::Processor::BRISC,
        api::Processor::NCRISC,
        api::Processor::TRISC0,
        api::Processor::TRISC1,
        api::Processor::TRISC2};
    return type == proto::CORE_TYPE_ETH ? (slot == 0 ? api::Processor::ERISC0 : api::Processor::ERISC1)
                                        : kTensix.at(slot);
}

uint32_t packed(const proto::CoreCoord& coord) { return kp::word_of(kp::NocXy{.x = coord.x(), .y = coord.y()}); }

}  // namespace

struct ExternalReceiver::Ring {
    std::unique_ptr<tt::umd::SysmemBuffer> buffer;
    const volatile uint32_t* data = nullptr;
    const volatile uint32_t* bytes_sent = nullptr;
    uint32_t ring_bytes = 0;
    uint32_t max_lead_bytes = 0;

    uint32_t read = 0;    // bytes_sent at the last read
    uint32_t offset = 0;  // where `read` is in the ring

    uint32_t dev = 0;
    bool clock_sync = false;

    // A window onto the endpoint's control block, which stays mapped on its own, so stopping the writes at exit needs
    // nothing of UMD's, whose statics may be gone by then.
    std::unique_ptr<tt::umd::IoWindow> window;
    uint64_t control_block = 0;  // its offset in the window
};

struct ExternalReceiver::Stream {
    std::unique_ptr<std::byte[]> fifo;  // kStreamBytes
    std::atomic<uint64_t> head{0};      // where the frame being written ends
    std::atomic<uint64_t> walked{0};    // where the last frame written ends
    std::vector<std::atomic<uint64_t>> marks;
    uint64_t mark_block = UINT64_MAX;
    // Where each lane's last run ends, in positions of the lane's own that start each run on a ring boundary, so no
    // run is a wrap image. By core, then RISC.
    std::vector<uint32_t> lane_tails;
};

// This process's own clock solver, which builds the map from the clock sync frames and from its own reads of the
// refclk. It joins the capture when this process attaches.
struct ExternalReceiver::Clock {
    RootRefclkWindow refclk;
    ClockSolver solver;
    std::chrono::steady_clock::time_point next_burst{};

    Clock(tt::umd::TTDevice& device, const proto::Counter& counter, const CaptureContext& ctx) :
        refclk(
            device,
            tt_xy_pair{counter.virtual_coord().x(), counter.virtual_coord().y()},
            counter.low_word_address(),
            counter.high_word_address()),
        solver(ctx, refclk.bases(), kRefclkReadPeriod) {
        solver.resume(refclk.bases().root_refclk);
    }
};

ExternalReceiver::ExternalReceiver(std::unique_ptr<external::ClientConnection> connection) :
    connection_(std::move(connection)),
    descriptor_(*connection_->descriptor()),
    frames_(descriptor_.frame_layout()),
    packets_(descriptor_.packet_format()) {
    TT_FATAL(
        frames_.packet_range_start_fields_size() == static_cast<int>(kLanes) &&
            frames_.packet_range_end_fields_size() == static_cast<int>(kLanes),
        "streaming profiler: Metal process {}'s frames have {} RISC slots, and this process reads {}",
        connection_->pid(),
        frames_.packet_range_start_fields_size(),
        kLanes);
    TT_FATAL(
        std::has_single_bit(frames_.risc_ring_buffer_size_words()),
        "streaming profiler: Metal process {}'s RISC ring buffers aren't a power of two",
        connection_->pid());
    header_words_ = round_up(
        frames_.prefix_length_words() + frames_.control_block_length_words(), frames_.socket_page_size_words());

    // The chips, as the decoder reads them. UMD numbers chips in the order it enumerates their PCIe devices.
    const std::vector<int> enumerated = tt::umd::PCIDevice::enumerate_devices();
    for (const proto::Chip& chip : descriptor_.chips()) {
        TT_FATAL(
            chip.has_clock_sync(),
            "streaming profiler: Metal process {}'s chip {} has its wall clock read by the host, which the external "
            "process doesn't support yet",
            connection_->pid(),
            chip.pci_device_number());
        auto& [tt_device, sysmem] = pcie_devices_[chip.pci_device_number()];
        tt_device = tt::umd::TTDevice::create(static_cast<int>(chip.pci_device_number()));
        sysmem = std::make_unique<tt::umd::SiliconSysmemManager>(tt_device.get(), 0);
        CaptureContext::Device& device = capture_.devices.emplace_back();
        const auto at = std::ranges::find(enumerated, static_cast<int>(chip.pci_device_number()));
        device.chip_id = static_cast<uint32_t>(at - enumerated.begin());
        for (const proto::Core& core : chip.cores()) {
            device.core_of_xy[packed(core.virtual_coord())] = static_cast<uint16_t>(device.clock_offsets.size());
            device.clock_offsets.push_back(core.clock_offset_ticks());
            for (uint32_t slot = 0; slot < kLanes; slot++) {
                device.lanes.push_back(api::Core{
                    .logical = CoreCoord(core.logical_coord().x(), core.logical_coord().y()),
                    .physical = CoreCoord(core.physical_coord().x(), core.physical_coord().y()),
                    .chip_id = static_cast<ChipId>(device.chip_id),
                    .processor = processor_of(core.core_type(), slot)});
            }
        }
        // The decoder widens a lane's first timer slot from the chip's wall clock high word, which every Tensix core
        // shows.
        const auto tensix = std::ranges::find(chip.cores(), proto::CORE_TYPE_TENSIX, &proto::Core::core_type);
        if (tensix != chip.cores().end()) {
            auto window = tt_device->create_io_window(
                tt::umd::TargetIoWindowConfig{
                    .core_start = {tensix->virtual_coord().x(), tensix->virtual_coord().y()},
                    .addr = kTensixWallClockLowWord},
                tt::umd::HostIoWindowConfig{
                    .mapping = tt::umd::HostMemoryCaching::UC,
                    .size = kTensixWallClockHighWord - kTensixWallClockLowWord + sizeof(uint32_t)});
            static_cast<void>(window->read32(0));
            device.start_timer_hi = window->read32(kTensixWallClockHighWord - kTensixWallClockLowWord);
        }
    }
    const auto chip_index = [&](uint32_t pci_device_number) {
        const auto& chips = descriptor_.chips();
        const auto at = std::ranges::find(chips, pci_device_number, &proto::Chip::pci_device_number);
        TT_FATAL(at != chips.end(), "streaming profiler: a link names chip {}, which has no Chip", pci_device_number);
        return static_cast<uint32_t>(at - chips.begin());
    };
    const auto core_index = [&](uint32_t dev, const proto::CoreCoord& coord) {
        const uint32_t core = capture_.devices[dev].core_of_xy.find(packed(coord));
        TT_FATAL(core != CoreTable::kNone, "streaming profiler: a link names a core its chip doesn't list");
        return core;
    };
    for (const proto::Link& link : descriptor_.clock_sync_links()) {
        const uint32_t dev_a = chip_index(link.transmitter().pci_device_number());
        const uint32_t dev_b = chip_index(link.receiver().pci_device_number());
        const uint32_t core_a = core_index(dev_a, link.transmitter().virtual_coord());
        const uint32_t core_b = core_index(dev_b, link.receiver().virtual_coord());
        const auto logical = [&](uint32_t dev, uint32_t core) {
            const proto::CoreCoord& coord =
                descriptor_.chips(static_cast<int>(dev)).cores(static_cast<int>(core)).logical_coord();
            return CoreCoord(coord.x(), coord.y());
        };
        capture_.links.push_back(CaptureContext::Link{
            .dev_a = dev_a,
            .dev_b = dev_b,
            .core_a = core_a,
            .core_b = core_b,
            .eth_a = logical(dev_a, core_a),
            .eth_b = logical(dev_b, core_b)});
    }

    for (uint32_t dev = 0; dev < capture_.devices.size(); dev++) {
        auto stream = std::make_unique<Stream>();
        stream->fifo = std::make_unique_for_overwrite<std::byte[]>(kStreamBytes);
        stream->marks = std::vector<std::atomic<uint64_t>>(kStreamBytes / kMarkBytes);
        for (auto& mark : stream->marks) {
            mark.store(UINT64_MAX, std::memory_order_relaxed);
        }
        stream->lane_tails.assign(capture_.devices[dev].lanes.size(), 0);
        streams_view_.push_back(ReceiverStream{
            .fifo = std::span<const std::byte>(stream->fifo.get(), kStreamBytes),
            .walked = &stream->walked,
            .dev = dev,
            .marks = stream->marks});
        streams_.push_back(std::move(stream));
    }
    connection_->start();

    // A ring per endpoint.
    const auto add_ring = [&](uint32_t dev, const proto::Endpoint& endpoint, bool clock_sync) {
        tt::umd::TTDevice& tt_device =
            *pcie_devices_.at(descriptor_.chips(static_cast<int>(dev)).pci_device_number()).first;
        tt::umd::SiliconSysmemManager& sysmem =
            *pcie_devices_.at(descriptor_.chips(static_cast<int>(dev)).pci_device_number()).second;
        Ring& ring = rings_.emplace_back();
        ring.dev = dev;
        ring.clock_sync = clock_sync;
        ring.max_lead_bytes = endpoint.max_lead_bytes();
        const uint32_t page_bytes = std::max<uint32_t>(endpoint.ring_size_multiple_bytes(), 64);
        ring.ring_bytes = (kRingBufferBytes - page_bytes) / page_bytes * page_bytes;
        ring.buffer = sysmem.allocate_sysmem_buffer(kRingBufferBytes, /*map_to_noc=*/true);
        ring.data = static_cast<const volatile uint32_t*>(ring.buffer->get_va());
        ring.bytes_sent = ring.data + ring.ring_bytes / 4;
        const_cast<volatile uint32_t*>(ring.bytes_sent)[0] = 0;
        const uint64_t base = endpoint.control_block_address() & ~(kControlBlockWindowBytes - 1);
        ring.control_block = endpoint.control_block_address() - base;
        ring.window = tt_device.create_io_window(
            tt::umd::TargetIoWindowConfig{
                .core_start = {endpoint.virtual_coord().x(), endpoint.virtual_coord().y()},
                .addr = base,
                .noc = tt::umd::NocId::NOC0},
            tt::umd::HostIoWindowConfig{.mapping = tt::umd::HostMemoryCaching::UC, .size = kControlBlockWindowBytes});
    };
    const proto::Counter* refclk = nullptr;
    for (uint32_t dev = 0; dev < capture_.devices.size(); dev++) {
        const proto::Chip& chip = descriptor_.chips(static_cast<int>(dev));
        for (const proto::Endpoint& endpoint : chip.record_endpoints()) {
            add_ring(dev, endpoint, false);
        }
        add_ring(dev, chip.clock_sync().endpoint(), true);
        if (chip.clock_sync().has_refclk()) {
            TT_FATAL(
                refclk == nullptr && dev == CaptureContext::kRootDevice,
                "streaming profiler: the external process maps clocks through the first chip's refclk only");
            refclk = &chip.clock_sync().refclk();
        }
    }
    TT_FATAL(refclk != nullptr, "streaming profiler: Metal process {} names no refclk", connection_->pid());
    clock_ = std::make_unique<Clock>(
        *pcie_devices_.at(descriptor_.chips(CaptureContext::kRootDevice).pci_device_number()).first, *refclk, capture_);
    lanes_.reset(&api::detail::CaptureLanes::create(capture_, connection_->call_sites()));
    // A consumer reading a record's time() needs the steady clock's map, so it has a sample before any record arrives.
    service().steady().sample();
    service().attach_receiver(*this);
    point_devices(true);
    log_info(
        tt::LogMetal,
        "[streaming profiler] receiving external from {} ring(s) on {} device(s)",
        rings_.size(),
        capture_.devices.size());
}

ExternalReceiver::~ExternalReceiver() {
    // run() has finished the map of a Metal process that closed the connection, but not of one this process dropped.
    try {
        end_capture();
    } catch (const std::exception& e) {
        log_error(tt::LogMetal, "[streaming profiler] Metal process {}'s clock map: {}", connection_->pid(), e.what());
    }
    service().detach_receiver(*this);
    // Relays that finished the capture write nothing more, and their cores are the next Metal process's.
    if (!relays_stopped_) {
        point_devices(false);
    }
    if (lost_bytes_ != 0) {
        log_warning(
            tt::LogMetal,
            "[streaming profiler] the devices wrote over {} bytes of the rings before this process read them",
            lost_bytes_);
    }
}

uint64_t ExternalReceiver::live_head(uint32_t stream) const {
    return streams_[stream]->head.load(std::memory_order_acquire);
}

const ClockMap& ExternalReceiver::clock_map() const { return clock_->solver.map(); }

void ExternalReceiver::drop(const std::exception& failure) {
    log_error(
        tt::LogMetal,
        "[streaming profiler] dropping the connection to Metal process {}: {}",
        connection_->pid(),
        failure.what());
    connection_->drop();
}

void ExternalReceiver::run(const std::stop_token& stop) {
    auto next_sample = std::chrono::steady_clock::now();
    while (!stop.stop_requested()) {
        // A Metal process that ends its capture sends CaptureEnded once its relays' last writes have landed, then
        // closes the connection. One that dies may leave its relays writing, so their writes are turned off first.
        // Either way, the rings are read once more after the close, which covers every write made before it. A
        // connection this process dropped over a bad message isn't read again: its later records may name call sites
        // that message held.
        if (connection_->closed()) {
            if (!connection_->capture_ended()) {
                point_devices(false);
            }
            relays_stopped_ = true;
            if (!connection_->dropped()) {
                for (Ring& ring : rings_) {
                    read_ring(ring);
                }
            }
            end_capture();
            service().wake_consumers();
            return;
        }
        bool any = false;
        for (Ring& ring : rings_) {
            any |= read_ring(ring);
        }
        const auto now = std::chrono::steady_clock::now();
        if (now >= clock_->next_burst) {
            clock_->next_burst = now + kRefclkReadPeriod;
            clock_->solver.on_refclk_burst(clock_->refclk.read_burst());
            solver_pending_ = true;
        }
        if (solver_pending_) {
            solver_pending_ = false;
            any |= clock_->solver.on_batch_end();
        }
        if (any) {
            service().wake_consumers();
        }
        if (now >= next_sample) {
            next_sample = now + kSteadySamplePeriod;
            service().steady().sample();
        }
        if (!any) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }
}

uint32_t ExternalReceiver::entry_words(const Ring& ring, const uint32_t* entry) const {
    const uint32_t page_words = frames_.socket_page_size_words();
    if (ring.clock_sync) {
        return round_up(
            frames_.prefix_length_words() + field_of(entry, frames_.frame_payload_length_field()), page_words);
    }
    const uint32_t first = field_of(entry, frames_.routed_first_page_field()) * page_words;
    const uint32_t end = field_of(entry, frames_.routed_end_page_field()) * page_words;
    return end < first ? 0 : std::min(first, header_words_) + end - first;
}

// Reads what the device has written to the ring since the last read. Like a BroadcastRing's writer, the device
// overwrites the ring without waiting for this reader, which drops what the device may have written over before or
// while it read. bytes_sent only stops where an entry ends, so a reader the device laps picks up there.
bool ExternalReceiver::read_ring(Ring& ring) {
    const uint32_t sent = *ring.bytes_sent;
    if (sent == ring.read) {
        return false;
    }
    // Whether the device may have written over the oldest unread byte, by the bytes_sent it shows.
    const auto lapped = [&](uint32_t shown) { return shown - ring.read > ring.ring_bytes - ring.max_lead_bytes; };
    uint32_t read = sent;
    if (lapped(sent)) {
        lost_bytes_ += sent - ring.read;
    } else {
        words_.resize((sent - ring.read) / 4);
        const auto* data = const_cast<const uint32_t*>(ring.data);
        const size_t from = ring.offset / 4;
        const size_t first = std::min(words_.size(), ring.ring_bytes / 4 - from);
        std::memcpy(words_.data(), data + from, first * 4);
        std::memcpy(words_.data() + first, data, (words_.size() - first) * 4);
        std::atomic_thread_fence(std::memory_order_acquire);
        if (lapped(*ring.bytes_sent)) {
            lost_bytes_ += sent - ring.read;
        } else {
            size_t at = 0;
            while (at < words_.size()) {
                const uint32_t n = entry_words(ring, &words_[at]);
                // An entry that runs past bytes_sent is left for a later read.
                if (n == 0 || at + n > words_.size()) {
                    break;
                }
                if (ring.clock_sync) {
                    on_clock_sync(&words_[at], ring.dev);
                } else {
                    repack(&words_[at], ring.dev);
                }
                at += n;
            }
            read = ring.read + static_cast<uint32_t>(at * sizeof(uint32_t));
        }
    }
    const bool moved = read != ring.read;
    ring.offset = (ring.offset + (read - ring.read)) % ring.ring_bytes;
    ring.read = read;
    return moved;
}

uint32_t ExternalReceiver::core_of(const uint32_t* frame, uint32_t dev) const {
    const uint32_t xy = kp::word_of(kp::NocXy{
        .x = field_of(frame, frames_.core_coord_x_field()), .y = field_of(frame, frames_.core_coord_y_field())});
    const uint32_t core = capture_.devices[dev].core_of_xy.find(xy);
    TT_FATAL(
        core != CoreTable::kNone,
        "streaming profiler: a frame from NoC core {:#x}, which the capture descriptor doesn't list",
        xy);
    return core;
}

// Rebuilds the record frame the entry holds pages of, then writes a frame of its routed records in this build's format
// for this process's decoder: each RISC's routed records whole, with this process's ids for their call sites. A packet
// range cut short ends at its last whole packet.
void ExternalReceiver::repack(const uint32_t* entry, uint32_t dev) {
    const uint32_t page_words = frames_.socket_page_size_words();
    const uint32_t first = field_of(entry, frames_.routed_first_page_field()) * page_words;
    const uint32_t end = field_of(entry, frames_.routed_end_page_field()) * page_words;
    const uint32_t head = std::min(first, header_words_);
    frame_.assign(end, 0);
    std::copy(entry, entry + head, frame_.begin());
    std::copy(entry + head, entry + head + end - first, frame_.begin() + first);
    const uint32_t core = core_of(frame_.data(), dev);
    Stream& stream = *streams_[dev];
    uint32_t* const tails = stream.lane_tails.data() + size_t{core} * kLanes;
    packed_.assign(kp::SPSC_SPAN_PREFIX_WORDS + kp::SPSC_SPAN_WIRE_CTRL_WORDS, 0);
    packed_[kp::SPSC_PREFIX_XY] = kp::word_of(kp::NocXy{
        .x = field_of(frame_.data(), frames_.core_coord_x_field()),
        .y = field_of(frame_.data(), frames_.core_coord_y_field())});

    const uint32_t ring_words = frames_.risc_ring_buffer_size_words();
    const auto& non_record = packets_.non_record_packet_lengths_words();
    uint32_t off = frames_.prefix_length_words() + frames_.control_block_length_words();
    bool kept_any = false;
    for (uint32_t risc = 0; risc < kLanes; risc++) {
        const uint32_t start = field_of(frame_.data(), frames_.packet_range_start_fields(static_cast<int>(risc)));
        const uint32_t take = field_of(frame_.data(), frames_.packet_range_end_fields(static_cast<int>(risc))) - start;
        uint32_t lane_start = tails[risc];
        size_t kept = 0;
        if (take != 0) {
            // A packet range that wraps ships as its whole ring buffer when little of the buffer is unused.
            const bool whole = (start & (ring_words - 1)) + take > ring_words &&
                               ring_words - take <= frames_.whole_ring_buffer_threshold_words();
            off += ((whole ? 0u : start) - off) & (kNocWriteAlignmentWords - 1);
            const uint32_t words = whole ? ring_words : take;
            // The device sends a whole ring buffer whole or not at all, and a packet range in ring order up to the
            // entry's end.
            const uint32_t sent = whole ? (off + words <= end ? take : 0) : (off < end ? std::min(take, end - off) : 0);
            if (off >= first && sent != 0) {
                lane_start = (tails[risc] + kSpscRingCap - 1) & ~(kSpscRingCap - 1);
                const size_t pad_at = packed_.size();
                packed_.resize(pad_at + kp::spsc_span_pack_pad(lane_start, static_cast<uint32_t>(pad_at)), 0);
                const size_t run_at = packed_.size();
                const auto word = [&](uint32_t k) {
                    return frame_[off + (whole ? (start + k) & (ring_words - 1) : k)];
                };
                for (uint32_t k = 0; k < sent;) {
                    const uint32_t w0 = word(k);
                    const uint32_t type = field_of(&w0, packets_.packet_type_field());
                    const bool record = type == packets_.record_packet_type();
                    uint32_t n = 0;
                    if (record) {
                        n = field_of(&w0, packets_.record_length_field());
                    } else if (const auto length = non_record.find(type); length != non_record.end()) {
                        n = length->second;
                    }
                    if (n == 0 || k + n > sent) {
                        break;
                    }
                    if (record && field_of(&w0, packets_.routed_field()) != 0) {
                        const uint32_t id = field_of(&w0, packets_.call_site_id_field());
                        const uint32_t packet_words =
                            connection_->wait_for_call_site(id).packet_layouts.front().packet_words;
                        // A record for the external process always has its call site's 64-bit packet, so a shorter
                        // one is corrupt.
                        TT_FATAL(
                            n >= packet_words,
                            "streaming profiler: Metal process {} sent a corrupt record of call site {}",
                            connection_->pid(),
                            id);
                        packed_.push_back(kp::ppfmt::record_w0(packet_words) | kp::ppfmt::TO_EXTERNAL | id);
                        for (uint32_t j = 1; j < packet_words; j++) {
                            packed_.push_back(word(k + j));
                        }
                    }
                    k += n;
                }
                kept = packed_.size() - run_at;
                if (kept == 0) {
                    packed_.resize(pad_at);
                    lane_start = tails[risc];
                }
            }
            off += words;
        }
        tails[risc] = lane_start + static_cast<uint32_t>(kept);
        packed_[kp::SPSC_PREFIX_HEAD_0 + risc] = lane_start;
        packed_[kp::SPSC_SPAN_PREFIX_WORDS + kp::SPSC_WIRE_TAIL_0 + risc] = tails[risc];
        kept_any |= kept != 0;
    }
    if (!kept_any) {
        return;
    }
    packed_[0] = kp::spsc_span_w0();
    packed_[kp::SPSC_PREFIX_PAYLOAD_WORDS] = static_cast<uint32_t>(packed_.size()) - kp::SPSC_SPAN_PREFIX_WORDS;
    packed_.resize(kp::spsc_span_frame_words(packed_[kp::SPSC_PREFIX_PAYLOAD_WORDS]), 0);
    write(stream);
}

// Appends packed_ to the stream. The head moves past the frame before its bytes are written, so a consumer this write
// laps sees so when it checks the head after its copy.
void ExternalReceiver::write(Stream& stream) {
    const uint64_t at = stream.walked.load(std::memory_order_relaxed);
    const size_t bytes = packed_.size() * sizeof(uint32_t);
    stream.head.store(at + bytes, std::memory_order_release);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    const size_t pos = at & (kStreamBytes - 1);
    const size_t first = std::min<size_t>(bytes, kStreamBytes - pos);
    std::memcpy(stream.fifo.get() + pos, packed_.data(), first);
    std::memcpy(stream.fifo.get(), reinterpret_cast<const std::byte*>(packed_.data()) + first, bytes - first);
    if (at / kMarkBytes != stream.mark_block) {
        stream.mark_block = at / kMarkBytes;
        stream.marks[stream.mark_block % stream.marks.size()].store(at, std::memory_order_release);
    }
    stream.walked.store(at + bytes, std::memory_order_release);
}

// Follows the endpoints' handshake: to start a device's writes, stop them and wait until it has, fill in the ring, then
// enable it; to stop them, clear the enable and wait for the acknowledgement.
void ExternalReceiver::point_devices(bool enable) {
    constexpr uint64_t kEnable = offsetof(kp::EndpointControlBlock, ring_write_enable);
    constexpr uint64_t kAcknowledged = offsetof(kp::EndpointControlBlock, ring_write_enable_acknowledged);
    const auto wait_for = [&](uint32_t acknowledged) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
        uint32_t followed = 0;
        for (Ring& ring : rings_) {
            while (ring.window->read32(ring.control_block + kAcknowledged) != acknowledged &&
                   std::chrono::steady_clock::now() < deadline) {
            }
            followed += ring.window->read32(ring.control_block + kAcknowledged) == acknowledged;
        }
        return followed;
    };
    for (Ring& ring : rings_) {
        ring.window->write32(ring.control_block + kEnable, 0);
    }
    wait_for(0);
    if (!enable) {
        return;
    }
    for (Ring& ring : rings_) {
        kp::EndpointControlBlock block{};
        block.ring_noc_address = ring.buffer->get_noc_address().value();
        block.bytes_sent_noc_address = block.ring_noc_address + ring.ring_bytes;
        block.ring_size_bytes = ring.ring_bytes;
        ring.window->write_block(ring.control_block, &block, offsetof(kp::EndpointControlBlock, ring_write_enable));
        ring.window->write32(ring.control_block + kEnable, 1);
    }
    if (const uint32_t followed = wait_for(1); followed != rings_.size()) {
        log_warning(
            tt::LogMetal, "[streaming profiler] {} of {} devices took this process's rings", followed, rings_.size());
    }
}

// Feeds a clock sync frame's records to this process's solver.
void ExternalReceiver::on_clock_sync(const uint32_t* frame, uint32_t dev) {
    const uint32_t core = core_of(frame, dev);
    const uint32_t records = field_of(frame, frames_.frame_payload_length_field()) / kp::kSyncRecordWords;
    const auto* record = reinterpret_cast<const kp::SyncRecord*>(frame + frames_.prefix_length_words());
    for (uint32_t r = 0; r < records; r++) {
        clock_->solver.on_record(dev, core, record[r]);
    }
    solver_pending_ = true;
}

void ExternalReceiver::end_capture() {
    if (std::exchange(ended_, true)) {
        return;
    }
    if (solver_pending_) {
        solver_pending_ = false;
        clock_->solver.on_batch_end();
    }
    // Bursts after the capture's last records extend the refclk-to-TSC map past them, until enough exist to fit it.
    while (!clock_->solver.on_refclk_burst(clock_->refclk.read_burst())) {
    }
    clock_->solver.on_capture_end();
}

namespace {

// The Metal processes connected to this process, each served by a thread of its own that owns its receiver. The thread
// destroys the receiver once its Metal process closes the connection, or once asked to stop, which lets go of the rings
// and the chips. A receiver's relays copy into its rings until it is destroyed, so the process stops them before it
// exits.
struct Clients {
    struct Client {
        std::vector<uint32_t> pcie_devices;
        std::atomic<bool> done{false};  // its thread has destroyed its receiver
        std::jthread serving;           // declared last, so it stops before the rest goes
    };
    std::mutex mu;
    std::list<Client> connected;

    // Stops the clients on any of `pcie_devices`, whose relays a Metal process that connects has booted anew, and
    // forgets every client whose receiver is gone.
    void drop_on(const std::vector<uint32_t>& pcie_devices) {
        std::lock_guard lock(mu);
        connected.remove_if([&](const Client& client) {
            return client.done.load(std::memory_order_acquire) ||
                   std::ranges::find_first_of(client.pcie_devices, pcie_devices) != client.pcie_devices.end();
        });
    }
    void add(std::vector<uint32_t> pcie_devices, std::unique_ptr<ExternalReceiver> receiver) {
        std::lock_guard lock(mu);
        Client& client = connected.emplace_back();
        client.pcie_devices = std::move(pcie_devices);
        client.serving =
            std::jthread([&done = client.done, receiver = std::move(receiver)](const std::stop_token& stop) mutable {
                try {
                    receiver->run(stop);
                } catch (const std::exception& e) {
                    receiver->drop(e);
                }
                receiver.reset();
                done.store(true, std::memory_order_release);
            });
    }
    void clear() {
        std::lock_guard lock(mu);
        connected.clear();
    }
};
Clients& clients() {
    static ttsl::Indestructible<Clients> instance;
    return instance.get();
}

// This process's socket, which it listens on once it is the external process, for the rest of its life.
external::ExternalListener& listener() {
    static ttsl::Indestructible<external::ExternalListener> instance;
    return instance.get();
}

}  // namespace

void serve_metal_processes() {
    static std::once_flag once;
    std::call_once(once, [] {
        TT_FATAL(listener().listen(), "streaming profiler: another process is the external process");
        std::atexit([] { clients().clear(); });
        std::thread([] {
            set_thread_name("sp-external");
            // The first call sleeps while it measures the rate, so it is made before any capture.
            static_cast<void>(ns_per_tsc_tick());
            // A connection that fails is dropped, and this process goes on to serve the next Metal process.
            for (int fd = listener().accept(); fd >= 0; fd = listener().accept()) {
                try {
                    auto connection = std::make_unique<external::ClientConnection>(fd);
                    if (!connection->descriptor()) {
                        continue;
                    }
                    std::vector<uint32_t> pcie_devices;
                    for (const external::proto::Chip& chip : connection->descriptor()->chips()) {
                        pcie_devices.push_back(chip.pci_device_number());
                    }
                    clients().drop_on(pcie_devices);
                    clients().add(std::move(pcie_devices), std::make_unique<ExternalReceiver>(std::move(connection)));
                } catch (const std::exception& e) {
                    log_error(tt::LogMetal, "[streaming profiler] couldn't serve a Metal process: {}", e.what());
                }
            }
        }).detach();
    });
}

}  // namespace tt::tt_metal::streaming_profiler
