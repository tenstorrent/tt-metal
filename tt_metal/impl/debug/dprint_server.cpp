// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <pthread.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include <enchantum/enchantum.hpp>
#include <tt-logger/tt-logger.hpp>
#include "context/metal_env_accessor.hpp"
#include <tt_stl/assert.hpp>
#include <tt_stl/fmt.hpp>
#include <umd/device/types/core_coordinates.hpp>
#include <umd/device/soc_descriptor.hpp>
#include <umd/device/types/xy_pair.hpp>

#include "core_coord.hpp"
#include "debug_helpers.hpp"
#include "dprint_server.hpp"
#include "dprint_parser.hpp"
#include "fmt/base.h"
#include "hal_types.hpp"
#include "hostdev/device_print_common.h"
#include "hostdev/device_print_structures.h"
#include "hostdevcommon/dprint_common.h"
#include "llrt.hpp"
#include "impl/context/metal_env_impl.hpp"
#include "impl/context/metal_context.hpp"
#include "dispatch/dispatch_query_manager.hpp"
#include <llrt/tt_cluster.hpp>
#include "llrt/hal.hpp"
#include "impl/debug/inspector/inspector.hpp"
#include "jit_build/build_env_manager.hpp"

using std::ofstream;
using std::ostream;
using std::string;
using std::to_string;
using std::uint32_t;

using namespace tt;

using RiscKey = std::tuple<ChipId, umd::CoreDescriptor, uint32_t>;  // Chip id, core, risc id

struct RiscKeyComparator {
    bool operator()(const RiscKey& x, const RiscKey& y) const {
        const ChipId x_device_id = get<0>(x);
        const ChipId y_device_id = get<0>(y);
        const uint32_t x_risc_id = get<2>(x);
        const uint32_t y_risc_id = get<2>(y);
        const umd::CoreDescriptor& x_core_desc = get<1>(x);
        const umd::CoreDescriptor& y_core_desc = get<1>(y);

        if (x_device_id != y_device_id) {
            return x_device_id < y_device_id;
        }

        tt::tt_metal::CoreDescriptorComparator core_desc_cmp;
        if (core_desc_cmp(x_core_desc, y_core_desc)) {
            return true;
        }
        if (core_desc_cmp(y_core_desc, x_core_desc)) {
            return false;
        }

        return x_risc_id < y_risc_id;
    }
};
namespace {

string logfile_path = "generated/dprint/";

string GetRiscName(
    const tt::Cluster& cluster,
    const tt_metal::Hal& hal,
    ChipId device_id,
    const umd::CoreDescriptor& logical_core,
    int risc_id,
    bool abbreviated = false) {
    CoreCoord virtual_core =
        cluster.get_virtual_coordinate_from_logical_coordinates(device_id, logical_core.coord, logical_core.type);
    auto programmable_core_type = llrt::get_core_type(device_id, virtual_core);
    return hal.get_processor_class_name(programmable_core_type, risc_id, abbreviated);
}

inline bool RiscEnabled(
    const llrt::RunTimeOptions& rtoptions, tt_metal::HalProgrammableCoreType core_type, int risc_index) {
    const auto& processors = rtoptions.get_feature_processors(tt::llrt::RunTimeDebugFeatureDprint);
    return processors.contains(core_type, risc_index);
}

// A null stream for when the print server is muted.
class NullBuffer : public std::streambuf {
public:
    int overflow(int c) override { return c; }
};
NullBuffer null_buffer;
std::ostream null_stream(&null_buffer);

// Writes a magic value at wpos ptr address for dprint buffer at the given address.
// Used for debug print server startup sequence.
void WriteInitMagic(
    tt::Cluster& cluster,
    ChipId device_id,
    const CoreCoord& virtual_core,
    const tt::tt_metal::DPrintBufferInfo& buffer_info,
    bool enabled) {
    // TODO(AP): this could use a cleanup - need a different mechanism to know if a kernel is running on device.
    // Force wait for first kernel launch by first writing a non-zero and waiting for a zero.
    std::vector<uint32_t> initbuf = std::vector<uint32_t>(buffer_info.structure_size / sizeof(uint32_t), 0);
    initbuf[0] = uint32_t(enabled ? DEBUG_PRINT_SERVER_STARTING_MAGIC : DEBUG_PRINT_SERVER_DISABLED_MAGIC);
    cluster.write_core(device_id, virtual_core, initbuf, buffer_info.structure_address);

    // Prevent race conditions during runtime by waiting until the init value is actually written
    // DPrint is only used for debug purposes so this delay should not be a big issue.
    // 1. host will read remote and think the wpos is 0. so it'll go and poll the data
    // 2. the packet will arrive to set the wpos = DEBUG_PRINT_SERVER_STARTING_MAGIC
    // 3. the actual host polling function will read wpos = DEBUG_PRINT_SERVER_STARTING_MAGIC
    // 4. now we will access wpos at the starting magic which is incorrect
    uint32_t num_tries = 100000;
    while (num_tries-- > 0) {
        auto result = cluster.read_core(device_id, virtual_core, buffer_info.structure_address, 4);
        if ((result[0] == DEBUG_PRINT_SERVER_STARTING_MAGIC && enabled) ||
            (result[0] == DEBUG_PRINT_SERVER_DISABLED_MAGIC && !enabled)) {
            return;
        }
    }
    TT_THROW("Timed out writing init magic");
}  // WriteInitMagic

}  // namespace

namespace tt::tt_metal {

using DevicePrintHeader = device_print_detail::structures::DevicePrintHeader;
static_assert(sizeof(DevicePrintHeader) == sizeof(uint32_t));

class DPrintServer::Impl {
public:
    Impl(MetalContext* context, MetalEnv& env, uint8_t num_hw_cqs, const DispatchCoreConfig& dispatch_core_config);
    Impl() = delete;
    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;
    ~Impl();

    void set_mute(bool mute_print_server) { mute_print_server_ = mute_print_server; }
    void await();
    void attach_devices();
    void detach_devices();
    void clear_log_file();
    bool reads_dispatch_cores(ChipId device_id) { return device_reads_dispatch_cores_[device_id]; }
    bool hang_detected() { return server_killed_due_to_hang_; }

    std::vector<umd::CoreDescriptor> get_print_cores(ChipId device_id) {
        std::lock_guard lock(device_to_core_range_lock_);
        auto it = device_to_core_range_.find(device_id);
        if (it == device_to_core_range_.end()) {
            return {};
        }
        return it->second;
    }

    // Returns the layout of every DPRINT buffer in a core's DPRINT_BUFFERS region. Most arch/core
    // combinations use a single buffer covering all of the core's RISC processors. Quasar TENSIX is
    // special: it splits its processors across two physically separate buffers (see
    // DevicePrintMemoryLayout in device_print_mem.h) — a TRISC/compute buffer followed by a DM
    // buffer — so it returns two entries, in memory order.
    std::vector<DPrintBufferInfo> get_core_buffers(ChipId device_id, const umd::CoreDescriptor& print_core) const {
        const auto& cluster = env_.get_cluster();
        const auto& hal = env_.get_hal();
        auto virtual_core =
            cluster.get_virtual_coordinate_from_logical_coordinates(device_id, print_core.coord, print_core.type);
        auto programmable_core_type = llrt::get_core_type(device_id, virtual_core);
        const uint64_t structure_address =
            hal.get_dev_noc_addr(programmable_core_type, HalL1MemAddrType::DPRINT_BUFFERS);
        const uint32_t structure_size = hal.get_dev_size(programmable_core_type, HalL1MemAddrType::DPRINT_BUFFERS);

        TT_FATAL(structure_size <= 0xFFFFu, "DPRINT buffer size {} doesn't fit uint16_t", structure_size);

        // Each buffer has the layout (see DevicePrintBuffer in device_print_common.h):
        // uint32_t wpos;
        // uint32_t rpos;
        // uint8_t risc_state[processor_count]; // Rounded up to nearest word
        // uint32_t lock;
        // byte print_buffer[remaining buffer];
        auto make_buffer = [](uint64_t address, uint16_t size, uint16_t processor_count, uint16_t processor_offset) {
            const uint16_t risc_state_bytes = ((processor_count + 3) / 4) * 4;
            const uint16_t buffer_offset = 8u + risc_state_bytes + sizeof(uint32_t);
            const uint16_t buffer_size = size - buffer_offset;
            return DPrintBufferInfo{address, size, 0, buffer_offset, buffer_size, processor_count, processor_offset};
        };

        // Quasar TENSIX uses two buffers: the compute (TRISC) processors first, then the DM
        // processors. The compute buffer's processors are offset by the DM count in the core's
        // global processor index space (DM processors occupy global indices 0..dm_count-1, compute the rest).
        if (hal.get_arch() == tt::ARCH::QUASAR && programmable_core_type == HalProgrammableCoreType::TENSIX) {
            const uint16_t dm_count = static_cast<uint16_t>(hal.get_processor_types_count(
                programmable_core_type, static_cast<uint32_t>(HalProcessorClassType::DM)));
            const uint16_t compute_count = static_cast<uint16_t>(hal.get_processor_types_count(
                programmable_core_type, static_cast<uint32_t>(HalProcessorClassType::COMPUTE)));
            const uint16_t compute_size = 3264;
            const uint16_t dm_size = 1632;
            TT_FATAL(
                static_cast<uint32_t>(compute_size) + dm_size == structure_size,
                "Quasar TENSIX DPRINT buffer split (compute {} + DM {}) doesn't match region size {}",
                compute_size,
                dm_size,
                structure_size);
            return {
                make_buffer(structure_address, compute_size, compute_count, dm_count),
                make_buffer(structure_address + compute_size, dm_size, dm_count, 0),
            };
        }

        const uint16_t num_processors = static_cast<uint16_t>(hal.get_num_risc_processors(programmable_core_type));
        return {make_buffer(structure_address, static_cast<uint16_t>(structure_size), num_processors, 0)};
    }

private:
    bool poll_one_core(ChipId device_id, const umd::CoreDescriptor& logical_core, bool new_data_this_iter);
    bool poll_print_buffer(
        ChipId device_id, const umd::CoreDescriptor& logical_core, const DPrintBufferInfo& buffer_info);
    void init_print_buffers_for_core(ChipId device_id, const umd::CoreDescriptor& logical_core);
    void enable_print_buffers_for_core(ChipId device_id, const umd::CoreDescriptor& logical_core);

    struct RiscData {
        std::string firmware_elf_path;
        std::shared_ptr<DevicePrintParser> firmware_elf_parser;
        std::string kernel_elf_path;
        std::shared_ptr<DevicePrintParser> kernel_elf_parser;
        std::string message_buffer;
        std::optional<std::string> line_prefix;
        int last_loaded_kernel_id = -1;
    };

    std::map<RiscKey, RiscData, RiscKeyComparator> risc_data_;

    void print_buffer_data(ChipId device_id, const umd::CoreDescriptor& logical_core, std::span<uint32_t> data);

    struct DispatchDramData {
        uint64_t rw_pointers_address = 0;
        uint64_t buffer_address = 0;
        uint32_t buffer_size = 0;
        int dram_view = 0;
        bool disabled = false;
        bool running_logged = false;
        std::map<std::pair<uint32_t, uint32_t>, umd::CoreDescriptor> noc_to_core;
    };
    std::map<ChipId, DispatchDramData> dispatch_dram_data_;
    std::mutex dispatch_dram_data_lock_;

    MetalContext* context_;

    // Flag for main thread to signal the print server thread to stop.
    std::atomic<bool> stop_print_server_ = false;
    // Flag for muting the print server. This doesn't disable reading print data from the device,
    // but it suppresses the output of that print data the user.
    std::atomic<bool> mute_print_server_ = false;
    // Flag for signalling whether the print server thread has recently processed data (and is
    // therefore likely to continue processing data in the next round of polling).
    std::atomic<bool> new_data_last_iter_ = false;
    std::thread* print_server_thread_;

    std::atomic<bool> profiler_is_running_ = false;

    // A flag to signal to the main thread if the print server detected a print-based hang.
    std::atomic<bool> server_killed_due_to_hang_ = false;

    // A counter to keep track of how many iterations the print server has gone through without
    std::atomic<int> wait_loop_iterations_ = 0;

    ofstream* outfile_ = nullptr;  // non-cout
    ostream* stream_ = nullptr;    // either == outfile_ or is &cout

    // For printing each risc's dprint to a separate file, a map from {device id, core, risc index} to files.
    std::map<RiscKey, ofstream*, RiscKeyComparator> risc_to_file_stream_;

    // A map from Device -> Core Range, which is used to determine which cores on which devices
    // to scan for print data. Also a lock for editing it.
    std::map<ChipId, std::vector<umd::CoreDescriptor>> device_to_core_range_;
    std::map<ChipId, bool> device_reads_dispatch_cores_;  // True if given device reads any dispatch cores. Used to
                                                          // know whether dprint can be compiled out.
    std::mutex device_to_core_range_lock_;

    MetalEnvImpl& env_;
    uint8_t num_hw_cqs_;
    DispatchCoreConfig dispatch_core_config_;

    // Polls specified cores/riscs on all attached devices and prints any new print data. This
    // function is the main loop for the print server thread.
    void poll_print_data();

    // Polls one device for print data via the dispatch_s DRAM aggregator. Returns true if some
    // data was read. Falls back to poll_device_print_data_l1 when dispatch_s is unavailable or
    // unhealthy.
    bool poll_device_print_data(ChipId device_id, const std::vector<umd::CoreDescriptor>& logical_cores);

    // Direct per-core L1 polling. Used as the fallback when dispatch_s aggregation is unavailable.
    bool poll_device_print_data_l1(ChipId device_id, const std::vector<umd::CoreDescriptor>& logical_cores);

    // Returns the stream that the dprint data should be output to. Can be auto-generated files, the user-selected file,
    // stdout, or nothing.
    ostream* get_output_stream(const RiscKey& risc_key);

    // Flushes the shared output stream and any per-risc file streams. print_buffer_data no longer
    // flushes per line (that turned into millions of write() syscalls under heavy load); both the
    // DRAM-aggregation path and the L1-fallback path call this once after processing instead.
    void flush_output_streams();

    // Helper functions to init/attach/detach a single device
    void init_device(ChipId device_id);
    void attach_device(ChipId device_id);
    void detach_device(ChipId device_id);
};

// DEVICE_PRINT implementations — single shared buffer per core.

void DPrintServer::Impl::print_buffer_data(
    ChipId device_id, const umd::CoreDescriptor& logical_core, std::span<uint32_t> data) {
    std::size_t word_index = 0;
    auto& cluster = env_.get_cluster();
    const auto& hal = env_.get_hal();
    auto virtual_core =
        cluster.get_virtual_coordinate_from_logical_coordinates(device_id, logical_core.coord, logical_core.type);
    auto programmable_core_type = llrt::get_core_type(device_id, virtual_core);
    uint32_t risc_count = env_.get_hal().get_num_risc_processors(programmable_core_type);
    uint32_t programmable_core_type_idx = hal.get_programmable_core_type_index(programmable_core_type);
    DevicePrintParser::FormatMessageBuffer format_message_buffer;

    while (word_index < data.size()) {
        // New data always starts with a DevicePrintHeader, so lets parse it.
        const DevicePrintHeader* header = reinterpret_cast<const DevicePrintHeader*>(data.data() + word_index);
        word_index++;

        // Check if we are loading new kernel
        if (header->is_kernel && header->message_payload == DevicePrintHeader::max_message_payload_size) {
            if (header->risc_id >= risc_count) {
                log_error(
                    tt::LogMetal,
                    "Data corruption detected in device print buffer while loading new kernel: invalid risc id {}, "
                    "risc count for core is {}. Parsed message: is_kernel={}, risc_id={}, message_payload={}, "
                    "info_id={}. Ignoring rest of the buffer.",
                    header->risc_id,
                    risc_count,
                    header->is_kernel,
                    header->risc_id,
                    header->message_payload,
                    header->info_id);
                return;
            }
            RiscKey risc_key = {device_id, logical_core, header->risc_id};
            RiscData& risc_data = risc_data_[risc_key];
            if (risc_data.last_loaded_kernel_id == static_cast<int>(header->info_id)) {
                // We have already loaded this kernel, no need to do load it again.
                continue;
            }

            if (risc_data.last_loaded_kernel_id != -1) {
                // Remove reference to previous kernel elf parser so that it can be freed if needed.
                risc_data.kernel_elf_parser = nullptr;
            }

            // Update kernel id
            auto kernel_id = static_cast<int>(header->info_id);
            risc_data.last_loaded_kernel_id = kernel_id;

            // Find the elf path for this risc from the inspector.
            auto elf_path = Inspector::get_kernel_elf_path(header->info_id, header->risc_id);

            if (elf_path.empty() || !std::filesystem::exists(elf_path)) {
                log_warning(
                    tt::LogMetal,
                    "DPRINT: could not resolve ELF path for kernel id {} risc {}; print messages for this kernel will "
                    "not be decoded.",
                    kernel_id,
                    header->risc_id);
                continue;
            }
            risc_data.kernel_elf_path = elf_path;
            risc_data.kernel_elf_parser = DevicePrintParser::get_parser_for_elf(elf_path);
        } else if (
            header->is_kernel == 0 && header->risc_id == 0 && header->message_payload == 0 &&
            header->info_id == DevicePrintHeader::max_info_id_value) {
            // This is a wrap around message, we should just ignore it and mark that we processed buffer.
            break;
        } else {
            if (header->risc_id >= risc_count) {
                log_error(
                    tt::LogMetal,
                    "Data corruption detected in device print buffer while processing message: invalid risc id {}, "
                    "risc count for core is {}. Parsed message: is_kernel={}, risc_id={}, message_payload={}, "
                    "info_id={}. Ignoring rest of the buffer.",
                    header->risc_id,
                    risc_count,
                    header->is_kernel,
                    header->risc_id,
                    header->message_payload,
                    header->info_id);
                return;
            }

            // This is a normal print message, we should parse it and print it out.
            RiscKey risc_key = {device_id, logical_core, header->risc_id};
            RiscData& risc_data = risc_data_[risc_key];

            // Find elf file
            std::shared_ptr<DevicePrintParser> elf_parser = nullptr;

            if (header->is_kernel) {
                elf_parser = risc_data.kernel_elf_parser;
            } else {
                // Check if firmware elf is already loaded for this risc.
                if (risc_data.firmware_elf_parser == nullptr) {
                    // Find firmware elf path from BuildEnvManager.
                    auto [processor_class, processor_type_idx] =
                        hal.get_processor_class_and_type_from_index(programmable_core_type, header->risc_id);
                    auto firmware_elf_path = BuildEnvManager::get_instance(context_->get_context_id())
                                                 .get_firmware_binary_path(
                                                     device_id,
                                                     programmable_core_type_idx,
                                                     static_cast<uint32_t>(processor_class),
                                                     processor_type_idx);

                    risc_data.firmware_elf_path = firmware_elf_path;
                    risc_data.firmware_elf_parser = DevicePrintParser::get_parser_for_elf(firmware_elf_path);
                }
                elf_parser = risc_data.firmware_elf_parser;
            }

            // Check if we found elf file for this print message.
            if (elf_parser != nullptr) {
                // Format message
                auto buffer_remaining_bytes = std::as_bytes(data.subspan(word_index));
                if (buffer_remaining_bytes.size() < header->message_payload) {
                    log_error(
                        tt::LogMetal,
                        "Data corruption detected in device print buffer while processing message: message payload "
                        "size {} exceeds remaining buffer size {}. Parsed message: is_kernel={}, risc_id={}, "
                        "message_payload={}, info_id={}. Ignoring rest of the buffer.",
                        header->message_payload,
                        buffer_remaining_bytes.size(),
                        header->is_kernel,
                        header->risc_id,
                        header->message_payload,
                        header->info_id);
                    return;
                }
                auto payload_bytes = buffer_remaining_bytes.subspan(0, header->message_payload);
                auto formatted_message =
                    elf_parser->format_message(header->info_id, payload_bytes, format_message_buffer);
                if (!formatted_message.empty()) {
                    // Append onto any buffered partial line and work in message_buffer;
                    std::string& buffer = risc_data.message_buffer;
                    buffer.append(formatted_message);

                    // DEVICE_PRINT will output '\r' when it wants to open a new line without
                    // flushing the host buffer for that core. This allows multiple calls to DEVICE_PRINT
                    // to span multiple lines without interleaving with prints from other cores
                    auto last_newline_pos = buffer.rfind('\n');
                    if (last_newline_pos != std::string::npos) {
                        // replace the '\r' before the '\n' because they will be flushed in this iteration
                        std::replace(buffer.begin(), buffer.begin() + last_newline_pos, '\r', '\n');
                    }

                    // Check if we hit new line
                    auto newline_pos = buffer.find('\n');
                    if (newline_pos != std::string::npos) {
                        // We will do message printing. Check if we have generated line prefix for this risc before,
                        // if not generate one.
                        std::string line_prefix;
                        if (!risc_data.line_prefix.has_value()) {
                            // Compute line prefix based on RTOptions
                            const bool prepend_device_core_risc =
                                env_.get_rtoptions().get_feature_prepend_device_core_risc(
                                    tt::llrt::RunTimeDebugFeatureDprint);
                            if (prepend_device_core_risc) {
                                const string& device_id_str = to_string(device_id);
                                const string& core_coord_str = logical_core.coord.str();
                                const string& risc_name =
                                    GetRiscName(cluster, hal, device_id, logical_core, header->risc_id, true);
                                line_prefix = fmt::format("{}:{}:{}: ", device_id_str, core_coord_str, risc_name);
                            }
                            risc_data.line_prefix = line_prefix;
                        } else {
                            line_prefix = risc_data.line_prefix.value();
                        }

                        // Are we printing the whole string, or we need to split it into multiple lines because of
                        // multiple new lines in the message or because we want to prepend line prefix to each line?
                        ostream* output_stream = get_output_stream(risc_key);
                        if (newline_pos == buffer.size() - 1) {
                            *output_stream << line_prefix << buffer;
                            buffer.clear();
                        } else {
                            std::size_t newline_start = 0;
                            std::string_view full_message_view = buffer;
                            while (newline_pos != std::string::npos) {
                                std::string_view line =
                                    full_message_view.substr(newline_start, newline_pos - newline_start);
                                *output_stream << line_prefix << line << '\n';
                                newline_start = newline_pos + 1;
                                newline_pos = full_message_view.find('\n', newline_start);
                            }
                            // Keep only the trailing partial line (everything after the last '\n') for next time.
                            buffer.erase(0, newline_start);
                        }
                    }
                }
            }

            // Move to the next message
            word_index += (header->message_payload + 3) / 4;  // round up to nearest word
        }
    }
}

bool DPrintServer::Impl::poll_print_buffer(
    ChipId device_id, const umd::CoreDescriptor& logical_core, const DPrintBufferInfo& buffer_info) {
    auto& cluster = env_.get_cluster();
    auto virtual_core =
        cluster.get_virtual_coordinate_from_logical_coordinates(device_id, logical_core.coord, logical_core.type);
    auto print_buffer_address = buffer_info.structure_address + buffer_info.buffer_offset;
    auto print_buffer_size = buffer_info.buffer_size;
    auto read_write_pointer_address = buffer_info.get_read_write_pointer_address();
    constexpr uint32_t eightbytes = 8;
    auto from_dev = cluster.read_core(device_id, virtual_core, read_write_pointer_address, eightbytes);
    uint32_t wpos = from_dev[0], rpos = from_dev[1];

    if (wpos == DEBUG_PRINT_SERVER_DISABLED_MAGIC || wpos == DEBUG_PRINT_SERVER_STARTING_MAGIC || wpos == rpos) {
        return false;
    }

    while (true) {
        bool stall = (wpos & DEVICE_PRINT_WRITE_STALL_FLAG) != 0;

        // Clear stall bit to get actual wpos value
        wpos = wpos & ~DEVICE_PRINT_WRITE_STALL_FLAG;

        if (rpos > wpos) {
            // Read until end of buffer and then from beginning until wpos
            auto data =
                cluster.read_core(device_id, virtual_core, print_buffer_address + rpos, print_buffer_size - rpos);

            // Process buffer data
            print_buffer_data(device_id, logical_core, data);

            // Update rpos, so that device knows it can use rest of the buffer
            rpos = 0;
        }
        if (rpos < wpos) {
            // Read until wpos
            auto data = cluster.read_core(device_id, virtual_core, print_buffer_address + rpos, wpos - rpos);

            // Process buffer data
            print_buffer_data(device_id, logical_core, data);

            // Update rpos, so that device knows it can use rest of the buffer
            rpos = wpos;
        }

        // Check if writer is in stall waiting for reader and send clear buffer.
        if (stall) {
            // Write clear buffer.
            rpos = DEVICE_PRINT_RESET_BUFFER_MAGIC;
            cluster.write_core(device_id, virtual_core, std::vector<uint32_t>{rpos}, read_write_pointer_address + 4);

            // We should probably drain while core is in stall state.
            // Read wpos and rpos again and repeat
            from_dev = cluster.read_core(device_id, virtual_core, read_write_pointer_address, eightbytes);
            wpos = from_dev[0];
            rpos = from_dev[1];
            continue;
        }

        // We have caught up to the writer, break out of the loop and wait for more data to arrive
        break;
    }
    cluster.write_core(device_id, virtual_core, std::vector<uint32_t>{rpos}, read_write_pointer_address + 4);
    return true;
}

bool DPrintServer::Impl::poll_one_core(
    ChipId device_id, const umd::CoreDescriptor& logical_core, bool /*new_data_this_iter*/) {
    bool result = false;
    for (auto& buffer_info : get_core_buffers(device_id, logical_core)) {
        result |= poll_print_buffer(device_id, logical_core, buffer_info);
    }
    return result;
}

void DPrintServer::Impl::init_print_buffers_for_core(ChipId device_id, const umd::CoreDescriptor& logical_core) {
    auto& cluster = env_.get_cluster();
    CoreCoord virtual_core =
        cluster.get_virtual_coordinate_from_logical_coordinates(device_id, logical_core.coord, logical_core.type);
    for (auto& buffer_info : get_core_buffers(device_id, logical_core)) {
        WriteInitMagic(cluster, device_id, virtual_core, buffer_info, false);
    }
}

void DPrintServer::Impl::enable_print_buffers_for_core(ChipId device_id, const umd::CoreDescriptor& logical_core) {
    auto& cluster = env_.get_cluster();
    CoreCoord virtual_core =
        cluster.get_virtual_coordinate_from_logical_coordinates(device_id, logical_core.coord, logical_core.type);
    auto programmable_core_type = llrt::get_core_type(device_id, virtual_core);
    for (auto& buffer_info : get_core_buffers(device_id, logical_core)) {
        WriteInitMagic(cluster, device_id, virtual_core, buffer_info, true);

        uint64_t risc_flags_address = buffer_info.get_read_write_pointer_address() + 8;  // sizeof(wpos) + sizeof(rpos)
        uint16_t num_processors = buffer_info.processor_count;
        std::vector<uint8_t> risc_flags(
            (num_processors + 3) / 4 * 4, static_cast<uint8_t>(DevicePrintRiscCoreState::KernelNotPrinted));
        for (int risc_index = 0; risc_index < num_processors; risc_index++) {
            // This buffer's risc_state slots are local; map them to the core's global processor indices
            // (which RiscEnabled / the feature processor set are keyed by) via the buffer's offset.
            if (!RiscEnabled(env_.get_rtoptions(), programmable_core_type, buffer_info.processor_offset + risc_index)) {
                risc_flags[risc_index] = static_cast<uint8_t>(DevicePrintRiscCoreState::PrintingDisabled);
            }
        }

        // We created array of flags for each risc. We will write it to the device at an offset from the print buffer
        // address. It is OK to do this write any time to update flags. Flags carry info about if kernel already sent
        // kernel loaded structure and if printing is enabled. If we overwrite flags while kernel is running, all we can
        // do is make kernel print more data (repeat kernel loaded structure). We will handle this on server side.
        cluster.write_core(device_id, virtual_core, risc_flags, risc_flags_address);
    }
}

bool DPrintServer::Impl::poll_device_print_data(
    ChipId device_id, const std::vector<umd::CoreDescriptor>& logical_cores) {
    // In case dispatch is not working, we want to fall back to per-L1 polling of device_print buffers to avoid losing
    // prints.
    auto fall_back_to_l1 = [&]() { return poll_device_print_data_l1(device_id, logical_cores); };

    // Check if dispatch is disabled for this device.
    std::lock_guard<std::mutex> lock(dispatch_dram_data_lock_);
    auto it = dispatch_dram_data_.find(device_id);
    if (it == dispatch_dram_data_.end()) {
        return fall_back_to_l1();
    }
    DispatchDramData& data = it->second;

    if (data.disabled) {
        return fall_back_to_l1();
    }

    // Read DRAM to get print data for this device.
    auto& cluster = env_.get_cluster();

    try {
        // Read 5 words from the rw-pointer cell:
        //   [0] wpos/magic       kernel writes; host reads (current write position).
        //   [1] rpos             host writes; kernel reads (host's consumed position).
        //   [2] provided_size    kernel writes alongside DISABLED_MAGIC.
        //   [3] required_size    kernel writes alongside DISABLED_MAGIC.
        //   [4] finished_flag    kernel writes 1 in shutdown() right before exiting.
        std::array<uint32_t, 5> rw = {0u, 0u, 0u, 0u, 0u};
        cluster.read_dram_vec(rw.data(), sizeof(rw), device_id, data.dram_view, data.rw_pointers_address);
        const uint32_t wpos = rw[0];
        const uint32_t rpos = rw[1];
        const bool dispatcher_finished = rw[4] != 0;

        // Check if it dispatch disabled itself due to insufficient buffer
        if (wpos == DEBUG_PRINT_SERVER_DISABLED_MAGIC) {
            data.disabled = true;
            const uint32_t provided = rw[2];
            const uint32_t required = rw[3];
            log_warning(
                tt::LogMetal,
                "DPRINT: dispatch DRAM aggregation self-disabled on device {} — L1 cache buffer was {} bytes but "
                "{} bytes or more are required. Increase the size via TT_METAL_DEVICE_PRINT_DISPATCH_L1_CACHE_BYTES. "
                "Falling back to per-core L1 polling.",
                device_id,
                provided,
                required);
            return fall_back_to_l1();
        }

        // Check if dispatch was initialized
        if (wpos == DEBUG_PRINT_SERVER_STARTING_MAGIC) {
            // Kernel hasn't booted yet. Fall back to per-L1 polling so prints emitted
            // before dispatch_s starts running don't get lost.
            return fall_back_to_l1();
        }

        // Log that dispatch is running on this device, bug only once.
        if (!data.running_logged) {
            data.running_logged = true;
            log_info(tt::LogMetal, "DPRINT: dispatch DRAM aggregation running on device {}", device_id);
        }

        // Is there something in DRAM that we should read?
        if (wpos == rpos) {
            // Check if dispatcher has finished.
            if (dispatcher_finished) {
                data.disabled = true;
                return fall_back_to_l1();
            }

            // No new data in DRAM, nothing to do.
            return false;
        }

        // Sanity-check the pointers — wpos/rpos must be inside the ring buffer.
        if (wpos >= data.buffer_size || rpos >= data.buffer_size) {
            log_warning(
                tt::LogMetal,
                "DPRINT: dispatch DRAM cell out of range on device {} — wpos={} rpos={} buffer_size={}.",
                device_id,
                wpos,
                rpos,
                data.buffer_size);
            return false;
        }

        // Read new data from DRAM buffer.
        std::vector<uint32_t> payload_vector;
        if (wpos > rpos) {
            payload_vector.resize((wpos - rpos + sizeof(uint32_t) - 1) / sizeof(uint32_t));
            cluster.read_dram_vec(
                payload_vector.data(), wpos - rpos, device_id, data.dram_view, data.buffer_address + rpos);
        } else {
            const uint32_t first = data.buffer_size - rpos;
            payload_vector.resize((first + wpos + sizeof(uint32_t) - 1) / sizeof(uint32_t));
            cluster.read_dram_vec(payload_vector.data(), first, device_id, data.dram_view, data.buffer_address + rpos);
            cluster.read_dram_vec(
                payload_vector.data() + first / sizeof(uint32_t), wpos, device_id, data.dram_view, data.buffer_address);
        }

        // Walk the payload as a sequence of {DramStreamMessageHeader, padding, payload}.
        // Each chunk is dram-aligned.
        using device_print_dispatch::DramStreamMessageHeader;
        const uint32_t dram_align = env_.get_hal().get_alignment(HalMemType::DRAM);
        auto round_up = [](uint32_t v, uint32_t a) { return (v + a - 1) & ~(a - 1); };
        constexpr size_t header_size = sizeof(DramStreamMessageHeader);
        constexpr size_t rw_ptrs_size = sizeof(uint16_t) * 2;
        size_t pos = 0;
        auto payload = std::as_bytes(std::span(payload_vector));

        while (pos + header_size <= payload.size()) {
            DramStreamMessageHeader header;
            std::memcpy(&header, payload.data() + pos, header_size);
            const uint32_t align = header.align;
            const uint32_t length = header.length;
            const bool buffer_wrapped = header.buffer_wrapped != 0;

            // Safety: if both align and length are zero, the parser would advance pos by
            // dram_align(0) = 0 and spin forever. That can only happen on garbage bytes
            // (the kernel never emits a zero-length non-wrap chunk). Bail out and let the
            // outer rpos update mark the rest of the window consumed.
            if (align == 0 && length == 0 && !buffer_wrapped) {
                log_warning(
                    tt::LogMetal,
                    "DPRINT: dispatch DRAM chunk has zero align+length at pos={} (likely "
                    "stale ring bytes); skipping rest of window",
                    pos);
                break;
            }

            // Find NOC location.
            auto core_it =
                data.noc_to_core.find(std::make_pair(static_cast<uint32_t>(header.x), static_cast<uint32_t>(header.y)));
            if (core_it == data.noc_to_core.end()) {
                log_warning(
                    tt::LogMetal,
                    "DPRINT: dispatch DRAM chunk references unknown core ({},{})",
                    static_cast<uint32_t>(header.x),
                    static_cast<uint32_t>(header.y));
            }

            // Check if whole message is here.
            const size_t body_start = pos + align;
            if (body_start + length > payload.size()) {
                log_warning(tt::LogMetal, "DPRINT: dispatch DRAM chunk truncated; stopping");
                break;
            }

            // When buffer_wrapped, the kernel issued a single NOC read of the *entire* L1
            // print buffer and stuffed two uint16_t {wpos, rpos} after the body (or inside
            // the alignment slot if it fits). The valid data is [rpos, length) ++ [0, wpos).
            bool rw_ptrs_inside_align = (align >= header_size + rw_ptrs_size);
            size_t chunk_end_bytes = align + length;
            if (buffer_wrapped && !rw_ptrs_inside_align) {
                chunk_end_bytes += rw_ptrs_size;
            }

            if (core_it != data.noc_to_core.end() && length > 0) {
                if (!buffer_wrapped) {
                    if (body_start % sizeof(uint32_t) != 0) {
                        log_warning(
                            tt::LogMetal, "DPRINT: dispatch DRAM chunk body not word-aligned; skipping this chunk");
                    } else if (length % sizeof(uint32_t) != 0) {
                        log_warning(
                            tt::LogMetal,
                            "DPRINT: dispatch DRAM chunk body length not multiple of word size; skipping this "
                            "chunk");
                    } else {
                        std::span<uint32_t> data =
                            std::span(payload_vector).subspan(body_start / sizeof(uint32_t), length / sizeof(uint32_t));
                        print_buffer_data(device_id, core_it->second, data);
                    }
                } else {
                    size_t rw_offset = rw_ptrs_inside_align ? (pos + header_size) : (body_start + length);
                    if (rw_offset + rw_ptrs_size > payload.size()) {
                        log_warning(tt::LogMetal, "DPRINT: dispatch DRAM wrap chunk rw_pointers truncated; stopping");
                        break;
                    }
                    uint16_t wrap_wpos = 0, wrap_rpos = 0;
                    std::memcpy(&wrap_wpos, payload.data() + rw_offset, sizeof(uint16_t));
                    std::memcpy(&wrap_rpos, payload.data() + rw_offset + sizeof(uint16_t), sizeof(uint16_t));
                    if (wrap_rpos <= length && wrap_wpos <= length) {
                        if (body_start % sizeof(uint32_t) != 0) {
                            log_warning(
                                tt::LogMetal, "DPRINT: dispatch DRAM chunk body not word-aligned; skipping this chunk");
                        } else if (length % sizeof(uint32_t) != 0) {
                            log_warning(
                                tt::LogMetal,
                                "DPRINT: dispatch DRAM chunk body length not multiple of word size; skipping "
                                "this chunk");
                        } else if (wrap_rpos % sizeof(uint32_t) != 0 || wrap_wpos % sizeof(uint32_t) != 0) {
                            log_warning(
                                tt::LogMetal,
                                "DPRINT: dispatch DRAM chunk wrap pointers not word-aligned; skipping this "
                                "chunk");
                        } else {
                            if (wrap_rpos > wrap_wpos) {
                                const size_t tail_len = length - wrap_rpos;
                                if (tail_len > 0) {
                                    std::span<uint32_t> data = std::span(payload_vector)
                                                                   .subspan(
                                                                       (body_start + wrap_rpos) / sizeof(uint32_t),
                                                                       tail_len / sizeof(uint32_t));
                                    print_buffer_data(device_id, core_it->second, data);
                                }
                                wrap_rpos = 0;
                            }
                            if (wrap_rpos < wrap_wpos) {
                                const size_t head_len = wrap_wpos;
                                if (head_len > 0) {
                                    std::span<uint32_t> data =
                                        std::span(payload_vector)
                                            .subspan(body_start / sizeof(uint32_t), head_len / sizeof(uint32_t));
                                    print_buffer_data(device_id, core_it->second, data);
                                }
                            }
                        }
                    }
                }
            }

            // Each chunk is dram-aligned in the kernel; advance accordingly.
            pos += round_up(chunk_end_bytes, dram_align);
        }
        // Flush once per drain window instead of per line (see print_buffer_data). This batches the
        // millions of lines a drain can emit into far fewer write() syscalls, which is what dominated
        // runtime on a journaled filesystem. Output still appears per drain (every few ms).
        flush_output_streams();

        // Update read pointer in DRAM.
        const uint32_t new_read_pointer = wpos;
        cluster.write_dram_vec(
            &new_read_pointer,
            sizeof(uint32_t),
            device_id,
            data.dram_view,
            data.rw_pointers_address + sizeof(uint32_t));

        if (dispatcher_finished) {
            // Dispatcher exited and we just drained its final batch. Also do one
            // per-L1 sweep to catch any prints written between dispatch's last
            // remote rw-pointer read and a kernel's final wpos update, then mark
            // sticky-disabled so subsequent polls skip the DRAM path.
            data.disabled = true;
            (void)fall_back_to_l1();
        }
        return true;
    } catch (std::runtime_error&) {
        if (env_.get_rtoptions().get_test_mode_enabled()) {
            server_killed_due_to_hang_ = true;
            return false;
        }
        throw;
    }
}

DPrintServer::Impl::Impl(
    MetalContext* context, MetalEnv& env, uint8_t num_hw_cqs, const DispatchCoreConfig& dispatch_core_config) :
    context_(context),
    env_(MetalEnvAccessor(env).impl()),
    num_hw_cqs_(num_hw_cqs),
    dispatch_core_config_(dispatch_core_config) {
    Inspector::enable_kernel_path_collection();

    // Read risc mask + log file from rtoptions
    string file_name = env_.get_rtoptions().get_feature_file_name(tt::llrt::RunTimeDebugFeatureDprint);
    bool one_file_per_risc = env_.get_rtoptions().get_feature_one_file_per_risc(tt::llrt::RunTimeDebugFeatureDprint);
    bool prepend_device_core_risc =
        env_.get_rtoptions().get_feature_prepend_device_core_risc(tt::llrt::RunTimeDebugFeatureDprint);

    // One file per risc auto-generates the output files and ignores the env var for it. Print a warning if both are
    // specified just in case.
    if (!file_name.empty() && one_file_per_risc) {
        log_warning(
            tt::LogMetal,
            "Both TT_METAL_DPRINT_FILE_NAME and TT_METAL_DPRINT_ONE_FILE_PER_RISC are specified. "
            "TT_METAL_DPRINT_FILE_NAME will be ignored.");
    }

    if (prepend_device_core_risc && one_file_per_risc) {
        log_warning(
            tt::LogMetal,
            "Both TT_METAL_DPRINT_PREPEND_DEVICE_CORE_RISC and TT_METAL_DPRINT_ONE_FILE_PER_RISC are specified. "
            "TT_METAL_DPRINT_PREPEND_DEVICE_CORE_RISC will be disabled.");
        env_.get_rtoptions().set_feature_prepend_device_core_risc(tt::llrt::RunTimeDebugFeatureDprint, false);
    }

    // Set the output stream according to RTOptions, either a file name or stdout if none specified.
    std::filesystem::path output_dir(env_.get_rtoptions().get_logs_dir() + logfile_path);
    std::filesystem::create_directories(output_dir);
    if (!file_name.empty() && !one_file_per_risc) {
        outfile_ = new ofstream(file_name);
    }
    stream_ = outfile_ ? outfile_ : &std::cout;

    // Spin off the thread that runs the print server.
    print_server_thread_ = new std::thread([this] { poll_print_data(); });
}  // Impl::Impl

DPrintServer::Impl::~Impl() {
    // Signal the print server thread to finish
    stop_print_server_ = true;

    // Wait for the thread to end, with a timeout
    auto future = std::async(std::launch::async, &std::thread::join, print_server_thread_);
    const int join_timeout_sec = debug_server_finish_timeout_sec(env_.get_rtoptions());
    if (future.wait_for(std::chrono::seconds(join_timeout_sec)) == std::future_status::timeout) {
        log_fatal(tt::LogMetal, "Timed out waiting on debug print thread to terminate ({}s).", join_timeout_sec);
    }
    delete print_server_thread_;
    print_server_thread_ = nullptr;

    if (outfile_) {
        outfile_->close();
        delete outfile_;
    }
    for (auto& key_and_stream : risc_to_file_stream_) {
        key_and_stream.second->close();
        delete key_and_stream.second;
    }
    // Parser instances are automatically cleaned up via unique_ptr
}  // Impl::~Impl

void DPrintServer::Impl::await() {
    // Wait for dispatch to collect all data into DRAM.
    // Since this happens during full dispatch pass, we need to wait for it to happen.
    if (!server_killed_due_to_hang_) {
        const uint32_t full_us = env_.get_rtoptions().get_device_print_dispatch_full_us();
        // 2 full-dispatch windows (full_us is in microseconds → divide by 1000 for ms), just in case.
        const uint32_t initial_wait_ms = 2 * ((full_us + 999) / 1000);
        std::this_thread::sleep_for(std::chrono::milliseconds(initial_wait_ms));
    }
    auto poll_until_no_new_data = [&]() {
        // Simply poll the flag every few ms to check whether new data is still being processed,
        // or whether any cores are waiting for a signal to be raised.
        // TODO(dma): once we have access to the device is there a way we can poll the device to
        // check whether more print data is coming?
        size_t num_riscs_waiting = 0;

        // Make sure to run at least one full iteration inside poll_print_data before returning.
        wait_loop_iterations_ = 0;

        do {
            // No need to await if the server was killed already due to a hang.
            if (server_killed_due_to_hang_) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        } while (num_riscs_waiting > 0 || new_data_last_iter_ || wait_loop_iterations_ < 2);
    };
    auto future = std::async(std::launch::async, poll_until_no_new_data);
    const int await_timeout_sec = debug_server_wait_timeout_sec(env_.get_rtoptions());
    if (future.wait_for(std::chrono::seconds(await_timeout_sec)) == std::future_status::timeout) {
        TT_THROW("Timed out waiting on debug print server to read data ({}s).", await_timeout_sec);
    }
}  // await

void DPrintServer::Impl::init_device(ChipId device_id) {
    auto& cluster = env_.get_cluster();
    auto& control_plane = env_.get_control_plane();
    CoreDescriptorSet all_cores = GetAllCores(cluster, control_plane, device_id);
    // Initialize all print buffers on all cores on the device to have print disabled magic. We
    // will then write print enabled magic for only the cores the user has specified to monitor.
    // This way in the kernel code (dprint.h) we can detect whether the magic value is present and
    // skip prints entirely to prevent kernel code from hanging waiting for the print buffer to be
    // flushed from the host.
    for (const auto& logical_core : all_cores) {
        init_print_buffers_for_core(device_id, logical_core);
    }
}

void DPrintServer::Impl::attach_devices() {
    auto all_devices = env_.get_cluster().all_chip_ids();

    // Always init all chips, to disable prints by default.
    for (ChipId device_id : all_devices) {
        init_device(device_id);
    }

    // If RTOptions enables all chips, then attach all chips. Otherwise only attach specified devices.
    if (env_.get_rtoptions().get_feature_all_chips(tt::llrt::RunTimeDebugFeatureDprint)) {
        for (ChipId device_id : all_devices) {
            attach_device(device_id);
        }
    } else {
        for (ChipId device_id : env_.get_rtoptions().get_feature_chip_ids(tt::llrt::RunTimeDebugFeatureDprint)) {
            attach_device(device_id);
        }
    }
}

void DPrintServer::Impl::attach_device(ChipId device_id) {
    // A set of all valid printable cores, used for checking the user input. Note that the coords
    // here are virtual.
    auto& cluster = env_.get_cluster();
    auto& control_plane = env_.get_control_plane();
    tt::tt_metal::CoreDescriptorSet all_cores = tt::tt_metal::GetAllCores(cluster, control_plane, device_id);
    tt::tt_metal::CoreDescriptorSet dispatch_cores =
        tt::tt_metal::GetDispatchCores(env_, device_id, num_hw_cqs_, dispatch_core_config_);

    // If RTOptions doesn't enable DPRINT on this device, return here and don't actually attach it
    // to the server.
    const auto& rtoptions = env_.get_rtoptions();
    std::vector<ChipId> chip_ids = rtoptions.get_feature_chip_ids(tt::llrt::RunTimeDebugFeatureDprint);
    if (!rtoptions.get_feature_all_chips(tt::llrt::RunTimeDebugFeatureDprint)) {
        if (std::find(chip_ids.begin(), chip_ids.end(), device_id) == chip_ids.end()) {
            return;
        }
    }

    // Core range depends on whether dprint_all_cores flag is set.
    std::vector<umd::CoreDescriptor> print_cores_sanitized;
    const auto& hal = env_.get_hal();
    std::vector<CoreType> core_types_to_check = {CoreType::WORKER, CoreType::ETH};
    if (hal.has_programmable_core_type(HalProgrammableCoreType::DRAM)) {
        core_types_to_check.push_back(CoreType::DRAM);
    }
    for (CoreType core_type : core_types_to_check) {
        if (rtoptions.get_feature_all_cores(tt::llrt::RunTimeDebugFeatureDprint, core_type) ==
            tt::llrt::RunTimeDebugClassAll) {
            // Print from all cores of the given type, cores returned here are guaranteed to be valid.
            for (umd::CoreDescriptor logical_core : all_cores) {
                if (logical_core.type == core_type) {
                    print_cores_sanitized.push_back(logical_core);
                }
            }
            log_info(
                tt::LogMetal,
                "DPRINT enabled on device {}, all {} cores.",
                device_id,
                tt::tt_metal::get_core_type_name(core_type));
        } else if (
            rtoptions.get_feature_all_cores(tt::llrt::RunTimeDebugFeatureDprint, core_type) ==
            tt::llrt::RunTimeDebugClassDispatch) {
            for (umd::CoreDescriptor logical_core : dispatch_cores) {
                if (logical_core.type == core_type) {
                    print_cores_sanitized.push_back(logical_core);
                }
            }
            log_info(
                tt::LogMetal,
                "DPRINT enabled on device {}, {} dispatch cores.",
                device_id,
                tt::tt_metal::get_core_type_name(core_type));
        } else if (
            rtoptions.get_feature_all_cores(tt::llrt::RunTimeDebugFeatureDprint, core_type) ==
            tt::llrt::RunTimeDebugClassWorker) {
            // For worker cores, take all cores and remove dispatch cores.
            for (umd::CoreDescriptor logical_core : all_cores) {
                if (!dispatch_cores.contains(logical_core)) {
                    if (logical_core.type == core_type) {
                        print_cores_sanitized.push_back(logical_core);
                    }
                }
            }
            log_info(
                tt::LogMetal,
                "DPRINT enabled on device {}, {} worker cores.",
                device_id,
                tt::tt_metal::get_core_type_name(core_type));
        } else {
            // No "all cores" option provided, which means print from the cores specified by the user
            const std::vector<CoreCoord>& print_cores =
                rtoptions.get_feature_cores(tt::llrt::RunTimeDebugFeatureDprint).at(core_type);

            // We should also validate that the cores the user specified are valid worker cores.
            for (const auto& logical_core : print_cores) {
                // Need to convert user-specified logical cores to virtual cores, this can throw
                // if the user gave bad coords.
                CoreCoord virtual_core;
                bool valid_logical_core = true;
                try {
                    virtual_core = env_.get_cluster().get_virtual_coordinate_from_logical_coordinates(
                        device_id, logical_core, core_type);
                } catch (std::runtime_error& error) {
                    valid_logical_core = false;
                }
                if (valid_logical_core && all_cores.contains({logical_core, core_type})) {
                    print_cores_sanitized.push_back({logical_core, core_type});
                    log_info(
                        tt::LogMetal,
                        "DPRINT enabled on device {}, {} core {} (virtual {}).",
                        device_id,
                        tt::tt_metal::get_core_type_name(core_type),
                        logical_core.str(),
                        virtual_core.str());
                } else {
                    log_warning(
                        tt::LogMetal,
                        "TT_METAL_DPRINT_CORES included {} core with logical coordinates {} (virtual coordinates {}), "
                        "which is not a valid core on device {}. This coordinate will be ignored by the dprint server.",
                        tt::tt_metal::get_core_type_name(core_type),
                        logical_core.str(),
                        valid_logical_core ? virtual_core.str() : "INVALID",
                        device_id);
                }
            }
        }
    }

    // Write print enable magic for the cores the user specified.
    for (auto& logical_core : print_cores_sanitized) {
        enable_print_buffers_for_core(device_id, logical_core);
        if (dispatch_cores.contains(logical_core)) {
            device_reads_dispatch_cores_[device_id] = true;
        }
    }

    // Save this device + core range to the print server
    {
        std::lock_guard lock(device_to_core_range_lock_);
        TT_ASSERT(
            !device_to_core_range_.contains(device_id), "Device {} added to DPRINT server more than once!", device_id);
        device_to_core_range_[device_id] = print_cores_sanitized;
    }
    log_info(tt::LogMetal, "DPRINT Server attached device {}", device_id);

    // Set up dispatch_s DRAM aggregation for this device (when dispatch_s is enabled).
    if (!context_->get_dispatch_query_manager().dispatch_s_enabled()) {
        return;
    }

    const uint32_t dram_alignment = hal.get_alignment(HalMemType::DRAM);
    DispatchDramData data;
    data.dram_view = 0;
    data.rw_pointers_address = hal.get_dev_addr(HalDramMemAddrType::DEVICE_PRINT_DISPATCH);
    data.buffer_address = data.rw_pointers_address + dram_alignment;
    data.buffer_size = hal.get_dev_size(HalDramMemAddrType::DEVICE_PRINT_DISPATCH) - dram_alignment;
    data.disabled = false;
    data.noc_to_core.clear();

    // Build the (virtual NOC1 x, y) -> CoreDescriptor map dispatch_s reports headers in.
    // Replicates Device::virtual_noc0_coordinate(NOC_1, virtual_core) without an IDevice
    // (IDevices aren't registered yet at attach_device time): for Blackhole NOC0 and NOC1
    // share the virtual coordinate space, and for other archs the cluster's soc_desc
    // grid_size + HAL noc_coordinate transform yields the NOC1 coord.
    const auto& print_cores = get_print_cores(device_id);
    for (const auto& core_desc : print_cores) {
        auto core_virtual =
            cluster.get_virtual_coordinate_from_logical_coordinates(device_id, core_desc.coord, core_desc.type);
        data.noc_to_core.emplace(std::make_pair(core_virtual.x, core_virtual.y), core_desc);
    }

    // Initialize read/write pointers
    std::array<uint32_t, 5> rw_init = {DEBUG_PRINT_SERVER_STARTING_MAGIC, 0u, 0u, 0u, 0u};
    cluster.write_dram_vec(rw_init.data(), sizeof(rw_init), device_id, data.dram_view, data.rw_pointers_address);

    // Store dispatch DRAM data for this device so that we can use it in poll_device_print_data.
    // There is race condition between poll_device_print_data and attach_device, so adding dispatch
    // DRAM data to map at the end of this function to make sure poll_device_print_data won't read
    // uninitialized data and update our structure.
    {
        std::lock_guard lock(dispatch_dram_data_lock_);
        dispatch_dram_data_[device_id] = std::move(data);
    }

    // In case we are running multiple tests, we want to make sure there is no race condition
    // between dispatch kernel and host. After writing STARTING_MAGIC, we want to make sure dispatch
    // observes it and updates rw pointers before we start polling.
    // In regular flow, we will do this initialization before dispatch kernel starts.
    const uint32_t full_us = env_.get_rtoptions().get_device_print_dispatch_full_us();
    const uint32_t wait_ms = (full_us + 999) / 1000 + 5;
    std::this_thread::sleep_for(std::chrono::milliseconds(wait_ms));
}  // attach_device

void DPrintServer::Impl::detach_devices() {
    // Make a copy of devices to detach, since we'll be modiying device_to_core_range_A
    std::set<ChipId> devices_to_detach;
    for (const auto& id_and_core_range : device_to_core_range_) {
        devices_to_detach.insert(id_and_core_range.first);
    }

    for (ChipId device_id : devices_to_detach) {
        detach_device(device_id);
    }
}

void DPrintServer::Impl::detach_device(ChipId device_id) {
    auto& cluster = env_.get_cluster();
    auto& control_plane = env_.get_control_plane();

    std::lock_guard lock(device_to_core_range_lock_);
    TT_ASSERT(
        device_to_core_range_.contains(device_id),
        "Device {} not present in DPRINT server but tried removing it!",
        device_id);
    device_to_core_range_.erase(device_id);
    log_info(LogMetal, "DPRINT Server detached device {}", device_id);

    // When detaching a device, disable prints on it.
    tt::tt_metal::CoreDescriptorSet all_cores = tt::tt_metal::GetAllCores(cluster, control_plane, device_id);
    for (const auto& logical_core : all_cores) {
        init_print_buffers_for_core(device_id, logical_core);
    }
}  // detach_device

void DPrintServer::Impl::clear_log_file() {
    if (outfile_) {
        auto& rtoptions = env_.get_rtoptions();
        // Just close the file and re-open it (without append) to clear it.
        outfile_->close();
        delete outfile_;

        string file_name = rtoptions.get_feature_file_name(tt::llrt::RunTimeDebugFeatureDprint);
        outfile_ = new ofstream(file_name);
        stream_ = outfile_ ? outfile_ : &std::cout;
    }
}  // clear_log_file

void DPrintServer::Impl::poll_print_data() {
    // Give the print server thread a reasonable name.
    pthread_setname_np(pthread_self(), "TT_DPRINT_SERVER");

    // Main print loop, go through all chips/cores/riscs on the device and poll for any print data
    // written.
    while (true) {
        if (stop_print_server_ && !new_data_last_iter_) {
            // If the stop signal was received, exit the print server thread after all new data has been processed.
            break;
        }

        // Make a copy of the device->core map, so that it can be modified while polling.
        std::map<ChipId, std::vector<umd::CoreDescriptor>> device_to_core_range_copy;
        {
            std::lock_guard lock(device_to_core_range_lock_);
            device_to_core_range_copy = device_to_core_range_;
        }

        // Flag for whether any new print data was found in this round of polling.
        bool new_data_this_iter = false;
        for (auto& device_and_cores : device_to_core_range_copy) {
            ChipId device_id = device_and_cores.first;
            new_data_this_iter |= poll_device_print_data(device_id, device_and_cores.second);

            // If this read detected a print hang, stop processing prints.
            if (server_killed_due_to_hang_) {
                return;
            }
        }

        // Signal whether the print server is currently processing data.
        new_data_last_iter_ = new_data_this_iter;
        // Sleep for a few ms if no data was processed.
        if (!new_data_last_iter_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        wait_loop_iterations_++;
    }
}  // poll_print_data

bool DPrintServer::Impl::poll_device_print_data_l1(
    ChipId device_id, const std::vector<umd::CoreDescriptor>& logical_cores) {
    bool new_data_this_iter = false;

    for (const auto& logical_core : logical_cores) {
        try {
            new_data_this_iter |= poll_one_core(device_id, logical_core, new_data_this_iter);
        } catch (std::runtime_error& e) {
            // Depending on if test mode is enabled, catch and stop server, or
            // re-throw the exception.
            if (env_.get_rtoptions().get_test_mode_enabled()) {
                server_killed_due_to_hang_ = true;
                flush_output_streams();
                return new_data_this_iter;  // Stop the print loop
            }  // Re-throw for instant exit
            throw e;
        }

        // If this read detected a print hang, stop processing prints.
        if (server_killed_due_to_hang_) {
            flush_output_streams();
            return new_data_this_iter;
        }
    }
    // Flush here too: the L1-fallback path (used when dispatch_s isn't running — early boot,
    // self-disabled, or after the dispatcher finished) is where prompt, complete output matters
    // most for hang debugging, and print_buffer_data no longer flushes per line.
    if (new_data_this_iter) {
        flush_output_streams();
    }
    return new_data_this_iter;
}  // poll_device_print_data_l1

void DPrintServer::Impl::flush_output_streams() {
    if (stream_ != nullptr) {
        stream_->flush();
    }
    for (auto& [risc_key, risc_stream] : risc_to_file_stream_) {
        if (risc_stream != nullptr) {
            risc_stream->flush();
        }
    }
}  // flush_output_streams

ostream* DPrintServer::Impl::get_output_stream(const RiscKey& risc_key) {
    ostream* output_stream = stream_;
    auto& cluster = env_.get_cluster();
    const auto& hal = env_.get_hal();
    const auto& rtoptions = env_.get_rtoptions();
    if (rtoptions.get_feature_one_file_per_risc(tt::llrt::RunTimeDebugFeatureDprint)) {
        if (!risc_to_file_stream_[risc_key]) {
            const ChipId chip_id = get<0>(risc_key);
            const umd::CoreDescriptor& logical_core = get<1>(risc_key);
            const int risc_id = get<2>(risc_key);
            string filename = rtoptions.get_logs_dir() + logfile_path;
            filename += fmt::format(
                "device-{}_{}-core-{}-{}_{}.txt",
                chip_id,
                tt::tt_metal::get_core_type_name(logical_core.type),
                logical_core.coord.x,
                logical_core.coord.y,
                GetRiscName(cluster, hal, chip_id, logical_core, risc_id));
            risc_to_file_stream_[risc_key] = new ofstream(filename);
        }
        output_stream = risc_to_file_stream_[risc_key];
    }

    if (mute_print_server_) {
        output_stream = &null_stream;
    }

    return output_stream;
}  // get_output_stream

// Wrapper class functions
DPrintServer::DPrintServer(
    MetalContext* context, MetalEnv& env, uint8_t num_hw_cqs, const DispatchCoreConfig& dispatch_core_config) {
    impl_ = std::make_unique<DPrintServer::Impl>(context, env, num_hw_cqs, dispatch_core_config);
}
DPrintServer::~DPrintServer() = default;
void DPrintServer::set_mute(bool mute_print_server) { impl_->set_mute(mute_print_server); }
void DPrintServer::await() { impl_->await(); }
void DPrintServer::attach_devices() { impl_->attach_devices(); }
void DPrintServer::detach_devices() { impl_->detach_devices(); }
void DPrintServer::clear_log_file() { impl_->clear_log_file(); }
bool DPrintServer::reads_dispatch_cores(ChipId device_id) { return impl_->reads_dispatch_cores(device_id); }
bool DPrintServer::hang_detected() { return impl_->hang_detected(); }
std::vector<umd::CoreDescriptor> DPrintServer::get_print_cores(ChipId device_id) const {
    return impl_->get_print_cores(device_id);
}
std::vector<DPrintBufferInfo> DPrintServer::get_core_buffers(
    ChipId device_id, const umd::CoreDescriptor& print_core) const {
    return impl_->get_core_buffers(device_id, print_core);
}

}  // namespace tt::tt_metal
