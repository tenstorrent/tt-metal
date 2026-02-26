// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cctype>
#include <cmath>
#include <pthread.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <future>
#include <iomanip>
#include <ios>
#include <iostream>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>
#include <fstream>

#include <enchantum/enchantum.hpp>
#include <tt-logger/tt-logger.hpp>
#include "impl/data_format/blockfloat_common.hpp"
#include <tt_stl/assert.hpp>
#include <umd/device/types/core_coordinates.hpp>
#include <umd/device/soc_descriptor.hpp>
#include <umd/device/types/xy_pair.hpp>

#include "core_coord.hpp"
#include "debug_helpers.hpp"
#include "dprint_server.hpp"
#include "dprint_parser.hpp"
#include "fmt/base.h"
#include "hal_types.hpp"
#include "hostdevcommon/dprint_common.h"
#include "hostdevcommon/kernel_structs.h"
#include "llrt.hpp"
#include "impl/context/metal_context.hpp"
#include "tt_backend_api_types.hpp"
#include <llrt/tt_cluster.hpp>
#include "impl/debug/inspector/inspector.hpp"
#include "tt_metal/llrt/tt_elffile.hpp"
#include "tt_stl/span.hpp"

using std::flush;
using std::int32_t;
using std::ofstream;
using std::ostream;
using std::ostringstream;
using std::set;
using std::string;
using std::to_string;
using std::tuple;
using std::uint32_t;

using namespace tt;

#define CAST_U8P(p) (reinterpret_cast<uint8_t*>(p))

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

string GetRiscName(ChipId device_id, const umd::CoreDescriptor& logical_core, int risc_id, bool abbreviated = false) {
    CoreCoord virtual_core =
        tt::tt_metal::MetalContext::instance().get_cluster().get_virtual_coordinate_from_logical_coordinates(
            device_id, logical_core.coord, logical_core.type);
    auto programmable_core_type = llrt::get_core_type(device_id, virtual_core);
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    return hal.get_processor_class_name(programmable_core_type, risc_id, abbreviated);
}

inline bool RiscEnabled(tt_metal::HalProgrammableCoreType core_type, int risc_index) {
    const auto& processors =
        tt::tt_metal::MetalContext::instance().rtoptions().get_feature_processors(tt::llrt::RunTimeDebugFeatureDprint);
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
    ChipId device_id,
    const CoreCoord& virtual_core,
    uint64_t base_addr,
    bool enabled,
    uint32_t buffer_size = DPRINT_BUFFER_SIZE) {
    // TODO(AP): this could use a cleanup - need a different mechanism to know if a kernel is running on device.
    // Force wait for first kernel launch by first writing a non-zero and waiting for a zero.
    std::vector<uint32_t> initbuf = std::vector<uint32_t>(buffer_size / sizeof(uint32_t), 0);
    initbuf[0] = uint32_t(enabled ? DEBUG_PRINT_SERVER_STARTING_MAGIC : DEBUG_PRINT_SERVER_DISABLED_MAGIC);
    tt::tt_metal::MetalContext::instance().get_cluster().write_core(device_id, virtual_core, initbuf, base_addr);

    // Prevent race conditions during runtime by waiting until the init value is actually written
    // DPrint is only used for debug purposes so this delay should not be a big issue.
    // 1. host will read remote and think the wpos is 0. so it'll go and poll the data
    // 2. the packet will arrive to set the wpos = DEBUG_PRINT_SERVER_STARTING_MAGIC
    // 3. the actual host polling function will read wpos = DEBUG_PRINT_SERVER_STARTING_MAGIC
    // 4. now we will access wpos at the starting magic which is incorrect
    uint32_t num_tries = 100000;
    while (num_tries-- > 0) {
        auto result =
            tt::tt_metal::MetalContext::instance().get_cluster().read_core(device_id, virtual_core, base_addr, 4);
        if ((result[0] == DEBUG_PRINT_SERVER_STARTING_MAGIC && enabled) ||
            (result[0] == DEBUG_PRINT_SERVER_DISABLED_MAGIC && !enabled)) {
            return;
        }
    }
    TT_THROW("Timed out writing init magic");
}  // WriteInitMagic

// Checks if our magic value was cleared by the device code at the given buffer address.
// The assumption is that if our magic number was cleared,
// it means there is a write in the queue and wpos/rpos are now valid
// Note that this is not a bulletproof way to bootstrap the print server (TODO(AP))
bool CheckInitMagicCleared(ChipId device_id, const CoreCoord& virtual_core, uint64_t base_addr) {
    auto result = tt::tt_metal::MetalContext::instance().get_cluster().read_core(device_id, virtual_core, base_addr, 4);
    return (result[0] != DEBUG_PRINT_SERVER_STARTING_MAGIC && result[0] != DEBUG_PRINT_SERVER_DISABLED_MAGIC);
}  // CheckInitMagicCleared

}  // namespace

namespace tt::tt_metal {

// Base class for DPrintServer implementations.
// Contains all common state and logic; subclasses only override peek_one_risc_non_blocking.
class DPrintServer::Impl {
public:
    Impl(llrt::RunTimeOptions& rtoptions);
    Impl() = delete;
    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;
    virtual ~Impl();

    void set_mute(bool mute_print_server) { mute_print_server_ = mute_print_server; }
    void await();
    void attach_devices();
    void detach_devices();
    void clear_log_file();
    bool reads_dispatch_cores(ChipId device_id) { return device_reads_dispatch_cores_[device_id]; }
    bool hang_detected() { return server_killed_due_to_hang_; }

protected:
    // Polls one core for any new print data and outputs it. Returns true if some data was read.
    // Old DPRINT iterates over per-RISC buffers; new DEVICE_PRINT reads a single per-core buffer.
    virtual bool poll_one_core(ChipId device_id, const umd::CoreDescriptor& logical_core, bool new_data_this_iter) = 0;

    // Writes disabled init magic to all print buffers on a single core.
    // DPrintImpl: iterates over all per-RISC buffers.
    // DevicePrintImpl: writes to the single shared buffer once.
    virtual void init_print_buffers_for_core(
        ChipId device_id, const CoreCoord& virtual_core, HalProgrammableCoreType core_type) = 0;

    // Writes enabled init magic to the appropriate print buffers on a single core.
    // DPrintImpl: uses RiscEnabled() to iterate over only the enabled per-RISC buffers.
    // DevicePrintImpl: writes to the single shared buffer once.
    virtual void enable_print_buffers_for_core(
        ChipId device_id, const CoreCoord& virtual_core, HalProgrammableCoreType core_type) = 0;

    // Returns true if the given core has any outstanding unread print data.
    // DPrintImpl: checks across all enabled per-RISC buffers.
    // DevicePrintImpl: checks the single shared buffer.
    virtual bool core_has_outstanding_prints(
        ChipId device_id, const CoreCoord& virtual_core, HalProgrammableCoreType core_type) = 0;

    // Flag for main thread to signal the print server thread to stop.
    std::atomic<bool> stop_print_server_ = false;
    // Flag for muting the print server. This doesn't disable reading print data from the device,
    // but it supresses the output of that print data the user.
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

    // Parser instances for each risc (handles parsing state, intermediate buffering, and line prefixing).
    std::map<RiscKey, std::unique_ptr<DPrintParser>, RiscKeyComparator> risc_to_parser_;

    // For printing each risc's dprint to a separate file, a map from {device id, core, risc index} to files.
    std::map<RiscKey, ofstream*, RiscKeyComparator> risc_to_file_stream_;

    // A map from Device -> Core Range, which is used to determine which cores on which devices
    // to scan for print data. Also a lock for editing it.
    std::map<ChipId, std::vector<umd::CoreDescriptor>> device_to_core_range_;
    std::map<ChipId, bool> device_reads_dispatch_cores_;  // True if given device reads any dispatch cores. Used to
                                                          // know whether dprint can be compiled out.
    std::mutex device_to_core_range_lock_;

    // Used to signal to the print server to flush all intermediate streams for a device so that any remaining prints
    // are printed out.
    std::map<ChipId, bool> device_intermediate_streams_force_flush_;
    std::mutex device_intermediate_streams_force_flush_lock_;

    // Polls specified cores/riscs on all attached devices and prints any new print data. This
    // function is the main loop for the print server thread.
    void poll_print_data();

    // Transfers data from all parser intermediate streams to output stream and flushes it.
    void transfer_all_streams_to_output(ChipId device_id);

    // Returns the stream that the dprint data should be output to. Can be auto-generated files, the user-selected file,
    // stdout, or nothing.
    ostream* get_output_stream(const RiscKey& risc_key);

    // Helper functions to init/attach/detach a single device
    void init_device(ChipId device_id);
    void attach_device(ChipId device_id);
    void detach_device(ChipId device_id);
};

// Original DPRINT implementation: reads DPRINT buffers written by device kernels.
class DPrintImpl : public DPrintServer::Impl {
public:
    DPrintImpl(llrt::RunTimeOptions& rtoptions) : DPrintServer::Impl(rtoptions) {}

protected:
    bool poll_one_core(ChipId device_id, const umd::CoreDescriptor& logical_core, bool new_data_this_iter) override;
    void init_print_buffers_for_core(
        ChipId device_id, const CoreCoord& virtual_core, HalProgrammableCoreType core_type) override;
    void enable_print_buffers_for_core(
        ChipId device_id, const CoreCoord& virtual_core, HalProgrammableCoreType core_type) override;
    bool core_has_outstanding_prints(
        ChipId device_id, const CoreCoord& virtual_core, HalProgrammableCoreType core_type) override;

private:
    // Reads the DPRINT buffer for a single RISC and outputs any new data.
    bool peek_one_risc_non_blocking(
        ChipId device_id, const umd::CoreDescriptor& logical_core, int risc_id, bool new_data_this_iter);
};

// New DEVICE_PRINT implementation (stub - to be implemented).

// TODO: Can we reuse structure defined for device?
struct DevicePrintHeader {
    static constexpr uint32_t max_info_id_value = 65535;
    static constexpr uint32_t max_message_payload_size = 1023;  // 10 bits for message_payload

    uint32_t is_kernel : 1;         // 0 = firmware, 1 = kernel
    uint32_t risc_id : 5;           // 0-31 risc id (supporting quasar)
    uint32_t message_payload : 10;  // Message payload size (<1024 bytes)
    uint32_t info_id : 16;          // Index into .device_print_strings_info (max 65536 entries)
};
static_assert(sizeof(DevicePrintHeader) == sizeof(uint32_t));

struct DevicePrintStringInfo {
    std::uint32_t format_string_ptr;
    std::uint32_t file;
    std::uint32_t line;
};

class DevicePrintImpl : public DPrintServer::Impl {
public:
    DevicePrintImpl(llrt::RunTimeOptions& rtoptions) : DPrintServer::Impl(rtoptions) {}

protected:
    bool poll_one_core(ChipId device_id, const umd::CoreDescriptor& logical_core, bool new_data_this_iter) override;
    void init_print_buffers_for_core(
        ChipId device_id, const CoreCoord& virtual_core, HalProgrammableCoreType core_type) override;
    void enable_print_buffers_for_core(
        ChipId device_id, const CoreCoord& virtual_core, HalProgrammableCoreType core_type) override;
    bool core_has_outstanding_prints(
        ChipId device_id, const CoreCoord& virtual_core, HalProgrammableCoreType core_type) override;

private:
    void print_buffer_data(
        ChipId device_id, const umd::CoreDescriptor& logical_core, const std::vector<uint32_t>& data);
    std::string format_message(std::string_view format_str, std::span<const std::byte> payload_bytes);

    struct ElfFileCacheEntry {
        uint32_t ref_count;
        ll_api::ElfFile elf_file;
        std::span<std::byte> format_strings_info_bytes;
        uint64_t format_strings_info_address;
        std::span<std::byte> format_strings_bytes;
        uint64_t format_strings_address;
        DevicePrintStringInfo* string_info_ptr = nullptr;
        size_t string_info_size = 0;

        void load_elf(const std::filesystem::path& elf_path) {
            try {
                elf_file.ReadImage(elf_path);
                format_strings_info_bytes =
                    elf_file.GetSectionContents(".device_print_strings_info", format_strings_info_address);
                format_strings_bytes = elf_file.GetSectionContents(".device_print_strings", format_strings_address);
                string_info_ptr = reinterpret_cast<DevicePrintStringInfo*>(format_strings_info_bytes.data());
                string_info_size = format_strings_info_bytes.size() / sizeof(DevicePrintStringInfo);
            } catch (...) {
                // Failed to load ELF file
                log_warning(tt::LogMetal, "Failed to load ELF file {}", elf_path.string());
            }
        }
    };

    struct RiscData {
        std::string firmare_elf_path;
        ElfFileCacheEntry* firmware_elf_cache_entry = nullptr;
        std::string kernel_elf_path;
        ElfFileCacheEntry* kernel_elf_cache_entry = nullptr;
        std::string message_buffer;
        std::optional<std::string> line_prefix;
        int last_loaded_kernel_id = -1;
    };

    std::map<std::string, ElfFileCacheEntry> elf_cache_;
    std::map<RiscKey, RiscData, RiscKeyComparator> risc_data_;
};

void DPrintImpl::init_print_buffers_for_core(
    ChipId device_id, const CoreCoord& virtual_core, HalProgrammableCoreType core_type) {
    uint32_t num_processors = MetalContext::instance().hal().get_num_risc_processors(core_type);
    for (int risc_index = 0; risc_index < num_processors; risc_index++) {
        WriteInitMagic(device_id, virtual_core, GetDprintBufAddr(device_id, virtual_core, risc_index), false);
    }
}

void DPrintImpl::enable_print_buffers_for_core(
    ChipId device_id, const CoreCoord& virtual_core, HalProgrammableCoreType core_type) {
    uint32_t num_processors = MetalContext::instance().hal().get_num_risc_processors(core_type);
    for (int risc_index = 0; risc_index < num_processors; risc_index++) {
        if (RiscEnabled(core_type, risc_index)) {
            WriteInitMagic(device_id, virtual_core, GetDprintBufAddr(device_id, virtual_core, risc_index), true);
        }
    }
}

bool DPrintImpl::core_has_outstanding_prints(
    ChipId device_id, const CoreCoord& virtual_core, HalProgrammableCoreType core_type) {
    uint32_t num_processors = MetalContext::instance().hal().get_num_risc_processors(core_type);
    for (int risc_id = 0; risc_id < num_processors; risc_id++) {
        if (!RiscEnabled(core_type, risc_id)) {
            continue;
        }
        uint64_t base_addr = GetDprintBufAddr(device_id, virtual_core, risc_id);
        if (!CheckInitMagicCleared(device_id, virtual_core, base_addr)) {
            continue;
        }
        constexpr int eightbytes = 8;
        auto from_dev =
            MetalContext::instance().get_cluster().read_core(device_id, virtual_core, base_addr, eightbytes);
        uint32_t wpos = from_dev[0], rpos = from_dev[1];
        if (rpos < wpos) {
            return true;
        }
    }
    return false;
}

// DEVICE_PRINT implementations — single shared buffer per core.

struct FormatPlaceholderInfo {
    uint32_t arg_id;
    char type_id;
    std::string_view format_spec;  // The part after ':' in the format string, if it exists, including ':' itself.
};

// TODO: Add more types here
using ArgumentValue =
    std::variant<bool, int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t, float, double>;

std::optional<FormatPlaceholderInfo> parse_placeholder(std::string_view format_str, std::size_t& pos) {
    if (pos >= format_str.size() || format_str[pos] != '{') {
        return std::nullopt;
    }

    // Start of a placeholder. Read until the closing '}' to extract the placeholder content.
    pos++;  // Skip '{'

    // We are trying to mimic fmtlib format specifiers here, but device already changed it a bit:
    // replacement_field ::= "{" arg_id "," type_id [":" (format_spec | chrono_format_spec)] "}"
    // type_id           ::= "a"..."z" | "A"..."Z"
    // arg_id            ::= integer
    // integer           ::= digit+
    // digit             ::= "0"..."9"
    // But we don't support using identifiers to reduce kernel size, only integers for arg_id.

    // Regarding format_spec:
    // format_spec ::= [[fill]align][sign]["#"]["0"][width]["." precision]["L"][type]
    // fill        ::= <a character other than '{' or '}'>
    // align       ::= "<" | ">" | "^"
    // sign        ::= "+" | "-" | " "
    // width       ::= integer | "{" [arg_id] "}"
    // precision   ::= integer | "{" [arg_id] "}"
    // type        ::= "a" | "A" | "b" | "B" | "c" | "d" | "e" | "E" | "f" | "F" |
    //                 "g" | "G" | "o" | "p" | "s" | "x" | "X" | "?"
    // We don't support using arg_id for width/precision.

    // As everything is verified during kernel compile time, we can parse format_spec just by reading until the closing
    // '}' without needing to fully understand it on the host side.
    uint32_t arg_id = 0;

    // arg_id parsing
    if (!std::isdigit(format_str[pos])) {
        return std::nullopt;
    }
    while (pos < format_str.size() && std::isdigit(format_str[pos])) {
        arg_id = arg_id * 10 + (format_str[pos] - '0');
        pos++;
    }

    // Read type_id (the character after arg_id and ',')
    if (pos >= format_str.size() || format_str[pos] != ',') {
        return std::nullopt;
    }
    pos++;  // Skip ','
    char type_id = format_str[pos++];

    uint32_t format_spec_start = pos;
    while (pos < format_str.size() && format_str[pos] != '}') {
        pos++;
    }
    pos++;  // Skip '}'
    return {{arg_id, type_id, format_str.substr(format_spec_start, pos - format_spec_start - 1)}};
}

std::vector<FormatPlaceholderInfo> parse_format_string(std::string_view format_str) {
    std::vector<FormatPlaceholderInfo> placeholders;
    for (size_t i = 0; i < format_str.size(); i++) {
        if (format_str[i] == '{' && i + 1 < format_str.size() && format_str[i + 1] == '{') {
            // Escaped '{', add a single '{' to the result and skip the next character.
            i++;
            continue;
        }
        if (format_str[i] == '}' && i + 1 < format_str.size() && format_str[i + 1] == '}') {
            // Escaped '}', add a single '}' to the result and skip the next character.
            i++;
            continue;
        }
        if (format_str[i] == '{') {
            auto placeholder = parse_placeholder(format_str, i);
            if (!placeholder) {
                TT_THROW("Invalid format string: failed to parse placeholder at position {}", i);
            }
            placeholders.push_back(*placeholder);
            i--;  // Step back so that the main loop can correctly identify the end of the placeholder
        } else {
            // Regular character, add it to the result.
            continue;
        }
    }
    return placeholders;
}

std::vector<ArgumentValue> read_arguments_from_payload(
    std::string_view format_str, std::span<const std::byte> payload_bytes) {
    auto placeholders = parse_format_string(format_str);
    uint32_t max_arg_id = 0;
    for (const auto& placeholder : placeholders) {
        max_arg_id = std::max(max_arg_id, placeholder.arg_id);
    }
    std::vector<char> argument_types(max_arg_id + 1);
    for (const auto& placeholder : placeholders) {
        argument_types[placeholder.arg_id] = placeholder.type_id;
    }
    std::vector<ArgumentValue> arguments;
    std::size_t payload_offset = 0;

    arguments.reserve(argument_types.size());
    for (char argument_type : argument_types) {
        switch (argument_type) {
            case 'b':  // int8_t
                arguments.push_back(*reinterpret_cast<const int8_t*>(payload_bytes.data() + payload_offset));
                payload_offset += sizeof(int8_t);
                break;
            case 'B':  // uint8_t
                arguments.push_back(*reinterpret_cast<const uint8_t*>(payload_bytes.data() + payload_offset));
                payload_offset += sizeof(uint8_t);
                break;
            case 'h':  // int16_t
                arguments.push_back(*reinterpret_cast<const int16_t*>(payload_bytes.data() + payload_offset));
                payload_offset += sizeof(int16_t);
                break;
            case 'H':  // uint16_t
                arguments.push_back(*reinterpret_cast<const uint16_t*>(payload_bytes.data() + payload_offset));
                payload_offset += sizeof(uint16_t);
                break;
            case 'i':  // int32_t
                arguments.push_back(*reinterpret_cast<const int32_t*>(payload_bytes.data() + payload_offset));
                payload_offset += sizeof(int32_t);
                break;
            case 'I':  // uint32_t
                arguments.push_back(*reinterpret_cast<const uint32_t*>(payload_bytes.data() + payload_offset));
                payload_offset += sizeof(uint32_t);
                break;
            case 'q':  // int64_t
                arguments.push_back(*reinterpret_cast<const int64_t*>(payload_bytes.data() + payload_offset));
                payload_offset += sizeof(int64_t);
                break;
            case 'Q':  // uint64_t
                arguments.push_back(*reinterpret_cast<const uint64_t*>(payload_bytes.data() + payload_offset));
                payload_offset += sizeof(uint64_t);
                break;
            case 'f':  // float
                arguments.push_back(*reinterpret_cast<const float*>(payload_bytes.data() + payload_offset));
                payload_offset += sizeof(float);
                break;
            case 'd':  // double
                arguments.push_back(*reinterpret_cast<const double*>(payload_bytes.data() + payload_offset));
                payload_offset += sizeof(double);
                break;
            case '?':  // bool
                arguments.push_back(*reinterpret_cast<const uint8_t*>(payload_bytes.data() + payload_offset) != 0);
                payload_offset += sizeof(uint8_t);
                break;
            // Add more cases here for other supported types.
            default: TT_THROW("Unsupported type_id in format placeholder: {}", argument_type);
        }
    }

    return arguments;
}

std::string DevicePrintImpl::format_message(std::string_view format_str, std::span<const std::byte> payload_bytes) {
    // Iterate over format_str and replace {} with format of payload values.
    std::stringstream result;
    auto argument_values = read_arguments_from_payload(format_str, payload_bytes);

    for (size_t i = 0; i < format_str.size(); i++) {
        if (format_str[i] == '{' && i + 1 < format_str.size() && format_str[i + 1] == '{') {
            // Escaped '{', add a single '{' to the result and skip the next character.
            result << '{';
            i++;
        } else if (format_str[i] == '}' && i + 1 < format_str.size() && format_str[i + 1] == '}') {
            // Escaped '}', add a single '}' to the result and skip the next character.
            result << '}';
            i++;
        } else if (format_str[i] == '{') {
            auto placeholder = parse_placeholder(format_str, i);
            if (!placeholder) {
                return {};
            }
            i--;  // Step back so that the main loop can correctly identify the end of the placeholder

            // Do the actual formatting of the argument
            auto format = fmt::runtime("{0" + std::string(placeholder->format_spec) + "}");

            switch (placeholder->type_id) {
                case 'b':  // int8_t
                    result << fmt::format(format, std::get<int8_t>(argument_values[placeholder->arg_id]));
                    break;
                case 'B':  // uint8_t
                    result << fmt::format(format, std::get<uint8_t>(argument_values[placeholder->arg_id]));
                    break;
                case 'h':  // int16_t
                    result << fmt::format(format, std::get<int16_t>(argument_values[placeholder->arg_id]));
                    break;
                case 'H':  // uint16_t
                    result << fmt::format(format, std::get<uint16_t>(argument_values[placeholder->arg_id]));
                    break;
                case 'i':  // int32_t
                    result << fmt::format(format, std::get<int32_t>(argument_values[placeholder->arg_id]));
                    break;
                case 'I':  // uint32_t
                    result << fmt::format(format, std::get<uint32_t>(argument_values[placeholder->arg_id]));
                    break;
                case 'q':  // int64_t
                    result << fmt::format(format, std::get<int64_t>(argument_values[placeholder->arg_id]));
                    break;
                case 'Q':  // uint64_t
                    result << fmt::format(format, std::get<uint64_t>(argument_values[placeholder->arg_id]));
                    break;
                case 'f':  // float
                    result << fmt::format(format, std::get<float>(argument_values[placeholder->arg_id]));
                    break;
                case 'd':  // double
                    result << fmt::format(format, std::get<double>(argument_values[placeholder->arg_id]));
                    break;
                case '?':  // bool
                    result << fmt::format(format, std::get<bool>(argument_values[placeholder->arg_id]));
                    break;
                default: TT_THROW("Unsupported type_id in format placeholder: {}", placeholder->type_id);
            }
        } else {
            // Regular character, add it to the result.
            result << format_str[i];
        }
    }
    return result.str();
}

void DevicePrintImpl::print_buffer_data(
    ChipId device_id, const umd::CoreDescriptor& logical_core, const std::vector<uint32_t>& data) {
    std::size_t word_index = 0;

    while (word_index < data.size()) {
        // New data always starts with a DevicePrintHeader, so lets parse it.
        const DevicePrintHeader* header = reinterpret_cast<const DevicePrintHeader*>(data.data() + word_index);
        word_index++;

        // Check if we are loading new kernel
        if (header->is_kernel && header->message_payload == DevicePrintHeader::max_message_payload_size) {
            RiscKey risc_key = {device_id, logical_core, header->risc_id};
            RiscData& risc_data = risc_data_[risc_key];
            if (risc_data.last_loaded_kernel_id == static_cast<int>(header->info_id)) {
                // We have already loaded this kernel, no need to do load it again.
                continue;
            }

            if (risc_data.last_loaded_kernel_id != -1 && risc_data.kernel_elf_cache_entry != nullptr) {
                // Decrease reference count of previously loaded kernel and remove it from cache if reference count
                // reaches 0.
                if (risc_data.kernel_elf_cache_entry->ref_count <= 1) {
                    elf_cache_.erase(risc_data.kernel_elf_path);
                } else {
                    risc_data.kernel_elf_cache_entry->ref_count--;
                }
                risc_data.kernel_elf_cache_entry = nullptr;
            }

            // Update kernel id
            auto kernel_id = static_cast<int>(header->info_id);
            risc_data.last_loaded_kernel_id = kernel_id;

            // Find elf path from inspector using kernel id
            auto kernel_path = Inspector::get_kernel_path_from_watcher_kernel_id(header->info_id);
            auto risc_name = GetRiscName(device_id, logical_core, header->risc_id);
            std::transform(
                risc_name.begin(), risc_name.end(), risc_name.begin(), [](auto c) { return std::tolower(c); });
            auto elf_path = std::filesystem::path(kernel_path) / risc_name / (risc_name + ".elf");
            risc_data.kernel_elf_path = elf_path.string();

            auto elf_cache_it = elf_cache_.find(risc_data.kernel_elf_path);
            if (elf_cache_it != elf_cache_.end()) {
                // Kernel already in cache, just increase reference count and continue.
                elf_cache_it->second.ref_count++;
                continue;
            }

            // Load elf file
            auto& elf_cache_entry = elf_cache_[risc_data.kernel_elf_path];
            elf_cache_entry.ref_count = 1;
            risc_data.kernel_elf_cache_entry = &elf_cache_entry;
            elf_cache_entry.load_elf(elf_path);
        } else if (
            header->is_kernel == 0 && header->risc_id == 0 && header->message_payload == 0 &&
            header->info_id == DevicePrintHeader::max_info_id_value) {
            // This is a wrap around message, we should just ignore it and mark that we processed buffer.
            break;
        } else {
            // This is a normal print message, we should parse it and print it out.
            RiscKey risc_key = {device_id, logical_core, header->risc_id};
            RiscData& risc_data = risc_data_[risc_key];

            // Find elf file
            ElfFileCacheEntry* elf_entry_ptr = nullptr;

            if (header->is_kernel) {
                elf_entry_ptr = risc_data.kernel_elf_cache_entry;
            } else {
                // TODO: Find firmware elf. If it is still not loaded, load it into cache.
            }

            // Check if we found elf file for this print message.
            if (elf_entry_ptr != nullptr) {
                if (elf_entry_ptr->string_info_ptr != nullptr && elf_entry_ptr->string_info_size > header->info_id) {
                    const DevicePrintStringInfo& info = elf_entry_ptr->string_info_ptr[header->info_id];
                    if (info.format_string_ptr >= elf_entry_ptr->format_strings_address &&
                        info.format_string_ptr <
                            elf_entry_ptr->format_strings_address + elf_entry_ptr->format_strings_bytes.size()) {
                        const char* format_string = reinterpret_cast<const char*>(
                            elf_entry_ptr->format_strings_bytes.data() +
                            (info.format_string_ptr - elf_entry_ptr->format_strings_address));
                        std::string_view format_str(format_string);
                        std::string_view file_str;
                        if (info.file >= elf_entry_ptr->format_strings_address &&
                            info.file <
                                elf_entry_ptr->format_strings_address + elf_entry_ptr->format_strings_bytes.size()) {
                            const char* file_string = reinterpret_cast<const char*>(
                                elf_entry_ptr->format_strings_bytes.data() +
                                (info.file - elf_entry_ptr->format_strings_address));
                            file_str = std::string_view(file_string);
                        }

                        // Format message
                        std::span<const std::byte> payload_bytes(
                            reinterpret_cast<const std::byte*>(data.data() + word_index), header->message_payload);
                        auto formatted_message = format_message(format_str, payload_bytes);

                        // Find if we have something buffered from before
                        if (!risc_data.message_buffer.empty()) {
                            // We have something in the buffer, prepend it to the current message and clear the buffer.
                            formatted_message = risc_data.message_buffer + formatted_message;
                            risc_data.message_buffer.clear();
                        }

                        // Check if we hit new line
                        auto newline_pos = formatted_message.find('\n');

                        if (newline_pos != std::string::npos) {
                            // We will do message printing. Check if we have generated line prefix for this risc before,
                            // if not generate one.
                            std::string line_prefix;
                            if (!risc_data.line_prefix.has_value()) {
                                // Compute line prefix based on RTOptions
                                const bool prepend_device_core_risc =
                                    tt::tt_metal::MetalContext::instance()
                                        .rtoptions()
                                        .get_feature_prepend_device_core_risc(tt::llrt::RunTimeDebugFeatureDprint);
                                if (prepend_device_core_risc) {
                                    const string& device_id_str = to_string(device_id);
                                    const string& core_coord_str = logical_core.coord.str();
                                    const string& risc_name =
                                        GetRiscName(device_id, logical_core, header->risc_id, true);
                                    line_prefix = fmt::format("{}:{}:{}: ", device_id_str, core_coord_str, risc_name);
                                }
                                risc_data.line_prefix = line_prefix;
                            } else {
                                line_prefix = risc_data.line_prefix.value();
                            }

                            // Are we printing the whole string, or we need to split it into multiple lines because of
                            // multiple new lines in the message or because we want to prepend line prefix to each line?
                            ostream* output_stream = get_output_stream(risc_key);
                            if (newline_pos == formatted_message.size() - 1) {
                                *output_stream << line_prefix << formatted_message << flush;
                            } else {
                                std::size_t newline_start = 0;
                                std::string_view full_message_view = formatted_message;
                                while (newline_pos != std::string::npos) {
                                    std::string_view line =
                                        full_message_view.substr(newline_start, newline_pos - newline_start);
                                    *output_stream << line_prefix << line << std::endl;
                                    newline_start = newline_pos + 1;
                                    newline_pos = full_message_view.find('\n', newline_start);
                                }
                                if (newline_start < full_message_view.size()) {
                                    risc_data.message_buffer = formatted_message.substr(newline_start);
                                }
                            }
                        } else {
                            // We don't have a complete line yet, buffer the message for next time.
                            risc_data.message_buffer = formatted_message;
                        }
                    }
                }
            }

            // Move to the next message
            word_index += (header->message_payload + 3) / 4;  // round up to nearest word
        }
    }
}

bool DevicePrintImpl::poll_one_core(
    ChipId device_id, const umd::CoreDescriptor& logical_core, bool /*new_data_this_iter*/) {
    auto virtual_core = MetalContext::instance().get_cluster().get_virtual_coordinate_from_logical_coordinates(
        device_id, logical_core.coord, logical_core.type);
    auto programmable_core_type = llrt::get_core_type(device_id, virtual_core);
    uint32_t num_processors = MetalContext::instance().hal().get_num_risc_processors(programmable_core_type);
    bool new_data = false;

    // Memory layout:
    // uint32_t wpos;
    // uint32_t rpos;
    // uint8_t risc_state[num_processors]; // Rounded up to nearest word

    auto buffer_address = GetDevicePrintBufAddr(device_id, virtual_core);
    uint32_t buffer_size = DPRINT_BUFFER_SIZE * num_processors;
    constexpr uint32_t eightbytes = 8;
    uint32_t risc_state_bytes = ((num_processors + 3) / 4) * 4;  // Round up to nearest word
    auto from_dev =
        MetalContext::instance().get_cluster().read_core(device_id, virtual_core, buffer_address, eightbytes);
    uint32_t wpos = from_dev[0], rpos = from_dev[1];
    uint32_t print_buffer_address = buffer_address + eightbytes + risc_state_bytes;  // Skip wpos, rpos, and risc state
    uint32_t print_buffer_size = buffer_size - eightbytes - risc_state_bytes;

    if (wpos == DEBUG_PRINT_SERVER_DISABLED_MAGIC || wpos == DEBUG_PRINT_SERVER_STARTING_MAGIC) {
        return false;
    }

    if (wpos != rpos) {
        new_data = true;
        if (rpos > wpos) {
            // Read until end of buffer and then from beginning until wpos
            auto data = MetalContext::instance().get_cluster().read_core(
                device_id, virtual_core, print_buffer_address + rpos, print_buffer_size - rpos);

            // Process buffer data
            print_buffer_data(device_id, logical_core, data);

            // Update rpos, so that device knows it can use rest of the buffer
            rpos = 0;
            MetalContext::instance().get_cluster().write_core(
                device_id, virtual_core, std::vector<uint32_t>{rpos}, buffer_address + 4);
        }
        if (rpos < wpos) {
            // Read until wpos
            auto data = MetalContext::instance().get_cluster().read_core(
                device_id, virtual_core, print_buffer_address + rpos, wpos - rpos);

            // Process buffer data
            print_buffer_data(device_id, logical_core, data);

            // Update rpos, so that device knows it can use rest of the buffer
            rpos = wpos;
            MetalContext::instance().get_cluster().write_core(
                device_id, virtual_core, std::vector<uint32_t>{rpos}, buffer_address + 4);
        }
    }
    return new_data;
}

void DevicePrintImpl::init_print_buffers_for_core(
    ChipId device_id, const CoreCoord& virtual_core, HalProgrammableCoreType core_type) {
    uint32_t num_processors = MetalContext::instance().hal().get_num_risc_processors(core_type);
    uint32_t buffer_size = DPRINT_BUFFER_SIZE * num_processors;
    WriteInitMagic(device_id, virtual_core, GetDevicePrintBufAddr(device_id, virtual_core), true, buffer_size);
}

void DevicePrintImpl::enable_print_buffers_for_core(
    ChipId device_id, const CoreCoord& virtual_core, HalProgrammableCoreType core_type) {
    uint32_t num_processors = MetalContext::instance().hal().get_num_risc_processors(core_type);
    uint64_t device_print_buffer_address = GetDevicePrintBufAddr(device_id, virtual_core);
    uint64_t risc_flags_address = device_print_buffer_address + 8;  // sizeof(wpos) + sizeof(rpos)
    std::vector<uint8_t> risc_flags((num_processors + 3) / 4 * 4, 0);
    for (int risc_index = 0; risc_index < num_processors; risc_index++) {
        if (!RiscEnabled(core_type, risc_index)) {
            risc_flags[risc_index] = 2;  // TODO: use enum value?!?
        }
    }

    // We created array of flags for each risc. We will write it to the device at an offset from the print buffer
    // address. It is OK to do this write any time to update flags. Flags carry info about if kernel already sent kernel
    // loaded structure and if printing is enabled. If we overwrite flags while kernel is running, all we can do is make
    // kernel print more data (repeat kernel loaded structure). We will handle this on server side.
    tt::tt_metal::MetalContext::instance().get_cluster().write_core(
        device_id, virtual_core, risc_flags, risc_flags_address);
}

bool DevicePrintImpl::core_has_outstanding_prints(
    ChipId device_id, const CoreCoord& virtual_core, HalProgrammableCoreType core_type) {
    // TODO: What does this function represent?!? Do we need this function in new implementation?
    uint32_t num_processors = MetalContext::instance().hal().get_num_risc_processors(core_type);
    bool any_risc_enabled = false;
    for (int risc_id = 0; risc_id < num_processors; risc_id++) {
        if (RiscEnabled(core_type, risc_id)) {
            any_risc_enabled = true;
            break;
        }
    }
    if (!any_risc_enabled) {
        return false;
    }
    uint64_t base_addr = GetDevicePrintBufAddr(device_id, virtual_core);
    if (CheckInitMagicCleared(device_id, virtual_core, base_addr)) {
        constexpr int eightbytes = 8;
        auto from_dev =
            MetalContext::instance().get_cluster().read_core(device_id, virtual_core, base_addr, eightbytes);
        uint32_t wpos = from_dev[0], rpos = from_dev[1];
        if (rpos < wpos) {
            return true;
        }
    }
    return false;
}

DPrintServer::Impl::Impl(llrt::RunTimeOptions& rtoptions) {
    // Read risc mask + log file from rtoptions
    string file_name = rtoptions.get_feature_file_name(tt::llrt::RunTimeDebugFeatureDprint);
    bool one_file_per_risc = rtoptions.get_feature_one_file_per_risc(tt::llrt::RunTimeDebugFeatureDprint);
    bool prepend_device_core_risc = rtoptions.get_feature_prepend_device_core_risc(tt::llrt::RunTimeDebugFeatureDprint);

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
        rtoptions.set_feature_prepend_device_core_risc(tt::llrt::RunTimeDebugFeatureDprint, false);
    }

    // Set the output stream according to RTOptions, either a file name or stdout if none specified.
    std::filesystem::path output_dir(rtoptions.get_logs_dir() + logfile_path);
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
    if (future.wait_for(std::chrono::seconds(2)) == std::future_status::timeout) {
        log_fatal(tt::LogMetal, "Timed out waiting on debug print thread to terminate.");
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
    if (future.wait_for(std::chrono::seconds(5)) == std::future_status::timeout) {
        TT_THROW("Timed out waiting on debug print server to read data.");
    }
}  // await

void DPrintServer::Impl::init_device(ChipId device_id) {
    tt::tt_metal::CoreDescriptorSet all_cores = tt::tt_metal::GetAllCores(device_id);
    // Initialize all print buffers on all cores on the device to have print disabled magic. We
    // will then write print enabled magic for only the cores the user has specified to monitor.
    // This way in the kernel code (dprint.h) we can detect whether the magic value is present and
    // skip prints entirely to prevent kernel code from hanging waiting for the print buffer to be
    // flushed from the host.
    for (const auto& logical_core : all_cores) {
        CoreCoord virtual_core =
            tt::tt_metal::MetalContext::instance().get_cluster().get_virtual_coordinate_from_logical_coordinates(
                device_id, logical_core.coord, logical_core.type);
        init_print_buffers_for_core(device_id, virtual_core, llrt::get_core_type(device_id, virtual_core));
    }
}

void DPrintServer::Impl::attach_devices() {
    auto all_devices = MetalContext::instance().get_cluster().all_chip_ids();

    // Always init all chips, to disable prints by default.
    for (ChipId device_id : all_devices) {
        init_device(device_id);
    }

    // If RTOptions enables all chips, then attach all chips. Otherwise only attach specified devices.
    if (MetalContext::instance().rtoptions().get_feature_all_chips(tt::llrt::RunTimeDebugFeatureDprint)) {
        for (ChipId device_id : all_devices) {
            attach_device(device_id);
        }
    } else {
        for (ChipId device_id :
             MetalContext::instance().rtoptions().get_feature_chip_ids(tt::llrt::RunTimeDebugFeatureDprint)) {
            attach_device(device_id);
        }
    }
}

void DPrintServer::Impl::attach_device(ChipId device_id) {
    // A set of all valid printable cores, used for checking the user input. Note that the coords
    // here are virtual.
    tt::tt_metal::CoreDescriptorSet all_cores = tt::tt_metal::GetAllCores(device_id);
    tt::tt_metal::CoreDescriptorSet dispatch_cores = tt::tt_metal::GetDispatchCores(device_id);

    // If RTOptions doesn't enable DPRINT on this device, return here and don't actually attach it
    // to the server.
    const auto& rtoptions = tt_metal::MetalContext::instance().rtoptions();
    std::vector<ChipId> chip_ids = rtoptions.get_feature_chip_ids(tt::llrt::RunTimeDebugFeatureDprint);
    if (!rtoptions.get_feature_all_chips(tt::llrt::RunTimeDebugFeatureDprint)) {
        if (std::find(chip_ids.begin(), chip_ids.end(), device_id) == chip_ids.end()) {
            return;
        }
    }

    // Core range depends on whether dprint_all_cores flag is set.
    std::vector<umd::CoreDescriptor> print_cores_sanitized;
    for (CoreType core_type : {CoreType::WORKER, CoreType::ETH}) {
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
                    virtual_core =
                        tt::tt_metal::MetalContext::instance()
                            .get_cluster()
                            .get_virtual_coordinate_from_logical_coordinates(device_id, logical_core, core_type);
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
        CoreCoord virtual_core =
            tt::tt_metal::MetalContext::instance().get_cluster().get_virtual_coordinate_from_logical_coordinates(
                device_id, logical_core.coord, logical_core.type);
        auto programmable_core_type = llrt::get_core_type(device_id, virtual_core);
        enable_print_buffers_for_core(device_id, virtual_core, programmable_core_type);
        if (dispatch_cores.contains(logical_core)) {
            device_reads_dispatch_cores_[device_id] = true;
        }
    }

    device_intermediate_streams_force_flush_lock_.lock();
    TT_ASSERT(
        !device_intermediate_streams_force_flush_.contains(device_id),
        "Device {} added to DPRINT server more than once!",
        device_id);
    device_intermediate_streams_force_flush_[device_id] = false;
    device_intermediate_streams_force_flush_lock_.unlock();

    // Save this device + core range to the print server
    device_to_core_range_lock_.lock();
    TT_ASSERT(
        !device_to_core_range_.contains(device_id), "Device {} added to DPRINT server more than once!", device_id);
    device_to_core_range_[device_id] = print_cores_sanitized;
    device_to_core_range_lock_.unlock();
    log_info(tt::LogMetal, "DPRINT Server attached device {}", device_id);
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
    // When we detach a device, we should poll to make sure there's no outstanding prints.
    bool outstanding_prints = true;
    while (outstanding_prints && !server_killed_due_to_hang_) {
        // Polling interval of 1ms
        std::this_thread::sleep_for(std::chrono::milliseconds(1));

        // Check all dprint-enabled cores on this device for outstanding prints.
        outstanding_prints = false;
        for (auto& logical_core : device_to_core_range_.at(device_id)) {
            CoreCoord virtual_core =
                MetalContext::instance().get_cluster().get_virtual_coordinate_from_logical_coordinates(
                    device_id, logical_core.coord, logical_core.type);
            auto programmable_core_type = llrt::get_core_type(device_id, virtual_core);
            if (core_has_outstanding_prints(device_id, virtual_core, programmable_core_type)) {
                outstanding_prints = true;
                break;
            }
        }
    }

    // When we detach a device, we should poll to make sure that any leftover prints in the intermediate stream are
    // transferred to the output stream and flushed to ensure that they are printed out to the user.
    if (!server_killed_due_to_hang_) {
        device_intermediate_streams_force_flush_lock_.lock();
        device_intermediate_streams_force_flush_[device_id] = true;
        device_intermediate_streams_force_flush_lock_.unlock();
        bool intermediate_streams_need_to_be_flushed = true;
        while (intermediate_streams_need_to_be_flushed) {
            // Polling interval of 1ms
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            device_intermediate_streams_force_flush_lock_.lock();
            intermediate_streams_need_to_be_flushed = device_intermediate_streams_force_flush_[device_id];
            device_intermediate_streams_force_flush_lock_.unlock();
        }
    }

    // Remove the device from relevant data structures.
    device_intermediate_streams_force_flush_lock_.lock();
    TT_ASSERT(
        device_to_core_range_.contains(device_id),
        "Device {} not present in DPRINT server but tried removing it!",
        device_id);
    device_intermediate_streams_force_flush_.erase(device_id);
    device_intermediate_streams_force_flush_lock_.unlock();

    device_to_core_range_lock_.lock();
    TT_ASSERT(
        device_to_core_range_.contains(device_id),
        "Device {} not present in DPRINT server but tried removing it!",
        device_id);
    device_to_core_range_.erase(device_id);
    log_info(LogMetal, "DPRINT Server detached device {}", device_id);

    // When detaching a device, disable prints on it.
    CoreDescriptorSet all_cores = GetAllCores(device_id);
    for (const auto& logical_core : all_cores) {
        CoreCoord virtual_core = MetalContext::instance().get_cluster().get_virtual_coordinate_from_logical_coordinates(
            device_id, logical_core.coord, logical_core.type);
        init_print_buffers_for_core(device_id, virtual_core, llrt::get_core_type(device_id, virtual_core));
    }
    device_to_core_range_lock_.unlock();
}  // detach_device

void DPrintServer::Impl::clear_log_file() {
    if (outfile_) {
        // Just close the file and re-open it (without append) to clear it.
        outfile_->close();
        delete outfile_;

        string file_name = tt::tt_metal::MetalContext::instance().rtoptions().get_feature_file_name(
            tt::llrt::RunTimeDebugFeatureDprint);
        outfile_ = new ofstream(file_name);
        stream_ = outfile_ ? outfile_ : &std::cout;
    }
}  // clear_log_file

bool DPrintImpl::poll_one_core(ChipId device_id, const umd::CoreDescriptor& logical_core, bool new_data_this_iter) {
    auto virtual_core = MetalContext::instance().get_cluster().get_virtual_coordinate_from_logical_coordinates(
        device_id, logical_core.coord, logical_core.type);
    auto programmable_core_type = llrt::get_core_type(device_id, virtual_core);
    uint32_t risc_count = MetalContext::instance().hal().get_num_risc_processors(programmable_core_type);
    bool new_data = false;
    for (int risc_index = 0; risc_index < risc_count; risc_index++) {
        if (RiscEnabled(programmable_core_type, risc_index)) {
            new_data |= peek_one_risc_non_blocking(device_id, logical_core, risc_index, new_data_this_iter || new_data);
        }
    }
    return new_data;
}  // poll_one_core

bool DPrintImpl::peek_one_risc_non_blocking(
    ChipId device_id, const umd::CoreDescriptor& logical_core, int risc_id, bool /*new_data_this_iter*/) {
    // If init magic isn't cleared for this risc, then dprint isn't enabled on it, don't read it.
    CoreCoord virtual_core =
        tt::tt_metal::MetalContext::instance().get_cluster().get_virtual_coordinate_from_logical_coordinates(
            device_id, logical_core.coord, logical_core.type);
    if (!CheckInitMagicCleared(device_id, virtual_core, GetDprintBufAddr(device_id, virtual_core, risc_id))) {
        return false;
    }

    // compute the buffer address for the requested risc
    uint32_t base_addr = tt::tt_metal::GetDprintBufAddr(device_id, virtual_core, risc_id);
    ChipId chip_id = device_id;
    RiscKey risc_key{chip_id, logical_core, risc_id};

    // Get or create parser for this RISC
    if (!risc_to_parser_[risc_key]) {
        // Compute line prefix based on RTOptions
        std::string line_prefix;
        const bool prepend_device_core_risc =
            tt::tt_metal::MetalContext::instance().rtoptions().get_feature_prepend_device_core_risc(
                tt::llrt::RunTimeDebugFeatureDprint);
        if (prepend_device_core_risc) {
            const string& device_id_str = to_string(device_id);
            const string& core_coord_str = logical_core.coord.str();
            const string& risc_name = GetRiscName(device_id, logical_core, risc_id, true);
            line_prefix = fmt::format("{}:{}:{}: ", device_id_str, core_coord_str, risc_name);
        }
        risc_to_parser_[risc_key] = std::make_unique<DPrintParser>(line_prefix);
    }
    DPrintParser* parser = risc_to_parser_[risc_key].get();

    // Device is incrementing wpos
    // Host is reading wpos and incrementing local rpos up to wpos
    // Device is filling the buffer and in the end waits on host to write rpos
    auto from_dev = tt::tt_metal::MetalContext::instance().get_cluster().read_core(
        chip_id, virtual_core, base_addr, DPRINT_BUFFER_SIZE);
    DebugPrintMemLayout* l = reinterpret_cast<DebugPrintMemLayout*>(from_dev.data());
    uint32_t rpos = l->aux.rpos;
    uint32_t wpos = l->aux.wpos;
    if (rpos < wpos) {
        // at this point rpos,wpos can be stale but not reset to 0 by the producer
        // it's ok for the consumer to be behind the latest wpos+rpos from producer
        // since the corresponding data in buffer for stale rpos+wpos will not be overwritten
        // until we update rpos and write it back
        // The producer only updates rpos in case of buffer overflow.
        // Then it waits for rpos to first catch up to wpos (rpos update by the consumer) before proceeding

        // Parse the data using DPrintParser
        uint32_t data_len = wpos - rpos;
        DPrintParser::ParseResult result = parser->parse(l->data + rpos, data_len);

        // Write each completed line to output
        for (const auto& line : result.completed_lines) {
            ostream* output_stream = get_output_stream(risc_key);
            *output_stream << line << flush;
        }

        // Update rpos based on bytes consumed
        rpos += result.bytes_consumed;

        // writes by the producer should've been atomic w.r.t code+size+payload
        // i.e at this point we shouldn't have piecemeal reads on code+size+payload
        // with rpos not aligned to wpos

        // write back to device - update rpos only
        std::vector<uint32_t> rposbuf;
        rposbuf.push_back(rpos);
        uint32_t offs = DebugPrintMemLayout::rpos_offs();
        tt::tt_metal::MetalContext::instance().get_cluster().write_core(
            chip_id, virtual_core, rposbuf, base_addr + offs);

        // Return true to signal that some print data was read
        return true;
    }  // if (rpos < wpos)

    // Return false to signal that no print data was ready this time.
    return false;
}  // peek_one_risc_non_blocking

void DPrintServer::Impl::poll_print_data() {
    // Give the print server thread a reasonable name.
    pthread_setname_np(pthread_self(), "TT_DPRINT_SERVER");

    // Main print loop, go through all chips/cores/riscs on the device and poll for any print data
    // written.
    const auto& rtoptions = tt_metal::MetalContext::instance().rtoptions();
    while (true) {
        if (stop_print_server_ && !new_data_last_iter_) {
            // If the stop signal was received, exit the print server thread after all new data has been processed.
            break;
        }

        // Make a copy of the device->core map, so that it can be modified while polling.
        std::map<ChipId, std::vector<umd::CoreDescriptor>> device_to_core_range_copy;
        device_to_core_range_lock_.lock();
        device_to_core_range_copy = device_to_core_range_;

        // Flag for whether any new print data was found in this round of polling.
        bool new_data_this_iter = false;
        for (auto& device_and_cores : device_to_core_range_copy) {
            ChipId device_id = device_and_cores.first;
            device_intermediate_streams_force_flush_lock_.lock();
            if (device_intermediate_streams_force_flush_[device_id]) {
                transfer_all_streams_to_output(device_id);
                device_intermediate_streams_force_flush_[device_id] = false;
            }
            device_intermediate_streams_force_flush_lock_.unlock();
            for (auto& logical_core : device_and_cores.second) {
                try {
                    new_data_this_iter |= poll_one_core(device_id, logical_core, new_data_this_iter);
                } catch (std::runtime_error& e) {
                    // Depending on if test mode is enabled, catch and stop server, or
                    // re-throw the exception.
                    if (rtoptions.get_test_mode_enabled()) {
                        server_killed_due_to_hang_ = true;
                        device_to_core_range_lock_.unlock();
                        return;  // Stop the print loop
                    }  // Re-throw for instant exit
                    throw e;
                }

                // If this read detected a print hang, stop processing prints.
                if (server_killed_due_to_hang_) {
                    return;
                }
            }
        }

        // Signal whether the print server is currently processing data.
        new_data_last_iter_ = new_data_this_iter;
        device_to_core_range_lock_.unlock();
        // Sleep for a few ms if no data was processed.
        if (!new_data_last_iter_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        wait_loop_iterations_++;
    }
}  // poll_print_data

void DPrintServer::Impl::transfer_all_streams_to_output(ChipId device_id) {
    for (auto& [risc_key, parser] : risc_to_parser_) {
        const ChipId risc_key_device_id = get<0>(risc_key);
        if (device_id == risc_key_device_id) {
            std::string remaining = parser->flush();
            if (!remaining.empty()) {
                ostream* output_stream = get_output_stream(risc_key);
                *output_stream << remaining << flush;
            }
        }
    }
}  // transfer_all_streams_to_output

ostream* DPrintServer::Impl::get_output_stream(const RiscKey& risc_key) {
    ostream* output_stream = stream_;
    const auto& rtoptions = tt_metal::MetalContext::instance().rtoptions();
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
                GetRiscName(chip_id, logical_core, risc_id));
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
DPrintServer::DPrintServer(llrt::RunTimeOptions& rtoptions) {
    if (rtoptions.get_use_device_print()) {
        impl_ = std::make_unique<DevicePrintImpl>(rtoptions);
    } else {
        log_warning(
            tt::LogMetal,
            "DPRINT is deprecated and will be removed in a future release. "
            "Please migrate to DEVICE_PRINT by setting TT_METAL_DEVICE_PRINT=1"
            " and using DEVICE_PRINT() macro when writing kernels.");
        impl_ = std::make_unique<DPrintImpl>(rtoptions);
    }
}
DPrintServer::~DPrintServer() = default;
void DPrintServer::set_mute(bool mute_print_server) { impl_->set_mute(mute_print_server); }
void DPrintServer::await() { impl_->await(); }
void DPrintServer::attach_devices() { impl_->attach_devices(); }
void DPrintServer::detach_devices() { impl_->detach_devices(); }
void DPrintServer::clear_log_file() { impl_->clear_log_file(); }
bool DPrintServer::reads_dispatch_cores(ChipId device_id) { return impl_->reads_dispatch_cores(device_id); }
bool DPrintServer::hang_detected() { return impl_->hang_detected(); }
}  // namespace tt::tt_metal
