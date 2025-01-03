// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <sstream>
#include <string>
#include <thread>
#include <future>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <set>
#include <filesystem>
#include <tuple>
#include "llrt/llrt.hpp"
#include "tt_metal/common/logger.hpp"

#include "dprint_server.hpp"
#include "debug_helpers.hpp"
#include "llrt/rtoptions.hpp"
#include "common/bfloat8.hpp"

#include "hostdevcommon/dprint_common.h"
#include "tt_metal/impl/device/device.hpp"

using std::cout;
using std::endl;
using std::flush;
using std::int32_t;
using std::ofstream;
using std::ostream;
using std::ostringstream;
using std::set;
using std::setw;
using std::string;
using std::to_string;
using std::tuple;
using std::uint32_t;

using tt::tt_metal::Device;
using namespace tt;

#define CAST_U8P(p) reinterpret_cast<uint8_t*>(p)

namespace {

static string logfile_path = "generated/dprint/";

static inline float bfloat16_to_float(uint16_t bfloat_val) {
    uint32_t uint32_data = ((uint32_t)bfloat_val) << 16;
    float f;
    std::memcpy(&f, &uint32_data, sizeof(f));
    return f;
}

static string GetRiscName(CoreType core_type, int hart_id, bool abbreviated = false) {
    if (core_type == CoreType::ETH) {
        switch (hart_id) {
            case DPRINT_RISCV_INDEX_ER:
                return abbreviated ? "ER" : "ERISC";
                // Default case falls through and handled at end.
        }
    } else {
        switch (hart_id) {
            case DPRINT_RISCV_INDEX_NC: return abbreviated ? "NC" : "NCRISC";
            case DPRINT_RISCV_INDEX_TR0: return abbreviated ? "TR0" : "TRISC0";
            case DPRINT_RISCV_INDEX_TR1: return abbreviated ? "TR1" : "TRISC1";
            case DPRINT_RISCV_INDEX_TR2: return abbreviated ? "TR2" : "TRISC2";
            case DPRINT_RISCV_INDEX_BR:
                return abbreviated ? "BR" : "BRISC";
                // Default case falls through and handled at end.
        }
    }
    return fmt::format("UNKNOWN_RISC_ID({})", hart_id);
}

static void AssertSize(uint8_t sz, uint8_t expected_sz) {
    TT_ASSERT(
        sz == expected_sz,
        "DPrint token size ({}) did not match expected ({}), potential data corruption in the DPrint buffer.",
        sz,
        expected_sz);
}

// A null stream for when the print server is muted.
class NullBuffer : public std::streambuf {
public:
    int overflow(int c) { return c; }
};
NullBuffer null_buffer;
std::ostream null_stream(&null_buffer);

using HartKey = std::tuple<chip_id_t, CoreDescriptor, uint32_t>;

struct HartKeyComparator {
    bool operator()(const HartKey& x, const HartKey& y) const {
        const chip_id_t x_device_id = get<0>(x);
        const chip_id_t y_device_id = get<0>(y);
        const uint32_t x_hart_id = get<2>(x);
        const uint32_t y_hart_id = get<2>(y);
        const CoreDescriptor& x_core_desc = get<1>(x);
        const CoreDescriptor& y_core_desc = get<1>(y);

        if (x_device_id != y_device_id) {
            return x_device_id < y_device_id;
        }

        CoreDescriptorComparator core_desc_cmp;
        if (core_desc_cmp(x_core_desc, y_core_desc)) {
            return true;
        }
        if (core_desc_cmp(y_core_desc, x_core_desc)) {
            return false;
        }

        return x_hart_id < y_hart_id;
    }
};

struct DebugPrintServerContext {
    // only one instance is allowed at the moment
    static DebugPrintServerContext* inst;
    static bool ProfilerIsRunning;

    // Constructor/destructor, reads dprint options from RTOptions.
    DebugPrintServerContext();
    ~DebugPrintServerContext();

    // Sets whether the print server is muted. Calling this function while a kernel is running may
    // result in a loss of print data.
    void SetMute(bool mute_print_server) { mute_print_server_ = mute_print_server; }

    // Waits for the prints erver to finish processing any current print data.
    void WaitForPrintsFinished();

    // Attaches a device to be monitored by the print server.
    // This device should not already be attached.
    void AttachDevice(Device* device);

    // Detaches a device from being monitored by the print server.
    // This device must have been attached previously.
    void DetachDevice(Device* device);

    // Clears the log file of a currently-running print server.
    void ClearLogFile();

    // Clears any raised signals (so they can be used again in a later run).
    void ClearSignals();

    bool ReadsDispatchCores(Device* device) { return device_reads_dispatch_cores_[device]; }

    int GetNumAttachedDevices() { return device_to_core_range_.size(); }

    bool PrintHangDetected() { return server_killed_due_to_hang_; }

private:
    // Flag for main thread to signal the print server thread to stop.
    std::atomic<bool> stop_print_server_;
    // Flag for muting the print server. This doesn't disable reading print data from the device,
    // but it supresses the output of that print data the user.
    std::atomic<bool> mute_print_server_;
    // Flag for signalling whether the print server thread has recently processed data (and is
    // therefore likely to continue processing data in the next round of polling).
    std::atomic<bool> new_data_last_iter_;
    std::thread* print_server_thread_;

    // A flag to signal to the main thread if the print server detected a print-based hang.
    bool server_killed_due_to_hang_;

    // A counter to keep track of how many iterations the print server has gone through without
    std::atomic<int> wait_loop_iterations_ = 0;

    // For keeping track of the previous dprint type read for each risc. In some cases, the way that the current dprint
    // type is parsed depends on the previous dprint type.
    std::map<HartKey, DPrintTypeID, HartKeyComparator> risc_to_prev_type_;

    ofstream* outfile_ = nullptr;  // non-cout
    ostream* stream_ = nullptr;    // either == outfile_ or is &cout

    // For buffering up partial dprints from each risc.
    std::map<HartKey, ostringstream*, HartKeyComparator> risc_to_intermediate_stream_;

    // For printing each risc's dprint to a separate file, a map from {device id, core, hart index} to files.
    std::map<HartKey, ofstream*, HartKeyComparator> risc_to_file_stream_;

    // A map from {device id, core, hart index} to the signal code it's waiting for.
    std::map<HartKey, uint32_t, HartKeyComparator> hart_waiting_on_signal_;
    // Keep a separate set of raised signal codes so that multiple harts can wait for the same
    // signal.
    std::set<uint32_t> raised_signals_;
    std::mutex raise_wait_lock_;  // A lock for these two objects since both server and main access.

    // A map from Device -> Core Range, which is used to determine which cores on which devices
    // to scan for print data. Also a lock for editing it.
    std::map<Device*, std::vector<CoreDescriptor>> device_to_core_range_;
    std::map<Device*, bool> device_reads_dispatch_cores_;  // True if given device reads any dispatch cores. Used to
                                                           // know whether dprint can be compiled out.
    std::mutex device_to_core_range_lock_;

    // Used to signal to the print server to flush all intermediate streams for a device so that any remaining prints
    // are printed out.
    std::map<Device*, bool> device_intermediate_streams_force_flush_;
    std::mutex device_intermediate_streams_force_flush_lock_;

    // Polls specified cores/harts on all attached devices and prints any new print data. This
    // function is the main loop for the print server thread.
    void PollPrintData(uint32_t hart_mask);

    // Peeks a specified hart for any debug prints present in the buffer, printing the contents
    // out to host-side stream. Returns true if some data was read out, and false if no new
    // print data was present on the device. Note that if an unanswered WAIT is present, the print
    // buffer on the device is only flushed  up to the WAIT, even if more print data is available
    // after it.
    bool PeekOneHartNonBlocking(
        Device* device, const CoreDescriptor& logical_core, int hart_index, bool new_data_this_iter);

    // Transfers data from each intermediate stream associated with the given device to the output stream and flushes
    // the output stream so that the data is visible to the user.
    void TransferIntermediateStreamsToOutputStreamAndFlush(Device* device);

    // Transfers data from the given intermediate stream to the output stream and flushes the output stream so that the
    // data is visible to the user.
    void TransferToAndFlushOutputStream(const HartKey& hart_key, ostringstream* intermediate_stream);

    // Returns the dprint data that should be outputted by the output stream.
    string GetDataToOutput(const HartKey& hart_key, const ostringstream* stream);

    // Returns the stream that the dprint data should be output to. Can be auto-generated files, the user-selected file,
    // stdout, or nothing.
    ostream* GetOutputStream(const HartKey& hart_key);

    // Stores the last value of setw, so that array elements can reuse the width.
    char most_recent_setw = 0;
};

static void ResetStream(ostringstream* stream) {
    stream->str("");
    stream->clear();
}  // ResetStream

static bool StreamEndsWithNewlineChar(const ostringstream* stream) {
    const string stream_str = stream->str();
    return !stream_str.empty() && stream_str.back() == '\n';
}  // StreamEndsWithNewlineChar

static void PrintTileSlice(ostringstream* stream, uint8_t* ptr) {
    TileSliceHostDev<0> ts_copy;  // Make a copy since ptr might not be properly aligned
    std::memcpy(&ts_copy, ptr, sizeof(TileSliceHostDev<0>));
    TileSliceHostDev<0>* ts = &ts_copy;
    TT_ASSERT(
        offsetof(TileSliceHostDev<0>, data) % sizeof(uint32_t) == 0,
        "TileSliceHostDev<0> data field is not properly aligned");
    uint8_t* data = ptr + offsetof(TileSliceHostDev<0>, data);

    // Read any error codes and handle accordingly
    enum CBIndex cb = static_cast<enum CBIndex>(ts->cb_id);
    switch (ts->return_code) {
        case DPrintOK: break;  // Continue to print the tile slice
        case DPrintErrorBadPointer: {
            uint32_t ptr = ts->cb_ptr;
            uint8_t count = ts->data_count;
            *stream << fmt::format("Tried printing {}: BAD TILE POINTER (ptr={}, count={})\n", cb, ptr, count);
            return;
        }
        case DPrintErrorUnsupportedFormat: {
            tt::DataFormat data_format = static_cast<tt::DataFormat>(ts->data_format);
            *stream << fmt::format("Tried printing {}: Unsupported data format ({})\n", cb, data_format);
            return;
        }
        case DPrintErrorMath:
            *stream << "Warning: MATH core does not support TileSlice printing, omitting print...\n";
            return;
        case DPrintErrorEthernet:
            *stream << "Warning: Ethernet core does not support TileSlice printing, omitting print...\n";
            return;
        default:
            *stream << fmt::format(
                "Warning: TileSlice printing failed with unknown return code {}, omitting print...\n", ts->return_code);
            return;
    }

    // No error codes, print the TileSlice
    uint32_t i = 0;
    bool count_exceeded = false;
    for (int h = ts->slice_range.h0; h < ts->slice_range.h1; h += ts->slice_range.hs) {
        for (int w = ts->slice_range.w0; w < ts->slice_range.w1; w += ts->slice_range.ws) {
            // If the number of data specified by the SliceRange exceeds the number that was
            // saved in the print buffer (set by the MAX_COUNT template parameter in the
            // TileSlice), then break early.
            if (i >= ts->data_count) {
                count_exceeded = true;
                break;
            }
            tt::DataFormat data_format = static_cast<tt::DataFormat>(ts->data_format);
            switch (data_format) {
                case tt::DataFormat::Float16_b: {
                    uint16_t* float16_b_ptr = reinterpret_cast<uint16_t*>(data);
                    *stream << bfloat16_to_float(float16_b_ptr[i]);
                    break;
                }
                case tt::DataFormat::Float32: {
                    float* float32_ptr = reinterpret_cast<float*>(data);
                    *stream << float32_ptr[i];
                    break;
                }
                case tt::DataFormat::Bfp4_b:
                case tt::DataFormat::Bfp8_b: {
                    // Saved the exponent and data together
                    uint16_t* data_ptr = reinterpret_cast<uint16_t*>(data);
                    uint8_t val = (data_ptr[i] >> 8) & 0xFF;
                    uint8_t exponent = data_ptr[i] & 0xFF;
                    uint32_t bit_val = convert_bfp_to_u32(data_format, val, exponent, false);
                    *stream << *reinterpret_cast<float*>(&bit_val);
                    break;
                }
                case tt::DataFormat::Int8: {
                    int8_t* data_ptr = reinterpret_cast<int8_t*>(data);
                    *stream << (int)data_ptr[i];
                    break;
                }
                case tt::DataFormat::UInt8: {
                    uint8_t* data_ptr = reinterpret_cast<uint8_t*>(data);
                    *stream << (unsigned int)data_ptr[i];
                    break;
                }
                case tt::DataFormat::UInt16: {
                    uint16_t* data_ptr = reinterpret_cast<uint16_t*>(data);
                    *stream << (unsigned int)data_ptr[i];
                    break;
                }
                case tt::DataFormat::Int32: {
                    int32_t* data_ptr = reinterpret_cast<int32_t*>(data);
                    *stream << (int)data_ptr[i];
                    break;
                }
                case tt::DataFormat::UInt32: {
                    uint32_t* data_ptr = reinterpret_cast<uint32_t*>(data);
                    *stream << (unsigned int)data_ptr[i];
                    break;
                }
                default: break;
            }
            if (w + ts->slice_range.ws < ts->slice_range.w1) {
                *stream << " ";
            }
            i++;
        }

        // Break outer loop as well if MAX COUNT exceeded, also print a message to let the user
        // know that the slice has been truncated.
        if (count_exceeded) {
            *stream << "<TileSlice data truncated due to exceeding max count (" << to_string(ts->data_count) << ")>\n";
            break;
        }

        if (ts->endl_rows) {
            *stream << "\n";
        }
    }
}  // PrintTileSlice

// Create a float from a given bit pattern, given the number of bits for the exponent and mantissa.
// Assumes the following order of bits in the input data:
//   [sign bit][mantissa bits][exponent bits]
static float make_float(uint8_t exp_bit_count, uint8_t mantissa_bit_count, uint32_t data) {
    int sign = (data >> (exp_bit_count + mantissa_bit_count)) & 0x1;
    const int exp_mask = (1 << (exp_bit_count)) - 1;
    int exp_bias = (1 << (exp_bit_count - 1)) - 1;
    int exp_val = (data & exp_mask) - exp_bias;
    const int mantissa_mask = ((1 << mantissa_bit_count) - 1) << exp_bit_count;
    int mantissa_val = (data & mantissa_mask) >> exp_bit_count;
    float result = 1.0 + ((float)mantissa_val / (float)(1 << mantissa_bit_count));
    result = result * pow(2, exp_val);
    if (sign) {
        result = -result;
    }
    return result;
}

// Prints a given datum in the array, given the data_format
static void PrintTensixRegisterData(ostringstream* stream, int setwidth, uint32_t datum, uint16_t data_format) {
    switch (data_format) {
        case static_cast<std::uint8_t>(tt::DataFormat::Float16):
        case static_cast<std::uint8_t>(tt::DataFormat::Bfp8):
        case static_cast<std::uint8_t>(tt::DataFormat::Bfp4):
        case static_cast<std::uint8_t>(tt::DataFormat::Bfp2):
        case static_cast<std::uint8_t>(tt::DataFormat::Lf8):
            *stream << setw(setwidth) << make_float(5, 10, datum & 0xffff) << " ";
            *stream << setw(setwidth) << make_float(5, 10, (datum >> 16) & 0xffff) << " ";
            break;
        case static_cast<std::uint8_t>(tt::DataFormat::Bfp8_b):
        case static_cast<std::uint8_t>(tt::DataFormat::Bfp4_b):
        case static_cast<std::uint8_t>(tt::DataFormat::Bfp2_b):
        case static_cast<std::uint8_t>(tt::DataFormat::Float16_b):
            *stream << setw(setwidth) << make_float(8, 7, datum & 0xffff) << " ";
            *stream << setw(setwidth) << make_float(8, 7, (datum >> 16) & 0xffff) << " ";
            break;
        case static_cast<std::uint8_t>(tt::DataFormat::Tf32):
            *stream << setw(setwidth) << make_float(8, 10, datum) << " ";
            break;
        case static_cast<std::uint8_t>(tt::DataFormat::Float32): {
            float value;
            memcpy(&value, &datum, sizeof(float));
            *stream << setw(setwidth) << value << " ";
        } break;
        case static_cast<std::uint8_t>(tt::DataFormat::UInt32): *stream << setw(setwidth) << datum << " "; break;
        case static_cast<std::uint8_t>(tt::DataFormat::UInt16):
            *stream << setw(setwidth) << (datum & 0xffff) << " ";
            *stream << setw(setwidth) << (datum >> 16) << " ";
            break;
        default: *stream << "Unknown data format " << data_format << " "; break;
    }
}

// Prints a typed uint32 array given the number of elements including the type.
// If force_element_type is set to a valid type, it is assumed that the type is not included in the
// data array, and the type is forced to be the given type.
static void PrintTypedUint32Array(
    ostringstream* stream,
    int setwidth,
    uint32_t raw_element_count,
    uint32_t* data,
    TypedU32_ARRAY_Format force_array_type = TypedU32_ARRAY_Format_INVALID) {
    uint16_t array_type = data[raw_element_count - 1] >> 16;
    uint16_t array_subtype = data[raw_element_count - 1] & 0xffff;

    raw_element_count = (force_array_type == TypedU32_ARRAY_Format_INVALID) ? raw_element_count : raw_element_count + 1;

    for (uint32_t i = 0; i < raw_element_count - 1; i++) {
        switch (array_type) {
            case TypedU32_ARRAY_Format_Raw: *stream << std::hex << "0x" << data[i] << " "; break;
            case TypedU32_ARRAY_Format_Tensix_Config_Register_Data_Format_Type:
                PrintTensixRegisterData(stream, setwidth, data[i], array_subtype);
                break;
            default: *stream << "Unknown type " << array_type; break;
        }
    }
}

// Writes a magic value at wpos ptr address for dprint buffer for a specific hart/core/chip
// Used for debug print server startup sequence.
void WriteInitMagic(Device* device, const CoreCoord& phys_core, int hart_id, bool enabled) {
    // compute the buffer address for the requested hart
    uint64_t base_addr = GetDprintBufAddr(device, phys_core, hart_id);

    // TODO(AP): this could use a cleanup - need a different mechanism to know if a kernel is running on device.
    // Force wait for first kernel launch by first writing a non-zero and waiting for a zero.
    std::vector<uint32_t> initbuf = std::vector<uint32_t>(DPRINT_BUFFER_SIZE / sizeof(uint32_t), 0);
    initbuf[0] = uint32_t(enabled ? DEBUG_PRINT_SERVER_STARTING_MAGIC : DEBUG_PRINT_SERVER_DISABLED_MAGIC);
    tt::llrt::write_hex_vec_to_core(device->id(), phys_core, initbuf, base_addr);

    // Prevent race conditions during runtime by waiting until the init value is actually written
    // DPrint is only used for debug purposes so this delay should not be a big issue.
    // 1. host will read remote and think the wpos is 0. so it'll go and poll the data
    // 2. the packet will arrive to set the wpos = DEBUG_PRINT_SERVER_STARTING_MAGIC
    // 3. the actual host polling function will read wpos = DEBUG_PRINT_SERVER_STARTING_MAGIC
    // 4. now we will access wpos at the starting magic which is incorrect
    uint32_t num_tries = 100000;
    while (num_tries-- > 0) {
        auto result = tt::llrt::read_hex_vec_from_core(device->id(), phys_core, base_addr, 4);
        if (result[0] == DEBUG_PRINT_SERVER_STARTING_MAGIC && enabled) {
            return;
        } else if (result[0] == DEBUG_PRINT_SERVER_DISABLED_MAGIC && !enabled) {
            return;
        }
    }
    TT_THROW("Timed out writing init magic");
}  // WriteInitMagic

// Checks if our magic value was cleared by the device code
// The assumption is that if our magic number was cleared,
// it means there is a write in the queue and wpos/rpos are now valid
// Note that this is not a bulletproof way to bootstrap the print server (TODO(AP))
bool CheckInitMagicCleared(Device* device, const CoreCoord& phys_core, int hart_id) {
    // compute the buffer address for the requested hart
    uint32_t base_addr = GetDprintBufAddr(device, phys_core, hart_id);

    auto result = tt::llrt::read_hex_vec_from_core(device->id(), phys_core, base_addr, 4);
    return (result[0] != DEBUG_PRINT_SERVER_STARTING_MAGIC && result[0] != DEBUG_PRINT_SERVER_DISABLED_MAGIC);
}  // CheckInitMagicCleared

DebugPrintServerContext::DebugPrintServerContext() {
    TT_ASSERT(inst == nullptr);
    inst = this;

    // Read hart mask + log file from rtoptions
    uint32_t hart_mask =
        tt::llrt::RunTimeOptions::get_instance().get_feature_riscv_mask(tt::llrt::RunTimeDebugFeatureDprint);
    string file_name =
        tt::llrt::RunTimeOptions::get_instance().get_feature_file_name(tt::llrt::RunTimeDebugFeatureDprint);
    bool one_file_per_risc =
        tt::llrt::RunTimeOptions::get_instance().get_feature_one_file_per_risc(tt::llrt::RunTimeDebugFeatureDprint);
    bool prepend_device_core_risc =
        tt::llrt::RunTimeOptions::get_instance().get_feature_prepend_device_core_risc(tt::llrt::RunTimeDebugFeatureDprint);

    // One file per risc auto-generates the output files and ignores the env var for it. Print a warning if both are
    // specified just in case.
    if (file_name != "" && one_file_per_risc) {
        log_warning(
            "Both TT_METAL_DPRINT_FILE_NAME and TT_METAL_DPRINT_ONE_FILE_PER_RISC are specified. "
            "TT_METAL_DPRINT_FILE_NAME will be ignored.");
    }

    if (prepend_device_core_risc && one_file_per_risc) {
        log_warning(
            "Both TT_METAL_DPRINT_PREPEND_DEVICE_CORE_RISC and TT_METAL_DPRINT_ONE_FILE_PER_RISC are specified. "
            "TT_METAL_DPRINT_PREPEND_DEVICE_CORE_RISC will be disabled.");
        tt::llrt::RunTimeOptions::get_instance().set_feature_prepend_device_core_risc(tt::llrt::RunTimeDebugFeatureDprint, false);
    }

    // Set the output stream according to RTOptions, either a file name or stdout if none specified.
    std::filesystem::path output_dir(tt::llrt::RunTimeOptions::get_instance().get_root_dir() + logfile_path);
    std::filesystem::create_directories(output_dir);
    if (file_name != "" && !one_file_per_risc) {
        outfile_ = new ofstream(file_name);
    }
    stream_ = outfile_ ? outfile_ : &cout;

    stop_print_server_ = false;
    mute_print_server_ = false;
    new_data_last_iter_ = false;
    server_killed_due_to_hang_ = false;

    // Spin off the thread that runs the print server.
    print_server_thread_ = new std::thread([this, hart_mask] { PollPrintData(hart_mask); });
}  // DebugPrintServerContext

DebugPrintServerContext::~DebugPrintServerContext() {
    // Signal the print server thread to finish
    stop_print_server_ = true;

    // Wait for the thread to end, with a timeout
    auto future = std::async(std::launch::async, &std::thread::join, print_server_thread_);
    if (future.wait_for(std::chrono::seconds(2)) == std::future_status::timeout) {
        TT_THROW("Timed out waiting on debug print thread to terminate.");
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
    for (auto& [key, intermediate_stream] : risc_to_intermediate_stream_) {
        delete intermediate_stream;
    }
    inst = nullptr;
}  // ~DebugPrintServerContext

void DebugPrintServerContext::WaitForPrintsFinished() {
    // Simply poll the flag every few ms to check whether new data is still being processed,
    // or whether any cores are waiting for a signal to be raised.
    // TODO(dma): once we have access to the device is there a way we can poll the device to
    // check whether more print data is coming?
    size_t num_harts_waiting = 0;

    // Make sure to run at least one full iteration inside PollPrintData before returning.
    wait_loop_iterations_ = 0;

    do {
        // No need to await if the server was killed already due to a hang.
        if (server_killed_due_to_hang_) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        raise_wait_lock_.lock();
        num_harts_waiting = hart_waiting_on_signal_.size();
        raise_wait_lock_.unlock();
    } while (num_harts_waiting > 0 || new_data_last_iter_ || wait_loop_iterations_ < 2);
}  // WaitForPrintsFinished

void DebugPrintServerContext::AttachDevice(Device* device) {
    chip_id_t device_id = device->id();

    // A set of all valid printable cores, used for checking the user input. Note that the coords
    // here are physical.
    CoreDescriptorSet all_cores = GetAllCores(device);
    CoreDescriptorSet dispatch_cores = GetDispatchCores(device);

    // Initialize all print buffers on all cores on the device to have print disabled magic. We
    // will then write print enabled magic for only the cores the user has specified to monitor.
    // This way in the kernel code (dprint.h) we can detect whether the magic value is present and
    // skip prints entirely to prevent kernel code from hanging waiting for the print buffer to be
    // flushed from the host.
    for (auto& logical_core : all_cores) {
        CoreCoord phys_core = device->virtual_core_from_logical_core(logical_core.coord, logical_core.type);
        for (int hart_index = 0; hart_index < GetNumRiscs(logical_core); hart_index++) {
            WriteInitMagic(device, phys_core, hart_index, false);
        }
    }

    // If RTOptions doesn't enable DPRINT on this device, return here and don't actually attach it
    // to the server.
    std::vector<chip_id_t> chip_ids =
        tt::llrt::RunTimeOptions::get_instance().get_feature_chip_ids(tt::llrt::RunTimeDebugFeatureDprint);
    if (!tt::llrt::RunTimeOptions::get_instance().get_feature_all_chips(tt::llrt::RunTimeDebugFeatureDprint)) {
        if (std::find(chip_ids.begin(), chip_ids.end(), device->id()) == chip_ids.end()) {
            return;
        }
    }

    // Core range depends on whether dprint_all_cores flag is set.
    std::vector<CoreDescriptor> print_cores_sanitized;
    for (CoreType core_type : {CoreType::WORKER, CoreType::ETH}) {
        if (tt::llrt::RunTimeOptions::get_instance().get_feature_all_cores(
                tt::llrt::RunTimeDebugFeatureDprint, core_type) == tt::llrt::RunTimeDebugClassAll) {
            // Print from all cores of the given type, cores returned here are guaranteed to be valid.
            for (CoreDescriptor logical_core : all_cores) {
                if (logical_core.type == core_type) {
                    print_cores_sanitized.push_back(logical_core);
                }
            }
            log_info(
                tt::LogMetal,
                "DPRINT enabled on device {}, all {} cores.",
                device->id(),
                tt::llrt::get_core_type_name(core_type));
        } else if (
            tt::llrt::RunTimeOptions::get_instance().get_feature_all_cores(
                tt::llrt::RunTimeDebugFeatureDprint, core_type) == tt::llrt::RunTimeDebugClassDispatch) {
            for (CoreDescriptor logical_core : dispatch_cores) {
                if (logical_core.type == core_type) {
                    print_cores_sanitized.push_back(logical_core);
                }
            }
            log_info(
                tt::LogMetal,
                "DPRINT enabled on device {}, {} dispatch cores.",
                device->id(),
                tt::llrt::get_core_type_name(core_type));
        } else if (
            tt::llrt::RunTimeOptions::get_instance().get_feature_all_cores(
                tt::llrt::RunTimeDebugFeatureDprint, core_type) == tt::llrt::RunTimeDebugClassWorker) {
            // For worker cores, take all cores and remove dispatch cores.
            for (CoreDescriptor logical_core : all_cores) {
                if (dispatch_cores.find(logical_core) == dispatch_cores.end()) {
                    if (logical_core.type == core_type) {
                        print_cores_sanitized.push_back(logical_core);
                    }
                }
            }
            log_info(
                tt::LogMetal,
                "DPRINT enabled on device {}, {} worker cores.",
                device->id(),
                tt::llrt::get_core_type_name(core_type));
        } else {
            // No "all cores" option provided, which means print from the cores specified by the user
            std::vector<CoreCoord>& print_cores = tt::llrt::RunTimeOptions::get_instance().get_feature_cores(
                tt::llrt::RunTimeDebugFeatureDprint)[core_type];

            // We should also validate that the cores the user specified are valid worker cores.
            for (auto& logical_core : print_cores) {
                // Need to convert user-specified logical cores to physical cores, this can throw
                // if the user gave bad coords.
                CoreCoord phys_core;
                bool valid_logical_core = true;
                try {
                    phys_core = device->virtual_core_from_logical_core(logical_core, core_type);
                } catch (std::runtime_error& error) {
                    valid_logical_core = false;
                }
                if (valid_logical_core && all_cores.count({logical_core, core_type}) > 0) {
                    print_cores_sanitized.push_back({logical_core, core_type});
                    log_info(
                        tt::LogMetal,
                        "DPRINT enabled on device {}, {} core {} (physical {}).",
                        device->id(),
                        tt::llrt::get_core_type_name(core_type),
                        logical_core.str(),
                        phys_core.str());
                } else {
                    log_warning(
                        tt::LogMetal,
                        "TT_METAL_DPRINT_CORES included {} core with logical coordinates {} (physical coordinates {}), "
                        "which is not a valid core on device {}. This coordinate will be ignored by the dprint server.",
                        tt::llrt::get_core_type_name(core_type),
                        logical_core.str(),
                        valid_logical_core ? phys_core.str() : "INVALID",
                        device->id());
                }
            }
        }
    }

    // Write print enable magic for the cores the user specified.
    uint32_t hart_mask =
        tt::llrt::RunTimeOptions::get_instance().get_feature_riscv_mask(tt::llrt::RunTimeDebugFeatureDprint);
    for (auto& logical_core : print_cores_sanitized) {
        CoreCoord phys_core = device->virtual_core_from_logical_core(logical_core.coord, logical_core.type);
        for (int hart_index = 0; hart_index < GetNumRiscs(logical_core); hart_index++) {
            if (hart_mask & (1 << hart_index)) {
                WriteInitMagic(device, phys_core, hart_index, true);
            }
        }
        if (dispatch_cores.count(logical_core)) {
            device_reads_dispatch_cores_[device] = true;
        }
    }

    device_intermediate_streams_force_flush_lock_.lock();
    TT_ASSERT(
        device_intermediate_streams_force_flush_.count(device) == 0,
        "Device {} added to DPRINT server more than once!",
        device_id);
    device_intermediate_streams_force_flush_[device] = false;
    device_intermediate_streams_force_flush_lock_.unlock();

    // Save this device + core range to the print server
    device_to_core_range_lock_.lock();
    TT_ASSERT(device_to_core_range_.count(device) == 0, "Device {} added to DPRINT server more than once!", device_id);
    device_to_core_range_[device] = print_cores_sanitized;
    device_to_core_range_lock_.unlock();
    log_info(tt::LogMetal, "DPRINT Server attached device {}", device_id);
}  // AttachDevice

void DebugPrintServerContext::DetachDevice(Device* device) {
    // Don't detach the device if it's disabled by env vars - in this case it wasn't attached.
    std::vector<chip_id_t> chip_ids =
        tt::llrt::RunTimeOptions::get_instance().get_feature_chip_ids(tt::llrt::RunTimeDebugFeatureDprint);
    if (!tt::llrt::RunTimeOptions::get_instance().get_feature_all_chips(tt::llrt::RunTimeDebugFeatureDprint)) {
        if (std::find(chip_ids.begin(), chip_ids.end(), device->id()) == chip_ids.end()) {
            return;
        }
    }

    // When we detach a device, we should poll to make sure there's no outstanding prints.
    chip_id_t chip_id = device->id();
    uint32_t risc_mask =
        tt::llrt::RunTimeOptions::get_instance().get_feature_riscv_mask(tt::llrt::RunTimeDebugFeatureDprint);
    bool outstanding_prints = true;
    while (outstanding_prints && !server_killed_due_to_hang_) {
        // Polling interval of 1ms
        std::this_thread::sleep_for(std::chrono::milliseconds(1));

        // Check all dprint-enabled cores on this device for outstanding prints.
        outstanding_prints = false;
        for (auto& logical_core : device_to_core_range_.at(device)) {
            CoreCoord phys_core = device->virtual_core_from_logical_core(logical_core.coord, logical_core.type);
            for (int risc_id = 0; risc_id < GetNumRiscs(logical_core); risc_id++) {
                if (risc_mask & (1 << risc_id)) {
                    // No need to check if risc is not dprint-enabled.
                    if (!CheckInitMagicCleared(device, phys_core, risc_id)) {
                        continue;
                    }

                    // Check if rpos < wpos, indicating unprocessed prints.
                    constexpr int eightbytes = 8;
                    uint32_t base_addr = GetDprintBufAddr(device, phys_core, risc_id);
                    auto from_dev = tt::llrt::read_hex_vec_from_core(chip_id, phys_core, base_addr, eightbytes);
                    uint32_t wpos = from_dev[0], rpos = from_dev[1];
                    if (rpos < wpos) {
                        outstanding_prints = true;
                        break;
                    }
                }
            }
            // If we already detected outstanding prints, no need to check the rest of the cores.
            if (outstanding_prints) {
                break;
            }
        }
    }

    // When we detach a device, we should poll to make sure that any leftover prints in the intermediate stream are
    // transferred to the output stream and flushed to ensure that they are printed out to the user.
    if (!server_killed_due_to_hang_) {
        device_intermediate_streams_force_flush_lock_.lock();
        device_intermediate_streams_force_flush_[device] = true;
        device_intermediate_streams_force_flush_lock_.unlock();
        bool intermediate_streams_need_to_be_flushed = true;
        while (intermediate_streams_need_to_be_flushed) {
            // Polling interval of 1ms
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            device_intermediate_streams_force_flush_lock_.lock();
            intermediate_streams_need_to_be_flushed = device_intermediate_streams_force_flush_[device];
            device_intermediate_streams_force_flush_lock_.unlock();
        }
    }

    // Remove the device from relevant data structures.
    device_intermediate_streams_force_flush_lock_.lock();
    TT_ASSERT(
        device_to_core_range_.count(device) > 0,
        "Device {} not present in DPRINT server but tried removing it!",
        device->id());
    device_intermediate_streams_force_flush_.erase(device);
    device_intermediate_streams_force_flush_lock_.unlock();

    device_to_core_range_lock_.lock();
    TT_ASSERT(
        device_to_core_range_.count(device) > 0,
        "Device {} not present in DPRINT server but tried removing it!",
        device->id());
    device_to_core_range_.erase(device);
    log_info(tt::LogMetal, "DPRINT Server dettached device {}", device->id());

    // When detaching a device, disable prints on it.
    CoreDescriptorSet all_cores = GetAllCores(device);
    for (auto& logical_core : all_cores) {
        CoreCoord phys_core = device->virtual_core_from_logical_core(logical_core.coord, logical_core.type);
        for (int hart_index = 0; hart_index < GetNumRiscs(logical_core); hart_index++) {
            WriteInitMagic(device, phys_core, hart_index, false);
        }
    }
    device_to_core_range_lock_.unlock();
}  // DetachDevice

void DebugPrintServerContext::ClearLogFile() {
    if (outfile_) {
        // Just close the file and re-open it (without append) to clear it.
        outfile_->close();
        delete outfile_;

        string file_name =
            tt::llrt::RunTimeOptions::get_instance().get_feature_file_name(tt::llrt::RunTimeDebugFeatureDprint);
        outfile_ = new ofstream(file_name);
        stream_ = outfile_ ? outfile_ : &cout;
    }
}  // ClearLogFile

void DebugPrintServerContext::ClearSignals() {
    raise_wait_lock_.lock();
    raised_signals_.clear();
    raise_wait_lock_.unlock();
}  // ClearSignals

bool DebugPrintServerContext::PeekOneHartNonBlocking(
    Device* device, const CoreDescriptor& logical_core, int hart_id, bool new_data_this_iter) {
    // If init magic isn't cleared for this risc, then dprint isn't enabled on it, don't read it.
    CoreCoord phys_core = device->virtual_core_from_logical_core(logical_core.coord, logical_core.type);
    if (!CheckInitMagicCleared(device, phys_core, hart_id)) {
        return false;
    }

    // compute the buffer address for the requested hart
    uint32_t base_addr = GetDprintBufAddr(device, phys_core, hart_id);
    chip_id_t chip_id = device->id();

    // Device is incrementing wpos
    // Host is reading wpos and incrementing local rpos up to wpos
    // Device is filling the buffer and in the end waits on host to write rpos

    // TODO(AP) - compare 8-bytes transfer and full buffer transfer latency
    // First probe only 8 bytes to see if there's anything to read
    constexpr int eightbytes = 8;
    auto from_dev = tt::llrt::read_hex_vec_from_core(device->id(), phys_core, base_addr, eightbytes);
    uint32_t wpos = from_dev[0], rpos = from_dev[1];
    uint32_t counter = 0;
    uint32_t sigval = 0;
    char val = 0;

    HartKey hart_key{chip_id, logical_core, hart_id};

    if (!risc_to_prev_type_[hart_key]) {
        risc_to_prev_type_[hart_key] = DPrintTypeID_Count;
    }

    if (!risc_to_intermediate_stream_[hart_key]) {
        risc_to_intermediate_stream_[hart_key] = new ostringstream;
    }
    ostringstream* intermediate_stream = risc_to_intermediate_stream_[hart_key];

    // Check whether this hart is currently waiting on a WAIT to be fulfilled.
    raise_wait_lock_.lock();
    if (hart_waiting_on_signal_.count(hart_key) > 0) {
        // Check if the signal the hart is waiting for has been raised.
        uint32_t wait_signal = hart_waiting_on_signal_[hart_key];
        if (raised_signals_.count(wait_signal) > 0) {
            // The signal has been raised, we can continue.
            hart_waiting_on_signal_.erase(hart_key);
        } else {
            // This hart is still waiting. This is fine as long as the print server (and therefore
            // the device) is still making progress. Unfortunetaly there's no way to check if the
            // print server is full because the next print that would overflow the buffer spins the
            // device until the buffer has more space, but checking for any new prints seems to work
            // for cases so far.
            if (!new_data_this_iter && !new_data_last_iter_) {
                // If no progress was made on both sides, then it could be an invalid wait
                // condition, which could cause a deadlock. Print a warning and set a flag to close
                // the print server in this case.
                string core_str = fmt::format(
                    "Device {}, {} core {}, riscv {}",
                    chip_id,
                    tt::llrt::get_core_type_name(logical_core.type),
                    logical_core.coord,
                    hart_id);
                string error_str = fmt::format(
                    "DPRINT server timed out on {}, waiting on a RAISE signal: {}\n", core_str, wait_signal);
                *intermediate_stream << error_str;
                TransferToAndFlushOutputStream(hart_key, intermediate_stream);
                log_warning(tt::LogMetal, "Debug Print Server encountered an error: {}", error_str);
                raise_wait_lock_.unlock();
                TT_THROW("{}", error_str);
                server_killed_due_to_hang_ = true;
            }

            // Since it's still waiting, return false here since no data was read.
            raise_wait_lock_.unlock();
            return false;
        }
    }
    raise_wait_lock_.unlock();

    if (rpos < wpos) {
        // Now read the entire buffer
        from_dev = tt::llrt::read_hex_vec_from_core(chip_id, phys_core, base_addr, DPRINT_BUFFER_SIZE);
        // at this point rpos,wpos can be stale but not reset to 0 by the producer
        // it's ok for the consumer to be behind the latest wpos+rpos from producer
        // since the corresponding data in buffer for stale rpos+wpos will not be overwritten
        // until we update rpos and write it back
        // The producer only updates rpos in case of buffer overflow.
        // Then it waits for rpos to first catch up to wpos (rpos update by the consumer) before proceeding

        DebugPrintMemLayout* l = reinterpret_cast<DebugPrintMemLayout*>(from_dev.data());
        constexpr uint32_t bufsize = sizeof(DebugPrintMemLayout::data);
        // parse the input codes
        while (rpos < wpos) {
            DPrintTypeID code = static_cast<DPrintTypeID>(l->data[rpos++]);
            TT_ASSERT(rpos <= bufsize);
            uint8_t sz = l->data[rpos++];
            TT_ASSERT(rpos <= bufsize);
            uint8_t* ptr = l->data + rpos;

            // Possible to break before rpos == wpos due to waiting on another core's raise.
            bool break_due_to_wait = false;

            // we are sharing the same output file between debug print threads for multiple cores
            switch (code) {
                case DPrintCSTR:  // const char*
                {
                    // null terminating char was included in size and should be present in the buffer
                    const char* cptr = reinterpret_cast<const char*>(ptr);
                    const size_t cptr_len = strnlen(cptr, sizeof(DebugPrintMemLayout::data) - 2);
                    if (cptr_len == sizeof(DebugPrintMemLayout::data) - 2) {
                        *intermediate_stream << "STRING BUFFER OVERFLOW DETECTED\n";
                        TransferToAndFlushOutputStream(hart_key, intermediate_stream);
                    } else {
                        // if we come across a newline char, we should transfer the data up to the newline to the output
                        // stream and flush it
                        const char* newline_pos = strchr(cptr, '\n');
                        bool contains_newline = newline_pos != nullptr;
                        while (contains_newline) {
                            const char* pos_after_newline = newline_pos + 1;
                            const uint32_t substr_len = pos_after_newline - cptr;
                            char substr_upto_newline[substr_len + 1];
                            strncpy(substr_upto_newline, cptr, substr_len);
                            substr_upto_newline[substr_len] = '\0';
                            *intermediate_stream << substr_upto_newline;
                            TransferToAndFlushOutputStream(hart_key, intermediate_stream);
                            cptr = pos_after_newline;
                            newline_pos = strchr(cptr, '\n');
                            contains_newline = newline_pos != nullptr;
                        }
                        *intermediate_stream << cptr;
                    }
                    AssertSize(sz, cptr_len + 1);
                    break;
                }
                case DPrintTILESLICE: PrintTileSlice(intermediate_stream, ptr); break;

                case DPrintENDL:
                    if (risc_to_prev_type_[hart_key] != DPrintTILESLICE ||
                        !StreamEndsWithNewlineChar(intermediate_stream)) {
                        *intermediate_stream << '\n';
                    }
                    TransferToAndFlushOutputStream(hart_key, intermediate_stream);
                    AssertSize(sz, 1);
                    break;
                case DPrintSETW:
                    val = CAST_U8P(ptr)[0];
                    *intermediate_stream << setw(val);
                    most_recent_setw = val;
                    AssertSize(sz, 1);
                    break;
                case DPrintSETPRECISION:
                    *intermediate_stream << std::setprecision(*ptr);
                    AssertSize(sz, 1);
                    break;
                case DPrintFIXED:
                    *intermediate_stream << std::fixed;
                    AssertSize(sz, 1);
                    break;
                case DPrintDEFAULTFLOAT:
                    *intermediate_stream << std::defaultfloat;
                    AssertSize(sz, 1);
                    break;
                case DPrintHEX:
                    *intermediate_stream << std::hex;
                    AssertSize(sz, 1);
                    break;
                case DPrintOCT:
                    *intermediate_stream << std::oct;
                    AssertSize(sz, 1);
                    break;
                case DPrintDEC:
                    *intermediate_stream << std::dec;
                    AssertSize(sz, 1);
                    break;
                case DPrintUINT8:
                    // iostream default uint8_t printing is as char, not an int
                    *intermediate_stream << *reinterpret_cast<uint8_t*>(ptr);
                    AssertSize(sz, 1);
                    break;
                case DPrintUINT16: {
                    uint16_t value;
                    memcpy(&value, ptr, sizeof(uint16_t));
                    *intermediate_stream << value;
                    AssertSize(sz, 2);
                } break;
                case DPrintUINT32: {
                    uint32_t value;
                    memcpy(&value, ptr, sizeof(uint32_t));
                    *intermediate_stream << value;
                    AssertSize(sz, 4);
                } break;
                case DPrintUINT64: {
                    uint64_t value;
                    memcpy(&value, ptr, sizeof(uint64_t));
                    *intermediate_stream << value;
                    AssertSize(sz, 8);
                } break;
                case DPrintINT8: {
                    int8_t value;
                    memcpy(&value, ptr, sizeof(int8_t));
                    *intermediate_stream << (int)value;  // Cast to int to ensure it prints as a number, not a char
                    AssertSize(sz, 1);
                } break;
                case DPrintINT16: {
                    int16_t value;
                    memcpy(&value, ptr, sizeof(int16_t));
                    *intermediate_stream << value;
                    AssertSize(sz, 2);
                } break;
                case DPrintINT32: {
                    int32_t value;
                    memcpy(&value, ptr, sizeof(int32_t));
                    *intermediate_stream << value;
                    AssertSize(sz, 4);
                } break;
                case DPrintINT64: {
                    int64_t value;
                    memcpy(&value, ptr, sizeof(int64_t));
                    *intermediate_stream << value;
                    AssertSize(sz, 8);
                } break;
                case DPrintFLOAT32: {
                    float value;
                    memcpy(&value, ptr, sizeof(float));
                    *intermediate_stream << value;
                    AssertSize(sz, 4);
                } break;
                case DPrintBFLOAT16: {
                    uint16_t rawValue;
                    memcpy(&rawValue, ptr, sizeof(uint16_t));
                    float value = bfloat16_to_float(rawValue);
                    *intermediate_stream << value;
                    AssertSize(sz, 2);
                } break;
                case DPrintCHAR:
                    *intermediate_stream << *reinterpret_cast<char*>(ptr);
                    AssertSize(sz, 1);
                    break;
                case DPrintU32_ARRAY:
                    PrintTypedUint32Array(
                        intermediate_stream,
                        most_recent_setw,
                        sz / 4,
                        reinterpret_cast<uint32_t*>(ptr),
                        TypedU32_ARRAY_Format_Raw);
                    break;
                case DPrintTYPED_U32_ARRAY:
                    PrintTypedUint32Array(
                        intermediate_stream, most_recent_setw, sz / 4, reinterpret_cast<uint32_t*>(ptr));
                    break;
                case DPrintRAISE:
                    memcpy(&sigval, ptr, sizeof(uint32_t));
                    // Add this newly raised signals to the set of raised signals.
                    raise_wait_lock_.lock();
                    raised_signals_.insert(sigval);
                    raise_wait_lock_.unlock();
                    AssertSize(sz, 4);
                    break;
                case DPrintWAIT: {
                    memcpy(&sigval, ptr, sizeof(uint32_t));
                    // Given that we break immediately on a wait, this core should never be waiting
                    // on multiple signals at the same time.
                    raise_wait_lock_.lock();
                    TT_ASSERT(hart_waiting_on_signal_.count(hart_key) == 0);
                    // Set that this hart is waiting on this signal, and then stop reading for now.
                    hart_waiting_on_signal_[hart_key] = sigval;
                    raise_wait_lock_.unlock();
                    break_due_to_wait = true;
                    AssertSize(sz, 4);
                } break;
                default:
                    TT_THROW(
                        "Unexpected debug print type wpos {:#x} rpos {:#x} code {} chip {} phy {}, {}",
                        wpos,
                        rpos,
                        (uint32_t)code,
                        chip_id,
                        phys_core.x,
                        phys_core.y);
            }

            risc_to_prev_type_[hart_key] = code;

            rpos += sz;  // parse the payload size
            TT_ASSERT(rpos <= wpos);

            // Break due to wait (we'll get the rest of the print buffer after the raise).
            if (break_due_to_wait) {
                break;
            }
        }  // while (rpos < wpos)

        // writes by the producer should've been atomic w.r.t code+size+payload
        // i.e at this point we shouldn't have piecemeal reads on code+size+payload
        // with rpos not aligned to wpos

        // write back to device - update rpos only
        std::vector<uint32_t> rposbuf;
        rposbuf.push_back(rpos);
        uint32_t offs = DebugPrintMemLayout().rpos_offs();
        tt::llrt::write_hex_vec_to_core(chip_id, phys_core, rposbuf, base_addr + offs);

        // Return true to signal that some print data was read
        return true;
    }  // if (rpos < wpos)

    // Return false to signal that no print data was ready this time.
    return false;
}  // PeekOneHartNonBlocking

void DebugPrintServerContext::PollPrintData(uint32_t hart_mask) {
    // Give the print server thread a reasonable name.
    pthread_setname_np(pthread_self(), "TT_DPRINT_SERVER");

    // Main print loop, go through all chips/cores/harts on the device and poll for any print data
    // written.
    while (true) {
        if (stop_print_server_) {
            // If the stop signal was received, exit the print server thread, but wait for any
            // existing prints to be wrapped up first.
            raise_wait_lock_.lock();
            size_t num_harts_waiting = hart_waiting_on_signal_.size();
            raise_wait_lock_.unlock();
            if (num_harts_waiting == 0 && !new_data_last_iter_) {
                break;
            }
        }

        // Make a copy of the device->core map, so that it can be modified while polling.
        std::map<Device*, std::vector<CoreDescriptor>> device_to_core_range_copy;
        device_to_core_range_lock_.lock();
        device_to_core_range_copy = device_to_core_range_;

        // Flag for whether any new print data was found in this round of polling.
        bool new_data_this_iter = false;
        for (auto& device_and_cores : device_to_core_range_copy) {
            Device* device = device_and_cores.first;
            device_intermediate_streams_force_flush_lock_.lock();
            if (device_intermediate_streams_force_flush_[device]) {
                TransferIntermediateStreamsToOutputStreamAndFlush(device);
                device_intermediate_streams_force_flush_[device] = false;
            }
            device_intermediate_streams_force_flush_lock_.unlock();
            for (auto& logical_core : device_and_cores.second) {
                int hart_count = GetNumRiscs(logical_core);
                for (int hart_index = 0; hart_index < hart_count; hart_index++) {
                    if (hart_mask & (1 << hart_index)) {
                        try {
                            new_data_this_iter |=
                                PeekOneHartNonBlocking(device, logical_core, hart_index, new_data_this_iter);
                        } catch (std::runtime_error& e) {
                            // Depending on if test mode is enabled, catch and stop server, or
                            // re-throw the exception.
                            if (tt::llrt::RunTimeOptions::get_instance().get_test_mode_enabled()) {
                                server_killed_due_to_hang_ = true;
                                device_to_core_range_lock_.unlock();
                                return;  // Stop the print loop
                            } else {
                                // Re-throw for instant exit
                                throw e;
                            }
                        }

                        // If this read detected a print hang, stop processing prints.
                        if (server_killed_due_to_hang_) {
                            return;
                        }
                    }
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
}  // PollPrintData

void DebugPrintServerContext::TransferIntermediateStreamsToOutputStreamAndFlush(Device* device) {
    const chip_id_t device_id = device->id();
    for (auto& [hart_key, intermediate_stream] : risc_to_intermediate_stream_) {
        const chip_id_t hart_key_device_id = get<0>(hart_key);
        if (device_id == hart_key_device_id) {
            TransferToAndFlushOutputStream(hart_key, intermediate_stream);
        }
    }
}  // TransferIntermediateStreamsToOutputStreamAndFlush

void DebugPrintServerContext::TransferToAndFlushOutputStream(
    const HartKey& hart_key, ostringstream* intermediate_stream) {
    const string& output_data = GetDataToOutput(hart_key, intermediate_stream);
    ostream* output_stream = GetOutputStream(hart_key);
    *output_stream << output_data << flush;
    ResetStream(intermediate_stream);
}  // TransferToAndFlushOutputStream

string DebugPrintServerContext::GetDataToOutput(const HartKey& hart_key, const ostringstream* stream) {
    string output;
    const bool prepend_device_core_risc =
        tt::llrt::RunTimeOptions::get_instance().get_feature_prepend_device_core_risc(tt::llrt::RunTimeDebugFeatureDprint);
    if (prepend_device_core_risc) {
        const chip_id_t device_id = get<0>(hart_key);
        const CoreDescriptor& core_desc = get<1>(hart_key);
        const uint32_t risc_id = get<2>(hart_key);

        const string& device_id_str = to_string(device_id);
        const string& core_coord_str = core_desc.coord.str();
        const string& risc_name = GetRiscName(core_desc.type, risc_id, true);
        output += fmt::format("{}:{}:{}: ", device_id_str, core_coord_str, risc_name);
    }

    if (stream->str().empty()) {
        output = "";
    } else {
        output += stream->str();
    }

    return output;
}

ostream* DebugPrintServerContext::GetOutputStream(const HartKey& hart_key) {
    ostream* output_stream = stream_;
    if (tt::llrt::RunTimeOptions::get_instance().get_feature_one_file_per_risc(tt::llrt::RunTimeDebugFeatureDprint)) {
        if (!risc_to_file_stream_[hart_key]) {
            const chip_id_t chip_id = get<0>(hart_key);
            const CoreDescriptor& logical_core = get<1>(hart_key);
            const int hart_id = get<2>(hart_key);
            string filename = tt::llrt::RunTimeOptions::get_instance().get_root_dir() + logfile_path;
            filename += fmt::format(
                "device-{}_{}-core-{}-{}_{}.txt",
                chip_id,
                tt::llrt::get_core_type_name(logical_core.type),
                logical_core.coord.x,
                logical_core.coord.y,
                GetRiscName(logical_core.type, hart_id));
            risc_to_file_stream_[hart_key] = new ofstream(filename);
        }
        output_stream = risc_to_file_stream_[hart_key];
    }

    if (mute_print_server_) {
        output_stream = &null_stream;
    }

    return output_stream;
}  // GetOutputStream

DebugPrintServerContext* DebugPrintServerContext::inst = nullptr;
bool DebugPrintServerContext::ProfilerIsRunning = false;

}  // namespace

// Implementation for functions available from dprint_server.hpp.
namespace tt {

void DprintServerAttach(Device* device) {
    // Skip if DPRINT not enabled, and make sure profiler is not running.
    if (!tt::llrt::RunTimeOptions::get_instance().get_feature_enabled(tt::llrt::RunTimeDebugFeatureDprint)) {
        return;
    }
    TT_FATAL(
        DebugPrintServerContext::ProfilerIsRunning == false,
        "Device side profiler is running, cannot start print server");

    // If no server is running, create one
    if (!DprintServerIsRunning()) {
        DebugPrintServerContext* ctx = new DebugPrintServerContext();
    }

    // Add this device to the server
    DebugPrintServerContext::inst->AttachDevice(device);
}

void DprintServerDetach(Device* device) {
    if (DprintServerIsRunning()) {
        DebugPrintServerContext::inst->DetachDevice(device);

        // Check if there's no devices left attached to the server, and close it if so.
        if (DebugPrintServerContext::inst->GetNumAttachedDevices() == 0) {
            delete DebugPrintServerContext::inst;
            DebugPrintServerContext::inst = nullptr;
        }
    }
}

void DprintServerSetProfilerState(bool profile_device) { DebugPrintServerContext::ProfilerIsRunning = profile_device; }

bool DprintServerIsRunning() { return DebugPrintServerContext::inst != nullptr; }

void DprintServerSetMute(bool mute_print_server) {
    if (DprintServerIsRunning()) {
        DebugPrintServerContext::inst->SetMute(mute_print_server);
    }
}

void DprintServerAwait() {
    if (DprintServerIsRunning()) {
        // Call the wait function for the print server, with a timeout
        auto future = std::async(
            std::launch::async, &DebugPrintServerContext::WaitForPrintsFinished, DebugPrintServerContext::inst);
        if (future.wait_for(std::chrono::seconds(1)) == std::future_status::timeout) {
            TT_THROW("Timed out waiting on debug print server to read data.");
        }
    }
}

bool DPrintServerHangDetected() {
    return DprintServerIsRunning() && DebugPrintServerContext::inst->PrintHangDetected();
}

void DPrintServerClearLogFile() {
    if (DprintServerIsRunning()) {
        DebugPrintServerContext::inst->ClearLogFile();
    }
}

void DPrintServerClearSignals() {
    if (DprintServerIsRunning()) {
        DebugPrintServerContext::inst->ClearSignals();
    }
}
bool DPrintServerReadsDispatchCores(Device* device) {
    return DprintServerIsRunning() && DebugPrintServerContext::inst->ReadsDispatchCores(device);
}

}  // namespace tt
