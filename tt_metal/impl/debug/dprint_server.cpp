// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>
#include <thread>
#include <future>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <set>
#include <filesystem>
#include "llrt/llrt.hpp"
#include "tt_metal/common/logger.hpp"

#include "dprint_server.hpp"
#include "llrt/tt_cluster.hpp"
#include "llrt/rtoptions.hpp"

#include "hostdevcommon/dprint_common.h"
#include "tt_metal/impl/device/device.hpp"
#include "tensix_types.h"

using std::uint32_t;
using std::int32_t;
using std::string;
using std::to_string;
using std::cout;
using std::endl;
using std::setw;
using std::flush;
using std::tuple;
using std::set;

using tt::tt_metal::Device;
using namespace tt;

#define CAST_U8P(p) reinterpret_cast<uint8_t*>(p)

namespace {

static string logfile_path = "generated/dprint/";

// Helper function for comparing CoreDescriptors for using in sets.
struct CoreDescriptorComparator {
    bool operator()(const CoreDescriptor &x, const CoreDescriptor &y) const {
        if (x.coord == y.coord) {
            return x.type < y.type;
        } else {
            return x.coord < y.coord;
        }
    }
};
#define CoreDescriptorSet set<CoreDescriptor, CoreDescriptorComparator>

static inline float bfloat16_to_float(uint16_t bfloat_val) {
    uint32_t uint32_data = ((uint32_t)bfloat_val) << 16;
    float f;
    std::memcpy(&f, &uint32_data, sizeof(f));
    return f;
}

static std::string GetRiscName(CoreType core_type, int hart_id) {
    if (core_type == CoreType::ETH) {
        switch (hart_id) {
            case DPRINT_RISCV_INDEX_ER:
                return "ERISC";
            // Default case falls through and handled at end.
        }
    } else {
        switch (hart_id) {
            case DPRINT_RISCV_INDEX_NC:
                return "NCRISC";
            case DPRINT_RISCV_INDEX_TR0:
                return "TRISC0";
            case DPRINT_RISCV_INDEX_TR1:
                return "TRISC1";
            case DPRINT_RISCV_INDEX_TR2:
                return "TRISC2";
            case DPRINT_RISCV_INDEX_BR:
                return "BRISC";
            // Default case falls through and handled at end.
        }
    }
    return fmt::format("UNKNOWN_RISC_ID({})", hart_id);
}

static inline uint64_t GetBaseAddr(Device *device, const CoreCoord &phys_core, int hart_id) {

    dprint_buf_msg_t *buf = device->get_dev_addr<dprint_buf_msg_t *>(phys_core, HalMemAddrType::DPRINT);

    return reinterpret_cast<uint64_t>(buf->data[hart_id]);
}

static inline int GetNumRiscs(const CoreDescriptor &core) {
    return (core.type == CoreType::ETH)? DPRINT_NRISCVS_ETH : DPRINT_NRISCVS;
}

// Helper function to get all (logical) printable cores on a device
static CoreDescriptorSet get_all_printable_cores(Device *device) {
    CoreDescriptorSet all_printable_cores;
    // The set of all printable cores is Tensix + Eth cores
    CoreCoord logical_grid_size = device->logical_grid_size();
    for (uint32_t x = 0; x < logical_grid_size.x; x++) {
        for (uint32_t y = 0; y < logical_grid_size.y; y++) {
            all_printable_cores.insert({{x, y}, CoreType::WORKER});
        }
    }
    for (const auto& logical_core : device->get_active_ethernet_cores()) {
        all_printable_cores.insert({logical_core, CoreType::ETH});
    }
    for (const auto& logical_core : device->get_inactive_ethernet_cores()) {
        all_printable_cores.insert({logical_core, CoreType::ETH});
    }

    return all_printable_cores;
}

// Helper function to get all (logical) printable cores that are used for dispatch. Should be a subset of
// get_all_printable_cores().
static CoreDescriptorSet get_dispatch_printable_cores(Device* device) {
    CoreDescriptorSet printable_dispatch_cores;
    unsigned num_cqs = tt::llrt::OptionsG.get_num_hw_cqs();
    CoreType dispatch_core_type = dispatch_core_manager::instance().get_dispatch_core_type(device->id());
    for (auto logical_core : tt::get_logical_dispatch_cores(device->id(), num_cqs, dispatch_core_type)) {
        printable_dispatch_cores.insert({logical_core, dispatch_core_type});
    }
    return printable_dispatch_cores;
}

// A null stream for when the print server is muted.
class NullBuffer : public std::streambuf {
public:
   int overflow(int c) { return c; }
};
NullBuffer null_buffer;
std::ostream null_stream(&null_buffer);

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

    std::ofstream* outfile_ = nullptr; // non-cout
    std::ostream* stream_ = nullptr; // either == outfile_ or is &cout
    std::ofstream* noc_log_ = nullptr;
    std::map<uint32_t, uint32_t> noc_xfer_counts;

    // For printing each riscs dprint to a separate file, a map from {device id, core coord x, y, hard index} to files.
    std::map<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>, std::ofstream *> risc_to_stream_;

    // A map to from {device id, core coord x, y, hart index} to the signal code it's waiting for.
    std::map<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>, uint32_t> hart_waiting_on_signal_;
    // Keep a separate set of raised signal codes so that multiple harts can wait for the same
    // signal.
    std::set<uint32_t> raised_signals_;
    std::mutex raise_wait_lock_; // A lock for these two objects since both server and main access.

    // A map from Device -> Core Range, which is used to determine which cores on which devices
    // to scan for print data. Also a lock for editing it.
    std::map<Device*, vector<CoreDescriptor>> device_to_core_range_;
    std::map<Device*, bool> device_reads_dispatch_cores_;  // True if given device reads any dispatch cores. Used to
                                                           // know whether dprint can be compiled out.
    std::mutex device_to_core_range_lock_;

    // Polls specified cores/harts on all attached devices and prints any new print data. This
    // function is the main loop for the print server thread.
    void PollPrintData(uint32_t hart_mask);

    // Peeks a specified hart for any debug prints present in the buffer, printing the contents
    // out to host-side stream. Returns true if some data was read out, and false if no new
    // print data was present on the device. Note that if an unanswered WAIT is present, the print
    // buffer on the device is only flushed  up to the WAIT, even if more print data is available
    // after it.
    bool PeekOneHartNonBlocking(
        Device *device,
        const CoreDescriptor &logical_core,
        int hart_index,
        bool new_data_this_iter
    );

    // Stores the last value of setw, so that array elements can reuse the width.
    char most_recent_setw = 0;
};

static void PrintTileSlice(ostream& stream, uint8_t* ptr, int hart_id) {
    // Since MATH RISCV doesn't have access to CBs, we can't print tiles from it. If the user still
    // tries to do this print a relevant message.
    if ((1 << hart_id) == tt::llrt::RISCV_TR1) {
        stream << "Warning: MATH core does not support TileSlice printing, omitting print..."
            << endl << std::flush;
        return;
    }

    TileSliceHostDev<0>ts_copy; // Make a copy since ptr might not be properly aligned
    std::memcpy(&ts_copy, ptr, sizeof(TileSliceHostDev<0>));
    TileSliceHostDev<0>* ts = &ts_copy;
    TT_ASSERT(offsetof(TileSliceHostDev<0>, samples_) % sizeof(uint16_t) == 0, "TileSliceHostDev<0> samples_ field is not properly aligned");
    uint16_t *samples_ = reinterpret_cast<uint16_t *>(ptr) + offsetof(TileSliceHostDev<0>, samples_) / sizeof(uint16_t);

    if (ts->w0_ == 0xFFFF) {
        stream << "BAD TILE POINTER" << std::flush;
        stream << " count=" << ts->count_ << std::flush;
    } else {
        uint32_t i = 0;
        bool count_exceeded = false;
        for (int h = ts->h0_; h < ts->h1_; h += ts->hs_) {
            for (int w = ts->w0_; w < ts->w1_; w += ts->ws_) {
                // If the number of data specified by the SliceRange exceeds the number that was
                // saved in the print buffer (set by the MAX_COUNT template parameter in the
                // TileSlice), then break early.
                if (i >= ts->count_) {
                    count_exceeded = true;
                    break;
                }
                stream << bfloat16_to_float(samples_[i]);
                if (w + ts->ws_ < ts->w1_)
                    stream << " ";
                i++;
            }

            // Break outer loop as well if MAX COUNT exceeded, also print a message to let the user
            // know that the slice has been truncated.
            if (count_exceeded) {
                stream << "<TileSlice data truncated due to exceeding max count ("
                    << to_string(ts->count_) << ")>" << endl;
                break;
            }

            if (ts->endl_rows_)
                stream << endl;
        }
    }
} // PrintTileSlice

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
    float result = 1.0 + ((float) mantissa_val / (float) (1 << mantissa_bit_count));
    result = result * pow(2, exp_val);
    if (sign) {
        result = -result;
    }
    return result;
}

// Prints a given datum in the array, given the data_format
static void PrintTensixRegisterData(ostream& stream, int setwidth, uint32_t raw_element_count, uint32_t datum, uint16_t data_format) {
    switch (data_format) {
        case static_cast<std::uint8_t>(tt::DataFormat::Float16):
        case static_cast<std::uint8_t>(tt::DataFormat::Bfp8):
        case static_cast<std::uint8_t>(tt::DataFormat::Bfp4):
        case static_cast<std::uint8_t>(tt::DataFormat::Bfp2):
        case static_cast<std::uint8_t>(tt::DataFormat::Lf8):
            stream << setw(setwidth) << make_float (5, 10, datum & 0xffff) << " ";
            stream << setw(setwidth) << make_float (5, 10, (datum >> 16) & 0xffff) << " ";
            break;
        case static_cast<std::uint8_t>(tt::DataFormat::Bfp8_b):
        case static_cast<std::uint8_t>(tt::DataFormat::Bfp4_b):
        case static_cast<std::uint8_t>(tt::DataFormat::Bfp2_b):
        case static_cast<std::uint8_t>(tt::DataFormat::Float16_b):
            stream << setw(setwidth) << make_float (8, 7, datum & 0xffff) << " ";
            stream << setw(setwidth) << make_float (8, 7, (datum >> 16) & 0xffff) << " ";
            break;
        case static_cast<std::uint8_t>(tt::DataFormat::Tf32):
            stream << setw(setwidth) << make_float(8, 10, datum) << " ";
            break;
        case static_cast<std::uint8_t>(tt::DataFormat::Float32):
            stream << setw(setwidth) << *reinterpret_cast<float*>(&datum) << " "; // Treat datum as if it stores bits for a float
            break;
        case static_cast<std::uint8_t>(tt::DataFormat::UInt32):
            stream << setw(setwidth) << datum << " ";
            break;
        case static_cast<std::uint8_t>(tt::DataFormat::UInt16):
            stream << setw(setwidth) << (datum & 0xffff) << " ";
            stream << setw(setwidth) << (datum >> 16) << " ";
            break;
        default:
            stream << "Unknown data format " << data_format << " ";
            break;
   }
}

// Prints a typed uint32 array given the number of elements including the type.
// If force_element_type is set to a valid type, it is assumed that the type is not included in the
// data array, and the type is forced to be the given type.
static void PrintTypedUint32Array(ostream& stream, int setwidth, uint32_t raw_element_count, uint32_t* data, TypedU32_ARRAY_Format force_array_type = TypedU32_ARRAY_Format_INVALID ) {
    uint16_t array_type = data[raw_element_count-1] >> 16;
    uint16_t array_subtype = data[raw_element_count-1] & 0xffff;

    raw_element_count = (force_array_type == TypedU32_ARRAY_Format_INVALID) ? raw_element_count : raw_element_count + 1;

    // stream << setwidth << "  ARRAY[len=" << std::dec << raw_element_count - 1 << ", type=" << array_type << "] = ";
    for (uint32_t i = 0; i < raw_element_count - 1; i++) {
        switch (array_type) {
            case TypedU32_ARRAY_Format_Raw:
                stream << std::hex << "0x" << data[i] << " ";
                break;
            case TypedU32_ARRAY_Format_Tensix_Config_Register_Data_Format_Type:
                PrintTensixRegisterData(stream, setwidth, raw_element_count, data[i], array_subtype);
                break;
            default:
                stream << "Unknown type " << array_type;
                break;
        }
    }
}

// Writes a magic value at wpos ptr address for dprint buffer for a specific hart/core/chip
// Used for debug print server startup sequence.
void WriteInitMagic(Device *device, const CoreCoord& phys_core, int hart_id, bool enabled) {
    // compute the buffer address for the requested hart
    uint64_t base_addr = GetBaseAddr(device, phys_core, hart_id);

    // TODO(AP): this could use a cleanup - need a different mechanism to know if a kernel is running on device.
    // Force wait for first kernel launch by first writing a non-zero and waiting for a zero.
    vector<uint32_t> initbuf = vector<uint32_t>(DPRINT_BUFFER_SIZE / sizeof(uint32_t), 0);
    initbuf[0] = uint32_t(enabled ? DEBUG_PRINT_SERVER_STARTING_MAGIC : DEBUG_PRINT_SERVER_DISABLED_MAGIC);
    tt::llrt::write_hex_vec_to_core(device->id(), phys_core, initbuf, base_addr);
} // WriteInitMagic

// Checks if our magic value was cleared by the device code
// The assumption is that if our magic number was cleared,
// it means there is a write in the queue and wpos/rpos are now valid
// Note that this is not a bulletproof way to bootstrap the print server (TODO(AP))
bool CheckInitMagicCleared(Device *device, const CoreCoord& phys_core, int hart_id) {
    // compute the buffer address for the requested hart
    uint32_t base_addr = GetBaseAddr(device, phys_core, hart_id);

    vector<uint32_t> initbuf = { DEBUG_PRINT_SERVER_STARTING_MAGIC };
    auto result = tt::llrt::read_hex_vec_from_core(device->id(), phys_core, base_addr, 4);
    return (result[0] != initbuf[0]);
} // CheckInitMagicCleared

DebugPrintServerContext::DebugPrintServerContext() {
    TT_ASSERT(inst == nullptr);
    inst = this;

    // Read hart mask + log file from rtoptions
    uint32_t hart_mask = tt::llrt::OptionsG.get_feature_riscv_mask(tt::llrt::RunTimeDebugFeatureDprint);
    string file_name = tt::llrt::OptionsG.get_feature_file_name(tt::llrt::RunTimeDebugFeatureDprint);
    bool one_file_per_risc = tt::llrt::OptionsG.get_feature_one_file_per_risc(tt::llrt::RunTimeDebugFeatureDprint);

    // One file per risc auto-generates the output files and ignores the env var for it. Print a warning if both are
    // specified just in case.
    if (file_name != "" && one_file_per_risc)
        log_warning(
            "Both TT_METAL_DPRINT_FILE_NAME and TT_METAL_DPRINT_ONE_FILE_PER_RISC are specified. "
            "TT_METAL_DPRINT_FILE_NAME will be ignored.");

    // Set the output stream according to RTOptions, either a file name or stdout if none specified.
    std::filesystem::path output_dir(tt::llrt::OptionsG.get_root_dir() + logfile_path);
    std::filesystem::create_directories(output_dir);
    if (file_name != "" && !one_file_per_risc) {
        outfile_ = new std::ofstream(file_name);
    }
    stream_ = outfile_ ? outfile_ : &cout;
    noc_log_ = new std::ofstream("noc_log.csv");

    stop_print_server_ = false;
    mute_print_server_ = false;
    new_data_last_iter_ = false;
    server_killed_due_to_hang_ = false;

    // Spin off the thread that runs the print server.
    print_server_thread_ = new std::thread(
        [this, hart_mask] { PollPrintData(hart_mask); }
    );
} // DebugPrintServerContext

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
    for (auto &key_and_stream : risc_to_stream_) {
        key_and_stream.second->close();
        delete key_and_stream.second;
    }
    for (auto &size_and_count : noc_xfer_counts)
        *noc_log_ << size_and_count.first << "," << size_and_count.second << "\n";
    noc_log_->close();
    delete noc_log_;
    inst = nullptr;
} // ~DebugPrintServerContext

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
        if (server_killed_due_to_hang_)
            break;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        raise_wait_lock_.lock();
        num_harts_waiting = hart_waiting_on_signal_.size();
        raise_wait_lock_.unlock();
    } while (num_harts_waiting > 0 || new_data_last_iter_ || wait_loop_iterations_ < 2);
} // WaitForPrintsFinished

void DebugPrintServerContext::AttachDevice(Device* device) {
    chip_id_t device_id = device->id();

    // A set of all valid printable cores, used for checking the user input. Note that the coords
    // here are physical.
    CoreDescriptorSet all_printable_cores = get_all_printable_cores(device);
    CoreDescriptorSet dispatch_printable_cores = get_dispatch_printable_cores(device);

    // Initialize all print buffers on all cores on the device to have print disabled magic. We
    // will then write print enabled magic for only the cores the user has specified to monitor.
    // This way in the kernel code (dprint.h) we can detect whether the magic value is present and
    // skip prints entirely to prevent kernel code from hanging waiting for the print buffer to be
    // flushed from the host.
    for (auto &logical_core : all_printable_cores) {
        CoreCoord phys_core = device->physical_core_from_logical_core(logical_core);
        for (int hart_index = 0; hart_index < GetNumRiscs(logical_core); hart_index++) {
            WriteInitMagic(device, phys_core, hart_index, false);
        }
    }

    // If RTOptions doesn't enable DPRINT on this device, return here and don't actually attach it
    // to the server.
    vector<chip_id_t> chip_ids = tt::llrt::OptionsG.get_feature_chip_ids(tt::llrt::RunTimeDebugFeatureDprint);
    if (!tt::llrt::OptionsG.get_feature_all_chips(tt::llrt::RunTimeDebugFeatureDprint))
        if (std::find(chip_ids.begin(), chip_ids.end(), device->id()) == chip_ids.end())
            return;

    // Core range depends on whether dprint_all_cores flag is set.
    vector<CoreDescriptor> print_cores_sanitized;
    for (CoreType core_type : {CoreType::WORKER, CoreType::ETH}) {
        if (tt::llrt::OptionsG.get_feature_all_cores(tt::llrt::RunTimeDebugFeatureDprint, core_type) ==
            tt::llrt::RunTimeDebugClassAll) {
            // Print from all cores of the given type, cores returned here are guaranteed to be valid.
            for (CoreDescriptor logical_core : all_printable_cores) {
                if (logical_core.type == core_type)
                    print_cores_sanitized.push_back(logical_core);
            }
            log_info(
                tt::LogMetal,
                "DPRINT enabled on device {}, all {} cores.",
                device->id(),
                tt::llrt::get_core_type_name(core_type));
        } else if (
            tt::llrt::OptionsG.get_feature_all_cores(tt::llrt::RunTimeDebugFeatureDprint, core_type) ==
            tt::llrt::RunTimeDebugClassDispatch) {
            for (CoreDescriptor logical_core : dispatch_printable_cores) {
                if (logical_core.type == core_type)
                    print_cores_sanitized.push_back(logical_core);
            }
            log_info(
                tt::LogMetal,
                "DPRINT enabled on device {}, {} dispatch cores.",
                device->id(),
                tt::llrt::get_core_type_name(core_type));
        } else if (
            tt::llrt::OptionsG.get_feature_all_cores(tt::llrt::RunTimeDebugFeatureDprint, core_type) ==
            tt::llrt::RunTimeDebugClassWorker) {
            // For worker cores, take all cores and remove dispatch cores.
            for (CoreDescriptor logical_core : all_printable_cores) {
                if (dispatch_printable_cores.find(logical_core) == dispatch_printable_cores.end()) {
                if (logical_core.type == core_type)
                    print_cores_sanitized.push_back(logical_core);
                }
            }
            log_info(
                tt::LogMetal,
                "DPRINT enabled on device {}, {} worker cores.",
                device->id(),
                tt::llrt::get_core_type_name(core_type));
        } else {
            // No "all cores" option provided, which means print from the cores specified by the user
            vector<CoreCoord>& print_cores =
                tt::llrt::OptionsG.get_feature_cores(tt::llrt::RunTimeDebugFeatureDprint)[core_type];

            // We should also validate that the cores the user specified are valid worker cores.
            for (auto &logical_core : print_cores) {
                // Need to convert user-specified logical cores to physical cores, this can throw
                // if the user gave bad coords.
                CoreCoord phys_core;
                bool valid_logical_core = true;
                try {
                    phys_core = device->physical_core_from_logical_core(logical_core, core_type);
                } catch (std::runtime_error& error) {
                    valid_logical_core = false;
                }
                if (valid_logical_core && all_printable_cores.count({logical_core, core_type}) > 0) {
                    print_cores_sanitized.push_back({logical_core, core_type});
                    log_info(
                        tt::LogMetal,
                        "DPRINT enabled on device {}, {} core {} (physical {}).",
                        device->id(),
                        tt::llrt::get_core_type_name(core_type),
                        logical_core.str(),
                        phys_core.str()
                    );
                } else {
                    log_warning(
                        tt::LogMetal,
                        "TT_METAL_DPRINT_CORES included {} core with logical coordinates {} (physical coordinates {}), which is not a valid core on device {}. This coordinate will be ignored by the dprint server.",
                        tt::llrt::get_core_type_name(core_type),
                        logical_core.str(),
                        valid_logical_core? phys_core.str() : "INVALID",
                        device->id()
                    );
                }
            }
        }
    }

    // Write print enable magic for the cores the user specified.
    uint32_t hart_mask = tt::llrt::OptionsG.get_feature_riscv_mask(tt::llrt::RunTimeDebugFeatureDprint);
    for (auto &logical_core : print_cores_sanitized) {
        CoreCoord phys_core = device->physical_core_from_logical_core(logical_core);
        for (int hart_index = 0; hart_index < GetNumRiscs(logical_core); hart_index++) {
            if (hart_mask & (1<<hart_index)) {
                WriteInitMagic(device, phys_core, hart_index, true);
            }
        }
        if (dispatch_cores.count(logical_core))
            device_reads_dispatch_cores_[device] = true;
    }

    // Save this device + core range to the print server
    device_to_core_range_lock_.lock();
    TT_ASSERT(device_to_core_range_.count(device) == 0, "Device {} added to DPRINT server more than once!", device_id);
    device_to_core_range_[device] = print_cores_sanitized;
    device_to_core_range_lock_.unlock();
    log_info(tt::LogMetal, "DPRINT Server attached device {}", device_id);
} // AttachDevice

void DebugPrintServerContext::DetachDevice(Device* device) {
    // Don't detach the device if it's disabled by env vars - in this case it wasn't attached.
    vector<chip_id_t> chip_ids = tt::llrt::OptionsG.get_feature_chip_ids(tt::llrt::RunTimeDebugFeatureDprint);
    if (!tt::llrt::OptionsG.get_feature_all_chips(tt::llrt::RunTimeDebugFeatureDprint))
        if (std::find(chip_ids.begin(), chip_ids.end(), device->id()) == chip_ids.end())
            return;

    // When we detach a device, we should poll to make sure there's no outstanding prints.
    chip_id_t chip_id = device->id();
    uint32_t risc_mask = tt::llrt::OptionsG.get_feature_riscv_mask(tt::llrt::RunTimeDebugFeatureDprint);
    bool outstanding_prints = true;
    while (outstanding_prints && !server_killed_due_to_hang_) {
        // Polling interval of 1ms
        std::this_thread::sleep_for(std::chrono::milliseconds(1));

        // Check all dprint-enabled cores on this device for outstanding prints.
        outstanding_prints = false;
        for (auto &logical_core : device_to_core_range_.at(device)) {
            CoreCoord phys_core = device->physical_core_from_logical_core(logical_core);
            for (int risc_id = 0; risc_id < GetNumRiscs(logical_core); risc_id++) {
                if (risc_mask & (1<<risc_id)) {
                    // No need to check if risc is not dprint-enabled.
                    if (!CheckInitMagicCleared(device, phys_core, risc_id))
                    continue;

                    // Check if rpos < wpos, indicating unprocessed prints.
                    constexpr int eightbytes = 8;
                    uint32_t base_addr = GetBaseAddr(device, phys_core, risc_id);
                    auto from_dev = tt::llrt::read_hex_vec_from_core(chip_id, phys_core, base_addr, eightbytes);
                    uint32_t wpos = from_dev[0], rpos = from_dev[1];
                    if (rpos < wpos) {
                        outstanding_prints = true;
                        break;
                    }
                }
            }
            // If we already detected outstanding prints, no need to check the rest of the cores.
            if (outstanding_prints)
                break;
        }
    }

    // Remove the device from relevant data structures.
    device_to_core_range_lock_.lock();
    TT_ASSERT(device_to_core_range_.count(device) > 0, "Device {} not present in DPRINT server but tried removing it!", device->id());
    device_to_core_range_.erase(device);
    log_info(tt::LogMetal, "DPRINT Server dettached device {}", device->id());

    // When detaching a device, disable prints on it.
    CoreDescriptorSet all_printable_cores = get_all_printable_cores(device);
    for (auto &logical_core : all_printable_cores) {
            CoreCoord phys_core = device->physical_core_from_logical_core(logical_core);
            for (int hart_index = 0; hart_index < GetNumRiscs(logical_core); hart_index++) {
                WriteInitMagic(device, phys_core, hart_index, false);
            }
    }
    device_to_core_range_lock_.unlock();
} // DetachDevice

void DebugPrintServerContext::ClearLogFile() {
    if (outfile_) {
        // Just close the file and re-open it (without append) to clear it.
        outfile_->close();
        delete outfile_;

        string file_name = tt::llrt::OptionsG.get_feature_file_name(tt::llrt::RunTimeDebugFeatureDprint);
        outfile_ = new std::ofstream(file_name);
        stream_ = outfile_ ? outfile_ : &cout;
    }
} // ClearLogFile

void DebugPrintServerContext::ClearSignals() {
    raise_wait_lock_.lock();
    raised_signals_.clear();
    raise_wait_lock_.unlock();
} // ClearSignals

bool DebugPrintServerContext::PeekOneHartNonBlocking(
    Device* device, const CoreDescriptor& logical_core, int hart_id, bool new_data_this_iter) {

    // If init magic isn't cleared for this risc, then dprint isn't enabled on it, don't read it.
    CoreCoord phys_core = device->physical_core_from_logical_core(logical_core);
    if (!CheckInitMagicCleared(device, phys_core, hart_id))
        return false;

    // compute the buffer address for the requested hart
    uint32_t base_addr = GetBaseAddr(device, phys_core, hart_id);
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

    // Choose which stream to output the dprint data to. Can be auto-generated files, the user-selected file, stdout, or
    // nothing.
    ostream *stream_ptr = stream_;
    std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> hart_key {chip_id, phys_core.x, phys_core.y, hart_id};
    if (tt::llrt::OptionsG.get_feature_one_file_per_risc(tt::llrt::RunTimeDebugFeatureDprint)) {
        if (!risc_to_stream_[hart_key]) {
            std::string filename = tt::llrt::OptionsG.get_root_dir() + logfile_path;
            filename += fmt::format(
                "device-{}_{}-core-{}-{}_{}.txt",
                chip_id,
                tt::llrt::get_core_type_name(logical_core.type),
                logical_core.coord.x,
                logical_core.coord.y,
                GetRiscName(logical_core.type, hart_id));
            risc_to_stream_[hart_key] = new std::ofstream(filename);
        }
        stream_ptr = risc_to_stream_[hart_key];
    }
    if (mute_print_server_) {
        stream_ptr = &null_stream;
    }
    ostream &stream = *stream_ptr;

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
                stream << error_str << flush;
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
        const char* cptr = nullptr;
        int nlen = 0;
        uint32_t i = 0, h = 0, w = 0;
        while (rpos < wpos) {
            auto code = static_cast<DPrintTypeID>(l->data[rpos++]); TT_ASSERT(rpos <= bufsize);
            uint8_t sz = l->data[rpos++]; TT_ASSERT(rpos <= bufsize);
            uint8_t* ptr = l->data + rpos;

            // Possible to break before rpos == wpos due to waiting on another core's raise.
            bool break_due_to_wait = false;

            // we are sharing the same output file between debug print threads for multiple cores
            switch(code) {
                case DPrintCSTR: // const char*
                    // terminating zero was included in size and should be present in the buffer
                    cptr = reinterpret_cast<const char*>(ptr);
                    nlen = strnlen(cptr, sizeof(DebugPrintMemLayout::data));
                    if (nlen >= 200)
                        stream << "STRING BUFFER OVERFLOW DETECTED" << endl;
                    else
                        stream << cptr;
                    TT_ASSERT(sz == strlen(cptr)+1);
                break;
                case DPrintTILESLICE:
                    PrintTileSlice(stream, ptr, hart_id);
                break;

                case DPrintENDL:
                    stream << endl;
                    TT_ASSERT(sz == 1);
                break;
                case DPrintSETW:
                    val = CAST_U8P(ptr)[0];
                    stream << setw(val);
                    most_recent_setw = val;
                    TT_ASSERT(sz == 1);
                break;
                case DPrintSETPRECISION:
                    stream << std::setprecision(*ptr);
                    TT_ASSERT(sz == 1);
                break;
                case DPrintNOC_LOG_XFER:
                    if (tt::llrt::OptionsG.get_dprint_noc_transfers())
                        noc_xfer_counts[*reinterpret_cast<uint32_t*>(ptr)]++;
                    TT_ASSERT(sz == 4);
                break;
                case DPrintFIXED:
                    stream << std::fixed;
                    TT_ASSERT(sz == 1);
                break;
                case DPrintDEFAULTFLOAT:
                    stream << std::defaultfloat;
                    TT_ASSERT(sz == 1);
                break;
                case DPrintHEX:
                    stream << std::hex;
                    TT_ASSERT(sz == 1);
                break;
                case DPrintOCT:
                    stream << std::oct;
                    TT_ASSERT(sz == 1);
                break;
                case DPrintDEC:
                    stream << std::dec;
                    TT_ASSERT(sz == 1);
                break;
                case DPrintUINT8:
                    // iostream default uint8_t printing is as char, not an int
                    stream << *reinterpret_cast<uint8_t*>(ptr);
                    TT_ASSERT(sz == 1);
                break;
                case DPrintUINT16:
                    {
                        uint16_t value;
                        memcpy(&value, ptr, sizeof(uint16_t));
                        stream << value;
                        TT_ASSERT(sz == 2);
                    }
                    break;
                case DPrintUINT32:
                    {
                        uint32_t value;
                        memcpy(&value, ptr, sizeof(uint32_t));
                        stream << value;
                        TT_ASSERT(sz == 4);
                    }
                    break;
                case DPrintUINT64:
                    {
                        uint64_t value;
                        memcpy(&value, ptr, sizeof(uint64_t));
                        stream << value;
                        TT_ASSERT(sz == 8);
                    }
                    break;
                case DPrintINT8:
                    {
                        int8_t value;
                        memcpy(&value, ptr, sizeof(int8_t));
                        stream << (int)value;  // Cast to int to ensure it prints as a number, not a char
                        TT_ASSERT(sz == 1);
                    }
                    break;
                case DPrintINT16:
                    {
                        int16_t value;
                        memcpy(&value, ptr, sizeof(int16_t));
                        stream << value;
                        TT_ASSERT(sz == 2);
                    }
                    break;
                case DPrintINT32:
                    {
                        int32_t value;
                        memcpy(&value, ptr, sizeof(int32_t));
                        stream << value;
                        TT_ASSERT(sz == 4);
                    }
                    break;
                case DPrintINT64:
                    {
                        int64_t value;
                        memcpy(&value, ptr, sizeof(int64_t));
                        stream << value;
                        TT_ASSERT(sz == 8);
                    }
                    break;
                case DPrintFLOAT32:
                    {
                        float value;
                        memcpy(&value, ptr, sizeof(float));
                        stream << value;
                        TT_ASSERT(sz == 4);
                    }
                    break;
                case DPrintBFLOAT16:
                    {
                        uint16_t rawValue;
                        memcpy(&rawValue, ptr, sizeof(uint16_t));
                        float value = bfloat16_to_float(rawValue);
                        stream << value;
                        TT_ASSERT(sz == 2);
                    }
                    break;
                case DPrintCHAR:
                    stream << *reinterpret_cast<char*>(ptr);
                    TT_ASSERT(sz == 1);
                break;
                case DPrintU32_ARRAY:
                    PrintTypedUint32Array(stream, most_recent_setw, sz/4, reinterpret_cast<uint32_t*>(ptr), TypedU32_ARRAY_Format_Raw);
                break;
                case DPrintTYPED_U32_ARRAY:
                    PrintTypedUint32Array(stream, most_recent_setw, sz/4, reinterpret_cast<uint32_t*>(ptr));
                break;
                case DPrintRAISE:
                    memcpy (&sigval, ptr, sizeof(uint32_t));
                    // Add this newly raised signals to the set of raised signals.
                    raise_wait_lock_.lock();
                    raised_signals_.insert(sigval);
                    raise_wait_lock_.unlock();
                    //stream << "\nRaised signal=" << sigval << endl;
                    TT_ASSERT(sz == 4);
                break;
                case DPrintWAIT:
                    {
                    memcpy (&sigval, ptr, sizeof(uint32_t));
                    // Given that we break immediately on a wait, this core should never be waiting
                    // on multiple signals at the same time.
                    std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> hart_key {chip_id, phys_core.x, phys_core.y, hart_id};
                    raise_wait_lock_.lock();
                    TT_ASSERT(hart_waiting_on_signal_.count(hart_key) == 0);
                    // Set that this hart is waiting on this signal, and then stop reading for now.
                    hart_waiting_on_signal_[hart_key] = sigval;
                    raise_wait_lock_.unlock();
                    break_due_to_wait = true;
                    //stream << "\nWaiting on signal=" << *reinterpret_cast<uint32_t*>(ptr);
                    TT_ASSERT(sz == 4);
                    }
                break;
                default:
                    TT_THROW("Unexpected debug print type code");
            }

            // TODO(AP): this is slow but leaving here for now for debugging the debug prints themselves
            stream << flush;

            rpos += sz; // parse the payload size
            TT_ASSERT(rpos <= wpos);

            // Break due to wait (we'll get the rest of the print buffer after the raise).
            if (break_due_to_wait)
                break;
        } // while (rpos < wpos)

        // writes by the producer should've been atomic w.r.t code+size+payload
        // i.e at this point we shouldn't have piecemeal reads on code+size+payload
        // with rpos not aligned to wpos

        // write back to device - update rpos only
        vector<uint32_t> rposbuf;
        rposbuf.push_back(rpos);
        uint32_t offs = DebugPrintMemLayout().rpos_offs();
        tt::llrt::write_hex_vec_to_core(chip_id, phys_core, rposbuf, base_addr+offs);

        // Return true to signal that some print data was read
        return true;
    } // if (rpos < wpos)

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
            if (num_harts_waiting == 0 && !new_data_last_iter_)
                break;
        }

        // Make a copy of the device->core map, so that it can be modified while polling.
        std::map<Device*, vector<CoreDescriptor>> device_to_core_range_copy;
        device_to_core_range_lock_.lock();
        device_to_core_range_copy = device_to_core_range_;

        // Flag for whether any new print data was found in this round of polling.
        bool new_data_this_iter = false;
        for (auto& device_and_cores : device_to_core_range_copy) {
            chip_id_t chip_id = device_and_cores.first->id();
            for (auto &logical_core : device_and_cores.second) {
                int hart_count = GetNumRiscs(logical_core);
                for (int hart_index = 0; hart_index < hart_count; hart_index++) {
                    if (hart_mask & (1<<hart_index)) {
                        try {
                            new_data_this_iter |= PeekOneHartNonBlocking(
                                device_and_cores.first,
                                logical_core,
                                hart_index,
                                new_data_this_iter
                            );
                        } catch (std::runtime_error& e) {
                            // Depending on if test mode is enabled, catch and stop server, or
                            // re-throw the exception.
                            if (tt::llrt::OptionsG.get_test_mode_enabled()) {
                                server_killed_due_to_hang_ = true;
                                device_to_core_range_lock_.unlock();
                                return; // Stop the print loop
                            } else {
                                // Re-throw for instant exit
                                throw e;
                            }

                        }

                        // If this read detected a print hang, stop processing prints.
                        if (server_killed_due_to_hang_)
                            return;
                    }
                }
            }
        }

        // Signal whether the print server is currently processing data.
        new_data_last_iter_ = new_data_this_iter;
        device_to_core_range_lock_.unlock();
        // Sleep for a few ms if no data was processed.
        if (!new_data_last_iter_)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));

        wait_loop_iterations_++;
    }
} // PollPrintData

DebugPrintServerContext* DebugPrintServerContext::inst = nullptr;
bool DebugPrintServerContext::ProfilerIsRunning = false;

} // anon namespace

// Implementation for functions available from dprint_server.hpp.
namespace tt {

void DprintServerAttach(Device* device) {
    // Skip if DPRINT not enabled, and make sure profiler is not running.
    if (!tt::llrt::OptionsG.get_feature_enabled(tt::llrt::RunTimeDebugFeatureDprint))
        return;
    TT_FATAL(
       DebugPrintServerContext::ProfilerIsRunning == false,
       "Device side profiler is running, cannot start print server"
    );

    // If no server is running, create one
    if (!DprintServerIsRunning())
        DebugPrintServerContext* ctx = new DebugPrintServerContext();

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

void DprintServerSetProfilerState(bool profile_device) {
    DebugPrintServerContext::ProfilerIsRunning = profile_device;
}

bool DprintServerIsRunning() {
    return DebugPrintServerContext::inst != nullptr;
}

void DprintServerSetMute(bool mute_print_server) {
    if (DprintServerIsRunning())
        DebugPrintServerContext::inst->SetMute(mute_print_server);
}

void DprintServerAwait() {
    if (DprintServerIsRunning()) {
        // Call the wait function for the print server, with a timeout
        auto future = std::async(
            std::launch::async,
            &DebugPrintServerContext::WaitForPrintsFinished,
            DebugPrintServerContext::inst
        );
        if (future.wait_for(std::chrono::seconds(1)) == std::future_status::timeout) {
            TT_THROW("Timed out waiting on debug print server to read data.");
        }
    }
}

bool DPrintServerHangDetected() {
    return DprintServerIsRunning() && DebugPrintServerContext::inst->PrintHangDetected();
}

void DPrintServerClearLogFile() {
    if (DprintServerIsRunning())
        DebugPrintServerContext::inst->ClearLogFile();
}

void DPrintServerClearSignals() {
    if (DprintServerIsRunning())
        DebugPrintServerContext::inst->ClearSignals();
}
bool DPrintServerReadsDispatchCores(Device* device) {
    return DprintServerIsRunning() && DebugPrintServerContext::inst->ReadsDispatchCores(device);
}

} // namespace tt
