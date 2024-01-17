// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <thread>
#include <future>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <set>
#include "llrt/llrt.hpp"
#include "common/logger.hpp"

#include "dprint_server.hpp"
#include "llrt/tt_cluster.hpp"
#include "llrt/rtoptions.hpp"

#include "hostdevcommon/common_runtime_address_map.h"
#include "hostdevcommon/dprint_common.h"

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

static inline float bfloat16_to_float(uint16_t bfloat_val) {
    uint32_t val = bfloat_val << 16;
    return *reinterpret_cast<float*>(&val);
}

static inline uint32_t GetBaseAddr(int chip_id, const CoreCoord &core, int hart_id) {
    // For tensix cores, compute the buffer address for the requested hart.
    uint32_t base_addr = PRINT_BUFFER_NC + hart_id*PRINT_BUFFER_SIZE;

    // Ethernet cores have a different address mapping
    if (tt::llrt::is_ethernet_core(core, chip_id))
        base_addr = eth_l1_mem::address_map::PRINT_BUFFER_ER;

    return base_addr;
}

static inline int GetNumRiscs(int chip_id, const CoreCoord &core) {
    return (tt::llrt::is_ethernet_core(core, chip_id))? DPRINT_NRISCVS_ETH : DPRINT_NRISCVS;
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

    std::ofstream* outfile_ = nullptr; // non-cout
    std::ostream* stream_ = nullptr; // either == outfile_ or is &cout

    // A map to from {core coord x, y, hart index} to the signal code it's waiting for.
    std::map<tuple<uint32_t, uint32_t, uint32_t>, uint32_t> hart_waiting_on_signal_;
    // Keep a separate set of raised signal codes so that multiple harts can wait for the same
    // signal.
    std::set<uint32_t> raised_signals_;

    // A map from Device -> Core Range, which is used to determine which cores on which devices
    // to scan for print data. Also a lock for editing it.
    std::map<Device*, vector<CoreCoord>> device_to_core_range_;
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
        int chip_id,
        const CoreCoord& core,
        int hart_index,
        bool new_data_this_iter
    );
};

static void PrintTileSlice(ostream& stream, uint8_t* ptr, int hart_id) {
    // Since MATH RISCV doesn't have access to CBs, we can't print tiles from it. If the user still
    // tries to do this print a relevant message.
    if ((1 << hart_id) == DPRINT_RISCV_TR1) {
        stream << "Warning: MATH core does not support TileSlice printing, omitting print..."
            << endl << std::flush;
        return;
    }

    TileSliceHostDev<0>* ts = reinterpret_cast<TileSliceHostDev<0>*>(ptr);
    stream << "TILE: (" << endl << std::flush;
    if (ts->w0_ == 0xFFFF) {
        stream << "BAD TILE POINTER" << std::flush;
        stream << " count=" << ts->count_ << std::flush;
    } else {
        uint32_t i = 0;
        bool count_exceeded = false;
        for (int h = ts->h0_; h < ts->h1_; h += ts->hs_) {
            if (ts->w0_ < ts->w1_)
                stream << "  ";
            for (int w = ts->w0_; w < ts->w1_; w += ts->ws_) {
                // If the number of data specified by the SliceRange exceeds the number that was
                // saved in the print buffer (set by the MAX_COUNT template parameter in the
                // TileSlice), then break early.
                if (i >= ts->count_) {
                    count_exceeded = true;
                    break;
                }
                stream << bfloat16_to_float(ts->samples_[i]);
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
    stream << endl << "  ptr=" << ts->ptr_ << ")" << endl;
} // PrintTileSlice

// Writes a magic value at wpos ptr address for dprint buffer for a specific hart/core/chip
// Used for debug print server startup sequence.
void WriteInitMagic(int chip_id, const CoreCoord& core, int hart_id, bool enabled) {
    // compute the buffer address for the requested hart
    uint32_t base_addr = GetBaseAddr(chip_id, core, hart_id);

    // TODO(AP): this could use a cleanup - need a different mechanism to know if a kernel is running on device.
    // Force wait for first kernel launch by first writing a non-zero and waiting for a zero.
    vector<uint32_t> initbuf = { uint32_t(enabled ? DEBUG_PRINT_SERVER_STARTING_MAGIC : DEBUG_PRINT_SERVER_DISABLED_MAGIC) };
    tt::llrt::write_hex_vec_to_core(chip_id, core, initbuf, base_addr);
} // WriteInitMagic

// Checks if our magic value was cleared by the device code
// The assumption is that if our magic number was cleared,
// it means there is a write in the queue and wpos/rpos are now valid
// Note that this is not a bulletproof way to bootstrap the print server (TODO(AP))
bool CheckInitMagicCleared(int chip_id, const CoreCoord& core, int hart_id) {
    // compute the buffer address for the requested hart
    uint32_t base_addr = GetBaseAddr(chip_id, core, hart_id);

    vector<uint32_t> initbuf = { DEBUG_PRINT_SERVER_STARTING_MAGIC };
    auto result = tt::llrt::read_hex_vec_from_core(chip_id, core, base_addr, 4);
    return (result[0] != initbuf[0]);
} // CheckInitMagicCleared

DebugPrintServerContext::DebugPrintServerContext() {
    TT_ASSERT(inst == nullptr);
    inst = this;

    // Read hart mask + log file from rtoptions
    uint32_t hart_mask = tt::llrt::OptionsG.get_dprint_riscv_mask();
    string file_name = tt::llrt::OptionsG.get_dprint_file_name();

    // Set the output stream according to RTOptions, either a file name or stdout if none specified.
    if (file_name != "") {
        outfile_ = new std::ofstream(file_name);
    }
    stream_ = outfile_ ? outfile_ : &cout;

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
        TT_FATAL(false && "Timed out waiting on debug print thread to terminate.");
    }
    delete print_server_thread_;
    print_server_thread_ = nullptr;

    if (outfile_) {
        outfile_->close();
        delete outfile_;
    }
    inst = nullptr;
} // ~DebugPrintServerContext

void DebugPrintServerContext::WaitForPrintsFinished() {
    // Simply poll the flag every few ms to check whether new data is still being processed,
    // or whether any cores are waiting for a signal to be raised.
    // TODO(dma): once we have access to the device is there a way we can poll the device to
    // check whether more print data is coming?
    do {
        // No need to await if the server was killed already due to a hang.
        if (server_killed_due_to_hang_)
            break;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    } while (hart_waiting_on_signal_.size() > 0 || new_data_last_iter_);
} // WaitForPrintsFinished

void DebugPrintServerContext::AttachDevice(Device* device) {
    chip_id_t device_id = device->id();

    // A set of all valid printable cores, used for checking the user input. Note that the coords
    // here are physical, which matches the user input for configuring the print server.
    set<CoreCoord> all_printable_cores;
    // The set of all printable cores is Tensix + Eth cores
    CoreCoord logical_grid_size = device->logical_grid_size();
    for (uint32_t x = 0; x < logical_grid_size.x; x++) {
        for (uint32_t y = 0; y < logical_grid_size.y; y++) {
            CoreCoord logical_coord(x, y);
            CoreCoord worker_core = device->worker_core_from_logical_core(logical_coord);
            all_printable_cores.insert(worker_core);
        }
    }
    for (const auto& eth_core : device->ethernet_cores()) {
        CoreCoord physical_core = device->ethernet_core_from_logical_core(eth_core);
        all_printable_cores.insert(physical_core);
    }

    // Core range depends on whether dprint_all_cores flag is set.
    vector<CoreCoord> print_cores_sanitized;
    if (tt::llrt::OptionsG.get_dprint_all_cores()) {
        // Print from all worker cores, cores returned here are guaranteed to be valid.
        print_cores_sanitized = vector<CoreCoord>(all_printable_cores.begin(), all_printable_cores.end());
    } else {
        // Only print from the cores specified by the user.
        vector<CoreCoord> print_cores = tt::llrt::OptionsG.get_dprint_cores();

        // We should also validate that the cores the user specified are valid worker cores.
        for (auto core : print_cores) {
            if (all_printable_cores.count(core) > 0) {
                print_cores_sanitized.push_back(core);
            } else {
                log_warning(
                    tt::LogMetal,
                    "TT_METAL_DPRINT_CORES included worker core ({}, {}), which is not a valid coordinate. This coordinate will be ignored by the dprint server.",
                    core.x,
                    core.y
                );
            }
        }
    }

    // Initialize all print buffers on all cores on the device to have print disabled magic. We
    // will then write print enabled magic for only the cores the user has specified to monitor.
    // This way in the kernel code (dprint.h) we can detect whether the magic value is present and
    // skip prints entirely to prevent kernel code from hanging waiting for the print buffer to be
    // flushed from the host.
    for (auto core : all_printable_cores) {
        int hart_count = GetNumRiscs(device_id, core);
        for (int hart_index = 0; hart_index < hart_count; hart_index++) {
            WriteInitMagic(device_id, core, hart_index, false);
        }
    }
    // Write print enable magic for the cores the user specified.
    uint32_t hart_mask = tt::llrt::OptionsG.get_dprint_riscv_mask();
    for (auto core : print_cores_sanitized) {
        int hart_count = GetNumRiscs(device_id, core);
        for (int hart_index = 0; hart_index < hart_count; hart_index++) {
            if (hart_mask & (1<<hart_index)) {
                WriteInitMagic(device_id, core, hart_index, true);
            }
        }
    }

    // Save this device + core range to the print server
    device_to_core_range_lock_.lock();
    TT_ASSERT(device_to_core_range_.count(device) == 0, "Device {} added to DPRINT server more than once!", device_id);
    device_to_core_range_[device] = print_cores_sanitized;
    device_to_core_range_lock_.unlock();
    log_info(tt::LogMetal, "DPRINT Server attached device {}", device_id);
} // AttachDevice

void DebugPrintServerContext::DetachDevice(Device* device) {
    device_to_core_range_lock_.lock();
    TT_ASSERT(device_to_core_range_.count(device) > 0, "Device {} not present in DPRINT server but tried removing it!", device->id());
    device_to_core_range_.erase(device);
    device_to_core_range_lock_.unlock();
    log_info(tt::LogMetal, "DPRINT Server dettached device {}", device->id());
} // DetachDevice

void DebugPrintServerContext::ClearLogFile() {
    if (outfile_) {
        // Just close the file and re-open it (without append) to clear it.
        outfile_->close();
        delete outfile_;

        string file_name = tt::llrt::OptionsG.get_dprint_file_name();
        outfile_ = new std::ofstream(file_name);
        stream_ = outfile_ ? outfile_ : &cout;
    }
} // ClearLogFile

void DebugPrintServerContext::ClearSignals() {
    raised_signals_.clear();
} // ClearSignals

bool DebugPrintServerContext::PeekOneHartNonBlocking(
    int chip_id,
    const CoreCoord& core,
    int hart_id,
    bool new_data_this_iter
) {
    // compute the buffer address for the requested hart
    uint32_t base_addr = GetBaseAddr(chip_id, core, hart_id);

    // Device is incrementing wpos
    // Host is reading wpos and incrementing local rpos up to wpos
    // Device is filling the buffer and in the end waits on host to write rpos

    // TODO(AP) - compare 8-bytes transfer and full buffer transfer latency
    // First probe only 8 bytes to see if there's anything to read
    constexpr int eightbytes = 8;
    auto from_dev = tt::llrt::read_hex_vec_from_core(chip_id, core, base_addr, eightbytes);
    uint32_t wpos = from_dev[0], rpos = from_dev[1];
    uint32_t counter = 0;
    uint32_t sigval = 0;
    char val = 0;

    // If the print server is muted, dump the output to a null stream instead.
    ostream& stream = (mute_print_server_)? null_stream : *stream_;

    // Check whether this hart is currently waiting on a WAIT to be fulfilled.
    tuple<uint32_t, uint32_t, uint32_t> hart_key {core.x, core.y, hart_id};
    if (hart_waiting_on_signal_.count(hart_key) > 0) {
        // Check if the signal the hart is wairint for has been raised.
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
                string core_str = "core (" + to_string(core.x) + "," + to_string(core.y) +
                    ") riscv " + to_string(hart_id);
                string error_str = "DPRINT server timed out on " +
                    core_str +
                    ", waiting on a RAISE signal: " +
                    to_string(wait_signal) + "\n";
                stream << error_str << flush;
                log_warning(tt::LogMetal, "Debug Print Server encountered an error: {}", error_str);
                server_killed_due_to_hang_ = true;
                return false;
            }

            // Since it's still waiting, return false here since no data was read.
            return false;
        }
    }

    if (rpos < wpos) {
        // Now read the entire buffer
        from_dev = tt::llrt::read_hex_vec_from_core(chip_id, core, base_addr, PRINT_BUFFER_SIZE);
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
            uint8_t code = l->data[rpos++]; TT_ASSERT(rpos <= bufsize);
            uint8_t sz = l->data[rpos++]; TT_ASSERT(rpos <= bufsize);
            uint8_t* ptr = l->data + rpos;

            // Possible to break before rpos == wpos due to waiting on another core's raise.
            bool break_due_to_wait = false;

            // we are sharing the same output file between debug print threads for multiple cores
            switch(code) {
                // TODO(AP): better code index sync with debug_print.h
                case DEBUG_PRINT_TYPEID_CSTR: // const char*
                    // terminating zero was included in size and should be present in the buffer
                    cptr = reinterpret_cast<const char*>(ptr);
                    nlen = strnlen(cptr, 200);
                    if (nlen >= 200)
                        stream << "STRING BUFFER OVERFLOW DETECTED" << endl;
                    else
                        stream << cptr;
                    TT_ASSERT(sz == strlen(cptr)+1);
                break;
                case DEBUG_PRINT_TYPEID_TILESLICE:
                    PrintTileSlice(stream, ptr, hart_id);
                break;

                case DEBUG_PRINT_TYPEID_ENDL:
                    stream << endl;
                    TT_ASSERT(sz == 1);
                break;
                case DEBUG_PRINT_TYPEID_SETW:
                    val = CAST_U8P(ptr)[0];
                    stream << setw(val);
                    TT_ASSERT(sz == 1);
                break;
                case DEBUG_PRINT_TYPEID_SETPRECISION:
                    stream << std::setprecision(*ptr);
                    TT_ASSERT(sz == 1);
                break;
                case DEBUG_PRINT_TYPEID_FIXED:
                    stream << std::fixed;
                    TT_ASSERT(sz == 1);
                break;
                case DEBUG_PRINT_TYPEID_DEFAULTFLOAT:
                    stream << std::defaultfloat;
                    TT_ASSERT(sz == 1);
                break;
                case DEBUG_PRINT_TYPEID_HEX:
                    stream << std::hex;
                    TT_ASSERT(sz == 1);
                break;
                case DEBUG_PRINT_TYPEID_OCT:
                    stream << std::oct;
                    TT_ASSERT(sz == 1);
                break;
                case DEBUG_PRINT_TYPEID_DEC:
                    stream << std::dec;
                    TT_ASSERT(sz == 1);
                break;
                case DEBUG_PRINT_TYPEID_UINT32:
                    stream << *reinterpret_cast<uint32_t*>(ptr);
                    TT_ASSERT(sz == 4);
                break;
                case DEBUG_PRINT_TYPEID_FLOAT32:
                    stream << *reinterpret_cast<float*>(ptr);
                    TT_ASSERT(sz == 4);
                break;
                case DEBUG_PRINT_TYPEID_BFLOAT16:
                    stream << bfloat16_to_float(*reinterpret_cast<uint16_t*>(ptr));
                    TT_ASSERT(sz == 2);
                break;
                case DEBUG_PRINT_TYPEID_CHAR:
                    stream << *reinterpret_cast<char*>(ptr);
                    TT_ASSERT(sz == 1);
                break;
                case DEBUG_PRINT_TYPEID_RAISE:
                    sigval = *reinterpret_cast<uint32_t*>(ptr);
                    // Add this newly raised signals to the set of raised signals.
                    raised_signals_.insert(sigval);
                    //stream << "\nRaised signal=" << sigval << endl;
                    TT_ASSERT(sz == 4);
                break;
                case DEBUG_PRINT_TYPEID_WAIT:
                    {
                    sigval = *reinterpret_cast<uint32_t*>(ptr);
                    // Given that we break immediately on a wait, this core should never be waiting
                    // on multiple signals at the same time.
                    tuple<uint32_t, uint32_t, uint32_t> hart_key {core.x, core.y, hart_id};
                    TT_ASSERT(hart_waiting_on_signal_.count(hart_key) == 0);
                    // Set that this hart is waiting on this signal, and then stop reading for now.
                    hart_waiting_on_signal_[hart_key] = sigval;
                    break_due_to_wait = true;
                    //stream << "\nWaiting on signal=" << *reinterpret_cast<uint32_t*>(ptr);
                    TT_ASSERT(sz == 4);
                    }
                break;
                case DEBUG_PRINT_TYPEID_INT32:
                    stream << *reinterpret_cast<int32_t*>(ptr);
                    TT_ASSERT(sz == 4);
                break;
                case DEBUG_PRINT_TYPEID_UINT64:
                    stream << *reinterpret_cast<uint64_t*>(ptr);
                    TT_ASSERT(sz == 8);
                break;
                default:
                    TT_FATAL("Unexpected debug print type code" && false);
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
        tt::llrt::write_hex_vec_to_core(chip_id, core, rposbuf, base_addr+offs);

        // Return true to signal that some print data was read
        return true;
    } // if (rpos < wpos)

    // Return false to signal that no print data was ready this time.
    return false;
} // PeekOneHartNonBlocking

void DebugPrintServerContext::PollPrintData(uint32_t hart_mask) {
    // Give the print server thread a reasonable name.
    pthread_setname_np(pthread_self(), "TT_DPRINT_SERVER");

    // Main print loop, go through all chips/cores/harts on the device and poll for any print data
    // written.
    while (true) {
        if (stop_print_server_) {
            // If the stop signal was received, exit the print server thread, but wait for any
            // existing prints to be wrapped up first.
            if (hart_waiting_on_signal_.size() == 0 && !new_data_last_iter_)
                break;
        }

        // Make a copy of the device->core map, so that it can be modified while polling.
        std::map<Device*, vector<CoreCoord>> device_to_core_range_copy;
        device_to_core_range_lock_.lock();
        device_to_core_range_copy = device_to_core_range_;
        device_to_core_range_lock_.unlock();

        // Flag for whether any new print data was found in this round of polling.
        bool new_data_this_iter = false;
        for (auto& device_and_cores : device_to_core_range_copy) {
            chip_id_t chip_id = device_and_cores.first->id();
            for (auto core: device_and_cores.second) {
                int hart_count = GetNumRiscs(chip_id, core);
                for (int hart_index = 0; hart_index < hart_count; hart_index++) {
                    if (hart_mask & (1<<hart_index)) {
                        if (!CheckInitMagicCleared(chip_id, core, hart_index))
                            continue;

                        new_data_this_iter |= PeekOneHartNonBlocking(
                            chip_id,
                            core,
                            hart_index,
                            new_data_this_iter
                        );

                        // If this read detected a print hang, stop processing prints.
                        if (server_killed_due_to_hang_)
                            return;
                    }
                }
            }
        }

        // Signal whether the print server is currently processing data.
        new_data_last_iter_ = new_data_this_iter;
        // Sleep for a few ms if no data was processed.
        if (!new_data_last_iter_)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
} // PollPrintData

DebugPrintServerContext* DebugPrintServerContext::inst = nullptr;
bool DebugPrintServerContext::ProfilerIsRunning = false;

} // anon namespace

// Implementation for functions available from dprint_server.hpp.
namespace tt {

void DprintServerAttach(Device* device) {
    // Skip if DPRINT not enabled, and make sure profiler is not running.
    if (!tt::llrt::OptionsG.get_dprint_enabled())
        return;
    TT_FATAL(
       DebugPrintServerContext::ProfilerIsRunning == false,
       "Device side profiler is running, cannot start print server"
    );

    // Skip if RTOptions doesn't enable DPRINT for this device
    vector<chip_id_t> chip_ids = tt::llrt::OptionsG.get_dprint_chip_ids();
    if (!tt::llrt::OptionsG.get_dprint_all_chips())
        if (std::find(chip_ids.begin(), chip_ids.end(), device->id()) == chip_ids.end())
            return;

    // If no server ir running, create one
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
            TT_FATAL(false && "Timed out waiting on debug print server to read data.");
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
} // namespace tt
