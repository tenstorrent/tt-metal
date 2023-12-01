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
using std::cout;
using std::endl;
using std::setw;
using std::flush;
using std::tuple;
using std::set;

#define CAST_U8P(p) reinterpret_cast<uint8_t*>(p)

namespace {

static inline float bfloat16_to_float(uint16_t bfloat_val) {
    uint32_t val = bfloat_val << 16;
    return *reinterpret_cast<float*>(&val);
}

// A null stream for when the print server is muted.
class NullBuffer : public std::streambuf {
public:
   int overflow(int c) { return c; }
};
NullBuffer null_buffer;
std::ostream null_stream(&null_buffer);

// Writes a magic value at wpos ptr address for dprint buffer for a specific hart/core/chip
// Used for debug print server startup sequence.
void write_init_magic(int chip_id, const CoreCoord& core, int hart_id, bool starting = true) {
    // compute the buffer address for the requested hart
    uint32_t base_addr = PRINT_BUFFER_NC + hart_id*PRINT_BUFFER_SIZE;

    // TODO(AP): this could use a cleanup - need a different mechanism to know if a kernel is running on device.
    // Force wait for first kernel launch by first writing a non-zero and waiting for a zero.
    vector<uint32_t> initbuf = { uint32_t(starting ? DEBUG_PRINT_SERVER_STARTING_MAGIC : DEBUG_PRINT_SERVER_DISABLED_MAGIC) };
    tt::llrt::write_hex_vec_to_core(chip_id, core, initbuf, base_addr);
} // write_init_magic


struct DebugPrintServerContext {

    // only one instance is allowed at the moment
    static DebugPrintServerContext* inst;
    static bool ProfilerIsRunning;

    DebugPrintServerContext(
        vector<int> chip_ids,
        const vector<CoreCoord>& cores,
        uint32_t hart_mask,
        string file_name
    ) {
        TT_ASSERT(inst == nullptr);
        inst = this;

        // the stream is shared between threads
        if (file_name != "") {
            outfile_ = new std::ofstream(file_name);
        }
        stream_ = outfile_ ? outfile_ : &cout;

        stop_print_server_ = false;
        mute_print_server_ = false;
        new_data_processed_ = false;
        print_server_thread_ = new std::thread(
            [this, chip_ids, cores, hart_mask] { thread_poll(chip_ids, cores, hart_mask); }
        );
    }

    ~DebugPrintServerContext() {
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
    }

    void SetMute(bool mute_print_server) { mute_print_server_ = mute_print_server; }

    void WaitForPrintsFinished() {
        // Simply poll the flag every few ms to check whether new data is still being processed,
        // or whether any cores are waiting for a signal to be raised.
        // TODO(dma): once we have access to the device is there a way we can poll the device to
        // check whether more print data is coming?
        do {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        } while (hart_waiting_on_signal_.size() > 0 || new_data_processed_);
    }

private:

    // Flag for main thread to signal the print server thread to stop.
    std::atomic<bool> stop_print_server_;
    // Flag for muting the print server. This doesn't disable reading print data from the device,
    // but it supresses the output of that print data the user.
    std::atomic<bool> mute_print_server_;
    // Flag for signalling whether the print server thread has recently processed data (and is
    // therefore likely to continue processing data in the next round of polling).
    std::atomic<bool> new_data_processed_;
    std::thread* print_server_thread_;

    std::ofstream* outfile_ = nullptr; // non-cout
    std::ostream* stream_ = nullptr; // either == outfile_ or is &cout

    // A map to from {core coord x, y, hart index} to the signal code it's waiting for.
    std::map<tuple<uint32_t, uint32_t, uint32_t>, uint32_t> hart_waiting_on_signal_;
    // Keep a separate set of raised signal codes so that multiple harts can wait for the same
    // signal.
    std::set<uint32_t> raised_signals_;

    void thread_poll(const vector<int>& chip_ids, const vector<CoreCoord>& cores, uint32_t hart_mask);
    bool peek_flush_one_hart_nonblocking(int chip_id, const CoreCoord& core, int hart_index);
};

static void print_tile_slice(ostream& stream, uint8_t* ptr) {
    TileSliceHostDev<0>* ts = reinterpret_cast<TileSliceHostDev<0>*>(ptr);
    stream << "TILE: (" << endl << std::flush;
    if (ts->w0_ == 0xFFFF) {
        stream << "BAD TILE POINTER" << std::flush;
        stream << " count=" << ts->count_ << std::flush;
    } else {
        uint32_t i = 0;
        for (int h = ts->h0_; h < ts->h1_; h += ts->hs_) {
            if (ts->w0_ < ts->w1_)
                stream << "  ";
            for (int w = ts->w0_; w < ts->w1_; w += ts->ws_) {
                if (i >= ts->count_)
                    goto done;
                stream << bfloat16_to_float(ts->samples_[i]);
                if (w + ts->ws_ < ts->w1_)
                    stream << " ";
                i++;
            }
            if (ts->endl_rows_)
                stream << endl;
        }
    }
done:
    stream << endl << "  ptr=" << ts->ptr_ << ")" << endl;
}

// Checks if our magic value was cleared by the device code
// The assumption is that if our magic number was cleared,
// it means there is a write in the queue and wpos/rpos are now valid
// Note that this is not a bulletproof way to bootstrap the print server (TODO(AP))
bool check_init_magic_cleared(int chip_id, const CoreCoord& core, int hart_id) {
    // compute the buffer address for the requested hart
    uint32_t base_addr = PRINT_BUFFER_NC + hart_id*PRINT_BUFFER_SIZE;

    vector<uint32_t> initbuf = { DEBUG_PRINT_SERVER_STARTING_MAGIC };
    auto result = tt::llrt::read_hex_vec_from_core(chip_id, core, base_addr, 4);
    return (result[0] != initbuf[0]);
} // check_init_magic_cleared

// Peeks a specified hart for any debug prints present in the buffer and flushes it, printing the
// contents out to host-side stream. Returns true if some data was read out, and false if no new
// print data was present on the device.
bool DebugPrintServerContext::peek_flush_one_hart_nonblocking(int chip_id, const CoreCoord& core, int hart_id) {
    // compute the buffer address for the requested hart
    uint32_t base_addr = PRINT_BUFFER_NC + hart_id*PRINT_BUFFER_SIZE;

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
    ostream& stream = (mute_print_server_)? null_stream : *stream_;
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
                    print_tile_slice(stream, ptr);
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
} // peek_one_hart_once_nonblocking

void DebugPrintServerContext::thread_poll(
    const vector<int>& chip_ids,
    const vector<CoreCoord>& cores,
    uint32_t hart_mask
) {
    // Give the print server thread a reasonable name.
    pthread_setname_np(pthread_self(), "TT_DPRINT_SERVER");

    // First, go through all cores and write init magic to set up dprint.
    for (auto chip: chip_ids) {
        for (auto core: cores) {
            for (int hart_index = 0; hart_index < DPRINT_NRISCVS; hart_index++) {
                if (hart_mask & (1<<hart_index)) {
                    write_init_magic(chip, core, hart_index);
                }
            }
        }
    }

    // Main print loop, go through all chips/cores/harts on the device and poll for any print data
    // written.
    while (true) {
        if (stop_print_server_) {
            // If the stop signal was received, exit the print server thread, but wait for any
            // existing prints to be wrapped up first.
            if (hart_waiting_on_signal_.size() == 0 && !new_data_processed_)
                break;
        }

        // Flag for whether any new print data was found in this round of polling.
        bool new_print_data = false;
        for (auto chip: chip_ids) {
            for (auto core: cores) {
                for (int hart_index = 0; hart_index < DPRINT_NRISCVS; hart_index++) {
                    if (hart_mask & (1<<hart_index)) {
                        if (!check_init_magic_cleared(chip, core, hart_index))
                            continue;

                        // Make sure that this core is not waiting on a raise signal to continue
                        // printing.
                        tuple<uint32_t, uint32_t, uint32_t> hart_key {core.x, core.y, hart_index};
                        if (hart_waiting_on_signal_.count(hart_key) > 0) {
                            uint32_t wait_signal = hart_waiting_on_signal_[hart_key];
                            if (raised_signals_.count(wait_signal) > 0) {
                                // The signal this hart is waiting for has been raised, it's not
                                // waiting anymore.
                                hart_waiting_on_signal_.erase(hart_key);
                            } else {
                                // Not raised yet, keep waiting.
                                continue;
                            }
                        }

                        new_print_data |= peek_flush_one_hart_nonblocking(chip, core, hart_index);
                    }
                }
            }
        }

        // Signal whether the print server is currently processing data.
        new_data_processed_ = new_print_data;
        // Sleep for a few ms if no data was processed.
        if (!new_print_data)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

DebugPrintServerContext* DebugPrintServerContext::inst = nullptr;
bool DebugPrintServerContext::ProfilerIsRunning = false;

} // anon namespace

void tt_stop_debug_print_server()
{
    // this can be called multiple times since we register it with atexit to make explicit call to stop optional
    if (DebugPrintServerContext::inst != nullptr) {
        delete DebugPrintServerContext::inst;
        DebugPrintServerContext::inst = nullptr;
    }
}

void tt_set_profiler_state_for_debug_print(bool profile_device)
{
    DebugPrintServerContext::ProfilerIsRunning = profile_device;
}

bool tt_is_print_server_running()
{
    return DebugPrintServerContext::inst != nullptr;
}

void tt_set_debug_print_server_mute(bool mute_print_server) {
    if (DebugPrintServerContext::inst != nullptr)
        DebugPrintServerContext::inst->SetMute(mute_print_server);
}

void tt_await_debug_print_server() {
    if (DebugPrintServerContext::inst != nullptr) {
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

// The print server is not valid without alive Cluster and tt_device
void tt_start_debug_print_server(
    std::function<CoreCoord ()>get_grid_size,
    std::function<CoreCoord (CoreCoord)>worker_from_logical
) {
    if (tt::llrt::OptionsG.get_dprint_enabled()) {
        TT_FATAL(DebugPrintServerContext::inst == nullptr, "Multiple print servers not allowed");
        TT_FATAL(DebugPrintServerContext::ProfilerIsRunning == false, "Device side profiler is running, cannot start print server");

        tt::Cluster::instance().reset_debug_print_server_buffers();

        // A set of all valid worker cores, used for checking the user input.
        auto compare_coords = [](const CoreCoord& a, const CoreCoord& b){
            if (a.x < b.x)
                return true;
            else if (a.x == b.x)
                return (a.y < b.y);
            else
                return false;
        };
        set<CoreCoord, decltype(compare_coords)> all_worker_cores(compare_coords);
        CoreCoord logical_grid_size = get_grid_size();
        for (uint32_t x = 0; x < logical_grid_size.x; x++) {
            for (uint32_t y = 0; y < logical_grid_size.y; y++) {
                CoreCoord logical_coord(x, y);
                CoreCoord worker_core = worker_from_logical(logical_coord);
                all_worker_cores.insert(worker_core);
            }
        }

        // Core range depends on whether dprint_all_cores flag is set.
        vector<CoreCoord> print_cores_sanitized;
        if (tt::llrt::OptionsG.get_dprint_all_cores()) {
            // Print from all worker cores, cores returned here are guaranteed to be valid.
            print_cores_sanitized = vector<CoreCoord>(all_worker_cores.begin(), all_worker_cores.end());
        } else {
            // Only print from the cores specified by the user.
            vector<CoreCoord> print_cores = tt::llrt::OptionsG.get_dprint_cores();

            // We should also validate that the cores the user specified are valid worker cores.
            for (auto core : print_cores) {
                if (all_worker_cores.count(core) > 0) {
                    print_cores_sanitized.push_back(core);
                } else {
                    log_info(
                        tt::LogDevice,
                        "TT_METAL_DPRINT_CORES included worker core ({}, {}), which is not a valid coordinate. This coordinate will be ignored by the dprint server.",
                        core.x,
                        core.y
                    );
                }
            }
        }

        DebugPrintServerContext* ctx = new DebugPrintServerContext(
            tt::llrt::OptionsG.get_dprint_chip_ids(),
            print_cores_sanitized,
            tt::llrt::OptionsG.get_dprint_riscv_mask(),
            tt::llrt::OptionsG.get_dprint_file_name()
        );
    }
}
