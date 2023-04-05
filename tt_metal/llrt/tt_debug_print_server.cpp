#include <thread>
#include <future>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <set>
#include "llrt.hpp"
#include "common/logger.hpp"

#include "tt_debug_print_server.hpp"

#include "hostdevcommon/common_runtime_address_map.h"
#include "hostdevcommon/debug_print_common.h"

using std::uint32_t;
using std::int32_t;
using std::cout;
using std::endl;
using std::setw;
using std::flush;

namespace {

static inline float bfloat16_to_float(uint16_t bfloat_val) {
    uint32_t val = bfloat_val << 16;
    return *reinterpret_cast<float*>(&val);
}


// TODO(AP): this shouldn't be necessary as the API itself should be thread-safe but otherwise currently the test fails
// with DMA mode (non-MMIO version of api enabled via export TT_PCI_DMA_BUF_SIZE=1048576)
std::vector<std::uint32_t> my_read_hex_vec_from_core(tt_cluster *cluster, int chip, const tt_xy_pair& core, uint64_t addr, uint32_t size) {
    static std::mutex r_lock;
    r_lock.lock();
    auto result = tt::llrt::read_hex_vec_from_core(cluster, chip, core, addr, size);
    r_lock.unlock();
    return result;
}

// TODO(AP): this shouldn't be necessary as the API itself should be thread-safe
void my_write_hex_vec_to_core(tt_cluster *cluster, int chip, const tt_xy_pair& core, std::vector<uint32_t> hex_vec, uint64_t addr) {
    static std::mutex w_lock;
    w_lock.lock();
    tt::llrt::write_hex_vec_to_core(cluster, chip, core, hex_vec, addr);
    w_lock.unlock();
}

// Writes a magic value at wpos ptr address for dprint buffer for a specific hart/core/chip
// Used for debug print server startup sequence.
void write_init_magic(tt_cluster* cluster, int chip_id, const tt_xy_pair& core, int hart_id, bool starting = true) {
    // compute the buffer address for the requested hart
    uint32_t base_addr = PRINT_BUFFER_NC + hart_id*PRINT_BUFFER_SIZE;

    // TODO(AP): this could use a cleanup - need a different mechanism to know if a kernel is running on device.
    // Force wait for first kernel launch by first writing a non-zero and waiting for a zero.
    vector<uint32_t> initbuf = { uint32_t(starting ? DEBUG_PRINT_SERVER_STARTING_MAGIC : DEBUG_PRINT_SERVER_DISABLED_MAGIC) };
    my_write_hex_vec_to_core(cluster, chip_id, core, initbuf, base_addr);
} // write_init_magic


struct DebugPrintServerContext {

    // only one instance is allowed at the moment
    static DebugPrintServerContext* inst;

    DebugPrintServerContext(
        tt_cluster* cluster, vector<int> chip_ids, const vector<tt_xy_pair>& cores, uint32_t hart_mask, const char* filename
    ) {
        TT_ASSERT(inst == nullptr);
        inst = this;

        cluster_ = cluster;
        chip_ids_ = chip_ids;
        cores_ = cores;
        hart_mask_ = hart_mask;

        // the stream is shared between threads
        if (filename != nullptr) {
            outfile_ = new std::ofstream(filename);
        }
        stream_ = outfile_ ? outfile_ : &cout;

        exit_threads_condition_ = false;
        for (auto chip: chip_ids) {
            for (auto core: cores) {
                for (int hart_index = 0; hart_index < DPRINT_NHARTS; hart_index++) {
                    if (hart_mask & (1<<hart_index)) {
                        // Cannot do this magic write inside the thread because of a possible race condition
                        // where the kernel subsequently both launches and terminates prior to magic write going through
                        write_init_magic(cluster_, chip, core, hart_index);

                        auto print_thread = new std::thread(
                            [this, chip, core, hart_index] { thread_poll(chip, core, hart_index); }
                        );
                        print_threads_.push_back(print_thread);
                    }
                }
            }
        }
    }

    ~DebugPrintServerContext() {
        exit_threads_condition_ = true;
        for (int i = 0; i < print_threads_.size(); i++) {
            auto future = std::async(std::launch::async, &std::thread::join, print_threads_[i]);
            if (future.wait_for(std::chrono::seconds(2)) == std::future_status::timeout) {
                TT_ASSERT(false && "Timed out waiting on debug print thread to terminate.");
            }
            delete print_threads_[i];
            print_threads_[i] = nullptr;
        }
        print_threads_.clear();

        if (outfile_) {
            outfile_->close();
            delete outfile_;
        }
        inst = nullptr;
    }

private:

    std::atomic<bool> exit_threads_condition_;
    std::vector<std::thread*> print_threads_;
    std::unordered_map<uint32_t, int> raised_signals_; // signalID -> count
    std::mutex signals_lock_;

    // we are sharing the same output file/cout between threads for multiple cores
    // so we need a lock for it since it may not be cout
    std::mutex output_lock_;
    std::ofstream* outfile_ = nullptr; // non-cout
    std::ostream* stream_ = nullptr; // either == outfile_ or is &cout

    // configuration of cores/harts to listen for
    tt_cluster* cluster_;
    vector<int> chip_ids_;
    vector<tt_xy_pair> cores_;
    uint32_t hart_mask_;

    void thread_poll(int chip_id, tt_xy_pair core, int hart_index);
    void peek_flush_one_hart_nonblocking(int chip_id, const tt_xy_pair& core, int hart_index);

    void lock_stream() {
        if (outfile_ != &cout)
            output_lock_.lock();
    }
    void unlock_stream() {
        if (outfile_ != &cout)
            output_lock_.unlock();
    }

    void raise_signal(uint32_t signal_code) {
        signals_lock_.lock();
        raised_signals_[signal_code] = 1;
        signals_lock_.unlock();
    }

    bool check_clear_signal(uint32_t signal_code) {
        signals_lock_.lock();
        bool is_raised = false;
        if (raised_signals_.find(signal_code) != raised_signals_.end()) {
            int& val = raised_signals_[signal_code];
            is_raised = (val > 0);
            val = 0;
        }
        signals_lock_.unlock();
        return is_raised;
    }
};

// Checks if our magic value was cleared by the device code
// The assumption is that if our magic number was cleared,
// it means there is a write in the queue and wpos/rpos are now valid
// Note that this is not a bulletproof way to bootstrap the print server (TODO(AP))
bool check_init_magic_cleared(tt_cluster* cluster, int chip_id, const tt_xy_pair& core, int hart_id) {
    // compute the buffer address for the requested hart
    uint32_t base_addr = PRINT_BUFFER_NC + hart_id*PRINT_BUFFER_SIZE;

    vector<uint32_t> initbuf = { DEBUG_PRINT_SERVER_STARTING_MAGIC };
    auto result = my_read_hex_vec_from_core(cluster, chip_id, core, base_addr, 4);
    return (result[0] != initbuf[0]);
} // check_init_magic_cleared


// rename the current thread so it's easier to distinguish in the debugger
// TODO(AP): this renaming didn't show up in VSCODE thread list
void rename_my_thread(int chip_id, const tt_xy_pair& core, int hart_id)
{
    std::string rn("DPRINT_C,X,Y,T{");
    rn += std::to_string(chip_id);
    rn += ",";
    rn += std::to_string(core.x);
    rn += ",";
    rn += std::to_string(core.y);
    rn += ",";
    rn += std::to_string(hart_id);
    rn += "}";
    pthread_setname_np(pthread_self(), rn.c_str());
}

// peeks a specified hart for any debug prints present in the buffer and flushes it, printing the contents out to host-side stream
void DebugPrintServerContext::peek_flush_one_hart_nonblocking(int chip_id, const tt_xy_pair& core, int hart_id) {
    // compute the buffer address for the requested hart
    uint32_t base_addr = PRINT_BUFFER_NC + hart_id*PRINT_BUFFER_SIZE;

    // Device is incrementing wpos
    // Host is reading wpos and incrementing local rpos up to wpos
    // Device is filling the buffer and in the end waits on host to write rpos

    // TODO(AP) - compare 8-bytes transfer and full buffer transfer latency
    // First probe only 8 bytes to see if there's anything to read
    constexpr int eightbytes = 8;
    auto from_dev = my_read_hex_vec_from_core(cluster_, chip_id, core, base_addr, eightbytes);
    uint32_t wpos = from_dev[0], rpos = from_dev[1];
    uint32_t counter = 0;
    uint32_t sigval = 0;
    ostream& stream = *stream_;
    if (rpos < wpos) {
        // Now read the entire buffer
        from_dev = my_read_hex_vec_from_core(cluster_, chip_id, core, base_addr, PRINT_BUFFER_SIZE);
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
        while (rpos < wpos) {
            uint8_t code = l->data[rpos++]; TT_ASSERT(rpos <= bufsize);
            uint8_t sz = l->data[rpos++]; TT_ASSERT(rpos <= bufsize);
            uint8_t* ptr = l->data + rpos;

            TileSamplesHostDev<8>* ts8 = nullptr; // need this decl ouside of switch :(
            TileSamplesHostDev<32>* ts32 = nullptr; // TODO(AP): unify template params
            // we are sharing the same output file between debug print threads for multiple cores
            switch(code) {
                // TODO(AP): better code index sync with debug_print.h
                case DEBUG_PRINT_TYPEID_CSTR: // const char*
                    // terminating zero was included in size and should be present in the buffer
                    cptr = reinterpret_cast<const char*>(ptr);
                    nlen = strnlen(cptr, 200);
                    lock_stream();
                    if (nlen >= 200)
                        stream << "STRING BUFFER OVERFLOW DETECTED" << endl;
                    else
                        stream << cptr;
                    unlock_stream();
                    TT_ASSERT(sz == strlen(cptr)+1);
                break;
                case DEBUG_PRINT_TYPEID_TILESAMPLES8:
                    lock_stream();
                    ts8 = reinterpret_cast< TileSamplesHostDev<8>* >(ptr);
                    stream << "TILE: (";
                    for (int j = 0; j < ts8->count_; j++) {
                        stream << bfloat16_to_float(reinterpret_cast<uint16_t*>(ptr)[j]);
                        stream << " ";
                    }
                    stream << "ptr=" << ts8->ptr_ << ")" << endl;
                    unlock_stream();
                break;
                case DEBUG_PRINT_TYPEID_TILESAMPLES32:
                    lock_stream();
                    ts32 = reinterpret_cast< TileSamplesHostDev<32>* >(ptr);
                    stream << "TILE: (";
                    for (int j = 0; j < ts32->count_; j++) {
                        stream << bfloat16_to_float(reinterpret_cast<uint16_t*>(ptr)[j]);
                        stream << " ";
                    }
                    stream << "ptr=" << ts32->ptr_ << ")" << endl;
                    unlock_stream();
                break;

                case DEBUG_PRINT_TYPEID_ENDL:
                    lock_stream();
                    stream << endl;
                    unlock_stream();
                    TT_ASSERT(sz == 1);
                break;
                case DEBUG_PRINT_TYPEID_SETW:
                    lock_stream();
                    stream << setw(*ptr);
                    unlock_stream();
                    TT_ASSERT(sz == 1);
                break;
                case DEBUG_PRINT_TYPEID_SETP:
                    lock_stream();
                    stream << std::setprecision(*ptr);
                    unlock_stream();
                    TT_ASSERT(sz == 1);
                break;
                case DEBUG_PRINT_TYPEID_FIXP:
                    lock_stream();
                    stream << std::fixed;
                    unlock_stream();
                    TT_ASSERT(sz == 1);
                break;
                case DEBUG_PRINT_TYPEID_HEX:
                    lock_stream();
                    stream << std::hex;
                    unlock_stream();
                    TT_ASSERT(sz == 1);
                break;

                case DEBUG_PRINT_TYPEID_UINT32:
                    lock_stream();
                    stream << *reinterpret_cast<uint32_t*>(ptr);
                    unlock_stream();
                    TT_ASSERT(sz == 4);
                break;
                case DEBUG_PRINT_TYPEID_FLOAT32:
                    lock_stream();
                    stream << *reinterpret_cast<float*>(ptr);
                    unlock_stream();
                    TT_ASSERT(sz == 4);
                break;
                case DEBUG_PRINT_TYPEID_FLOAT16:
                    lock_stream();
                    stream << bfloat16_to_float(*reinterpret_cast<uint16_t*>(ptr));
                    unlock_stream();
                    TT_ASSERT(sz == 2);
                break;
                case DEBUG_PRINT_TYPEID_CHAR:
                    lock_stream();
                    stream << *reinterpret_cast<char*>(ptr);
                    unlock_stream();
                    TT_ASSERT(sz == 1);
                break;
                case DEBUG_PRINT_TYPEID_RAISE:
                    sigval = *reinterpret_cast<uint32_t*>(ptr);
                    //stream << "\nRaised signal=" << sigval << endl;
                    raise_signal(sigval);
                    TT_ASSERT(sz == 4);
                break;
                case DEBUG_PRINT_TYPEID_WAIT:
                    sigval = *reinterpret_cast<uint32_t*>(ptr);
                    //stream << "\nWaiting on signal=" << *reinterpret_cast<uint32_t*>(ptr);
                    counter = 0;
                    while (!check_clear_signal(sigval)) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(1)); // sleep for a few ms
                        counter ++;
                        if (counter == 20000) {
                            lock_stream();
                            stream << "\n*** Timed out waiting on signal " << sigval << " ***" << endl;
                            unlock_stream();
                            break;
                        }
                    }
                    //cout << "SIG=" << *reinterpret_cast<uint32_t*>(ptr) << " " << std::flush;

                    TT_ASSERT(sz == 4);
                break;
                case DEBUG_PRINT_TYPEID_INT32:
                    lock_stream();
                    stream << *reinterpret_cast<int32_t*>(ptr);
                    unlock_stream();
                    TT_ASSERT(sz == 4);
                break;
                default:
                    TT_ASSERT("Unexpected debug print type code" && false);
            }

            // TODO(AP): this is slow but leaving here for now for debugging the debug prints themselves
            lock_stream();
            stream << flush;
            unlock_stream();

            rpos += sz; // parse the payload size
            TT_ASSERT(rpos <= wpos);
        } // while (rpos < wpos)

        // writes by the producer should've been atomic w.r.t code+size+payload
        // i.e at this point we shouldn't have piecemeal reads on code+size+payload
        // with rpos not aligned to wpos
        TT_ASSERT(rpos == wpos);

        // write back to device - update rpos only
        vector<uint32_t> rposbuf;
        rposbuf.push_back(rpos);
        uint32_t offs = DebugPrintMemLayout().rpos_offs();
        my_write_hex_vec_to_core(cluster_, chip_id, core, rposbuf, base_addr+offs);
    } // if (rpos < wpos)
} // peek_one_hart_once_nonblocking

// TODO(AP): investigate if we can reduce the number of threads using coroutines
void DebugPrintServerContext::thread_poll(
    int chip_id, tt_xy_pair core, int hart_index) {

    rename_my_thread(chip_id, core, hart_index);

    // Main print loop - poll all the harts on the device as specified by mask for any data written
    while (true) {
        if (exit_threads_condition_ != false)
            break;

        std::this_thread::sleep_for(std::chrono::milliseconds(1)); // sleep for a few ms

        if (!check_init_magic_cleared(cluster_, chip_id, core, hart_index))
            continue;

        peek_flush_one_hart_nonblocking(chip_id, core, hart_index);
    }
}

DebugPrintServerContext* DebugPrintServerContext::inst = nullptr;

} // anon namespace

void tt_stop_debug_print_server(tt_cluster*)
{
    // this can be called multiple times since we register it with atexit to make explicit call to stop optional
    if (DebugPrintServerContext::inst != nullptr) {
        delete DebugPrintServerContext::inst;
        DebugPrintServerContext::inst = nullptr;
    }
}

void tt_stop_debug_print_server_per_device(tt_cluster* cluster, int)
{
    tt_stop_debug_print_server(cluster);
}

// The print server is not valid without alive tt_cluster and tt_device
void tt_start_debug_print_server(
    tt_cluster* cluster, const vector<int>& chip_ids, const vector<tt_xy_pair>& cores, uint32_t hart_mask, const char* filename)
{
    TT_ASSERT(DebugPrintServerContext::inst == nullptr);

    DebugPrintServerContext* ctx = new DebugPrintServerContext(cluster, chip_ids, cores, hart_mask, filename);

    // currently there's only one device per cluster
    // in some tests close_device is not called but we still need to flush the debug prints
    // in other tests, the device is destroyed before the cluster is destroyed.
    // so by the time we get to cluster destructor the device is destroyed and in between the state is invalid
    // so we add the callback to both paths since it doesn't hurt to call it twice
    cluster->on_close_device(tt_stop_debug_print_server_per_device);
    cluster->on_destroy(tt_stop_debug_print_server);
}
