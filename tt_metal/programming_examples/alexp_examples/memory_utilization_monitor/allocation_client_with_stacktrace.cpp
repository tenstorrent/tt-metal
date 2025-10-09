// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Enhanced allocation client with stack trace capture for debugging
// Shows where each allocation originates from in the code

#include <iostream>
#include <fstream>
#include <sstream>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <iomanip>
#include <atomic>

// For stack traces
#include <execinfo.h>
#include <cxxabi.h>

#define TT_ALLOC_SERVER_SOCKET "/tmp/tt_allocation_server.sock"
#define STACK_TRACE_FILE "/tmp/tt_alloc_stack_traces.log"

struct __attribute__((packed)) AllocMessage {
    enum Type : uint8_t { ALLOC = 1, FREE = 2, QUERY = 3, RESPONSE = 4 };

    Type type;
    uint8_t pad1[3];
    int32_t device_id;
    uint64_t size;
    uint8_t buffer_type;
    uint8_t pad2[3];
    int32_t process_id;
    uint64_t buffer_id;
    uint64_t timestamp;

    uint64_t dram_allocated;
    uint64_t l1_allocated;
    uint64_t l1_small_allocated;
    uint64_t trace_allocated;
};

class AllocationClientWithStackTrace {
private:
    int socket_fd_;
    bool enabled_;
    bool connected_;
    std::atomic<uint64_t> alloc_counter_{0};
    std::ofstream stack_log_;

    // Capture stack trace
    std::string get_stack_trace(int skip_frames = 2) {
        const int max_frames = 64;
        void* callstack[max_frames];
        int frames = backtrace(callstack, max_frames);
        char** symbols = backtrace_symbols(callstack, frames);

        std::stringstream ss;

        // Skip first few frames (this function, wrapper functions)
        for (int i = skip_frames; i < frames && i < skip_frames + 15; i++) {
            // Try to demangle C++ names
            char* mangled_name = nullptr;
            char* offset_begin = nullptr;
            char* offset_end = nullptr;

            // Find the parentheses and +address offset
            for (char* p = symbols[i]; *p; ++p) {
                if (*p == '(') {
                    mangled_name = p;
                } else if (*p == '+') {
                    offset_begin = p;
                } else if (*p == ')') {
                    offset_end = p;
                    break;
                }
            }

            if (mangled_name && offset_begin && offset_end && mangled_name < offset_begin) {
                *mangled_name++ = '\0';
                *offset_begin++ = '\0';
                *offset_end = '\0';

                int status;
                char* demangled = abi::__cxa_demangle(mangled_name, nullptr, nullptr, &status);

                if (status == 0 && demangled) {
                    ss << "  " << symbols[i] << " : " << demangled << "+" << offset_begin << "\n";
                    free(demangled);
                } else {
                    ss << "  " << symbols[i] << " : " << mangled_name << "+" << offset_begin << "\n";
                }
            } else {
                // Couldn't parse, just print the whole line
                ss << "  " << symbols[i] << "\n";
            }
        }

        free(symbols);
        return ss.str();
    }

    bool connect_to_server() {
        if (connected_) {
            return true;
        }

        socket_fd_ = socket(AF_UNIX, SOCK_STREAM, 0);
        if (socket_fd_ < 0) {
            return false;
        }

        struct sockaddr_un addr;
        memset(&addr, 0, sizeof(addr));
        addr.sun_family = AF_UNIX;
        strncpy(addr.sun_path, TT_ALLOC_SERVER_SOCKET, sizeof(addr.sun_path) - 1);

        if (connect(socket_fd_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            close(socket_fd_);
            socket_fd_ = -1;

            static bool warned = false;
            if (!warned) {
                std::cerr << "[AllocationClient] Server not available at " << TT_ALLOC_SERVER_SOCKET << std::endl;
                warned = true;
            }
            return false;
        }

        connected_ = true;
        std::cout << "[AllocationClient] Connected to tracking server" << std::endl;
        return true;
    }

    void log_allocation_with_trace(int device_id, uint64_t size, uint8_t buffer_type, uint64_t buffer_id) {
        uint64_t alloc_id = alloc_counter_++;

        if (!stack_log_.is_open()) {
            stack_log_.open(STACK_TRACE_FILE, std::ios::app);
            stack_log_ << "=== TT-Metal Buffer Allocation Stack Traces ===" << std::endl;
            stack_log_ << "PID: " << getpid() << std::endl << std::endl;
        }

        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);

        const char* type_names[] = {"DRAM", "L1", "SYSTEM_MEMORY", "L1_SMALL", "TRACE"};
        const char* type_str = (buffer_type <= 4) ? type_names[buffer_type] : "UNKNOWN";

        stack_log_ << "─────────────────────────────────────────────────────" << std::endl;
        stack_log_ << "Allocation #" << alloc_id << std::endl;
        stack_log_ << "Time: " << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S") << std::endl;
        stack_log_ << "Device: " << device_id << std::endl;
        stack_log_ << "Type: " << type_str << std::endl;
        stack_log_ << "Size: " << size << " bytes (" << (size / 1024.0) << " KB)" << std::endl;
        stack_log_ << "Address: 0x" << std::hex << buffer_id << std::dec << std::endl;
        stack_log_ << "Stack trace:" << std::endl;
        stack_log_ << get_stack_trace(3);  // Skip 3 frames (this function + wrappers)
        stack_log_ << std::endl;
        stack_log_.flush();
    }

    void log_deallocation_with_trace(int device_id, uint64_t buffer_id) {
        if (!stack_log_.is_open()) {
            stack_log_.open(STACK_TRACE_FILE, std::ios::app);
        }

        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);

        stack_log_ << "─────────────────────────────────────────────────────" << std::endl;
        stack_log_ << "Deallocation" << std::endl;
        stack_log_ << "Time: " << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S") << std::endl;
        stack_log_ << "Device: " << device_id << std::endl;
        stack_log_ << "Address: 0x" << std::hex << buffer_id << std::dec << std::endl;
        stack_log_ << "Stack trace:" << std::endl;
        stack_log_ << get_stack_trace(3);
        stack_log_ << std::endl;
        stack_log_.flush();
    }

public:
    AllocationClientWithStackTrace() : socket_fd_(-1), enabled_(false), connected_(false) {
        const char* env_enabled = std::getenv("TT_ALLOC_TRACKING_ENABLED");
        if (env_enabled && std::string(env_enabled) == "1") {
            enabled_ = true;
            connect_to_server();
        }
    }

    ~AllocationClientWithStackTrace() {
        if (socket_fd_ >= 0) {
            close(socket_fd_);
        }
        if (stack_log_.is_open()) {
            stack_log_.close();
        }
    }

    static AllocationClientWithStackTrace& instance() {
        static AllocationClientWithStackTrace inst;
        return inst;
    }

    static void report_allocation(int device_id, uint64_t size, uint8_t buffer_type, uint64_t buffer_id) {
        auto& inst = instance();
        if (!inst.enabled_) {
            return;
        }

        // Log stack trace for debugging
        const char* detailed = std::getenv("TT_ALLOC_STACK_TRACE");
        if (detailed && std::string(detailed) == "1") {
            inst.log_allocation_with_trace(device_id, size, buffer_type, buffer_id);
        }

        // Send to server
        if (!inst.connect_to_server()) {
            return;
        }

        AllocMessage msg;
        memset(&msg, 0, sizeof(msg));
        msg.type = AllocMessage::ALLOC;
        msg.device_id = device_id;
        msg.size = size;
        msg.buffer_type = buffer_type;
        msg.process_id = getpid();
        msg.buffer_id = buffer_id;
        msg.timestamp = std::chrono::system_clock::now().time_since_epoch().count();

        ssize_t sent = send(inst.socket_fd_, &msg, sizeof(msg), MSG_DONTWAIT);
        if (sent < 0) {
            inst.connected_ = false;
        }
    }

    static void report_deallocation(int device_id, uint64_t buffer_id) {
        auto& inst = instance();
        if (!inst.enabled_) {
            return;
        }

        // Log stack trace for debugging
        const char* detailed = std::getenv("TT_ALLOC_STACK_TRACE");
        if (detailed && std::string(detailed) == "1") {
            inst.log_deallocation_with_trace(device_id, buffer_id);
        }

        // Send to server
        if (!inst.connect_to_server()) {
            return;
        }

        AllocMessage msg;
        memset(&msg, 0, sizeof(msg));
        msg.type = AllocMessage::FREE;
        msg.device_id = device_id;
        msg.buffer_id = buffer_id;
        msg.process_id = getpid();

        ssize_t sent = send(inst.socket_fd_, &msg, sizeof(msg), MSG_DONTWAIT);
        if (sent < 0) {
            inst.connected_ = false;
        }
    }

    static bool is_enabled() { return instance().enabled_; }
};

// Example usage
int main() {
    std::cout << "Example: Buffer Allocation with Stack Trace Capture\n" << std::endl;
    std::cout << "Usage:" << std::endl;
    std::cout << "  export TT_ALLOC_TRACKING_ENABLED=1" << std::endl;
    std::cout << "  export TT_ALLOC_STACK_TRACE=1" << std::endl;
    std::cout << "  ./allocation_client_with_stacktrace" << std::endl;
    std::cout << "\nStack traces will be written to: " << STACK_TRACE_FILE << std::endl;
    std::cout << "\nSimulating allocations..." << std::endl;

    // Simulate some allocations
    AllocationClientWithStackTrace::report_allocation(0, 1024 * 1024, 1, 0x1000000);
    AllocationClientWithStackTrace::report_allocation(0, 512 * 1024, 1, 0x1100000);
    AllocationClientWithStackTrace::report_allocation(1, 256 * 1024, 0, 0x2000000);

    // Simulate deallocations
    AllocationClientWithStackTrace::report_deallocation(0, 0x1000000);
    AllocationClientWithStackTrace::report_deallocation(0, 0x1100000);

    std::cout << "\nDone! Check " << STACK_TRACE_FILE << " for stack traces." << std::endl;

    return 0;
}
