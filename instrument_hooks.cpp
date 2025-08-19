#include <dlfcn.h>
#include <cxxabi.h>
#include <cstdlib>
#include <pthread.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <atomic>

// A simple, lock-free single-producer, multi-consumer ring buffer.
// This is safe for our use case where the hooks are the producers and one thread is the consumer.
constexpr size_t BUFFER_SIZE = 65536;  // 64k entries, should be enough for most cases
static void* g_call_buffer[BUFFER_SIZE];
static std::atomic<size_t> g_buffer_head(0);
static std::atomic<size_t> g_buffer_tail(0);

// File descriptor for the final JSONL trace log
static int g_log_fd = -1;

// Flag to signal the processing thread to exit
static std::atomic<bool> g_done(false);

// The processing thread
static pthread_t g_processing_thread;

/**
 * @brief Demangles a C++ symbol name.
 *
 * @param name The mangled symbol name.
 * @return A pointer to the demangled name string. The caller is responsible for
 * freeing this memory using free(). Returns a malloc'd copy of the
 * original name if demangling fails.
 */
static char* demangle(const char* name) {
    int status = 0;
    char* demangled = abi::__cxa_demangle(name, nullptr, nullptr, &status);
    if (status == 0 && demangled) {
        return demangled;
    }
    if (demangled) {
        free(demangled);
    }
    size_t name_len = strlen(name) + 1;
    char* name_copy = (char*)malloc(name_len);
    if (name_copy) {
        memcpy(name_copy, name, name_len);
    }
    return name_copy;
}

/**
 * @brief Converts a function address to its string representation.
 *
 * @param addr The memory address of the function.
 * @param buf The buffer to write the string representation into.
 * @param size The size of the buffer.
 */
static void addr_to_string(void* addr, char* buf, size_t size) {
    Dl_info info;
    if (dladdr(addr, &info) && info.dli_sname) {
        char* demangled_name = demangle(info.dli_sname);
        if (demangled_name) {
            snprintf(buf, size, "%s", demangled_name);
            free(demangled_name);
        } else {
            snprintf(buf, size, "??");
        }
    } else {
        snprintf(buf, size, "??");
    }
}

/**
 * @brief The main function for the background processing thread.
 *
 * This thread consumes addresses from the ring buffer, resolves them,
 * and writes them to the log file.
 */
static void* processing_thread_func(void* arg) {
    g_log_fd = open("execution_trace_clang.jsonl", O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (g_log_fd < 0) {
        return nullptr;
    }

    // Process items until the main program is done AND the buffer is empty.
    while (!g_done.load(std::memory_order_acquire) || g_buffer_head.load() != g_buffer_tail.load()) {
        size_t tail = g_buffer_tail.load(std::memory_order_relaxed);
        if (tail != g_buffer_head.load(std::memory_order_acquire)) {
            void* caller = g_call_buffer[tail];

            Dl_info caller_info;
            dladdr(caller, &caller_info);
            const char* caller_file = caller_info.dli_fname ? caller_info.dli_fname : "";

            // FILTER: Only log if the caller is part of the target library.
            if (strstr(caller_file, "libtt_metal.so")) {
                char caller_name_buf[1024];
                addr_to_string(caller, caller_name_buf, sizeof(caller_name_buf));
                dprintf(
                    g_log_fd,
                    "{\"event\":\"enter\",\"func\":\"%s\",\"caller_path\":\"%s\"}\n",
                    caller_name_buf,
                    caller_file);
            }
            g_buffer_tail.store((tail + 1) % BUFFER_SIZE, std::memory_order_release);
        } else {
            // Buffer is empty, sleep for a bit to avoid busy-waiting
            usleep(100);
        }
    }

    close(g_log_fd);
    g_log_fd = -1;
    return nullptr;
}

/**
 * @brief Function to be called at program exit to clean up.
 */
static void profiler_shutdown() {
    g_done.store(true, std::memory_order_release);
    pthread_join(g_processing_thread, nullptr);
}

/**
 * @brief A constructor function that runs when the library is loaded.
 */
__attribute__((constructor)) static void profiler_init() {
    // Register the shutdown function to be called at normal program termination.
    atexit(profiler_shutdown);
    // Create the background thread that will process the buffer.
    pthread_create(&g_processing_thread, nullptr, processing_thread_func, nullptr);
}

extern "C" {

// Clang's sanitizer coverage hook, called to initialize guard variables.
void __sanitizer_cov_trace_pc_guard_init(uint32_t* start, uint32_t* stop) {
    static uint64_t n;
    if (start == stop || *start) {
        return;
    }
    for (uint32_t* x = start; x < stop; x++) {
        *x = ++n;
    }
}

// Clang's sanitizer coverage hook, called on function entry.
// This function must be extremely fast and safe. It only writes to the buffer.
void __sanitizer_cov_trace_pc_guard(uint32_t* guard) {
    if (!*guard) {
        return;
    }

    // Get the address of the function that called this one.
    void* caller = __builtin_return_address(0);

    // Add the address to the ring buffer. This is the only work done in the hook.
    size_t head = g_buffer_head.load(std::memory_order_relaxed);
    size_t next_head = (head + 1) % BUFFER_SIZE;

    // Check for buffer overflow. If the buffer is full, we drop the trace
    // to avoid blocking the instrumented program.
    if (next_head != g_buffer_tail.load(std::memory_order_acquire)) {
        g_call_buffer[head] = caller;
        g_buffer_head.store(next_head, std::memory_order_release);
    }

    // Mark this guard as processed.
    *guard = 0;
}

}  // extern "C"
