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
#include <string>
#include <inttypes.h>

// A simple, lock-free single-producer, single-consumer ring buffer.
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

// For debugging
static std::atomic<uint64_t> g_total_hook_calls{0};
static std::atomic<uint64_t> g_total_enqueued{0};
static std::atomic<uint64_t> g_dropped_full{0};

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
    // __cxa_demangle allocates memory using malloc
    char* demangled = abi::__cxa_demangle(name, nullptr, nullptr, &status);
    if (status == 0 && demangled) {
        return demangled;
    }
    if (demangled) {
        free(demangled);
    }
    return strdup(name);
}

/**
 * @brief Gets the function name, source file, and line number for a given address.
 *
 * This function shells out to the `llvm-addr2line-17` utility to resolve the address
 * using the debug symbols in the shared object.
 *
 * @param info Dl_info struct for the address, containing the object path.
 * @param addr The absolute memory address to resolve.
 * @param func_buf Buffer to store the resulting function name.
 * @param func_buf_size Size of the function name buffer.
 * @param file_buf Buffer to store the resulting file path.
 * @param file_buf_size Size of the file buffer.
 * @param line_buf Buffer to store the resulting line number.
 * @param line_buf_size Size of the line buffer.
 */
static void get_source_info(
    const Dl_info& info,
    void* addr,
    char* func_buf,
    size_t func_buf_size,
    char* file_buf,
    size_t file_buf_size,
    char* line_buf,
    size_t line_buf_size) {
    // Default to unknown
    snprintf(func_buf, func_buf_size, "??");
    snprintf(file_buf, file_buf_size, "??");
    snprintf(line_buf, line_buf_size, "0");

    // Calculate the relative address within the shared object
    void* relative_addr = (void*)((char*)addr - (char*)info.dli_fbase);

    // Construct the addr2line command. -f shows function name, -e specifies the executable.
    char command[2048];
    snprintf(command, sizeof(command), "llvm-addr2line-17 -f -i -e %s %p", info.dli_fname, relative_addr);

    FILE* pipe = popen(command, "r");
    if (!pipe) {
        return;
    }

    char func_name_output[4096];
    char file_line_output[4096];

    // The output of `addr2line -f` is two lines:
    // 1. Mangled function name
    // 2. file:line
    if (fgets(func_name_output, sizeof(func_name_output), pipe) != nullptr &&
        fgets(file_line_output, sizeof(file_line_output), pipe) != nullptr) {
        // Remove newline characters
        func_name_output[strcspn(func_name_output, "\n")] = 0;
        file_line_output[strcspn(file_line_output, "\n")] = 0;

        // Demangle the function name
        char* demangled_name = demangle(func_name_output);
        if (demangled_name) {
            snprintf(func_buf, func_buf_size, "%s", demangled_name);
            free(demangled_name);
        }

        // Split "filepath:linenumber"
        char* colon = strrchr(file_line_output, ':');
        if (colon) {
            *colon = '\0';
            snprintf(file_buf, file_buf_size, "%s", file_line_output);
            snprintf(line_buf, line_buf_size, "%s", colon + 1);
        }
    }
    pclose(pipe);
}

/**
 * @brief The main function for the background processing thread.
 */
static void* processing_thread_func(void* arg) {
    g_log_fd = open("execution_trace_clang.jsonl", O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (g_log_fd < 0) {
        return nullptr;
    }
    uint64_t counter = 0;

    while (!g_done.load(std::memory_order_acquire) || g_buffer_head.load() != g_buffer_tail.load()) {
        size_t tail = g_buffer_tail.load(std::memory_order_relaxed);
        if (tail != g_buffer_head.load(std::memory_order_acquire)) {
            void* caller_addr = g_call_buffer[tail];
            Dl_info caller_info;
            if (dladdr(caller_addr, &caller_info) && caller_info.dli_fname) {
                const char* caller_so_path = caller_info.dli_fname;

                if (true || strstr(caller_so_path, "libtt_metal.so") || strstr(caller_so_path, "_ttnn.so") ||
                    strstr(caller_so_path, "_ttnncpp.so")) {
                    char func_name_buf[1024];
                    char file_path_buf[1024];
                    char line_num_buf[32];

                    const char* mangled_name = caller_info.dli_sname ? caller_info.dli_sname : "??";

                    get_source_info(
                        caller_info,
                        caller_addr,
                        func_name_buf,
                        sizeof(func_name_buf),
                        file_path_buf,
                        sizeof(file_path_buf),
                        line_num_buf,
                        sizeof(line_num_buf));

                    if (true || strstr(file_path_buf, "/home/ubuntu/tt-metal/tt_metal") ||
                        strstr(file_path_buf, "/home/ubuntu/tt-metal/ttnn")) {
                        dprintf(
                            g_log_fd,
                            "{\"event\":\"enter\",\"func\":\"%s\",\"func_mangled\":\"%s\",\"file\":\"%s:%s\"}\n",
                            func_name_buf,
                            mangled_name,
                            file_path_buf,
                            line_num_buf);
                    }
                }
            }
            g_buffer_tail.store((tail + 1) % BUFFER_SIZE, std::memory_order_release);
        } else {
            counter++;
            if (counter > 10000) {
                usleep(10);
                counter = 0;
            }
        }
    }

    // Emit a single final summary entry before closing the log.
    {
        uint64_t total_calls = g_total_hook_calls.load(std::memory_order_relaxed);
        uint64_t total_enqueued = g_total_enqueued.load(std::memory_order_relaxed);
        uint64_t dropped_full = g_dropped_full.load(std::memory_order_relaxed);
        const char* lost = (dropped_full > 0) ? "true" : "false";
        dprintf(
            g_log_fd,
            "{\"event\":\"summary\",\"hook_calls\":%" PRIu64 ",\"enqueued\":%" PRIu64 ",\"dropped_ring_full\":%" PRIu64
            ",\"lost\":%s}\n",
            total_calls,
            total_enqueued,
            dropped_full,
            lost);
    }

    close(g_log_fd);
    g_log_fd = -1;
    return nullptr;
}

/**
 * @brief Function to be called at program exit to clean up.
 */
__attribute__((no_instrument_function)) static void profiler_shutdown() {
    g_done.store(true, std::memory_order_release);
    if (g_processing_thread) {
        pthread_join(g_processing_thread, nullptr);
    }
}

/**
 * @brief A constructor function that runs when the library is loaded.
 */
__attribute__((constructor, no_instrument_function)) static void profiler_init() {
    atexit(profiler_shutdown);
    pthread_create(&g_processing_thread, nullptr, processing_thread_func, nullptr);
}

extern "C" {

// Clang's sanitizer coverage hook, called to initialize guard variables.
void __attribute__((used, no_instrument_function)) __sanitizer_cov_trace_pc_guard_init(
    uint32_t* start, uint32_t* stop) {
    static uint64_t n;
    if (start == stop || *start) {
        return;
    }
    for (uint32_t* x = start; x < stop; x++) {
        *x = ++n;
    }
}

// Clang's sanitizer coverage hook, called on function entry.
void __attribute__((used, no_instrument_function)) __sanitizer_cov_trace_pc_guard(uint32_t* guard) {
    if (!*guard) {
        return;
    }

    // Count every hook invocation
    g_total_hook_calls.fetch_add(1, std::memory_order_relaxed);

    void* caller = __builtin_return_address(0);

    size_t head = g_buffer_head.load(std::memory_order_relaxed);
    size_t next_head = (head + 1) % BUFFER_SIZE;

    if (next_head != g_buffer_tail.load(std::memory_order_acquire)) {
        g_call_buffer[head] = caller;
        g_buffer_head.store(next_head, std::memory_order_release);
        g_total_enqueued.fetch_add(1, std::memory_order_relaxed);
    } else {
        // Ring is full; record the drop
        g_dropped_full.fetch_add(1, std::memory_order_relaxed);
    }
}

}  // extern "C"
