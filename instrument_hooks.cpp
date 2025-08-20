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
 * @brief Gets the source file and line number for a given address.
 *
 * This function shells out to the `llvm-addr2line-17` utility to resolve the address
 * to a file and line number using the debug symbols in the shared object.
 *
 * @param info Dl_info struct for the address, containing the object path.
 * @param addr The absolute memory address to resolve.
 * @param file_buf Buffer to store the resulting file path.
 * @param file_buf_size Size of the file buffer.
 * @param line_buf Buffer to store the resulting line number.
 * @param line_buf_size Size of the line buffer.
 */
static void get_source_info(
    const Dl_info& info, void* addr, char* file_buf, size_t file_buf_size, char* line_buf, size_t line_buf_size) {
    // Default to unknown
    snprintf(file_buf, file_buf_size, "??");
    snprintf(line_buf, line_buf_size, "0");

    // Calculate the relative address within the shared object
    void* relative_addr = (void*)((char*)addr - (char*)info.dli_fbase);

    // Construct the addr2line command using the version-specific llvm tool
    char command[1024];
    snprintf(command, sizeof(command), "llvm-addr2line-17 -e %s %p", info.dli_fname, relative_addr);

    FILE* pipe = popen(command, "r");
    if (!pipe) {
        return;
    }

    char output[512];
    if (fgets(output, sizeof(output), pipe) != nullptr) {
        // The output is typically in the format "filepath:linenumber"
        char* colon = strrchr(output, ':');
        if (colon) {
            // Split the string at the colon
            *colon = '\0';
            snprintf(file_buf, file_buf_size, "%s", output);
            snprintf(line_buf, line_buf_size, "%s", colon + 1);
            // Remove trailing newline from line number if present
            line_buf[strcspn(line_buf, "\n")] = 0;
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

    while (!g_done.load(std::memory_order_acquire) || g_buffer_head.load() != g_buffer_tail.load()) {
        size_t tail = g_buffer_tail.load(std::memory_order_relaxed);
        if (tail != g_buffer_head.load(std::memory_order_acquire)) {
            void* caller_addr = g_call_buffer[tail];

            Dl_info caller_info;
            dladdr(caller_addr, &caller_info);
            const char* caller_so_path = caller_info.dli_fname ? caller_info.dli_fname : "";

            // First, check if the call is coming from our target library.
            if (strstr(caller_so_path, "libtt_metal.so")) {
                char func_name_buf[1024];
                char file_path_buf[1024];
                char line_num_buf[32];

                // Get the source info first, so we can filter on it.
                get_source_info(
                    caller_info, caller_addr, file_path_buf, sizeof(file_path_buf), line_num_buf, sizeof(line_num_buf));

                // REFINED FILTER: Now, also check if the source file path contains "tt-metal".
                // This excludes inlined standard library headers.
                if (strstr(file_path_buf, "tt-metal/tt_metal")) {
                    addr_to_string(caller_addr, func_name_buf, sizeof(func_name_buf));

                    dprintf(
                        g_log_fd,
                        "{\"event\":\"enter\",\"func\":\"%s\",\"file\":\"%s:%s\",\"so_path\":\"%s\"}\n",
                        func_name_buf,
                        file_path_buf,
                        line_num_buf,
                        caller_so_path);
                }
            }
            g_buffer_tail.store((tail + 1) % BUFFER_SIZE, std::memory_order_release);
        } else {
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
__attribute__((no_instrument_function)) static void profiler_shutdown() {
    g_done.store(true, std::memory_order_release);
    pthread_join(g_processing_thread, nullptr);
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
void __attribute__((used)) __sanitizer_cov_trace_pc_guard_init(uint32_t* start, uint32_t* stop) {
    static uint64_t n;
    if (start == stop || *start) {
        return;
    }
    for (uint32_t* x = start; x < stop; x++) {
        *x = ++n;
    }
}

// Clang's sanitizer coverage hook, called on function entry.
void __attribute__((used)) __sanitizer_cov_trace_pc_guard(uint32_t* guard) {
    if (!*guard) {
        return;
    }

    void* caller = __builtin_return_address(0);

    size_t head = g_buffer_head.load(std::memory_order_relaxed);
    size_t next_head = (head + 1) % BUFFER_SIZE;

    if (next_head != g_buffer_tail.load(std::memory_order_acquire)) {
        g_call_buffer[head] = caller;
        g_buffer_head.store(next_head, std::memory_order_release);
    }

    *guard = 0;
}

}  // extern "C"
