#pragma once

#include <iostream>
#include <thread>
#include <execinfo.h>
#include <cstdlib>
#include <cxxabi.h>

// Thread-local flag to track API entry
inline thread_local bool g_inPublicAPI = false;

// RAII Guard Class
struct APICallGuard {
    APICallGuard() { g_inPublicAPI = true; }
    ~APICallGuard() { g_inPublicAPI = false; }
};

// Simple stack trace dumper for Unix/Linux
inline void DumpCallStack(int maxFrames = 10) {
    void* frames[maxFrames];
    int numFrames = backtrace(frames, maxFrames);
    char** symbols = backtrace_symbols(frames, numFrames);

    if (symbols) {
        std::cerr << "Call Stack (max " << maxFrames << " frames):\n";

        for (int i = 0; i < numFrames; ++i) {
            std::string symbol(symbols[i]);

            // Attempt to find the mangled name in the symbol string
            size_t begin = symbol.find('(');
            size_t end = symbol.find('+', begin);

            if (begin != std::string::npos && end != std::string::npos) {
                std::string mangled = symbol.substr(begin + 1, end - begin - 1);
                int status = 0;

                // Demangle the symbol
                char* demangled = abi::__cxa_demangle(mangled.c_str(), nullptr, nullptr, &status);
                if (status == 0 && demangled) {
                    // Replace mangled name with demangled name
                    symbol.replace(begin + 1, end - begin - 1, demangled);
                    std::free(demangled);
                }
            }

            std::cerr << symbol << "\n";
        }
        std::free(symbols);
    } else {
        std::cerr << "Unable to capture stack trace.\n";
    }
}

inline bool IsCallGuardEnabled() {
    static bool enabled = (std::getenv("EN_CALL_GUARD") != nullptr);
    return enabled;
}

// Macro to ensure lower-level functions are called from a public API
#define ENSURE_CALLED_FROM_API()                                                                               \
    do {                                                                                                       \
        if (IsCallGuardEnabled()) {                                                                            \
            if (!g_inPublicAPI) {                                                                              \
                std::cerr << "KCM ERROR: Function called outside public API.  " << __PRETTY_FUNCTION__ << " (" \
                          << __FILE__ << ":" << __LINE__ << ")\n";                                             \
                DumpCallStack(10);                                                                             \
            } else {                                                                                           \
                std::cerr << "KCM INFO: Function called inside public API.  " << __PRETTY_FUNCTION__ << " ("   \
                          << __FILE__ << ":" << __LINE__ << ")\n";                                             \
            }                                                                                                  \
        }                                                                                                      \
    } while (0)

// Macro for public API entry with automatic guard and logging
#define PUBLIC_API_ENTRY()                                                                                     \
    APICallGuard _apiGuard;                                                                                    \
    if (IsCallGuardEnabled()) {                                                                                \
        std::cout << "KCM Entering public API: " << __PRETTY_FUNCTION__ << " (" << __FILE__ << ":" << __LINE__ \
                  << ")\n";                                                                                    \
    }

// Macro for flagging detail API entry. Don't create guard since not public API.
#define DETAIL_API_ENTRY()                                                                                     \
    if (IsCallGuardEnabled()) {                                                                                \
        std::cout << "KCM Entering detail API: " << __PRETTY_FUNCTION__ << " (" << __FILE__ << ":" << __LINE__ \
                  << ")\n";                                                                                    \
    }
