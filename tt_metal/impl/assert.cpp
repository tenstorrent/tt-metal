// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cxxabi.h>
#include <execinfo.h>
#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <sstream>
#include <tt-logger/tt-logger.hpp>
#include <vector>

namespace tt::assert {

    namespace {
// NOLINTBEGIN(cppcoreguidelines-no-malloc)
std::string demangle(const char* str) {
    size_t size = 0;
    int status = 0;
    std::string rt(256, '\0');
    if (1 == sscanf(str, "%*[^(]%*[^_]%255[^)+]", &rt[0])) {
        char* v = abi::__cxa_demangle(&rt[0], nullptr, &size, &status);
        if (v) {
            std::string result(v);
            free(v);
            return result;
        }
    }
    return str;
    }
// NOLINTEND(cppcoreguidelines-no-malloc)
}

// https://www.fatalerrors.org/a/backtrace-function-and-assert-assertion-macro-encapsulation.html

/**
 * @brief Get the current call stack
 * @param[out] bt Save Call Stack
 * @param[in] size Maximum number of return layers
 * @param[in] skip Skip the number of layers at the top of the stack
 */
// NOLINTBEGIN(cppcoreguidelines-no-malloc)
std::vector<std::string> backtrace(int size , int skip ) {
    std::vector<std::string> bt;
    void** array = (void**)malloc((sizeof(void*) * size));
    size_t s = ::backtrace(array, size);
    char** strings = backtrace_symbols(array, s);
    if (strings == NULL) {
        std::cout << "backtrace_symbols error." << std::endl;
        return bt;
    }
    for (size_t i = skip; i < s; ++i) {
        bt.push_back(demangle(strings[i]));
    }
    free(strings);  // NOLINT(bugprone-multi-level-implicit-pointer-conversion)
    free(array);    // NOLINT(bugprone-multi-level-implicit-pointer-conversion)

    return bt;
}
// NOLINTEND(cppcoreguidelines-no-malloc)

std::string backtrace_to_string(int size , int skip , const std::string& prefix) {
    std::vector<std::string> bt = backtrace(size, skip);
    std::stringstream ss;
    for (size_t i = 0; i < bt.size(); ++i) {
        ss << prefix << bt[i] << std::endl;
    }
    return ss.str();
}

}  // namespace tt::assert
