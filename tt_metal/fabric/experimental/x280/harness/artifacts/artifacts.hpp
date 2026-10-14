// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// harness/artifacts/artifacts.hpp -- image/kernel path discovery.
//
// One-place resolution for:
//   - X280 firmware image path (env-var, empty when unset).
//   - Tensix kernel source directory (tests/device/kernels), the same
//     path the host cases already compute from __FILE__.
//   - env_u32/env_str helpers and the strict decimal parser behind them.

#ifndef X280_HARNESS_ARTIFACTS_HPP
#define X280_HARNESS_ARTIFACTS_HPP

#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>

namespace x280::harness::artifacts {

// Resolve X280 firmware image path from an env var. Returns an empty string
// when the variable is unset. Tests treat "" as "GTEST_SKIP()".
inline std::string image_path(const char* env_var) {
    const char* p = std::getenv(env_var);
    return p ? std::string(p) : std::string{};
}

// The tests/device/kernels directory relative to a specific host test.
// A tests/device/*.cpp caller passes __FILE__ and gets kernels/ next to it.
inline std::filesystem::path kernel_dir_of(const char* test_file) {
    return std::filesystem::absolute(std::filesystem::path(test_file).parent_path() / "kernels");
}

// The x280/ root, resolved from a `kernel_dir_of` result: tests/device/kernels
// goes up three levels to x280/.
inline std::filesystem::path x280_root_of(const std::filesystem::path& kernel_dir) {
    return std::filesystem::absolute(kernel_dir
                                         .parent_path()    // tests/device
                                         .parent_path()    // tests
                                         .parent_path());  // x280
}

// The kernels/<subdir> directory relative to the x280/ root, resolved from
// a `kernel_dir_of` result.
inline std::filesystem::path shared_kernel_dir(const std::filesystem::path& kernel_dir, const char* subdir) {
    return x280_root_of(kernel_dir) / "kernels" / subdir;
}

/**
 * Parse unsigned decimal input without signs/whitespace or narrowing overflow.
 * Throws std::runtime_error naming `name` and `str` when `str` is empty, holds
 * a non-digit or trailing character, or exceeds `max`.
 */
inline uint64_t parse_u64_strict(const std::string& str, const char* name, uint64_t max = UINT64_MAX) {
    uint64_t val = 0;
    const auto parsed = std::from_chars(str.data(), str.data() + str.size(), val);
    if (str.empty() || parsed.ec != std::errc{} || parsed.ptr != str.data() + str.size() || val > max) {
        throw std::runtime_error(std::string("invalid or out-of-range number in ") + name + ": " + str);
    }
    return val;
}

/**
 * Read an env var as u32; empty when unset/empty. Throws std::runtime_error
 * naming the variable and its value unless the whole value is a decimal
 * number no larger than UINT32_MAX.
 */
inline std::optional<uint32_t> env_u32(const char* name) {
    const char* v = std::getenv(name);
    if (v == nullptr || *v == '\0') {
        return std::nullopt;
    }
    return static_cast<uint32_t>(parse_u64_strict(v, name, UINT32_MAX));
}

// Read an env var as u32 with a fallback when unset/empty.
inline uint32_t env_u32(const char* name, uint32_t fallback) { return env_u32(name).value_or(fallback); }

// Read an env var as a string with a fallback when unset/empty.
inline std::string env_str(const char* name, const char* fallback = "") {
    const char* v = std::getenv(name);
    if (v == nullptr || *v == '\0') {
        return std::string(fallback);
    }
    return std::string(v);
}

}  // namespace x280::harness::artifacts

#endif  // X280_HARNESS_ARTIFACTS_HPP
