// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Sets or unsets one environment variable for a scope and restores it after.

#ifndef X280_TESTS_HOST_SCOPED_ENV_HPP
#define X280_TESTS_HOST_SCOPED_ENV_HPP

#include <cstdlib>
#include <string>

namespace x280::testing {

/** A null `value` unsets `name`. Not safe against concurrent environment use. */
class ScopedEnv {
public:
    ScopedEnv(const char* name, const char* value) : name_(name) {
        const char* prev = std::getenv(name);
        if (prev != nullptr) {
            has_prev_ = true;
            prev_value_ = prev;
        }
        if (value != nullptr) {
            setenv(name, value, 1);
        } else {
            unsetenv(name);
        }
    }

    ~ScopedEnv() {
        if (has_prev_) {
            setenv(name_.c_str(), prev_value_.c_str(), 1);
        } else {
            unsetenv(name_.c_str());
        }
    }

    ScopedEnv(const ScopedEnv&) = delete;
    ScopedEnv& operator=(const ScopedEnv&) = delete;

private:
    std::string name_;
    bool has_prev_ = false;
    std::string prev_value_;
};

}  // namespace x280::testing

#endif  // X280_TESTS_HOST_SCOPED_ENV_HPP
