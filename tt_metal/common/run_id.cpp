// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/run_id.hpp>

#include <fmt/format.h>

#include <cstdint>
#include <cstdlib>
#include <random>
#include <string>

#include "common/env_lib.hpp"

namespace tt::tt_metal {

namespace {

std::string mint_run_id() {
    std::random_device rd;
    std::uniform_int_distribution<std::uint64_t> dist;
    return fmt::format("{:016x}{:016x}", dist(rd), dist(rd));
}

std::string resolve_run_id() {
    std::string from_env = tt::parse_env<std::string>("TT_METAL_RUN_ID", "");
    // A blank value is treated as unset so that `TT_METAL_RUN_ID=` does not produce artefacts
    // carrying an empty identifier that would then appear to pair with each other.
    if (!from_env.empty()) {
        return from_env;
    }
    return mint_run_id();
}

}  // namespace

const std::string& get_or_create_run_id() {
    static const std::string run_id = resolve_run_id();
    return run_id;
}

}  // namespace tt::tt_metal
