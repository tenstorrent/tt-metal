// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/session_id.hpp>

#include <fmt/format.h>

#include <cstdint>
#include <random>
#include <string>

#include "common/env_lib.hpp"

namespace tt::tt_metal {
namespace {

std::string mint_session_id() {
    std::random_device rd;
    std::uniform_int_distribution<std::uint64_t> dist;
    return fmt::format("{:016x}{:016x}", dist(rd), dist(rd));
}

std::string resolve_session_id() {
    std::string from_env = tt::parse_env<std::string>("TTNN_RUN_SESSION_ID", "");
    return from_env.empty() ? mint_session_id() : from_env;
}

}  // namespace

const std::string& get_or_create_session_id() {
    static const std::string session_id = resolve_session_id();
    return session_id;
}

}  // namespace tt::tt_metal
