// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace tt::tt_metal::jit_server {

struct KernelKey {
    std::uint64_t build_key = 0;
    std::string kernel_name;

    friend bool operator==(const KernelKey&, const KernelKey&) = default;
};

struct FirmwareKey {
    std::uint64_t build_key = 0;
    std::string server_endpoint;

    friend bool operator==(const FirmwareKey&, const FirmwareKey&) = default;
};

enum class FirmwareState : std::uint8_t {
    PRESENT = 0,
    ABSENT = 1,
};

enum class FirmwareUploadAction : std::uint8_t {
    SKIP_ALREADY_PRESENT = 0,
    YOU_UPLOAD = 1,
    WAIT_FOR_OTHER = 2,
};

struct BrokerAssignment {
    std::string server_endpoint;
    std::uint64_t handle = 0;
    FirmwareState firmware_state = FirmwareState::ABSENT;
};

struct BrokerAssignRequest {
    std::uint64_t build_key = 0;
    std::vector<std::string> kernel_keys;
};

struct BrokerAssignResponse {
    std::vector<BrokerAssignment> assignments;
};

struct KernelKeyHash {
    std::size_t operator()(const KernelKey& key) const {
        const std::size_t h1 = std::hash<std::uint64_t>{}(key.build_key);
        const std::size_t h2 = std::hash<std::string>{}(key.kernel_name);
        return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    }
};

struct FirmwareKeyHash {
    std::size_t operator()(const FirmwareKey& key) const {
        const std::size_t h1 = std::hash<std::uint64_t>{}(key.build_key);
        const std::size_t h2 = std::hash<std::string>{}(key.server_endpoint);
        return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    }
};

}  // namespace tt::tt_metal::jit_server
