// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include "tt_metal/fabric/physical_system_descriptor.hpp"

struct TestParams {
    uint32_t packet_size_bytes = 0;
    uint32_t data_size = 0;
};

struct RetrainFailure {
    uint32_t retrain_count = 0;
    TestParams test_params;
};

struct CrcErrorFailure {
    uint32_t crc_error_count = 0;
    TestParams test_params;
};

struct UncorrectedCodewordFailure {
    uint32_t uncorrected_codeword_count = 0;
    TestParams test_params;
};

struct DataMismatchFailure {
    uint32_t num_mismatched_words = 0;
    TestParams test_params;
};

struct LinkFailure {
    RetrainFailure retrain_failure;
    CrcErrorFailure crc_error_failure;
    UncorrectedCodewordFailure uncorrected_codeword_failure;
    DataMismatchFailure data_mismatch_failure;
};

struct EthChannelIdentifier {
    std::string host;
    tt::tt_metal::AsicID asic_id;
    tt::tt_metal::TrayID tray_id;
    tt::tt_metal::ASICLocation asic_location;
    uint8_t channel;
};

inline bool operator==(const EthChannelIdentifier& lhs, const EthChannelIdentifier& rhs) {
    return lhs.host == rhs.host && lhs.asic_id == rhs.asic_id && lhs.tray_id == rhs.tray_id &&
           lhs.asic_location == rhs.asic_location && lhs.channel == rhs.channel;
}

namespace std {
template <>
struct hash<EthChannelIdentifier> {
    size_t operator()(const EthChannelIdentifier& identifier) const {
        return std::hash<std::string>()(identifier.host) ^ std::hash<tt::tt_metal::AsicID>()(identifier.asic_id) ^
               std::hash<tt::tt_metal::TrayID>()(identifier.tray_id) ^
               std::hash<tt::tt_metal::ASICLocation>()(identifier.asic_location) ^
               std::hash<uint8_t>()(identifier.channel);
    }
};
}  // namespace std

struct FaultyLink {
    EthChannelIdentifier channel_identifier;
    LinkFailure link_failure;
};
