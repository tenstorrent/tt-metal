// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

namespace tt::tt_metal::profiler {

struct LosslessNocOperation {
    uint64_t runtime_id;
    std::optional<uint64_t> trace_id;
    std::optional<uint64_t> trace_replay_session_id;
    std::string name;
};

struct LosslessNocCore {
    uint32_t x;
    uint32_t y;
};

struct LosslessNocTransaction {
    LosslessNocOperation operation;
    uint32_t device_id;
    LosslessNocCore core;
    std::string risc;
    uint64_t issue_timestamp;
    std::string type;
    std::string noc;
    int32_t vc;
    nlohmann::ordered_json destinations;
    uint32_t num_bytes;
    nlohmann::ordered_json debug_metadata;
};

inline nlohmann::ordered_json serializeLosslessNocTransaction(const LosslessNocTransaction& transaction) {
    nlohmann::ordered_json operation = {
        {"runtime_id", transaction.operation.runtime_id},
        {"trace_id",
         transaction.operation.trace_id ? nlohmann::ordered_json(*transaction.operation.trace_id) : nullptr},
        {"trace_replay_session_id",
         transaction.operation.trace_replay_session_id
             ? nlohmann::ordered_json(*transaction.operation.trace_replay_session_id)
             : nullptr},
        {"name", transaction.operation.name},
    };

    return {
        {"operation", operation},
        {"device_id", transaction.device_id},
        {"core", {{"x", transaction.core.x}, {"y", transaction.core.y}}},
        {"risc", transaction.risc},
        {"issue_timestamp", transaction.issue_timestamp},
        {"type", transaction.type},
        {"noc", transaction.noc},
        {"vc", transaction.vc},
        {"destinations", transaction.destinations},
        {"num_bytes", transaction.num_bytes},
        {"debug_metadata", transaction.debug_metadata},
    };
}

inline nlohmann::ordered_json makeLosslessNocManifest(uint32_t device_frequency_mhz, uint64_t event_count) {
    return {
        {"schema_version", 1},
        {"capture_mode", "non_dropping"},
        {"complete", true},
        {"timestamp_semantics", "issue_cycles"},
        {"device_frequency_mhz", device_frequency_mhz},
        {"events", {{"path", "lossless_noc_events.jsonl"}, {"count", event_count}}},
        {"npe_modeled_semantics",
         {
             {"input", "noc_trace*.json"},
             {"input_timestamps", "issue_cycles"},
             {"completion_timestamps", "modeled_by_tt_npe"},
         }},
    };
}

inline void writeLosslessNocArtifactsAtomically(
    const std::filesystem::path& output_dir,
    const std::vector<LosslessNocTransaction>& transactions,
    uint32_t device_frequency_mhz) {
    constexpr std::string_view events_filename = "lossless_noc_events.jsonl";
    constexpr std::string_view manifest_filename = "lossless_noc_manifest.json";

    const auto events_path = output_dir / events_filename;
    const auto manifest_path = output_dir / manifest_filename;
    const auto events_tmp_path = output_dir / "lossless_noc_events.jsonl.tmp";
    const auto manifest_tmp_path = output_dir / "lossless_noc_manifest.json.tmp";

    if (std::ifstream existing_manifest_stream(manifest_path); existing_manifest_stream) {
        const auto existing_manifest = nlohmann::json::parse(existing_manifest_stream);
        if (existing_manifest.value("device_frequency_mhz", device_frequency_mhz) != device_frequency_mhz) {
            throw std::runtime_error("Lossless NoC capture contains devices with different frequencies");
        }
    }

    uint64_t event_count = 0;
    std::ofstream events_stream(events_tmp_path, std::ios::trunc);
    if (!events_stream) {
        throw std::runtime_error("Could not open lossless NoC events temporary file");
    }

    if (std::ifstream existing_events(events_path); existing_events) {
        std::string line;
        while (std::getline(existing_events, line)) {
            if (!line.empty()) {
                events_stream << line << '\n';
                ++event_count;
            }
        }
    }

    for (const auto& transaction : transactions) {
        events_stream << serializeLosslessNocTransaction(transaction).dump() << '\n';
        ++event_count;
    }
    events_stream.close();
    if (!events_stream) {
        throw std::runtime_error("Could not write lossless NoC events temporary file");
    }

    std::ofstream manifest_stream(manifest_tmp_path, std::ios::trunc);
    if (!manifest_stream) {
        throw std::runtime_error("Could not open lossless NoC manifest temporary file");
    }
    manifest_stream << makeLosslessNocManifest(device_frequency_mhz, event_count).dump(2) << '\n';
    manifest_stream.close();
    if (!manifest_stream) {
        throw std::runtime_error("Could not write lossless NoC manifest temporary file");
    }

    std::error_code error;
    std::filesystem::remove(manifest_path, error);
    if (error) {
        throw std::runtime_error("Could not invalidate lossless NoC manifest: " + error.message());
    }
    std::filesystem::rename(events_tmp_path, events_path, error);
    if (error) {
        throw std::runtime_error("Could not publish lossless NoC events file: " + error.message());
    }
    std::filesystem::rename(manifest_tmp_path, manifest_path, error);
    if (error) {
        throw std::runtime_error("Could not publish lossless NoC manifest file: " + error.message());
    }
}

}  // namespace tt::tt_metal::profiler
