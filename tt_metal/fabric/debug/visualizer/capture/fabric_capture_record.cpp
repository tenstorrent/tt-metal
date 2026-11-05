// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture_record.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <string_view>
#include <utility>

#include <tt_stl/assert.hpp>

#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture_json.hpp"
#include "tt_metal/fabric/debug/visualizer/common/fabric_visualizer_files.hpp"

namespace tt::tt_fabric::capture {

namespace {

using json = JsonReader::json;
using json_pointer = JsonReader::json_pointer;

// ============ Enum names ============

template <typename E>
struct EnumName {
    E value;
    std::string_view name;
};

constexpr std::array<EnumName<ReadStep>, 3> k_read_step_names = {{
    {ReadStep::REGISTERS_BEFORE, "registers_before"},
    {ReadStep::L1, "l1"},
    {ReadStep::REGISTERS_AFTER, "registers_after"},
}};

constexpr std::array<EnumName<ChipLocation>, 3> k_chip_location_names = {{
    {ChipLocation::FOUND, "found"},
    {ChipLocation::MISSING, "missing"},
    {ChipLocation::ARCH_MISMATCH, "arch_mismatch"},
}};

constexpr std::array<EnumName<CoreLocation>, 4> k_core_location_names = {{
    {CoreLocation::FOUND, "found"},
    {CoreLocation::CORE_MISMATCH, "core_mismatch"},
    {CoreLocation::CHANNEL_NOT_FOUND, "channel_not_found"},
    {CoreLocation::CORE_NOT_FOUND, "core_not_found"},
}};

template <typename E, size_t N>
std::string get_enum_name(const std::array<EnumName<E>, N>& names, E value) {
    const auto it = std::ranges::find(names, value, &EnumName<E>::value);
    TT_FATAL(it != names.end(), "Fabric capture: no name for enum value {}", static_cast<int>(value));
    return std::string(it->name);
}

template <typename E, size_t N>
E read_enum(const JsonReader& record_reader, const json_pointer& pointer, const std::array<EnumName<E>, N>& names) {
    const std::string name = record_reader.read_string(pointer);
    const auto it = std::ranges::find(names, name, &EnumName<E>::name);
    TT_FATAL(
        it != names.end(),
        "Fabric capture: the capture record's {} is \"{}\", which it doesn't know",
        pointer.to_string(),
        name);
    return it->value;
}

// A router's key in its chip, such as "E0" for "M0/C7/E0".
std::string get_router_key(const std::string& router_path) { return router_path.substr(router_path.rfind('/') + 1); }

// ============ Writing ============

template <typename T>
json make_optional_json(const std::optional<T>& value) {
    return value.has_value() ? json(*value) : json(nullptr);
}

json make_manifest_identity_json(const ManifestIdentity& manifest) {
    json out;
    out["file"] = manifest.path.string();
    out["blake3"] = manifest.blake3;
    out["mpi_rank"] = manifest.mpi_rank;
    out["world_size"] = manifest.world_size;
    out["host_name"] = manifest.host_name;
    out["git_sha"] = manifest.git_sha;
    return out;
}

json make_liveness_sample_json(const LivenessSample& sample) {
    json out;
    out["round"] = sample.round;
    out["t_ns"] = sample.t_ns;
    out["heartbeat"] = make_optional_json(sample.heartbeat);
    out["core_registers"] = sample.core_registers;
    out["error"] = make_optional_json(sample.error);
    return out;
}

json make_read_record_json(const ReadRecord& read) {
    json out;
    out["step"] = get_enum_name(k_read_step_names, read.step);
    out["address"] = read.address;
    out["size_bytes"] = read.size_bytes;
    out["t_ns"] = read.t_ns;
    out["bin_offset"] = make_optional_json(read.bin_offset);
    out["error"] = make_optional_json(read.error);
    return out;
}

// A router: where its core was found, then its liveness samples and its reads. Its path is its key and its chip's.
json make_router_record_json(const RouterRecord& router) {
    const RouterTarget& target = router.target;
    json out;
    out["eth_chan"] = target.router.eth_chan;
    out["manifest_core"] = make_core_json(target.router.virtual_core);
    out["stream_register_addresses"] = target.router.stream_register_addresses;
    out["reader_core"] = target.reader_core.has_value() ? make_core_json(*target.reader_core) : json(nullptr);
    out["location"] =
        target.location.has_value() ? json(get_enum_name(k_core_location_names, *target.location)) : json(nullptr);
    out["discovery_skipped"] = target.discovery_skipped;
    json liveness = json::array();
    for (const LivenessSample& sample : router.liveness) {
        liveness.push_back(make_liveness_sample_json(sample));
    }
    out["liveness"] = std::move(liveness);
    json reads = json::array();
    for (const ReadRecord& read : router.reads) {
        reads.push_back(make_read_record_json(read));
    }
    out["reads"] = std::move(reads);
    return out;
}

json make_chip_record_json(const ChipRecord& chip) {
    json out;
    out["asic_id"] = make_asic_id_text(chip.asic_id);
    // Chip ids are never negative, so the record writes them unsigned, as it reads them.
    out["reader_chip_id"] = chip.chip.has_value() ? json(static_cast<uint32_t>(*chip.chip)) : json(nullptr);
    out["location"] = get_enum_name(k_chip_location_names, chip.location);
    json routers = json::object();
    for (const RouterRecord& router : chip.routers) {
        routers[get_router_key(router.target.router.path)] = make_router_record_json(router);
    }
    out["routers"] = std::move(routers);
    return out;
}

// ============ Reading ============

std::optional<std::string> read_optional_string(const JsonReader& record_reader, const json_pointer& pointer) {
    return record_reader.is_null(pointer) ? std::nullopt : std::optional(record_reader.read_string(pointer));
}

std::optional<uint64_t> read_optional_unsigned(const JsonReader& record_reader, const json_pointer& pointer) {
    return record_reader.is_null(pointer) ? std::nullopt : std::optional(record_reader.read_unsigned(pointer));
}

std::vector<uint64_t> read_unsigned_array(const JsonReader& record_reader, const json_pointer& pointer) {
    const size_t num_values = record_reader.read_array(pointer).size();
    std::vector<uint64_t> values;
    values.reserve(num_values);
    for (size_t i = 0; i < num_values; ++i) {
        values.push_back(record_reader.read_unsigned(pointer / i));
    }
    return values;
}

ManifestIdentity read_manifest_identity(const JsonReader& record_reader) {
    const json_pointer pointer("/manifest");
    return ManifestIdentity{
        .path = record_reader.read_string(pointer / "file"),
        .blake3 = record_reader.read_string(pointer / "blake3"),
        .mpi_rank = record_reader.read_uint32(pointer / "mpi_rank"),
        .world_size = record_reader.read_uint32(pointer / "world_size"),
        .host_name = record_reader.read_string(pointer / "host_name"),
        .git_sha = record_reader.read_string(pointer / "git_sha"),
    };
}

LivenessSample read_liveness_sample(const JsonReader& record_reader, const json_pointer& pointer) {
    std::vector<uint32_t> core_registers;
    for (const uint64_t value : read_unsigned_array(record_reader, pointer / "core_registers")) {
        core_registers.push_back(static_cast<uint32_t>(value));
    }
    const std::optional<uint64_t> heartbeat = read_optional_unsigned(record_reader, pointer / "heartbeat");
    return LivenessSample{
        .round = record_reader.read_uint32(pointer / "round"),
        .t_ns = record_reader.read_unsigned(pointer / "t_ns"),
        .heartbeat = heartbeat.has_value() ? std::optional(static_cast<uint32_t>(*heartbeat)) : std::nullopt,
        .core_registers = std::move(core_registers),
        .error = read_optional_string(record_reader, pointer / "error"),
    };
}

ReadRecord read_read_record(const JsonReader& record_reader, const json_pointer& pointer) {
    return ReadRecord{
        .step = read_enum(record_reader, pointer / "step", k_read_step_names),
        .address = record_reader.read_unsigned(pointer / "address"),
        .size_bytes = record_reader.read_uint32(pointer / "size_bytes"),
        .t_ns = record_reader.read_unsigned(pointer / "t_ns"),
        .bin_offset = read_optional_unsigned(record_reader, pointer / "bin_offset"),
        .error = read_optional_string(record_reader, pointer / "error"),
    };
}

RouterRecord read_router_record(
    const JsonReader& record_reader,
    const json_pointer& pointer,
    const std::string& chip_path,
    const std::string& key) {
    RouterRecord router{
        .target =
            RouterTarget{
                .router =
                    PlannedRouter{
                        .path = chip_path + "/" + key,
                        .eth_chan = record_reader.read_uint32(pointer / "eth_chan"),
                        .virtual_core = record_reader.read_core(pointer / "manifest_core"),
                        .stream_register_addresses =
                            read_unsigned_array(record_reader, pointer / "stream_register_addresses"),
                    },
                .reader_core = record_reader.is_null(pointer / "reader_core")
                                   ? std::nullopt
                                   : std::optional(record_reader.read_core(pointer / "reader_core")),
                .location = record_reader.is_null(pointer / "location")
                                ? std::nullopt
                                : std::optional(read_enum(record_reader, pointer / "location", k_core_location_names)),
                .discovery_skipped = record_reader.read_bool(pointer / "discovery_skipped"),
            },
    };
    for (size_t i = 0; i < record_reader.read_array(pointer / "liveness").size(); ++i) {
        router.liveness.push_back(read_liveness_sample(record_reader, pointer / "liveness" / i));
    }
    for (size_t i = 0; i < record_reader.read_array(pointer / "reads").size(); ++i) {
        router.reads.push_back(read_read_record(record_reader, pointer / "reads" / i));
    }
    return router;
}

ChipRecord read_chip_record(const JsonReader& record_reader, const json_pointer& pointer, const std::string& path) {
    ChipRecord chip{
        .path = path,
        .asic_id = record_reader.read_asic_id(pointer / "asic_id"),
        .chip = record_reader.is_null(pointer / "reader_chip_id")
                    ? std::nullopt
                    : std::optional(static_cast<int32_t>(record_reader.read_uint32(pointer / "reader_chip_id"))),
        .location = read_enum(record_reader, pointer / "location", k_chip_location_names),
    };
    for (const auto& [key, router] : record_reader.read_object(pointer / "routers").items()) {
        chip.routers.push_back(read_router_record(record_reader, pointer / "routers" / key, path, key));
    }
    return chip;
}

}  // namespace

json make_capture_record_json(const CaptureRecord& record) {
    json out;
    out["capture_version"] = visualizer::k_capture_version;
    out["file_kind"] = std::string(visualizer::k_capture_file_kind);
    out["manifest"] = make_manifest_identity_json(record.manifest);

    json capture;
    capture["git_sha"] = record.capture_git_sha;
    capture["cluster_id"] = make_optional_json(record.cluster_id);
    capture["started_at"] = record.started_at;
    capture["finished_at"] = record.finished_at;
    capture["liveness"] = {
        {"num_rounds", record.liveness_options.num_rounds},
        {"interval_ms", static_cast<uint64_t>(record.liveness_options.interval.count())},
    };
    out["capture"] = std::move(capture);

    json bin;
    bin["file"] = record.bin_file;
    bin["size_bytes"] = record.bin_size_bytes;
    bin["blake3"] = record.bin_blake3;
    out["bin"] = std::move(bin);

    json skipped = json::array();
    for (const DiscoverySkippedCore& core : record.discovery_skipped_cores) {
        skipped.push_back(
            {{"asic_id", make_asic_id_text(core.asic_id)},
             {"core", make_core_json(core.core)},
             {"reason", core.reason}});
    }
    out["discovery_skipped_cores"] = std::move(skipped);

    json chips = json::object();
    for (const ChipRecord& chip : record.chips) {
        chips[chip.path] = make_chip_record_json(chip);
    }
    out["chips"] = std::move(chips);
    return out;
}

CaptureRecord parse_capture_record(const json& document) {
    const JsonReader record_reader(document, "capture record");
    const std::string file_kind = record_reader.read_string(json_pointer("/file_kind"));
    TT_FATAL(
        file_kind == visualizer::k_capture_file_kind,
        "Fabric capture: the capture record's /file_kind is \"{}\", not \"{}\"",
        file_kind,
        visualizer::k_capture_file_kind);
    const uint64_t version = record_reader.read_unsigned(json_pointer("/capture_version"));
    TT_FATAL(
        version == visualizer::k_capture_version,
        "Fabric capture: the capture record's /capture_version is {}, but this build reads version {}",
        version,
        visualizer::k_capture_version);

    CaptureRecord record{
        .manifest = read_manifest_identity(record_reader),
        .capture_git_sha = record_reader.read_string(json_pointer("/capture/git_sha")),
        .cluster_id = read_optional_string(record_reader, json_pointer("/capture/cluster_id")),
        .started_at = record_reader.read_string(json_pointer("/capture/started_at")),
        .finished_at = record_reader.read_string(json_pointer("/capture/finished_at")),
        .liveness_options =
            LivenessOptions{
                .num_rounds = record_reader.read_uint32(json_pointer("/capture/liveness/num_rounds")),
                .interval = std::chrono::milliseconds(
                    record_reader.read_unsigned(json_pointer("/capture/liveness/interval_ms"))),
            },
        .bin_file = record_reader.read_string(json_pointer("/bin/file")),
        .bin_size_bytes = record_reader.read_unsigned(json_pointer("/bin/size_bytes")),
        .bin_blake3 = record_reader.read_string(json_pointer("/bin/blake3")),
    };
    const json_pointer skipped_pointer("/discovery_skipped_cores");
    for (size_t i = 0; i < record_reader.read_array(skipped_pointer).size(); ++i) {
        record.discovery_skipped_cores.push_back(DiscoverySkippedCore{
            .asic_id = record_reader.read_asic_id(skipped_pointer / i / "asic_id"),
            .core = record_reader.read_core(skipped_pointer / i / "core"),
            .reason = record_reader.read_string(skipped_pointer / i / "reason"),
        });
    }
    const json_pointer chips_pointer("/chips");
    for (const auto& [path, chip] : record_reader.read_object(chips_pointer).items()) {
        record.chips.push_back(read_chip_record(record_reader, chips_pointer / path, path));
    }
    return record;
}

}  // namespace tt::tt_fabric::capture
