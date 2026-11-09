// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture_record.hpp"

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "tt_metal/fabric/debug/visualizer/common/fabric_visualizer_files.hpp"
#include "tt_metal/fabric/debug/visualizer/common/fabric_visualizer_json_reader.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_names.hpp"

namespace tt::tt_fabric::capture {

namespace {

using json = nlohmann::ordered_json;
using visualizer::JsonObjectReader;
using visualizer::JsonValueReader;

// How the record's read errors name it.
constexpr std::string_view k_record_document_name = "Fabric capture record";

// A router's key in its chip, such as "E0" for "M0/C7/E0".
std::string get_router_key(const std::string& router_path) { return router_path.substr(router_path.rfind('/') + 1); }

// A core's coordinates as [x, y].
json make_core_json(const CoreXY& core) { return json::array({core.x, core.y}); }

// ============ Writing ============

template <typename T>
json make_optional_json(const std::optional<T>& value) {
    return value.has_value() ? json(*value) : json(nullptr);
}

json make_manifest_identity_json(const ManifestIdentity& manifest) {
    json out;
    out["file"] = manifest.path.string();
    out["blake3"] = manifest.blake3;
    out["run_id"] = manifest.run_id;
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
    out["step"] = manifest::lower_enum_name(read.step);
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
    out["location"] = target.location.has_value() ? json(manifest::lower_enum_name(*target.location)) : json(nullptr);
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
    out["asic_id"] = manifest::make_asic_id_text(chip.asic_id);
    // Chip ids are never negative, so the record writes them unsigned, as it reads them.
    out["reader_chip_id"] = chip.chip.has_value() ? json(static_cast<uint32_t>(*chip.chip)) : json(nullptr);
    out["location"] = manifest::lower_enum_name(chip.location);
    json routers = json::object();
    for (const RouterRecord& router : chip.routers) {
        routers[get_router_key(router.target.router.path)] = make_router_record_json(router);
    }
    out["routers"] = std::move(routers);
    return out;
}

// ============ Reading ============

// An enum as manifest::lower_enum_name writes it.
template <typename E>
E read_enum(const JsonValueReader& value) {
    const std::string name = value.read<std::string>();
    const std::optional<E> parsed = manifest::parse_lower_enum_name<E>(name);
    if (!parsed.has_value()) {
        value.fail("is \"{}\", which names no value this reader knows", name);
    }
    return *parsed;
}

template <typename E>
std::optional<E> read_nullable_enum(const JsonValueReader& value) {
    return value.is_null() ? std::nullopt : std::optional(read_enum<E>(value));
}

// An ASIC id as manifest::make_asic_id_text writes it.
uint64_t read_asic_id(const JsonValueReader& value) {
    const std::string text = value.read<std::string>();
    const std::optional<uint64_t> asic_id = manifest::parse_asic_id_text(text);
    if (!asic_id.has_value()) {
        value.fail("is \"{}\", not an ASIC id written as 0x and 16 hex digits", text);
    }
    return *asic_id;
}

CoreXY read_core(const JsonValueReader& value) {
    const std::vector<uint32_t> coordinates = value.read_array_of<uint32_t>();
    if (coordinates.size() != 2) {
        value.fail("holds {} numbers, not an [x, y] pair", coordinates.size());
    }
    return CoreXY{.x = coordinates[0], .y = coordinates[1]};
}

std::optional<CoreXY> read_nullable_core(const JsonValueReader& value) {
    return value.is_null() ? std::nullopt : std::optional(read_core(value));
}

ManifestIdentity read_manifest_identity(const JsonValueReader& value) {
    return value.read_object([](JsonObjectReader& manifest_json) {
        return ManifestIdentity{
            .path = manifest_json.take<std::string>("file"),
            .blake3 = manifest_json.take<std::string>("blake3"),
            .run_id = manifest_json.take<std::string>("run_id"),
            .mpi_rank = manifest_json.take<uint32_t>("mpi_rank"),
            .world_size = manifest_json.take<uint32_t>("world_size"),
            .host_name = manifest_json.take<std::string>("host_name"),
            .git_sha = manifest_json.take<std::string>("git_sha"),
        };
    });
}

LivenessSample read_liveness_sample(const JsonValueReader& value) {
    return value.read_object([](JsonObjectReader& sample_json) {
        return LivenessSample{
            .round = sample_json.take<uint32_t>("round"),
            .t_ns = sample_json.take<uint64_t>("t_ns"),
            .heartbeat = sample_json.take("heartbeat").read_nullable<uint32_t>(),
            .core_registers = sample_json.take("core_registers").read_array_of<uint32_t>(),
            .error = sample_json.take("error").read_nullable<std::string>(),
        };
    });
}

ReadRecord read_read_record(const JsonValueReader& value) {
    return value.read_object([](JsonObjectReader& read_json) {
        return ReadRecord{
            .step = read_enum<ReadStep>(read_json.take("step")),
            .address = read_json.take<uint64_t>("address"),
            .size_bytes = read_json.take<uint32_t>("size_bytes"),
            .t_ns = read_json.take<uint64_t>("t_ns"),
            .bin_offset = read_json.take("bin_offset").read_nullable<uint64_t>(),
            .error = read_json.take("error").read_nullable<std::string>(),
        };
    });
}

// A router, at `path`: its chip's path and its key.
RouterRecord read_router_record(const JsonValueReader& value, std::string path) {
    return value.read_object([&path](JsonObjectReader& router_json) {
        RouterRecord router{
            .target =
                RouterTarget{
                    .router =
                        PlannedRouter{
                            .path = std::move(path),
                            .eth_chan = router_json.take<uint32_t>("eth_chan"),
                            .virtual_core = read_core(router_json.take("manifest_core")),
                            .stream_register_addresses =
                                router_json.take("stream_register_addresses").read_array_of<uint64_t>(),
                        },
                    .reader_core = read_nullable_core(router_json.take("reader_core")),
                    .location = read_nullable_enum<CoreLocation>(router_json.take("location")),
                    .discovery_skipped = router_json.take<bool>("discovery_skipped"),
                },
        };
        router_json.take("liveness").read_array([&router](const JsonValueReader& sample) {
            router.liveness.push_back(read_liveness_sample(sample));
        });
        router_json.take("reads").read_array(
            [&router](const JsonValueReader& read) { router.reads.push_back(read_read_record(read)); });
        return router;
    });
}

ChipRecord read_chip_record(const JsonValueReader& value, std::string path) {
    return value.read_object([&path](JsonObjectReader& chip_json) {
        ChipRecord chip{
            .path = std::move(path),
            .asic_id = read_asic_id(chip_json.take("asic_id")),
            .chip = chip_json.take("reader_chip_id").read_nullable<int32_t>(),
            .location = read_enum<ChipLocation>(chip_json.take("location")),
        };
        chip_json.take("routers").read_object([&chip](JsonObjectReader& routers_json) {
            routers_json.take_each([&chip](std::string_view key, const JsonValueReader& router) {
                chip.routers.push_back(read_router_record(router, chip.path + "/" + std::string(key)));
            });
        });
        return chip;
    });
}

}  // namespace

json make_capture_record_json(const CaptureRecord& record) {
    json out;
    out["capture_version"] = visualizer::k_capture_version;
    out["file_kind"] = std::string(visualizer::k_capture_file_kind);
    out["manifest"] = make_manifest_identity_json(record.manifest);

    json capture;
    capture["capture_id"] = record.capture_id;
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
            {{"asic_id", manifest::make_asic_id_text(core.asic_id)},
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
    return JsonValueReader(document, k_record_document_name).read_object([](JsonObjectReader& record_json) {
        record_json.check("capture_version", visualizer::k_capture_version);
        record_json.check("file_kind", std::string(visualizer::k_capture_file_kind));
        CaptureRecord record{.manifest = read_manifest_identity(record_json.take("manifest"))};
        record_json.take("capture").read_object([&record](JsonObjectReader& capture_json) {
            record.capture_id = capture_json.take<std::string>("capture_id");
            record.capture_git_sha = capture_json.take<std::string>("git_sha");
            record.cluster_id = capture_json.take("cluster_id").read_nullable<std::string>();
            record.started_at = capture_json.take<std::string>("started_at");
            record.finished_at = capture_json.take<std::string>("finished_at");
            record.liveness_options = capture_json.take("liveness").read_object([](JsonObjectReader& liveness_json) {
                return LivenessOptions{
                    .num_rounds = liveness_json.take<uint32_t>("num_rounds"),
                    .interval = std::chrono::milliseconds(liveness_json.take<uint64_t>("interval_ms")),
                };
            });
        });
        record_json.take("bin").read_object([&record](JsonObjectReader& bin_json) {
            record.bin_file = bin_json.take<std::string>("file");
            record.bin_size_bytes = bin_json.take<uint64_t>("size_bytes");
            record.bin_blake3 = bin_json.take<std::string>("blake3");
        });
        record_json.take("discovery_skipped_cores").read_array([&record](const JsonValueReader& core) {
            record.discovery_skipped_cores.push_back(core.read_object([](JsonObjectReader& core_json) {
                return DiscoverySkippedCore{
                    .asic_id = read_asic_id(core_json.take("asic_id")),
                    .core = read_core(core_json.take("core")),
                    .reason = core_json.take<std::string>("reason"),
                };
            }));
        });
        record_json.take("chips").read_object([&record](JsonObjectReader& chips_json) {
            chips_json.take_each([&record](std::string_view path, const JsonValueReader& chip) {
                record.chips.push_back(read_chip_record(chip, std::string(path)));
            });
        });
        return record;
    });
}

}  // namespace tt::tt_fabric::capture
