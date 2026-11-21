// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// fabric_decode: decodes one rank's capture against its manifest, and writes the rank's decoded file. It opens no
// chip, so it runs anywhere the files are, on the same build that wrote the manifest. Given no capture record, it
// finds this rank's the way capture finds its manifest, so the run's tt-run command finds each rank's capture. The
// manifest is the one the capture record names, unless --manifest gives another.
//
// Exit status: 0 when every router was read in full and every region decoded, 1 when the decoded file was written
// with something missing or wrong, which its findings say, 2 when it failed and wrote nothing.

#include <algorithm>
#include <exception>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>

#include <cxxopts.hpp>
#include <tt_stl/assert.hpp>

#include "tt_metal/fabric/debug/visualizer/common/fabric_visualizer_build_identity.hpp"
#include "tt_metal/fabric/debug/visualizer/common/fabric_visualizer_files.hpp"
#include "tt_metal/fabric/debug/visualizer/common/fabric_visualizer_time.hpp"
#include "tt_metal/fabric/debug/visualizer/decode/fabric_decode_findings.hpp"
#include "tt_metal/fabric/debug/visualizer/decode/fabric_decode_inputs.hpp"
#include "tt_metal/fabric/debug/visualizer/decode/fabric_decode_rank.hpp"

namespace {

namespace decode = tt::tt_fabric::decode;
namespace visualizer = tt::tt_fabric::visualizer;

constexpr int k_exit_complete = 0;
constexpr int k_exit_findings_recorded = 1;
constexpr int k_exit_failed = 2;

// The build this tool is, and the time now.
decode::DecodeIdentity make_decode_identity() {
    const visualizer::BuildIdentity build = visualizer::get_build_identity();
    return decode::DecodeIdentity{
        .git_sha = std::string(build.git_sha),
        .git_dirty = build.git_dirty,
        .version = std::string(build.version),
        .build_type = std::string(build.build_type),
        .decoded_at = visualizer::get_utc_now_iso8601(),
    };
}

// Whether an error's message is already in the log.
bool was_logged(const std::exception& error) {
    const std::string_view what = error.what();
    return what.starts_with("TT_FATAL @") || what.starts_with("TT_THROW @");
}

// How many of the rank's findings have `severity`.
size_t count_findings(const decode::DecodedRank& rank, decode::FindingSeverity severity) {
    return std::ranges::count_if(rank.findings, [&](const decode::Finding& finding) {
        return decode::get_finding_severity(finding.code) == severity;
    });
}

// The capture record to decode: --capture's, else this rank's record of --capture-id, else of its latest capture, in
// the newest run under the logs folder that has one.
std::filesystem::path find_capture_record(const cxxopts::ParseResult& options) {
    if (options.contains("capture")) {
        return options["capture"].as<std::string>();
    }
    const std::filesystem::path logs_dir = visualizer::get_logs_directory();
    const visualizer::JobRank rank = visualizer::get_job_rank();
    const std::string record_file_name = visualizer::make_capture_record_file_name(rank.mpi_rank, rank.world_size);
    if (options.contains("capture-id")) {
        const std::string capture_id = options["capture-id"].as<std::string>();
        const std::optional<std::filesystem::path> record_path =
            visualizer::find_capture_record(logs_dir, rank, capture_id);
        TT_FATAL(
            record_path.has_value(),
            "Fabric decode: no run under {} has capture {} with {}; give the record with --capture",
            visualizer::make_runs_directory(logs_dir).string(),
            capture_id,
            record_file_name);
        return *record_path;
    }
    const std::optional<std::filesystem::path> record_path = visualizer::find_latest_capture_record(logs_dir, rank);
    if (!record_path.has_value()) {
        const std::string manifest_file_name = visualizer::make_manifest_file_name(rank.mpi_rank, rank.world_size);
        const std::optional<std::filesystem::path> manifest_path = visualizer::find_latest_manifest(logs_dir, rank);
        TT_FATAL(
            manifest_path.has_value(),
            "Fabric decode: no run under {} has {}; give the record with --capture",
            visualizer::make_runs_directory(logs_dir).string(),
            manifest_file_name);
        TT_THROW(
            "Fabric decode: the newest run with {}, {}, has no capture with {}; give the record with --capture",
            manifest_file_name,
            manifest_path->parent_path().string(),
            record_file_name);
    }
    return *record_path;
}

int run_decode(const cxxopts::ParseResult& options) {
    const decode::DecodeRequest request{
        .capture_record_path = find_capture_record(options),
        .manifest_path = options.contains("manifest")
                             ? std::optional<std::filesystem::path>(options["manifest"].as<std::string>())
                             : std::nullopt,
    };
    const decode::DecodeInputs inputs = decode::load_decode_inputs(request);
    // Beside the capture record, under the rank's decoded file name, unless --output says otherwise.
    const tt::tt_fabric::manifest::RunInfo& run = inputs.manifest.facts.run;
    const std::filesystem::path output_path =
        options.contains("output") ? std::filesystem::path(options["output"].as<std::string>())
                                   : request.capture_record_path.parent_path() /
                                         visualizer::make_decoded_file_name(run.mpi_rank, run.world_size);
    std::cerr << "fabric_decode: decoding " << request.capture_record_path.string() << " against "
              << inputs.manifest.path.string() << "\n";

    const decode::DecodedRank rank = decode::decode_rank(request, inputs, make_decode_identity());
    decode::write_decoded_file(
        output_path, decode::make_decoded_json(rank, inputs.manifest.facts), options["replace"].as<bool>());

    std::cerr << "fabric_decode: wrote " << output_path.string() << ": " << rank.routers.size() << " routers, "
              << rank.findings.size() << " findings (" << count_findings(rank, decode::FindingSeverity::ERROR)
              << " errors, " << count_findings(rank, decode::FindingSeverity::WARNING) << " warnings)\n";
    if (!rank.is_complete()) {
        std::cerr << "fabric_decode: not everything was read and decoded; the file's analysis.findings say what\n";
        return k_exit_findings_recorded;
    }
    return k_exit_complete;
}

}  // namespace

int main(int argc, char** argv) {
    cxxopts::Options parser("fabric_decode", "Decodes one rank's fabric capture against its manifest.");
    parser.add_options()(
        "capture",
        "This rank's capture record. Its .bin is beside it. Defaults to this rank's record of --capture-id, else of "
        "the newest capture of the newest run that has this rank's manifest, under TT_METAL_LOGS_PATH (else the "
        "working folder), for its rank from TT_RUN_RANK or OMPI_COMM_WORLD_RANK",
        cxxopts::value<std::string>())(
        "capture-id",
        "The capture to decode, when --capture isn't given. Its record is found in the newest run that has it",
        cxxopts::value<std::string>())(
        "manifest",
        "The manifest the capture was taken from. Defaults to the one the capture record names, in the run's folder "
        "the record is in",
        cxxopts::value<std::string>())(
        "output",
        "Where to write the decoded file. Defaults to beside the capture record, as "
        "fabric_decoded_rank_<r>_of_<n>.json for the manifest's rank",
        cxxopts::value<std::string>())(
        "replace", "Replace the decoded file if it's already there", cxxopts::value<bool>()->default_value("false"))(
        "h,help", "Print this help");

    try {
        const cxxopts::ParseResult options = parser.parse(argc, argv);
        if (options.contains("help")) {
            std::cout << parser.help() << "\n";
            return k_exit_complete;
        }
        if (options.contains("capture") && options.contains("capture-id")) {
            std::cerr << "fabric_decode: give --capture or --capture-id, not both; see --help\n";
            return k_exit_failed;
        }
        return run_decode(options);
    } catch (const cxxopts::exceptions::exception& error) {
        std::cerr << "fabric_decode: " << error.what() << "; see --help\n";
        return k_exit_failed;
    } catch (const std::exception& error) {
        // Don't duplicate logged errors.
        if (was_logged(error)) {
            std::cerr << "fabric_decode: failed, and wrote no decoded file\n";
        } else {
            std::cerr << "fabric_decode: failed, and wrote no decoded file: " << error.what() << "\n";
        }
        return k_exit_failed;
    }
}
