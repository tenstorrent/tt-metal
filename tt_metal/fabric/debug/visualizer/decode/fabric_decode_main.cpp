// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// fabric_decode: decodes one rank's capture against its manifest, and writes the rank's decoded file. It opens no
// chip, so it runs anywhere the two files are, on the same build that wrote the manifest.
//
// Exit status: 0 when every router was read in full and every region decoded, 1 when the decoded file was written
// with something missing or wrong, which its findings say, 2 when it failed and wrote nothing.

#include <algorithm>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>

#include <cxxopts.hpp>

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

int run_decode(const cxxopts::ParseResult& options) {
    const decode::DecodeRequest request{
        .manifest_path = options["manifest"].as<std::string>(),
        .capture_record_path = options["capture"].as<std::string>(),
    };
    const decode::DecodeInputs inputs = decode::load_decode_inputs(request);
    // Beside the capture record, under the rank's decoded file name, unless --output says otherwise.
    const tt::tt_fabric::manifest::RunInfo& run = inputs.manifest.facts.run;
    const std::filesystem::path output_path =
        options.contains("output")
            ? std::filesystem::path(options["output"].as<std::string>())
            : request.capture_record_path.parent_path() /
                  visualizer::make_decoded_file_name(run.mpi_rank, run.world_size);
    std::cerr << "fabric_decode: decoding " << request.capture_record_path.string() << " against "
              << request.manifest_path.string() << "\n";

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
    parser.add_options()("manifest", "This rank's manifest", cxxopts::value<std::string>())(
        "capture", "This rank's capture record. Its .bin is beside it", cxxopts::value<std::string>())(
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
        for (const char* required : {"manifest", "capture"}) {
            if (!options.contains(required)) {
                std::cerr << "fabric_decode: --" << required << " is required; see --help\n";
                return k_exit_failed;
            }
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
