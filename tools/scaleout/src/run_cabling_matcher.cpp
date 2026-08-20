// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <cxxopts.hpp>
#include <fmt/format.h>
#include <fmt/ranges.h>

#include <cabling_matcher/cabling_matcher.hpp>

using namespace tt::scaleout_tools;
using namespace tt::scaleout_tools::matcher;

namespace {

struct Config {
    std::string cabling_path;
    std::string deployment_path;
    std::string pattern_cabling_path;
    std::string pattern_deployment_path;
    std::string pattern_template;
    TierScope tier = TierScope::Full;
    MatchOptions options;
};

template <typename T>
T parse_choice(const std::string& flag, const std::string& value, const std::map<std::string, T>& choices) {
    auto it = choices.find(value);
    if (it == choices.end()) {
        std::vector<std::string> names;
        for (const auto& [name, unused] : choices) {
            names.push_back(name);
        }
        throw std::invalid_argument(
            "Invalid --" + flag + " '" + value + "'; expected one of: " + fmt::format("{}", fmt::join(names, ", ")));
    }
    return it->second;
}

Config parse_arguments(int argc, char** argv) {
    cxxopts::Options options(
        "run_cabling_matcher",
        "Find a cabling scheme inside another cabling scheme.\n\n"
        "The pattern is a cabling descriptor, or one graph_template out of one, and the target is the\n"
        "descriptor to search. A match is an assignment of pattern hosts to target hosts under which\n"
        "every pattern cable is a cable the target actually has.");

    options.add_options()(
        "c,cabling",
        "Target cabling descriptor (.textproto file, or directory to merge)",
        cxxopts::value<std::string>()->default_value(""))(
        "d,deployment",
        "Target deployment descriptor (.textproto). Without it, target hosts are named host_0..host_N-1 and "
        "--cabling must be a single file",
        cxxopts::value<std::string>()->default_value(""))(
        "p,pattern-cabling",
        "Pattern cabling descriptor (.textproto). Defaults to --cabling, which asks whether a template of a "
        "descriptor recurs elsewhere in it",
        cxxopts::value<std::string>()->default_value(""))(
        "t,pattern-template",
        "graph_template within the pattern descriptor to use as the pattern. Without it, the descriptor's own "
        "root_instance is the pattern",
        cxxopts::value<std::string>()->default_value(""))(
        "pattern-deployment",
        "Deployment descriptor for the pattern; only valid without --pattern-template, since a template is "
        "instantiated with synthetic hosts",
        cxxopts::value<std::string>()->default_value(""))(
        "i,identity",
        "How strictly ports must correspond: strict (same port id), chip (same ASIC reached, treating a port "
        "that spans two ASICs as reaching either), relaxed (port ids ignored)",
        cxxopts::value<std::string>()->default_value("strict"))(
        "tray-symmetry",
        "fixed (a pattern tray must map to the same tray number) or any (search tray relabellings that "
        "preserve board types)",
        cxxopts::value<std::string>()->default_value("fixed"))(
        "m,mode",
        "contains (the pattern appears somewhere in the target) or exact (it also accounts for every target "
        "host and cable)",
        cxxopts::value<std::string>()->default_value("contains"))(
        "tier",
        "full (the pattern is every cable in the template's subtree) or own-level (only the cables the template "
        "declares itself)",
        cxxopts::value<std::string>()->default_value("full"))(
        "max-matches",
        "Distinct target host sets to report before stopping; 0 for no limit. Proving there are no further "
        "placements is the expensive part under --identity chip/relaxed, so pass 1 to ask a cheap does-it-fit "
        "question",
        cxxopts::value<size_t>()->default_value("16"))(
        "allow-disconnected",
        "Match each connected component of the pattern separately instead of refusing a pattern whose cables do "
        "not tie all its hosts together",
        cxxopts::value<bool>()->default_value("false")->implicit_value("true"))(
        "list-templates",
        "List the graph templates in the pattern descriptor and exit",
        cxxopts::value<bool>()->default_value("false")->implicit_value("true"))("h,help", "Print usage information");

    auto result = options.parse(argc, argv);
    if (result.contains("help") || argc == 1) {
        std::cout << options.help() << std::endl;
        std::cout << "\nExit status: 0 if the pattern matched, 1 if it did not, 2 on error, 3 if the search ran out "
                     "of budget without deciding.\n";
        std::cout << "\nExamples:\n";
        std::cout << "  " << argv[0] << " -c system.textproto -t bh_galaxy_sp\n";
        std::cout << "  # Does the superpod template recur elsewhere in the system it came from?\n\n";
        std::cout << "  " << argv[0] << " -c st_sc36.textproto -p sc36.textproto -t bh_glx_pod --identity chip\n";
        std::cout << "  # Does an SC36 pod sit inside an ST-SC36 system, comparing chips rather than port ids?\n";
        exit(0);
    }

    Config config;
    config.cabling_path = result["cabling"].as<std::string>();
    config.deployment_path = result["deployment"].as<std::string>();
    config.pattern_cabling_path = result["pattern-cabling"].as<std::string>();
    config.pattern_deployment_path = result["pattern-deployment"].as<std::string>();
    config.pattern_template = result["pattern-template"].as<std::string>();

    // Listing what a descriptor offers is a question about one file, so it needs neither a target to
    // search nor a template to look for.
    if (result["list-templates"].as<bool>()) {
        const std::string& path =
            config.pattern_cabling_path.empty() ? config.cabling_path : config.pattern_cabling_path;
        if (path.empty()) {
            throw std::invalid_argument(
                "--list-templates needs a descriptor, given with --cabling or --pattern-cabling");
        }
        for (const auto& name : list_graph_templates(path)) {
            std::cout << name << std::endl;
        }
        exit(0);
    }

    if (config.cabling_path.empty()) {
        throw std::invalid_argument("--cabling is required");
    }
    if (config.pattern_cabling_path.empty()) {
        config.pattern_cabling_path = config.cabling_path;
        if (config.pattern_template.empty()) {
            throw std::invalid_argument(
                "Without --pattern-cabling the pattern comes from --cabling, which then needs "
                "--pattern-template to say which part of it to look for");
        }
    }
    for (const auto& path : {config.cabling_path, config.pattern_cabling_path}) {
        if (!std::filesystem::exists(path)) {
            throw std::invalid_argument("Cabling descriptor path not found: '" + path + "'");
        }
    }

    config.options.port_identity = parse_choice<PortIdentity>(
        "identity",
        result["identity"].as<std::string>(),
        {{"strict", PortIdentity::Strict}, {"chip", PortIdentity::Chip}, {"relaxed", PortIdentity::Relaxed}});
    config.options.tray_symmetry = parse_choice<TraySymmetry>(
        "tray-symmetry",
        result["tray-symmetry"].as<std::string>(),
        {{"fixed", TraySymmetry::None}, {"any", TraySymmetry::Full}});
    config.options.mode = parse_choice<MatchMode>(
        "mode", result["mode"].as<std::string>(), {{"contains", MatchMode::Contains}, {"exact", MatchMode::Exact}});
    config.tier = parse_choice<TierScope>(
        "tier", result["tier"].as<std::string>(), {{"full", TierScope::Full}, {"own-level", TierScope::OwnLevel}});
    config.options.max_matches = result["max-matches"].as<size_t>();
    config.options.allow_disconnected = result["allow-disconnected"].as<bool>();
    return config;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Config config = parse_arguments(argc, argv);

        std::string pattern_label = config.pattern_cabling_path;
        if (!config.pattern_template.empty()) {
            pattern_label += " :: " + config.pattern_template;
        }
        if (config.tier == TierScope::OwnLevel) {
            pattern_label += " (own level only)";
        }

        MatchGraph pattern = MatchGraph::load(
            config.pattern_cabling_path,
            config.pattern_deployment_path,
            config.pattern_template,
            config.tier,
            pattern_label);
        MatchGraph target =
            MatchGraph::load(config.cabling_path, config.deployment_path, "", TierScope::Full, config.cabling_path);

        MatchResult result = match(pattern, target, config.options);
        std::cout << format_result(pattern, target, result, config.options);
        if (result.matched) {
            return 0;
        }
        return result.inconclusive() ? 3 : 1;
    } catch (const cxxopts::exceptions::exception& e) {
        std::cerr << "Error parsing arguments: " << e.what() << std::endl;
        return 2;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 2;
    }
}
