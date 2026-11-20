// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// SPEC-0014, per-chip SoC descriptors inside metal: the refusals E0 to E7 of § Contract on Mock (§ Verification 3),
// and that a chip's descriptor and the package's give a core one translated coordinate (§ Verification 4).
//
// Metal reads its environment once a process, so each Mock case opens metal in a child process of its own (a
// threadsafe death test). The child generates the descriptors, changes one thing, exports the variables, opens, and
// exits with an Outcome; the parent's environment is not changed.

#include <gtest/gtest.h>

#include <yaml-cpp/yaml.h>

#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <llrt/tt_cluster.hpp>
#include <tt-umd/soc_arch_descriptor.hpp>
#include <tt-umd/soc_descriptor.hpp>
#include <tt-umd/types/cluster_descriptor_types.hpp>
#include <tt-umd/types/core_coordinates.hpp>

#include "impl/context/metal_context.hpp"
#include "tt_metal/tt_fabric/keraunos/vcluster/descriptors.hpp"
#include "tt_metal/tt_fabric/keraunos/vcluster/manifest.hpp"

namespace tt::tt_fabric::vcluster_metal_desc_tests {
namespace {

namespace fs = std::filesystem;
namespace vc = tt::tt_fabric::vcluster;

const fs::path kVclusterDir = fs::path(__FILE__).parent_path();

// The package descriptor, UMD descriptors/soc_descs/mmk.yaml; VCLUSTER_PACKAGE_SOC_DESC overrides it.
fs::path package_soc_desc() {
    if (const char* path = std::getenv("VCLUSTER_PACKAGE_SOC_DESC"); path != nullptr && *path != '\0') {
        return path;
    }
    return kVclusterDir.parent_path().parent_path().parent_path().parent_path().parent_path() /
           "tt_metal/third_party/umd/descriptors/soc_descs/mmk.yaml";
}

fs::path temp_dir(const std::string& name) {
    return fs::path(::testing::TempDir()) / ("vcluster_metal_desc_" + std::to_string(::getpid()) + "_" + name);
}

// The files one case opens metal on. `package` is what TT_METAL_EMU_PACKAGE_SOC_DESC names; empty leaves it unset.
struct Files {
    fs::path dir;
    std::vector<fs::path> socs;  // TT_METAL_EMU_SOC_DESC, in order
    fs::path cluster;
    fs::path mesh_graph;
    fs::path package;
};

Files generate(const std::string& manifest, const fs::path& dir) {
    const vc::GeneratedDescriptors g =
        vc::generate_descriptors(vc::load_manifest(kVclusterDir / "manifests" / manifest), package_soc_desc(), dir);
    return Files{
        .dir = dir,
        .socs = g.soc_descs,
        .cluster = g.cluster_desc,
        .mesh_graph = g.mesh_graph_desc,
        .package = fs::absolute(package_soc_desc())};
}

void edit(const fs::path& file, const std::function<void(YAML::Node&)>& change) {
    YAML::Node node = YAML::LoadFile(file.string());
    change(node);
    std::ofstream out(file);
    out << node << "\n";
    if (!out.good()) {
        throw std::runtime_error("cannot write " + file.string());
    }
}

// A copy of the package descriptor in the case's directory, changed, and named as the package.
void edit_package_copy(Files& files, const std::function<void(YAML::Node&)>& change) {
    const fs::path copy = files.dir / "package_soc_desc.yaml";
    fs::copy_file(files.package, copy, fs::copy_options::overwrite_existing);
    edit(copy, change);
    files.package = copy;
}

// `sequence` without the items `drop` holds for.
YAML::Node without(const YAML::Node& sequence, const std::function<bool(const YAML::Node&)>& drop) {
    YAML::Node kept(YAML::NodeType::Sequence);
    for (const auto& item : sequence) {
        if (!drop(item)) {
            kept.push_back(item);
        }
    }
    return kept;
}

// The package's keraunos.cores entry for `core`.
YAML::Node package_keraunos_entry(const std::string& core) {
    for (const auto& entry : YAML::LoadFile(package_soc_desc().string())["keraunos"]["cores"]) {
        if (entry["core"].as<std::string>() == core) {
            return YAML::Clone(entry);
        }
    }
    throw std::runtime_error("the package has no keraunos.cores entry for " + core);
}

// Chip `from` becomes chip `to` everywhere in a cluster descriptor of the generator's shape.
void renumber_chip(YAML::Node& cluster, uint32_t from, uint32_t to) {
    const std::string from_key = std::to_string(from);
    const std::string to_key = std::to_string(to);
    for (const char* by_chip : {"arch", "chip_unique_ids", "harvesting", "chip_to_boardtype", "asic_locations"}) {
        YAML::Node map = cluster[by_chip];
        if (!map.IsMap() || !map[from_key]) {
            throw std::runtime_error(std::string("cluster descriptor: no chip ") + from_key + " in " + by_chip);
        }
        const YAML::Node value = YAML::Clone(map[from_key]);
        map.remove(from_key);
        map[to_key] = value;
    }
    for (auto connection : cluster["ethernet_connections"]) {
        for (auto end : connection) {
            if (end["chip"].as<uint32_t>() == from) {
                end["chip"] = to;
            }
        }
    }
    YAML::Node mmio(YAML::NodeType::Sequence);
    for (const auto& entry : cluster["chips_with_mmio"]) {
        YAML::Node renamed(YAML::NodeType::Map);
        for (const auto& kv : entry) {
            renamed[kv.first.as<uint32_t>() == from ? to_key : kv.first.Scalar()] = kv.second;
        }
        mmio.push_back(renamed);
    }
    cluster["chips_with_mmio"] = mmio;
    for (auto board : cluster["boards"]) {
        for (auto field : board) {
            if (!field["chips"]) {
                continue;
            }
            YAML::Node chips(YAML::NodeType::Sequence);
            for (const auto& chip : field["chips"]) {
                chips.push_back(chip.as<uint32_t>() == from ? to : chip.as<uint32_t>());
            }
            field["chips"] = chips;
        }
    }
}

enum Outcome : int { kOpened = 0, kRefused = 3, kWrongView = 4, kSetupFailed = 5 };

// The Mock environment of SPEC-0014 § Contract, Environment: no emulator server, the cluster descriptor in the Mock
// variable, the list when `files.socs` is not empty, and the package variable when `files.package` is not empty.
void export_mock(const Files& files) {
    for (const char* unset :
         {"TT_METAL_EMU_SERVER", "TT_METAL_EMU_CLUSTER_DESC", "TT_METAL_SIMULATOR", "TT_METAL_EMU_PACKAGE_SOC_DESC"}) {
        ::unsetenv(unset);
    }
    std::string socs;
    for (const auto& soc : files.socs) {
        socs += (socs.empty() ? "" : ",") + soc.string();
    }
    ::setenv("TT_METAL_MOCK_CLUSTER_DESC_PATH", files.cluster.c_str(), 1);
    if (socs.empty()) {
        ::unsetenv("TT_METAL_EMU_SOC_DESC");
    } else {
        ::setenv("TT_METAL_EMU_SOC_DESC", socs.c_str(), 1);
    }
    ::setenv("TT_MESH_GRAPH_DESC_PATH", files.mesh_graph.c_str(), 1);
    if (!files.package.empty()) {
        ::setenv("TT_METAL_EMU_PACKAGE_SOC_DESC", files.package.c_str(), 1);
    }
}

using Change = std::function<void(Files&)>;
using View = std::function<bool(const tt::Cluster&)>;

// Runs in the death test's child and ends it. Generates `manifest`'s descriptors, applies `change`, opens metal on
// Mock, and exits kRefused with "REFUSED: <the exception's text on one line>" on stderr when the open throws, kOpened
// when it opens and `view` (if given) holds of the cluster, kWrongView when it opens and `view` does not hold, and
// kSetupFailed when anything before the open throws.
[[noreturn]] void open_on_mock(
    const Change& change, const View& view = {}, const std::string& manifest = "line3_eth.yaml") {
    int outcome = kSetupFailed;
    const fs::path dir = temp_dir("case");
    try {
        fs::remove_all(dir);
        Files files = generate(manifest, dir);
        change(files);
        export_mock(files);
        try {
            const tt::Cluster& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
            std::cerr << "\nMETAL OPENED\n";
            outcome = kOpened;
            try {
                if (view && !view(cluster)) {
                    outcome = kWrongView;
                }
            } catch (const std::exception& e) {
                std::cerr << "VIEW THREW: " << e.what() << "\n";
                outcome = kWrongView;
            }
        } catch (const std::exception& e) {
            std::string text = e.what();
            for (char& c : text) {
                if (c == '\n' || c == '\r') {
                    c = ' ';
                }
            }
            std::cerr << "\nREFUSED: " << text << "\n";
            outcome = kRefused;
        }
    } catch (const std::exception& e) {
        std::cerr << "\nSETUP FAILED: " << e.what() << "\n";
    }
    std::error_code ignored;
    fs::remove_all(dir, ignored);
    std::cerr.flush();
    std::_Exit(outcome);
}

class VclusterMetalDescDeathTest : public ::testing::Test {
protected:
    void SetUp() override { GTEST_FLAG_SET(death_test_style, "threadsafe"); }
};

#define EXPECT_REFUSED(change, message) \
    EXPECT_EXIT(open_on_mock(change), ::testing::ExitedWithCode(kRefused), std::string("REFUSED: .*") + (message))
#define EXPECT_OPENS(...) EXPECT_EXIT(open_on_mock(__VA_ARGS__), ::testing::ExitedWithCode(kOpened), "METAL OPENED")

std::string core_at(const tt::Cluster& cluster, uint32_t chip, uint32_t channel) {
    const auto core = cluster.get_soc_desc(chip).get_eth_core_for_channel(channel, tt::CoordSystem::NOC0);
    return std::to_string(core.x) + "-" + std::to_string(core.y);
}

// --- The generated files open, so each refusal below is the one change's -----------------------------------------

// § Contract, Environment, and § Design: with the list and the package variable metal's view a chip is Phase 2's.
// § Why it is sound: "chip 1's channel 0 is 4-1 to metal".
TEST_F(VclusterMetalDescDeathTest, GeneratedFilesOpen) {
    for (const std::string manifest : {"line3_eth.yaml", "line3_am.yaml", "line3_noc.yaml"}) {
        SCOPED_TRACE(manifest);
        EXPECT_OPENS(
            [](Files&) {},
            [](const tt::Cluster& cluster) {
                return core_at(cluster, 0, 0) == "4-0" && core_at(cluster, 1, 0) == "4-1" &&
                       core_at(cluster, 1, 1) == "4-2" && core_at(cluster, 2, 0) == "4-3" &&
                       cluster.get_soc_desc(1).get_num_eth_channels() == 2;
            },
            manifest);
    }
}

// § Contract, Environment: one entry and no package variable is the package descriptor for every chip.
TEST_F(VclusterMetalDescDeathTest, OneEntryNoPackageVariableOpens) {
    EXPECT_OPENS(
        [](Files& files) {
            files.socs = {files.package};
            files.package.clear();
        },
        [](const tt::Cluster& cluster) {
            return cluster.get_soc_desc(0).get_num_eth_channels() == 4 &&
                   cluster.get_soc_desc(2).get_num_eth_channels() == 4 && core_at(cluster, 1, 0) == "4-0";
        });
}

// --- § Contract, Refusals, and § Verification 3 ------------------------------------------------------------------

// E0, first condition: Phase 2's environment, the list with no package variable.
TEST_F(VclusterMetalDescDeathTest, E0ListWithoutPackageVariable) {
    EXPECT_REFUSED(
        [](Files& files) { files.package.clear(); },
        "TT_METAL_EMU_SOC_DESC lists 3 descriptors: TT_METAL_EMU_PACKAGE_SOC_DESC must name the package descriptor");
}

// E0, second condition: the package variable with one entry in the list.
TEST_F(VclusterMetalDescDeathTest, E0PackageVariableWithOneEntry) {
    EXPECT_REFUSED(
        [](Files& files) { files.socs = {files.package}; },
        "TT_METAL_EMU_PACKAGE_SOC_DESC is set and TT_METAL_EMU_SOC_DESC has one entry");
}

// E1: four entries and three chips open. The fourth lists no core and no view, so E5 and E6 hold nothing against it.
TEST_F(VclusterMetalDescDeathTest, E1ListLongerThanTheChips) {
    EXPECT_REFUSED(
        [](Files& files) {
            const fs::path extra = files.dir / "chip_3_soc_desc.yaml";
            fs::copy_file(files.socs[0], extra);
            edit(extra, [](YAML::Node& soc) {
                soc["eth"] = YAML::Node(YAML::NodeType::Sequence);
                soc["keraunos"]["cores"] = YAML::Node(YAML::NodeType::Sequence);
                soc["dram_views"] = YAML::Node(YAML::NodeType::Sequence);
            });
            files.socs.push_back(extra);
        },
        "TT_METAL_EMU_SOC_DESC lists 4 chip descriptors and 3 chips are open");
}

// E2: three entries and three chips open, with ids 0, 1 and 3.
TEST_F(VclusterMetalDescDeathTest, E2ChipIdBeyondTheList) {
    EXPECT_REFUSED(
        [](Files& files) { edit(files.cluster, [](YAML::Node& cluster) { renumber_chip(cluster, 2, 3); }); },
        "TT_METAL_EMU_SOC_DESC has no descriptor for chip 3");
}

// E3: chip 1's Ethernet harvesting mask in the cluster descriptor is 1. On Mock UMD refuses the mask when it builds
// its own descriptor of the chip, before metal's check, so the text is UMD's and E3's own is not reached here.
TEST_F(VclusterMetalDescDeathTest, E3EthernetHarvestingMask) {
    EXPECT_REFUSED(
        [](Files& files) {
            edit(files.cluster, [](YAML::Node& cluster) { cluster["harvesting"]["1"]["eth_harvesting_mask"] = 1; });
        },
        "ETH harvesting is supported only for Blackhole");
}

std::string e4(const std::string& key) {
    return "Chip 1 descriptor [^ ]*chip_1_soc_desc\\.yaml differs from the package descriptor [^ ]*mmk\\.yaml in key " +
           key;
}

Change in_chip(size_t chip, const std::function<void(YAML::Node&)>& change) {
    return [chip, change](Files& files) { edit(files.socs.at(chip), change); };
}

// E4: a top-level key other than eth, keraunos and dram_views differs from the package's.
TEST_F(VclusterMetalDescDeathTest, E4TopLevelKeyDiffers) {
    EXPECT_REFUSED(in_chip(1, [](YAML::Node& soc) { soc["eth_l1_size"] = 2097152; }), e4("eth_l1_size"));
}

// E4: a key in the chip's file and not in the package's.
TEST_F(VclusterMetalDescDeathTest, E4KeyOnlyInTheChipsFile) {
    EXPECT_REFUSED(in_chip(1, [](YAML::Node& soc) { soc["spare_key"] = 1; }), e4("spare_key"));
}

// E4: a key in the package's file and not in the chip's.
TEST_F(VclusterMetalDescDeathTest, E4KeyOnlyInThePackage) {
    EXPECT_REFUSED(in_chip(1, [](YAML::Node& soc) { soc.remove("dram_view_size"); }), e4("dram_view_size"));
}

// E4: a key of keraunos other than cores differs, and is named keraunos.<key>.
TEST_F(VclusterMetalDescDeathTest, E4KeraunosKeyDiffers) {
    EXPECT_REFUSED(
        in_chip(1, [](YAML::Node& soc) { soc["keraunos"]["tile_stride"] = 1048576; }), e4("keraunos\\.tile_stride"));
}

// E4: each of eth, keraunos and dram_views is in both files or in neither.
TEST_F(VclusterMetalDescDeathTest, E4DramViewsInOneFileOnly) {
    EXPECT_REFUSED(in_chip(1, [](YAML::Node& soc) { soc.remove("dram_views"); }), e4("dram_views"));
}

// E5: an eth core the package's eth lacks.
TEST_F(VclusterMetalDescDeathTest, E5EthernetCoreNotThePackages) {
    EXPECT_REFUSED(
        in_chip(0, [](YAML::Node& soc) { soc["eth"].push_back("5-0"); }),
        "Chip 0 descriptor [^ ]*chip_0_soc_desc\\.yaml: Ethernet core 5-0 is not the package descriptor's");
}

// E5, § Verification 3: a chip's keraunos.cores entry with another hsio.
TEST_F(VclusterMetalDescDeathTest, E5KeraunosEntryWithAnotherHsio) {
    EXPECT_REFUSED(
        in_chip(
            1,
            [](YAML::Node& soc) {
                for (auto entry : soc["keraunos"]["cores"]) {
                    if (entry["core"].as<std::string>() == "4-1") {
                        entry["hsio"] = 3;
                    }
                }
            }),
        "Chip 1 descriptor [^ ]*chip_1_soc_desc\\.yaml: keraunos\\.cores entry \\{core: 4-1, hsio: 3, eth_cce: 0\\} is "
        "not the package descriptor's");
}

// E5: a keraunos.cores entry, the package's own, for a core the chip's eth lacks.
TEST_F(VclusterMetalDescDeathTest, E5KeraunosEntryForACoreEthLacks) {
    EXPECT_REFUSED(
        in_chip(0, [](YAML::Node& soc) { soc["keraunos"]["cores"].push_back(package_keraunos_entry("4-1")); }),
        "Chip 0 descriptor [^ ]*chip_0_soc_desc\\.yaml: keraunos\\.cores entry \\{core: 4-1, hsio: 1, eth_cce: 0\\} names "
        "a core its eth list lacks");
}

// E5: a dram_views entry that equals no entry of the package's list.
TEST_F(VclusterMetalDescDeathTest, E5DramViewNotThePackages) {
    EXPECT_REFUSED(
        in_chip(0, [](YAML::Node& soc) { soc["dram_views"][0]["address_offset"] = 4096; }),
        "Chip 0 descriptor [^ ]*chip_0_soc_desc\\.yaml: DRAM view \\{.*address_offset: 4096.*\\} is not the package "
        "descriptor's");
}

// E6, § Verification 3: two chips list one Ethernet core, each with the package's keraunos.cores entry for it.
TEST_F(VclusterMetalDescDeathTest, E6EthernetCoreInTwoChips) {
    EXPECT_REFUSED(
        in_chip(
            0,
            [](YAML::Node& soc) {
                soc["eth"].push_back("4-1");
                soc["keraunos"]["cores"].push_back(package_keraunos_entry("4-1"));
            }),
        "Chips 0 and 1 both list Ethernet core 4-1");
}

// E6: two chips list one programmable DRAM view.
TEST_F(VclusterMetalDescDeathTest, E6ProgrammableDramViewInTwoChips) {
    EXPECT_REFUSED(
        [](Files& files) {
            const YAML::Node view = YAML::Clone(YAML::LoadFile(files.socs[1].string())["dram_views"][0]);
            if (!view["programmable"].as<bool>()) {
                throw std::runtime_error("chip 1's first DRAM view is not programmable");
            }
            edit(files.socs[0], [&](YAML::Node& soc) { soc["dram_views"].push_back(view); });
        },
        "Chips 0 and 1 both list DRAM view \\{");
}

// E6: "One in no chip's file is allowed", an Ethernet core. Chip 1 gives up 4-2 and the link that used it.
TEST_F(VclusterMetalDescDeathTest, E6EthernetCoreInNoChipOpens) {
    EXPECT_OPENS(
        [](Files& files) {
            edit(files.socs[1], [](YAML::Node& soc) {
                soc["eth"] = without(soc["eth"], [](const YAML::Node& core) { return core.Scalar() == "4-2"; });
                soc["keraunos"]["cores"] = without(
                    soc["keraunos"]["cores"], [](const YAML::Node& entry) { return entry["core"].Scalar() == "4-2"; });
            });
            edit(files.cluster, [](YAML::Node& cluster) {
                cluster["ethernet_connections"] =
                    without(cluster["ethernet_connections"], [](const YAML::Node& connection) {
                        return connection[0]["chip"].as<uint32_t>() == 2 || connection[1]["chip"].as<uint32_t>() == 2;
                    });
            });
        },
        [](const tt::Cluster& cluster) {
            return cluster.get_soc_desc(1).get_num_eth_channels() == 1 && core_at(cluster, 1, 0) == "4-1";
        });
}

// E6: "One in no chip's file is allowed", a programmable DRAM view. The chip that holds two gives one up.
TEST_F(VclusterMetalDescDeathTest, E6ProgrammableDramViewInNoChipOpens) {
    EXPECT_OPENS([](Files& files) {
        for (const auto& soc_file : files.socs) {
            if (YAML::LoadFile(soc_file.string())["dram_views"].size() < 2) {
                continue;
            }
            edit(soc_file, [](YAML::Node& soc) {
                bool first = true;
                soc["dram_views"] =
                    without(soc["dram_views"], [&](const YAML::Node&) { return std::exchange(first, false); });
            });
            return;
        }
        throw std::runtime_error("no chip's descriptor has two DRAM views");
    });
}

// E6 names a programmable DRAM view: one that is not programmable, the package's, may be in two chips' files.
TEST_F(VclusterMetalDescDeathTest, E6NonProgrammableDramViewInTwoChipsOpens) {
    EXPECT_OPENS([](Files& files) {
        YAML::Node view = YAML::Clone(YAML::LoadFile(files.package.string())["dram_views"][0]);
        view["address_offset"] = 536870912;
        view["programmable"] = false;
        edit_package_copy(files, [&](YAML::Node& package) { package["dram_views"].push_back(view); });
        for (const size_t chip : {0, 1}) {
            edit(files.socs[chip], [&](YAML::Node& soc) { soc["dram_views"].push_back(view); });
        }
    });
}

// --- § Contract, Refusals, "Further conditions, from the code review" ---------------------------------------------

// E0 on Mock: the package variable set and the list unset.
TEST_F(VclusterMetalDescDeathTest, E0PackageVariableWithoutTheList) {
    EXPECT_REFUSED(
        [](Files& files) { files.socs.clear(); },
        "TT_METAL_EMU_PACKAGE_SOC_DESC is set and TT_METAL_EMU_SOC_DESC is not( |$)");
}

// E4's message for chip 1 with `key` the whole key named, not a prefix of it.
std::string e4_kind(const std::string& key) { return e4(key) + "( |$)"; }

YAML::Node a_map() {
    YAML::Node map(YAML::NodeType::Map);
    map["a"] = 1;
    return map;
}

YAML::Node a_sequence() {
    YAML::Node sequence(YAML::NodeType::Sequence);
    sequence.push_back("a");
    return sequence;
}

// E4, node kinds: each of these is refused with E4's text and never as a yaml-cpp exception.
TEST_F(VclusterMetalDescDeathTest, E4EthIsAMap) {
    EXPECT_REFUSED(in_chip(1, [](YAML::Node& soc) { soc["eth"] = a_map(); }), e4_kind("eth"));
}

TEST_F(VclusterMetalDescDeathTest, E4EthIsAScalar) {
    EXPECT_REFUSED(in_chip(1, [](YAML::Node& soc) { soc["eth"] = "4-1"; }), e4_kind("eth"));
}

TEST_F(VclusterMetalDescDeathTest, E4DramViewsIsAMap) {
    EXPECT_REFUSED(in_chip(1, [](YAML::Node& soc) { soc["dram_views"] = a_map(); }), e4_kind("dram_views"));
}

TEST_F(VclusterMetalDescDeathTest, E4KeraunosCoresIsAMap) {
    EXPECT_REFUSED(
        in_chip(1, [](YAML::Node& soc) { soc["keraunos"]["cores"] = a_map(); }), e4_kind("keraunos\\.cores"));
}

TEST_F(VclusterMetalDescDeathTest, E4KeraunosIsAScalar) {
    EXPECT_REFUSED(in_chip(1, [](YAML::Node& soc) { soc["keraunos"] = "k0"; }), e4_kind("keraunos"));
}

TEST_F(VclusterMetalDescDeathTest, E4KeraunosIsASequence) {
    EXPECT_REFUSED(in_chip(1, [](YAML::Node& soc) { soc["keraunos"] = a_sequence(); }), e4_kind("keraunos"));
}

TEST_F(VclusterMetalDescDeathTest, E4DocumentIsASequence) {
    EXPECT_REFUSED(
        [](Files& files) { std::ofstream(files.socs.at(1)) << "- eth\n- keraunos\n"; },
        e4_kind("\\(the document is not a map\\)"));
}

// E5: a `programmable` value that is not the package's text makes the view unequal to the package's, before it is
// read as a boolean.
TEST_F(VclusterMetalDescDeathTest, E5ProgrammableNotABoolean) {
    EXPECT_REFUSED(
        in_chip(0, [](YAML::Node& soc) { soc["dram_views"][0]["programmable"] = "maybe"; }),
        "Chip 0 descriptor [^ ]*chip_0_soc_desc\\.yaml: DRAM view \\{.*programmable: maybe.*\\} is not the package "
        "descriptor's");
}

// E5: an Ethernet core listed twice in one chip's file.
TEST_F(VclusterMetalDescDeathTest, E5EthernetCoreTwice) {
    EXPECT_REFUSED(
        in_chip(0, [](YAML::Node& soc) { soc["eth"].push_back("4-0"); }),
        "Chip 0 descriptor [^ ]*chip_0_soc_desc\\.yaml lists Ethernet core 4-0 twice");
}

// E5: a keraunos.cores entry listed twice in one chip's file.
TEST_F(VclusterMetalDescDeathTest, E5KeraunosEntryTwice) {
    EXPECT_REFUSED(
        in_chip(0, [](YAML::Node& soc) { soc["keraunos"]["cores"].push_back(package_keraunos_entry("4-0")); }),
        "Chip 0 descriptor [^ ]*chip_0_soc_desc\\.yaml lists keraunos\\.cores entry \\{core: 4-0, hsio: 0, eth_cce: 0\\} "
        "twice");
}

// E5: a DRAM view listed twice in one chip's file.
TEST_F(VclusterMetalDescDeathTest, E5DramViewTwice) {
    EXPECT_REFUSED(
        in_chip(0, [](YAML::Node& soc) { soc["dram_views"].push_back(YAML::Clone(soc["dram_views"][0])); }),
        "Chip 0 descriptor [^ ]*chip_0_soc_desc\\.yaml lists DRAM view \\{.*\\} twice");
}

// E6 compares views by the package's entry they equal, whatever order their keys are written in: chip 1's first view
// in chip 0's file with `programmable` written first.
TEST_F(VclusterMetalDescDeathTest, E6ProgrammableDramViewInTwoChipsByKeyOrder) {
    EXPECT_REFUSED(
        [](Files& files) {
            const YAML::Node view = YAML::LoadFile(files.socs[1].string())["dram_views"][0];
            if (!view["programmable"].as<bool>() || view.begin()->first.Scalar() == "programmable") {
                throw std::runtime_error("chip 1's first DRAM view is not programmable, or has programmable first");
            }
            YAML::Node reordered(YAML::NodeType::Map);
            reordered["programmable"] = YAML::Clone(view["programmable"]);
            for (const auto& kv : view) {
                if (kv.first.Scalar() != "programmable") {
                    reordered[kv.first.Scalar()] = YAML::Clone(kv.second);
                }
            }
            edit(files.socs[0], [&](YAML::Node& soc) { soc["dram_views"].push_back(reordered); });
        },
        "Chips 0 and 1 both list DRAM view ");
}

// E6: a chip's file that lists no programmable DRAM view.
TEST_F(VclusterMetalDescDeathTest, E6NoProgrammableDramView) {
    EXPECT_REFUSED(
        in_chip(1, [](YAML::Node& soc) { soc["dram_views"] = YAML::Node(YAML::NodeType::Sequence); }),
        "Chip 1 descriptor [^ ]*chip_1_soc_desc\\.yaml lists no programmable DRAM view; every DRAM core of the package "
        "would be its own");
}

// E7, § Verification 3: the package's and every chip's architecture changed alike to Blackhole.
TEST_F(VclusterMetalDescDeathTest, E7ArchitectureNotQuasar) {
    EXPECT_REFUSED(
        [](Files& files) {
            const auto to_blackhole = [](YAML::Node& soc) { soc["arch_name"] = "BLACKHOLE"; };
            edit_package_copy(files, to_blackhole);
            for (const auto& soc : files.socs) {
                edit(soc, to_blackhole);
            }
        },
        "TT_METAL_EMU_SOC_DESC lists 3 descriptors on architecture [^;]*; a descriptor a chip is read on Quasar only");
}

// --- § Verification 4: the two views agree, no device --------------------------------------------------------------

tt::umd::SocDescriptor descriptor(const fs::path& file, const tt::ChipInfo& chip_info) {
    return tt::umd::SocDescriptor(std::make_shared<tt::umd::SocArchDescriptor>(file.string()), chip_info);
}

std::string xy(const tt::umd::CoreCoord& core) { return std::to_string(core.x) + "-" + std::to_string(core.y); }

// For each manifest's generated descriptors and each chip: UMD's descriptor of the chip is the package's, and metal's
// is built from the chip's file as § Contract, `get_metal_desc_from_tt_desc` states, with `noc_translation_enabled`
// and `harvesting_masks` from UMD's and `board_type` and `asic_location` from the cluster descriptor. Every core of
// the chip's descriptor, and the core of every Ethernet channel, has the translated coordinate the package's
// descriptor gives the same NOC0 core. `noc_translation` overrides the cluster descriptor's value when given.
void expect_views_agree(std::optional<bool> noc_translation) {
    for (const std::string manifest : {"line3_eth.yaml", "line3_am.yaml", "line3_noc.yaml"}) {
        SCOPED_TRACE(manifest);
        const fs::path dir = temp_dir("views");
        fs::remove_all(dir);
        const Files files = generate(manifest, dir);
        const YAML::Node cluster = YAML::LoadFile(files.cluster.string());
        std::set<std::string> eth_cores_of_all_chips;
        size_t num_eth_cores = 0;
        for (uint32_t chip = 0; chip < files.socs.size(); ++chip) {
            SCOPED_TRACE("chip " + std::to_string(chip));
            tt::ChipInfo umd_info;
            umd_info.noc_translation_enabled =
                noc_translation.value_or(cluster["harvesting"][chip]["noc_translation"].as<bool>());
            umd_info.board_type = tt::board_type_name_map.at(cluster["chip_to_boardtype"][chip].as<std::string>());
            umd_info.asic_location = cluster["asic_locations"][chip].as<uint32_t>();
            const tt::umd::SocDescriptor package = descriptor(files.package, umd_info);

            tt::ChipInfo metal_info;
            metal_info.noc_translation_enabled = package.noc_translation_enabled;
            metal_info.harvesting_masks = package.harvesting_masks;
            metal_info.board_type = umd_info.board_type;
            metal_info.asic_location = umd_info.asic_location;
            const tt::umd::SocDescriptor own = descriptor(files.socs[chip], metal_info);

            const auto in_package = [&](const tt::umd::CoreCoord& noc0) {
                return package.translate_coord_to(
                    package.get_coord_at({noc0.x, noc0.y}, tt::CoordSystem::NOC0), tt::CoordSystem::TRANSLATED);
            };

            const std::vector<tt::umd::CoreCoord> cores = own.get_all_cores(tt::CoordSystem::NOC0);
            ASSERT_FALSE(cores.empty());
            for (const auto& core : cores) {
                SCOPED_TRACE("NOC0 core " + core.str());
                const tt::umd::CoreCoord mine = own.translate_coord_to(core, tt::CoordSystem::TRANSLATED);
                const tt::umd::CoreCoord packages = in_package(core);
                EXPECT_EQ(xy(mine), xy(packages)) << "chip's " << mine.str() << ", package's " << packages.str();
                EXPECT_EQ(mine, packages) << "chip's " << mine.str() << ", package's " << packages.str();
            }

            const uint32_t num_channels = own.get_num_eth_channels();
            ASSERT_GT(num_channels, 0u);
            for (uint32_t channel = 0; channel < num_channels; ++channel) {
                SCOPED_TRACE("Ethernet channel " + std::to_string(channel));
                const tt::umd::CoreCoord noc0 = own.get_eth_core_for_channel(channel, tt::CoordSystem::NOC0);
                const tt::umd::CoreCoord mine = own.get_eth_core_for_channel(channel, tt::CoordSystem::TRANSLATED);
                const tt::umd::CoreCoord packages = in_package(noc0);
                EXPECT_EQ(xy(mine), xy(packages))
                    << "NOC0 " << noc0.str() << ": chip's " << mine.str() << ", package's " << packages.str();
                EXPECT_EQ(mine, packages)
                    << "NOC0 " << noc0.str() << ": chip's " << mine.str() << ", package's " << packages.str();
                eth_cores_of_all_chips.insert(xy(noc0));
                ++num_eth_cores;
            }
        }
        // The comparison is of four distinct cores, and the chips' channel numbers are not the package's:
        // § Why it is sound, "chip 1's channel 0 is 4-1 to metal and 4-0 to UMD".
        EXPECT_EQ(num_eth_cores, 4u);
        EXPECT_EQ(eth_cores_of_all_chips, (std::set<std::string>{"4-0", "4-1", "4-2", "4-3"}));
        fs::remove_all(dir);
    }
}

TEST(VclusterMetalDesc, TwoViewsAgree) { expect_views_agree(std::nullopt); }

// The same with NOC translation on, which the generated cluster descriptor does not ask for: § Why it is sound reads
// an Ethernet core's translated coordinate on Quasar as its NOC0 coordinate whatever the chip reports.
TEST(VclusterMetalDesc, TwoViewsAgreeWithNocTranslation) { expect_views_agree(true); }

// § Why it is sound, "Channel numbers never cross": chip 1's channel 0 is 4-1 in its own descriptor and 4-0 in the
// package's, so a channel number taken to UMD's descriptor names another tile.
TEST(VclusterMetalDesc, ChannelNumbersDifferBetweenTheViews) {
    const fs::path dir = temp_dir("channels");
    fs::remove_all(dir);
    const Files files = generate("line3_eth.yaml", dir);
    const tt::umd::SocDescriptor package = descriptor(files.package, {});
    const tt::umd::SocDescriptor chip_1 = descriptor(files.socs.at(1), {});
    EXPECT_EQ(xy(chip_1.get_eth_core_for_channel(0, tt::CoordSystem::NOC0)), "4-1");
    EXPECT_EQ(xy(package.get_eth_core_for_channel(0, tt::CoordSystem::NOC0)), "4-0");
    fs::remove_all(dir);
}

}  // namespace
}  // namespace tt::tt_fabric::vcluster_metal_desc_tests
