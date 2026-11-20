// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The descriptor generator, without a device. VclusterGenerate.FromEnvironment is the generator's entry point for a
// run; it is skipped unless VCLUSTER_MANIFEST is set.

#include <gtest/gtest.h>

#include <yaml-cpp/yaml.h>

#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "tt_metal/tt_fabric/keraunos/vcluster/descriptors.hpp"
#include "tt_metal/tt_fabric/keraunos/vcluster/manifest.hpp"
#include "tt_metal/tt_fabric/keraunos/vcluster/test_support.hpp"

namespace tt::tt_fabric::vcluster {
namespace {

namespace fs = std::filesystem;

// The Keraunos fabric test directory, located from the source path, so the test runs from any working directory.
const fs::path kKeraunosDir = fs::path(__FILE__).parent_path().parent_path();

using test::package_soc_desc;

// A fresh, empty directory under gtest's temp dir, named for this process so that two runs cannot remove each other's,
// and removed when the test ends.
class TempDir {
public:
    explicit TempDir(std::string_view name) :
        path_(
            fs::path(::testing::TempDir()) /
            ("vcluster_descriptors_" + std::to_string(::getpid()) + "_" + std::string(name))) {
        fs::remove_all(path_);
        fs::create_directories(path_);
    }
    ~TempDir() { fs::remove_all(path_); }
    const fs::path& path() const { return path_; }

private:
    fs::path path_;
};

std::string read_text(const fs::path& path) {
    std::ifstream in(path);
    EXPECT_TRUE(in.good()) << "cannot read " << path;
    std::stringstream text;
    text << in.rdbuf();
    return text.str();
}

void write_text(const fs::path& path, std::string_view text) { std::ofstream(path) << text; }

// `text` with `from` replaced by `to`; `from` must occur exactly once.
std::string replaced(std::string text, std::string_view from, std::string_view to) {
    return test::edited(std::move(text), {{from, to}});
}

// YAML equality as parsed data: maps compare by key set and values, whatever the key order; sequences in order;
// scalars by text. Comments are not part of the parsed data. Returns the first difference, empty when equal.
std::string yaml_difference(const YAML::Node& a, const YAML::Node& b, const std::string& at = "") {
    if (a.Type() != b.Type()) {
        return at + ": node types differ";
    }
    switch (a.Type()) {
        case YAML::NodeType::Scalar:
            return a.Scalar() == b.Scalar() ? "" : at + ": '" + a.Scalar() + "' vs '" + b.Scalar() + "'";
        case YAML::NodeType::Sequence: {
            if (a.size() != b.size()) {
                return at + ": " + std::to_string(a.size()) + " vs " + std::to_string(b.size()) + " items";
            }
            for (size_t i = 0; i < a.size(); ++i) {
                if (auto d = yaml_difference(a[i], b[i], at + "[" + std::to_string(i) + "]"); !d.empty()) {
                    return d;
                }
            }
            return "";
        }
        case YAML::NodeType::Map: {
            std::map<std::string, YAML::Node> am, bm;
            for (const auto& kv : a) {
                am[kv.first.Scalar()] = kv.second;
            }
            for (const auto& kv : b) {
                bm[kv.first.Scalar()] = kv.second;
            }
            for (const auto& [key, value] : am) {
                if (!bm.contains(key)) {
                    return at + "." + key + ": missing on the right";
                }
                if (auto d = yaml_difference(value, bm[key], at + "." + key); !d.empty()) {
                    return d;
                }
            }
            for (const auto& [key, value] : bm) {
                if (!am.contains(key)) {
                    return at + "." + key + ": missing on the left";
                }
            }
            return "";
        }
        default: return "";  // Null and Undefined: equal by type
    }
}

std::string yaml_file_difference(const fs::path& a, const fs::path& b) {
    return yaml_difference(YAML::LoadFile(a.string()), YAML::LoadFile(b.string()));
}

// Protobuf text format, enough for a mesh graph descriptor: `name: value`, `name: [ v, ... ]` and `name { ... }`,
// comments from `#` to the end of the line. A message is its fields sorted by name then value, so field order is
// ignored; repeated fields keep their values.
struct TextprotoMessage {
    std::vector<std::pair<std::string, std::string>> fields;  // nested messages rendered to text in canonical form
};

std::vector<std::string> textproto_tokens(std::string_view text) {
    std::vector<std::string> tokens;
    size_t i = 0;
    while (i < text.size()) {
        const char c = text[i];
        if (c == '#') {
            while (i < text.size() && text[i] != '\n') {
                ++i;
            }
        } else if (std::isspace(static_cast<unsigned char>(c))) {
            ++i;
        } else if (c == '{' || c == '}' || c == '[' || c == ']' || c == ':' || c == ',') {
            tokens.emplace_back(1, c);
            ++i;
        } else if (c == '"') {
            const size_t end = text.find('"', i + 1);
            tokens.emplace_back(text.substr(i, end - i + 1));
            i = end + 1;
        } else {
            size_t end = i;
            while (end < text.size() && !std::isspace(static_cast<unsigned char>(text[end])) &&
                   std::string_view("{}[]:,#\"").find(text[end]) == std::string_view::npos) {
                ++end;
            }
            tokens.emplace_back(text.substr(i, end - i));
            i = end;
        }
    }
    return tokens;
}

// Canonical text of the message starting at tokens[i], up to its closing `}` or the end; advances i past it.
std::string textproto_canonical(const std::vector<std::string>& tokens, size_t& i) {
    std::vector<std::string> fields;
    while (i < tokens.size() && tokens[i] != "}") {
        const std::string name = tokens[i++];
        if (i < tokens.size() && tokens[i] == ":") {
            ++i;
        }
        if (i < tokens.size() && tokens[i] == "{") {
            ++i;
            fields.push_back(name + "{" + textproto_canonical(tokens, i) + "}");
            ++i;  // the closing brace
        } else if (i < tokens.size() && tokens[i] == "[") {
            std::string list = name + "[";
            for (++i; i < tokens.size() && tokens[i] != "]"; ++i) {
                list += tokens[i] == "," ? "," : tokens[i];
            }
            ++i;
            fields.push_back(list + "]");
        } else {
            fields.push_back(name + "=" + tokens[i++]);
        }
    }
    std::sort(fields.begin(), fields.end());
    std::string text;
    for (const auto& field : fields) {
        text += field + ";";
    }
    return text;
}

std::string textproto_canonical(const fs::path& path) {
    const auto tokens = textproto_tokens(read_text(path));
    size_t i = 0;
    return textproto_canonical(tokens, i);
}

// The three-device line as text, A on tile 0, M on tiles 1 and 2, B on tile 3, both links on the NOC: the
// base the refusal tests edit. It passes check_manifest with the package's facts.
constexpr std::string_view kLine = R"yaml(
version: 1
devices:
  - name: A
    fabric_chip: 0
    slots:
      - {tile: 0, cce: 0, hart: 0, role: router, port: 0}
      - {tile: 0, cce: 0, hart: 1, role: worker}
    regions:
      - {tile: 0, cce: 0, base: 0x0, size: 0x400000}
  - name: M
    fabric_chip: 1
    slots:
      - {tile: 1, cce: 0, hart: 0, role: router, port: 0}
      - {tile: 2, cce: 0, hart: 0, role: router, port: 1}
    regions:
      - {tile: 1, cce: 0, base: 0x0, size: 0x400000}
      - {tile: 2, cce: 0, base: 0x0, size: 0x400000}
  - name: B
    fabric_chip: 2
    slots:
      - {tile: 3, cce: 0, hart: 0, role: router, port: 0}
      - {tile: 3, cce: 0, hart: 1, role: worker}
    regions:
      - {tile: 3, cce: 0, base: 0x0, size: 0x400000}
links:
  - {a: {device: A, port: 0}, b: {device: M, port: 0}, backend: noc}
  - {a: {device: M, port: 1}, b: {device: B, port: 0}, backend: noc}
)yaml";

std::string line_with(std::initializer_list<std::pair<std::string_view, std::string_view>> edits) {
    return test::edited(std::string(kLine), edits);
}

using Refusals = std::set<uint32_t>;

// What generate_descriptors() refused `manifest` with: the check_manifest() rows and the G refusals.
struct Refused {
    Refusals rows;
    Refusals g;
};

// The refusal of `manifest`, and that nothing was written to the output directory.
Refused refused(const Manifest& manifest, const fs::path& package = package_soc_desc()) {
    const TempDir out("refusal");
    Refused found;
    try {
        generate_descriptors(manifest, package, out.path());
        ADD_FAILURE() << "generated, expected a refusal";
    } catch (const DescriptorGenerationError& error) {
        for (const auto& row : error.rows()) {
            found.rows.insert(static_cast<uint32_t>(row.check));
        }
        for (const auto& finding : error.findings()) {
            found.g.insert(static_cast<uint32_t>(finding.refusal));
        }
    }
    EXPECT_TRUE(fs::is_empty(out.path())) << "a refusal wrote a file";
    return found;
}

// The G refusals of a manifest that passes the checks, so no row is reported with them.
Refusals refusals(const Manifest& manifest, const fs::path& package = package_soc_desc()) {
    const Refused found = refused(manifest, package);
    EXPECT_EQ(found.rows, Refusals{}) << "rows reported for a manifest the checks accept";
    return found.g;
}

// Every refusal test's manifest is one the checks accept, so the refusal is the generator's alone.
Manifest checked(std::string_view yaml, const fs::path& package = package_soc_desc()) {
    Manifest manifest = parse_manifest(yaml);
    const auto findings = check_manifest(manifest, package_facts(package));
    EXPECT_TRUE(findings.empty()) << "the manifest fails row " << static_cast<uint32_t>(findings.front().check) << ": "
                                  << findings.front().detail;
    return manifest;
}

// --- package_facts -----------------------------------------------------------------------------------------------

// The package facts: the cores from keraunos.cores, 8 harts a CCE, the SRAM size from eth_l1_size, and the Ethernet
// tile pairs (0, 1) and (2, 3) of 26ww45.eth-pcs-output-inter.
TEST(VclusterDescriptors, PackageFacts) {
    const PackageFacts facts = package_facts(package_soc_desc());
    EXPECT_EQ(facts.cores, (std::vector<CceId>{{0, 0}, {1, 0}, {2, 0}, {3, 0}, {0, 1}, {1, 1}, {2, 1}, {3, 1}}));
    EXPECT_EQ(facts.harts_per_cce, 8u);
    EXPECT_EQ(facts.sram_bytes_per_cce, 4194304u);
    std::set<std::pair<uint32_t, uint32_t>> pairs;
    for (auto [a, b] : facts.ethernet_tile_pairs) {
        pairs.insert({std::min(a, b), std::max(a, b)});
    }
    EXPECT_EQ(pairs, (std::set<std::pair<uint32_t, uint32_t>>{{0, 1}, {2, 3}}));
}

// --- noc_links ---------------------------------------------------------------------------------------------------

// Each pair is its ends' fabric_chip, lower first, in link order, and only noc links count.
TEST(VclusterDescriptors, NocLinksLowerFirstInLinkOrder) {
    const Manifest m = parse_manifest(line_with(
        {{"  - {a: {device: A, port: 0}, b: {device: M, port: 0}, backend: noc}\n"
          "  - {a: {device: M, port: 1}, b: {device: B, port: 0}, backend: noc}\n",
          "  - {a: {device: B, port: 0}, b: {device: M, port: 1}, backend: noc}\n"
          "  - {a: {device: M, port: 0}, b: {device: A, port: 0}, backend: noc}\n"}}));
    EXPECT_EQ(noc_links(m), (std::vector<std::pair<uint32_t, uint32_t>>{{1, 2}, {0, 1}}));
    const Manifest one = parse_manifest(
        line_with({{"b: {device: B, port: 0}, backend: noc", "b: {device: B, port: 0}, backend: ethernet"}}));
    EXPECT_EQ(noc_links(one), (std::vector<std::pair<uint32_t, uint32_t>>{{0, 1}}));
}

// --- generate_descriptors: the reference output ------------------------------------------------------------------

// Each device's eth and keraunos.cores are its
// routers' package entries in port order, the four DRAM views go 2, 1, 1 in list order, the cluster descriptor has one
// connection a link and the mesh is [ 1, 3 ].
TEST(VclusterDescriptors, Line3) {
    const TempDir out("line3");
    const GeneratedDescriptors g = generate_descriptors(checked(kLine), package_soc_desc(), out.path());
    const YAML::Node package = YAML::LoadFile(package_soc_desc().string());
    ASSERT_EQ(package["dram_views"].size(), 4u);
    ASSERT_EQ(g.soc_descs.size(), 3u);
    const std::vector<std::vector<size_t>> tiles = {{0}, {1, 2}, {3}};
    const std::vector<std::vector<size_t>> views = {{0, 1}, {2}, {3}};
    for (size_t chip = 0; chip < 3; ++chip) {
        SCOPED_TRACE(chip);
        EXPECT_EQ(g.soc_descs[chip], out.path() / ("chip_" + std::to_string(chip) + "_soc_desc.yaml"));
        const YAML::Node soc = YAML::LoadFile(g.soc_descs[chip].string());
        ASSERT_EQ(soc["eth"].size(), tiles[chip].size());
        ASSERT_EQ(soc["keraunos"]["cores"].size(), tiles[chip].size());
        for (size_t port = 0; port < tiles[chip].size(); ++port) {
            // The package lists CCE0 of tile t at index t.
            const YAML::Node entry = package["keraunos"]["cores"][tiles[chip][port]];
            EXPECT_EQ(soc["eth"][port].Scalar(), entry["core"].Scalar());
            EXPECT_EQ(yaml_difference(soc["keraunos"]["cores"][port], entry), "");
        }
        ASSERT_EQ(soc["dram_views"].size(), views[chip].size());
        for (size_t i = 0; i < views[chip].size(); ++i) {
            EXPECT_EQ(yaml_difference(soc["dram_views"][i], package["dram_views"][views[chip][i]]), "");
        }
    }
    EXPECT_EQ(g.cluster_desc, out.path() / "cluster_desc.yaml");
    const YAML::Node cluster = YAML::LoadFile(g.cluster_desc.string());
    EXPECT_EQ(
        yaml_difference(
            cluster["ethernet_connections"],
            YAML::Load("[[{chip: 0, chan: 0}, {chip: 1, chan: 0}], [{chip: 1, chan: 1}, {chip: 2, chan: 0}]]")),
        "");
    EXPECT_EQ(
        yaml_difference(
            cluster["chip_unique_ids"], YAML::Load("{0: 5497558138880, 1: 5497558138881, 2: 5497558138882}")),
        "");
    EXPECT_EQ(g.mesh_graph_desc, out.path() / "mesh_graph_descriptor.textproto");
    EXPECT_NE(textproto_canonical(g.mesh_graph_desc).find("dims[1,3]"), std::string::npos)
        << textproto_canonical(g.mesh_graph_desc);
}

// "Every other key is copied unchanged": a generated SoC descriptor differs from the package's only in
// eth, keraunos.cores and dram_views.
TEST(VclusterDescriptors, SocDescCopiesThePackage) {
    const TempDir out("copies");
    const GeneratedDescriptors g =
        generate_descriptors(checked(kLine), package_soc_desc(), out.path());
    const YAML::Node package = YAML::LoadFile(package_soc_desc().string());
    for (const auto& soc : g.soc_descs) {
        SCOPED_TRACE(soc.string());
        YAML::Node generated = YAML::LoadFile(soc.string());
        YAML::Node expected = YAML::Clone(package);
        expected["eth"] = generated["eth"];
        expected["keraunos"]["cores"] = generated["keraunos"]["cores"];
        expected["dram_views"] = generated["dram_views"];
        EXPECT_EQ(yaml_difference(generated, expected), "");
    }
}

// A backend is in none of the files, so the line generates the same descriptors whatever its links'
// backends.
TEST(VclusterDescriptors, BackendInNoFile) {
    const TempDir noc("noc");
    const GeneratedDescriptors ref = generate_descriptors(checked(kLine), package_soc_desc(), noc.path());
    const std::string am = line_with(
        {{"b: {device: B, port: 0}, backend: noc", "b: {device: B, port: 0}, backend: ethernet"}});
    const std::string eth =
        replaced(am, "b: {device: M, port: 0}, backend: noc", "b: {device: M, port: 0}, backend: ethernet");
    for (const auto& text : {am, eth}) {
        const TempDir out("backend");
        const GeneratedDescriptors g = generate_descriptors(checked(text), package_soc_desc(), out.path());
        ASSERT_EQ(g.soc_descs.size(), ref.soc_descs.size());
        for (size_t chip = 0; chip < g.soc_descs.size(); ++chip) {
            EXPECT_EQ(read_text(g.soc_descs[chip]), read_text(ref.soc_descs[chip]));
        }
        EXPECT_EQ(read_text(g.cluster_desc), read_text(ref.cluster_desc));
        EXPECT_EQ(read_text(g.mesh_graph_desc), read_text(ref.mesh_graph_desc));
    }
}

// The NAME=value lines of an env file.
std::map<std::string, std::string> env_vars(const fs::path& path) {
    std::map<std::string, std::string> vars;
    std::istringstream lines(read_text(path));
    for (std::string line; std::getline(lines, line);) {
        if (line.empty()) {
            continue;
        }
        const size_t eq = line.find('=');
        EXPECT_NE(eq, std::string::npos) << line;
        EXPECT_FALSE(vars.contains(line.substr(0, eq))) << "repeated " << line;
        vars[line.substr(0, eq)] = line.substr(eq + 1);
    }
    return vars;
}

// The non-empty lines of an env file, in file order.
std::vector<std::string> env_lines(const fs::path& path) {
    std::vector<std::string> found;
    std::istringstream lines(read_text(path));
    for (std::string line; std::getline(lines, line);) {
        if (!line.empty()) {
            found.push_back(line);
        }
    }
    return found;
}

// Three lines NAME=value with absolute paths, the SoC descriptors
// comma-separated in chip order; env.mock names the cluster descriptor TT_METAL_MOCK_CLUSTER_DESC_PATH where env.emu
// names it TT_METAL_EMU_CLUSTER_DESC. TT_METAL_EMU_PACKAGE_SOC_DESC, the
// absolute path of the package descriptor given, is the last line of both, after those three.
TEST(VclusterDescriptors, EnvFiles) {
    const TempDir out("env");
    const GeneratedDescriptors g =
        generate_descriptors(checked(kLine), package_soc_desc(), out.path());
    EXPECT_EQ(g.env_emu, out.path() / "env.emu");
    EXPECT_EQ(g.env_mock, out.path() / "env.mock");
    std::string socs;
    for (const auto& soc : g.soc_descs) {
        socs += (socs.empty() ? "" : ",") + fs::absolute(soc).string();
    }
    const std::map<std::string, std::string> shared = {
        {"TT_METAL_EMU_SOC_DESC", socs},
        {"TT_MESH_GRAPH_DESC_PATH", fs::absolute(g.mesh_graph_desc).string()},
        {"TT_METAL_EMU_PACKAGE_SOC_DESC", fs::absolute(package_soc_desc()).string()}};
    auto emu = shared;
    emu["TT_METAL_EMU_CLUSTER_DESC"] = fs::absolute(g.cluster_desc).string();
    auto mock = shared;
    mock["TT_METAL_MOCK_CLUSTER_DESC_PATH"] = fs::absolute(g.cluster_desc).string();
    EXPECT_EQ(env_vars(g.env_emu), emu);
    EXPECT_EQ(env_vars(g.env_mock), mock);

    const std::vector<std::string> after_the_cluster = {
        "TT_METAL_EMU_SOC_DESC=" + socs,
        "TT_MESH_GRAPH_DESC_PATH=" + fs::absolute(g.mesh_graph_desc).string(),
        "TT_METAL_EMU_PACKAGE_SOC_DESC=" + fs::absolute(package_soc_desc()).string()};
    auto emu_lines = after_the_cluster;
    emu_lines.insert(emu_lines.begin(), "TT_METAL_EMU_CLUSTER_DESC=" + fs::absolute(g.cluster_desc).string());
    auto mock_lines = after_the_cluster;
    mock_lines.insert(mock_lines.begin(), "TT_METAL_MOCK_CLUSTER_DESC_PATH=" + fs::absolute(g.cluster_desc).string());
    EXPECT_EQ(env_lines(g.env_emu), emu_lines);
    EXPECT_EQ(env_lines(g.env_mock), mock_lines);
}

// The package line is absolute when the package descriptor is given by a
// relative path.
TEST(VclusterDescriptors, EnvFilesPackagePathAbsolute) {
    const TempDir out("env_relative");
    const fs::path relative = fs::relative(package_soc_desc());
    ASSERT_TRUE(relative.is_relative()) << relative;
    const GeneratedDescriptors g = generate_descriptors(checked(kLine), relative, out.path());
    for (const auto& env_file : {g.env_emu, g.env_mock}) {
        const std::vector<std::string> lines = env_lines(env_file);
        ASSERT_EQ(lines.size(), 4u) << env_file;
        const std::string name = "TT_METAL_EMU_PACKAGE_SOC_DESC=";
        ASSERT_EQ(lines.back().substr(0, name.size()), name) << env_file;
        const fs::path written = lines.back().substr(name.size());
        EXPECT_TRUE(written.is_absolute()) << written;
        EXPECT_TRUE(fs::equivalent(written, package_soc_desc())) << written;
    }
}

// Files are replaced if present.
TEST(VclusterDescriptors, ReplacesExistingFiles) {
    const TempDir out("replace");
    write_text(out.path() / "cluster_desc.yaml", "stale: true\n");
    write_text(out.path() / "env.emu", "STALE=1\n");
    const GeneratedDescriptors g =
        generate_descriptors(checked(kLine), package_soc_desc(), out.path());
    EXPECT_FALSE(YAML::LoadFile(g.cluster_desc.string())["stale"]);
    EXPECT_EQ(read_text(g.env_emu).find("STALE"), std::string::npos);
}

// A two-device manifest generates a [ 1, 2 ] mesh and two SoC descriptors whose DRAM views are
// disjoint and together are the package's. With 4 programmable views and 2 devices, device 0 takes views 0 and 1 and
// device 1 views 2 and 3: Metal takes a power of two of DRAM banks.
TEST(VclusterDescriptors, TwoDevices) {
    const Manifest m = checked(R"yaml(
version: 1
devices:
  - name: A
    fabric_chip: 0
    slots:
      - {tile: 0, cce: 0, hart: 0, role: router, port: 0}
      - {tile: 0, cce: 0, hart: 1, role: worker}
    regions:
      - {tile: 0, cce: 0, base: 0x0, size: 0x400000}
  - name: B
    fabric_chip: 1
    slots:
      - {tile: 1, cce: 0, hart: 0, role: router, port: 0}
      - {tile: 1, cce: 0, hart: 1, role: worker}
    regions:
      - {tile: 1, cce: 0, base: 0x0, size: 0x400000}
links:
  - {a: {device: A, port: 0}, b: {device: B, port: 0}, backend: ethernet}
)yaml");
    const TempDir out("two");
    const GeneratedDescriptors g = generate_descriptors(m, package_soc_desc(), out.path());
    ASSERT_EQ(g.soc_descs.size(), 2u);
    EXPECT_NE(textproto_canonical(g.mesh_graph_desc).find("dims[1,2]"), std::string::npos)
        << textproto_canonical(g.mesh_graph_desc);
    const YAML::Node package_views = YAML::LoadFile(package_soc_desc().string())["dram_views"];
    const YAML::Node a = YAML::LoadFile(g.soc_descs[0].string());
    const YAML::Node b = YAML::LoadFile(g.soc_descs[1].string());
    ASSERT_EQ(package_views.size(), 4u);
    ASSERT_EQ(a["dram_views"].size(), 2u);
    ASSERT_EQ(b["dram_views"].size(), 2u);
    for (size_t i = 0; i < 2; ++i) {
        EXPECT_EQ(yaml_difference(a["dram_views"][i], package_views[i]), "") << "device 0 view " << i;
        EXPECT_EQ(yaml_difference(b["dram_views"][i], package_views[2 + i]), "") << "device 1 view " << i;
    }
    EXPECT_EQ(YAML::LoadFile(g.soc_descs[0].string())["eth"][0].Scalar(), "4-0");
    EXPECT_EQ(YAML::LoadFile(g.soc_descs[1].string())["eth"][0].Scalar(), "4-1");
    const YAML::Node cluster = YAML::LoadFile(g.cluster_desc.string());
    EXPECT_EQ(cluster["arch"].size(), 2u);
    ASSERT_EQ(cluster["ethernet_connections"].size(), 1u);
    EXPECT_EQ(cluster["ethernet_connections"][0][0]["chip"].as<int>(), 0);
    EXPECT_EQ(cluster["ethernet_connections"][0][1]["chip"].as<int>(), 1);
}

// A device's router cores in port order, and keraunos.cores in the same order, whatever the
// order of its slots. M with its tile-2 router written first.
TEST(VclusterDescriptors, EthInPortOrder) {
    const Manifest m = checked(line_with(
        {{"      - {tile: 1, cce: 0, hart: 0, role: router, port: 0}\n      - {tile: 2, cce: 0, hart: 0, role: router, "
          "port: 1}\n",
          "      - {tile: 2, cce: 0, hart: 0, role: router, port: 1}\n      - {tile: 1, cce: 0, hart: 0, role: router, "
          "port: 0}\n"}}));
    const TempDir out("order");
    const TempDir line("order_line");
    const GeneratedDescriptors g = generate_descriptors(m, package_soc_desc(), out.path());
    const GeneratedDescriptors ref = generate_descriptors(checked(kLine), package_soc_desc(), line.path());
    EXPECT_EQ(yaml_file_difference(g.soc_descs[1], ref.soc_descs[1]), "");
}

// --- generate_descriptors: refusals G1 to G5 ---------------------------------------------------------------------

// G1: fabric_chip values not 0 to N-1 (B is chip 3).
TEST(VclusterDescriptors, G1ChipsNotDense) {
    EXPECT_EQ(refusals(checked(line_with({{"fabric_chip: 2", "fabric_chip: 3"}}))), Refusals{1});
}

// G1: the links are not one between chips i and i+1: M is chip 2 and B chip 1, so the links join 0-2 and 2-1.
TEST(VclusterDescriptors, G1NotALine) {
    EXPECT_EQ(
        refusals(checked(line_with(
            {{"fabric_chip: 1", "fabric_chip: 2"},
             {"    fabric_chip: 2\n    slots:\n      - {tile: 3",
              "    fabric_chip: 1\n    slots:\n      - {tile: 3"}}))),
        Refusals{1});
}

// G2: a device's router ports are not 0 to k-1 (M's are 1 and 2).
TEST(VclusterDescriptors, G2PortsNotDense) {
    EXPECT_EQ(
        refusals(checked(line_with(
            {{"{tile: 1, cce: 0, hart: 0, role: router, port: 0}", "{tile: 1, cce: 0, hart: 0, role: router, port: 2}"},
             {"b: {device: M, port: 0}", "b: {device: M, port: 2}"}}))),
        Refusals{2});
}

// The package descriptor with its dram_views replaced by `views`, written to `path`.
void write_package_with_views(const fs::path& path, const YAML::Node& views) {
    YAML::Node package = YAML::LoadFile(package_soc_desc().string());
    package["dram_views"] = views;
    YAML::Emitter out;
    out << package;
    write_text(path, out.c_str());
}

// G3: fewer DRAM views than devices: a package with the first two of its four views, both programmable.
TEST(VclusterDescriptors, G3TooFewDramViews) {
    const TempDir dir("g3");
    const YAML::Node views = YAML::LoadFile(package_soc_desc().string())["dram_views"];
    YAML::Node two(YAML::NodeType::Sequence);
    two.push_back(YAML::Clone(views[0]));
    two.push_back(YAML::Clone(views[1]));
    write_package_with_views(dir.path() / "package.yaml", two);
    EXPECT_EQ(refusals(checked(kLine, dir.path() / "package.yaml"), dir.path() / "package.yaml"), Refusals{3});
}

// G6: a DRAM view of the package is not programmable: the last of its four, so there are still enough views (no G3).
TEST(VclusterDescriptors, G6ViewNotProgrammable) {
    const TempDir dir("g6");
    YAML::Node views = YAML::Clone(YAML::LoadFile(package_soc_desc().string())["dram_views"]);
    views[views.size() - 1]["programmable"] = false;
    write_package_with_views(dir.path() / "package.yaml", views);
    EXPECT_EQ(refusals(checked(kLine, dir.path() / "package.yaml"), dir.path() / "package.yaml"), Refusals{6});
}

// G4: a router slot's core is not an Ethernet core of the package: a package whose eth list leaves out 4-3, which
// keraunos.cores still lists, so the checks still accept B on tile 3.
TEST(VclusterDescriptors, G4NotAnEthernetCore) {
    const TempDir dir("g4");
    write_text(
        dir.path() / "package.yaml",
        replaced(read_text(package_soc_desc()), "[4-0, 4-1, 4-2, 4-3]", "[4-0, 4-1, 4-2]"));
    EXPECT_EQ(refusals(checked(kLine, dir.path() / "package.yaml"), dir.path() / "package.yaml"), Refusals{4});
}

// G5: the manifest has a capture key.
TEST(VclusterDescriptors, G5Capture) {
    EXPECT_EQ(refusals(checked(line_with({{"version: 1\n", "version: 1\ncapture: line3.capture\n"}}))), Refusals{5});
}

// The generator runs check_manifest() itself and refuses on any row, reporting the rows. A worker on
// hart 8 of an 8-hart CCE is row 3 and touches no G refusal.
TEST(VclusterDescriptors, RefusesOnARow) {
    const Manifest m = parse_manifest(
        line_with({{"{tile: 3, cce: 0, hart: 1, role: worker}", "{tile: 3, cce: 0, hart: 8, role: worker}"}}));
    const Refused found = refused(m);
    EXPECT_EQ(found.rows, Refusals{3});
    EXPECT_EQ(found.g, Refusals{});
}

// The rows are check_manifest()'s, staged: a reserved value (row 1) reports row 1 alone.
TEST(VclusterDescriptors, RefusesOnRowOneAlone) {
    const Manifest m = parse_manifest(line_with(
        {{"{tile: 3, cce: 0, hart: 1, role: worker}", "{tile: 3, cce: 0, hart: 8, role: worker}"},
         {"b: {device: B, port: 0}, backend: noc", "b: {device: B, port: 0}, backend: local"}}));
    const Refused found = refused(m);
    EXPECT_EQ(found.rows, Refusals{1});
    EXPECT_EQ(found.g, Refusals{});
}

// Each ethernet_connections entry puts the end with the lower chip first, whatever
// the manifest's order of the two ends.
TEST(VclusterDescriptors, ConnectionLowerChipFirst) {
    const Manifest m = checked(line_with(
        {{"{a: {device: A, port: 0}, b: {device: M, port: 0}, backend: noc}",
          "{a: {device: M, port: 0}, b: {device: A, port: 0}, backend: noc}"},
         {"{a: {device: M, port: 1}, b: {device: B, port: 0}, backend: noc}",
          "{a: {device: B, port: 0}, b: {device: M, port: 1}, backend: noc}"}}));
    const TempDir out("lower_first");
    const TempDir line("lower_first_line");
    const GeneratedDescriptors g = generate_descriptors(m, package_soc_desc(), out.path());
    const GeneratedDescriptors ref = generate_descriptors(checked(kLine), package_soc_desc(), line.path());
    EXPECT_EQ(yaml_file_difference(g.cluster_desc, ref.cluster_desc), "");
}

// The generator lists every refusal that holds: G2 and G5 together.
TEST(VclusterDescriptors, EveryRefusalListed) {
    EXPECT_EQ(
        refusals(checked(line_with(
            {{"version: 1\n", "version: 1\ncapture: line3.capture\n"},
             {"{tile: 1, cce: 0, hart: 0, role: router, port: 0}", "{tile: 1, cce: 0, hart: 0, role: router, port: 2}"},
             {"b: {device: M, port: 0}", "b: {device: M, port: 2}"}}))),
        (Refusals{2, 5}));
}

// --- The generator's entry point for a run ------------------------------------------------------------------------

// Reads VCLUSTER_MANIFEST, VCLUSTER_PACKAGE_SOC_DESC and VCLUSTER_OUT, and generates; skipped when
// VCLUSTER_MANIFEST is unset. The generator runs check_manifest() itself; each row or refusal is a failure.
TEST(VclusterGenerate, FromEnvironment) {
    const char* manifest_path = std::getenv("VCLUSTER_MANIFEST");
    if (manifest_path == nullptr || *manifest_path == '\0') {
        GTEST_SKIP() << "VCLUSTER_MANIFEST is not set";
    }
    const char* package = std::getenv("VCLUSTER_PACKAGE_SOC_DESC");
    const char* out = std::getenv("VCLUSTER_OUT");
    ASSERT_TRUE(package != nullptr && *package != '\0') << "VCLUSTER_PACKAGE_SOC_DESC is not set";
    ASSERT_TRUE(out != nullptr && *out != '\0') << "VCLUSTER_OUT is not set";
    const Manifest manifest = load_manifest(manifest_path);
    GeneratedDescriptors g;
    try {
        g = generate_descriptors(manifest, package, out);
    } catch (const DescriptorGenerationError& error) {
        for (const auto& row : error.rows()) {
            ADD_FAILURE() << "row " << static_cast<uint32_t>(row.check) << ": " << row.detail;
        }
        for (const auto& finding : error.findings()) {
            ADD_FAILURE() << "G" << static_cast<uint32_t>(finding.refusal) << ": " << finding.detail;
        }
        FAIL() << "the generator refused " << manifest_path;
    }
    std::cout << "export the variables of " << g.env_emu.string() << " (emulator) or " << g.env_mock.string()
              << " (Mock)" << std::endl;
}

}  // namespace
}  // namespace tt::tt_fabric::vcluster
