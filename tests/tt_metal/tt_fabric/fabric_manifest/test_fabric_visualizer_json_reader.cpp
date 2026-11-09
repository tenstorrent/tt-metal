// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The strict JSON reader the visualizer's stages read each other's files with: a reader built on it fails when the
// file and the reader disagree about which keys there are, naming the key by its JSON pointer. No device.

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <cstdint>
#include <exception>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "tt_metal/fabric/debug/visualizer/common/fabric_visualizer_json_reader.hpp"

namespace tt::tt_fabric::visualizer {
namespace {

using json = nlohmann::ordered_json;

constexpr std::string_view k_document_name = "Test document";

// Reads `document`'s top-level object with `read`.
template <typename Read>
auto read_document(const json& document, Read&& read) {
    return JsonValueReader(document, k_document_name).read_object(std::forward<Read>(read));
}

// What reading `document` with `read` fails with, or empty when it doesn't fail.
template <typename Read>
std::string read_document_error(const json& document, Read&& read) {
    try {
        read_document(document, std::forward<Read>(read));
    } catch (const std::exception& error) {
        return error.what();
    }
    return {};
}

// Fails the test unless reading `document` with `read` fails with a message holding `expected`.
template <typename Read>
void expect_read_fails_with(const json& document, Read&& read, std::string_view expected) {
    const std::string message = read_document_error(document, std::forward<Read>(read));
    EXPECT_FALSE(message.empty()) << "expected a failure with: " << expected;
    EXPECT_NE(message.find(expected), std::string::npos) << message;
}

TEST(VisualizerJsonReader, ReadsEachKindOfValue) {
    const json document = json::parse(R"({
        "name": "fabric_2d",
        "enabled": true,
        "count": 3,
        "address": 18446744073709551615,
        "offset": -4,
        "peer": null,
        "list": [1, 2, 3],
        "nested": {"inner": "x"}
    })");
    read_document(document, [](JsonObjectReader& object) {
        EXPECT_EQ(object.take<std::string>("name"), "fabric_2d");
        EXPECT_TRUE(object.take<bool>("enabled"));
        EXPECT_EQ(object.take<uint8_t>("count"), 3);
        EXPECT_EQ(object.take<uint64_t>("address"), UINT64_MAX);
        EXPECT_EQ(object.take<int32_t>("offset"), -4);
        EXPECT_EQ(object.take("peer").read_nullable<std::string>(), std::nullopt);
        EXPECT_EQ(object.take("list").read_array_of<uint32_t>(), (std::vector<uint32_t>{1, 2, 3}));
        const std::string inner = object.take("nested").read_object(
            [](JsonObjectReader& nested) { return nested.take<std::string>("inner"); });
        EXPECT_EQ(inner, "x");
    });
}

// A document built in memory holds a non-negative int as signed, while one parsed from text holds it as unsigned.
// Both read as unsigned, so a reader can read the writer's JSON with or without writing it out.
TEST(VisualizerJsonReader, ReadsAnInMemorySignedIntegerAsUnsigned) {
    json document;
    document["count"] = static_cast<int>(7);
    read_document(document, [](JsonObjectReader& object) { EXPECT_EQ(object.take<uint32_t>("count"), 7u); });
}

TEST(VisualizerJsonReader, MissingKeyFailsNamingItsPointer) {
    const json document = json::parse(R"({"run": {"run_id": "r"}})");
    expect_read_fails_with(
        document,
        [](JsonObjectReader& object) {
            object.take("run").read_object([](JsonObjectReader& run) {
                run.take<std::string>("run_id");
                run.take<uint32_t>("mpi_rank");
            });
        },
        "Test document: /run/mpi_rank is missing");
}

// The drift the reader exists to catch: a key the writer writes that the reader doesn't read fails, whatever its
// value, including one that equals a default.
TEST(VisualizerJsonReader, UntakenKeysFailNamingEachPointer) {
    const json document = json::parse(R"({"run": {"run_id": "r", "added": false, "renamed": 0, "listed": []}})");
    expect_read_fails_with(
        document,
        [](JsonObjectReader& object) {
            object.take("run").read_object([](JsonObjectReader& run) { run.take<std::string>("run_id"); });
        },
        "Test document: /run has keys this reader doesn't read: /run/added, /run/renamed, /run/listed");
}

TEST(VisualizerJsonReader, UntakenTopLevelKeyFails) {
    const json document = json::parse(R"({"manifest_version": 1, "extra": {}})");
    expect_read_fails_with(
        document,
        [](JsonObjectReader& object) { object.take<uint32_t>("manifest_version"); },
        "Test document: the top level has keys this reader doesn't read: /extra");
}

TEST(VisualizerJsonReader, OptionalKeyMayBeMissing) {
    const json document = json::parse(R"({"torus": {"y": true}})");
    read_document(document, [](JsonObjectReader& object) {
        const std::optional<JsonValueReader> torus = object.take_optional("torus");
        ASSERT_TRUE(torus.has_value());
        EXPECT_TRUE(torus->read_object([](JsonObjectReader& axes) { return axes.take<bool>("y"); }));
        EXPECT_FALSE(object.take_optional("credit_transport").has_value());
    });
}

TEST(VisualizerJsonReader, SkippedKeyMustBeThere) {
    const json document = json::parse(R"({"vocabulary": {"status": ["active"]}})");
    read_document(document, [](JsonObjectReader& object) { object.skip("vocabulary"); });
    expect_read_fails_with(
        json::object(),
        [](JsonObjectReader& object) { object.skip("vocabulary"); },
        "Test document: /vocabulary is missing");
}

TEST(VisualizerJsonReader, CheckedKeyMustHoldTheExpectedValue) {
    const json document = json::parse(R"({"manifest_version": 2, "file_kind": "fabric_manifest"})");
    expect_read_fails_with(
        document,
        [](JsonObjectReader& object) {
            object.check("file_kind", "fabric_manifest");
            object.check("manifest_version", 1);
        },
        "Test document: /manifest_version is 2, not 1");
}

TEST(VisualizerJsonReader, ValueOfAnotherTypeFails) {
    const json document = json::parse(R"({"text": "3", "negative": -1, "large": 300, "fraction": 1.5, "empty": null})");
    expect_read_fails_with(
        document,
        [](JsonObjectReader& object) { object.take<uint32_t>("text"); },
        "Test document: /text is \"3\", not an unsigned integer");
    expect_read_fails_with(
        document,
        [](JsonObjectReader& object) { object.take<uint32_t>("negative"); },
        "Test document: /negative is -1, not an unsigned integer");
    expect_read_fails_with(
        document,
        [](JsonObjectReader& object) { object.take<uint8_t>("large"); },
        "Test document: /large is 300, which doesn't fit 8 bits");
    expect_read_fails_with(
        document,
        [](JsonObjectReader& object) { object.take<int32_t>("fraction"); },
        "Test document: /fraction is 1.5, not an integer");
    expect_read_fails_with(
        document,
        [](JsonObjectReader& object) { object.take<std::string>("empty"); },
        "Test document: /empty is null, not a string");
    expect_read_fails_with(
        document,
        [](JsonObjectReader& object) { object.take("text").read_object([](JsonObjectReader&) {}); },
        "Test document: /text is \"3\", not an object");
}

TEST(VisualizerJsonReader, NullableReadsNullAsNullopt) {
    const json document = json::parse(R"({"ring_buffer": null, "peer": "M0/C1/W0", "local_sync": null})");
    read_document(document, [](JsonObjectReader& object) {
        EXPECT_EQ(object.take("ring_buffer").read_nullable<uint32_t>(), std::nullopt);
        EXPECT_EQ(object.take("peer").read_nullable<std::string>(), "M0/C1/W0");
        EXPECT_EQ(
            object.take("local_sync").read_nullable_object([](JsonObjectReader& sync) { return sync.take<int>("x"); }),
            std::nullopt);
    });
}

// An object keyed by id, such as meshes keyed M<mesh_id>, is read by taking its fixed keys and then each of the rest,
// in the document's order.
TEST(VisualizerJsonReader, TakeEachReadsTheRestInOrder) {
    const json document = json::parse(R"({"count": 2, "M1": {"n": 1}, "M0": {"n": 0}})");
    read_document(document, [](JsonObjectReader& object) {
        EXPECT_EQ(object.take<uint32_t>("count"), 2u);
        std::vector<std::string> keys;
        object.take_each([&keys](std::string_view key, const JsonValueReader& value) {
            keys.emplace_back(key);
            value.read_object([](JsonObjectReader& mesh) { mesh.take<uint32_t>("n"); });
        });
        EXPECT_EQ(keys, (std::vector<std::string>{"M1", "M0"}));
    });
}

TEST(VisualizerJsonReader, KeyTakenTwiceFails) {
    const json document = json::parse(R"({"run_id": "r"})");
    expect_read_fails_with(
        document,
        [](JsonObjectReader& object) {
            object.take<std::string>("run_id");
            object.skip("run_id");
        },
        "Test document: /run_id is read twice");
}

TEST(VisualizerJsonReader, ArrayElementsAreNamedByIndex) {
    const json document = json::parse(R"({"registers": [{"name": "a"}, {"name": "b", "extra": 1}]})");
    expect_read_fails_with(
        document,
        [](JsonObjectReader& object) {
            object.take("registers").read_array([](const JsonValueReader& element) {
                element.read_object([](JsonObjectReader& entry) { entry.take<std::string>("name"); });
            });
        },
        "Test document: /registers/1 has keys this reader doesn't read: /registers/1/extra");
}

TEST(VisualizerJsonReader, FailNamesTheValue) {
    const json document = json::parse(R"({"peer": "M0/C9/W0"})");
    expect_read_fails_with(
        document,
        [](JsonObjectReader& object) {
            const JsonValueReader peer = object.take("peer");
            peer.fail("names {}, a chip this manifest doesn't have", peer.read<std::string>());
        },
        "Test document: /peer names M0/C9/W0, a chip this manifest doesn't have");
}

}  // namespace
}  // namespace tt::tt_fabric::visualizer
