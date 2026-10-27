// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <map>
#include <string_view>
#include <variant>

#include <tt_stl/assert.hpp>
#include <tt_stl/overloaded.hpp>

#include "tt_metal/fabric/debug/visualizer/manifest/struct_layout.hpp"

// The manifest's types block holds every struct, enum and tagged union its schemas name. The writer adds the type of
// each schema it writes to a TypeCollector, which follows each element to its description (struct_layout.hpp), and
// writes the types block from what it collected.

namespace tt::tt_fabric::manifest {

class TypeCollector {
public:
    // Adds `type`'s element names, if it names one, and every type that each element names.
    void add(const layout::Type& type) {
        std::visit(
            ttsl::overloaded{
                [this](const layout::element::Struct& element) { add_struct(*element.description); },
                [this](const layout::element::Enum& element) { add_enum(*element.description); },
                [this](const layout::element::TaggedUnion& element) { add_tagged_union(*element.description); },
                [](const auto&) {},
            },
            type.element);
    }

    // Each type kind, by name.
    const std::map<std::string_view, const layout::StructDescription*>& get_structs() const { return structs_; }
    const std::map<std::string_view, const layout::EnumDescription*>& get_enums() const { return enums_; }
    const std::map<std::string_view, const layout::TaggedUnionDescription*>& get_tagged_unions() const {
        return tagged_unions_;
    }

private:
    void add_struct(const layout::StructDescription& description) {
        if (insert(structs_, description)) {
            for (const auto& member : description.members) {
                add(member.type);
            }
        }
    }

    void add_enum(const layout::EnumDescription& description) { insert(enums_, description); }

    void add_tagged_union(const layout::TaggedUnionDescription& description) {
        if (insert(tagged_unions_, description)) {
            add_enum(*description.tag);
            for (const auto& union_case : description.cases) {
                add(union_case.member.type);
            }
        }
    }

    // True when `description` is unique. Two different types with one name would make its schema ambiguous.
    template <typename Description>
    static bool insert(std::map<std::string_view, const Description*>& by_name, const Description& description) {
        const auto [it, inserted] = by_name.emplace(description.name, &description);
        TT_FATAL(
            inserted || it->second == &description,
            "Fabric manifest: two different types are named {}, so its schema is ambiguous",
            description.name);
        return inserted;
    }

    std::map<std::string_view, const layout::StructDescription*> structs_;
    std::map<std::string_view, const layout::EnumDescription*> enums_;
    std::map<std::string_view, const layout::TaggedUnionDescription*> tagged_unions_;
};

}  // namespace tt::tt_fabric::manifest
