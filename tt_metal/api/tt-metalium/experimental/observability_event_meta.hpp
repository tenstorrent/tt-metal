// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// An observability event's metadata as the host holds it, and the way back to the struct the device declared it with.
// The metadata types and their rules are shared with the device: hostdev/api/observability_event_meta.h.
//
// This API is experimental and may change or be removed without notice.
//
//     if (auto m = event.meta->as<tt::observability_event::ZoneColorMeta>()) {
//         use(m->name, m->color);
//     }
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <hostdev/api/observability_event_meta.h>

namespace tt::observability_event {

// One unmarshalled field: s for 's', bits for the integer codes.
struct FieldValue {
    char code = 0;
    uint64_t bits = 0;
    const char* s = nullptr;  // valid for the life of the process
};

// An event's metadata as the host holds it before anyone asks for a type.
struct Metadata {
    std::string_view signature;      // "<type name>:<field codes>"
    std::vector<FieldValue> fields;  // one per leaf field, nested structs flattened

    std::string_view type_name() const { return signature.substr(0, signature.rfind(':')); }

    // The value as T, or nullopt when the event carries another type.
    template <class T>
    std::optional<T> as() const;
};

template <class T>
const std::string& signature_of() {
    static const std::string sig = [] {
        struct Sink {
            std::string s;
            void put(char c) { s.push_back(c == '"' || c == '\\' ? '_' : c); }
        } out;
        detail::write_signature<T>(out);
        return std::move(out.s);
    }();
    return sig;
}

template <class T>
std::optional<T> Metadata::as() const {
    if (signature != signature_of<T>()) {
        return std::nullopt;
    }
    T t{};
    size_t i = 0;
    detail::for_each_leaf(t, [&](auto& f) {
        using F = detail::field_t<decltype(f)>;
        const FieldValue& v = fields[i++];  // the signature matched, so there is one value per leaf
        if constexpr (std::is_same_v<F, const char*>) {
            f = v.s;
        } else if constexpr (std::is_same_v<F, bool>) {
            f = v.bits != 0;
        } else if constexpr (std::is_enum_v<F>) {
            f = static_cast<F>(static_cast<std::make_unsigned_t<std::underlying_type_t<F>>>(v.bits));
        } else {
            f = static_cast<F>(static_cast<std::make_unsigned_t<F>>(v.bits));
        }
    });
    return t;
}

}  // namespace tt::observability_event
