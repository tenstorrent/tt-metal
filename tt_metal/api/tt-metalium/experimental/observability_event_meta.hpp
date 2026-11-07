// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// An observability event's metadata as the host holds it, and the way back to the struct the device declared it with.
// The metadata types and their rules are shared with the device: observability/api/observability_event_meta.h.
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

#include <observability/api/observability_event_meta.h>

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

    // The value as T, or nullopt when the event carries another type. When the event's type has T's name but not its
    // fields, or a value that does not fit T's field, the type differs between the device and the host: as() logs a
    // warning naming both signatures, once, and returns nullopt.
    template <class T>
    std::optional<T> as() const;
};

namespace detail {
// Logs, once per signature pair and reason, that a record's type has T's name but does not read back as T.
void report_meta_mismatch(std::string_view record_signature, std::string_view host_signature, std::string_view reason);
}  // namespace detail

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
    const std::string& expected = signature_of<T>();
    if (signature != expected) {
        if (type_name() == std::string_view(expected).substr(0, expected.rfind(':'))) {
            detail::report_meta_mismatch(
                signature, expected, "the type's fields differ between the device and the host");
        }
        return std::nullopt;
    }
    T t{};
    size_t i = 0;
    size_t first_unfit = 0;  // 1-based, 0 when every value fits
    detail::for_each_leaf(t, [&](auto& f) {
        using F = detail::field_t<decltype(f)>;
        const FieldValue& v = fields[i++];  // the signature matched, so there is one value per leaf
        if constexpr (std::is_same_v<F, const char*>) {
            f = v.s;
        } else if (!detail::field_from_bits(v.bits, f) && first_unfit == 0) {
            first_unfit = i;
        }
    });
    if (first_unfit != 0) {
        detail::report_meta_mismatch(
            signature,
            expected,
            "leaf field " + std::to_string(first_unfit) + " holds a value that does not fit the host's field type");
        return std::nullopt;
    }
    return t;
}

}  // namespace tt::observability_event
