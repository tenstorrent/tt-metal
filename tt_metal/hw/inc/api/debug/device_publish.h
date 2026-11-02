// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// DevicePublish sends a record to the host. The record's payload is the call's arguments, and its metadata is a struct
// value given as a template argument.
//
//     DevicePublish<NocSite{.api = "noc_async_read"_s, .type = NocType::Read}>(NocEvent{x, y, bytes});
//     DevicePublish<Name{"tiles"_s}>(count, ratio);
//     DevicePublish<>(count);                                  // No metadata.
//     DevicePublish<Name{"tiles"_s}, kPublishBatched>(count);  // With an option.
//
// Options:
//   kPublishBatched         Defers the record until the RISC's next batch or the end of the launch, saving a fence.
//   kNoInline               Prevents inlining the publish at this call site, saving code size.
//
// A host callback registered for Record<Payload, Metadata> receives the records of calls whose metadata type is
// Metadata (void for calls without metadata) and whose arguments match Payload:
//   void                     No arguments.
//   T                        One argument of type T.
//   std::tuple<T1, T2, ...>  Two or more arguments of those types.
//   Variadic<T>              Any number of arguments of type T.
// Each T above can also be one of:
//   std::array<E, N>         A std::array or C array of N elements of type E.
//   std::span<const E>       A std::array or C array of any length.
//   std::variant<E1, E2...>  An argument of any of those types.
// Use std::variant to receive, in one callback, a value whose type differs between call sites:
//
//     DevicePublish<Name{"temperature"_s}>(celsius);  // A float.
//     DevicePublish<Name{"tiles"_s}>(tiles);          // A uint32_t.
//
//     RegisterCallback([](const Batch<Record<std::variant<float, uint32_t>, Name>>& batch) { ... });
//
// Supported types for arguments and for the fields of payload and metadata structs:
//   - int8_t to int64_t, uint8_t to uint64_t, char, float, double and bool.
//   - Enums whose underlying type is one of the above.
//   - Timestamp, from Timestamp::now(), in arguments and payloads only.
//   - Structs, std::arrays and C arrays of the above.
// Metadata fields can also be String, written "..."_s; on the host, a String is a std::string_view. Payloads can't
// contain String. To publish text chosen at runtime, use one DevicePublish call per text, with the text in its metadata
// and kNoInline.
//
// Types whose size differs between the device and the host, such as long and size_t, are not supported.
//
// Structs, and enums passed as arguments, must be named, non-template types that are not declared in a function or in
// an anonymous or inline namespace. Unions are not supported. Bit-fields, packed members and alignas members are
// supported. A struct is sent up to the end of its last field, including padding between fields; order fields from
// largest to smallest to avoid padding.
//
// To pass a string literal from a macro to DevicePublish, wrap it in TT_RECORD_STRING, which appends _s.
#pragma once

#include "internal/debug/device_publish_impl.h"

// A record's location is the file, line and function where DevicePublish is written.
#define DevicePublish TT_DEVICE_PUBLISH_INTERNAL
