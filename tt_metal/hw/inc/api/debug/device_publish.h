// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// DevicePublish sends a payload struct, or a few arguments, over device observability, with metadata fixed at compile
// time.
//
//     DevicePublish<NocSite{.api = "noc_async_read"_s, .type = NocType::Read}>(NocEvent{x, y, bytes});
//     DevicePublish<Name{"tiles"_s}>(count, ratio);  // the host reads Record<std::tuple<uint32_t, float>, Name>
//     DevicePublish<Name{"x"_s}>(Bf16{bits}, count);  // an argument can be a struct too
//     DevicePublish<Name{"y"_s}, kPublishBatched>(count);  // options follow the metadata
//     DevicePublish<kPublishBatched>(count);  // or stand alone when there is none
//     DevicePublish<>(count);  // with neither, the host reads Record<uint32_t, void>
//     DevicePublish<Name{"z"_s}, kNoInline>(count);  // only a call here, even when DevicePublish is inlined
//
// A record is visible to the host as soon as it is written. kPublishBatched lets it wait for its RISC's next batch of
// records, or the launch's end, which saves a fence per record for a tool that publishes often.
//
// Payload and metadata fields, and arguments, use fixed-width types: int8_t to int64_t, uint8_t to uint64_t, char,
// float, double, bool, enums with one of those as their underlying type, in metadata, String for text, and in a
// payload, Timestamp for a time Timestamp::now() takes. A payload carries no String: a call that would pick its text at
// runtime branches over DevicePublish calls instead, one per text, each with the text in its metadata, and kNoInline
// keeps each branch to a call. long, size_t and the like differ in width between the device (RV32) and the host
// (x86-64): a struct with one fails to load, and an argument arrives at its device width. A struct is a named,
// non-template aggregate outside functions and anonymous or inline namespaces, and so is an enum passed as an argument:
// the host matches call sites by type name, and the two compilers spell those other names differently. A struct, as a
// payload or as an argument, travels as its bytes up to its last field's end, wherever the compiler put its fields:
// padding between its fields goes out, so fields ordered largest first cost nothing, and padding after them doesn't.
// Bit-fields, packed and alignas members are fine. A union is not: send the member it holds. The host
// reads a String in metadata as a std::string_view and a Timestamp on its own timeline too; every other field arrives
// as its own bits.
//
// A string is written "..."_s, which keeps the text in a section the device never loads and makes the String the text's
// 64-bit hash, by which the host looks the text up. A tool's macro that takes a literal passes it through
// TT_RECORD_STRING, which adds the _s, so kernel code passes the macro a plain literal.
#pragma once

#include "internal/debug/device_publish_impl.h"

// Publishes from the line it is written on, whether called there or handed to a scope that calls it when it closes.
#define DevicePublish TT_DEVICE_PUBLISH_INTERNAL
