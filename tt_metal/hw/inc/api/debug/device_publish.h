// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// DevicePublish sends a payload struct, or a few arguments, over the streaming profiler, with metadata fixed at compile
// time.
//
//     DevicePublish<NocSite{.api = "noc_async_read"_s, .type = NocType::Read}>(NocEvent{x, y, bytes});
//     DevicePublish<Name{"tiles"_s}>(count, ratio);  // the host reads Record<Arguments<V>, Name>
//     DevicePublish<Name{"x"_s}>(Bf16{bits}, count);  // an argument can be a struct too
//     DevicePublish<Name{"y"_s}, kPublishImmediately>(count);  // options follow the metadata
//
// Payload and metadata fields, and arguments, use fixed-width types: int8_t to int64_t, uint8_t to uint64_t, char,
// float, double, bool, enums with one of those as their underlying type, String for text, and in a payload, Timestamp
// for a time Timestamp::now() takes. long, size_t and the like differ in width between the device (RV32) and the host
// (x86-64): a struct with one fails to load, and an argument arrives at its device width. A struct is a named,
// non-template aggregate outside functions and anonymous or inline namespaces, and so is an enum passed as an argument:
// the host matches call sites by type name, and the two compilers spell those other names differently. A struct, as a
// payload or as an argument, travels as its bytes up to its last field's end: padding between its fields goes out, so
// fields ordered largest first cost nothing, and padding after them doesn't. Bit-fields, packed and alignas members are
// fine, and a struct its field types don't lay out travels whole. A union is not: send the member it holds. The host
// reads a String as a std::string_view and a Timestamp on its own timeline too; every other field arrives as its own
// bits.
//
// A string is written "..."_s, which copies the literal into a named array in a section the device never loads, where
// the host reads it; a template argument can point only at such an array. A tool's macro that takes a literal passes it
// through TT_RECORD_STRING, which adds the _s, so kernel code passes the macro a plain literal.
#pragma once

#include "api/debug/kernel_profiler.hpp"
#include "internal/debug/device_publish.h"
