// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Op-to-op R/W inference (POC): the device-side NoC read/write APIs emit one fixed 8-byte (slot, kind)
// record into the non-allocatable ".tt.BUF_RW" section (api/dataflow/buf_rw_note.h, which reopens this
// namespace to add the device-only emit machinery); the host parses it
// (tt_metal/llrt/binary_metadata.hpp). The constants live here, reachable from both, so the two sides
// read/write the same values directly instead of hand-duplicating them under a "must match" comment.
namespace tt_buf_rw {

// kOpaque is 0 so a zero-filled record degrades to the conservative "un-analyzable" state rather
// than being mistaken for a plain read; malformed section sizes are rejected separately by the
// parser. Neither covers a record that was never emitted: an un-instrumented raw-address call site
// produces no bytes at all, so a missing note does NOT set opaque
constexpr uint32_t kOpaque = 0;
constexpr uint32_t kRead = 1;
constexpr uint32_t kWrite = 2;

// Sentinel slot value a device-emitted OPAQUE record carries. The host does not currently key off this
// (it only switches on kind), but the two sides' choice of sentinel is kept shared regardless.
constexpr uint32_t kOpaqueSlot = 0xFFFFFFFFu;

}  // namespace tt_buf_rw
