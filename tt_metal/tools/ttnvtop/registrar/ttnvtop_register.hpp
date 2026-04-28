// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// ttnvtop program registrar — public API.
//
// Workloads call `register_program(runtime_id, name)` to publish a
// `{runtime_id -> name}` mapping into the shared-memory registry at
// `/dev/shm/tt_program_registry`. The ttnvtop viewer reads that file to
// show human-readable names next to the numeric runtime_ids it samples
// from device mailboxes.
//
// Behavior is gated on the `TTNVTOP_REGISTER_PROGRAMS` env var. When the
// env var is unset or not equal to "1", `register_program` is a hot-path
// no-op (one atomic load and return). This header is intentionally
// dependency-free so any caller — including tt-metal core — can include
// it without pulling additional libraries.

#pragma once

#include <cstdint>

namespace ttnvtop {

// Publish a `{runtime_id -> name}` mapping to the registry. Safe to call
// from any thread. Safe to call with `name == nullptr` (stored as
// "(unnamed)"). Names longer than `kRegistryNameMax - 1` bytes are
// truncated. When the env var gate is off this returns after a single
// atomic load with no allocation or syscall.
void register_program(uint32_t runtime_id, const char* name);

}  // namespace ttnvtop
