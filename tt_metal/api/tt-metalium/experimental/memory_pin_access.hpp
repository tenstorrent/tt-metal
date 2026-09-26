// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace tt::tt_metal {

class MemoryPin;

namespace experimental {

/**
 * @brief Marks the memory kept alive by `pin` as never written while any copy of the pin is alive.
 *
 * Every copy of `pin` shares the mark. A host-to-device upload from a HostBuffer whose pin is marked may return
 * before the device has finished reading the memory; the upload keeps a copy of the pin until the reads complete.
 * Without the mark, uploads return only once the device no longer reads the caller's memory.
 *
 * Only code that created the storage read-only (for example a PROT_READ file mapping it owns) may mark it: writing
 * the memory while a copy of the pin is alive can corrupt data still being uploaded. `pin` must not be empty.
 */
void MemoryPinMarkDeviceImmutable(MemoryPin& pin);

/**
 * @brief Returns whether `pin` (or any copy of it) was marked with MemoryPinMarkDeviceImmutable.
 */
bool MemoryPinIsDeviceImmutable(const MemoryPin& pin);

}  // namespace experimental

}  // namespace tt::tt_metal
