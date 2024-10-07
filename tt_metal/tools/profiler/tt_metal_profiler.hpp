// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace tt::tt_metal {

inline namespace v0 {
class Device;
class Program;

void DumpDeviceProfileResults(Device* device, const Program& program);
}  // namespace v0


}  // namespace tt::tt_metal
