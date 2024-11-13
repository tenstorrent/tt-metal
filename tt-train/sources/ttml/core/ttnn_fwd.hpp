// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace tt::tt_metal {
struct Tensor;
class CommandQueue;
struct MemoryConfig;
class DeviceMesh;
class LegacyShape;
inline namespace v0 {
class Device;
}  // namespace v0
}  // namespace tt::tt_metal

namespace ttnn {
using Tensor = tt::tt_metal::Tensor;  // not sure if it works but we can use original tensor namespace

}  // namespace ttnn
