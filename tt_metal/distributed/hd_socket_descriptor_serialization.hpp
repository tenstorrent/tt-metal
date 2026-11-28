// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/experimental/sockets/hd_socket_descriptor.hpp>

#include "hd_socket_descriptor_generated.h"

namespace tt::tt_metal::distributed {

// Flatbuffer conversion for HDSocketDescriptor, shared by the socket file and the
// H2D/D2H service descriptors that embed one per coord.
flatbuffers::Offset<flatbuffer::HDSocketDescriptor> to_flatbuffer(
    flatbuffers::FlatBufferBuilder& builder, const HDSocketDescriptor& desc);
HDSocketDescriptor from_flatbuffer(const flatbuffer::HDSocketDescriptor& fb);

}  // namespace tt::tt_metal::distributed
