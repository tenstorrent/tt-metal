// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_page_mapping.hpp>

namespace tt::tt_metal {

UncompressedBufferPageMapping generate_buffer_page_mapping(const Buffer& buffer);

}  // namespace tt::tt_metal
