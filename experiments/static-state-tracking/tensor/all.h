// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// sst::tensor — umbrella header.
// ----------------------------------------------------------------------------
// Includes every public header of the SST tensor layer in dependency order, so
// a consumer can `#include ".../tensor/all.h"` and get the full surface.
//
// Order:
//   1. format_traits.h  — DataFormat enum.
//   2. face_shape.h     — FaceShape<…> (depends on format_traits).
//   3. tile_shape.h     — TileShape<…> + tile_size_bytes_of + aliases.
//   4. resolver.h       — TileConfig + Resolver<TileShape<…>>.
//   5. tensor.h         — Tensor<…> + dataflow wrappers + free pop/push.
// ----------------------------------------------------------------------------

#include "format_traits.h"
#include "face_shape.h"
#include "tile_shape.h"
#include "resolver.h"
#include "tensor.h"
