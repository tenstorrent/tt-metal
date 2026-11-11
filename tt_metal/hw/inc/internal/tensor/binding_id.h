// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace tensor_accessor {

// Sentinel BindingId meaning "no op-to-op binding id tracked" for this accessor. Real ids are small
// per-binding CRTA byte offsets (the binding's base-address word), so 0xFFFFFFFF never collides. See
// DistributionSpec::binding_id (internal/tensor/dspec.h) and the op-to-op R/W inference note emit
// (api/dataflow/buf_rw_note.h).
inline constexpr uint32_t NO_BINDING_ID = 0xFFFFFFFFu;

}  // namespace tensor_accessor
