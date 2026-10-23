// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// NoC address backend for Quasar, selected by include path the same way as
// noc_parameters.h. Call sites use the backend-neutral noc_address_backend
// alias and never name a concrete backend directly.
//
// Quasar NoC V1/V2 use the legacy XY encoding. Under NOC_ATT_ENABLED every
// operand is a complete translated 64-bit address built by the ATT backend
// over the selected map configuration - no mixing.

#if defined(NOC_ATT_ENABLED)
#include "internal/tt-2xx/quasar/noc/att/noc_address_backend_att.h"

namespace noc_address_backend = noc_address_backend_att;
#else
#include "internal/dataflow/noc_address_backend_xy.h"

namespace noc_address_backend = noc_address_backend_xy;
#endif
