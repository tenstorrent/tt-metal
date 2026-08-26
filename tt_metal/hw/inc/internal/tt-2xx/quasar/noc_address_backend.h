// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// NoC address backend for Quasar, selected by include path the same way as
// noc_parameters.h. Call sites use the backend-neutral noc_address_backend
// alias and never name a concrete backend directly.
//
// Quasar NoC V1/V2 use the legacy XY encoding. NoC V3 will select between the
// XY encoding and the ATT (Address Translation Table) encoding
// (noc_address_backend_att) here.

#include "internal/dataflow/noc_address_backend_xy.h"

namespace noc_address_backend = noc_address_backend_xy;
