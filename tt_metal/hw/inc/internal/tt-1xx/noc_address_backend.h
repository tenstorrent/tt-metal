// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// NoC address backend for Wormhole and Blackhole, selected by include path the
// same way as noc_parameters.h. Both architectures encode addresses with raw
// (x, y) coordinates; call sites use the backend-neutral noc_address_backend
// alias and never name a concrete backend directly.

#include "internal/dataflow/noc_address_backend_xy.h"

namespace noc_address_backend = noc_address_backend_xy;
