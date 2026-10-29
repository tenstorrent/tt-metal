// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// TEST ONLY. Found by a tt-metal kernel's #include "api/dataflow/dataflow_buffer.h" when the tt-llk harness builds it
// (tests/helpers is on the include path): tt-metal's DataflowBuffer, as the harness's stand-in.

#pragma once

#include "test_dataflow_buffer.h"

using DataflowBuffer = llk2_test::DataflowBuffer;
