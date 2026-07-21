// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Stub for the CMAC rtoptions unit test referenced by sources.cmake (commit
// ae0d9483). The intended tests for has_external_cmac_ports() / parsing of
// TT_METAL_EXTERNAL_CMAC_PORTS / TT_METAL_CMAC_RS_FEC / TT_METAL_CMAC_TX_RATE
// have not been written yet; this stub keeps the build green so the smoke
// rig can run.

#include <gtest/gtest.h>

TEST(CmacRtoptions, Placeholder) { SUCCEED() << "Stub — implement real coverage of rtoptions CMAC parsing."; }
