// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compiles the layouts' coverage checks into the library, so a struct change fails the build even where nothing
// else includes the layouts.
#include "tt_metal/fabric/manifest/fabric_struct_layouts.hpp"
