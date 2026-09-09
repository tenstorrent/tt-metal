// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Temporary negative CI test: this commit must fail all four added API checks.
// Revert the probe after recording the diagnostics from header verification.
namespace tt::tt_metal::experimental::api_hygiene_probe {

struct ImportedType {};

// cppcoreguidelines-avoid-non-const-global-variables
inline int mutable_state = 0;

// readability-named-parameter requires a visible definition in clang-tidy 20.
inline int unnamed_parameter(int) { return 0; }

}  // namespace tt::tt_metal::experimental::api_hygiene_probe

// google-global-names-in-headers
using tt::tt_metal::experimental::api_hygiene_probe::ImportedType;

namespace tt::tt_metal::experimental::api_hygiene_probe_consumer {

// google-build-using-namespace
using namespace tt::tt_metal::experimental::api_hygiene_probe;

}  // namespace tt::tt_metal::experimental::api_hygiene_probe_consumer
