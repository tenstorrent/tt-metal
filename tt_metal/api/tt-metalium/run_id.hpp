// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

namespace tt::tt_metal {

// Identifier shared by every reporting artefact a single run produces, so that a TT-NN memory
// report and a Tracy performance report captured from the same process can be paired afterwards
// instead of guessed at. Currently emitted into the profiler device log preamble and the TT-NN
// graph report metadata.
//
// Seeded from TT_METAL_RUN_ID when that is set, which is how the ranks of a multi-process run are
// made to agree on one value (launchers such as mpirun propagate the environment). When it is
// unset a value is minted on first use, so each process gets its own; single-process runs, which
// are the common case, need no configuration.
const std::string& get_or_create_run_id();

}  // namespace tt::tt_metal
