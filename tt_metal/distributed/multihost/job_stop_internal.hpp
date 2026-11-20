// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The parts of the job stop (distributed_termination.cpp) that only a context implementation uses.

namespace tt::tt_metal::distributed::multihost::job_stop {

// What the stop watchdog calls when a rank is still running TT_METAL_STOP_TIMEOUT_S after its stop began: the MPI
// context sets MPI_Abort here. Without one the watchdog ends this process alone.
void set_force_abort(void (*force_abort)(int exit_code));

// Whether this process was started by a launcher that can stop the job (PMIx job control).
bool launcher_reachable();

// MPI reported a failed peer (MPI_ERR_PROC_FAILED): this rank's exit fence is then bounded short, since the dead
// rank's stand-in may never reach it.
void note_peer_failed();

// One line to the tt-run-rank wrapper this rank runs in (TT_RUN_RANK_FD), when there is one: "context" once the
// distributed context exists (the wrapper then stands in for this rank at the exit fence if it dies), "stopped" when
// this rank left a stopping job itself.
void tell_rank_wrapper(const char* word);

}  // namespace tt::tt_metal::distributed::multihost::job_stop
