// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The one part of a rank's exit (distributed_termination.cpp) that only a context implementation uses.

namespace tt::tt_metal::distributed::multihost::job_stop {

// Whether this rank runs inside tt-run-rank (TT_RUN_RANK_FD set): the wrapper then ends the job for a rank that
// fails; without one, the launcher's own abort is the only way to end the job.
bool in_rank_wrapper();

// Whether the stop signal (SIGTERM) itself has reached this rank. A peer's failure can reach an MPI wait (ULFM's
// MPI_ERR_PROC_FAILED) a few ms before the wrapper's SIGTERM does; the wait gives the signal that long to arrive, so
// the rank leaves as a stopped rank and not as a second failure.
bool stop_signal_seen();

}  // namespace tt::tt_metal::distributed::multihost::job_stop
