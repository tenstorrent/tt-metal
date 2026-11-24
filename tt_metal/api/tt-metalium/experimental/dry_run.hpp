// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <exception>

namespace tt::tt_metal::experimental::dry_run {

/**
 * Start a dry run. Until the matching end(), on any thread, command queues send no work to a device: programs are not
 * launched, and buffer writes and reads, events, finishes and trace replays return without touching it. A read
 * leaves its destination unfilled. Opening and closing devices works as usual.
 *
 * A program that would have been launched has its kernel builds queued instead. They run in the background, outlive
 * the device, and fill the kernel cache. Up-front precompile uses this to run test bodies only for the programs
 * they build. Dry runs nest.
 */
void begin();

/** End the dry run started by the matching begin(). Extra calls do nothing. */
void end();

bool active();

/** Block until every kernel build queued so far has finished. A failed build is dropped: the program that needs the
 * kernel builds it again when it compiles, and reports the failure then. */
void wait_for_queued_builds();

/**
 * Thrown in a dry run in place of constructing a Program or MeshWorkload from a ProgramSpec. Such an object is
 * compiled when it is constructed; a dry run queues its kernel builds and lets no uncompiled one come to exist.
 */
struct BuildsQueued : std::exception {
    const char* what() const noexcept override { return "dry run: kernel builds queued, no program constructed"; }
};

}  // namespace tt::tt_metal::experimental::dry_run
