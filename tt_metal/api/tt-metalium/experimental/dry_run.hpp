// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <exception>
#include <string>
#include <vector>

namespace tt::tt_metal::experimental::dry_run {

/**
 * Start a dry run. Until the matching end(), on any thread, command queues send no work to a device: programs are not
 * launched, and buffer writes and reads, events, finishes and trace replays return without touching it. A read
 * leaves its destination unfilled. Opening and closing devices works as usual.
 *
 * A program that would have been launched has its kernel builds queued instead. They run in the background in the
 * order they were queued, outlive the device, and fill the kernel cache. Work that follows the dry run does not have
 * to wait for them: a program that needs a kernel whose build is still queued builds it itself, and one that needs a
 * kernel being built waits for that build. Up-front precompile uses this to run test bodies only for the programs
 * they build. Dry runs nest.
 */
void begin();

/** End the dry run started by the matching begin(). Extra calls do nothing. */
void end();

bool active();

/**
 * No dry run for the lifetime of the object. Opening and closing a device must reach it also when a test does so
 * inside its body: the device's own programs are built and launched for real.
 */
class Suspended {
public:
    Suspended();
    ~Suspended();
    Suspended(const Suspended&) = delete;
    Suspended& operator=(const Suspended&) = delete;

private:
    int depth_;
};

/** A dry run for the lifetime of the object. */
struct Scope {
    Scope() { begin(); }
    ~Scope() { end(); }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
};

/**
 * End the queued kernel builds: the running ones are waited for, and the ones that have not started are dropped, or
 * with drop_queued false run and waited for as well. Dropping is for when nothing that follows needs the kernels.
 * Returns how many kernels queued builds have built in this process. A failed build is dropped as well: the program
 * that needs the kernel builds it again when it compiles, and reports the failure then.
 */
size_t finish_queued_builds(bool drop_queued = true);

/** Diagnostic: the kernels ("<name>/<hash>") whose builds dry runs queued for the first time since the last call. */
std::vector<std::string> take_queued_builds();

/**
 * Mark the stretch of a process in which dry runs happen, device opens and closes between them included. A fabric
 * mesh opened in a dry pass without a router launch (FabricManagerMode::ENABLED) queues the routers' kernel builds,
 * so the first open that launches the routers finds them built.
 */
void set_pass(bool in_dry_pass);
bool in_pass();

/**
 * Thrown in a dry run in place of constructing a Program or MeshWorkload from a ProgramSpec. Such an object is
 * compiled when it is constructed; a dry run queues its kernel builds and lets no uncompiled one come to exist.
 */
struct BuildsQueued : std::exception {
    const char* what() const noexcept override { return "dry run: kernel builds queued, no program constructed"; }
};

}  // namespace tt::tt_metal::experimental::dry_run
