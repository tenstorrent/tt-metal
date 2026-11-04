// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace tt::tt_metal {

// Makes SIGINT/SIGTERM exit through atexit when TT_METAL_SIMULATOR is set. Dying from the signal
// skips MetalContext teardown, so the simulator is never told to exit. Only installed over a
// default disposition. Call before registering the teardown atexit handler.
void install_simulator_termination_handler();

// Marks the process as exiting, so a later signal bounds that teardown instead of starting a second
// exit. Call after registering the teardown atexit handler, so this one runs first.
void install_simulator_exit_guard();

// Exits through atexit instead of returning to a caller that would keep running on a dead context.
// Returns false when there is nothing to release; otherwise it does not return.
bool release_simulator_session(int exit_code, const char* reason);

}  // namespace tt::tt_metal
