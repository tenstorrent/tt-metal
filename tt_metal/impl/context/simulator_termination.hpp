// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace tt::tt_metal {

// When TT_METAL_SIMULATOR is set, a default SIGINT/SIGTERM kills the process before MetalContext's
// atexit handler runs, so libttsim_exit never runs and the Zebu session stays reserved. Install a
// handler that std::exit()s instead, for any signal that still has the default disposition.
void install_simulator_termination_handler();

// A TT_FATAL during device setup throws, and gtest catches it, so the process keeps running and the
// Zebu job stays reserved. When a simulator session is active, exit instead so atexit releases it.
// Returns false when TT_METAL_SIMULATOR is unset; otherwise it does not return.
bool release_simulator_session(int exit_code, const char* reason);

}  // namespace tt::tt_metal
