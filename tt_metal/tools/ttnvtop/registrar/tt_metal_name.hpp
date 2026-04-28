// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Bridge header for deriving a short human-readable name from a
// tt-metal Program. Separated from `ttnvtop_register.hpp` so the core
// registrar library stays tt-metal-free; only this header (and its
// corresponding `tt_metal_name.cpp`) touches tt-metal types.
//
// The implementation is linked into libtt_metal (see
// tt_metal/impl/sources.cmake) — callers inside tt-metal include this
// header; callers outside tt-metal should use `register_program`
// directly with their own name.

#pragma once

#include <string>

namespace tt::tt_metal::detail {
class ProgramImpl;
}  // namespace tt::tt_metal::detail

namespace ttnvtop {

// Derive a short name for a program: basename of the first kernel's
// source file, stripped of extension. Falls back to "prog_<runtime_id>"
// if the program has no kernels with a file-path source.
std::string ttnvtop_program_name(const tt::tt_metal::detail::ProgramImpl& program);

}  // namespace ttnvtop
