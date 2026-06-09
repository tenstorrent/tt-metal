// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <string>
#include <vector>

// Parser for the TT_KERNEL-tagged kernel entry point.
//
// This is the Phase 1 "signature parser" from the design doc
// (tech_reports/NamedKernelArgs/kernel_args_as_parameters.md, §4). It runs as a step before
// JIT compilation: it scans the raw (un-preprocessed) kernel source for the single TT_KERNEL
// marker and extracts the entry's name, its template non-type parameters (the compile-time
// args / CTAs) and its function parameters (the runtime args / RTAs+CRTAs).
//
// Binding is by name and — in Phase 1 — every parameter type is uint32_t, so the parser only
// needs names, their order, and which list each is in. It is a name extractor, not a type
// parser; a lightweight tokenizer suffices. Richer types (Phase 2) will require a real parser.

namespace tt::tt_metal {

// The extracted TT_KERNEL entry signature.
struct KernelMainSignature {
    std::string name;                               // entry function name
    std::vector<std::string> template_param_names;  // non-type template params -> CTAs (in order)
    std::vector<std::string> fn_param_names;        // function params -> RTAs/CRTAs (in order)
};

// Scan kernel source for the TT_KERNEL marker and return the parsed entry signature.
//
// Returns std::nullopt when the source contains no TT_KERNEL marker (a legacy / hand-written
// `void kernel_main()` kernel — fully backward compatible).
//
// Throws std::runtime_error when the source has a TT_KERNEL marker but its signature is
// malformed or outside the Phase 1 surface, e.g.: more than one marker; a non-`void` return
// type; a non-`uint32_t` parameter type; a `typename`/`class` template parameter; a defaulted,
// unnamed, variadic, pointer/reference, or otherwise unsupported parameter.
std::optional<KernelMainSignature> parse_kernel_main_signature(const std::string& source);

}  // namespace tt::tt_metal
