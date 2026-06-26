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

// Cross-check a parsed TT_KERNEL signature against the host-registered argument schema, by name.
//
// The kernel's template parameters must name exactly the registered compile-time args (CTAs), and
// its function parameters must name exactly the registered runtime args — the union of per-core
// RTAs and common CRTAs, since the kernel can't (and shouldn't) tell the two apart. Resource
// bindings (tensor/DFB/semaphore accessors) are reached through their own namespaces, not passed
// as parameters, so they are deliberately not part of this check. The comparison is set-based:
// the generated shim binds every argument by name, so declaration order is irrelevant.
//
// Throws std::runtime_error naming the offending arguments when either set doesn't match. This
// converts two otherwise-poor failure modes into one clear, early error: a kernel parameter the
// host never registered (otherwise a compile error inside generated code), and a registered
// argument the kernel never takes (otherwise silently generated-but-unused).
void validate_signature_against_schema(
    const KernelMainSignature& sig,
    const std::vector<std::string>& cta_names,
    const std::vector<std::string>& rta_names,
    const std::vector<std::string>& crta_names);

// Generate the kernel_main() shim text for a parsed TT_KERNEL entry: a void kernel_main() that
// calls the user entry, passing every argument by name through get_arg(args::<name>) — template
// parameters (CTAs) in the angle brackets (constexpr), function parameters (RTAs/CRTAs) in the
// parentheses (runtime L1 reads). The args:: accessors come from the generated
// kernel_args_generated.h, so the shim must be emitted after that header and after the user
// source. Parameter declaration order is preserved exactly — it is load-bearing, since the call
// binds positionally to the entry's parameters.
std::string generate_kernel_main_shim(const KernelMainSignature& sig);

}  // namespace tt::tt_metal
