# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Compile-time gate for the descriptor-only output of the migration flow."""

import json
import re
import shlex
from pathlib import Path

from tools.generic_op_to_factory.export_run import ExportError, _hash_file, _safe_path


PROBE = r"""
#include "@HEADER@"
#include "ttnn/operation_concepts.hpp"
#include <concepts>
#include <optional>
#include <utility>
#include <variant>

template <class Op, class Factory>
consteval bool check_factory() {
    static_assert(ttnn::device_operation::ProgramDescriptorFactoryConcept<Factory>,
                  "Migration requires a ProgramDescriptor factory, not CachedProgram or ProgramSpec");
    static_assert(!requires { &Factory::create_workload_descriptor; },
                  "WorkloadDescriptor requires a separately approved migration flow");
    static_assert(!requires { &Factory::apply_descriptor; },
                  "Factory must not shadow the descriptor adapter");
    using A = typename Op::operation_attributes_t;
    using T = typename Op::tensor_args_t;
    using O = typename Op::tensor_return_value_t;
    using C = std::optional<ttnn::MeshCoordinate>;
    static_assert(!requires(const A& a, const T& t, O& o, const ttnn::MeshCoordinateRangeSet& c) {
        Factory::create_workload_descriptor(a, t, o, c);
    }, "Callable workload descriptor factories require a separately approved flow");
    constexpr bool plain = requires(const A& a, const T& t, O& o) {
        { Factory::create_descriptor(a, t, o) } -> std::same_as<tt::tt_metal::ProgramDescriptor>;
    };
    constexpr bool per_coordinate = requires(const A& a, const T& t, O& o, const C& c) {
        { Factory::create_descriptor(a, t, o, c) } -> std::same_as<tt::tt_metal::ProgramDescriptor>;
    };
    static_assert(plain || per_coordinate, "Factory must return ProgramDescriptor with a supported signature");
    static_assert(requires(tt::tt_metal::Program& p, const A& a, const T& t, O& o, const C& c) {
        { Factory::override_runtime_arguments(p, a, t, o, c) } -> std::same_as<void>;
    }, "Factory must provide the adapter's explicit per-Program cache refresh hook");
    static_assert(!requires(tt::tt_metal::Program& p, const A& a, const T& t, O& o, const C& c) {
        Op::override_runtime_arguments(p, a, t, o, c);
    }, "Refresh hook belongs on the factory, not on the device operation");
    static_assert(!requires(const A& a, const T& t, O& o, const C& c) {
        Op::get_dynamic_runtime_args(a, t, o, c);
    }, "Legacy dynamic runtime args cannot be combined with the explicit refresh hook");
    return true;
}

template <class Op, std::size_t... I>
consteval bool check_operation(std::index_sequence<I...>) {
    static_assert(sizeof...(I) > 0, "Operation must declare a program factory");
    return (check_factory<Op, std::variant_alternative_t<I, typename Op::program_factory_t>>() && ...);
}

using MigrationOperation = @OPERATION@;
static_assert(check_operation<MigrationOperation>(
    std::make_index_sequence<std::variant_size_v<typename MigrationOperation::program_factory_t>>{}));
"""


def validate(specification, runtime):
    required = {"operation_header", "operation_type", "factory_source"}
    if not isinstance(specification, dict) or specification.keys() != required:
        raise ExportError("factory_contract requires operation_header, operation_type and factory_source")
    name = specification["operation_type"]
    if not isinstance(name, str) or not re.fullmatch(r"(?:::)?[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*", name, re.ASCII):
        raise ExportError("operation_type must be a qualified C++ type name, not an expression")
    runtime = Path(runtime).resolve()
    for key, suffixes in (("operation_header", (".h", ".hpp")), ("factory_source", (".cpp", ".cc"))):
        relative = _safe_path(specification[key])
        if not re.fullmatch(r"[A-Za-z0-9_./-]+", relative) or not relative.startswith("ttnn/"):
            raise ExportError("Factory contract paths must be tt-metal-relative paths below ttnn/")
        path = runtime / relative
        if path.suffix not in suffixes or not path.is_file() or path.resolve() != path:
            raise ExportError(f"Factory contract source is missing, redirected or has the wrong type: {relative}")
    return dict(specification)


def render(specification):
    return PROBE.replace("@HEADER@", specification["operation_header"]).replace(
        "@OPERATION@", specification["operation_type"]
    )


def compile_invocation(specification, runtime, probe):
    """Reuse the actual factory TU's build flags, including CMake unity builds.

    No shell evaluation, fallback flags, build-system edits or linking. Ambiguous
    or unsupported compilation databases fail explicitly.
    """
    runtime = Path(runtime).resolve()
    database = runtime / "build_Release/compile_commands.json"
    if not database.is_file() or database.resolve() != database:
        raise ExportError("Descriptor contract needs build_Release/compile_commands.json from the target build")
    desired = runtime / specification["factory_source"]
    candidates = []
    for entry in json.loads(database.read_bytes()):
        directory = Path(entry["directory"])
        if not directory.is_absolute() or not directory.resolve().is_relative_to(runtime):
            continue
        source = Path(entry["file"])
        source = source if source.is_absolute() else directory / source
        source = source.resolve()
        if not source.is_relative_to(runtime):
            continue
        matches = source == desired
        if not matches and re.fullmatch(r"unity_\d+_cxx\.cxx", source.name) and source.is_file():
            includes = re.findall(r'^\s*#include\s+"([^"\n]+)"\s*$', source.read_text(), re.MULTILINE)
            matches = any((source.parent / include).resolve() == desired for include in includes)
        if matches:
            candidates.append((entry, directory, source))
    if len(candidates) != 1:
        raise ExportError("Factory source must match exactly one target compile command (direct or CMake unity)")
    entry, directory, source = candidates[0]
    arguments = entry.get("arguments")
    if arguments is None:
        arguments = shlex.split(entry["command"])
    if (
        not isinstance(arguments, list)
        or not arguments
        or any(not isinstance(arg, str) or not arg for arg in arguments)
        or any(arg in (";", "&&", "||", "|", ">", "<") or arg.startswith("@") for arg in arguments)
    ):
        raise ExportError("Unsupported compiler command; shell operators and response files are not accepted")
    output = []
    skip = False
    compile_flags = 0
    for arg in arguments:
        if skip:
            skip = False
            continue
        if arg in ("-o", "-MF", "-MT", "-MQ", "-MJ"):
            skip = True
        elif arg == "-c":
            compile_flags += 1
        elif arg in ("-MD", "-MMD", "-MP"):
            continue
        elif arg.startswith(("-o", "-MF", "-MT", "-MQ", "-MJ")):
            continue
        elif not arg.startswith("-") and (directory / arg).resolve() == source:
            continue
        elif arg.startswith(("-Wp,", "--output", "-save-temps")) or arg in ("-M", "-MM", "-E", "-S"):
            raise ExportError("Unsupported side-effect or non-compilation flags in factory command")
        else:
            output.append(arg)
    if skip or compile_flags != 1:
        raise ExportError("Expected one ordinary -c compiler invocation")
    output += ["-I", str(runtime), "-fsyntax-only", str(probe)]
    evidence = {str(database): _hash_file(database)[0]}
    if source != desired:
        evidence[str(source)] = _hash_file(source)[0]
    return output, directory, evidence
