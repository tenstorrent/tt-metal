# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Descriptor-only flow contract: real host compilation plus command safety."""

import json
import shlex
import shutil
import subprocess

import pytest

from tools.generic_op_to_factory import factory_contract as contract
from tools.generic_op_to_factory.export_run import ExportError


@pytest.fixture
def inputs(tmp_path):
    runtime = tmp_path / "runtime"
    operation = runtime / "ttnn/sample.hpp"
    operation.parent.mkdir(parents=True)
    operation.write_text("// fixture\n")
    source = runtime / "ttnn/sample.cpp"
    source.write_text("// fixture\n")
    specification = {
        "operation_header": "ttnn/sample.hpp",
        "operation_type": "sample::Op",
        "factory_source": "ttnn/sample.cpp",
    }
    return runtime, specification


def database(runtime, source, arguments):
    build = runtime / "build_Release"
    build.mkdir(exist_ok=True)
    path = build / "compile_commands.json"
    path.write_text(json.dumps([{"directory": str(runtime), "file": str(source), "arguments": arguments}]))
    return path


@pytest.mark.parametrize("unity", [False, True])
def test_uses_exact_factory_flags_without_link_or_object_output(inputs, tmp_path, unity):
    runtime, specification = inputs
    source = runtime / specification["factory_source"]
    if unity:
        source = runtime / "unity_0_cxx.cxx"
        source.write_text('#include "' + str(runtime / specification["factory_source"]) + '"\n')
    arguments = [
        "clang++",
        "-std=c++20",
        "-DIMPORTANT=1",
        "-MD",
        "-MF",
        "object.d",
        "-o",
        "object.o",
        "-c",
        str(source),
    ]
    db = database(runtime, source, arguments)
    probe = tmp_path / "probe.cpp"
    argv, cwd, evidence = contract.compile_invocation(specification, runtime, probe)
    assert argv[:3] == arguments[:3]
    assert argv[-2:] == ["-fsyntax-only", str(probe)]
    assert not set(argv) & {"-c", "-o", "-MD", "-MF", "object.o", "object.d", str(source)}
    assert str(db) in evidence and cwd == runtime
    assert (str(source) in evidence) == unity


def test_command_string_is_tokenized_not_executed(inputs, tmp_path):
    runtime, specification = inputs
    source = runtime / specification["factory_source"]
    db = database(runtime, source, [])
    db.write_text(
        json.dumps(
            [
                {
                    "directory": str(runtime),
                    "file": str(source),
                    "command": shlex.join(["clang++", "-DVALUE=two words", "-c", str(source), "-oout.o"]),
                }
            ]
        )
    )
    argv, _, _ = contract.compile_invocation(specification, runtime, tmp_path / "probe.cpp")
    assert "-DVALUE=two words" in argv and "-oout.o" not in argv


def test_cmake_env_ccache_launcher_is_preserved(inputs, tmp_path):
    runtime, specification = inputs
    source = runtime / specification["factory_source"]
    prefix = ["/usr/bin/cmake", "-E", "env", "CCACHE_COMPRESS=true", "CCACHE_BASEDIR=/a path", "ccache", "clang++"]
    database(runtime, source, [*prefix, "-std=c++20", "-c", str(source), "-o", "out.o"])
    argv, _, _ = contract.compile_invocation(specification, runtime, tmp_path / "probe.cpp")
    assert argv[: len(prefix)] == prefix
    assert argv[-2:] == ["-fsyntax-only", str(tmp_path / "probe.cpp")]


@pytest.mark.parametrize("extra", [["&&", "false"], ["@flags.rsp"], ["-save-temps"], ["-E"]])
def test_unsupported_commands_fail_closed(inputs, tmp_path, extra):
    runtime, specification = inputs
    source = runtime / specification["factory_source"]
    database(runtime, source, ["clang++", "-c", str(source), *extra])
    with pytest.raises(ExportError, match="Unsupported"):  # allow-pytest.raises: host-only command validation
        contract.compile_invocation(specification, runtime, tmp_path / "probe.cpp")


def test_missing_or_ambiguous_compile_entry_refused(inputs, tmp_path):
    runtime, specification = inputs
    with pytest.raises(ExportError, match="compile_commands"):  # allow-pytest.raises: host-only command validation
        contract.compile_invocation(specification, runtime, tmp_path / "probe.cpp")
    source = runtime / specification["factory_source"]
    db = database(runtime, source, ["clang++", "-c", str(source)])
    entries = json.loads(db.read_text())
    db.write_text(json.dumps(entries * 2))
    with pytest.raises(ExportError, match="exactly one"):  # allow-pytest.raises: host-only command validation
        contract.compile_invocation(specification, runtime, tmp_path / "probe.cpp")


@pytest.mark.parametrize("source_count", [0, 2])
def test_exactly_one_factory_source_argument_is_required(inputs, tmp_path, source_count):
    runtime, specification = inputs
    source = runtime / specification["factory_source"]
    database(runtime, source, ["clang++", "-c", *([str(source)] * source_count)])
    with pytest.raises(ExportError, match="one ordinary -c"):  # allow-pytest.raises: compiler command validation
        contract.compile_invocation(specification, runtime, tmp_path / "probe.cpp")


@pytest.mark.parametrize(
    "key,value",
    [
        ("operation_type", "sample::Op;evil()"),
        ("operation_header", "../outside.hpp"),
        ("factory_source", "ttnn/missing.cpp"),
    ],
)
def test_config_cannot_inject_code_or_escape(inputs, key, value):
    runtime, specification = inputs
    specification[key] = value
    with pytest.raises(ExportError):  # allow-pytest.raises: host-only config validation
        contract.validate(specification, runtime)


@pytest.mark.parametrize(
    "kind",
    [
        "descriptor",
        "per_coordinate",
        "legacy",
        "spec",
        "workload",
        "overloaded_workload",
        "missing_hook",
        "wrong_return",
        "mixed",
        "operation_hook",
        "both_hooks",
        "dynamic",
    ],
)
def test_probe_rejects_non_descriptor_or_missing_refresh_with_real_compiler(inputs, tmp_path, kind):
    runtime, specification = inputs
    compiler = shutil.which("clang++-20") or shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("Host C++20 compiler unavailable")
    framework = runtime / "ttnn/operation_concepts.hpp"
    framework.write_text(
        """#pragma once
#include <optional>
#include <variant>
namespace tt::tt_metal { struct Program {}; struct ProgramDescriptor {}; }
namespace ttnn { struct MeshCoordinate {}; struct MeshCoordinateRangeSet {}; }
namespace ttnn::device_operation {
template<class F> concept ProgramDescriptorFactoryConcept =
    (requires { &F::create_descriptor; } || requires { &F::create_workload_descriptor; }) &&
    !requires { typename F::cached_program_t; };
}
"""
    )
    hook = "static void override_runtime_arguments(tt::tt_metal::Program&, const int&, const int&, int&, const std::optional<ttnn::MeshCoordinate>&);"
    create = "static tt::tt_metal::ProgramDescriptor create_descriptor(const int&, const int&, int&);"
    factory = create + hook
    if kind == "per_coordinate":
        factory = (
            "static tt::tt_metal::ProgramDescriptor create_descriptor(const int&, const int&, int&, const std::optional<ttnn::MeshCoordinate>&);"
            + hook
        )
    elif kind == "legacy":
        factory = "using cached_program_t = int; static int create(const int&, const int&, int&);" + hook
    elif kind == "spec":
        factory = "static int create_program_artifacts(const int&, const int&, int&);" + hook
    elif kind == "workload":
        factory = "static int create_workload_descriptor(const int&, const int&, int&);" + hook
    elif kind == "overloaded_workload":
        factory += "static int create_workload_descriptor(const int&, const int&, int&, const ttnn::MeshCoordinateRangeSet&); static int create_workload_descriptor();"
    elif kind in ("missing_hook", "operation_hook"):
        factory = create
    elif kind == "wrong_return":
        factory = create.replace("tt::tt_metal::ProgramDescriptor", "int") + hook
    operation_extra = hook if kind in ("operation_hook", "both_hooks") else ""
    if kind == "dynamic":
        operation_extra = "static int get_dynamic_runtime_args(const int&, const int&, int&, const std::optional<ttnn::MeshCoordinate>&);"
    alternatives = "Factory, BadFactory" if kind == "mixed" else "Factory"
    (runtime / specification["operation_header"]).write_text(
        '#include "ttnn/operation_concepts.hpp"\nnamespace sample {\n'
        + "struct Factory {"
        + factory
        + "}; struct BadFactory {};\n"
        + "struct Op { using operation_attributes_t = int; using tensor_args_t = int; using tensor_return_value_t = int; "
        + "using program_factory_t = std::variant<"
        + alternatives
        + ">;"
        + operation_extra
        + "}; }\n"
    )
    contract.validate(specification, runtime)
    probe = tmp_path / "probe.cpp"
    probe.write_text(contract.render(specification))
    result = subprocess.run(
        [compiler, "-std=c++20", "-I", str(runtime), "-fsyntax-only", str(probe)], capture_output=True, text=True
    )
    assert (result.returncode == 0) == (kind in ("descriptor", "per_coordinate")), result.stderr
    if result.returncode:
        assert "static assertion failed" in result.stderr
