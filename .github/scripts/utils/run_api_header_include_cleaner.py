#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""clang-tidy misc-include-cleaner on tt_metal/api headers, one header at a time.

    run_api_header_include_cleaner.py <build-dir> --all
    run_api_header_include_cleaner.py <build-dir> <header> [<header> ...]

Options:
    --fail-on-findings   exit 1 if any header has a finding, or any header could not be
                         analyzed (compile error, clang-tidy crash, header with no
                         verification TU). This is what CI runs.
    --fix                apply clang-tidy's fix-its to the headers (local use). All
                         headers are analyzed against the unmodified tree first, then
                         the edits are applied, with inserted includes respelled the
                         way the repo writes them (<tt-metalium/...>, <tt_stl/...>)
    -j N                 parallel clang-tidy processes (default: nproc)
    --clang-tidy BIN     clang-tidy binary (default: clang-tidy-20, else clang-tidy)

<build-dir> must have been configured with the `iwyu` preset (which turns on
CMAKE_VERIFY_INTERFACE_HEADER_SETS and CMAKE_EXPORT_COMPILE_COMMANDS) and have had
the `metalium_GeneratedHeaders` target built.

misc-include-cleaner only reports on the main file of a translation unit, so it
cannot be pointed at CMake's header verification stubs (#include <hdr>) the way
IWYU's `pragma: associated` is: it would only look at the stub's single include.
Instead each header becomes its own main file: the stub's compilation database
entry is copied with the stub path replaced by the header and `-x c++` by
`-x c++-header`, and `-o <obj>` and `-c` dropped. All other flags are the real
build's; none are assembled here. Per header this runs

    clang-tidy -p <build-dir>/report-api-include-cleaner \
        --config-file=.github/api-include-cleaner.clang-tidy --quiet \
        --export-fixes=<tmp>.yaml <repo>/tt_metal/api/<header>

`-x c++-header` (not `-x c++`) is what keeps `#pragma once` quiet: as a C++ main
file clang reports "#pragma once in main file" as an error
(clang-diagnostic-pragma-once-outside-header); the findings are otherwise the
same. The stubs' `// IWYU pragma: associated` is irrelevant here.

The check configuration is .github/api-include-cleaner.clang-tidy, passed with
--config-file so that misc-include-cleaner is enabled for these headers only.

Writes to <build-dir>/report-api-include-cleaner/: the derived
compile_commands.json, include-cleaner.txt (clang-tidy output), findings.json,
clang-tidy-version.txt and exit-code.txt. Prints a Markdown summary on stdout.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time

CHECK = "misc-include-cleaner"
HEADER_SUFFIXES = (".h", ".hpp", ".tpp", ".inl")
MISSING = re.compile(r'^no header providing "(?P<symbol>.+)" is directly included$')
COVERED = "(no fix-it of its own)"
UNUSED = re.compile(r"^included header (?P<header>.+) is not used directly$")

# clang-tidy spells a header it asks to add as a quoted path relative to the
# include directory it was found through ("tt_stl/span.hpp", "core_coord.hpp");
# the repo writes <tt_stl/span.hpp> and <tt-metalium/core_coord.hpp>, and
# tt_metal/api has no quoted includes at all. (root relative to the repo,
# include prefix), in lookup order: the include directories of the verification TUs.
PROJECT_INCLUDE_ROOTS = (
    ("tt_metal/api/tt-metalium", "tt-metalium/"),
    ("tt_metal/api", ""),
    ("tt_stl", ""),
    ("tt_metal/hostdevcommon/api", ""),
)
QUOTED_INCLUDE = re.compile(r'^#include "([^"]+)"', re.M)


def repo_spelling(include: str, repo_root: str) -> str:
    """'#include "tt_stl/span.hpp"' -> '#include <tt_stl/span.hpp>'."""

    def angled(match: re.Match) -> str:
        path = match.group(1)
        for root, prefix in PROJECT_INCLUDE_ROOTS:
            if os.path.exists(os.path.join(repo_root, root, path)):
                return f"#include <{prefix}{path}>"
        return match.group(0)

    return QUOTED_INCLUDE.sub(angled, include)


@dataclasses.dataclass
class Result:
    header: str  # relative to tt_metal/api
    seconds: float = 0.0
    output: str = ""
    missing: dict = dataclasses.field(default_factory=dict)  # include line -> [symbols]
    unused: list = dataclasses.field(default_factory=list)  # (line number, include line)
    errors: list = dataclasses.field(default_factory=list)  # compiler errors
    edits: list = dataclasses.field(default_factory=list)  # (offset, length, text) fix-its

    @property
    def failed(self) -> bool:
        return bool(self.errors)

    @property
    def failure_kind(self) -> str:
        if not self.errors:
            return ""
        first = self.errors[0]
        if first.startswith("clang-tidy was killed"):
            return "clang-tidy crashed"
        if first.startswith("clang-tidy exited"):
            return "clang-tidy failed"
        if first.startswith(f"{CHECK}: unrecognised"):
            return "unrecognised diagnostic"
        if first.startswith("no verification TU"):
            return "not analyzed (no verification TU)"
        return "compile error"


def unquote(value: str) -> str:
    value = value.strip()
    if value.startswith("'"):
        return value[1:-1].replace("''", "'")
    if value.startswith('"'):
        return json.loads(value)
    return value


def parse_export_fixes(text: str) -> list[dict]:
    """Just enough of clang-tidy's --export-fixes YAML: name, message, offset, replacements."""
    diagnostics = []
    for line in text.splitlines():
        stripped = line.strip()
        key, _, value = stripped.partition(":")
        if stripped.startswith("- DiagnosticName:"):
            diagnostics.append({"name": unquote(stripped.split(":", 1)[1]), "replacements": []})
        elif not diagnostics:
            continue
        elif key == "Message":
            diagnostics[-1]["message"] = unquote(value)
        elif key == "FileOffset":
            diagnostics[-1]["offset"] = int(value)
        elif key == "Level":
            diagnostics[-1]["level"] = unquote(value)
        elif key == "Offset":
            diagnostics[-1].setdefault("edits", []).append([int(value), 0, ""])
        elif key == "Length":
            diagnostics[-1]["edits"][-1][1] = int(value)
        elif key == "ReplacementText":
            diagnostics[-1]["replacements"].append(unquote(value))
            diagnostics[-1]["edits"][-1][2] = unquote(value)
    return diagnostics


def select(build: str, root: str, headers: list[str]) -> tuple[list[dict], list[str], list[str]]:
    """Header compile commands derived from the verification stubs.

    Returns (entries, requested headers with no stub, headers under tt_metal/api
    that are not in the public header set and so have no stub; --all only).
    """
    stub_dir = os.path.join(build, "tt_metal", "tt_metal_verify_interface_header_sets") + os.sep
    api_dir = os.path.join(root, "tt_metal", "api") + os.sep
    stubs = {}
    with open(os.path.join(build, "compile_commands.json")) as db:
        for entry in json.load(db):
            path = os.path.realpath(os.path.join(entry["directory"], entry["file"]))
            if path.startswith(stub_dir):
                stubs[path[len(stub_dir) : -len(".cxx")]] = (path, entry)
    if not stubs:
        sys.exit(
            f"::error::no header verification TUs under {stub_dir}; was the build configured with the iwyu preset?"
        )

    no_stub, outside_set = [], []
    if headers == ["--all"]:
        names = sorted(stubs)
        listed = subprocess.run(
            ["git", "-C", root, "ls-files", "--", "tt_metal/api"], capture_output=True, text=True
        ).stdout.split()
        outside_set = sorted(
            f for f in listed if f.endswith(HEADER_SUFFIXES) and f[len("tt_metal/api/") :] not in stubs
        )
    else:
        names = []
        for header in headers:
            path = os.path.realpath(os.path.join(root, header))
            name = path[len(api_dir) :] if path.startswith(api_dir) else None
            if name in stubs:
                names.append(name)
            else:
                no_stub.append(header)

    derived = []
    for name in names:
        stub, entry = stubs[name]
        args = entry["arguments"] if "arguments" in entry else shlex.split(entry["command"])
        args = list(args)
        if "-o" in args:
            i = args.index("-o")
            del args[i : i + 2]
        args = [a for a in args if a != "-c"]
        if "-x" in args:
            args[args.index("-x") + 1] = "c++-header"
        else:
            args.insert(1, "-x")
            args.insert(2, "c++-header")
        header = os.path.join(api_dir, name)
        args = [header if os.path.realpath(os.path.join(entry["directory"], a)) == stub else a for a in args]
        derived.append({"directory": entry["directory"], "arguments": args, "file": header})
    return derived, no_stub, outside_set


def run_one(entry: dict, args: argparse.Namespace, out: str, root: str) -> Result:
    api_dir = os.path.join(root, "tt_metal", "api") + os.sep
    with open(entry["file"], "rb") as source:
        content = source.read()
    with tempfile.NamedTemporaryFile("r", suffix=".yaml") as fixes:
        command = [
            args.clang_tidy,
            "-p",
            out,
            f"--config-file={args.config}",
            f"--export-fixes={fixes.name}",
            "--quiet",
            entry["file"],
        ]
        start = time.monotonic()
        proc = subprocess.run(command, capture_output=True, text=True, errors="replace")
        seconds = time.monotonic() - start
        exported = fixes.read()
    result = analyze(entry["file"][len(api_dir) :], proc.returncode, proc.stdout + proc.stderr, exported, content, root)
    result.seconds = seconds
    return result


def analyze(header: str, returncode: int, output: str, exported: str, content: bytes, root: str) -> Result:
    """Turn one clang-tidy run (exit status, console output, --export-fixes YAML) into a Result."""
    result = Result(header=header, output=output)
    promoted = False  # a misc-include-cleaner finding reported at error level
    for diagnostic in parse_export_fixes(exported):
        message = diagnostic.get("message", "")
        if diagnostic["name"] != CHECK:
            result.errors.append(f"{diagnostic['name']}: {message}")
            continue
        promoted |= diagnostic.get("level") == "Error"
        result.edits += [tuple(edit) for edit in diagnostic.get("edits", [])]
        if match := MISSING.match(message):
            # clang-tidy attaches the insertion to the first symbol a header
            # provides; later symbols from the same header carry no fix-it.
            # A few diagnostics have no fix-it at all and need a manual change.
            for text in diagnostic["replacements"] or [COVERED]:
                include = repo_spelling(text.strip(), root)
                result.missing.setdefault(include, [])
                if match["symbol"] not in result.missing[include]:
                    result.missing[include].append(match["symbol"])
        elif UNUSED.match(message):
            offset = diagnostic.get("offset", 0)
            line_no = content.count(b"\n", 0, offset) + 1
            end = content.find(b"\n", offset)
            result.unused.append((line_no, content[offset : end if end >= 0 else None].decode(errors="replace")))
        else:
            result.errors.append(f"{CHECK}: unrecognised message: {message}")
    # Compiler errors are not always exported as fixes; take them from the output too.
    for line in output.splitlines():
        if re.search(r": (fatal )?error: ", line) and line not in result.errors:
            result.errors.append(line)

    # Exit status, as measured with clang-tidy 20.1.8 and this config
    # (WarningsAsErrors ''): 0 when the header parsed, with or without findings
    # (they are warnings); 1 when the header has a compile error, even if
    # findings were reported too ("Error while processing ..."); negative when
    # killed by a signal. So a nonzero exit is a failure unless it is fully
    # explained: by a compile error recorded above (already a failure), or by
    # findings promoted to errors through WarningsAsErrors. Findings never
    # excuse a nonzero exit on their own.
    if returncode < 0:
        result.errors.append(f"clang-tidy was killed by signal {-returncode}")
    elif returncode != 0 and not result.errors and not promoted:
        result.errors.append(f"clang-tidy exited with {returncode} without reporting an error")
    return result


def apply_edits(path: str, edits: list, root: str) -> None:
    """Apply one header's fix-its (byte offsets into the unmodified file), last first."""
    with open(path, "rb") as source:
        content = source.read()
    # Respell first so insertions sort by their final text. Applied from the
    # end: at one offset the removal goes first, then the insertions, each in
    # front of the previous one, so they end up in ascending order.
    respelled = {(offset, length, repo_spelling(text, root)) for offset, length, text in edits}
    for offset, length, text in sorted(respelled, reverse=True):
        content = content[:offset] + text.encode() + content[offset + length :]
    with open(path, "wb") as source:
        source.write(content)


def exit_status(results: list[Result], gating: bool) -> int:
    """With --fail-on-findings: 1 if any header has a finding or could not be analyzed."""
    return 1 if gating and any(r.missing or r.unused or r.failed for r in results) else 0


REPRODUCE = """\
cmake --preset iwyu
cmake --build .build/iwyu --target metalium_GeneratedHeaders
.github/scripts/utils/run_api_header_include_cleaner.py .build/iwyu --all --fail-on-findings"""


def render(results: list[Result], status: int, artifact: str, gating: bool, outside_set: list[str] = ()) -> str:
    to_add = sum(len(r.missing.keys() - {COVERED}) for r in results)
    to_remove = sum(len(r.unused) for r in results)
    flagged = [r for r in results if r.missing or r.unused]
    failed = [r for r in results if r.failed]
    if not gating:
        verdict = "Findings are reported but do not fail this run (no `--fail-on-findings`)."
    elif status:
        verdict = (
            f"**FAILED**: {len(flagged)} header(s) with findings, {len(failed)} header(s) failed. "
            "Every tt_metal public header must have exactly the includes it uses."
        )
    else:
        verdict = "**PASSED**: no findings, every header analyzed."
    lines = [
        f"### misc-include-cleaner on tt_metal public headers: {'FAILED' if status else 'passed'}",
        "",
        verdict,
        "",
        "| Headers | Count |",
        "| --- | --- |",
        f"| analyzed | {sum(1 for r in results if not r.failure_kind.startswith('not analyzed'))} |",
        f"| with findings | {len(flagged)} |",
        f"| missing includes | {sum(1 for r in results if r.missing)} headers, {to_add} includes to add |",
        f"| unused includes | {sum(1 for r in results if r.unused)} headers, {to_remove} includes to remove |",
        f"| failed (compile error, crash, no verification TU) | {len(failed)} |",
        "",
        f"Exit status: {status}. Wall time per header: "
        f"max {max((r.seconds for r in results), default=0):.1f} s, "
        f"total {sum(r.seconds for r in results):.1f} s (CPU-serial).",
    ]
    if failed:
        lines += ["", "#### Failed", "", "| Header | Kind | First error |", "| --- | --- | --- |"]
        for r in failed:
            first = r.errors[0].replace("|", "\\|")[:200]
            lines.append(f"| `{r.header}` | {r.failure_kind} | `{first}` |")
    if flagged:
        lines += ["", "#### Findings", "", "| Header | Add | Remove | Without fix-it |", "| --- | --- | --- | --- |"]
        for r in flagged:
            add = "<br>".join(f"`{inc}`" for inc in sorted(r.missing.keys() - {COVERED}))
            rm = "<br>".join(f"`{inc}` (line {n})" for n, inc in r.unused)
            manual = ", ".join(f"`{sym}`" for sym in r.missing.get(COVERED, []))
            lines.append(f"| `{r.header}` | {add} | {rm} | {manual} |")
    if outside_set:
        lines += [
            "",
            f"Not checked: {len(outside_set)} header(s) under tt_metal/api are not in the tt_metal public header "
            "set (TT_METAL_PUBLIC_API), so CMake generates no verification TU for them: "
            + ", ".join(f"`{h}`" for h in outside_set)
            + ".",
        ]
    lines += [
        "",
        "Reproduce locally (from the repo root):",
        "",
        "```",
        REPRODUCE,
        "```",
        "",
        "Apply the fix-its, then review the diff (inserted includes may need regrouping; "
        "symbols listed under *Without fix-it* that are not covered by an added include need a manual include, "
        "or `// NOLINT(misc-include-cleaner)` / `// IWYU pragma: keep` with a reason):",
        "",
        "```",
        ".github/scripts/utils/run_api_header_include_cleaner.py .build/iwyu --fix "
        + (" ".join(f"tt_metal/api/{r.header}" for r in flagged) if 0 < len(flagged) <= 10 else "--all"),
        "```",
        "",
        f"Full clang-tidy output: `include-cleaner.txt` and `findings.json` in the `{artifact}` artifact.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("build_dir")
    parser.add_argument("headers", nargs="*", help="headers relative to the repo root (or absolute)")
    parser.add_argument("--all", action="store_true", help="every tt_metal public header")
    parser.add_argument("--fail-on-findings", action="store_true")
    parser.add_argument("--fix", action="store_true")
    parser.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--clang-tidy", default=shutil.which("clang-tidy-20") or "clang-tidy")
    parser.add_argument("--artifact", default="include-cleaner-api-headers-results")
    args = parser.parse_intermixed_args()

    root = subprocess.run(
        ["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    args.config = os.path.join(root, ".github", "api-include-cleaner.clang-tidy")
    build = os.path.realpath(args.build_dir)
    out = os.path.join(build, "report-api-include-cleaner")
    os.makedirs(out, exist_ok=True)

    if args.all == bool(args.headers):
        parser.error("give either --all or a list of headers")
    entries, no_stub, outside_set = select(build, root, ["--all"] if args.all else args.headers)
    with open(os.path.join(out, "compile_commands.json"), "w") as db:
        json.dump(entries, db, indent=1)
    print(f"{len(entries)} header(s) selected for {CHECK}", file=sys.stderr)

    version = subprocess.run([args.clang_tidy, "--version"], capture_output=True, text=True).stdout
    with open(os.path.join(out, "clang-tidy-version.txt"), "w") as f:
        f.write(version)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(lambda e: run_one(e, args, out, root), entries))
    # A requested header without a verification TU cannot be analyzed: that is a failure, not a skip.
    results += [
        Result(header=h, errors=[f"no verification TU: {h} is not in the tt_metal public header set"]) for h in no_stub
    ]
    if args.fix:
        for entry, result in zip(entries, results):
            if result.edits:
                apply_edits(entry["file"], result.edits, root)

    status = exit_status(results, args.fail_on_findings)
    failed = any(r.failed for r in results)
    with open(os.path.join(out, "include-cleaner.txt"), "w") as f:
        for r in results:
            f.write(f"=== {r.header} ({r.seconds:.2f} s)\n{r.output}\n")
    with open(os.path.join(out, "findings.json"), "w") as f:
        json.dump([dataclasses.asdict(r) | {"output": None} for r in results], f, indent=1)
    with open(os.path.join(out, "exit-code.txt"), "w") as f:
        f.write(f"{status}\n")
    print(render(results, status, args.artifact, args.fail_on_findings, outside_set))
    level = "error" if args.fail_on_findings else "warning"
    flagged = sum(1 for r in results if r.missing or r.unused)
    if flagged:
        print(f"::{level}::{CHECK}: {flagged} tt_metal public header(s) have include findings", file=sys.stderr)
    if failed:
        print(f"::{level}::{CHECK}: {sum(r.failed for r in results)} header(s) could not be analyzed", file=sys.stderr)
    return status


if __name__ == "__main__":
    sys.exit(main())
