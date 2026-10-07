#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""clang-tidy misc-include-cleaner on tt_metal/api headers, one header at a time.

    run_api_header_include_cleaner.py <build-dir> --all
    run_api_header_include_cleaner.py <build-dir> <header> [<header> ...]

Options:
    --fail-on-findings   exit 1 if any header has a finding or fails to parse
                         (the switch that turns this report into a gate)
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
MISSING = re.compile(r'^no header providing "(?P<symbol>.+)" is directly included$')
COVERED = "(provided by an include already listed)"
UNUSED = re.compile(r"^included header (?P<header>.+) is not used directly$")

# clang-tidy spells a header it asks to add as a quoted path relative to the
# include directory it was found through ("tt_stl/span.hpp", "core_coord.hpp");
# the repo writes <tt_stl/span.hpp> and <tt-metalium/core_coord.hpp>, and
# tt_metal/api has no quoted includes at all. (root relative to the repo,
# include prefix), in lookup order. Same table as summarize_host_iwyu.py.
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
        elif key == "Offset":
            diagnostics[-1].setdefault("edits", []).append([int(value), 0, ""])
        elif key == "Length":
            diagnostics[-1]["edits"][-1][1] = int(value)
        elif key == "ReplacementText":
            diagnostics[-1]["replacements"].append(unquote(value))
            diagnostics[-1]["edits"][-1][2] = unquote(value)
    return diagnostics


def select(build: str, root: str, headers: list[str]) -> list[dict]:
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

    if headers == ["--all"]:
        names = sorted(stubs)
    else:
        names = []
        for header in headers:
            path = os.path.realpath(os.path.join(root, header))
            name = path[len(api_dir) :] if path.startswith(api_dir) else None
            if name in stubs:
                names.append(name)
            else:
                print(f"::warning::{header} has no header verification TU; skipped", file=sys.stderr)

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
    return derived


def run_one(entry: dict, args: argparse.Namespace, out: str, root: str) -> Result:
    api_dir = os.path.join(root, "tt_metal", "api") + os.sep
    result = Result(header=entry["file"][len(api_dir) :])
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
        ]
        command.append(entry["file"])
        start = time.monotonic()
        proc = subprocess.run(command, capture_output=True, text=True, errors="replace")
        result.seconds = time.monotonic() - start
        result.output = proc.stdout + proc.stderr
        exported = fixes.read()

    for diagnostic in parse_export_fixes(exported):
        message = diagnostic.get("message", "")
        if diagnostic["name"] != CHECK:
            result.errors.append(f"{diagnostic['name']}: {message}")
            continue
        result.edits += [tuple(edit) for edit in diagnostic.get("edits", [])]
        if match := MISSING.match(message):
            # clang-tidy attaches the insertion to the first symbol a header
            # provides; later symbols from the same header carry no fix-it.
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
    for line in result.output.splitlines():
        if re.search(r": (fatal )?error: ", line) and line not in result.errors:
            result.errors.append(line)
    if proc.returncode != 0 and not result.errors and not (result.missing or result.unused):
        result.errors.append(f"clang-tidy exited with {proc.returncode}")

    return result


def apply_edits(path: str, edits: list, root: str) -> None:
    """Apply one header's fix-its (byte offsets into the unmodified file), last first."""
    with open(path, "rb") as source:
        content = source.read()
    # At one offset: the removal first, then insertions, which end up in sorted order.
    for offset, length, text in sorted(set(edits), key=lambda e: (e[0], e[1], e[2]), reverse=True):
        text = repo_spelling(text, root).encode()
        content = content[:offset] + text + content[offset + length :]
    with open(path, "wb") as source:
        source.write(content)


def render(results: list[Result], status: int, artifact: str, gating: bool) -> str:
    missing = sum(len(r.missing.keys() - {COVERED}) for r in results)
    unused = sum(len(r.unused) for r in results)
    flagged = [r for r in results if r.missing or r.unused]
    failed = [r for r in results if r.failed]
    mode = "gate" if gating else "report-only"
    lines = [
        f"### misc-include-cleaner (tt_metal API headers, {mode})",
        "",
        "| Headers | Count |",
        "| --- | --- |",
        f"| analyzed | {len(results)} |",
        f"| with findings | {len(flagged)} |",
        f"| with missing includes | {sum(1 for r in results if r.missing)} ({missing} includes to add) |",
        f"| with unused includes | {sum(1 for r in results if r.unused)} ({unused} includes to remove) |",
        f"| failed to parse | {len(failed)} |",
        "",
        f"Exit status: {status}. Wall time per header: "
        f"max {max((r.seconds for r in results), default=0):.1f} s, "
        f"total {sum(r.seconds for r in results):.1f} s (CPU-serial).",
    ]
    if flagged or failed:
        lines += ["", "| Header | Add | Remove |", "| --- | --- | --- |"]
        for r in flagged + [r for r in failed if r not in flagged]:
            add = "<br>".join(f"`{inc}`" for inc in sorted(r.missing.keys() - {COVERED}))
            rm = "<br>".join(f"`{inc}` (line {n})" for n, inc in r.unused)
            if r.failed:
                rm += (" " if rm else "") + "**parse failed**"
            lines.append(f"| `{r.header}` | {add} | {rm} |")
    lines += [
        "",
        "Apply locally: `.github/scripts/utils/run_api_header_include_cleaner.py .build/iwyu --fix <header>...`.",
        f"Full output: `include-cleaner.txt` and `findings.json` in the `{artifact}` artifact.",
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
    parser.add_argument("--artifact", default="include-cleaner-api-headers-report")
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
    entries = select(build, root, ["--all"] if args.all else args.headers)
    with open(os.path.join(out, "compile_commands.json"), "w") as db:
        json.dump(entries, db, indent=1)
    print(f"{len(entries)} header(s) selected for {CHECK}", file=sys.stderr)
    if not entries:
        print("::warning::no header selected; nothing analyzed", file=sys.stderr)
        return 0

    version = subprocess.run([args.clang_tidy, "--version"], capture_output=True, text=True).stdout
    with open(os.path.join(out, "clang-tidy-version.txt"), "w") as f:
        f.write(version)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(lambda e: run_one(e, args, out, root), entries))
    if args.fix:
        for entry, result in zip(entries, results):
            if result.edits:
                apply_edits(entry["file"], result.edits, root)

    flagged = any(r.missing or r.unused for r in results)
    failed = any(r.failed for r in results)
    status = 1 if args.fail_on_findings and (flagged or failed) else 0
    with open(os.path.join(out, "include-cleaner.txt"), "w") as f:
        for r in results:
            f.write(f"=== {r.header} ({r.seconds:.2f} s)\n{r.output}\n")
    with open(os.path.join(out, "findings.json"), "w") as f:
        json.dump([dataclasses.asdict(r) | {"output": None} for r in results], f, indent=1)
    with open(os.path.join(out, "exit-code.txt"), "w") as f:
        f.write(f"{status}\n")
    print(render(results, status, args.artifact, args.fail_on_findings))
    if failed:
        print(f"::warning::{CHECK} could not parse some headers; see include-cleaner.txt", file=sys.stderr)
    return status


if __name__ == "__main__":
    sys.exit(main())
