#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Command-line HAL explorer: compile HAL code and print what it issues.

    python3 tools/python/hal_explorer/cli.py kernel.cpp
    python3 tools/python/hal_explorer/cli.py --mode body --var "bool partial_face" body.cpp
    cat kernel.cpp | python3 tools/python/hal_explorer/cli.py - --json

Exit status: 0 complete result and every comparison identical; 1 compile error;
2 a comparison differs or a reference_<name> has no partner; 3 invalid arguments
or unreadable input; 4 the result is incomplete (see the warnings).
"""

import argparse
import json
import sys
from pathlib import Path

# Run as a script, the package's parent directory is not yet importable.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hal_explorer import pipeline  # noqa: E402  pylint: disable=wrong-import-position
from hal_explorer.model import (  # noqa: E402  pylint: disable=wrong-import-position
    to_json,
)
from hal_explorer.ui import render  # noqa: E402  pylint: disable=wrong-import-position

EXIT_OK = 0
EXIT_COMPILE_ERROR = 1
EXIT_MISMATCH = 2
EXIT_USAGE = 3
EXIT_INCOMPLETE = 4


class Parser(argparse.ArgumentParser):
    """argparse exits 2 on bad arguments, which this tool uses for a mismatch."""

    def error(self, message):
        self.print_usage(sys.stderr)
        self.exit(EXIT_USAGE, f"{self.prog}: error: {message}\n")


def parse_variable(text):
    """'const std::uint32_t n' or 'bool a' -> {"type": ..., "name": ...}."""
    words = text.split()
    if words and words[0] == "const":
        words = words[1:]
    if len(words) < 2:
        raise argparse.ArgumentTypeError(f"expected 'TYPE NAME', got {text!r}")
    return {"type": " ".join(words[:-1]), "name": words[-1]}


def read_source(path):
    try:
        if path == "-":
            return sys.stdin.buffer.read().decode("utf-8")
        with open(path, encoding="utf-8") as f:
            return f.read()
    except (OSError, UnicodeDecodeError) as error:
        raise SystemExit(f"cannot read {path}: {error}") from error


def exit_status(result):
    if not result.ok:
        return EXIT_COMPILE_ERROR
    if any(c.verdict in ("differs", "unpaired") for c in result.comparisons):
        return EXIT_MISMATCH
    if result.incomplete:
        return EXIT_INCOMPLETE
    return EXIT_OK


def select(result, name):
    """Keep only the named function and the comparisons that involve it."""
    if not name:
        return result
    keep = [f for f in result.functions if name in (f.name, f.signature)]
    if not keep:
        names = ", ".join(f.signature for f in result.functions)
        raise SystemExit(f"no function {name!r}; compiled: {names}")
    signatures = {f.signature for f in keep}
    result.functions = keep
    result.comparisons = [
        c for c in result.comparisons if signatures & {c.name, c.reference}
    ]
    return result


def main():
    parser = Parser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input", help="source file, or - for stdin")
    parser.add_argument(
        "--mode",
        choices=pipeline.MODES,
        default="code",
        help="code: top-level functions (default); body: the body of a replay::record lambda",
    )
    parser.add_argument(
        "--var",
        action="append",
        default=[],
        type=parse_variable,
        help="body mode: a runtime variable the body reads, e.g. --var 'bool partial_face'",
    )
    parser.add_argument(
        "--function", help="print only this function (name or signature)"
    )
    parser.add_argument(
        "--counts", action="store_true", help="one tab-separated row per path"
    )
    parser.add_argument(
        "--gimple", action="store_true", help="also print the compiler output"
    )
    parser.add_argument(
        "--json", action="store_true", help="print the full result as JSON"
    )
    args = parser.parse_args()

    try:
        pipeline.check_variables(args.var)
        if args.var and args.mode != "body":
            raise pipeline.InputError(
                "--var applies to --mode body; in code mode use function parameters"
            )
    except pipeline.InputError as error:
        parser.error(str(error))
    try:
        source = read_source(args.input)
    except SystemExit as error:
        print(error, file=sys.stderr)
        return EXIT_USAGE

    result = pipeline.run(args.mode, source, pipeline.load_tables(), args.var)
    try:
        result = select(result, args.function) if result.ok else result
    except SystemExit as error:
        print(error, file=sys.stderr)
        return EXIT_USAGE

    if args.json:
        print(json.dumps(to_json(result), indent=1, ensure_ascii=False))
        return exit_status(result)
    if result.diagnostics:
        print(result.diagnostics, file=sys.stderr)
    if not result.ok:
        return EXIT_COMPILE_ERROR
    for function in result.functions:
        if args.counts:
            print("\n".join(render.counts_rows(function)))
        else:
            print(render.function_text(function, args.gimple))
    stream = sys.stderr if args.counts else sys.stdout
    for comparison in result.comparisons:
        print(render.comparison_text(comparison), file=stream)
    return exit_status(result)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BrokenPipeError:
        sys.exit(0)
