# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Command-line interface for the LLK API analyzer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .analyzer import LlkAnalyzer
from .extractor import ExtractorConfig
from .model import ApiLayer
from .report import render_csv, render_table, render_text, to_json
from .runner import ModelRunner

_DEFAULT_CACHE = Path.home() / ".cache" / "tt-metal-cache"

_LAYER_CHOICES = {
    "llk_core": ApiLayer.LLK_CORE,
    "llk_api": ApiLayer.LLK_API,
    "compute_api": ApiLayer.COMPUTE_API,
    "other": ApiLayer.OTHER,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llk_api_analyzer",
        description=(
            "Analyze compiled tt-metal compute kernels and report which LLK APIs "
            "were used, with their template arguments (data formats, sync scheme, "
            "dest/L1 accumulation, math fidelity, ...) and runtime arguments."
        ),
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help=(
            "Path to analyze: the tt-metal cache root, a single <build_key>, a "
            "kernels/ directory, or one kernel/compile-hash directory "
            f"(default: {_DEFAULT_CACHE}). Ignored when --run is given."
        ),
    )
    run_group = parser.add_argument_group("running a model")
    run_group.add_argument(
        "-r",
        "--run",
        metavar="COMMAND",
        help=(
            "Shell command that runs the model/test (e.g. "
            '"pytest tests/foo.py -k bar"). The command is run with an isolated '
            "TT_METAL_CACHE and TT_METAL_RISCV_DEBUG_INFO=1, then the resulting "
            "kernels are analyzed."
        ),
    )
    run_group.add_argument(
        "--cache-dir",
        help=(
            "Directory to use as TT_METAL_CACHE for --run (default: a fresh temp "
            "directory). Use this to reuse/inspect a persistent cache."
        ),
    )
    run_group.add_argument(
        "--run-cwd",
        help="Working directory to run the --run command from (default: current directory).",
    )
    run_group.add_argument(
        "--keep-cache",
        action="store_true",
        help="Do not delete the temporary cache created for --run after analysis.",
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=("text", "json", "table", "csv"),
        default="text",
        help=("Output format (default: text). 'table'/'csv' collapse the run into a " "single row-per-LLK-call table."),
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Write the report to this file instead of stdout.",
    )
    parser.add_argument(
        "-l",
        "--layers",
        default="llk_api",
        help=(
            "Comma-separated API layers to collect, from " f"{{{','.join(sorted(_LAYER_CHOICES))}}} (default: llk_api)."
        ),
    )
    return parser


def _parse_layers(spec: str) -> frozenset[ApiLayer]:
    layers = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if token not in _LAYER_CHOICES:
            raise ValueError(f"unknown layer '{token}' (choose from {sorted(_LAYER_CHOICES)})")
        layers.add(_LAYER_CHOICES[token])
    return frozenset(layers)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.path is not None and args.run:
        print("error: provide either a path or --run, not both", file=sys.stderr)
        return 2

    try:
        include_layers = _parse_layers(args.layers)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    analyzer = LlkAnalyzer(extractor_config=ExtractorConfig(include_layers=include_layers))

    run_result = None
    if args.run:
        runner = ModelRunner(cache_dir=args.cache_dir, working_directory=args.run_cwd)
        try:
            run_result = runner.run(args.run)
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        analyze_target = str(run_result.cache_dir)
    else:
        analyze_target = args.path or str(_DEFAULT_CACHE)

    try:
        try:
            analysis = analyzer.analyze_run(analyze_target)
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

        if not analysis.kernels:
            print(f"warning: no compute kernels found under {analyze_target}", file=sys.stderr)
            if args.run:
                print(
                    "hint: the run may have failed before compiling kernels, or used a "
                    "different cache; check the command output above.",
                    file=sys.stderr,
                )

        renderers = {
            "json": to_json,
            "table": render_table,
            "csv": render_csv,
            "text": render_text,
        }
        output = renderers[args.format](analysis)

        if args.output:
            Path(args.output).write_text(output)
        else:
            print(output)
    finally:
        if run_result is not None and not args.keep_cache:
            ModelRunner.cleanup(run_result)
        elif run_result is not None and run_result.is_temporary:
            print(f"[llk-analyzer] kept cache at {run_result.cache_dir}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
