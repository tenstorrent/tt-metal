# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Prepare a complete HF checkpoint or individual matrices for TT BFP loading."""
import argparse
import json
from pathlib import Path
import sys

import torch

from . import __version__, gptq_search, search_linear, search_packed, to_bf16_exact, validate_repacking
from .native import build_native


def _tensor(path):
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{path}: expected a single Torch tensor, not a checkpoint dictionary")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="Compile the optional fast CPU backend")
    build.add_argument("--openmp", choices=("auto", "on", "off"), default="auto")
    export = commands.add_parser("export", help="Create a complete HF safetensors checkpoint")
    export.add_argument("--model", type=Path, required=True, help="Local original checkpoint directory")
    export.add_argument("--output", type=Path, required=True, help="New checkpoint directory")
    export.add_argument("--recipe", type=Path, required=True, help="JSON selection and precision rules")
    export.add_argument("--hessians", type=Path, help="JSON map of checkpoint weight names to saved H.pt files")
    export.add_argument(
        "--dry-run", action="store_true", help="Check headers/recipe and print selected weights; write nothing"
    )
    export.add_argument("--backend", choices=("auto", "numpy", "native"), default="auto")
    export.add_argument("--threads", type=int, default=8)
    export.add_argument("--block-size", type=int, default=128)
    export.add_argument("--damping", type=float, default=0.01)
    export.add_argument(
        "--validate-ttnn", action="store_true", help="Also validate every prepared matrix with installed TTNN"
    )
    q = commands.add_parser("quantize", help="Prepare one weight tensor and a metadata sidecar")
    q.add_argument("--weight", type=Path, required=True)
    q.add_argument("--output", type=Path, required=True)
    q.add_argument("--method", choices=("round", "max-minus-one", "gptq-search"), required=True)
    q.add_argument("--bits", type=int, choices=(4, 8), default=4)
    q.add_argument("--layout", choices=("linear", "packed"), default="linear")
    q.add_argument("--hessian", type=Path)
    q.add_argument("--output-splits", type=int, nargs="+")
    q.add_argument("--backend", choices=("auto", "numpy", "native"), default="auto")
    q.add_argument("--threads", type=int, default=8)
    q.add_argument("--block-size", type=int, default=128)
    q.add_argument("--damping", type=float, default=0.01)
    q.add_argument("--validate-ttnn", action="store_true", help="Also run installed TTNN's host-packing check")
    args = parser.parse_args()
    if args.command == "build":
        print(json.dumps(build_native(args.openmp), indent=2))
        return
    if args.command == "export":
        from .checkpoint import export_checkpoint

        try:
            info = export_checkpoint(
                args.model,
                args.output,
                args.recipe,
                hessians=args.hessians,
                dry_run=args.dry_run,
                backend=args.backend,
                threads=args.threads,
                block_size=args.block_size,
                damping=args.damping,
                validate_ttnn=args.validate_ttnn,
                progress=lambda message: print(message, file=sys.stderr, flush=True),
            )
        except (ValueError, OSError, ImportError) as error:
            parser.error(str(error))
        print(
            json.dumps(
                info
                if args.dry_run
                else {
                    "output": str(args.output),
                    "selected_count": info["selected_count"],
                    "seconds": info["seconds"],
                    "report": str(args.output / "tt_bfp_quantization.json"),
                },
                indent=2,
            )
        )
        return
    if args.threads <= 0:
        parser.error("--threads must be positive")
    if args.output.resolve() == args.weight.resolve() or args.output.exists():
        parser.error("Choose a new output file; input/previous exports are never overwritten")
    if args.method == "gptq-search" and (args.bits != 4 or args.layout != "linear" or args.hessian is None):
        parser.error("gptq-search requires --bits 4 --layout linear --hessian H.pt")
    if args.method != "gptq-search" and args.hessian is not None:
        parser.error("Only gptq-search uses --hessian")
    if args.layout == "packed" and args.output_splits is not None:
        parser.error("For packed layout, export and process each physical shard separately")
    torch.set_num_threads(args.threads)
    weight = _tensor(args.weight)
    if args.method == "gptq-search":
        result, info = gptq_search(
            weight,
            _tensor(args.hessian),
            damping=args.damping,
            block_size=args.block_size,
            output_splits=args.output_splits,
            backend=args.backend,
            threads=args.threads,
        )
    else:
        function = search_linear if args.layout == "linear" else search_packed
        extra = {"output_splits": args.output_splits} if args.layout == "linear" else {}
        result, info = function(
            weight,
            args.bits,
            (0,) if args.method == "round" else (0, -1),
            backend=args.backend,
            threads=args.threads,
            **extra,
        )
    info["validation"] = validate_repacking(
        result, args.bits, layout=args.layout, output_splits=args.output_splits, native=args.validate_ttnn
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(to_bf16_exact(result), args.output)
    info.update(
        package_version=__version__,
        carrier_dtype="bfloat16",
        source_weight=str(args.weight),
        source_hessian=str(args.hessian) if args.hessian else None,
    )
    args.output.with_suffix(args.output.suffix + ".json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
