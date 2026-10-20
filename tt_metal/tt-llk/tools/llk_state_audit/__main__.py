"""Command-line entry point for the LLK state audit inventory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import audit, build_effects, generate, verify
from .candidates import enforce_candidates, load_candidate_model
from .inventory import (
    AuditModelError,
    inventory,
    load_effect_model,
    scan_functions,
    scan_helpers,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inventory LLK stateful API definitions."
    )
    parser.add_argument(
        "command",
        choices=(
            "inventory",
            "check",
            "effects",
            "audit",
            "generate",
            "verify",
            "bh-effects",
            "bh-hazards",
        ),
        nargs="?",
        default="inventory",
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--effects", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    # Blackhole state-hazard audit (bh-effects / bh-hazards); no git required.
    parser.add_argument(
        "--arch",
        choices=("blackhole",),
        default="blackhole",
        help="architecture for bh-* commands",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="tt-metal tree root for bh-* commands",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="output JSON file for bh-* commands"
    )
    parser.add_argument(
        "--families",
        default=None,
        help="comma-separated subset of bh/families modules to load (default: all)",
    )
    args = parser.parse_args()
    if args.command in ("bh-effects", "bh-hazards"):
        if args.repo_root is None or args.out is None:
            parser.error(f"{args.command} requires --repo-root and --out")
        from .bh import effects_db

        families = (
            None
            if args.families is None
            else [f for f in args.families.split(",") if f]
        )
        if args.command == "bh-effects":
            data = effects_db.write_effects(args.repo_root, args.out, families=families)
            summary = {
                "arch": args.arch,
                "counts": data["counts"],
                "tree_fingerprint": data["tree_fingerprint"],
            }
        else:
            data = effects_db.write_hazards(args.repo_root, args.out, families=families)
            summary = {
                "arch": args.arch,
                "findings": len(data["findings"]),
                "tree_fingerprint": data["tree_fingerprint"],
            }
        print(json.dumps(summary, sort_keys=True))
        return 0
    try:
        if args.command == "check":
            model = load_effect_model(args.effects, root=args.root)
            candidate_model = load_candidate_model()
            records, scan_model = scan_functions(args.root, args.effects)
            helpers = scan_helpers(args.root, args.effects, _model=scan_model)
            candidates = enforce_candidates(records, helpers, candidate_model)
            print(
                json.dumps(
                    {
                        "schema_version": 1,
                        "model_valid": True,
                        "effect_count": len(model["effects"]),
                        "candidate_count": len(candidates),
                    },
                    sort_keys=True,
                )
            )
        elif args.command == "effects":
            print(
                json.dumps(
                    build_effects(args.root, args.effects), indent=2, sort_keys=True
                )
            )
        elif args.command == "audit":
            print(json.dumps(audit(args.root, args.effects), indent=2, sort_keys=True))
        elif args.command == "generate":
            generated = generate(
                args.root, output_dir=args.output_dir, effect_model=args.effects
            )
            print(
                json.dumps(
                    {
                        "csv_path": str(generated["csv_path"]),
                        "json_path": str(generated["json_path"]),
                        "readme_path": str(generated["readme_path"]),
                        "summary": generated["summary"],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        elif args.command == "verify":
            result = verify(
                args.root, output_dir=args.output_dir, effect_model=args.effects
            )
            print(json.dumps(result, sort_keys=True))
            if not result["valid"]:
                return 1
        else:
            print(
                json.dumps(inventory(args.root, args.effects), indent=2, sort_keys=True)
            )
    except AuditModelError as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
