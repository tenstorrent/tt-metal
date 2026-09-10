from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def cmd_scaffold(args) -> int:
    from ..scaffold import (
        ScaffoldError,
        apply_scaffold,
        plan_scaffold,
        render_apply,
        render_json as render_scaffold_json,
        render_patch,
        render_text,
    )

    try:
        plan = plan_scaffold(
            args.model_id,
            force_already_supported=getattr(args, "force_already_supported", False),
        )
    except ScaffoldError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    if args.format == "json":
        applied = apply_scaffold(plan) if args.apply else None
        print(render_scaffold_json(plan, applied))
        return 0
    if args.format == "patch":
        print(render_patch(plan))
        if args.apply:
            applied = apply_scaffold(plan)
            print(render_apply(plan, applied), file=sys.stderr)
        return 0

    print(render_text(plan, show_diff=not args.no_diff))

    if args.apply:
        applied = apply_scaffold(plan)
        print()
        print(render_apply(plan, applied))

    return 0
