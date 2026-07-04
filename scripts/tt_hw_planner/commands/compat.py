from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def cmd_compat(args) -> int:
    from ..cli import check_compatibility, evaluate_kernels, probe_model, render_compat_json, render_compat_table

    probe = probe_model(args.model_id)
    if not probe.raw_config:
        print(
            f"ERROR: could not load config.json for {args.model_id}. "
            "Compatibility analysis needs the HuggingFace config; check that "
            "the repo is public (or HF_TOKEN is set) and that model_type is "
            "exposed in config.json.",
            file=sys.stderr,
        )
        return 1

    report = check_compatibility(args.model_id, probe.raw_config)

    kernel_report = None
    if not args.skip_kernel_check:
        tp_grid = args.tp_grid if args.tp_grid else None
        kernel_report = evaluate_kernels(probe.raw_config, tp_grid=tp_grid)

    _chips = None
    _mesh_arg = getattr(args, "mesh", None)
    if _mesh_arg:
        try:
            _prod = 1
            for _x in str(_mesh_arg).lower().split("x"):
                _prod *= int(_x)
            _chips = _prod
        except Exception:
            _chips = None

    if args.format == "json":
        print(render_compat_json(report, kernel_report))
    else:
        print(render_compat_table(report, kernel_report, verbose=args.verbose, chips=_chips))

    if report.overall == "BLOCKED":
        return 2

    try:
        from ..compatibility import Status as _CompatStatus

        _arch_blocked = any(
            r.needed and r.status == _CompatStatus.MISSING for r in (getattr(report, "results", None) or [])
        )
    except Exception:
        _arch_blocked = False
    if _arch_blocked:
        return 2
    if kernel_report is not None:
        _mesh_tp = 1
        _mesh_arg = getattr(args, "mesh", None)
        if _mesh_arg:
            try:
                _dims = [int(x) for x in str(_mesh_arg).lower().split("x")]
                if _dims:
                    _mesh_tp = 1
                    for d in _dims:
                        _mesh_tp *= d
            except Exception:
                _mesh_tp = 1
        if kernel_report.has_blockers(tp=_mesh_tp):
            return 2
    return 0
