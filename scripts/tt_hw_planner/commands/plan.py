from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def cmd_plan(args) -> int:
    from ..cli import (
        HARDWARE,
        _dtypes_for,
        _filter_verdict_by_divisibility,
        _render_weights_only,
        evaluate_all,
        evaluate_kernels,
        find_box,
        probe_model,
        render_json,
        render_markdown,
        render_table,
    )

    probe = probe_model(args.model_id)

    boxes = HARDWARE if not args.box else [find_box(n) for n in args.box]
    dtypes = _dtypes_for(probe.category, args.dtype, probe.saved_dtype)

    if probe.memory_model is None:
        return _render_weights_only(probe, boxes, dtypes, args)

    kv_bpe = 2.0 if args.kv_dtype == "bf16" else (4.0 if args.kv_dtype == "fp32" else 2.0)
    verdict = evaluate_all(
        model=probe.memory_model,
        boxes=boxes,
        dtypes=dtypes,
        batch=args.batch,
        seq=args.seq,
        kv_dtype_bytes=kv_bpe,
        all_meshes=args.all_meshes,
        explore_pp=args.explore_pp,
    )

    if probe.raw_config:
        all_mesh_verdict = (
            verdict
            if args.all_meshes
            else evaluate_all(
                model=probe.memory_model,
                boxes=boxes,
                dtypes=dtypes,
                batch=args.batch,
                seq=args.seq,
                kv_dtype_bytes=kv_bpe,
                all_meshes=True,
                explore_pp=args.explore_pp,
            )
        )
        tps = sorted({max(1, int(r.mesh_shape[1])) for r in all_mesh_verdict.rows} | {1})
        kernel_report = evaluate_kernels(probe.raw_config, tp_grid=tps)
        verdict = _filter_verdict_by_divisibility(verdict, kernel_report, all_mesh_verdict)

    if args.format == "json":
        print(render_json(probe, verdict, args.batch, args.seq, args.kv_dtype, dtypes))
    elif args.format == "markdown":
        print(render_markdown(probe, verdict, args.batch, args.seq, args.kv_dtype, dtypes))
    else:
        print(
            render_table(
                probe, verdict, args.batch, args.seq, args.kv_dtype, dtypes, show_overhead=not args.no_overhead_detail
            )
        )
    return 0
