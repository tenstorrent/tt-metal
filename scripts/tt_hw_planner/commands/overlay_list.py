from __future__ import annotations


def cmd_overlay_list(args) -> int:
    from datetime import datetime

    from ..overlay_manager import list_overlays

    rows = list_overlays(model_id=getattr(args, "model_id", None) or None)
    if not rows:
        target = args.model_id if getattr(args, "model_id", None) else "any model"
        print(f"  no overlays found for {target}")
        return 0

    print(f"  {'MODEL':<48s}  {'FILE':<60s}  {'LINES':>6s}  CAPTURED")
    print("  " + "-" * 120)
    for r in rows:
        ts = r.get("captured_ts")
        ts_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if isinstance(ts, (int, float)) else "?"
        print(
            f"  {r['model_dir_slug']:<48s}  "
            f"{r['rel_path']:<60s}  "
            f"{str(r.get('line_count', '?')):>6s}  "
            f"{ts_str}"
        )
    return 0
