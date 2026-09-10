from __future__ import annotations


def cmd_overlay_drop(args) -> int:
    """Drop overlays.

    Two modes:
      * ``overlay-drop <model_id> <rel_path>`` — drop ONE file
        (surgical, original behavior).
      * ``overlay-drop <model_id>`` (rel_path omitted) — drop EVERY
        overlay for the scope. Useful when overlays were captured
        under a broken regime and the operator wants a clean slate.

    Returns rc=0 on success (even when no overlays existed — a
    "nothing to drop" no-op isn't an error). rc=1 when a specific
    rel_path was requested but didn't exist in the index.
    """
    rel_path = getattr(args, "rel_path", None)
    if rel_path:
        from ..overlay_manager import drop

        ok = drop(args.model_id, rel_path)
        if not ok:
            print(f"  no overlay for {rel_path} under {args.model_id}")
            return 1
        print(f"  dropped overlay: {args.model_id} / {rel_path}")
        return 0

    # Scope-wide drop (rel_path omitted). A composite's parts each hold their own
    # overlay scope, so dropping only the parent's leaves them to be replayed on the
    # next run -- the drop covers those too, or "clean slate" is not clean.
    from ..overlay_manager import component_scopes, drop_scope

    count, dropped = drop_scope(args.model_id)
    if count:
        print(f"  dropped {count} overlay(s) under {args.model_id}:")
        for rp in dropped:
            print(f"    - {rp}")

    comp_total = 0
    for scope in component_scopes(args.model_id):
        c_count, c_dropped = drop_scope(scope)
        if not c_count:
            continue
        comp_total += c_count
        print(f"  dropped {c_count} overlay(s) under component scope {scope}:")
        for rp in c_dropped:
            print(f"    - {rp}")

    if not count and not comp_total:
        print(f"  no overlays registered under {args.model_id} (nothing to drop)")
    return 0
