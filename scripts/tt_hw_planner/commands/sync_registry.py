"""``sync-registry`` command — deterministic-registry drift detection (fixes-plan Point 2a).

Verifies that every path the Layer-2 registries point at still exists in the
checkout, and (reverse) hints at reusable ``tt_transformers/tt`` modules that no
registry entry maps. ``--check`` exits non-zero on hard drift (a mapped path
that is gone) so CI / a pre-plan gate fails loudly instead of the planner
silently mis-pointing at a stale sibling.
"""

from ..discovery import REPO_ROOT
from ..registry_sync import add_source, check_registry_drift, format_drift, has_hard_drift, load_extra_sources


def cmd_sync_registry(args) -> int:
    new_sources = getattr(args, "add_source", None) or []
    if new_sources:
        kind = getattr(args, "kind", None) or "component"
        default = getattr(args, "default", None) or "adapt"
        persisted = []
        for src in new_sources:
            persisted = add_source(src, kind=kind, default=default)
            print(f"[sync-registry] registered source: {src}  (kind={kind}, default={default})")
        print(f"[sync-registry] {len(persisted)} extra source(s) now scanned on every up/auto-up:")
        for r in persisted:
            print(f"    {r.path:40s} kind={r.kind} default={r.default}")
        print(
            "\nThese roots are fetched from tenstorrent/tt-metal main and their PROVIDES/FAMILY "
            "manifests + heuristic modules extend the reuse map + sibling families on the next run."
        )
        return 0

    issues = check_registry_drift(REPO_ROOT, include_unmapped=not getattr(args, "no_unmapped", False))
    print(format_drift(issues))
    if getattr(args, "check", False) and has_hard_drift(issues):
        n = sum(1 for i in issues if i.kind == "missing_path")
        print(
            f"\n[sync-registry] FAIL: {n} registry path(s) missing from the checkout — fix the registry or restore the paths."
        )
        return 1
    return 0
