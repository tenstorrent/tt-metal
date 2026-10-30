"""Read-only visibility CLI: ``view-skips``.

Prints the persistent skip-list (verified KERNEL_MISSING components
blocked by TTNN op gaps). No mutations — purely diagnostic.
"""

from __future__ import annotations


def cmd_view_skips(args) -> int:
    """Pretty-print the skip-list. Only KERNEL_MISSING entries are
    persisted now; every line is a verified TTNN op gap blocking a
    component from running on device."""
    from ..overlay_manager import load_persistent_skips

    model_id = args.model_id
    skips = load_persistent_skips(model_id)
    if not skips:
        print(f"no skip-list entries for `{model_id}` — every component is on the bringup queue or graduated.")
        return 0

    print(f"skip-list for {model_id}  ({len(skips)} component(s) blocked by missing TTNN kernels)")
    print()
    print(f"  {'component':<40}  reason")
    print(f"  {'-'*40}  {'-'*60}")
    for name in sorted(skips.keys()):
        entry = skips[name]
        reason = (entry.get("reason") or "(no reason recorded)")[:90]
        print(f"  {name:<40}  {reason}")
    print()
    print("  These components stay on CPU until the missing TTNN op(s) land.")
    print("  After TTNN ships the kernel, run `overlay-clear-skips` then `up --auto`.")
    return 0
