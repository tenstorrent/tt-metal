from __future__ import annotations

import sys


def cmd_auto_onboard(args) -> int:
    """LLM-draft a FamilyBackend for a brand-new HF model_type and
    optionally write it into family_backends.py."""
    from ..auto_onboard import auto_onboard, write_backend_into_registry

    sep = "=" * 78
    print(sep)
    print(f"  AUTO-ONBOARD  drafting FamilyBackend entry for {args.model_id}")
    print(sep)

    try:
        proposal = auto_onboard(
            args.model_id,
            agent_bin=args.agent_bin,
            model=args.auto_model,
            timeout_s=args.timeout_s,
            skip_llm=getattr(args, "skip_llm", False),
        )
    except Exception as exc:
        print(f"  auto-onboard failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    print(f"\n  Probe summary:")
    print(f"    model_type    = {proposal.new_model_type!r}")
    print(f"    pipeline_tag  = {proposal.new_pipeline_tag!r}")
    print(f"    category      = {proposal.inferred_category}")
    print(f"    closest backend (informational): {proposal.closest_existing_backend!r}")
    print(f"\n  Discovered components ({len(proposal.discovered_components)}):")
    for c in proposal.discovered_components[:15]:
        print(
            f"    - {c.get('name', '?'):<28s} class={c.get('class_name', '?'):<30s}"
            f" path={c.get('submodule_path', '?'):<35s} occ={c.get('occurrences', 0)}"
            f" leaves={c.get('leaf_op_count', 0)}"
        )
    if len(proposal.discovered_components) > 15:
        print(f"    ... ({len(proposal.discovered_components) - 15} more truncated)")

    if proposal.validation_errors:
        print(f"\n  Validation errors ({len(proposal.validation_errors)}):")
        for e in proposal.validation_errors:
            print(f"    - {e}")
        print(f"\n  Raw LLM response (first 500 chars):")
        print(f"  {proposal.llm_raw_response[:500]!r}")
        return 2

    print(f"\n  Proposed FamilyBackend (validated):")
    print()
    print(proposal.backend_dataclass_source.rstrip())

    if getattr(args, "accept", False):
        ok, msg = write_backend_into_registry(proposal)
        print(f"\n  {msg}")
        if not ok:
            return 2
        print(
            f"\n  Next step: `python -m scripts.tt_hw_planner up "
            f"{args.model_id} --auto ...` will now find this backend "
            f"by model_type={proposal.new_model_type!r}."
        )
    else:
        print(f"\n  (Proposal only; pass --accept to splice into " f"family_backends.py.)")
    return 0
