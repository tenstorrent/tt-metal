#!/usr/bin/env python3
"""
Fail-fast check that a (SKU, tier, model) filter combination selects at least one
test leg from a pipeline_reorg tests YAML.

Runs before the expensive build so an impossible combination (e.g. a model that
isn't configured for the chosen SKU/tier) fails in seconds with a GitHub warning
annotation that lists the valid models, tiers, and SKUs — instead of building and
then silently skipping an empty matrix.

Usage:
    python validate_test_combination.py <tests_yaml> <enabled_skus_csv> <tier> <model>

  enabled_skus_csv : comma-separated logical SKUs already resolved by the caller
                     (a single SKU, or the pipeline's full ALL_SKUS list).
  tier             : "all", or "1"/"2"/"3".
  model            : "all", or a comma-separated OR-list of case-insensitive
                     substrings matched against each entry's `model` (hub id).

Exit 0 when >=1 leg matches; exit 1 (with ::error:: + ::warning::) when none do.
"""

import sys

import yaml


def parse_csv(s):
    return [t.strip() for t in (s or "").split(",") if t.strip()]


def model_matches(entry_model, model_filter):
    if model_filter.strip().lower() == "all":
        return True
    mm = (entry_model or "").lower()
    return any(tok in mm for tok in [t.lower() for t in parse_csv(model_filter)])


def expand(tests):
    """Flatten to (model, sku, tier) rows — one per SKU under each test entry."""
    rows = []
    for test in tests:
        skus = test.get("skus")
        if not isinstance(skus, dict):
            continue
        for sku, cfg in skus.items():
            if not isinstance(cfg, dict):
                continue
            rows.append({"model": test.get("model"), "sku": sku, "tier": cfg.get("tier")})
    return rows


def gh_warning(message):
    # GitHub annotations are single-line; encode newlines so the multi-line body
    # renders in the annotation. Also echo the plain text to the step log.
    print(message)
    print("::warning title=No matching test combination::" + message.replace("\n", "%0A"))


def main(argv):
    if len(argv) != 5:
        print(f"::error::usage: {argv[0]} <tests_yaml> <enabled_skus_csv> <tier> <model>")
        return 2

    tests_yaml, enabled_skus_csv, tier, model = argv[1], argv[2], argv[3], argv[4]

    with open(tests_yaml, "r") as f:
        tests = yaml.safe_load(f) or []

    rows = expand(tests)
    enabled = set(parse_csv(enabled_skus_csv))

    def tier_ok(row_tier):
        return tier.strip().lower() == "all" or str(row_tier) == str(tier).strip()

    matched = [
        r for r in rows if r["sku"] in enabled and tier_ok(r["tier"]) and model_matches(r["model"], model)
    ]

    print(f"Filters → SKUs={sorted(enabled)} tier={tier} model={model}")
    if matched:
        print(f"✅ {len(matched)} test leg(s) match the selected combination.")
        return 0

    # Empty combination — fail with the catalogue of what IS selectable.
    models = sorted({r["model"] for r in rows if r["model"]})
    tiers = sorted({str(r["tier"]) for r in rows if r["tier"] is not None})
    skus = sorted({r["sku"] for r in rows})
    combos = sorted({(r["model"], r["sku"], str(r["tier"])) for r in rows})

    lines = [
        f"No test leg matches SKUs={sorted(enabled)}, tier={tier}, model={model}.",
        "",
        f"Valid models: {', '.join(models)}",
        f"Valid tiers:  {', '.join(tiers)}",
        f"Valid SKUs:   {', '.join(skus)}",
        "",
        "Valid combinations (model | SKU | tier):",
    ]
    lines += [f"  - {m} | {s} | {t}" for (m, s, t) in combos]
    gh_warning("\n".join(lines))
    print("::error::Selected filter combination matches no tests — see the warning above for valid combinations.")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
