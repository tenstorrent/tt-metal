"""SKU selection for issue verification runs.

Runner labels are never written here: `.github/sku_config.yaml` is the single
source of truth, and this module only decides which subset of it a verification
run is allowed to ask for.
"""

from __future__ import annotations

from pathlib import Path

import yaml

SKU_CONFIG_PATH = Path(__file__).resolve().parent.parent / "sku_config.yaml"

# Verification runs are untrusted-input driven (an issue body written by anyone),
# so they may only land on this narrow set of pools. Scarce multi-card and
# perf-pipeline SKUs are deliberately excluded.
ALLOWED_SKUS = {
    "github_hosted_cpu": "No Tenstorrent device. Host-only checks: golden re-execution, "
    "source reading, git history. Sufficient for any claim that reduces to "
    "'the reference value in the report is wrong'.",
    "wh_n150_civ2": "Single Wormhole B0 card. Use when the claim must be observed on "
    "silicon and names Wormhole, names no architecture at all, or cites a "
    "wormhole_b0 source path.",
    "bh_p150b_civ2_viommu": "Single Blackhole card. Use only when the claim is "
    "Blackhole-specific or cites a blackhole source path.",
}

DEFAULT_SKU = "github_hosted_cpu"


class UnknownSku(ValueError):
    pass


def load_runs_on(sku: str) -> list[str]:
    """Resolve a SKU to its runner labels, rejecting anything off the allowlist."""
    if sku not in ALLOWED_SKUS:
        raise UnknownSku(f"SKU {sku!r} is not verification-allowed. Pick one of: {sorted(ALLOWED_SKUS)}")

    config = yaml.safe_load(SKU_CONFIG_PATH.read_text())
    try:
        runs_on = config["skus"][sku]["runs_on"]
    except KeyError as exc:
        raise UnknownSku(f"SKU {sku!r} is allowlisted but missing from {SKU_CONFIG_PATH}") from exc

    return list(runs_on)


def needs_hardware(sku: str) -> bool:
    return sku != "github_hosted_cpu"


def describe_choices() -> str:
    """Render the allowlist for injection into the planner prompt."""
    return "\n".join(f"- `{sku}` — {why}" for sku, why in ALLOWED_SKUS.items())
