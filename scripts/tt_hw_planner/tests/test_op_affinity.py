"""Unit tests for ttnn-op affinity scoring.

Pins the catalog of device-favorable vs device-unfavorable ops and the
component-level scoring rule (sum of count × affinity over opplan).
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.tt_hw_planner.op_affinity import (  # noqa: E402
    DEVICE_FAVORABLE,
    DEVICE_UNFAVORABLE,
    NEUTRAL,
    affinity_label,
    component_affinity_score,
    op_affinity_score,
)


# ---------------------------------------------------------------------------
# op_affinity_score: per-op scoring
# ---------------------------------------------------------------------------


def test_matmul_is_device_favorable():
    assert op_affinity_score("ttnn.matmul") == 1


def test_conv2d_is_device_favorable():
    assert op_affinity_score("ttnn.conv2d") == 1


def test_scaled_dot_product_attention_is_favorable():
    assert op_affinity_score("ttnn.scaled_dot_product_attention") == 1


def test_permute_is_device_unfavorable():
    assert op_affinity_score("ttnn.permute") == -1


def test_reshape_is_device_unfavorable():
    assert op_affinity_score("ttnn.reshape") == -1


def test_gather_is_device_unfavorable():
    assert op_affinity_score("ttnn.gather") == -1


def test_tilize_is_device_unfavorable():
    """Pure layout op — transfer/dispatch overhead with no compute payoff."""
    assert op_affinity_score("ttnn.tilize") == -1


def test_from_torch_is_device_unfavorable():
    """ttnn.from_torch is a host->device transfer — never a compute win."""
    assert op_affinity_score("ttnn.from_torch") == -1


def test_gelu_is_neutral():
    assert op_affinity_score("ttnn.gelu") == 0


def test_unknown_op_defaults_neutral():
    """Unknown ops score 0 by safe-default."""
    assert op_affinity_score("ttnn.brand_new_op_xyz") == 0


def test_non_ttnn_op_scores_zero():
    """Raw torch ops aren't scored; they'd need to be ported to ttnn first."""
    assert op_affinity_score("torch.matmul") == 0


def test_bare_op_name_normalized_to_ttnn():
    """Bare names like ``matmul`` are normalized to ``ttnn.matmul``."""
    assert op_affinity_score("matmul") == 1


def test_empty_input_scores_zero():
    assert op_affinity_score("") == 0
    assert op_affinity_score(None) == 0  # type: ignore[arg-type]


def test_no_overlap_between_buckets():
    """Catalog correctness: no op should be in multiple buckets at once."""
    assert DEVICE_FAVORABLE.isdisjoint(DEVICE_UNFAVORABLE)
    assert DEVICE_FAVORABLE.isdisjoint(NEUTRAL)
    assert DEVICE_UNFAVORABLE.isdisjoint(NEUTRAL)


# ---------------------------------------------------------------------------
# component_affinity_score: sum over opplan
# ---------------------------------------------------------------------------


def test_component_score_from_counts_dict():
    """When an opplan has counts {op: N}, score = Σ N × affinity."""
    counts = {"ttnn.matmul": 5, "ttnn.permute": 2}
    # +5 (matmul) + (-2) (permute) = +3
    assert component_affinity_score(counts) == 3


def test_component_score_unfavorable_dominated():
    """When unfavorable ops dominate, score is negative — component
    should land in COLD even if it has SOME compute."""
    counts = {"ttnn.matmul": 1, "ttnn.permute": 5, "ttnn.reshape": 3}
    # +1 - 5 - 3 = -7
    assert component_affinity_score(counts) == -7


def test_component_score_from_palette_list():
    """A list-form opplan (palette) scores each op once."""
    palette = ["ttnn.conv2d", "ttnn.layer_norm", "ttnn.gelu"]
    # +1 (conv2d) + 1 (layer_norm) + 0 (gelu) = +2
    assert component_affinity_score(palette) == 2


def test_component_score_handles_none():
    assert component_affinity_score(None) == 0


def test_component_score_neutral_when_only_neutral_ops():
    counts = {"ttnn.gelu": 10, "ttnn.relu": 5}
    assert component_affinity_score(counts) == 0


def test_component_score_ignores_non_ttnn_entries():
    """Mixed lists with non-ttnn ops: non-ttnn contributes 0."""
    counts = {"ttnn.matmul": 3, "torch.matmul": 100}
    assert component_affinity_score(counts) == 3


# ---------------------------------------------------------------------------
# affinity_label
# ---------------------------------------------------------------------------


def test_affinity_label_for_positive():
    assert affinity_label(5) == "device-favorable"


def test_affinity_label_for_negative():
    assert affinity_label(-3) == "device-unfavorable"


def test_affinity_label_for_zero():
    assert affinity_label(0) == "neutral"


# ---------------------------------------------------------------------------
# Integration with classify_evidence
# ---------------------------------------------------------------------------


def test_classify_with_negative_affinity_is_cold():
    """Bug-J style: a HOT-path bandwidth-bound component should COLD
    via the new affinity signal too. vision_neck (permute-heavy) →
    negative affinity → COLD even if frequency=1 and latency=high."""
    from scripts.tt_hw_planner.cold_evidence import classify_evidence

    kind, ev = classify_evidence(
        frequency=1.0,
        cpu_latency_pct=6.0,  # latency-significant
        compute_density=1e-5,  # density-favorable
        affinity_score=-5,  # ops are bandwidth-bound
    )
    assert kind == "COLD"
    assert any("affinity_score=-5" in e for e in ev)


def test_classify_with_positive_affinity_strengthens_hot():
    """All positive signals AND positive affinity → HOT."""
    from scripts.tt_hw_planner.cold_evidence import classify_evidence

    kind, ev = classify_evidence(
        frequency=1.0,
        cpu_latency_pct=10.0,
        compute_density=1e-5,
        affinity_score=8,
    )
    assert kind == "HOT"
    assert any("affinity_score=8" in e and "device-favorable ops" in e for e in ev)


def test_classify_with_only_affinity_signal():
    """Affinity alone can decide when other signals absent."""
    from scripts.tt_hw_planner.cold_evidence import classify_evidence

    kind, _ = classify_evidence(frequency=None, cpu_latency_pct=None, compute_density=None, affinity_score=-3)
    assert kind == "COLD"

    kind2, _ = classify_evidence(frequency=None, cpu_latency_pct=None, compute_density=None, affinity_score=5)
    assert kind2 == "HOT"


def test_classify_zero_affinity_is_neutral_not_cold():
    """A score of 0 (no opplan data OR only neutral ops) should NOT
    force COLD — let other signals decide. The classifier penalizes
    only NEGATIVE affinity (more unfavorable than favorable ops)."""
    from scripts.tt_hw_planner.cold_evidence import classify_evidence

    # Strong positive signals + affinity_score=0 → still HOT
    kind, ev = classify_evidence(frequency=1.0, cpu_latency_pct=10.0, compute_density=1e-5, affinity_score=0)
    assert kind == "HOT"
    # No affinity-COLD reason should appear
    assert not any("affinity_score=0 (lacks" in e for e in ev)

    # Only affinity_score=0 with no other signals → UNKNOWN (probe didn't
    # measure enough). The has_any_signal flag IS set by affinity=0 being
    # present, so this case falls to HOT-with-no-positive-signals path.
    # Verify it doesn't COLD:
    kind2, _ = classify_evidence(frequency=None, cpu_latency_pct=None, compute_density=None, affinity_score=0)
    assert kind2 != "COLD"
