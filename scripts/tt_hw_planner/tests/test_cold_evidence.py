"""Unit tests for the evidence-based COLD/HOT classifier.

Pins the rule: COLD requires POSITIVE evidence (never-fired,
negligible latency, or bandwidth-bound), not just absence of data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.tt_hw_planner.cold_evidence import (  # noqa: E402
    COLD_DENSITY_THRESHOLD,
    COLD_FREQUENCY_THRESHOLD,
    COLD_LATENCY_PCT_THRESHOLD,
    ComponentEvidence,
    _read_io_bytes,
    _read_ops_count,
    _walk_tensor_shapes,
    classify_evidence,
    measure_compute_density,
    measure_cpu_latency,
    measure_frequency,
)


# ---------------------------------------------------------------------------
# classify_evidence — the core rule combining all 3 signals
# ---------------------------------------------------------------------------


def test_classify_never_fired_is_cold() -> None:
    """Frequency=0 → COLD (Signal 1)."""
    kind, ev = classify_evidence(frequency=0.0, cpu_latency_pct=10.0, compute_density=1.0)
    assert kind == "COLD"
    assert any("frequency=0" in e for e in ev)


def test_classify_low_latency_is_cold() -> None:
    """Latency < 0.5% → COLD (Signal 2), even if frequency > 0."""
    kind, ev = classify_evidence(frequency=1.0, cpu_latency_pct=0.1, compute_density=1.0)
    assert kind == "COLD"
    assert any("cpu_latency_pct=0.10" in e for e in ev)


def test_classify_bandwidth_bound_is_cold() -> None:
    """Density < threshold → COLD (Signal 3)."""
    kind, ev = classify_evidence(frequency=1.0, cpu_latency_pct=5.0, compute_density=1e-9)
    assert kind == "COLD"
    assert any("compute_density" in e and "bandwidth-bound" in e for e in ev)


def test_classify_hot_when_all_signals_strong() -> None:
    """Frequency > 0 AND latency >= threshold AND density >= threshold → HOT."""
    kind, ev = classify_evidence(frequency=1.0, cpu_latency_pct=10.0, compute_density=1e-5)
    assert kind == "HOT"
    assert any("frequency=1" in e and "hot path" in e for e in ev)


def test_classify_unknown_when_no_signals() -> None:
    """No probe data → UNKNOWN, fall back to safe default downstream."""
    kind, ev = classify_evidence(frequency=None, cpu_latency_pct=None, compute_density=None)
    assert kind == "UNKNOWN"
    assert "no probe data" in ev[0]


def test_classify_partial_signal_can_still_decide() -> None:
    """Only frequency available — if 0, still COLD."""
    kind, _ = classify_evidence(frequency=0.0, cpu_latency_pct=None, compute_density=None)
    assert kind == "COLD"


def test_classify_threshold_boundaries() -> None:
    """Frequency just above 0 and latency just at threshold → HOT.
    Exactly at COLD_FREQUENCY_THRESHOLD (0.0) → COLD."""
    # Just at frequency=0
    kind, _ = classify_evidence(frequency=0.0, cpu_latency_pct=10.0, compute_density=1e-5)
    assert kind == "COLD"
    # Just above frequency=0 with healthy latency → HOT
    kind, _ = classify_evidence(frequency=0.5, cpu_latency_pct=10.0, compute_density=1e-5)
    assert kind == "HOT"
    # At latency threshold → HOT (strictly less is COLD)
    kind, _ = classify_evidence(frequency=1.0, cpu_latency_pct=COLD_LATENCY_PCT_THRESHOLD, compute_density=1e-5)
    assert kind == "HOT"


# ---------------------------------------------------------------------------
# measure_frequency — hooks fire counts across multi-pass forward
# ---------------------------------------------------------------------------


def test_measure_frequency_counts_per_pass() -> None:
    """A component invoked once per forward fires exactly n_passes times."""
    import torch

    class _M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Linear(4, 1)

        def forward(self, pixel_values):
            return self.head(pixel_values)

    model = _M()
    freq = measure_frequency(
        model,
        pixel_values_fn=lambda i: torch.zeros(1, 4),
        component_paths={"head_comp": "head"},
        n_passes=3,
    )
    assert freq["head_comp"] == 1.0


def test_measure_frequency_zero_for_uninvoked_path() -> None:
    """A component that the forward never reaches gets frequency=0."""
    import torch

    class _M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.used = torch.nn.Linear(4, 1)
            self.unused = torch.nn.Linear(4, 1)  # never called

        def forward(self, pixel_values):
            return self.used(pixel_values)

    model = _M()
    freq = measure_frequency(
        model,
        pixel_values_fn=lambda i: torch.zeros(1, 4),
        component_paths={"used": "used", "unused": "unused"},
        n_passes=4,
    )
    assert freq["used"] == 1.0
    assert freq["unused"] == 0.0  # never_fired = COLD evidence


# ---------------------------------------------------------------------------
# measure_cpu_latency — pre/post hooks time each component
# ---------------------------------------------------------------------------


def test_measure_cpu_latency_returns_positive_for_invoked() -> None:
    """A component that runs has mean_ms > 0; one that doesn't has 0."""
    import torch

    class _M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = torch.nn.Linear(64, 64)
            self.layer2 = torch.nn.Linear(64, 64)

        def forward(self, pixel_values):
            return self.layer2(self.layer1(pixel_values))

    model = _M()
    lat = measure_cpu_latency(
        model,
        pixel_values_fn=lambda i: torch.randn(1, 64),
        component_paths={"L1": "layer1", "L2": "layer2"},
        n_iters=3,
    )
    assert lat["L1"]["mean_ms"] > 0
    assert lat["L2"]["mean_ms"] > 0
    # pct should be in (0, 100]
    assert 0 < lat["L1"]["pct"] <= 100
    assert 0 < lat["L2"]["pct"] <= 100


def test_measure_cpu_latency_uninvoked_is_zero() -> None:
    import torch

    class _M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.used = torch.nn.Linear(4, 1)
            self.unused = torch.nn.Linear(4, 1)

        def forward(self, pixel_values):
            return self.used(pixel_values)

    lat = measure_cpu_latency(
        _M(),
        pixel_values_fn=lambda i: torch.zeros(1, 4),
        component_paths={"used": "used", "unused": "unused"},
        n_iters=2,
    )
    assert lat["used"]["mean_ms"] > 0
    assert lat["unused"]["mean_ms"] == 0.0
    assert lat["unused"]["pct"] == 0.0


# ---------------------------------------------------------------------------
# measure_compute_density — ops / io_bytes from manifests
# ---------------------------------------------------------------------------


def test_walk_tensor_shapes_simple_tensor() -> None:
    """A single bf16 tensor of shape (1,3,1024,1024) is 6 MB."""
    node = {"kind": "tensor", "shape": [1, 3, 1024, 1024], "dtype": "torch.bfloat16"}
    assert _walk_tensor_shapes(node) == 1 * 3 * 1024 * 1024 * 2


def test_walk_tensor_shapes_nested_dict_and_tuple() -> None:
    node = {
        "kind": "dict",
        "items": {
            "x": {"kind": "tensor", "shape": [1, 4], "dtype": "torch.float32"},
            "extra": {
                "kind": "tuple",
                "items": [
                    {"kind": "tensor", "shape": [2, 2], "dtype": "torch.float32"},
                ],
            },
        },
    }
    expected = 1 * 4 * 4 + 2 * 2 * 4
    assert _walk_tensor_shapes(node) == expected


def test_read_ops_count_sums_counts_dict(tmp_path: Path) -> None:
    """Sum REUSE+ADAPT+NEW from opplan manifest counts dict."""
    stub_dir = tmp_path / "_stubs"
    stub_dir.mkdir()
    (stub_dir / "comp_a.opplan.json").write_text(json.dumps({"counts": {"op-REUSE": 10, "op-ADAPT": 5, "op-NEW": 3}}))
    assert _read_ops_count(tmp_path, "comp_a") == 18


def test_read_ops_count_falls_back_to_palette_length(tmp_path: Path) -> None:
    stub_dir = tmp_path / "_stubs"
    stub_dir.mkdir()
    (stub_dir / "comp_a.opplan.json").write_text(json.dumps({"palette": ["op1", "op2", "op3"]}))
    assert _read_ops_count(tmp_path, "comp_a") == 3


def test_read_io_bytes_sums_args_kwargs_output(tmp_path: Path) -> None:
    """io_bytes is the sum of bytes across captured args + kwargs + output tensors."""
    cap = tmp_path / "_captured" / "comp_a"
    cap.mkdir(parents=True)
    (cap / "manifest.json").write_text(
        json.dumps(
            {
                "args": {
                    "kind": "tuple",
                    "items": [{"kind": "tensor", "shape": [1, 4], "dtype": "torch.float32"}],
                },
                "kwargs": {"kind": "dict", "items": {}},
                "output": {"kind": "tensor", "shape": [1, 1], "dtype": "torch.float32"},
            }
        )
    )
    # 4 fp32 elements + 1 fp32 element = 5 * 4 bytes = 20 bytes
    assert _read_io_bytes(tmp_path, "comp_a") == 20


def test_measure_affinity_score_handles_pre_bound_schema(tmp_path: Path) -> None:
    """The op-synth opplan schema has `pre_bound: [{ttnn_target: ...}]`
    rather than counts/palette. Affinity scoring must extract from
    pre_bound too."""
    from scripts.tt_hw_planner.cold_evidence import measure_affinity_score

    stub_dir = tmp_path / "_stubs"
    stub_dir.mkdir()
    (stub_dir / "comp_a.opplan.json").write_text(
        json.dumps(
            {
                "pre_bound": [
                    {"name": "c1", "ttnn_target": "ttnn.conv2d", "helper": "x"},
                    {"name": "ln1", "ttnn_target": "ttnn.layer_norm", "helper": "x"},
                    {"name": "lin1", "ttnn_target": "ttnn.linear", "helper": "x"},
                    {"name": "g1", "ttnn_target": "ttnn.gelu", "helper": "x"},
                ]
            }
        )
    )
    out = measure_affinity_score(tmp_path, {"comp_a": "x.y"})
    # conv2d (+1) + layer_norm (+1) + linear (+1) + gelu (0) = +3
    assert out["comp_a"] == 3


def test_measure_affinity_score_pre_bound_with_unfavorable(tmp_path: Path) -> None:
    """A pre_bound list mostly of unfavorable ops scores negative."""
    from scripts.tt_hw_planner.cold_evidence import measure_affinity_score

    stub_dir = tmp_path / "_stubs"
    stub_dir.mkdir()
    (stub_dir / "comp_a.opplan.json").write_text(
        json.dumps(
            {
                "pre_bound": [
                    {"ttnn_target": "ttnn.permute"},
                    {"ttnn_target": "ttnn.reshape"},
                    {"ttnn_target": "ttnn.matmul"},
                ]
            }
        )
    )
    out = measure_affinity_score(tmp_path, {"comp_a": "x.y"})
    # -1 (permute) - 1 (reshape) + 1 (matmul) = -1
    assert out["comp_a"] == -1


def test_measure_compute_density_combines_ops_and_io(tmp_path: Path) -> None:
    """density = ops_count / io_bytes; zero io_bytes → zero density."""
    stub_dir = tmp_path / "_stubs"
    stub_dir.mkdir()
    cap = tmp_path / "_captured" / "comp_a"
    cap.mkdir(parents=True)
    (stub_dir / "comp_a.opplan.json").write_text(json.dumps({"counts": {"op-NEW": 100}}))
    (cap / "manifest.json").write_text(
        json.dumps(
            {
                "args": {"kind": "tensor", "shape": [1024], "dtype": "torch.float32"},
                "kwargs": {"kind": "dict", "items": {}},
                "output": {"kind": "tensor", "shape": [1024], "dtype": "torch.float32"},
            }
        )
    )
    out = measure_compute_density(tmp_path, {"comp_a": "some.path"})
    # 100 ops / (1024 * 4 + 1024 * 4) bytes = 100 / 8192 = 0.0122
    assert out["comp_a"]["ops_count"] == 100
    assert out["comp_a"]["io_bytes"] == 8192
    assert out["comp_a"]["density"] == pytest_approx(100 / 8192)


def pytest_approx(expected, tol=1e-9):
    """Tiny shim — pytest.approx without the import roundabout."""

    class _A:
        def __eq__(self, other):
            return abs(other - expected) < tol

    return _A()


# ---------------------------------------------------------------------------
# Per-workload-mode evidence: union of evidence across modes
# ---------------------------------------------------------------------------


def test_union_hot_if_any_mode_is_hot() -> None:
    """A component is HOT if it's HOT in ANY measured mode (even if COLD
    in others). Use-case: a video module that's HOT in video-mode and
    COLD in image-mode — should graduate so video workload runs fast."""
    from scripts.tt_hw_planner.cold_evidence import ComponentEvidence, union_evidence_across_modes

    image = ComponentEvidence(kind="COLD", frequency=0.0, cpu_latency_pct=0.0, workload_mode="image")
    video = ComponentEvidence(kind="HOT", frequency=1.0, cpu_latency_pct=15.0, workload_mode="video")

    union = union_evidence_across_modes({"image": image, "video": video})
    assert union.kind == "HOT"
    # Max across modes for each signal
    assert union.frequency == 1.0
    assert union.cpu_latency_pct == 15.0


def test_union_cold_only_if_cold_in_every_mode() -> None:
    """A component is COLD only if it's COLD in EVERY measured mode."""
    from scripts.tt_hw_planner.cold_evidence import ComponentEvidence, union_evidence_across_modes

    image = ComponentEvidence(kind="COLD", frequency=0.0, workload_mode="image")
    video = ComponentEvidence(kind="COLD", frequency=0.0, workload_mode="video")

    union = union_evidence_across_modes({"image": image, "video": video})
    assert union.kind == "COLD"


def test_union_unknown_when_no_signals() -> None:
    """A component with UNKNOWN in every mode stays UNKNOWN at the union."""
    from scripts.tt_hw_planner.cold_evidence import ComponentEvidence, union_evidence_across_modes

    a = ComponentEvidence(kind="UNKNOWN", workload_mode="image")
    b = ComponentEvidence(kind="UNKNOWN", workload_mode="video")

    union = union_evidence_across_modes({"image": a, "video": b})
    assert union.kind == "UNKNOWN"


def test_union_evidence_includes_per_mode_tag() -> None:
    """The union's evidence reasons are tagged with the mode they came from
    so the operator can see which mode contributed which signal."""
    from scripts.tt_hw_planner.cold_evidence import ComponentEvidence, union_evidence_across_modes

    image = ComponentEvidence(kind="COLD", evidence=["frequency=0 (never-fired)"], workload_mode="image")
    video = ComponentEvidence(kind="HOT", evidence=["frequency=1 (on hot path)"], workload_mode="video")

    union = union_evidence_across_modes({"image": image, "video": video})
    assert any("[image]" in e for e in union.evidence)
    assert any("[video]" in e for e in union.evidence)


def test_merge_evidence_into_persisted_creates_multi_mode_entry() -> None:
    """merge_evidence_into_persisted: new run creates a multi-mode entry
    with the new run's mode under modes[<mode>]."""
    from scripts.tt_hw_planner.cold_evidence import ComponentEvidence, merge_evidence_into_persisted

    existing = {}  # no prior evidence
    new_report = {"comp_a": ComponentEvidence(kind="HOT", frequency=1.0, workload_mode="video")}

    merged = merge_evidence_into_persisted(existing, new_report, workload_mode="video")
    assert "comp_a" in merged
    entry = merged["comp_a"]
    assert "modes" in entry
    assert "video" in entry["modes"]
    assert entry["kind"] == "HOT"  # union of single mode = that mode's kind


def test_merge_legacy_entry_promotes_to_multi_mode() -> None:
    """When the existing entry is a flat (single-mode) dict, the merge
    promotes it to multi-mode and adds the new run alongside."""
    from scripts.tt_hw_planner.cold_evidence import ComponentEvidence, merge_evidence_into_persisted

    existing = {
        "comp_a": {
            "kind": "COLD",
            "frequency": 0.0,
            "cpu_latency_pct": 0.0,
            "workload_mode": "image",
            "evidence": ["frequency=0 (never-fired)"],
        }
    }
    new_report = {"comp_a": ComponentEvidence(kind="HOT", frequency=1.0, cpu_latency_pct=20.0, workload_mode="video")}

    merged = merge_evidence_into_persisted(existing, new_report, workload_mode="video")
    entry = merged["comp_a"]
    assert "image" in entry["modes"]
    assert "video" in entry["modes"]
    # Union of COLD + HOT → HOT
    assert entry["kind"] == "HOT"


def test_merge_same_mode_overwrites() -> None:
    """Re-running profile-cold with the same workload-mode OVERWRITES
    that mode's record (latest measurement wins per mode)."""
    from scripts.tt_hw_planner.cold_evidence import ComponentEvidence, merge_evidence_into_persisted

    existing = {
        "comp_a": {
            "kind": "HOT",
            "frequency": 0.5,
            "modes": {"image": {"kind": "HOT", "frequency": 0.5, "workload_mode": "image"}},
        }
    }
    # Re-run image mode with different freq
    new_report = {"comp_a": ComponentEvidence(kind="HOT", frequency=1.0, workload_mode="image")}

    merged = merge_evidence_into_persisted(existing, new_report, workload_mode="image")
    # The newer image-mode record overwrites
    assert merged["comp_a"]["modes"]["image"]["frequency"] == 1.0
