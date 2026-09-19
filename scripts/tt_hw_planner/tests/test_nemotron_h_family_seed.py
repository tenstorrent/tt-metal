# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The graduated Nano Nemotron-H bring-up is part of the tool: routing and the
sibling map must be in the checkout, not only on one machine.

Overlays are deliberately NOT asserted here. A third test used to require the
Lightning overlay index to carry Nano's seeded stubs, on the premise that overlays
belong in the checkout. That premise was retired: the overlay store is per-run
state, untracked and gitignored, so such a test can only pass on a machine that
happens to have run that bring-up — and what it pinned (carrying graduation
markers from one model to another) is the defect that reported a model as
graduated for work it had never done. The durable guarantee is below: routing
resolves to a demo directory that is really in the repo."""
from pathlib import Path

from scripts.tt_hw_planner.compatibility import SUPPORTED_HF_MODELS, closest_supported_model
from scripts.tt_hw_planner.family_backends import pick_backend_with_quality

_REPO = Path(__file__).resolve().parents[3]
_NANO_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"


def test_nemotron_h_routes_exactly_to_the_graduated_nano_demo() -> None:
    backend, quality = pick_backend_with_quality(category="LLM", model_type="nemotron_h")
    assert quality == "exact"
    assert backend is not None
    assert backend.demo_path == "models/demos/nvidia_nemotron_3_nano_30b_a3b_bf16"
    assert (_REPO / backend.demo_path).is_dir()
    assert (_REPO / backend.demo_path / "_stubs" / "nemotron_h_mamba2_mixer.py").is_file()


def test_nemotron_h_sibling_map_points_at_nano() -> None:
    assert _NANO_ID in SUPPORTED_HF_MODELS
    assert closest_supported_model("someone/new-nemotron-h", {"model_type": "nemotron_h"}) == _NANO_ID
