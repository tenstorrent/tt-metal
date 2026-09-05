# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The graduated Nano Nemotron-H bring-up is part of the tool: routing, sibling
map, and Lightning overlays must be in the checkout, not only on one machine."""
from pathlib import Path

from scripts.tt_hw_planner.compatibility import SUPPORTED_HF_MODELS, closest_supported_model
from scripts.tt_hw_planner.family_backends import pick_backend_with_quality
from scripts.tt_hw_planner.overlay_manager import _load_index

_REPO = Path(__file__).resolve().parents[3]
_NANO_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
_LIGHTNING_ID = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"


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


def test_lightning_overlay_carries_nano_graduated_stubs() -> None:
    idx = _load_index(_LIGHTNING_ID)
    stubs = "models/tt_transformers/demo/nvidia_nemotron_3_5_lightning_30b_a3b_bf16/_stubs"
    for stem in (
        "nemotron_h_mamba2_mixer.py",
        "nemotron_h_mo_e.py",
        "nemotron_h_topk_router.py",
        "nemotron_h_block.py",
    ):
        assert f"{stubs}/{stem}" in idx, f"Lightning overlay missing seeded stub {stem}"
